import os
import random

import numpy as np
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator


def init_cls_datasets_dali(cfg):
    train_pipe = ClsTrainPipe(cfg=cfg)
    val_pipe = ClsValPipe(cfg=cfg)
    train_pipe.build()
    val_pipe.build()

    dataloaders = {}
    dataloaders['train'] = DALIGenericIterator(train_pipe, ["images", "labels"],
                                               train_pipe.eii.n,
                                               fill_last_batch=True)
    dataloaders['val'] = DALIGenericIterator(val_pipe, ["images", "labels"],
                                             val_pipe.eii.n,
                                             fill_last_batch=True)
    cfg.dataset.train_num = dataloaders['train']._size
    cfg.dataset.val_num = dataloaders['val']._size
    return dataloaders                        


class ClsInputIterator(object):
    '''
        定义外部输入的数据迭代器，类似于Pytorch的自定义Dataset
    '''
    def __init__(self, cfg, is_train=False):
        self.cfg = cfg
        self.is_train = is_train
        self._get_label_dict()
        self._split_trainval_set(cfg.dataset.train_perc)

        if is_train:
            txt_file = os.path.join(cfg.dataset.txt_root, 'train.txt')
        else:
            txt_file = os.path.join(cfg.dataset.txt_root, 'val.txt')
        with open(txt_file, 'r') as f:
            self.f_list_all = f.readlines()

    def shuffle(self, seed):
        '''
        如果是多卡并行，则在每次epoch循环时，重新对data进行shuffle操作并分块
        '''
        random.seed(seed)
        random.shuffle(self.f_list_all)
        self.f_list = self._get_data_split()    

    def _get_data_split(self):
        if self.cfg.device.num > 1:
            idx_start = self.cfg.device.local_rank * self.cfg.dataset.loader.batch_size * self.num_batch
            idx_end = (self.cfg.device.local_rank+1) * self.cfg.dataset.loader.batch_size * self.num_batch
            f_list = self.f_list_all[idx_start:idx_end]
        else:
            f_list = self.f_list_all
        return f_list

    def __iter__(self):
        self.i = 0
        if self.cfg.device.num > 1:
            self.num_batch = int(len(self.f_list_all) / (self.cfg.device.num * self.cfg.dataset.loader.batch_size))
            self.n = self.cfg.dataset.loader.batch_size * self.num_batch
        else:
            self.n = len(self.f_list_all)
        self.f_list = self._get_data_split()    
        return self

    def __next__(self):
        images = [] 
        labels = []
        for idx in range(self.cfg.dataset.loader.batch_size):
            image_path, label = self.f_list[self.i].strip().split(' ')
            image_data = np.fromstring(open(image_path, 'rb').read(), np.uint8)
            images.append(np.frombuffer(image_data, dtype = np.uint8))
            labels.append(np.array(int(label), dtype = np.int32))
            self.i = (self.i + 1) % self.n
        return (images, labels)

    def _get_label_dict(self):
        self.class_names = os.listdir(self.cfg.dataset.root)
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        return

    def _split_trainval_set(self, train_perc):
        '''
        如果train-set和val-set的文件列表不存在，则随机生成文件列表
        如果已经存在，则沿用已有的列表
        '''
        txt_dir = self.cfg.dataset.txt_root
        if (os.path.isfile(os.path.join(txt_dir, 'train.txt'))
            and os.path.isfile(os.path.join(txt_dir, 'val.txt'))):
            return
        else:
            file_list = []
            for class_name in self.class_names:
                sub_dir = os.path.join(self.cfg.dataset.root, class_name)
                if os.path.isdir(sub_dir):
                    file_sub_list = os.listdir(sub_dir)
                    file_sub_list = [os.path.join(self.cfg.dataset.root, class_name, name) \
                                        + ' ' + str(self.class_dict[class_name]) + '\n'
                                        for name in file_sub_list]
                    file_list += file_sub_list
                else:
                    raise Exception('No samples belong to class {}'.format(class_name))
            random.shuffle(file_list)
            num_train = int(train_perc*len(file_list))
            file_train = file_list[:num_train]
            file_val = file_list[num_train:]
            if len(file_train)>0 and len(file_val)>0:
                train_txt = os.path.join(txt_dir, 'train.txt')
                val_txt = os.path.join(txt_dir, 'val.txt')
                with open(train_txt, 'w') as ftrain:
                    for example in file_train:
                        ftrain.write(example)
                with open(val_txt, 'w') as fval:
                    for example in file_val:
                        fval.write(example)     
            else:
                raise Exception('Not enough samples.')
        return


class ClsTrainPipe(Pipeline):
    def __init__(self, cfg):
        super(ClsTrainPipe, self).__init__(
            batch_size=cfg.dataset.loader.batch_size,
            num_threads=cfg.dataset.loader.num_workers,
            device_id=cfg.device.local_rank)

        self.eii = ClsInputIterator(cfg=cfg, is_train=True)
        self.source = ops.ExternalSource(source = self.eii, num_outputs = 2)
        
        self.decode = ops.ImageDecoderRandomCrop(device='mixed', output_type=types.RGB,
                                                 random_aspect_ratio=[0.8, 1.25],
                                                 random_area=[0.3, 1.0],
                                                 num_attempts=100)
        self.rotate = ops.Rotate(device = 'gpu', fill_value=127.5)
        self.res = ops.Resize(device='gpu',
                              resize_x=cfg.dataset.transform.image_size,
                              resize_y=cfg.dataset.transform.image_size,
                              interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device='gpu',
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(cfg.dataset.transform.image_size, cfg.dataset.transform.image_size),
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        self.angle = ops.Uniform(range=(-1*cfg.dataset.transform.max_rotate_angle, cfg.dataset.transform.max_rotate_angle))

    def define_graph(self):
        inputs, labels = self.source()
        images = self.decode(inputs)
        images = self.rotate(images, angle=self.angle())
        images = self.res(images)
        output = self.cmnp(images, mirror=self.coin())
        return [output, labels]


class ClsValPipe(Pipeline):
    def __init__(self, cfg):
        super(ClsValPipe, self).__init__(
            batch_size=cfg.dataset.loader.batch_size,
            num_threads=cfg.dataset.loader.num_workers,
            device_id=cfg.device.local_rank)

        self.eii = ClsInputIterator(cfg=cfg, is_train=False)
        self.source = ops.ExternalSource(source=self.eii, num_outputs = 2)

        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu",
                              resize_shorter=int(1.15*cfg.dataset.transform.image_size),
                              interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(cfg.dataset.transform.image_size, cfg.dataset.transform.image_size),
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        inputs, labels = self.source()
        images = self.decode(inputs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, labels]


