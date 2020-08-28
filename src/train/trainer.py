import os
import math
import logging

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel

from src.datasets.cls_dataset_dali import init_cls_datasets_dali
from src.losses.cls_loss import ClsLoss
from src.train.saver import Saver
from src.train.logger import Logger
from src.train.evaluator import Evaluator
from src.utils.utils import DdpPrinter

from src.models.shufflenetv2 import shufflenet_v2_x1_0
from src.models.mobilenet import MobileNetV2


class Trainer:
    def __init__(self, config):
        self.cfg = config
        # init printer
        self.printer = DdpPrinter(self.cfg['device']['local_rank'])
        self.printer(self.cfg)
        # init device
        self.device = self._init_device()
        # init datasets
        self.dataloaders = self._init_datasets()
        # init checkpoint saver
        self.saver = Saver(self.cfg, self.device)
        # init model
        self.model = self._init_model()
        # init optimizer
        self.optimizer = self._init_optimizer(self.model.parameters())
        # init loss function
        self.criterion = self._init_loss()
        # set amp
        self._set_amp()
        # load model parameters
        self.start_epoch = self._load_model_params()
        # set parallel
        self._set_parallel()
        # init logger
        self.logger = Logger(self.cfg, self.start_epoch)
        # init evaluator
        self.evaluator = Evaluator()

    def _init_device(self):
        if torch.cuda.is_available():
            self.cfg.device.num = torch.cuda.device_count()
            torch.distributed.init_process_group(backend="nccl", init_method='env://')
            self.cfg.device.world_size = dist.get_world_size()
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_device(self.cfg.device.local_rank)
            device = torch.device('cuda:{}'.format(self.cfg.device.local_rank))
            assert torch.backends.cudnn.enabled
            logging.info('Training device : {}, world num : {}'.format(
                            device, self.cfg.device.world_size))
        else:
            self.printer('No cuda device found.')
            raise NotImplementedError
        return device

    def _init_datasets(self):
        return init_cls_datasets_dali(self.cfg)

    def _init_model(self):
        '''
        网络结构定义来自于torchvision库，参见：
        https://github.com/pytorch/vision/tree/master/torchvision/models
        '''
        name = self.cfg.model.name
        num_classes = self.cfg.dataset.num_classes
        if name == 'shufflenet_v2':
            model = shufflenet_v2_x1_0()
        elif name == 'mobilenet_v2':
            model = MobileNetV2(num_classes=num_classes)
        else:
            raise NotImplementedError
        self.printer('Build model : {}'.format(name))
        return model.to(self.device)

    def _load_model_params(self, verbose=True):
        epoch = 0
        model_file = self.cfg.model.model_file
        # 首先看是否指定加载某一个checkpoint
        if 'checkpoint' in model_file.keys():
            epoch = self.saver.load_checkpoint(model_file['checkpoint'], self.model, self.optimizer)
        # 如果未指定，但存在checkpoint，则加载最新的
        elif len(self.saver.epoch_list) > 0:
            epoch = self.saver.load_checkpoint(self.saver.epoch_list[0], self.model, self.optimizer)
        # 如无checkpoint，则看是否有指定的预训练模型
        elif 'pretrain' in model_file.keys():
            model_dict = self.model.state_dict()
            pretrained_dict = torch.load(model_file['pretrain'], map_location=torch.device(self.device))
            loaded_dict = {k: v for k, v in pretrained_dict.items() 
                               if k in model_dict and k.split('.')[0] not in model_file['exclude']}
            model_dict.update(loaded_dict)
            self.model.load_state_dict(model_dict)
            self.printer('Load pretrained model : {}'.format(model_file['pretrain']))
            self.printer('Pretrained model has {} keys, {} loaded.'.format(len(pretrained_dict.keys()), len(loaded_dict.keys())))
            if verbose and self.cfg['device']['local_rank']==0:
                params_file = os.path.join(self.cfg['dataset']['txt_root'], 'loaded_params.txt')
                self.printer('Write params loaded in file : {}'.format(params_file))
                with open(params_file, 'w') as f:
                    for k, v in loaded_dict.items():
                        f.write('{} : {}\n'.format(k, v.size()))
        else:
            self.printer('Start training from scratch')
        # 可以设置冻结部分参数
        self.model = self._freeze_params(self.model, self.cfg.model.freeze_branch, True)
        return epoch

    def _freeze_params(self, model, freeze_branch, verbose=False):
        freeze_branch = freeze_branch.split(",")
        for name, param in model.named_parameters():
            if name.split('.')[0] in freeze_branch:
                param.requires_grad = False
        if verbose and self.cfg.device.local_rank==0:
            params_file = os.path.join(self.cfg.dataset.txt_root, 'learnable_params.txt')
            logging.info('Write params to be updated in file : {}'.format(params_file))
            with open(params_file, 'w') as f:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        f.write('{} : {}\n'.format(name, param.size()))
        return model

    def _init_loss(self): 
        return ClsLoss()

    def _init_optimizer(self, params):
        op_name = self.cfg.optimizer.name
        lr = self.cfg.optimizer.lr
        mom = self.cfg.optimizer.momentum
        wd = self.cfg.optimizer.weight_decay
        if op_name == 'SGD':
            optimizer = optim.SGD(params, lr, momentum=mom, weight_decay=wd)
        elif op_name == 'Adam':
            optimizer = optim.Adam(params, lr)
        elif op_name == 'RMSprop':
            optimizer = optim.RMSprop(params, lr, momentum=mom, weight_decay=wd)
        elif op_name == 'Adadelta':
            optimizer = optim.Adadelta(params, lr)
        else:
            raise NotImplementedError
        self.printer('Using optimizer: {}, initial lr : {}'.format(op_name, lr))
        return optimizer

    def _set_amp(self):
        if self.cfg.device.use_amp:
            self.scaler = GradScaler()

    def _set_parallel(self):
        if self.cfg.device.num > 1:
            self.printer('Using {} GPUs to train'.format(self.cfg.device.num))
            self.model = DistributedDataParallel(self.model, device_ids=[self.cfg.device.local_rank],
                                                 output_device=self.cfg.device.local_rank)

    def train(self):
        if self.cfg.train.eval_only:
            self.printer('Start evaluating...')
            self._epoch_eval(self.start_epoch)
            self.printer('Stop evaluating...')
            return

        self.printer('Start training...')
        for epoch in range(self.start_epoch, self.cfg.train.epoch):
            tloss, tspeed = self._epoch_train(epoch)
            if epoch % self.cfg.train.eval_interval == 0:
                eloss, espeed = self._epoch_eval(epoch)
                self.saver.save_checkpoint(epoch, self.model, (tloss, eloss))
                self.printer('Epoch: [{}], Train-loss: {:.3f}, aver-speed: {:.3f} iters/s; Eval=loss: {:.3f}, aver-speed: {:.3f} iters/s\n'.format(
                        epoch, tloss, tspeed, eloss, espeed))
            else:
                self.saver.save_checkpoint(epoch, self.model, tloss)
                logging.info('Epoch: [{}], Train-loss: {:.3f}, aver-speed: {:.3f} iters/s\n'.format(
                        epoch, tloss, tspeed))
        self.printer('Stop training...')
        if self.cfg.train.output:
            self.saver.save_model_dict(self.model)
        return
    
    def _epoch_train(self, epoch):
        self.model.train()
        total_iter = math.ceil(self.cfg.dataset.train_num / self.cfg.dataset.loader.batch_size)
        self.logger.log_epoch_train_start()
        # 在每个epoch开始时，手动shuffle并切割数据集
        self.dataloaders['train']._pipes[0].eii.shuffle(epoch)  
        for idx, data in enumerate(self.dataloaders['train']):
            cur_loss, cur_speed = self._batch_train(data, idx)
            if self.cfg.device.local_rank == 0:
                print('\rTrain Epoch: [{}], Step: [{}]/[{}] loss = {:.4f} Speed = {:.4f} iters/s'.format(
                        epoch, idx, total_iter, cur_loss, cur_speed), end='', flush=True)
        else:
            if self.cfg.device.local_rank == 0: print()
        avg_loss, avg_speed = self.logger.log_epoch_train_end()
        self.saver.save_checkpoint(epoch, self.model, avg_loss)
        self._updata_lr(self.cfg.optimizer, epoch)
        self.dataloaders['train'].reset()
        return avg_loss, avg_speed

    def _epoch_eval(self, epoch):
        self.model.eval()
        total_iter = math.ceil(self.cfg.dataset.val_num / self.cfg.dataset.loader.batch_size)
        self.logger.log_epoch_val_start()
        # 在每个epoch开始时，手动shuffle并切割数据集
        self.dataloaders['val']._pipes[0].eii.shuffle(epoch)  
        for idx, data in enumerate(self.dataloaders['val']):
            cur_loss, cur_speed = self._batch_eval(data, idx)
            if self.cfg.device.local_rank == 0:
                print('\r  Val Epoch: [{}], Step: [{}]/[{}] loss = {:.4f} Speed = {:.4f} iters/s'.format(
                        epoch, idx, total_iter, cur_loss, cur_speed), end='', flush=True)
        else:
            if self.cfg.device.local_rank == 0: print()
        avg_loss, avg_speed, _ = self.logger.log_epoch_eval_end()
        self.dataloaders['val'].reset()
        return avg_loss, avg_speed

    def _batch_train(self, data, idx):
        images = data[0]["images"].to(self.device)
        targets = data[0]["labels"].squeeze().long().to(self.device)
        self.optimizer.zero_grad()
        if self.cfg.device.use_amp:
            with autocast():
                preds = self.model(images)
                losses = self.criterion(preds, targets)
            total_loss = losses
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            preds = self.model(images)
            losses = self.criterion(preds, targets)
            total_loss = losses
            total_loss.backward()
            self.optimizer.step()
        rloss, speed = self.logger.log_batch_train(losses)
        return rloss, speed

    def _batch_eval(self, data, idx):
        images = data[0]["images"].to(self.device)
        targets = data[0]["labels"].squeeze().long().to(self.device)
        with torch.no_grad(): 
            if self.cfg.device.use_amp:
                with autocast():
                    preds = self.model(images)
                    losses = self.criterion(preds, targets)
            else:
                preds = self.model(images)
                losses = self.criterion(preds, targets)
            perf = self.evaluator(preds, targets, (1, 5))
            rloss, speed = self.logger.log_batch_eval(losses, [0,0])
        return rloss, speed

    def _updata_lr(self, cfg, epoch):
        index = int(epoch / cfg['lr_step'])
        lr_new = cfg['lr'] * pow(cfg['lr_gamma'], index)
        self.optimizer.lr = lr_new
        return
