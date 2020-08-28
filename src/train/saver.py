import os
import logging
import datetime
from glob import glob

import torch


class Saver(object):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.local_rank = cfg.device.local_rank
        self.device = device
        self.cp_dir = cfg.saver.cp_root
        if self.local_rank == 0:
            if not os.path.isdir(self.cp_dir):
                os.mkdir(self.cp_dir)
        self.epoch_list = self._update_status()

    def _update_status(self):
        '''
            获取checkpoint文件夹内的文件列表
            将提取出的epoch参数按照从大到小的顺序排列
        '''
        cp_list = os.listdir(self.cp_dir)
        epoch_list = []
        for cp in cp_list:
            name, ext = os.path.splitext(cp)
            if ext == '.pth':
                idx = int(name.split('_')[1])
                epoch_list.append(idx)
        epoch_list.sort(reverse = True)
        return epoch_list

    def _remove_outdated_checkpoint(self):
        '''
            按从先到后的顺序删除超过保存上限的checkpoint文件
        '''
        self.epoch_list = self._update_status()
        while(len(self.epoch_list) > self.cfg.saver.max_keep):
            cp_paths = glob(os.path.join(self.cp_dir, 'checkpoint_{}*.pth'.format(self.epoch_list[-1])))
            os.remove(cp_paths[0])
            self.epoch_list.pop()
        return

    def save_checkpoint(self, epoch, model, loss=0):
        '''
            保存当前checkpoint，记录epoch数和当前loss值
            保存后检查保存数量是否超过上限
        '''
        if self.local_rank == 0:
            if epoch % self.cfg.saver.save_interval == 0:
                cp_path = os.path.join(self.cp_dir, 'checkpoint_{}_{:.3f}.pth'.format(epoch, loss))
                if hasattr(model, 'module'):
                    model_dict = model.module.state_dict()
                else:
                    model_dict = model.state_dict()
                checkpoint = {
                    'epoch': epoch,
                    'model': model_dict,
                }
                torch.save(checkpoint, cp_path)
                print()
                logging.info('Save checkpoint : {}'.format(cp_path)) 
                self._remove_outdated_checkpoint()
        return
 
    def _load_state_dict(self, cp, model):
        checkpoint = torch.load(cp, map_location=torch.device(self.device))
        model.load_state_dict(checkpoint['model'])
        return


    def load_checkpoint(self, epoch, model):
        '''
            加载指定的checkpoint
            如果指定的checkpoint存在，
                则正常加载，并删除已经存在的epoch-index超过当前epoch的checkpoint
            如果指定的checkpoint不存在，
                则尝试加载最新的checkpoint
        '''
        if not isinstance(epoch, int):
            raise Exception('Epoch of checkpoint should be an Int')

        cp_paths = glob(os.path.join(self.cp_dir, 'checkpoint_{}*.pth'.format(epoch)))
        self.epoch_list = self._update_status()

        if len(cp_paths) == 1:
            self._load_state_dict(cp_paths[0], model)
            logging.info('Rank {}, Load checkpoint : {}'.format(self.local_rank, cp_paths[0]))
            # delete checkpoint later than epoch
            while(self.epoch_list[0] > epoch):
                rm_paths = glob(os.path.join(self.cp_dir, 'checkpoint_{}*.pth'.format(self.epoch_list[0])))
                os.remove(rm_paths[0])
                self.epoch_list.pop(0)
            return epoch + 1

        elif len(cp_paths) == 0:
            if len(self.epoch_list) > 0:
                cp_paths = glob(os.path.join(self.cp_dir, 'checkpoint_{}*.pth'.format(self.epoch_list[0])))
                self._load_state_dict(cp_paths[0], model)
                logging.info('Rank {}, Load the latest checkpoint : {}'.format(self.local_rank, cp_paths[0]))
                return self.epoch_list[0] + 1
            else:
                return 0
        else:
            raise Exception('More than one checkpoint have same epoch-index')
    
    def save_model_dict(self, model):
        '''
            保存最终训练完成的模型
        '''
        if self.local_rank == 0:
            gen_time = datetime.datetime.now().strftime('%m%d%H%M')
            save_name = self.cfg.model.name + '_' + self.cfg.dataset.name + '_' + gen_time + '.pth'
            save_path = os.path.join(self.cfg.saver.output_root, save_name)
            if hasattr(model, 'module'):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            logging.info('Save output model : {}'.format(save_path)) 
        return