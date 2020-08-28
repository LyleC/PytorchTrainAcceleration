import os
import time
import datetime

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter


class  Logger(object):
    '''
        记录数据，计算均值,
        写入tensorboard
    '''
    def __init__(self, cfg, start_epoch=0):
        now = datetime.datetime.now().strftime('%m%d-%H%M%S')
        self.cfg = cfg
        self.local_rank = cfg.device.local_rank
        self.train_batch = 0
        self.val_batch = 0
        self.current_epoch = start_epoch
        self.world_size = dist.get_world_size()
        if self.local_rank == 0:
            self.writer = SummaryWriter(os.path.join(cfg.logger.root, now))
        self.train_loss_meter = AverageMeter()
        self.val_loss_meter = AverageMeter()
        self.timer = Timer()

    def _reduce_tensor(self, tensor: torch.Tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.world_size
        return rt

    def list_of_scalars_summary(self, prefix, tag_value_pairs, step):
        for tag, value in tag_value_pairs:
            self.writer.add_scalar(prefix + '/' + tag, value, step)
    
    def log_epoch_train_start(self):
        self.timer.reset()
        self.timer.record()

    def log_epoch_train_end(self):
        epoch_speed = self.timer.get_epoch_speed()
        self.train_loss_meter.reset()
        self.current_epoch += 1
        return epoch_speed, self.train_loss_meter.avg
    
    def log_epoch_val_start(self):
        self.timer.reset()
        self.timer.record()

    def log_epoch_eval_end(self):
        epoch_speed = self.timer.get_epoch_speed()
        self.val_loss_meter.reset()
        return epoch_speed, self.val_loss_meter.avg
        
    def log_batch_train(self, loss):
        reduced_loss = self._reduce_tensor(loss)
        self.train_loss_meter.update(reduced_loss.item())
        self.timer.record()
        if self.local_rank == 0:
            self.list_of_scalars_summary("train", [("loss", reduced_loss.item())], self.train_batch)
        self.train_batch += 1
        return reduced_loss, self.timer.get_cur_speed()

    def log_batch_eval(self, loss, precision):
        reduced_loss = self._reduce_tensor(loss)
        self.val_loss_meter.update(reduced_loss.item())
        self.top1_meter.update(precision[0])
        self.top5_meter.update(precision[1])
        self.timer.record()
        if self.local_rank == 0:
            self.list_of_scalars_summary("val", [("loss", reduced_loss.item())], self.val_batch)
        self.val_batch += 1
        return reduced_loss, self.timer.get_cur_speed()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.time_list = []
    
    def record(self):
        self.time_list.append(time.time())
    
    def get_epoch_speed(self):
        if len(self.time_list) > 1:
            epoch_speed = (len(self.time_list) -1) / (self.time_list[-1] - self.time_list[0])
        else:
            epoch_speed = 0
        return epoch_speed

    def get_cur_speed(self):
        if len(self.time_list) <= 10:
            cur_speed = self.get_epoch_speed()
        else:
            cur_speed = 10 / (self.time_list[-1] - self.time_list[-11])
        return cur_speed