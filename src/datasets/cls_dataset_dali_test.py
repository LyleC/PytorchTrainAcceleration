import os
import argparse

from src.utils.utils import load_config
from src.datasets.cls_dataset_dali import *


def test_pipeline(cfg):
    # input_iter = ClsInputIterator(cfg, True)
    # for idx, (images, labels) in enumerate(input_iter):
    #     print('{}'.format(idx), end='', flush=True)
    # print()

    train_pipe = ClsTrainPipe(cfg=cfg)
    val_pipe = ClsValPipe(cfg=cfg)
    train_pipe.build()
    val_pipe.build()

    dataloaders = {}
    dataloaders['train'] = DALIGenericIterator(train_pipe, ["images", "labels"],
                                               train_pipe.eii.n,
                                            #    reader_name="Reader",
                                               fill_last_batch=True)
    dataloaders['val'] = DALIGenericIterator(val_pipe, ["images", "labels"],
                                             val_pipe.eii.n,
                                            #  reader_name="Reader",
                                             fill_last_batch=True)

    for i in range(3):
        dataloaders['val']._pipes[0].eii.shuffle(i)  
        for idx, data in enumerate(dataloaders['val']):
            if cfg.device.local_rank == 0:
                print('\r{}'.format(idx), end='', flush=True)
        if cfg.device.local_rank == 0:
            print()

        dataloaders['val'].reset()

    return


def main():
    cfg = load_config(os.path.join('./config', config_file))
    cfg.device.local_rank = args.local_rank
    cfg.device.num = torch.cuda.device_count()
    torch.cuda.set_device(cfg.device.local_rank)
    print(cfg.device.local_rank, cfg.device.num)

    test_pipeline(cfg)

    return


if __name__ == '__main__':
    config_file = 'config_imagenet_dali_ddp.json'

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int,
                        help='node rank for distributed training')
    args = parser.parse_args()

    main()
