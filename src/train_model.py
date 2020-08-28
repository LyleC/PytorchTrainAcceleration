import os
import logging
import argparse

from src.utils.utils import load_config
from src.train.trainer import Trainer


def main():
    # load configs
    cfg = load_config(os.path.join('./config', config_file))
    cfg.device.local_rank = args.local_rank
    
    # start training
    trainer = Trainer(cfg)
    trainer.train()
    return


if __name__ == "__main__":
    logging.basicConfig(level = logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m-%d %H:%M:%S')

    config_file = 'config_imagenet_dali_ddp.json'

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int,
                        help='node rank for distributed training')
    args = parser.parse_args()

    main()