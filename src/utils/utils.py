import json
import logging
import pprint

import torch
import torchvision
from easydict import EasyDict


def load_config(json_file):
    ''' 
        load config files
    '''
    with open(json_file, 'r', encoding='utf-8') as f:
        config = EasyDict(json.load(f))
    return config


class DdpPrinter:
    def __init__(self, local_rank):
        self.local_rank = local_rank
        if self.local_rank == 0:
            logging.info("PyTorch Version: {}".format(torch.__version__))
            logging.info("Torchvision Version: {}".format(torchvision.__version__))

    def __call__(self, message):
        if self.local_rank == 0:
            if isinstance(message, dict):
                pprint.pprint(message, indent=2, width=80)
            elif isinstance(message, str):
                logging.info(message)
