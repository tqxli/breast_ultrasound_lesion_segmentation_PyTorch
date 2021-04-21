from __future__ import print_function
import os, time
import numpy as np

import argparse
from parse_config import ConfigParser

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import dataloader.dataloader as dataloader
import model.models as models
import model.losses as losses
import model.metrics as metrics
from utils import prepare_device
from trainer import Trainer, Trainer_extra_classification


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', dataloader)
    valid_data_loader = data_loader.split_validation()
    
    # build model architecture, then print to console
    model = config.init_obj('arch', models)
    logger.info(model)

    # prepare for GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(losses, config['loss'])
    selected_metrics = [getattr(metrics, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # Start training
    trainer = Trainer_extra_classification(model, criterion, selected_metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Training configuration')

    args.add_argument('--config', type=str, default='options/default.json', 
                      help='config path to correct json file')
    args.add_argument('--device', type=str, default=None,
                      help='indices of GPUs to enable')
    args.add_argument('--resume', type=str, default=None,
                      help='path to latest checkpoint')

    config = ConfigParser.from_args(args)

    main(config)
    