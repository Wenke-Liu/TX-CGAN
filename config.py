#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 13:05:26 2018

@author: lwk
"""
import numpy as np

HYPERPARAMS = {"batch_size": 8,
        "learning_rate": 1E-5,
        "dropout": 0.8,
        "pool_size": 64,
        "print_step": 1000,
        "lr_decay": 0.2
        }

ARCHITECTURE = {
        'feature_size': 17859,
        'G_a2b':[1000,17859],
        'G_b2a':[1000,17859],
        'D_a':[100],
        'D_b':[100]
        }


MAX_ITER = 100#2**16
MAX_EPOCHS = 10

LOG_DIR = '190315/log'
METAGRAPH_DIR = '190315/out'

TRAIN_FILES = {
        'A': '/media/lwk/data/pancancer/data/breast_n_centered_100.txt',
        'B': '/media/lwk/data/pancancer/data/breast_t_centered_100.txt'
        }

MODEL_NAME = 'test'