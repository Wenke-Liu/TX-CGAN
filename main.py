#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:01:39 2018

@author: lwk
"""
import os
import sys
import numpy as np

#import numpy as np
import tensorflow as tf


import cgan
import dataset

CONFIG_FILE_NAME = 'config'
config = __import__(CONFIG_FILE_NAME)


ARCHITECTURE = config.ARCHITECTURE
HYPERPARAMS = config.HYPERPARAMS

MAX_ITER = config.MAX_ITER
MAX_EPOCHS = config.MAX_EPOCHS

LOG_DIR = config.LOG_DIR
METAGRAPH_DIR = config.METAGRAPH_DIR

TRAIN_FILES = config.TRAIN_FILES

MODEL_NAME = config.MODEL_NAME

def main(to_reload=None):
    
    cgan_train = dataset.load_data(TRAIN_FILES['A'], TRAIN_FILES['B'],
                                   skiprows =1, transpose = True)
    
    

    if to_reload: # restore, get fake and cycle outputs
        
        m=cgan.CGAN(meta_graph=to_reload,config_name=CONFIG_FILE_NAME)
        print("Loaded!", flush=True)
        
        ### use training data as testing data
        test_a, test_b = cgan_train.data
        
        fake_a = m.generate_fake_a(test_b)
        print('generated fake a dimension: {}'.format(fake_a.shape))
        
        np.savetxt(fname=MODEL_NAME +'_' + 'fake_a.txt',
           X=fake_a.T,
           delimiter='\t',
           comments='')
        
        fake_b=m.generate_fake_b(test_a)
        print('generated fake b dimension: {}'.format(fake_b.shape))
        
        np.savetxt(fname = MODEL_NAME +'_' + 'fake_b.txt',
           X= fake_b.T,
           delimiter='\t',
           comments='')
        
        cycle_a=m.generate_cycle_a(test_a)
        print('generated cycle a dimension: {}'.format(cycle_a.shape))
        
        np.savetxt(fname=MODEL_NAME +'_' + 'cycle_a.txt',
           X=cycle_a.T,
           delimiter='\t',
           comments='')
        
        cycle_b=m.generate_cycle_b(test_b)
        print('generated cycle b dimension: {}'.format(cycle_b.shape))
        
        np.savetxt(fname=MODEL_NAME +'_' + 'cycle_b.txt',
           X=cycle_b.T,
           delimiter='\t',
           comments='')
        
        
        a2b_final_weights=m.a2b_final_w()
        print('a2b generator output weights size: {}'.format(a2b_final_weights.shape))
        
        np.savetxt(fname=MODEL_NAME +'_' + 'a2b_output_weights.txt',
           X=a2b_final_weights.T,
           delimiter='\t',
           comments='')
        
        b2a_final_weights=m.b2a_final_w()
        print('b2a generator output weights size: {}'.format(b2a_final_weights.shape))
        
        np.savetxt(fname=MODEL_NAME +'_'+ 'b2a_output_weights.txt',
           X=b2a_final_weights.T,
           delimiter='\t',
           comments='')
        
        

    else: # train
        """to try cont'd training, load data from previously saved meta graph"""
        
        m = cgan.CGAN(ARCHITECTURE,HYPERPARAMS,
                      config_name=CONFIG_FILE_NAME+'_'+MODEL_NAME,
                      log_dir=LOG_DIR)
        
        m.train(cgan_train, max_iter=MAX_ITER, max_epochs=MAX_EPOCHS, 
                verbose=True, save=True, outdir=METAGRAPH_DIR)
        print("Trained!",flush=True)
       
   

if __name__ == "__main__":
    
    tf.reset_default_graph()

    for DIR in (LOG_DIR, METAGRAPH_DIR):
        try:
            os.mkdir(DIR)
        except(FileExistsError):
            pass

    try:
        to_reload = sys.argv[1]
        main(to_reload=to_reload)
    except(IndexError):
        main()