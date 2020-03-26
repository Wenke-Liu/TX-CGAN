#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:37:45 2018

@author: lwk

"""
import numpy as np
from tensorflow.python.framework import dtypes

def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype=float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
               next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data

class DataSet(object):
  def __init__(self,
               data_A, data_B,
               epoch_size = None,
               dtype=dtypes.float16):
      
    """Construct a DataSet for CycleGAN training
       Return two tensors of shape [batch_size,feature_size] from 
       datasets A and B, respectively
       
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float16):
      raise TypeError('Invalid image dtype %r, expected uint8 or float16' %
                      dtype)
    
    
    self._num_examples_A = data_A.shape[0]
    self._num_examples_B = data_B.shape[0]

    self._data_A = data_A
    self._data_B = data_B
    self._epochs_completed = 0
    self._index_in_epoch = 0
    
    """If epoch_size is not specified, define as size of the smaller domain """
    self._epoch_size = min(self._num_examples_A, self._num_examples_B)
    
    if epoch_size:
        self._epoch_size = epoch_size
        

  @property
  def data(self):
    return self._data_A, self._data_B


  @property
  def num_examples(self):
    return self._num_examples_A, self._num_examples_B

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._epoch_size:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm_A = np.arange(self._num_examples_A)
      perm_B = np.arange(self._num_examples_B)
      np.random.shuffle(perm_A)
      np.random.shuffle(perm_B)
      self._data_A = self._data_A[perm_A]
      self._data_B = self._data_B[perm_B]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._epoch_size
    end = self._index_in_epoch
    return self._data_A[start:end], self._data_B[start:end]


class DataSets(object):
    def __init__(self):
        pass
    pass


def load_data(train_file_a,train_file_b,skiprows=0,transpose = False):
    
    train_dat_a=iter_loadtxt(train_file_a,delimiter='\t',skiprows = skiprows)
    train_dat_b=iter_loadtxt(train_file_b,delimiter='\t',skiprows = skiprows)
    
    if transpose:
        train_dat_a=np.transpose(train_dat_a)
        train_dat_b=np.transpose(train_dat_b)
    
    assert train_dat_a.shape[1] == train_dat_b.shape[1], \
    "No. of features different between dataset A and B!"
    
    print('-------start loading data--------')
    print('Feature size: {}'.format(train_dat_a.shape[1]))
    print('Dataset A n: {}'.format(train_dat_a.shape[0]))
    print('Dataset B n: {}'.format(train_dat_b.shape[0]))
    
    data_set = DataSet(train_dat_a,train_dat_b)
    print('----------data loaded -----------')
        
    return data_set


def load_multiple_datasets(train_file_a_list,
                           train_file_b_list,
                           skiprows=0, transpose = False):
    
    
    
    pass
