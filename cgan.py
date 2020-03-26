#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:28:58 2018

@author: lwk
"""

from datetime import datetime
import os
import sys
import random
import numpy as np
import tensorflow as tf
import layers
import losses



class CGAN():
    
    """
    CycleGAN model for normal/cancer gene expression profiles
       Fully-connected layers as generator and discriminator
       
    see: 
       Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent 
       Adversarial Networks", 
       in IEEE International Conference on Computer Vision (ICCV), 2017. 
       arxiv 1703.10593, 2017. 
       https://arxiv.org/abs/1703.10593
       
       Ghahramani et al., "Generative adversarial networks simulate 
       gene expression and predict perturbations in single cells",
       bioRxiv 2018, 
       https://www.biorxiv.org/content/early/2018/07/30/262501
    
    
    """
    DEFAULTS = {
        "batch_size": 1,
        "learning_rate": 1E-3,
        "dropout": 1.,
        "pool_size": 50,
        "lambda_a": 1.,
        "lambda_b": 1.,
        "logger_step": 10,
        "print_step": 10,
        "lr_decay":0.1
    }
    
    RESTORE_KEY = "to_restore"

    def __init__(self, 
                 architecture={},
                 d_hyperparams={},
                 meta_graph=None,
                 config_name = None,
                 save_graph_def=True, 
                 log_dir="./log"):
        
        """(Re)build a CycleGAN model model with given:

            * generator and discriminator functions defined in layers.py
              
            * hyperparameters (optional dictionary of updates to `DEFAULTS`)
            
            
        """
        
        self.__dict__.update(CGAN.DEFAULTS, **d_hyperparams)
        self.architecture = architecture
        self.sesh = tf.Session()
        self.config_name = config_name
        
        self.num_fakes = 0
        
        
        if not meta_graph: # new model
            self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")
            assert all(key in self.architecture for \
                       key in ['feature_size','G_a2b','D_b','G_b2a','D_a']), \
                "Architecture must specify feature_size, Generator and Discriminator latent size"
            
            
            self.feature_size = self.architecture['feature_size']
            self.fake_pool_a = np.zeros((self.pool_size,self.feature_size))
            self.fake_pool_b = np.zeros((self.pool_size,self.feature_size))
            
            # build graph
            handles = self._buildGraph()
            for handle in handles:
                tf.add_to_collection(CGAN.RESTORE_KEY, handle)

            
            self.sesh.run(tf.global_variables_initializer())

        else: # restore saved model
            model_datetime, model_name = os.path.basename(meta_graph).split("_cgan_")
            model_name = model_name.split('-')[0]
            self.datetime = "{}_reloaded".format(model_datetime)
            
            assert (config_name is None)|(config_name==model_name), \
                "Double check configuration!"
            
            # rebuild graph
            meta_graph = os.path.abspath(meta_graph)
            tf.train.import_meta_graph(meta_graph + ".meta").restore(
                self.sesh, meta_graph)
            handles = self.sesh.graph.get_collection(CGAN.RESTORE_KEY)

        # unpack handles for tensor ops to feed or fetch
        (self.real_a, self.real_b, 
         self.dropout_, self.train_status,
         self.fake_a, self.fake_b,
         self.cycle_a, self.cycle_b,
         self.fake_pool_a_sample, self.fake_pool_b_sample, 
         self.g_loss_a2b, self.d_loss_b,self.g_loss_b2a, self.d_loss_a,
         self.g_a2b_trainer, self.d_b_trainer, self.g_b2a_trainer, self.d_a_trainer,self.curr_lr_,      
         self.global_step, self.merged_summary) = handles
         

        if save_graph_def: # tensorboard
            try:
                os.mkdir(log_dir)
             
                
            except(FileExistsError):
                pass
            self.logger = tf.summary.FileWriter(log_dir, self.sesh.graph)
            
    
    def __del__(self):
        print('CGAN object destructed.')
        
    
    @property
    def step(self):
        """Train step"""
        return self.global_step.eval(session=self.sesh)
    
    
    def past_fake_pool(self, num_fakes, fake, fake_pool):
        ''' This function saves the generated image to corresponding pool of images.
        In starting. It keeps on filling the pool till it is full and then randomly selects an
        already stored image and replace it with new one.
        
        fake is a tensor of size [batch_size,feature_size]
        fake_pool is a tensor of size [num_fakes, feature_size] until it is filled
        to be of size [pool_size, feature_size]
        '''
        
        if(num_fakes < self.pool_size):
            fake_pool[(num_fakes-self.batch_size):num_fakes,]=fake
            return fake
        else :
            p = random.random()
            if p > 0.5:
                random_id = np.random.choice(self.pool_size,self.batch_size,
                                             replace = False)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                
                return temp
            else :
                return fake
    

    def _buildGraph(self):
        
        # placeholder variables for feeding
        
        real_a = tf.placeholder(tf.float32, shape=[None, # enables variable batch size
                                                 self.feature_size], name="A")
    
        real_b = tf.placeholder(tf.float32, shape=[None, # same feature size for both domains
                                                 self.feature_size], name="B")
        
        
        dropout = tf.placeholder_with_default(1., shape=[], name="dropout")
        
        train_status = tf.placeholder_with_default(True, shape=[], name="train_status")
        
        # random intial values for fake pools
        
        fake_pool_a_sample = tf.placeholder(tf.float32, shape=[None, # enables variable batch size
                                                 self.feature_size], name="fake_pool_a_sample")
        fake_pool_b_sample = tf.placeholder(tf.float32, shape=[None, # same feature size for both domains
                                                 self.feature_size], name="fake_pool_b_sample")
        

        # construct generator and discriminator functions as class instances
        
        G_a2b = layers.generator(name = 'G_a2b',
                                 architecture = self.architecture['G_a2b'],
                                 activation = tf.nn.leaky_relu,
                                 output_activation = tf.nn.leaky_relu,
                                 dropout = dropout) 
        # linear discriminator
        D_a = layers.discriminator(name = 'D_a',
                                   architecture = self.architecture['D_a'],
                                   activation = None,
                                   dropout = dropout)
        
        G_b2a = layers.generator(name = 'G_b2a',
                                 architecture = self.architecture['G_b2a'],
                                 activation = tf.nn.leaky_relu,
                                 output_activation = tf.nn.leaky_relu,
                                 dropout = dropout)
        # linear discriminator
        D_b = layers.discriminator(name = 'D_b',
                                   architecture = self.architecture['D_b'],
                                   activation = None,
                                   dropout = dropout)
        
        
        # print generator and discriminator architecture
        for f in [G_a2b,D_a,G_b2a,D_b]:
            f.print_architecture()
        
        
        # generate fake and cycle profiles
        fake_a = G_b2a(real_b, is_train = train_status)
        fake_b = G_a2b(real_a, is_train = train_status)
        
        cycle_a = G_b2a(fake_b, is_train = train_status)
        cycle_b = G_a2b(fake_a, is_train = train_status)
        
        
        # construct recent fake profile pool
        """
        See Zhu et al., "Training Details": 
        to reduce model oscillation [15], 
        we follow Shrivastava et al.’s strategy [46] and update the 
        discriminators using a history of generated images rather than 
        the ones produced by the latest generators. We keep an image
        buffer that stores the 50 previously created images.
        
        """
        
                
        # get discriminator predictions
        
        prob_real_a_is_a=D_a(real_a, is_train=train_status)
        prob_fake_a_is_a=D_a(fake_a, is_train=train_status)
        prob_real_b_is_b=D_b(real_b, is_train=train_status)
        prob_fake_b_is_b=D_b(fake_b, is_train=train_status)
        
        prob_fake_pool_a_is_a = D_a(fake_pool_a_sample, is_train = train_status)
        prob_fake_pool_b_is_b = D_b(fake_pool_b_sample, is_train = train_status)
        
        
        # calculate losses
        
        ## Generator loss
        lsgan_loss_a = losses.lsgan_loss_generator(prob_fake_a_is_a)
        lsgan_loss_b = losses.lsgan_loss_generator(prob_fake_b_is_b)
        
        ## Cycle-consistency loss
        ## Weighted by quality of the generated image
        cycle_consistency_loss_a =\
            self.lambda_a * losses.cycle_consistency_loss(
                real_images=real_a, generated_images=cycle_a) *\
                    tf.maximum(0.,tf.reduce_mean(prob_fake_b_is_b))
                    
            
        cycle_consistency_loss_b =\
            self.lambda_b * losses.cycle_consistency_loss(
                real_images=real_b, generated_images=cycle_b) *\
                    tf.maximum(0.,tf.reduce_mean(prob_fake_a_is_a))
                    
            
            
        g_loss_a2b =\
            cycle_consistency_loss_a + cycle_consistency_loss_b + lsgan_loss_b*2
        g_loss_b2a =\
            cycle_consistency_loss_b + cycle_consistency_loss_a + lsgan_loss_a*2
            
        ## Discriminator loss
        d_loss_a = losses.lsgan_loss_discriminator(
            prob_real_is_real=prob_real_a_is_a,
            prob_fake_is_real=prob_fake_pool_a_is_a,
        )
        d_loss_b = losses.lsgan_loss_discriminator(
            prob_real_is_real=prob_real_b_is_b,
            prob_fake_is_real=prob_fake_pool_b_is_b,
        )


        # construct optimizers
        
        global_step = tf.Variable(0, trainable=False)
        
        # make learning rate adjustable
        curr_lr_ = tf.Variable(self.learning_rate, trainable=False)
        
        optimizer = tf.train.AdamOptimizer(curr_lr_, beta1=0.5)

        model_vars = tf.trainable_variables()

        
        g_a2b_vars = [var for var in model_vars if 'G_a2b' in var.name]
        d_b_vars = [var for var in model_vars if 'D_b' in var.name]
        g_b2a_vars = [var for var in model_vars if 'G_b2a' in var.name]
        d_a_vars = [var for var in model_vars if 'D_a' in var.name]

        g_a2b_trainer = optimizer.minimize(g_loss_a2b, var_list=g_a2b_vars)
        d_b_trainer = optimizer.minimize(d_loss_b, var_list=d_b_vars)
        g_b2a_trainer = optimizer.minimize(g_loss_b2a, var_list=g_b2a_vars)
        d_a_trainer = optimizer.minimize(d_loss_a, var_list=d_a_vars, global_step=global_step)

        for var in model_vars:
            print(var.name)

        # Summary variables for tensorboard
        tf.summary.scalar("D_A(x)", tf.reduce_mean(prob_fake_a_is_a))
        tf.summary.scalar("D_B(x)", tf.reduce_mean(prob_fake_b_is_b))
        tf.summary.scalar("G_A2B_loss", g_loss_a2b)
        tf.summary.scalar("G_B2A_loss", g_loss_b2a)
        tf.summary.scalar("D_A_loss", d_loss_a)
        tf.summary.scalar("D_B_loss", d_loss_b)
       
        
        
        
        # Merge summary
        
        merged_summary = tf.summary.merge_all()
               
        # return handles
        return (real_a, real_b, 
                dropout,train_status,
                fake_a, fake_b,
                cycle_a, cycle_b,
                fake_pool_a_sample, fake_pool_b_sample,
                g_loss_a2b, d_loss_b,g_loss_b2a, d_loss_a,
                g_a2b_trainer, d_b_trainer, g_b2a_trainer, d_a_trainer, curr_lr_,
                global_step, merged_summary)

    def save_fake(self,step,save_dir):
        """Save fake profiles during training"""
        
        pass
    
    def generate_fake_a(self, new_b):
        """generate new fake a profiles"""
        feed_dict = {self.real_b: new_b, self.train_status: False}
        return self.sesh.run(self.fake_a, feed_dict=feed_dict)
    
    def generate_fake_b(self, new_a):
        """generate new fake b profiles"""
        feed_dict = {self.real_a: new_a, self.train_status: False}
        return self.sesh.run(self.fake_b, feed_dict=feed_dict)
    
    def generate_cycle_a(self, new_a):
        feed_dict = {self.real_a: new_a, self.train_status: False}
        return self.sesh.run(self.cycle_a, feed_dict=feed_dict)
    
    def generate_cycle_b(self, new_b):
        feed_dict = {self.real_b: new_b, self.train_status: False}
        return self.sesh.run(self.cycle_b, feed_dict=feed_dict)
    
    def a2b_final_w(self):
        return self.sesh.run('G_a2b/output/fully_connected/weights:0')
        
    
    def b2a_final_w(self):
        return self.sesh.run('G_b2a/output/fully_connected/weights:0')
        
    

    def train(self, X, max_iter=np.inf, max_epochs=np.inf,
              verbose=True, save=True, save_fake=False,outdir="./out"):
        
        if save:
            saver = tf.train.Saver(tf.global_variables(),max_to_keep=None)

        try:
            
            now = datetime.now().isoformat()
            print("------- Training begin: {} -------\n".format(now),flush=True)
                        

            while True:
                real_a, real_b = X.next_batch(self.batch_size)
                
                # adjust learning rate
                curr_epoch = X.epochs_completed
                if curr_epoch < max_epochs*self.lr_decay:
                    curr_lr = self.learning_rate
                else:
                    curr_lr = self.learning_rate - \
                    self.learning_rate*(curr_epoch/max_epochs - self.lr_decay)
                
                # Optimizing the G_A2B network
                
                _, g_loss_a2b,fake_b = self.sesh.run([self.g_a2b_trainer,self.g_loss_a2b,
                                                      self.fake_b],
                                                     feed_dict={self.real_a: real_a, self.real_b: real_b, 
                                                                self.dropout_: self.dropout,
                                                                self.curr_lr_: curr_lr})
                
                
                # get fake_b from fake image pool
                if self.num_fakes < self.pool_size:   
                    self.num_fakes += self.batch_size
                
                fake_pool_b_sample = self.past_fake_pool(self.num_fakes,
                                                         fake_b,
                                                         self.fake_pool_b)
                
                
                # Optimizing the D_B network
                
                _,d_loss_b = self.sesh.run([self.d_b_trainer,self.d_loss_b],
                                           feed_dict={self.real_a: real_a, self.real_b: real_b, 
                                                      self.dropout_: self.dropout,
                                                      self.curr_lr_: curr_lr,
                                                      self.fake_pool_b_sample: fake_pool_b_sample})
                
                # Optimizing the G_B2A network
                
                _, g_loss_b2a,fake_a = self.sesh.run([self.g_b2a_trainer,self.g_loss_b2a,self.fake_a],
                                                     feed_dict={self.real_a: real_a, self.real_b: real_b, 
                                                                self.dropout_: self.dropout,
                                                                self.curr_lr_: curr_lr})
                
                
                # get fake_a from fake image pool
                
                fake_pool_a_sample = self.past_fake_pool(self.num_fakes,
                                                         fake_a,
                                                         self.fake_pool_a)
                
                # Optimizing the D_A network, get global_step and summary
                
                _,i,d_loss_a,merged_summary = self.sesh.run([self.d_a_trainer,
                                                             self.global_step,
                                                             self.d_loss_a,
                                                             self.merged_summary],
                
                                                            feed_dict={self.real_a: real_a, self.real_b: real_b, 
                                                                       self.dropout_: self.dropout,
                                                                       self.curr_lr_: curr_lr,
                                                                       self.fake_pool_a_sample:fake_pool_a_sample,
                                                                       self.fake_pool_b_sample:fake_pool_b_sample})
                
                
                if i%self.logger_step == 0:
                    self.logger.add_summary(merged_summary,i)
                    
                if i%self.print_step == 0 and verbose:
                    
                    print("round {} --> Generator A2B loss:{}".format(i,g_loss_a2b), flush=True)
                    print("round {} --> Discriminator B loss:{}".format(i,d_loss_b), flush=True)
                    print("round {} --> Generator B2A loss:{}".format(i,g_loss_b2a), flush=True)
                    print("round {} --> Discriminator A loss:{}".format(i,d_loss_a), flush=True)
                    print("epoch {} --> Learning rate: {}\n".format(curr_epoch,curr_lr), flush=True)
                    #print(self.fake_pool_a[0:5,0:5])
                    #print(self.fake_pool_b[0:5,0:5])
                    
                if i%100 == 0 and save_fake:
                    self.save_fake(i,outdir)
                   
                if i >= max_iter or X.epochs_completed >= max_epochs:
                    print("final step (@ step {} = epoch {})".format(
                        i, X.epochs_completed),flush=True)
                    
                    now = datetime.now().isoformat()[11:]
                    print("------- Training end: {} -------\n".format(now),flush=True)
                    
                    print("num_fakes: {}".format(self.num_fakes))
                    

                    if save:
                        outfile = os.path.join(os.path.abspath(outdir), "{}_cgan_{}".format(
                            self.datetime,self.config_name))
                        
                        saver.save(self.sesh, outfile, global_step=self.step)
                    try:
                        self.logger.flush()
                        self.logger.close()
                       
                    except(AttributeError): # not logging
                        continue
                    break

        except(KeyboardInterrupt):
            print("final avg cost (@ step {} = epoch {})".format(
                i, X.epochs_completed),flush=True)
            
            now = datetime.now().isoformat()[11:]
            print("------- Training end: {} -------\n".format(now),flush=True)
            
            if save:
                outfile = os.path.join(os.path.abspath(outdir), "{}_cgan_{}".format(
                            self.datetime,self.config_name))
                      
                saver.save(self.sesh, outfile, global_step=self.step)
            try:
                self.logger.flush()
                self.logger.close()
                
                

            except(AttributeError): # not logging
                print('Not logging',flush=True)
            
                        
            sys.exit(0)

