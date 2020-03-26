import tensorflow as tf



def fc_bn_dropout(x,
                  scope = "fc_bn_dropout",
                  size = None,
                  dropout = 1.,
                  activation = tf.nn.elu,
                  training = True):
    
    assert size, "Must specify layer size (num nodes)"
    # use linear activation for pre-activation batch_normalization
    
    with tf.variable_scope(scope):
        
        fc = tf.contrib.layers.fully_connected(inputs= x,
                                                   num_outputs = size,
                                                   activation_fn = None)
        fc_bn = tf.contrib.layers.batch_norm(inputs = fc,
                                                 is_training = training,
                                                 activation_fn = activation)
        
        fc_bn_drop= tf.contrib.layers.dropout(inputs = fc_bn,
                                          keep_prob = dropout,
                                          is_training = training)
        
        
    return fc_bn_drop


def fc_dropout(x,
                  scope = "fc_dropout",
                  size = None,
                  dropout = 1.,
                  activation = tf.nn.elu,
                  training = True):
    
    assert size, "Must specify layer size (num nodes)"
    # use linear activation for pre-activation batch_normalization
    
    with tf.variable_scope(scope):
        
        fc = tf.contrib.layers.fully_connected(inputs= x,
                                                   num_outputs = size,
                                                   activation_fn = activation)
        
        
        fc_drop= tf.contrib.layers.dropout(inputs = fc,
                                          keep_prob = dropout,
                                          is_training = training)
        
        
    return fc_drop



class generator:
    
    def __init__(self, name,architecture,dropout = 1., 
                 activation = tf.nn.relu, output_activation = None):
        self.name = name
        self.architecture = architecture
        self.dropout = dropout
        self.activation = activation
        self.output_activation = output_activation
        self.reuse = None
        
    def __call__(self,inputs,is_train = True):
        
        with tf.variable_scope(self.name,reuse=self.reuse):
            
            net = inputs
            for idx, hidden_size in enumerate(self.architecture[:-1]):
                net = fc_dropout(x = net,
                                 size = int(hidden_size),
                                 scope = "h"+str(idx)+"_fc_drop_"+ str(hidden_size),
                                 activation = self.activation,
                                 dropout = self.dropout,
                                 training = is_train)
            net = fc_dropout(x = net,
                                 size = int(self.architecture[-1]),
                                 scope = "output",
                                 activation = self.output_activation,
                                 dropout = self.dropout,
                                 training = is_train)
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return net
    
    def print_architecture(self):        
        print(self.name + ' architecture: {}'.format(self.architecture),
              flush=True)
    
    def print_variables(self):        
        print(self.name + ' variables: {}'.format(self.variables))
        

class discriminator:
    
    def __init__(self, name, architecture, dropout =1., 
                 activation = tf.nn.relu,output_activation = None):
        self.name = name
        self.architecture = architecture
        self.dropout = dropout
        self.activation = activation
        self.output_activation = output_activation
        self.reuse = None
    
    def __call__(self, inputs, is_train = True):
        
        with tf.variable_scope(self.name,reuse=self.reuse):
            net = inputs
            for idx, hidden_size in enumerate(self.architecture):
                net = fc_dropout(x = net,
                                 size = int(hidden_size),
                                 scope = "h"+str(idx)+"_fc_drop_"+ str(hidden_size),
                                 activation = self.activation,
                                 dropout = self.dropout,
                                 training = is_train)
            
            net = fc_dropout(x = net,
                                size = 1,
                                scope = "output",
                                activation = self.output_activation,
                                dropout = self.dropout,
                                training = is_train)
            
            
        self.reuse = True
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return net
    
    def print_architecture(self):       
        print(self.name + ' architecture: {}'.format(self.architecture),
              flush=True)
    def print_variables(self):        
        print(self.name + ' variables: {}'.format(self.variables))
        
        
        
        
        


