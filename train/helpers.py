import numpy as np
import tensorflow as tf

def new_variable(shape, name):
    return tf.get_variable(name=name, 
                           shape=shape, 
                           dtype=tf.float32, 
                           initializer=tf.truncated_normal_initializer())
    
def flatten_layer(layer):
    layer_shape  = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat   = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


def new_fc_layer(_input,          
                 num_inputs,     
                 num_outputs,
                 prefix,
                 use_relu=True,
                 batch_norm=False):
    
    with tf.variable_scope(prefix):
        weights = new_variable(shape=[num_inputs, num_outputs], name='fc_weights')
        biases  = new_variable(shape=[num_outputs], name='fc_bias')
    
    if batch_norm:
        layer = tf.contrib.layers.batch_norm(layer, epsilon=1e-5)
    
    if use_relu:
        layer = tf.nn.leaky_relu(tf.add(tf.matmul(_input, weights), biases))
    else:
        layer = tf.add(tf.matmul(_input, weights), biases)

    return layer

def conv_pool_1d_layer(_input,
                       filter_size, 
                       num_input_channels, 
                       num_output,
                       prefix,
                       max_pooling=True,
                       use_relu=True):
    
    '''
    If this function is called multiple times with the same variable names,
    ValueErrors will rise, hence the need of a prefix.
    '''
    assert isinstance(prefix, str)
    assert prefix != ''
    
    CONV_STRIDES = 1
    POOL_STRIDES = 3
    POOL_SIZE    = 3
    LEAKY_ALPHA  = 0.2
    
    with tf.variable_scope(prefix):
        conv_matrix = new_variable(name='conv_matrix', shape=[filter_size, num_input_channels, num_output])
        bias        = new_variable(name='bias', shape=[num_output])
        

    conv_layer = tf.nn.conv1d(value=_input,
                              filters=conv_matrix,
                              stride=CONV_STRIDES,
                              padding='SAME',
                              name='conv_layer')
    

    if max_pooling:
        pooled_layer = tf.layers.max_pooling1d(intputs=conv_layer,
                                               pool_size=POOL_SIZE,
                                               strides=POOL_STRIDES,
                                               name='pooled_layer')
    else:
        pooled_layer = conv_layer
        
    pooled_layer += bias
    
    if use_relu:
        return tf.nn.leaky_relu(pooled_layer, alpha=LEAKY_ALPHA)
    
    return pooled_layer