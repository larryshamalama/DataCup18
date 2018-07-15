from helpers_datacup import *

'''
The files here are manual add-ons to the tensorflow package. Every function
begins with tf_
'''

def tf_new_kl(y, pred, reg_constant=1e-4):
    kbl      = tf.reduce_sum(tf.multiply(y, tf.log(y/pred)), 1, keepdims=True)
    reg_term = tf.multiply(tf.norm(pred), tf.constant(reg_constant))
    
    return tf.add(kbl, -reg_term)

def tf_new_jsd(y, pred):
    '''
    Symmetry is added
    '''
    return tf_new_kl(y, pred) + tf_new_kl(pred, y)

def tf_l2_norm(y, pred):
    return tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(y, pred), axis=1)))
