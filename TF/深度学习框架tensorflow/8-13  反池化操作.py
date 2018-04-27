# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 10:35:37 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""

import tensorflow as tf
import numpy as np


def max_pool_with_argmax(net, stride):
    _, mask = tf.nn.max_pool_with_argmax( net,ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1],padding='SAME')
    mask = tf.stop_gradient(mask)
    net = tf.nn.max_pool(net, ksize=[1, stride, stride, 1],strides=[1, stride, stride, 1], padding='SAME') 
    return net, mask
 

def unpool(net, mask, stride):

    ksize = [1, stride, stride, 1]
    input_shape = net.get_shape().as_list()
    #  calculation new shape
    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
    # calculation indices for batch, height, width and feature maps
    one_like_mask = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    f = one_like_mask * feature_range
    # transpose indices & reshape update values to one dimension
    updates_size = tf.size(net)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(net, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return ret
    
    
img=tf.constant([  
        [[0.0,4.0],[0.0,4.0],[0.0,4.0],[0.0,4.0]],  
        [[1.0,5.0],[1.0,5.0],[1.0,5.0],[1.0,5.0]],  
        [[2.0,6.0],[2.0,6.0],[2.0,6.0],[2.0,6.0]],  
        [[3.0,7.0],[3.0,7.0], [3.0,7.0],[3.0,7.0]]
    ])  
  
img=tf.reshape(img,[1,4,4,2])  
pooling2=tf.nn.max_pool(img,[1,2,2,1],[1,2,2,1],padding='SAME')  
encode, mask = max_pool_with_argmax(img, 2)
img2 = unpool(encode,mask,2)
print(img.shape)
print(encode.shape)
print(mask.shape)
print(img2.shape)
with tf.Session() as sess:  
    print("image:")  
    print (sess.run(img))     
    result=sess.run(pooling2)  
    print ("pooling2:\n",result)
    result,mask2=sess.run([encode, mask])  
    print ("encode:\n",result,mask2)
    result=sess.run(img2)  
    print ("reslut:\n",result)