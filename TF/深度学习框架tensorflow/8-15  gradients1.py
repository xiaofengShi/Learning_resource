# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 09:32:15 2017


@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""

import tensorflow as tf


tf.reset_default_graph()
w1 = tf.get_variable('w1', shape=[2])
w2 = tf.get_variable('w2', shape=[2])

w3 = tf.get_variable('w3', shape=[2])
w4 = tf.get_variable('w4', shape=[2])

y1 = w1 + w2+ w3
y2 = w3 + w4

a = w1+w2
a_stoped = tf.stop_gradient(a)
y3= a_stoped+w3

gradients = tf.gradients([y1, y2], [w1, w2, w3, w4], grad_ys=[tf.convert_to_tensor([1.,2.]),
                                                          tf.convert_to_tensor([3.,4.])])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(gradients))

    
 
  
    
