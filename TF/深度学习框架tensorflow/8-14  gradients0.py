# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:47:13 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""
import tensorflow as tf
w1 = tf.Variable([[1,2]])
w2 = tf.Variable([[3,4]])

y = tf.matmul(w1, [[9],[10]])
#grads = tf.gradients(y,[w1,w2])#w2不相干，会报错
grads = tf.gradients(y,[w1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    gradval = sess.run(grads)
    print(gradval)
