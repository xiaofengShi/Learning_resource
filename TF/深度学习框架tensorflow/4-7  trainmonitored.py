# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 22:27:15 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""

import tensorflow as tf
tf.reset_default_graph()
global_step = tf.train.get_or_create_global_step()
step = tf.assign_add(global_step, 1)

with tf.train.MonitoredTrainingSession(checkpoint_dir='log/checkpoints',save_checkpoint_secs  = 2) as sess:
    print(sess.run([global_step]))
    while not sess.should_stop():
        i = sess.run( step)
        print( i)
