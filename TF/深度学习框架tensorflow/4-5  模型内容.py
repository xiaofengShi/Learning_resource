# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 05:38:39 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
savedir = "log/"
print_tensors_in_checkpoint_file(savedir+"linermodel.cpkt", None, True)

W = tf.Variable(1.0, name="weight")
b = tf.Variable(2.0, name="bias")

# 放到一个字典里:
saver = tf.train.Saver({'weight': b, 'bias': W})

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver.save(sess, savedir+"linermodel.cpkt")

print_tensors_in_checkpoint_file(savedir+"linermodel.cpkt", None, True)