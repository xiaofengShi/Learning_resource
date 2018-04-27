# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 06:00:19 2017
@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""


import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import batch_norm
# 导入 MINST 数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data/", one_hot=True)


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')  
def avg_pool_7x7(x):
  return tf.nn.avg_pool(x, ksize=[1, 7, 7, 1],
                        strides=[1, 7, 7, 1], padding='SAME')
def batch_norm_layer(value,train = None, name = 'batch_norm'): 
  if train is not None:       
      return batch_norm(value, decay = 0.9,updates_collections=None, is_training = True)
  else:
      return batch_norm(value, decay = 0.9,updates_collections=None, is_training = False)
# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data维度 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 数字=> 10 classes
train = tf.placeholder(tf.float32)


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(batch_norm_layer((conv2d(x_image, W_conv1) + b_conv1),train))
h_pool1 = max_pool_2x2(h_conv1)
######################################################多卷积核
W_conv2_5x5 = weight_variable([5, 5, 32, 32]) 
b_conv2_5x5 = bias_variable([32]) 
W_conv2_7x7 = weight_variable([7, 7, 32, 32]) 
b_conv2_7x7 = bias_variable([32]) 
h_conv2_5x5 = tf.nn.relu(batch_norm_layer((conv2d(h_pool1, W_conv2_5x5) + b_conv2_5x5),train))
h_conv2_7x7 = tf.nn.relu(batch_norm_layer((conv2d(h_pool1, W_conv2_7x7) + b_conv2_7x7),train))
h_conv2 = tf.concat([h_conv2_5x5,h_conv2_7x7],3)

h_pool2 = max_pool_2x2(h_conv2)
#########################################################new 池化

W_conv3 = weight_variable([5, 5, 64, 10])
b_conv3 = bias_variable([10])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

nt_hpool3=avg_pool_7x7(h_conv3)#10
nt_hpool3_flat = tf.reshape(nt_hpool3, [-1, 10])
y_conv=tf.nn.softmax(nt_hpool3_flat)




keep_prob = tf.placeholder("float")


cross_entropy = -tf.reduce_sum(y*tf.log(y_conv))

decaylearning_rate = tf.train.exponential_decay(0.04, 20000,1000, 0.9)
train_step = tf.train.AdamOptimizer(decaylearning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):#20000
      batch = mnist.train.next_batch(50)
      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y: batch[1], keep_prob: 1.0})
        print( "step %d, training accuracy %g"%(i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
    
    print ("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
