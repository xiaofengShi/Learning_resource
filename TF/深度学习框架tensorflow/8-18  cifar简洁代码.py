# -*- coding: utf-8 -*-
"""
Created on Tue May  9 16:54:32 2017
@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""


import cifar10_input
import tensorflow as tf
import numpy as np


batch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'
print("begin")
images_train, labels_train = cifar10_input.inputs(eval_data = False,data_dir = data_dir, batch_size = batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data = True, data_dir = data_dir, batch_size = batch_size)
print("begin data")



# tf Graph Input
x = tf.placeholder(tf.float32, [None, 24,24,3]) # cifar data image of shape 24*24*3
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 数字=> 10 classes


x_image = tf.reshape(x, [-1,24,24,3])

h_conv1 =tf.contrib.layers.conv2d(x_image,64,[5,5],1,'SAME',activation_fn=tf.nn.relu)
h_pool1 = tf.contrib.layers.max_pool2d(h_conv1,[2,2],stride=2,padding='SAME')


h_conv2 =tf.contrib.layers.conv2d(h_pool1,64,[5,5],1,'SAME',activation_fn=tf.nn.relu)
h_pool2 = tf.contrib.layers.max_pool2d(h_conv2,[2,2],stride=2,padding='SAME')


nt_hpool2 = tf.contrib.layers.avg_pool2d(h_pool2,[6,6],stride=6,padding='SAME')

nt_hpool2_flat = tf.reshape(nt_hpool2, [-1, 64])

y_conv = tf.contrib.layers.fully_connected(nt_hpool2_flat,10,activation_fn=tf.nn.softmax)

cross_entropy = -tf.reduce_sum(y*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


sess = tf.Session()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)
for i in range(15000):#20000
  image_batch, label_batch = sess.run([images_train, labels_train])
  label_b = np.eye(10,dtype=float)[label_batch] #one hot
  
  train_step.run(feed_dict={x:image_batch, y: label_b},session=sess)
  
  if i%200 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:image_batch, y: label_b},session=sess)
    print( "step %d, training accuracy %g"%(i, train_accuracy))


image_batch, label_batch = sess.run([images_test, labels_test])
label_b = np.eye(10,dtype=float)[label_batch]#one hot
print ("finished！ test accuracy %g"%accuracy.eval(feed_dict={
     x:image_batch, y: label_b},session=sess))


