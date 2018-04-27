# -*- coding: utf-8 -*-
__author__ = "代码医生"
"""
Created on Sun Jul 16 10:14:36 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data/", one_hot=True)

print(__author__)

batch_size = 30   # 获取样本的批次大小
n_input = 784 # MNIST data 输入 (img shape: 28*28)
n_classes = 10  # MNIST 列别 (0-9 ，一共10类)

# 待输入的样本图片
x = tf.placeholder("float", [None, n_input])
#x = mnist.train.image
img = tf.reshape(x,[-1,28,28,1])
# corrupted image
x_small = tf.image.resize_bicubic(img, (14, 14))#  缩小2倍



x_bicubic = tf.image.resize_bicubic(x_small, (28, 28))#双立方插值算法变化
x_nearest = tf.image.resize_nearest_neighbor(x_small, (28, 28))
x_bilin = tf.image.resize_bilinear(x_small, (28, 28))


#espcn
net = slim.conv2d(x_small, 64, 5)
net =slim.conv2d(net, 32, 3)
net = slim.conv2d(net, 4, 3)
net = tf.depth_to_space(net,2)
print("net.shape",net.shape)

y_pred = tf.reshape(net,[-1,784])


cost = tf.reduce_mean(tf.pow(x - y_pred, 2))
optimizer = tf.train.AdamOptimizer(0.01 ).minimize(cost)

training_epochs =100
display_step =20

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(mnist.train.num_examples/batch_size)
    # 启动循环开始训练
    for epoch in range(training_epochs):
        # 遍历全部数据集
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs})
        # 显示训练中的详细信息
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("完成!")

    show_num = 10
    encode_s,encode_b,encode_n ,encode_bi,y_predv= sess.run(
        [x_small,x_bicubic,x_nearest,x_bilin,y_pred], feed_dict={x: mnist.test.images[:show_num]})
    
    f, a = plt.subplots(6, 10, figsize=(10, 6))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_s[i], (14, 14)))
        a[2][i].imshow(np.reshape(encode_b[i], (28, 28)))
        a[3][i].imshow(np.reshape(encode_n[i], (28, 28)))
        a[4][i].imshow(np.reshape(encode_bi[i], (28, 28)))
        a[5][i].imshow(np.reshape(y_predv[i], (28, 28)))
    plt.show()
