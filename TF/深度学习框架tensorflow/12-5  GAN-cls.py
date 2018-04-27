# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 07:05:40 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow.contrib.slim as slim

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data/")#, one_hot=True)



tf.reset_default_graph()

def generator(x):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('generator')]) > 0
    with tf.variable_scope('generator', reuse = reuse):
        #两个全连接
        x = slim.fully_connected(x, 1024)
        x = slim.batch_norm(x, activation_fn=tf.nn.relu)
        x = slim.fully_connected(x, 7*7*128)
        x = slim.batch_norm(x, activation_fn=tf.nn.relu)
        x = tf.reshape(x, [-1, 7, 7, 128])
        #两个转置卷积
        x = slim.conv2d_transpose(x, 64, kernel_size=[4,4], stride=2, activation_fn = None)
        x = slim.batch_norm(x, activation_fn = tf.nn.relu)
        z = slim.conv2d_transpose(x, 1, kernel_size=[4, 4], stride=2, activation_fn=tf.nn.sigmoid)
    return z

def leaky_relu(x):
     return tf.where(tf.greater(x, 0), x, 0.01 * x)

batch_size = 10             # 最小批次
classes_dim = 10            # 10 个分类

rand_dim = 38  
n_input  = 784

def discriminator(x,y):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('discriminator')]) > 0
    with tf.variable_scope('discriminator', reuse=reuse):
        y = slim.fully_connected(y, num_outputs=n_input, activation_fn = leaky_relu)
        y = tf.reshape(y, shape=[-1, 28, 28, 1])
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        
        x= tf.concat(axis=3, values=[x,y])
        x = slim.conv2d(x, num_outputs = 64, kernel_size=[4,4], stride=2, activation_fn=leaky_relu)
        x = slim.conv2d(x, num_outputs=128, kernel_size=[4,4], stride=2, activation_fn=leaky_relu)
        x = slim.flatten(x)
        shared_tensor = slim.fully_connected(x, num_outputs=1024, activation_fn = leaky_relu)
        disc = slim.fully_connected(shared_tensor, num_outputs=1, activation_fn=tf.nn.sigmoid)
        disc = tf.squeeze(disc, -1)

    return disc




x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.int32, [None])
misy = tf.placeholder(tf.int32, [None])


z_rand = tf.random_normal((batch_size, rand_dim))#38列
z = tf.concat(axis=1, values=[tf.one_hot(y, depth = classes_dim), z_rand])#50列
gen = generator(z)
genout= tf.squeeze(gen, -1)



# 判别器
xin=tf.concat([x, tf.reshape(gen, shape=[-1,784]),x],0)
yin=tf.concat([tf.one_hot(y, depth = classes_dim),tf.one_hot(y, depth = classes_dim),tf.one_hot(misy, depth = classes_dim)],0)
disc_all = discriminator(xin,yin)
disc_real,disc_fake,disc_mis =tf.split(disc_all,3)


loss_d = tf.reduce_sum(tf.square(disc_real-1) + ( tf.square(disc_fake)+tf.square(disc_mis))/2 )/2
loss_g = tf.reduce_sum(tf.square(disc_fake-1))/2

# 获得各个网络中各自的训练参数
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'discriminator' in var.name]
g_vars = [var for var in t_vars if 'generator' in var.name]


#disc_global_step = tf.Variable(0, trainable=False)
gen_global_step = tf.Variable(0, trainable=False)

global_step = tf.train.get_or_create_global_step()#使用MonitoredTrainingSession，必须有

train_disc = tf.train.AdamOptimizer(0.0001).minimize(loss_d , var_list = d_vars, global_step = global_step)
train_gen = tf.train.AdamOptimizer(0.001).minimize(loss_g , var_list = g_vars, global_step = gen_global_step)



training_epochs = 3
display_step = 1


with tf.train.MonitoredTrainingSession(checkpoint_dir='log/checkpointsnew',save_checkpoint_secs  =60) as sess:

    total_batch = int(mnist.train.num_examples/batch_size)
    print("global_step.eval(session=sess)",global_step.eval(session=sess),int(global_step.eval(session=sess)/total_batch))
    for epoch in range( int(global_step.eval(session=sess)/total_batch),training_epochs):
        avg_cost = 0.

        # 遍历全部数据集
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)#取数据
            _, mis_batch_ys = mnist.train.next_batch(batch_size)#取数据
            feeds = {x: batch_xs, y: batch_ys,misy:mis_batch_ys}

            # Fit training using batch data
            l_disc, _, l_d_step = sess.run([loss_d, train_disc, global_step],feeds)
            l_gen, _, l_g_step = sess.run([loss_g, train_gen, gen_global_step],feeds)

        # 显示训练中的详细信息
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f} ".format(l_disc),l_gen)

    print("完成!")
    
    # 测试
    _, mis_batch_ys = mnist.train.next_batch(batch_size)
    print ("result:", loss_d.eval({x: mnist.test.images[:batch_size],y:mnist.test.labels[:batch_size],misy:mis_batch_ys},session = sess)
                        , loss_g.eval({x: mnist.test.images[:batch_size],y:mnist.test.labels[:batch_size],misy:mis_batch_ys},session = sess))
    
    # 根据图片模拟生成图片
    show_num = 10
    gensimple,inputx,inputy = sess.run(
        [genout,x,y], feed_dict={x: mnist.test.images[:batch_size],y: mnist.test.labels[:batch_size]})

    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(inputx[i], (28, 28)))
        a[1][i].imshow(np.reshape(gensimple[i], (28, 28)))
        
        
    plt.draw()
    plt.show()  
    


    
    
    
    
    
    
    
    
    
    
    