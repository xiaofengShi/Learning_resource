# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 21:56:25 2017

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
#生成器函数
def generator(x):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('generator')]) > 0
    with tf.variable_scope('generator', reuse = reuse):
        #两个带bn的全连接
        x = slim.fully_connected(x, 1024)
        x = slim.batch_norm(x, activation_fn=tf.nn.relu)
        x = slim.fully_connected(x, 7*7*128)
        x = slim.batch_norm(x, activation_fn=tf.nn.relu)
        #两个转置卷积
        x = tf.reshape(x, [-1, 7, 7, 128])
        x = slim.conv2d_transpose(x, 64, kernel_size=[4,4], stride=2, activation_fn = None)
        x = slim.batch_norm(x, activation_fn = tf.nn.relu)
        z = slim.conv2d_transpose(x, 1, kernel_size=[4, 4], stride=2, activation_fn=tf.nn.sigmoid)
    return z

#反向生成器定义，结构与判别器类似
def inversegenerator(x):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('inversegenerator')]) > 0
    with tf.variable_scope('inversegenerator', reuse=reuse):
        #两个卷积
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        x = slim.conv2d(x, num_outputs = 64, kernel_size=[4,4], stride=2, activation_fn=leaky_relu)
        x = slim.conv2d(x, num_outputs=128, kernel_size=[4,4], stride=2, activation_fn=leaky_relu)
        #两个全连接
        x = slim.flatten(x)        
        shared_tensor = slim.fully_connected(x, num_outputs=1024, activation_fn = leaky_relu)
        z = slim.fully_connected(shared_tensor, num_outputs=50, activation_fn = leaky_relu)
    return z    

#leaky relu定义
def leaky_relu(x):
     return tf.where(tf.greater(x, 0), x, 0.01 * x)

#判别器定义
def discriminator(x, num_classes=10, num_cont=2):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('discriminator')]) > 0

    with tf.variable_scope('discriminator', reuse=reuse):
        #两个卷积
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        x = slim.conv2d(x, num_outputs = 64, kernel_size=[4,4], stride=2, activation_fn=leaky_relu)
        x = slim.conv2d(x, num_outputs=128, kernel_size=[4,4], stride=2, activation_fn=leaky_relu)
        #print ("conv2d",x.get_shape())
        x = slim.flatten(x)
        #两个全连接
        shared_tensor = slim.fully_connected(x, num_outputs=1024, activation_fn = leaky_relu)
        recog_shared = slim.fully_connected(shared_tensor, num_outputs=128, activation_fn = leaky_relu)
        #通过全连接变换，生成输出信息。
        disc = slim.fully_connected(shared_tensor, num_outputs=1, activation_fn=None)
        disc = tf.squeeze(disc, -1)
        #print ("disc",disc.get_shape())#0 or 1
        recog_cat = slim.fully_connected(recog_shared, num_outputs=num_classes, activation_fn=None)
        recog_cont = slim.fully_connected(recog_shared, num_outputs=num_cont, activation_fn=tf.nn.sigmoid)
    return disc, recog_cat, recog_cont


batch_size = 10      # 最小批次
classes_dim = 10     # 10类数字
con_dim = 2          # total continuous factor
rand_dim = 38  
n_input  = 784


x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.int32, [None])

z_con = tf.random_normal((batch_size, con_dim))      #2列
z_rand = tf.random_normal((batch_size, rand_dim))    #38列
z = tf.concat(axis=1, values=[tf.one_hot(y, depth = classes_dim), z_con, z_rand])#50列
gen = generator(z)
genout= tf.squeeze(gen, -1)


#自编码网络
aelearning_rate =0.01
igen = generator(inversegenerator(generator(z)))
loss_ae = tf.reduce_mean(tf.pow(gen - igen, 2))

#输出
igenout = generator(inversegenerator(x))



# labels for discriminator
y_real = tf.ones(batch_size)          #真
y_fake = tf.zeros(batch_size)         #假

# 判别器
disc_real, class_real, _ = discriminator(x)
disc_fake, class_fake, con_fake = discriminator(gen)
pred_class = tf.argmax(class_fake, dimension=1)

# 判别器 loss
loss_d_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=y_real))
loss_d_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=y_fake))
loss_d = (loss_d_r + loss_d_f) / 2

# generator loss
loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=y_real))
# categorical factor loss
loss_cf = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=class_fake, labels=y))#class ok 图片对不上
loss_cr = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=class_real, labels=y))#生成的图片与class ok 与输入的class对不上
loss_c =(loss_cf + loss_cr) / 2
# continuous factor loss
loss_con =tf.reduce_mean(tf.square(con_fake-z_con))

# 获得各个网络中各自的训练参数
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'discriminator' in var.name]
g_vars = [var for var in t_vars if 'generator' in var.name]
ae_vars =  [var for var in t_vars if 'inversegenerator' in var.name]

#disc_global_step = tf.Variable(0, trainable=False)
gen_global_step = tf.Variable(0, trainable=False)
#ae_global_step = tf.Variable(0, trainable=False)
global_step = tf.train.get_or_create_global_step()#使用MonitoredTrainingSession，必须有

train_disc = tf.train.AdamOptimizer(0.0001).minimize(loss_d + loss_c + loss_con, var_list = d_vars, global_step = global_step)
train_gen = tf.train.AdamOptimizer(0.001).minimize(loss_g + loss_c + loss_con, var_list = g_vars, global_step = gen_global_step)
train_ae = tf.train.AdamOptimizer(aelearning_rate).minimize(loss_ae, var_list = ae_vars, global_step = global_step)


training_GANepochs = 3   #训练GAN迭代3次数据集
training_aeepochs = 6    #训练AE迭代3次数据集(从3开始到6)
display_step = 1


with tf.train.MonitoredTrainingSession(checkpoint_dir='log/aecheckpoints',save_checkpoint_secs  =120) as sess:
    
    total_batch = int(mnist.train.num_examples/batch_size)
    print("ae_global_step.eval(session=sess)",global_step.eval(session=sess),int(global_step.eval(session=sess)/total_batch))
    
    for epoch in range( int(global_step.eval(session=sess)/total_batch),training_GANepochs):
        avg_cost = 0.

        # 遍历全部数据集
        for i in range(total_batch):

            batch_xs, batch_ys = mnist.train.next_batch(batch_size)#取数据
            feeds = {x: batch_xs, y: batch_ys}

            # Fit training using batch data
            l_disc, _, l_d_step = sess.run([loss_d, train_disc, global_step],feeds)
            l_gen, _, l_g_step = sess.run([loss_g, train_gen, gen_global_step],feeds)

        # 显示训练中的详细信息
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f} ".format(l_disc),l_gen)

    print("GAN完成!")
    # 测试
    print ("Result:", loss_d.eval({x: mnist.test.images[:batch_size],y:mnist.test.labels[:batch_size]},session = sess)
                        , loss_g.eval({x: mnist.test.images[:batch_size],y:mnist.test.labels[:batch_size]},session = sess))

    # 根据图片模拟生成图片
    show_num = 10
    gensimple,inputx = sess.run(
        [genout,x], feed_dict={x: mnist.test.images[:batch_size],y: mnist.test.labels[:batch_size]})

    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(inputx[i], (28, 28)))
        a[1][i].imshow(np.reshape(gensimple[i], (28, 28)))
        
    plt.draw()
    plt.show()                          
                        
                        
                        
    #begin ae
    print("ae_global_step.eval(session=sess)",global_step.eval(session=sess),int(global_step.eval(session=sess)/total_batch))
    for epoch in range(int(global_step.eval(session=sess)/total_batch),training_aeepochs):
        avg_cost = 0.

        # 遍历全部数据集
        for i in range(total_batch):

            batch_xs, batch_ys = mnist.train.next_batch(batch_size)#取数据
            feeds = {x: batch_xs, y: batch_ys}

            # Fit training using batch data
            l_ae, _, ae_step = sess.run([loss_ae, train_ae, global_step],feeds)

        # 显示训练中的详细信息
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f} ".format(l_ae))
    
    # 测试
    print ("Result:", loss_ae.eval({x: mnist.test.images[:batch_size],y:mnist.test.labels[:batch_size]},session = sess)  )
    
    # 根据图片模拟生成图片
    show_num = 10
    gensimple,inputx = sess.run(
        [igenout,x], feed_dict={x: mnist.test.images[:batch_size],y: mnist.test.labels[:batch_size]})

    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(inputx[i], (28, 28)))
        a[1][i].imshow(np.reshape(gensimple[i], (28, 28)))
        
    plt.draw()
    plt.show()  
    

    

    
    
    
    
    
    
    
    
    
    
    