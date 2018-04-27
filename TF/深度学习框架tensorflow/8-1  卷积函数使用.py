# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:45:26 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""

import tensorflow as tf  
  
# [batch, in_height, in_width, in_channels] [训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]  
input = tf.Variable(tf.constant(1.0,shape = [1, 5, 5, 1])) 
input2 = tf.Variable(tf.constant(1.0,shape = [1, 5, 5, 2]))
input3 = tf.Variable(tf.constant(1.0,shape = [1, 4, 4, 1])) 

# [filter_height, filter_width, in_channels, out_channels] [卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]   
filter1 =  tf.Variable(tf.constant([-1.0,0,0,-1],shape = [2, 2, 1, 1]))
filter2 =  tf.Variable(tf.constant([-1.0,0,0,-1,-1.0,0,0,-1],shape = [2, 2, 1, 2])) 
filter3 =  tf.Variable(tf.constant([-1.0,0,0,-1,-1.0,0,0,-1,-1.0,0,0,-1],shape = [2, 2, 1, 3])) 
filter4 =  tf.Variable(tf.constant([-1.0,0,0,-1,
                                   -1.0,0,0,-1,
                                   -1.0,0,0,-1,
                                   -1.0,0,0,-1],shape = [2, 2, 2, 2])) 
filter5 =  tf.Variable(tf.constant([-1.0,0,0,-1,-1.0,0,0,-1],shape = [2, 2, 2, 1])) 



# padding的值为‘VALID’，表示边缘不填充, 当其为‘SAME’时，表示填充到卷积核可以到达图像边缘  
op1 = tf.nn.conv2d(input, filter1, strides=[1, 2, 2, 1], padding='SAME') #1个通道输入，生成1个feature ma
op2 = tf.nn.conv2d(input, filter2, strides=[1, 2, 2, 1], padding='SAME') #1个通道输入，生成2个feature map
op3 = tf.nn.conv2d(input, filter3, strides=[1, 2, 2, 1], padding='SAME') #1个通道输入，生成3个feature map

op4 = tf.nn.conv2d(input2, filter4, strides=[1, 2, 2, 1], padding='SAME') # 2个通道输入，生成2个feature
op5 = tf.nn.conv2d(input2, filter5, strides=[1, 2, 2, 1], padding='SAME') # 2个通道输入，生成一个feature map

vop1 = tf.nn.conv2d(input, filter1, strides=[1, 2, 2, 1], padding='VALID') # 5*5 对于pading不同而不同
op6 = tf.nn.conv2d(input3, filter1, strides=[1, 2, 2, 1], padding='SAME') 
vop6 = tf.nn.conv2d(input3, filter1, strides=[1, 2, 2, 1], padding='VALID')  #4*4与pading无关
  


init = tf.global_variables_initializer()  
with tf.Session() as sess:  
    sess.run(init)  
    
    print("op1:\n",sess.run([op1,filter1]))#1-1  后面补0
    print("------------------")
    
    print("op2:\n",sess.run([op2,filter2])) #1-2多卷积核 按列取
    print("op3:\n",sess.run([op3,filter3])) #1-3
    print("------------------")   
    
    print("op4:\n",sess.run([op4,filter4]))#2-2    通道叠加
    print("op5:\n",sess.run([op5,filter5]))#2-1        
    print("------------------")
  
    print("op1:\n",sess.run([op1,filter1]))#1-1
    print("vop1:\n",sess.run([vop1,filter1]))
    print("op6:\n",sess.run([op6,filter1]))
    print("vop6:\n",sess.run([vop6,filter1]))    