# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 10:20:45 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""

import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import tensorflow as tf  


myimg = mpimg.imread('img.jpg') # 读取和代码处于同一目录下的图片
plt.imshow(myimg) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()
print(myimg.shape)


full=np.reshape(myimg,[1,3264,2448,3])  
inputfull = tf.Variable(tf.constant(1.0,shape = [1, 3264, 2448, 3]))

filter =  tf.Variable(tf.constant([[-1.0,-1.0,-1.0],  [0,0,0],  [1.0,1.0,1.0],
                                    [-2.0,-2.0,-2.0], [0,0,0],  [2.0,2.0,2.0],
                                    [-1.0,-1.0,-1.0], [0,0,0],  [1.0,1.0,1.0]],shape = [3, 3, 3, 1]))                                    

op = tf.nn.conv2d(inputfull, filter, strides=[1, 1, 1, 1], padding='SAME') #3个通道输入，生成1个feature ma
o=tf.cast(  ((op-tf.reduce_min(op))/(tf.reduce_max(op)-tf.reduce_min(op)) ) *255 ,tf.uint8)



with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer()  )  

    t,f=sess.run([o,filter],feed_dict={ inputfull:full})
    #print(f)
    t=np.reshape(t,[3264,2448]) 
 
    plt.imshow(t,cmap='Greys_r') # 显示图片
    plt.axis('off') # 不显示坐标轴
    plt.show()

