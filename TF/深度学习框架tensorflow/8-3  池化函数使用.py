# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:59:29 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""

import tensorflow as tf  
  
img=tf.constant([  
        [[0.0,4.0],[0.0,4.0],[0.0,4.0],[0.0,4.0]],  
        [[1.0,5.0],[1.0,5.0],[1.0,5.0],[1.0,5.0]],  
        [[2.0,6.0],[2.0,6.0],[2.0,6.0],[2.0,6.0]],  
        [[3.0,7.0],[3.0,7.0], [3.0,7.0],[3.0,7.0]]
    ])  
  
img=tf.reshape(img,[1,4,4,2])  
  
pooling=tf.nn.max_pool(img,[1,2,2,1],[1,2,2,1],padding='VALID')  
pooling1=tf.nn.max_pool(img,[1,2,2,1],[1,1,1,1],padding='VALID')
pooling2=tf.nn.avg_pool(img,[1,4,4,1],[1,1,1,1],padding='SAME')  
pooling3=tf.nn.avg_pool(img,[1,4,4,1],[1,4,4,1],padding='SAME') 
nt_hpool2_flat = tf.reshape(tf.transpose(img), [-1, 16]) 
pooling4=tf.reduce_mean(nt_hpool2_flat,1) #1对行求均值（1表示轴是列）   0 对列求均值


with tf.Session() as sess:  
    print("image:")  
    image=sess.run(img)  
    print (image)  
    result=sess.run(pooling)  
    print ("reslut:\n",result)  
    result=sess.run(pooling1)  
    print ("reslut1:\n",result)     
    result=sess.run(pooling2)  
    print ("reslut2:\n",result)
    result=sess.run(pooling3)  
    print ("reslut3:\n",result) 
    flat,result=sess.run([nt_hpool2_flat,pooling4])  
    print ("reslut4:\n",result) 
    print("flat:\n",flat)     
    
    
    