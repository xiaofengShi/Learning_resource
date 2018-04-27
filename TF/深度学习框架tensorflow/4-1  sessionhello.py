# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 06:09:31 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')  #定义一个常量
sess = tf.Session()                             #建立一个session
print (sess.run(hello))                        #通过session里面的run来运行结果
sess.close()                                     #关闭session



