# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 17:44:42 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""
import tensorflow as tf
import numpy as np

# 网络结构：2维输入 --> 2维隐藏层 --> 1维输出

learning_rate = 1e-4
n_input  = 2
n_label  = 2#1
n_hidden = 100


x = tf.placeholder(tf.float32, [None,n_input])
y = tf.placeholder(tf.float32, [None, n_label])

weights = {
    'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden], stddev=0.1)),
    'h2': tf.Variable(tf.truncated_normal([n_hidden, n_label], stddev=0.1))
	} 
biases = {
    'h1': tf.Variable(tf.zeros([n_hidden])),
    'h2': tf.Variable(tf.zeros([n_label]))
    }    


layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['h1']))

#1 softmax 方法
#y_pred = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']),biases['h2'])) 
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits( labels=y,logits=y_pred)
#loss = tf.reduce_mean(cross_entropy)

#2 sigmoid方法+平方差
y_pred = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']),biases['h2']))
loss=tf.reduce_mean((y_pred-y)**2)

#3 relu方法+平方差
#layer2 =tf.add(tf.matmul(layer_1, weights['h2']),biases['h2'])
#y_pred = tf.maximum(layer2,0.01*layer2) 
#loss=tf.reduce_mean((y_pred-y)**2)
 

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#生成数据
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
X=np.array(X).astype('float32')
Y=np.array(Y).astype('int16')

#加载
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

#训练
for i in range(10000):
    sess.run(train_step,feed_dict={x:X,y:Y} )

     
#计算预测值
print(sess.run(y_pred,feed_dict={x:X}))
#输出：已训练100000次

