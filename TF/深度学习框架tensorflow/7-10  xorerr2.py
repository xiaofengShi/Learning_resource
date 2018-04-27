# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 08:06:57 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(55)
np.random.seed(55)

input_data = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]  # XOR input
output_data = [[0.], [1.], [1.], [0.]]  # XOR output


hidden_nodes =2

n_input = tf.placeholder(tf.float32, shape=[None, 2], name="n_input")
n_output = tf.placeholder(tf.float32, shape=[None, 1], name="n_output")

# hidden layer's bias neuron
b_hidden = tf.Variable(0.1, name="hidden_bias")  


W_hidden = tf.Variable(tf.random_normal([2, hidden_nodes]), name="hidden_weights")

hidden = tf.sigmoid(tf.matmul(n_input, W_hidden) + b_hidden)

################
# output layer #
################
W_output = tf.Variable(tf.random_normal([hidden_nodes, 1]), name="output_weights")  # output layer's weight matrix


#不影响
b_output =  tf.Variable(0.1, name="output_bias")



output = tf.nn.tanh(tf.matmul(hidden, W_output)+b_output)  # 

#softmax
y = tf.matmul(hidden, W_output)+b_output
output = tf.nn.softmax(tf.matmul(hidden, W_output)+b_output)



#交叉熵
loss = -(n_output * tf.log(output) + (1 - n_output) * tf.log(1 - output))



optimizer = tf.train.AdamOptimizer(0.01) 
train = optimizer.minimize(loss)  # let the optimizer train

#####################
# train the network #
#####################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(0, 2001):  
        # run the training operation
        cvalues = sess.run([train, loss, W_hidden, b_hidden, W_output],
                       feed_dict={n_input: input_data, n_output: output_data})

    # print some debug stuff
        if epoch % 200 == 0:
            print("")
            print("step: {:>3}".format(epoch))
            print("loss: {}".format(cvalues[1]))
            # print("b_hidden: {}".format(cvalues[3]))
            # print("W_hidden: {}".format(cvalues[2]))
            # print("W_output: {}".format(cvalues[4]))


    print("")
    print("input: {} | output: {}".format(input_data[0], sess.run(output, feed_dict={n_input: [input_data[0]]})))
    print("input: {} | output: {}".format(input_data[1], sess.run(output, feed_dict={n_input: [input_data[1]]})))
    print("input: {} | output: {}".format(input_data[2], sess.run(output, feed_dict={n_input: [input_data[2]]})))
    print("input: {} | output: {}".format(input_data[3], sess.run(output, feed_dict={n_input: [input_data[3]]})))
 





































