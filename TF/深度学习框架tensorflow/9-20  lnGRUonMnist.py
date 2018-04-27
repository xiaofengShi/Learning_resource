# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 06:16:24 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""
import numpy as np
import tensorflow as tf
# 导入 MINST 数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data/", one_hot=True)

from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear 


print(tf.__version__)

tf.reset_default_graph()

def ln(tensor, scope = None, epsilon = 1e-5):
    """ Layer normalizes a 2D tensor along its second axis """
    assert(len(tensor.get_shape()) == 2)
    m, v = tf.nn.moments(tensor, [1], keep_dims=True)
    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'layer_norm'):
        scale = tf.get_variable('scale',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    LN_initial = (tensor - m) / tf.sqrt(v + epsilon)

    return LN_initial * scale + shift


class LNGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, num_units, input_size=None, activation=tanh):
        if input_size is not None:
            print("%s: The input_size parameter is deprecated." % self)
        self._num_units = num_units
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""
        with vs.variable_scope("Gates"):  # Reset gate and update gate.,reuse=True
            # We start with bias of 1.0 to not reset and not update.
            value =_linear([inputs, state], 2 * self._num_units, True, 1.0)
            r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
            r = ln(r, scope = 'r/')
            u = ln(u, scope = 'u/')
            r, u = sigmoid(r), sigmoid(u)
        with vs.variable_scope("Candidate"):
#            with vs.variable_scope("Layer_Parameters"):
            Cand = _linear([inputs,  r *state], self._num_units, True)
            c_pre = ln(Cand,  scope = 'new_h/')
            c = self._activation(c_pre)
        new_h = u * state + (1 - u) * c
        return new_h, new_h







n_input = 28 # MNIST data 输入 (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10  # MNIST 列别 (0-9 ，一共10类)

tf.reset_default_graph()

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])


x1 = tf.unstack(x, n_steps, 1)

#1 BasicLSTMCell
#lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
#outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x1, dtype=tf.float32)

#2 LSTMCell
#lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0)
#outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x1, dtype=tf.float32)

#3 gru
#gru = tf.contrib.rnn.GRUCell(n_hidden)
gru = LNGRUCell(n_hidden)
#outputs = tf.contrib.rnn.static_rnn(gru, x1, dtype=tf.float32)

#4 创建动态RNN
outputs,_  = tf.nn.dynamic_rnn(gru,x,dtype=tf.float32)
outputs = tf.transpose(outputs, [1, 0, 2])

pred = tf.contrib.layers.fully_connected(outputs[-1],n_classes,activation_fn = None)



learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # 计算批次数据的准确率
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print (" Finished!")

    # 计算准确率 for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print ("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
