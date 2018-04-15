#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _Author_: xiaofengShi
# Date: 2018-03-18 20:35:20
# Last Modified by:   xiaofengShi
# Last Modified time: 2018-03-18 20:35:20

import tensorflow as tf


def dense(x, input_size, output_size, activation):
    W = tf.Variable(tf.truncated_normal(
        [input_size, output_size], stddev=0.1), name="weight")
    b = tf.Variable(tf.constant(0.1, shape=[output_size]), name="bias")
    y = activation(tf.matmul(x, W) + b)
    return y


def highway(x, size, activation, carry_bias=-1.0):
    W_T = tf.Variable(tf.truncated_normal(
        [size, size], stddev=0.1), name="weight_transform")
    b_T = tf.Variable(tf.constant(carry_bias, shape=[
                      size]), name="bias_transform")

    W = tf.Variable(tf.truncated_normal(
        [size, size], stddev=0.1), name="weight")
    b = tf.Variable(tf.constant(0.1, shape=[size]), name="bias")

    T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name="transform_gate")
    H = activation(tf.matmul(x, W) + b, name="activation")
    C = tf.sub(1.0, T, name="carry_gate")

    y = tf.add(tf.mul(H, T), tf.mul(x, C), "y")
    return y
