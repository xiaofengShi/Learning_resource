# _*_ coding:utf-8 _*_
# author :Xiaofeng shi
import tensorflow as tf

slim = tf.contrib.slim

# trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
batch_nums = 32
num_bathces = 100


def alex_slim(input, labels, nums_output, keep_prob=0.5, is_training=True, scope=None):
    end_points = {}
    with tf.variable_scope(scope, 'AlexNet', [input]):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                stride=1,
                                padding='SAME'):
                net = slim.conv2d(input, 96, [11, 11], stride=4, scope='conv1')
                end_points['conv1'] = net
                net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.01 / 9, beta=0.75, name='lrn')
                end_points['lrn1'] = net
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1', padding='VALID')
                net = slim.conv2d(net, 256, [5, 5], scope='conv2')
                end_points['conv2'] = net
                net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.01 / 9, beta=0.75, name='lrn2')
                end_points['lrn2'] = net
                net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool2')
                net = slim.repeat(net, 2, slim.conv2d, 384, [3, 3], stride=1, pooling='SAME', scope='conv3')
                end_points['conv3'] = net
                net = slim.conv2d(net, 256, [3, 3], scope='conv4')
                end_points['conv4'] = net
                net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool3')
                net = tf.transpose(net, [0, 3, 1, 2], name='transpose')
                net = slim.flatten(net, scope='flateen')
                net = slim.fully_connected(net, 4096, scope='fc1')
                end_points['fc1'] = net
                net = slim.fully_connected(net, 4096, scope='fc2')
                end_points['fc2'] = net
                net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training, scope='drop_out')
                net = slim.fully_connected(net, nums_output, scope='fc3')
                end_points['fc3'] = net
                loss = tf.reduce_mean(
                    tf.cast(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=net), tf.float32),
                    name='loss')
                end_points['loss'] = loss
                tf.add_to_collection('loss', loss)
    return net, end_points, loss


input = tf.placeholder(tf.float32, [None, 224 * 224])
labels = tf.placeholder(tf.float32, [None, 10])
net, end_points, loss = alex_slim(input=input, labels=labels, nums_output=10)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
accurate=tf.reduce_mean(tf.equal(tf.arg_max(labels,1),tf.arg_max(net,1)),tf.float32)