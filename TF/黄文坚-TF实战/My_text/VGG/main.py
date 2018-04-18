import tensorflow as tf
import numpy as np
import TFRecord

# 定义网络参数
learning_rate = 0.001
display_step = 5
epochs = 10
keep_prob = 0.5


def conv_op(input_op, name, kh, kw, n_out, dh, dw, par):
    input_op = tf.convert_to_tensor(input_op)
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w",
                                 shape=[kh, kw, n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        par += [kernel, biases]
        return activation


# 定义全连接操作


def fc_op(input_op, name, n_out, par):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + 'w',
                                 shape=[n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        # tf.nn.relu_layer对输入变量input_op与kernel做矩阵乘法加上bias，再做RELU非线性变换得到activation
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        par += [kernel, biases]
        return activation


# 定义池化层
def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)


# net
def inference_op(input_op, keep_prob):
    p = []
    # block 1 -- outputs 112x112x64
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, par=p)
    conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, par=p)
    pool1 = mpool_op(conv1_2, name="pool1", kh=2, kw=2, dw=2, dh=2)

    # block 2 -- outputs 56x56x128
    conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, par=p)
    conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, par=p)
    pool2 = mpool_op(conv2_2, name="pool2", kh=2, kw=2, dh=2, dw=2)

    # # block 3 -- outputs 28x28x256
    conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, par=p)
    conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, par=p)
    conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, par=p)
    pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dh=2, dw=2)

    # block 4 -- outputs 14x14x512
    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, par=p)
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, par=p)
    conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, par=p)
    pool4 = mpool_op(conv4_3, name="pool4", kh=2, kw=2, dh=2, dw=2)

    # block 5 -- outputs 7x7x512
    conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, par=p)
    conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, par=p)
    conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, par=p)
    pool5 = mpool_op(conv5_3, name="pool5", kh=2, kw=2, dw=2, dh=2)

    # flatten
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

    # fully connected
    fc6 = fc_op(resh1, name="fc6", n_out=4096, par=p)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

    fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, par=p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

    logits = fc_op(fc7_drop, name="fc8", n_out=2, par=p)
    return logits


def train(logits, labels):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return optimizer, cost, accuracy


if __name__ == "__main__":
    train_filename = "/home/wc/DataSet/traffic/testTFRecord/train.tfrecords"
    test_filename = "/home/wc/DataSet/traffic/testTFRecord/test.tfrecords"
    image_batch, label_batch = TFRecord.createBatch(filename=train_filename, batchsize=2)
    test_image, test_label = TFRecord.createBatch(filename=test_filename, batchsize=20)
    pred = inference_op(input_op=image_batch, keep_prob=keep_prob)
    test_pred = inference_op(input_op=test_image, keep_prob=keep_prob)
    optimizer, cost, accuracy = train(logits=pred, labels=label_batch)
    test_optimizer, test_cost, test_acc = train(logits=test_pred, labels=test_label)
    initop = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(initop)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        step = 0
        while step < epochs:
            step += 1
            print(step)
            _, loss, acc = sess.run([optimizer, cost, accuracy])
            if step % display_step == 0:
                print(loss, acc)
        print("training finish!")
        _, testLoss, testAcc = sess.run([test_optimizer, test_cost, test_acc])
        print("Test acc = " + str(testAcc))
        print("Test Finish!")
