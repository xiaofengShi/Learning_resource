import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_DATA/', one_hot=True)


sess = tf.InteractiveSession()


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None,data_format=None, name=None)
# x-输入,
# w-卷积参数[卷积高度，卷积宽度，输入数据含有几个通道，卷积核数量]
# strides-移动步长[]格式默认格式”NHWC“
# strides的设置为[1,stride,stride,1]对应[batch,in_height, in_width, in_channels]
# 第一个表示在一个样本的特征图上的移动，第二三个是在filter在特征图上的移动的跨度，第四个表示在一个样本的一个通道上移动。
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# 最大池化函数
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
# [-1,28,28,1]将1x784的数据变成28x28形式，-1表示样本数量不固定，1表示颜色通道数
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一层卷积参数
# 卷积核尺寸5x5,1个颜色通道，32个卷积核
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# relu 激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
# 最大池化层，经过2x2池化之后，原始尺寸变为14x14
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积参数
# 卷积核尺寸5x5,32个颜色通道，64个卷积核
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
# 最大池化层，经过2x2池化之后，原始尺寸变为7x7
h_pool2 = max_pool_2x2(h_conv2)  # 经过第二层的之后的图像输出尺寸为7x7x64

# 全链接层
w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

# 连接softmax
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_dropout, w_fc2) + b_fc2)

# cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# correct properity
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 训练之前初始化参数
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())


# 进行训练
for i in range(2000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d,training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})  # 训练的dropout率为0.5

print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


saver.save(sess, './save/my_first_cnn_model')
# saver.restore(sess, '/home/sxf/MyProject_Python/TFtest/卷积/save/my_first_cnn_model')