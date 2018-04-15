import tensorflow as tf

# v1 = tf.Variable(tf.random_normal([1, 2]), name="v1")
# v2 = tf.Variable(tf.random_normal([2, 3]), name="v2")
# saver = tf.train.Saver()  # 声明tf.train.Saver类用于保存模型
# with tf.Session() as sess:
#     saver.restore(sess, "model.ckpt")  # 即将固化到硬盘中的Session从保存路径再读取出来
#     print("v1:", sess.run(v1))  # 打印v1、v2的值和之前的进行对比
#     print("v2:", sess.run(v2))
#     print("Model Restored")


saver = tf.train.import_meta_graph('my_first_cnn_model.meta')
with tf.Session() as sess:
    saver.restore(sess, '/')

    print(sess.run(tf.get_default_graph().get_tensor_by_name('accuracy')))