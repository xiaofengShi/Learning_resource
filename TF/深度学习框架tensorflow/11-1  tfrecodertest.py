# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 10:53:51 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""

import tensorflow as tf
from datasets import flowers
import pylab 

slim = tf.contrib.slim

DATA_DIR="D:/own/python/flower_photosos"

#选择数据集validation
dataset = flowers.get_split('validation', DATA_DIR)

#创建一个provider
provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
#通过provider的get拿到内容
[image, label] = provider.get(['image', 'label'])
print(image.shape)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
#启动队列
tf.train.start_queue_runners()
#获取数据
image_batch, label_batch = sess.run([image, label])
#显示
print(label_batch)
pylab.imshow(image_batch)
pylab.show()

