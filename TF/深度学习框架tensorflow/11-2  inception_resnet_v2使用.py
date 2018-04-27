# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:05:06 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""

import tensorflow as tf

from PIL import Image
from matplotlib import pyplot as plt
from nets import inception
import numpy as np
from datasets import imagenet

tf.reset_default_graph()
image_size = inception.inception_resnet_v2.default_image_size
names = imagenet.create_readable_names_for_imagenet_labels()


slim = tf.contrib.slim

checkpoint_file = 'inception_resnet_v2/inception_resnet_v2_2016_08_30.ckpt'
sample_images = ['img.jpg', 'ps.jpg']

input_imgs = tf.placeholder("float", [None, image_size,image_size,3])

#Load the model
sess = tf.Session()
arg_scope = inception.inception_resnet_v2_arg_scope()

with slim.arg_scope(arg_scope):
  logits, end_points = inception.inception_resnet_v2(input_imgs, is_training=False)

saver = tf.train.Saver()
saver.restore(sess, checkpoint_file)


for image in sample_images:
    reimg = Image.open(image).resize((image_size,image_size))
    reimg = np.array(reimg)
    reimg = reimg.reshape(-1,image_size,image_size,3)
    
    plt.figure()  
    p1 = plt.subplot(121)
    p2 = plt.subplot(122)
    

    p1.imshow(reimg[0])# 显示图片
    p1.axis('off') 
    p1.set_title("organization image")

    reimg_norm = 2 *(reimg / 255.0)-1.0 
    
    p2.imshow(reimg_norm[0])# 显示图片
    p2.axis('off') 
    p2.set_title("input image")  

    plt.show()
 
    predict_values, logit_values = sess.run([end_points['Predictions'], logits], feed_dict={input_imgs: reimg_norm})
     
    print (np.max(predict_values), np.max(logit_values))
    print (np.argmax(predict_values), np.argmax(logit_values),names[np.argmax(logit_values)])

