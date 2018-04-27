# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 09:45:41 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""


import numpy as np
import os
import tensorflow as tf

from PIL import Image
from datasets import imagenet
from nets import vgg
# 加载像素均值及相关函数
from preprocessing.vgg_preprocessing import (_mean_image_subtraction,
                                            _R_MEAN, _G_MEAN, _B_MEAN)

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif']=['SimHei']#用来正常显示中文标签  
mpl.rcParams['font.family'] = 'STSong'
mpl.rcParams['font.size'] = 12

tf.reset_default_graph()

slim = tf.contrib.slim

# 网络模型的输入图像有默认的尺寸
# 先调整输入图片的尺寸

names = imagenet.create_readable_names_for_imagenet_labels()
checkpoints_dir = 'vgg_19_2016_08_28'
sample_images = ['hy.jpg', 'ps.jpg']



def showobjlab(img, labels_str=[], title=""):
    minval = np.min(img)
    maxval = np.max(img)
    #获取离散化的色彩表
    plt.figure(figsize=(3,3)) 
    cmap = plt.get_cmap('Paired', np.max(img)-np.min(img)+1)
    mat = plt.matshow(img, cmap=cmap,vmin = minval-0.5,vmax = maxval +0.5)
    
    #定义colorbar
    cax = plt.colorbar(mat,ticks=np.arange(minval,maxval+1),shrink=2)

    # 添加类别名称
    if labels_str:
        cax.ax.set_yticklabels(labels_str)

    if title:
        plt.suptitle(title, fontsize=14, fontweight='bold')






input_imgs = tf.placeholder("float", [None,None,3])
# 每个像素减去像素的均值
processed_image = _mean_image_subtraction(input_imgs,
                                          [_R_MEAN, _G_MEAN, _B_MEAN])

input_image = tf.expand_dims(processed_image, 0)
#print(input_image.shape)
with slim.arg_scope(vgg.vgg_arg_scope()):# spatial_squeeze选项指定是否压缩结果的空间维度将不必要的空间维度删除
    logits, _ = vgg.vgg_19(input_image,
                           num_classes=1000,
                           is_training=False,
                           spatial_squeeze=False)


pred = tf.argmax(logits, dimension=3)

init_fn = slim.assign_from_checkpoint_fn(
    os.path.join(checkpoints_dir, 'vgg_19.ckpt'),
    slim.get_model_variables('vgg_19'))

with tf.Session() as sess:
    init_fn(sess)
    for image in sample_images:
        reimg = Image.open(image)
        plt.suptitle("原始图片", fontsize=14, fontweight='bold')
        plt.imshow(reimg) # 显示图片
        plt.axis('off') # 不显示坐标轴
        plt.show()        
        
        #reimg = np.array(reimg)
        reimg = np.asarray(reimg, dtype='float')
        #print(np.shape(reimg))

        obj,inpt= sess.run([pred,input_image],feed_dict={input_imgs: reimg})
        
       
        obj = np.squeeze(obj)
        
        unique_classes, relabeled_image = np.unique(obj,
                                                    return_inverse=True)
        
        obj_size = obj.shape
        relabeled_image = relabeled_image.reshape(obj_size)
        labels_names = []

        for index, current_class_number in enumerate(unique_classes):
            labels_names.append(str(index) + ' ' + names[current_class_number+1])
        
        showobjlab(img=relabeled_image, labels_str=labels_names, title="画面识别")
        plt.show()





















