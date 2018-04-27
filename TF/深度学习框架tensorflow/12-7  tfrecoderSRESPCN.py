# -*- coding: utf-8 -*-
__author__ = "代码医生"
"""
Created on Fri Jun  2 10:53:51 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""

print(__author__)

import tensorflow as tf
from datasets import flowers
import numpy as np
import matplotlib.pyplot as plt

slim = tf.contrib.slim



def batch_mse_psnr(dbatch):
    im1,im2=np.split(dbatch,2)
    mse=((im1-im2)**2).mean(axis=(1,2))
    psnr=np.mean(20*np.log10(255.0/np.sqrt(mse)))
    return np.mean(mse),psnr
def batch_y_psnr(dbatch):
    r,g,b=np.split(dbatch,3,axis=3)
    y=np.squeeze(0.3*r+0.59*g+0.11*b)
    im1,im2=np.split(y,2)
    mse=((im1-im2)**2).mean(axis=(1,2))
    psnr=np.mean(20*np.log10(255.0/np.sqrt(mse)))
    return psnr
def batch_ssim(dbatch):
    im1,im2=np.split(dbatch,2)
    imgsize=im1.shape[1]*im1.shape[2]
    avg1=im1.mean((1,2),keepdims=1)
    avg2=im2.mean((1,2),keepdims=1)
    std1=im1.std((1,2),ddof=1)
    std2=im2.std((1,2),ddof=1)
    cov=((im1-avg1)*(im2-avg2)).mean((1,2))*imgsize/(imgsize-1)
    avg1=np.squeeze(avg1)
    avg2=np.squeeze(avg2)
    k1=0.01
    k2=0.03
    c1=(k1*255)**2
    c2=(k2*255)**2
    c3=c2/2
    return np.mean((2*avg1*avg2+c1)*2*(cov+c3)/(avg1**2+avg2**2+c1)/(std1**2+std2**2+c2))

def showresult(subplot,title,orgimg,thisimg,dopsnr = True):
    p =plt.subplot(subplot)
    p.axis('off') 
    p.imshow(np.asarray(thisimg[0], dtype='uint8'))
    if dopsnr :
        conimg =  np.concatenate((orgimg,thisimg))
        mse,psnr=batch_mse_psnr(conimg)
        ypsnr=batch_y_psnr(conimg)
        ssim=batch_ssim(conimg)
        p.set_title(title+str(int(psnr))+" y:"+str(int(ypsnr))+" s:"+str(ssim))
    else:
        p.set_title(title)





height = width = 200
batch_size = 4
DATA_DIR="D:/own/python/flower_photosos"

#选择数据集validation
dataset = flowers.get_split('validation', DATA_DIR)
#创建一个provider
provider = slim.dataset_data_provider.DatasetDataProvider(dataset,num_readers = 2)
#通过provider的get拿到内容
[image, label] = provider.get(['image', 'label'])
print(image.shape)
 

# 剪辑图片为统一大小 
distorted_image = tf.image.resize_image_with_crop_or_pad(image, height, width)#剪辑尺寸，不够填充 
################################################
images, labels = tf.train.batch([distorted_image, label], batch_size=batch_size)
print(images.shape)

x_smalls = tf.image.resize_images(images, (np.int32(height/2), np.int32(width/2)))#  缩小2*2倍
x_smalls2 = x_smalls/255.0
#还原
x_nearests = tf.image.resize_images(x_smalls, (height, width),tf.image.ResizeMethod.NEAREST_NEIGHBOR)
x_bilins = tf.image.resize_images(x_smalls, (height, width),tf.image.ResizeMethod.BILINEAR)
x_bicubics = tf.image.resize_images(x_smalls, (height, width),tf.image.ResizeMethod.BICUBIC)


net = slim.conv2d(x_smalls2, 64, 5,activation_fn = tf.nn.tanh)
net =slim.conv2d(net, 32, 3,activation_fn = tf.nn.tanh)
net = slim.conv2d(net, 12, 3,activation_fn = None)#2*2*3
y_predt = tf.depth_to_space(net,2)

y_pred = y_predt*255.0
y_pred = tf.maximum(y_pred,0)
y_pred = tf.minimum(y_pred,255)

dbatch=tf.concat([tf.cast(images,tf.float32),y_pred],0)


cost = tf.reduce_mean(tf.pow( tf.cast(images,tf.float32)/255.0  - y_predt, 2))
optimizer = tf.train.AdamOptimizer(0.000001 ).minimize(cost)
training_epochs =150000
display_step =200


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

#启动队列
tf.train.start_queue_runners(sess=sess)

# 启动循环开始训练
for epoch in range(training_epochs):
    
    _, c = sess.run([optimizer, cost])

    # 显示训练中的详细信息
    if epoch % display_step == 0:
        d_batch=dbatch.eval()
        mse,psnr=batch_mse_psnr(d_batch)
        ypsnr=batch_y_psnr(d_batch)
        ssim=batch_ssim(d_batch)
        print("Epoch:", '%04d' % (epoch+1),
              "cost=", "{:.9f}".format(c),"psnr",psnr,"ypsnr",ypsnr,"ssim",ssim)

print("完成!")


imagesv, label_batch,x_smallv,x_nearestv,x_bilinv,x_bicubicv,y_predv = sess.run([images, labels,x_smalls,x_nearests,x_bilins,x_bicubics,y_pred])
print("原",np.shape(imagesv),"缩放后的",np.shape(x_smallv),label_batch)

###显示
plt.figure(figsize=(20,10))  

showresult(161,"org",imagesv,imagesv,False)
showresult(162,"small/4",imagesv,x_smallv,False)
showresult(163,"near",imagesv,x_nearestv)
showresult(164,"biline",imagesv,x_bilinv)
showresult(165,"bicubicv",imagesv,x_bicubicv)
showresult(166,"pred",imagesv,y_predv)

    
plt.show()


#
## 可视化结果
#plt.figure(figsize=(20,10))  
#p1 = plt.subplot(161)
#p2 = plt.subplot(162)
#p3 = plt.subplot(163) 
#p4 = plt.subplot(164) 
#p5 = plt.subplot(165) 
#p6 = plt.subplot(166) 
#p1.axis('off') 
#p2.axis('off') 
#p3.axis('off') 
#p4.axis('off') 
#p5.axis('off') 
#p6.axis('off') 
#
#
#p1.imshow(imagesv[0])# 显示图片
#p2.imshow(np.asarray(x_smallv[0], dtype='uint8') )# 显示图片,必须转成uint8才能打印出来  
#p3.imshow(np.asarray(x_nearestv[0], dtype='uint8')  )# 显示图片
#p4.imshow(np.asarray(x_bilinv[0], dtype='uint8') )# 显示图片
#p5.imshow(np.asarray(x_bicubicv[0], dtype='uint8') )# 显示图片
#p6.imshow(np.asarray(y_predv[0], dtype='uint8') )# 显示图片
#
#p1.set_title("org")
#p2.set_title("small/4")
#p3.set_title("near")
#p4.set_title("biline")
#p5.set_title("bicubicv")
#p6.set_title("pred")    
#plt.show()


