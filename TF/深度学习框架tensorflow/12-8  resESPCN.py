# -*- coding: utf-8 -*-
__author__ = "代码医生"
"""
Created on Fri Jun  2 10:53:51 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""

print("作者:",__author__)

import tensorflow as tf
from datasets import flowers
import numpy as np
import matplotlib.pyplot as plt
import os

slim = tf.contrib.slim

tf.reset_default_graph()

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
        p.set_title(title+str(int(psnr))+" y:"+str(int(ypsnr))+" s:"+" s:%.4f"%ssim)
    else:
        p.set_title(title)


height = width = 256
batch_size = 16
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

x_smalls = tf.image.resize_images(images, (np.int32(height/4), np.int32(width/4)))#  缩小4*4倍
x_smalls2 = x_smalls/255.0
#还原
x_nearests = tf.image.resize_images(x_smalls, (height, width),tf.image.ResizeMethod.NEAREST_NEIGHBOR)
x_bilins = tf.image.resize_images(x_smalls, (height, width),tf.image.ResizeMethod.BILINEAR)
x_bicubics = tf.image.resize_images(x_smalls, (height, width),tf.image.ResizeMethod.BICUBIC)

####################################
#net = slim.conv2d(x_smalls2, 64, 5,activation_fn = tf.nn.tanh)
#net =slim.conv2d(net, 256, 3,activation_fn = tf.nn.tanh)
#net = tf.depth_to_space(net,2) #64
#net =slim.conv2d(net, 64, 3,activation_fn = tf.nn.tanh)
#net = tf.depth_to_space(net,2) #16
#y_predt = slim.conv2d(net, 3, 3,activation_fn = None)#2*2*3

######################################
def leaky_relu(x,alpha=0.1,name='lrelu'):
     with tf.name_scope(name):
         x=tf.maximum(x,alpha*x)
         return x
def residual_block(nn,i,name='resblock'):
    with tf.variable_scope(name+str(i)):
        conv1=slim.conv2d(nn, 64, 3,activation_fn = leaky_relu,normalizer_fn=slim.batch_norm)
        conv2=slim.conv2d(conv1, 64, 3,activation_fn = leaky_relu,normalizer_fn=slim.batch_norm)
        return tf.add(nn,conv2)

net = slim.conv2d(x_smalls2, 64, 5,activation_fn = leaky_relu)
block=[]
for i in range(16):
    block.append(residual_block(block[-1] if i else net,i))
conv2=slim.conv2d(block[-1], 64, 3,activation_fn = leaky_relu,normalizer_fn=slim.batch_norm)
sum1=tf.add(conv2,net)

conv3=slim.conv2d(sum1, 256, 3,activation_fn = None)
ps1=tf.depth_to_space(conv3,2) 
relu2=leaky_relu(ps1)
conv4=slim.conv2d(relu2, 256, 3,activation_fn = None)
ps2=tf.depth_to_space(conv4,2)#再放大两倍 64
relu3=leaky_relu(ps2)
y_predt=slim.conv2d(relu3, 3, 3,activation_fn = None)#输出



y_pred = y_predt*255.0
y_pred = tf.maximum(y_pred,0)
y_pred = tf.minimum(y_pred,255)

dbatch=tf.concat([tf.cast(images,tf.float32),y_pred],0)

learn_rate =0.001

cost = tf.reduce_mean(tf.pow( tf.cast(images,tf.float32)/255.0  - y_predt, 2))
optimizer = tf.train.AdamOptimizer(learn_rate ).minimize(cost)
#training_epochs =100000
#display_step =5000

training_epochs =10000
display_step =400


flags='b'+str(batch_size)+'_h'+str(height/4)+'_r'+str(learn_rate)+'_res'#set for practicers to try different setups 
#flags='b'+str(batch_size)+'_r'+str(height/4)+'_depth_conv2d'#set for practicers to try different setups 
if not os.path.exists('save'):
    os.mkdir('save')
save_path='save/tf_'+flags
if not os.path.exists(save_path):
    os.mkdir(save_path)
saver = tf.train.Saver(max_to_keep=1) # 生成saver

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

kpt = tf.train.latest_checkpoint(save_path)
print(kpt)
startepo= 0
if kpt!=None:
    saver.restore(sess, kpt) 
    ind = kpt.find("-")
    startepo = int(kpt[ind+1:])
    print("startepo=",startepo)


#启动队列
tf.train.start_queue_runners(sess=sess)

# 启动循环开始训练
for epoch in range(startepo,training_epochs):
    
    _, c = sess.run([optimizer, cost])

    # 显示训练中的详细信息
    if epoch % display_step == 0:
        d_batch=dbatch.eval()
        mse,psnr=batch_mse_psnr(d_batch)
        ypsnr=batch_y_psnr(d_batch)
        ssim=batch_ssim(d_batch)
        print("Epoch:", '%04d' % (epoch+1),
              "cost=", "{:.9f}".format(c),"psnr",psnr,"ypsnr",ypsnr,"ssim",ssim)

        saver.save(sess, save_path+"/tfrecord.cpkt", global_step=epoch)
print("完成!")
saver.save(sess, save_path+"/tfrecord.cpkt", global_step=epoch)


imagesv, label_batch,x_smallv,x_nearestv,x_bilinv,x_bicubicv,y_predv = sess.run([images, labels,x_smalls,x_nearests,x_bilins,x_bicubics,y_pred])
print("原",np.shape(imagesv),"缩放后的",np.shape(x_smallv),label_batch)

#        print(np.max(imagesv[0]),np.max(x_bilinv[0]),np.max(x_bicubicv[0]),np.max(y_predv[0]))
#        print(np.min(imagesv[0]),np.min(x_bilinv[0]),np.min(x_bicubicv[0]),np.min(y_predv[0]))

###显示
plt.figure(figsize=(20,10))  

showresult(161,"org",imagesv,imagesv,False)
showresult(162,"small/4",imagesv,x_smallv,False)
showresult(163,"near",imagesv,x_nearestv)
showresult(164,"biline",imagesv,x_bilinv)
showresult(165,"bicubicv",imagesv,x_bicubicv)
showresult(166,"pred",imagesv,y_predv)

    
plt.show()





