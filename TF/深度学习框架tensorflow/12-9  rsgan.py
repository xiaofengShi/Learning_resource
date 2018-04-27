# -*- coding: utf-8 -*-
__author__ = "代码医生"
"""
Created on Mon Jul 24 09:59:30 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""

print("作者:",__author__)

import tensorflow as tf
import time
import os 
import numpy as np
import matplotlib.pyplot as plt
from nets import vgg
from datasets import flowers


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

images = tf.cast(images,tf.float32)
x_smalls=tf.image.resize_bicubic(images,[np.int32(height/4), np.int32(width/4)])#  缩小4*4倍
x_smalls2 = x_smalls/127.5-1 #（0--1）

#还原
x_nearests = tf.image.resize_images(x_smalls, (height, width),tf.image.ResizeMethod.NEAREST_NEIGHBOR)
x_bilins = tf.image.resize_images(x_smalls, (height, width),tf.image.ResizeMethod.BILINEAR)
x_bicubics = tf.image.resize_images(x_smalls, (height, width),tf.image.ResizeMethod.BICUBIC)


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

def gen(x_smalls2 ):
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
    return y_predt
#################################
def rgbmeanfun(rgb):
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    print("build model started")
    # Convert RGB to BGR
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb)
    rgbmean = tf.concat(axis=3, values=[red - _R_MEAN,green -_G_MEAN, blue - _B_MEAN,])
    return rgbmean
    
resnetimg=gen(x_smalls2)
result=(resnetimg+1)*127.5
gen_var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


y_pred = tf.maximum(result,0)
y_pred = tf.minimum(y_pred,255)


dbatch=tf.concat([images,result],0)
rgbmean = rgbmeanfun(dbatch)

#vgg 特征值
_, end_points = vgg.vgg_19(rgbmean, num_classes=1000,is_training=False,spatial_squeeze=False)                    
conv54=end_points['vgg_19/conv5/conv5_4']
print("vgg.conv5_4",conv54.shape)
fmap=tf.split(conv54,2)

content_loss=tf.losses.mean_squared_error(fmap[0],fmap[1])
######################################

def Discriminator(dbatch, name ="Discriminator"):
    with tf.variable_scope(name):
        net = slim.conv2d(dbatch, 64, 1,activation_fn = leaky_relu)

        ochannels=[64,128,128,256,256,512,512]
        stride=[2,1]

        for i in range(7):
            net = slim.conv2d(net, ochannels[i], 3,stride = stride[i%2],activation_fn = leaky_relu,normalizer_fn=slim.batch_norm,scope='block'+str(i))

        dense1 = slim.fully_connected(net, 1024, activation_fn=leaky_relu)
        dense2 = slim.fully_connected(dense1, 1, activation_fn=tf.nn.sigmoid)
        
        return dense2


disc=Discriminator(dbatch)
D_x,D_G_z=tf.split(tf.squeeze(disc),2) 
  
adv_loss=tf.reduce_mean(tf.square(D_G_z-1.0))

gen_loss=(adv_loss+content_loss)
disc_loss=(tf.reduce_mean(tf.square(D_x-1.0)+tf.square(D_G_z)))

disc_var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
print("len-----",len(disc_var_list),len(gen_var_list))
for x in gen_var_list:
    disc_var_list.remove(x)


learn_rate =0.001
global_step=tf.Variable(0,trainable=0,name='global_step')
gen_train_step=tf.train.AdamOptimizer(learn_rate).minimize(gen_loss,global_step,gen_var_list)
disc_train_step=tf.train.AdamOptimizer(learn_rate).minimize(disc_loss,global_step,disc_var_list)

#res 检查点
flags='b'+str(batch_size)+'_r'+str(np.int32(height/4))+'_r'+str(learn_rate)+'rsgan'
save_path='save/srgan_'+flags
if not os.path.exists(save_path):
    os.mkdir(save_path)
saver = tf.train.Saver(max_to_keep=1) # 生成saver

srResNet_path='./save/tf_b16_h64.0_r0.001_res/'
srResNetloader = tf.train.Saver(var_list=gen_var_list) # 生成saver

#vgg 检查点
checkpoints_dir = 'vgg_19_2016_08_28'
init_fn = slim.assign_from_checkpoint_fn(
    os.path.join(checkpoints_dir, 'vgg_19.ckpt'),
    slim.get_model_variables('vgg_19'))



log_steps=100
training_epochs=16000

with tf.Session() as sess:
 
    sess.run(tf.global_variables_initializer())  
    
    init_fn(sess)    
    
    kpt = tf.train.latest_checkpoint(srResNet_path)
    print("srResNet_path",kpt,srResNet_path)
    startepo= 0
    if kpt!=None:
        srResNetloader.restore(sess, kpt) 
        ind = kpt.find("-")
        startepo = int(kpt[ind+1:])
        print("srResNetloader global_step=",global_step.eval(),startepo)     
    

    kpt = tf.train.latest_checkpoint(save_path)
    print("srgan",kpt)
    startepo= 0
    if kpt!=None:
        saver.restore(sess, kpt) 
        ind = kpt.find("-")
        startepo = int(kpt[ind+1:])
        print("global_step=",global_step.eval(),startepo)      
      
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    
    try:
        def train(endpoint,gen_step,disc_step):
            #print(global_step.eval(),endpoint)
            while global_step.eval()<=endpoint:
                #print(global_step.eval(),global_step.eval()%log_steps)
                if((global_step.eval()/2)%log_steps==0):# 一次走两步
                    #print(global_step.eval(),log_steps)
                    d_batch=dbatch.eval()
                    mse,psnr=batch_mse_psnr(d_batch)
                    ssim=batch_ssim(d_batch)
                    s=time.strftime('%Y-%m-%d %H:%M:%S:',time.localtime(time.time()))+'step='+str(global_step.eval())+' mse='+str(mse)+' psnr='+str(psnr)+' ssim='+str(ssim)+' gen_loss='+str(gen_loss.eval())+' disc_loss='+str(disc_loss.eval())
                    print(s)
                    f=open('info.train_'+flags,'a')
                    f.write(s+'\n')
                    f.close()
                    saver.save(sess, save_path+"/srgan.cpkt", global_step=global_step.eval())
                    #save()
                sess.run(disc_step)
                sess.run(gen_step)
        train(training_epochs,gen_train_step,disc_train_step)
        print('训练完成')
 

    ###显示
        resultv,imagesv,x_smallv,x_nearestv,x_bilinv,x_bicubicv,y_predv = sess.run([result,images,x_smalls,x_nearests,x_bilins,x_bicubics,y_pred])
        print("原",np.shape(imagesv),"缩放后的",np.shape(x_smallv))
    

        conimg1 =  np.concatenate((imagesv,x_bilinv))
        ssim1=batch_ssim(conimg1)
        conimg2 =  np.concatenate((imagesv,y_predv))
        ssim2=batch_ssim(conimg2)  

    
        plt.figure(figsize=(20,10))  
        showresult(161,"org",imagesv,imagesv,False)
        showresult(162,"small/4",imagesv,x_smallv,False)
        showresult(163,"near",imagesv,x_nearestv)
        showresult(164,"biline",imagesv,x_bilinv)
        showresult(165,"bicubicv",imagesv,x_bicubicv)
        showresult(166,"pred",imagesv,y_predv)
        plt.show()
   
    
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    except KeyboardInterrupt:
        print("Ending Training...")
        saver.save(sess, save_path+"/srgan.cpkt", global_step=global_step.eval())
    finally:
        coord.request_stop()

    coord.join(threads)
