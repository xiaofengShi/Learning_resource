# _*_ coding:utf-8 _*_
# AUTHOR:Xiaofeng Shi
# writhe the inception net v3 in my own way

import tensorflow as tf

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
slim = tf.contrib.slim


# define the scope
# 对各种参数进行定义
def arg_scope(weight_decay=0.04,
              stddev=0.1,
              batch_norm_var_collection='moving_vars'):
    # 对bn进行参数定义--批规范化
    batch_norm_params = {'decay': 0.997,  # 衰减系数
                         'epsilon': 0.001,  #
                         'updates_collections': tf.GraphKeys.UPDATE_OPS,
                         'variables_collections': {'beta': None,
                                                   'gama': None,
                                                   'moving_mean': [batch_norm_var_collection],
                                                   'moving_variance': [batch_norm_var_collection]}}
    # 卷积和全连接的权重正则化进行定义
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        # 卷积的初始参数进行定义
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=tf.truncated_normal_initializer(stddev),
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params) as sc:
            return sc


def InceptionNet_v3_frame(input, scope=None):
    '''
    对整体的网络结构进行搭建
    1.5个卷积+2个pool层交替的普通结构，输出为35*35*192
    2.三个inception模块层，逐个模块，逐个module向前传播
    >模块1：包含3个inception module，这个模块不对feature map的尺寸进行压缩,每个module的输出为35*35*(256/288/288)；
    >模块2：包含5个inception module，这个模块对feature map进行尺寸压缩，输出分别为17*17*(768/768/768/768/192),逐个module传递；
    >模块3：包含3个inception module，该模块继续对feature map进行压缩，输出分别为8*8*(1280/2048/2048)；
    a.将模块2的输出进行保存，作为分类的feature map 分析层
    b.在模块2中引入了1xn，nx1的卷积方式，但是均为单向传递
    c.在模块3中引入了级联卷积方式，使用tf.concat()将conv1x3和conv3x1在输出通道维度上进行连接，并将连接后的结果作为一个整体
    d.在各个模块中，各个module之间均为前向传递，在每个module的分支之间，均为级联，并通过tf.concat的合并经过作为当前module的输出。
    e.inception module 的设计原则是图片的尺寸不断被压缩，与此同时，输出的通道数不断增加。
    f.在inception module中的分支结构一般为，branch0：conv1x1，branch1：conv1x1再接1xn和nx1卷积，branch2：与branch1类似，一般通道数更多，
      branch3：一般为最大或者平均池化，再接1x1卷积；也就是每一个module是将简单的特征抽象，比较复杂的特征抽象，简化结构的池化进行特征的提取。
    '''
    end_points = {}
    with tf.variable_scope(scope, 'Inception', [input]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='VALID'):
            # 299*299*3
            net = slim.conv2d(input, 32, 3, stride=2, scope='conv1')
            # ceil(（229-3+1）/2)=149
            # 149*149*32
            net = slim.conv2d(net, 32, 3, scope='conv2')
            # 147*147*32
            net = slim.conv2d(net, 64, 3, padding='SAME', scope='conv3')
            # 147*147*64
            net = slim.max_pool2d(net, 3, stride=2, scope='maxpool1')
            # ceil((147-3+1)/2)=73
            # 73*73*64
            net = slim.conv2d(net, 80, 1, scope='conv4')
            # 73*73*80
            net = slim.conv2d(net, 192, 3, scope='conv5')
            # ceil((73-3+1)/1)
            # 71*71*192
            net = slim.max_pool2d(net, 3, stride=2, scope='maxpool2')
            # ceil((71-3+1)/2)=35
            # 35*35*192
        #  inception 1
        # inception module 1
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            # ====================================================================#
            # inception 模块1，包含3个inception module
            # ===================================================================#
            # 35*35*256
            with tf.variable_scope('inception_module_11'):
                with tf.variable_scope('baracn_0'):
                    branch_0 = slim.conv2d(net, 64, 1, scope='module10_conv1')
                with tf.variable_scope('branch_1'):
                    branch_1 = slim.conv2d(net, 48, 1, scope='module11_conv1')
                    branch_1 = slim.conv2d(branch_1, 64, 5, scope='module11_conv2')
                with tf.variable_scope('branch_2'):
                    branch_2 = slim.conv2d(net, 64, 1, scope='module12_conv1')
                    branch_2 = slim.conv2d(branch_2, 96, 3, scope='module12_conv2')
                    branch_2 = slim.conv2d(branch_2, 96, 3, scope='module12_conv3')
                with tf.variable_scope('branch_3'):
                    branch_3 = slim.avg_pool2d(net, 3, scope='module13_avgpool1')
                    branch_3 = slim.conv2d(branch_3, 32, 1, scope='module13_conv1')
                # 将几个分支在输出通道的维度上进行合并
                # 几个分支采用的步长均为1，padding为SAME，没有减小输入图片的尺寸，仍未35*35；
                # 输出通道数为64+64+96+32=256
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                # 输出的feature map为 35*35*256
            # 35*35*288
            with tf.variable_scope('inception_module_12'):
                with tf.variable_scope('baracn_0'):
                    branch_0 = slim.conv2d(net, 64, 1, scope='module20_conv1')
                with tf.variable_scope('branch_1'):
                    branch_1 = slim.conv2d(net, 48, 1, scope='module21_conv1')
                    branch_1 = slim.conv2d(branch_1, 64, 5, scope='module21_conv2')
                with tf.variable_scope('branch_2'):
                    branch_2 = slim.conv2d(net, 64, 1, scope='module22_conv1')
                    branch_2 = slim.conv2d(branch_2, 96, 3, scope='module22_conv2')
                    branch_2 = slim.conv2d(branch_2, 96, 3, scope='module22_conv3')
                with tf.variable_scope('branch_3'):
                    branch_3 = slim.avg_pool2d(net, 3, scope='module23_avgpool1')
                    # 与'inception_module_1'不同的地方在于，branch_3的输出为64的conv1*1
                    branch_3 = slim.conv2d(branch_3, 64, 1, scope='module23_conv1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                # 输出为35*35*288
            # 'inception_module_3'和'inception_module_2'完全相同
            # 35*35*288
            with tf.variable_scope('inception_module_13'):
                with tf.variable_scope('baracn_0'):
                    branch_0 = slim.conv2d(net, 64, 1, scope='module30_conv1')
                with tf.variable_scope('branch_1'):
                    branch_1 = slim.conv2d(net, 48, 1, scope='module31_conv1')
                    branch_1 = slim.conv2d(branch_1, 64, 5, scope='module31_conv2')
                with tf.variable_scope('branch_2'):
                    branch_2 = slim.conv2d(net, 64, 1, scope='module32_conv1')
                    branch_2 = slim.conv2d(branch_2, 96, 3, scope='module32_conv2')
                    branch_2 = slim.conv2d(branch_2, 96, 3, scope='module32_conv3')
                with tf.variable_scope('branch_3'):
                    branch_3 = slim.avg_pool2d(net, 3, scope='module33_avgpool1')
                    # 与'inception_module_1'不同的地方在于，branch_3的输出为64的conv1*1
                    branch_3 = slim.conv2d(branch_3, 64, 1, scope='module33_conv1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            # 上面三个module 构成第一个inception模块，输出尺寸分别为35*35*(256/288/288)
            # 经过上面的几个inception module 得到的feature map尺寸为35*35*288
            # ====================================================================#
            # inception 模块2，接inception模块1继续进行,在该模块中存在5个inception module
            # ====================================================================#
            with tf.variable_scope('inception_module_21'):
                with tf.variable_scope('branch_0'):
                    branch_0 = slim.conv2d(net, 384, 3, stride=2, padding='VALID', scope='module40_conv1')
                    # 17*17*384
                with tf.variable_scope('branch_1'):
                    branch_1 = slim.conv2d(net, 64, 1, scope='module41_conv1')
                    branch_1 = slim.conv2d(branch_1, 96, 3, scope='module41_conv2')
                    branch_1 = slim.conv2d(branch_1, 96, 3, stride=2, padding='VALID', scope='module41_conv3')
                    # 17*17*96
                with tf.variable_scope('branch_2'):
                    branch_2 = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='module42_conv1')
                    # 17*17*288
                net = tf.concat([branch_0, branch_1, branch_2], 3)
                # 17*17*768
            with tf.variable_scope('inception_module_22'):
                with tf.variable_scope('branch_0'):
                    branch_0 = slim.conv2d(net, 192, 1, scope='module50_conv1')
                    # 17*17*192
                with tf.variable_scope('branch_1'):
                    branch_1 = slim.conv2d(net, 128, 1, scope='module51_conv2')
                    branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope='module51_conv1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='module51_conv7x1')
                    # 17*17*192
                with tf.variable_scope('branch_2'):
                    branch_2 = slim.conv2d(net, 128, 1, scope='module52_conv1')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='module52_conv7x1a')
                    branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='module52_conv1x7b')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='module52_conv7x1c')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='module52_conv1x7d')
                    # 17*17*192
                with tf.variable_scope('branch_3'):
                    branch_3 = slim.avg_pool2d(net, 3, scope='module53_avgpool')
                    branch_3 = slim.conv2d(branch_3, 192, 1, scope='module53_conv1')
                    # 17*17*192
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                # 本module 输出为17*17*768
            # 与'indeption 2'的区别在于将'branch2，branch3'的通道数进行了修改
            with tf.variable_scope('inception_23'):
                with tf.variable_scope('branch_0'):
                    branch_0 = slim.conv2d(net, 192, 1, scope='module60_conv1')
                    # 17*17*192
                with tf.variable_scope('branch_1'):
                    branch_1 = slim.conv2d(net, 160, 1, scope='module61_conv2')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='module61_conv1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='module61_conv7x1')
                    # 17*17*192
                with tf.variable_scope('branch_2'):
                    branch_2 = slim.conv2d(net, 160, 1, scope='module62_conv1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='module62_conv7x1a')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='module62_conv1x7b')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='module62_conv7x1c')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='module62_conv1x7d')
                    # 17*17*192
                with tf.variable_scope('branch_3'):
                    branch_3 = slim.avg_pool2d(net, 3, scope='module63_avgpool')
                    branch_3 = slim.conv2d(branch_3, 192, 1, scope='module63_conv1')
                    # 17*17*192
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                # 本module的输出为17*17*768
            with tf.variable_scope('inception_24'):
                with tf.variable_scope('branch_0'):
                    branch_0 = slim.conv2d(net, 192, 1, scope='module70_conv1')
                    # 17*17*192
                with tf.variable_scope('branch_1'):
                    branch_1 = slim.conv2d(net, 160, 1, scope='module71_conv2')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='module71_conv1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='module71_conv7x1')
                    # 17*17*192
                with tf.variable_scope('branch_2'):
                    branch_2 = slim.conv2d(net, 160, 1, scope='module72_conv1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='module72_conv7x1a')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='module72_conv1x7b')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='module72_conv7x1c')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='module72_conv1x7d')
                    # 17*17*192
                with tf.variable_scope('branch_3'):
                    branch_3 = slim.avg_pool2d(net, 3, scope='module73_avgpool')
                    branch_3 = slim.conv2d(branch_3, 192, 1, scope='module73_conv1')
                    # 17*17*192
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                # 本module的输出为17*17*768
            with tf.variable_scope('inception_25'):
                with tf.variable_scope('branch_0'):
                    branch_0 = slim.conv2d(net, 192, 1, scope='module80_conv1')
                    # 17*17*192
                with tf.variable_scope('branch_1'):
                    branch_1 = slim.conv2d(net, 160, 1, scope='module81_conv2')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='module81_conv1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='module81_conv7x1')
                    # 17*17*192
                with tf.variable_scope('branch_2'):
                    branch_2 = slim.conv2d(net, 160, 1, scope='module82_conv1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='module82_conv7x1a')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='module82_conv1x7b')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='module82_conv7x1c')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='module82_conv1x7d')
                    # 17*17*192
                with tf.variable_scope('branch_3'):
                    branch_3 = slim.avg_pool2d(net, 3, scope='module83_avgpool')
                    branch_3 = slim.conv2d(branch_3, 192, 1, scope='module83_conv1')
                    # 17*17*192
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                # 本module的输出为17*17*768
            # 将该层的信心进行保存，进行整体模型的后续辅助分类计算
            end_points['inception_25'] = net
            # ====================================================================#
            # inception 模块3，接inception模块2继续进行,在该模块中存在3个inception module
            # ====================================================================#
            with tf.variable_scope('inception_31'):
                with tf.variable_scope('branch_0'):
                    branch_0 = slim.conv2d(net, 192, 1, scope='module90_conv1')
                    branch_0 = slim.conv2d(branch_0, 320, 3, stride=2, padding='VALID', scope='module90_conv2')
                    # 输出为：8*8*320
                with tf.variable_scope('branch_1'):
                    branch_1 = slim.conv2d(net, 192, 1, scope='module91_conv1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='module91_conv1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='module91_conv7x1')
                    branch_1 = slim.conv2d(branch_1, 192, 3, stride=2, padding='VALID', scope='module91_conv3')
                    # 输出为：8*8*192
                with tf.variable_scope('branc_2'):
                    branch_2 = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='module92_maxpool1')
                    # 输出为：8*8768
                net = tf.concat([branch_0, branch_1, branch_2], 3)
                # 该module输出为：8*8*1280
            with tf.variable_scope('inception_32'):
                with tf.variable_scope('branch_0'):
                    branch_0 = slim.conv2d(net, 320, 1, scope='module100_conv1')
                    # 输出为：8*8*320
                with tf.variable_scope('branch_1'):
                    branch_1 = slim.conv2d(net, 384, 1, scope='module101_conv1')
                    # 输出为：8*8*384
                    # 进行两个卷积的合并，在输出通道的维度上进行合并
                    branch_1 = tf.concat([slim.conv2d(branch_1, 384, [1, 3], scope='module101_conv21'),
                                          slim.conv2d(branch_1, 384, [3, 1], scope='module101_conv22')], 3)
                    # 输出为：8*8*768
                with tf.variable_scope('branch_2'):
                    branch_2 = slim.conv2d(net, 448, 1, scope='module102_conv1')
                    branch_2 = slim.conv2d(branch_2, 384, 3, scope='module102_conv2')

                    branch_2 = tf.concat([slim.conv2d(branch_2, 384, [1, 3], scope='module102_conv21'),
                                          slim.conv2d(branch_2, 384, [3, 1], scope='module102_conv22')], 3)
                    # 输出为： 8*8*768
                with tf.variable_scope('branch_3'):
                    branch_3 = slim.avg_pool2d(net, 3, scope='module103_avgpool1')
                    branch_3 = slim.conv2d(branch_3, 192, 1, scope='module103_conv1')
                    # 输出为：8*8*192
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                # 输出为： 8*8*2048
            with tf.variable_scope('inception_33'):
                with tf.variable_scope('branch_0'):
                    branch_0 = slim.conv2d(net, 320, 1, scope='module110_conv1')
                    # 输出为：8*8*320
                with tf.variable_scope('branch_1'):
                    branch_1 = slim.conv2d(net, 384, 1, scope='module111_conv1')
                    # 输出为：8*8*384
                    # 进行两个卷积的合并，在输出通道的维度上进行合并
                    branch_1 = tf.concat([slim.conv2d(branch_1, 384, [1, 3], scope='module111_conv21'),
                                          slim.conv2d(branch_1, 384, [3, 1], scope='module111_conv22')], 3)
                    # 输出为：8*8*768
                with tf.variable_scope('branch_2'):
                    branch_2 = slim.conv2d(net, 448, 1, scope='module112_conv1')
                    branch_2 = slim.conv2d(branch_2, 384, 3, scope='module112_conv2')

                    branch_2 = tf.concat([slim.conv2d(branch_2, 384, [1, 3], scope='module112_conv21'),
                                          slim.conv2d(branch_2, 384, [3, 1], scope='module112_conv22')], 3)
                    # 输出为： 8*8*768
                with tf.variable_scope('branch_3'):
                    branch_3 = slim.avg_pool2d(net, 3, scope='module113_avgpool1')
                    branch_3 = slim.conv2d(branch_3, 192, 1, scope='module113_conv1')
                    # 输出为：8*8*192
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                # 输出为：8*8*2048
            return net, end_points


num_classes = 100


def Inception_v3_final(input,
                       num_classes=num_classes,
                       is_training=True,  # 表示是否为训练过程，该状态影响BN和dropout
                       dropout_prob=0.8,
                       prediction_fn=slim.softmax,
                       squeeze_or_not=True,  # 是否对输出进行去除维度为1的操作，比如讲5x2x1变成5x2
                       reuse=None,  # 是否对网络和variable进行重复使用
                       scope='Inception'):  # scope为包含默认参数的环境
    with tf.variable_scope(scope, 'Inception', [input, num_classes], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net, end_points = InceptionNet_v3_frame(input, scope=scope)
            # net为8*8*2048
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            axu_logits = end_points['inception_25']
            # 接下来对该层的输出进行进一步的特征提取及维度处理
            # 17*17*768
            with tf.variable_scope('auxlogits'):
                aux_logits = slim.max_pool2d(axu_logits, 5, stride=3, padding='VALID', scope='maxpool5x5')
                # 5*5*768
                aux_logits = slim.conv2d(aux_logits, 128, 1, scope='conv1x1')
                # 5*5*128
                axu_logits = slim.conv2d(aux_logits, 768, 5,
                                         # 默认的weights_initializer已经在arg_scope中进行了设置，在本层feature map要进行参数的修改
                                         weights_initializer=tf.truncated_normal_initializer(0.01),
                                         padding='VALID',
                                         scope='conv5x5')
                # 1x1x768
                axu_logits = slim.conv2d(axu_logits, num_classes, 1, activation_fn=None, normalizer_fn=None,
                                         weights_initializer=tf.truncated_normal_initializer(0.001),
                                         scope='conv1x1_out')
                # 1x1xnum_classes
                # 对前两个为1的维度进行去除，也就是将整个铺平，是不是可以使用slim.flatten
                if squeeze_or_not:
                    axu_logits = tf.squeeze(axu_logits, [1, 2], name='squeeze')
                    # aux_logits=slim.flatten(axu_logits,scope='flatten')
                end_points['axu_logits'] = axu_logits
        with tf.variable_scope('logits'):
            net = slim.avg_pool2d(net, 8, padding='VALID', scope='avgpool8x8')
            # 1x1x2048
            net = slim.dropout(net, keep_prob=dropout_prob, is_training=is_training, scope='dropout')
            end_points['prelogits'] = net
            logits = slim.conv2d(net, num_classes, 1, activation_fn=None, normalizer_fn=None,
                                 scope='conv1x1xnum_classes')
            if squeeze_or_not:
                logits = tf.squeeze(logits, [1, 2], name='squeeze')
            end_points['logits'] = logits
            end_points['prediction'] = prediction_fn(logits, scope='prediction')
    return logits, end_points


from datetime import datetime
import math
import time


def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                      (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, num_batches, mn, sd))


batch_size = 32
height, width = 299, 299
inputs = tf.random_uniform((batch_size, height, width, 3))
with slim.arg_scope(arg_scope()):
    logits, end_points = Inception_v3_final(inputs, is_training=False)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
num_batches = 10
time_tensorflow_run(sess, logits, "Forward")
