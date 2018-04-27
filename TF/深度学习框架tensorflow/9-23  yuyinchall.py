# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 18:52:37 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""

import numpy as np
import time
import tensorflow as tf
from tensorflow.python.ops import ctc_ops
from collections import Counter
## 自定义
yuyinutils = __import__("9-24  yuyinutils")
sparse_tuple_to_texts_ch = yuyinutils.sparse_tuple_to_texts_ch
ndarray_to_text_ch = yuyinutils.ndarray_to_text_ch
get_audio_and_transcriptch = yuyinutils.get_audio_and_transcriptch
pad_sequences = yuyinutils.pad_sequences
sparse_tuple_from = yuyinutils.sparse_tuple_from
get_wavs_lables = yuyinutils.get_wavs_lables





tf.reset_default_graph()


b_stddev = 0.046875
h_stddev = 0.046875

n_hidden = 1024
n_hidden_1 = 1024
n_hidden_2 =1024
n_hidden_5 = 1024
n_cell_dim = 1024
n_hidden_3 = 2 * 1024

keep_dropout_rate=0.95
relu_clip = 20


def BiRNN_model( batch_x, seq_length, n_input, n_context,n_character ,keep_dropout):

    # batch_x_shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
    batch_x_shape = tf.shape(batch_x)

   
    # 将输入转成时间序列优先
    batch_x = tf.transpose(batch_x, [1, 0, 2])
    # 再转成2维传入第一层
    batch_x = tf.reshape(batch_x,
                         [-1, n_input + 2 * n_input * n_context])  # (n_steps*batch_size, n_input + 2*n_input*n_context)

    # 使用clipped RELU activation and dropout.
    # 1st layer
    with tf.name_scope('fc1'):
        b1 = variable_on_cpu('b1', [n_hidden_1], tf.random_normal_initializer(stddev=b_stddev))
        h1 = variable_on_cpu('h1', [n_input + 2 * n_input * n_context, n_hidden_1],
                             tf.random_normal_initializer(stddev=h_stddev))
        layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), relu_clip)
        layer_1 = tf.nn.dropout(layer_1, keep_dropout)

    # 2nd layer
    with tf.name_scope('fc2'):
        b2 = variable_on_cpu('b2', [n_hidden_2], tf.random_normal_initializer(stddev=b_stddev))
        h2 = variable_on_cpu('h2', [n_hidden_1, n_hidden_2], tf.random_normal_initializer(stddev=h_stddev))
        layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), relu_clip)
        layer_2 = tf.nn.dropout(layer_2, keep_dropout)


    # 3rd layer
    with tf.name_scope('fc3'):
        b3 = variable_on_cpu('b3', [n_hidden_3], tf.random_normal_initializer(stddev=b_stddev))
        h3 = variable_on_cpu('h3', [n_hidden_2, n_hidden_3], tf.random_normal_initializer(stddev=h_stddev))
        layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, h3), b3)), relu_clip)
        layer_3 = tf.nn.dropout(layer_3, keep_dropout)

    # 双向rnn
    with tf.name_scope('lstm'):
        # Forward direction cell:
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell,
                                                     input_keep_prob=keep_dropout)
        # Backward direction cell:
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell,
                                                     input_keep_prob=keep_dropout)

        # `layer_3`  `[n_steps, batch_size, 2*n_cell_dim]`
        layer_3 = tf.reshape(layer_3, [-1, batch_x_shape[0], n_hidden_3])

        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                 cell_bw=lstm_bw_cell,
                                                                 inputs=layer_3,
                                                                 dtype=tf.float32,
                                                                 time_major=True,
                                                                 sequence_length=seq_length)

        # 连接正反向结果[n_steps, batch_size, 2*n_cell_dim]
        outputs = tf.concat(outputs, 2)
        # to a single tensor of shape [n_steps*batch_size, 2*n_cell_dim]        
        outputs = tf.reshape(outputs, [-1, 2 * n_cell_dim])

    with tf.name_scope('fc5'):
        b5 = variable_on_cpu('b5', [n_hidden_5], tf.random_normal_initializer(stddev=b_stddev))
        h5 = variable_on_cpu('h5', [(2 * n_cell_dim), n_hidden_5], tf.random_normal_initializer(stddev=h_stddev))
        layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, h5), b5)), relu_clip)
        layer_5 = tf.nn.dropout(layer_5, keep_dropout)

    with tf.name_scope('fc6'):
        # 全连接层用于softmax分类
        b6 = variable_on_cpu('b6', [n_character], tf.random_normal_initializer(stddev=b_stddev))
        h6 = variable_on_cpu('h6', [n_hidden_5, n_character], tf.random_normal_initializer(stddev=h_stddev))
        layer_6 = tf.add(tf.matmul(layer_5, h6), b6)


    # 将2维[n_steps*batch_size, n_character]转成3维 time-major [n_steps, batch_size, n_character].
    layer_6 = tf.reshape(layer_6, [-1, batch_x_shape[0], n_character])

    # Output shape: [n_steps, batch_size, n_character]
    return layer_6

"""
used to create a variable in CPU memory.
"""    
def variable_on_cpu(name, shape, initializer):
    # Use the /cpu:0 device for scoped operations
    with tf.device('/cpu:0'):
        # Create or get apropos variable
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var
    

wav_path='D:/ data_thchs30/data_thchs30/train'
label_file='D: /data_thchs30/doc/trans/train.word.txt'

   
wav_files, labels = get_wavs_lables(wav_path,label_file) 
print(wav_files[0], labels[0])  
# wav/train/A11/A11_0.WAV -> 绿 是 阳春 烟 景 大块 文章 的 底色 四月 的 林 峦 更是 绿 得 鲜活 秀媚 诗意 盎然  

print("wav:",len(wav_files),"label",len(labels))


# 字表 
all_words = []  
for label in labels:  
    #print(label)    
    all_words += [word for word in label]  
counter = Counter(all_words)  
words = sorted(counter)
words_size= len(words)
word_num_map = dict(zip(words, range(words_size))) 

print('字表大小:', words_size) 
 

n_input = 26#计算美尔倒谱系数的个数
n_context = 9#对于每个时间点，要包含上下文样本的个数
batch_size =8
def next_batch(labels, start_idx = 0,batch_size=1,wav_files = wav_files):
    filesize = len(labels)
    end_idx = min(filesize, start_idx + batch_size)
    idx_list = range(start_idx, end_idx)
    txt_labels = [labels[i] for i in idx_list]
    wav_files = [wav_files[i] for i in idx_list]
    (source, audio_len, target, transcript_len) = get_audio_and_transcriptch(None,
                                                      wav_files,
                                                      n_input,
                                                      n_context,word_num_map,txt_labels)
    
    start_idx += batch_size
    # Verify that the start_idx is not larger than total available sample size
    if start_idx >= filesize:
        start_idx = -1

    # Pad input to max_time_step of this batch
    source, source_lengths = pad_sequences(source)#如果多个文件将长度统一，支持按最大截断或补0
    sparse_labels = sparse_tuple_from(target)

    return start_idx,source, source_lengths, sparse_labels

next_idx,source,source_len,sparse_lab = next_batch(labels,0,batch_size)
print(len(sparse_lab))
print(np.shape(source))
#print(sparse_lab)
t = sparse_tuple_to_texts_ch(sparse_lab,words)
print(t[0])
#source已经将变为前9（不够补空）+本身+后9，每个26，第一个顺序是第10个的数据。



# shape = [batch_size, max_stepsize, n_input + (2 * n_input * n_context)]
# the batch_size and max_stepsize每步都是变长的。
input_tensor = tf.placeholder(tf.float32, [None, None, n_input + (2 * n_input * n_context)], name='input')#语音log filter bank or MFCC features
# Use sparse_placeholder; will generate a SparseTensor, required by ctc_loss op.
targets = tf.sparse_placeholder(tf.int32, name='targets')#文本
# 1d array of size [batch_size]
seq_length = tf.placeholder(tf.int32, [None], name='seq_length')#序列长
keep_dropout= tf.placeholder(tf.float32)

# logits is the non-normalized output/activations from the last layer.
# logits will be input for the loss function.
# nn_model is from the import statement in the load_model function
logits = BiRNN_model( input_tensor, tf.to_int64(seq_length), n_input, n_context,words_size +1,keep_dropout)



#调用ctc loss
avg_loss = tf.reduce_mean(ctc_ops.ctc_loss(targets, logits, seq_length))


#[optimizer]
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(avg_loss)


with tf.name_scope("decode"):    
    decoded, log_prob = ctc_ops.ctc_beam_search_decoder( logits, seq_length, merge_repeated=False)
    

with tf.name_scope("accuracy"):
    distance = tf.edit_distance( tf.cast(decoded[0], tf.int32), targets)
    # 计算label error rate (accuracy)
    ler = tf.reduce_mean(distance, name='label_error_rate')
   

epochs = 100
savedir = "log/yuyinchalltest/"
saver = tf.train.Saver(max_to_keep=1) # 生成saver
# create the session
sess = tf.Session()
# 没有模型的话，就重新初始化
sess.run(tf.global_variables_initializer())

kpt = tf.train.latest_checkpoint(savedir)
print("kpt:",kpt)
startepo= 0
if kpt!=None:
    saver.restore(sess, kpt) 
    ind = kpt.find("-")
    startepo = int(kpt[ind+1:])
    print(startepo)

# 准备运行训练步骤
section = '\n{0:=^40}\n'
print(section.format('Run training epoch'))

train_start = time.time()
for epoch in range(epochs):#样本集迭代次数
    epoch_start = time.time()
    if epoch<startepo:
        continue
   
    print("epoch start:",epoch,"total epochs= ",epochs)
#######################run batch####
    n_batches_per_epoch = int(np.ceil(len(labels) / batch_size))
    print("total loop ",n_batches_per_epoch,"in one epoch，",batch_size,"items in one loop") 
    
    train_cost = 0
    train_ler = 0
    next_idx =0
    
    for batch in range(n_batches_per_epoch):#一次batch_size，取多少次
        #取数据
        next_idx,source,source_lengths,sparse_labels = \
            next_batch(labels,next_idx ,batch_size)
        feed = {input_tensor: source, targets: sparse_labels,seq_length: source_lengths,keep_dropout:keep_dropout_rate}
        
        #计算 avg_loss optimizer ;
        batch_cost, _ = sess.run([avg_loss, optimizer],  feed_dict=feed )
        train_cost += batch_cost 
             
        if (batch +1)%20 == 0:
            print('loop:',batch, 'Train cost: ', train_cost/(batch+1))
            feed2 = {input_tensor: source, targets: sparse_labels,seq_length: source_lengths,keep_dropout:1.0}

            d,train_ler = sess.run([decoded[0],ler], feed_dict=feed2)
            dense_decoded = tf.sparse_tensor_to_dense( d, default_value=-1).eval(session=sess)
            dense_labels = sparse_tuple_to_texts_ch(sparse_labels,words)
            
            counter =0
            print('Label err rate: ', train_ler)
            for orig, decoded_arr in zip(dense_labels, dense_decoded):
                # convert to strings
                decoded_str = ndarray_to_text_ch(decoded_arr,words)
                print(' file {}'.format( counter))
                print('Original: {}'.format(orig))
                print('Decoded:  {}'.format(decoded_str))
                counter=counter+1
                break
            
        
    epoch_duration = time.time() - epoch_start
    
    log = 'Epoch {}/{}, train_cost: {:.3f}, train_ler: {:.3f}, time: {:.2f} sec'
    print(log.format(epoch ,epochs, train_cost,train_ler,epoch_duration))
    saver.save(sess, savedir+"yuyinch.cpkt", global_step=epoch)
    
        
    
train_duration = time.time() - train_start
print('Training complete, total duration: {:.2f} min'.format(train_duration / 60))

sess.close()   


    

