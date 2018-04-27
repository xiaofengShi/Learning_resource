# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 19:36:41 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""

import numpy as np
import tensorflow as tf
import random
import collections
from collections import Counter
import jieba

from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.sans-serif']=['SimHei']#用来正常显示中文标签  
mpl.rcParams['font.family'] = 'STSong'
mpl.rcParams['font.size'] = 20


training_file = '人体阴阳与电能.txt'
    
#中文字
def get_ch_lable(txt_file):  
    labels= ""
    with open(txt_file, 'rb') as f:
        for label in f: 
            #labels =label.decode('utf-8')
            labels =labels+label.decode('gb2312')
           
    return  labels
   
#分词
def fenci(training_data):
    seg_list = jieba.cut(training_data)  # 默认是精确模式  
    training_ci = " ".join(seg_list)
    training_ci = training_ci.split()
    #以空格将字符串分开
    training_ci = np.array(training_ci)
    training_ci = np.reshape(training_ci, [-1, ])
    return training_ci


def build_dataset(words, n_words):

  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

    
training_data =get_ch_lable(training_file)
print("总字数",len(training_data))
training_ci =fenci(training_data)
#print(training_ci)
print("总词数",len(training_ci))    
training_label, count, dictionary, words = build_dataset(training_ci, 350)

words_size = len(dictionary)
print("字典词数",words_size)
#print(training_label)#将文本转为词向量 
#print(words)#每个编号对应的词
#print(dictionary)#每个词对应的编号
#print(count)#每个词对应的个数
####################################################
print('Sample data', training_label[:10], [words[i] for i in training_label[:10]])
data_index = 0
def generate_batch(data,batch_size, num_skips, skip_window):

  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window

  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)

  if data_index + span > len(data):
    data_index = 0

  buffer.extend(data[data_index:data_index + span])
  data_index += span

  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)

      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]

    if data_index == len(data):
      #print(data_index,len(data),span,len(data[:span]))
      #buffer[:] = data[:span]
      buffer = data[:span]
      data_index = span
    else:
      buffer.append(data[data_index])
      data_index += 1

  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

batch, labels = generate_batch(training_label,batch_size=8, num_skips=2, skip_window=1)

for i in range(8):# 取第一个字，后一个是标签，再取其前一个字当标签，
  print(batch[i], words[batch[i]], '->', labels[i, 0], words[labels[i, 0]])



batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

valid_size = 16     # Random set of words to evaluate similarity on.
valid_window =np.int32( words_size/2 ) # Only pick dev samples in the head of the distribution.
print("valid_window",valid_window)
valid_examples = np.random.choice(valid_window, valid_size, replace=False)#0-words_size/2,中的数取16个。不能重复。
num_sampled = 64    # Number of negative examples to sample.

tf.reset_default_graph()

train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# Ops and variables pinned to the CPU because of missing GPU implementation
with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(tf.random_uniform([words_size, embedding_size], -1.0, 1.0))#94个，每个128个向量
    
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable( tf.truncated_normal([words_size, embedding_size],
                            stddev=1.0 / tf.sqrt(np.float32(embedding_size))))
                            
    nce_biases = tf.Variable(tf.zeros([words_size]))


# Compute the average NCE loss for the batch.
# tf.nce_loss automatically draws a new sample of the negative labels each
# time we evaluate the loss.
loss = tf.reduce_mean(
tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                 labels=train_labels, inputs=embed,
                 num_sampled=num_sampled, num_classes=words_size))


#loss = tf.reduce_mean(
#tf.nn.sampled_softmax_loss(weights=nce_weights, biases=nce_biases,
#                          labels=train_labels, inputs=embed,
#                num_sampled=num_sampled, num_classes=words_size))



# Construct the SGD optimizer using a learning rate of 1.0.
optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

# Compute the cosine similarity between minibatch examples and all embeddings.
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
print("________________________",similarity.shape)


#Begin training.
num_steps = 100001
with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )
    print('Initialized')
    
    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(training_label, batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val
        
#通过打印测试可以看到  embed的值在逐渐的被调节      
#        emv = sess.run(embed,feed_dict = {train_inputs: [37,18]})
#        print("emv-------------------",emv[0])

        if step % 2000 == 0:
          if step > 0:
            average_loss /= 2000
          # The average loss is an estimate of the loss over the last 2000 batches.
          print('Average loss at step ', step, ': ', average_loss)
          average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        
        if step % 10000 == 0:
          sim = similarity.eval(session=sess)
          #print(valid_size)
          for i in range(valid_size):
            valid_word = words[valid_examples[i]]
            #print("valid_word",valid_word)#16
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]  #argsort函数返回的是数组值从小到大的索引值
            #print("nearest",nearest,top_k)
            log_str = 'Nearest to %s:' % valid_word

            for k in range(top_k):
              close_word = words[nearest[k]]
              log_str = '%s,%s' % (log_str, close_word)
            print(log_str)
        
    final_embeddings = normalized_embeddings.eval()


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,xy=(x, y),xytext=(5, 2), textcoords='offset points',
                     ha='right',va='bottom')
    plt.savefig(filename)

try:
  # pylint: disable=g-import-not-at-top
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 80#输出100个词
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [words[i] for i in range(plot_only)]
      #print(labels)
    plot_with_labels(low_dim_embs, labels)

except ImportError:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
















