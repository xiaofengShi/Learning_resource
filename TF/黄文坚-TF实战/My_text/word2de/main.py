# _*_ coding:utf-8 _*_
# Author:Xiaofeng Shi

import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib.request
import tensorflow as tf

url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        print(url + filename)
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
        print('downloading done')
    if os.path.exists(filename):
        print(filename, 'has already exist!!!')
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify' + filename +
                        '. can you get it with a browser?')
    return filename


filename = maybe_download('text8.zip', 31344016)


# extract data
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


words = read_data(filename)
# print('Data_size', len(words))

# vocabulary made
vocabulary_size = 50000


# create the dataset
def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    # print('count_size,sorted by frequency ', np.shape(count))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    # print('key:value--word:frequency', dictionary)
    # dictionary: key是count中的word，value是按照词出现的频数进行从大到小排序，当前word对应的位置
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            # 如果当前word在前5000个出现的词汇中，index为当前词汇出现的频数的排序所得的位置
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    # data中为每个word出现的频次，并按照word在words中从前向后的顺序存储
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


# count:选取整个文本中的所有单子，统计出现的次数，并按照出现的次数从多到少进行排序，
# ------得到单词出现的次数，对于50000个之外的单词，进行数量统计，存在第一位
# dictionary：存储为前50000个筛选出来的单词(key)以及按照出现次数从多到少进行编号(value)
# data:在输入的words文本中，按照顺序读入，按照dictionary字符的编号，data存储为单词的编号
# reverse_dictionary：对dictionary的key和value进行反转

# count

data, count, dictionary, reverse_dictionary = build_dataset(words)

# print('most common words (+UNK)', count[:5])
# print('sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
# print('sample words', words[:20], [dictionary[i] for i in words[:20]])
# print('data_length', np.shape(data))
del words

data_index = 0


def generate_bacth(batch_size, num_skips, skip_window):
    '''
    :param batch_size: how many samples in a batch
    :param num_skips: how many samples were made for a single word .
         it can not be bigger than the parameter'skip_window' for two times,and the parameter'batch_size' must be
         integer multiple of it(ensure a batch has the all the samples for a single word).
    :param skip_window: the distance of  a single word can contact with others
    :return:
    '''
    global data_index  # ensure the parameter'data_index' can be edited when refer to this function
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    # =======================================
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # how many words can be used when create samples for a single word
    span = 2 * skip_window + 1
    # create the deque(双向队列，最大容量为span)
    buffer = collections.deque(maxlen=span)
    # 将span个单词读入buffer中作为初始值，buffer开始定义的容量为span
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # 按照每个单词进行样本的制作
    for i in range(batch_size // num_skips):
        # buffer中的第skip_window个变量是目标单词
        target = skip_window
        # 定义生成样本时需要避免的单词列表，列表中一开始包括第skip_window个单词(目标单词)，因为要预测的是语境单词，不包括目标单词本身
        target_to_avoid = [skip_window]
        for j in range(num_skips):
            # 每次循环对一个语境单词生成样本，先产生随机数，直到随机数不在target_to_avoid中，代表可以使用的语境单词
            while target in target_to_avoid:
                target = random.randint(0, span - 1)
            target_to_avoid.append(target)
            # bacth为目标词汇
            # label为在目标词汇周围skip_window范围内的词汇
            # 区别于以前的多个输入对应一个输出；该训练集为一个相同的输入对应周围skip_window个label
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        # 一个单词生成num_skip个样本，循环完成后，对目标词汇进行更新，buffer加入一个新的词汇，原来的第一个词汇被挤掉，实现对下一个目标词汇的更新
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


# batch, labels = generate_bacth(batch_size=8, num_skips=2, skip_window=1)
# print(np.shape(batch), np.shape(labels))
# for i in range(8):
#     print('NO', i, '---', batch[i], reverse_dictionary[batch[i]], '-->', labels[i, 0], reverse_dictionary[labels[i, 0]])

batch_size = 128
# 将单词转化为稠密向量的维度，一般为50~1000
embedding_size = 128
skip_window = 1
num_skip = 2
# 生成验证数据
valid_size = 16  # 抽取的验证单词数
valid_window = 200  # 验证单词只从频数最高的100个单词中抽取
valid_examples = np.random.choice(
    valid_window, valid_size, replace=False)  # 从0-100之间随机选取16个数，返回数组
num_sampled = 64  # 训练是用作负样本的噪声单词的数量

graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(dtype=tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    with tf.device('/cpu:0'):
        # 生成词向量
        embeddings = tf.Variable(tf.random_uniform(
            [vocabulary_size, embedding_size], -1, 1.0))
        # 查找输入的train_inputs对应的向量
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_bias = tf.Variable(tf.zeros([vocabulary_size]))
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_bias,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    merge = tf.summary.merge_all()
    # accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax())))
    # 嵌入向量embedding的L2范数
    norm = tf.sqrt(tf.reduce_sum(
        tf.square(embeddings), axis=1, keep_dims=True))
    # 标准化
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)
    init = tf.global_variables_initializer()

# train
num_steps = 100001

with tf.Session(graph=graph) as session:
    init.run()
    saver = tf.train.Saver(max_to_keep=1)
    summary_writer = tf.summary.FileWriter('./log/summary')
    print('initialized')
    avg_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_bacth(
            batch_size, num_skip, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        _, loss_val, summary = session.run(
            [optimizer, loss, merge], feed_dict=feed_dict)
        summary_writer.add_summary(summary=summary, global_step=step)
        avg_loss += loss_val
        if step % 200 == 0:
            if step > 0:
                avg_loss /= 200
            print('average loss at step', step, ':', avg_loss)
            saver.save(session, './log/out/save', global_step=step % 200)
        if step % 5000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s , %s' % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()

# import matplotlib.pyplot as plt


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

    # %%


try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 200
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels)


except ImportError:
    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
