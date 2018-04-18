# _*_ coding:utf-8 _*_
# Author: Xiaofeng Shi
# Data:2018.02.15


import time
import numpy as np
import tensorflow as tf
import reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size  # 每个batch包含的样本数
        self.num_steps = num_steps = config.num_steps  # LSTM的展开步数
        self.epoch_size = (len(data) // batch_size - 1) // num_steps  # 每个epoch内需要多少轮迭代
        self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name=name)  # 获得input和labels


class PTBModel(object):

    def __init__(self, is_training, config, input_):
        self._input = input_

        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size  # lstm节点数量
        vocab_size = config.vocab_size  # 词汇表的大小

        # 定义LSTM默认单元，隐含节点为 hidden size，forget_bias表示forget gate 的bias，state_is_tuple表示接受和返回的state都是2-tuple的形式
        # LSTM单元可以读入一个单词并结合之前存储的状态state机选下一个单词出现的概率分布，每次读取一个单词之后，状态state被更新
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)

        attn_cell = lstm_cell
        # 如果是训练状态，并且keep_prob<1 在lstm.cell()之后再接一个dropout层
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DroputWapper(lstm_cell(), out_keep_rob=config.keep_prob)
        # RNN堆叠函数，堆叠次数为num_layers
        cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
        # 设置LSTM的初始状态
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        # 进行词嵌入，限制在cpu上进行，行数为词汇表数vocab_size,列数(每个单词的向量表达的维数)为hidden_size，和LSTM的隐含节点数一致
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [vocab_size, size], dtype=tf.float32)
            # 查找对应单词的向量表达
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
        # 如果是训练状态，添加一层dropout层
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)
        outputs = []
        state = self._initial_state
        # 为了控制训练过程，设定梯度在反向传播时的展开步数为一个固定的值num_steps，设定一个循环，长度为num_step来控制梯度的传播
        with tf.variable_scope('RNN'):
            for time_step in range(num_steps):
                # 从第二次循环开始设置为复用变量
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                # 每次循环，传入input和state到堆叠的LSTM单元(cell)
                # inputs的维度为3，分别表示batch中的第几个样本，样本中第几个单词，单词的向量表达的维度
                # 得到cell_output和更新后的state
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        # 将outputs连接在一起，并reshape成1维向量
        output = tf.reshape(tf.concat(outputs, 1), [-1, size])
        # 权重
        softmax_w = tf.get_variable('softmax_w', [size, vocab_size], dtype=tf.float32)
        # 偏置
        softmax_b = tf.get_variable('softmax_b', [vocab_size], dtype=tf.float32)
        # 类似于全连接wx+b
        logits = tf.matmul(output, softmax_w) + softmax_b
        # 计算logits和targets之间的误差
        self._loss = loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(input_.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=tf.float32)])
        tf.summary.scalar('loss', self._loss)
        # 计算一个batch的总误差，再平均到每个样本的误差
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        tf.summary.scalar('cost', self._cost)
        # 保留最终的状态为final_state
        self._final_state = state
        # 如果不是训练状态，直接返回
        if not is_training:
            return
        # 学习率为不可训练
        self._lr = tf.Variable(0.0, trainable=False)
        tf.summary.scalar('learning_rate', self._lr)
        # 获得全部可以训练的参数
        tvars = tf.trainable_variables()
        # 针对前面的batch内的每个样本的误差，计算tvars的梯度
        # 使用tf.clip_by_global_norm设置梯度的最大范数，控制梯度的最大范数，起到正则化的作用，防止梯度爆炸，
        # 在rnn中如果不对梯度进行限制，由于梯度下降基于BPTT，很可能会导致梯度过大，训练无法进行
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        # 定义优化器
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        # 将clip过得梯度应用到可训练参数中, 使用tf.contrib.framework.get_or_create_global_step()生成全局统一的训练步数
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                   global_step=tf.contrib.framework.get_or_create_global_step())
        # 控制学习率
        self._new_lr = tf.placeholder(tf.float32, shape=[], name='new_learning_rate')
        # 将_new_lr的值赋值给当前的学习率_lr，用来实现对_lr的更新
        self._lr_update = tf.assign(self._lr, self._new_lr)
        tf.summary.scalar('learning_rate_update', self._lr_update)
        summary = tf.summary.merge_all()

    # 在外部控制学习率
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    # @property 可以实现将返回的变量设置为只读模式，防止修改变量引发的问题

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def loss(self):
        return self._loss

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


# 定义集中不同的参数
class SmallConfig(object):
    """Small config."""
    init_scale = 0.1  # 网络权重值的初始scale
    learning_rate = 1.0  # 学习速率初始值
    max_grad_norm = 5  # 梯度的最大范数
    num_layers = 2  # LSTM可以堆叠的层数
    num_steps = 20  # LSTM反向传播的展开步数
    hidden_size = 200  # LSTM内的隐含节点数
    max_epoch = 4  # 初始学习速率可训练的epoch数
    max_max_epoch = 13  # 总共可以训练的epoch数
    keep_prob = 1.0  # dropout层的保留节点比例
    lr_decay = 0.5  # 学习速率衰减率
    batch_size = 20  # 每个batch中样本数量
    vocab_size = 10000  # 词汇表中单词数量


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


def run_epoch(session, model, eval_op=None, verbose=False):
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    # 输出结果存入进feches中
    feches = {'cost': model.cost, 'final_state': model.final_state, }
    if eval_op is not None:
        feches['eval_op'] = eval_op
    for step in range(model.input.epoch_size):
        feed_dict = {}
        # 将全部的state存入feed_dict
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        #
        vals = session.run(feches, feed_dict)
        cost = vals['cost']
        state = vals['final_state']

        # 损失，迭代数
        costs += cost
        iters += model.input.num_steps
        # np.exp(costs/iters)平均cost的自然常数指数，语言模型中用来比较模型性能的重要指标，越低表示模型输出的概率分布在预测样本上越好
        if verbose and step % (model.input.epoch_size // 10) == 10:
            print('%.3f perplexity : %.3f speed : %.0f wps' %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.bacth_size / (time.time() - start_time)))
    return np.exp(costs / iters)


# 读取原始数据
raw_data = reader.ptb_raw_data('simple-examples/data/')
train_data, valid_data, test_data, _ = raw_data
config = SmallConfig()
eval_config = SmallConfig()
# 修改配置中的参数
eval_config.batch_size = 1
eval_config.num_steps = 1  # 修改反向传播的展开次数

# 创建默认的graph
with tf.Graph().as_default():
    # 初始化参数的范围[-0.1,0.1]
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
    with tf.name_scope("Train"):
        train_input = PTBInput(config=config, data=train_data, name="TrainInput")
        with tf.variable_scope('Model', reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config, input_=train_input)
            tf.summary.scalar('train_cost', m.cost)
            tf.summary.scalar('train_learning_rate', m.lr)

    with tf.name_scope('Valid'):
        valid_input = PTBInput(config=config, data=valid_data, name='ValidInput')
        with tf.variable_scope('Model', reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=config, input_=valid_input)

    with tf.name_scope('Test'):
        test_input = PTBInput(config=eval_config, data=test_data, name='TestInput')
        with tf.variable_scope('Model', reuse=True, initializer=initializer):
            mtest = PTBModel(is_training=False, config=eval_config, input_=test_input)
            tf.summary.scalar('test_cost', mtest.cost)
    merged = tf.summary.merge_all()

    # 创建训练管理器
    sv = tf.train.Supervisor(logdir='./log/summary', summary_op=merged)
    # 创建默认session
    with sv.managed_session() as session:
        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i + 1 - config.max_max_epoch, 0)
            # cost, summary, learning_rate = session.run([m.cost, merged, ])
            m.assign_lr(session=session, lr_value=config.learning_rate * lr_decay)
            print('Epoch: %d Learning rate :.3f' % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                         verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = run_epoch(session, mvalid)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
        test_perplexity = run_epoch(session, mtest)
        print("Test Perplexity: %.3f" % test_perplexity)

        if FLAGS.save_path:
            print("Saving model to %s." % FLAGS.save_path)
            sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)

if __name__ == '__main__':
    tf.app.run()
