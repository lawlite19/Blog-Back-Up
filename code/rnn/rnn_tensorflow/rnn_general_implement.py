# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import urllib.request
import os
import reader
import time
from LayerNormalizedLSTMCell import LayerNormalizedLSTMCell

'''下载数据并读取数据'''
file_url = 'https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt'
#file_url = 'https://gist.githubusercontent.com/spitis/59bfafe6966bfe60cc206ffbb760269f/'+\
#'raw/030a08754aada17cef14eed6fac7797cda830fe8/variousscripts.txt'
file_name = 'tinyshakespeare.txt'
#file_name = 'variousscripts.txt'
if not os.path.exists(file_name):
    urllib.request.urlretrieve(file_url, filename=file_name)
with open(file_name, 'r') as f:
    raw_data = f.read()
    print("数据长度", len(raw_data))

'''处理字符数据，转换为数字'''
vocab = set(raw_data)                    # 使用set去重，这里就是去除重复的字母(大小写是区分的)
vocab_size = len(vocab)      
idx_to_vocab = dict(enumerate(vocab))    # 这里将set转为了字典，每个字符对应了一个数字0,1,2,3..........(vocab_size-1)
vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys())) # 这里将字典的(key, value)转换成(value, key)

data = [vocab_to_idx[c] for c in raw_data]   # 处理raw_data, 根据字符，得到对应的value,就是数字
del raw_data

'''超参数'''
num_steps=200             # 学习的步数
batch_size=32
state_size=100            # cell的size
num_classes = vocab_size
learning_rate = 1e-4
keep_prob = 0.9           # drop参数

'''生成eopchs数据'''
def gen_epochs(num_epochs, num_steps, batch_size):
    for i in range(num_epochs):
        yield reader.ptb_iterator_oldversion(data, batch_size, num_steps)

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()   # 重设计算图

'''训练rnn网络的函数'''
def train_rnn(g, num_epochs, num_steps=num_steps, batch_size=batch_size, verbose=True, save=False):
    tf.set_random_seed(2345)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps, batch_size)):
            training_loss = 0
            steps = 0
            training_state = None
            for X, Y in epoch:
                steps += 1
                feed_dict = {g['x']: X, g['y']: Y, g['keep_prob']: keep_prob}
                if training_state is not None:
                    feed_dict[g['init_state']] = training_state
                training_loss_, training_state, _ = sess.run([g['total_loss'],
                                                           g['final_state'],
                                                           g['train_step']],
                                                          feed_dict=feed_dict)
                training_loss += training_loss_ 
            if verbose:
                print('epoch: {0}的平均损失值：{1}'.format(idx, training_loss/steps))
            training_losses.append(training_loss/steps)
        
        if isinstance(save, str):
            g['saver'].save(sess, save)
    return training_losses

'''使用list的方式,static_rnn'''
def build_basic_rnn_graph_with_list(
    state_size = state_size,
    num_classes = num_classes,
    batch_size = batch_size,
    num_steps = num_steps,
    num_layers = 3,
    learning_rate = learning_rate):

    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='x')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    x_one_hot = tf.one_hot(x, num_classes)   # (batch_size, num_steps, num_classes)
    '''这里按第二维拆开num_steps*(batch_size, num_classes)'''
    rnn_inputs = [tf.squeeze(i,squeeze_dims=[1]) for i in tf.split(x_one_hot, num_steps, 1)]
    '''dropout rnn inputs'''
    rnn_inputs = [tf.nn.dropout(rnn_input, keep_prob) for rnn_input in rnn_inputs]
    
    cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
    init_state = cell.zero_state(batch_size, tf.float32)
    '''使用static_rnn方式'''
    rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell=cell, inputs=rnn_inputs, 
                                                        initial_state=init_state)
    '''dropout rnn outputs'''
    rnn_outputs = [tf.nn.dropout(rnn_output, keep_prob) for rnn_output in rnn_outputs]
    #rnn_outputs, final_state = tf.nn.rnn(cell, rnn_inputs, initial_state=init_state) # tensorflow 1.0的方式
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]

    y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(y, num_steps, 1)]

    #loss_weights = [tf.ones([batch_size]) for i in range(num_steps)]
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_as_list, 
                                                  logits=logits)
    #losses = tf.nn.seq2seq.sequence_loss_by_example(logits, y_as_list, loss_weights)  # tensorflow 1.0的方式
    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x = x,
        y = y,
        keep_prob = keep_prob,
        init_state = init_state,
        final_state = final_state,
        total_loss = total_loss,
        train_step = train_step
    )


'''使用dynamic_rnn方式
   - 之前我们自己实现的cell和static_rnn的例子都是将得到的tensor使用list存起来，这种方式构建计算图时很慢
   - dynamic可以在运行时构建计算图
'''
def build_multilayer_lstm_graph_with_dynamic_rnn(
    state_size = state_size,
    num_classes = num_classes,
    batch_size = batch_size,
    num_steps = num_steps,
    num_layers = 3,
    learning_rate = learning_rate
    ):
    reset_graph()
    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='x')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='y')
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    embeddings = tf.get_variable(name='embedding_matrix', shape=[num_classes, state_size])
    '''这里的输入是三维的[batch_size, num_steps, state_size]
        - embedding_lookup(params, ids)函数是在params中查找ids的表示， 和在matrix中用array索引类似,
          这里是在二维embeddings中找二维的ids, ids每一行中的一个数对应embeddings中的一行，所以最后是[batch_size, num_steps, state_size]
    '''
    rnn_inputs = tf.nn.embedding_lookup(params=embeddings, ids=x)
    
    cell = tf.nn.rnn_cell.LSTMCell(num_units=state_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell]*num_layers, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=keep_prob)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    '''使用dynamic_rnn方式'''
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=rnn_inputs, 
                                                 initial_state=init_state)    
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])   # 转成二维的矩阵
    y_reshape = tf.reshape(y, [-1])
    logits = tf.matmul(rnn_outputs, W) + b                    # 进行矩阵运算
    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshape))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    
    return dict(x = x,
                y = y,
                keep_prob = keep_prob,
                init_state = init_state,
                final_state = final_state,
                total_loss = total_loss,
                train_step = train_step)

'''使用scan实现dynamic_rnn的效果'''
def build_multilayer_lstm_graph_with_scan(
    state_size = state_size,
    num_classes = num_classes,
    batch_size = batch_size,
    num_steps = num_steps,
    num_layers = 3,
    learning_rate = learning_rate
    ):
    reset_graph()
    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='x')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='y')
    embeddings = tf.get_variable(name='embedding_matrix', shape=[num_classes, state_size])
    '''这里的输入是三维的[batch_size, num_steps, state_size]
    '''
    rnn_inputs = tf.nn.embedding_lookup(params=embeddings, ids=x)
    '''构建多层的cell, 先构建一个cell, 然后使用MultiRNNCell函数构建即可'''
    cell = tf.nn.rnn_cell.LSTMCell(num_units=state_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell]*num_layers, state_is_tuple=True)  
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    '''使用tf.scan方式
       - tf.transpose(rnn_inputs, [1,0,2])  是将rnn_inputs的第一个和第二个维度调换，即[num_steps,batch_size, state_size],
           在dynamic_rnn函数有个time_major参数，就是指定num_steps是否在第一个维度上，默认是false的,即不在第一维
       - tf.scan会将elems按照第一维拆开，所以一次就是一个step的数据（和我们static_rnn的例子类似）
       - a的结构和initializer的结构一致，所以a[1]就是对应的state，cell需要传入x和state计算
       - 每次迭代cell返回的是一个rnn_output(batch_size,state_size)和对应的state,num_steps之后的rnn_outputs的shape就是(num_steps, batch_size, state_size)
       - 每次输入的x都会得到的state(final_states)，我们只要的最后的final_state
    '''
    def testfn(a, x):
        return cell(x, a[1])
    rnn_outputs, final_states = tf.scan(fn=testfn, elems=tf.transpose(rnn_inputs, [1,0,2]),
                                        initializer=(tf.zeros([batch_size,state_size]),init_state)
                                        )
    '''或者使用lambda的方式'''
    #rnn_outputs, final_states = tf.scan(lambda a,x: cell(x, a[1]), tf.transpose(rnn_inputs, [1,0,2]),
                                        #initializer=(tf.zeros([batch_size, state_size]),init_state))
    final_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(
        tf.squeeze(tf.slice(c, [num_steps-1,0,0], [1,batch_size,state_size])),
        tf.squeeze(tf.slice(h, [num_steps-1,0,0], [1,batch_size,state_size]))) for c, h in final_states])

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshape = tf.reshape(y, [-1])
    logits = tf.matmul(rnn_outputs, W) + b
    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshape))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    
    return dict(x = x,
                y = y,
                init_state = init_state,
                final_state = final_state,
                total_loss = total_loss,
                train_step = train_step)


'''最终的整合模型，
   - 普通RNN，GRU，LSTM
   - dropout
   - BN
'''
def build_final_graph(
    cell_type = None,
    state_size = state_size,
    num_classes = num_classes,
    batch_size = batch_size,
    num_steps = num_steps,
    num_layers = 3,
    build_with_dropout = False,
    learning_rate = learning_rate):
    
    reset_graph()
    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='x')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)
    if cell_type == 'GRU':
        cell = tf.nn.rnn_cell.GRUCell(state_size)
    elif cell_type == 'LSTM':
        cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    elif cell_type == 'LN_LSTM':
        cell = LayerNormalizedLSTMCell(state_size)
    else:
        cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
    if build_with_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
        
    init_state = cell.zero_state(batch_size, tf.float32)
    '''dynamic_rnn'''
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(y, [-1])
    logits = tf.matmul(rnn_outputs, W) + b

    predictions = tf.nn.softmax(logits)

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x = x,
        y = y,
        keep_prob = keep_prob,
        init_state = init_state,
        final_state = final_state,
        total_loss = total_loss,
        train_step = train_step,
        preds = predictions,
        saver = tf.train.Saver()
    )    

'''生成文本'''
def generate_characters(g, checkpoint, num_chars, prompt='A', pick_top_chars=None):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        g['saver'].restore(sess, checkpoint)   # 读取文件
        state = None
        current_char = vocab_to_idx[prompt]    # 得到给出的字母对应的数字
        chars = [current_char]                          
        for i in range(num_chars):             # 总共生成多少数字
            if state is not None:              # 第一次state为None,因为计算图中定义了刚开始为0
                feed_dict={g['x']: [[current_char]], g['init_state']: state} # 传入当前字符
            else:
                feed_dict={g['x']: [[current_char]]}
            preds, state = sess.run([g['preds'],g['final_state']], feed_dict)   # 得到预测结果（概率）preds的shape就是（1，num_classes）
            if pick_top_chars is not None:              # 如果设置了概率较大的前多少个
                p = np.squeeze(preds)
                p[np.argsort(p)[:-pick_top_chars]] = 0  # 其余的置为0
                p = p / np.sum(p)                       # 因为下面np.random.choice函数p的概率和要求是1，处理一下
                current_char = np.random.choice(vocab_size, 1, p=p)[0]    # 根据概率选择一个
            else:
                current_char = np.random.choice(vocab_size, 1, p=np.squeeze(preds))[0]

            chars.append(current_char)
    chars = map(lambda x: idx_to_vocab[x], chars)
    result = "".join(chars)
    print(result)
    return result
                


'''1、构建图和训练'''
#start_time = time.time()
###g = build_basic_rnn_graph_with_list()
###g = build_multilayer_lstm_graph_with_dynamic_rnn()
###g = build_multilayer_lstm_graph_with_scan()
#g = build_final_graph(cell_type='LN_LSTM', state_size=state_size, 
                     #num_classes=num_classes, 
                     #batch_size=batch_size, 
                     #num_steps=num_steps, num_layers=3, 
                     #build_with_dropout=False, 
                     #learning_rate=learning_rate)
#print("构建图耗时", time.time()-start_time)
#start_time = time.time()
#losses = train_rnn(g, 2, save='saves/LN_LSTM_2_epochs')
#print("训练耗时：", time.time()-start_time)
#print('1',losses[-1])



'''2、生成文本'''
g = build_final_graph(cell_type='LN_LSTM', num_steps=1, batch_size=1)
text = generate_characters(g, "saves/LN_LSTM_2_epochs", 750, prompt='A', pick_top_chars=5)
print(text)
