#-*- coding: utf-8 -*-
'''此文件可视化神经网络的结构'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

print("tensorflow版本：", tf.__version__)

'''加载数据'''
data = input_data.read_data_sets('MNIST_data', one_hot=True)
print("Size of:")
print("\t\t training set:\t\t{}".format(len(data.train.labels)))
print("\t\t test set: \t\t\t{}".format(len(data.test.labels)))
print("\t\t validation set:\t{}".format(len(data.validation.labels)))

'''超参数'''
img_size = 28
img_flatten_size = img_size ** 2
img_shape = (img_size, img_size)
num_classes = 10
learning_rate = 1e-4

'''定义添加一层'''
def add_fully_layer(inputs, input_size, output_size, num_layer, activation=None):
    with tf.name_scope('layer_'+num_layer):
        with tf.name_scope('Weights'):
            W = tf.Variable(initial_value=tf.random_normal(shape=[input_size, output_size]), name='W')
        with tf.name_scope('biases'):
            b = tf.Variable(initial_value=tf.zeros(shape=[1, output_size]) + 0.1, name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, W) + b
        if activation is not None:
            outputs = activation(Wx_plus_b)
        else:
            outputs = Wx_plus_b
        return outputs

'''placehoder'''
with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, shape=[None, img_flatten_size], name='x')
    y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')

'''结构'''
hidden_layer1 = add_fully_layer(x, img_flatten_size, 20, '1', activation=tf.nn.relu)
outputs = add_fully_layer(hidden_layer1, 20, num_classes, '2')
predictions = tf.nn.softmax(outputs)

'''loss'''
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=predictions)
cross_entropy = tf.reduce_sum(tf.square(y-predictions), reduction_indices=[1])
with tf.name_scope('losses'):
    losses = tf.reduce_mean(cross_entropy)
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(losses)

'''session'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logs', sess.graph)  # 将计算图写入文件
