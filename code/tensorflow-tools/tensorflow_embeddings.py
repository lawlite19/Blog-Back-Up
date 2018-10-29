#-*- coding: utf-8 -*-
# Author: Lawlite
# Date: 2017/07/26
# Associate Blog: http://lawlite.me/2017/06/24/Tensorflow学习-工具相关/#1、可视化embedding
# License: MIT

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data
import os

MNIST_DATA_PATH = 'MNIST_data'
LOG_DIR = 'log'
SPRITE_IMAGE_FILE = 'mnist_10k_sprite.png'
META_DATA_FILE = 'metadata.tsv'
IMAGE_NUM = 100

mnist = input_data.read_data_sets(MNIST_DATA_PATH, one_hot=False)
plot_array = mnist.test.images[:IMAGE_NUM]   # 取前100个图片
np.savetxt(os.path.join(LOG_DIR, META_DATA_FILE), mnist.test.labels[:IMAGE_NUM], fmt='%d')  # label 保存为metadata.tsv文件


'''可视化embedding, 3个步骤'''
with tf.Session() as sess:
    '''1、 将2D矩阵放入Variable中'''
    embeddings_var = tf.Variable(plot_array, name='embedding')
    tf.global_variables_initializer().run()
    
    '''2、 保存到文件中'''
    saver = tf.train.Saver()
    sess.run(embeddings_var.initializer)
    saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
    
    '''3、 关联metadata和sprite图片'''
    summary_writer = tf.summary.FileWriter(LOG_DIR)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embeddings_var.name
    embedding.metadata_path = META_DATA_FILE
    embedding.sprite.image_path = SPRITE_IMAGE_FILE
    embedding.sprite.single_image_dim.extend([28, 28])
    projector.visualize_embeddings(summary_writer, config)