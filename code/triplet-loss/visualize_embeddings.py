# -*- coding: utf-8 -*-
# Author: Lawlite
# Date: 2018/10/20
# Associate Blog: http://lawlite.me/2018/10/16/Triplet-Loss原理及其实现/#more
# License: MIT

import os
import shutil
import numpy as np
import tensorflow as tf
import argparse
import json
from triplet_loss import batch_all_triplet_loss
from triplet_loss import batch_hard_triplet_loss
import mnist_dataset
from train_with_triplet_loss import my_model
from train_with_triplet_loss import test_input_fn
from tensorflow.contrib.tensorboard.plugins import projector


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',type=str, help="数据地址")
parser.add_argument('--model_dir', default='experiment/model', type=str, help="模型地址")
parser.add_argument('--model_config', default='experiment/params.json', type=str, help="模型参数")
parser.add_argument('--sprite_filename', default='experiment/mnist_10k_sprite.png', help="Sprite image for the projector")
parser.add_argument('--log_dir', default='experiment/log', type=str, help='可视化embeddings log文件夹')

def main(argv):
    args = parser.parse_args(argv[1:])
    '''创建模型'''
    with open(args.model_config) as f:
        params = json.load(f)
    tf.logging.info("创建模型....")
    config = tf.estimator.RunConfig(model_dir=args.model_dir, tf_random_seed=100)  # config
    cls = tf.estimator.Estimator(model_fn=my_model, config=config, params=params)  # 建立模型
    
    '''预测得到embeddings'''
    tf.logging.info("预测....")
    predictions = cls.predict(input_fn=lambda: test_input_fn(args.data_dir, params))
    embeddings = np.zeros((10000, params['embedding_size']))
    for i, p in enumerate(predictions):
        embeddings[i] = p['embeddings']
    tf.logging.info("embeddings shape: {}".format(embeddings.shape))
    
    '''获得testset 的label 数据，并保存为metadata.tsv 文件'''
    with tf.Session() as sess:
        # Obtain the test labels
        dataset = mnist_dataset.test(args.data_dir)
        dataset = dataset.map(lambda img, lab: lab)
        dataset = dataset.batch(10000)
        labels_tensor = dataset.make_one_shot_iterator().get_next()
        labels = sess.run(labels_tensor)   
    np.savetxt(os.path.join(args.log_dir, 'metadata.tsv'), labels, fmt='%d')
    shutil.copy(args.sprite_filename, args.log_dir)
    '''可视化embeddings'''
    with tf.Session() as sess:
        # 1. Variable
        embedding_var = tf.Variable(embeddings, name="mnist_embeddings")
        #tf.global_variables_initializer().run()  # 不需要
        
        # 2. 保存到文件中，embeddings.ckpt
        saver = tf.train.Saver()
        sess.run(embedding_var.initializer)
        saver.save(sess, os.path.join(args.log_dir, 'embeddings.ckpt'))
        
        # 3. 关联metadata.tsv, 和mnist_10k_sprite.png
        summary_writer = tf.summary.FileWriter(args.log_dir)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = 'metadata.tsv'
        embedding.sprite.image_path = 'mnist_10k_sprite.png'
        embedding.sprite.single_image_dim.extend([28, 28])
        projector.visualize_embeddings(summary_writer, config)


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)    
    tf.app.run(main)
        
