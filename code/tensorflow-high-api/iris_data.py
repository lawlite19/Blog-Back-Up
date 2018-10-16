# -*- coding: utf-8 -*-
# Author: Lawlite
# Date: 2018/05/31
# Associate Blog: 
# License: MIT
import tensorflow as tf
import pandas as pd

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

def maybe_download():
    """下载数据"""
    train_path = tf.keras.utils.get_file(fname=TRAIN_URL.split("/")[-1], origin=TRAIN_URL)
    test_path  = tf.keras.utils.get_file(fname=TEST_URL.split("/")[-1], origin=TEST_URL)
    return train_path, test_path

def load_data(y_name = "Species"):
    """返回训练集和测试集"""
    train_path, test_path = maybe_download()
    train = pd.read_csv(filepath_or_buffer=train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)
    
    test = pd.read_csv(filepath_or_buffer=test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)
    
    return (train_x, train_y), (test_x, test_y)

def train_input_fn(features, labels, batch_size):
    """训练集输入函数"""
    dataset = tf.data.Dataset.from_tensor_slices((dict(features,), labels))   # 转化为Dataset
    
    dataset = dataset.shuffle(buffer_size=1000).repeat().batch(batch_size)    # Shuffle, batch
    
    return dataset
def eval_input_fn(features, labels, batch_size):
    """评价或者预测数据集"""
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    return dataset

CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]
def _parse_line(line):
    '''解析一行数据'''
    field = tf.decode_csv(line, record_defaults=CSV_TYPES)
    features = dict(zip(CSV_COLUMN_NAMES, field))
    labels = features.pop("Species")
    return features, labels

def csv_input_fn(csv_path, batch_size):
    '''csv文件输入函数'''
    dataset = tf.data.TextLineDataset(csv_path).skip(1)   # 跳过第一行
    dataset = dataset.map(_parse_line)        # 应用map函数处理dataset中的每一个元素
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset
    