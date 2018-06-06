# -*- coding: utf-8 -*-
# Author: Lawlite
# Date: 2018/05/31
# Associate Blog: http://lawlite.me/2018/05/31/Tensorflow%E9%AB%98%E7%BA%A7API/#3-1-%E9%A2%84%E5%88%9B%E5%BB%BA%E6%A8%A1%E5%9E%8B
# License: MIT
import tensorflow as tf
import argparse
import iris_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=100, type=int, help='number of training steps')


def my_model(features, labels, mode, params):
    '''自定义模型'''
    net = tf.feature_column.input_layer(features=features, 
                                        feature_columns=params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(inputs=net, units=units, activation=tf.nn.relu)
    
    logits = tf.layers.dense(net, params['num_classes'], activation=None)
    pred = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': pred[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # 计算loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # 计算评价信息
    accuracy = tf.metrics.accuracy(labels=labels, predictions=pred, 
                                  name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar(name='accuracy', tensor=accuracy[1])
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    
    # 训练操作
    assert mode == tf.estimator.ModeKeys.TRAIN
    
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    
def main(argv):
    args = parser.parse_args(args=argv[1:])
    # 加载数据
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()
    # feature columns
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    # 建立模型
    cls = tf.estimator.Estimator(model_fn=my_model, 
                                 params={
                                    'feature_columns': my_feature_columns,
                                    'hidden_units': [10, 10],
                                    'num_classes': 3
                                    })
    # 训练模型
    cls.train(input_fn=lambda: iris_data.train_input_fn(train_x, train_y, args.batch_size), steps=args.train_steps)
    # 评价模型
    eval_res = cls.evaluate(input_fn=lambda: iris_data.eval_input_fn(test_x, test_y, args.batch_size))
    print("\n Test set accuracy: {:0.3f}\n".format(eval_res['accuracy']))
    # 预测
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }  
    predictions = cls.predict(input_fn=lambda: iris_data.eval_input_fn(
                                                                      features=predict_x, 
                                                                      labels=None, 
                                                                      batch_size=args.batch_size))
    template =  ('\nPrediction is "{}" ({:.1f}%), expected "{}"')    
    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        print(template.format(iris_data.SPECIES[class_id], 100*probability, expec))
        
    
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)