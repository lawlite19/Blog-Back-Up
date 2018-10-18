# -*- coding: utf-8 -*-
# Author: Lawlite
# Date: 2018/10/20
# Associate Blog: http://lawlite.me/2018/10/16/Triplet-Loss原理及其实现/#more
# License: MIT
import tensorflow as tf

def _pairwise_distance(embeddings, squared=False):
    '''
       计算两两embedding的距离
       ------------------------------------------
       Args：
          embedding: 特征向量， 大小（batch_size, vector_size）
          squared:   是否距离的平方，即欧式距离
    
       Returns：
          distances: 两两embeddings的距离矩阵，大小 （batch_size, batch_size）
    '''    
    # 矩阵相乘,得到（batch_size, batch_size），因为计算欧式距离|a-b|^2 = a^2 -2ab + b^2, 
    # 其中 ab 可以用矩阵乘表示
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))   
    # dot_product对角线部分就是 每个embedding的平方
    square_norm = tf.diag_part(dot_product)
    # |a-b|^2 = a^2 - 2ab + b^2
    # tf.expand_dims(square_norm, axis=1)是（batch_size, 1）大小的矩阵，减去 （batch_size, batch_size）大小的矩阵，相当于每一列操作
    distances = tf.expand_dims(square_norm, axis=1) - 2.0 * dot_product + tf.expand_dims(square_norm, axis=0)
    distances = tf.maximum(distances, 0.0)   # 小于0的距离置为0
    if not squared:          # 如果不平方，就开根号，但是注意有0元素，所以0的位置加上 1e*-16
        mask = tf.to_float(tf.equal(dot_product, 0.0))
        distances = distances + mask * 1e-16
        distances = tf.sqrt(distances)
        distances = distances * (1.0 - mask)    # 0的部分仍然置为0
    return distances


def _get_triplet_mask(labels):
    '''
       得到一个3D的mask [a, p, n], 对应triplet（a, p, n）是valid的位置是True
       ----------------------------------
       Args:
          labels: 对应训练数据的labels, shape = (batch_size,)
       
       Returns:
          mask: 3D,shape = (batch_size, batch_size, batch_size)
    
    '''
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)
    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))
    mask = tf.logical_and(distinct_indices, valid_labels)
    return mask

def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    pairwise_dis = _pairwise_distance(embeddings, squared=squared)
    anchor_positive_dist = tf.exoand_dims(pairwise_dis, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    anchor_negative_dist = tf.expand_dims(pairwise_dis, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    mask = _get_triplet_mask(labels)
    mask = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)
    triplet_loss = tf.maximum(triplet_loss, 0.0)
    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_postive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)
    return triplet_loss, fraction_postive_triplets