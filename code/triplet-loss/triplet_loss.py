# -*- coding: utf-8 -*-
# Author: Lawlite
# Date: 2018/10/20
# Associate Blog: http://lawlite.me/2018/10/16/Triplet-Loss原理及其实现/#more
# License: MIT
import tensorflow as tf

def _pairwise_distances(embeddings, squared=False):
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
        mask = tf.to_float(tf.equal(distances, 0.0))
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
    
    # 初始化一个二维矩阵，坐标(i, j)不相等置为1，得到indices_not_equal
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    # 因为最后得到一个3D的mask矩阵(i, j, k)，增加一个维度，则 i_not_equal_j 在第三个维度增加一个即，(batch_size, batch_size, 1), 其他同理
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2) 
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)
    # 想得到i!=j!=k, 三个不等取and即可, 最后可以得到当下标（i, j, k）不相等时才取True
    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # 同样根据labels得到对应i=j, i!=k
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)
    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))
    # mask即为满足上面两个约束，所以两个3D取and
    mask = tf.logical_and(distinct_indices, valid_labels)
    return mask

def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    '''
       triplet loss of a batch, 注意这里的loss一般不是收敛的，因为是计算的semi-hard和hard的距离均值，因为每次是先选择出semi-hard和hard
       triplet, 那么上次优化后的可能就选择不到了，所以loss并不会收敛，但是fraction_postive_triplets是收敛的，因为随着优化占的比例是越来越少的
       -------------------------------
       Args:
          labels:     标签数据，shape = （batch_size,）
          embeddings: 提取的特征向量， shape = (batch_size, vector_size)
          margin:     margin大小， scalar
          
       Returns:
          triplet_loss: scalar, 一个batch的损失值
          fraction_postive_triplets : valid的triplets占的比例
    '''
    
    # 得到每两两embeddings的距离，然后增加一个维度，一维需要得到（batch_size, batch_size, batch_size）大小的3D矩阵
    # 然后再点乘上valid 的 mask即可
    pairwise_dis = _pairwise_distances(embeddings, squared=squared)
    anchor_positive_dist = tf.expand_dims(pairwise_dis, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    anchor_negative_dist = tf.expand_dims(pairwise_dis, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    
    mask = _get_triplet_mask(labels)
    mask = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)
    triplet_loss = tf.maximum(triplet_loss, 0.0)
    
    # 计算valid的triplet的个数，然后对所有的triplet loss求平均
    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_postive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
    
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)
    return triplet_loss, fraction_postive_triplets

def _get_anchor_positive_triplet_mask(labels):
    ''' 
       得到合法的positive的mask， 即2D的矩阵，[a, p], a!=p and a和p相同labels
       ------------------------------------------------
       Args:
          labels: 标签数据，shape = (batch_size, )
          
       Returns:
          mask: 合法的positive mask, shape = (batch_size, batch_size)
    '''
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)                 # （i, j）不相等
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))  # labels相等，
    mask = tf.logical_and(indices_not_equal, labels_equal)            # 取and即可
    return mask

def _get_anchor_negative_triplet_mask(labels):
    '''
       得到negative的2D mask, [a, n] 只需a, n不同且有不同的labels
       ------------------------------------------------
       Args:
          labels: 标签数据，shape = (batch_size, )
          
       Returns:
          mask: negative mask, shape = (batch_size, batch_size)
    '''
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    mask = tf.logical_not(labels_equal)
    return mask

def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    '''
       batch hard triplet loss of a batch， 每个样本最大的positive距离 - 对应样本最小的negative距离
       ------------------------------------
       Args:
          labels:     标签数据，shape = （batch_size,）
          embeddings: 提取的特征向量， shape = (batch_size, vector_size)
          margin:     margin大小， scalar
          
       Returns:
          triplet_loss: scalar, 一个batch的损失值
    '''
    pairwise_distances = _pairwise_distances(embeddings, squared=squared)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_distances)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)  # 取每一行最大的值即为最大positive距离
    tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))
    
    '''取每一行最小值得时候，因为invalid [a, n]置为了0， 所以不能直接取，这里对应invalid位置加上每一行的最大值即可，然后再取最小的值'''
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)
    max_anchor_negative_dist = tf.reduce_max(pairwise_distances, axis=1, keepdims=True)   # 每一样最大值
    anchor_negative_dist = pairwise_distances + max_anchor_negative_dist * (1.0 - mask_anchor_negative)  # (1.0 - mask_anchor_negative)即为invalid位置
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))
    
    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)
    triplet_loss = tf.reduce_mean(triplet_loss)
    return triplet_loss
    