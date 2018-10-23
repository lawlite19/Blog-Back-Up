# -*- coding: utf-8 -*-
# Author: Lawlite
# Date: 2018/10/20
# Associate Blog: http://lawlite.me/2018/10/16/Triplet-Loss原理及其实现/#more
# License: MIT
import numpy as np


def test_pairwise_distances(squared = False):
    '''两两embedding的距离，比如第一行， 0和0距离为0， 0和1距离为8， 0和2距离为16 （注意开过根号）
    [[ 0.  8. 16.]
     [ 8.  0.  8.]
     [16.  8.  0.]]
    '''    
    embeddings = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)
    dot_product = np.dot(embeddings, np.transpose(embeddings))
    square_norm = np.diag(dot_product)
    distances = np.expand_dims(square_norm, axis=1) - 2.0*dot_product + np.expand_dims(square_norm, 0)
    mask = np.float32(np.equal(distances, 0.0))
    if not squared:
        distances = distances + mask * 1e-16
        distances = np.sqrt(distances)
        distances = distances * (1.0 - mask)
    print(distances)
    return distances

def test_get_triplet_mask(labels):
    '''
    valid （i, j, k）满足
         - i, j, k 不相等
         - labels[i] == labels[j]  && labels[i] != labels[k]
    
    '''
    # 初始化一个二维矩阵，坐标(i, j)不相等置为1，得到indices_not_equal    
    indices_equal = np.cast[np.bool](np.eye(np.shape(labels)[0], dtype=np.int32))
    indices_not_equal = np.logical_not(indices_equal)
    # 因为最后得到一个3D的mask矩阵(i, j, k)，增加一个维度，则 i_not_equal_j 在第三个维度增加一个即，(batch_size, batch_size, 1), 其他同理    
    i_not_equal_j = np.expand_dims(indices_not_equal, 2)
    i_not_equal_k = np.expand_dims(indices_not_equal, 1)
    j_not_equal_k = np.expand_dims(indices_not_equal, 0)
    # 想得到i!=j!=k, 三个不等取and即可
    # 比如这里得到
    '''array([[[False, False, False],
               [False, False,  True],
               [False,  True, False]],
              [[False, False,  True],
               [False, False, False],
               [ True, False, False]],
              [[False,  True, False],
              [ True, False, False],
              [False, False, False]]])'''
    # 只有下标(i, j, k)不相等时才是True
    distinct_indices = np.logical_and(np.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)
    
    # 同样根据labels得到对应i=j, i!=k
    label_equal = np.equal(np.expand_dims(labels, 0), np.expand_dims(labels, 1))
    i_equal_j = np.expand_dims(label_equal, 2)
    i_equal_k = np.expand_dims(label_equal, 1)
    valid_labels = np.logical_and(i_equal_j, np.logical_not(i_equal_k))
    
    # mask即为满足上面两个约束，所以两个3D取and
    mask = np.logical_and(valid_labels, distinct_indices)
    return mask
    

def test_batch_all_triplet_loss(margin):
    
    # 得到每两两embeddings的距离，然后增加一个维度，一维需要得到（batch_size, batch_size, batch_size）大小的3D矩阵
    # 然后再点乘上valid 的 mask即可    
    labels = np.array([1, 0, 1])   # 比如1，3是正例，2是负例，这样计算出的loss应该是16-8 = 8
    pairwise_distances = test_pairwise_distances()
    anchor_positive = np.expand_dims(pairwise_distances, axis=2)
    anchor_negative = np.expand_dims(pairwise_distances, axis=1)
    triplet_loss = anchor_positive - anchor_negative + margin
    
    mask = test_get_triplet_mask(labels)
    mask = np.cast[np.float32](mask)
    triplet_loss = np.multiply(mask, triplet_loss)
    triplet_loss = np.maximum(triplet_loss, 0.0)
    
    valid_triplet_loss = np.cast[np.float32](np.greater(triplet_loss, 1e-16))
    num_positive_triplet = np.sum(valid_triplet_loss)
    num_valid_triplet_loss = np.sum(mask)
    fraction_positive_triplet = num_positive_triplet / (num_valid_triplet_loss + 1e-16)
    
    triplet_loss = np.sum(triplet_loss) / (num_positive_triplet + 1e-16)
    return triplet_loss, fraction_positive_triplet

def test_anchor_positive_triplet_mask(labels):  
    # 得到positive的2D mask， i!=j and i和j有相同labels
    indices_equal = np.cast[np.bool](np.eye(np.shape(labels)[0]))
    indices_not_equal = np.logical_not(indices_equal)
    labels_equal = np.equal(np.expand_dims(labels, 0), np.expand_dims(labels, 1))
    mask = np.logical_and(indices_not_equal, labels_equal)
    return mask
    
def test_get_anchor_negative_triplet_mask(labels):
    # 得到negative的2D mask
    labels_equal = np.equal(np.expand_dims(labels, 0), np.expand_dims(labels, 1))
    mask = np.logical_not(labels_equal)
    return mask
    
def test_batch_hard_triplet_loss(margin):
    # 还是得到两两的距离pairwise_distances
    # 计算最大的positive距离，只需要取每行最大元素即可
    # 计算最小的negative距离，不能直接取每行最小的元素，因为invalid的[a, n]设置为0，这里设置invalid的位置为每一行最大的值，这样就可以取每一行最小的值了
    labels = np.array([1, 0, 1])
    pairwise_distances = test_pairwise_distances()
    mask_anchor_positive = test_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = np.cast[np.float](mask_anchor_positive)
    anchor_positive_dist = np.multiply(mask_anchor_positive, pairwise_distances)
    hardest_positive_dist = np.max(anchor_positive_dist, axis=1, keepdims=True)
    
    mask_anchor_negative = test_get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = np.cast[np.float](mask_anchor_negative)
    
    max_anchor_negative_dist = np.max(pairwise_distances, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_distances + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
    hardest_negative_dist = np.min(anchor_negative_dist, axis=1, keepdims=True)
    triplet_loss = np.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)
    triplet_loss = np.mean(triplet_loss)
    
    return triplet_loss

if __name__ == '__main__':
    #test_batch_all_triplet_loss(margin = 0.0)
    test_batch_hard_triplet_loss(margin = 0.0)
