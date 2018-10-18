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
         - i, j, k都不相等
         - labels[i] == labels[j]  && labels[i] != labels[k]
    
    array([[[False, False, False],
            [False, False, False],
            [False,  True, False]],

          [[False, False, False],
           [False, False, False],
           [False, False, False]],

          [[False,  True, False],
           [False, False, False],
           [False, False, False]]])
    '''
    indices_equal = np.cast[np.bool](np.eye(np.shape(labels)[0], dtype=np.int32))
    indices_not_equal = np.logical_not(indices_equal)
    i_not_equal_j = np.expand_dims(indices_not_equal, 2)
    i_not_equal_k = np.expand_dims(indices_not_equal, 1)
    j_not_equal_k = np.expand_dims(indices_not_equal, 0)

    distinct_indices = np.logical_and(np.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)
    label_equal = np.equal(np.expand_dims(labels, 0), np.expand_dims(labels, 1))
    i_equal_j = np.expand_dims(label_equal, 2)
    i_equal_k = np.expand_dims(label_equal, 1)
    
    valid_labels = np.logical_and(i_equal_j, np.logical_not(i_equal_k))
    mask = np.logical_and(valid_labels, distinct_indices)
    return mask
    

def test_batch_all_triplet_loss(margin):
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
    
    

if __name__ == '__main__':
    # test_pairwise_distances()
    test_batch_all_triplet_loss(margin = 0.0)
