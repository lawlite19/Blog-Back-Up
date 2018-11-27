#-*- coding: utf-8 -*-
# https://github.com/ShaoqingRen/faster_rcnn/blob/master/functions/nms/nms.m
# https://github.com/ShaoqingRen/faster_rcnn/blob/master/functions/nms/nms_multiclass.m
import numpy as np
import torch

def nms_oneclass(detections, threshold):
    '''
    单个类别非极大值抑制抑制算法
    -------------------------------------------
    Args:
        detections: 检测框坐标和预测的得分，array, [x1, y1, x2, y2, score]
        threshold:  两个boxes重叠区域的阈值, scalar, 小于阈值的保留
    
    Returns:
        keep: 保留的结果框集合， list
    '''
    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 2]
    y2 = detections[:, 3]
    scores = detections[:, 4]
    
    areas = (x2-x1+1) * (y2-y1+1)    # 所有候选框的面积，注意+1，因为是每个点是一个像素，包含这个点
    order = scores.argsort()[::-1]   # 得分按 index 降序
    keep = []
    while order.size>0:  # 不断删除IOU大于阈值的index
        i = order[0]
        keep.append(i)
        
        '''计算其余boxes与已经选出得分最高的重叠的面积'''
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter_area = w * h
        
        IOU = inter_area / (areas[i] + areas[order[1:]] - inter_area)   # 计算IOU = inter/(area1+area2-inter)
        indices = np.where(IOU<=threshold)[0]
        order = order[indices + 1]   # 注意indices + 1, 因为iou数组的元素相比order是少一个的
    return keep


def nms_multiclass(detections, threshold):
    '''
    多个类别非极大值抑制抑制算法
    -------------------------------------------------
    Args:
        detections: 检测框坐标和每个类别的得分，array, [x1, y1, x2, y2, score1, score2...]
        threshold:  两个boxes重叠区域的阈值, scalar, 小于阈值的保留
    
    Returns:
        res: 每个类别保留的额候选框index，dict, {class_id: [候选框list]}
    '''
    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 2]
    y2 = detections[:, 3]
    
    areas = (x2-x1+1) * (y2-y1+1)
    res = {}
    for j in range(4, detections.shape[1]):
        scores = detections[:, j]
        order = scores.argsort()[::-1]
        keep = []
        while order.size>0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2-xx1+1)
            h = np.maximum(0.0, yy2-yy1+1)
            inter_area = w * h
            
            IOU = inter_area / (areas[i] + areas[order[1:]] - inter_area)
            indices = np.where(IOU<=threshold)[0]
            order = order[indices + 1]
        res[j-4] = keep
    return res


def rescore(IOU, scores, threshold, type='gaussian'):
    '''
    根据IOU和本来的score重新打分，两种打分方式：
    （1） linear:   score * (1-IOU)
     (2) gaussian: score * exp(IOU*IOU/threshold)
    ----------------------------------------------------------
    Args:
        IOU:       计算得到的IOU
        scores:    候选框对应的得分
        threshold: 高斯函数的方差
        type:      string, 重新打分的方式
    Returns:
        scores：计算后的得分
    '''
    assert IOU.shape[0] == scores.shape[0]
    if type == 'linear':
        indices = np.where(IOU >= threshold)[0]
        scores[indices] = scores[indices] * (1-IOU[indices])
    else:
        scores = scores * np.exp(- IOU ** 2 / threshold)
    return scores

def soft_nms(detections, threshold, max_detections):
    '''
    soft_nms实现
    -----------------------------------------------------------
    Args:
        detections:     检测框坐标和每个类别的得分，array, [x1, y1, x2, y2, score1, score2...]
        threshold:      这里是重新打分高斯函数的方差，一般是0.5
        max_detections: 最多保留的候选框，-1代表和原detections个数一样，只是重新打分
    Returns:

    '''
    if detections.shape[0] == 0:
        return np.zeros((0, 5))
    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 2]
    y2 = detections[:, 3]
    scores = detections[:, 4]
    areas = (x2-x1+1) * (y2-y1+1)
    order = scores.argsort()[::-1]
    scores = scores[order]

    if max_detections == -1:
        max_detections = order.size

    keep = np.zeros(max_detections, dtype=np.intp)
    keep_cnt = 0

    while order.size > 0 and keep_cnt < max_detections:
        i = order[0]
        detections[i, 4] = scores[0]
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w * h
        IOU = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[1:]
        scores = rescore(IOU, scores[1:], threshold)

        tmp = scores.argsort()[::-1]
        order = order[tmp]
        scores = scores[tmp]

        keep[keep_cnt] = i
        keep_cnt += 1

    keep = keep[:keep_cnt]
    detections = detections[keep, :]
    return detections


if __name__ == '__main__':
    detections = np.array([[1,1,3,3,0.8,0.3],
                           [2,1,3,2,0.7,0.4],
                           [2,1,4,2,0.5,0.5]])
    print(nms_oneclass(detections, threshold=0.4))
    print(nms_multiclass(detections, threshold=0.4))
    print(soft_nms(detections, threshold=0.4, max_detections=-1))