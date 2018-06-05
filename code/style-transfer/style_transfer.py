# -*- coding: utf-8 -*-
# Author: Lawlite
# Date: 2018/02/28
# Associate Blog: http://lawlite.me/2018/02/28/%E9%A3%8E%E6%A0%BC%E8%BF%81%E7%A7%BBStyle-transfer/
# License: MIT
import numpy as np
from keras import backend as K
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

from keras.applications import VGG16
from scipy.optimize import fmin_l_bfgs_b
from matplotlib import pyplot as plt

'''图片路径'''
content_image_path = './data/buildings.jpg'
style_image_path = './data/starry-sky.jpg'
generate_image_path = './data/output.jpg'


'''加载图片并初始化输出图片'''
target_height = 512
target_width = 512
target_size = (target_height, target_width)

content_image = load_img(content_image_path, target_size=target_size)
content_image_array = img_to_array(content_image)
content_image_array = K.variable(preprocess_input(np.expand_dims(content_image_array, 0)), dtype='float32')

style_image = load_img(style_image_path, target_size=target_size)
style_image_array = img_to_array(style_image)
style_image_array = K.variable(preprocess_input(np.expand_dims(style_image_array, 0)), dtype='float32')

generate_image = np.random.randint(256, size=(target_width, target_height, 3)).astype('float64')
generate_image = preprocess_input(np.expand_dims(generate_image, 0))
generate_image_placeholder = K.placeholder(shape=(1, target_width, target_height, 3))


def get_feature_represent(x, layer_names, model):
    '''图片的特征图表示
    
    参数
    ----------------------------------------------
    x : 输入，
        这里并没有使用，可以看作一个输入的标识
    layer_names : list
        CNN网络层的名字
    model : CNN模型
    
    返回值
    ----------------------------------------------
    feature_matrices : list
        经过CNN卷积层的特征表示，这里大小是(filter个数, feature map的长*宽)
    
    '''
    feature_matrices = []
    for ln in layer_names:
        select_layer = model.get_layer(ln)
        feature_raw = select_layer.output
        feature_raw_shape = K.shape(feature_raw).eval(session=tf_session)
        N_l = feature_raw_shape[-1]
        M_l = feature_raw_shape[1]*feature_raw_shape[2]
        feature_matrix = K.reshape(feature_raw, (M_l, N_l))
        feature_matrix = K.transpose(feature_matrix)
        feature_matrices.append(feature_matrix)
    return feature_matrices

def get_content_loss(F, P):
    '''计算内容损失
    
    参数
    ---------------------------------------
    F : tensor, float32
        生成图片特征图矩阵
    P : tensor, float32
        内容图片特征图矩阵
    
    返回值
    ---------------------------------------
    content_loss : tensor, float32
        内容损失
    '''
    content_loss = 0.5*K.sum(K.square(F-P))
    return content_loss

def get_gram_matrix(F):
    '''计算gram矩阵'''
    G = K.dot(F, K.transpose(F))
    return G

def get_style_loss(ws, Gs, As):
    '''计算风格损失
    
    参数
    ---------------------------------------
    ws : array
         每一层layer的权重
    Gs : list
         生成图片每一层得到的特征表示组成的list
    As : list
         风格图片每一层得到的特征表示组成的list
    '''
    style_loss = K.variable(0.)
    for w, G, A in zip(ws, Gs, As):
        M_l = K.int_shape(G)[1]
        N_l = K.int_shape(G)[0]
        G_gram = get_gram_matrix(G)
        A_gram = get_gram_matrix(A)
        style_loss += w*0.25*K.sum(K.square(G_gram-A_gram))/(N_l**2*M_l**2)
    return style_loss



def get_total_loss(generate_image_placeholder, alpha=1.0, beta=10000.0):
    '''总损失
    '''
    F = get_feature_represent(generate_image_placeholder, layer_names=[content_layer_name], model=gModel)[0]
    Gs = get_feature_represent(generate_image_placeholder, layer_names=style_layer_names, model=gModel)
    content_loss = get_content_loss(F, P)
    style_loss = get_style_loss(ws, Gs, As)
    total_loss = alpha*content_loss + beta*style_loss
    return total_loss


def calculate_loss(gen_image_array):
    '''调用总损失函数，计算得到总损失数值'''
    if gen_image_array != (1, target_width, target_height, 3):
        gen_image_array = gen_image_array.reshape((1, target_width, target_height, 3))
    loss_fn = K.function(inputs=[gModel.input], outputs=[get_total_loss(gModel.input)])
    return loss_fn([gen_image_array])[0].astype('float64')

def get_grad(gen_image_array):
    '''计算损失函数的梯度'''
    if gen_image_array != (1, target_width, target_height, 3):
        gen_image_array = gen_image_array.reshape((1, target_width, target_height, 3))
    grad_fn = K.function([gModel.input], K.gradients(get_total_loss(gModel.input), [gModel.input]))
    grad = grad_fn([gen_image_array])[0].flatten().astype('float64')
    return grad


def postprocess_array(x):
    '''生成图片后处理，因为之前preprocess_input函数中做了处理，这里进行逆处理还原
    
    '''
    if x.shape != (target_width, target_height, 3):
        x = x.reshape((target_width, target_height, 3))
    x[..., 0] += 103.939
    x[..., 1] += 116.779
    x[..., 2] += 123.68
    
    x = x[..., ::-1]  # BGR-->RGB
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    return x

'''定义VGG模型'''
tf_session = K.get_session()
cModel = VGG16(include_top=False, input_tensor=content_image_array)
sModel = VGG16(include_top=False, input_tensor=style_image_array)
gModel = VGG16(include_top=False, input_tensor=generate_image_placeholder)
content_layer_name = 'block4_conv2'
style_layer_names = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1'
]

'''得到对应的representation矩阵'''
P = get_feature_represent(x=content_image_array, layer_names=[content_layer_name], model=cModel)[0]
As = get_feature_represent(x=style_image_array, layer_names=style_layer_names, model=sModel)
ws = np.ones(len(style_layer_names))/float(len(style_layer_names))

'''使用fmin_l_bfgs_b进行损失函数优化'''
iterations = 600
x_val = generate_image.flatten()
xopt, f_val, info = fmin_l_bfgs_b(func=calculate_loss, x0=x_val, fprime=get_grad, maxiter=iterations, disp=True)

x_out = postprocess_array(xopt)
plt.imshow(x_out)

plt.show()