import numpy as np
import struct
import os
import time
from numpy.lib.stride_tricks import as_strided

def show_matrix(mat, name):
    #print(name + str(mat.shape) + ' mean %f, std %f' % (mat.mean(), mat.std()))
    pass

def show_time(time, name):
    #print(name + str(time))
    pass

# N,C,K,out
def im2col(input, kszie, stride):
    N, C, H, W = input.shape
    H_out = (H - kszie) // stride + 1
    W_out = (W - kszie) // stride + 1

    shape = (N, C, kszie, kszie, H_out, W_out)
    strides = (
        input.strides[0],  # Batch 维度
        input.strides[1],  # Channel 维度
        input.strides[2],  # Kernel 纵向步长 
        input.strides[3],   # Kernel 横向步长
        input.strides[2] * stride,  # 滑动窗口在 H 方向移动
        input.strides[3] * stride  # 滑动窗口在 W 方向移动
    )
    patches = as_strided(input, shape=shape, strides=strides)
    return patches.reshape(N, C,kszie * kszie, H_out * W_out)

#N,CKK,HoWo->NCHiWi
def col2im(input, height_pad, width_pad, kszie, channel, padding, stride):#hp wp都是self.input的
    N = input.shape[0]
    output = np.zeros([N, channel, height_pad, width_pad])
    input = input.reshape(N, channel, kszie * kszie, -1)#N,C,KK,HW
    height = (height_pad - kszie) // stride + 1
    width = (width_pad - kszie) // stride + 1
    for idx in range(kszie * kszie):
        i, j = divmod(idx, kszie)  # i, j 分别是核内的行列索引
        output[:, :, i: i + height * stride: stride, j: j + width * stride: stride] += \
            input[:, :, idx, :].reshape(N, channel, height, width)#height_out
    return output[:, :, padding : height_pad - padding, padding : width_pad - padding]
    
class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride, type=0):
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride
        self.forward = self.forward_raw
        self.backward = self.backward_raw
        if type == 1:  # type 设为 1 时，使用优化后的 foward 和 backward 函数
            self.forward = self.forward_speedup
            self.backward = self.backward_speedup
        print('\tConvolutional layer with kernel size %d, input channel %d, output channel %d.' % (self.kernel_size, self.channel_in, self.channel_out))
    def init_param(self, std=0.01):
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out))
        self.bias = np.zeros([self.channel_out])
        show_matrix(self.weight, 'conv weight ')
        show_matrix(self.bias, 'conv bias ')
    def forward_raw(self, input):
        start_time = time.time()
        self.input = input # [N, C, H, W]
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]] = self.input
        height_out = (height - self.kernel_size) // self.stride + 1
        width_out = (width - self.kernel_size) // self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.channel_out, height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.channel_out):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO: 计算卷积层的前向传播，特征图与卷积核的内积再加偏置
                        self.output[idxn, idxc, idxh, idxw] = np.sum(self.weight[:, :, :, idxc] * self.input_pad[idxn, :, idxh * self.stride : idxh * self.stride + self.kernel_size, idxw * self.stride : idxw * self.stride + self.kernel_size]) + self.bias[idxc]
        self.forward_time = time.time() - start_time
        return self.output
    def forward_speedup(self, input):
        # TODO: 改进forward函数，使得计算加速
        start_time = time.time()
        self.input = input # [N, C, H, W]
        N, C, H, W = input.shape
        height = H + self.padding * 2
        width = W + self.padding * 2
        self.input_pad = np.zeros([N, C, height, width])
        self.input_pad[:, :, self.padding:self.padding+H, self.padding:self.padding+W] = self.input
        height_out = (height - self.kernel_size) // self.stride + 1
        width_out = (width - self.kernel_size) // self.stride + 1
        #im2col+gemm
        self.input_col = im2col(self.input_pad, self.kernel_size, self.stride) # N,Cin,KK,HW
        self.weights_col = self.weight.transpose(3, 0, 1, 2).reshape(self.channel_out, -1) # Cout,CinKK
        # N,Cout,HW
        output = np.matmul(self.weights_col, self.input_col.reshape(N, -1, self.input_col.shape[3])) + self.bias.reshape(-1, 1)
        self.output = output.reshape(N, self.channel_out, height_out, width_out)  
        self.forward_time = time.time() - start_time
        return self.output
    def backward_speedup(self, top_diff):
        start_time = time.time()
        N, C, H, W = self.input.shape
        height = H + self.padding * 2
        width = W + self.padding * 2
        bottom_diff_col = np.matmul(self.weights_col.T, top_diff.reshape(N,self.channel_out,-1))# N,CoutKK,HW
        bottom_diff = col2im(bottom_diff_col, height, width, self.kernel_size, self.channel_in, self.padding, self.stride)
        self.backward_time = time.time() - start_time
        return bottom_diff
    def backward_raw(self, top_diff):
        start_time = time.time()
        self.d_weight = np.zeros(self.weight.shape)
        self.d_bias = np.zeros(self.bias.shape)
        bottom_diff = np.zeros(self.input_pad.shape)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        # TODO： 计算卷积层的反向传播， 权重、偏置的梯度和本层损失
                        self.d_weight[:, :, :, idxc] += top_diff[idxn, idxc, idxh, idxw] * self.input_pad[idxn, :, idxh * self.stride : idxh * self.stride + self.kernel_size, idxw * self.stride : idxw * self.stride + self.kernel_size]
                        self.d_bias[idxc] += top_diff[idxn, idxc, idxh, idxw]
                        bottom_diff[idxn, :, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size] += top_diff[idxn, idxc, idxh, idxw] * self.weight[:, :, :, idxc]
        bottom_diff = bottom_diff[:, :, self.padding : self.padding + self.input.shape[2], self.padding : self.padding + self.input.shape[3]]
        self.backward_time = time.time() - start_time
        return bottom_diff
    def get_gradient(self):
        return self.d_weight, self.d_bias
    def update_param(self, lr):
        self.weight += - lr * self.d_weight
        self.bias += - lr * self.d_bias
    def load_param(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
        show_matrix(self.weight, 'conv weight ')
        show_matrix(self.bias, 'conv bias ')
    def get_forward_time(self):
        return self.forward_time
    def get_backward_time(self):
        return self.backward_time

class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride, type=0):
        self.kernel_size = kernel_size
        self.stride = stride
        ### adding
        self.forward = self.forward_raw
        self.backward = self.backward_raw_book
        if type == 1: # type 设为 1 时，使用优化后的 foward 和 backward 函数
            self.forward = self.forward_speedup
            self.backward = self.backward_speedup

        print('\tMax pooling layer with kernel size %d, stride %d.' % (self.kernel_size, self.stride))
    def forward_raw(self, input):
        start_time = time.time()
        self.input = input # [N, C, H, W]
        self.max_index = np.zeros(self.input.shape)
        height_out = (self.input.shape[2] - self.kernel_size) // self.stride + 1
        width_out = (self.input.shape[3] - self.kernel_size) // self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.input.shape[1], height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.input.shape[1]):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO： 计算最大池化层的前向传播， 取池化窗口内的最大值
                        pool_window = self.input[idxn, idxc, 
                            idxh * self.stride : idxh * self.stride + self.kernel_size, 
                            idxw * self.stride : idxw * self.stride + self.kernel_size]
                        max_val = np.max(pool_window)
                        max_pos = np.where(pool_window == max_val)    
                        self.output[idxn, idxc, idxh, idxw] = maxval
                        selected_idx = np.random.choice(len(max_pos[0]))  
                        sel_h, sel_w = max_pos[0][selected_idx], max_pos[1][selected_idx]
                        self.max_index[idxn, idxc, idxh * self.stride + sel_h, idxw * self.stride + sel_w] = 1
        return self.output
    def forward_speedup(self, input):
        # TODO: 改进forward函数，使得计算加速
        start_time = time.time()
        self.input = input # [N, C, H, W]
        N, C, H, W = self.input.shape
        self.height_out = (H - self.kernel_size) // self.stride + 1
        self.width_out = (W - self.kernel_size) // self.stride + 1
        self.input_col = im2col(self.input,self.kernel_size, self.stride).reshape(N, C, -1, self.height_out, self.width_out)
        output = self.input_col.max(axis=2, keepdims=True)
        self.output = output.reshape(N, C, self.height_out, self.width_out)        
        self.max_index = (self.input_col == output)
        return self.output
    def backward_speedup(self, top_diff):
        # TODO: 改进backward函数，使得计算加速
        N, C, H, W = self.input.shape
        pool_diff = (self.max_index * top_diff[:, :, np.newaxis, :, :]).reshape(N, -1, self.height_out * self.width_out)
        bottom_diff = col2im(pool_diff, H, W, self.kernel_size, C, 0, self.stride)
        return bottom_diff
    def backward_raw_book(self, top_diff):
        bottom_diff = np.zeros(self.input.shape)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        pool_window = self.input[idxn, idxc,
                            idxh * self.stride : idxh * self.stride + self.kernel_size, 
                            idxw * self.stride : idxw * self.stride + self.kernel_size]
                        max_index = np.argmax(pool_window)
                        max_index = np.unravel_index(max_index, [self.kernel_size, self.kernel_size])
                        bottom_diff[idxn, idxc, 
                            idxh * self.stride + max_index[0], 
                            idxw * self.stride + max_index[1]] = top_diff[idxn, idxc, idxh, idxw] 
        show_matrix(top_diff, 'top_diff--------')
        show_matrix(bottom_diff, 'max pooling d_h ')
        return bottom_diff

class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert np.prod(self.input_shape) == np.prod(self.output_shape)
        print('\tFlatten layer with input shape %s, output shape %s.' % (str(self.input_shape), str(self.output_shape)))
    def forward(self, input):
        assert list(input.shape[1:]) == list(self.input_shape)
        # matconvnet feature map dim: [N, height, width, channel]
        # ours feature map dim: [N, channel, height, width]
        self.input = np.transpose(input, [0, 2, 3, 1])
        self.output = self.input.reshape([self.input.shape[0]] + list(self.output_shape))
        show_matrix(self.output, 'flatten out ')
        return self.output
    def backward(self, top_diff):
        assert list(top_diff.shape[1:]) == list(self.output_shape)
        top_diff = np.transpose(top_diff, [0, 3, 1, 2])
        bottom_diff = top_diff.reshape([top_diff.shape[0]] + list(self.input_shape))
        show_matrix(bottom_diff, 'flatten d_h ')
        return bottom_diff
