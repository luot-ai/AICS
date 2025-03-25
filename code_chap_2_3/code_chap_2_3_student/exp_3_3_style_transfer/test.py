import numpy as np
from numpy.lib.stride_tricks import as_strided

def im2col_fast(input, kernel_size, stride):
    N, C, H, W = input.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1

    shape = (N, C, H_out, W_out, kernel_size, kernel_size)
    strides = (
        input.strides[0],  # Batch 维度
        input.strides[1],  # Channel 维度
        input.strides[2] * stride,  # 滑动窗口在 H 方向移动
        input.strides[3] * stride,  # 滑动窗口在 W 方向移动
        input.strides[2],  # Kernel 纵向步长 ✅ 正确
        input.strides[3]   # Kernel 横向步长 ✅ 正确
    )

    patches = as_strided(input, shape=shape, strides=strides)
    #print(patches[0,0])
    return patches.reshape(N, C, H_out * W_out,kernel_size * kernel_size)

# 测试
input = np.arange(1, 17).reshape(1, 1, 4, 4).astype(np.float32)
print(input)
output = im2col_fast(input, 3, 1)

print(output.shape)  # (1, 1, 9, 9)
print("First im2col patch:\n", output[0, 0])