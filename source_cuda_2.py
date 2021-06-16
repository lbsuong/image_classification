from re import U
import time
import math
from typing import BinaryIO
from unittest import signals
import numba
from numba.cuda.stubs import blockDim, blockIdx, gridsize, threadIdx
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.testing._private.utils import gisfinite
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from numba import cuda, jit
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# có thể sử dụng SMEM để tối ưu ở version 2


@cuda.jit
def dot_kernel(A, B, C):
    '''
    Nhân ma trận A và B, số cột của A phải bằng số dòng của B

    Input:
        @ "A" là ma trận.
        @ "B" là ma trận.
    Output:

        @ Ma trận của tích hai ma trận A và B.
    '''
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if y >= C.shape[0] or x >= C.shape[1]:
        return
    C[x, y] = 0
    for col in range(A.shape[1]):
        C[x, y] += A[x, col] * B[col, y]


@cuda.jit
def normalize_kernel(X):
    '''
    Chuẩn hoá các phần tử trong mảng một chiều X về dạng [0,1] bằng cách chia cho "max".

    Input:
        @ "X" là ma trận.
        @ "max" là giá trị tối đa.

    Output:
        @ Mảng các giá trị đã được normalize.
    '''
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row > X.shape[0] or col > X.shape[1]:
        return
    X[row, col] = (X[row, col]/255) - 0.5

# có thể sử dụng SMEM để tối ưu trên version 2


@cuda.jit
def conv_forward_kernel(input, filters, output, outputHeight, outputWidth):
    '''
        Thực hiện lan truyền xuôi qua conv layer.

    Input:
            @ "input" là ma trận các giá trị của hình ảnh đầu vào sau khi được chuẩn hoá.
            @ "filters" là mảng các ma trận filter được tạo bởi hàm "gen_conv_filters".

    Output:
            @ Mảng các output sau khi đi qua conv layer.
    '''

    node = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    outputRow = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    outputCol = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

    #output = np.zeros(shape=(filters.shape[0], outputHeight, outputWidth))

    if node > filter.shape[0] or outputRow > outputHeight or outputCol > outputWidth:
        return

    for filterRow in range(filters.shape[1]):
        for filterCol in range(filters.shape[2]):
            output[node, outputRow, outputCol] += input[filterRow + outputRow,
                                                        filterCol + outputCol] * filters[node, filterRow, filterCol]

# hàm này chỉ tìm max - cần thêm một hàm tìm vị trí - ma trận X sẽ bị sửa đổi
# gía trị max là giá trị nằm ở đầu mảng X - X[0]
# blkIdx = gridSize.x - 1
# unfinishBlk = gridSize.x : số lượng những block chưa hoàn thành


@cuda.jit
def matrix_max_kernel(X, blkIdx, unfinsishBlk):
    # blkIdx = cuda.device_array(shape=(1,1), dtype=np.int)
    newBlkIdx = cuda.shared.array((1, 1), np.dtype(int))
    # if cuda.threadIdx.x == 0 and (blkIdx[0] < 0 or blkIdx[0] > cuda.blockDim.x):
    #     blkIdx[0] = 0

    if cuda.threadIdx == 0:
        newBlkIdx[0] = blkIdx
        blkIdx = blkIdx - 1
    # cuda.threadfence_system()
    cuda.syncthreads()

    nBefore = cuda.blockDim.x * newBlkIdx[0] * 2

    stride = 1
    while stride < cuda.blockDim.x:
        curr_idx = nBefore + stride * cuda.threadIdx.x
        next_idx = nBefore + stride * cuda.threadIdx.x + stride
        curr_row = curr_idx // X.shape[1]
        curr_col = curr_idx % X.shape[1]
        next_row = next_idx // X.shape[1]
        next_col = next_idx % X.shape[1]

        if X[curr_row, curr_col] < X[next_row, next_col]:
            X[curr_row, curr_col] = X[next_row, next_col]
        cuda.syncthreads()
        stride = stride * 2

    if cuda.threadIdx.x == 0:
        if newBlkIdx[0] < cuda.blockDim.x - 1:
            # chở block phía sau thực hiện xong
            while unfinsishBlk > newBlkIdx[0] + 1:
                pass
            curr_idx = nBefore
            next_idx = nBefore + 2 * cuda.blockDim.x
            curr_row = curr_idx // X.shape[1]
            curr_col = curr_idx % X.shape[1]
            next_row = next_idx // X.shape[1]
            next_col = next_idx % X.shape[1]
            if X[curr_row, curr_col] < X[next_row, next_col]:
                X[curr_row, curr_col] = X[next_row, next_col]
            cuda.threadfence()
        unfinsishBlk = unfinsishBlk - 1
        cuda.threadfence_system()


@cuda.jit
def find_max_index_kernel(X, max_value, index):
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if X[row, col] == max_value:
        index = (row, col)


# hàm này chỉ được gọi trong hàm kernel
@cuda.jit(device=True)
def gen_softmax_weights_kernel(numNode, inputLength):
    return np.random.rand(numNode, inputLength) / inputLength


# hàm này tính tổng của ma trận X - hỗ trợ cho hàm softmax()
@cuda.jit
def total_kernel(X, blkIdx, unfinsishBlk):
    # blkIdx = cuda.device_array(shape=(1,1), dtype=np.int)
    newBlkIdx = cuda.shared.array((1, 1), np.dtype(int))
    # if cuda.threadIdx.x == 0 and (blkIdx[0] < 0 or blkIdx[0] > cuda.blockDim.x):
    #     blkIdx[0] = 0

    if cuda.threadIdx == 0:
        newBlkIdx[0] = blkIdx
        blkIdx = blkIdx - 1
    # cuda.threadfence_system()
    cuda.syncthreads()

    nBefore = cuda.blockDim.x * newBlkIdx[0] * 2

    stride = 1
    while stride < cuda.blockDim.x:
        curr_idx = nBefore + stride * cuda.threadIdx.x
        next_idx = nBefore + stride * cuda.threadIdx.x + stride
        X[curr_idx] += X[next_idx]

        cuda.syncthreads()
        stride = stride * 2

    if cuda.threadIdx.x == 0:
        if newBlkIdx[0] < cuda.blockDim.x - 1:
            # chở block phía sau thực hiện xong
            while unfinsishBlk > newBlkIdx[0] + 1:
                pass
            curr_idx = nBefore
            next_idx = nBefore + 2 * cuda.blockDim.x
            X[curr_idx] += X[next_idx]
            cuda.threadfence()
        unfinsishBlk = unfinsishBlk - 1
        cuda.threadfence_system()

# hàm này tính ma trận output của softmax - hỗ trợ cho hàm softmax()


@cuda.jit
def output_softmax_kernel(X, total, output):
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if idx > len(X):
        return
    output[idx] = math.exp(X[idx]) / total


@jit
def softmax(X, blockSize):
    X_d = cuda.to_device(X)
    X_d_2 = cuda.to_device(X)
    gridSize = (len(X) + (blockSize - 1)) // blockSize
    total_kernel[gridSize, blockSize](X_d, gridSize - 1, gridSize)
    cuda.synchronize()
    output = np.empty(shape=(1, len(X)))
    # output = cuda.device_array(shape=(1,len(X)))
    output_softmax_kernel[gridSize, blockSize](X_d_2, X_d[0], output)
    cuda.synchronize()
    return output

# hỗ trợ cho hàm softmax_forward
# preSoftmax = dot(input, weights.transpose()).flatten()


@cuda.jit
def pre_softmax_kernel(biases, preSoftmax):
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if idx >= len(preSoftmax):
        return
    preSoftmax[idx] += biases[idx]


@jit
def softmax_forward(input, weights, biases, blockSize):
    input = input.reshape(1, input.shape[0] * input.shape[1] * input.shape[2])
    preSoftmax = np.empty(shape=(input.shape[0], weights.transpose().shape[1]))
    blockspergrid_x = math.ceil(preSoftmax.shape[0] / blockSize[0])
    blockspergrid_y = math.ceil(preSoftmax.shape[1] / blockSize[1])
    gridSize = (blockspergrid_x, blockspergrid_y)
    dot_kernel[gridSize, blockSize](input, weights.transpose(), preSoftmax)
    cuda.synchronize()
    postSoftmax = softmax(preSoftmax, blockSize.x)
    return preSoftmax, postSoftmax


@jit
def cost_entropy_loss(x):
    '''
    Hàm tính đỘ lỗi.

    Input:
          @ "x" là giá trị lớn nhất của mảng trả về từ hàm "softmax_forward". 

    Output:
          @ Độ lỗi cost-entropy loss.
    '''
    return -math.log(x)


def main():
    print('main function')


if __name__ == "__main__":
    main()