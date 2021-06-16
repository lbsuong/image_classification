from re import U
import time
import math
from unittest import signals
import numba
from numba.cuda.stubs import blockDim, threadIdx
import numpy as np
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
    if y > C.shape[0] or x > C.shape[1]:
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
# blkIdx = blockDim.x - 1
# unfinishBlk = blockDim.x : số lượng những block chưa hoàn thành


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


#     Thực hiện lan truyền xuôi qua maxpool layer.

#   Input:
#   	@ "input" là mảng các output của hàm "conv_forward".
#   	@ "poolSize" là kích thước của maxpool, maxpool là ma trận vuông.
  
#   Output:
#   	@ Mảng các output sau khi đi qua maxpool layer.

#   outputShape = (input.shape[0], math.ceil(input.shape[1] / poolSize), math.ceil(input.shape[2] / poolSize))
#   output = np.zeros(shape=outputShape)
#   for node in range(input.shape[0]):
#     for row in range(outputShape[1]):
#       for col in range(outputShape[2]):
#         max, _ = matrix_max(input[node, (row * poolSize):(poolSize * (row + 1)), (col * poolSize):(poolSize * (col + 1))])
#         output[node, row, col] = max
#   return output     

@cuda.jit
def maxpool_forward_kernel(input, output, poolSize):
    input_node = input.shape[0]
    input_h = input.shape[1]
    input_w = input.shape[2]
    output_h= input_h/poolSize
    output_w= input_w/poolSize
    node = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    outputRow = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    outputCol = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    # c, r = cuda.grid(2)

    if node > input_node or outputRow > output_h or outputCol > output_w:
        return
    output[node, outputRow, outputCol]= input[node,outputRow*poolSize, outputCol*poolSize]
    temp_max = input[node,outputRow*poolSize, outputCol*poolSize]

    for filterRow in range(poolSize):
        for filterCol in range(1,poolSize):
            if(input[node,outputRow*poolSize + filterRow, outputCol*poolSize + filterCol] > max):
                temp_max = input[node,outputRow*poolSize + filterRow, outputCol*poolSize + filterCol]
    output[node, outputRow, outputCol] = temp_max

    return output
def main():
    print('main function')


if __name__ == "__main__":
    main()
