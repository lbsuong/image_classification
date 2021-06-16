from re import U
import time
import math
# from typing_extensions import ParamSpecArgs
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
    X[row, col] = (X[row, col] / 255) - 0.5


@cuda.jit
def divide_max_kernel(X, _max, X_return):
    """
    Chuẩn hoá các phần tử trong mảng một chiều X về dạng [0,1] bằng cách chia cho "_max".

    Input:
        @ "X" là ma trận.
        @ "max" là giá trị tối đa.

    Output:
        @ Mảng các giá trị đã được normalize.
    """
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row > X.shape[0] or col > X.shape[1]:
        return
    X_return[row, col] = (X[row, col] / _max)


# có thể sử dụng SMEM để tối ưu trên version 2


@cuda.jit
def update_weights_kernel(W, gradient_w, learning_rate):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row > W.shape[0] or col > W.shape[1]:
        return
    W[row, col] = W[row, col] - learning_rate * gradient_w[row, col]


@cuda.jit
def update_biases_kernel(B, gradient_b, learning_rate):
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if idx < B.shape[0]:
        B[idx] = B[idx] - learning_rate * gradient_b[idx]


'''
@cuda.jit
def devide_all_kernel(X,z):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row > X.shape[0] or col > X.shape[1]:'''


@cuda.jit
def conv_forward_kernel(_input, filters, output, outputHeight, outputWidth):
    """
        Thực hiện lan truyền xuôi qua conv layer.

    Input:
            @ "input" là ma trận các giá trị của hình ảnh đầu vào sau khi được chuẩn hoá.
            @ "filters" là mảng các ma trận filter được tạo bởi hàm "gen_conv_filters".

    Output:
            @ Mảng các output sau khi đi qua conv layer.
    """

    node = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    outputRow = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    outputCol = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

    # output = np.zeros(shape=(filters.shape[0], outputHeight, outputWidth))

    if node > filter.shape[0] or outputRow > outputHeight or outputCol > outputWidth:
        return

    for filterRow in range(filters.shape[1]):
        for filterCol in range(filters.shape[2]):
            output[node, outputRow, outputCol] += _input[filterRow + outputRow,
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
def softmax_backprop_use_kernel(gradient_out, learningRate, weights, biases, maxpoolOutputs):
    """
	Thực hiện lan truyền ngược qua softmax layer.

	Input:
		@ "gradient_out" là gradient của hàm lỗi so với output của hàm "softmax_forward".
		@ "learningRate" là tốc độ học.
		@ "weights" là mảng các trọng số của những node trong softmax layer, các trọng số trong một node là mảng một chiều.
		@ "biases" là mảng các bias của những node trong softmax layer.
		@ "softmaxForwardFlattenedInputs" là mảng các ma trận input của hàm "softmax_forward" đã được duỗi thẳng thành mảng một chiều.
		@ "softmaxForwardInputsShape" là một tuple chứa hình dạng của input của hàm "softmax_forward".
		@ "maxpoolOutputs" là mảng các giá trị trước khi tính softmax trong hàm "softmax_forward".

	Output:
		@ "d_L_d_inputs" là gradient của hàm lỗi so với input của hàm "softmax_forward".
	"""
    maxpoolOutputsLength = maxpoolOutputs.shape[1] * maxpoolOutputs.shape[2] * maxpoolOutputs.shape[3]
    gradient_err_weights = np.zeros(gradient_out.shape[1], maxpoolOutputsLength)
    gradient_err_biases = np.zeros(gradient_out.shape[1])
    gradient_err_inputs = np.zeros((maxpoolOutputs.shape[0], 1, maxpoolOutputsLength))
    for i in range(maxpoolOutputs.shape[0]):
        block_size = (32, 32)
        grid_size = (
            math.ceil(maxpoolOutputsLength / block_size[1]), math.ceil(gradient_out[i].shape[1] / block_size[0]))
        gradient_err_weights_temp = np.zeros((gradient_out.shape[1], maxpoolOutputsLength))
        dot_kernel[grid_size, block_size](gradient_out[i].reshape(gradient_out.shape[1], 1),
                                          maxpoolOutputs[i].reshape(1, maxpoolOutputsLength),
                                          gradient_err_weights_temp)
        grid_size_1 = (math.ceil(gradient_err_weights.shape[0] / block_size[0]),
                       math.ceil(gradient_err_weights.shape[1] / block_size[1]))
        divide_max_kernel[grid_size_1, block_size](gradient_err_weights_temp,
                                                   maxpoolOutputs.shape[0],
                                                   gradient_err_weights)
        for j in range(gradient_out.shape[1]):
            gradient_err_biases[j] = gradient_out[i, j] / maxpoolOutputs.shape[0]
        grid_size_2 = (1, 1)
        dot_kernel[grid_size_2, block_size](gradient_out[i].reshape(1, gradient_out.shape[1]), weights,
                                            gradient_err_inputs[i])
    update_weights_kernel(weights, gradient_err_weights, learning_rate=learningRate)
    update_biases_kernel(biases, gradient_err_biases, learning_rate=learningRate)
    return gradient_err_inputs.reshape(maxpoolOutputs.shape)


def conv_backward_kernel(d_L_d_out, learningRate, convFilters, normalizedImages):
    d_L_d_filters = np.zeros(convFilters.shape)
    node = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    outputRow = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    outputCol = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    if node > d_L_d_filters.shape[0] or outputRow > d_L_d_filters.shape[1] or outputCol > d_L_d_filters.shape[2]:
        return
    for image in range(normalizedImages.shape[0]):
        for d_L_d_out_row in range(d_L_d_out.shape[2]):
            for d_L_d_out_col in range(d_L_d_out.shape[3]):
                d_L_d_filters[node, outputRow, outputCol] = d_L_d_filters[node, outputRow, outputCol] + d_L_d_out[
                    image, node, d_L_d_out_row, d_L_d_out_col] * normalizedImages[
                                                                image, d_L_d_out_row + outputRow, d_L_d_out_col + outputCol]
    d_L_d_filters[node, outputRow, outputCol] = d_L_d_filters[node, outputRow, outputCol] / normalizedImages.shape[0]
    convFilters[node, outputRow, outputCol] = convFilters[node, outputRow, outputCol] - learningRate * d_L_d_filters[
        node, outputRow, outputCol]


# def conv_backprop_use_gpu(d_L_d_out, learningRate, convFilters, normalizedImages):
#     d_L_d_filters = np.zeros(convFilters.shape)


def train():
    pass


def main():
    print('main function')


if __name__ == "__main__":
    main()
