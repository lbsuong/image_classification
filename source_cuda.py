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
def max_matrix_kernel(result, values):
    """
    hàm hỗ trợ cho hàm matrix_max
    tìm max trong mảng values, lưu vào result[0,1]
    """
    i, j = cuda.grid(2)
    # Atomically store to result[0,1,2] from values[i, j, k]
    cuda.atomic.max(result, (0, 1), values[i, j])


@cuda.jit
def max_index_kernel(max_value, X, index):
    '''
    hàm hỗ trợ cho hàm matrix_max
    tìm chỉ số của giá trị max trong mảng X
    '''
    row, col = cuda.grid(2)
    if row >= X.shape[0] or col >= X.shape[1]:
        return
    if X[row, col] == max_value:
        index[0] = row
        index[1] = col


def matrix_max(X, blockSize=(32, 32)):
    '''
    sử dụng hàm này để gọi trên host
    cấu tham số tương tự như hàm trên cpu
    '''
    result = np.zeros((2, 2))
    index = np.array([0, 0])
    blockspergrid_y = math.ceil(X.shape[1] / blockSize[1])
    blockspergrid_x = math.ceil(X.shape[0] / blockSize[0])
    gridSize = (blockspergrid_x, blockspergrid_y)
    max_matrix_kernel[gridSize, blockSize](result, X)
    max_value = result[0, 1]
    max_index_kernel[gridSize, blockSize](max_value, X, index)
    return max_value, (index[0], index[1])


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
def maxpool_forward_kernel(input, output, poolSize):
    c, r = cuda.grid(2)
    for node in range(input.shape[0]):
        if r < output.shape[1] and c < output.shape[2]:
            output[node, r, c] = 0
    temp_max = input[node,r*poolSize, c*poolSize]

    for filterRow in range(poolSize):
        for filterCol in range(1,poolSize):
            if(input[node,r*poolSize + filterRow, c*poolSize + filterCol] > max):
                temp_max = input[node,r*poolSize + filterRow, c*poolSize + filterCol]
    output[node, r, c] = temp_max

    return output

@jit
def maxpool_forward(input, poolSize,block_size = (32, 32)):
    input_num = input.shape[0]
    input_h = input.shape[1]
    input_w = input.shape[2]
    output_num = input_num
    output_h= input_h/poolSize
    output_w= input_w/poolSize
    output = np.zeros(shape=(output_num, output_h, output_w))
    grid_size = (math.ceil(output_h / block_size[0]),math.ceil(output_w / block_size[1]))
    maxpool_forward_kernel[grid_size, block_size](input, poolSize, output)
    return output


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


@cuda.jit
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

	

def main():
    print('main function')


if __name__ == "__main__":
    main()
