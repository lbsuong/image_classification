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
    if row < X.shape[0] and col < X.shape[1]:
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
def conv_forward_kernel(input, filters, output):
    c, r = cuda.grid(2)
    for node in range(filters.shape[0]):
        if r < output.shape[1] and c < output.shape[2]:
            output[node, r, c] = 0
            for filter_row in range(filters.shape[1]):
                for filter_col in range(filters.shape[2]):
                    output[node, r, c] += input[r + filter_row, c +
                                                filter_col] * filters[node, filter_row, filter_col]


def conv_forward(convInput, filters, block_size=(32, 32)):
    '''
    for host calling
    '''
    convOutput = np.empty((filters.shape[0], convInput.shape[0] - filters.shape[1] +
                          1, convInput.shape[1] - filters.shape[2] + 1), dtype=float)

    grid_size = (math.ceil(convOutput.shape[2] / block_size[0]),
                 math.ceil(convOutput.shape[1] / block_size[1]))
    conv_forward_kernel[grid_size, block_size](convInput, filters, convOutput)
    return convOutput


@cuda.jit
def maxpool_forward_kernel(input, output, poolSize):
    c, r = cuda.grid(2)
    for node in range(input.shape[0]):
        if r < output.shape[1] and c < output.shape[2]:
            temp_max = input[node, r*poolSize, c*poolSize]

            for filterRow in range(poolSize):
                for filterCol in range(1, poolSize):
                    if(input[node, r*poolSize + filterRow, c*poolSize + filterCol] > temp_max):
                        temp_max = input[node, r*poolSize +
                                         filterRow, c*poolSize + filterCol]
            output[node, r, c] = temp_max


def maxpool_forward(input, poolSize, block_size=(32, 32)):
    input_num = input.shape[0]
    input_h = input.shape[1]
    input_w = input.shape[2]
    output_num = input_num
    output_h = input_h//poolSize
    output_w = input_w//poolSize
    output = np.empty((output_num, output_h, output_w), dtype=float)
    grid_size = (
        math.ceil(output_h / block_size[0]), math.ceil(output_w / block_size[1]))
    maxpool_forward_kernel[grid_size, block_size](input, poolSize, output)
    return output


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
    if row < X.shape[0] and col < X.shape[1]:
        X_return[row, col] = X_return[row, col] + (X[row, col] / _max)


@cuda.jit
def update_weights_kernel(W, gradient_w, learning_rate):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row < W.shape[0] and col < W.shape[1]:
        W[row, col] = W[row, col] - learning_rate * gradient_w[row, col]


@cuda.jit
def update_biases_kernel(B, gradient_b, learning_rate):
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if idx < B.shape[0]:
        B[idx] = B[idx] - learning_rate * gradient_b[idx]


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
    maxpoolOutputsLength = maxpoolOutputs.shape[1] * \
        maxpoolOutputs.shape[2] * maxpoolOutputs.shape[3]
    gradient_err_biases = np.zeros(gradient_out.shape[1])
    cuda_gradient_err_inputs = cuda.device_array(
        (maxpoolOutputs.shape[0], 1, maxpoolOutputsLength), dtype=float)

    gradient_err_weights = np.zeros(
        (gradient_out.shape[1], maxpoolOutputsLength))
    cuda_gradient_err_weights = cuda.to_device(gradient_err_weights)

    cuda_gradient_err_biases = cuda.device_array(gradient_out.shape[1])

    cuda_weights = cuda.to_device(weights)
    cuda_biases = cuda.to_device(biases)
    block_size = (32, 32)
    for i in range(maxpoolOutputs.shape[0]):
        stream = cuda.stream()
        grid_size = (
            math.ceil(maxpoolOutputsLength / block_size[0]), math.ceil(gradient_out.shape[1] / block_size[1]))
        cuda_gradient_out_ = cuda.to_device(
            np.array(gradient_out[i].reshape(gradient_out.shape[1], 1)), stream=stream)
        cuda_maxpoolOutputs = cuda.to_device(
            np.array(maxpoolOutputs[i].reshape(1, maxpoolOutputsLength)))
        cuda_gradient_err_weights_temp = cuda.device_array(
            (gradient_out.shape[1], maxpoolOutputsLength))
        dot_kernel[grid_size, block_size](cuda_gradient_out_,
                                          cuda_maxpoolOutputs,
                                          cuda_gradient_err_weights_temp)
        # a = dot(gradient_out[i].reshape(gradient_out.shape[1], 1), maxpoolOutputs[i].reshape(1, maxpoolOutputsLength))
        # cuda.synchronize()
        # b = cuda_gradient_err_weights_temp.copy_to_host()
        # cuda_gradient_err_weights_temp = cuda.to_device(gradient_err_weights_temp)
        grid_size_1 = (math.ceil(cuda_gradient_err_weights.shape[1] / block_size[0]),
                       math.ceil(cuda_gradient_err_weights.shape[0] / block_size[1]))
        divide_max_kernel[grid_size_1, block_size](
            cuda_gradient_err_weights_temp, maxpoolOutputs.shape[0], cuda_gradient_err_weights)
        # gradient_err_weights_1 = cuda_gradient_err_weights.copy_to_host()
        for j in range(gradient_out.shape[1]):
            gradient_err_biases[j] = gradient_out[i, j] / \
                maxpoolOutputs.shape[0]

        cuda_gradient_out_1 = cuda.to_device(
            gradient_out[i].reshape(1, gradient_out.shape[1]))
        grid_size_2 = (math.ceil(cuda_weights.shape[1] / block_size[0]), math.ceil(
            cuda_gradient_out_1.shape[0] / block_size[1]))
        dot_kernel[grid_size_2, block_size](cuda_gradient_out_1, cuda_weights,
                                            cuda_gradient_err_inputs[i])
        # b = dot(gradient_out[i].reshape(1, gradient_out.shape[1]), weights)
        # p = gradient_out[i].reshape(1, gradient_out.shape[1]) @ weights
        # g = cuda_gradient_err_inputs[i].copy_to_host()
        # print(np.mean(np.abs(g-b)))
        # print("sdsdsd: ", np.mean(np.abs(p-b)))

    gradient_err_inputs = cuda_gradient_err_inputs.copy_to_host()

    grid_size_update_weights = (math.ceil(
        cuda_weights.shape[1] / block_size[0]), math.ceil(cuda_weights.shape[0] / block_size[1]))
    grid_size_update_biases = math.ceil(cuda_biases.shape[0] / block_size[0])

    update_weights_kernel[grid_size_update_weights, block_size](
        cuda_weights, cuda_gradient_err_weights, learningRate)
    update_biases_kernel[grid_size_update_biases, block_size[0]](
        cuda_biases, cuda_gradient_err_biases, learningRate)

    weights = cuda_weights.copy_to_host()
    biases = cuda_biases.copy_to_host()
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
    d_L_d_filters[node, outputRow, outputCol] = d_L_d_filters[node,
                                                              outputRow, outputCol] / normalizedImages.shape[0]
    convFilters[node, outputRow, outputCol] = convFilters[node, outputRow, outputCol] - learningRate * d_L_d_filters[
        node, outputRow, outputCol]


def main():
    print('main function')


if __name__ == "__main__":
    main()
