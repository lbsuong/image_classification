from numba import cuda, jit, float64
from numba.cuda.stubs import gridsize
import numpy as np
import math
from tensorflow.keras.datasets import mnist
import time


@jit
def gen_conv_filters(numConvFilter, convFilterSize):
    return np.random.rand(numConvFilter, convFilterSize,
                          convFilterSize) / (convFilterSize * convFilterSize)


@jit
def gen_softmax_weights(numNode, inputLength):
    return np.random.rand(numNode, inputLength) / inputLength


@cuda.jit
def dot_2D_kernel(A, B, C):
    c, r = cuda.grid(2)
    if r >= C.shape[0] or c >= C.shape[1]:
        return
    C[r, c] = 0
    for col in range(A.shape[1]):
        C[r, c] += A[r, col] * B[col, c]


@cuda.jit
def dot_3D2D_kernel(A, B, C):
    c, r, node = cuda.grid(3)
    if node >= C.shape[0] or r >= C.shape[1] or c >= C.shape[2]:
        return
    C[node, r, c] = 0
    for col in range(B.shape[0]):
        C[node, r, c] += A[node, r, col] * B[col, c]


@cuda.jit
def dot_3D_kernel(A, B, C):
    c, r, node = cuda.grid(3)
    if node >= C.shape[0] or r >= C.shape[1] or c >= C.shape[2]:
        return
    C[node, r, c] = 0
    for col in range(B.shape[1]):
        C[node, r, c] += A[node, r, col] * B[node, col, c]


@cuda.jit
def normalize_kernel(images, output):
    c, r, image = cuda.grid(3)
    if image >= output.shape[0] or r >= output.shape[1] or c >= output.shape[2]:
        return
    output[image, r, c] = images[image, r, c] / 255 - 0.5


def normalize_wrapper(images, blockSize=(32, 32)):
    d_images = cuda.to_device(images)
    d_output = cuda.device_array(d_images.shape, dtype=float)
    gridSize = ((math.ceil(d_images.shape[2] / blockSize[0]),
                 math.ceil(d_images.shape[1] / blockSize[1]),
                 d_images.shape[0]))
    normalize_kernel[gridSize, blockSize](d_images, d_output)
    return d_output


@cuda.jit
def conv_forward_kernel(input, filters, numImage, output):
    c, r, node = cuda.grid(3)
    if node >= output.shape[1] or r >= output.shape[2] or c >= output.shape[3]:
        return
    for image in range(numImage):
        output[image, node, r, c] = 0
        for filterRow in range(filters.shape[1]):
            for filterCol in range(filters.shape[2]):
                output[image, node, r,
                       c] += input[image, r + filterRow,
                                   c + filterCol] * filters[node, filterRow,
                                                            filterCol]


def conv_forward_kernel_wrapper(d_input,
                                d_filters,
                                numImage,
                                d_output,
                                blockSize=(32, 32)):
    gridSize = (math.ceil(d_output.shape[3] / blockSize[0]),
                math.ceil(d_output.shape[2] / blockSize[1]), d_output.shape[1])
    conv_forward_kernel[gridSize, blockSize](d_input, d_filters, numImage,
                                             d_output)


@cuda.jit
def maxpool_forward_kernel(input, poolSize, numImage, output, maxPosition):
    c, r, node = cuda.grid(3)
    if node >= output.shape[1] or r >= output.shape[2] or c >= output.shape[3]:
        return
    for image in range(numImage):
        startRow = r * poolSize
        startCol = c * poolSize
        maxRow = startRow
        maxCol = startCol
        maxValue = input[image, node, maxRow, maxCol]
        for filterRow in range(poolSize):
            for filterCol in range(poolSize):
                tempRow = startRow + filterRow
                tempCol = startCol + filterCol
                if input[image, node, tempRow, tempCol] > maxValue:
                    maxValue = input[image, node, tempRow, tempCol]
                    maxRow = tempRow
                    maxCol = tempCol
        output[image, node, r, c] = maxValue
        maxPosition[image, node, r, c] = (maxRow, maxCol)


def maxpool_forward_kernel_wrapper(d_input,
                                   poolSize,
                                   numImage,
                                   d_output,
                                   d_maxPosition,
                                   blockSize=(32, 32)):
    gridSize = (math.ceil(d_output.shape[3] / blockSize[0]),
                math.ceil(d_output.shape[2] / blockSize[1]), d_output.shape[1])
    maxpool_forward_kernel[gridSize, blockSize](d_input, poolSize, numImage,
                                                d_output, d_maxPosition)


@cuda.jit
def softmax_kernel(input, d_bias, h_total, numImage, output):
    c, image = cuda.grid(2)
    if image >= numImage or c >= output.shape[1]:
        return
    output[image, c] = input[image, 0, c] + d_bias[c]
    output[image, c] = math.exp(output[image, c])
    cuda.syncthreads()
    cuda.atomic.add(h_total, image, output[image, c])
    cuda.syncthreads()
    output[image, c] /= h_total[image]


def softmax_forward_kernel_wrapper(d_input,
                                   d_weight_transpose,
                                   d_bias,
                                   numImage,
                                   d_output,
                                   blockSize=(32, 32)):
    d_input_reshape = d_input[:numImage].reshape(
        numImage, 1,
        d_input.shape[1] * d_input.shape[2] * d_input.shape[3])
    d_preSoftmax = cuda.device_array(
        (d_input_reshape.shape[0], d_input_reshape.shape[1],
         d_weight_transpose.shape[1]),
        dtype=float)

    gridSize = (math.ceil(d_preSoftmax.shape[2] / blockSize[0]),
                math.ceil(d_preSoftmax.shape[1] / blockSize[1]),
                d_preSoftmax.shape[0])
    dot_3D2D_kernel[gridSize, blockSize](d_input_reshape, d_weight_transpose,
                                         d_preSoftmax)

    total = np.zeros(d_preSoftmax.shape[0], dtype=np.float64)
    gridSize = (math.ceil(d_preSoftmax.shape[2] / blockSize[0]),
                math.ceil(d_preSoftmax.shape[0] / blockSize[1]))
    softmax_kernel[gridSize, blockSize](d_preSoftmax, d_bias, total, numImage,
                                        d_output)


@cuda.jit
def cal_d_L_d_out_kernel(postSoftmax, numImage, labelSubBatch, d_L_d_out):
    i, image = cuda.grid(2)
    if image >= numImage or i >= d_L_d_out.shape[1]:
        return
    if i == labelSubBatch[image]:
        d_L_d_out[image, i] = -1 / postSoftmax[image, i]
    else:
        d_L_d_out[image, i] = 0


def cal_d_L_d_out(d_postSoftmax,
                  d_labelSubBatch,
                  numImage,
                  d_d_L_d_out,
                  blockSize=(32, 32)):
    gridSize = (math.ceil(d_d_L_d_out.shape[1] / blockSize[0]),
                math.ceil(numImage / blockSize[1]))
    cal_d_L_d_out_kernel[gridSize, blockSize](d_postSoftmax, numImage, d_labelSubBatch,
                                              d_d_L_d_out)


@cuda.jit
def softmax_backprop_kernel(d_L_d_out, postSoftmax, numImage,
                            d_L_d_preSoftmax):
    image = cuda.grid(1)
    if image >= numImage:
        return
    for i in range(d_L_d_out.shape[1]):
        if d_L_d_out[image, i] == 0:
            continue
        d_out_d_preSoftmax = cuda.local.array(10, float64)
        for j in range(d_out_d_preSoftmax.shape[0]):
            d_out_d_preSoftmax[j] = -postSoftmax[image, j] * postSoftmax[image,
                                                                         i]
        d_out_d_preSoftmax[i] = postSoftmax[image,
                                            i] * (1 - postSoftmax[image, i])
        for j in range(d_L_d_preSoftmax.shape[1]):
            d_L_d_preSoftmax[image,
                             j] = d_L_d_out[image, i] * d_out_d_preSoftmax[j]
        break


def softmax_backprop_kernel_wrapper(d_d_L_d_out,
                                    d_weight,
                                    d_maxpoolOutput,
                                    d_postSoftmax,
                                    numImage,
                                    d_d_L_d_input,
                                    d_d_L_d_w,
                                    d_d_L_d_b,
                                    blockSize=(32, 32)):
    # Tính d_d_L_d_preSoftmax cũng là tính d_d_L_d_b vì nó trỏ cùng 1 vùng nhớ
    d_d_L_d_preSoftmax = d_d_L_d_b
    gridSize = math.ceil(numImage / blockSize[1])
    softmax_backprop_kernel[gridSize,
                            blockSize[1]](d_d_L_d_out, d_postSoftmax, numImage,
                                          d_d_L_d_preSoftmax)

    # Tính d_d_L_d_w
    d_d_L_d_preSoftmaxReshape = d_d_L_d_preSoftmax[:numImage].reshape(
        numImage, d_d_L_d_preSoftmax.shape[1], 1)
    d_maxpoolOutputReshape = d_maxpoolOutput[:numImage].reshape(
        numImage, 1, d_weight.shape[1])
    gridSize = (math.ceil(d_maxpoolOutputReshape.shape[2] / blockSize[0]),
                math.ceil(d_d_L_d_preSoftmaxReshape.shape[1] / blockSize[1]),
                d_d_L_d_preSoftmaxReshape.shape[0])
    dot_3D_kernel[gridSize, blockSize](d_d_L_d_preSoftmaxReshape,
                                       d_maxpoolOutputReshape, d_d_L_d_w)

    # Tính d_d_L_d_input
    d_d_L_d_input_temp = cuda.device_array((numImage, 1, d_weight.shape[1]),
                                           dtype=float)
    d_d_L_d_preSoftmaxReshape = d_d_L_d_preSoftmax[:numImage].reshape(
        numImage, 1, d_d_L_d_preSoftmax.shape[1])
    gridSize = (math.ceil(d_d_L_d_input_temp.shape[2] / blockSize[0]),
                math.ceil(d_d_L_d_input_temp.shape[1] / blockSize[1]),
                d_d_L_d_input_temp.shape[0])
    dot_3D2D_kernel[gridSize, blockSize](d_d_L_d_preSoftmaxReshape, d_weight,
                                         d_d_L_d_input_temp)
    d_d_L_d_input[0, :numImage] = d_d_L_d_input_temp[:numImage].reshape(
        d_maxpoolOutput[:numImage].shape)


@cuda.jit
def maxpool_backprop_zeros_kernel(d_L_d_input, numImage):
    c, r, node = cuda.grid(3)
    if node >= d_L_d_input.shape[1] or r >= d_L_d_input.shape[
            2] or c >= d_L_d_input.shape[3]:
        return
    for image in range(numImage):
        d_L_d_input[image, node, r, c] = 0


@cuda.jit
def maxpool_backprop_kernel(d_L_d_out, numImage, maxPosition, d_L_d_input):
    c, r, node = cuda.grid(3)
    if node >= d_L_d_out.shape[1] or r >= d_L_d_out.shape[
            2] or c >= d_L_d_out.shape[3]:
        return
    for image in range(numImage):
        d_L_d_input[image, node, maxPosition[image, node, r, c, 0],
                    maxPosition[image, node, r, c, 1]] = d_L_d_out[image, node,
                                                                   r, c]


def maxpool_backprop_kernel_wrapper(d_d_L_d_out,
                                    d_maxPosition,
                                    numImage,
                                    d_d_L_d_input,
                                    blockSize=(32, 32)):
    gridSize = (math.ceil(d_d_L_d_input.shape[3] / blockSize[0]),
                math.ceil(d_d_L_d_input.shape[2] / blockSize[1]),
                d_d_L_d_input.shape[1])
    maxpool_backprop_zeros_kernel[gridSize, blockSize](d_d_L_d_input, numImage)
    gridSize = (math.ceil(d_d_L_d_out.shape[3] / blockSize[0]),
                math.ceil(d_d_L_d_out.shape[2] / blockSize[1]),
                d_d_L_d_out.shape[1])
    maxpool_backprop_kernel[gridSize, blockSize](d_d_L_d_out, numImage,
                                                 d_maxPosition, d_d_L_d_input)


@cuda.jit
def conv_backprop_kernel(d_L_d_out, numImage, normalizedImage, d_L_d_filters):
    c, r, node = cuda.grid(3)
    if node >= d_L_d_filters.shape[1] or r >= d_L_d_filters.shape[
            2] or c >= d_L_d_filters.shape[3]:
        return
    for image in range(numImage):
        d_L_d_filters[image, node, r, c] = 0
        for d_L_d_out_row in range(d_L_d_out.shape[2]):
            for d_L_d_out_col in range(d_L_d_out.shape[3]):
                d_L_d_filters[image, node, r, c] += d_L_d_out[
                    image, node, d_L_d_out_row,
                    d_L_d_out_col] * normalizedImage[image, d_L_d_out_row + r,
                                                     d_L_d_out_col + c]


def conv_backprop_kernel_wrapper(d_d_L_d_out,
                                 numImage,
                                 d_normalizedImage,
                                 d_d_L_d_filters,
                                 blockSize=(32, 32)):
    gridSize = (math.ceil(d_d_L_d_filters.shape[3] / blockSize[0]),
                math.ceil(d_d_L_d_filters.shape[2] / blockSize[1]),
                d_d_L_d_filters.shape[1])
    conv_backprop_kernel[gridSize,
                         blockSize](d_d_L_d_out, numImage, d_normalizedImage,
                                    d_d_L_d_filters)


@cuda.jit
def average_sum_softmaxweight_gradient_kernel(softmaxWeightGradient,
                                              newSoftmaxWeightGradient,
                                              numImage):
    i, node, image = cuda.grid(3)
    if image >= numImage or node >= softmaxWeightGradient.shape[
            0] or i >= softmaxWeightGradient.shape[1]:
        return
    cuda.atomic.add(softmaxWeightGradient, (node, i), newSoftmaxWeightGradient[image, node, i] / numImage)


@cuda.jit
def average_sum_softmaxbias_gradient_kernel(softmaxBiasGradient,
                                            newSoftmaxBiasGradient, numImage):
    i, image = cuda.grid(2)
    if image >= numImage or i >= softmaxBiasGradient.shape[0]:
        return
    cuda.atomic.add(softmaxBiasGradient, i, newSoftmaxBiasGradient[image, i] / numImage)


@cuda.jit
def average_sum_convfilter_gradient_kernel(convFilterGradient,
                                           newConvFilterGradient, numImage):
    c, r, node = cuda.grid(3)
    if node >= convFilterGradient.shape[0] or r >= convFilterGradient.shape[
            1] or c >= convFilterGradient.shape[2]:
        return
    for image in range(numImage):
        cuda.atomic.add(convFilterGradient, (node, r, c), newConvFilterGradient[image, node, r, c] / numImage)


def average_sum_gradient(d_softmaxWeightGradient,
                         d_newSoftmaxWeightGradient,
                         d_softmaxBiasGradient,
                         d_newSoftmaxBiasGradient,
                         d_convFilterGradient,
                         d_newConvFilterGradient,
                         numImage,
                         blockSize=(32, 32)):
    gridSize = (math.ceil(d_softmaxWeightGradient.shape[1] / blockSize[0]),
                math.ceil(d_softmaxWeightGradient.shape[0] / blockSize[1]),
                numImage)
    average_sum_softmaxweight_gradient_kernel[gridSize, blockSize](
        d_softmaxWeightGradient, d_newSoftmaxWeightGradient, numImage)
    gridSize = (math.ceil(d_softmaxBiasGradient.shape[0] / blockSize[0]),
                math.ceil(numImage / blockSize[1]))
    average_sum_softmaxbias_gradient_kernel[gridSize, blockSize](
        d_softmaxBiasGradient, d_newSoftmaxBiasGradient, numImage)
    gridSize = (math.ceil(d_convFilterGradient.shape[2] / blockSize[0]),
                math.ceil(d_convFilterGradient.shape[1] / blockSize[1]),
                d_convFilterGradient.shape[0])
    average_sum_convfilter_gradient_kernel[gridSize,
                                           blockSize](d_convFilterGradient,
                                                      d_newConvFilterGradient,
                                                      numImage)


# @cuda.jit
# def update_softmax_weight_kernel(weigth, learningRate, d_L_d_w):
#     c, r = cuda.grid(2)
#     if r >= weigth.shape[0] or c >= weigth.shape[1]:
#         return
#     weigth[r, c] -= learningRate * d_L_d_w[r, c]


# @cuda.jit
# def update_softmax_bias_kernel(bias, learningRate, d_L_d_b):
#     i = cuda.grid(1)
#     if i >= bias.shape[0]:
#         return
#     bias[i] -= learningRate * d_L_d_b[i]


# def update_softmax_layer(d_weigth,
#                          d_bias,
#                          learningRate,
#                          d_d_L_d_w,
#                          d_d_L_d_b,
#                          blockSize=(32, 32)):
#     gridsize = (math.ceil(d_weigth.shape[1] / blockSize[0]),
#                 math.ceil(d_weigth.shape[0] / blockSize[1]))
#     update_softmax_weight_kernel[gridsize, blockSize](d_weigth, learningRate,
#                                                       d_d_L_d_w)
#     gridsize = math.ceil(d_bias.shape[0] / blockSize[1])
#     update_softmax_bias_kernel[gridsize, blockSize[1]](d_bias, learningRate,
#                                                        d_d_L_d_b)


# @cuda.jit
# def update_conv_filters_kernel(convFilters, learningRate, d_L_d_filters):
#     c, r, node = cuda.grid(3)
#     if node >= convFilters.shape[0] or r >= convFilters.shape[
#             1] or c >= convFilters.shape[2]:
#         return
#     convFilters[node, r, c] -= learningRate * d_L_d_filters[node, r, c]



# def update_conv_layer(d_convFilters, learningRate, d_d_L_d_filters, blockSize=(32, 32)):
#     gridsize = (
#         math.ceil(d_convFilters.shape[2] / blockSize[0]),
#         math.ceil(d_convFilters.shape[1] / blockSize[1]),
#         d_convFilters.shape[0]
#     )
#     update_conv_filters_kernel[gridsize, blockSize](d_convFilters, learningRate, d_d_L_d_filters)


@jit
def update_softmax_layer(weigth, bias, learningRate, d_L_d_w, d_L_d_b):
    # Cập nhật weigth
    for row in range(weigth.shape[0]):
        for col in range(weigth.shape[1]):
            weigth[row, col] -= learningRate * d_L_d_w[row, col]

    # Cập nhật bias
    for i in range(len(d_L_d_b)):
        bias[i] -= learningRate * d_L_d_b[i]


@jit
def update_conv_layer(convFilters, learningRate, d_L_d_filters):
    for node in range(convFilters.shape[0]):
        for row in range(convFilters.shape[1]):
            for col in range(convFilters.shape[2]):
                convFilters[node, row, col] -= learningRate * \
                    d_L_d_filters[node, row, col]


@cuda.jit(device=True)
def cost_entropy_loss(x):
    return -math.log(x)


@cuda.jit(device=True)
def max_value_index(X):
    max = X[0]
    index = 0
    for i in range(len(X)):
        if (X[i] > max):
            max = X[i]
            index = i
    return index


@cuda.jit
def cal_loss_and_accuracy_kernel(postSoftmax, labelBatch, numImage, h_loss, h_accuracy):
    image = cuda.grid(1)
    if image >= numImage:
        return
    cuda.atomic.add(h_loss, 0, cost_entropy_loss(postSoftmax[image, labelBatch[image]]))
    predictedLabel = max_value_index(postSoftmax[image])
    if predictedLabel == labelBatch[image]:
        cuda.atomic.add(h_accuracy, 0, 1)


def cal_loss_and_accuracy(d_postSoftmax, d_labelBatch, numImage, loss, accuracy, blockSize=32):
    gridsize = math.ceil(numImage / blockSize)
    cal_loss_and_accuracy_kernel[gridsize, blockSize](d_postSoftmax, d_labelBatch, numImage, loss, accuracy)


def train(d_trainImages, d_trainLabels, learningRate, batchSize, convFilters,
          maxpoolSize, softmaxWeight, softmaxBias):
    loss = np.zeros(1)
    accuracy = np.zeros(1)

    #print(convFilters[0])

    ''' Phần cấp phát vùng nhớ cố định cho device '''
    d_convOutput = cuda.device_array(
        (batchSize, convFilters.shape[0],
         d_trainImages.shape[1] - convFilters.shape[1] + 1,
         d_trainImages.shape[2] - convFilters.shape[2] + 1),
        dtype=float)
    d_maxpoolOutput = cuda.device_array(
        (d_convOutput.shape[0], d_convOutput.shape[1],
         math.ceil(d_convOutput.shape[2] / maxpoolSize),
         math.ceil(d_convOutput.shape[3] / maxpoolSize)),
        dtype=float)
    d_maxPosition = cuda.device_array(
        (d_maxpoolOutput.shape[0], d_maxpoolOutput.shape[1],
         d_maxpoolOutput.shape[2], d_maxpoolOutput.shape[3], 2),
        dtype=int)
    d_postSoftmax = cuda.device_array(
        (d_maxpoolOutput.shape[0], softmaxBias.shape[0]), dtype=float)

    d_d_L_d_out = cuda.to_device(np.zeros(d_postSoftmax.shape))

    # Thêm 1 chiều đầu tiên cho d_newSoftmaxGradient để có thể thay đỔi giá trị của nó trong hàm khác
    d_newSoftmaxGradient = cuda.device_array(
        (1, d_maxpoolOutput.shape[0], d_maxpoolOutput.shape[1],
         d_maxpoolOutput.shape[2], d_maxpoolOutput.shape[3]),
        dtype=float)
    d_newSoftmaxWeightGradient = cuda.device_array(
        (d_d_L_d_out.shape[0], softmaxWeight.shape[0],
         softmaxWeight.shape[1]),
        dtype=float)
    d_newSoftmaxBiasGradient = cuda.device_array_like(d_postSoftmax)

    d_maxpoolGradient = cuda.device_array_like(d_convOutput)

    d_newConvFilterGradient = cuda.device_array(
        (d_d_L_d_out.shape[0], convFilters.shape[0], convFilters.shape[1],
         convFilters.shape[2]),
        dtype=float)
    ''' Kết thúc phần cấp phát vùng nhớ cố định cho device '''

    for batch in range(0, d_trainImages.shape[0], batchSize):
        # Tạo mini-batch
        d_imageBatch = d_trainImages[batch:batch + batchSize]
        d_labelBatch = d_trainLabels[batch:batch + batchSize]
        numImage = d_imageBatch.shape[0]

        # Copy các tham số sang device
        d_convFilters = cuda.to_device(convFilters)
        d_softmaxWeightTranspose = cuda.to_device(softmaxWeight.transpose())
        d_softmaxWeight = cuda.to_device(softmaxWeight)
        d_softmaxBias = cuda.to_device(softmaxBias)

        # Cấp phát vùng nhớ cho các gradient tổng trung bình
        d_softmaxWeightGradient = cuda.to_device(
            np.zeros(d_softmaxWeight.shape))
        d_softmaxBiasGradient = cuda.to_device(np.zeros(d_softmaxBias.shape))
        d_convFilterGradient = cuda.to_device(np.zeros(d_convFilters.shape))

        # Lan truyền xuôi
        conv_forward_kernel_wrapper(d_imageBatch, d_convFilters, numImage,
                                    d_convOutput)
        maxpool_forward_kernel_wrapper(d_convOutput, maxpoolSize, numImage,
                                       d_maxpoolOutput, d_maxPosition)
        softmax_forward_kernel_wrapper(d_maxpoolOutput,
                                       d_softmaxWeightTranspose, d_softmaxBias,
                                       numImage, d_postSoftmax)

        # Tính độ lỗi và độ chính xác
        cal_loss_and_accuracy(d_postSoftmax, d_labelBatch, numImage, loss, accuracy)

        # Lan truyền ngược
        cal_d_L_d_out(d_postSoftmax, d_labelBatch, numImage, d_d_L_d_out)
        softmax_backprop_kernel_wrapper(d_d_L_d_out, d_softmaxWeight,
                                        d_maxpoolOutput, d_postSoftmax,
                                        numImage, d_newSoftmaxGradient,
                                        d_newSoftmaxWeightGradient,
                                        d_newSoftmaxBiasGradient)
        maxpool_backprop_kernel_wrapper(d_newSoftmaxGradient[0], d_maxPosition,
                                        numImage, d_maxpoolGradient)
        conv_backprop_kernel_wrapper(d_maxpoolGradient, numImage, d_imageBatch,
                                     d_newConvFilterGradient)

        # Cộng trung bình các gradient
        average_sum_gradient(d_softmaxWeightGradient,
                             d_newSoftmaxWeightGradient, d_softmaxBiasGradient,
                             d_newSoftmaxBiasGradient, d_convFilterGradient,
                             d_newConvFilterGradient, numImage)

        # Copy các trọng số qua host và cập nhật
        softmaxWeightGradient = d_softmaxWeightGradient.copy_to_host()
        softmaxBiasGradient = d_softmaxBiasGradient.copy_to_host()
        convFilterGradient = d_convFilterGradient.copy_to_host()
        update_softmax_layer(softmaxWeight, softmaxBias, learningRate, softmaxWeightGradient, softmaxBiasGradient)
        update_conv_layer(convFilters, learningRate, convFilterGradient)

    return loss[0] / d_trainImages.shape[0], (accuracy[0] / d_trainImages.shape[0])


def validate(d_validateImages, d_validateLabels, batchSize, convFilters, maxpoolSize,
             softmaxWeight, softmaxBias):
    loss = np.zeros(1)
    accuracy = np.zeros(1)

    # Copy các tham số sang device
    d_convFilters = cuda.to_device(convFilters)
    d_softmaxWeightTranspose = cuda.to_device(softmaxWeight.transpose())
    d_softmaxBias = cuda.to_device(softmaxBias)

    d_convOutput = cuda.device_array(
        (batchSize, convFilters.shape[0],
         d_validateImages.shape[1] - convFilters.shape[1] + 1,
         d_validateImages.shape[2] - convFilters.shape[2] + 1),
        dtype=float)
    d_maxpoolOutput = cuda.device_array(
        (d_convOutput.shape[0], d_convOutput.shape[1],
         math.ceil(d_convOutput.shape[2] / maxpoolSize),
         math.ceil(d_convOutput.shape[3] / maxpoolSize)),
        dtype=float)
    d_maxPosition = cuda.device_array(
        (d_maxpoolOutput.shape[0], d_maxpoolOutput.shape[1],
         d_maxpoolOutput.shape[2], d_maxpoolOutput.shape[3], 2),
        dtype=int)
    d_postSoftmax = cuda.device_array(
        (d_maxpoolOutput.shape[0], softmaxBias.shape[0]), dtype=float)

    for batch in range(0, d_validateImages.shape[0], batchSize):
        d_imageBatch = d_validateImages[batch:batch + batchSize]
        d_labelBatch = d_validateLabels[batch:batch + batchSize]
        numImage = d_imageBatch.shape[0]

        # Lan truyền xuôi
        conv_forward_kernel_wrapper(d_imageBatch, d_convFilters, numImage,
                                    d_convOutput)
        maxpool_forward_kernel_wrapper(d_convOutput, maxpoolSize, numImage,
                                       d_maxpoolOutput, d_maxPosition)
        softmax_forward_kernel_wrapper(d_maxpoolOutput,
                                       d_softmaxWeightTranspose, d_softmaxBias,
                                       numImage, d_postSoftmax)

        # Tính độ lỗi và độ chính xác
        cal_loss_and_accuracy(d_postSoftmax, d_labelBatch, numImage, loss, accuracy)

    return loss[0] / d_validateImages.shape[0], (accuracy[0] /
                                              d_validateImages.shape[0])


def main():
    # Khởi tạo các tham số
    convFiltersSize = 5
    numConvFilter = 32
    maxpoolSize = 2
    numClass = 10
    learningRate = 0.005
    batchSize = 100
    epoch = 20

    print("Loading data...")
    (trainImages, trainLabels), (validateImages,
                                 validateLabels) = mnist.load_data()

    print("Shuffle training data")
    trainingShuffler = np.random.permutation(len(trainLabels))
    trainImages = trainImages[trainingShuffler]
    trainLabels = trainLabels[trainingShuffler]

    print("Normalizing and copying data to GPU...")
    d_trainImages = normalize_wrapper(trainImages)
    d_validateImages = normalize_wrapper(validateImages)
    d_trainLabels = cuda.to_device(trainLabels)
    d_validateLabels = cuda.to_device(validateLabels)

    print("Initiating parameters...")
    convFilters = gen_conv_filters(numConvFilter, convFiltersSize)
    softmaxWeightsLength = (math.ceil(
        (trainImages.shape[1] - convFiltersSize + 1) /
        maxpoolSize)) * math.ceil(
            ((trainImages.shape[2] - convFiltersSize + 1) /
             maxpoolSize)) * numConvFilter
    softmaxWeights = gen_softmax_weights(numClass, softmaxWeightsLength)
    softmaxBiases = np.zeros(numClass)

    print("Start training...")
    print("\n--------------Our model--------------\n")
    start = time.time()
    for e in range(epoch):
        if epoch != 1:
            print("Epoch {e}/{epoch}".format(e=e + 1, epoch=epoch))
            print('\t', end='')
        epochStart = time.time()
        trainLoss, trainAccuracy = train(d_trainImages, d_trainLabels,
                                         learningRate, batchSize, convFilters,
                                         maxpoolSize, softmaxWeights,
                                         softmaxBiases)
        validateLoss, validateAccuracy = validate(d_validateImages,
                                                  d_validateLabels, batchSize, convFilters,
                                                  maxpoolSize, softmaxWeights,
                                                  softmaxBiases)
        epochEnd = time.time()
        print(
            "{second}s - loss: {trainingLoss:.4f} - accuracy: {trainingAccuracy:.4f} - val_loss: {validationLoss:.4f} - val_accuracy: {validationAccuracy:.4f}"
            .format(second=int(epochEnd - epochStart),
                    trainingLoss=trainLoss,
                    trainingAccuracy=trainAccuracy,
                    validationLoss=validateLoss,
                    validationAccuracy=validateAccuracy))
    stop = time.time()
    print("Total runtime:", time.strftime("%H:%M:%S",
                                          time.gmtime(stop - start)))
    return 0


if __name__ == '__main__':
    main()