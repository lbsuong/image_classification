from numba import cuda, jit
import numpy as np
import math
from numpy.lib.function_base import average
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
def dot_kernel(A, B, C):
    c, r = cuda.grid(2)
    if r >= C.shape[0] or c >= C.shape[1]:
        return
    C[r, c] = 0
    for col in range(A.shape[1]):
        C[r, c] += A[r, col] * B[col, c]


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
def conv_forward_kernel(input, filters, output):
    c, r, node = cuda.grid(3)
    if node >= output.shape[0] or r >= output.shape[1] or c >= output.shape[2]:
        return
    output[node, r, c] = 0
    for filterRow in range(filters.shape[1]):
        for filterCol in range(filters.shape[2]):
            output[node, r,
                   c] += input[r + filterRow, c +
                               filterCol] * filters[node, filterRow, filterCol]


def conv_forward_kernel_wrapper(d_input, d_filters, blockSize=(32, 32)):
    d_output = cuda.device_array(
        (d_filters.shape[0], d_input.shape[0] - d_filters.shape[1] + 1,
         d_input.shape[1] - d_filters.shape[2] + 1),
        dtype=float)
    gridSize = (math.ceil(d_output.shape[2] / blockSize[0]),
                math.ceil(d_output.shape[1] / blockSize[1]), d_output.shape[0])
    conv_forward_kernel[gridSize, blockSize](d_input, d_filters, d_output)
    return d_output


@cuda.jit
def maxpool_forward_kernel(input, poolSize, output, maxPosition):
    c, r, node = cuda.grid(3)
    if node >= output.shape[0] or r >= output.shape[1] or c >= output.shape[2]:
        return
    startRow = r * poolSize
    startCol = c * poolSize
    maxRow = startRow
    maxCol = startCol
    maxValue = input[node, maxRow, maxCol]
    for filterRow in range(poolSize):
        for filterCol in range(poolSize):
            tempRow = startRow + filterRow
            tempCol = startCol + filterCol
            if input[node, tempRow, tempCol] > maxValue:
                maxValue = input[node, tempRow, tempCol]
                maxRow = tempRow
                maxCol = tempCol
    output[node, r, c] = maxValue
    maxPosition[node, r, c] = (maxRow, maxCol)


def maxpool_forward_kernel_wrapper(d_input, poolSize, blockSize=(32, 32)):
    d_output = cuda.device_array(
        (d_input.shape[0], math.ceil(d_input.shape[1] / poolSize),
         math.ceil(d_input.shape[2] / poolSize)),
        dtype=float)
    d_maxPosition = cuda.device_array(
        (d_input.shape[0], math.ceil(d_input.shape[1] / poolSize),
         math.ceil(d_input.shape[2] / poolSize), 2),
        dtype=int)
    gridSize = (math.ceil(d_output.shape[2] / blockSize[0]),
                math.ceil(d_output.shape[1] / blockSize[1]), d_output.shape[0])
    maxpool_forward_kernel[gridSize, blockSize](d_input, poolSize, d_output,
                                                d_maxPosition)
    return d_output, d_maxPosition


@cuda.jit
def softmax_kernel(d_input, d_output, d_bias):
    image = cuda.grid(1)
    if image >= d_output.shape[0]:
        return
    for j in range(d_output.shape[1]):
        d_output[image, j] = d_input[image, j] + d_bias[image]
        d_output[image, j] = math.exp(d_output[image, j])
    total = 0
    for j in range(d_output.shape[1]):
        total += d_output[image, j]
    for j in range(d_output.shape[1]):
        d_output[image, j] /= total


@jit
def softmax(X):
    total = 0
    for i in X:
        total += math.exp(i)
    output = np.zeros(len(X))
    for i in range(len(X)):
        output[i] = math.exp(X[i]) / total
    return output


def softmax_forward_kernel_wrapper(d_input, d_weight, bias,
                                   blockSize=(32, 32)):
    d_input_reshape = d_input.reshape(
        1, d_input.shape[0] * d_input.shape[1] * d_input.shape[2])
    d_preSoftmax = cuda.device_array(
        (d_input_reshape.shape[0], d_weight.shape[1]), dtype=float)

    gridSize = (math.ceil(d_preSoftmax.shape[1] / blockSize[0]),
                math.ceil(d_preSoftmax.shape[0] / blockSize[1]))
    dot_kernel[gridSize, blockSize](d_input_reshape, d_weight, d_preSoftmax)

    output = d_preSoftmax.copy_to_host().flatten()
    for i in range(len(output)):
        output[i] += bias[i]
    return softmax(output)


@jit
def cost_entropy_loss(x):
    return -math.log(x)


@jit
def max_value_index(X):
    max = X[0]
    index = 0
    for i in range(len(X)):
        if (X[i] > max):
            max = X[i]
            index = i
    return index


def softmax_backprop_kernel_wrapper(d_L_d_out,
                                    d_weight,
                                    d_maxpoolOutput,
                                    postSoftmax,
                                    blockSize=(32, 32)):
    for i, gradient in enumerate(d_L_d_out):
        if gradient == 0:
            continue

        # Gradient của output so với biến preSoftmax
        d_out_d_preSoftmax = np.array(
            [-x * postSoftmax[i] for x in postSoftmax], dtype=float)
        d_out_d_preSoftmax[i] = postSoftmax[i] * (1 - postSoftmax[i])

        # Gradient của hàm lỗi so với biến preSoftmax
        d_d_L_d_preSoftmax = cuda.to_device(
            np.array([gradient * x for x in d_out_d_preSoftmax], dtype=float))

        d_maxpoolOutputsLength = d_maxpoolOutput.shape[0] * \
            d_maxpoolOutput.shape[1] * d_maxpoolOutput.shape[2]

        # Gradient của hàm lỗi so với biến weigth
        d_d_L_d_w = cuda.device_array(
            (len(d_d_L_d_preSoftmax), d_maxpoolOutputsLength), dtype=float)
        gridSize = (math.ceil(d_maxpoolOutputsLength / blockSize[0]),
                    math.ceil(len(d_d_L_d_preSoftmax) / blockSize[1]))
        dot_kernel[gridSize, blockSize](d_d_L_d_preSoftmax.reshape(
            len(d_d_L_d_preSoftmax),
            1), d_maxpoolOutput.reshape(1, d_maxpoolOutputsLength), d_d_L_d_w)
        cuda.synchronize()

        # Gradient của hàm lỗi so với biến bias
        d_d_L_d_b = d_d_L_d_preSoftmax

        # Gradient của hàm lỗi so với biến inputs
        d_d_L_d_input = cuda.device_array((1, d_weight.shape[1]), dtype=float)
        gridSize = (math.ceil(d_weight.shape[1] / blockSize[0]),
                    math.ceil(1 / blockSize[1]))
        dot_kernel[gridSize, blockSize](d_d_L_d_preSoftmax.reshape(
            1, len(d_d_L_d_preSoftmax)), d_weight, d_d_L_d_input)
        cuda.synchronize()

        return d_d_L_d_input.reshape(
            d_maxpoolOutput.shape), d_d_L_d_w.copy_to_host(
            ), d_d_L_d_b.copy_to_host()


@cuda.jit
def maxpool_backprop_kernel(d_L_d_out, d_L_d_input, maxPosition):
    c, r, node = cuda.grid(3)
    if node >= d_L_d_out.shape[0] or r >= d_L_d_out.shape[
            1] or c >= d_L_d_out.shape[2]:
        return
    d_L_d_input[node, maxPosition[node, r, c, 0],
                maxPosition[node, r, c, 1]] = d_L_d_out[node, r, c]


def maxpool_backprop_kernel_wrapper(d_d_L_d_out,
                                    d_maxPosition,
                                    convOutputShape,
                                    blockSize=(32, 32)):
    d_d_L_d_input = cuda.to_device(np.zeros(convOutputShape))
    gridSize = (math.ceil(d_d_L_d_out.shape[2] / blockSize[0]),
                math.ceil(d_d_L_d_out.shape[1] / blockSize[1]),
                d_d_L_d_out.shape[0])
    maxpool_backprop_kernel[gridSize, blockSize](d_d_L_d_out, d_d_L_d_input,
                                                 d_maxPosition)
    return d_d_L_d_input


@cuda.jit
def conv_backprop_kernel(d_L_d_out, d_L_d_filters, normalizedImage):
    c, r, node = cuda.grid(3)
    if node >= d_L_d_filters.shape[0] or r >= d_L_d_filters.shape[
            1] or c >= d_L_d_filters.shape[2]:
        return
    for d_L_d_out_row in range(d_L_d_out.shape[1]):
        for d_L_d_out_col in range(d_L_d_out.shape[2]):
            d_L_d_filters[node, r, c] += d_L_d_out[
                node, d_L_d_out_row,
                d_L_d_out_col] * normalizedImage[d_L_d_out_row + r,
                                                 d_L_d_out_col + c]


def conv_backprop_kernel_wrapper(d_d_L_d_out,
                                 convFiltersShape,
                                 d_normalizedImage,
                                 blockSize=(32, 32)):
    d_d_L_d_filters = cuda.to_device(np.zeros(convFiltersShape))
    gridSize = (math.ceil(d_d_L_d_filters.shape[2] / blockSize[0]),
                math.ceil(d_d_L_d_filters.shape[1] / blockSize[1]),
                d_d_L_d_filters.shape[0])
    conv_backprop_kernel[gridSize, blockSize](d_d_L_d_out, d_d_L_d_filters,
                                              d_normalizedImage)
    return d_d_L_d_filters.copy_to_host()


@jit
def average_sum_gradient(softmaxWeightGradient, newSoftmaxWeightGradient,
                         softmaxBiasGradient, newSoftmaxBiasGradient,
                         convFilterGradient, newConvFilterGradient, numImage):
    for r in range(softmaxWeightGradient.shape[0]):
        for c in range(softmaxWeightGradient.shape[1]):
            softmaxWeightGradient[r, c] += (newSoftmaxWeightGradient[r, c] /
                                            numImage)
    for i in range(len(softmaxBiasGradient)):
        softmaxBiasGradient[i] += (newSoftmaxBiasGradient[i] / numImage)
    for node in range(convFilterGradient.shape[0]):
        for r in range(convFilterGradient.shape[1]):
            for c in range(convFilterGradient.shape[2]):
                convFilterGradient[node, r,
                                   c] += (newConvFilterGradient[node, r, c] /
                                          numImage)


@jit
def update_softmax_layer(weigth, bias, learningRate, d_L_d_w, d_L_d_b):
    for row in range(weigth.shape[0]):
        for col in range(weigth.shape[1]):
            weigth[row, col] -= learningRate * d_L_d_w[row, col]

    for i in range(len(d_L_d_b)):
        bias[i] -= learningRate * d_L_d_b[i]


@jit
def update_conv_layer(convFilters, learningRate, d_L_d_filters):
    for node in range(convFilters.shape[0]):
        for row in range(convFilters.shape[1]):
            for col in range(convFilters.shape[2]):
                convFilters[node, row, col] -= learningRate * \
                    d_L_d_filters[node, row, col]


def train(d_trainImages, trainLabels, learningRate, batchSize, convFilters,
          maxpoolSize, softmaxWeight, softmaxBias):
    loss = 0
    accuracy = 0

    for batch in range(0, d_trainImages.shape[0], batchSize):
        # Tạo mini-batch
        d_imageBatch = d_trainImages[batch:batch + batchSize]
        labelBatch = trainLabels[batch:batch + batchSize]

        d_convFilters = cuda.to_device(convFilters)
        d_softmaxWeightTranspose = cuda.to_device(softmaxWeight.transpose())
        d_softmaxWeight = cuda.to_device(softmaxWeight)

        softmaxWeightGradient = np.zeros(softmaxWeight.shape)
        softmaxBiasGradient = np.zeros(softmaxBias.shape)
        convFilterGradient = np.zeros(convFilters.shape)

        for i in range(d_imageBatch.shape[0]):
            # Lan truyền xuôi
            d_convOutput = conv_forward_kernel_wrapper(d_imageBatch[i],
                                                       d_convFilters)
            cuda.synchronize()
            d_maxpoolOutput, d_maxPosition = maxpool_forward_kernel_wrapper(
                d_convOutput, maxpoolSize)
            cuda.synchronize()
            postSoftmax = softmax_forward_kernel_wrapper(
                d_maxpoolOutput, d_softmaxWeightTranspose, softmaxBias)
            cuda.synchronize()

            # Tính tổng cost-entropy loss và đếm số lượng các dự đoán đúng
            loss += cost_entropy_loss(postSoftmax[labelBatch[i]])
            predictedLabel = max_value_index(postSoftmax)
            if predictedLabel == labelBatch[i]:
                accuracy += 1

            # Tính gradient
            d_L_d_out = np.zeros(softmaxWeight.shape[0])
            d_L_d_out[labelBatch[i]] = -1 / postSoftmax[labelBatch[i]]
            d_newSoftmaxGradient, newSoftmaxWeightGradient, newSoftmaxBiasGradient = softmax_backprop_kernel_wrapper(
                d_L_d_out, d_softmaxWeight, d_maxpoolOutput, postSoftmax)
            cuda.synchronize()
            d_maxpoolGradient = maxpool_backprop_kernel_wrapper(
                d_newSoftmaxGradient, d_maxPosition, d_convOutput.shape)
            cuda.synchronize()
            newConvFilterGradient = conv_backprop_kernel_wrapper(
                d_maxpoolGradient, d_convFilters.shape, d_imageBatch[i])
            cuda.synchronize()

            # Cộng trung bình gradient
            average_sum_gradient(softmaxWeightGradient,
                                 newSoftmaxWeightGradient, softmaxBiasGradient,
                                 newSoftmaxBiasGradient, convFilterGradient,
                                 newConvFilterGradient, d_imageBatch.shape[0])

        update_softmax_layer(softmaxWeight, softmaxBias, learningRate,
                             softmaxWeightGradient, softmaxBiasGradient)
        update_conv_layer(convFilters, learningRate, convFilterGradient)

    return loss / d_trainImages.shape[0], (accuracy / d_trainImages.shape[0])


def validate(d_validateImages, validateLabels, convFilters, maxpoolSize,
             softmaxWeight, softmaxBias):
    loss = 0
    accuracy = 0

    d_convFilters = cuda.to_device(convFilters)
    d_softmaxWeightTranspose = cuda.to_device(softmaxWeight.transpose())
    for i in range(d_validateImages.shape[0]):
        # Lan truyền xuôi
        d_convOutput = conv_forward_kernel_wrapper(d_validateImages[i],
                                                   d_convFilters)
        cuda.synchronize()
        d_maxpoolOutput, _ = maxpool_forward_kernel_wrapper(
            d_convOutput, maxpoolSize)
        cuda.synchronize()
        postSoftmax = softmax_forward_kernel_wrapper(d_maxpoolOutput,
                                                     d_softmaxWeightTranspose,
                                                     softmaxBias)
        cuda.synchronize()

        # Tính tổng cost-entropy loss và đếm số lượng các dự đoán đúng
        loss += cost_entropy_loss(postSoftmax[validateLabels[i]])
        predictedLabel = max_value_index(postSoftmax)
        if predictedLabel == validateLabels[i]:
            accuracy += 1

    return loss / d_validateImages.shape[0], (accuracy /
                                              d_validateImages.shape[0])


@jit
def normalize(X):
    '''
    Chuẩn hoá các phần tử trong mảng một chiều X về dạng [0,1] bằng cách chia cho "max".

    Input:
          @ "X" là ma trận.
          @ "max" là giá trị tối đa.

    Output:
          @ Mảng các giá trị đã được normalize.
    '''
    for row in range(X.shape[0]):
        for col in range(X.shape[1]):
            X[row, col] = (X[row, col] / 255) - 0.5
    return X.astype(np.float32)


def main():
    # Khởi tạo các tham số
    convFiltersSize = 5
    numConvFilter = 32
    maxpoolSize = 2
    numClass = 10
    learningRate = 0.005
    batchSize = 100
    epoch = 10

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
        trainLoss, trainAccuracy = train(d_trainImages, trainLabels,
                                         learningRate, batchSize, convFilters,
                                         maxpoolSize, softmaxWeights,
                                         softmaxBiases)
        validateLoss, validateAccuracy = validate(d_validateImages,
                                                  validateLabels, convFilters,
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