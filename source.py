import time
import math
from numba.np.ufunc import parallel
import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from numba import jit, prange
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


@jit(parallel=True)
def dot(A, B):
    '''
    Nhân ma trận A và B, số cột của A phải bằng số dòng của B

    Input:
          @ "A" là ma trận.
          @ "B" là ma trận.
    Output:

          @ Ma trận của tích hai ma trận A và B.
    '''
    if (A.shape[1] != B.shape[0]):
        raise Exception('Số dòng của A không bằng số cột của B')

    output = np.zeros((A.shape[0], B.shape[1]))
    for row in prange(A.shape[0]):
        for col in prange(B.shape[1]):
            for i in range(A.shape[1]):
                output[row, col] += A[row, i] * B[i, col]
    return output


@jit
def gen_conv_filters(numConvFilter, convFilterSize):
    '''
    Khởi tạo "numConvFilter" filter với kích thước là h*w được gán các giá trị ngẫu nhiên.

    Input:
            @ "h" là số dòng của một filter.
            @ "w" là số cột của một filter.
            @ "numConvFilter" là số lượng filter cần tạo.

    Output:
            @ Mảng các ma trận filter với kích thước là h*w nhưng các filter được duỗi thẳng thành mảng một chiều.
    '''
    return np.random.rand(numConvFilter, convFilterSize, convFilterSize) / (convFilterSize * convFilterSize)


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


@jit(parallel=True)
def conv_forward(input, filters):
    '''
    Thực hiện lan truyền xuôi qua conv layer.

Input:
    @ "input" là ma trận các giá trị của hình ảnh đầu vào sau khi được chuẩn hoá.
      @ "filters" là mảng các ma trận filter được tạo bởi hàm "gen_conv_filters".

Output:
      @ Mảng các output sau khi đi qua conv layer.
'''
    outputHeight = input.shape[0] - filters.shape[1] + 1
    outputWidth = input.shape[1] - filters.shape[2] + 1
    output = np.zeros((filters.shape[0], outputHeight, outputWidth))
    for node in prange(filters.shape[0]):
        for outputRow in prange(outputHeight):
            for outputCol in prange(outputWidth):
                for filterRow in range(filters.shape[1]):
                    for filterCol in range(filters.shape[2]):
                        output[node, outputRow, outputCol] += input[filterRow + outputRow,
                                                                    filterCol + outputCol] * filters[node, filterRow, filterCol]
    return output


@jit
def matrix_max(X):
    '''
    Tìm max trong ma trận X.
    Input:
          @ "X" là ma trận.
    Output:
          @ Max của X
        @ Vị trí của max của X
    '''
    max = X[0][0]
    maxRow = 0
    maxCol = 0
    for row in range(X.shape[0]):
        for col in range(X.shape[1]):
            if (X[row][col] > max):
                max = X[row][col]
                maxRow = row
                maxCol = col
    return max, (maxRow, maxCol)


@jit(parallel=True)
def maxpool_forward(input, poolSize):
    '''
    Thực hiện lan truyền xuôi qua maxpool layer.

    Input:
          @ "input" là mảng các output của hàm "conv_forward".
          @ "poolSize" là kích thước của maxpool, maxpool là ma trận vuông.

    Output:
          @ Mảng các output sau khi đi qua maxpool layer.
    '''
    outputShape = (input.shape[0], math.ceil(
        input.shape[1] / poolSize), math.ceil(input.shape[2] / poolSize))
    output = np.zeros(outputShape)
    maxPositions = np.zeros(
        (outputShape[0], outputShape[1], outputShape[2], 2))
    for node in prange(input.shape[0]):
        for row in prange(outputShape[1]):
            for col in prange(outputShape[2]):
                output[node, row, col], maxPositions[node, row, col] = matrix_max(
                    input[node, (row * poolSize):(poolSize * (row + 1)), (col * poolSize):(poolSize * (col + 1))])
    return output, maxPositions


@jit
def gen_softmax_weights(numNode, inputLength):
    '''
    Khởi tạo "numNode" trọng số với kích thước là "length" được gán các giá trị ngẫu nhiên.

    Input:
            @ "inputLength" là số trọng số có trong một node.
            @ "numNode" số lượng các node.

    Output:
            @ Mảng "numNode" phần tử có kích thước là "inputLength".
    '''
    return np.random.rand(numNode, inputLength) / inputLength


@jit
def softmax(X):
    '''
    Tính giá trị softmax của mảng một chiều X.  
    Input:
          @ "X" là mảng một chiều.

    Output:
          @ Mảng các giá trị softmax được tính từ X.
    '''
    total = 0
    for i in X:
        total += math.exp(i)
    output = np.zeros(len(X))
    for i in range(len(X)):
        output[i] = math.exp(X[i]) / total
    return output


@jit
def softmax_forward(input, weights, biases):
    '''
    Thực hiện lan truyền xuôi qua softmax layer, softmax layer là một fully connected layer.

    Input:
          @ "input" là mảng các output của hàm "maxpool_forward".
          @ "weights" là mảng các trọng số của những node trong softmax layer.
          @ "biases" là mảng các bias của những node trong softmax layer.

    Output:
          @ Mảng các giá trị trước khi tính softmax.
          @ Mảng các giá trị sau khi tính softmax.
    '''
    input_reshape = input.reshape(
        1, input.shape[0] * input.shape[1] * input.shape[2])
    preSoftmax = dot(input_reshape, weights.transpose()).flatten()
    for i in range(len(preSoftmax)):
        preSoftmax[i] += biases[i]
    postSoftmax = softmax(preSoftmax)
    return postSoftmax


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


@jit
def compute_gradient(d_L_d_out, postSoftmax):
    for i, gradient in enumerate(d_L_d_out):
        if gradient == 0:
            continue

        # Gradient của output so với biến preSoftmax
        d_out_d_preSoftmax = np.zeros(len(postSoftmax))
        for k in range(len(postSoftmax)):
            d_out_d_preSoftmax[k] = -postSoftmax[k] * postSoftmax[i]
        d_out_d_preSoftmax[i] = postSoftmax[i] * (1 - postSoftmax[i])

        # Gradient của hàm lỗi so với biến preSoftmax
        d_L_d_preSoftmax = np.zeros(len(d_out_d_preSoftmax))
        for j in range(len(d_out_d_preSoftmax)):
            d_L_d_preSoftmax[j] = gradient * d_out_d_preSoftmax[j]

        return d_L_d_preSoftmax


@jit(parallel=True)
def softmax_backprop(d_L_d_preSoftmax, learningRate, weights, biases, maxpoolOutputs):
    '''
          Thực hiện lan truyền ngược qua softmax layer.

          Input:
                  @ "d_L_d_out" là gradient của hàm lỗi so với output của hàm "softmax_forward".
                  @ "learningRate" là tốc độ học.
                  @ "weights" là mảng các trọng số của những node trong softmax layer, các trọng số trong một node là mảng một chiều.
                  @ "biases" là mảng các bias của những node trong softmax layer.
                  @ "softmaxForwardFlattenedInputs" là mảng các ma trận input của hàm "softmax_forward" đã được duỗi thẳng thành mảng một chiều.
                  @ "softmaxForwardInputsShape" là một tuple chứa hình dạng của input của hàm "softmax_forward".
                  @ "preSoftmax" là mảng các giá trị trước khi tính softmax trong hàm "softmax_forward".

          Output:
                  @ "d_L_d_inputs" là gradient của hàm lỗi so với input của hàm "softmax_forward".
          '''
    maxpoolOutputsLength = maxpoolOutputs.shape[1] * \
        maxpoolOutputs.shape[2] * maxpoolOutputs.shape[3]
    d_L_d_w = np.zeros((d_L_d_preSoftmax.shape[1], maxpoolOutputsLength))
    d_L_d_b = np.zeros(d_L_d_preSoftmax.shape[1])
    d_L_d_inputs = np.zeros((maxpoolOutputs.shape[0], 1, maxpoolOutputsLength))

    for i in prange(maxpoolOutputs.shape[0]):
        # Gradient của hàm lỗi so với biến weights
        d_L_d_w_temp = dot(d_L_d_preSoftmax[i].reshape(
            d_L_d_preSoftmax.shape[1], 1), maxpoolOutputs[i].reshape(1, maxpoolOutputsLength))
        for row in prange(d_L_d_w.shape[0]):
            for col in prange(d_L_d_w.shape[1]):
                d_L_d_w[row, col] += d_L_d_w_temp[row, col] / \
                    maxpoolOutputs.shape[0]

        # Gradient của hàm lỗi so với biến biases
        for j in range(d_L_d_preSoftmax.shape[1]):
            d_L_d_b[j] += d_L_d_preSoftmax[i, j] / maxpoolOutputs.shape[0]

        # Gradient của hàm lỗi so với biến inputs
        d_L_d_inputs[i] = dot(d_L_d_preSoftmax[i].reshape(
            1, d_L_d_preSoftmax.shape[1]), weights)

        # Cập nhật weights
    for row in prange(weights.shape[0]):
        for col in prange(weights.shape[1]):
            weights[row, col] -= learningRate * d_L_d_w[row, col]

            # Cập nhật biases
    for j in range(len(d_L_d_b)):
        biases[j] -= learningRate * d_L_d_b[j]

    return d_L_d_inputs.reshape(maxpoolOutputs.shape)


@jit(parallel=True)
def maxpool_backprop(d_L_d_out, maxPositions, maxpoolSize, convOutputsShape):
    '''
    Thực hiện lan truyền ngược qua maxpool layer. 

    Input:
          @ "d_L_d_out" là gradient của hàm lỗi so với output của hàm "maxpool_forward".
          @ "convForwardOutputs" là mảng các input của hàm "maxpool_forward". 

    Output:
          @ "d_L_d_input" là gradient của hàm lỗi so với input của hàm "maxpool_forward".
    '''
    d_L_d_inputs = np.zeros(
        (d_L_d_out.shape[0], convOutputsShape[0], convOutputsShape[1], convOutputsShape[2]))
    for image in prange(d_L_d_out.shape[0]):
        for node in prange(d_L_d_out.shape[1]):
            for row in prange(d_L_d_out.shape[2]):
                for col in prange(d_L_d_out.shape[3]):
                    d_L_d_inputs[image, node, maxPositions[image, node, row, col, 0] + row * maxpoolSize,
                                 maxPositions[image, node, row, col, 1] + col * maxpoolSize] = d_L_d_out[image, node, row, col]
    return d_L_d_inputs


@jit(parallel=True)
def conv_backprop(d_L_d_out, learningRate, convFilters, normalizedImages):
    '''
    Thực hiện lan truyền ngược qua maxpool layer.

    Input:
          @ "d_L_d_out" là gradient của hàm lỗi so với output của hàm "conv_forward".
          @ "learningRate" là tốc độ học.
          @ "convFilters" là mảng các conv filter trong hàm "conv_forward".
          @ "normalizedImage" là ma trận các giá trị của hình ảnh đầu vào sau khi được chuẩn hoá.

    Output: None
    '''
    d_L_d_filters = np.zeros(convFilters.shape)
    for image in prange(normalizedImages.shape[0]):
        for node in prange(convFilters.shape[0]):
            for filter_row in range(convFilters.shape[1]):
                for filter_col in range(convFilters.shape[2]):
                    for d_L_d_out_row in prange(d_L_d_out.shape[2]):
                        for d_L_d_out_col in prange(d_L_d_out.shape[3]):
                            d_L_d_filters[node, filter_row, filter_col] += d_L_d_out[image, node, d_L_d_out_row,
                                                                                     d_L_d_out_col] * normalizedImages[image, d_L_d_out_row + filter_row, d_L_d_out_col + filter_col]
    for node in prange(d_L_d_filters.shape[0]):
        for row in prange(d_L_d_filters.shape[1]):
            for col in prange(d_L_d_filters.shape[2]):
                d_L_d_filters[node, row, col] /= normalizedImages.shape[0]
    for node in prange(convFilters.shape[0]):
        for row in prange(convFilters.shape[1]):
            for col in prange(convFilters.shape[2]):
                convFilters[node, row, col] -= learningRate * \
                    d_L_d_filters[node, row, col]


@jit
def max_value_index(X):
    max = X[0]
    index = 0
    for i in range(len(X)):
        if (X[i] > max):
            max = X[i]
            index = i
    return index


@jit(parallel=True)
def test(imageBatch, labelBatch, convFilters, maxpoolSize, maxpoolOutputs, maxPositions, softmaxWeights, softmaxBiases, convOutputsShape, loss, accuracy, d_L_d_preSoftmax):
    for i in prange(imageBatch.shape[0]):
        # Lan truyền xuôi.
        convOutputs = conv_forward(imageBatch[i], convFilters)
        maxpoolOutputs[i], maxPositions[i] = maxpool_forward(
            convOutputs, maxpoolSize)
        postSoftmax = softmax_forward(
            maxpoolOutputs[i], softmaxWeights, softmaxBiases)

        convOutputsShape[0] = convOutputs.shape[0]
        convOutputsShape[1] = convOutputs.shape[1]
        convOutputsShape[2] = convOutputs.shape[2]

        # Tính tổng cost-entropy loss và đếm số lượng các dự đoán đúng.
        loss += cost_entropy_loss(postSoftmax[labelBatch[i]])
        predictedLabel = max_value_index(postSoftmax)
        if predictedLabel == labelBatch[i]:
            accuracy += 1

        # Tính gradient
        d_L_d_out = np.zeros(softmaxWeights.shape[0])
        d_L_d_out[labelBatch[i]] = -1 / postSoftmax[labelBatch[i]]
        d_L_d_preSoftmax[i] = compute_gradient(d_L_d_out, postSoftmax)


@jit(parallel=True)
def train(trainImages, trainLabels, learningRate, batchSize, convFilters, maxpoolSize, softmaxWeights, softmaxBiases, loss, accuracy):
    for batch in prange(math.ceil(trainImages.shape[0] / batchSize)):
        # Tạo mini-batch
        imageBatch = trainImages[batch * batchSize: batchSize * (batch + 1)]
        labelBatch = trainLabels[batch * batchSize: batchSize * (batch + 1)]

        maxpoolOutputs = np.empty((imageBatch.shape[0], convFilters.shape[0], math.ceil(
            (imageBatch.shape[1] - convFilters.shape[1] + 1) / maxpoolSize), math.ceil((imageBatch.shape[2] - convFilters.shape[2] + 1) / maxpoolSize)))
        maxPositions = np.empty((imageBatch.shape[0], convFilters.shape[0], math.ceil(
            (imageBatch.shape[1] - convFilters.shape[1] + 1) / maxpoolSize), math.ceil((imageBatch.shape[2] - convFilters.shape[2] + 1) / maxpoolSize), 2), dtype=np.int_)
        d_L_d_preSoftmax = np.empty((imageBatch.shape[0], len(softmaxBiases)))
        convOutputsShape = (0, 0, 0)

        for i in range(imageBatch.shape[0]):
            # Lan truyền xuôi.
            convOutputs = conv_forward(imageBatch[i], convFilters)
            maxpoolOutputs[i], maxPositions[i] = maxpool_forward(
                convOutputs, maxpoolSize)
            postSoftmax = softmax_forward(
                maxpoolOutputs[i], softmaxWeights, softmaxBiases)
            convOutputsShape = convOutputs.shape

            # Tính tổng cost-entropy loss và đếm số lượng các dự đoán đúng.
            loss[0] += cost_entropy_loss(postSoftmax[labelBatch[i]])
            predictedLabel = max_value_index(postSoftmax)
            if predictedLabel == labelBatch[i]:
                accuracy[0] += 1

            # Tính gradient
            d_L_d_out = np.zeros(softmaxWeights.shape[0])
            d_L_d_out[labelBatch[i]] = -1 / postSoftmax[labelBatch[i]]
            d_L_d_preSoftmax[i] = compute_gradient(d_L_d_out, postSoftmax)

        # Cập nhật trọng số và bias
        gradients = softmax_backprop(
            d_L_d_preSoftmax, learningRate, softmaxWeights, softmaxBiases, maxpoolOutputs)
        gradients = maxpool_backprop(
            gradients, maxPositions, maxpoolSize, convOutputsShape)
        conv_backprop(
            gradients, learningRate, convFilters, imageBatch)

    # Tính trung bình cost-entropy loss và phần trăm số dự đoán đúng.
    numImage = len(trainImages)
    loss[0] /= numImage
    accuracy[0] /= numImage
    # return avgLoss, accuracy


@jit(parallel=True)
def validate(validateImages, validateLabels, convFilters, maxpoolSize, softmaxWeights, softmaxBiases, loss, accuracy):
    for i in prange(validateImages.shape[0]):
        # Lan truyền xuôi.
        convOutputs = conv_forward(validateImages[i], convFilters)
        maxpoolOutputs, _ = maxpool_forward(convOutputs, maxpoolSize)
        postSoftmax = softmax_forward(
            maxpoolOutputs, softmaxWeights, softmaxBiases)

        # Tính tổng cost-entropy loss và đếm số lượng các dự đoán đúng.
        loss[0] += cost_entropy_loss(postSoftmax[validateLabels[i]])
        predictedLabel = max_value_index(postSoftmax)
        if predictedLabel == validateLabels[i]:
            accuracy[0] += 1

    # Tính trung bình cost-entropy loss và phần trăm số dự đoán đúng.
    numImage = len(validateImages)
    loss[0] /= numImage
    accuracy[0] /= numImage


def main():
    # Khởi tạo các tham số
    convFiltersSize = 5
    numConvFilter = 32
    maxpoolSize = 2
    numClass = 10
    learningRate = 0.005
    batchSize = 100
    epoch = 20

    # Load dữ liệu, chia tập train test
    print("Loading data...")
    (trainImages, trainLabels), (validateImages, validateLabels) = mnist.load_data()
    normalizeTrainImages = np.zeros(trainImages.shape)
    normalizeValidateImages = np.zeros(validateImages.shape)
    print("Normalizing...")
    for i in range(len(trainLabels)):
        normalizeTrainImages[i] = normalize(trainImages[i].astype(np.float32))
    for i in range(len(validateLabels)):
        normalizeValidateImages[i] = normalize(
            validateImages[i].astype(np.float32))

    # Khởi tạo các filter và trọng số
    print("Initiating parameters...")
    convFilters = gen_conv_filters(numConvFilter, convFiltersSize)
    softmaxWeightsLength = (math.ceil((trainImages.shape[1] - convFiltersSize + 1) / maxpoolSize)) * math.ceil(
        ((trainImages.shape[2] - convFiltersSize + 1) / maxpoolSize)) * numConvFilter
    softmaxWeights = gen_softmax_weights(numClass, softmaxWeightsLength)
    softmaxBiases = np.zeros(numClass)

    trainingLoss = np.zeros(1)
    trainingAccuracy = np.zeros(1)
    validationLoss = np.zeros(1)
    validationAccuracy = np.zeros(1)

    # Compile hàm train() và validate()
    print("Compiling...")
    train(normalizeTrainImages, trainLabels, learningRate, batchSize,
          convFilters, maxpoolSize, softmaxWeights, softmaxBiases, trainingLoss, trainingAccuracy)
    validate(normalizeValidateImages, validateLabels, convFilters,
             maxpoolSize, softmaxWeights, softmaxBiases, validationLoss, validationAccuracy)

    # Fit
    print("\n--------------Our model--------------\n")
    start = time.time()
    for e in range(epoch):
        if epoch != 1:
            print("Epoch {e}/{epoch}".format(e=e+1, epoch=epoch))
            print('\t', end='')
        epochStart = time.time()
        trainingShuffler = np.random.permutation(len(trainLabels))
        validationShuffler = np.random.permutation(len(validateLabels))
        train(
            normalizeTrainImages[trainingShuffler], trainLabels[trainingShuffler], learningRate, batchSize, convFilters, maxpoolSize, softmaxWeights, softmaxBiases, trainingLoss, trainingAccuracy)
        validate(
            normalizeValidateImages[validationShuffler], validateLabels[validationShuffler], convFilters, maxpoolSize, softmaxWeights, softmaxBiases, validationLoss, validationAccuracy)
        epochEnd = time.time()
        print("{second}s - loss: {trainingLoss:.4f} - accuracy: {trainingAccuracy:.4f} - val_loss: {validationLoss:.4f} - val_accuracy: {validationAccuracy:.4f}".format(
            second=int(epochEnd-epochStart), trainingLoss=trainingLoss[0], trainingAccuracy=trainingAccuracy[0], validationLoss=validationLoss[0], validationAccuracy=validationAccuracy[0]))
    stop = time.time()
    print("Total runtime:", time.strftime(
        "%H:%M:%S", time.gmtime(stop - start)))

    # keras
    print("\n--------------Keras model--------------\n")
    model = Sequential()
    model.add(Conv2D(numConvFilter, kernel_size=convFiltersSize, input_shape=(
        trainImages.shape[1], trainImages.shape[2], 1), use_bias=False))
    model.add(MaxPooling2D(pool_size=maxpoolSize))
    model.add(Flatten())
    model.add(Dense(numClass, activation='softmax'))

    model.compile(SGD(learning_rate=learningRate),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    start = time.time()
    model.fit(
        np.expand_dims(normalizeTrainImages, axis=3),
        to_categorical(trainLabels),
        batch_size=batchSize, epochs=epoch,
        validation_data=(np.expand_dims(normalizeValidateImages,
                                        axis=3), to_categorical(validateLabels)),
        verbose=2
    )
    stop = time.time()
    print("Total runtime:", time.strftime(
        "%H:%M:%S", time.gmtime(stop - start)))


if __name__ == "__main__":
    main()
