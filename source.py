from source_cuda import matrix_max_kernel
from keras.datasets import mnist
from numba import jit
from math import exp
from math import log
import numpy as np
import warnings

warnings.filterwarnings("ignore")


@jit
def dot(A, B):
    """
	Nhân ma trận A và B, số cột của A phải bằng số dòng của B

	Input:
		@ "A" là ma trận.
		@ "B" là ma trận.
	Output:
		@ Ma trận của tích hai ma trận A và B.
	"""

    if A.shape[1] != B.shape[0]:
        return None

    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i, j] = C[i, j] + A[i, k] * B[k, j]
    return C


@jit
def gen_conv_filters(h, w, numConvFilter):
    """
	Khởi tạo "numConvFilter" filter với kích thước là h*w được gán các giá trị ngẫu nhiên.
	Input:
		@ "h" là số dòng của một filter.
		@ "w" là số cột của một filter.
		@ "numConvFilter" là số lượng filter cần tạo.
	Output:
		@ Mảng các ma trận filter với filter có kích thước là h*w.
	"""
    C = [0] * numConvFilter
    for i in range(len(C)):
        C[i] = np.random.rand(h, w)
    return np.array(C)


@jit
def conv_forward(_input, filters):
    """
	Thực hiện lan truyền xuôi qua conv layer.
    Input:
    	@ "input" là ma trận các giá trị của hình ảnh đầu vào sau khi được chuẩn hoá.
		@ "filters" là mảng các ma trận filter được tạo bởi hàm "gen_conv_filters".
    Output:
		@ Mảng các output sau khi đi qua conv layer.
    """
    print("max of input", np.max(_input))
    conv_h = _input.shape[0] - filters.shape[1] + 1
    conv_w = _input.shape[1] - filters.shape[2] + 1
    Output_conv = np.zeros((filters.shape[0], conv_h, conv_w))
    for id_filter in range(filters.shape[0]):
        for conv_r in range(conv_h):
            for conv_c in range(conv_w):
                for filter_r in range(filters.shape[1]):
                    for filter_c in range(filters.shape[2]):
                        Output_conv[id_filter, conv_r, conv_c] = Output_conv[id_filter, conv_r, conv_c] + _input[
                            conv_r + filter_r, conv_c + filter_c] * filters[id_filter, filter_r, filter_c]
    return Output_conv


@jit
def maxpool_forward(_input, poolSize):
    """
	Thực hiện lan truyền xuôi qua maxpool layer.

	Input:
		@ "input" là mảng các output của hàm "conv_forward".
    	@ "poolSize" là kích thước của maxpool, maxpool là ma trận vuông.

    Output:
		@ Mảng các output sau khi đi qua maxpool layer.
	"""
    print("max: ", np.max(_input))
    output_forward = np.zeros((_input.shape[0], int(_input.shape[1] / poolSize), int(_input.shape[2] / poolSize)))
    for id_conv in range(output_forward.shape[0]):
        for id_row in range(0, output_forward.shape[1]):
            for id_col in range(0, output_forward.shape[2]):
                temp_arr = []
                for id_pool_row in range(poolSize):
                    for id_pool_col in range(poolSize):
                        temp_arr.append(_input[id_conv, id_row + id_pool_row, id_col + id_pool_col])
                output_forward[id_conv, id_row, id_col] = max(temp_arr)
        return output_forward


@jit
def softmax(X):
    """
	Tính giá trị softmax của mảng một chiều X.
	Input:
		@ "X" là mảng một chiều.
	Output:
		@ Mảng các giá trị softmax được tính từ X.
	"""
    output_softmax = np.zeros(X.shape[0])
    for i in range(output_softmax.shape[0]):
        print(X[i])
        output_softmax[i] = np.exp(X[i]) / sum(np.exp(X))
    return output_softmax


@jit
def gen_softmax_weights(inputLength, numNode):
    """
	Khởi tạo "numNode" trọng số với kích thước là "length" được gán các giá trị ngẫu nhiên.
	Input:
		@ "inputLength" là số trọng số có trong một node.
		@ "numNode" số lượng các node.
	Output:
		@ Mảng "numNode" phần tử có kích thước là "inputLength".
	"""
    softmax_weights = np.random.rand(numNode, inputLength) / inputLength
    return softmax_weights


@jit
def flatten(X):
    """
	Duỗi thẳng X thành mảng một chiều.
	Input:
		@ "X" là mảng với bất cứ chiều nào.
	Output:
		@ Mảng một chiều.
	"""
    return np.array(X).flatten()


@jit
def softmax_forward(_input, weights, biases):
    """
	Thực hiện lan truyền xuôi qua softmax layer, softmax layer là một fully connected layer.
	Input:
		@ "input" là mảng các output của hàm "maxpool_forward".
		@ "weights" là mảng các trọng số của những node trong softmax layer.
		@ "biases" là mảng các bias của những node trong softmax layer.

    Output:
		@ Mảng các giá trị trước khi tính softmax.
		@ Mảng các giá trị sau khi tính softmax.
		@ Shape của "input".
	"""
    temp_input = []
    print("Hello softmax forward")
    for i in range(_input.shape[0]):
        temp_input.append(_input[i].flatten())
    temp_input = _input.flatten()
    preSoftmax = np.zeros(weights.shape[0], dtype='float64')
    print("biases", biases)
    for i in range(preSoftmax.shape[0]):
        preSoftmax[i] = dot(np.array([temp_input], dtype='float64'), np.transpose([weights[i]])) + biases[i]
        print(preSoftmax[i])
    results_softmax = np.zeros(preSoftmax.shape[0])
    sum_exp = np.sum(np.exp(preSoftmax))
    for i in range(preSoftmax.shape[0]):
        results_softmax[i] = np.exp(preSoftmax[i]) / sum_exp
    return preSoftmax, results_softmax


@jit
def max(X):
    """
	Tìm max trong mảng một chiều X.
	Input:
		@ "X" là mảng một chiều.
	Output:
		@ Max của X
	"""
    return np.max(X)
def matrix_max(X):
    _max = X[0][0]
    _maxRow = 0
    _maxCol = 0
    for row in range(X.shape[0]):
        for col in range(X.shape[1]):
            if(X[row, col] > _max):
                _max = X[row, col]
                _maxRow = row
                _maxCol = col
    return _max, (_maxRow, _maxCol)

@jit
def cost_entropy_loss(x):
    """
	Hàm tính đỘ lỗi.

	Input:
		@ "x" là giá trị lớn nhất của mảng trả về từ hàm "softmax_forward".
	Output:
		@ Độ lỗi cost-entropy loss.
	"""
    return -np.log(x)


@jit
def normalize(X, _max):
    """
	Chuẩn hoá các phần tử trong mảng một chiều X về dạng [0,1] bằng cách chia cho "max".
	Input:
		@ "X" là mảng một chiều có "n" phần tử.
		@ "max" là giá trị tối đa.
	Output:
		@ Mảng các giá trị đã được normalize.
	"""
    X_new = np.zeros(X.shape)
    for i in range(X_new.shape[0]):
        for j in range(X_new.shape[1]):
            X_new[i] = X[i] / _max
    return X_new


@jit
def reshape(X, shape):
    """
	Chuyển mảng một chiều "X" sang mảng "shape" chiều.
	Input:
		@ "X" là mảng một chiều.
		@ "shape" là một tuple chứa hình dạng của mảng sau khi reshape.
	Output:
		@ Mảng có hình dạng là "shape".
	"""


@jit
def softmax_backprop(gradient_out, learningRate, weights, biases, softmaxForwardFlattenedInputs,
                     softmaxForwardInputsShape, preSoftmax):
    """
	Thực hiện lan truyền ngược qua softmax layer.
	Input:
		@ "gradient_out" là gradient của hàm lỗi so với output của hàm "softmax_forward".
		@ "learningRate" là tốc độ học.
		@ "weights" là mảng các trọng số của những node trong softmax layer, các trọng số trong một node là mảng một chiều.
		@ "biases" là mảng các bias của những node trong softmax layer.
		@ "softmaxForwardFlattenedInputs" là mảng các ma trận input của hàm "softmax_forward" đã được duỗi thẳng thành mảng một chiều.
		@ "softmaxForwardInputsShape" là một tuple chứa hình dạng của input của hàm "softmax_forward".
		@ "preSoftmax" là mảng các giá trị trước khi tính softmax trong hàm "softmax_forward".
	Output:
		@ "gradient_in" là gradient của hàm lỗi so với input của hàm "softmax_forward".
	"""
    for i, gradient in enumerate(gradient_out):
        if gradient == 0:
            continue
        exp_softmax = np.exp(preSoftmax)
        sum_exp_softmax = np.sum(exp_softmax)
        db_dPreSoftmax = 1
        gradient_err_exp_softmax = exp_softmax[i] * exp_softmax / (sum_exp_softmax ** 2)
        print("gradient_err_exp_softmax", gradient_err_exp_softmax)
        gradient_err_exp_softmax[i] = exp_softmax[i] * (sum_exp_softmax - exp_softmax[i]) / (sum_exp_softmax ** 2)
        gradient_loss_vs_totals = gradient * gradient_err_exp_softmax
        dWeights = dot(gradient_loss_vs_totals.reshape(len(gradient_loss_vs_totals), 1),
                       softmaxForwardFlattenedInputs.reshape(1, len(softmaxForwardFlattenedInputs)))
        gradient_err_inputs = dot(gradient_err_exp_softmax.reshape(1, len(gradient_err_exp_softmax)), weights)
        print("gradient_err_inputs: ", gradient_err_inputs)
        gradient_err_biases = gradient_err_exp_softmax * db_dPreSoftmax
        print("gradient_err_biases: ", gradient_err_biases)
        for r in range(weights.shape[0]):
            for c in range(weights.shape[1]):
                weights[r, c] = weights[r, c] - learningRate * dWeights[r, c]
        for j in range(biases.shape[0]):
            biases[j] = biases[j] - learningRate * gradient_err_biases[j]
    return gradient_err_inputs.reshape(softmaxForwardInputsShape)


@jit
def maxpool_backprop(d_L_d_out, maxpoolForwardInputs, maxpoolSize):
    """
	Thực hiện lan truyền ngược qua maxpool layer.
	Input:
		@ "d_L_d_out" là gradient của hàm lỗi so với output của hàm "maxpool_forward".
		@ "maxpoolForwardInputs" là mảng các input của hàm "maxpool_forward".
	Output:
		@ "d_L_d_input" là gradient của hàm lỗi so với input của hàm "maxpool_forward".
	"""
    dld_input = np.zeros(maxpoolForwardInputs.shape)
    for node in range(dld_input.shape[0]):
        for row in range(dld_input.shape[1]):
            for col in range(dld_input.shape[2]):
                _max, (maxPoolOutput_max_R, maxPoolOuput_max_C) = matrix_max(maxpoolForwardInputs[node,
                row * maxpoolSize : (row+1) * maxpoolSize,
                 col * maxpoolSize : (col+1)*maxpoolSize])
                dld_input[node,
                 maxPoolOutput_max_R + row * maxpoolSize, 
                 maxPoolOuput_max_C + col *maxpoolSize] = d_L_d_out[node , row, col]
    return dld_input


@jit
def conv_backprop(d_L_d_out, learningRate, convFilters, normalizedImage):
    """
	Thực hiện lan truyền ngược qua maxpool layer.
	Input:
		@ "d_L_d_out" là gradient của hàm lỗi so với output của hàm "conv_forward".
		@ "learningRate" là tốc độ học.
		@ "convFilters" là mảng các conv filter trong hàm "conv_forward".
		@ "normalizedImage" là ma trận các giá trị của hình ảnh đầu vào sau khi được chuẩn hoá.

	Output: None
	"""
    d_filters = np.zeros(convFilters.shape)
    for idx in range(convFilters.shape[0]):
        for filter_row in range(convFilters.shape[1]):
            for filter_col in range(convFilters.shape[2]):
                for d_L_d_out_row in range(d_L_d_out.shape[1]):
                    for d_L_d_out_col in range(d_L_d_out.shape[2]):
                        d_filters[idx, filter_row, filter_col] = d_L_d_out[idx,filter_row, filter_col] * normalizedImage[d_L_d_out_row + filter_row,
                           d_L_d_out_col + filter_col]
    for idx in range(convFilters.shape[0]):
        for filter_row in range(convFilters.shape[1]):
            for filter_col in range(convFilters.shape[2]):
                convFilters[idx, filter_row, filter_row] -= learningRate * d_L_d_out[idx, filter_row, filter_row]
    return None

@jit
def train(trainImages, trainLabels, learningRate, convFilters, maxpoolSize, softmaxWeights, softmaxBiases, numNode):
    loss = 0
    accuracy = 0
    for i in range(trainImages.shape[0]):
        # Lan truyền xuôi.
        # Chuẩn hoá mảng image về [0,1] để tính toán dễ dàng hơn.
        convInput = normalize(trainImages[i], 255)
        print("convInput", convInput)
        maxpoolInputs = conv_forward(convInput, convFilters)
        print("maxpoolInputs", maxpoolInputs)
        softmaxInputs = maxpool_forward(maxpoolInputs, maxpoolSize)
        print("softmaxInputs", softmaxInputs)
        preSoftmax, postSoftmax = softmax_forward(softmaxInputs, softmaxWeights, softmaxBiases)
        # Tính tổng cost-entropy loss và đếm số lượng các dự đoán đúng.
        loss += cost_entropy_loss(postSoftmax[trainLabels[i]])
        predictedLabel = max(postSoftmax)
        if predictedLabel == trainLabels[i]:
            accuracy += 1

        # Khởi tạo gradient
        gradient = np.zeros(numNode)
        gradient[trainLabels[i]] = -1 / postSoftmax[trainLabels[i]]
        print("gradient", gradient)
        # Lan truyền ngược.
        gradient = softmax_backprop(gradient, learningRate, softmaxWeights, softmaxBiases, softmaxInputs.flatten(),
                                    softmaxInputs.shape, preSoftmax)
        print("gradient: ", gradient)
        gradient = maxpool_backprop(gradient, maxpoolInputs, maxpoolSize)
        gradient = conv_backprop(gradient, learningRate, convFilters, convInput)

    # Tính trung bình cost-entropy loss và phần trăm số dự đoán đúng.
    numImage = len(trainImages)
    avgLoss = loss / numImage
    accuracy = accuracy / numImage
    return avgLoss, accuracy


@jit
def predict(image, convFilters, maxpoolSize, softmaxWeights, softmaxBiases):
    # Lan truyền xuôi.
    # Chuẩn hoá mảng image về [0,1] để tính toán dễ dàng hơn.
    convInput = normalize(image, 255)
    maxpoolInputs = conv_forward(convInput, convFilters)
    softmaxInputs = maxpool_forward(maxpoolInputs, maxpoolSize)
    _, postSoftmax, _ = softmax_forward(softmaxInputs, softmaxWeights, softmaxBiases)

    # Nhãn sẽ là phần tử có giá trị softmax cao nhất.
    predictedLabel = max(postSoftmax)
    return predictedLabel


def main():
    (trainImages, trainLabels), (testImages, testLabels) = mnist.load_data()

    # Lấy 1000 phần tử đầu tiên của tập train và test
    trainImages = trainImages[:1000]
    trainLabels = trainLabels[:1000]
    print("trainLabels:",trainLabels)
    convFiltersH = 3
    convFiltersW = 3
    numConvFilter = 8
    convFilters = gen_conv_filters(convFiltersH, convFiltersW, numConvFilter)

    maxpoolSize = 2

    numNode = 10
    h = 28
    w = 28
    softmaxWeightsLength = ((h - convFiltersH + 1) // maxpoolSize) * (
            (w - convFiltersW + 1) // maxpoolSize) * numConvFilter
    softmaxWeights = gen_softmax_weights(softmaxWeightsLength, numNode)
    softmaxBiases = np.array([0]*numNode)

    learningRate = 0.005
    avgLoss, accuracy = train(trainImages, trainLabels, learningRate, convFilters, maxpoolSize, softmaxWeights,
                              softmaxBiases, numNode)
    print("Average loss: {avgLoss:.3f} | Accuracy: {accuracy:.2f}".format(avgLoss=avgLoss, accuracy=accuracy * 100))


if __name__ == "__main__":
    main()
