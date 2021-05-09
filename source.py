from keras.datasets import mnist
from numba import jit
from math import exp
from math import log
import numpy as np

@jit
def dot(A, B):
	'''
	Nhân ma trận A và B, số cột của A phải bằng số dòng của B
	
	Input:
		@ "A" là ma trận.
		@ "B" là ma trận.
	Output:
		@ Ma trận của tích hai ma trận A và B.
	'''
	
	if A.shape[0] != B.shape[1]:
		return None
	
	C = np.zeros((A.shape[1],B.shape[0]))
	for i in range(A.shape[1]):
		for j in range(B.shape[0]):
			for k in range(A.shape[0]):
				C[i,j] = C[i,j] + A[i,k] * B[k,j]
	return C
@jit
def gen_conv_filters(h, w, numConvFilter):
	'''
	Khởi tạo "numConvFilter" filter với kích thước là h*w được gán các giá trị ngẫu nhiên.
	Input:
		@ "h" là số dòng của một filter.
		@ "w" là số cột của một filter.
		@ "numConvFilter" là số lượng filter cần tạo.
	Output:
		@ Mảng các ma trận filter với filter có kích thước là h*w.
	'''
	C = np.zeros(numConvFilter)
	for ConvFilter in C:
		ConvFilter = np.random.rand(h,w)
	return C
@jit
def conv_forward(_input, filters):
	'''
	Thực hiện lan truyền xuôi qua conv layer.
    Input:
    	@ "input" là ma trận các giá trị của hình ảnh đầu vào sau khi được chuẩn hoá.
		@ "filters" là mảng các ma trận filter được tạo bởi hàm "gen_conv_filters".
    Output:
		@ Mảng các output sau khi đi qua conv layer.
    '''
	
	conv_h = _input.shape[0] - filters.shape[1] + 1
	conv_w = _input.shape[1] - filters.shape[2] + 1
	Output_conv = np.zeros((_input.shape[0],conv_h,conv_w))
	for id_filter in range(filters.shape[0]):
		for conv_r in range(conv_h):
			for conv_c in range(conv_w):
				for filter_r in range(filters.shape[1]):
					for filter_c in range(filters.shape[2]):
						Output_conv[conv_r,conv_c] = Output_conv[conv_r,conv_c] + _input[conv_r+filter_r,conv_c+filter_c] * filters[id_filter,filter_r,filter_c]
	return Output_conv
@jit
def maxpool_forward(_input, poolSize):
	'''
	Thực hiện lan truyền xuôi qua maxpool layer.
	
	Input:
		@ "input" là mảng các output của hàm "conv_forward".
    	@ "poolSize" là kích thước của maxpool, maxpool là ma trận vuông.
		
    Output:
		@ Mảng các output sau khi đi qua maxpool layer.
	'''
	output_forward = np.zeros((_input.shape[0], int(_input.shape[1] / poolSize) + 1, int(_input.shape[2] / poolSize) + 1))
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
	'''
	Tính giá trị softmax của mảng một chiều X.
	Input:
		@ "X" là mảng một chiều.
	Output:
		@ Mảng các giá trị softmax được tính từ X.
	'''
	output_softmax = np.zeros(X.shape[0])
	for i in range(output_softmax.shape[0]):
		print(X[i])
		output_softmax[i] = np.exp(X[i])/ sum(np.exp(X))
	return output_softmax
@jit

def gen_softmax_weights(inputLength, numNode):
	'''
	Khởi tạo "numNode" trọng số với kích thước là "length" được gán các giá trị ngẫu nhiên.
	Input:
		@ "inputLength" là số trọng số có trong một node.
		@ "numNode" số lượng các node.
	Output:
		@ Mảng "numNode" phần tử có kích thước là "inputLength".
	'''
	softmax_weights = np.random.rand(100,size=(numNode,inputLength))
	return softmax_weights
@jit
def flatten(X):
	'''
	Duỗi thẳng X thành mảng một chiều.
	Input:
		@ "X" là mảng với bất cứ chiều nào.
	Output:
		@ Mảng một chiều.
	'''
	return np.array(X).flatten()
@jit
def softmax_forward(_input, weights, biases):
	'''
	Thực hiện lan truyền xuôi qua softmax layer, softmax layer là một fully connected layer.
	Input:
		@ "input" là mảng các output của hàm "maxpool_forward".
		@ "weights" là mảng các trọng số của những node trong softmax layer.
		@ "biases" là mảng các bias của những node trong softmax layer.
    
    Output:
		@ Mảng các giá trị trước khi tính softmax.
		@ Mảng các giá trị sau khi tính softmax.
		@ Shape của "input".
	'''
	temp_input = []
	for i in range(_input.shape[0]):
		temp_input.append(_input[i].flatten())
	temp_input = np.array(temp_input)
	results_softmax = np.zeros(_input.shape[0])
	for i in range(_input.shape[0]):
		results_softmax[i] = np.sum(_input[i]*weights[i]) + biases[i]
	return _input, results_softmax, _input.shape
@jit
def max(X):
	'''
	Tìm max trong mảng một chiều X.
	Input:
		@ "X" là mảng một chiều.
	Output:
		@ Max của X
	'''
	return np.max(X)

@jit
def cost_entropy_loss(x):
	'''
	Hàm tính đỘ lỗi.
	
	Input:
		@ "x" là giá trị lớn nhất của mảng trả về từ hàm "softmax_forward".
	Output:
		@ Độ lỗi cost-entropy loss.
	'''
	return -np.log(x)

@jit
def normalize(X, _max):
	'''
	Chuẩn hoá các phần tử trong mảng một chiều X về dạng [0,1] bằng cách chia cho "max".
	Input:
		@ "X" là mảng một chiều có "n" phần tử.
		@ "max" là giá trị tối đa.
	Output:
		@ Mảng các giá trị đã được normalize.
	'''
	X_new = np.zeros(X.shape)
	for i in range(X_new.shape[0]):
		X_new[i] = X_new[i]/_max
	return X_new
@jit
def reshape(X, shape):
	'''
	Chuyển mảng một chiều "X" sang mảng "shape" chiều.
	Input:
		@ "X" là mảng một chiều.
		@ "shape" là một tuple chứa hình dạng của mảng sau khi reshape.
	Output:
		@ Mảng có hình dạng là "shape".
	'''

@jit
def softmax_backprop(gradient_out, learningRate, weights, biases, softmaxForwardFlattenedInputs, softmaxForwardInputsShape, preSoftmax):
	'''
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
	'''
	for i, gradient in enumerate(gradient_out):
		if gradient == 0:
			continue
	pass

@jit
def maxpool_backprop(d_L_d_out, maxpoolForwardInputs):
	'''
	Thực hiện lan truyền ngược qua maxpool layer.
	Input:
		@ "d_L_d_out" là gradient của hàm lỗi so với output của hàm "maxpool_forward".
		@ "maxpoolForwardInputs" là mảng các input của hàm "maxpool_forward".
	Output:
		@ "d_L_d_input" là gradient của hàm lỗi so với input của hàm "maxpool_forward".
	'''
	pass

@jit
def conv_backprop(d_L_d_out, learningRate, convFilters, normalizedImage):
	'''
	Thực hiện lan truyền ngược qua maxpool layer.
	Input:
		@ "d_L_d_out" là gradient của hàm lỗi so với output của hàm "conv_forward".
		@ "learningRate" là tốc độ học.
		@ "convFilters" là mảng các conv filter trong hàm "conv_forward".
		@ "normalizedImage" là ma trận các giá trị của hình ảnh đầu vào sau khi được chuẩn hoá.
	
	Output: None
	'''

@jit
def train(trainImages, trainLabels, learningRate, convFilters, maxpoolSize, softmaxWeights, softmaxBiases, numNode):
	loss = 0
	accuracy = 0

	for (image, label) in zip(trainImages, trainLabels):
		# Lan truyền xuôi.
		## Chuẩn hoá mảng image về [0,1] để tính toán dễ dàng hơn.
		convInput = normalize(image, 255)
		maxpoolInputs = conv_forward(convInput, convFilters)
		softmaxInputs = maxpool_forward(maxpoolInputs, maxpoolSize)
		preSoftmax, postSoftmax, softmaxInputsShape = softmax_forward(softmaxInputs, softmaxWeights, softmaxBiases)

		# Tính tổng cost-entropy loss và đếm số lượng các dự đoán đúng.
		loss += cost_entropy_loss(postSoftmax[label])
		predictedLabel = max(postSoftmax)
		if predictedLabel == label:
			accuracy += 1

		# Khởi tạo gradient
		gradient = [0] * numNode
		gradient[label] = 1 / postSoftmax[label]

		# Lan truyền ngược.
		gradient = softmax_backprop(gradient, learningRate, softmaxWeights, softmaxBiases, softmaxInputs, softmaxInputsShape, preSoftmax)
		gradient = maxpool_backprop(gradient, maxpoolInputs)
		gradient = conv_backprop(gradient, learningRate, convFilters, convInput)

	#Tính trung bình cost-entropy loss và phần trăm số dự đoán đúng.
	numImage = len(trainImages)
	avgLoss = loss / numImage
	accuracy = accuracy / numImage
	return avgLoss, accuracy
	
@jit
def predict(image, convFilters, maxpoolSize, softmaxWeights, softmaxBiases):
	# Lan truyền xuôi.
	## Chuẩn hoá mảng image về [0,1] để tính toán dễ dàng hơn.
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

	convFiltersH = 3
	convFiltersW = 3
	numConvFilter = 8
	convFilters = gen_conv_filters(convFiltersH, convFiltersW, numConvFilter)

	maxpoolSize = 2

	numNode = 10
	softmaxWeightsLength = ((h - convFiltersH + 1) // maxpoolSize) * ((w - convFiltersW + 1) // maxpoolSize)
	softmaxWeights = gen_softmax_weights(softmaxWeightsLength, numNode)
	softmaxBiases = [0] * numNode

	learningRate = 0.005
	avgLoss, accuracy = train(trainImages, learningRate, convFilters, maxpoolSize, softmaxWeights, softmaxBiases)
	print("Average loss: {avgLoss:.3f} | Accuracy: {accuravy:.2f}".format(avgLoss=avgLoss, accuracy=accuracy*100))

if __name__ == "__main__":
	main()