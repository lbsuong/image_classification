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
	pass

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
	pass
	
@jit
def conv_forward(input, filters):
	'''
	Thực hiện lan truyền xuôi qua conv layer.

    Input:
    	@ "input" là ma trận các giá trị của hình ảnh đầu vào sau khi được chuẩn hoá.
		@ "filters" là mảng các ma trận filter được tạo bởi hàm "gen_conv_filters".

    Output:
		@ Mảng các output sau khi đi qua conv layer.
    '''
	pass
    
@jit
def maxpool_forward(input, poolSize):
	'''
	Thực hiện lan truyền xuôi qua maxpool layer.
	
	Input:
		@ "input" là mảng các output của hàm "conv_forward".
    	@ "poolSize" là kích thước của maxpool, maxpool là ma trận vuông.
		
    Output:
		@ Mảng các output sau khi đi qua maxpool layer.
	'''
	pass

@jit
def softmax(X):
	'''
	Tính giá trị softmax của mảng một chiều X.

	Input:
		@ "X" là mảng một chiều.

	Output:
		@ Mảng các giá trị softmax được tính từ X.
	'''
	pass

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
	pass
	
@jit
def flatten(X):
	'''
	Duỗi thẳng X thành mảng một chiều.

	Input:
		@ "X" là mảng với bất cứ chiều nào.

	Output:
		@ Mảng một chiều.
	'''
	pass

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
		@ Shape của "input".
	'''
	pass

@jit
def max(X):
	'''
	Tìm max trong mảng một chiều X.

	Input:
		@ "X" là mảng một chiều.

	Output:
		@ Max của X
	'''
	pass

@jit
def cost_entropy_loss(x):
	'''
	Hàm tính đỘ lỗi.
	
	Input:
		@ "x" là giá trị lớn nhất của mảng trả về từ hàm "softmax_forward".

	Output:
		@ Độ lỗi cost-entropy loss.
	'''
	pass

@jit
def normalize(X, max):
	'''
	Chuẩn hoá các phần tử trong mảng một chiều X về dạng [0,1] bằng cách chia cho "max".

	Input:
		@ "X" là mảng một chiều có "n" phần tử.
		@ "max" là giá trị tối đa.

	Output:
		@ Mảng các giá trị đã được normalize.
	'''
	pass

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
	pass

@jit
def softmax_backprop(d_L_d_out, learningRate, weights, biases, softmaxForwardFlattenedInputs, softmaxForwardInputsShape, preSoftmax):
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
	for i, gradient in enumerate(d_L_d_out):
		if gradient == 0:
			continue
		
		# e^preSoftmax
		e_preSoftmax = [exp(x) for x in preSoftmax]

		# Tổng tất cả phần tử trong e^preSoftmax
		S = sum(e_preSoftmax)

		# Gradient của out[i] trên biến preSoftmax
		temp = [-e_preSoftmax[i] * x for x in e_preSoftmax]
		d_out_d_preSoftmax = [x / (S ** 2) for x in temp]
		d_out_d_preSoftmax[i] = e_preSoftmax[i] * (S - e_preSoftmax[i]) / (S ** 2)

		# Gradient của preSoftmax trên biến weights/biases/input
		d_preSoftmax_d_w = softmaxInputs
		d_preSoftmax_d_b = 1
		d_preSoftmax_d_inputs = weights

		# Gradient của loss trên biến preSoftmax
		d_L_d_preSoftmax = [gradient * x for x in d_out_d_preSoftmax]

		# Gradient của loss trên biến weights/biases/input
		d_L_d_w, d_L_d_w_Height, d_L_d_w_Width = dot(d_preSoftmax_d_w, len(d_preSoftmax_d_w) * softmaxInputsH * softmaxInputsW, 1, d_L_d_preSoftmax, 1, len(d_L_d_preSoftmax))
		d_L_d_b = [x * d_preSoftmax_d_b for x in d_L_d_preSoftmax]
		d_L_d_inputs, _, _ = dot(d_preSoftmax_d_inputs, weightsLength, numNode,d_L_d_preSoftmax, numNode, 1)

		# Update weights / biases
		weights = [x - learningRate * y for (x, y) in zip(weights, d_L_d_w)]
		biases = [x - learningRate * y for (x, y) in zip(biases, d_L_d_b)]
		return d_L_d_inputs

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
	pass

@jit
def train(trainImages, trainLabels, learningRate, convFilters, maxpoolSize, softmaxWeights, softmaxBiases):
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
		gredient = maxpool_backprop(gradient, maxpoolInputs)
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