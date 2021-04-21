from keras.datasets import mnist
from numba import jit
from math import exp
from math import log

@jit
def dot(A, Ah, Aw, B, Bh, Bw):
	'''
	Nhân ma trận A và B
	
	Input:
		@ "A" là ma trận với kích thước là Ah*Aw nhưng được biểu diễn dưới dạng mảng một chiều.
		@ "Ah" là số dòng của ma trận A.
		@ "Aw" là số cột của ma trận B.
		@ "B" là ma trận với kích thước là Bh*Bw nhưng được biểu diễn dưới dạng mảng một chiều.
		@ "Bh" là số dòng của ma trận B.
		@ "Bw" là số cột của ma trận B.

	Output:
		@ Ma trận của tích hai ma trận A và B nhưng được duỗi thẳng thành mảng một chiều.
		@ Số dòng của ma trận vừa tìm được.
		@ Số cột của ma trận vừa tìm được.
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
		@ Mảng các ma trận filter với kích thước là h*w nhưng các filter được duỗi thẳng thành mảng một chiều.
	'''
	pass
	
@jit
def conv_forward(input, h, w, filters, filtersH, filtersW, numConvFilter):
	'''
	Thực hiện lan truyền xuôi qua conv layer.

    Input:
    	@ "input" là mảng một chiều của hình ảnh đầu vào, có kích thước là h*w.
		@ "h" là số dòng của "input".
		@ "w" là số cột của "input".
		@ "filters" là mảng các filter được tạo bởi hàm "gen_conv_filters".
		@ "filtersH" là số dòng của một filter.
		@ "filtersW" là số cột của một filter.
		@ "numConvFilter" là số lượng các filter trong "filters".

    Output:
		@ Mảng các output sau khi đi qua conv layer nhưng các output được duỗi thẳng thành mảng một chiều.
    	@ Số dòng của một output vừa tìm được.
		@ Số cột của một output vừa tìm được.
    '''
	pass
    
@jit
def maxpool_forward(input, h, w, numConvFilter, poolSize):
	'''
	Thực hiện lan truyền xuôi qua maxpool layer.
	
	Input:
		@ "input" là mảng các output của hàm "conv_forward".
		@ "h" là số dòng của "input".
		@ "w" là số cột của "input".
		@ "numConvFilter" là số lượng các phần tử trong mảng "input".
    	@ "poolSize" là kích thước của maxpool, maxpool là ma trận vuông nhưng được duỗi thẳng thành mảng một chiều.
		
    Output:
		@ Mảng các output sau khi đi qua maxpool layer nhưng các output được duỗi thẳng thành mảng một chiều.
    	@ Số dòng của một output vừa tìm được.
    	@ Số cột của một output vừa tìm được.
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
def gen_softmax_weights(length, numNode):
	'''
	Khởi tạo "numNode" trọng số với kích thước là "length" được gán các giá trị ngẫu nhiên.

	Input:
		@ "length" là số trọng số có trong một node.
		@ "numNode" số lượng các node.

	Output:
		@ Mảng "numNode" trọng số có kích thước là "length".
	'''
	pass
	
@jit
def softmax_forward(input, h, w, numConvFilter, weights, weightsLength, biases, numNode):
	'''
	Thực hiện lan truyền xuôi qua softmax layer, softmax layer là một fully connected layer.

	Input:
		@ "input" là mảng các output của hàm "maxpool_forward".
		@ "h" là số dòng của "input".
		@ "w" là số cột của "input".
		@ "numConvFilter" là số lượng các phần tử trong mảng "input".
		@ "weights" là mảng các trọng số của những node trong softmax layer, các trọng số trong một node là mảng một chiều.
		@ "weightsLength" là số trọng số có trong một node.
		@ "biases" là mảng các bias của những node trong softmax layer.
		@ "numNode" là số lượng các node có trong softmax layer.
    
    Output:
		@ Mảng các giá trị trước khi tính softmax.
		@ Mảng các giá trị sau khi tính softmax.
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
def softmax_backprop(d_L_d_out, learningRate, weights, weightsLength, biases, numNode, softmaxInputs, softmaxInputsH, softmaxInputsW, preSoftmax):
	'''
	Thực hiện lan truyền ngược qua softmax layer.

	Input:
		@ "d_L_d_out" là loss gradient cho output của lớp này.
		@ "learningRate" là tốc độ học.
		@ "weights" là mảng các trọng số của những node trong softmax layer, các trọng số trong một node là mảng một chiều.
		@ "weightsLength" là số trọng số có trong một node.
		@ "biases" là mảng các bias của những node trong softmax layer.
		@ "numNode" là số lượng các node có trong softmax layer.
		@ "preSoftmax" là mảng các giá trị trước khi tính softmax trong hàm "softmax_forward".

	Output:
		@ "d_L_d_inputs"
	'''
	pass

@jit
def maxpool_backprop(d_L_d_out, maxpoolInputs, maxpoolInputsH, maxpoolInputsW, softmaxInputs, softmaxInputsH, softmaxInputsW):
	'''
	Thực hiện lan truyền ngược qua maxpool layer.

	Input:
		@ "d_L_d_out" là loss gradient cho output của lớp này.
		@ "maxpoolInputs" là mảng các input được truyền vào trong hàm "maxpool_forward".
		@ "maxpoolInputsH" là số dòng trong "maxpoolInputs".
		@ "maxpoolInputsW" là số cột trong "maxpoolInputs".

	Output:
		@ "d_L_d_input"
	'''
	pass

@jit
def conv_backprop(d_L_d_out, learningRate, convFilters, convFiltersH, convFiltersW, numConvFilter, convInput, h, w):
	'''
	Thực hiện lan truyền ngược qua maxpool layer.

	Input:
		@ "d_L_d_out" là loss gradient cho output của lớp này.
		@ "learningRate" là tốc độ học.
		@ "convFilters" là mảng các conv filter trong hàm "conv_forward".
		@ "convFiltersH" là số dòng của một filter.
		@ "convFiltersW" là số cột của một filter.
		@ "numConvFilter" là số lượng các filter.
		@ "convInput" là mảng input trong hàm "conv_forward".
		@ "h" là số dòng của "convInput".
		@ "w" là số cột của "convInput".
	
	Output: None
	'''
	pass

@jit
def train(trainImages, h, w, trainLabels, learningRate, convFilters, convFiltersH, convFiltersW, maxpoolSize, softmaxWeights, softmaxWeightsLength, softmaxBiases, numNode):
	numConvFilter = len(convFilters)
	loss = 0
	accuracy = 0

	for (image, label) in zip(trainImages, trainLabels):
		# Lan truyền xuôi.
		## Chuẩn hoá mảng image về [0,1] để tính toán dễ dàng hơn.
		convInput = normalize(image, 255)
		maxpoolInputs, maxpoolInputsH, maxpoolInputsW = conv_forward(convInput, h, w, convFilters, convFiltersH, convFiltersW, numConvFilter)
		softmaxInputs, softmaxInputsH, softmaxInputsW = maxpool_forward(maxpoolInputs, maxpoolInputsH, maxpoolInputsW, numConvFilter, maxpoolSize)
		preSoftmax, postSoftmax = softmax_forward(softmaxInputs, softmaxInputsH, softmaxInputsW, numConvFilter, softmaxWeights, softmaxWeightsLength, softmaxBiases, numNode)

		# Tính tổng cost-entropy loss và đếm số lượng các dự đoán đúng.
		loss += cost_entropy_loss(postSoftmax[label])
		predictedLabel = max(postSoftmax)
		if predictedLabel == label:
			accuracy += 1

		# Khởi tạo gradient
		gradient = [0] * numNode
		gradient[label] = 1 / postSoftmax[label]

		# Lan truyền ngược.
		gradient = softmax_backprop(gradient, learningRate, softmaxWeights, softmaxBiases, softmaxInputs, softmaxInputsH, softmaxInputsW, preSoftmax)
		gredient = maxpool_backprop(gradient, maxpoolInputs, maxpoolInputsH, maxpoolInputsW, softmaxInputs, softmaxInputsH, softmaxInputsW)
		gradient = conv_backprop(gradient, learningRate, convFilters, convFiltersH, convFiltersW, numConvFilter, convInput, h, w)

	#Tính trung bình cost-entropy loss và phần trăm số dự đoán đúng.
	numImage = len(trainImages)
	avgLoss = loss / numImage
	accuracy = accuracy / numImage
	return avgLoss, accuracy
	
@jit
def predict(image, h, w, convFilters, convFiltersH, convFiltersW, maxpoolSize, softmaxWeights, softmaxWeightsLength, softmaxBiases, numNode):
	# Lan truyền xuôi.
	## Chuẩn hoá mảng image về [0,1] để tính toán dễ dàng hơn.
	numConvFilter = len(convFilters)
	convInput = normalize(image, 255)
	maxpoolInputs, maxpoolInputsH, maxpoolInputsW = conv_forward(convInput, h, w, convFilters, convFiltersH, convFiltersW, numConvFilter)
	softmaxInputs, softmaxInputsH, softmaxInputsW = maxpool_forward(maxpoolInputs, maxpoolInputsH, maxpoolInputsW, numConvFilter, maxpoolSize)
	_, postSoftmax = softmax_forward(softmaxInputs, softmaxInputsH, softmaxInputsW, numConvFilter, softmaxWeights, softmaxWeightsLength, softmaxBiases, numNode)

	# Nhãn sẽ là phần tử có giá trị softmax cao nhất.
	predictedLabel = max(postSoftmax)
	return predictedLabel

def main():
	(trainImages, trainLabels), (testImages, testLabels) = mnist.load_data()
	h = trainImages.shape[1]
	w = trainImages.shape[2]
	# trainImages = trainImages.reshape(trainImages.shape[0], h*w)
	# trainImages = trainImages.tolist()
	# trainLabels = trainLabels.tolist()
	# testImages = testImages.reshape(testImages.shape[0], h*w)
	# testImages = testImages.tolist()
	# testLabels = testLabels.tolist()

	# Lấy 1000 phần tử đầu tiên của tập train và test
	trainImages = trainImages.reshape(trainImages.shape[0], h*w)[0:1000,:]
	trainImages = trainImages.tolist()
	trainLabels = trainLabels.tolist()[0:1000]
	testImages = testImages.reshape(testImages.shape[0], h*w)[0:1000,:]
	testImages = testImages.tolist()
	testLabels = testLabels.tolist()[0:1000]

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
	avgLoss, accuracy = train(trainImages, h, w, learningRate, convFilters, convFiltersH, convFiltersW, numConvFilter, maxpoolSize, softmaxWeights, softmaxWeightsLength, softmaxBiases, numNode)
	print("Average loss: {avgLoss:.3f} | Accuracy: {accuravy:.2f}".format(avgLoss=avgLoss, accuracy=accuracy*100))

if __name__ == "__main__":
	main()