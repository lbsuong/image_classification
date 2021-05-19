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
def dot(A, B):
	C  = [x[:] for x in [[0]*len(A)]*len(B[0])]
	for r in range(len(C)):
		for c in range(len(C[0])):
			temp = 0
			for i in range(len(A[0])):
				temp += A[r][i] * B [i][c]
			C[r][c] = temp
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
	# for i in range(0, width):
    # for j in range(0, height):
    #     val = random.choice([0, 1])
    #     print("%d %d\t%d" % (i, j, val))
	listConv=[]
	for num in range(numConvFilter):
		conv = [[0 for x in range(w)] for y in range(h)] 
		listConv.append(conv)
	return listConv
				

	
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
	numConv = filters.shape[0]
	filters_h = filters.shape[1]
	filters_w = filters.shape[2]
	input_h=input.shape[0]
	input_w=input.shape[1]
	output_h = input_h - filters_h + 1
	output_w = input_w - filters_w + 1
	output = np.zeros(shape=(numConv, output_h, output_w))
	for conv in range(numConv):
		for output_r in range(output_h):
			for output_c in range(output_w):
				for filter_r in range(filters_h):
					for filter_c in range(filters_w):
						output[conv, output_r, output_c] += input[filter_r + output_r, filter_c + output_c] * filters[conv, filter_r, filter_c]
	return output
	    
@jit
def maxMat(X):
  '''
  Tìm max trong ma trận X.
  Input:
  	@ "X" là ma trận.
  Output:
  	@ Max của X
  '''
  max = 0
  maxRow = 0
  maxCol = 0
  for row in range(X.shape[0]):
    for col in range(X.shape[1]):
      if (X[row][col] > max):
        max = X[row][col]
        maxRow = row
        maxCol = col
  return max

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
	input_num = input.shape[0]
	input_h = input.shape[1]
	input_w = input.shape[2]
	output_num = input_num
	output_h= input_h/poolSize
	output_w= input_w/poolSize

	output = np.zeros(shape=(output_num, output_h, output_w))
	for num in range(output_num):
		for output_r in range(output_h):
			for output_c in range(output_w):
				output[num][output_r][output_c]= maxMat(input[num, (output_r * poolSize):(poolSize * (output_r + 1)), (output_c * poolSize):(poolSize * (output_c + 1))])
	return output
	

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