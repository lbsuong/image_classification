from numba import jit
from tensorflow.keras.datasets import mnist
import numpy as np
import math

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
  if (A.shape[1] != B.shape[0]):
    raise Exception('Số dòng của A không bằng số cột của B')

  output = np.zeros(shape=(A.shape[0], B.shape[1]))
  for row in range(A.shape[0]):
    for col in range(B.shape[1]):
      for i in range(A.shape[1]):
        output[row, col] += A[row, i] * B[i, col]
  return output

@jit
def gen_conv_filters(numConvFilter, h, w):
	'''
	Khởi tạo "numConvFilter" filter với kích thước là h*w được gán các giá trị ngẫu nhiên.

	Input:
		@ "h" là số dòng của một filter.
		@ "w" là số cột của một filter.
		@ "numConvFilter" là số lượng filter cần tạo.

	Output:
		@ Mảng các ma trận filter với kích thước là h*w nhưng các filter được duỗi thẳng thành mảng một chiều.
	'''
	return np.random.rand(numConvFilter, h, w) / 9

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
  return X

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
	outputHeight = input.shape[0] - filters.shape[1] + 1
	outputWidth = input.shape[1] - filters.shape[2] + 1
	output = np.zeros(shape=(filters.shape[0], outputHeight, outputWidth))
	for node in range(filters.shape[0]):
		for outputRow in range(outputHeight):
			for outputCol in range(outputWidth):
				for filterRow in range(filters.shape[1]):
					for filterCol in range(filters.shape[2]):
						output[node, outputRow, outputCol] += input[filterRow + outputRow, filterCol + outputCol] * filters[node, filterRow, filterCol]
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
  outputShape = (input.shape[0], math.ceil(input.shape[1] / poolSize), math.ceil(input.shape[2] / poolSize))
  output = np.zeros(shape=outputShape)
  for node in range(input.shape[0]):
    for row in range(outputShape[1]):
      for col in range(outputShape[2]):
        max, _ = matrix_max(input[node, (row * poolSize):(poolSize * (row + 1)), (col * poolSize):(poolSize * (col + 1))])
        output[node, row, col] = max
  return output

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
  input = input.reshape(1, input.shape[0] * input.shape[1] * input.shape[2])
  preSoftmax = dot(input, weights.transpose()).flatten()
  for i in range(len(preSoftmax)):
    preSoftmax[i] += biases[i]
  postSoftmax = softmax(preSoftmax)
  return preSoftmax, postSoftmax

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
def softmax_backprop(d_L_d_out, learningRate, weights, biases, maxpoolFlattenedOutputs, maxpoolOutputsShape, preSoftmax):
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

    # e^(mỗi phần tử trong preSoftmax)
    e_preSoftmax = np.zeros(len(preSoftmax))
    for j in range(len(preSoftmax)):
      e_preSoftmax[j] = math.exp(preSoftmax[i])

    # Tổng của tất cả phần tử trong e_preSoftmax
    S = 0
    for j in e_preSoftmax:
      S += j

    # Gradient của hàm lỗi so với biến preSoftmax
    d_out_d_preSoftmax = np.zeros(len(e_preSoftmax))
    for k in range(len(e_preSoftmax)):
      d_out_d_preSoftmax[k] = (-e_preSoftmax[i] * e_preSoftmax[k]) / (S ** 2)
    d_out_d_preSoftmax[i] = e_preSoftmax[i] * (S - e_preSoftmax[i]) / (S ** 2)

    # Gradient của preSoftmax so với biến weights/biases/inputs
    d_preSoftmax_d_w = maxpoolFlattenedOutputs
    d_preSoftmax_d_b = 1
    d_preSoftmax_d_inputs = weights

		# Gradient của hàm lỗi so với biến preSoftmax
		# xem lại
    d_L_d_preSoftmax = np.zeros(len(d_out_d_preSoftmax))
    for j in range(len(d_out_d_preSoftmax)):
      d_L_d_preSoftmax[j] = gradient * d_out_d_preSoftmax[j]

		# Gradient của hàm lỗi so với biến weights
    d_L_d_w = dot(d_L_d_preSoftmax.reshape(len(d_L_d_preSoftmax), 1), d_preSoftmax_d_w.reshape(1, len(d_preSoftmax_d_w)))

		# Gradient của hàm lỗi so với biến biases
    d_L_d_b = np.zeros(len(d_L_d_preSoftmax))
    for j in range(len(d_L_d_preSoftmax)):
      d_L_d_b[j] = d_L_d_preSoftmax[j] * d_preSoftmax_d_b

		# Gradient của hàm lỗi so với biến inputs
    d_L_d_inputs = dot(d_L_d_preSoftmax.reshape(1, len(d_L_d_preSoftmax)), d_preSoftmax_d_inputs)

		# Cập nhật weights
    for row in range(weights.shape[0]):
      for col in range(weights.shape[1]):
        weights[row, col] -= learningRate * d_L_d_w[row, col]
        
		# Cập nhật biases
    for j in range(len(d_L_d_b)):
      biases[j] -= learningRate * d_L_d_b[j]

    return d_L_d_inputs.reshape(maxpoolOutputsShape)

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
def max_value_index(X):
  max = X[0]
  index = 0
  for i in range(len(X)):
    if (X[i] > max):
      max = X[i]
      index = i
  return index

@jit
def train(trainImages, trainLabels, learningRate, convFilters, maxpoolSize, softmaxWeights, softmaxBiases):
  loss = 0
  accuracy = 0
  
  for i in range(trainImages.shape[0]):
	  # Lan truyền xuôi.
	  ## Chuẩn hoá mảng image về [0,1] để tính toán dễ dàng hơn.
    normalizedImage = normalize(trainImages[i].astype(np.float32))
    convOutputs = conv_forward(normalizedImage, convFilters)
    maxpoolOutputs = maxpool_forward(convOutputs, maxpoolSize)
    preSoftmax, postSoftmax = softmax_forward(maxpoolOutputs, softmaxWeights, softmaxBiases)

    # Tính tổng cost-entropy loss và đếm số lượng các dự đoán đúng.
    loss += cost_entropy_loss(postSoftmax[trainLabels[i]])
    predictedLabel = max_value_index(postSoftmax)
    if predictedLabel == trainLabels[i]:
    	accuracy += 1

    # Khởi tạo gradient
    gradient = np.zeros(softmaxWeights.shape[0])
    gradient[trainLabels[i]] = -1 / postSoftmax[trainLabels[i]]

    # Lan truyền ngược.
    gradient = softmax_backprop(gradient, learningRate, softmaxWeights, softmaxBiases, maxpoolOutputs.flatten(), maxpoolOutputs.shape, preSoftmax)
    #gredient = maxpool_backprop(gradient, maxpoolInputs)
    #gradient = conv_backprop(gradient, learningRate, convFilters, convInput)

  #Tính trung bình cost-entropy loss và phần trăm số dự đoán đúng.
  numImage = len(trainImages)
  avgLoss = loss / numImage
  accuracy = accuracy / numImage
  return avgLoss, accuracy

def main():
  (trainImages, trainLabels), (testImages, testLabels) = mnist.load_data()
  
  # Lấy 1000 phần tử đầu tiên của tập train và test
  trainImages = trainImages[:1000]
  trainLabels = trainLabels[:1000]

  convFiltersH = 3
  convFiltersW = 3
  numConvFilter = 8
  convFilters = gen_conv_filters(numConvFilter, convFiltersH, convFiltersW)

  maxpoolSize = 2

  numNode = 10
  softmaxWeightsLength = (math.ceil((trainImages.shape[1] - convFiltersH + 1) / maxpoolSize)) * math.ceil(((trainImages.shape[2] - convFiltersW + 1) / maxpoolSize)) * numConvFilter
  softmaxWeights = gen_softmax_weights(numNode, softmaxWeightsLength)
  softmaxBiases = np.zeros(numNode)

  learningRate = 0.005
  avgLoss, accuracy = train(trainImages, trainLabels, learningRate, convFilters, maxpoolSize, softmaxWeights, softmaxBiases)
  print("Average loss: {avgLoss:.3f} | Accuracy: {accuracy:.2f}%".format(avgLoss=avgLoss, accuracy=accuracy*100))

if __name__ == "__main__":
	main()