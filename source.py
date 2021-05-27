import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from numba import jit
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
import numpy as np
import math
import time

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
      e_preSoftmax[j] = math.exp(preSoftmax[j])

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
def maxpool_backprop(d_L_d_out, convForwardOutputs, maxpoolSize):
  '''
  Thực hiện lan truyền ngược qua maxpool layer. 

  Input:
  	@ "d_L_d_out" là gradient của hàm lỗi so với output của hàm "maxpool_forward".
  	@ "convForwardOutputs" là mảng các input của hàm "maxpool_forward". 

  Output:
  	@ "d_L_d_input" là gradient của hàm lỗi so với input của hàm "maxpool_forward".
  '''
  d_L_d_inputs = np.zeros(convForwardOutputs.shape)
  for node in range(d_L_d_out.shape[0]):
    for row in range(d_L_d_out.shape[1]):
      for col in range(d_L_d_out.shape[2]):
        _, (convForwardOutputs_maxRow, convForwardOutputs_maxCol) = matrix_max(convForwardOutputs[node, row * maxpoolSize : maxpoolSize * (row + 1), col * maxpoolSize : maxpoolSize * (col + 1)])
        d_L_d_inputs[node, convForwardOutputs_maxRow + row * maxpoolSize, convForwardOutputs_maxCol + col * maxpoolSize] = d_L_d_out[node, row, col]
  return d_L_d_inputs

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
  d_L_d_filters = np.zeros(convFilters.shape)
  for node in range(convFilters.shape[0]):
    for filter_row in range(convFilters.shape[1]):
      for filter_col in range(convFilters.shape[2]):
        for d_L_d_out_row in range(d_L_d_out.shape[1]):
          for d_L_d_out_col in range(d_L_d_out.shape[2]):
            d_L_d_filters[node, filter_row, filter_col] += d_L_d_out[node, d_L_d_out_row, d_L_d_out_col] * normalizedImage[d_L_d_out_row + filter_row, d_L_d_out_col + filter_col]
  for node in range(convFilters.shape[0]):
    for row in range(convFilters.shape[1]):
      for col in range(convFilters.shape[2]):
        convFilters[node, row, col] -= learningRate * d_L_d_filters[node, row, col]
  return None

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
def train(trainImages, trainLabels, learningRate, convFilters1, convFilters2, maxpoolSize, softmaxWeights, softmaxBiases):
  loss = 0
  accuracy = 0
  
  for i in range(trainImages.shape[0]):
	  # Lan truyền xuôi.
    convOutputs = conv_forward(trainImages[i], convFilters1)
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
    gradient = maxpool_backprop(gradient, convOutputs, maxpoolSize)
    gradient = conv_backprop(gradient, learningRate, convFilters1, trainImages[i])

  #Tính trung bình cost-entropy loss và phần trăm số dự đoán đúng.
  numImage = len(trainImages)
  avgLoss = loss / numImage
  accuracy = accuracy / numImage
  return avgLoss, accuracy

@jit
def validate(validateImages, validateLabels, convFilters1, convFilters2, maxpoolSize, softmaxWeights, softmaxBiases):
  loss = 0
  accuracy = 0
  
  for i in range(validateImages.shape[0]):
    # Lan truyền xuôi.
    convOutputs = conv_forward(validateImages[i], convFilters1)
    maxpoolOutputs = maxpool_forward(convOutputs, maxpoolSize)
    _, postSoftmax = softmax_forward(maxpoolOutputs, softmaxWeights, softmaxBiases)
    
    # Tính tổng cost-entropy loss và đếm số lượng các dự đoán đúng.
    loss += cost_entropy_loss(postSoftmax[validateLabels[i]])
    predictedLabel = max_value_index(postSoftmax)
    if predictedLabel == validateLabels[i]:
    	accuracy += 1
  
  #Tính trung bình cost-entropy loss và phần trăm số dự đoán đúng.
  numImage = len(validateImages)
  avgLoss = loss / numImage
  accuracy = accuracy / numImage
  return avgLoss, accuracy

def main():
  # Khởi tạo các tham số
  convFiltersSize = 5
  numConvFilter = 32
  maxpoolSize = 2
  numClass = 10
  learningRate = 0.005
  epoch = 100

  # Load dữ liệu, chia tập train test
  print("Loading data...")
  (trainImages, trainLabels), (validateImages, validateLabels) = mnist.load_data()
  normalizeTrainImages = np.zeros(trainImages.shape)
  normalizeValidateImages = np.zeros(validateImages.shape)
  print("Normalizing...")
  for i in range(len(trainLabels)):
    normalizeTrainImages[i] = normalize(trainImages[i].astype(np.float32))
  for i in range(len(validateLabels)):
    normalizeValidateImages[i] = normalize(validateImages[i].astype(np.float32))
  
  # Khởi tạo các filter và trọng số
  print("Initiating parameters...")
  convFilters = gen_conv_filters(numConvFilter, convFiltersSize)
  softmaxWeightsLength = (math.ceil((trainImages.shape[1] - convFiltersSize + 1) / maxpoolSize)) * math.ceil(((trainImages.shape[2] - convFiltersSize + 1) / maxpoolSize)) * numConvFilter
  softmaxWeights = gen_softmax_weights(numClass, softmaxWeightsLength)
  softmaxBiases = np.zeros(numClass)

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
    trainingLoss, trainingAccuracy = train(normalizeTrainImages[trainingShuffler], trainLabels[trainingShuffler], learningRate, convFilters, convFilters, maxpoolSize, softmaxWeights, softmaxBiases)
    validationLoss, validationAccuracy = validate(normalizeValidateImages[validationShuffler], validateLabels[validationShuffler], convFilters, convFilters, maxpoolSize, softmaxWeights, softmaxBiases)
    epochEnd = time.time()
    print("{second}s - loss: {trainingLoss:.4f} - accuracy: {trainingAccuracy:.4f} - val_loss: {validationLoss:.4f} - val_accuracy: {validationAccuracy:.4f}".format(second=int(epochEnd-epochStart), trainingLoss=trainingLoss, trainingAccuracy=trainingAccuracy, validationLoss=validationLoss, validationAccuracy=validationAccuracy))
  stop = time.time()
  print("Total runtime:", time.strftime("%H:%M:%S", time.gmtime(stop - start)))

  # keras
  print("\n--------------Keras model--------------\n")
  model = Sequential()
  model.add(Conv2D(numConvFilter, kernel_size=convFiltersSize, input_shape=(trainImages.shape[1], trainImages.shape[2], 1), use_bias=False))
  model.add(MaxPooling2D(pool_size=maxpoolSize))
  model.add(Flatten())
  model.add(Dense(numClass, activation='softmax'))
  
  model.compile(SGD(learning_rate=learningRate), loss='categorical_crossentropy', metrics=['accuracy'])

  start = time.time()
  model.fit(
    np.expand_dims(normalizeTrainImages, axis=3),
    to_categorical(trainLabels),
    batch_size=1, epochs=epoch,
    validation_data=(np.expand_dims(normalizeValidateImages, axis=3), to_categorical(validateLabels)),
    verbose=2
  )
  stop = time.time()
  print("Total runtime:", time.strftime("%H:%M:%S", time.gmtime(stop - start)))


if __name__ == "__main__":
	main()