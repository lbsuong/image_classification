import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from numba import cuda, jit
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
import numpy as np
import math
import time



@cuda.jit
def dot_kernel(A, B, C):
    """
    Nhân ma trận A và B, số cột của A phải bằng số dòng của B

    Input:
        @ "A" là ma trận.
        @ "B" là ma trận.
    Output:

        @ Ma trận của tích hai ma trận A và B.
    """
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if y > C.shape[0] or x > C.shape[1]:
        return
    C[x, y] = 0
    for col in range(A.shape[1]):
        C[x, y] += A[x, col] * B[col, y]


@cuda.jit
def normalize_kernel(X):
    """
    Chuẩn hoá các phần tử trong mảng một chiều X về dạng [0,1] bằng cách chia cho "max".

    Input:
        @ "X" là ma trận.
        @ "max" là giá trị tối đa.

    Output:
        @ Mảng các giá trị đã được normalize.
    """
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row > X.shape[0] or col > X.shape[1]:
        return
    X[row, col] = (X[row, col] / 255) - 0.5


@cuda.jit
def divide_max_kernel(X, _max, X_return):
    """
    Chuẩn hoá các phần tử trong mảng một chiều X về dạng [0,1] bằng cách chia cho "_max".

    Input:
        @ "X" là ma trận.
        @ "max" là giá trị tối đa.

    Output:
        @ Mảng các giá trị đã được normalize.
    """
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row >= X.shape[0] or col >= X.shape[1]:
        return
    X_return[row, col] = X_return[row, col] + (X[row, col] / _max)


# có thể sử dụng SMEM để tối ưu trên version 2


@cuda.jit
def update_weights_kernel(W, gradient_w, learning_rate):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row > W.shape[0] or col > W.shape[1]:
        return
    W[row, col] = W[row, col] - learning_rate * gradient_w[row, col]


@cuda.jit
def update_biases_kernel(B, gradient_b, learning_rate):
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if idx < B.shape[0]:
        B[idx] = B[idx] - learning_rate * gradient_b[idx]


def softmax_backprop_use_kernel(gradient_out, learningRate, weights, biases, maxpoolOutputs):
    """
	Thực hiện lan truyền ngược qua softmax layer.

	Input:
		@ "gradient_out" là gradient của hàm lỗi so với output của hàm "softmax_forward".
		@ "learningRate" là tốc độ học.
		@ "weights" là mảng các trọng số của những node trong softmax layer, các trọng số trong một node là mảng một chiều.
		@ "biases" là mảng các bias của những node trong softmax layer.
		@ "softmaxForwardFlattenedInputs" là mảng các ma trận input của hàm "softmax_forward" đã được duỗi thẳng thành mảng một chiều.
		@ "softmaxForwardInputsShape" là một tuple chứa hình dạng của input của hàm "softmax_forward".
		@ "maxpoolOutputs" là mảng các giá trị trước khi tính softmax trong hàm "softmax_forward".

	Output:
		@ "d_L_d_inputs" là gradient của hàm lỗi so với input của hàm "softmax_forward".
	"""
    maxpoolOutputsLength = maxpoolOutputs.shape[1] * maxpoolOutputs.shape[2] * maxpoolOutputs.shape[3]
    d_L_d_inputs = np.zeros((maxpoolOutputs.shape[0], 1, maxpoolOutputsLength), dtype=np.float32)
    device_gradient_err_inputs = cuda.to_device(d_L_d_inputs)

    gradient_err_weights = np.zeros((gradient_out.shape[1], maxpoolOutputsLength), dtype=np.float32)
    gradient_err_biases = np.zeros(gradient_out.shape[1], dtype=np.float32)

    device_gradient_err_weights = cuda.to_device(gradient_err_weights)
    device_gradient_err_biases = cuda.to_device(gradient_err_biases)

    device_weights = cuda.to_device(weights)
    device_biases = cuda.to_device(biases)
    block_size = (32, 32)
    for i in range(maxpoolOutputs.shape[0]):
        stream = cuda.stream()
        grid_size = (
            math.ceil(maxpoolOutputsLength / block_size[0]), math.ceil(gradient_out.shape[1] / block_size[1]))
        device_gradient_out_ = cuda.to_device(np.array(gradient_out[i].reshape(gradient_out.shape[1], 1)), stream=stream)
        device_maxpoolOutputs = cuda.to_device(np.array(maxpoolOutputs[i].reshape(1, maxpoolOutputsLength)), stream=stream)
        device_gradient_err_weights_temp = cuda.device_array((gradient_out.shape[1], maxpoolOutputsLength), dtype=np.float32)
        dot_kernel[grid_size, block_size](device_gradient_out_,
                                          device_maxpoolOutputs,
                                          device_gradient_err_weights_temp)
        # a = dot(gradient_out[i].reshape(gradient_out.shape[1], 1), maxpoolOutputs[i].reshape(1, maxpoolOutputsLength))
        # cuda.synchronize()
        # b = device_gradient_err_weights_temp.copy_to_host()
        # print(np.mean(np.abs(a-b)))
        # cuda_gradient_err_weights_temp = cuda.to_device(gradient_err_weights_temp)
        grid_size_1 = (math.ceil(device_gradient_err_weights.shape[0] / block_size[0]),
                       math.ceil(device_gradient_err_weights.shape[1] / block_size[1]))
        divide_max_kernel[grid_size_1, block_size](device_gradient_err_weights_temp, maxpoolOutputs.shape[0], device_gradient_err_weights)
        # gradient_err_weights_1 = cuda_gradient_err_weights.copy_to_host()
        for j in range(gradient_out.shape[1]):
            gradient_err_biases[j] = gradient_err_biases[j] + gradient_out[i, j] / maxpoolOutputs.shape[0]

        device_gradient_out_1 = cuda.to_device(gradient_out[i].reshape(1, gradient_out.shape[1]))
        grid_size_2 = (math.ceil(device_weights.shape[1] / block_size[0]), 
        math.ceil(device_gradient_out_1.shape[0] / block_size[1]))
        dot_kernel[grid_size_2, block_size](device_gradient_out_1, device_weights,
                                            device_gradient_err_inputs[i])

    
    # print(device_biases.shape[0])
    grid_size_update_weights = (math.ceil(device_weights.shape[1] / block_size[0]), math.ceil(device_weights.shape[0] / block_size[1]))
    grid_size_update_biases = math.ceil(device_biases.shape[0] / block_size[0])

    update_weights_kernel[grid_size_update_weights, block_size](device_weights, device_gradient_err_weights, learningRate)
    update_biases_kernel[grid_size_update_biases, block_size[0]](device_biases, device_gradient_err_biases, learningRate)
    weights = device_weights.copy_to_host()
    biases = device_biases.copy_to_host()
    gradient_err_inputs = device_gradient_err_inputs.copy_to_host()
    
    return gradient_err_inputs.reshape(maxpoolOutputs.shape)


# def softmax_backprop_kernel_wrapper(d_d_L_d_out,
    #                                 d_weight,
    #                                 d_maxpoolOutput,
    #                                 d_postSoftmax,
    #                                 numImage,
    #                                 d_d_L_d_input,
    #                                 d_d_L_d_w,
    #                                 d_d_L_d_b,
    #                                 blockSize=(32, 32)):
    # # Tính d_d_L_d_preSoftmax cũng là tính d_d_L_d_b vì nó trỏ cùng 1 vùng nhớ
    # d_d_L_d_preSoftmax = d_d_L_d_b
    # gridSize = math.ceil(numImage / blockSize[1])
    # softmax_backprop_kernel[gridSize,
    #                         blockSize[1]](d_d_L_d_out, d_postSoftmax, numImage,
    #                                       d_d_L_d_preSoftmax)

    # # Tính d_d_L_d_w
    # d_d_L_d_preSoftmaxReshape = d_d_L_d_preSoftmax[:numImage].reshape(
    #     numImage, d_d_L_d_preSoftmax.shape[1], 1)
    # d_maxpoolOutputReshape = d_maxpoolOutput[:numImage].reshape(
    #     numImage, 1, d_weight.shape[1])
    # gridSize = (math.ceil(d_maxpoolOutputReshape.shape[2] / blockSize[0]),
    #             math.ceil(d_d_L_d_preSoftmaxReshape.shape[1] / blockSize[1]),
    #             d_d_L_d_preSoftmaxReshape.shape[0])
    # dot_3D_kernel[gridSize, blockSize](d_d_L_d_preSoftmaxReshape,
    #                                    d_maxpoolOutputReshape, d_d_L_d_w)

    # # Tính d_d_L_d_input
    # d_d_L_d_input_temp = cuda.device_array((numImage, 1, d_weight.shape[1]),
    #                                        dtype=float)
    # d_d_L_d_preSoftmaxReshape = d_d_L_d_preSoftmax[:numImage].reshape(
    #     numImage, 1, d_d_L_d_preSoftmax.shape[1])
    # gridSize = (math.ceil(d_d_L_d_input_temp.shape[2] / blockSize[0]),
    #             math.ceil(d_d_L_d_input_temp.shape[1] / blockSize[1]),
    #             d_d_L_d_input_temp.shape[0])
    # dot_3D2D_kernel[gridSize, blockSize](d_d_L_d_preSoftmaxReshape, d_weight,
    #                                      d_d_L_d_input_temp)
    # d_d_L_d_input[0] = d_d_L_d_input_temp[:numImage].reshape(
    #     d_maxpoolOutput[:numImage].shape)

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

  output = np.zeros((A.shape[0], B.shape[1]))
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
	return np.array(np.random.rand(numConvFilter, convFilterSize, convFilterSize) / (convFilterSize * convFilterSize), dtype=np.float32)

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
	output = np.zeros((filters.shape[0], outputHeight, outputWidth))
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
  output = np.zeros(outputShape)
  maxPositions = np.zeros((outputShape[0], outputShape[1], outputShape[2], 2))
  for node in range(input.shape[0]):
    for row in range(outputShape[1]):
      for col in range(outputShape[2]):
        output[node, row, col], maxPositions[node, row, col] = matrix_max(input[node, (row * poolSize):(poolSize * (row + 1)), (col * poolSize):(poolSize * (col + 1))])
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
	return np.array(np.random.rand(numNode, inputLength) / inputLength, dtype=np.float32)

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
  input_reshape = input.reshape(1, input.shape[0] * input.shape[1] * input.shape[2])
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

@jit
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
  maxpoolOutputsLength = maxpoolOutputs.shape[1] * maxpoolOutputs.shape[2] * maxpoolOutputs.shape[3]
  d_L_d_w = np.zeros((d_L_d_preSoftmax.shape[1], maxpoolOutputsLength))
  d_L_d_b = np.zeros(d_L_d_preSoftmax.shape[1])
  d_L_d_inputs = np.zeros((maxpoolOutputs.shape[0], 1, maxpoolOutputsLength))

  for i in range(maxpoolOutputs.shape[0]):
    # Gradient của hàm lỗi so với biến weights
    d_L_d_w_temp = dot(d_L_d_preSoftmax[i].reshape(d_L_d_preSoftmax.shape[1], 1), maxpoolOutputs[i].reshape(1, maxpoolOutputsLength))
    for row in range(d_L_d_w.shape[0]):
      for col in range(d_L_d_w.shape[1]):
        d_L_d_w[row, col] += d_L_d_w_temp[row, col] / maxpoolOutputs.shape[0]

    # Gradient của hàm lỗi so với biến biases
    for j in range(d_L_d_preSoftmax.shape[1]):
      d_L_d_b[j] += d_L_d_preSoftmax[i, j] / maxpoolOutputs.shape[0]

    # Gradient của hàm lỗi so với biến inputs
    d_L_d_inputs[i] = dot(d_L_d_preSoftmax[i].reshape(1, d_L_d_preSoftmax.shape[1]), weights)

	# Cập nhật weights
  for row in range(weights.shape[0]):
    for col in range(weights.shape[1]):
      weights[row, col] -= learningRate * d_L_d_w[row, col]
      
	# Cập nhật biases
  for j in range(len(d_L_d_b)):
    biases[j] -= learningRate * d_L_d_b[j]

  return d_L_d_inputs.reshape(maxpoolOutputs.shape)

@jit
def maxpool_backprop(d_L_d_out, maxPositions, maxpoolSize, convOutputsShape):
  '''
  Thực hiện lan truyền ngược qua maxpool layer. 

  Input:
  	@ "d_L_d_out" là gradient của hàm lỗi so với output của hàm "maxpool_forward".
  	@ "convForwardOutputs" là mảng các input của hàm "maxpool_forward". 

  Output:
  	@ "d_L_d_input" là gradient của hàm lỗi so với input của hàm "maxpool_forward".
  '''
  d_L_d_inputs = np.zeros((d_L_d_out.shape[0], convOutputsShape[0], convOutputsShape[1], convOutputsShape[2]))
  for image in range(d_L_d_out.shape[0]):
    for node in range(d_L_d_out.shape[1]):
      for row in range(d_L_d_out.shape[2]):
        for col in range(d_L_d_out.shape[3]):
          d_L_d_inputs[image, node, maxPositions[image, node, row, col, 0] + row * maxpoolSize, maxPositions[image, node, row, col, 1] + col * maxpoolSize] = d_L_d_out[image, node, row, col]
  return d_L_d_inputs

@jit
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
  for image in range(normalizedImages.shape[0]):
    for node in range(convFilters.shape[0]):
      for filter_row in range(convFilters.shape[1]):
        for filter_col in range(convFilters.shape[2]):
          for d_L_d_out_row in range(d_L_d_out.shape[2]):
            for d_L_d_out_col in range(d_L_d_out.shape[3]):
              d_L_d_filters[node, filter_row, filter_col] += d_L_d_out[image, node, d_L_d_out_row, d_L_d_out_col] * normalizedImages[image, d_L_d_out_row + filter_row, d_L_d_out_col + filter_col]
  for node in range(d_L_d_filters.shape[0]):
    for row in range(d_L_d_filters.shape[1]):
      for col in range(d_L_d_filters.shape[2]):
        d_L_d_filters[node, row, col] /= normalizedImages.shape[0]
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
def train(trainImages, trainLabels, learningRate, batchSize, convFilters, maxpoolSize, softmaxWeights, softmaxBiases):
  loss = 0
  accuracy = 0

  for batch in range(0, trainImages.shape[0], batchSize):
    # Tạo mini-batch
    imageBatch = trainImages[batch : batch + batchSize]
    labelBatch = trainLabels[batch : batch + batchSize]

    maxpoolOutputs = np.zeros((imageBatch.shape[0], convFilters.shape[0], math.ceil((imageBatch.shape[1] - convFilters.shape[1] + 1) / maxpoolSize), math.ceil((imageBatch.shape[2] - convFilters.shape[2] + 1) / maxpoolSize)))
    maxPositions = np.zeros((imageBatch.shape[0], convFilters.shape[0], math.ceil((imageBatch.shape[1] - convFilters.shape[1] + 1) / maxpoolSize), math.ceil((imageBatch.shape[2] - convFilters.shape[2] + 1) / maxpoolSize), 2), dtype=np.int_)
    d_L_d_preSoftmax = np.zeros((imageBatch.shape[0], len(softmaxBiases)))
    convOutputsShape = (0, 0, 0)

    for i in range(imageBatch.shape[0]):
	    # Lan truyền xuôi.
      convOutputs = conv_forward(imageBatch[i], convFilters)
      maxpoolOutputs[i], maxPositions[i] = maxpool_forward(convOutputs, maxpoolSize)
      postSoftmax = softmax_forward(maxpoolOutputs[i], softmaxWeights, softmaxBiases)
      convOutputsShape = convOutputs.shape

      # Tính tổng cost-entropy loss và đếm số lượng các dự đoán đúng.
      loss += cost_entropy_loss(postSoftmax[labelBatch[i]])
      predictedLabel = max_value_index(postSoftmax)
      if predictedLabel == labelBatch[i]:
      	accuracy += 1

      # Tính gradient
      d_L_d_out = np.zeros(softmaxWeights.shape[0])
      d_L_d_out[labelBatch[i]] = -1 / postSoftmax[labelBatch[i]]
      d_L_d_preSoftmax[i] = compute_gradient(d_L_d_out, postSoftmax)

    # Cập nhật trọng số và bias  
    gradients = softmax_backprop_use_kernel(d_L_d_preSoftmax, learningRate, softmaxWeights, softmaxBiases, maxpoolOutputs)
    gradients = maxpool_backprop(gradients, maxPositions, maxpoolSize, convOutputsShape)
    gradients = conv_backprop(gradients, learningRate, convFilters, imageBatch)

  #Tính trung bình cost-entropy loss và phần trăm số dự đoán đúng.
  numImage = len(trainImages)
  avgLoss = loss / numImage
  accuracy = accuracy / numImage
  return avgLoss, accuracy

@jit
def validate(validateImages, validateLabels, convFilters, maxpoolSize, softmaxWeights, softmaxBiases):
  loss = 0
  accuracy = 0
  
  for i in range(validateImages.shape[0]):
    # Lan truyền xuôi.
    convOutputs = conv_forward(validateImages[i], convFilters)
    maxpoolOutputs, _ = maxpool_forward(convOutputs, maxpoolSize)
    postSoftmax = softmax_forward(maxpoolOutputs, softmaxWeights, softmaxBiases)
    
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
    normalizeValidateImages[i] = normalize(validateImages[i].astype(np.float32))
  
  # Khởi tạo các filter và trọng số
  print("Initiating parameters...")
  convFilters = gen_conv_filters(numConvFilter, convFiltersSize)
  softmaxWeightsLength = (math.ceil((trainImages.shape[1] - convFiltersSize + 1) / maxpoolSize)) * math.ceil(((trainImages.shape[2] - convFiltersSize + 1) / maxpoolSize)) * numConvFilter
  softmaxWeights = gen_softmax_weights(numClass, softmaxWeightsLength)
  softmaxBiases = np.zeros(numClass, dtype=np.float32)

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
    trainingLoss, trainingAccuracy = train(normalizeTrainImages[trainingShuffler], trainLabels[trainingShuffler], learningRate, batchSize, convFilters, maxpoolSize, softmaxWeights, softmaxBiases)
    validationLoss, validationAccuracy = validate(normalizeValidateImages[validationShuffler], validateLabels[validationShuffler], convFilters, maxpoolSize, softmaxWeights, softmaxBiases)
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
    batch_size=batchSize, epochs=epoch,
    validation_data=(np.expand_dims(normalizeValidateImages, axis=3), to_categorical(validateLabels)),
    verbose=2
  )
  stop = time.time()
  print("Total runtime:", time.strftime("%H:%M:%S", time.gmtime(stop - start)))


if __name__ == "__main__":
	main()