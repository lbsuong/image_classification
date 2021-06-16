from re import U
import time
import math
from unittest import signals
import numba
from numba.cuda.stubs import blockDim, grid, threadIdx
import numpy as np
from numba import cuda, jit
import os

from numpy.core.defchararray import index
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


@cuda.jit
def max_matrix_kernel(result, values):
    """
    Find the maximum value in values and store in result[0].
    Both result and values are 2d arrays.
    """
    i, j = cuda.grid(2)
    # Atomically store to result[0,1,2] from values[i, j, k]
    cuda.atomic.max(result, (0, 1), values[i, j])


@cuda.jit
def max_index_kernel(max_value, X, index):
    row, col = cuda.grid(2)
    if row >= X.shape[0] or col >= X.shape[1]:
        return
    if X[row, col] == max_value:
        index[0] = row
        index[1] = col


def matrix_max(X, blockSize=(32, 32)):
    result = np.zeros((2, 2))
    index = np.array([0, 0])
    blockspergrid_y = math.ceil(X.shape[1] / blockSize[1])
    blockspergrid_x = math.ceil(X.shape[0] / blockSize[0])
    gridSize = (blockspergrid_x, blockspergrid_y)
    max_matrix_kernel[gridSize, blockSize](result, X)
    max_value = result[0, 1]
    max_index_kernel[gridSize, blockSize](max_value, X, index)
    return max_value, (index[0], index[1])


def main():
    arr = np.random.randint(10000, size=(100, 100))
    maxGPU, maxGPUIndex = matrix_max(arr)
    max_value = np.max(arr)
    print('max in gpu: ', maxGPU)
    print('max index gpu: ', maxGPUIndex)
    index = (0, 0)
    print('max in cpu ', max_value)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i][j] == max_value:
                index = (i, j)
    print('max index in cpu: ', index)


if __name__ == "__main__":
    main()
