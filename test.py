import time
import math
import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from numba import jit
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def main():
    A = np.arange(12).reshape(3,4)
    print(A)
    print(A.shape[0])


if __name__ == "__main__":
    main()

