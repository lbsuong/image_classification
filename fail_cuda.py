
@cuda.jit
def matrix_max_kernel(X, blkIdx, unfinsishBlk):
    # blkIdx = cuda.device_array(shape=(1,1), dtype=np.int)
    newBlkIdx = cuda.shared.array((1, 1), np.dtype(int))
    # if cuda.threadIdx.x == 0 and (blkIdx[0] < 0 or blkIdx[0] > cuda.blockDim.x):
    #     blkIdx[0] = 0

    if cuda.threadIdx == 0:
        newBlkIdx[0] = blkIdx
        blkIdx = blkIdx - 1
    # cuda.threadfence_system()
    cuda.syncthreads()

    nBefore = cuda.blockDim.x * newBlkIdx[0] * 2

    stride = 1
    while stride < cuda.blockDim.x:
        curr_idx = nBefore + stride * cuda.threadIdx.x
        next_idx = nBefore + stride * cuda.threadIdx.x + stride
        curr_row = curr_idx // X.shape[1]
        curr_col = curr_idx % X.shape[1]
        next_row = next_idx // X.shape[1]
        next_col = next_idx % X.shape[1]

        if X[curr_row, curr_col] < X[next_row, next_col]:
            X[curr_row, curr_col] = X[next_row, next_col]
        cuda.syncthreads()
        stride = stride * 2

    if cuda.threadIdx.x == 0:
        if newBlkIdx[0] < cuda.blockDim.x - 1:
            # chở block phía sau thực hiện xong
            while unfinsishBlk > newBlkIdx[0] + 1:
                pass
            curr_idx = nBefore
            next_idx = nBefore + 2 * cuda.blockDim.x
            curr_row = curr_idx // X.shape[1]
            curr_col = curr_idx % X.shape[1]
            next_row = next_idx // X.shape[1]
            next_col = next_idx % X.shape[1]
            if X[curr_row, curr_col] < X[next_row, next_col]:
                X[curr_row, curr_col] = X[next_row, next_col]
            cuda.threadfence()
        unfinsishBlk = unfinsishBlk - 1
        cuda.threadfence_system()
