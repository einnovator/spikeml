import numpy as np

def one_hot(yy, n):
    yy_ = np.zeros((yy.shape[0], n), dtype=int)
    yy_[np.arange(yy.shape[0]), yy] = 1
    return yy_
