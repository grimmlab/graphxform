import numpy as np

def softmax(x: np.array):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)