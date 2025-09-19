import numpy as np

def scaled_sigmoid(x, a):
    """
    Sigmoid function with scaling parameter `a`.
    """
    return a / (1 + np.exp(-x))