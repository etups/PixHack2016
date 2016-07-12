import numpy as np

def sigmoid(x):
    x = 1. / (1 + np.exp(-x))

    return x

def grad_sigmoid(f):
    f = f * (1-f)
    
    return f