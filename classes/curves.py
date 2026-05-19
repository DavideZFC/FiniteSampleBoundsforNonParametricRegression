import numpy as np

class Gaussian:
    '''
    d-variate Gaussian function. This class is callable on an np.array
    of dimension (k,d), returning the evaluation of the function on each row of the array
    '''
    def __init__(self, d):
        self.dim = d
    
    def __call__(self, x):
        x_squared = x**2
        return np.exp(-5*x_squared.sum(axis=1))