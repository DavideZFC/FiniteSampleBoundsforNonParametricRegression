import numpy as np

class Gaussian:
    def __init__(self, d):
        self.dim = d
    
    def __call__(self, x):
        x_squared = x**2
        return np.exp(-5*x_squared.sum(axis=1))