import numpy as np

class Kernel:
    def __init__(self, abs_func, func, disc = 1000):        

        self.x = np.linspace(-1,1,disc)
        self.abs_func = abs_func
        self.func = func
    
    def sample_noise(self, n, dim=2):
        p = self.abs_func(self.x)/self.abs_func(self.x).sum()

        samples = np.random.choice(self.x, p=p, size = (n,dim))
        signs = np.ones(n)
        for i in range(samples.shape[1]):
            signs *= np.sign(self.func(samples[:,i]))
        return samples, signs
