import numpy as np

class Kernel:
    """
    A class to represent a kernel function, compute its properties, and sample 
    points tailored for noise generation based on the kernel's profile.

    Attributes
    ----------
    x : ndarray
        Discretized domain grid between -1 and 1.
    abs_func : function
        The absolute value of the kernel function.
    func : function
        The standard kernel function.
    lebeconst : float
        Numerical approximation of the Lebesgue constant of the kernel.
    """
    def __init__(self, kernel_builder, degree, disc = 1000):
        abs_func, func = kernel_builder(degree)

        self.x = np.linspace(-1,1,disc)
        self.abs_func = abs_func
        self.func = func
        self.lebeconst = 2*np.mean(self.abs_func(self.x))
    
    def sample_noise(self, n, dim=2):
        """
        Samples points from the domain grid according to the probability 
        distribution induced by the absolute kernel value.

        Parameters
        ----------
        n : int
            Number of samples to draw.
        dim : int, optional
            Dimensionality of each sample (default is 2).

        Returns
        -------
        samples : ndarray
            Shape (n, dim) array of sampled points from the grid.
        signs : ndarray
            Shape (n,) array containing the product of the kernel signs 
            across all dimensions for each sample.
        """
        p = self.abs_func(self.x)/self.abs_func(self.x).sum()

        samples = np.random.choice(self.x, p=p, size = (n,dim))
        signs = np.ones(n)
        for i in range(samples.shape[1]):
            signs *= np.sign(self.func(samples[:,i]))
        return samples, signs
    
    def sample_noise_plus_minus(self, n, dim=2, c=4):
        """
        Generates balanced datasets of positive and negative sign samples 
        by oversampling and splitting.

        Parameters
        ----------
        n : int
            Target number of samples for each sign pool.
        dim : int, optional
            Dimensionality of the samples (default is 2).
        c : int, optional
            Oversampling factor to ensure enough candidates are generated (default is 4).

        Returns
        -------
        samples_plus : ndarray
            Shape (n, dim) array of samples with positive sign products.
        samples_minus : ndarray
            Shape (n, dim) array of samples with negative sign products.
        """
        large_n = c*n
        samples, signs = self.sample_noise(large_n, dim=2)
        return self.split_samples(samples, signs, n, dim)

    def split_samples(self, samples, signs, n, dim):
        """
        Splits samples into positive and negative pools, ensuring both outputs 
        are filled to exactly `n` samples via tiling/repetition if necessary.
        """
        plus_pool = samples[signs == 1]
        minus_pool = samples[signs == -1]
        
        samples_plus = np.zeros((n, dim))
        samples_minus = np.zeros((n, dim))
        
        def fill_matrix(target, pool):
            num_available = len(pool)
            if num_available == 0:
                return target
            
            num_repeats = n // num_available
            remainder = n % num_available
            
            full_repeats = np.tile(pool, (num_repeats, 1))
            rest = pool[:remainder]
            
            return np.vstack([full_repeats, rest]) if num_repeats > 0 else rest

        samples_plus = fill_matrix(samples_plus, plus_pool)
        samples_minus = fill_matrix(samples_minus, minus_pool)
        
        return samples_plus, samples_minus

