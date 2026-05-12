import numpy as np

class FitterEnv:
    """Simulates an environment for sampling points from a curve with optional Gaussian noise."""
    def __init__(self, curve):
        self.curve = curve
        self.problem_dim = curve.dim

    def query_points(self, points, noise_sd=0.0):
        if not (points.shape[1] == self.problem_dim):
            raise ValueError('Ambient dimension different than query points')

        return self.curve(points) + np.random.normal(0, noise_sd, size=points.shape[0])