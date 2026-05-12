import numpy as np

class FitterEnv:
    def __init__(self, curve):
        self.curve = curve
        self.problem_dim = curve.dim

    def query_points(self, points, noise_sd=0.0):
        if not (points.shape[1] == self.problem_dim):
            raise ValueError('ambient diemnsion different than query points')

        return self.curve(points) + np.random.normal(0, noise_sd, size=points.shape[0])
