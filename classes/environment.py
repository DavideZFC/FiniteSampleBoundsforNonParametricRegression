import numpy as np

class FitterEnv:
    """Simulates an environment for sampling points from a curve with optional Gaussian noise."""
    def __init__(self, curve, k=30):
        """
        Initializes the environment and evaluates the curve on a grid.
        """
        self.curve = curve
        self.problem_dim = curve.dim

        # Discretize the multi-dimensional domain [-1, 1]^d
        self.k = k
        lin = np.linspace(-1, 1, k)
        grid_coords = np.meshgrid(*(lin for _ in range(self.problem_dim)), indexing='ij')
        self.x_all = np.stack(grid_coords, axis=-1).reshape(-1, self.problem_dim)
        self.y_all = self.curve(self.x_all)

    def query_points(self, points, noise_sd=0.0):
        """
        Evaluates the curve at specific points with optional Gaussian noise.
        """
        if not (points.shape[1] == self.problem_dim):
            raise ValueError('Ambient dimension different than query points')

        return self.curve(points) + np.random.normal(0, noise_sd, size=points.shape[0])
    
    def compute_derivatives(self, y):
        """
        Computes the first-order forward finite differences.
        """
        df_dx = np.zeros_like(y)
        # Note: This assumes null function at the frontier; boundary element is left at 0
        df_dx[:-1] = (y[1:] - y[:-1])*self.k/2
        return df_dx

