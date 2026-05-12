import numpy as np
from classes.curves import Gaussian
from classes.environment import FitterEnv
from functions.three_d_plotter import threedplotter
from functions.fourier_features import fourier_feature_map
from functions.optimal_design import *

gaussian_curve = Gaussian(d=2)
env = FitterEnv(gaussian_curve)

n = 20
x_query = np.random.uniform(-1,1,size=(n,2))
y = env.query_points(x_query)

N_grado = 1

Phi = fourier_feature_map(x_query, N_grado)

print(f"Forma originale di X: {x_query.shape}")
print(f"Forma della Feature Map Phi: {Phi.shape}")

pi = find_optimal_design(Phi)
print(pi)
# threedplotter(x_query, y)

counts = sample_from_pi_deterministic(pi, 100)
print(counts)



