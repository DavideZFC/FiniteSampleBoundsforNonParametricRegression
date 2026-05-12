import numpy as np
from classes.curves import Gaussian
from classes.environment import FitterEnv
from functions.threedplotter import threedplotter
from functions.fourierfeatures import fourier_feature_map

gaussian_curve = Gaussian(d=2)
env = FitterEnv(gaussian_curve)

n = 5000
x_query = np.random.uniform(-1,1,size=(n,2))
y = env.query_points(x_query)

N_grado = 2

Phi = fourier_feature_map(x_query, N_grado)

print(f"Forma originale di X: {x_query.shape}")
print(f"Forma della Feature Map Phi: {Phi.shape}")

# threedplotter(x_query, y)



