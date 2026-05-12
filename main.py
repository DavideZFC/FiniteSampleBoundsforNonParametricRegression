import numpy as np
from classes.curves import Gaussian
from classes.environment import FitterEnv
from functions.three_d_plotter import threedplotter
from functions.fourier_features import fourier_feature_map
from functions.optimal_design import *
from classes.passive_design_query import PassiveDesign

gaussian_curve = Gaussian(d=2)
env = FitterEnv(gaussian_curve)

n = 20
N_grado = 1


design = PassiveDesign(fourier_feature_map, N_grado)
idxs = design.choose_query_points(n)
print(idxs)


# threedplotter(x_query, y)


