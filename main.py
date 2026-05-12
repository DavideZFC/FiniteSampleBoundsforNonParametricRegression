import numpy as np
from classes.curves import Gaussian
from classes.environment import FitterEnv
from functions.three_d_plotter import threedplotter
from functions.fourier_features import fourier_feature_map
from functions.optimal_design import *
from classes.passive_design_query import PassiveDesign
from classes.experiment import Experiment

gaussian_curve = Gaussian(d=2)
env = FitterEnv(gaussian_curve)

n = 1000
N_grado = 4

threedplotter(env.x_all, env.y_all)

design = PassiveDesign(fourier_feature_map, N_grado)
expe = Experiment(design=design, env=env)
y_pred_all = expe.make_experiment(n)
threedplotter(env.x_all, y_pred_all)

'''
design = PassiveDesign(fourier_feature_map, N_grado)
idxs = design.choose_query_points(n)
print(idxs)
'''


