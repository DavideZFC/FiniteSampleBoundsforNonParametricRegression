from classes.curves import Gaussian
from classes.environment import FitterEnv
from functions.fourier_features import fourier_feature_map
from functions.optimal_design import *
from classes.passive_design_query import PassiveDesign
from classes.experiment import Experiment
from classes.kernel import Kernel
from functions.dirichlet import dirichlet_func
from functions.poussin import poussin_func

gaussian_curve = Gaussian(d=2)
env = FitterEnv(gaussian_curve)

n = 10000
N_degree = 6


design = PassiveDesign(fourier_feature_map, N_degree)
expe = Experiment(design=design, env=env)

dir_kernel = Kernel(dirichlet_func, N_degree)
dvp_kernel = Kernel(poussin_func, 2*N_degree)

n_vec = [1000, 2000, 5000, 10000, 20000, 50000]
seeds = 5
expe.make_multiple_experiments(dvp_kernel, n_vec, seeds, 'DVP', noise_sd=0.05)




