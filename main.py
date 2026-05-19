from classes.curves import Gaussian
from classes.environment import FitterEnv
from functions.fourier_features import fourier_feature_map
from functions.optimal_design import *
from classes.passive_design_query import PassiveDesign
from classes.experiment import Experiment
from classes.kernel import Kernel
from functions.dirichlet import dirichlet_func
from functions.poussin import poussin_func

# Build target function and environment to generate samples
gaussian_curve = Gaussian(d=2)
env = FitterEnv(gaussian_curve)

# Choose experiment parameters
# Degree of the feature map/kernels
N_degree = 6
# Sample sizes to test
n_vec = [1000, 2000, 5000, 10000, 20000, 50000]
# Number of random seeds
seeds = 5

# Define design environment specifying which points to query
design = PassiveDesign(fourier_feature_map, N_degree)
# Define experiment environment to set-up the regression experiment
expe = Experiment(design=design, env=env)

# Define kernels for data perturbation
dir_kernel = Kernel(dirichlet_func, N_degree)
dvp_kernel = Kernel(poussin_func, 2*N_degree)

# Make experiments
expe.make_multiple_experiments(dir_kernel, n_vec, seeds, 'Dir', noise_sd=0.05)
expe.make_multiple_experiments(dvp_kernel, n_vec, seeds, 'DVP', noise_sd=0.05)




