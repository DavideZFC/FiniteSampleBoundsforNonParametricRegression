from classes.curves import Gaussian
from classes.environment import FitterEnv
from functions.three_d_plotter import threedplotter
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
N_grado = 6


design = PassiveDesign(fourier_feature_map, N_grado)
expe = Experiment(design=design, env=env)
# abs_func, func = poussin_func(n_pous=2*N_grado)
dir_kernel = Kernel(dirichlet_func, N_grado)
dvp_kernel = Kernel(poussin_func, 2*N_grado)

n_vec = [1000, 2000, 5000, 10000, 20000, 50000]
seeds = 5
expe.make_multiple_experiments(dvp_kernel, n_vec, seeds, 'DVP', noise_sd=0.05)

'''
y_pred_all = expe.make_experiment_kernel(dir_kernel, n, noise_sd=0.05)
threedplotter(env.x_all, y_pred_all, 'figures/dir_app.pdf')
y_pred_all = expe.make_experiment_kernel(dvp_kernel, n, noise_sd=0.05)
threedplotter(env.x_all, y_pred_all, 'figures/dvp_app.pdf')


rmse = np.sqrt(((y_pred_all-env.y_all)**2).mean())
maxerr = np.max(np.abs(y_pred_all-env.y_all))
print('Function: L2 error {} Linfty error {}'.format(rmse,maxerr))



df_dx = env.compute_derivatives(env.y_all)
df_dx_pred = env.compute_derivatives(y_pred_all)
threedplotter(env.x_all, df_dx_pred-df_dx)

rmse = np.sqrt(((df_dx_pred-df_dx)**2).mean())
maxerr = np.max(np.abs(df_dx_pred-df_dx))
print('Derivative: L2 error {} Linfty error {}'.format(rmse, maxerr))
'''



