from classes.curves import Gaussian
from classes.environment import FitterEnv
from functions.three_d_plotter import threedplotter

gaussian_curve = Gaussian(d=2)
env = FitterEnv(gaussian_curve)

threedplotter(env.x_all, env.y_all, name='figures/benckmark.pdf')