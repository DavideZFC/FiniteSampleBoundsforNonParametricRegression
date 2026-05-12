import numpy as np
from classes.curves import Gaussian
from classes.environment import FitterEnv
from functions.threedplotter import threedplotter

gaussian_curve = Gaussian(d=2)
env = FitterEnv(gaussian_curve)

n = 5000
x_query = np.random.uniform(-1,1,size=(n,2))
y = env.query_points(x_query)
print(y)

threedplotter(x_query, y)



