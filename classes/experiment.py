from classes.passive_design_query import PassiveDesign
from classes.environment import FitterEnv
from sklearn.linear_model import LinearRegression

class Experiment:
    def __init__(self, design, env):
        self.design = design
        self.env = env
    
    def make_experiment(self, n, noise_sd=0.0):
        query_idxs = self.design.choose_query_points(self.env.x_all, n)
        
        y_noisy = self.env.query_points(self.design.x_all[query_idxs], noise_sd)
        X_feat = self.design.features[query_idxs]

        model = LinearRegression()
        model.fit(X_feat, y_noisy)

        y_pred_all = model.predict(self.design.features)
        return y_pred_all
    

    def make_experiment_kernel(self, kernel, n, noise_sd=0.0):
        dim=self.env.problem_dim
        query_idxs = self.design.choose_query_points(self.env.x_all, n)
        points = self.design.x_all[query_idxs]
        noise_plus, noise_minus = kernel.sample_noise_plus_minus(n, dim=dim)
        points_plus, points_minus = points + noise_plus, points + noise_minus
        
        y_noisy_plus = self.env.query_points(points_plus, noise_sd)
        y_noisy_minus = self.env.query_points(points_minus, noise_sd)
        X_feat = self.design.features[query_idxs]

        L = kernel.lebeconst**dim
        c_plus = (L+1)/2
        c_minus = (L-1)/2
        y_noisy = c_plus*y_noisy_plus - c_minus*y_noisy_minus

        model = LinearRegression()
        model.fit(X_feat, y_noisy)

        y_pred_all = model.predict(self.design.features)
        return y_pred_all
