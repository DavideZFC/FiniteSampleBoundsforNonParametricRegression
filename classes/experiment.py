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
