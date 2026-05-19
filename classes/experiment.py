import numpy as np
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
        dim = self.env.problem_dim
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
    
    def compute_errors(self, haty, y):
        rmse = np.sqrt(((haty-y)**2).mean())
        maxerr = np.max(np.abs(haty-y))
        return rmse, maxerr
    
    def make_multiple_experiments(self, kernel, n_vec, seeds, kernel_name, noise_sd=0.0):
        '''
        return results of the experiment as a vector of means ans standard deviations
        '''
        results = {
            "rmse_mean": [], "rmse_std": [],
            "max_mean": [], "max_std": [],
            "rmse_dx_mean": [], "rmse_dx_std": [],
            "max_dx_mean": [], "max_dx_std": []
        }

        y = self.env.y_all
        dy_dx = self.env.compute_derivatives(self.env.y_all)

        for n in n_vec:

            tmp_rmse, tmp_max = [], []
            tmp_rmse_dx, tmp_max_dx = [], []

            for _ in range(seeds):
                haty = self.make_experiment_kernel(kernel, n, noise_sd)
                rmse, maxerr = self.compute_errors(haty, y)

                hatdy_dx = self.env.compute_derivatives(haty)
                rmse_dx, maxerr_dx = self.compute_errors(hatdy_dx, dy_dx)

                tmp_rmse.append(rmse)
                tmp_max.append(maxerr)
                tmp_rmse_dx.append(rmse_dx)
                tmp_max_dx.append(maxerr_dx)

            results["rmse_mean"].append(np.mean(tmp_rmse))
            results["rmse_std"].append(np.std(tmp_rmse))
            
            results["max_mean"].append(np.mean(tmp_max))
            results["max_std"].append(np.std(tmp_max))
            
            results["rmse_dx_mean"].append(np.mean(tmp_rmse_dx))
            results["rmse_dx_std"].append(np.std(tmp_rmse_dx))
            
            results["max_dx_mean"].append(np.mean(tmp_max_dx))
            results["max_dx_std"].append(np.std(tmp_max_dx))

        import json
        # Definisci il nome del file (magari includendo il nome del kernel)
        filename = f"results_{kernel_name}.json"
        results['n'] = n_vec

        # Salvataggio su disco
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)

