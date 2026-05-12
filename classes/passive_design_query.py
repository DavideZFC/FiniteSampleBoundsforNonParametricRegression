import numpy as np
from functions.optimal_design import *

class PassiveDesign:
    def __init__(self, feature_map, Ndegree, domain_dim=2, k=50):
        self.feature_map = feature_map
        self.Ndegree = Ndegree
        lin = np.linspace(-1, 1, k)
        grid_coords = np.meshgrid(*(lin for _ in range(domain_dim)), indexing='ij')
        self.x_all = np.stack(grid_coords, axis=-1).reshape(-1, domain_dim)
    
    def choose_query_points(self, n):
        features = self.feature_map(self.x_all, self.Ndegree)
        idxs = get_design_idxs(features, n)
        return idxs
        
