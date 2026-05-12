import numpy as np
from functions.optimal_design import *

class PassiveDesign:
    def __init__(self, feature_map, Ndegree):
        self.feature_map = feature_map
        self.Ndegree = Ndegree
        
    
    def choose_query_points(self, x_all, n):
        self.x_all = x_all
        self.features = self.feature_map(self.x_all, self.Ndegree)
        idxs = get_design_idxs(self.features, n)
        return idxs
        
