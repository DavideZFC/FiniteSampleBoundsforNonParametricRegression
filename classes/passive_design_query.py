import numpy as np
from functions.optimal_design import *

class PassiveDesign:
    def __init__(self, feature_map, N_degree):
        '''
        Parameters
        ----------
        feature_map : a function taking as input a set of evaluation points and a degree and resulting a vector for each query point
        N_degree : int, the degree of the feature map
        '''
        self.feature_map = feature_map
        self.Ndegree = N_degree
        
    
    def choose_query_points(self, x_all, n):
        '''
        Queries points from X_all according to the optimal design of the feature map self.feature_map.

        Parameters
        ----------
        x_all : np.ndarray, a matrix (k,d) of k points in R^d
        n : int, how many points the agent is allower to query

        Returns
        ----------
        idxs : a vector of n indeces corresponding to the points of x_all that the agent has choosen to query
        '''
        self.x_all = x_all
        self.features = self.feature_map(self.x_all, self.Ndegree)
        idxs = get_design_idxs(self.features, n)
        return idxs
        
