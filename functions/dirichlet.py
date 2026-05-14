import numpy as np

def dirichlet_func(n_pous):

    def DirichletWrapper(n):
        def Dirichlet(x):
            return np.sin(np.pi*(n+1/2)*x)/np.sin(np.pi*x/2)
        return Dirichlet

    def DirichletAbsWrapper(n):
        def AbsDirichlet(x):
            return np.abs(np.sin(np.pi*(n+1/2)*x)/np.sin(np.pi*x/2))
        return AbsDirichlet

    f = DirichletAbsWrapper(n_pous)
    g = DirichletWrapper(n_pous)

    return f,g


    