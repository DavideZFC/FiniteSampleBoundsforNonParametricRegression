import numpy as np

def dirichlet_func(N_degree):
    """
    Generates the Dirichlet kernel function and its absolute value function.

    This function acts as a factory that returns two specialized, pre-configured 
    functions (`f` and `g`) for a given parameter `N_degree`, representing the 
    order of the kernel.

    Parameters
    ----------
    N_degree : int, the order parameter that controls the degree 
        of the Dirichlet kernel.

    Returns
    -------
    f : function
        A function `AbsDirichlet(x)` that computes the absolute value of the 
        Dirichlet kernel for a given input `x`.
    g : function
        A function `Dirichlet(x)` that computes the standard un-normalized 
        Dirichlet kernel for a given input `x`.
    """

    def DirichletWrapper(n):
        def Dirichlet(x):
            return np.sin(np.pi*(n+1/2)*x)/np.sin(np.pi*x/2)
        return Dirichlet

    def DirichletAbsWrapper(n):
        def AbsDirichlet(x):
            return np.abs(np.sin(np.pi*(n+1/2)*x)/np.sin(np.pi*x/2))
        return AbsDirichlet

    f = DirichletAbsWrapper(N_degree)
    g = DirichletWrapper(N_degree)

    return f,g


    