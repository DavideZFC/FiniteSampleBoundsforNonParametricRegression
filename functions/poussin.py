import numpy as np

def poussin_func(N_degree):
    """
    Generates the De La Vallèe Poussin (DVP) kernel function and its absolute value function.

    This function acts as a factory that returns two specialized, pre-configured 
    functions (`f` and `g`) for a given parameter `N_degree`, representing the 
    order of the kernel.

    Parameters
    ----------
    N_degree : int, the order parameter that controls the degree 
        of the DVP kernel.

    Returns
    -------
    f : function
        A function `AbsPoussin(x)` that computes the absolute value of the 
        Poussin kernel for a given input `x`.
    g : function
        A function `Poussin(x)` that computes the standard un-normalized 
        Poussin kernel for a given input `x`.
    """

    def PoussinWrapper(n,p):
        c1 = (2*n+1-p)/2
        c2 = (p+1)/2
        c3 = 2*(p+1)
        def Poussin(x):
            return np.sin(np.pi*(c1*x))*np.sin(np.pi*(c2*x))/(c3*np.sin(np.pi*(x/2))**2)
        return Poussin

    def PoussinAbsWrapper(n,p):
        c1 = (2*n+1-p)/2
        c2 = (p+1)/2
        c3 = 2*(p+1)
        def AbsPoussin(x):
            return np.abs(np.sin(np.pi*(c1*x))*np.sin(np.pi*(c2*x))/(c3*np.sin(np.pi*(x/2))**2))
        return AbsPoussin

    f = PoussinAbsWrapper(N_degree,N_degree//2)
    g = PoussinWrapper(N_degree,N_degree//2)

    return f,g


    