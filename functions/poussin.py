import numpy as np

def poussin_func(n_pous):

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

    f = PoussinAbsWrapper(n_pous,n_pous//2)
    g = PoussinWrapper(n_pous,n_pous//2)

    return f,g


    