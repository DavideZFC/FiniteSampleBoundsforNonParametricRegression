import numpy as np

def fourier_feature_map(X, N):
    """
    Build multivariate Fourier feature map.
    
    Parameters:
    - X: numpy array (n_samples, 2)
    - N: total degree
    
    Returns:
    - Phi: (n_samples, n_features) feature matrix
    """
    n_samples, d = X.shape
    features = [np.ones(n_samples)] # Inizia con il Bias
    
    for k1 in range(-N, N + 1):
        for k2 in range(-N, N + 1):
            # Vincolo grado totale
            if 0 < abs(k1) + abs(k2) <= N:
                # Condizione per evitare ridondanza (frequenze "coniugate")
                # Prendiamo solo metà dello spazio: 
                # o k1 > 0, oppure (k1 == 0 e k2 > 0)
                if k1 > 0 or (k1 == 0 and k2 > 0):
                    dot_product = k1 * X[:, 0] + k2 * X[:, 1]
                    features.append(np.cos(dot_product))
                    features.append(np.sin(dot_product))
    
    return np.column_stack(features)

