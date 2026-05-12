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
    features = []
    
    # Generiamo le combinazioni di frequenze (k1, k2) tale che |k1| + |k2| <= N
    # Usiamo un range da -N a N per coprire tutte le direzioni
    for k1 in range(-N, N + 1):
        for k2 in range(-N, N + 1):
            # Vincolo sul grado totale (Norma L1 delle frequenze)
            if 0 < abs(k1) + abs(k2) <= N:
                # Prodotto scalare K * X^T -> (n_samples,)
                dot_product = k1 * X[:, 0] + k2 * X[:, 1]
                
                # Aggiungiamo seno e coseno per ogni frequenza
                features.append(np.cos(dot_product))
                features.append(np.sin(dot_product))
            
            elif k1 == 0 and k2 == 0:
                # Aggiungiamo la componente bias (costante 1) una sola volta
                features.append(np.ones(n_samples))

    # Trasponiamo per avere la forma (n_samples, n_features)
    return np.column_stack(features)

