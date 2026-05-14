import numpy as np

class Kernel:
    def __init__(self, abs_func, func, disc = 1000):        

        self.x = np.linspace(-1,1,disc)
        self.abs_func = abs_func
        self.func = func
        self.lebeconst = 2*np.mean(self.abs_func(self.x))
    
    def sample_noise(self, n, dim=2):
        p = self.abs_func(self.x)/self.abs_func(self.x).sum()

        samples = np.random.choice(self.x, p=p, size = (n,dim))
        signs = np.ones(n)
        for i in range(samples.shape[1]):
            signs *= np.sign(self.func(samples[:,i]))
        return samples, signs
    
    def sample_noise_plus_minus(self, n, dim=2, c=4):
        large_n = c*n
        samples, signs = self.sample_noise(large_n, dim=2)
        return self.split_samples(samples, signs, n, dim)

    def split_samples(self, samples, signs, n, dim):
        # 1. Separazione dei campioni in base al segno
        plus_pool = samples[signs == 1]
        minus_pool = samples[signs == -1]
        
        # 2. Inizializzazione delle matrici di output
        samples_plus = np.zeros((n, dim))
        samples_minus = np.zeros((n, dim))
        
        # 3. Funzione di supporto per il riempimento con ripetizione (oversampling)
        def fill_matrix(target, pool):
            num_available = len(pool)
            if num_available == 0:
                return target # Gestione caso limite: nessun segno corrispondente
            
            # Calcoliamo quante volte il pool entra interamente in n
            num_repeats = n // num_available
            remainder = n % num_available
            
            # Riempimento tramite concatenazione di ripetizioni e il resto finale
            # Questo equivale a "ripetere le prime righe" fino a saturazione
            full_repeats = np.tile(pool, (num_repeats, 1))
            rest = pool[:remainder]
            
            return np.vstack([full_repeats, rest]) if num_repeats > 0 else rest

        # Applicazione del riempimento
        samples_plus = fill_matrix(samples_plus, plus_pool)
        samples_minus = fill_matrix(samples_minus, minus_pool)
        
        return samples_plus, samples_minus

