import numpy as np

def compute_induced_norm(Ainv, v):
    """Calculates the quadratic form v[i]^T * Ainv * v[i] for each row in v."""
    results = np.zeros(v.shape[0])
    for i in range(v.shape[0]):
        results[i] = np.dot(v[i,:].T, np.dot(Ainv, v[i,:]))
    return results

def compute_design_matrix(A, pi):
    """Computes the weighted information matrix sum(pi[i] * a_i * a_i^T)."""
    D = np.zeros((A.shape[1],A.shape[1]))
    for i in range(A.shape[0]):
        # Outer product update of the information matrix
        D += pi[i]*np.dot(A[i:i+1,:].T,A[i:i+1,:])
    return D

def squeeze_distribution(pi, n):
    """Sparsifies the distribution by keeping only the top n weights."""
    # apply noise injection to avoid ties
    pi = pi + np.random.normal(0,scale=1e-4,size=len(pi))

    sorted_vals = sorted(pi, reverse=True)
    nth_largest = sorted_vals[min(n, len(sorted_vals))-1]
    pi[pi<nth_largest] = 0
    pi = pi/np.sum(pi)
    return pi

def onehot(idx, k):
    """Returns a basis vector with 1 at idx and 0 elsewhere."""
    v = np.zeros(k)
    v[idx] = 1
    return v

def eval_pi(pi, A):
    """Evaluates the G-optimality objective: the maximum variance across points."""
    D = compute_design_matrix(A, pi)
    Dinv = np.linalg.inv(D)
    v = compute_induced_norm(Dinv, A)
    return np.max(v)

def find_optimal_design(A, iter=100, thresh=0):
    """
    Finds the G-optimal design distribution pi over rows of A using 
    the Frank-Wolfe algorithm based on the Kiefer-Wolfowitz theorem.
    """
    k = A.shape[0]
    d = A.shape[1]
    pi = np.ones(k)/k

    for it in range(iter):
        D = compute_design_matrix(A, pi)
        Dinv = np.linalg.inv(D)
        v = compute_induced_norm(Dinv, A)

        # Find the point with the highest variance (directional derivative)
        best_index = np.argmax(v)
        current = v[best_index]
        
        # Convergence check: in G-optimality, max variance should approach dimension d
        if current < (thresh + 1)*A.shape[1]:
            break
            
        # Optimal step size (gamma) for updating the distribution
        gamma = (current/d-1)/(current-1)

        # Update distribution toward the best point
        pi = (1-gamma)*pi + gamma*onehot(best_index, k)
        
    # Sparsify result: Caratheodory's theorem suggests d(d+1)/2 points suffice
    pi = squeeze_distribution(pi, 2*A.shape[1])
    return pi

def sample_from_pi_deterministic(pi, n):
    """
    Allocates exactly n integer samples based on the distribution pi.
    """
    # 1. Initial floor allocation
    counts = np.floor(n * pi).astype(int)
    
    # 2. Distribute remaining samples based on largest fractional remainders
    remainder = n - np.sum(counts)
    if remainder > 0:
        # Sort indices by the decimal part of n * pi
        fractions = (n * pi) - counts
        extra_indices = np.argsort(fractions)[-remainder:]
        counts[extra_indices] += 1
        
    return counts.astype(int)

def from_counts_to_idxs(vector):
    """
        The expanded array of indices, sorted in non-decreasing order.
    """
    ans = np.zeros(np.sum(vector))
    curr = 0
    for i,num in enumerate(vector):
        for j in range(num):
            ans[curr] = i
            curr += 1
    return ans

def get_design_idxs(A,n):
    """
    Composition of the functions
    - from_counts_to_idxs
    - sample_from_pi_deterministic
    - find_optimal_design
    """
    return from_counts_to_idxs(sample_from_pi_deterministic(find_optimal_design(A), n)).astype(int)