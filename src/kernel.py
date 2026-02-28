import numpy as np

def rbf_kernel(X1, X2, theta):
    """
    Computes the RBF (Radial Basis Function) Kernel matrix between X1 and X2.
    
    Formula: \Sigma_{\theta}(x, x') = \exp(-\frac{1}{2} \sum_{d=1}^D \frac{(x_d - x'_d)^2}{\theta_d^2})
    
    Args:
        X1 (np.ndarray): First set of points, shape (N1, D)
        X2 (np.ndarray): Second set of points, shape (N2, D)
        theta (np.ndarray): Length scales for each dimension, shape (D,)
        
    Returns:
        np.ndarray: Kernel covariance matrix of shape (N1, N2)
    """
    if X1.ndim == 1:
        X1 = X1.reshape(1, -1)
    if X2.ndim == 1:
        X2 = X2.reshape(1, -1)
        
    # Scale inputs by length scales
    X1_scaled = X1 / theta
    X2_scaled = X2 / theta
    
    # Compute squared Euclidean distance efficiently
    sq_dists = np.sum(X1_scaled**2, axis=1).reshape(-1, 1) + \
               np.sum(X2_scaled**2, axis=1) - \
               2 * np.dot(X1_scaled, X2_scaled.T)
    
    # Ensure no small negative values due to numerical issues
    sq_dists = np.maximum(sq_dists, 0)
    
    return np.exp(-0.5 * sq_dists)

def compute_covariance(X, theta, sigma_sq):
    """
    Computes the covariance matrix for a single dataset X with added noise variance
    on the diagonal.
    
    Formula: \Sigma_{\theta_k} + \sigma_k^2 I
    
    Args:
        X (np.ndarray): Dataset, shape (N, D)
        theta (np.ndarray): Kernel length scales, shape (D,)
        sigma_sq (float): Noise variance \sigma_k^2
        
    Returns:
        np.ndarray: Covariance matrix of shape (N, N)
    """
    N = X.shape[0]
    K = rbf_kernel(X, X, theta)
    return K + sigma_sq * np.eye(N)
