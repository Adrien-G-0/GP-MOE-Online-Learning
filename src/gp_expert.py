import numpy as np
from .kernel import rbf_kernel, compute_covariance

class GPExpert:
    """
    Represents an individual Gaussian Process cluster (Expert) in the GP-MOE model.
    """
    def __init__(self, expert_id, D, prior_mean, prior_cov):
        """
        Args:
            expert_id (int): Unique identifier for the cluster.
            D (int): Dimensionality of the input space.
            prior_mean (np.ndarray): Mean vector of the log-normal prior for [theta_1...theta_D, sigma_sq].
            prior_cov (np.ndarray): Covariance matrix of the log-normal prior.
        """
        self.expert_id = expert_id
        self.D = D
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.prior_cov_cholesky = np.linalg.cholesky(prior_cov + 1e-6 * np.eye(len(prior_cov)))
        
        self.X = []
        self.y = []
        
        # Initialize hyperparameters (theta, sigma_sq) drawing from the log-normal prior
        init_f = self.prior_mean + np.dot(self.prior_cov_cholesky, np.random.randn(len(self.prior_mean)))
        init_params = np.exp(init_f)
        
        self.theta = init_params[:-1]
        self.sigma_sq = init_params[-1]
        
    def add_observation(self, x_i, y_i):
        """Adds a new observation to this cluster."""
        self.X.append(x_i)
        self.y.append(y_i)
        
    def get_X(self):
        return np.array(self.X)
        
    def get_y(self):
        return np.array(self.y)
        
    @property
    def N(self):
        """Number of observations in this cluster."""
        return len(self.X)

    def marginal_likelihood(self, theta, sigma_sq, B=None, exclude_last=False):
        """
        Computes the log marginal likelihood of the observations given hyperparameters.
        
        Args:
            theta (np.ndarray): Kernel length scales.
            sigma_sq (float): Noise variance.
            B (int, optional): Minibatch limit for stochastic approximation.
            
        Returns:
            float: Log marginal likelihood.
        """
        X = self.get_X()
        y = self.get_y()
        
        if exclude_last and len(X) > 0:
            X = X[:-1]
            y = y[:-1]
            
        N = len(X)
        if N == 0:
            return 0.0
            
        if B is not None and N > B:
            # Stochastic approximation (Minibatching)
            indices = np.random.choice(N, size=B, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]
            
            # Adjusted noise variance: (N_k * sigma_k^2) / B
            adjusted_sigma_sq = (N * sigma_sq) / B
            K = compute_covariance(X_batch, theta, adjusted_sigma_sq)
            
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-6 * np.eye(B)
                L = np.linalg.cholesky(K)
                
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_batch))
            log_det = 2 * np.sum(np.log(np.diag(L)))
            log_ml = -0.5 * np.dot(y_batch, alpha) - 0.5 * log_det - 0.5 * B * np.log(2 * np.pi)
            
            # Scale up to full dataset size approximation
            return log_ml * (N / B)
        else:
            # Exact marginal likelihood
            K = compute_covariance(X, theta, sigma_sq)
            
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-6 * np.eye(N)
                L = np.linalg.cholesky(K)
                
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            log_det = 2 * np.sum(np.log(np.diag(L)))
            log_ml = -0.5 * np.dot(y, alpha) - 0.5 * log_det - 0.5 * N * np.log(2 * np.pi)
            return log_ml

    def update_hyperparameters(self, B=None):
        """
        Updates the log-normal hyperparameters using the Elliptical Slice Sampler (ESS).
        """
        if self.N == 0:
            return  # Nothing to update if no data
            
        # Form the current state vector f = [log(theta_1), ..., log(theta_D), log(sigma_sq)]
        f = np.concatenate([np.log(self.theta), [np.log(self.sigma_sq)]])
        f_centered = f - self.prior_mean
        
        # 1. Draw nu from the prior N(0, log_prior_cov)
        nu = np.dot(self.prior_cov_cholesky, np.random.randn(len(f)))
        
        # 2. Define the log-likelihood threshold
        log_L_current = self.marginal_likelihood(self.theta, self.sigma_sq, B=B)
        log_y = log_L_current + np.log(np.random.rand())
        
        # 3. Draw initial parameter u from [0, 2*pi]
        u = np.random.uniform(0, 2 * np.pi)
        u_min = u - 2 * np.pi
        u_max = u
        
        # 4. Bracket and slice
        while True:
            f_prime_centered = f_centered * np.cos(u) + nu * np.sin(u)
            f_prime = f_prime_centered + self.prior_mean
            
            # Reconstruct natural parameters
            theta_prime = np.exp(f_prime[:-1])
            sigma_sq_prime = np.exp(f_prime[-1])
            
            log_L_prime = self.marginal_likelihood(theta_prime, sigma_sq_prime, B=B)
            
            if log_L_prime > log_y:
                self.theta = theta_prime
                self.sigma_sq = sigma_sq_prime
                break
            else:
                # Shrink the bracket
                if u < 0:
                    u_min = u
                else:
                    u_max = u
                u = np.random.uniform(u_min, u_max)
