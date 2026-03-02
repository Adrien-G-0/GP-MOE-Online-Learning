import numpy as np
from .kernel import rbf_kernel, compute_covariance

class GPExpert:
    def __init__(self, expert_id, D, prior_mean, prior_cov):
        self.expert_id = expert_id
        self.D = D
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.prior_cov_cholesky = np.linalg.cholesky(prior_cov + 1e-6 * np.eye(len(prior_cov)))
        
        self.X = []
        self.y = []
        
        init_f = self.prior_mean + np.dot(self.prior_cov_cholesky, np.random.randn(len(self.prior_mean)))
        init_params = np.exp(init_f)
        
        self.theta = init_params[:-1]
        self.sigma_sq = init_params[-1]
        
    def add_observation(self, x_i, y_i):
        self.X.append(x_i)
        self.y.append(y_i)
        
    def get_X(self):
        return np.array(self.X)
        
    def get_y(self):
        return np.array(self.y)
        
    @property
    def N(self):
        return len(self.X)

    def marginal_likelihood(self, theta, sigma_sq, indices=None, exclude_last=False):
        """
        Computes the log marginal likelihood with exact Eq 19 stochastic approximation.
        """
        X = self.get_X()
        y = self.get_y()
        N_total = len(X)
        
        if exclude_last and N_total > 0:
            X = X[:-1]
            y = y[:-1]
            N_total -= 1
            
        if N_total == 0:
            return 0.0
            
        if indices is not None:
            valid_indices = [idx for idx in indices if idx < N_total]
            if len(valid_indices) == 0:
                return 0.0
                
            X_batch = X[valid_indices]
            y_batch = y[valid_indices]
            B_actual = len(valid_indices)
            
            # Adjusted noise variance according to Eq 19: (N_k * sigma_k^2) / B
            adjusted_sigma_sq = (N_total * sigma_sq) / B_actual
            K = compute_covariance(X_batch, theta, adjusted_sigma_sq)
            y_eval = y_batch
            N_eval = B_actual
        else:
            K = compute_covariance(X, theta, sigma_sq)
            y_eval = y
            N_eval = N_total
            
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K += 1e-6 * np.eye(N_eval)
            L = np.linalg.cholesky(K)
            
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_eval))
        log_det = 2 * np.sum(np.log(np.diag(L)))
        log_ml = -0.5 * np.dot(y_eval, alpha) - 0.5 * log_det - 0.5 * N_eval * np.log(2 * np.pi)
        
        return log_ml

    def update_hyperparameters(self, B=None):
        """
        Updates hyperparameters using the Elliptical Slice Sampler (ESS) with fixed batch.
        """
        if self.N == 0:
            return 
            
        # FIXER LE BATCH POUR TOUTE LA DUREE DE L'ESS
        batch_indices = None
        if B is not None and self.N > B:
            batch_indices = np.random.choice(self.N, size=B, replace=False)

        f = np.concatenate([np.log(self.theta), [np.log(self.sigma_sq)]])
        f_centered = f - self.prior_mean
        nu = np.dot(self.prior_cov_cholesky, np.random.randn(len(f)))
        
        log_L_current = self.marginal_likelihood(self.theta, self.sigma_sq, indices=batch_indices)
        log_y = log_L_current + np.log(np.random.rand())
        
        u = np.random.uniform(0, 2 * np.pi)
        u_min = u - 2 * np.pi
        u_max = u
        
        while True:
            f_prime_centered = f_centered * np.cos(u) + nu * np.sin(u)
            f_prime = f_prime_centered + self.prior_mean
            
            theta_prime = np.exp(f_prime[:-1])
            sigma_sq_prime = np.exp(f_prime[-1])
            
            log_L_prime = self.marginal_likelihood(theta_prime, sigma_sq_prime, indices=batch_indices)
            
            if log_L_prime > log_y:
                self.theta = theta_prime
                self.sigma_sq = sigma_sq_prime
                break
            else:
                if u < 0:
                    u_min = u
                else:
                    u_max = u
                u = np.random.uniform(u_min, u_max)