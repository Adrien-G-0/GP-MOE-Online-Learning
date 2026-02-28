import numpy as np
from .smc_sampler import SMCSampler

class GPMoE:
    """
    High-level API for Sequential Gaussian Processes for Online Learning
    of Nonstationary functions.
    """
    def __init__(self, D, J=20, prior_mean=None, prior_cov=None, alpha_init=1.0, crp_params=None, B=None):
        """
        Args:
            D (int): Dimensionality of input space.
            J (int): Number of particles for SMC.
            prior_mean (np.ndarray): Mean of log-normal prior for GP hyperparams (L scales + noise variance).
            prior_cov (np.ndarray): Covariance of log-normal prior for GP hyperparams.
            alpha_init (float): Initial concentration parameter for CRP.
            crp_params (dict): Parameters for the base measure of CRP (mu_0, kappa_0, nu_0, Psi_0).
            B (int): Minibatch size for stochastic approximation.
        """
        self.D = D
        self.J = J
        self.B = B
        
        # Default priors if not provided
        if prior_mean is None:
            # D length scales + 1 noise variance
            prior_mean = np.zeros(D + 1)
        if prior_cov is None:
            prior_cov = np.eye(D + 1)
            
        if crp_params is None:
            crp_params = {
                'mu_0': np.zeros(D),
                'kappa_0': 1.0,
                'nu_0': float(D + 2.0),
                'Psi_0': np.eye(D) * 5.0  
            }
            
        self.smc = SMCSampler(J, D, prior_mean, prior_cov, alpha_init, crp_params)
        
    def update(self, x_i, y_i, update_hyperparams=True):
        """
        Feed a new observation sequentially into the model.
        """
        self.smc.update(x_i, y_i, stochastic_B=self.B, update_hyperparams=update_hyperparams)
        
    def predict(self, x_test):
        """
        Predict the target value for a new test point averaging over particles and clusters.
        
        Args:
            x_test (np.ndarray): Shape (D,)
            
        Returns:
            tuple: (mean_pred, var_pred)
        """
        x_test = np.array(x_test)
        
        mean_preds = []
        var_preds = []
        
        for p in self.smc.particles:
            clusters_X = p._get_clusters_X()
            # Obtenir les probabilités non normalisées
            unnorm_probs = p.crp.compute_assignment_probabilities(x_test, clusters_X)
            
            # CORRECTION (Algorithme 2) : Ne conserver que les clusters existants (K^+)
            unnorm_probs_existing = unnorm_probs[:-1] 
            
            prob_sum = sum(unnorm_probs_existing)
            if prob_sum > 0:
                p_k = np.array(unnorm_probs_existing) / prob_sum
            else:
                p_k = np.ones(len(unnorm_probs_existing)) / max(1, len(unnorm_probs_existing))
                
            p_mean_k = []
            p_var_k = []
            
            for k in range(p.next_cluster_id):
                expert = p.experts[k]
                if expert.N == 0:
                    p_mean_k.append(0.0)
                    p_var_k.append(expert.sigma_sq)
                    continue
                    
                X_k = expert.get_X()
                y_k = expert.get_y()
                
                from .kernel import rbf_kernel, compute_covariance
                
                N_k = expert.N
                if self.B is not None and N_k > self.B:
                    indices = np.random.choice(N_k, size=self.B, replace=False)
                    X_train = X_k[indices]
                    y_train = y_k[indices]
                    adjusted_sigma_sq = (N_k * expert.sigma_sq) / self.B
                else:
                    X_train = X_k
                    y_train = y_k
                    adjusted_sigma_sq = expert.sigma_sq
                    
                K = compute_covariance(X_train, expert.theta, adjusted_sigma_sq)
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    K += 1e-6 * np.eye(len(K))
                    L = np.linalg.cholesky(K)
                    
                k_star = rbf_kernel(X_train, x_test, expert.theta).flatten()
                
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                mu = np.dot(k_star, alpha)
                
                v = np.linalg.solve(L, k_star)
                variance = rbf_kernel(x_test, x_test, expert.theta)[0, 0] + expert.sigma_sq - np.dot(v, v)
                
                p_mean_k.append(mu)
                p_var_k.append(max(0.0, variance))
                
            # Les calculs "For the new cluster" ont été supprimés pour suivre l'Algorithme 2.
            
            # Si aucun cluster existant (au tout début), fallback à 0
            if len(p_mean_k) == 0:
                p_mean_k = [0.0]
                new_sigma_sq = np.exp(p.prior_mean[-1])
                new_theta = np.exp(p.prior_mean[:-1])
                from .kernel import rbf_kernel
                p_var_k = [new_sigma_sq + rbf_kernel(x_test, x_test, new_theta)[0, 0]]
                p_k = np.array([1.0])
            
            p_mean = np.sum(p_k * np.array(p_mean_k))
            p_var = np.sum(p_k * (np.array(p_var_k) + np.array(p_mean_k)**2)) - p_mean**2
            
            mean_preds.append(p_mean)
            var_preds.append(max(0.0, p_var))
            
        weights = [p.weight for p in self.smc.particles]
        return np.average(mean_preds, weights=weights), np.average(var_preds, weights=weights)
