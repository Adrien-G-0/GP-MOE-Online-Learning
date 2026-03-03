import numpy as np
from .smc_sampler import SMCSampler

class GPMoE:
    """
    High-level API for Sequential Gaussian Processes for Online Learning.
    """
    def __init__(self, D, J=20, prior_mean=None, prior_cov=None, alpha_init=1.0, crp_params=None, B=None):
        self.D = D
        self.J = J
        self.B = B
        
        if prior_mean is None:
            prior_mean = np.zeros(D + 1)
        if prior_cov is None:
            prior_cov = np.eye(D + 1)
            
        if crp_params is None:
            crp_params = {
                'mu_0': np.zeros(D),
                'kappa_0': 1.0,
                'nu_0': float(D + 2.0),
                'Psi_0': np.eye(D) * 0.2  # Réduit l'étalement (0.1 ou 0.2 est idéal)
            }
            
        self.smc = SMCSampler(J, D, prior_mean, prior_cov, alpha_init, crp_params)
        
    def update(self, x_i, y_i, update_hyperparams=True):
        self.smc.update(x_i, y_i, stochastic_B=self.B, update_hyperparams=update_hyperparams)
        
    def predict(self, x_test):
        x_test = np.array(x_test)
        mean_preds = []
        var_preds = []
        
        for p in self.smc.particles:
            clusters_X = p._get_clusters_X()
            unnorm_probs = p.crp.compute_assignment_probabilities(x_test, clusters_X)
            
            prob_sum = sum(unnorm_probs)
            if prob_sum > 0:
                p_k = np.array(unnorm_probs) / prob_sum
            else:
                p_k = np.ones(len(unnorm_probs)) / len(unnorm_probs)
                
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
                
            p_mean_k.append(0.0)
            from .kernel import rbf_kernel
            new_sigma_sq = np.exp(p.prior_mean[-1])
            new_theta = np.exp(p.prior_mean[:-1])
            p_var_k.append(new_sigma_sq + rbf_kernel(x_test, x_test, new_theta)[0, 0])
            
            p_mean = np.sum(p_k * np.array(p_mean_k))
            p_var = np.sum(p_k * (np.array(p_var_k) + np.array(p_mean_k)**2)) - p_mean**2
            
            mean_preds.append(p_mean)
            var_preds.append(max(0.0, p_var))
            
        weights = np.array([p.weight for p in self.smc.particles])
        
        final_mean = np.sum(weights * np.array(mean_preds))
        final_var = np.sum(weights * (np.array(var_preds) + np.array(mean_preds)**2)) - final_mean**2
        
        return final_mean, max(0.0, final_var)