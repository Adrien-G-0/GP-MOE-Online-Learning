import numpy as np
import scipy.special as sp

def multivariate_t_distribution(x, mu, Sigma, nu):
    """
    Computes the PDF of the multivariate t-distribution.
    """
    D = len(x)
    try:
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            Sigma = Sigma + 1e-6 * np.eye(D)
            sign, logdet = np.linalg.slogdet(Sigma)
        inv_Sigma = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        Sigma = Sigma + 1e-6 * np.eye(D)
        sign, logdet = np.linalg.slogdet(Sigma)
        inv_Sigma = np.linalg.inv(Sigma)
        
    diff = x - mu
    mahalanobis = np.dot(diff, np.dot(inv_Sigma, diff))
    
    log_gamma_num = sp.gammaln((nu + D) / 2.0)
    log_gamma_den = sp.gammaln(nu / 2.0)
    
    log_pdf = (
        log_gamma_num
        - log_gamma_den
        - (D / 2.0) * np.log(nu)
        - (D / 2.0) * np.log(np.pi)
        - 0.5 * logdet
        - ((nu + D) / 2.0) * np.log(1.0 + mahalanobis / nu)
    )
    return np.exp(log_pdf)

class CRP:
    """
    Handles the Chinese Restaurant Process prior for cluster assignments
    using multivariate-T marginal predictive distributions.
    """
    def __init__(self, alpha, mu_0, kappa_0, nu_0, Psi_0):
        self.alpha = alpha
        self.mu_0 = np.array(mu_0)
        self.kappa_0 = kappa_0
        self.nu_0 = nu_0
        self.Psi_0 = np.array(Psi_0)
        self.D = len(self.mu_0)
        
    def _compute_posterior_predictive(self, X_k):
        """
        Computes the parameters of the posterior predictive Student-t distribution
        given the points in cluster k (Equation 14).
        """
        N = len(X_k)
        if N == 0:
            df = self.nu_0 - self.D + 1
            scale = self.Psi_0 * (self.kappa_0 + 1) / (self.kappa_0 * df)
            return self.mu_0, scale, df
            
        x_bar = np.mean(X_k, axis=0)
        
        kappa_N = self.kappa_0 + N
        nu_N = self.nu_0 + N
        mu_N = (self.kappa_0 * self.mu_0 + N * x_bar) / kappa_N
        
        # S: scatter matrix
        S = np.zeros((self.D, self.D))
        for x in X_k:
            diff = (x - x_bar).reshape(-1, 1)
            S += np.dot(diff, diff.T)
            
        diff_mu = (x_bar - self.mu_0).reshape(-1, 1)
        Psi_N = self.Psi_0 + S + (self.kappa_0 * N / kappa_N) * np.dot(diff_mu, diff_mu.T)
        
        # Predictive t-distribution parameters
        df = nu_N - self.D + 1
        scale_matrix = Psi_N * (kappa_N + 1) / (kappa_N * df)
        
        return mu_N, scale_matrix, df

    def compute_assignment_probabilities(self, x_i, clusters_X):
        """
        Computes the probability of assigning x_i to each existing cluster
        and to a new cluster (Equation 13).
        """
        probs = []
        for X_k in clusters_X:
            N_k = len(X_k)
            mu_k, Sigma_k, df_k = self._compute_posterior_predictive(X_k)
            prob_k = N_k * multivariate_t_distribution(x_i, mu_k, Sigma_k, df_k)
            probs.append(prob_k)
            
        # Probability for a new cluster
        mu_new, Sigma_new, df_new = self._compute_posterior_predictive([])
        prob_new = self.alpha * multivariate_t_distribution(x_i, mu_new, Sigma_new, df_new)
        probs.append(prob_new)
        
        return probs