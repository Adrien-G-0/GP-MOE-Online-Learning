import numpy as np
from scipy.optimize import minimize
from .kernel import rbf_kernel, compute_covariance

class SimpleGP:
    """
    Un Processus Gaussien standard (Exact GP) pour servir de modèle de référence (baseline).
    Il accumule toutes les données historiques et recalcule la matrice de covariance 
    complète à chaque prédiction, ce qui illustre le problème de complexité O(N^3).
    """
    def __init__(self, D, prior_mean=0.0, init_theta=1.0, init_sigma_sq=0.1):
        self.D = D
        self.prior_mean = prior_mean
        # Initialisation des hyperparamètres
        self.theta = np.ones(D) * init_theta
        self.sigma_sq = init_sigma_sq
        
        # Stockage de l'historique complet (pas de clusters, pas d'oubli)
        self.X = []
        self.y = []
        
    def _marginal_likelihood(self, params):
        """Calcule la log-vraisemblance marginale négative pour l'optimisation."""
        theta = np.exp(params[:-1])
        sigma_sq = np.exp(params[-1])
        
        X_arr = np.array(self.X)
        y_arr = np.array(self.y) - self.prior_mean
        N = len(X_arr)
        
        K = compute_covariance(X_arr, theta, sigma_sq)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K += 1e-6 * np.eye(N)
            L = np.linalg.cholesky(K)
            
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_arr))
        log_det = 2 * np.sum(np.log(np.diag(L)))
        
        # Negative Log Marginal Likelihood
        nll = 0.5 * np.dot(y_arr, alpha) + 0.5 * log_det + 0.5 * N * np.log(2 * np.pi)
        return nll

    def update(self, x_i, y_i, optimize=False):
        """
        Ajoute une nouvelle observation.
        Si optimize=True, recalcule les hyperparamètres (très lourd en ligne).
        """
        self.X.append(x_i)
        self.y.append(y_i)
        
        # L'optimisation complète en ligne est le point faible du GP standard.
        # On l'évite à chaque étape pour que le code puisse tourner, mais vous 
        # pouvez l'activer ponctuellement.
        if optimize and len(self.X) > 5:
            init_params = np.concatenate([np.log(self.theta), [np.log(self.sigma_sq)]])
            res = minimize(self._marginal_likelihood, init_params, method='L-BFGS-B')
            if res.success:
                self.theta = np.exp(res.x[:-1])
                self.sigma_sq = np.exp(res.x[-1])

    def predict(self, x_test):
        """Prédit la moyenne et la variance pour un nouveau point."""
        if len(self.X) == 0:
            return self.prior_mean, self.sigma_sq + 1.0 
            
        X_arr = np.array(self.X)
        y_arr = np.array(self.y) - self.prior_mean
        x_test = np.atleast_2d(x_test)
        
        # 1. Calcul de la covariance sur tout l'historique
        K = compute_covariance(X_arr, self.theta, self.sigma_sq)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K += 1e-6 * np.eye(len(K))
            L = np.linalg.cholesky(K)
            
        # 2. Covariance croisée avec le point de test
        k_star = rbf_kernel(X_arr, x_test, self.theta).flatten()
        
        # 3. Moyenne prédictive
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_arr))
        mu = np.dot(k_star, alpha) + self.prior_mean
        
        # 4. Variance prédictive
        v = np.linalg.solve(L, k_star)
        k_star_star = rbf_kernel(x_test, x_test, self.theta)[0, 0]
        variance = k_star_star + self.sigma_sq - np.dot(v, v)
        
        return mu, max(0.0, variance)