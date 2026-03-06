import numpy as np
from scipy.optimize import minimize

class GP_MOE_Bandit:
    """Implémentation de l'optimisation par bandit UCB utilisant GP-MOE."""

    def __init__(self, model, domain_bounds, kappa=2.0):
        """
        Initialise l'optimiseur par bandit.

        Args:
            model: Une instance entraînée de GP_MOE.
            domain_bounds: Liste de tuples [(min, max), ...] pour chaque dimension de l'espace d'entrée.
            kappa: Paramètre de compromis pour UCB (trade-off exploration/exploitation). kappa > 0.
        """
        self.model = model
        self.bounds = domain_bounds
        self.kappa = kappa
        self.num_dimensions = len(domain_bounds)

    def acquisition_function(self, x_test):
        """Calcule UCB : On veut maximiser la récompense, donc on minimise -UCB."""
        x_reshaped = np.atleast_2d(x_test)
        
        # Le modèle renvoie la variance, il faut la convertir en écart-type
        pred_mean, pred_var = self.model.predict(x_reshaped)
        pred_std = np.sqrt(max(0, pred_var))
        
        # Upper Confidence Bound
        ucb_value = pred_mean + self.kappa * pred_std
        return -ucb_value 

    def select_next_point(self, num_restarts=5):
        best_x = None
        best_acq_value = np.inf
        
        for _ in range(num_restarts):
            x_start = np.random.uniform(np.array(self.bounds)[:, 0], np.array(self.bounds)[:, 1], size=self.num_dimensions)
            res = minimize(self.acquisition_function, x_start, bounds=self.bounds, method='L-BFGS-B')
            
            if res.success and res.fun < best_acq_value:
                best_acq_value = res.fun
                best_x = res.x
                
        if best_x is None:
            best_x = np.random.uniform(np.array(self.bounds)[:, 0], np.array(self.bounds)[:, 1], size=self.num_dimensions)
            
        return best_x

# --- Exemple d'utilisation (Code Seul - Non Testé) ---
def run_bandit_optimization_loop(unknown_function, domain, model_initial, num_steps=20, kappa=2.5):
    """
    Simule une boucle d'optimisation par bandit.
    
    Note: 'unknown_function' représente la fonction objectif inconnue que nous essayons d'optimiser.
    """
    model = model_initial # Un GP_MOE avec un sampler SMC
    bandit = GP_MOE_Bandit(model, domain, kappa=kappa)
    
    # Historique des observations
    X_history = []
    y_history = []
    
    for i in range(num_steps):
        print(f"Étape Bandit {i+1}/{num_steps}...")
        
        # 1. Sélectionner le prochain point à interroger
        x_next = bandit.select_next_point()
        
        # 2. Évaluer la fonction inconnue au point x_next
        y_next = unknown_function(x_next)
        
        # 3. Mettre à jour le modèle GP-MOE avec la nouvelle observation
        model.update(x_next, y_next)
        
        # Enregistrement
        X_history.append(x_next)
        y_history.append(y_next)
        
    return X_history, y_history