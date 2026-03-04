import numpy as np
import copy
from .crp import CRP
from .gp_expert import GPExpert

class SMCParticle:
    def __init__(self, particle_id, D, prior_mean, prior_cov, alpha, crp_params):
        self.particle_id = particle_id
        self.D = D
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.alpha = alpha
        
        self.crp = CRP(alpha, **crp_params)
        
        self.z = [] 
        self.experts = {} 
        self.weight = 1.0
        self.log_weight = 0.0
        self.next_cluster_id = 0
        
    def _get_clusters_X(self):
        return [self.experts[k].get_X() for k in range(self.next_cluster_id)]

    def update_alpha(self):
        """Met à jour alpha selon l'Équation 15 de l'article."""
        K = self.next_cluster_id
        N_total = sum(e.N for e in self.experts.values())
        if N_total == 0: 
            return
            
        a_0, b_0 = 1.0, 1.0
        rho = np.random.beta(self.alpha + 1, N_total)
        rho_safe = max(rho, 1e-15)
        
        ratio = (a_0 + K - 1) / (N_total * (b_0 - np.log(rho_safe)))
        pi_alpha = ratio / (1 + ratio)
        
        if np.random.rand() < pi_alpha:
            self.alpha = np.random.gamma(a_0 + K, 1.0 / (b_0 - np.log(rho_safe)))
        else:
            self.alpha = np.random.gamma(a_0 + K - 1, 1.0 / (b_0 - np.log(rho_safe)))
            
        self.crp.alpha = self.alpha

    def process_observation(self, x_i, y_i, stochastic_B=None, update_hyperparams=True):
        # 1. CRP Assignment
        clusters_X = self._get_clusters_X()
        unnorm_probs = self.crp.compute_assignment_probabilities(x_i, clusters_X)
        
        prob_sum = sum(unnorm_probs)
        if prob_sum == 0 or np.isnan(prob_sum):
            probs = np.ones(len(unnorm_probs)) / len(unnorm_probs)
        else:
            probs = np.array(unnorm_probs) / prob_sum
            
        z_i = np.random.choice(len(probs), p=probs)
        self.z.append(z_i)
        
        p_x_i_given_z_i_normalized = probs[z_i] 
        
        if z_i == self.next_cluster_id:
            new_expert = GPExpert(self.next_cluster_id, self.D, self.prior_mean, self.prior_cov)
            self.experts[self.next_cluster_id] = new_expert
            self.next_cluster_id += 1
            
        expert = self.experts[z_i]
        
        # 2. Add observation and update hyperparameters
        # 2. Add observation
        expert.add_observation(x_i, y_i)
        
        # 3. DÉFINIR LE BATCH AVANT DE CALCULER LA VRAISEMBLANCE
        batch_indices = None
        if stochastic_B is not None and expert.N > stochastic_B:
            old_indices = np.random.choice(expert.N - 1, size=stochastic_B - 1, replace=False)
            batch_indices = np.append(old_indices, expert.N - 1)
            
        # CALCULER LA VRAISEMBLANCE AVANT LA MISE A JOUR DES PARAMÈTRES
        log_ml_before = expert.marginal_likelihood(expert.theta, expert.sigma_sq, indices=batch_indices, exclude_last=True)
        
        # 4. Update hyperparameters
        if update_hyperparams:
            expert.update_hyperparameters(B=stochastic_B)
            
        # CALCULER LA VRAISEMBLANCE APRÈS LA MISE A JOUR
        log_ml_after = expert.marginal_likelihood(expert.theta, expert.sigma_sq, indices=batch_indices, exclude_last=False)
        
        # 5. Update Weight & Alpha (Correction du poids ici aussi)
        log_weight_update = (log_ml_after - log_ml_before) + np.log(max(prob_sum, 1e-300))
        self.log_weight += log_weight_update
        self.update_alpha()

class SMCSampler:
    def __init__(self, J, D, prior_mean, prior_cov, alpha_init, crp_params, 
                    enable_retro=False, retro_freq=20, retro_threshold=3, retro_B=None):
            self.J = J
            self.particles = [SMCParticle(j, D, prior_mean, prior_cov, alpha_init, crp_params) for j in range(J)]
            
            self.t = 0
            self.enable_retro = enable_retro
            self.retro_freq = retro_freq
            self.retro_threshold = retro_threshold
            self.retro_B = retro_B
            
            self.normalize_weights()
        
    def normalize_weights(self):
        log_weights = np.array([p.log_weight for p in self.particles])
        max_log_weight = np.max(log_weights)
        weights = np.exp(log_weights - max_log_weight)
        sum_weights = np.sum(weights)
        
        if sum_weights > 0:
            weights = weights / sum_weights
        else:
            weights = np.ones(self.J) / self.J
            
        for p, w in zip(self.particles, weights):
            p.weight = w
            
    def compute_N_eff(self):
        weights = np.array([p.weight for p in self.particles])
        return 1.0 / np.sum(weights ** 2)
        
    def resample(self):
        weights = np.array([p.weight for p in self.particles])
        indices = np.random.choice(self.J, size=self.J, p=weights, replace=True)
        
        new_particles = []
        for i, idx in enumerate(indices):
            p_copy = copy.deepcopy(self.particles[idx])
            p_copy.particle_id = i
            p_copy.weight = 1.0 / self.J
            p_copy.log_weight = 0.0
            new_particles.append(p_copy)
            
        self.particles = new_particles

    def update(self, x_i, y_i, stochastic_B=None, update_hyperparams=True):
        for p in self.particles:
            p.process_observation(x_i, y_i, stochastic_B, update_hyperparams)
            
        self.normalize_weights()
        
        N_eff = self.compute_N_eff()
        if N_eff < self.J / 2.0:
            self.resample()

        # Déclenchement conditionnel du rétrospective sampling
        if self.enable_retro and (self.t % self.retro_freq == 0):
            self.retrospective_sample()
            

    def retrospective_sample(self):
        """
        Nettoie les particules en réassignant point par point les données des 
        petits clusters aberrants vers les experts les plus probables.
        """
        if not self.enable_retro:
            return

        for particle in self.particles:
            # 1. Identifier les clusters actifs et légitimes (au-dessus du seuil)
            valid_clusters = [k for k in range(particle.next_cluster_id) 
                              if particle.experts[k].N >= self.retro_threshold]
            
            if len(valid_clusters) == 0:
                continue

            # 2. Identifier les clusters suspects (sous le seuil)
            suspect_clusters = [k for k in range(particle.next_cluster_id) 
                                if 0 < particle.experts[k].N < self.retro_threshold]

            for k_suspect in suspect_clusters:
                expert_suspect = particle.experts[k_suspect]
                
                # Extraire les points un par un pour les partager si besoin
                points_X = list(expert_suspect.X)
                points_y = list(expert_suspect.y)
                
                # Vider l'expert suspect
                expert_suspect.X = []
                expert_suspect.y = []
                
                # Réassigner chaque point individuellement
                for px, py in zip(points_X, points_y):
                    best_k = -1
                    best_ll = -float('inf')
                    
                    # Tester la vraisemblance de ce point sous chaque expert valide
                    for k_valid in valid_clusters:
                        expert_valid = particle.experts[k_valid]
                        
                        # Calcul de l'erreur prédictive (Vraisemblance marginale approchée)
                        from .kernel import rbf_kernel, compute_covariance
                        try:
                            # On utilise le minibatch pour que ça reste rapide
                            N_k = expert_valid.N
                            if self.retro_B is not None and N_k > self.retro_B:
                                indices = np.random.choice(N_k, size=self.retro_B, replace=False)
                                X_train = expert_valid.get_X()[indices]
                                y_train = expert_valid.get_y()[indices]
                                adjusted_sigma_sq = (N_k * expert_valid.sigma_sq) / self.retro_B
                            else:
                                X_train = expert_valid.get_X()
                                y_train = expert_valid.get_y()
                                adjusted_sigma_sq = expert_valid.sigma_sq

                            K = compute_covariance(X_train, expert_valid.theta, adjusted_sigma_sq)
                            L = np.linalg.cholesky(K + 1e-6 * np.eye(len(K)))
                            
                            k_star = rbf_kernel(X_train, px, expert_valid.theta).flatten()
                            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                            
                            # Prédiction GP
                            mu_pred = np.dot(k_star, alpha)
                            v = np.linalg.solve(L, k_star)
                            var_pred = rbf_kernel(px, px, expert_valid.theta)[0, 0] + expert_valid.sigma_sq - np.dot(v, v)
                            
                            # Log-vraisemblance Gaussienne de la prédiction
                            var_pred = max(var_pred, 1e-6)
                            ll = -0.5 * np.log(2 * np.pi * var_pred) - ((py - mu_pred)**2) / (2 * var_pred)
                            
                        except Exception:
                            ll = -float('inf')
                            
                        if ll > best_ll:
                            best_ll = ll
                            best_k = k_valid
                            
                    # 3. Assigner et nettoyer
                    if best_k != -1:
                        expert_receveur = particle.experts[best_k]
                        expert_receveur.add_observation(px, py)
                        # Optionnel : ré-optimiser l'expert après injection de nouveaux points
                        # expert_receveur.update_hyperparameters(B=self.retro_B)
                        
                        # Mise à jour historique heuristique
                        particle.z = [best_k if z == k_suspect else z for z in particle.z]