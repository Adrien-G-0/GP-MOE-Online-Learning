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
        expert.add_observation(x_i, y_i)
        if update_hyperparams:
            expert.update_hyperparameters(B=stochastic_B)
            
        # 3. Fixed batch for Weight Ratio (Eq 17)
        batch_indices = None
        if stochastic_B is not None and expert.N > stochastic_B:
            old_indices = np.random.choice(expert.N - 1, size=stochastic_B - 1, replace=False)
            batch_indices = np.append(old_indices, expert.N - 1)
            
        log_ml_after = expert.marginal_likelihood(expert.theta, expert.sigma_sq, indices=batch_indices, exclude_last=False)
        log_ml_before = expert.marginal_likelihood(expert.theta, expert.sigma_sq, indices=batch_indices, exclude_last=True)
        
        # 4. Update Weight & Alpha
        log_weight_update = (log_ml_after - log_ml_before) + np.log(max(p_x_i_given_z_i_normalized, 1e-300))
        self.log_weight += log_weight_update
        self.update_alpha()

class SMCSampler:
    def __init__(self, J, D, prior_mean, prior_cov, alpha_init, crp_params):
        self.J = J
        self.particles = [SMCParticle(j, D, prior_mean, prior_cov, alpha_init, crp_params) for j in range(J)]
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
            