import numpy as np
import copy
from .crp import CRP
from .gp_expert import GPExpert

class SMCParticle:
    """Represents a single particle in the SMC sampler."""
    def __init__(self, particle_id, D, prior_mean, prior_cov, alpha, crp_params):
        self.particle_id = particle_id
        self.D = D
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.alpha = alpha
        
        # CRP prior
        self.crp = CRP(alpha, **crp_params)
        
        # State
        self.z = [] # List of cluster assignments sequentially
        self.experts = {} # Map from cluster_id to GPExpert
        self.weight = 1.0
        self.log_weight = 0.0
        
        self.next_cluster_id = 0
        
    def _get_clusters_X(self):
        """Returns list of X arrays for existing clusters."""
        return [self.experts[k].get_X() for k in range(self.next_cluster_id)]

    def process_observation(self, x_i, y_i, stochastic_B=None, update_hyperparams=True):
        """
        Process a new observation sequentially.
        """
        # 1. Sample indicator z_i
        clusters_X = self._get_clusters_X()
        unnorm_probs = self.crp.compute_assignment_probabilities(x_i, clusters_X)
        
        # Normalize to get probabilities
        prob_sum = sum(unnorm_probs)
        if prob_sum == 0 or np.isnan(prob_sum):
            # Fallback to uniform if probabilities vanish (edge case)
            K_total = len(unnorm_probs)
            probs = np.ones(K_total) / K_total
        else:
            probs = np.array(unnorm_probs) / prob_sum
            
        z_i = np.random.choice(len(probs), p=probs)
        self.z.append(z_i)
        
        # Value of P(x_i | z_i, alpha) approx given by unnormalized prob element
        p_x_i_given_z_i = unnorm_probs[z_i]
        
        # 2. Update cluster with new observation
        if z_i == self.next_cluster_id:
            # Create new cluster
            new_expert = GPExpert(self.next_cluster_id, self.D, self.prior_mean, self.prior_cov)
            self.experts[self.next_cluster_id] = new_expert
            self.next_cluster_id += 1
            
        expert = self.experts[z_i]
        
        # Compute marginal likelihood before adding the point (for weight ratio)
        log_ml_before = expert.marginal_likelihood(expert.theta, expert.sigma_sq, B=stochastic_B)
        
        # Add observation
        expert.add_observation(x_i, y_i)
        
        # Update hyperparameters via Elliptical Slice Sampler
        if update_hyperparams:
            expert.update_hyperparameters(B=stochastic_B)
            
        # Compute marginal likelihood after adding and updating
        log_ml_after = expert.marginal_likelihood(expert.theta, expert.sigma_sq, B=stochastic_B)
        
        # 3. Update Weight
        # Weight update ratio: P(y_k | X_k, theta_k) / P(y'_k | X'_k, theta'_k) * P(x_i | z_i)
        
        log_weight_update = (log_ml_after - log_ml_before) + np.log(max(p_x_i_given_z_i, 1e-300))
        self.log_weight += log_weight_update

class SMCSampler:
    def __init__(self, J, D, prior_mean, prior_cov, alpha_init, crp_params):
        """
        Initializes the SMC Sampler with J particles.
        """
        self.J = J
        self.particles = [SMCParticle(j, D, prior_mean, prior_cov, alpha_init, crp_params) for j in range(J)]
        self.normalize_weights()
        
    def normalize_weights(self):
        # Subtract max log weight for numerical stability
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
        """Multinomial Resampling to avoid particle degeneracy."""
        weights = np.array([p.weight for p in self.particles])
        indices = np.random.choice(self.J, size=self.J, p=weights, replace=True)
        
        new_particles = []
        for i, idx in enumerate(indices):
            # Deep copy the particle to maintain separate divergent paths
            p_copy = copy.deepcopy(self.particles[idx])
            p_copy.particle_id = i
            # Reset weight
            p_copy.weight = 1.0 / self.J
            p_copy.log_weight = 0.0
            new_particles.append(p_copy)
            
        self.particles = new_particles

    def update(self, x_i, y_i, stochastic_B=None, update_hyperparams=True):
        """Process one step for all particles."""
        for p in self.particles:
            p.process_observation(x_i, y_i, stochastic_B, update_hyperparams)
            
        self.normalize_weights()
        
        N_eff = self.compute_N_eff()
        if N_eff < self.J / 2.0:
            self.resample()
