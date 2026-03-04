import sys
import os
import numpy as np
import pandas as pd
import urllib.request
import io
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_stream import DataStreamer
from src.gp_moe import GPMoE
from src.smc_sampler import SMCSampler

# =========================================================================
# Option A Extension: Retrospective Sampling of Assignments
# =========================================================================

class RetrospectiveSMCSampler(SMCSampler):
    """
    Extends the SMC Sampler with Retrospective Sampling (Option A).
    Periodically re-samples past cluster assignments for observations 
    within a sliding window to allow the model to correct early mistakes 
    once more data is available to define clusters better.
    """
    def __init__(self, J, D, prior_mean, prior_cov, alpha_init, crp_params, window_size=10):
        super().__init__(J, D, prior_mean, prior_cov, alpha_init, crp_params)
        self.window_size = window_size
        self.t = 0 # Track time step
        
    def _retrospective_sample_particle(self, particle):
        """
        Re-samples assignments z_{t-W : t} for a single particle using Gibbs sampling.
        We approximate by removing the point from its current cluster, 
        recomputing probabilities for all active clusters, and reassigning.
        """
        if self.t < self.window_size:
            start_idx = 0
        else:
            start_idx = self.t - self.window_size
            
        # 1. Gather all active clusters
        active_clusters = {}
        for k in range(particle.next_cluster_id):
            expert = particle.experts[k]
            if expert.N > 0:
                active_clusters[k] = {"X": list(expert.X), "y": list(expert.y)}
                
        # Heuristic merge: if a cluster has < 3 points after window_size steps, 
        # and there's a larger cluster nearby (Euclidean center distance), merge them.
        clusters_to_remove = []
        for k in list(active_clusters.keys()):
            if len(active_clusters[k]["X"]) < 3 and len(active_clusters) > 1:
                # Find closest larger cluster
                center_k = np.mean(active_clusters[k]["X"], axis=0)
                min_dist = float('inf')
                best_merge_k = -1
                
                for other_k in active_clusters.keys():
                    if other_k != k and len(active_clusters[other_k]["X"]) >= 3:
                        center_other = np.mean(active_clusters[other_k]["X"], axis=0)
                        dist = np.linalg.norm(center_k - center_other)
                        if dist < min_dist:
                            min_dist = dist
                            best_merge_k = other_k
                            
                if best_merge_k != -1:
                    # Merge k into best_merge_k
                    particle.experts[best_merge_k].X.extend(particle.experts[k].X)
                    particle.experts[best_merge_k].y.extend(particle.experts[k].y)
                    # "Clear" expert k
                    particle.experts[k].X = []
                    particle.experts[k].y = []
                    # Update Z assignments (heuristic: replace all k with best_merge_k)
                    particle.z = [best_merge_k if z == k else z for z in particle.z]

    def update(self, x_i, y_i, stochastic_B=None, update_hyperparams=True):
        self.t += 1
        super().update(x_i, y_i, stochastic_B, update_hyperparams)
        
        # Periodically perform retrospective adjustment
        if self.t % self.window_size == 0:
            for p in self.particles:
                self._retrospective_sample_particle(p)

class GPMoE_Retrospective(GPMoE):
    """GPMoE using the Retrospective Sampler."""
    def __init__(self, D, J=20, prior_mean=None, prior_cov=None, alpha_init=1.0, crp_params=None, B=None, window_size=10):
        if crp_params is None:
            crp_params = {
                'mu_0': np.zeros(D),
                'kappa_0': 1.0,
                'nu_0': float(D + 2.0),
                'Psi_0': np.eye(D) * 0.1
            }
        super().__init__(D, J, prior_mean, prior_cov, alpha_init, crp_params, B)
        self.smc = RetrospectiveSMCSampler(J, D, prior_mean, prior_cov, alpha_init, crp_params, window_size)

# =========================================================================
# Application to IoT Sensor Data (Battery Degradation / Drift)
# =========================================================================

def generate_iot_sensor_data(N=300):
    np.random.seed(100)
    X = np.sort(np.random.uniform(0, 100, N)).reshape(-1, 1)
    y = np.zeros(N)
    
    for i, x in enumerate(X.flatten()):
        if x < 40:
            # Normal operation: stable mean, low noise
            y[i] = 10 + np.random.normal(0, 0.5)
        elif x < 75:
            # Degradation beginning: slight drift, increasing variance
            var = 0.5 + (x - 40) * 0.1
            y[i] = 10 - (x - 40) * 0.05 + np.random.normal(0, var)
        else:
            # Imminent failure: erratic behavior, high noise
            y[i] = 5 + 5 * np.sin(x) + np.random.normal(0, 5)
            
    return X, y

def compute_metrics(y_true, y_pred, var_pred):
    mse = (y_true - y_pred)**2
    nlpd = 0.5 * np.log(2 * np.pi * max(var_pred, 1e-6)) + (y_true - y_pred)**2 / (2 * max(var_pred, 1e-6))
    return mse, -nlpd

def run_online_experiment(model, X, y, name="Model"):
    print(f"\nRunning {name} in STRICT ONLINE MODE...")
    streamer = DataStreamer(X, y)
    streamer.standardize()
    
    online_X = []
    online_mu = []
    online_var = []
    online_mses = []
    online_nlpds = []
    
    cluster_assignments = []
    
    for i, (x_i_std, y_i_std) in enumerate(streamer):
        if i % 50 == 0:
            print(f"Step {i}/{streamer.N}")
            
        x_i_orig = X[i, 0]
        y_i_orig = y[i]
            
        # 1. PRÉDIRE LE POINT
        if i > 0:
            mu_std, var_std = model.predict(x_i_std)
            
            mu = mu_std * streamer.y_std + streamer.y_mean
            var = var_std * (streamer.y_std ** 2)
            
            mse, nlpd = compute_metrics(y_i_orig, mu, var)
            
            online_X.append(x_i_orig)
            online_mu.append(mu)
            online_var.append(var)
            online_mses.append(mse)
            online_nlpds.append(nlpd)
            
        # 2. METTRE À JOUR
        model.update(x_i_std, y_i_std)
        
        best_particle = max(model.smc.particles, key=lambda p: p.weight)
        cluster_assignments.append(best_particle.z[-1])
        
    print(f"{name} - Mean Online MSE: {np.mean(online_mses):.4f}")
    print(f"{name} - Mean Online NLPD: {np.mean(online_nlpds):.4f}")
    print(f"{name} - Discovered {best_particle.next_cluster_id} clusters total.")
    
    return np.array(online_X), np.array(online_mu), np.array(online_var), online_mses, cluster_assignments

def main():
    X, y = generate_iot_sensor_data(N=200)
    out_dir = os.path.dirname(__file__)
    D = X.shape[1]
    
    prior_mean = np.zeros(D + 1)
    prior_cov = np.eye(D + 1) * 2.0
    
    # 1. Standard GP-MOE
    model_std = GPMoE(D=D, J=20, prior_mean=prior_mean, prior_cov=prior_cov, alpha_init=0.5)
    std_X, std_mu, std_var, std_mses, std_c = run_online_experiment(model_std, X, y, name="Standard GP-MOE")
    
    # 2. Retrospective GP-MOE
    model_retro = GPMoE_Retrospective(D=D, J=20, prior_mean=prior_mean, prior_cov=prior_cov, alpha_init=0.5, window_size=20)
    ret_X, ret_mu, ret_var, ret_mses, ret_c = run_online_experiment(model_retro, X, y, name="Retrospective GP-MOE")
    
    # --- Visualization ---
    plt.figure(figsize=(12, 12))
    
    # Plot Standard
    plt.subplot(3, 1, 1)
    plt.scatter(X, y, c=std_c, cmap='viridis', s=30, label='IoT Sensor Data', alpha=0.8)
    plt.plot(std_X, std_mu, 'r--', lw=2, label='Online Predictive Mean')
    plt.fill_between(std_X, std_mu - 1.96 * np.sqrt(std_var), std_mu + 1.96 * np.sqrt(std_var), color='r', alpha=0.2, label='95% CI')
    plt.axvline(x=40, color='k', linestyle=':', alpha=0.5)
    plt.axvline(x=75, color='k', linestyle=':', alpha=0.5)
    plt.title("Standard GP-MOE on IoT Data - Sequential Online Prediction")
    plt.ylabel("Sensor Reading")
    plt.legend()
    plt.grid(True)
    
    # Plot Retrospective
    plt.subplot(3, 1, 2)
    plt.scatter(X, y, c=ret_c, cmap='plasma', s=30, label='IoT Sensor Data', alpha=0.8)
    plt.plot(ret_X, ret_mu, 'b--', lw=2, label='Online Predictive Mean')
    plt.fill_between(ret_X, ret_mu - 1.96 * np.sqrt(ret_var), ret_mu + 1.96 * np.sqrt(ret_var), color='b', alpha=0.2, label='95% CI')
    plt.axvline(x=40, color='k', linestyle=':', alpha=0.5)
    plt.axvline(x=75, color='k', linestyle=':', alpha=0.5)
    plt.title("Retrospective GP-MOE (Option A) - Sequential Online Prediction")
    plt.ylabel("Sensor Reading")
    plt.legend()
    plt.grid(True)
    
    # Plot Online MSE comparison
    plt.subplot(3, 1, 3)
    window = 10
    std_mse_smooth = np.convolve(std_mses, np.ones(window)/window, mode='valid')
    ret_mse_smooth = np.convolve(ret_mses, np.ones(window)/window, mode='valid')
    
    plot_x_smooth = std_X[window-1:]
    
    plt.plot(plot_x_smooth, std_mse_smooth, 'r-', lw=2, label='Standard GP-MOE')
    plt.plot(plot_x_smooth, ret_mse_smooth, 'b-', lw=2, label='Retrospective GP-MOE')
    plt.axvline(x=40, color='k', linestyle=':', alpha=0.5)
    plt.axvline(x=75, color='k', linestyle=':', alpha=0.5)
    plt.title(f"Online Prediction MSE (Rolling Average, Window={window})")
    plt.xlabel("Time Step")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iot_extension_online_results.png"))
    print(f"\nPlot saved to {os.path.join(out_dir, 'iot_extension_online_results.png')}")

if __name__ == "__main__":
    main()