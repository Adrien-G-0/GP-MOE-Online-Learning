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

def load_motorcycle_data():
    url = "https://raw.githubusercontent.com/dfm/gp/master/data/mcycle.csv"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        df = pd.read_csv(io.StringIO(response.read().decode('utf-8')))
        X = df['times'].values.reshape(-1, 1)
        y = df['accel'].values
        print("Successfully loaded Motorcycle Crash dataset.")
    except Exception as e:
        print("Could not download Motorcycle dataset. Generating synthetic equivalent.", e)
        np.random.seed(42)
        X = np.sort(np.random.uniform(0, 60, 133)).reshape(-1, 1)
        y = np.zeros(133)
        for i, x in enumerate(X.flatten()):
            if x < 15:
                y[i] = np.random.normal(0, 5)
            elif x < 40:
                y[i] = -70 * np.sin((x - 15) * 0.15) + np.random.normal(0, 20)
            else:
                y[i] = np.random.normal(0, 10)
    return X, y

def compute_metrics(y_true, y_pred, var_pred):
    mse = (y_true - y_pred)**2
    # log N(y_true | y_pred, var_pred)
    nlpd = 0.5 * np.log(2 * np.pi * var_pred) + (y_true - y_pred)**2 / (2 * var_pred)
    return mse, -nlpd

def run_experiment(X, y, J, name="Model"):
    print(f"\nRunning {name} (J={J})...")
    streamer = DataStreamer(X, y)
    streamer.standardize()
    
    D = X.shape[1]
    
    # Priors for standardized data
    prior_mean = np.zeros(D + 1)
    # Variance in prior
    prior_cov = np.eye(D + 1) * 2.0
    
    model = GPMoE(D=D, J=J, prior_mean=prior_mean, prior_cov=prior_cov, alpha_init=0.5)
    
    mses = []
    nlpds = []
    
    # For plotting
    X_test = np.linspace(min(streamer.X_stdized), max(streamer.X_stdized), 100).reshape(-1, 1)
    X_test_orig = X_test * streamer.X_std + streamer.X_mean
    
    # Track points and their cluster assignments for the best particle
    cluster_assignments = []
    
    for i, (x_i, y_i) in enumerate(streamer):
        if i % 20 == 0:
            print(f"Step {i}/{streamer.N}")
            
        # Predict before update
        if i > 0:
            mu, var = model.predict(x_i)
            mse, nlpd = compute_metrics(y_i, mu, var)
            mses.append(mse)
            nlpds.append(nlpd)
            
        # Update
        model.update(x_i, y_i)
        
        # Record best particle's assignment for this point
        best_particle = max(model.smc.particles, key=lambda p: p.weight)
        cluster_assignments.append(best_particle.z[-1])
        
    print(f"{name} - Mean Online MSE: {np.mean(mses):.4f}")
    print(f"{name} - Mean Online NLPD: {np.mean(nlpds):.4f}")
    print(f"{name} - Discovered {best_particle.next_cluster_id} clusters total.")
    
    # Final predictions
    mu_test_std, var_test_std = [], []
    for x_t in X_test:
        m, v = model.predict(x_t)
        mu_test_std.append(m)
        var_test_std.append(v)
        
    mu_test_std = np.array(mu_test_std)
    var_test_std = np.array(var_test_std)
    
    # Destandardize predictions
    mu_test = mu_test_std * streamer.y_std + streamer.y_mean
    var_test = var_test_std * (streamer.y_std ** 2)
    
    return mses, nlpds, X_test_orig, mu_test, var_test, cluster_assignments

def main():
    X, y = load_motorcycle_data()
    
    # Provide an output dir if writing locally
    out_dir = os.path.dirname(__file__)
    
    # Run Baseline (J=1) -> Product of Experts approach / Single path
    base_mses, base_nlpds, X_t, mu_base, var_base, c_base = run_experiment(X, y, J=1, name="Baseline (J=1)")
    
    # Run GP-MOE (J=20)
    moe_mses, moe_nlpds, _, mu_moe, var_moe, c_moe = run_experiment(X, y, J=20, name="GP-MOE (J=20)")
    
    # --- Visualization ---
    plt.figure(figsize=(12, 10))
    
    # Plot Baseline
    plt.subplot(2, 1, 1)
    plt.scatter(X, y, c=c_base, cmap='tab10', label='Observations')
    plt.plot(X_t, mu_base, 'k-', label='Predictive Mean')
    plt.fill_between(X_t.flatten(), mu_base - 1.96 * np.sqrt(var_base), mu_base + 1.96 * np.sqrt(var_base), color='k', alpha=0.2, label='95% CI')
    plt.title("Baseline POE (SMC J=1, Single Particle Trace)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Acceleration (g)")
    plt.legend()
    plt.grid(True)
    
    # Plot GP-MOE
    plt.subplot(2, 1, 2)
    plt.scatter(X, y, c=c_moe, cmap='tab10', label='Observations')
    plt.plot(X_t, mu_moe, 'r-', label='Predictive Mean')
    plt.fill_between(X_t.flatten(), mu_moe - 1.96 * np.sqrt(var_moe), mu_moe + 1.96 * np.sqrt(var_moe), color='r', alpha=0.2, label='95% CI')
    plt.title("GP-MOE (SMC J=20)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Acceleration (g)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "motorcycle_results.png"))
    print(f"Plot saved to {os.path.join(out_dir, 'motorcycle_results.png')}")
    
if __name__ == "__main__":
    main()
