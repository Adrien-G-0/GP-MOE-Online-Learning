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
from src.simple_gp import SimpleGP

def load_motorcycle_data():
    url = "https://raw.githubusercontent.com/dfm/gp/master/data/mcycle.csv"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        df = pd.read_csv(io.StringIO(response.read().decode('utf-8')))
        df = df.sort_values(by='times')
        
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
        if i > 0 and i % 50 == 0:
            print(f"Step {i}/{streamer.N}")
            
        x_i_orig = X[i, 0]
        y_i_orig = y[i]
            
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
            
        if isinstance(model, SimpleGP):
            model.update(x_i_std, y_i_std, optimize=False)
            cluster_assignments.append(0)
        else:
            model.update(x_i_std, y_i_std)
            best_particle = max(model.smc.particles, key=lambda p: p.weight)
            cluster_assignments.append(best_particle.z[-1])
            
    print(f"{name} - Mean Online MSE: {np.mean(online_mses):.4f}")
    
    return np.array(online_X), np.array(online_mu), np.array(online_var), online_mses, cluster_assignments

def main():
    X, y = load_motorcycle_data()
    out_dir = os.path.dirname(__file__)
    D = X.shape[1]
    
    prior_mean = np.zeros(D + 1)
    prior_mean[-1] = -2.0  
    prior_cov = np.eye(D + 1) * 0.5
    
    crp_params = {
        'mu_0': np.zeros(D),
        'kappa_0': 1.0,
        'nu_0': float(D + 2.0),
        'Psi_0': np.eye(D) * 0.2
    }
    
    metrics_table = []
    
    # --- 1. Run Simple GP ---
    print("\n--- Initializing Simple GP ---")
    simple_gp = SimpleGP(D=D)
    res_x_sgp, res_mu_sgp, res_var_sgp, res_mses_sgp, res_c_sgp = run_online_experiment(
        simple_gp, X, y, name="Simple GP"
    )
    
    metrics_table.append({
        "Model": "Simple GP",
        "Mean MSE": np.mean(res_mses_sgp),
        "Final MSE": res_mses_sgp[-1],
        "Mean Variance": np.mean(res_var_sgp),
        "Final Variance": res_var_sgp[-1]
    })
    
    # --- 2. Run GP-MOE ---
    j_val = 60
    print(f"\n--- Initializing GP-MOE with J={j_val} ---")
    gp_moe = GPMoE(D=D, J=j_val, prior_mean=prior_mean, prior_cov=prior_cov, alpha_init=0.5, 
                  crp_params=crp_params, enable_retro=True, retro_freq=10, retro_threshold=4)
    
    res_x_moe, res_mu_moe, res_var_moe, res_mses_moe, res_c_moe = run_online_experiment(
        gp_moe, X, y, name=f"GP-MOE J={j_val}"
    )
    
    metrics_table.append({
        "Model": f"GP-MOE (J={j_val})",
        "Mean MSE": np.mean(res_mses_moe),
        "Final MSE": res_mses_moe[-1],
        "Mean Variance": np.mean(res_var_moe),
        "Final Variance": res_var_moe[-1]
    })
    
    # --- Export Metrics ---
    df_metrics = pd.DataFrame(metrics_table)
    csv_path = os.path.join(out_dir, "motorcycle_comparison_metrics.csv")
    df_metrics.to_csv(csv_path, index=False)
    print("\n" + "="*50)
    print(f"Metrics successfully saved to {csv_path}")
    print(df_metrics.to_string(index=False))
    print("="*50 + "\n")
    
    # --- Visualization ---
    plt.figure(figsize=(14, 10))
    
    # Plot Simple GP
    plt.subplot(2, 1, 1)
    plt.plot(X, y, 'k.', markersize=8, label='True Observations', alpha=0.6)
    plt.plot(res_x_sgp, res_mu_sgp, 'b-', lw=2, label='Simple GP Predictive Mean')
    plt.fill_between(res_x_sgp.flatten(), 
                     res_mu_sgp - 1.96 * np.sqrt(res_var_sgp), 
                     res_mu_sgp + 1.96 * np.sqrt(res_var_sgp), 
                     color='blue', alpha=0.2, label='95% Confidence Interval')
    plt.title("Simple GP: Online Prediction vs True Value")
    plt.ylabel("Acceleration (g)")
    plt.legend()
    plt.grid(True)
    
    # Plot GP-MOE
    plt.subplot(2, 1, 2)
    plt.scatter(X, y, c=res_c_moe, cmap='viridis', s=30, label='True Observations (Colored by Cluster)', zorder=2)
    plt.plot(res_x_moe, res_mu_moe, 'r-', lw=2, label=f'GP-MOE (J={j_val}) Predictive Mean', zorder=1)
    plt.fill_between(res_x_moe.flatten(), 
                     res_mu_moe - 1.96 * np.sqrt(res_var_moe), 
                     res_mu_moe + 1.96 * np.sqrt(res_var_moe), 
                     color='red', alpha=0.2, label='95% Confidence Interval', zorder=0)
    plt.title(f"GP-MOE (J={j_val}): Online Prediction vs True Value (with Clustering)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Acceleration (g)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "motorcycle_models_comparison_plot.png")
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")

if __name__ == "__main__":
    main()