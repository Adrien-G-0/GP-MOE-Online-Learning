import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_stream import DataStreamer
from src.gp_moe import GPMoE
from src.simple_gp import SimpleGP

def load_extended_eurusd_data(period="5y", max_points=500):
    """
    Downloads EUR/USD daily closing prices using yfinance.
    We extend the limit to capture more market dynamics and regime shifts.
    """
    print(f"Downloading EUR/USD data for period: {period}...")
    ticker = "EURUSD=X"
    data = yf.download(ticker, period=period)
    
    # Take the Close price
    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close'][ticker].values
    else:
        closes = data['Close'].values
        
    # Drop NaNs if any
    closes = closes[~np.isnan(closes)]
    
    # Take the last max_points for the online streaming test
    n_points = min(max_points, len(closes))
    closes = closes[-n_points:]
    
    # X is just time steps
    X = np.arange(n_points).reshape(-1, 1)
    y = closes
    
    print(f"Loaded {n_points} daily closing prices.")
    return X, y

def compute_metrics(y_true, y_pred, var_pred):
    mse = (y_true - y_pred)**2
    # log N(y_true | y_pred, var_pred)
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
        if i > 0 and i % 100 == 0:
            print(f"Step {i}/{streamer.N}")
            
        x_i_orig = X[i, 0]
        y_i_orig = y[i]
            
        # 1. PRÉDIRE LE POINT (avant d'incorporer y_i)
        if i > 0:
            mu_std, var_std = model.predict(x_i_std)
            
            # Dé-standardisation
            mu = mu_std * streamer.y_std + streamer.y_mean
            var = var_std * (streamer.y_std ** 2)
            
            mse, nlpd = compute_metrics(y_i_orig, mu, var)
            
            online_X.append(x_i_orig)
            online_mu.append(mu)
            online_var.append(var)
            online_mses.append(mse)
            online_nlpds.append(nlpd)
            
        # 2. METTRE À JOUR LE MODÈLE
        # Note: We don't force full optimize=True for SimpleGP here to keep it running reasonably fast,
        # but the standard GP will still suffer from O(N^3) predictions.
        if isinstance(model, SimpleGP):
            model.update(x_i_std, y_i_std, optimize=False)
            cluster_assignments.append(0) # Simple GP has no distinct clusters
        else:
            model.update(x_i_std, y_i_std)
            # Tracking des clusters pour GP-MOE
            best_particle = max(model.smc.particles, key=lambda p: p.weight)
            cluster_assignments.append(best_particle.z[-1])
            
    print(f"{name} - Mean Online MSE: {np.mean(online_mses):.6f}")
    print(f"{name} - Mean Online NLPD: {np.mean(online_nlpds):.4f}")
    
    return np.array(online_X), np.array(online_mu), np.array(online_var), online_mses, cluster_assignments

def main():
    X, y = load_extended_eurusd_data(period="5y", max_points=500)
    out_dir = os.path.dirname(__file__)
    D = X.shape[1]
    
    prior_mean = np.zeros(D + 1)
    prior_cov = np.eye(D + 1) * 0.5
    
    results = {}
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
    j_val = 50
    print(f"\n--- Initializing GP-MOE with J={j_val} ---")
    gp_moe = GPMoE(D=D, J=j_val, prior_mean=prior_mean, prior_cov=prior_cov, alpha_init=0.5, B=50)
    
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
    
    # --- Export Metrics to CSV ---
    df_metrics = pd.DataFrame(metrics_table)
    csv_path = os.path.join(out_dir, "comparison_metrics.csv")
    df_metrics.to_csv(csv_path, index=False)
    print("\n" + "="*50)
    print(f"Metrics successfully saved to {csv_path}")
    print(df_metrics.to_string(index=False))
    print("="*50 + "\n")
    
    # --- Visualization ---
    plt.figure(figsize=(14, 10))
    
    # Subplot 1: Simple GP
    plt.subplot(2, 1, 1)
    plt.plot(X, y, 'k.', markersize=6, label='True EUR/USD Close', alpha=0.6)
    plt.plot(res_x_sgp, res_mu_sgp, 'b-', lw=2, label='Simple GP Predictive Mean')
    plt.fill_between(res_x_sgp.flatten(), 
                     res_mu_sgp - 1.96 * np.sqrt(res_var_sgp), 
                     res_mu_sgp + 1.96 * np.sqrt(res_var_sgp), 
                     color='blue', alpha=0.2, label='95% Confidence Interval')
                     
    plt.title("Simple GP: Online Prediction vs True Value")
    plt.ylabel("Exchange Rate")
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: GP-MOE
    plt.subplot(2, 1, 2)
    # CORRECTION ICI: X et y passés dans leur intégralité pour correspondre à res_c_moe
    plt.scatter(X, y, c=res_c_moe, cmap='viridis', s=25, label='True EUR/USD Close (Colored by Cluster)', zorder=2)
    plt.plot(res_x_moe, res_mu_moe, 'r-', lw=2, label=f'GP-MOE (J={j_val}) Predictive Mean', zorder=1)
    plt.fill_between(res_x_moe.flatten(), 
                     res_mu_moe - 1.96 * np.sqrt(res_var_moe), 
                     res_mu_moe + 1.96 * np.sqrt(res_var_moe), 
                     color='red', alpha=0.2, label='95% Confidence Interval', zorder=0)
                     
    plt.title(f"GP-MOE (J={j_val}): Online Prediction vs True Value (with Clustering)")
    plt.xlabel("Time Step (Days)")
    plt.ylabel("Exchange Rate")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "models_comparison_plot.png")
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")

if __name__ == "__main__":
    main()