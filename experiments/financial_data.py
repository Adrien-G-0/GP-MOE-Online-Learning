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
from experiments.iot_extension import GPMoE_Retrospective

def load_eurusd_data(period="1y"):
    """
    Downloads EUR/USD daily closing prices using yfinance.
    We limit to the last 200 days to keep the online learning experiment fast.
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
    
    # Let's take the last 200 points for the online streaming test
    n_points = min(200, len(closes))
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
        if i % 50 == 0:
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
        model.update(x_i_std, y_i_std)
        
        # Tracking des clusters
        best_particle = max(model.smc.particles, key=lambda p: p.weight)
        cluster_assignments.append(best_particle.z[-1])
        
    print(f"{name} - Mean Online MSE: {np.mean(online_mses):.6f}")
    print(f"{name} - Mean Online NLPD: {np.mean(online_nlpds):.4f}")
    print(f"{name} - Discovered {best_particle.next_cluster_id} clusters total.")
    
    return np.array(online_X), np.array(online_mu), np.array(online_var), online_mses, cluster_assignments

def main():
    X, y = load_eurusd_data(period="1y")
    out_dir = os.path.dirname(__file__)
    D = X.shape[1]
    
    prior_mean = np.zeros(D + 1)
    prior_cov = np.eye(D + 1) * 3.0
    
    # 1. Run Standard GP-MOE
    model_std = GPMoE(D=D, J=20, prior_mean=prior_mean, prior_cov=prior_cov, alpha_init=0.5)
    std_X, std_mu, std_var, std_mses, std_c = run_online_experiment(model_std, X, y, name="Standard GP-MOE")
    
    # 2. Run Retrospective GP-MOE
    model_retro = GPMoE_Retrospective(D=D, J=20, prior_mean=prior_mean, prior_cov=prior_cov, alpha_init=0.5, window_size=20)
    ret_X, ret_mu, ret_var, ret_mses, ret_c = run_online_experiment(model_retro, X, y, name="Retrospective GP-MOE")
    
    # --- Visualization ---
    plt.figure(figsize=(12, 12))
    
    # Plot Standard
    plt.subplot(3, 1, 1)
    # Vrais points colorés par cluster
    plt.scatter(X, y, c=std_c, cmap='viridis', s=30, label='True EUR/USD Close', alpha=0.8)
    # Ligne de prédiction en ligne
    plt.plot(std_X, std_mu, 'r--', lw=2, label='Online Predictive Mean')
    plt.fill_between(std_X, std_mu - 1.96 * np.sqrt(std_var), std_mu + 1.96 * np.sqrt(std_var), color='r', alpha=0.2, label='95% CI')
    plt.title("Standard GP-MOE on EUR/USD - Sequential Online Prediction")
    plt.ylabel("Exchange Rate")
    plt.legend()
    plt.grid(True)
    
    # Plot Retrospective
    plt.subplot(3, 1, 2)
    plt.scatter(X, y, c=ret_c, cmap='plasma', s=30, label='True EUR/USD Close', alpha=0.8)
    plt.plot(ret_X, ret_mu, 'b--', lw=2, label='Online Predictive Mean')
    plt.fill_between(ret_X, ret_mu - 1.96 * np.sqrt(ret_var), ret_mu + 1.96 * np.sqrt(ret_var), color='b', alpha=0.2, label='95% CI')
    plt.title("Retrospective GP-MOE on EUR/USD - Sequential Online Prediction")
    plt.ylabel("Exchange Rate")
    plt.legend()
    plt.grid(True)
    
    # Plot Online MSE comparison
    plt.subplot(3, 1, 3)
    window = 10
    std_mse_smooth = np.convolve(std_mses, np.ones(window)/window, mode='valid')
    ret_mse_smooth = np.convolve(ret_mses, np.ones(window)/window, mode='valid')
    
    # On ajuste l'axe X pour le smoothing
    plot_x_smooth = std_X[window-1:]
    
    plt.plot(plot_x_smooth, std_mse_smooth, 'r-', lw=2, label='Standard GP-MOE')
    plt.plot(plot_x_smooth, ret_mse_smooth, 'b-', lw=2, label='Retrospective GP-MOE')
    plt.title(f"Online Prediction MSE (Rolling Average, Window={window})")
    plt.xlabel("Days")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "financial_data_online_results.png"))
    print(f"\nPlot saved to {os.path.join(out_dir, 'financial_data_online_results.png')}")

if __name__ == "__main__":
    main()