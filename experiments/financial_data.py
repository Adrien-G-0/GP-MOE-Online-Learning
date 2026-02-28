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
    # yfinance sometimes returns a MultiIndex column for Close, so we handle it gracefully
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

def run_experiment(model, X, y, name="Model"):
    print(f"\nRunning {name}...")
    streamer = DataStreamer(X, y)
    streamer.standardize()
    
    mses = []
    nlpds = []
    
    X_test = np.linspace(min(streamer.X_stdized), max(streamer.X_stdized), 200).reshape(-1, 1)
    X_test_orig = X_test * streamer.X_std + streamer.X_mean
    
    cluster_assignments = []
    
    for i, (x_i, y_i) in enumerate(streamer):
        if i % 50 == 0:
            print(f"Step {i}/{streamer.N}")
            
        if i > 0:
            mu, var = model.predict(x_i)
            mse, nlpd = compute_metrics(y_i, mu, var)
            mses.append(mse)
            nlpds.append(nlpd)
            
        model.update(x_i, y_i)
        best_particle = max(model.smc.particles, key=lambda p: p.weight)
        cluster_assignments.append(best_particle.z[-1])
        
    print(f"{name} - Mean Online MSE: {np.mean(mses):.6f}")
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
    
    mu_test = mu_test_std * streamer.y_std + streamer.y_mean
    var_test = var_test_std * (streamer.y_std ** 2)
    
    return mses, nlpds, X_test_orig, mu_test, var_test, cluster_assignments

def main():
    X, y = load_eurusd_data(period="1y")
    out_dir = os.path.dirname(__file__)
    D = X.shape[1]
    
    prior_mean = np.zeros(D + 1)
    # We allow slightly more variance in prior to adapt to financial shocks
    prior_cov = np.eye(D + 1) * 3.0
    
    # 1. Run Standard GP-MOE
    model_std = GPMoE(D=D, J=20, prior_mean=prior_mean, prior_cov=prior_cov, alpha_init=0.5)
    std_mses, std_nlpds, X_t, std_mu, std_var, std_c = run_experiment(model_std, X, y, name="Standard GP-MOE")
    
    # 2. Run Retrospective GP-MOE
    model_retro = GPMoE_Retrospective(D=D, J=20, prior_mean=prior_mean, prior_cov=prior_cov, alpha_init=0.5, window_size=20)
    ret_mses, ret_nlpds, X_t, ret_mu, ret_var, ret_c = run_experiment(model_retro, X, y, name="Retrospective GP-MOE")
    
    # --- Visualization ---
    plt.figure(figsize=(12, 12))
    
    # Plot Standard
    plt.subplot(3, 1, 1)
    plt.scatter(X, y, c=std_c, cmap='viridis', label='EUR/USD Close', alpha=0.6)
    plt.plot(X_t, std_mu, 'r-', label='Predictive Mean')
    plt.fill_between(X_t.flatten(), std_mu - 1.96 * np.sqrt(std_var), std_mu + 1.96 * np.sqrt(std_var), color='r', alpha=0.2, label='95% CI')
    plt.title("Standard GP-MOE on EUR/USD (Daily Close)")
    plt.ylabel("Exchange Rate")
    plt.legend()
    plt.grid(True)
    
    # Plot Retrospective
    plt.subplot(3, 1, 2)
    plt.scatter(X, y, c=ret_c, cmap='plasma', label='EUR/USD Close', alpha=0.6)
    plt.plot(X_t, ret_mu, 'b-', label='Predictive Mean')
    plt.fill_between(X_t.flatten(), ret_mu - 1.96 * np.sqrt(ret_var), ret_mu + 1.96 * np.sqrt(ret_var), color='b', alpha=0.2, label='95% CI')
    plt.title("Retrospective GP-MOE on EUR/USD")
    plt.ylabel("Exchange Rate")
    plt.legend()
    plt.grid(True)
    
    # Plot Online MSE comparison
    plt.subplot(3, 1, 3)
    window = 10
    std_mse_smooth = np.convolve(std_mses, np.ones(window)/window, mode='valid')
    ret_mse_smooth = np.convolve(ret_mses, np.ones(window)/window, mode='valid')
    plt.plot(std_mse_smooth, 'r-', label='Standard GP-MOE')
    plt.plot(ret_mse_smooth, 'b-', label='Retrospective GP-MOE')
    plt.title(f"Online MSE (Rolling Average, Window={window})")
    plt.xlabel("Days")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "financial_data_results.png"))
    print(f"Plot saved to {os.path.join(out_dir, 'financial_data_results.png')}")

if __name__ == "__main__":
    main()
