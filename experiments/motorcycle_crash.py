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
        
        # S'assurer que les données sont bien triées chronologiquement pour l'apprentissage en ligne !
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
    # log N(y_true | y_pred, var_pred)
    nlpd = 0.5 * np.log(2 * np.pi * max(var_pred, 1e-6)) + (y_true - y_pred)**2 / (2 * max(var_pred, 1e-6))
    return mse, -nlpd

def run_online_experiment(X, y, J, name="Model", enable_retro=False, retro_freq=20, retro_threshold=3, retro_B=50):
    print(f"\nRunning {name} (J={J}) in STRICT ONLINE MODE...")
    
    streamer = DataStreamer(X, y)
    streamer.standardize()
    
    D = X.shape[1]
    
    # Priors calibrés
    prior_mean = np.zeros(D + 1)
    prior_mean[-1] = -2.0  
    prior_cov = np.eye(D + 1) * 0.5
    
    # Paramètres CRP ajustés pour forcer la segmentation
    crp_params = {
        'mu_0': np.zeros(D),
        'kappa_0': 1.0,
        'nu_0': float(D + 2.0),
        'Psi_0': np.eye(D) * 0.2
    }
    
    # Initialisation du modèle avec les nouveaux paramètres de rétrospection
    model = GPMoE(D=D, J=J, prior_mean=prior_mean, prior_cov=prior_cov, alpha_init=0.5, 
                  crp_params=crp_params,
                  enable_retro=enable_retro, 
                  retro_freq=retro_freq, 
                  retro_threshold=retro_threshold, 
                  retro_B=retro_B)
    
    # Listes pour stocker les prédictions en ligne
    online_X = []
    online_mu = []
    online_var = []
    online_mse = []
    online_nlpd = []
    
    # Liste pour stocker le cluster assigné après avoir vu le point
    cluster_assignments = []
    
    for i, (x_i_std, y_i_std) in enumerate(streamer):
        # Récupération des vraies valeurs pour l'affichage
        x_i_orig = X[i, 0]
        y_i_orig = y[i]
        
        # 1. PRÉDIRE LE POINT (avant de le connaître)
        if i > 0: # On ne peut pas prédire le tout premier point
            mu_std, var_std = model.predict(x_i_std)
            
            # Dé-standardisation pour les métriques et l'affichage
            mu = mu_std * streamer.y_std + streamer.y_mean
            var = var_std * (streamer.y_std ** 2)
            
            mse, nlpd = compute_metrics(y_i_orig, mu, var)
            
            online_X.append(x_i_orig)
            online_mu.append(mu)
            online_var.append(var)
            online_mse.append(mse)
            online_nlpd.append(nlpd)
            
        # 2. RÉVÉLER LE POINT ET METTRE À JOUR LE MODÈLE
        model.update(x_i_std, y_i_std)
        
        # Enregistre le cluster choisi par la meilleure particule
        best_particle = max(model.smc.particles, key=lambda p: p.weight)
        cluster_assignments.append(best_particle.z[-1])
        
    print(f"{name} - Mean Online MSE: {np.mean(online_mse):.4f}")
    print(f"{name} - Mean Online NLPD: {np.mean(online_nlpd):.4f}")
    print(f"{name} - Discovered {best_particle.next_cluster_id} clusters total.")
    
    return np.array(online_X), np.array(online_mu), np.array(online_var), cluster_assignments

def main():
    X, y = load_motorcycle_data()
    out_dir = os.path.dirname(__file__)
    
    # 1. Run Baseline (J=1) sans retro
    online_X_base, mu_base, var_base, c_base = run_online_experiment(
        X, y, J=1, name="Baseline POE (J=1)", enable_retro=False)
    
    # 2. Run GP-MOE Standard (J=20) sans retro
    online_X_moe, mu_moe, var_moe, c_moe = run_online_experiment(
        X, y, J=20, name="Standard GP-MOE (J=20)", enable_retro=False)
        
    # 3. Run GP-MOE Retrospectif (J=20)
    online_X_ret, mu_ret, var_ret, c_ret = run_online_experiment(
        X, y, J=60, name="Retrospective GP-MOE (J=60)", 
        enable_retro=True, retro_freq=10, retro_threshold=4)
    
    # --- Visualization ---
    plt.figure(figsize=(12, 14))
    
    # Plot Baseline
    plt.subplot(3, 1, 1)
    plt.scatter(X, y, c=c_base, cmap='tab10', s=30, label='True Observations')
    plt.plot(online_X_base, mu_base, 'k--', lw=2, label='Online Predictive Mean')
    plt.fill_between(online_X_base, mu_base - 1.96 * np.sqrt(var_base), mu_base + 1.96 * np.sqrt(var_base), color='k', alpha=0.15, label='95% CI')
    plt.title("Baseline POE (J=1) - Sequential Online Prediction")
    plt.ylabel("Acceleration (g)")
    plt.legend()
    plt.grid(True)
    
    # Plot Standard GP-MOE
    plt.subplot(3, 1, 2)
    plt.scatter(X, y, c=c_moe, cmap='tab10', s=30, label='True Observations')
    plt.plot(online_X_moe, mu_moe, 'r-', lw=2, label='Online Predictive Mean')
    plt.fill_between(online_X_moe, mu_moe - 1.96 * np.sqrt(var_moe), mu_moe + 1.96 * np.sqrt(var_moe), color='r', alpha=0.15, label='95% CI')
    plt.title("Standard GP-MOE (J=20) - Sequential Online Prediction")
    plt.ylabel("Acceleration (g)")
    plt.legend()
    plt.grid(True)
    
    # Plot Retrospective GP-MOE
    plt.subplot(3, 1, 3)
    plt.scatter(X, y, c=c_ret, cmap='tab10', s=30, label='True Observations')
    plt.plot(online_X_ret, mu_ret, 'b-', lw=2, label='Online Predictive Mean')
    plt.fill_between(online_X_ret, mu_ret - 1.96 * np.sqrt(var_ret), mu_ret + 1.96 * np.sqrt(var_ret), color='b', alpha=0.15, label='95% CI')
    plt.title("Retrospective GP-MOE (J=60) - With Heuristic Cleaning")
    plt.xlabel("Time (ms)")
    plt.ylabel("Acceleration (g)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "motorcycle_online_results_with_retro.png"))
    print(f"\nPlot saved to {os.path.join(out_dir, 'motorcycle_online_results_with_retro.png')}")
    
if __name__ == "__main__":
    main()