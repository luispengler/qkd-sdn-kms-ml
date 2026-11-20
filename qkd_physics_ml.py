"""
Physical Layer & Neural Surrogate for Finite-Key Decoy-State BB84 QKD.
Computes non-asymptotic secret key rates and trains a PyTorch MLP to predict
optimal protocol parameters (mu*, nu*, p_mu*) and achievable SKR from channel telemetry.
"""

# Define the imports
import argparse
import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Machine learning preprocessing and splitting tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Constants and protocol setup - Define fixed hardware/security numbers
N = 10**10 # Finite-size block fixed
alpha = 0.2 # db/Km, fiber attenuation fixed
w = 0.0 # vacuum intensity
P_w = 0.1 # vacuum probability
#P_mu = 0.7 # Signal probability. Is this number ok? As initial guess yep
P_z = 0.5 # Probability of generating pulse in Z basis
P_x = 0.5 # Probability of generating pulse in X basis
f_ec = 1.16 # Error correction efficiency. Why this number? standard empirical benchmark in QKD literature
eps_sec = 10**(-10) # Security epsilon number 1
eps_cor = 10**(-15) # Security epsilon number
e_zero = 0.5 # Background random error rate of dark counts

# Channel simulation - Given a channel L, Y0, e_det, eta_d, and protocol settings mu, nu, P_mu, simulate the expected click counts n_basis_probs
def simulate_channel(L, Y0, e_det, eta_d, mu, nu, P_mu):
    total_transmittance = eta_d * 10**(-alpha * L / 10.0) # Total transmittance
    P_nu = 1.0 - P_mu - P_w # Since all three probabilities must sum to 1
    intensities = {"mu": mu, "nu": nu, "w": w} # Intensity dictionary
    probs = {"mu": P_mu, "nu": P_nu, "w": P_w} # Probability dictionary
    n = {"Z": {}, "X": {}} # Number detection counts dictionary
    m = {"Z": {}, "X": {}} # Error detection counts dictionary

    for label, k in intensities.items():
        Q_k = 1.0 - (1.0 - Y0) * np.exp(-k * total_transmittance) # Gain (detection probability per pulse) for intensity k
        E_kQ_k = e_zero * Y0 + e_det * (1.0 - np.exp(-k * total_transmittance)) # Error probability per pulse for intensity k
        Pk = probs[label] # Probability of intensity k
        for basis, Pb in [("Z", P_z), ("X", P_x)]:
            n[basis][label] = N * Pb**2 * Pk * Q_k # Total expected detection counts in basis
            m[basis][label] = N * Pb**2 * Pk * E_kQ_k # Total expected error detection counts in basis

    return n, m

# Define functions to calculate SKR using Lim et al. 2014's 1-5 equations, to compute s0_l, s1_l, binary entropy, phi_z_u, leak_ec, delta_finite, and return R (theta)
# In Lim et al. (2014), the X basis is used for key generation and the Z basis is used for parameter estimation, I invert. No problem bc of symmetry.

def binary_entropy(x: float) -> float:
    """Compute binary Shannon entropy H2(x)."""
    x = np.clip(x, 1e-12, 1.0 - 1e-12) # So that ANY x is between 1e-12 and 1 - 1e-12. Any number smaller or greater than the bound gets mapped to the bound.
    return -x * np.log2(x) - (1.0 - x) * np.log2(1.0 - x)

def compute_skr(n, m, mu, nu, P_mu, eps_sec=10**(-10), eps_cor=10**(-15)):
    """Compute finite-key SKR lower bound (Lim et al., 2014, Eqs 1-5)."""
    P_nu = 1.0 - P_mu - P_w # Since all three probabilities must sum to 1
    if P_nu <= 0.0 or not (0.0 <= w < nu < mu): # Rejects invalid probabilities or invalid intensities
        return 0.0

    intensities = {"mu": mu, "nu": nu, "w": w} # Intensity dictionary
    probs = {"mu": P_mu, "nu": P_nu, "w": P_w} # Probability dictionary

    # Total counts & errors per basis
    n_tot = {b: sum(n[b].values()) for b in ["Z", "X"]}
    m_tot = {b: sum(m[b].values()) for b in ["Z", "X"]}
    if n_tot["Z"] <= 0.0 or n_tot["X"] <= 0.0: # Rejects invalid n counts for Z/X basis. Must be greater than 0.
        return 0.0

    # Hoeffding finite-key bounds (Eq. 4)
    log_term = np.log(21.0 / eps_sec)
    n_up, n_low = {"Z": {}, "X": {}}, {"Z": {}, "X": {}}
    m_up, m_low = {"Z": {}, "X": {}}, {"Z": {}, "X": {}}

    for b in ["Z", "X"]:
        delta_n = np.sqrt((n_tot[b] / 2) * log_term)
        delta_m = np.sqrt((m_tot[b] / 2) * log_term)
        for label, k in intensities.items():
            scale = np.exp(k) / probs[label]
            n_up[b][label] = scale * (n[b][label] + delta_n)
            n_low[b][label] = scale * max(0.0, n[b][label] - delta_n)
            m_up[b][label] = scale * (m[b][label] + delta_m)
            m_low[b][label] = scale * max(0.0, m[b][label] - delta_m)

    # Total photon-number probabilities tau_0 and tau_1
    tau_0 = sum(probs[lbl] * np.exp(-k) for lbl, k in intensities.items())
    tau_1 = sum(probs[lbl] * k * np.exp(-k) for lbl, k in intensities.items())

    # Vacuum (Eq. 2) and Single-Photon (Eq. 3) lower bounds for Z and X
    s0_L, s1_L = {}, {}
    denom_1 = mu * (nu - w) - (nu**2 - w**2)
    if denom_1 <= 0.0:
        return 0.0

    for b in ["Z", "X"]:
        s0_L[b] = max(0.0, (tau_0 / (nu - w)) * (nu * n_low[b]["w"] - w * n_up[b]["nu"]))
        bracket = (
            n_low[b]["nu"] - n_up[b]["w"] - ((nu**2 - w**2) / (mu**2)) * (n_up[b]["mu"] - s0_L[b] / tau_0)
        )
        s1_L[b] = max(0.0, (tau_1 * mu / denom_1) * bracket)
    if s1_L["Z"] <= 0.0 or s1_L["X"] <= 0.0:
        return 0.0

    # Single-photon phase error upper bound phi_Z^U (Eq. 5)
    v_X1_U = max(0.0, (tau_1 / (nu - w)) * (m_up["X"]["nu"] - m_low["X"]["w"]))
    e_X1_est = np.clip(v_X1_U / s1_L["X"], 1e-12, 0.5 - 1e-12)

    c, d = s1_L["Z"], s1_L["X"]
    gamma_inside = (((c + d) * (1.0 - e_X1_est) * e_X1_est) / (c * d * np.log(2.0))) * np.log2(((c + d) / (c * d * (1.0 - e_X1_est) * e_X1_est)) * (21.0**2 / eps_sec**2))
    gamma = np.sqrt(max(0.0, gamma_inside))
    phi_Z_U = min(0.5, e_X1_est + gamma)

    # Error correction leakage & final SKR lower bound (Eq. 1)
    leak_ec = n_tot["Z"] * f_ec * binary_entropy(m_tot["Z"] / n_tot["Z"])
    delta_finite = 6.0 * np.log2(21.0 / eps_sec) + np.log2(2.0 / eps_cor)

    secret_bits = s0_L["Z"] + s1_L["Z"] * (1.0 - binary_entropy(phi_Z_U)) - leak_ec - delta_finite
    return max(0.0, secret_bits / N)


# Dataset generation - where we sample random channels, run scipy.optimize with penalty bounds for C and save (x, y) pairs

def optimize_protocol(L, Y0, e_det, eta_d):
    """Find optimal (mu*, nu*, P_mu*) and max SKR* for a given channel using multi-start Nelder-Mead."""
    def objective(theta):
        mu, nu, P_mu = theta
        # Enforce feasible domain C: 0 < w < nu < mu < 1 and P_mu + P_nu + P_w = 1, because Nelder-Mead is an unconstrained optimizer
        if not (0.05 <= mu <= 0.95 and 0.005 <= nu <= mu - 0.02 and 0.1 <= P_mu <= 0.88): # These are more realistic, and are safe enough to avoid divide by zero
            return 1e6 # Returning 1e6 is  better than returning 0.0 here, for if SKR = 0 at long distances, returning 1e6 ensures an invalid guess looks strictly worse than a valid zero-key-rate point.
        n, m = simulate_channel(L, Y0, e_det, eta_d, mu, nu, P_mu)
        skr = compute_skr(n, m, mu, nu, P_mu)
        return -skr # Because minimize -skr = maximize skr

    initial_guesses = [
        [0.60, 0.12, 0.70], # high-intensity guess
        [0.45, 0.08, 0.65], # medium guess
        [0.30, 0.05, 0.50], # conservative guess
    ]

    best_skr = -1.0 # SKR landscape is non-convex and at long distances can be 0 everywhere, so initialize at -1.0 not to get stuck in plateus.
    best_theta = None

    for x0 in initial_guesses: # multi-start to avoid flat plateus and avoid local minima
        res = minimize(
            objective,
            x0=x0,
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-9, "maxiter": 400},
        )
        current_skr = -res.fun # flipping sign to accomodate minimize -skr = maximize skr
        if current_skr > best_skr:
            best_skr = current_skr
            best_theta = res.x

    return best_theta, best_skr # Here returns y belonging to R^4

def generate_dataset(n_samples=5000, seed=42):
    rng = np.random.default_rng(seed)
    rows = [] # Empty list to be filled

    while len(rows) < n_samples: # while instead of for loop to skip invalid channels and always return n_samples
        L = rng.uniform(5.0, 120.0) # Picks L between 5km and 120km uniformly
        Y0 = 10.0 ** rng.uniform(-7.0, -4.5)  # Log-uniform sampling across orders of magnitude, to ensure equal mix of low-noise (10^-7), medium (10^-6), and high (10^-5)
        e_det = rng.uniform(0.005, 0.035) # Picks optical misalignment error between 0.5% and 3.5% uniformly
        eta_d = rng.uniform(0.10, 0.30) # Picks a single-photon detector efficiency between 10% and 30%.

        theta_opt, skr_opt = optimize_protocol(L, Y0, e_det, eta_d) # We find y in R^4 optimized from x in R^4

        # Keep only physically viable links with positive SKR, like rejecting L = 120km and Y_0 = 3e-5
        if skr_opt > 1e-12 and theta_opt is not None:
            rows.append({
                "L": L,
                "Y0": Y0,
                "e_det": e_det,
                "eta_d": eta_d,
                "mu_opt": theta_opt[0],
                "nu_opt": theta_opt[1],
                "P_mu_opt": theta_opt[2],
                "log10_SKR": np.log10(skr_opt), # Stores the key rate on a base-10 logarithmic scale to avoid numerical instability in MLP training
            })

    return pd.DataFrame(rows)


# Preprocessing - log-transform + 80/10/10 split + scaling

def prepare_data(df, seed=42):
    # Transform Y0 to log10 scale so all 4 inputs are smooth & linear
    X_raw = np.column_stack([
        df["L"].values,
        np.log10(df["Y0"].values),
        df["e_det"].values,
        df["eta_d"].values,
    ])
    y_raw = df[["mu_opt", "nu_opt", "P_mu_opt", "log10_SKR"]].values

    # 80% train, 20% temp -> split 20% in half for 10% val and 10% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed
    )

    # Fit scalers ONLY on training data to prevent data leakage
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s = scaler_X.transform(X_val)
    X_test_s = scaler_X.transform(X_test)

    y_train_s = scaler_y.fit_transform(y_train)
    y_val_s = scaler_y.transform(y_val)
    y_test_s = scaler_y.transform(y_test)

    return (
        (X_train_s, y_train_s),
        (X_val_s, y_val_s),
        (X_test_s, y_test_s),
        (X_test, y_test),
        scaler_X,
        scaler_y,
    )

# MLP Architecture (4 -> 64 -> 64 -> 64 -> 4)
# 4 in and 4 out match the vector x,y sizes.
# 3*64 hidden layers is a choice with around 8800 parameters, enough to capture non-linearity in 4D space, while small enough for sub ms.
class QKDSurrogateMLP(nn.Module):
    def __init__(self, in_dim=4, out_dim=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),  # Smooth non-linearity ideal for continuous physics surfaces instead of ReLU...
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim), # Linear activation so output unbounded continuous numbers
        )

    def forward(self, x):
        return self.net(x)

# Training Loop with Early Stopping / Best Checkpoint
def train_mlp(df, epochs=250, batch_size=128, lr=1e-3, seed=42):
    # Hyperparemeters. 250 epochs might be sweet-spot for 2k-20k rows datasets, 128 batch size sweet-spot for gradient updates, learning rate 1e-3 gold-strd for AdamW
    torch.manual_seed(seed)
    device = torch.device("cpu") # No need for GPU, introduces CPU-GPU-CPU overhead and slows it down for small NN

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), (X_test_unscaled, y_test_unscaled), scaler_X, scaler_y = prepare_data(df, seed)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)), # No need for Float64, our precisions are capped at 10^-12
        batch_size=batch_size,
        shuffle=True, # Ramdomizes the rows before splitting into batches for every epoch
    )
    X_val_t = torch.tensor(X_va, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_va, dtype=torch.float32, device=device)

    model = QKDSurrogateMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5) # AdamW for adaptive lr, weight decay 10^-5 doesnt penalize too harshly
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15) # Lower validation_loss is better, instd of greater accuracy
    # no new val_loss each 15 (patience) epochs, update lr, cutting in half (factor).
    criterion = nn.HuberLoss(delta=1.0)  # Robust to occasional Nelder-Mead outliers: MSE for normal errors and L1 (MAE) for outliers

    best_val_loss = float("inf") # Inital val_loss is +infinity, so the next one is already better and will be saved
    best_weights = None # Copy new parameters for every save
    history = {"train_loss": [], "val_loss": []} # two lists for training_loss and validation_loss

    for epoch in range(epochs): # Loops the entire dataset epoch=250 times
        model.train() # Goes into training NN mode inside the loop
        batch_losses = [] #
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device) # From the training dataset!! Not the validation...
            optimizer.zero_grad() # clears gradients from prevs. batch
            pred = model(bx) # forward pass to get new predictions
            loss = criterion(pred, by) # Using huber loss to compare predictions against "by"
            loss.backward() # backpropagation
            optimizer.step() # takes direction that optimizes/reduces error
            batch_losses.append(loss.item())

        model.eval() # Goes into testing NN mode inside the loop
        with torch.no_grad(): # dont track gradients, since we are not training
            val_loss = criterion(model(X_val_t), y_val_t).item() # Finds validation loss

        train_loss = float(np.mean(batch_losses)) # Finds training loss
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    joblib.dump(scaler_X, "scaler_X.pkl")
    joblib.dump(scaler_y, "scaler_y.pkl")
    return model, scaler_X, scaler_y, X_test_unscaled, y_test_unscaled, history


# Physical Evaluation: Parameter Error, SKR Regret and Speedup

def evaluate_physics(model, scaler_X, scaler_y, X_test_unscaled, y_test_unscaled): # trained model using input/output scalers and raw test data
    device = next(model.parameters()).device
    model.eval()

    # Measure MLP inference latency
    X_test_s = torch.tensor(scaler_X.transform(X_test_unscaled), dtype=torch.float32, device=device) # standardize raw test data using mean and std deviation learned in training
    t0 = time.perf_counter() # timing the MLP
    with torch.no_grad():
        preds_s = model(X_test_s).cpu().numpy()
    mlp_time_ms = (time.perf_counter() - t0) * 1000.0 # get time in ms

    preds = scaler_y.inverse_transform(preds_s) # reverts standarize raw test data to real physical values

    # Clip predicted controls to strictly valid physical bounds C
    mu_hat = np.clip(preds[:, 0], 0.05, 0.95)
    nu_hat = np.clip(preds[:, 1], 0.005, mu_hat - 0.02)
    P_mu_hat = np.clip(preds[:, 2], 0.10, 0.88)
    log10_skr_hat = preds[:, 3]

    # Compare true optimal SKR* vs actual physical SKR achieved by MLP's predicted theta_hat
    true_skr = 10.0 ** y_test_unscaled[:, 3] # convert from log to linear, the nelder-mead SKR
    achieved_skr = np.zeros_like(true_skr) # placeholder for the predicted skr, same size as true_skr

    for i in range(len(X_test_unscaled)): # for every single test channel
        L, log10_Y0, e_det, eta_d = X_test_unscaled[i] # unpacks the 4 parameters
        Y0 = 10.0 ** log10_Y0 # from log to linear
        n, m = simulate_channel(L, Y0, e_det, eta_d, mu_hat[i], nu_hat[i], P_mu_hat[i]) # finds n, m using predicted mu, nu, Pmu
        achieved_skr[i] = compute_skr(n, m, mu_hat[i], nu_hat[i], P_mu_hat[i]) # finds the skr for predicted mu, nu, Pmu

    # Relative SKR optimality ratio (1.0 = 100% of Nelder-Mead's optimal key rate)
    optimality_ratio = achieved_skr / true_skr # ratio between predicted and real
    rel_regret_pct = (1.0 - optimality_ratio) * 100.0 # regret is the loss of key rate compared to exact optimum

    print(f"Total inference time for {len(X_test_unscaled)} links: {mlp_time_ms:.2f} ms "
          f"({mlp_time_ms / len(X_test_unscaled):.4f} ms/link)")
    print(f"MAE mu*:        {np.mean(np.abs(mu_hat - y_test_unscaled[:, 0])):.5f}")
    print(f"MAE nu*:        {np.mean(np.abs(nu_hat - y_test_unscaled[:, 1])):.5f}")
    print(f"MAE P_mu*:      {np.mean(np.abs(P_mu_hat - y_test_unscaled[:, 2])):.5f}")
    print(f"MAE log10(SKR): {np.mean(np.abs(log10_skr_hat - y_test_unscaled[:, 3])):.5f}")
    print(f"Median SKR Optimality: {np.median(optimality_ratio) * 100:.2f}% "
          f"(Median Regret: {np.median(rel_regret_pct):.3f}%)")
    print(f"Mean SKR Optimality: {np.mean(optimality_ratio) * 100:.3f}%")
    print(f"95th pct Regret:     {np.percentile(rel_regret_pct, 95):.3f}%")
    print(f"99th pct Regret:     {np.percentile(rel_regret_pct, 99):.3f}%")
    print(f"Min Optimality:      {np.min(optimality_ratio) * 100:.3f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Finite-Key Decoy-State BB84 Neural Surrogate")
    parser.add_argument("--samples", type=int, default=10000, help="Number of physical channel samples")
    parser.add_argument("--epochs", type=int, default=250, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--run-sanity", action="store_true", help="Run the L=50/120/138km sanity checks & baseline plotting")
    args = parser.parse_args()

    if args.run_sanity:
        # Quick sanity test at L = 50 km, 120km, 138km
        n_test, m_test = simulate_channel(L=50.0, Y0=1e-6, e_det=0.015, eta_d=0.2, mu=0.5, nu=0.1, P_mu=0.7)
        print("SKR at 50 km:", compute_skr(n_test, m_test, mu=0.5, nu=0.1, P_mu=0.7))

        n_test, m_test = simulate_channel(L=120.0, Y0=1e-6, e_det=0.015, eta_d=0.2, mu=0.5, nu=0.1, P_mu=0.7)
        print("SKR at 120 km:", compute_skr(n_test, m_test, mu=0.5, nu=0.1, P_mu=0.7))

        n_test, m_test = simulate_channel(L=138, Y0=1e-6, e_det=0.015, eta_d=0.2, mu=0.5, nu=0.1, P_mu=0.7)
        print("SKR at 138 km (cutoff):", compute_skr(n_test, m_test, mu=0.5, nu=0.1, P_mu=0.7))

        # CODE CELL GENERATED BY GEMINI 3.5 FLASH INSIDE GOOGLE COLAB - HIGHLIGHTING FIXED BASELINE PARAMETERS
        # Define distance array from 0 to 145 km
        L_values = np.linspace(0.0, 145.0, 200)

        # --- FIXED BASELINE PARAMETERS (TO BE OPTIMIZED LATER) ---
        Y0_val = 1e-6
        e_det_val = 0.015
        eta_d_val = 0.2
        mu_val = 0.5
        nu_val = 0.1
        P_mu_val = 0.7
        # ---------------------------------------------------------

        skr_values = []
        for L in L_values:
            n, m = simulate_channel(L=L, Y0=Y0_val, e_det=e_det_val, eta_d=eta_d_val, mu=mu_val, nu=nu_val, P_mu=P_mu_val)
            rate = compute_skr(n, m, mu=mu_val, nu=nu_val, P_mu=P_mu_val)
            skr_values.append(rate)

        # Plotting the results
        plt.figure(figsize=(8, 5))
        plt.plot(L_values, skr_values, label=r"Fixed Baseline ($\mu=0.5, \nu=0.1, P_\mu=0.7$)", color="red", linestyle="--", linewidth=2)
        plt.yscale("log")
        plt.xlabel("Fiber Distance L (km)")
        plt.ylabel("Secret Key Rate (SKR) per pulse")
        plt.title("SKR vs. Distance (Fixed Parameters Baseline - To Be Optimized)")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.xlim(0, 145)
        plt.ylim(1e-8, 1.0)  # focusing on positive non-zero key rates on a log scale
        plt.legend(loc="upper right")
        plt.show()

        # Test on 10 samples first
        df_test = generate_dataset(n_samples=10)
        print("Test DataFrame (10 samples):\n", df_test.head())

    # Generate the dataset (saving to CSV)
    print(f"Generating the dataset with {args.samples} samples... Wait.")
    df = generate_dataset(n_samples=args.samples, seed=42)
    csv_filename = f"qkd_decoy_dataset_{args.samples}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Dataset saved to {csv_filename}! Shape:", df.shape)

    # Train the MLP surrogate on the 80/10/10 split
    print("Training the MLP surrogate... Wait.")
    model, scaler_X, scaler_y, X_test_unscaled, y_test_unscaled, history = train_mlp(
        df, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=42
    )

    # Evaluate physical SKR optimality, parameter errors, and inference speedup
    evaluate_physics(model, scaler_X, scaler_y, X_test_unscaled, y_test_unscaled)

    print("\nTest Sample Description:")
    print(df[["mu_opt", "nu_opt", "P_mu_opt", "log10_SKR"]].describe())
