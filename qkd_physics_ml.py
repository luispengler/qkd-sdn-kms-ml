"""
Physical Layer & Neural Surrogate for Finite-Key Decoy-State BB84 QKD.
Computes non-asymptotic secret key rates and trains a PyTorch MLP to predict
optimal protocol parameters (mu*, nu*, p_mu*) and achievable SKR from channel telemetry.
"""

import argparse
import os
import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# =====================================================================
# 1. FINITE-KEY DECOY-STATE BB84 PHYSICS ENGINE
# =====================================================================

def binary_entropy(x: float) -> float:
    """Compute binary Shannon entropy H2(x)."""
    x = np.clip(x, 1e-12, 1.0 - 1e-12)
    return -x * np.log2(x) - (1.0 - x) * np.log2(1.0 - x)


def decoy_bb84_skr(
    params: np.ndarray,
    L_km: float,
    Y0: float = 1e-6,
    e_det: float = 0.015,
    eta_det: float = 0.15,
    alpha_db: float = 0.2,
    N_pulses: float = 1e9,
    f_ec: float = 1.16,
    eps_sec: float = 1e-10,
) -> float:
    """
    Calculate the Secret Key Rate (SKR) per pulse for 3-intensity Decoy-State BB84
    with finite-key statistical fluctuation bounds (Hoeffding/Chernoff approximation).
    params: [mu (signal intensity), nu (decoy intensity), p_mu (signal probability)]
    """
    mu, nu, p_mu = params
    if nu <= 0.005 or mu <= nu + 0.02 or mu > 1.2 or not (0.1 <= p_mu <= 0.9):
        return 0.0

    # Channel transmittance: eta_ch = 10^(-alpha * L / 10)
    eta = eta_det * (10.0 ** (-alpha_db * L_km / 10.0))

    # Overall gains Q_k and Quantum Bit Error Rates E_k for intensity k in {mu, nu}
    Q_mu = 1.0 - (1.0 - Y0) * np.exp(-mu * eta)
    E_mu = (0.5 * Y0 + e_det * (1.0 - np.exp(-mu * eta))) / max(Q_mu, 1e-15)

    Q_nu = 1.0 - (1.0 - Y0) * np.exp(-nu * eta)
    E_nu = (0.5 * Y0 + e_det * (1.0 - np.exp(-nu * eta))) / max(Q_nu, 1e-15)

    # Finite-key statistical fluctuation penalty on decoy gain Q_nu
    p_nu = (1.0 - p_mu) * 0.8  # 80% of remaining pulses to weak decoy, 20% to vacuum
    N_nu = max(N_pulses * p_nu * Q_nu, 1.0)
    delta_nu = np.sqrt(np.log(1.0 / eps_sec) / (2.0 * N_nu))
    Q_nu_lower = max(Q_nu * (1.0 - delta_nu), 1e-15)
    Q_nu_upper = Q_nu * (1.0 + delta_nu)

    # Analytical lower bound on single-photon yield Y1^L and upper bound on phase error e1^U
    denom = mu * nu - nu**2
    Y1_lower = (mu / denom) * (
        Q_nu_lower * np.exp(nu)
        - Q_mu * np.exp(mu) * (nu**2 / mu**2)
        - ((mu**2 - nu**2) / mu**2) * Y0
    )
    Y1_lower = max(Y1_lower, 0.0)

    # Single-photon gain lower bound Q1^L
    Q1_lower = mu * np.exp(-mu) * Y1_lower
    if Q1_lower <= 1e-15:
        return 0.0

    e1_upper = (E_nu * Q_nu_upper * np.exp(nu) - 0.5 * Y0) / max(Y1_lower * nu, 1e-15)
    e1_upper = np.clip(e1_upper, 0.0, 0.5)
    if e1_upper >= 0.11 or E_mu >= 0.11:
        return 0.0

    # Basis sifting factor (asymmetric BB84 with Z-basis preference q = 0.8)
    q_sift = 0.64
    finite_overhead = 6.0 * np.log2(21.0 / eps_sec) / N_pulses

    skr = q_sift * p_mu * (
        -Q_mu * f_ec * binary_entropy(E_mu)
        + Q1_lower * (1.0 - binary_entropy(e1_upper))
    ) - finite_overhead

    return float(max(skr, 0.0))


def optimize_channel_parameters(L_km: float, Y0: float, e_det: float, eta_det: float):
    """Numerically find optimal (mu*, nu*, p_mu*) maximizing finite-key SKR."""
    best_skr = 0.0
    best_params = np.array([0.48, 0.12, 0.75], dtype=np.float32)

    res = minimize(
        lambda x: -decoy_bb84_skr(x, L_km=L_km, Y0=Y0, e_det=e_det, eta_det=eta_det),
        x0=best_params,
        bounds=[(0.15, 0.90), (0.02, 0.30), (0.30, 0.90)],
        method="Nelder-Mead",
        options={"maxiter": 120},
    )
    cand_skr = -res.fun
    if cand_skr > best_skr:
        best_skr = cand_skr
        best_params = res.x.astype(np.float32)

    return best_params[0], best_params[1], best_params[2], float(best_skr)


# =====================================================================
# 2. PYTORCH NEURAL SURROGATE ARCHITECTURE
# =====================================================================

class DecoyBB84SurrogateMLP(nn.Module):
    """
    Physics-Informed MLP mapping normalized channel observables:
      Input (4): [L_km / 150, log10(Y0) / -10, e_det / 0.05, eta_det]
      Output (4): [mu*, nu*, p_mu*, log10(SKR* + 1e-10)]
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def normalize_inputs(L_km: np.ndarray, Y0: np.ndarray, e_det: np.ndarray, eta_det: np.ndarray) -> np.ndarray:
    return np.stack([
        L_km / 150.0,
        np.log10(np.clip(Y0, 1e-8, 1e-3)) / -8.0,
        e_det / 0.05,
        eta_det,
    ], axis=-1).astype(np.float32)


def train_surrogate(n_samples: int = 2000, epochs: int = 50, model_path: str = "qkd_surrogate.pt"):
    """Generate physical dataset via numerical optimization and train the PyTorch surrogate."""
    rng = np.random.default_rng(42)
    L_samples = rng.uniform(5.0, 120.0, size=n_samples)
    Y0_samples = 10.0 ** rng.uniform(-7.0, -4.5, size=n_samples)
    edet_samples = rng.uniform(0.005, 0.035, size=n_samples)
    eta_samples = rng.uniform(0.10, 0.30, size=n_samples)

    X = normalize_inputs(L_samples, Y0_samples, edet_samples, eta_samples)
    Y = np.zeros((n_samples, 4), dtype=np.float32)

    print(f"[Physics Engine] Generating {n_samples} finite-key Decoy-State BB84 channel configurations...")
    for i in range(n_samples):
        mu_opt, nu_opt, pmu_opt, skr_opt = optimize_channel_parameters(
            L_samples[i], Y0_samples[i], edet_samples[i], eta_samples[i]
        )
        Y[i] = [mu_opt, nu_opt, pmu_opt, np.log10(skr_opt + 1e-10)]

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = DecoyBB84SurrogateMLP(hidden_dim=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print(f"[PyTorch Training] Training neural surrogate for {epochs} epochs...")
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(bx)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d}/{epochs:03d} | MSE Loss: {epoch_loss / n_samples:.6f}")

    torch.save(model.state_dict(), model_path)
    print(f"[Success] Saved trained weights to {model_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Finite-Key Decoy-State BB84 Neural Surrogate")
    parser.add_argument("--samples", type=int, default=1500, help="Number of physical channel samples")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    args = parser.parse_args()
    train_surrogate(n_samples=args.samples, epochs=args.epochs)