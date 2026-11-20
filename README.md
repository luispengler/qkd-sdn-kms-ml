# Physics-Informed Neural SDN Controller & ETSI-014 KMS for Multi-Hop QKD Networks

An integrated quantum network emulation, machine learning optimization, and Key Management System (KMS) framework bridging **physical-layer finite-key Decoy-State BB84 dynamics** with **ETSI GS QKD 014 REST APIs** and **Software-Defined Networking (SDN)** multi-hop routing.

## Architecture Overview

1. **Physical Layer & Fundamental ML Surrogate (`qkd_physics_ml.py`):**
   - Models realistic fiber-optic quantum channels characterized by fiber length $L$ (km), attenuation $\alpha$ (dB/km), detector efficiency $\eta_d$, dark count probability $Y_0$, optical misalignment error $e_{\text{det}}$, and finite block size $N$.
   - Evaluates the non-asymptotic Secret Key Rate (SKR) for **3-intensity Decoy-State BB84** ($\mu$ signal, $\nu$ weak decoy, $\omega=0$ vacuum) under strict finite-key statistical fluctuations bounds (Lim et al. 2014).
   - Solves the non-convex parameter landscape using Multi-start Nelder-Mead to generate ground-truth physics datasets.
   - Trains a **PyTorch Multi-Layer Perceptron (MLP)** with GELU activations and Huber Loss to bypass slow real-time numerical optimization. The network dynamically maps standardized real-time channel observables $(L, \log_{10}Y_0, e_{\text{det}}, \eta_d) \mapsto (\mu^*, \nu^*, P_\mu^*, \log_{10}\text{SKR}^*)$ in sub-millisecond inference time.

2. **ETSI GS QKD 014 Key Management System & SDN Controller (`kms_sdn_api.py`):**
   - Exposes standard **ETSI GS QKD 014 REST API** endpoints (`/api/v1/keys/{slave_SAE_ID}/status`, `/api/v1/keys/{slave_SAE_ID}/enc_keys`, and `/api/v1/keys/{master_SAE_ID}/dec_keys`) for Key Management Entities (KME/KMS).
   - Implements a **Software-Defined Networking (SDN) Controller** over a multi-domain quantum network topology (`NetworkX`). It queries the PyTorch neural surrogate and `scikit-learn` scalers to weight optical links by their real-time achievable Secret Key Rate and computes optimal multi-hop key relay paths.
   - Supports hybrid Post-Quantum Cryptography (PQC) + QKD key mixing (`SHA3-256` KDF) and simulated QRNG entropy seeding for crypto-agile KMS testing.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate physical decoy-state BB84 dataset and train the PyTorch MLP surrogate
python qkd_physics_ml.py --samples 5000 --epochs 250 --batch_size 128 --lr 0.001

# 2. Launch the ETSI-014 KMS & SDN Controller REST API server
uvicorn kms_sdn_api:app --reload --port 8000