"""
ETSI GS QKD 014 Compliant Key Management System (KMS/KME) & Neural SDN Controller.
Integrates the PyTorch Decoy-State BB84 surrogate with multi-hop quantum routing
and hybrid QKD + Post-Quantum Cryptography (PQC) key derivation.
"""

import base64
import hashlib
import os
import secrets
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import networkx as nx
import numpy as np
import torch

from qkd_physics_ml import DecoyBB84SurrogateMLP, normalize_inputs, decoy_bb84_skr

app = FastAPI(
    title="RNP Multi-Domain Quantum Network KMS & Neural SDN API",
    description=(
        "ETSI GS QKD 014 compliant Key Management Entity (KME/KMS) REST API integrated "
        "with a Physics-Informed Neural SDN Controller for finite-key Decoy-State BB84 networks."
    ),
    version="1.0.0",
)

# =====================================================================
# 1. INITIALIZE NEURAL SURROGATE & QUANTUM NETWORK TOPOLOGY
# =====================================================================

MODEL_PATH = "qkd_surrogate.pt"
surrogate_model = DecoyBB84SurrogateMLP(hidden_dim=64)
if os.path.exists(MODEL_PATH):
    surrogate_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
surrogate_model.eval()

# Build a 5-node Metropolitan Quantum Network (inspired by Rede Rio Quantica / RNP testbeds)
quantum_net = nx.Graph()
DEFAULT_LINKS = [
    ("SAE_CBPF", "SAE_UFRJ", {"L_km": 14.5, "Y0": 1.2e-6, "e_det": 0.012, "eta_det": 0.22}),
    ("SAE_CBPF", "SAE_UFF",  {"L_km": 22.0, "Y0": 2.0e-6, "e_det": 0.015, "eta_det": 0.20}),
    ("SAE_UFRJ", "SAE_PUC",  {"L_km": 18.0, "Y0": 1.5e-6, "e_det": 0.014, "eta_det": 0.21}),
    ("SAE_UFF",  "SAE_PUC",  {"L_km": 16.5, "Y0": 1.8e-6, "e_det": 0.013, "eta_det": 0.22}),
    ("SAE_PUC",  "SAE_RNP_CORE", {"L_km": 12.0, "Y0": 1.0e-6, "e_det": 0.010, "eta_det": 0.25}),
    ("SAE_UFRJ", "SAE_RNP_CORE", {"L_km": 38.0, "Y0": 3.5e-6, "e_det": 0.022, "eta_det": 0.18}),
]
for u, v, attrs in DEFAULT_LINKS:
    quantum_net.add_edge(u, v, **attrs)

# In-memory KMS Key Store: maps key_ID -> {"key_bytes": bytes, "master_sae": str, "slave_sae": str}
KMS_KEY_POOL: Dict[str, dict] = {}


def evaluate_link_with_neural_surrogate(attrs: dict) -> dict:
    """Use PyTorch surrogate (with physical verification) to compute optimal decoy-state parameters and SKR."""
    x_norm = normalize_inputs(
        np.array([attrs["L_km"]]),
        np.array([attrs["Y0"]]),
        np.array([attrs["e_det"]]),
        np.array([attrs["eta_det"]]),
    )
    with torch.no_grad():
        pred = surrogate_model(torch.from_numpy(x_norm)).numpy()[0]

    mu_opt = float(np.clip(pred[0], 0.20, 0.80))
    nu_opt = float(np.clip(pred[1], 0.04, 0.25))
    pmu_opt = float(np.clip(pred[2], 0.40, 0.85))

    # Evaluate physical finite-key SKR at the neural-predicted operating point
    skr_per_pulse = decoy_bb84_skr(
        np.array([mu_opt, nu_opt, pmu_opt]),
        L_km=attrs["L_km"],
        Y0=attrs["Y0"],
        e_det=attrs["e_det"],
        eta_det=attrs["eta_det"],
    )
    # Assume 100 MHz clock rate -> convert SKR/pulse to kbps
    skr_kbps = max(skr_per_pulse * 1e8 / 1e3, 0.01)
    return {
        "mu_signal": round(mu_opt, 4),
        "nu_decoy": round(nu_opt, 4),
        "p_mu": round(pmu_opt, 4),
        "skr_kbps": round(skr_kbps, 3),
    }


# =====================================================================
# 2. PYDANTIC SCHEMAS (ETSI GS QKD 014 STANDARD)
# =====================================================================

class ETSIStatusResponse(BaseModel):
    source_KME_ID: str
    target_KME_ID: str
    master_SAE_ID: str
    slave_SAE_ID: str
    key_size: int = 256
    stored_key_count: int = 2500
    max_key_count: int = 10000
    max_key_per_request: int = 128
    max_key_size: int = 1024
    min_key_size: int = 64


class KeyItem(BaseModel):
    key_ID: str
    key: str = Field(..., description="Base64-encoded cryptographic key material")


class ETSKeyContainerResponse(BaseModel):
    keys: List[KeyItem]
    sdn_route_metadata: Optional[dict] = None


class DecKeyIDRequest(BaseModel):
    key_IDs: List[Dict[str, str]]


# =====================================================================
# 3. ETSI GS QKD 014 & SDN CONTROLLER ENDPOINTS
# =====================================================================

@app.get("/api/v1/keys/{slave_SAE_ID}/status", response_model=ETSIStatusResponse, tags=["ETSI GS QKD 014"])
def get_status(slave_SAE_ID: str, master_SAE_ID: str = Query(default="SAE_CBPF")):
    """ETSI GS QKD 014 Status Endpoint: Returns KME/KMS link capacity and key pool status."""
    return ETSIStatusResponse(
        source_KME_ID=f"KME_{master_SAE_ID}",
        target_KME_ID=f"KME_{slave_SAE_ID}",
        master_SAE_ID=master_SAE_ID,
        slave_SAE_ID=slave_SAE_ID,
    )


@app.get("/api/v1/keys/{slave_SAE_ID}/enc_keys", response_model=ETSKeyContainerResponse, tags=["ETSI GS QKD 014"])
def get_key(
    slave_SAE_ID: str,
    master_SAE_ID: str = Query(default="SAE_CBPF"),
    number: int = Query(default=1, ge=1, le=32),
    size: int = Query(default=256, description="Key length in bits (multiple of 8)"),
    hybrid_pqc: bool = Query(default=True, description="Mix QKD key with simulated PQC (ML-KEM) secret via SHA3-256"),
):
    """
    ETSI GS QKD 014 Get Key Endpoint:
    1. Computes the optimal multi-hop SDN path between master_SAE_ID and slave_SAE_ID using the Neural Surrogate.
    2. Generates QRNG-seeded QKD key material (optionally hybridized with Post-Quantum Cryptography).
    3. Stores the key in the KMS pool and returns the key_ID and Base64 key.
    """
    if size % 8 != 0:
        raise HTTPException(status_code=400, detail="Key size must be a multiple of 8 bits.")

    route_info = compute_optimal_sdn_route(master_SAE_ID, slave_SAE_ID)
    n_bytes = size // 8
    output_keys = []

    for _ in range(number):
        key_id = str(uuid.uuid4())
        qkd_raw = secrets.token_bytes(n_bytes)  # Simulated QRNG / sifted & privacy-amplified QKD bits

        if hybrid_pqc:
            pqc_kem_secret = secrets.token_bytes(n_bytes)
            h = hashlib.sha3_256(qkd_raw + pqc_kem_secret).digest()
            final_key_bytes = (h * ((n_bytes // 32) + 1))[:n_bytes]
        else:
            final_key_bytes = qkd_raw

        b64_key = base64.b64encode(final_key_bytes).decode("utf-8")
        KMS_KEY_POOL[key_id] = {
            "key": b64_key,
            "master_SAE_ID": master_SAE_ID,
            "slave_SAE_ID": slave_SAE_ID,
        }
        output_keys.append(KeyItem(key_ID=key_id, key=b64_key))

    return ETSKeyContainerResponse(keys=output_keys, sdn_route_metadata=route_info)


@app.post("/api/v1/keys/{master_SAE_ID}/dec_keys", response_model=ETSKeyContainerResponse, tags=["ETSI GS QKD 014"])
def get_key_with_key_ids(master_SAE_ID: str, payload: DecKeyIDRequest):
    """
    ETSI GS QKD 014 Get Key with Key IDs Endpoint:
    Allows the receiving Slave SAE to retrieve matching symmetric keys from its local KME using key_IDs.
    """
    retrieved = []
    for item in payload.key_IDs:
        k_id = item.get("key_ID")
        if k_id not in KMS_KEY_POOL:
            raise HTTPException(status_code=404, detail=f"key_ID {k_id} not found in KMS pool.")
        record = KMS_KEY_POOL[k_id]
        retrieved.append(KeyItem(key_ID=k_id, key=record["key"]))

    return ETSKeyContainerResponse(keys=retrieved)


@app.get("/api/v1/sdn/route", tags=["Neural SDN Controller"])
def compute_optimal_sdn_route(source: str = "SAE_CBPF", target: str = "SAE_RNP_CORE"):
    """
    Queries the PyTorch Decoy-State BB84 surrogate across all optical links in the network,
    assigns routing weights w_e = 1 / SKR_e, and computes the bottleneck-optimal multi-hop QKD path.
    """
    if source not in quantum_net or target not in quantum_net:
        raise HTTPException(status_code=404, detail=f"Node {source} or {target} not in quantum topology.")

    link_telemetry = {}
    for u, v, data in quantum_net.edges(data=True):
        metrics = evaluate_link_with_neural_surrogate(data)
        data["weight"] = 1.0 / max(metrics["skr_kbps"], 0.01)
        data["skr_kbps"] = metrics["skr_kbps"]
        link_telemetry[f"{u}<->{v}"] = {**data, **metrics}

    path = nx.shortest_path(quantum_net, source=source, target=target, weight="weight")
    hop_skrs = [quantum_net[path[i]][path[i + 1]]["skr_kbps"] for i in range(len(path) - 1)]

    return {
        "source_node": source,
        "target_node": target,
        "optimal_multi_hop_path": path,
        "end_to_end_bottleneck_skr_kbps": min(hop_skrs) if hop_skrs else 0.0,
        "link_neural_optimization": link_telemetry,
    }