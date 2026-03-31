from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


def display(obj):
    if hasattr(obj, "to_string"):
        try:
            print(obj.to_string(index=False))
            return
        except Exception:
            pass
    print(obj)


def _configure_output_dir() -> Path:
    parser = argparse.ArgumentParser(
        description="Run the single-start FlipFlop FOCEI comparison used in the manuscript."
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where figures and tables will be written.",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)
    print(f"Output directory: {output_dir}")
    return output_dir


OUTPUT_DIR = _configure_output_dir()
# %% [markdown] cell 1
# # Single-start FOCEI comparison for the absorption/elimination-ambiguity example
# 
# This standalone script runs only the **single-start** comparison used in the manuscript. It does
# **not** reproduce the full paper workflow, multistart experiments, or the full set of manuscript
# figures.
# 
# The script compares four FOCEI variants from the same initial point:
# 
# - **FOCEI-FD**: outer optimization uses finite-difference gradients in $\theta$, and the inner
#   conditional modes $\hat\eta_i(\theta)$ are also computed using finite differences.
# - **FOCEI-AD-stop (STOP)**: autodiff gradients in $\theta$ with stop-gradient through
#   $\hat\eta(\theta)$.
# - **FOCEI-AD-full-unroll (FULL-unroll)**: autodiff gradients in $\theta$ through the unrolled
#   inner solver.
# - **FOCEI-AD-full-implicit (FULL-implicit)**: autodiff gradients in $\theta$ using implicit
#   differentiation of the inner mode conditions.
# 
# The example is a one-compartment subcutaneous absorption model with absorption/elimination
# ambiguity that produces two competing parameter sets.
# 
# _Extracted from the manuscript code for public single-start reproducibility._

# %% [code] cell 2
# ============================================================
# 0) Imports + global settings
# ============================================================
import os
import time
import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize

import torch

# --- Reproducibility ---
SEED = 123
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

# Use float64 for numerical stability (Hessians/logdets)
torch.set_default_dtype(torch.float64)

# --- Device ---
DEVICE = torch.device("cpu")  # keep CPU for reproducibility

# --- Output dirs ---
os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)

print("Python:", f"{os.sys.version.split()[0]}")
print("Torch:", torch.__version__, "| device:", DEVICE, "| dtype:", torch.get_default_dtype())

# A large penalty used when Laplace terms become invalid
BIG = 1e12

# -----------------------------
# Profile optimization controls
# -----------------------------

GTOL = 1e-4
FTOL = 1e-9
MAXLS = 40 # Higher numbers give the line search more chances to find a safe step instead of bailing/"ABNORMAL"

# Outer iteration budgets (adjust)
MAXITER_SINGLE_AD = 200
MAXITER_SINGLE_FD = MAXITER_SINGLE_AD  # FD is much slower

# Inner eta-mode Newton steps
MAXITER_ETA_AD = 50
MAXITER_ETA_FD = MAXITER_ETA_AD

# FD step sizes
EPS_THETA_FD = 1e-3
EPS_ETA_FD   = 1e-3


# Run the 2D countour?
RUN_CONTOUR = False

# ----------------------------
# Multi-start controls
# ----------------------------
RUN_MULTISTART = False
RUN_SINGLE = True

N_STARTS = 100
OUTER_MAXITER_AD = 50
OUTER_MAXITER_FD = OUTER_MAXITER_AD#20

# FD is expensive: optionally run fewer FD starts
N_STARTS_FD = N_STARTS #min(8, N_STARTS)

# Inner Newton limits
MAXITER_ETA_MS_AD = OUTER_MAXITER_AD
MAXITER_ETA_MS_FD = MAXITER_ETA_MS_AD #4

# FD steps
EPS_THETA_MS_FD = 2e-4
EPS_ETA_MS_FD   = 2e-5

# Cache paths (resume)
MS_CACHE_CSV = "tables/multistart_results.csv"
MS_STARTS_NPY = "tables/multistart_theta_starts.npy"

# Helper: treat penalty objectives as invalid solutions
def is_penalty(x: float, big: float = BIG) -> bool:
    return (not np.isfinite(x)) or (x >= big / 10.0)

METHODS = ["FULL_IMPLICIT", "FULL_UNROLL", "STOP", "FD"]
METHOD_LABEL = {
    "FULL_IMPLICIT": "FULL-implicit",
    "FULL_UNROLL":   "FULL-unroll",
    "STOP":          "STOP",
    "FD":            "FD",
}

def norm_method(m) -> str:
    s = str(m).strip().upper().replace("-", "_").replace(" ", "_")
    # allow legacy names you used later in the notebook
    if s in {"AD_STOP", "FOCEI_AD_STOP"}:
        return "STOP"
    if s in {"AD_FULL", "FOCEI_AD_FULL", "FULL"}:
        return "FULL_UNROLL"
    if s in {"AD_FULL_IMPLICIT", "FOCEI_AD_FULL_IMPLICIT"}:
        return "FULL_IMPLICIT"
    if s in {"FOCEI_FD"}:
        return "FD"
    return s

# Optional: if you sometimes want "FULL" as a group
METHOD_GROUP = {
    "FD": "FD",
    "STOP": "STOP",
    "FULL_UNROLL": "FULL",
    "FULL_IMPLICIT": "FULL",
}


# %% [markdown] cell 3
# ## 1) Model and simulated flip--flop dataset
# 
# We use a 1-compartment model with first-order absorption and elimination:
# 
# \[
# C(t) = \frac{D}{V}\,\frac{k_a}{k_a-k_e}\,\left(e^{-k_e t} - e^{-k_a t}\right),\quad k_e = \frac{CL}{V}.
# \]
# 
# Inter-individual variability (IIV) is log-normal on $(k_a, CL, V)$ via random effects $\eta_i \in \mathbb{R}^3$:
# \[
# \log k_{a,i} = \log k_a + \eta_{i,1},\quad
# \log CL_i = \log CL + \eta_{i,2},\quad
# \log V_i = \log V + \eta_{i,3},
# \]
# with diagonal $\Omega = \mathrm{diag}(\omega_{ka}^2, \omega_{CL}^2, \omega_V^2)$.
# 
# Residual error is additive: $y_{ij} = C_i(t_{ij}) + \epsilon_{ij}$, $\epsilon_{ij}\sim\mathcal{N}(0,\sigma^2)$ (clipped at a small positive value for log-scale plotting).
# 
# Two ``true'' population parameter sets are constructed to represent a flip--flop ambiguity:
# - True1: $(k_a, CL, V)$
# - True2: swap $k_a$ and $k_e$ while keeping $CL$ fixed, so $V$ rescales accordingly.

# %% [code] cell 4
# ============================================================
# 1) Model helpers (torch + numpy)
# ============================================================

DOSE = 100.0  # arbitrary units (e.g., mg)

# Sampling schedule (hours) - adjust as needed
TIME_POINTS_HR = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 24.0, 48.0], dtype=float)

N_SUBJ = 50

TIMES_NP = TIME_POINTS_HR
TIMES_T  = torch.tensor(TIME_POINTS_HR, device=DEVICE)


import numpy as np
import torch

# --- Numerics guards (same idea as warfarin notebook) ---
LOG_EXP_CLIP = 50.0      # conservative; loosen to 100 or 700 if you want
MIN_POS      = 1e-12     # keeps V, CL, ka away from 0

def safe_exp_np(x, lo=-LOG_EXP_CLIP, hi=LOG_EXP_CLIP):
    # Works for scalars or arrays
    return np.exp(np.clip(x, lo, hi))

def safe_exp_torch(x, lo=-LOG_EXP_CLIP, hi=LOG_EXP_CLIP):
    return torch.exp(torch.clamp(x, min=lo, max=hi))


def conc_one_comp_sc_np(t, ka, CL, V, dose, tol=1e-6):
    """
    1CMT first-order absorption, stable for ka~=ke and extreme (CL,V) from eta-exploration.
    Uses the same 'abs(delta)' formulation as the torch implementation.
    """
    t = np.asarray(t, dtype=float)

    ka = np.float64(ka)
    CL = np.float64(CL)
    V  = np.float64(V)

    # If something is non-finite, return NaNs (caller will turn objective into penalty)
    if (not np.isfinite(ka)) or (not np.isfinite(CL)) or (not np.isfinite(V)):
        return np.full_like(t, np.nan, dtype=float)

    # Clamp away from zero to prevent divide-by-zero / overflow cascades
    ka = np.maximum(ka, MIN_POS)
    CL = np.maximum(CL, MIN_POS)
    V  = np.maximum(V,  MIN_POS)

    ke = CL / V

    absd = np.abs(ka - ke)                 # |ka - ke|
    base_rate = np.minimum(ka, ke)         # min(ka, ke)
    base = np.exp(-base_rate * t)

    x = absd * t                           # >= 0
    denom = np.maximum(absd, 1e-12)
    ratio = -np.expm1(-x) / denom          # (1 - exp(-x)) / |ka-ke|

    # Series only where needed (avoids overflow warnings from computing unused branch)
    mask = x < tol
    if np.any(mask):
        tm = t[mask]
        ratio[mask] = tm - 0.5 * absd * tm**2 + (absd**2) * tm**3 / 6.0

    C = (dose / V) * ka * base * ratio
    return np.where(np.isfinite(C), C, np.nan)


def conc_one_comp_sc_torch(t, ka, CL, V, dose, tol=1e-6):
    """
    Torch version matched to conc_one_comp_sc_np (abs-delta formulation + clamps).
    """
    ka = torch.clamp(ka, min=MIN_POS)
    CL = torch.clamp(CL, min=MIN_POS)
    V  = torch.clamp(V,  min=MIN_POS)

    ke = CL / V

    absd = torch.abs(ka - ke)
    base_rate = torch.minimum(ka, ke)
    base = torch.exp(-base_rate * t)

    x = absd * t
    denom = torch.clamp(absd, min=1e-12)
    ratio_raw = -torch.expm1(-x) / denom

    mask = x < tol
    # compute series safely only for small-x elements
    absd_small = torch.where(mask, absd, torch.zeros_like(absd))
    ratio_series = t - 0.5 * absd_small * t**2 + (absd_small**2) * t**3 / 6.0
    ratio = torch.where(mask, ratio_series, ratio_raw)

    C = (dose / V) * ka * base * ratio
    return C






# ============================================================
# 2) True parameters (flip--flop construction)
# ============================================================

TRUE1 = dict(
    ka_pop = 0.05,
    CL_pop = 3.0,
    V_pop  = 20.0,
    omega_ka = 0.1,
    omega_CL = 0.1,
    omega_V  = 0.1,
    sigma    = 0.1,
)

def make_theta_from_TRUE(TRUE: dict) -> np.ndarray:
    return np.array([
        math.log(TRUE["ka_pop"]),
        math.log(TRUE["CL_pop"]),
        math.log(TRUE["V_pop"]),
        math.log(TRUE["omega_ka"]),
        math.log(TRUE["omega_CL"]),
        math.log(TRUE["omega_V"]),
        math.log(TRUE["sigma"]),
    ], dtype=float)

theta_true1 = make_theta_from_TRUE(TRUE1)

# Flip--flop: swap ka and ke while keeping CL fixed => V rescales
ke1 = TRUE1["CL_pop"] / TRUE1["V_pop"]
ka2 = ke1
ke2 = TRUE1["ka_pop"]
V2  = TRUE1["CL_pop"] / ke2

TRUE2 = dict(TRUE1)
TRUE2.update(dict(ka_pop=ka2, V_pop=V2))
theta_true2 = make_theta_from_TRUE(TRUE2)

print("True1 (ka, CL, V):", TRUE1["ka_pop"], TRUE1["CL_pop"], TRUE1["V_pop"], "ke=", ke1)
print("True2 (ka, CL, V):", TRUE2["ka_pop"], TRUE2["CL_pop"], TRUE2["V_pop"], "ke=", ke2)

# ============================================================
# 3) Simulate dataset
# ============================================================

def simulate_dataset(theta_true: np.ndarray, n_subj: int = N_SUBJ, times_np: np.ndarray = TIMES_NP):
    """Simulate one dataset under theta_true."""
    logka, logCL, logV, logwka, logwCL, logwV, logsig = theta_true
    w = np.exp([logwka, logwCL, logwV])
    sigma = float(np.exp(logsig))

    etas = rng.normal(0.0, w, size=(n_subj, 3))
    Y = np.zeros((n_subj, len(times_np)), dtype=float)

    for i in range(n_subj):
        logka_i = logka + etas[i, 0]
        logCL_i = logCL + etas[i, 1]
        logV_i  = logV  + etas[i, 2]
        ka_i, CL_i, V_i = np.exp([logka_i, logCL_i, logV_i])

        c = conc_one_comp_sc_np(times_np, ka_i, CL_i, V_i, dose=DOSE)
        y = c + rng.normal(0.0, sigma, size=c.shape)
        # clip for log-scale plots only (keeps likelihood well-defined)
        y = np.maximum(y, 1e-8)
        Y[i, :] = y

    return Y, etas

# Choose which truth generates the simulated dataset
DATA_GENERATING_THETA = theta_true1  # change to theta_true2 if desired

Y_NP, ETA_TRUE_NP = simulate_dataset(DATA_GENERATING_THETA, n_subj=N_SUBJ, times_np=TIMES_NP)
Y_T = torch.tensor(Y_NP, device=DEVICE)

print("Simulated Y shape:", Y_NP.shape)

# %% [code] cell 5
# ============================================================
# 4) Data summary plot (median and 2.5-97.5% band)
# ============================================================
q_lo = np.quantile(np.maximum(Y_NP, 0.01), 0.025, axis=0)
q_md = np.quantile(np.maximum(Y_NP, 0.01), 0.50, axis=0)
q_hi = np.quantile(np.maximum(Y_NP, 0.01), 0.975, axis=0)

plt.figure(figsize=(6,4))
plt.fill_between(TIMES_NP, q_lo, q_hi, alpha=0.25, label="2.5-97.5%")
plt.plot(TIMES_NP, q_md, marker="o", label="Median")
plt.yscale("log")
plt.xlabel("Time (h)")
plt.ylabel("Concentration")
plt.title("Simulated concentrations (population)")
plt.legend()
plt.tight_layout()
fp = os.path.join("figures", "fig_simulated_data_summary.png")
# make y axis from 0.01 to 10
plt.ylim(0.01, 2.0)
plt.savefig(fp, dpi=200)
plt.show()
print("Saved:", fp)

# %% [markdown] cell 6
# ## 2) FOCEI objective and eta-mode solvers
# %% [code] cell 7
# ============================================================
# 2) FOCEI / Laplace objective
# ============================================================
# theta = [log ka, log CL, log V, log wka, log wCL, log wV, log sigma]
# eta   = [eta_ka, eta_CL, eta_V]  (IIV on log-scale)

def stable_logdet_and_chol_3x3(
    H: torch.Tensor,
    jitter: float = 1e-8,
    max_tries: int = 20,
):
    """
    Returns (logdet, Hpd, Lchol, ok, used_shift).
    If stabilization fails, returns ok=False and logdet=+inf.
    """
    Hpd, Lchol, chol_ok, used_shift = _stabilize_hessian_for_cholesky_3x3(
        H, jitter=jitter, max_tries=max_tries
    )

    if (not chol_ok) or (Lchol is None):
        logdet = torch.as_tensor(float("inf"), device=H.device, dtype=H.dtype)
        return logdet, Hpd, None, False, used_shift

    diag = torch.diagonal(Lchol)
    logdet = 2.0 * torch.sum(torch.log(diag))
    ok = bool(torch.isfinite(logdet).detach().cpu().item())
    return logdet, Hpd, Lchol, ok, used_shift


def _stabilize_hessian_for_cholesky_3x3(
    H: torch.Tensor,
    *,
    jitter: float = 1e-8,
    max_tries: int = 8,
    floor_rel: float = 1e-4,
    floor_abs: float = 1e-6,
):
    """
    Robust PD stabilization for 3x3 Hessians without eigendecompositions.

    Returns (Hpd, L, chol_ok, used_shift):
      - Hpd = sym(H) + shift*I
      - L = cholesky(Hpd) (lower)
      - chol_ok indicates success in the ladder
      - used_shift is the final scalar shift

    Key idea:
      1) Symmetrize
      2) Add a curvature floor based on a scale proxy (mean abs diag)
      3) Escalate shift by x10 until cholesky_ex says success
      4) Last-resort Gershgorin-like diagonal dominance shift (still no eigvalsh)
    """
    Hs = 0.5 * (H + H.T)

    # If H has NaNs/Infs, replace them (still allows a "safe" PD matrix)
    if not torch.all(torch.isfinite(Hs)):
        Hs = torch.nan_to_num(Hs, nan=0.0, posinf=0.0, neginf=0.0)

    eye = torch.eye(3, device=Hs.device, dtype=Hs.dtype)

    # Scale proxy for a reasonable "minimum curvature" floor
    diag = torch.diagonal(Hs)
    scale = torch.mean(torch.abs(diag)).detach()
    scale = torch.clamp(scale, min=1.0)

    base_floor = torch.as_tensor(floor_abs, device=Hs.device, dtype=Hs.dtype) + \
                torch.as_tensor(floor_rel, device=Hs.device, dtype=Hs.dtype) * scale

    # Try a simple jitter ladder with cholesky_ex
    shift = (base_floor + torch.as_tensor(jitter, device=Hs.device, dtype=Hs.dtype)).detach()

    for _ in range(int(max_tries)):
        Hpd = Hs + shift * eye
        L, info = torch.linalg.cholesky_ex(Hpd)
        if int(info.detach().cpu().item()) == 0 and torch.all(torch.isfinite(L)):
            return Hpd, L, True, float(shift.detach().cpu().item())
        shift = (10.0 * shift).detach()

    # Last-resort: Gershgorin / diagonal-dominance style shift (still no eig)
    off = torch.sum(torch.abs(Hs), dim=1) - torch.abs(torch.diagonal(Hs))
    need = off - torch.diagonal(Hs) + base_floor
    gshift = torch.clamp(torch.max(need), min=0.0) + torch.as_tensor(jitter, device=Hs.device, dtype=Hs.dtype)
    gshift = gshift.detach()

    Hpd = Hs + gshift * eye
    L, info = torch.linalg.cholesky_ex(Hpd)

    chol_ok = (int(info.detach().cpu().item()) == 0) and torch.all(torch.isfinite(L))
    # Even if chol_ok is False, we return L and let caller decide; but typically it won't be.
    return Hpd, L, bool(chol_ok), float(gshift.detach().cpu().item())




def theta_to_params_np(theta_np: np.ndarray) -> dict:
    theta_np = np.asarray(theta_np, dtype=float)
    logka, logCL, logV, logwka, logwCL, logwV, logsig = theta_np
    ka, CL, V = np.exp([logka, logCL, logV])
    ke = CL / V
    omega_ka, omega_CL, omega_V = np.exp([logwka, logwCL, logwV])
    sigma = float(np.exp(logsig))
    return dict(
        log_ka=logka, log_CL=logCL, log_V=logV,
        ka_pop=ka, CL_pop=CL, V_pop=V, ke_pop=ke,
        omega_ka=omega_ka, omega_CL=omega_CL, omega_V=omega_V,
        sigma=sigma,
    )

def theta_to_params_torch(theta_t: torch.Tensor) -> dict:
    logka, logCL, logV, logwka, logwCL, logwV, logsig = torch.unbind(theta_t)
    ka = torch.exp(logka)
    CL = torch.exp(logCL)
    V  = torch.exp(logV)
    ke = CL / V
    omega_ka = torch.exp(logwka)
    omega_CL = torch.exp(logwCL)
    omega_V  = torch.exp(logwV)
    sigma    = torch.exp(logsig)
    return dict(
        log_ka=logka, log_CL=logCL, log_V=logV,
        ka_pop=ka, CL_pop=CL, V_pop=V, ke_pop=ke,
        omega_ka=omega_ka, omega_CL=omega_CL, omega_V=omega_V,
        sigma=sigma,
    )

# ----------------------------
# Subject negative log joint (up to constants)
# ----------------------------

def subject_nll_np(eta_np: np.ndarray, theta_np: np.ndarray, y_i_np: np.ndarray) -> float:
    """Negative log p(y_i, eta_i | theta) up to constants."""
    p = theta_to_params_np(theta_np)

    eta_np = np.asarray(eta_np, dtype=float)
    logka_i = p["log_ka"] + eta_np[0]
    logCL_i = p["log_CL"] + eta_np[1]
    logV_i  = p["log_V"]  + eta_np[2]

    ka_i = safe_exp_np(logka_i)
    CL_i = safe_exp_np(logCL_i)
    V_i  = safe_exp_np(logV_i)

    c = conc_one_comp_sc_np(TIMES_NP, ka_i, CL_i, V_i, dose=DOSE)
    resid = y_i_np - c

    sig = p["sigma"]
    nll_like = 0.5 * np.sum((resid / sig) ** 2) + y_i_np.size * math.log(sig)

    # diagonal Omega prior
    w = np.array([p["omega_ka"], p["omega_CL"], p["omega_V"]], dtype=float)
    nll_prior = 0.5 * np.sum((eta_np / w) ** 2) + np.sum(np.log(w))

    val = nll_like + nll_prior
    if not np.isfinite(val):
        return float("inf")
    return float(val)

def subject_nll_torch(eta_t: torch.Tensor, theta_t: torch.Tensor, y_i_t: torch.Tensor) -> torch.Tensor:
    p = theta_to_params_torch(theta_t)

    logka_i = p["log_ka"] + eta_t[0]
    logCL_i = p["log_CL"] + eta_t[1]
    logV_i  = p["log_V"]  + eta_t[2]

    ka_i = safe_exp_torch(logka_i)
    CL_i = safe_exp_torch(logCL_i)
    V_i  = safe_exp_torch(logV_i)

    c = conc_one_comp_sc_torch(TIMES_T, ka_i, CL_i, V_i, dose=DOSE)
    resid = y_i_t - c

    sig = p["sigma"]
    nll_like = 0.5 * torch.sum((resid / sig) ** 2) + y_i_t.numel() * torch.log(sig)

    w = torch.stack([p["omega_ka"], p["omega_CL"], p["omega_V"]])
    nll_prior = 0.5 * torch.sum((eta_t / w) ** 2) + torch.sum(torch.log(w))

    return nll_like + nll_prior

# ----------------------------
# Torch Hessian wrt eta (3x3)
# ----------------------------

def hessian_3x3_torch(scalar: torch.Tensor, vec: torch.Tensor, create_graph: bool) -> torch.Tensor:
    """H = d^2 scalar / d vec^2, for vec in R^3."""
    g = torch.autograd.grad(scalar, vec, create_graph=True, retain_graph=True)[0]
    rows = []
    for k in range(3):
        row = torch.autograd.grad(g[k], vec, create_graph=create_graph, retain_graph=True)[0]
        rows.append(row)
    return torch.stack(rows, dim=0)

def grad_hess_3x3_torch(scalar: torch.Tensor, vec: torch.Tensor, create_graph_hess: bool):
    """Return (g, H) where g = d scalar / d vec and H = d^2 scalar / d vec^2 for vec in R^3.

    Important autograd detail:
      - To compute a Hessian via reverse-mode autograd, the *first* derivative must be created with
        create_graph=True; otherwise the gradient tensor has no grad_fn and cannot be differentiated.
      - We set retain_graph=True so we can take grads of each g[k] without triggering
        "Trying to backward through the graph a second time".
    """
    # First derivative: always create_graph=True so second derivatives exist
    g = torch.autograd.grad(scalar, vec, create_graph=True, retain_graph=True)[0]

    rows = []
    for k in range(3):
        row = torch.autograd.grad(g[k], vec, create_graph=create_graph_hess, retain_graph=True)[0]
        rows.append(row)
    H = torch.stack(rows, dim=0)

    if not create_graph_hess:
        return g.detach(), H.detach()
    return g, H

# ----------------------------
# Safe logdet for PD matrices
# ----------------------------

#def safe_logdet_pd_torch(H: torch.Tensor, jitter0: float = 1e-8, max_tries: int = 8):
#    eye = torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
#    jitter = jitter0
#    for _ in range(max_tries):
#        Hj = H + jitter * eye
#        sign, ld = torch.linalg.slogdet(Hj)
#        if torch.isfinite(ld) and float(sign.detach().cpu()) > 0:
#            return ld, True
#        jitter *= 10.0
#    return torch.as_tensor(float("inf"), device=H.device, dtype=H.dtype), False

def safe_logdet_pd_np(H: np.ndarray, jitter0: float = 1e-8, max_tries: int = 8):
    H = np.asarray(H, dtype=float)
    eye = np.eye(H.shape[0], dtype=float)
    jitter = jitter0
    for _ in range(max_tries):
        Hj = H + jitter * eye
        sign, ld = np.linalg.slogdet(Hj)
        if np.isfinite(ld) and sign > 0:
            return float(ld), True
        jitter *= 10.0
    return float("inf"), False

# ----------------------------
# Inner solve: eta_mode via autodiff Newton (used for STOP/FULL)
# ----------------------------
def eta_mode_newton_ad(
    theta_for_mode: torch.Tensor,
    y_i_t: torch.Tensor,
    *,
    eta_init=None,
    maxiter: int = 6,
    tol: float = 1e-6,
    base_damp: float = 1e-6,
    create_graph: bool = False,
) -> torch.Tensor:
    """
+    Shared damped Newton solver for eta<U+0302>_i(theta).
+
+    Fidelity goals:
+      - STOP and FULL-unroll use the same numeric solver.
+      - FULL-unroll differentiates through the implemented K Newton steps:
+          * H depends on (eta, theta)
+          * the linear solve depends on H
+        so we do NOT detach H or the solve matrix when create_graph=True.
+    """
    if eta_init is None:
        eta = torch.zeros(3, device=DEVICE, dtype=theta_for_mode.dtype)
    else:
        eta = torch.as_tensor(eta_init, device=DEVICE, dtype=theta_for_mode.dtype)
    eta = eta.detach().clone().requires_grad_(True)

    eye = torch.eye(3, device=DEVICE, dtype=theta_for_mode.dtype)

    for _ in range(int(maxiter)):
        nll = subject_nll_torch(eta, theta_for_mode, y_i_t)
        g = torch.autograd.grad(nll, eta, create_graph=True, retain_graph=True)[0]

        if float(torch.linalg.norm(g).detach().cpu()) < float(tol):
            break

        # FULL-unroll needs a differentiable Hessian when create_graph=True
        H = hessian_3x3_torch(nll, eta, create_graph=create_graph)
        H = 0.5 * (H + H.T)

        damp = float(base_damp)
        step = None
        for _try in range(6):
            try:
                step = torch.linalg.solve(H + damp * eye, g)
                # --- safety: cap Newton step norm to prevent extreme eta excursions ---
                MAX_STEP_NORM = 10.0
                step_norm = torch.linalg.norm(step)
                if torch.isfinite(step_norm) and step_norm > MAX_STEP_NORM:
                    step = step * (MAX_STEP_NORM / (step_norm + 1e-12))
            except RuntimeError:
                step = None
            if step is not None and torch.all(torch.isfinite(step)):
                break
            damp *= 10.0

        if step is None or (not torch.all(torch.isfinite(step))):
            break

        # Backtracking line search (branch decisions are not differentiated)
        alpha = 1.0
        eta_new = eta - alpha * step
        nll_new = subject_nll_torch(eta_new, theta_for_mode, y_i_t)
        for _ls in range(8):
            if torch.isfinite(nll_new) and (float(nll_new.detach().cpu()) <= float(nll.detach().cpu())):
                break
            alpha *= 0.5
            eta_new = eta - alpha * step
            nll_new = subject_nll_torch(eta_new, theta_for_mode, y_i_t)

        if create_graph:
            eta = eta_new
        else:
            eta = eta_new.detach().requires_grad_(True)

    return eta if create_graph else eta.detach()



# ----------------------------
# Inner solve: eta_mode via finite-difference Newton (NONMEM-like)
# ----------------------------

def fd_grad_hess_eta(eta: np.ndarray, theta_np: np.ndarray, y_i_np: np.ndarray,
                     eps: float = 1e-4):
    """Central-difference gradient and Hessian of subject_nll_np wrt eta (dim=3)."""
    eta = np.asarray(eta, dtype=float)
    f0 = subject_nll_np(eta, theta_np, y_i_np)
    if not np.isfinite(f0):
        return np.full(3, np.nan), np.full((3,3), np.nan), f0

    # Precompute +/- along axes for reuse
    f_plus = np.zeros(3, dtype=float)
    f_minus = np.zeros(3, dtype=float)
    for j in range(3):
        d = np.zeros(3, dtype=float); d[j] = eps
        f_plus[j]  = subject_nll_np(eta + d, theta_np, y_i_np)
        f_minus[j] = subject_nll_np(eta - d, theta_np, y_i_np)

    g = (f_plus - f_minus) / (2.0 * eps)

    H = np.zeros((3,3), dtype=float)
    # diagonals
    for j in range(3):
        H[j,j] = (f_plus[j] - 2.0*f0 + f_minus[j]) / (eps**2)

    # off-diagonals
    for j in range(3):
        for k in range(j+1, 3):
            dj = np.zeros(3, dtype=float); dj[j] = eps
            dk = np.zeros(3, dtype=float); dk[k] = eps
            f_pp = subject_nll_np(eta + dj + dk, theta_np, y_i_np)
            f_pm = subject_nll_np(eta + dj - dk, theta_np, y_i_np)
            f_mp = subject_nll_np(eta - dj + dk, theta_np, y_i_np)
            f_mm = subject_nll_np(eta - dj - dk, theta_np, y_i_np)
            H_jk = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps**2)
            H[j,k] = H_jk
            H[k,j] = H_jk

    return g, H, f0

def eta_mode_newton_fd(theta_np: np.ndarray, y_i_np: np.ndarray, eta_init=None,
                       maxiter: int = 6, tol: float = 1e-6, eps: float = 1e-4):
    """Damped Newton in eta using finite-difference gradients/Hessians."""
    if eta_init is None:
        eta = np.zeros(3, dtype=float)
    else:
        eta = np.asarray(eta_init, dtype=float).copy()

    base_damp = 1e-4
    f_best = subject_nll_np(eta, theta_np, y_i_np)
    if not np.isfinite(f_best):
        # fall back to zero init
        eta = np.zeros(3, dtype=float)
        f_best = subject_nll_np(eta, theta_np, y_i_np)

    H_last = None
    for _ in range(int(maxiter)):
        g, H, f0 = fd_grad_hess_eta(eta, theta_np, y_i_np, eps=eps)
        H_last = H

        if (not np.all(np.isfinite(g))) or (not np.all(np.isfinite(H))) or (not np.isfinite(f0)):
            break

        # Symmetrize Hessian to reduce FD noise/asymmetry
        H = 0.5 * (H + H.T)
        
        if np.linalg.norm(g) < tol:
            f_best = f0
            break

        # damped solve
        # print an 'a'
        #print("a")
        damp = base_damp
        step = None
        eye = np.eye(3, dtype=float)
        for _try in range(8):
            try:
                step = np.linalg.solve(H + damp * eye, g)
                # --- safety: cap Newton step norm to prevent extreme eta excursions ---
                MAX_STEP_NORM = 10.0
                step_norm = np.linalg.norm(step)
                if np.isfinite(step_norm) and step_norm > MAX_STEP_NORM:
                    step = step * (MAX_STEP_NORM / (step_norm + 1e-12))
            except np.linalg.LinAlgError:
                step = None
            if step is not None and np.all(np.isfinite(step)):
                break
            damp *= 10.0

        if step is None or (not np.all(np.isfinite(step))):
            break

        # line search
        alpha = 1.0
        accepted = False
        for _ls in range(10):
            eta_new = eta - alpha * step
            f_new = subject_nll_np(eta_new, theta_np, y_i_np)
            if np.isfinite(f_new) and (f_new <= f0):
                eta = eta_new
                f_best = f_new
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            # no progress
            f_best = f0
            break

        if np.linalg.norm(alpha * step) < tol:
            break

    # Compute final Hessian at eta (for Laplace logdet)
    g, H, f0 = fd_grad_hess_eta(eta, theta_np, y_i_np, eps=eps)
    if np.all(np.isfinite(H)):
        H_last = H
        f_best = f0

    return eta, H_last, f_best

# ----------------------------
# Laplace contribution: one subject
# ----------------------------

def _penalty_torch(theta_t: torch.Tensor) -> torch.Tensor:
    # connect to theta_t so backward() always works
    # return torch.as_tensor(BIG, device=theta_t.device, dtype=theta_t.dtype) * (1.0 + 0.0 * theta_t.sum())
    return torch.as_tensor(2.0 * BIG, device=theta_t.device, dtype=theta_t.dtype) * (1.0 + 0.0 * theta_t.sum())

def laplace_contrib_ad(theta_t: torch.Tensor, y_i_t: torch.Tensor,
                       mode: str = "stop", eta_init=None,
                       need_theta_grad: bool = False, maxiter_eta: int = 6):
    """(STOP/FULL) One-subject Laplace contribution as torch tensor."""
    if mode == "stop":
        theta_for_mode = theta_t.detach()
        eta_hat = eta_mode_newton_ad(theta_for_mode, y_i_t, eta_init=eta_init, create_graph=False, maxiter=maxiter_eta).detach()
        eta_eval = eta_hat.detach().requires_grad_(True)
    elif mode == "full":
        theta_for_mode = theta_t
        eta_hat = eta_mode_newton_ad(theta_for_mode, y_i_t, eta_init=eta_init, create_graph=True, maxiter=maxiter_eta)
        eta_eval = eta_hat
    else:
        raise ValueError(mode)

    nll = subject_nll_torch(eta_eval, theta_t, y_i_t)
    H = hessian_3x3_torch(nll, eta_eval, create_graph=(need_theta_grad or (mode == "full")))
    #logdet, ok = safe_logdet_pd_torch(H)
    #logdet, Hpd, Lchol, ok, used = stable_logdet_and_chol_3x3(H, jitter=1e-8)
    #Hpd, Lchol, _, _ = _stabilize_hessian_for_cholesky_3x3(H, jitter=float(1e-8))
    logdet, Hpd, Lchol, ok, used = stable_logdet_and_chol_3x3(H, jitter=1e-8)
    # step = torch.cholesky_solve(g.unsqueeze(-1), Lchol).squeeze(-1)


    if (not ok) or (not torch.isfinite(nll)) or (not torch.isfinite(logdet)):
        return _penalty_torch(theta_t), eta_hat.detach()

    return nll + 0.5 * logdet, eta_hat.detach()


def laplace_contrib_fd(
    theta_np: np.ndarray,
    y_i_np: np.ndarray,
    *,
    eta_init=None,
    maxiter_eta: int = 6,
    eps_eta: float = 1e-4,
    logdet_jitter: float = 1e-8,
):
    """
    FD baseline:
      - inner eta<U+0302> uses FD derivatives (legacy-like),
      - BUT the evaluated Laplace objective uses the SAME stabilized logdet(H) policy as AD methods,
        so all strategies compare on the same implemented objective.
    """
    eta_hat, _H_hat_unused, _nll_unused = eta_mode_newton_fd(
        theta_np, y_i_np, eta_init=eta_init, maxiter=maxiter_eta, eps=eps_eta
    )

    # Recompute h_i and H_i exactly (via torch autograd) at (theta, eta_hat) for curvature consistency
    theta_t = torch.as_tensor(theta_np, device=DEVICE, dtype=torch.float64)
    y_t = torch.as_tensor(y_i_np, device=DEVICE, dtype=torch.float64)
    eta_t = torch.as_tensor(eta_hat, device=DEVICE, dtype=torch.float64).detach().clone().requires_grad_(True)

    nll = subject_nll_torch(eta_t, theta_t, y_t)
    if not torch.isfinite(nll):
        return float(BIG), eta_hat

    H = hessian_3x3_torch(nll, eta_t, create_graph=False)
    logdet, _Hpd, _Lchol, ok, _used = stable_logdet_and_chol_3x3(H, jitter=float(logdet_jitter))
    if (not ok) or (not torch.isfinite(logdet)):
        return float(BIG), eta_hat

    val = nll + 0.5 * logdet
    if not torch.isfinite(val):
        return float(BIG), eta_hat
    return float(val.detach().cpu().numpy()), eta_hat



# ----------------------------
# Full FOCEI objective
# ----------------------------

def focei_objective_torch(theta_t: torch.Tensor, mode: str,
                          eta_cache: np.ndarray, update_cache: bool,
                          need_theta_grad: bool, maxiter_eta: int = 6) -> torch.Tensor:
    total = torch.zeros((), device=DEVICE, dtype=theta_t.dtype)
    eta_new = np.zeros_like(eta_cache)

    for i in range(Y_T.shape[0]):
        eta_init = eta_cache[i] if eta_cache is not None else None
        c, eta_hat = laplace_contrib_ad(theta_t, Y_T[i], mode=mode, eta_init=eta_init,
                                        need_theta_grad=need_theta_grad, maxiter_eta=maxiter_eta)
        total = total + c
        eta_new[i] = eta_hat.detach().cpu().numpy()

    if update_cache and (eta_cache is not None):
        eta_cache[:] = eta_new

    return 2.0*total

def focei_objective_fd(theta_np: np.ndarray, eta_cache: np.ndarray,
                       update_cache: bool, maxiter_eta: int = 6, eps_eta: float = 1e-4):
    total = 0.0
    eta_new = np.zeros_like(eta_cache)

    for i in range(Y_NP.shape[0]):
        eta_init = eta_cache[i] if eta_cache is not None else None
        c, eta_hat = laplace_contrib_fd(theta_np, Y_NP[i], eta_init=eta_init,
                                        maxiter_eta=maxiter_eta, eps_eta=eps_eta)
        total += c
        eta_new[i] = eta_hat

    if update_cache and (eta_cache is not None):
        eta_cache[:] = eta_new

    return 2.0*float(total)

# ----------------------------
# Convenience wrappers used throughout the notebook
# ----------------------------

def focei_obj_only(theta_np: np.ndarray, mode_for_eta: str = "stop", maxiter_eta: int = 6) -> float:
    """Objective-only evaluation with a fresh cache (useful for plotting)."""
    eta_cache = np.zeros((Y_NP.shape[0], 3), dtype=float)

    if mode_for_eta in ["stop", "full"]:
        theta_t = torch.tensor(theta_np, device=DEVICE, requires_grad=False)
        val = focei_objective_torch(theta_t, mode=mode_for_eta, eta_cache=eta_cache,
                                    update_cache=True, need_theta_grad=False, maxiter_eta=maxiter_eta)
        f = float(val.detach().cpu().numpy())
        return f if np.isfinite(f) else float(BIG)

    elif mode_for_eta == "fd":
        f = focei_objective_fd(np.asarray(theta_np, dtype=float), eta_cache=eta_cache,
                               update_cache=True, maxiter_eta=maxiter_eta, eps_eta=1e-4)
        return f if np.isfinite(f) else float(BIG)

    else:
        raise ValueError(mode_for_eta)

# %% [code] cell 8
# ============================================================
# 2b) FOCEI FULL-IMPLICIT (poppkpd-style) gradient
# ============================================================
# Adds a new outer-gradient method without unrolling the Newton eta steps.
#
# FULL-IMPLICIT recipe:
#   L_i(theta) = h_i(theta, eta_hat(theta)) + 0.5 * logdet(H_i)
#   where eta_hat solves:  d h_i / d eta = 0
#
# Outer gradient:
#   dL/dtheta = dh/dtheta + 0.5*( d logdet/dtheta + (d logdet/deta)^T * d eta_hat/dtheta )
#   d eta_hat/dtheta = - H^{-1} * d^2 h / d eta d theta
#
# We compute the needed derivatives with torch.autograd, but we do NOT backprop
# through the Newton iterations (implicit differentiation).

FULL_IMPLICIT_DEBUG = False
FULL_IMPLICIT_DEBUG_MAX_EVALS = 999999           # print full trace for the first N outer evaluations
FULL_IMPLICIT_DEBUG_N_SUBJ = 3              # per-subject prints for the first N subjects
FULL_IMPLICIT_DEBUG_PRINT_ETA_NEWTON = True
FULL_IMPLICIT_DEBUG_PRINT_PER_SUBJ = True
FULL_IMPLICIT_DEBUG_PRINT_SUMMARY = True
FULL_IMPLICIT_DEBUG_PRINT_IF_PENALTY = True
FULL_IMPLICIT_DEBUG_PRINT_GRAD_PARTS = True

# For extra safety during debugging
FULL_IMPLICIT_USE_PD_SOLVE_FOR_MODE = True   # mode Newton uses PD-stabilized Cholesky solves
FULL_IMPLICIT_MODE_TOL = 1e-8
FULL_IMPLICIT_MODE_MAXITER = 20
FULL_IMPLICIT_MODE_MAX_STEP_NORM = 10
FULL_IMPLICIT_MODE_DAMPING = 1.0

_full_impl_eval_counter = 0

def _fullimp_should_print() -> bool:
    return bool(FULL_IMPLICIT_DEBUG) and (_full_impl_eval_counter <= int(FULL_IMPLICIT_DEBUG_MAX_EVALS))

def _eta_init_to_torch(eta_init, *, dtype, device):
    """Robust conversion of eta_init (None / np.ndarray / torch.Tensor) -> torch.Tensor (no grad)."""
    if eta_init is None:
        return torch.zeros(3, device=device, dtype=dtype)
    if torch.is_tensor(eta_init):
        return eta_init.detach().clone().to(device=device, dtype=dtype)
    return torch.as_tensor(eta_init, device=device, dtype=dtype).detach().clone()




def eta_mode_newton_stop_pd(theta_det: torch.Tensor,
                            y_i_t: torch.Tensor,
                            *,
                            eta_init=None,
                            maxiter: int = 20,
                            tol: float = 1e-8,
                            damping: float = 1.0,
                            logdet_jitter: float = 1e-8,
                            max_step_norm: float = 10.0,
                            debug: bool = False,
                            debug_prefix: str = "") -> torch.Tensor:
    """
    Damped Newton solve for eta_hat using DETACHED theta (STOP-like).
    Uses PD-stabilized Hessian solves to reduce blow-ups when far from the mode.

    Returns a DETACHED eta_hat tensor (no grad).
    """
    eta = _eta_init_to_torch(eta_init, dtype=theta_det.dtype, device=DEVICE).requires_grad_(True)

    for it in range(int(maxiter)):
        nll = subject_nll_torch(eta, theta_det, y_i_t)

        if not torch.isfinite(nll):
            if debug:
                print(f"{debug_prefix}etaNewton it={it:02d} nll=nan/inf -> break")
            break

        # detached g,H (create_graph_hess=False)
        g, H = grad_hess_3x3_torch(nll, eta, create_graph_hess=False)
        gnorm = float(torch.linalg.norm(g).detach().cpu())

        if debug:
            try:
                eigs = torch.linalg.eigvalsh(0.5 * (H + H.T)).detach().cpu().numpy()
                min_eig = float(np.min(eigs))
            except Exception:
                min_eig = float("nan")
            print(f"{debug_prefix}etaNewton it={it:02d} nll={float(nll.detach().cpu()):.6f} |g|={gnorm:.3e} minEig(H)={min_eig:.3e}")

        if (not torch.all(torch.isfinite(g))) or (not torch.all(torch.isfinite(H))):
            if debug:
                print(f"{debug_prefix}  non-finite g/H -> break")
            break

        if gnorm < float(tol):
            break

        # PD stabilize for step
        Hpd, L, chol_ok, used = _stabilize_hessian_for_cholesky_3x3(H, jitter=float(logdet_jitter))

        step = torch.cholesky_solve(g.unsqueeze(-1), L).squeeze(-1)

        # clip huge steps (safety)
        step_norm = float(torch.linalg.norm(step).detach().cpu())
        if torch.isfinite(torch.as_tensor(step_norm)) and (step_norm > float(max_step_norm)):
            step = step * (float(max_step_norm) / (step_norm + 1e-12))
            step_norm = float(torch.linalg.norm(step).detach().cpu())

        # backtracking line search on nll
        alpha = float(damping)
        nll0 = float(nll.detach().cpu())

        eta_new = eta - alpha * step
        nll_new = subject_nll_torch(eta_new, theta_det, y_i_t)
        nll_new_f = float(nll_new.detach().cpu()) if torch.isfinite(nll_new) else float("inf")

        ls_it = 0
        while (ls_it < 10) and ((not np.isfinite(nll_new_f)) or (nll_new_f > nll0)):
            alpha *= 0.5
            eta_new = eta - alpha * step
            nll_new = subject_nll_torch(eta_new, theta_det, y_i_t)
            nll_new_f = float(nll_new.detach().cpu()) if torch.isfinite(nll_new) else float("inf")
            ls_it += 1

        if debug:
            print(f"{debug_prefix}  stepNorm={step_norm:.3e} alpha={alpha:.3e} nllNew={nll_new_f:.6f} (chol={'ok' if chol_ok else 'shift'} used={used:.1e})")

        eta = eta_new.detach().requires_grad_(True)

        if (alpha * step_norm) < float(tol):
            break

    return eta.detach()

 

def _penalty_fg(theta_np: np.ndarray, *, weight: float = 100.0) -> tuple:
    """
    Return a smooth penalty (f,g) so L-BFGS-B does NOT see g=0 at a huge f.
    """
    th = np.asarray(theta_np, float)
    f0 = float(BIG) + 0.5 * float(weight) * float(np.dot(th, th))
    g0 = float(weight) * th
    f = 2.0 * f0
    g = 2.0 * g0
    return f, g

def laplace_contrib_full_implicit(theta_t: torch.Tensor,
                                  y_i_t: torch.Tensor,
                                  *,
                                  eta_init=None,
                                  maxiter_eta: int = 20,
                                  tol_eta: float = 1e-8,
                                  damping_eta: float = 1.0,
                                  logdet_jitter: float = 1e-8,
                                  max_step_norm: float = 10.0,
                                  debug: bool = False,
                                  debug_prefix: str = ""):
    """
    One-subject Laplace contribution and FULL-IMPLICIT outer gradient term.

    Returns:
      f_i_float, g_i_torch(p,), eta_hat_np(3,)
    """
    # 1) Solve mode with DETACHED theta (STOP-like)
    theta_det = theta_t.detach()

    # Use the SAME inner solver as STOP (numeric equivalence across methods)
    eta_hat = eta_mode_newton_ad(
        theta_det,
        y_i_t,
        eta_init=eta_init,
        maxiter=maxiter_eta,
        tol=tol_eta,
        base_damp=1e-6,
        create_graph=False,
    )

    eta_hat_np = eta_hat.detach().cpu().numpy().astype(float)

    # 2) Evaluate objective and derivatives at (theta, eta_hat)
    eta_leaf = eta_hat.detach().clone().requires_grad_(True)

    nll = subject_nll_torch(eta_leaf, theta_t, y_i_t)

    # g_eta and H need create_graph for implicit terms
    g_eta, H = grad_hess_3x3_torch(nll, eta_leaf, create_graph_hess=True)

    # Stabilize H once (reuse for logdet and solves)
    Hpd, Lchol, chol_ok, used = _stabilize_hessian_for_cholesky_3x3(
        H, jitter=float(logdet_jitter), max_tries=20
    )

    if (not chol_ok) or (Lchol is None):
        if debug and FULL_IMPLICIT_DEBUG_PRINT_IF_PENALTY:
            print(f"{debug_prefix}  PENALTY: Cholesky failed after ladder. used_shift={used}")
        f_i, g_i = _penalty_fg(theta_t.detach().cpu().numpy(), weight=100.0)
        g_i_t = torch.as_tensor(g_i, device=theta_t.device, dtype=theta_t.dtype)
        return float(f_i), g_i_t, eta_hat_np

    diag = torch.diagonal(Lchol)
    logdet = 2.0 * torch.sum(torch.log(diag))
    ok = bool(torch.isfinite(nll).detach().cpu().item() and torch.isfinite(logdet).detach().cpu().item())



    if (not ok) or (not torch.isfinite(nll)) or (not torch.isfinite(logdet)):
        # Hard failure: we could not even get a finite objective contribution.
        # Return a large constant contribution and ZERO gradient from this subject.
        # (Let other subjects drive the optimization.)
        f_bad = float(BIG)
        g_bad = torch.zeros_like(theta_t).detach()
        return f_bad, g_bad, eta_hat_np


    f_i = nll + 0.5 * logdet

    # Envelope term for nll (valid at the true mode). We still print |g_eta| to check convergence.
    dh_dtheta = torch.autograd.grad(nll, theta_t, retain_graph=True, create_graph=False)[0]

    # logdet partials
    dld_dtheta = torch.autograd.grad(logdet, theta_t, retain_graph=True, create_graph=False)[0]
    dld_deta   = torch.autograd.grad(logdet, eta_leaf, retain_graph=True, create_graph=False)[0]

    # Cross derivative J = d/dtheta (d nll / d eta)  => shape (3,p)
    J_rows = []
    for k in range(3):
        Jk = torch.autograd.grad(g_eta[k], theta_t, retain_graph=True, create_graph=False)[0]
        J_rows.append(Jk)
    J = torch.stack(J_rows, dim=0)  # (3,p)

    # Default: STOP gradient components (always available if objective is finite)
    g_theta = dh_dtheta + 0.5 * dld_dtheta

    try:
        V = -torch.cholesky_solve(J, Lchol)          # (3,p) using stabilized Hpd
        chain = dld_deta @ V                         # (p,)

        # Guard: if chain produced NaNs/Infs or is absurdly large, drop it
        if torch.all(torch.isfinite(chain)):
            # Optional magnitude guard (prevents one subject from exploding the whole gradient)
            CHAIN_MAX = 1e6
            if float(torch.linalg.norm(chain).detach().cpu().item()) <= CHAIN_MAX:
                g_theta = g_theta + 0.5 * chain
            #if float(torch.linalg.norm(chain)) <= 1e6:   # guard
            #    g_theta = g_theta + 0.5 * chain

    except RuntimeError:
        # If the implicit correction fails numerically, keep STOP gradient for this subject.
        pass


    if bool(debug and FULL_IMPLICIT_DEBUG_PRINT_GRAD_PARTS):
        ge_norm = float(torch.linalg.norm(g_eta.detach()).cpu())
        print(f"{debug_prefix}  f_i={float(f_i.detach().cpu()):.6f} |g_eta|={ge_norm:.3e} (chol={'ok' if chol_ok else 'shift'} used={used:.1e})")
        print(f"{debug_prefix}  |dh_dtheta|={float(torch.linalg.norm(dh_dtheta.detach()).cpu()):.3e} "
              f"|dld_dtheta|={float(torch.linalg.norm(dld_dtheta.detach()).cpu()):.3e} "
              f"|chain|={float(torch.linalg.norm(chain.detach()).cpu()):.3e} "
              f"|g_theta|={float(torch.linalg.norm(g_theta.detach()).cpu()):.3e}")

    return float(f_i.detach().cpu().numpy()), g_theta.detach(), eta_hat_np

def focei_objective_full_implicit_eval(theta_np: np.ndarray,
                                       *,
                                       eta_cache: np.ndarray,
                                       update_cache: bool = True,
                                       maxiter_eta: int = 20,
                                       tol_eta: float = 1e-8,
                                       damping_eta: float = 1.0,
                                       logdet_jitter: float = 1e-8,
                                       max_step_norm: float = 10.0):
    """
    Full FOCEI objective and FULL-IMPLICIT gradient w.r.t theta.

    Returns: (f_float, g_np(p,))
    """
    global _full_impl_eval_counter
    _full_impl_eval_counter += 1
    do_print = _fullimp_should_print()

    theta_np = np.asarray(theta_np, float)
    theta_t = torch.as_tensor(theta_np, device=DEVICE, dtype=torch.float64).detach().clone().requires_grad_(True)

    n_subj = Y_T.shape[0]
    p = theta_t.numel()
    total_f = 0.0
    total_g = torch.zeros(p, device=DEVICE, dtype=theta_t.dtype)
    eta_new = np.zeros_like(eta_cache)

    if do_print:
        print("=" * 80)
        print(f"[FULL-IMPLICIT] eval {_full_impl_eval_counter}  theta={theta_np}")

    # some summary stats
    n_pen = 0

    for i in range(int(n_subj)):
        eta_init = None if eta_cache is None else eta_cache[i]
        debug_i = bool(do_print and FULL_IMPLICIT_DEBUG_PRINT_PER_SUBJ and (i < int(FULL_IMPLICIT_DEBUG_N_SUBJ)))
        prefix = f"[FULL-IMPLICIT eval{_full_impl_eval_counter} subj{i:03d}] "

        f_i, g_i_t, eta_hat_np = laplace_contrib_full_implicit(
            theta_t,
            Y_T[i],
            eta_init=eta_init,
            maxiter_eta=maxiter_eta,
            tol_eta=tol_eta,
            damping_eta=damping_eta,
            logdet_jitter=logdet_jitter,
            max_step_norm=max_step_norm,
            debug=debug_i,
            debug_prefix=prefix,
        )

        #if (not np.isfinite(f_i)) or (not torch.all(torch.isfinite(g_i_t))):
        #    n_pen += 1
        if (not np.isfinite(f_i)) or is_penalty(float(f_i)) or (not torch.all(torch.isfinite(g_i_t))):
            n_pen += 1


        total_f += float(f_i)
        total_g = total_g + g_i_t
        eta_new[i] = eta_hat_np

        if debug_i:
            print(f"{prefix}eta_hat={eta_hat_np}")

    if update_cache and (eta_cache is not None):
        eta_cache[:] = eta_new

    g_np = total_g.detach().cpu().numpy().astype(float)

    if do_print and FULL_IMPLICIT_DEBUG_PRINT_SUMMARY:
        print(f"[FULL-IMPLICIT] eval {_full_impl_eval_counter} total_f={total_f:.6f} |g|={float(np.linalg.norm(g_np)):.3e} penalties={n_pen}")

    # If we hit penalties, return a smooth penalty so L-BFGS-B doesn't "converge" with g=0.
    #if n_pen > 0 or (not np.isfinite(total_f)) or (not np.all(np.isfinite(g_np))):
    #    f_pen, g_pen = _penalty_fg(theta_np, weight=100.0)
    #    if do_print:
    #        print(f"[FULL-IMPLICIT] returning penalty f={f_pen:.3e} |g|={float(np.linalg.norm(g_pen)):.3e}")
    #    return float(f_pen), np.asarray(g_pen, float)
    # If objective or gradient is unusable, then and only then return a global penalty.
    #if (not np.isfinite(total_f)) or (not np.all(np.isfinite(g_np))) or is_penalty(float(total_f)):
    #    f_pen, g_pen = _penalty_fg(theta_np, weight=100.0)
    #    print(f"[FULL-IMPLICIT] returning GLOBAL penalty f={f_pen:.3e} |g|={float(np.linalg.norm(g_pen)):.3e}")
    #    return float(f_pen), np.asarray(g_pen, float)

    # Global bailout ONLY if the whole evaluation is numerically unusable
    if (not np.isfinite(total_f)) or (not np.all(np.isfinite(g_np))):
        f_pen, g_pen = _penalty_fg(theta_np, weight=100.0)
        print(f"[FULL-IMPLICIT] returning GLOBAL penalty (non-finite) f={f_pen:.3e} |g|={float(np.linalg.norm(g_pen)):.3e}")
        return float(f_pen), np.asarray(g_pen, float)

    # Even if some subjects returned f_bad=BIG (finite), keep going.
    # That lets the optimizer move into regions where fewer subjects fail.
    return float(2.0 * total_f), np.asarray(2.0 * g_np, float)

    # Otherwise: even if some subjects used STOP fallback, return what we have.
    #return float(total_f), np.asarray(g_np, float)



    return float(2.0 * total_f), np.asarray(2.0 * g_np, float)

# %% [markdown] cell 9
# ## 3) Outer optimization wrappers
# %% [code] cell 10
# ============================================================
# 3) Outer optimization wrappers
# ============================================================

# Parameter bounds in log-space (same as earlier notebooks)
BOUNDS = [
    (math.log(1e-3), math.log(2.0)),    # ka
    (math.log(0.05), math.log(30.0)),   # CL
    (math.log(1.0),  math.log(300.0)),  # V
    (math.log(0.01), math.log(1.0)),    # omega_ka
    (math.log(0.01), math.log(1.0)),    # omega_CL
    (math.log(0.01), math.log(1.0)),    # omega_V
    (math.log(0.01), math.log(5.0)),    # sigma
]

def clip_to_bounds(theta: np.ndarray, bounds=BOUNDS) -> np.ndarray:
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return np.minimum(np.maximum(np.asarray(theta, dtype=float), lo), hi)

def theta_to_record(theta_np: np.ndarray) -> dict:
    p = theta_to_params_np(theta_np)
    return dict(
        log_ka=float(p["log_ka"]), log_V=float(p["log_V"]),
        ka_pop=float(p["ka_pop"]), CL_pop=float(p["CL_pop"]), V_pop=float(p["V_pop"]), ke_pop=float(p["ke_pop"]),
        omega_ka=float(p["omega_ka"]), omega_CL=float(p["omega_CL"]), omega_V=float(p["omega_V"]),
        sigma=float(p["sigma"]),
    )

# Method labels for plots/tables
#METHODS = ["AD-full-implicit", "FD", "AD-stop", "AD-full"]
#METHOD_LABEL = {"AD-full-implicit": "FOCEI-AD-full-implicit", "FD": "FOCEI-FD", "AD-stop": "FOCEI-AD-stop", "AD-full": "FOCEI-AD-full"}

# ----------------------------
# AD fun+jac (STOP/FULL)
# ----------------------------




def make_funjac_ad(mode: str, maxiter_eta: int = 6):
    eta_cache = np.zeros((Y_NP.shape[0], 3), dtype=float)

    def funjac(theta_np: np.ndarray):
        theta_np = clip_to_bounds(theta_np, BOUNDS)
        theta_t = torch.tensor(theta_np, device=DEVICE, requires_grad=True)

        val = focei_objective_torch(theta_t, mode=mode, eta_cache=eta_cache,
                                    update_cache=True, need_theta_grad=True,
                                    maxiter_eta=maxiter_eta)
        f = float(val.detach().cpu().numpy())

        # backward always works because penalties are connected to theta_t
        val.backward()
        g = theta_t.grad.detach().cpu().numpy().astype(float)
        g = np.where(np.isfinite(g), g, 0.0)

        return f, g

    return funjac

# ----------------------------
# FD fun+jac (outer FD + inner FD eta modes)
# ----------------------------

def make_funjac_fd(bounds=BOUNDS, maxiter_eta: int = 6, eps_eta: float = 1e-4,
                   eps_theta: float = 1e-3, scheme: str = "central"):
    """
    NONMEM-like finite-difference outer gradients *and* finite-difference eta-mode inner solves.

    scheme:
      - 'forward'  : g_j = (f(theta + h e_j) - f(theta)) / h
      - 'central'  : g_j = (f(theta + h e_j) - f(theta - h e_j)) / (2h)
    """
    eta_cache = np.zeros((Y_NP.shape[0], 3), dtype=float)

    def obj(theta: np.ndarray, eta_init_cache: np.ndarray):
        eta_tmp = eta_init_cache.copy()
        f = focei_objective_fd(theta, eta_cache=eta_tmp, update_cache=True,
                               maxiter_eta=maxiter_eta, eps_eta=eps_eta)
        return f

    def funjac(theta_np: np.ndarray):
        theta = clip_to_bounds(theta_np, bounds)
        # base objective (and update main cache)
        f = focei_objective_fd(theta, eta_cache=eta_cache, update_cache=True,
                               maxiter_eta=maxiter_eta, eps_eta=eps_eta)
        if is_penalty(f):
            return f, np.zeros_like(theta)

        base_eta = eta_cache.copy()
        g = np.zeros_like(theta)

        for j in range(len(theta)):
            lo, hi = bounds[j]
            h = eps_theta * max(1.0, abs(theta[j]))
            # choose a feasible direction
            if theta[j] + h > hi:
                h = -h
            if theta[j] + h < lo:
                h = abs(h)

            if scheme == "forward":
                th_p = theta.copy()
                th_p[j] = np.clip(theta[j] + h, lo, hi)
                step = th_p[j] - theta[j]
                if step == 0:
                    g[j] = 0.0
                    continue
                f_p = obj(th_p, base_eta)
                g[j] = 0.0 if (not np.isfinite(f_p)) or is_penalty(f_p) else (f_p - f) / step

            elif scheme == "central":
                th_p = theta.copy(); th_m = theta.copy()
                th_p[j] = np.clip(theta[j] + h, lo, hi)
                th_m[j] = np.clip(theta[j] - h, lo, hi)
                step_p = th_p[j] - theta[j]
                step_m = theta[j] - th_m[j]
                if step_p == 0 or step_m == 0:
                    # fall back to forward
                    f_p = obj(th_p, base_eta)
                    g[j] = 0.0 if (not np.isfinite(f_p)) or is_penalty(f_p) else (f_p - f) / (step_p if step_p != 0 else 1.0)
                    continue
                f_p = obj(th_p, base_eta)
                f_m = obj(th_m, base_eta)
                if (not np.isfinite(f_p)) or (not np.isfinite(f_m)) or is_penalty(f_p) or is_penalty(f_m):
                    g[j] = 0.0
                else:
                    g[j] = (f_p - f_m) / (step_p + step_m)

            else:
                raise ValueError("scheme must be 'forward' or 'central'")

        g = np.where(np.isfinite(g), g, 0.0)
        return f, g

    return funjac

# ----------------------------
# Run one optimization
# ----------------------------
def make_funjac_ad_line_search_safe(mode="stop", maxiter_eta=6, bounds=BOUNDS):
    """
    L-BFGS-B safe:
      - funjac does NOT mutate eta_cache (uses a local copy)
      - callback updates eta_cache only at accepted iterates
      - guards against detached penalty constants (no grad_fn)
    """
    n_subj = Y_T.shape[0]
    eta_cache = np.zeros((n_subj, 3), dtype=float)

    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)

    def funjac(theta_np):
        theta_np = np.asarray(theta_np, dtype=float)
        theta_np = np.minimum(np.maximum(theta_np, lo), hi)

        theta_t = torch.tensor(theta_np, device=DEVICE, requires_grad=True)

        # deterministic evaluation: warmstart from last accepted iterate only
        eta_local = eta_cache.copy()

        val = focei_objective_torch(
            theta_t,
            mode=mode,
            eta_cache=eta_local,      # updated locally only
            update_cache=True,
            need_theta_grad=True,
            maxiter_eta=maxiter_eta,
        )

        f = float(val.detach().cpu().numpy())

        # IMPORTANT: penalties may return constants that are detached
        if (not hasattr(val, "requires_grad")) or (not val.requires_grad):
            g = np.zeros_like(theta_np)
        else:
            theta_t.grad = None
            val.backward()
            if theta_t.grad is None:
                g = np.zeros_like(theta_np)
            else:
                g = theta_t.grad.detach().cpu().numpy().astype(float)
                g = np.where(np.isfinite(g), g, 0.0)

        funjac._last_theta = theta_np.copy()
        funjac._last_eta   = eta_local.copy()
        return f, g

    funjac._last_theta = None
    funjac._last_eta   = None

    def callback(xk):
        xk = np.asarray(xk, dtype=float)

        # If last eval corresponds to accepted xk, adopt cached eta
        if funjac._last_theta is not None and np.allclose(xk, funjac._last_theta, atol=1e-10, rtol=0):
            eta_cache[:] = funjac._last_eta
            return

        # fallback: recompute eta_hat at accepted iterate
        theta_t = torch.tensor(xk, device=DEVICE, requires_grad=False)
        _ = focei_objective_torch(
            theta_t,
            mode="stop",             # cheaper; eta_hat should be the same numerically
            eta_cache=eta_cache,
            update_cache=True,
            need_theta_grad=False,
            maxiter_eta=maxiter_eta,
        )

    return funjac, callback


import numpy as np

def make_funjac_fd_line_search_safe(
    bounds=BOUNDS,
    maxiter_eta: int = 12,
    eps_eta: float = 1e-4,
    eps_theta: float = 1e-3,
    scheme: str = "central",#"forward",     # "forward" or "central"
    max_shrink: int = 8,         # shrink FD step up to this many times
):
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)

    # eta cache updated ONLY at accepted iterates via callback
    n_subj = Y_NP.shape[0]
    eta_cache = np.zeros((n_subj, 3), dtype=float)

    def _clip(th):
        th = np.asarray(th, dtype=float)
        return np.minimum(np.maximum(th, lo), hi)

    def _safe_obj(th, eta_init):
        """
        Evaluate objective with its own local eta cache copy.
        Returns (f, eta_hat_cache).
        """
        th = _clip(th)
        eta_local = eta_init.copy()
        f = focei_objective_fd(
            th,
            eta_cache=eta_local,
            update_cache=True,
            maxiter_eta=maxiter_eta,
            eps_eta=eps_eta,
        )
        f = float(f)
        return f, eta_local

    def _bad(f):
        return (not np.isfinite(f)) or is_penalty(f)

    def funjac(theta_np):
        theta = _clip(theta_np)

        # base objective + eta_hat(theta) into eta0
        f0, eta0 = _safe_obj(theta, eta_cache)
        funjac._last_theta = theta
        funjac._last_eta = eta0

        if _bad(f0):
            return float(f0), np.zeros_like(theta)

        base_eta = eta0.copy()
        g = np.zeros_like(theta)

        for j in range(theta.size):
            # relative step in parameter space
            h0 = eps_theta * max(1.0, abs(theta[j]))
            h = h0

            for _ in range(max_shrink):
                if scheme == "forward":
                    th_p = theta.copy()
                    th_p[j] = np.clip(theta[j] + h, lo[j], hi[j])
                    step = th_p[j] - theta[j]
                    if step == 0.0:
                        g[j] = 0.0
                        break

                    fp, _ = _safe_obj(th_p, base_eta)
                    if _bad(fp):
                        h *= 0.5
                        continue

                    g[j] = (fp - f0) / step
                    break

                elif scheme == "central":
                    th_p = theta.copy()
                    th_m = theta.copy()
                    th_p[j] = np.clip(theta[j] + h, lo[j], hi[j])
                    th_m[j] = np.clip(theta[j] - h, lo[j], hi[j])

                    step_p = th_p[j] - theta[j]
                    step_m = theta[j] - th_m[j]
                    if step_p == 0.0 or step_m == 0.0:
                        h *= 0.5
                        continue

                    fp, _ = _safe_obj(th_p, base_eta)
                    fm, _ = _safe_obj(th_m, base_eta)
                    if _bad(fp) or _bad(fm):
                        h *= 0.5
                        continue

                    g[j] = (fp - fm) / (step_p + step_m)
                    break
                else:
                    raise ValueError("scheme must be 'forward' or 'central'")

            # if it never found a finite bracket, g[j] stays 0.0

        g = np.where(np.isfinite(g), g, 0.0)
        return float(f0), g

    funjac._last_theta = None
    funjac._last_eta = None

    def callback(xk):
        xk = _clip(xk)
        # Adopt eta cache from the last evaluation if it matches xk
        if funjac._last_theta is not None and np.allclose(xk, funjac._last_theta, atol=1e-10, rtol=0):
            eta_cache[:] = funjac._last_eta
            return

        # Fallback: recompute eta_hat at accepted iterate and update cache
        _, eta_new = _safe_obj(xk, eta_cache)
        eta_cache[:] = eta_new

    return funjac, callback



import time
import numpy as np
from scipy.optimize import minimize

def make_funjac_full_implicit_line_search_safe(maxiter_eta=FULL_IMPLICIT_MODE_MAXITER,
                                               tol_eta=FULL_IMPLICIT_MODE_TOL,
                                               damping_eta=FULL_IMPLICIT_MODE_DAMPING,
                                               logdet_jitter=1e-8,
                                               max_step_norm=FULL_IMPLICIT_MODE_MAX_STEP_NORM,
                                               bounds=BOUNDS):
    """
    L-BFGS-B safe FULL-IMPLICIT wrapper:
      - funjac does NOT mutate eta_cache (uses a local copy)
      - callback updates eta_cache only at accepted iterates
      - lots of debugging prints (controlled by FULL_IMPLICIT_DEBUG* flags)
    """
    n_subj = Y_T.shape[0]
    eta_cache = np.zeros((n_subj, 3), dtype=float)

    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)

    def funjac(theta_np):
        theta_np = np.asarray(theta_np, dtype=float)
        theta_np = np.minimum(np.maximum(theta_np, lo), hi)

        # deterministic evaluation: warmstart from last accepted iterate only
        eta_local = eta_cache.copy()

        f, g = focei_objective_full_implicit_eval(
            theta_np,
            eta_cache=eta_local,
            update_cache=True,
            maxiter_eta=int(maxiter_eta),
            tol_eta=float(tol_eta),
            damping_eta=float(damping_eta),
            logdet_jitter=float(logdet_jitter),
            max_step_norm=float(max_step_norm),
        )

        # store the local cache (commit happens in callback)
        funjac._last_theta = np.array(theta_np, float)
        funjac._last_eta = eta_local

        if FULL_IMPLICIT_DEBUG:
            print(f"[FULL-IMPLICIT outer] f={float(f):.6f}  |g|={float(np.linalg.norm(g)):.3e}")

        return float(f), np.asarray(g, dtype=float)

    funjac._last_theta = None
    funjac._last_eta = None

    def cb(theta_xk):
        th = np.asarray(theta_xk, dtype=float)
        # only commit if this callback corresponds to the most recent evaluation at the accepted point
        if funjac._last_theta is None or funjac._last_eta is None:
            return
        if np.allclose(th, funjac._last_theta, rtol=1e-12, atol=1e-12):
            eta_cache[:] = funjac._last_eta

    return funjac, cb

def run_focei(
    method: str,
    theta0: np.ndarray,
    bounds=BOUNDS,
    maxiter_outer: int = 50,
    maxiter_eta: int = 12,
    eps_theta_fd: float = 1e-3,
    eps_eta_fd: float = 1e-4,
    #fd_scheme: str = "forward",   # forward is usually more robust than central here
    fd_scheme = "central",
    ftol: float = 1e-9,
    gtol: float = 1e-7,
    maxls: int = 20,
):
    method = str(method)

    if method == "FD":
        funjac, cb = make_funjac_fd_line_search_safe(
            bounds=bounds,
            maxiter_eta=maxiter_eta,
            eps_eta=eps_eta_fd,
            eps_theta=eps_theta_fd,
            scheme=fd_scheme,
        )
    elif method == "STOP":
        funjac, cb = make_funjac_ad_line_search_safe(mode="stop", maxiter_eta=maxiter_eta)
    elif method == "FULL_UNROLL":
        funjac, cb = make_funjac_ad_line_search_safe(mode="full", maxiter_eta=maxiter_eta)
    elif method == "FULL_IMPLICIT":
        funjac, cb = make_funjac_full_implicit_line_search_safe(
            bounds=bounds,
            maxiter_eta=maxiter_eta,
            tol_eta=FULL_IMPLICIT_MODE_TOL,
            damping_eta=FULL_IMPLICIT_MODE_DAMPING,
            logdet_jitter=1e-8,
            max_step_norm=FULL_IMPLICIT_MODE_MAX_STEP_NORM,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    t0 = time.perf_counter()
    res = minimize(
        fun=funjac,                    # returns (f,g)
        x0=np.asarray(theta0, float),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        callback=cb,
        options={"maxiter": int(maxiter_outer),
                     "ftol": ftol,  # disable f-based stopping (we have a custom penalty handling)
                     "gtol": gtol, 
                     "maxls": maxls, # Maximum number of line search steps per iteration. The default is 20. 
                     },
    )
    dt = time.perf_counter() - t0
    return res, dt



# %% [markdown] cell 11
# ## 4) Single-fit comparison (one initialization)
# 
# We compare runtime and final objective for the four FOCEI variants starting from a perturbed initial point near True1.

# %% [code] cell 12

if False:
    # ============================================================
    # 4) FULL-IMPLICIT debug run (single start)
    # ============================================================

    # Debug: run ONLY FULL_IMPLICIT for now
    METHODS = ["FULL_IMPLICIT"]
    print("Running METHODS =", METHODS)

    # Initial point near True1 but perturbed
    theta0 = clip_to_bounds(theta_true1 + rng.normal(0.0, 0.2, size=theta_true1.shape), BOUNDS)
    print("theta0 =", theta0)
    print("Init params (ka, CL, V):", np.exp(theta0[:3]))

    # Outer iteration budget (adjust as needed)
    MAXITER_SINGLE_AD = 100

    # Inner eta-mode Newton steps (adjust as needed)
    MAXITER_ETA_AD = 25

    # ------------------------------------------------------------------
    # Sanity check: evaluate FULL_IMPLICIT objective/grad at theta0
    # ------------------------------------------------------------------
    try:
        _full_impl_eval_counter = 0  # reset for cleaner debug headers
    except NameError:
        pass

    print("\nSanity check: FULL_IMPLICIT objective/grad at theta0 (verbose prints for first subjects)")
    eta_tmp = np.zeros((N_SUBJ, 3), dtype=float)
    f0, g0 = focei_objective_full_implicit_eval(
        theta0,
        eta_cache=eta_tmp,
        update_cache=False,
        maxiter_eta=MAXITER_ETA_AD,
        tol_eta=1e-8,
        damping_eta=1.0,
        logdet_jitter=1e-8,
        max_step_norm=10.0,
    )
    print(f"[theta0] f={f0:.6g}  ||g||={np.linalg.norm(g0):.6g}  g={g0}")

    # ------------------------------------------------------------------
    # Run SciPy minimize for FULL_IMPLICIT (watch per-eval debug prints)
    # ------------------------------------------------------------------
    print("\nNow running SciPy minimize for FULL_IMPLICIT...")
    res, dt = run_focei(
        "FULL_IMPLICIT",
        theta0,
        bounds=BOUNDS,
        maxiter_outer=MAXITER_SINGLE_AD,
        maxiter_eta=MAXITER_ETA_AD,
        #logdet_jitter=1e-8,
    )

    print("\n========== FULL_IMPLICIT result ==========")
    print("success:", res.success)
    print("message:", res.message)
    print("nit:", res.nit, "nfev:", res.nfev)
    print("f_hat:", res.fun)
    print("theta_hat:", np.array(res.x, dtype=float))
    print("params_hat (ka, CL, V):", np.exp(np.array(res.x, dtype=float)[:3]))
    print("wall_time_sec:", dt)

# %% [code] cell 13
# ============================================================
# 4) Bounds + single-run comparison
# ============================================================

# Initial point near True1 but perturbed
theta0 = clip_to_bounds(theta_true1 + rng.normal(0.0, 0.2, size=theta_true1.shape), BOUNDS)
print("Init params (ka, CL, V):", np.exp(theta0[:3]))

import traceback

if RUN_SINGLE:
    records = []
    for m in METHODS:
        # display m
        print("\n==============================")
        print("Running method:", m)
        print("==============================")
        try:
            maxiter_outer = MAXITER_SINGLE_FD if m == "FD" else MAXITER_SINGLE_AD
            res, dt = run_focei(
                m, theta0,
                bounds=BOUNDS,
                maxiter_outer=maxiter_outer,
                maxiter_eta=(MAXITER_ETA_FD if m == "FD" else MAXITER_ETA_AD),
                eps_theta_fd=EPS_THETA_FD,
                eps_eta_fd=EPS_ETA_FD,
                #fd_scheme="forward",
                fd_scheme="central",
                ftol = FTOL,
                gtol = GTOL,
                maxls = MAXLS,
            )
        except Exception as e:
            print(f"\nFAILED method={m}: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise  # re-raise so you see the full stack in the notebook
        th_hat = np.array(res.x, dtype=float)
        rec = dict(
            method=m,
            method_label=METHOD_LABEL[m],
            success=bool(res.success),
            status=int(getattr(res, "status", -999)),
            message=str(getattr(res, "message", "")),
            objective=float(res.fun),
            runtime_s=float(dt),
            nit=int(getattr(res, "nit", -1)),
            nfev=int(getattr(res, "nfev", -1)),
            njev=int(getattr(res, "njev", -1)),
        )
        rec.update(theta_to_record(th_hat))
        rec["is_penalty"] = is_penalty(rec["objective"])
        records.append(rec)

    df_single = pd.DataFrame(records)
    display(df_single)

    df_single.to_csv("tables/focei_single_run_comparison.csv", index=False)
    df_single.to_latex("tables/focei_single_run_comparison.tex", index=False, float_format="%.4g")
    print("Saved: tables/focei_single_run_comparison.csv")
    print("Saved: tables/focei_single_run_comparison.tex")

# %% [code] cell 14
if RUN_SINGLE:
    # ============================================================
    # 5) Runtime bar chart (single-fit)
    # ============================================================

    df_rt = df_single.copy()
    df_rt = df_rt.sort_values("method")
    plt.figure(figsize=(6,4))
    plt.bar(df_rt["method_label"], df_rt["runtime_s"] / 60.0)
    plt.ylabel("Runtime (minutes)")
    plt.title("Single-fit runtime (FOCEI variants)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    fp = os.path.join("figures", "fig_singlefit_runtime_bar_minutes.png")
    plt.savefig(fp, dpi=200)
    plt.show()
    print("Saved:", fp)

if __name__ == "__main__":
    summary_path = Path("tables") / "flipflop_single_start_summary.csv"
    if "df_single" in globals() and isinstance(df_single, pd.DataFrame):
        df_single.to_csv(summary_path, index=False)
        print(f"Saved: {summary_path}")
