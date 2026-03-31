from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from scipy.optimize import OptimizeResult


def display(obj):
    if hasattr(obj, "to_string"):
        try:
            print(obj.to_string(index=False))
            return
        except Exception:
            pass
    print(obj)


def _configure_output_dir() -> tuple[Path, Path]:
    parser = argparse.ArgumentParser(
        description="Run the single-start Warfarin FOCEI comparison used in the manuscript."
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where figures and tables will be written.",
    )
    parser.add_argument(
        "--max-method-hours",
        type=float,
        default=None,
        help="Optional wall-time limit per method. If reached, the script returns the best point seen so far for that method.",
    )
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    bundled_data = script_dir / "data" / "warfarin_dat.csv"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if bundled_data.exists():
        shutil.copy2(bundled_data, output_dir / "warfarin_dat.csv")
    os.chdir(output_dir)
    print(f"Output directory: {output_dir}")
    global MAX_METHOD_RUNTIME_SEC
    MAX_METHOD_RUNTIME_SEC = None if args.max_method_hours is None else max(float(args.max_method_hours), 0.0) * 3600.0
    if MAX_METHOD_RUNTIME_SEC is not None:
        print(f"Per-method wall-time limit: {args.max_method_hours:.3f} hours")
    return output_dir, bundled_data


OUTPUT_DIR, BUNDLED_DATA = _configure_output_dir()


class MethodTimeLimitReached(RuntimeError):
    pass
# %% [markdown] cell 1
# # Single-start FOCEI comparison for the public Warfarin PK/PD example
# 
# This standalone script runs only the **single-start** joint PK/PD estimation problem for the
# public Warfarin dataset. It does **not** reproduce the multistart analyses, slice calculations,
# or the full manuscript figure set.
# 
# The script evaluates the same outer-gradient strategies used in the manuscript:
# 
# - **FULL-unrolled**: reverse-mode differentiation through the unrolled inner Newton solver
# - **FULL-implicit**: implicit differentiation of the inner mode conditions
# - **STOP**: detach mode sensitivity while retaining the dominant curvature derivative
# - **STOP+FULL**: STOP warm start followed by FULL-implicit polishing
# - **FULL+STOP**: FULL-implicit warm start followed by STOP polishing
# - **FD**: finite differences of the STOP objective
# 
# The model estimates PK and PD parameters simultaneously.

# %% [code] cell 2

import os
import math
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ---------- Internal Bools ----------
boolReadResults = False # if you want to read results from disk instead of re-running the optimization (which takes a few hours)
METHODS_TO_READ = ["FULL_UNROLL", "FULL_IMPLICIT", "STOP", "STOP+FULL", "FULL+STOP","FD"]
boolWriteResults = True

# Which methods to run (if boolReadResults is False)
METHODS = ["FULL_UNROLL", "FULL_IMPLICIT", "STOP", "STOP+FULL", "FULL+STOP", "FD"]


# 2D objective function
DO_OBJECTIVE_SLICE = False # do the objective slice for the STOP method (takes a few hours, so we can skip if we just want to read results from disk)
DO_OBJECTIVE_SLICE_PLOT = True


# multistart
RUN_MULTISTART = False
RUN_MULTISTART_PLOT = False
N_STARTS = 10
METHODS_MS = ["FULL_UNROLL","STOP+FULL", "FULL+STOP","STOP", "FULL_IMPLICIT","FD"]
# METHODS_MS = ["FULL_UNROLL"]

# ---------- Reproducibility ----------
np.random.seed(0)
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

# ---------- Data (public warfarin example) ----------
WARFARIN_DATA_URL = "https://holford.fmhs.auckland.ac.nz/research/nlmixr/warfarin/warfarin_dat.csv"
DATA_PATH = "warfarin_dat.csv"   # downloaded automatically if missing

# ---------- Output ----------
from pathlib import Path

FIG_DIR = Path("figures_out_warfarin")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------- FOCEI inner-solve settings (same as AMG notebook) ----------

full_unroll_steps = 50
inner_max_iter = 50
inner_tol = 1e-6
inner_damping = 0.8
logdet_jitter = 1e-4 # If you're close to singularities, different methods "feel" different curvature; a bit more jitter makes them behave more similarly
floor_rel = 1e-4
floor_abs = 1e-6 # This makes the implicit solve / eta Newton steps / logdet more stable and reduces path sensitivity
max_step_norm = 3.0

# FD steps
fd_eps_outer = float(1e-3)
fd_eps_eta = float(1e-3)


INNER_MAX_ITER = inner_max_iter
INNER_TOL = inner_tol
INNER_DAMPING = inner_damping

LOGDET_JITTER = logdet_jitter
ETA_SOLVE_FLOOR_REL = floor_rel
ETA_SOLVE_FLOOR_ABS = floor_abs
ETA_MAX_STEP_NORM = max_step_norm

max_outer_iter = 50
gtol = 1e-4
ftol = 1e-9
maxls = 40 # Higher numbers give the line search more chances to find a safe step instead of bailing/"ABNORMAL"
gtol_STOP = gtol#0.1
ftol_STOP = ftol#1e-10
maxls_STOP = maxls#20

# ---------- PD forcing grid ----------
PD_T0 = 0.0
PD_T1 = 144.0
PD_DT = 0.25  # hours

DEVICE = torch.device("cpu")


# %% [code] cell 3
results: Dict[str, dict] = {}

from pathlib import Path
import numpy as np
import pandas as pd
import ast
import csv
from typing import Dict, Any, List, Optional

import re
from pathlib import Path
from typing import Optional, List

def _norm_token(s: str) -> str:
    # normalize aggressively: remove all non-alphanumerics, uppercase
    return re.sub(r"[^A-Za-z0-9]+", "", str(s)).upper()

def _find_field_file(prefix: str, directory: str, method: str, field_aliases: List[str]) -> Optional[Path]:
    """
    Find a file for (method, field) by scanning directory and matching by normalized tokens.
    Works even if method has '+' or other punctuation, and if field name differs.
    """
    d = Path(directory)
    want_m = _norm_token(method)

    # scan all candidate files once
    candidates = list(d.glob(f"{prefix}_*.csv"))

    # build list of acceptable field normalized tokens
    field_norms = [_norm_token(f) for f in field_aliases]

    best = None
    for p in candidates:
        name = p.stem  # without .csv

        # Expect something like: {prefix}_{METHOD}_{FIELD}
        # We'll parse by splitting on '_' and taking:
        #   method_part = everything after prefix_ up to last '_' chunk(s) that form field
        parts = name.split("_")
        if not parts or parts[0] != prefix:
            continue

        # Try all possible splits of (method_tokens, field_tokens)
        # so this works if field is "runtime_sec" or "final_runtime_sec" etc.
        rest = parts[1:]
        for k in range(1, len(rest) + 1):
            method_tokens = rest[:k]
            field_tokens = rest[k:]

            if not field_tokens:
                continue

            m_guess = _norm_token("_".join(method_tokens))
            f_guess = _norm_token("_".join(field_tokens))

            if m_guess == want_m and f_guess in field_norms:
                return p  # exact match; return immediately

            # fallback: sometimes method token includes extra words; allow "contains"
            if want_m and want_m in m_guess and f_guess in field_norms:
                best = p

    return best

def _read_text_csvish(path: Path) -> str:
    """Read a tiny text/csv file as text, stripping an optional header row like 'value'."""
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return ""
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip() != ""]
    if not lines:
        return ""

    # If it's a simple 2-line CSV with header
    if len(lines) >= 2 and lines[0].lower() in ("value", "message", "method"):
        return "\n".join(lines[1:]).strip()

    # If it's one-line like "value,CONVERGENCE: ..."
    if len(lines) == 1 and lines[0].lower().startswith("value,"):
        return lines[0].split(",", 1)[1].strip()

    return "\n".join(lines).strip()

def _flatten_csv_values(path: Path) -> List[Any]:
    """
    Read CSV and flatten into a list of values, handling:
      - with/without header
      - one-column or multi-column
      - values that are strings like '[1,2,3]'
    """
    # Read with header=None so we capture header rows too (we'll drop them if needed)
    df = pd.read_csv(path, header=None)
    vals = df.values.reshape(-1).tolist()

    # Drop blanks/NAs
    cleaned = []
    for v in vals:
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        cleaned.append(v)

    # Drop common header tokens if they appear as first element
    if cleaned and isinstance(cleaned[0], str) and cleaned[0].strip().lower() in ("value", "x", "trace", "final_x"):
        cleaned = cleaned[1:]

    # If it's a single string that looks like a Python list, parse it
    if len(cleaned) == 1 and isinstance(cleaned[0], str):
        s = cleaned[0].strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, np.ndarray)):
                    return list(parsed)
            except Exception:
                pass

        # If it's comma-separated numbers in a single string
        if "," in s:
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            # try numeric parse
            ok = True
            nums = []
            for p in parts:
                try:
                    nums.append(float(p))
                except Exception:
                    ok = False
                    break
            if ok:
                return nums

    return cleaned

def _coerce_bool(x: Any) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    # fallback: non-empty string -> True
    return bool(s)

def _coerce_int(x: Any, default: int = -1) -> int:
    try:
        return int(float(x))
    except Exception:
        return default

def _coerce_float(x: Any, default: float = np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _is_float_like(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False
    
def load_one_method_results_from_csv(
    method: str,
    directory: str = ".",
    prefix: str = "result",
) -> Dict[str, Any]:
    method = str(method)
    d = Path(directory)

    # Keep the original label for reporting
    out: Dict[str, Any] = {"method": method}

    # Try a few filename variants (most robust approach)
    method_candidates = [
        method,                         # e.g. STOP+FULL
        method.replace("+", "_"),        # e.g. STOP_FULL
        method.replace("+", ""),         # e.g. STOPFULL
        method.replace("+", "PLUS"),     # e.g. STOPPLUSFULL
        method.replace(" ", ""),         # just in case
    ]

    def _file(field: str, mname: str) -> Path:
        return d / f"{prefix}_{mname}_{field}.csv"

    def _first_existing(fields: List[str]) -> Optional[Path]:
        for mname in method_candidates:
            for f in fields:
                p = _file(f, mname)
                if p.exists():
                    return p
        return None

    # Scalars
    p = _first_existing(["success", "final_success"])
    out["success"] = _coerce_bool(_flatten_csv_values(p)[0]) if p else False

    p = _first_existing(["niter", "final_niter", "nit"])
    out["niter"] = _coerce_int(_flatten_csv_values(p)[0]) if p else -1

    p = _first_existing(["nfev", "final_nfev"])
    out["nfev"] = _coerce_int(_flatten_csv_values(p)[0]) if p else -1

    #p = _first_existing(["runtime_sec", "final_runtime_sec"])
    #print(f"Loading runtime_sec for method {method} from {p}")
    #out["runtime_sec"] = _coerce_float(_flatten_csv_values(p)[0]) if p else np.nan
    p = _find_field_file(
        prefix=prefix,
        directory=str(d),
        method=method,
        field_aliases=["runtime_sec", "final_runtime_sec", "runtime", "runtime_s", "runtime_seconds", "walltime_sec", "walltime_s"]
    )
    #print(f"Loading runtime_sec for method {method} from {p}")
    out["runtime_sec"] = _coerce_float(_flatten_csv_values(p)[0]) if p else np.nan

    p = _first_existing(["final_f", "fun", "OFV(=2*L)"])
    out["final_f"] = _coerce_float(_flatten_csv_values(p)[0]) if p else np.nan

    p = _first_existing(["message", "final_message"])
    out["message"] = _read_text_csvish(p) if p else ""

    # final_x (vector)
    p = _first_existing(["final_x", "x_opt", "x"])
    if p:
        vals = _flatten_csv_values(p)
        out["final_x"] = [float(v) for v in vals if _is_float_like(v)]
    else:
        out["final_x"] = []

    # traces (vectors)
    for trace_key, possible_fields in [
        ("trace_f", ["trace_f", "final_trace_f"]),
        ("trace_gnorm", ["trace_gnorm", "final_trace_gnorm"]),
    ]:
        p = _first_existing(possible_fields)
        if p:
            vals = _flatten_csv_values(p)
            out[trace_key] = [float(v) for v in vals if _is_float_like(v)]
        else:
            out[trace_key] = []

    return out

def load_results_from_csv(
    methods: List[str],
    directory: str = ".",
    prefix: str = "result",
) -> Dict[str, Dict[str, Any]]:
    """Bulk load for multiple methods."""
    results: Dict[str, Dict[str, Any]] = {}
    for m in methods:
        out = load_one_method_results_from_csv(m, directory=directory, prefix=prefix)
        results[str(m)] = out
    return results


if boolReadResults:

    #unique_methods = ["FD", "STOP", "FULL_UNROLL", "FULL_IMPLICIT"]
    unique_methods = METHODS_TO_READ

    # in inuque_methods, replace the "+" with "_" for file loading, but keep the original method name in the results dict
    #unique_methods_for_files = [m.replace("+", "_") for m in unique_methods]
    # list the unique_methods_for_files
    #print("unique_methods_for_files:", unique_methods_for_files)

    results = load_results_from_csv(unique_methods, directory=".", prefix="result")

    for m in unique_methods:
        out = results[m]
        # change STOP_FULL to STOP+FULL in out
        #if m == "STOP_FULL":
        #    out["method"] = m.replace("_", "+")
        

        print(f"{out['method']}: success={out['success']}  niter={out['niter']}  nfev={out['nfev']}  "
            f"runtime={out['runtime_sec']:.1f}s  final_f={out['final_f']:.3f}")

            

# %% [code] cell 4
from pathlib import Path

DATA_PATH = Path("warfarin_dat.csv")
if not DATA_PATH.exists():
    alt = Path("/mnt/data/warfarin_dat.csv")
    if alt.exists():
        DATA_PATH = alt
    else:
        # Optional: download if internet is available (will fail in offline environments)
        import urllib.request
        print("Local warfarin_dat.csv not found; attempting download ...")
        urllib.request.urlretrieve(WARFARIN_DATA_URL, "warfarin_dat.csv")
        DATA_PATH = Path("warfarin_dat.csv")

df = pd.read_csv(DATA_PATH)

# Basic sanity checks / summary
print("rows:", len(df), "subjects:", df["id"].nunique())
print("dvid counts:", df["dvid"].value_counts().to_dict())
print("evid counts:", df["evid"].value_counts().to_dict())

# Split observations vs dosing records
dose_df = df[df["evid"] == 1][["id", "amt"]].drop_duplicates().sort_values("id")
pk_df   = df[(df["evid"] == 0) & (df["dvid"] == 1)].sort_values(["id", "time"])
pd_df   = df[(df["evid"] == 0) & (df["dvid"] == 2)].sort_values(["id", "time"])

dose_map = {int(r.id): float(r.amt) for r in dose_df.itertuples(index=False)}

# %% [markdown] cell 5
# ## Joint structural model
# 
# ### PK
# One-compartment oral PK with first-order absorption:
# 
# - individual parameters: $k_{a,i}, CL_i, V_i$
# - concentration $C_i(t)$ is computed in **closed form**.
# 
# ### PD (PCA)
# An effect-compartment + Emax inhibition model:
# 
# - effect-site concentration $C_{e,i}(t)$ solves $dC_e/dt = k_{e0,i}(C(t) - C_e)$, $C_e(0)=0$
# - PCA prediction: $\widehat{\mathrm{PCA}}_i(t) = E_{0,i}\left(1 - E_{\max}\,\frac{C_{e,i}(t)}{C_{50,i}+C_{e,i}(t)}\right)$
# 
# We estimate PK+PD fixed effects **jointly** and use FOCEI with diagonal random-effect covariance.
# 
# Random effects (dimension $q=6$):
# 
# - PK: $\eta_{ka,i},\eta_{CL,i},\eta_{V,i}$
# - PD: $\eta_{E0,i},\eta_{C50,i},\eta_{ke0,i}$

# %% [code] cell 6

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, NamedTuple

# -------------------------
# Random-effect dimension
# -------------------------
ETA_DIM = 6  # (ka, CL, V, E0, C50, ke0)

@dataclass
class SubjectData:
    sid: int
    dose_mg: float
    pk_times: np.ndarray
    pk_obs: np.ndarray
    pd_times: np.ndarray
    pd_obs: np.ndarray

# Build per-subject objects
subjects: List[SubjectData] = []
for sid in sorted(df["id"].unique()):
    sid = int(sid)
    dose = float(dose_map.get(sid, 0.0))
    pk = pk_df[pk_df["id"] == sid]
    pd = pd_df[pd_df["id"] == sid]
    subjects.append(
        SubjectData(
            sid=sid,
            dose_mg=dose,
            pk_times=pk["time"].to_numpy(dtype=float),
            pk_obs=pk["dv"].to_numpy(dtype=float),
            pd_times=pd["time"].to_numpy(dtype=float),
            pd_obs=pd["dv"].to_numpy(dtype=float),
        )
    )

print("Built subjects:", len(subjects))
print("Example subject:", subjects[0])

# -------------------------
# Joint parameter vector
# -------------------------
# Order matters: keep this consistent with unpack_theta().
PARAM_NAMES = [
    # PK fixed effects
    "logKA", "logCL", "logV",
    # PD fixed effects
    "logE0", "logC50", "logKE0", "logitEMAX",
    # Residual SDs
    "logSIGMA_PK", "logSIGMA_PD",
    # Random-effect SDs (diagonal Omega)
    "logOM_KA", "logOM_CL", "logOM_V",
    "logOM_E0", "logOM_C50", "logOM_KE0",
]

class Theta(NamedTuple):
    logKA: torch.Tensor
    logCL: torch.Tensor
    logV: torch.Tensor
    logE0: torch.Tensor
    logC50: torch.Tensor
    logKE0: torch.Tensor
    logitEMAX: torch.Tensor
    logSIGMA_PK: torch.Tensor
    logSIGMA_PD: torch.Tensor
    logOM_KA: torch.Tensor
    logOM_CL: torch.Tensor
    logOM_V: torch.Tensor
    logOM_E0: torch.Tensor
    logOM_C50: torch.Tensor
    logOM_KE0: torch.Tensor

def unpack_theta(x: torch.Tensor) -> Theta:
    assert x.numel() == len(PARAM_NAMES)
    return Theta(*[x[i] for i in range(x.numel())])

def sigmoid(z: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-z))

def _phi_exprel_neg(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    phi(x) = (1 - exp(-x))/x  for x>=0, stable around x~0.
    Uses -expm1(-x)/x form; avoids division by 0 by clamping x.
    """
    x = torch.clamp(x, min=0.0)
    x_safe = torch.clamp(x, min=eps)
    return -torch.expm1(-x) / x_safe

def diffexp_over_diff(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute (exp(-a t) - exp(-b t)) / (b - a) stably for a,b>0, t>=0.
    """
    t = torch.clamp(t, min=0.0)
    m = torch.minimum(a, b)
    d = torch.abs(a - b)
    x = d * t                       # >= 0
    return torch.exp(-m * t) * t * _phi_exprel_neg(x, eps=eps)


# -------------------------
# PK: 1-comp oral closed form
# -------------------------
def pk_conc_oral_1c_torch(t: torch.Tensor, dose: torch.Tensor,
                         ka: torch.Tensor, cl: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    t = torch.clamp(t, min=0.0)
    k = cl / v

    # C(t) = (dose/v) * ka * (exp(-k t) - exp(-ka t)) / (ka - k)
    #      = (dose/v) * ka * (exp(-k t) - exp(-ka t)) / (ka - k)
    # Use stable ratio:
    ratio = diffexp_over_diff(k, ka, t)    # (exp(-k t)-exp(-ka t))/(ka-k)
    c = (dose / v) * ka * ratio
    return c





# -------------------------
# Effect compartment: closed form for Ce(t) under oral 1-comp Cp(t)
# -------------------------
def ce_effect_compartment_torch(
    t: torch.Tensor, dose: torch.Tensor,
    ka: torch.Tensor, cl: torch.Tensor, v: torch.Tensor,
    ke0: torch.Tensor,
    eps_rate: float = 1e-8,
) -> torch.Tensor:
    t = torch.clamp(t, min=0.0)
    k = cl / v

    # Stable building blocks:
    term1 = diffexp_over_diff(k,   ke0, t)   # (exp(-k t)  - exp(-ke0 t)) / (ke0 - k)
    term2 = diffexp_over_diff(ka,  ke0, t)   # (exp(-ka t) - exp(-ke0 t)) / (ke0 - ka)

    # Ce_general = A * ke0 * (term1 - term2), A = dose*ka/(v*(ka-k))
    # We must avoid explicit 1/(ka-k) blowing up when ka~k.
    delta = ka - k
    absdelta = torch.abs(delta)

    # Sign-preserving clamp so we never divide by ~0 even when ka < k.
    sign = torch.sign(delta)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign) # ones_like: 
    delta_safe = torch.where(absdelta < eps_rate, eps_rate * sign, delta)

    A_safe = dose * ka / (v * delta_safe)
    ce_general_safe = A_safe * ke0 * (term1 - term2)

    # True ka=k limit (your existing formula, but computed safely):
    a = k - ke0
    a_safe = torch.where(torch.abs(a) < eps_rate, torch.as_tensor(eps_rate, dtype=t.dtype, device=t.device), a)

    exp_k   = torch.exp(-k   * t)
    exp_ke0 = torch.exp(-ke0 * t)

    # ce_limit corresponds to ka==k:
    ce_limit = (dose / v) * k * ke0 * (exp_ke0 - exp_k * (1.0 + a * t)) / (a_safe * a_safe)

    # If also ke0==k, limit reduces further (t^2/2 term)
    ce_limit2 = (dose / v) * (k * k) * torch.exp(-k * t) * (t * t) / 2.0
    ce_limit = torch.where(torch.abs(a) < eps_rate, ce_limit2, ce_limit)

    # Smooth blend (avoid a hard where that might still backprop weirdly at the boundary):
    w = torch.clamp(absdelta / eps_rate, 0.0, 1.0)  # 0 -> limit, 1 -> general
    ce = w * ce_general_safe + (1.0 - w) * ce_limit

    return torch.clamp(ce, min=0.0)


def predict_pd_pca_torch(subj: SubjectData, theta: Theta, eta: torch.Tensor) -> torch.Tensor:
    """
    PCA prediction at subj.pd_times using joint PK/PD individual params.
    """
    # Individual params
    ka  = torch.exp(theta.logKA + eta[0])
    cl  = torch.exp(theta.logCL + eta[1])
    v   = torch.exp(theta.logV  + eta[2])

    E0  = torch.exp(theta.logE0   + eta[3])
    C50 = torch.exp(theta.logC50  + eta[4])
    ke0 = torch.exp(theta.logKE0  + eta[5])

    EMAX = sigmoid(theta.logitEMAX)

    t_pd = torch.as_tensor(subj.pd_times, dtype=torch.float64, device=DEVICE)
    dose = torch.as_tensor(subj.dose_mg, dtype=torch.float64, device=DEVICE)

    ce = ce_effect_compartment_torch(t_pd, dose, ka, cl, v, ke0)

    # Emax inhibition on PCA
    effect = EMAX * ce / (C50 + ce + 1e-12)
    pca_hat = E0 * (1.0 - effect)
    return pca_hat

def predict_pk_conc_torch(subj: SubjectData, theta: Theta, eta: torch.Tensor) -> torch.Tensor:
    ka  = torch.exp(theta.logKA + eta[0])
    cl  = torch.exp(theta.logCL + eta[1])
    v   = torch.exp(theta.logV  + eta[2])

    t_pk = torch.as_tensor(subj.pk_times, dtype=torch.float64, device=DEVICE)
    dose = torch.as_tensor(subj.dose_mg, dtype=torch.float64, device=DEVICE)
    c_hat = pk_conc_oral_1c_torch(t_pk, dose, ka, cl, v)
    return c_hat

# -------------------------
# Subject-level negative log joint (up to constants)
# -------------------------
def h_i(subj: SubjectData, x: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    theta = unpack_theta(x)

    # Residual SDs
    sigma_pk = torch.exp(theta.logSIGMA_PK)
    sigma_pd = torch.exp(theta.logSIGMA_PD)

    # Random-effect SDs (diagonal)
    omega = torch.exp(torch.stack([
        theta.logOM_KA, theta.logOM_CL, theta.logOM_V,
        theta.logOM_E0, theta.logOM_C50, theta.logOM_KE0,
    ]))

    # PK likelihood
    y_pk = torch.as_tensor(subj.pk_obs, dtype=torch.float64, device=DEVICE)
    c_hat = predict_pk_conc_torch(subj, theta, eta)
    r_pk = (y_pk - c_hat) / (sigma_pk + 1e-12)
    nll_pk = y_pk.numel() * torch.log(sigma_pk + 1e-12) + 0.5 * torch.sum(r_pk * r_pk)

    # PD likelihood
    y_pd = torch.as_tensor(subj.pd_obs, dtype=torch.float64, device=DEVICE)
    pca_hat = predict_pd_pca_torch(subj, theta, eta)
    r_pd = (y_pd - pca_hat) / (sigma_pd + 1e-12)
    nll_pd = y_pd.numel() * torch.log(sigma_pd + 1e-12) + 0.5 * torch.sum(r_pd * r_pd)

    # Random-effect prior (diag Omega)
    nll_prior = torch.sum(torch.log(omega + 1e-12)) + 0.5 * torch.sum((eta / (omega + 1e-12)) ** 2)

    return nll_pk + nll_pd + nll_prior

# %% [code] cell 7
# =========================
# Inner solvers for eta-hat
# =========================


# =========================
# PATCH: Stabilized FULL_UNROLL
# =========================

def grad_hess_eta_ad_for_step(
    subj: SubjectData,
    x: torch.Tensor,
    eta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (g, H_detached, hi) where:
      - g has create_graph=True so eta updates remain differentiable wrt x
      - H is computed numerically via autograd
        so the Newton solve does NOT backprop through H (removes unstable 3rd-derivative transients).
    """
    hi = h_i(subj, x, eta)
    g = torch.autograd.grad(hi, eta, create_graph=True, retain_graph=True)[0]

    q = g.numel()
    H_rows = []
    for j in range(q):
        Hj = torch.autograd.grad(g[j], eta, create_graph=True, retain_graph=True)[0]
        H_rows.append(Hj)
    H = torch.stack(H_rows, dim=0)  # requires_grad=False already
    return g, H, hi


def solve_newton_pd_autograd(H, g, *, jitter=1e-6, max_tries=8,
                             floor_rel=1e-4, floor_abs=1e-6, max_step_norm=3.0):
    Hs = 0.5*(H + H.T)
    q = Hs.shape[0]; I = torch.eye(q, dtype=Hs.dtype, device=Hs.device)
    scale = torch.clamp(torch.mean(torch.abs(torch.diagonal(Hs))).detach(), min=1.0)
    lam = (floor_abs + floor_rel*scale + jitter)  # detach shift magnitude
    for _ in range(int(max_tries)):
        Lchol, info = torch.linalg.cholesky_ex(Hs + lam*I)
        if int(info.detach().max().cpu().item()) == 0:
            step = torch.cholesky_solve(g.unsqueeze(-1), Lchol).squeeze(-1)
            break
        lam = lam * 10.0
    else:
        step = torch.linalg.solve(Hs + lam*I, g)
    n = torch.linalg.vector_norm(step)
    step = step * torch.clamp(max_step_norm/(n+1e-12), max=1.0)
    return step, lam


def newton_eta_full_unroll(
    subj: SubjectData,
    x: torch.Tensor,
    eta0: torch.Tensor,
    *,
    unroll_steps: int = 12,
    tol: float = 1e-6,          # <-- default now matches your intended inner_tol
    damping: float = 1.0,
    jitter: float = 1e-6,
    floor_rel: float = 1e-3,    # <-- more conservative than 1e-5 (important for stability)
    floor_abs: float = 1e-6,
    max_step_norm: float = 2.0, # <-- more conservative default
    debug: bool = False,
    **kwargs: object,
) -> torch.Tensor:
    """
    Stabilized FULL_UNROLL inner solve.

    """
    # Ensure dtype/device consistency
    eta = eta0.detach().clone().to(dtype=x.dtype, device=x.device)
    eta = eta.requires_grad_(True)

    x_det = x.detach()

    for k in range(int(unroll_steps)):
        # g has graph; H is detached (no graph)
        #g, H_det, hi = grad_hess_eta_ad_for_step(subj, x, eta)

        #step = _solve_newton_damped(
        #    H_det, g,
        #    jitter=float(jitter),
        #    floor_rel=float(floor_rel),
        #    floor_abs=float(floor_abs),
        #    max_step_norm=float(max_step_norm),
        #)
        g, H, hi = grad_hess_eta_ad_for_step(subj, x, eta)
        step, used_shift = solve_newton_pd_autograd(H, g,
            jitter=jitter, max_tries=8, floor_rel=floor_rel, floor_abs=floor_abs,
            max_step_norm=max_step_norm)
        step = float(damping) * step

        # convergence check based on the step we *intend* to take
        step_norm = float(torch.norm(step).detach().cpu().item())
        if step_norm < float(tol):
            if debug:
                print(f"[sid={subj.sid}] break: step_norm={step_norm:.3e} < tol")
            break

        # Backtracking line search on h_i (decisions detached)
        hi0 = float(hi.detach().cpu().item())
        alpha = 1.0
        accepted = False
        with torch.no_grad():
            eta_det = eta.detach()
            step_det = step.detach()
            for _ in range(12):
                eta_try = eta_det - alpha * step_det
                hi_try = h_i(subj, x_det, eta_try)
                hi_try_val = float(hi_try.detach().cpu().item())
                if np.isfinite(hi_try_val) and (hi_try_val <= hi0):
                    accepted = True
                    break
                alpha *= 0.5
    
        if not accepted:
            # STOP behavior: if no improving step found, stop early.
            break

        # Apply update (alpha is a python float -> stop-grad through acceptance decisions)
        eta = eta - alpha * step

        # stop if accepted step is tiny
        if float(alpha) * step_norm < float(tol):
            if debug:
                print(f"[sid={subj.sid}] break: alpha*step_norm={(alpha*step_norm):.3e} < tol")
            break

        # optional gradient norm check (post-update would require recompute; keep cheap)
        gnorm = float(torch.norm(g.detach()).cpu().item())
        if gnorm < float(tol):
            if debug:
                print(f"[sid={subj.sid}] break: gnorm={gnorm:.3e} < tol")
            break

    return eta


def stabilize_for_cholesky(
    H: torch.Tensor,
    *,
    jitter: float = 1e-6,
    floor_rel: float = 1e-4,
    floor_abs: float = 1e-6,
    max_tries: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, bool, float]:
    """
    Return (Hpd, L, ok, used_shift) where Hpd = sym(H) + shift*I is PD enough for Cholesky.
    Uses a shift ladder with cholesky_ex, no eigendecomposition.
    """
    q = H.shape[0]
    I = torch.eye(q, dtype=H.dtype, device=H.device)

    Hs = 0.5 * (H + H.T)
    if not torch.all(torch.isfinite(Hs)):
        Hs = torch.nan_to_num(Hs, nan=0.0, posinf=0.0, neginf=0.0)

    diag = torch.diagonal(Hs)
    scale = torch.clamp(torch.mean(torch.abs(diag)).detach(), min=1.0)
    base_floor = (float(floor_abs) + float(floor_rel) * float(scale))

    shift = torch.as_tensor(base_floor + float(jitter), dtype=Hs.dtype, device=Hs.device).detach()

    for _ in range(int(max_tries)):
        Hpd = Hs + shift * I
        L, info = torch.linalg.cholesky_ex(Hpd)  # info==0 means success <U+E200>cite<U+E202>turn2view0<U+E201>
        ok = (int(info.detach().cpu().item()) == 0) and bool(torch.all(torch.isfinite(L)))
        if ok:
            return Hpd, L, True, float(shift.detach().cpu().item())
        shift = (10.0 * shift).detach()

    # last resort: diagonal dominance (Gershgorin-style)
    off = torch.sum(torch.abs(Hs), dim=1) - torch.abs(torch.diagonal(Hs))
    need = off - torch.diagonal(Hs) + (base_floor + float(jitter))
    gshift = torch.clamp(torch.max(need), min=0.0).detach()

    Hpd = Hs + gshift * I
    L, info = torch.linalg.cholesky_ex(Hpd)
    ok = (int(info.detach().cpu().item()) == 0) and bool(torch.all(torch.isfinite(L)))

    return Hpd, L, ok, float(gshift.detach().cpu().item())



def _stabilize_hessian_for_cholesky(
    H: torch.Tensor,
    *,
    jitter: float = 1e-6,
    floor_rel: float = 1e-6,
    floor_abs: float = 1e-12,
    max_tries: int = 6,
    jitter_mult: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (Hpd, L) where:
      - Hpd is a symmetrized + diagonally shifted version of H meant to be PD
      - L is a Cholesky factor of Hpd

    We detach the shift magnitude so the stabilization itself does not dominate
    higher-order derivatives (we're only using this inside FULL-implicit).
    """
    q = H.shape[0]
    eye = torch.eye(q, dtype=H.dtype, device=H.device)

    Hs = 0.5 * (H + H.T)

    # Eigenvalue-based shift to push minimum eigenvalue above a floor
    eigs = torch.linalg.eigvalsh(Hs)
    min_eig = torch.min(eigs)
    scale = torch.max(torch.abs(eigs)).detach()
    floor = torch.as_tensor(floor_abs, dtype=Hs.dtype, device=Hs.device) + torch.as_tensor(floor_rel, dtype=Hs.dtype, device=Hs.device) * scale
    shift = torch.clamp(floor - min_eig, min=0.0) + torch.as_tensor(jitter, dtype=Hs.dtype, device=Hs.device)
    shift = shift.detach()

    # Try Cholesky with escalating extra jitter
    for k in range(int(max_tries)):
        extra = torch.as_tensor(jitter * (jitter_mult ** k), dtype=Hs.dtype, device=Hs.device)
        Hpd = Hs + (shift + extra) * eye
        L, info = torch.linalg.cholesky_ex(Hpd)
        if int(info) == 0:
            return Hpd, L

    # Last resort: return the most shifted version (might still fail later)
    Hpd = Hs + (shift + torch.as_tensor(jitter * (jitter_mult ** (max_tries-1)), dtype=Hs.dtype, device=Hs.device)) * eye
    L = torch.linalg.cholesky(Hpd)  # let this throw if truly hopeless
    return Hpd, L


def stable_logdet_and_chol(H, *, jitter: float, floor_rel: float, floor_abs: float):
    # stabilize_for_cholesky returns: (Hpd, Lchol, ok, used_shift)
    Hpd, Lchol, ok, used_shift = stabilize_for_cholesky(
        H, jitter=jitter, floor_rel=floor_rel, floor_abs=floor_abs
    )

    # If stabilization claims failure, try a last-resort eigen-based PD shift
    # (q is small, so this is cheap and prevents hard crashes).
    if not ok or (not torch.all(torch.isfinite(Lchol))):
        Hs = 0.5 * (H + H.T)
        Hs = torch.nan_to_num(Hs, nan=0.0, posinf=0.0, neginf=0.0)
        q = Hs.shape[0]
        I = torch.eye(q, dtype=Hs.dtype, device=Hs.device)

        w = torch.linalg.eigvalsh(Hs)
        min_eig = torch.min(w)
        scale = torch.clamp(torch.max(torch.abs(w)).detach(), min=1.0)
        target = torch.as_tensor(float(floor_abs), dtype=Hs.dtype, device=Hs.device) + float(floor_rel) * scale
        shift = torch.clamp(target - min_eig, min=0.0).detach() + float(jitter)

        Hpd = Hs + shift * I
        Lchol = torch.linalg.cholesky(Hpd)  # let this raise if truly hopeless

    # logdet |Hpd| = 2 * sum(log(diag(Lchol)))
    logdet = 2.0 * torch.sum(torch.log(torch.diagonal(Lchol)))
    return logdet, Hpd, Lchol

def focei_objective_full(
    subjects: List[SubjectData],
    x: torch.Tensor,
    *,
    eta_cache: Dict[int, torch.Tensor],
    inner_unroll_steps: int = 12,
    inner_tol: float = 1e-6,        # <-- NEW
    inner_damping: float = 1.0,
    logdet_jitter: float = 1e-6,
    floor_rel: float = 1e-3,
    floor_abs: float = 1e-6,
    max_step_norm: float = 2.0,
) -> torch.Tensor:
    """FULL objective via stabilized unrolled Newton steps."""
    L = torch.zeros((), dtype=x.dtype, device=x.device)

    for subj in subjects:
        eta0 = eta_cache.get(
            subj.sid,
            torch.zeros(ETA_DIM, dtype=x.dtype, device=x.device),
        )

        eta_hat = newton_eta_full_unroll(
            subj, x, eta0,
            unroll_steps=inner_unroll_steps,
            tol=inner_tol,
            damping=inner_damping,
            jitter=logdet_jitter,
            floor_rel=floor_rel,
            floor_abs=floor_abs,
            max_step_norm=max_step_norm,
        )
        eta_cache[subj.sid] = eta_hat.detach()

        # Laplace term at eta_hat (full graph here, as before)
        g_eta, H_eta, hi = grad_hess_eta_ad(
            subj, x, eta_hat,
            create_graph_hess=True,
            retain_graph=True,
        )

        logdet, Hpd, Lchol = stable_logdet_and_chol(
            H_eta, jitter=logdet_jitter, floor_rel=floor_rel, floor_abs=floor_abs
        )

        L = L + hi + 0.5 * logdet

    return 2.0 * L


# Keep your alias consistent
focei_objective_full_unroll = focei_objective_full


def grad_hess_eta_ad(
    subj: SubjectData,
    x: torch.Tensor,
    eta: torch.Tensor,
    *,
    create_graph_hess: bool,
    retain_graph: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (g, H, hi) where g = d hi / d eta and H = d^2 hi / d eta^2.

    Autograd gotcha:
      To compute Hessians via autograd, the *first* derivative must be computed with
      create_graph=True; otherwise g has no grad_fn and you get:
        'element 0 of tensors does not require grad ...'

    Another gotcha (the one that caused your 'backward through the graph a second time'):
      If we call autograd.grad() to build H and do NOT retain the graph, PyTorch will
      free saved tensors. Later, when we compute dL/dtheta, PyTorch may need those saved tensors
      and throws the 'second time' error. So for outer differentiation we keep retain_graph=True.
    """
    hi = h_i(subj, x, eta)

    # Always create graph for the first derivative so second derivatives exist
    g = torch.autograd.grad(hi, eta, create_graph=True, retain_graph=True)[0]

    q = g.numel()
    H_rows = []
    for j in range(q):
        Hj = torch.autograd.grad(
            g[j],
            eta,
            create_graph=create_graph_hess,
            retain_graph=True if retain_graph else True,  # keep True (q is tiny)
        )[0]
        H_rows.append(Hj)
    H = torch.stack(H_rows, dim=0)

    if not create_graph_hess:
        g = g.detach()
        H = H.detach()

    return g, H, hi

def newton_eta_stop(
    subj: SubjectData,
    x_detached: torch.Tensor,
    eta0: torch.Tensor,
    *,
    max_iter: int = 25,
    tol: float = 1e-4,
    damping: float = 1.0,
    jitter: float = 1e-6,
    floor_rel: float = 0.1,
    floor_abs: float = 1e-3,
    max_step_norm: float = 5.0,
) -> torch.Tensor:
    """Damped Newton solve for eta with theta treated as constant (STOP inner).

    Key stability features:
      - Levenberg-Marquardt damping via _solve_newton_damped (prevents enormous steps)
      - Backtracking line search on h_i (prevents divergence when curvature is bad)
    """
    eta = eta0.detach().clone()

    with torch.no_grad():
        hi_prev = _safe_float(h_i(subj, x_detached, eta))

    for _ in range(int(max_iter)):
        eta_t = eta.detach().clone().requires_grad_(True)
        g, H, hi = grad_hess_eta_ad(
            subj, x_detached, eta_t,
            create_graph_hess=False,
            retain_graph=False,
        )
        hi_val = _safe_float(hi)

        # Newton step (solve H step = g)
        step = _solve_newton_damped(
            H, g,
            jitter=jitter,
            floor_rel=floor_rel,
            floor_abs=floor_abs,
            max_step_norm=max_step_norm,
        )
        step_norm = float(torch.norm(step).detach().cpu().item())
        if step_norm < float(tol):
            break

        direction = -float(damping) * step

        # backtracking line search on the inner objective h_i
        alpha = 1.0
        accepted = False
        for _ls in range(12):
            eta_new = eta + alpha * direction
            with torch.no_grad():
                hi_new = _safe_float(h_i(subj, x_detached, eta_new))
            if np.isfinite(hi_new) and (hi_new < hi_val):
                eta = eta_new.detach()
                hi_prev = hi_new
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            # No improvement even with heavy step-halving: stop early.
            break

    return eta.detach()



# ---------- FD inner derivatives (NONMEM-like) ----------

def grad_hess_eta_fd(
    subj: SubjectData,
    x_detached: torch.Tensor,
    eta: torch.Tensor,
    eps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    # print("DEBUG grad_hess_eta_fd eps:", eps, "type:", type(eps))
    if not isinstance(eps, (float, int)):
        # log once so we can find the caller passing bad type
        import warnings
        warnings.warn(f"grad_hess_eta_fd called with eps of wrong type: {type(eps)}. Forcing eps=1e-3")
        eps = 1e-3
    else:
        eps = float(eps)
    """Finite-difference gradient + Hessian of h_i wrt eta at fixed theta."""
    q = int(eta.numel())
    eta_np = eta.detach().cpu().numpy().astype(float)

    def f(eta_vec: np.ndarray) -> float:
        #et = torch.tensor(eta_vec, dtype=torch.float64)
        et = torch.tensor(eta_vec, dtype=torch.float64, device=x_detached.device)
        val = _safe_float(h_i(subj, x_detached, et))
        return float(val)

    f0 = f(eta_np)
    g = np.zeros(q, dtype=float)
    H = np.zeros((q, q), dtype=float)

    # gradient
    for j in range(q):
        ej = np.zeros(q); ej[j] = 1.0
        fp = f(eta_np + eps*ej)
        fm = f(eta_np - eps*ej)
        g[j] = (fp - fm) / (2*eps)

    # Hessian diag
    for j in range(q):
        ej = np.zeros(q); ej[j] = 1.0
        fp = f(eta_np + eps*ej)
        fm = f(eta_np - eps*ej)
        H[j, j] = (fp - 2*f0 + fm) / (eps**2)

    # Hessian off-diag
    for j in range(q):
        for k in range(j+1, q):
            ej = np.zeros(q); ej[j] = 1.0
            ek = np.zeros(q); ek[k] = 1.0
            fpp = f(eta_np + eps*ej + eps*ek)
            fpm = f(eta_np + eps*ej - eps*ek)
            fmp = f(eta_np - eps*ej + eps*ek)
            fmm = f(eta_np - eps*ej - eps*ek)
            Hjk = (fpp - fpm - fmp + fmm) / (4*eps**2)
            H[j, k] = Hjk
            H[k, j] = Hjk

    return g, H

def newton_eta_fd(
    subj: SubjectData,
    x_detached: torch.Tensor,
    eta0: torch.Tensor,
    *,
    max_iter: int = 25,
    tol: float = 1e-4,
    damping: float = 1.0,
    jitter: float = 1e-6,
    eps_eta: float = 1e-3,
    floor_rel: float = 1e-2,
    floor_abs: float = 1e-3,
    max_step_norm: float = 5.0,
    ls_max_backtracks: int = 8,
) -> torch.Tensor:
    """
    Robust Newton solve for eta using FD gradient/Hessian (NONMEM-like inner derivatives).

    Why this is needed:
      - With FD Hessians, H can be indefinite/noisy.
      - A raw Newton step can explode eta, leading to overflow (exp) and NaNs/Infs.
      - Once the objective becomes non-finite, the outer FD gradient becomes zero
        (all perturbed evaluations hit the same penalty), and L-BFGS-B terminates.

    Stabilization strategy (mirrors STOP/FULL):
      1) Compute g,H by FD.
      2) Solve a damped/shifted Newton system via _solve_newton_damped (PD floor).
      3) Clip the step norm (trust region).
      4) Backtracking line-search on h_i to ensure descent.
    """
    eta = eta0.detach().clone()

    # Current conditional objective value (no Laplace term)
    f_curr = float(h_i(subj, x_detached, eta).detach().cpu().item())
    if not np.isfinite(f_curr):
        f_curr = 1e100

    for _ in range(int(max_iter)):
        g_np, H_np = grad_hess_eta_fd(subj, x_detached, eta, eps=eps_eta)
        #eps = float(eps)
        # Convert to torch and compute a stabilized Newton step
        g = torch.tensor(g_np, dtype=torch.float64)
        H = torch.tensor(H_np, dtype=torch.float64)

        step = _solve_newton_damped(
            H, g,
            jitter=float(jitter),
            floor_rel=float(floor_rel),
            floor_abs=float(floor_abs),
            max_step_norm=float(max_step_norm),
        )

        # If the step is tiny, we are done
        step_norm = float(torch.norm(step).detach().cpu().item())
        if step_norm < float(tol):
            break

        # Backtracking line-search in eta-space (conditional objective)
        alpha = float(damping)
        accepted = False
        for _bt in range(int(ls_max_backtracks) + 1):
            eta_try = eta - alpha * step
            f_try = float(h_i(subj, x_detached, eta_try).detach().cpu().item())
            if np.isfinite(f_try) and (f_try <= f_curr):
                eta = eta_try
                f_curr = f_try
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            # If we cannot find descent, take a very small step anyway (prevents stalling)
            eta_try = eta - 0.1 * step
            f_try = float(h_i(subj, x_detached, eta_try).detach().cpu().item())
            if np.isfinite(f_try):
                eta = eta_try
                f_curr = f_try
            else:
                break

        if float(alpha * step_norm) < float(tol):
            break

    return eta.detach()



# %% [code] cell 8
# =========================
# FOCEI / Laplace objectives
# =========================





def focei_objective_stop_value(
    subjects: List[SubjectData],
    x_np: np.ndarray,
    *,
    inner_max_iter: int = 25,
    inner_tol: float = 1e-6,
    inner_damping: float = 1.0,
    logdet_jitter: float = 1e-3,
    floor_rel: float = 1e-5,
    floor_abs: float = 1e-8,
    max_step_norm: float = 5.0,
) -> float:
    """Value-only STOP objective (no theta gradients). Returns OFV-like value (~ NONMEM scale)."""
    x_det = torch.tensor(np.asarray(x_np, dtype=float), dtype=torch.float64)

    L = torch.zeros((), dtype=torch.float64)
    eta_cache: Dict[int, torch.Tensor] = {}

    for subj in subjects:
        eta0 = eta_cache.get(subj.sid, torch.zeros(ETA_DIM, dtype=torch.float64))
        eta_hat = newton_eta_stop(
            subj, x_det, eta0,
            max_iter=inner_max_iter, tol=inner_tol, damping=inner_damping,
            jitter=logdet_jitter, floor_rel=floor_rel, floor_abs=floor_abs, max_step_norm=max_step_norm,
        )
        eta_cache[subj.sid] = eta_hat.detach()

        # evaluate hi + 0.5 logdet(H) at eta_hat (no outer grads needed)
        eta_leaf = eta_hat.detach().clone().requires_grad_(True)
        g, H, hi = grad_hess_eta_ad(
            subj, x_det, eta_leaf,
            create_graph_hess=False,
            retain_graph=False,
        )
        #logdet = _logdet_sym_posdef(H, jitter=logdet_jitter)

        logdet, Hpd, Lchol = stable_logdet_and_chol(
            H, jitter=logdet_jitter, floor_rel=floor_rel, floor_abs=floor_abs
        )
        L_i = hi + 0.5 * logdet

        L = L + L_i #hi.detach() + 0.5 * logdet.detach()

    # Return on NONMEM-like (-2 loglik) scale
    return float(2.0 * L.detach().cpu().item())

def focei_objective_stop(
    subjects: List[SubjectData],
    x: torch.Tensor,
    *,
    eta_cache: Dict[int, torch.Tensor],
    inner_max_iter: int = 25,
    inner_tol: float = 1e-4,
    inner_damping: float = 1.0,
    logdet_jitter: float = 1e-6,
    floor_rel: float = 0.1,
    floor_abs: float = 1e-3,
    max_step_norm: float = 5.0,
) -> torch.Tensor:
    """STOP objective with theta gradients (eta-hats computed with theta detached)."""
    x_det = x.detach()
    #L = torch.zeros((), dtype=torch.float64)
    L = torch.zeros((), dtype=x.dtype, device=x.device)

    for subj in subjects:
        eta0 = eta_cache.get(subj.sid, torch.zeros(ETA_DIM, dtype=torch.float64))

        eta_hat = newton_eta_stop(
            subj, x_det, eta0,
            max_iter=inner_max_iter,
            tol=inner_tol,
            damping=inner_damping,
            jitter=logdet_jitter,
            floor_rel=floor_rel,
            floor_abs=floor_abs,
            max_step_norm=max_step_norm,
        )
        eta_cache[subj.sid] = eta_hat.detach()

        # Evaluate objective with x (grad-enabled), eta treated constant
        eta_leaf = eta_hat.detach().clone().requires_grad_(True)
        g, H, hi = grad_hess_eta_ad(
            subj, x, eta_leaf,
            create_graph_hess=True,
            retain_graph=True,
        )
        #logdet = _logdet_sym_posdef(H, jitter=logdet_jitter)

        logdet, Hpd, Lchol = stable_logdet_and_chol(
            H, jitter=logdet_jitter, floor_rel=floor_rel, floor_abs=floor_abs
        )
        L_i = hi + 0.5 * logdet

        L = L + L_i #hi + 0.5 * logdet

    return 2.0 * L  # return OFV-like scale


# Backwards-compatible alias used by run_optimization_notebook
focei_objective_full_unroll = focei_objective_full

def focei_objective_fd_value(
    subjects: List[SubjectData],
    x_np: np.ndarray,
    *,
    eta_cache: Dict[int, torch.Tensor],
    inner_max_iter: int = 25,
    inner_tol: float = 1e-4,
    inner_damping: float = 1.0,
    logdet_jitter: float = 1e-6,
    eps_eta: float = 1e-3,
    # FD-inner stabilization knobs (match STOP/FULL as much as possible)
    floor_rel: float = 1e-2,
    floor_abs: float = 1e-3,
    max_step_norm: float = 5.0,
) -> float:
    """
    Value-only FOCEI/Laplace objective for the FD baseline:

      - Inner eta mode-finding uses FD derivatives (g,H).
      - Outer theta derivatives will also be computed by FD in run_optimization_notebook.
      - We return 2*L to roughly match NONMEM's -2 log-likelihood scaling.

    Notes:
      * If any numerical failure occurs, return np.nan so the caller can apply a
        soft penalty (instead of returning a huge finite sentinel which causes
        the outer FD gradient to become exactly zero).
    """
    x_detached = torch.tensor(np.array(x_np, dtype=float), dtype=torch.float64)

    L = 0.0
    for subj in subjects:
        eta0 = eta_cache.get(subj.sid, torch.zeros(ETA_DIM, dtype=torch.float64))

        eta_hat = newton_eta_fd(
            subj, x_detached, eta0,
            max_iter=inner_max_iter,
            tol=inner_tol,
            damping=inner_damping,
            jitter=logdet_jitter,
            eps_eta=eps_eta,#float(1e-3),  # you can tune this; start with 1e-3
            floor_rel=floor_rel,
            floor_abs=floor_abs,
            max_step_norm=max_step_norm,
        )


        eta_cache[subj.sid] = eta_hat.detach()

        # objective value at eta_hat
        eta_leaf = eta_hat.detach().clone().requires_grad_(True)
        hi = h_i(subj, x_detached, eta_leaf).detach().cpu().item()

        # curvature / logdet term via FD Hessian at eta_hat
        #_, H_np = grad_hess_eta_fd(subj, x_detached, eta_leaf, eps=eps_eta)
        #H_np = 0.5 * (H_np + H_np.T)  # symmetrize
        #Hj = H_np + float(logdet_jitter) * np.eye(3)
        #sign, ld = np.linalg.slogdet(Hj)
        #if (not np.isfinite(ld)) or (sign <= 0):
        #    ld = 1e6  # keep going, but penalize

        #L += float(hi) + 0.5 * float(ld)
        _, H = grad_hess_eta_fd(subj, x_detached, eta_leaf, eps=eps_eta)
        H = 0.5 * (H + H.T)

        # --- PD floor (mirrors your torch PD-floor logic) ---
        w = np.linalg.eigvalsh(H)
        min_eig = float(w.min())
        scale = float(max(1.0, np.max(np.abs(w))))
        target = max(float(floor_abs), float(floor_rel) * scale)

        shift = max(0.0, target - min_eig) + float(logdet_jitter)
        Hpd = H + shift * np.eye(H.shape[0])

        # --- logdet via Cholesky (robust) ---
        ld = None
        for k in range(6):
            try:
                Lc = np.linalg.cholesky(Hpd)
                ld = 2.0 * float(np.sum(np.log(np.diag(Lc))))
                break
            except np.linalg.LinAlgError:
                Hpd = Hpd + (10.0**k) * float(logdet_jitter) * np.eye(H.shape[0])

        if ld is None or (not np.isfinite(ld)):
            ld = 1e6  # last resort fallback

        L = L + hi + 0.5 * ld

    if not np.isfinite(L):
        return np.nan
    return float(2.0 * L)





def focei_objective_full_implicit_eval(
    subjects: List[SubjectData],
    x_np: np.ndarray,
    *,
    eta_cache: Dict[int, torch.Tensor],
    inner_max_iter: int = 25,
    inner_tol: float = 1e-4,
    inner_damping: float = 1.0,
    logdet_jitter: float = 1e-6,
    floor_rel: float = 1e-6,
    floor_abs: float = 1e-3,
    max_step_norm: float = 5.0,
    verbose_inner: bool = False,
) -> Tuple[float, np.ndarray]:
    """
    FULL (implicit) evaluation: returns (OFV, grad) as numpy.

    We:
      1) compute eta<U+0302>_i by Newton with x detached (no graph),
      2) evaluate FOCEI objective at (x, eta<U+0302>_i),
      3) compute dOFV/dx using IFT:
            deta<U+0302>/dx = - H^{-1} * (<U+2202>^2 h / <U+2202>eta <U+2202>x)
         and chain it into the curvature term derivative.

    This avoids unrolling Newton steps, but still requires:
      - Hessian wrt eta
      - cross-derivatives wrt (eta, x)
      - gradients of logdet(H) wrt eta and x (third derivatives)
    """
    x = torch.tensor(np.asarray(x_np, dtype=float), dtype=torch.float64, requires_grad=True)
    p = int(x.numel())

    x_det = x.detach()

    total_L_val = 0.0
    total_g = torch.zeros(p, dtype=torch.float64)

    new_cache: Dict[int, torch.Tensor] = {}

    for subj in subjects:
        # ---- 1) inner solve (no graph) ----
        eta0 = eta_cache.get(subj.sid, torch.zeros(ETA_DIM, dtype=torch.float64))
        eta_hat = newton_eta_stop(
            subj, x_det, eta0,
            max_iter=inner_max_iter,
            tol=inner_tol,
            damping=inner_damping,
            jitter=logdet_jitter,
            floor_rel=floor_rel,
            floor_abs=floor_abs,
            max_step_norm=max_step_norm,
        )

        new_cache[subj.sid] = eta_hat.detach()

        # ---- 2) build leaf eta for derivatives at the mode ----
        eta = eta_hat.detach().clone().requires_grad_(True)

        # objective pieces
        g_eta, H, hi = grad_hess_eta_ad(subj, x, eta, create_graph_hess=True, retain_graph=True)
        
        logdet, Hpd, Lchol = stable_logdet_and_chol(
            H, jitter=logdet_jitter, floor_rel=floor_rel, floor_abs=floor_abs
        )

        L_sub = hi + 0.5 * logdet

        dh_dx       = torch.autograd.grad(hi,     x,        retain_graph=True)[0]
        dlogdet_dx  = torch.autograd.grad(logdet, x,        retain_graph=True)[0]
        dlogdet_deta= torch.autograd.grad(logdet, eta, retain_graph=True)[0]

        # v = Hpd^{-1} * dlogdet_deta   (numerically stable solve)
        v = torch.cholesky_solve(dlogdet_deta.unsqueeze(1), Lchol).squeeze(1)

        # J^T v where J = <U+2202>g_eta/<U+2202>x and g_eta = <U+2202>hi/<U+2202>eta
        Jt_v = torch.autograd.grad(g_eta, x, grad_outputs=v.detach(), retain_graph=False)[0]

        g_theta_sub = dh_dx + 0.5 * dlogdet_dx - 0.5 * Jt_v

        
        #logdet = _logdet_sym_posdef(H, jitter=logdet_jitter)
        L_sub = hi + 0.5 * logdet

        total_L_val += float(L_sub.detach().cpu().item())

        # ---- 3) IFT gradient pieces ----
        # partials at fixed eta
        #dh_dx = torch.autograd.grad(hi, x, retain_graph=True)[0]
        #dlogdet_dx = torch.autograd.grad(logdet, x, retain_graph=True)[0]

        # derivative of curvature term wrt eta (third derivatives)
        #dlogdet_deta = torch.autograd.grad(logdet, eta, retain_graph=True)[0]
        # mixed term via VJP (avoid forming full J and V)
        # v = H^{-1} * dlogdet_deta
        #Hpd, Lchol = _stabilize_hessian_for_cholesky(H, jitter=logdet_jitter, floor_rel=floor_rel, floor_abs=floor_abs)
        #v = torch.cholesky_solve(dlogdet_deta.unsqueeze(-1), Lchol).squeeze(-1)  # (q,)

        # J^T v where J = d g_eta / d x
        #Jt_v = torch.autograd.grad(g_eta, x, grad_outputs=v.detach(), retain_graph=False, allow_unused=True)[0]
        #if Jt_v is None:
        #    Jt_v = torch.zeros_like(x)

        # chain term: 0.5 * (<U+2202> logdet / <U+2202>eta)^T * d eta_hat / d x
        #chain = -Jt_v  # (p,)

        
        
        ETA_BAD = 1e-1  # tune
        if torch.linalg.norm(g_eta.detach()) > ETA_BAD:
            # fallback: pretend eta is fixed (STOP-like) for this subject
            g_theta_sub = dh_dx + 0.5 * dlogdet_dx


        g_sub = g_theta_sub#dh_dx + 0.5 * dlogdet_dx + 0.5 * chain

        total_g = total_g + g_sub.detach()

    # update warm-start cache only with the base-point eta<U+0302> (not with +/- FD evaluations)
    eta_cache.clear()
    eta_cache.update(new_cache)

    ofv = 2.0 * total_L_val
    grad = (2.0 * total_g).detach().cpu().numpy().astype(float)
    return float(ofv), grad

# %% [code] cell 9
# =========================
# Utilities
# =========================

def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)

def _as_float(x, default=0.0):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default

def _sym(H: torch.Tensor) -> torch.Tensor:
    return 0.5 * (H + H.T)

def _solve_newton_damped(
    H: torch.Tensor,
    g: torch.Tensor,
    *,
    jitter: float = 1e-6,
    floor_rel: float = 0.1,
    floor_abs: float = 1e-3,
    max_step_norm: float = 5.0,
) -> torch.Tensor:
    q = H.shape[0]
    Hs = 0.5 * (H + H.T)

    try:
        Hpd, Lchol, ok, _ = stabilize_for_cholesky(
            Hs, jitter=jitter, floor_rel=floor_rel, floor_abs=floor_abs
        )

        if ok:
            step = torch.cholesky_solve(g.unsqueeze(-1), Lchol).squeeze(-1)
        else:
            step = torch.linalg.solve(Hpd, g.unsqueeze(-1)).squeeze(-1)

    except Exception:
        # fallback: gradient direction
        step = g.clone()

    # clamp huge steps
    n = torch.norm(step)
    n_det = n.detach()
    if torch.isfinite(n_det) and float(n_det) > float(max_step_norm):
        step = step * (float(max_step_norm) / (float(n_det) + 1e-12))

    return step

def _logdet_sym_posdef(
    H: torch.Tensor,
    *,
    jitter: float = 1e-6,
    max_tries: int = 8,
    jitter_mult: float = 10.0,
    min_eig_floor: float = 1e-10,
) -> torch.Tensor:
    """Robust logdet for symmetric PD-ish matrices.

    Tries Cholesky with escalating jitter; if that fails, shifts eigenvalues so the
    minimum is at least min_eig_floor. The shift is detached so stabilization does not
    dominate outer gradients.
    """
    q = H.shape[0]
    I = torch.eye(q, dtype=H.dtype, device=H.device)
    Hs = _sym(H)

    # Cholesky with escalating jitter
    for k in range(int(max_tries)):
        j = float(jitter) * (float(jitter_mult) ** k)
        try:
            L = torch.linalg.cholesky(Hs + j * I)
            return 2.0 * torch.sum(torch.log(torch.diag(L)))
        except RuntimeError:
            pass

    # Eigen shift fallback
    eigs = torch.linalg.eigvalsh(Hs)
    min_eig = torch.min(eigs)

    floor = torch.as_tensor(min_eig_floor, dtype=Hs.dtype, device=Hs.device)
    zero = torch.as_tensor(0.0, dtype=Hs.dtype, device=Hs.device)
    shift = torch.clamp(floor - min_eig, min=zero)
    shift = (shift + torch.as_tensor(jitter, dtype=Hs.dtype, device=Hs.device)).detach()

    eigs_pd = eigs + shift
    return torch.sum(torch.log(eigs_pd))

def _safe_float(x):
    """
    Safely convert tensor or array-like to float.
    """
    try:
        if hasattr(x, "detach"):
            return float(x.detach().cpu().item())
        return float(x)
    except Exception:
        return float("nan")

# %% [code] cell 10

# =========================
# Outer optimization wrapper (stable-cache + robust penalties)
# =========================
#
# Key fixes vs earlier versions:
#   1) Do NOT update eta_cache during line-search evaluations.
#      (Line search may probe many points; mutating warm-starts makes the objective path-dependent.)
#      We use a local copy per evaluation and only "accept" the cache in the SciPy callback.
#   2) If we hit NaN/Inf, return a *differentiable quadratic penalty* with a non-zero gradient
#      pointing back to the last finite point (instead of f=1e100, g=0).
#   3) FD: add progress prints + option to run FD on a subset to avoid "it looks stuck".


from tabnanny import verbose


class OptimizationResult:
    def __init__(self, x_opt, success, message, niter, ofv):
        self.x_opt = x_opt
        self.success = success
        self.message = message
        self.niter = niter
        self.ofv = ofv


def projected_grad_inf(x, g, bounds, eps=1e-12):
    pg = g.copy()
    if bounds is None:
        return float(np.max(np.abs(pg)))
    for j, (lo, hi) in enumerate(bounds):
        if x[j] <= lo + eps:
            pg[j] = min(pg[j], 0.0)   # only allow movement into feasible region
        elif x[j] >= hi - eps:
            pg[j] = max(pg[j], 0.0)
    return float(np.max(np.abs(pg)))

def run_optimization_notebook(
    subjects: List[SubjectData],
    *,
    method: str,
    x0: np.ndarray,
    max_outer_iter: int = 60,
    # inner params
    inner_max_iter: int = 25,
    inner_tol: float = 1e-4,
    inner_damping: float = 1.0,
    # FULL_UNROLL controls
    full_unroll_steps: int = 20,
    full_damping: float = 1.0,
    # numerical
    logdet_jitter: float = 1e-6,
    hess_floor_rel: float = 1e-5,
    hess_floor_abs: float = 1e-8,
    max_step_norm: float = 10.0,
    # FD steps
    fd_eps_outer: float = 1e-3,
    fd_eps_eta: float = 1e-3,
    # verbosity
    verbose: bool = True,
    # optional bounds
    bounds: Optional[List[Tuple[float,float]]] = None,
    # FD controls
    fd_max_subjects: Optional[int] = None,
    fd_outer_scheme: str = "central",
    # outer minimization options
    gtol: float = 1e-4,
    ftol: float = 1e-8,
    maxls: int = 10,
    max_runtime_sec: Optional[float] = None,
    deadline_perf_counter: Optional[float] = None,
):
    """
    Returns (scipy_result, out_dict) where out_dict contains trace, runtime, etc.
    """
    method = method.upper()
    x0 = np.array(x0, dtype=float)
    trace_f = []
    trace_gnorm = []
    t_start = time.perf_counter()
    if deadline_perf_counter is None and max_runtime_sec is not None:
        deadline_perf_counter = t_start + float(max_runtime_sec)

    # Stable warm-start cache: only updated at accepted iterates via callback
    eta_cache_accepted: Dict[int, torch.Tensor] = {}
    last_eval_cache: Optional[Dict[int, torch.Tensor]] = None
    last_eval_x: Optional[np.ndarray] = None

    # For robust penalties, remember last finite evaluation
    last_finite_x = x0.copy()
    last_finite_f = np.inf

    last_eval_f = None
    last_eval_gnorm = None

    best_f = np.inf
    best_x = None

    # Penalty scaling: typical OFV ~ 1e4-1e5, so 1e8 is "large but not astronomically huge"
    PEN_BASE = 1e8
    PEN_QUAD = 1e4

    def _penalty(z: np.ndarray, ref: np.ndarray) -> Tuple[float, np.ndarray]:
        dz = (z - ref).astype(float)
        f = float(PEN_BASE + PEN_QUAD * float(np.dot(dz, dz)))
        g = (2.0 * PEN_QUAD) * dz
        return f, g

    def _check_time_limit(where: str = "") -> None:
        if deadline_perf_counter is None:
            return
        if time.perf_counter() >= deadline_perf_counter:
            where_txt = f" during {where}" if where else ""
            raise MethodTimeLimitReached(
                f"Stopped after reaching the {max_runtime_sec / 3600.0:.3f}-hour wall-time limit{where_txt}."
            )

    # -----------------------
    # STOP + FULL_UNROLL: AD outer gradient, stable cache
    # -----------------------
    if method in ["STOP", "FULL_UNROLL"]:

        def eval_f_g(z: np.ndarray) -> Tuple[float, np.ndarray]:
            nonlocal last_eval_cache, last_eval_x, last_finite_x, last_finite_f
            nonlocal last_eval_f, last_eval_gnorm, best_f, best_x

            _check_time_limit(f"{method} objective evaluation")
            z = np.array(z, dtype=float)
            #if bounds is not None:
            #    lo = np.array([b[0] for b in bounds], dtype=float)
            #    hi = np.array([b[1] for b in bounds], dtype=float)
            #    z = np.minimum(np.maximum(z, lo), hi)


            # Local cache copy for this evaluation (so line-search doesn't mutate accepted cache)
            #eta_cache_local = eta_cache_accepted.copy()
            eta_cache_local = {sid: t.detach().clone() for sid, t in eta_cache_accepted.items()}

            try:
                x = torch.tensor(z, dtype=torch.float64, requires_grad=True)
                if method == "STOP":
                    L = focei_objective_stop(
                        subjects, x,
                        eta_cache=eta_cache_local,
                        inner_max_iter=inner_max_iter,
                        inner_tol=inner_tol,
                        inner_damping=inner_damping,
                        logdet_jitter=logdet_jitter,
                        floor_rel=hess_floor_rel,
                        floor_abs=hess_floor_abs,
                        max_step_norm=max_step_norm,
                    )
                else:
                    L = focei_objective_full_unroll(
                        subjects, x,
                        eta_cache=eta_cache_local,
                        inner_unroll_steps=full_unroll_steps,
                        inner_tol=inner_tol,
                        inner_damping=full_damping,
                        logdet_jitter=logdet_jitter,
                        floor_rel=hess_floor_rel,
                        floor_abs=hess_floor_abs,
                        max_step_norm=max_step_norm,
                    )

                # Backprop once (SciPy with jac=True will memoize)
                L.backward()
                g = x.grad.detach().cpu().numpy().astype(float)
                f = float(L.detach().cpu().item())

            except Exception as e:
                import traceback
                traceback.print_exc()   # prints full stack trace to stderr
                f = np.nan
                g = np.full_like(z, np.nan, dtype=float)
                if verbose:
                    print(f"[{method}] Exception during eval: {type(e).__name__}: {e}")

            # --- ADD HERE: store last eval + best-so-far (seen points) ---
            gnorm = float(np.linalg.norm(g))
            last_eval_f = float(f)
            last_eval_gnorm = gnorm

            if np.isfinite(last_eval_f) and (last_eval_f < best_f):
                best_f = last_eval_f
                best_x = z.copy()


            # Robust non-finite handling
            if (not np.isfinite(f)) or (not np.all(np.isfinite(g))):
                f_pen, g_pen = _penalty(z, last_finite_x)
                f, g = f_pen, g_pen
                if verbose:
                    print(f"[{method}] non-finite -> penalty f={f:.3e}  |g|={np.linalg.norm(g):.3e}")
            else:
                last_finite_x = z.copy()
                last_finite_f = float(f)
                last_eval_cache = eta_cache_local
                last_eval_x = z.copy()
                #if verbose:
                #    print(f"[{method}] VISIT f={f:.6f}  |g|={np.linalg.norm(g):.3e}")
                pginf = projected_grad_inf(z, g, bounds)
                if verbose:
                    print(f"[{method}] VISIT f={f:.6f}  |g|2={np.linalg.norm(g):.3e}  |proj g|inf={pginf:.3e}")


            trace_f.append(float(f))
            trace_gnorm.append(gnorm)
            _check_time_limit(f"{method} objective evaluation")
            return float(f), g.astype(float)


        def callback(xk: np.ndarray):
            nonlocal eta_cache_accepted, last_eval_cache, last_eval_x
            nonlocal last_eval_f, last_eval_gnorm, best_f, best_x
            #nonlocal accepted_iter

            _check_time_limit(f"{method} callback")
            if (last_eval_cache is not None) and (last_eval_x is not None):
                xk = np.array(xk, dtype=float)

                if np.allclose(xk, last_eval_x, atol=1e-7, rtol=0.0):
                    eta_cache_accepted = last_eval_cache

                    #accepted_iter += 1

                    # best among *accepted* iterates
                    if (last_eval_f is not None) and np.isfinite(last_eval_f) and (last_eval_f < best_f):
                        best_f = float(last_eval_f)
                        best_x = xk.copy()

                    #if print_accepted_only:
                    print(f"[{method}] ACCEPT f={last_eval_f:.6f}  |g|={last_eval_gnorm:.3e}  best_f={best_f:.6f}")


        
        try:
            res = minimize(
                fun=eval_f_g,
                x0=x0,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                callback=callback,
                options={"maxiter": int(max_outer_iter),
                         "ftol": ftol,  # disable f-based stopping (we have a custom penalty handling)
                         "gtol": gtol, 
                         "maxls": maxls, # Maximum number of line search steps per iteration. The default is 20. 
                         },
            )
        except MethodTimeLimitReached as exc:
            x_timeout = best_x.copy() if best_x is not None else last_finite_x.copy()
            fun_timeout = best_f if np.isfinite(best_f) else float(last_finite_f)
            res = OptimizeResult(
                x=x_timeout,
                fun=fun_timeout,
                success=False,
                status=1,
                message=str(exc),
                nit=len(trace_f),
                nfev=len(trace_f),
                timed_out=True,
            )
        x_hat = res.x.copy()
        final_L = float(res.fun)

        def check_inner_eta_quality(subjects, x_np, eta_cache, *,
                            inner_max_iter=25, inner_tol=1e-4, inner_damping=1e-3,
                            floor_rel=1e-5, floor_abs=1e-8, max_step_norm=1.0,
                            ncheck=5):

            x = torch.tensor(x_np, dtype=torch.float64, device=DEVICE)
            rows = []

            for subj in subjects[:ncheck]:
                eta0 = eta_cache[subj.sid]
                eta_hat = newton_eta_stop(
                    subj, x, eta0,
                    max_iter=inner_max_iter,
                    tol=inner_tol,
                    damping=inner_damping,
                    floor_rel=floor_rel,
                    floor_abs=floor_abs,
                    max_step_norm=max_step_norm,
                )

                eta = eta_hat.detach().clone().requires_grad_(True)
                g_eta, H_eta, hi = grad_hess_eta_ad(subj, x, eta, create_graph_hess=False, retain_graph=False)
                Hs = 0.5 * (H_eta + H_eta.T)
                eigs = torch.linalg.eigvalsh(Hs).detach().cpu().numpy()

                rows.append({
                    "sid": subj.sid,
                    "||g_eta||": float(torch.linalg.norm(g_eta.detach()).cpu()),
                    "min_eig(H)": float(eigs.min()),
                    "max_eig(H)": float(eigs.max()),
                })

            rows = sorted(rows, key=lambda r: r["||g_eta||"], reverse=True)
            for r in rows:
                print(r)


    # -----------------------
    # FULL_IMPLICIT: implicit differentiation outer gradient, stable cache
    # -----------------------
    elif method == "FULL_IMPLICIT":

        def eval_f_g(z: np.ndarray) -> Tuple[float, np.ndarray]:
            nonlocal last_eval_cache, last_eval_x, last_finite_x, last_finite_f
            nonlocal last_eval_f, last_eval_gnorm, best_f, best_x

            _check_time_limit(f"{method} objective evaluation")
            z = np.array(z, dtype=float)
            #if bounds is not None:
            #    lo = np.array([b[0] for b in bounds], dtype=float)
            #    hi = np.array([b[1] for b in bounds], dtype=float)
            #    z = np.minimum(np.maximum(z, lo), hi)


            #eta_cache_local = eta_cache_accepted.copy()
            eta_cache_local = {sid: t.detach().clone() for sid, t in eta_cache_accepted.items()}

            try:
                f, g = focei_objective_full_implicit_eval(
                    subjects, z,
                    eta_cache=eta_cache_local,
                    inner_max_iter=inner_max_iter,
                    inner_tol=inner_tol,
                    inner_damping=inner_damping,
                    logdet_jitter=logdet_jitter,
                    floor_rel=hess_floor_rel,
                    floor_abs=hess_floor_abs,
                    max_step_norm=max_step_norm,
                )
                f = float(f)
                g = np.array(g, dtype=float)
            except Exception as e:
                import traceback
                traceback.print_exc()   # prints full stack trace to stderr
                f = np.nan
                g = np.full_like(z, np.nan, dtype=float)
                if verbose:
                    print(f"[{method}] Exception during eval: {type(e).__name__}: {e}")

            if (not np.isfinite(f)) or (not np.all(np.isfinite(g))):
                f_pen, g_pen = _penalty(z, last_finite_x)
                f, g = f_pen, g_pen
                if verbose:
                    print(f"[FULL_IMPLICIT] non-finite -> penalty f={f:.3e}  |g|={np.linalg.norm(g):.3e}")
            else:
                last_finite_x = z.copy()
                last_finite_f = float(f)
                last_eval_cache = eta_cache_local
                last_eval_x = z.copy()
                #if verbose:
                #    print(f"[FULL_IMPLICIT] VISIT f={f:.6f}  |g|={np.linalg.norm(g):.3e}")
                pginf = projected_grad_inf(z, g, bounds)
                if verbose:
                    print(f"[{method}] VISIT f={f:.6f}  |g|2={np.linalg.norm(g):.3e}  |proj g|inf={pginf:.3e}")
                    

            # --- ADD HERE ---
            gnorm = float(np.linalg.norm(g))
            last_eval_f = float(f)
            last_eval_gnorm = gnorm
            if np.isfinite(last_eval_f) and (last_eval_f < best_f):
                best_f = last_eval_f
                best_x = z.copy()
            # --- END ADD ---

            trace_f.append(float(f))
            trace_gnorm.append(gnorm)
            _check_time_limit(f"{method} objective evaluation")
            return float(f), g.astype(float)


        def callback(xk: np.ndarray):
            nonlocal eta_cache_accepted, last_eval_cache, last_eval_x
            nonlocal last_eval_f, last_eval_gnorm, best_f, best_x
            #nonlocal accepted_iter

            _check_time_limit(f"{method} callback")
            if (last_eval_cache is not None) and (last_eval_x is not None):
                xk = np.array(xk, dtype=float)

                if np.allclose(xk, last_eval_x, atol=1e-7, rtol=0.0):
                    eta_cache_accepted = last_eval_cache

                    #accepted_iter += 1

                    # best among *accepted* iterates
                    if (last_eval_f is not None) and np.isfinite(last_eval_f) and (last_eval_f < best_f):
                        best_f = float(last_eval_f)
                        best_x = xk.copy()

                    #if print_accepted_only:
                    print(f"[{method}] ACCEPT f={last_eval_f:.6f}  |g|={last_eval_gnorm:.3e}  best_f={best_f:.6f}")


        try:
            res = minimize(
                fun=eval_f_g,
                x0=x0,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                callback=callback,
                options={"maxiter": int(max_outer_iter),
                         "ftol": ftol,  # disable f-based stopping (we have a custom penalty handling)
                         "gtol": gtol, 
                         "maxls": maxls, # Maximum number of line search steps per iteration. The default is 20. 
                         },
            )
        except MethodTimeLimitReached as exc:
            x_timeout = best_x.copy() if best_x is not None else last_finite_x.copy()
            fun_timeout = best_f if np.isfinite(best_f) else float(last_finite_f)
            res = OptimizeResult(
                x=x_timeout,
                fun=fun_timeout,
                success=False,
                status=1,
                message=str(exc),
                nit=len(trace_f),
                nfev=len(trace_f),
                timed_out=True,
            )
        x_hat = res.x.copy()
        final_L = float(res.fun)

        def check_inner_eta_quality(subjects, x_np, eta_cache, *,
                            inner_max_iter=25, inner_tol=1e-4, inner_damping=1e-3,
                            floor_rel=1e-5, floor_abs=1e-8, max_step_norm=1.0,
                            ncheck=5):

            x = torch.tensor(x_np, dtype=torch.float64, device=DEVICE)
            rows = []

            for subj in subjects[:ncheck]:
                eta0 = eta_cache[subj.sid]
                eta_hat = newton_eta_stop(
                    subj, x, eta0,
                    max_iter=inner_max_iter,
                    tol=inner_tol,
                    damping=inner_damping,
                    floor_rel=floor_rel,
                    floor_abs=floor_abs,
                    max_step_norm=max_step_norm,
                )

                eta = eta_hat.detach().clone().requires_grad_(True)
                g_eta, H_eta, hi = grad_hess_eta_ad(subj, x, eta, create_graph_hess=False, retain_graph=False)
                Hs = 0.5 * (H_eta + H_eta.T)
                eigs = torch.linalg.eigvalsh(Hs).detach().cpu().numpy()
                rows.append({
                    "sid": subj.sid,
                    "||g_eta||": float(torch.linalg.norm(g_eta.detach()).cpu()),
                    "min_eig(H)": float(eigs.min()),
                    "max_eig(H)": float(eigs.max()),
                })

            rows = sorted(rows, key=lambda r: r["||g_eta||"], reverse=True)
            for r in rows:
                print(r)


    # -----------------------
    # MIXED: STOP warm-start -> FULL_IMPLICIT polish
    # -----------------------
    elif method == "STOP+FULL":
        
        if verbose:
            print("\n==============================")
            print("Running method: MIXED (STOP -> FULL_IMPLICIT)")
            print("==============================")
        
        # ---- Stage 1: STOP ----
        res_stop, out_stop = run_optimization_notebook(
            subjects,
            method="STOP",
            x0=x0,
            max_outer_iter=int(1),

            # inner params
            inner_max_iter=inner_max_iter,
            inner_tol=inner_tol,
            inner_damping=inner_damping,

            # numerical
            logdet_jitter=logdet_jitter,
            hess_floor_rel=hess_floor_rel,
            hess_floor_abs=hess_floor_abs,
            max_step_norm=max_step_norm,

            # bounds + verbosity
            bounds=bounds,
            verbose=verbose,

            # outer options for stage 1
            gtol=float(gtol_STOP),
            ftol=float(ftol_STOP),
            maxls=int(5),
            max_runtime_sec=max_runtime_sec,
            deadline_perf_counter=deadline_perf_counter,

            # pass initial cache (if any), and request cache back
            #eta_cache_init=eta_cache_init,
            #return_eta_cache=True,
        )

        x1 = np.array(out_stop["final_x"], dtype=float)
        #cache1 = out_stop.get("_eta_cache_final", None)

        if verbose:
            print(f"[STOP+FULL] Stage 1 done: STOP final_f={out_stop['final_f']:.6f}, niter={out_stop['niter']}, nfev={out_stop['nfev']}")

        if out_stop.get("timed_out", False):
            out_mixed = dict(out_stop)
            out_mixed["method"] = "STOP+FULL"
            out_mixed["success"] = False
            out_mixed["message"] = f"STOP warm start timed out: {out_stop.get('message', '')}"
            out_mixed["timed_out"] = True
            return res_stop, out_mixed

        # ---- Stage 2: FULL_IMPLICIT ----
        res_full, out_full = run_optimization_notebook(
            subjects,
            method="FULL_IMPLICIT",
            x0=x1,
            max_outer_iter=int(max_outer_iter),

            # inner params
            inner_max_iter=inner_max_iter,
            inner_tol=inner_tol,
            inner_damping=inner_damping,

            # numerical
            logdet_jitter=logdet_jitter,
            hess_floor_rel=hess_floor_rel,
            hess_floor_abs=hess_floor_abs,
            max_step_norm=max_step_norm,

            # bounds + verbosity
            bounds=bounds,
            verbose=verbose,

            # outer options for stage 2
            gtol=float(gtol),
            ftol=float(ftol),
            maxls=int(maxls),
            max_runtime_sec=max_runtime_sec,
            deadline_perf_counter=deadline_perf_counter,

            # warm-start FULL_IMPLICIT from STOP cache
            #eta_cache_init=cache1,
            #return_eta_cache=return_eta_cache,
        )

        # ---- Combine outputs (single out dict) ----
        out_mixed = {
            "method": "STOP+FULL",
            "success": bool(out_full.get("success", False)),
            "message": f"STOP: {out_stop.get('message','')} | FULL_IMPLICIT: {out_full.get('message','')}",
            "niter": int(out_stop.get("niter", 0)) + int(out_full.get("niter", 0)),
            "nfev": int(out_stop.get("nfev", 0)) + int(out_full.get("nfev", 0)),
            "final_f": float(out_full.get("final_f", np.nan)),
            "final_x": out_full.get("final_x", x1.tolist()),
            "runtime_sec": float(out_stop.get("runtime_sec", 0.0)) + float(out_full.get("runtime_sec", 0.0)),
            "trace_f": list(out_stop.get("trace_f", [])) + list(out_full.get("trace_f", [])),
            "trace_gnorm": list(out_stop.get("trace_gnorm", [])) + list(out_full.get("trace_gnorm", [])),
            "timed_out": bool(out_stop.get("timed_out", False) or out_full.get("timed_out", False)),

            # useful audit fields
            "stop_stage_final_f": float(out_stop.get("final_f", np.nan)),
            "stop_stage_final_x": out_stop.get("final_x", None),
            "full_stage_final_f": float(out_full.get("final_f", np.nan)),
            "full_stage_final_x": out_full.get("final_x", None),

            "settings": {
                "mixed_stop_max_outer_iter": int(max_outer_iter),
                "mixed_stop_gtol": float(gtol_STOP),
                "mixed_stop_ftol": float(ftol_STOP),
                "mixed_stop_maxls": int(maxls_STOP),
                "mixed_full_max_outer_iter": int(max_outer_iter),
                "inner_max_iter": inner_max_iter,
                "inner_tol": inner_tol,
                "inner_damping": inner_damping,
                "logdet_jitter": logdet_jitter,
                "floor_rel": hess_floor_rel,
                "floor_abs": hess_floor_abs,
                "max_step_norm": max_step_norm,
            }
        }

        # optionally expose final cache (ONLY if requested)
        #if return_eta_cache:
        #    out_mixed["_eta_cache_final"] = out_full.get("_eta_cache_final", None)

        return res_full, out_mixed

    # -----------------------
    # MIXED: FULL_IMPLICIT warm-start -> STOP polish
    # -----------------------
    elif method == "FULL+STOP":
        
        if verbose:
            print("\n==============================")
            print("Running method: MIXED (FULL_IMPLICIT -> STOP)")
            print("==============================")
        
        # ---- Stage 1: FULL_IMPLICIT ----
        res_full, out_full = run_optimization_notebook(
            subjects,
            method="FULL_IMPLICIT",
            x0=x0,
            max_outer_iter=int(14),

            # inner params
            inner_max_iter=inner_max_iter,
            inner_tol=inner_tol,
            inner_damping=inner_damping,

            # numerical
            logdet_jitter=logdet_jitter,
            hess_floor_rel=hess_floor_rel,
            hess_floor_abs=hess_floor_abs,
            max_step_norm=max_step_norm,

            # bounds + verbosity
            bounds=bounds,
            verbose=verbose,

            # outer options for stage 2
            gtol=float(gtol),
            ftol=float(ftol),
            maxls=int(maxls),
            max_runtime_sec=max_runtime_sec,
            deadline_perf_counter=deadline_perf_counter,

            # warm-start FULL_IMPLICIT from STOP cache
            #eta_cache_init=cache1,
            #return_eta_cache=return_eta_cache,
        )

        x1 = np.array(out_full["final_x"], dtype=float)
        #cache1 = out_stop.get("_eta_cache_final", None)

        if verbose:
            print(f"[FULL+STOP] Stage 1 done: FULL final_f={out_full['final_f']:.6f}, niter={out_full['niter']}, nfev={out_full['nfev']}")

        if out_full.get("timed_out", False):
            out_mixed = dict(out_full)
            out_mixed["method"] = "FULL+STOP"
            out_mixed["success"] = False
            out_mixed["message"] = f"FULL-implicit warm start timed out: {out_full.get('message', '')}"
            out_mixed["timed_out"] = True
            return res_full, out_mixed

        # ---- Stage 1: STOP ----
        res_stop, out_stop = run_optimization_notebook(
            subjects,
            method="STOP",
            x0=x1,
            max_outer_iter=int(max_outer_iter),

            # inner params
            inner_max_iter=inner_max_iter,
            inner_tol=inner_tol,
            inner_damping=inner_damping,

            # numerical
            logdet_jitter=logdet_jitter,
            hess_floor_rel=hess_floor_rel,
            hess_floor_abs=hess_floor_abs,
            max_step_norm=max_step_norm,

            # bounds + verbosity
            bounds=bounds,
            verbose=verbose,

            # outer options for stage 1
            gtol=float(gtol_STOP),
            ftol=float(ftol_STOP),
            maxls=int(maxls),
            max_runtime_sec=max_runtime_sec,
            deadline_perf_counter=deadline_perf_counter,

            # pass initial cache (if any), and request cache back
            #eta_cache_init=eta_cache_init,
            #return_eta_cache=True,
        )



        

        # ---- Combine outputs (single out dict) ----
        out_mixed = {
            "method": "FULL+STOP",
            "success": bool(out_stop.get("success", False)),
            "message": f"FULL_IMPLICIT: {out_full.get('message','')} | STOP: {out_stop.get('message','')}",
            "niter": int(out_full.get("niter", 0)) + int(out_stop.get("niter", 0)),
            "nfev": int(out_stop.get("nfev", 0)) + int(out_full.get("nfev", 0)),
            "final_f": float(out_stop.get("final_f", np.nan)),
            "final_x": out_stop.get("final_x", x1.tolist()),
            "runtime_sec": float(out_stop.get("runtime_sec", 0.0)) + float(out_full.get("runtime_sec", 0.0)),
            "trace_f": list(out_full.get("trace_f", [])) + list(out_stop.get("trace_f", [])),
            "trace_gnorm": list(out_full.get("trace_gnorm", [])) + list(out_stop.get("trace_gnorm", [])),
            "timed_out": bool(out_stop.get("timed_out", False) or out_full.get("timed_out", False)),

            # useful audit fields
            "stop_stage_final_f": float(out_stop.get("final_f", np.nan)),
            "stop_stage_final_x": out_stop.get("final_x", None),
            "full_stage_final_f": float(out_full.get("final_f", np.nan)),
            "full_stage_final_x": out_full.get("final_x", None),

            "settings": {
                "mixed_stop_max_outer_iter": int(max_outer_iter),
                "mixed_stop_gtol": float(gtol_STOP),
                "mixed_stop_ftol": float(ftol_STOP),
                "mixed_stop_maxls": int(maxls_STOP),
                "mixed_full_max_outer_iter": int(max_outer_iter),
                "inner_max_iter": inner_max_iter,
                "inner_tol": inner_tol,
                "inner_damping": inner_damping,
                "logdet_jitter": logdet_jitter,
                "floor_rel": hess_floor_rel,
                "floor_abs": hess_floor_abs,
                "max_step_norm": max_step_norm,
            }
        }

        # optionally expose final cache (ONLY if requested)
        #if return_eta_cache:
        #    out_mixed["_eta_cache_final"] = out_full.get("_eta_cache_final", None)

        return res_stop, out_mixed
    # -----------------------
    # FD (NONMEM-like): FD outer, FD inner
    # -----------------------
    elif method == "FD":
        # FD is extremely expensive on all subjects.
        subs = subjects
        if fd_max_subjects is not None and fd_max_subjects < len(subjects):
            rng = np.random.default_rng(0)
            idx = rng.choice(len(subjects), size=int(fd_max_subjects), replace=False)
            subs = [subjects[i] for i in sorted(idx)]
            if verbose:
                print(f"[FD] Using subset of subjects: {len(subs)}/{len(subjects)} (fd_max_subjects={fd_max_subjects})")

        fd_outer_scheme = str(fd_outer_scheme).lower().strip()
        if fd_outer_scheme not in ["forward", "central"]:
            raise ValueError("fd_outer_scheme must be 'forward' or 'central'")

        def f_only(z: np.ndarray, *, warm_cache: Optional[Dict[int, torch.Tensor]] = None) -> Tuple[float, Dict[int, torch.Tensor]]:
            """Return (f, eta_cache_out). Does NOT modify eta_cache_accepted."""
            _check_time_limit("FD base objective")
            z = np.array(z, dtype=float)
            eta_cache_local = (warm_cache.copy() if warm_cache is not None else eta_cache_accepted.copy())
            f = focei_objective_fd_value(
                subs, z,
                eta_cache=eta_cache_local,
                inner_max_iter=inner_max_iter,
                inner_tol=inner_tol,
                inner_damping=inner_damping,
                logdet_jitter=logdet_jitter,
                eps_eta=fd_eps_eta,
                floor_rel=hess_floor_rel,
                floor_abs=hess_floor_abs,
                max_step_norm=max_step_norm,
            )
            _check_time_limit("FD base objective")
            return float(f), eta_cache_local

        def eval_f_g(z: np.ndarray) -> Tuple[float, np.ndarray]:
            nonlocal last_eval_cache, last_eval_x, last_finite_x, last_finite_f
            nonlocal last_eval_f, last_eval_gnorm, best_f, best_x

            _check_time_limit("FD objective evaluation")
            z = np.array(z, dtype=float)
            #if bounds is not None:
            #    lo = np.array([b[0] for b in bounds], dtype=float)
            #    hi = np.array([b[1] for b in bounds], dtype=float)
            #    z = np.minimum(np.maximum(z, lo), hi)


            # Base evaluation (also gives a base eta-cache we can reuse as warm-start)
            try:
                f0, base_cache = f_only(z)
            except Exception as e:
                f0 = np.nan
                base_cache = None
                import traceback
                traceback.print_exc()   # prints full stack trace to stderr
                if verbose:
                    print(f"[{method}] Exception during eval: {type(e).__name__}: {e}")

            if not np.isfinite(f0) or base_cache is None:
                f_pen, g_pen = _penalty(z, last_finite_x)
                f0, g = f_pen, g_pen
                if verbose:
                    print(f"[FD] base non-finite -> penalty f={f0:.3e}  |g|={np.linalg.norm(g):.3e}")
                trace_f.append(float(f0))
                trace_gnorm.append(float(np.linalg.norm(g)))
                return float(f0), g.astype(float)

            # Outer FD gradient
            p = len(z)
            g = np.zeros(p, dtype=float)

            if fd_outer_scheme == "forward":
                if verbose:
                    print(f"[FD] VISIT f0={f0:.6f}. Computing forward-diff gradient (p={p}) ...")
                for j in range(p):
                    _check_time_limit(f"FD forward gradient component {j+1}/{p}")

                    zp = z.copy()
                    zp[j] = z[j] + fd_eps_outer

                    if bounds is not None:
                        lo_j, hi_j = bounds[j]
                        zp[j] = min(max(zp[j], lo_j), hi_j)

                    h = zp[j] - z[j]
                    if h == 0.0 :
                        g[j] = 0.0
                    else:
                        fp, _ = f_only(zp, warm_cache=base_cache)
                        g[j] = (fp - f0) / h

                    #ej = np.zeros(p); ej[j] = 1.0
                    #zp = z + fd_eps_outer * ej
                    #fp, _ = f_only(zp, warm_cache=base_cache)
                    #g[j] = (fp - f0) / fd_eps_outer
                        if verbose:
                            print(f"    [FD] VISIT grad {j+1:02d}/{p}: fp={fp:.3f}  g={g[j]:.3e}")
            else:
                if verbose:
                    print(f"[FD] VISIT f0={f0:.6f}. Computing central-diff gradient (2p evals, p={p}) ...")
                for j in range(p):
                    _check_time_limit(f"FD central gradient component {j+1}/{p}")
                    #ej = np.zeros(p); ej[j] = 1.0
                    #zp = z + fd_eps_outer * ej
                    #zm = z - fd_eps_outer * ej
                    #fp, _ = f_only(zp, warm_cache=base_cache)
                    #fm, _ = f_only(zm, warm_cache=base_cache)
                    #g[j] = (fp - fm) / (2.0 * fd_eps_outer)

                    zp = z.copy()
                    zm = z.copy()
                    zp[j] = z[j] + fd_eps_outer
                    zm[j] = z[j] - fd_eps_outer

                    if bounds is not None:
                        lo_j, hi_j = bounds[j]
                        zp[j] = min(max(zp[j], lo_j), hi_j)
                        zm[j] = min(max(zm[j], lo_j), hi_j)

                    h = zp[j] - zm[j]
                    if h == 0.0:
                        g[j] = 0.0
                    else:
                        fp, _ = f_only(zp, warm_cache=base_cache)
                        fm, _ = f_only(zm, warm_cache=base_cache)
                        g[j] = (fp - fm) / (h)


                        if verbose:
                            print(f"    [FD] VISIT grad {j+1:02d}/{p}: fp={fp:.3f} fm={fm:.3f} g={g[j]:.3e}")
                        

            # Store cache from base point as the "accepted" candidate for callback
            last_eval_cache = base_cache
            last_eval_x = z.copy()

            # Non-finite guard
            if (not np.isfinite(f0)) or (not np.all(np.isfinite(g))):
                f_pen, g_pen = _penalty(z, last_finite_x)
                f0, g = f_pen, g_pen
                if verbose:
                    print(f"[FD] non-finite after grad -> penalty f={f0:.3e}  |g|={np.linalg.norm(g):.3e}")
            else:
                last_finite_x = z.copy()
                last_finite_f = float(f0)
                if verbose:
                    print(f"[FD] f={f0:.6f}  |g|={np.linalg.norm(g):.3e}")

                        # --- ADD HERE ---
            gnorm = float(np.linalg.norm(g))
            last_eval_f = float(f0)
            last_eval_gnorm = gnorm
            if np.isfinite(last_eval_f) and (last_eval_f < best_f):
                best_f = last_eval_f
                best_x = z.copy()
            # --- END ADD ---

            trace_f.append(float(f0))
            trace_gnorm.append(gnorm)
            _check_time_limit("FD objective evaluation")
            return float(f0), g.astype(float)


        def callback(xk: np.ndarray):
            nonlocal eta_cache_accepted, last_eval_cache, last_eval_x
            nonlocal last_eval_f, last_eval_gnorm, best_f, best_x
            #nonlocal accepted_iter

            _check_time_limit("FD callback")
            if (last_eval_cache is not None) and (last_eval_x is not None):
                xk = np.array(xk, dtype=float)

                if np.allclose(xk, last_eval_x, atol=1e-7, rtol=0.0):
                    eta_cache_accepted = last_eval_cache

                    #accepted_iter += 1

                    # best among *accepted* iterates
                    if (last_eval_f is not None) and np.isfinite(last_eval_f) and (last_eval_f < best_f):
                        best_f = float(last_eval_f)
                        best_x = xk.copy()

                    #if print_accepted_only:
                    print(f"[{method}] ACCEPT f={last_eval_f:.6f}  |g|={last_eval_gnorm:.3e}  best_f={best_f:.6f}")


        try:
            res = minimize(
                fun=eval_f_g,
                x0=x0,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                callback=callback,
                options={"maxiter": int(max_outer_iter),
                         "ftol": ftol,  # disable f-based stopping (we have a custom penalty handling)
                         "gtol": gtol, 
                         "maxls": maxls, # Maximum number of line search steps per iteration. The default is 20. 
                         },
            )
        except MethodTimeLimitReached as exc:
            x_timeout = best_x.copy() if best_x is not None else last_finite_x.copy()
            fun_timeout = best_f if np.isfinite(best_f) else float(last_finite_f)
            res = OptimizeResult(
                x=x_timeout,
                fun=fun_timeout,
                success=False,
                status=1,
                message=str(exc),
                nit=len(trace_f),
                nfev=len(trace_f),
                timed_out=True,
            )
        x_hat = res.x.copy()
        final_L = float(res.fun)

    else:
        raise ValueError("method must be STOP, FULL_UNROLL, FULL_IMPLICIT, or FD")

    runtime = time.perf_counter() - t_start

    out = {
        "method": method,
        "success": bool(getattr(res, "success", False)),
        "message": str(getattr(res, "message", "")),
        "niter": int(getattr(res, "nit", -1)),
        "nfev": int(getattr(res, "nfev", -1)),
        "final_f": float(getattr(res, "fun", np.nan)),
        "final_x": x_hat.tolist(),
        "runtime_sec": float(runtime),
        "trace_f": trace_f,
        "trace_gnorm": trace_gnorm,
        "timed_out": bool(getattr(res, "timed_out", False)),
    }
    out["settings"] = dict(
        inner_max_iter=inner_max_iter,
        inner_tol=inner_tol,
        inner_damping=inner_damping,
        logdet_jitter=logdet_jitter,
        floor_rel=hess_floor_rel,
        floor_abs=hess_floor_abs,
        max_step_norm=max_step_norm,
    )
    return res, out

# %% [markdown] cell 11
# ## Joint estimation with 4 outer-gradient strategies
# %% [code] cell 12
# METHODS = ["FULL_IMPLICIT"]

# -------------------------
# Initialization & bounds
# -------------------------
# Start from the same rough values used in the staged PK and PD fits.
x0 = np.array([
    # logKA, logCL, logV
    np.log(1.0), np.log(0.2), np.log(10.0),

    # logE0, logC50, logKE0, logitEMAX
    np.log(100.0), np.log(2.0), np.log(0.05), np.log(0.9 / (1.0 - 0.9)),  # logit(0.9)

    # logSIGMA_PK, logSIGMA_PD
    np.log(0.5), np.log(5.0),

    # logOM_KA, logOM_CL, logOM_V, logOM_E0, logOM_C50, logOM_KE0
    np.log(0.3), np.log(0.3), np.log(0.3),
    np.log(0.3), np.log(0.3), np.log(0.3),
], dtype=float)

BOUNDS = [
    # PK
    (np.log(1e-3), np.log(50.0)),  # logKA
    (np.log(1e-4), np.log(10.0)),  # logCL
    (np.log(1e-2), np.log(1e3)),   # logV

    # PD
    (np.log(1.0),  np.log(200.0)), # logE0
    (np.log(1e-3), np.log(1e3)),   # logC50
    (np.log(1e-4), np.log(10.0)),  # logKE0
    (-10.0, 10.0),                 # logitEMAX  -> sigmoid in (0,1)

    # residual SDs
    (np.log(1e-4), np.log(50.0)),  # logSIGMA_PK
    (np.log(1e-4), np.log(200.0)), # logSIGMA_PD

    # omegas
    (np.log(1e-4), np.log(5.0)),   # logOM_KA
    (np.log(1e-4), np.log(5.0)),   # logOM_CL
    (np.log(1e-4), np.log(5.0)),   # logOM_V
    (np.log(1e-4), np.log(5.0)),   # logOM_E0
    (np.log(1e-4), np.log(5.0)),   # logOM_C50
    (np.log(1e-4), np.log(5.0)),   # logOM_KE0
]

x_by_method = {}



for m in METHODS:
    print("\n==============================")
    print("Running method:", m)
    print("==============================\n")
    if True:#m != "STOP":
        res, out = run_optimization_notebook(
            method=m,
            subjects=subjects,
            x0=x0,
            bounds=BOUNDS,
            verbose=True,
            # inner params
            inner_max_iter = inner_max_iter,
            inner_tol  = inner_tol,
            inner_damping  = inner_damping,
            # FULL_UNROLL controls
            full_unroll_steps = full_unroll_steps,
            full_damping  = inner_damping,
            # numerical
            logdet_jitter = logdet_jitter,
            hess_floor_rel  = floor_rel,
            hess_floor_abs  = floor_abs,
            max_step_norm  = max_step_norm,
            # FD steps
            fd_eps_outer = fd_eps_outer,
            fd_eps_eta = fd_eps_eta,
            # verbosity
            # FD controls
            fd_max_subjects = None,
            fd_outer_scheme = "central",
            # outer minimization options
            max_outer_iter = max_outer_iter,
            gtol = gtol,
            ftol = ftol,
            maxls = maxls,
            max_runtime_sec = MAX_METHOD_RUNTIME_SEC,
        )
    else:
        res, out = run_optimization_notebook(
            method=m,
            subjects=subjects,
            x0=x0,
            bounds=BOUNDS,
            verbose=True,
            # inner params
            inner_max_iter = inner_max_iter,
            inner_tol  = inner_tol,
            inner_damping  = inner_damping,
            # FULL_UNROLL controls
            full_unroll_steps = 50,
            full_damping  = 1.0,
            # numerical
            logdet_jitter = logdet_jitter,
            hess_floor_rel  = floor_rel,
            hess_floor_abs  = floor_abs,
            max_step_norm  = max_step_norm,
            # FD steps
            fd_eps_outer = fd_eps_outer,
            fd_eps_eta = fd_eps_eta,
            # verbosity
            # FD controls
            fd_max_subjects = None,
            fd_outer_scheme = "central",
            # outer minimization options
            max_outer_iter = max_outer_iter,
            gtol = gtol_STOP,
            ftol = ftol_STOP,
            maxls = maxls_STOP,
            max_runtime_sec = MAX_METHOD_RUNTIME_SEC,
        )
    results[m] = out

    print(f"Done {m}: success={out['success']}  timed_out={out.get('timed_out', False)}  niter={out['niter']}  nfev={out['nfev']}  runtime={out['runtime_sec']:.1f}s  final_f={out['final_f']:.3f}")
print(f"  message: {out['message']}")

# %% [code] cell 13
# print final results for all methods
# print all methods in results
unique_methods = sorted(set(results.keys()))

print("\n==============================")
print("Summary of results:")
print("==============================\n")

for m in unique_methods:
    out = results[m]
    print(f"{m}: success={out['success']}  timed_out={out.get('timed_out', False)}  niter={out['niter']}  nfev={out['nfev']}  runtime={out['runtime_sec']:.1f}s  final_f={out['final_f']:.3f}")
    print(f"  message: {out['message']}")

# %% [code] cell 14
import csv

if boolWriteResults:
    for m in METHODS:
        out = results[m]
        for metric in ["final_f", "final_x", "message", "method", "success", "nfev", "niter", "runtime_sec"]:
            print(f"writing metric {metric} from {m}")
            value = out.get(metric, "")

            out_path = f"result_{m}_{metric}.csv"
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["value"])
                if isinstance(value, list):
                    for v in value:
                        w.writerow([v])
                else:
                    w.writerow([value])
    

# %% [code] cell 15
# --- imports (fixes NameError for Dict/Optional) ---
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
import ast

# (optional) ensure display exists outside Jupyter
try:
    display
except NameError:
    def display(x): 
        print(x)

def _as_float_scalar(v: Any, default: float = np.nan) -> float:
    """Robust scalar coercion: handles python scalars, 0d arrays, 1-element lists, strings."""
    if v is None:
        return float(default)
    # torch tensor
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    # numpy / list
    arr = np.asarray(v)
    if arr.shape == ():  # scalar
        try:
            return float(arr)
        except Exception:
            return float(default)
    # if it's a 1-element container
    if arr.size == 1:
        try:
            return float(arr.reshape(-1)[0])
        except Exception:
            return float(default)
    # otherwise not a scalar
    return float(default)

def _coerce_x(v: Any) -> np.ndarray:
    """Robust conversion of final_x/x_opt to 1D float numpy array."""
    if v is None:
        return np.array([], dtype=float)

    # torch tensor
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()

    # string cases: "[...]" or "1,2,3"
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return np.array([], dtype=float)
        # try literal eval
        try:
            parsed = ast.literal_eval(s)
            return np.asarray(parsed, dtype=float).reshape(-1)
        except Exception:
            # try comma-separated parsing
            try:
                return np.fromstring(s, sep=",", dtype=float).reshape(-1)
            except Exception:
                return np.array([], dtype=float)

    # list/np.array cases
    try:
        x = np.asarray(v, dtype=float).reshape(-1)
        return x
    except Exception:
        # last resort: flatten and keep numeric-only
        flat = []
        try:
            for item in np.asarray(v).reshape(-1).tolist():
                try:
                    flat.append(float(item))
                except Exception:
                    pass
        except Exception:
            pass
        return np.asarray(flat, dtype=float).reshape(-1)

def build_results_df(results: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for m, out in results.items():
        if not isinstance(out, dict):
            out = {}  # defensive

        # Support both "final_x" and "x_opt" naming
        x_raw = out.get("final_x", out.get("x_opt", out.get("x", [])))
        x_opt = _coerce_x(x_raw)

        rows.append({
            "method": str(m),
            "success": bool(out.get("success", False)),
            "final_f": _as_float_scalar(out.get("final_f", np.nan)),
            "runtime_sec": _as_float_scalar(out.get("runtime_sec", np.nan)),
            "niter": int(_as_float_scalar(out.get("niter", -1), default=-1)),
            "nfev": int(_as_float_scalar(out.get("nfev", -1), default=-1)),
            "message": str(out.get("message", "")),
            "x_opt": x_opt,
            "x_dim": int(x_opt.size),
        })

    df = pd.DataFrame(rows)
    # Keep methods with missing final_f at bottom
    df = df.sort_values("final_f", na_position="last").reset_index(drop=True)
    return df

# ---- build df ----
res_df_single = build_results_df(results)
display(res_df_single[["method","success","final_f","runtime_sec","niter","nfev","x_dim"]])

# ---- Apples-to-apples STOP eval ----
def stop_eval_at_x(x_np: Any, max_subjects: Optional[int] = None,
                   inner_max_iter: int = 25, inner_tol: float = 1e-6,
                   inner_damping: float = 1.0,
                   logdet_jitter: float = 1e-3,
                   floor_rel: float = 1e-5,
                   floor_abs: float = 1e-8,
                   max_step_norm: float = 5.0) -> float: 
    x_np = _coerce_x(x_np)
    if x_np.size == 0:
        return float("nan")

    subs = subjects if max_subjects is None else subjects[:int(max_subjects)]
    return float(focei_objective_stop_value(
        subs, x_np,
        inner_max_iter=inner_max_iter,
        inner_tol=inner_tol,
        inner_damping=inner_damping,
        logdet_jitter=logdet_jitter,
        floor_rel=floor_rel,
        floor_abs=floor_abs,
        max_step_norm=max_step_norm,
    ))

if __name__ == "__main__":
    tables_dir = Path("tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    if "res_df_single" in globals() and isinstance(res_df_single, pd.DataFrame):
        summary_path = tables_dir / "warfarin_single_start_summary.csv"
        res_df_single.to_csv(summary_path, index=False)
        print(f"Saved: {summary_path}")
    for result_csv in Path(".").glob("result_*.csv"):
        target = tables_dir / result_csv.name
        if target.exists():
            target.unlink()
        result_csv.replace(target)
        print(f"Saved: {target}")
