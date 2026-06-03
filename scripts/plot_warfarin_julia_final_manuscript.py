from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_CSV = (
    ROOT
    / "WarfarinJulia"
    / "tables"
    / "warfarin_ode_method_contours_6methods_20260525"
    / "warfarin_julia_multistart_methods.csv"
)
DEFAULT_OUTDIR = ROOT / "WarfarinJulia" / "figures" / "warfarin_julia_final_manuscript_20260527"

METHOD_ORDER = ["FD", "FULL-implicit", "FULL-unroll", "STOP", "STOP+FULL", "FULL+STOP"]
METHOD_COLORS = {
    "FD": "#e67e22",
    "FULL-implicit": "#1f7a8c",
    "FULL-unroll": "#6a4c93",
    "STOP": "#b23a48",
    "STOP+FULL": "#0b6bcb",
    "FULL+STOP": "#607d3b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create final Warfarin Julia manuscript OFV/time figure.")
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--time-metric", choices=["cpu", "wall"], default="cpu")
    return parser.parse_args()


def method_label(value: object) -> str:
    mapping = {
        "FD": "FD",
        "FULL_IMPLICIT": "FULL-implicit",
        "FULL_UNROLL": "FULL-unroll",
        "STOP": "STOP",
        "STOP+FULL_IMPLICIT": "STOP+FULL",
        "STOP+FULL": "STOP+FULL",
        "FULL_IMPLICIT+STOP": "FULL+STOP",
        "FULL+STOP": "FULL+STOP",
    }
    return mapping.get(str(value), str(value))


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def delta_log_display(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    positive = arr[np.isfinite(arr) & (arr > 0.0)]
    floor = max(float(np.nanmin(positive)) * 0.5, 1e-8) if positive.size else 1e-8
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return np.where(arr > 0.0, arr, floor)


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["method_label"] = df["method"].map(method_label)
    df = df.loc[df["method_label"].isin(METHOD_ORDER)].copy()
    for col in ["objective_ad_eval", "wall_sec", "cpu_sec", "start_id", "success"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    best = float(df["objective_ad_eval"].min(skipna=True))
    df["endpoint_gap"] = df["objective_ad_eval"] - best
    df["endpoint_gap_plot"] = delta_log_display(df["endpoint_gap"])
    df["cpu_min"] = df["cpu_sec"] / 60.0
    df["wall_min"] = df["wall_sec"] / 60.0
    return df


def boxplot_with_points(ax: plt.Axes, df: pd.DataFrame, value: str, ylabel: str, logy: bool = False) -> None:
    data = []
    labels = []
    colors = []
    for method in METHOD_ORDER:
        vals = df.loc[df["method_label"] == method, value].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        data.append(vals)
        labels.append(method)
        colors.append(METHOD_COLORS[method])
    bp = ax.boxplot(data, tick_labels=labels, widths=0.55, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.22)
        patch.set_edgecolor("#5f6b78")
    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_color("#4b5662")
            item.set_linewidth(0.9)
    rng = np.random.default_rng(123)
    for i, (vals, color) in enumerate(zip(data, colors), start=1):
        ax.scatter(
            i + rng.normal(0.0, 0.07, size=len(vals)),
            vals,
            s=17,
            c=color,
            edgecolors="white",
            linewidths=0.35,
            alpha=0.72,
            zorder=3,
        )
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.18)
    ax.tick_params(axis="x", rotation=15)
    for spine in ax.spines.values():
        spine.set_color("#4b5662")


def plot_boxplots(df: pd.DataFrame, outdir: Path, time_metric: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), constrained_layout=True)
    boxplot_with_points(axes[0], df, "endpoint_gap_plot", "Endpoint objective gap", logy=True)
    time_col = "wall_min" if time_metric == "wall" else "cpu_min"
    time_label = "Wall time (min)" if time_metric == "wall" else "CPU time (min)"
    boxplot_with_points(axes[1], df, time_col, time_label, logy=True)
    outbase = outdir / f"warfarin_julia_gap_{time_metric}_boxplots"
    save_figure(fig, outbase)
    return outbase.with_suffix(".png")


def write_summary(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    best = float(df["objective_ad_eval"].min(skipna=True))
    for method in METHOD_ORDER:
        g = df.loc[df["method_label"] == method].copy()
        endpoint = g["objective_ad_eval"].replace([np.inf, -np.inf], np.nan).dropna()
        cpu = g["cpu_min"].replace([np.inf, -np.inf], np.nan).dropna()
        wall = g["wall_min"].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "method_label": method,
                "n": int(len(g)),
                "min_endpoint_ofv": float(endpoint.min()) if len(endpoint) else np.nan,
                "median_endpoint_ofv": float(endpoint.median()) if len(endpoint) else np.nan,
                "near_optimal_gap_le_3": int((g["objective_ad_eval"] - best <= 3.0).sum()),
                "near_optimal_gap_le_5": int((g["objective_ad_eval"] - best <= 5.0).sum()),
                "median_cpu_min": float(cpu.median()) if len(cpu) else np.nan,
                "median_wall_min": float(wall.median()) if len(wall) else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / "warfarin_julia_final_summary.csv", index=False)
    df.to_csv(outdir / "warfarin_julia_final_points.csv", index=False)
    return summary


def save_figure(fig: plt.Figure, outbase: Path) -> None:
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = load_results(args.results_csv)
    plot_boxplots(df, args.outdir, args.time_metric)
    summary = write_summary(df, args.outdir)
    print(f"Wrote final Warfarin Julia figure under {args.outdir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
