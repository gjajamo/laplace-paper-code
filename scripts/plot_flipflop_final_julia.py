from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

try:
    from scipy.interpolate import RectBivariateSpline
except Exception:  # pragma: no cover
    RectBivariateSpline = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLICE_NPZ = (
    ROOT
    / "20260121_ConferenceEMBC2026_Abstract2"
    / "reproducibility_freeze"
    / "slice"
    / "frozen_inputs"
    / "flipflop_objective_slice_logka_logv_autodiff_parallel.npz"
)

METHOD_ORDER = ["FD", "FULL-implicit", "FULL-unroll", "STOP"]
METHOD_COLORS = {
    "FD": "#e67e22",
    "FULL-implicit": "#1f7a8c",
    "FULL-unroll": "#6a4c93",
    "STOP": "#b23a48",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create final Julia-only flip-flop manuscript figures.")
    parser.add_argument("--julia-csv", type=Path, required=True)
    parser.add_argument("--slice-npz", type=Path, default=DEFAULT_SLICE_NPZ)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--smooth-factor", type=int, default=4)
    parser.add_argument("--delta-max", type=float, default=500.0)
    return parser.parse_args()


def method_label(value: object) -> str:
    mapping = {
        "FULL_IMPLICIT": "FULL-implicit",
        "FULL_UNROLL": "FULL-unroll",
        "FULL": "FULL-unroll",
        "FD": "FD",
        "STOP": "STOP",
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


def load_julia(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["method_label"] = df["method"].map(method_label)
    numeric_cols = [
        "start_id",
        "objective",
        "objective_ad_eval",
        "objective_stop_eval",
        "wall_sec",
        "cpu_sec",
        "theta_logka",
        "theta_logv",
        "success",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "objective_ad_eval" not in df.columns:
        df["objective_ad_eval"] = df.get("objective_stop_eval", df["objective"])
    df = df.loc[df["method_label"].isin(METHOD_ORDER)].copy()
    best = df["objective_ad_eval"].min(skipna=True)
    df["common_eval_gap"] = df["objective_ad_eval"] - best
    df["common_eval_gap_plot"] = df["common_eval_gap"].clip(lower=1e-8)
    return df


def load_slice(path: Path, delta_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float], tuple[float, float]]:
    z = np.load(path, allow_pickle=True)
    objective = np.asarray(z["objective"] if "objective" in z.files else -np.asarray(z["Z"], dtype=float), dtype=float)
    x = np.asarray(z["ka_vals"] if "ka_vals" in z.files else z["grid1"], dtype=float)
    y = np.asarray(z["v_vals"] if "v_vals" in z.files else z["grid2"], dtype=float)
    if "true1_logka" in z.files:
        true1 = (float(z["true1_logka"]), float(z["true1_logV"]))
        true2 = (float(z["true2_logka"]), float(z["true2_logV"]))
    else:
        true1 = (float(np.log(0.05)), float(np.log(20.0)))
        true2 = (float(np.log(3.0 / 20.0)), float(np.log(3.0 / 0.05)))
    delta = np.clip(objective - np.nanmin(objective), 0.0, delta_max)
    return x, y, delta, true1, true2


def smooth_grid(x: np.ndarray, y: np.ndarray, z: np.ndarray, factor: int, delta_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    factor = max(1, int(factor))
    if factor <= 1 or x.size < 4 or y.size < 4 or RectBivariateSpline is None or not np.all(np.isfinite(z)):
        return x, y, z
    xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), (len(x) - 1) * factor + 1)
    ys = np.linspace(float(np.nanmin(y)), float(np.nanmax(y)), (len(y) - 1) * factor + 1)
    spline = RectBivariateSpline(y, x, z, kx=3, ky=3, s=0.0)
    return xs, ys, np.clip(spline(ys, xs), 0.0, delta_max)


def rounded_ticks(lo: float, hi: float, step: float = 0.5) -> np.ndarray:
    start = np.ceil(lo / step) * step
    end = np.floor(hi / step) * step
    if end < start:
        return np.array([], dtype=float)
    return np.round(np.arange(start, end + 0.25 * step, step), 3)


def draw_background(ax: plt.Axes, x: np.ndarray, y: np.ndarray, delta: np.ndarray, true1, true2) -> None:
    xmesh, ymesh = np.meshgrid(x, y, indexing="xy")
    minor_levels = np.unique(np.r_[np.linspace(5.0, 120.0, 24), np.linspace(140.0, 500.0, 10)])
    major_levels = np.array([5, 10, 20, 30, 40, 60, 80, 120, 200, 300, 400, 500], dtype=float)
    ax.contour(xmesh, ymesh, delta, levels=minor_levels, colors="#d9dfe7", linewidths=0.55, alpha=0.92, zorder=0)
    cs_major = ax.contour(xmesh, ymesh, delta, levels=major_levels, colors="#758292", linewidths=0.95, alpha=0.88, zorder=1)
    ax.contour(xmesh, ymesh, delta, levels=[40.0], colors="#2b3440", linewidths=1.45, alpha=0.9, zorder=2)
    present = [level for level in [10.0, 40.0, 120.0, 200.0] if np.nanmin(delta) <= level <= np.nanmax(delta)]
    if present:
        ax.clabel(cs_major, levels=present, inline=True, fmt=lambda value: f"{value:g}", fontsize=7, colors="#394553")
    ax.scatter(
        [true1[0], true2[0]],
        [true1[1], true2[1]],
        s=120,
        marker="X",
        facecolors="white",
        edgecolors="black",
        linewidths=1.0,
        zorder=10,
    )


def style_panel(ax: plt.Axes, title: str, left: bool, bottom: bool, x: np.ndarray, y: np.ndarray) -> None:
    ax.text(
        0.03,
        0.97,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="semibold",
        color="#1d2733",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.8},
        zorder=20,
    )
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(float(np.min(y)), float(np.max(y)))
    ax.set_xticks(rounded_ticks(float(np.min(x)), float(np.max(x))))
    ax.set_yticks(rounded_ticks(float(np.min(y)), float(np.max(y))))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(alpha=0.08)
    ax.tick_params(direction="out", length=3.0, width=0.8, color="#55606d", labelcolor="#36424f")
    if not bottom:
        ax.tick_params(labelbottom=False)
    if not left:
        ax.tick_params(labelleft=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#4b5662")


def save_figure(fig: plt.Figure, outbase: Path) -> None:
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def clipped_points(sub: pd.DataFrame, xlim: tuple[float, float], ylim: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = sub["theta_logka"].to_numpy(dtype=float)
    ys = sub["theta_logv"].to_numpy(dtype=float)
    inside = np.isfinite(xs) & np.isfinite(ys) & (xs >= xlim[0]) & (xs <= xlim[1]) & (ys >= ylim[0]) & (ys <= ylim[1])
    xpad = 0.018 * (xlim[1] - xlim[0])
    ypad = 0.018 * (ylim[1] - ylim[0])
    return np.clip(xs, xlim[0] + xpad, xlim[1] - xpad), np.clip(ys, ylim[0] + ypad, ylim[1] - ypad), inside


def plot_contours(df: pd.DataFrame, slice_data: tuple, outdir: Path) -> None:
    x, y, delta, true1, true2 = slice_data
    panels = [("FD", (0, 0)), ("FULL-implicit", (0, 1)), ("FULL-unroll", (1, 0)), ("STOP", (1, 1))]
    xlim = (float(np.min(x)), float(np.max(x)))
    ylim = (float(np.min(y)), float(np.max(y)))
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 8.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.985, bottom=0.14, wspace=0.0, hspace=0.0)
    for label, (row, col) in panels:
        ax = axes[row, col]
        draw_background(ax, x, y, delta, true1, true2)
        style_panel(ax, label, left=(col == 0), bottom=(row == 1), x=x, y=y)
        sub = df.loc[df["method_label"] == label].dropna(subset=["theta_logka", "theta_logv"])
        xplot, yplot, inside = clipped_points(sub, xlim, ylim)
        rng = np.random.default_rng(100 + row * 10 + col)
        xplot = xplot + rng.normal(0.0, 0.012 * (xlim[1] - xlim[0]), size=len(xplot))
        yplot = yplot + rng.normal(0.0, 0.012 * (ylim[1] - ylim[0]), size=len(yplot))
        color = METHOD_COLORS[label]
        if np.any(inside):
            ax.scatter(xplot[inside], yplot[inside], s=28, c=color, edgecolors="white", linewidths=0.55, alpha=0.92, zorder=6)
        if np.any(~inside):
            ax.scatter(xplot[~inside], yplot[~inside], s=44, marker="D", c=color, edgecolors="black", linewidths=0.55, alpha=0.92, zorder=7)
            ax.text(
                0.97,
                0.04,
                f"{int(np.sum(~inside))} outside",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="#394553",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.8},
                zorder=20,
            )
    fig.supxlabel(r"$\log k_a$", y=0.055, fontsize=12)
    fig.supylabel(r"$\log V$", x=0.045, fontsize=12)
    handles = [
        Line2D([0], [0], color="#2b3440", linewidth=1.45, label="Delta OFV contours"),
        Line2D([0], [0], marker="o", color="#4b5662", markerfacecolor="#4b5662", markeredgecolor="white", markersize=7, linewidth=0, label="Endpoint in panel"),
        Line2D([0], [0], marker="X", color="black", markerfacecolor="white", markeredgecolor="black", markersize=8, linewidth=0, label="Reference basins"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005))
    save_figure(fig, outdir / "flipflop_final_method_contours")


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
        ax.scatter(i + rng.normal(0.0, 0.07, size=len(vals)), vals, s=17, c=color, edgecolors="white", linewidths=0.35, alpha=0.72, zorder=3)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.18)
    ax.tick_params(axis="x", rotation=15)
    for spine in ax.spines.values():
        spine.set_color("#4b5662")


def plot_boxplots(df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), constrained_layout=True)
    boxplot_with_points(axes[0], df, "common_eval_gap_plot", "Endpoint objective gap", logy=True)
    boxplot_with_points(axes[1], df, "wall_sec", "Wall time (s)", logy=True)
    save_figure(fig, outdir / "flipflop_final_gap_wall_boxplots")

    fig, ax = plt.subplots(figsize=(6.4, 4.1), constrained_layout=True)
    boxplot_with_points(ax, df, "cpu_sec", "CPU time (s)", logy=True)
    save_figure(fig, outdir / "flipflop_final_cpu_boxplot")


def write_summary(df: pd.DataFrame, outdir: Path, x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        g = df.loc[df["method_label"] == method].copy()
        common = g["objective_ad_eval"].replace([np.inf, -np.inf], np.nan).dropna()
        wall = g["wall_sec"].replace([np.inf, -np.inf], np.nan).dropna()
        cpu = g["cpu_sec"].replace([np.inf, -np.inf], np.nan).dropna()
        outside = (
            (g["theta_logka"] < float(np.min(x))) |
            (g["theta_logka"] > float(np.max(x))) |
            (g["theta_logv"] < float(np.min(y))) |
            (g["theta_logv"] > float(np.max(y)))
        )
        rows.append(
            {
                "method_label": method,
                "n": int(len(g)),
                "n_success": int(g["success"].astype(str).str.lower().eq("true").sum()),
                "min_endpoint_ofv": float(common.min()) if len(common) else np.nan,
                "median_endpoint_ofv": float(common.median()) if len(common) else np.nan,
                "near_optimal_gap_le_5": int((g["common_eval_gap"] <= 5.0).sum()),
                "median_wall_sec": float(wall.median()) if len(wall) else np.nan,
                "median_cpu_sec": float(cpu.median()) if len(cpu) else np.nan,
                "n_outside_slice": int(outside.sum()),
            }
        )
    summary = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / "flipflop_final_summary.csv", index=False)
    df.to_csv(outdir / "flipflop_final_points.csv", index=False)
    return summary


def main() -> None:
    args = parse_args()
    setup_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = load_julia(args.julia_csv)
    x, y, delta, true1, true2 = load_slice(args.slice_npz, args.delta_max)
    xs, ys, zs = smooth_grid(x, y, delta, args.smooth_factor, args.delta_max)
    plot_contours(df, (xs, ys, zs, true1, true2), args.outdir)
    plot_boxplots(df, args.outdir)
    summary = write_summary(df, args.outdir, x, y)
    print(f"Wrote final flip-flop figures under {args.outdir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
