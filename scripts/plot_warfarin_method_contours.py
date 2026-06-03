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
DEFAULT_RESULTS_CSV = (
    ROOT
    / "WarfarinJulia"
    / "tables"
    / "warfarin_ode_clean_10starts_per_method_20260523"
    / "warfarin_julia_multistart_methods.csv"
)
DEFAULT_SLICE_NPZ = ROOT / "WarfarinJulia" / "tables" / "slice_STOP_logitEMAX_logC50_latest.npz"
DEFAULT_OUTDIR = ROOT / "WarfarinJulia" / "figures" / "warfarin_ode_clean_10starts_per_method_20260523"

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
    parser = argparse.ArgumentParser(description="Plot Warfarin endpoint solutions over a 2D objective contour.")
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--slice-npz", type=Path, default=DEFAULT_SLICE_NPZ)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--smooth-factor", type=int, default=4)
    parser.add_argument("--delta-max", type=float, default=300.0)
    return parser.parse_args()


def method_label(value: object) -> str:
    raw = str(value)
    mapping = {
        "FULL_IMPLICIT": "FULL-implicit",
        "FULL_UNROLL": "FULL-unroll",
        "FULL": "FULL-unroll",
        "FD": "FD",
        "STOP": "STOP",
        "STOP+FULL_IMPLICIT": "STOP+FULL",
        "STOP+FULL": "STOP+FULL",
        "FULL_IMPLICIT+STOP": "FULL+STOP",
        "FULL+STOP": "FULL+STOP",
    }
    return mapping.get(raw, raw)


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


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["method_label"] = df["method"].map(method_label)
    for col in ["start_id", "objective_ad_eval", "logitEMAX", "logC50", "success"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.loc[df["method_label"].isin(METHOD_ORDER)].copy()


def load_slice(path: Path, delta_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str, tuple[float, float]]:
    data = np.load(path, allow_pickle=True)
    x = np.asarray(data["grid1"], dtype=float)
    y = np.asarray(data["grid2"], dtype=float)
    if "Zrel" in data.files:
        z = np.asarray(data["Zrel"], dtype=float)
    elif "Z" in data.files:
        z_raw = np.asarray(data["Z"], dtype=float)
        z = z_raw - np.nanmin(z_raw)
    else:
        z = np.asarray(data["Zplot"], dtype=float)
    z = np.clip(z, 0.0, delta_max)
    param1 = str(data["param1"]) if "param1" in data.files else "logitEMAX"
    param2 = str(data["param2"]) if "param2" in data.files else "logC50"
    basin = (float(data["base_x"][int(data["idx1"])]), float(data["base_x"][int(data["idx2"])]))
    return x, y, z, param1, param2, basin


def smooth_grid(x: np.ndarray, y: np.ndarray, z: np.ndarray, factor: int, delta_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    factor = max(1, int(factor))
    if factor <= 1 or RectBivariateSpline is None or x.size < 4 or y.size < 4 or not np.all(np.isfinite(z)):
        return x, y, z
    xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), (x.size - 1) * factor + 1)
    ys = np.linspace(float(np.nanmin(y)), float(np.nanmax(y)), (y.size - 1) * factor + 1)
    spline = RectBivariateSpline(y, x, z, kx=3, ky=3, s=0.0)
    zs = np.clip(spline(ys, xs), 0.0, delta_max)
    return xs, ys, zs


def rounded_ticks(lo: float, hi: float, step: float = 0.5) -> np.ndarray:
    start = np.ceil(lo / step) * step
    end = np.floor(hi / step) * step
    if end < start:
        return np.array([], dtype=float)
    return np.round(np.arange(start, end + 0.25 * step, step), 3)


def draw_background(ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray, basin: tuple[float, float]) -> None:
    xmesh, ymesh = np.meshgrid(x, y, indexing="xy")
    max_level = float(np.nanmax(z))
    minor_levels = np.unique(np.r_[np.linspace(5.0, 120.0, 24), np.linspace(140.0, 300.0, 9)])
    minor_levels = minor_levels[minor_levels <= max_level]
    major_levels = np.array([5, 10, 20, 40, 80, 120, 200, 300], dtype=float)
    major_levels = major_levels[major_levels <= max_level]
    ax.contour(xmesh, ymesh, z, levels=minor_levels, colors="#d9dfe7", linewidths=0.55, alpha=0.92, zorder=0)
    cs_major = ax.contour(xmesh, ymesh, z, levels=major_levels, colors="#758292", linewidths=0.95, alpha=0.88, zorder=1)
    ax.contour(xmesh, ymesh, z, levels=[40.0], colors="#2b3440", linewidths=1.45, alpha=0.9, zorder=2)
    present = [level for level in [10.0, 40.0, 120.0, 300.0] if np.nanmin(z) <= level <= np.nanmax(z)]
    if present:
        ax.clabel(cs_major, levels=present, inline=True, fmt=lambda value: f"{value:g}", fontsize=7, colors="#394553")


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
    x_span = float(np.max(x) - np.min(x))
    x_step = 2.0 if x_span >= 6.0 else 1.0
    ax.set_xticks(rounded_ticks(float(np.min(x)), float(np.max(x)), step=x_step))
    ax.set_yticks(rounded_ticks(float(np.min(y)), float(np.max(y)), step=0.5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
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


def clipped_points(sub: pd.DataFrame, xlim: tuple[float, float], ylim: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = sub["logitEMAX"].to_numpy(dtype=float)
    ys = sub["logC50"].to_numpy(dtype=float)
    inside = np.isfinite(xs) & np.isfinite(ys) & (xs >= xlim[0]) & (xs <= xlim[1]) & (ys >= ylim[0]) & (ys <= ylim[1])
    xpad = 0.018 * (xlim[1] - xlim[0])
    ypad = 0.018 * (ylim[1] - ylim[0])
    xclip = np.clip(xs, xlim[0] + xpad, xlim[1] - xpad)
    yclip = np.clip(ys, ylim[0] + ypad, ylim[1] - ypad)
    return xclip, yclip, inside


def plot_method_contours(df: pd.DataFrame, slice_data: tuple, outdir: Path) -> Path:
    x, y, z, param1, param2, basin = slice_data
    method_labels = [label for label in METHOD_ORDER if label in set(df["method_label"])]
    if not method_labels:
        raise RuntimeError("No recognized methods found for contour plotting.")
    n_panels = len(method_labels)
    ncols = 2 if n_panels > 2 else n_panels
    nrows = int(np.ceil(n_panels / ncols))
    xlim = (float(np.min(x)), float(np.max(x)))
    ylim = (float(np.min(y)), float(np.max(y)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.95 * ncols, 3.8 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    bottom_margin = 0.25 if nrows == 1 else 0.14
    fig.subplots_adjust(left=0.12, right=0.985, top=0.985, bottom=bottom_margin, wspace=0.0, hspace=0.0)

    for idx, label in enumerate(method_labels):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        draw_background(ax, x, y, z, basin)
        style_panel(ax, label, left=(col == 0), bottom=(row == nrows - 1), x=x, y=y)
        sub = df.loc[df["method_label"] == label].dropna(subset=["logitEMAX", "logC50"]).copy()
        xplot, yplot, inside = clipped_points(sub, xlim, ylim)
        rng = np.random.default_rng(1000 + row * 10 + col)
        xplot = xplot + rng.normal(0.0, 0.012 * (xlim[1] - xlim[0]), size=len(xplot))
        yplot = yplot + rng.normal(0.0, 0.012 * (ylim[1] - ylim[0]), size=len(yplot))
        color = METHOD_COLORS[label]
        if np.any(inside):
            ax.scatter(
                xplot[inside],
                yplot[inside],
                s=30,
                c=color,
                edgecolors="white",
                linewidths=0.55,
                alpha=0.92,
                zorder=6,
            )
        if np.any(~inside):
            ax.scatter(
                xplot[~inside],
                yplot[~inside],
                s=46,
                marker="D",
                c=color,
                edgecolors="black",
                linewidths=0.55,
                alpha=0.92,
                zorder=7,
            )

    for idx in range(n_panels, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    xlabel = r"$\mathrm{logit}(E_{\max})$" if param1 == "logitEMAX" else param1
    ylabel = r"$\log C_{50}$" if param2 == "logC50" else param2
    xlabel_y = 0.125 if nrows == 1 else 0.055
    legend_y = 0.005 if nrows > 1 else 0.015
    fig.supxlabel(xlabel, y=xlabel_y, fontsize=12)
    fig.supylabel(ylabel, x=0.035, fontsize=12)
    legend_handles = [
        Line2D([0], [0], color="#2b3440", linewidth=1.45, label="Delta OFV contours"),
        Line2D([0], [0], marker="o", color="#4b5662", markerfacecolor="#4b5662", markeredgecolor="white", markersize=7, linewidth=0, label="Endpoint in panel"),
        Line2D([0], [0], marker="D", color="black", markerfacecolor="#4b5662", markeredgecolor="black", markersize=7, linewidth=0, label="Endpoint outside panel"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, legend_y))

    outdir.mkdir(parents=True, exist_ok=True)
    outbase = outdir / "warfarin_ode_method_contours_logitEMAX_logC50"
    fig.savefig(outbase.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return outbase.with_suffix(".png")


def main() -> None:
    args = parse_args()
    setup_style()
    df = load_results(args.results_csv)
    slice_data_raw = load_slice(args.slice_npz, args.delta_max)
    x, y, z, param1, param2, basin = slice_data_raw
    xs, ys, zs = smooth_grid(x, y, z, args.smooth_factor, args.delta_max)
    out_png = plot_method_contours(df, (xs, ys, zs, param1, param2, basin), args.outdir)
    summary = (
        df.groupby("method_label")
        .agg(
            n=("method_label", "size"),
            min_endpoint_ofv=("objective_ad_eval", "min"),
            median_endpoint_ofv=("objective_ad_eval", "median"),
            n_outside_slice=(
                "logitEMAX",
                lambda s: int(
                    ((s < float(np.min(x))) | (s > float(np.max(x))) |
                     (df.loc[s.index, "logC50"] < float(np.min(y))) |
                     (df.loc[s.index, "logC50"] > float(np.max(y)))).sum()
                ),
            ),
        )
        .reset_index()
    )
    summary.to_csv(args.outdir / "warfarin_ode_method_contours_summary.csv", index=False)
    print(f"Wrote {out_png}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
