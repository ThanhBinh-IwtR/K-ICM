"""All six visualization functions for KicmBaseModel.

Non-interactive Agg backend is forced at module import so this file is safe
to import in headless training environments.
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib

matplotlib.use("Agg")  # must be set before importing pyplot

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------

_STYLE = "seaborn-v0_8-whitegrid"
_DPI = 200


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _apply_style() -> None:
    try:
        plt.style.use(_STYLE)
    except OSError:
        plt.style.use("seaborn-whitegrid")


# ---------------------------------------------------------------------------
# Plot 1 — Training curves
# ---------------------------------------------------------------------------


def plot_training_curves(
    log_data: dict[str, list[float]],
    save_dir: str = "outputs/plots/",
) -> str:
    """Plot training/validation loss, val R², and LR schedule side-by-side.

    Parameters
    ----------
    log_data:
        Dict with any subset of keys:
        ``train_loss``, ``val_loss``, ``val_r2``, ``lr``, ``steps``.
        Each value is a list of scalar floats (one per epoch or step).
    save_dir:
        Directory where ``training_curves.png`` is written.

    Returns
    -------
    str
        Absolute path to the saved figure.
    """
    _apply_style()
    _ensure_dir(save_dir)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Training Diagnostics", fontsize=14, fontweight="bold", y=1.02)

    # --- Panel 1: loss ---
    ax = axes[0]
    train_loss = log_data.get("train_loss", [])
    val_loss = log_data.get("val_loss", [])
    if train_loss:
        ax.plot(train_loss, label="Train loss", linewidth=1.8, color="#2196F3")
    if val_loss:
        ax.plot(val_loss, label="Val loss", linewidth=1.8, color="#F44336", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE, log10 space)")
    ax.set_title("Train / Val Loss")
    ax.legend()
    ax.grid(True, linewidth=0.5)

    # --- Panel 2: val R² ---
    ax = axes[1]
    val_r2 = log_data.get("val_r2", [])
    if val_r2:
        ax.plot(val_r2, linewidth=1.8, color="#4CAF50")
        ax.axhline(y=max(val_r2), color="#888", linestyle=":", linewidth=1)
        ax.text(
            len(val_r2) - 1,
            max(val_r2),
            f"  peak {max(val_r2):.3f}",
            va="bottom",
            fontsize=8,
            color="#555",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R²")
    ax.set_title("Validation R²")
    ax.grid(True, linewidth=0.5)

    # --- Panel 3: LR schedule ---
    ax = axes[2]
    lr = log_data.get("lr", [])
    steps = log_data.get("steps", list(range(len(lr))))
    if lr:
        ax.plot(steps, lr, linewidth=1.8, color="#FF9800")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("LR Schedule")
    ax.grid(True, linewidth=0.5)

    fig.tight_layout()
    out_path = os.path.join(save_dir, "training_curves.png")
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(out_path)


# ---------------------------------------------------------------------------
# Plot 2 — Prediction scatter
# ---------------------------------------------------------------------------


def plot_prediction_scatter(
    targets: np.ndarray,
    preds: np.ndarray,
    metrics: dict[str, float] | None = None,
    save_dir: str = "outputs/plots/",
) -> str:
    """Scatter plot of predicted vs true log10 values with marginal histograms.

    Parameters
    ----------
    targets:
        Ground-truth log10 kinetic values, shape ``(N,)``.
    preds:
        Model predictions, shape ``(N,)``.
    metrics:
        Optional dict with keys ``r2``, ``pcc``, ``scc``, ``rmse``.
        If omitted, metrics are computed internally.
    save_dir:
        Output directory.

    Returns
    -------
    str
        Absolute path to ``pred_vs_true_scatter.png``.
    """
    _apply_style()
    _ensure_dir(save_dir)

    targets = np.asarray(targets, dtype=float).ravel()
    preds = np.asarray(preds, dtype=float).ravel()

    if len(targets) == 0 or len(preds) == 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        out_path = os.path.join(save_dir, "pred_vs_true_scatter.png")
        fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return os.path.abspath(out_path)

    # Compute metrics if not supplied
    if metrics is None:
        pcc, _ = stats.pearsonr(targets, preds)
        scc, _ = stats.spearmanr(targets, preds)
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - targets.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = float(np.sqrt(np.mean((targets - preds) ** 2)))
        metrics = {"r2": r2, "pcc": float(pcc), "scc": float(scc), "rmse": rmse}

    # Build figure with gridspec for marginals
    fig = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(
        2, 2, width_ratios=[5, 1], height_ratios=[1, 5], hspace=0.05, wspace=0.05
    )
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Main panel — hexbin
    hb = ax_main.hexbin(
        targets, preds, gridsize=40, cmap="YlOrRd", mincnt=1, linewidths=0.2
    )
    fig.colorbar(hb, ax=ax_right, label="Count", shrink=0.8, pad=0.05)

    # Reference diagonal
    lo = min(targets.min(), preds.min())
    hi = max(targets.max(), preds.max())
    ax_main.plot([lo, hi], [lo, hi], "r--", linewidth=1.4, label="y = x")

    # Metrics text box
    txt = (
        f"R²   = {metrics.get('r2', float('nan')):.3f}\n"
        f"PCC = {metrics.get('pcc', float('nan')):.3f}\n"
        f"SCC = {metrics.get('scc', float('nan')):.3f}\n"
        f"RMSE = {metrics.get('rmse', float('nan')):.3f}"
    )
    ax_main.text(
        0.04,
        0.96,
        txt,
        transform=ax_main.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85, ec="#ccc"),
        family="monospace",
    )
    ax_main.set_xlabel("True log10 value")
    ax_main.set_ylabel("Predicted log10 value")
    ax_main.legend(fontsize=8, loc="lower right")

    # Marginal histograms
    ax_top.hist(targets, bins=40, color="#2196F3", alpha=0.75, edgecolor="none")
    ax_top.set_ylabel("Count")
    plt.setp(ax_top.get_xticklabels(), visible=False)

    ax_right.hist(
        preds,
        bins=40,
        color="#F44336",
        alpha=0.75,
        orientation="horizontal",
        edgecolor="none",
    )
    ax_right.set_xlabel("Count")
    plt.setp(ax_right.get_yticklabels(), visible=False)

    fig.suptitle("Predicted vs True", fontsize=13, fontweight="bold")
    out_path = os.path.join(save_dir, "pred_vs_true_scatter.png")
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(out_path)


# ---------------------------------------------------------------------------
# Plot 3 — Residuals
# ---------------------------------------------------------------------------


def plot_residuals(
    targets: np.ndarray,
    preds: np.ndarray,
    save_dir: str = "outputs/plots/",
) -> str:
    """Residual plot (predicted - true) vs true values, coloured by density.

    Parameters
    ----------
    targets:
        Ground-truth log10 values, shape ``(N,)``.
    preds:
        Model predictions, shape ``(N,)``.
    save_dir:
        Output directory.

    Returns
    -------
    str
        Absolute path to ``residuals_plot.png``.
    """
    _apply_style()
    _ensure_dir(save_dir)

    targets = np.asarray(targets, dtype=float).ravel()
    preds = np.asarray(preds, dtype=float).ravel()
    residuals = preds - targets

    fig, ax = plt.subplots(figsize=(7, 5))

    if len(targets) <= 1:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
    else:
        # Density coloring via Gaussian KDE
        try:
            xy = np.vstack([targets, residuals])
            density = stats.gaussian_kde(xy)(xy)
            order = density.argsort()
            sc = ax.scatter(
                targets[order],
                residuals[order],
                c=density[order],
                cmap="viridis",
                s=8,
                alpha=0.7,
                linewidths=0,
            )
            fig.colorbar(sc, ax=ax, label="Density")
        except np.linalg.LinAlgError:
            ax.scatter(targets, residuals, s=8, alpha=0.5, color="#2196F3")

        ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Zero residual")

    ax.set_xlabel("True log10 value")
    ax.set_ylabel("Residual (pred − true)")
    ax.set_title("Residuals vs True Values")
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.5)

    fig.tight_layout()
    out_path = os.path.join(save_dir, "residuals_plot.png")
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(out_path)


# ---------------------------------------------------------------------------
# Plot 4 — Error distribution
# ---------------------------------------------------------------------------


def plot_error_distribution(
    targets: np.ndarray,
    preds: np.ndarray,
    save_dir: str = "outputs/plots/",
) -> str:
    """Histogram + KDE of residuals with normal reference curve.

    Parameters
    ----------
    targets:
        Ground-truth log10 values, shape ``(N,)``.
    preds:
        Model predictions, shape ``(N,)``.
    save_dir:
        Output directory.

    Returns
    -------
    str
        Absolute path to ``error_distribution.png``.
    """
    _apply_style()
    _ensure_dir(save_dir)

    targets = np.asarray(targets, dtype=float).ravel()
    preds = np.asarray(preds, dtype=float).ravel()
    residuals = preds - targets

    fig, ax = plt.subplots(figsize=(7, 5))

    if len(residuals) <= 1:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
    else:
        sns.histplot(
            residuals,
            kde=True,
            stat="density",
            color="#5C6BC0",
            edgecolor="white",
            linewidth=0.4,
            alpha=0.7,
            ax=ax,
            label="Residuals (KDE)",
        )

        # Normal reference
        mu, sigma = residuals.mean(), residuals.std(ddof=1)
        x_ref = np.linspace(residuals.min(), residuals.max(), 300)
        normal_pdf = stats.norm.pdf(x_ref, mu, sigma)
        ax.plot(x_ref, normal_pdf, "r--", linewidth=1.6, label="Normal reference")

        # Annotation text box
        median = float(np.median(residuals))
        txt = f"mean  = {mu:.3f}\nstd   = {sigma:.3f}\nmedian = {median:.3f}"
        ax.text(
            0.97,
            0.97,
            txt,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85, ec="#ccc"),
            family="monospace",
        )

    ax.set_xlabel("Residual (pred − true)")
    ax.set_ylabel("Density")
    ax.set_title("Error Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.5)

    fig.tight_layout()
    out_path = os.path.join(save_dir, "error_distribution.png")
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(out_path)


# ---------------------------------------------------------------------------
# Plot 5 — Metric summary table
# ---------------------------------------------------------------------------

_METRIC_DISPLAY = {
    "r2": "R²",
    "pcc": "PCC",
    "scc": "SCC",
    "rmse": "RMSE",
}

_METRIC_HIGH_GOOD = {"r2", "pcc", "scc"}   # higher = better
_METRIC_LOW_GOOD = {"rmse"}                  # lower = better


def _cell_color(metric_key: str, value: float) -> str:
    """Return a hex colour between red (#FF6B6B) and green (#69DB7C)."""
    if not np.isfinite(value):
        return "#DDDDDD"
    if metric_key in _METRIC_HIGH_GOOD:
        # value in [-inf, 1] but practically [0, 1] for well-behaved models
        t = float(np.clip(value, 0.0, 1.0))
    else:
        # RMSE: 0 is perfect, ≥2 is bad
        t = float(np.clip(1.0 - value / 2.0, 0.0, 1.0))
    r = int(255 * (1.0 - t) + 105 * t)
    g = int(107 * (1.0 - t) + 219 * t)
    b = int(107 * (1.0 - t) + 124 * t)
    return f"#{r:02X}{g:02X}{b:02X}"


def plot_metric_summary_table(
    metrics_dict: dict[str, dict[str, float]],
    save_dir: str = "outputs/plots/",
) -> str:
    """Render a colour-coded metric table as an image.

    Parameters
    ----------
    metrics_dict:
        Dict of ``{split_name: {metric_name: value}}``.
        Example: ``{"Train": {"r2": 0.9, "pcc": 0.95, "scc": 0.94, "rmse": 0.35}}``.
    save_dir:
        Output directory.

    Returns
    -------
    str
        Absolute path to ``metric_summary.png``.
    """
    _ensure_dir(save_dir)

    splits = list(metrics_dict.keys())
    row_keys = ["r2", "pcc", "scc", "rmse"]
    row_labels = [_METRIC_DISPLAY.get(k, k.upper()) for k in row_keys]

    # Build cell values and colours
    cell_text: list[list[str]] = []
    cell_colours: list[list[str]] = []

    for rk in row_keys:
        row_vals: list[str] = []
        row_cols: list[str] = []
        for split in splits:
            val = metrics_dict[split].get(rk, float("nan"))
            row_vals.append(f"{val:.4f}" if np.isfinite(val) else "N/A")
            row_cols.append(_cell_color(rk, val))
        cell_text.append(row_vals)
        cell_colours.append(row_cols)

    n_rows = len(row_keys)
    n_cols = len(splits)
    fig_w = max(5.0, 1.5 * n_cols + 1.5)
    fig_h = max(2.5, 0.5 * n_rows + 1.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=splits,
        cellColours=cell_colours,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.8)

    # Style header cells
    for j, split in enumerate(splits):
        cell = tbl[0, j]
        cell.set_facecolor("#37474F")
        cell.set_text_props(color="white", fontweight="bold")

    # Style row label cells
    for i in range(n_rows):
        cell = tbl[i + 1, -1]
        cell.set_facecolor("#ECEFF1")
        cell.set_text_props(fontweight="bold")

    ax.set_title("Metric Summary", fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    out_path = os.path.join(save_dir, "metric_summary.png")
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(out_path)


# ---------------------------------------------------------------------------
# Plot 6 — Attention weights
# ---------------------------------------------------------------------------


def plot_attention_weights(
    attention_weights: Any,
    save_dir: str = "outputs/plots/",
    top_k: int = 10,
) -> str:
    """Plot mean AttentivePooling weights vs residue position.

    Parameters
    ----------
    attention_weights:
        Array-like of shape ``(N_samples, L)`` where ``L`` is sequence length
        (or max-padded length). Zero-padded positions should be masked to NaN
        before passing.
    save_dir:
        Output directory.
    top_k:
        Number of top positions to highlight.

    Returns
    -------
    str
        Absolute path to ``attn_pool_weights.png``.
    """
    _apply_style()
    _ensure_dir(save_dir)

    weights = np.asarray(attention_weights, dtype=float)
    if weights.ndim == 1:
        weights = weights[np.newaxis, :]

    if weights.size == 0 or weights.shape[0] == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        out_path = os.path.join(save_dir, "attn_pool_weights.png")
        fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return os.path.abspath(out_path)

    # Compute mean; ignore NaN (from padding)
    mean_w = np.nanmean(weights, axis=0)
    positions = np.arange(len(mean_w))

    fig, ax = plt.subplots(figsize=(max(10, len(mean_w) // 10), 4))

    ax.plot(positions, mean_w, linewidth=1.2, color="#5C6BC0", alpha=0.85, label="Mean attention")
    ax.fill_between(positions, mean_w, alpha=0.15, color="#5C6BC0")

    # Highlight top-k positions
    actual_k = min(top_k, len(mean_w))
    if actual_k > 0:
        top_indices = np.argsort(mean_w)[-actual_k:]
        ax.scatter(
            top_indices,
            mean_w[top_indices],
            color="#F44336",
            zorder=5,
            s=40,
            label=f"Top-{actual_k} positions",
        )
        # Annotate the single highest position
        peak_idx = top_indices[np.argmax(mean_w[top_indices])]
        ax.annotate(
            f"pos {peak_idx}",
            xy=(peak_idx, mean_w[peak_idx]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#B71C1C",
        )

    ax.set_xlabel("Residue Position")
    ax.set_ylabel("Mean Attention Weight")
    ax.set_title(f"AttentivePooling Weights  (n={weights.shape[0]} samples)")
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.5)

    fig.tight_layout()
    out_path = os.path.join(save_dir, "attn_pool_weights.png")
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(out_path)
