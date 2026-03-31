"""Standalone CLI for generating all applicable diagnostic plots.

Usage
-----
    python -m src.visualization.generate_plots \\
        --predictions_csv outputs/test_predictions.csv \\
        --target_col log10_value \\
        --pred_col prediction \\
        --output_dir outputs/plots/

The CSV must contain at minimum the columns specified by ``--target_col``
and ``--pred_col``.  Optional column ``--split_col`` (default: "split") is
used to filter rows; pass ``--split_name`` to select a specific split.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

from src.visualization.plots import (
    plot_error_distribution,
    plot_metric_summary_table,
    plot_prediction_scatter,
    plot_residuals,
)


def _compute_metrics(targets: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    from scipy import stats

    if len(targets) < 2:
        return {"r2": float("nan"), "pcc": float("nan"), "scc": float("nan"), "rmse": float("nan")}

    pcc, _ = stats.pearsonr(targets, preds)
    scc, _ = stats.spearmanr(targets, preds)
    ss_res = float(np.sum((targets - preds) ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((targets - preds) ** 2)))
    return {"r2": float(r2), "pcc": float(pcc), "scc": float(scc), "rmse": rmse}


def main(argv: Optional[list[str]] = None) -> None:  # noqa: D103
    parser = argparse.ArgumentParser(
        description="Generate KicmBaseModel diagnostic plots from a predictions CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictions_csv",
        required=True,
        help="Path to CSV file containing ground-truth and predicted values.",
    )
    parser.add_argument(
        "--target_col",
        default="log10_value",
        help="Column name for ground-truth log10 values.",
    )
    parser.add_argument(
        "--pred_col",
        default="prediction",
        help="Column name for model predictions.",
    )
    parser.add_argument(
        "--split_col",
        default="split",
        help="Column name for split labels (optional; used with --split_name).",
    )
    parser.add_argument(
        "--split_name",
        default=None,
        help="If set, filter to rows where split_col == split_name before plotting.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/plots/",
        help="Directory where plot PNGs are saved.",
    )
    args = parser.parse_args(argv)

    # --- Load data ---
    if not os.path.isfile(args.predictions_csv):
        print(f"ERROR: file not found: {args.predictions_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.predictions_csv)

    for col in (args.target_col, args.pred_col):
        if col not in df.columns:
            print(f"ERROR: column '{col}' not found in {args.predictions_csv}", file=sys.stderr)
            print(f"Available columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)

    if args.split_name is not None and args.split_col in df.columns:
        df = df[df[args.split_col] == args.split_name].reset_index(drop=True)
        if len(df) == 0:
            print(
                f"WARNING: no rows matched split '{args.split_name}' in column "
                f"'{args.split_col}'. Plotting all rows.",
                file=sys.stderr,
            )
            df = pd.read_csv(args.predictions_csv)

    targets = df[args.target_col].to_numpy(dtype=float)
    preds = df[args.pred_col].to_numpy(dtype=float)

    # Drop NaN pairs
    valid = np.isfinite(targets) & np.isfinite(preds)
    targets, preds = targets[valid], preds[valid]

    if len(targets) == 0:
        print("ERROR: no valid (finite) rows after filtering NaNs.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(targets)} samples from {args.predictions_csv}")

    os.makedirs(args.output_dir, exist_ok=True)
    metrics = _compute_metrics(targets, preds)

    # --- Generate plots ---
    saved: list[str] = []

    saved.append(plot_prediction_scatter(targets, preds, metrics, args.output_dir))
    saved.append(plot_residuals(targets, preds, args.output_dir))
    saved.append(plot_error_distribution(targets, preds, args.output_dir))

    split_label = args.split_name or "Test"
    saved.append(
        plot_metric_summary_table(
            {split_label: metrics},
            args.output_dir,
        )
    )

    print(f"\nAll plots saved to {os.path.abspath(args.output_dir)}/")
    for p in saved:
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
