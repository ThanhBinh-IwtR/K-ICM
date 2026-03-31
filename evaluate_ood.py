"""Post-hoc OOD evaluation script.

Loads a results CSV produced after inference, filters test samples by CD-HIT
cluster membership relative to the training set, then reports metrics on the
full test set, the IID (in-cluster) subset, and the OOD subset.

Usage::

    python evaluate_ood.py \\
        --results_csv outputs/test_predictions.csv \\
        --train_csv   data/train.csv \\
        --threshold   40 \\
        [--output_csv outputs/ood_40_results.csv]

Supported thresholds: 20, 40, 60, 80, 99.
Multiple thresholds can be evaluated in a single call::

    python evaluate_ood.py \\
        --results_csv outputs/test_predictions.csv \\
        --train_csv   data/train.csv \\
        --threshold   20 40 60 80 99
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import pandas as pd

try:
    from setproctitle import setproctitle
    setproctitle("kicm evaluate_ood")
except ImportError:
    pass

from src.evaluation.ood_evaluator import THRESHOLD_TO_COLUMN, OODEvaluator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_METRIC_KEYS: list[str] = ["r2", "rmse", "mae", "pearson", "spearman"]
_HEADER_ROW = (
    f"{'Split':<20} | {'R2':>7} | {'RMSE':>7} | {'MAE':>7} | "
    f"{'PCC':>7} | {'SCC':>7} | {'N':>7}"
)
_DIVIDER = "-" * len(_HEADER_ROW)


def _fmt(value: float, width: int = 7) -> str:
    """Format a float, displaying 'nan' if the value is not finite."""
    if math.isnan(value) or math.isinf(value):
        return f"{'N/A':>{width}}"
    return f"{value:>{width}.4f}"


def _print_table(
    rows: list[tuple[str, dict[str, float]]],
    threshold: int,
) -> None:
    """Print a formatted metrics comparison table to stdout.

    Args:
        rows: List of (split_label, metrics_dict) pairs.
        threshold: The CD-HIT threshold used (e.g. 40).
    """
    header = f"OOD-{threshold} EVALUATION RESULTS"
    border = "=" * len(_HEADER_ROW)

    print(f"\n{border}")
    print(f"{header:^{len(_HEADER_ROW)}}")
    print(border)
    print(_HEADER_ROW)
    print(_DIVIDER)

    for label, m in rows:
        n = int(m.get("n_samples", 0))
        print(
            f"{label:<20} | "
            f"{_fmt(m['r2'])} | "
            f"{_fmt(m['rmse'])} | "
            f"{_fmt(m['mae'])} | "
            f"{_fmt(m['pearson'])} | "
            f"{_fmt(m['spearman'])} | "
            f"{n:>7}"
        )

    print(border)

    # Summary line for OOD fraction
    ood_row = next((m for lbl, m in rows if "OOD" in lbl), None)
    full_row = next((m for lbl, m in rows if lbl == "Full Test"), None)
    if ood_row is not None and full_row is not None:
        n_ood = int(ood_row.get("n_samples", 0))
        n_full = int(full_row.get("n_samples", 0))
        pct = 100.0 * n_ood / max(n_full, 1)
        print(f"OOD-{threshold}: {n_ood}/{n_full} samples ({pct:.1f}%)")
    print(border)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc OOD evaluation: filter test predictions by CD-HIT cluster "
            "membership and compute metrics on full / IID / OOD subsets."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        required=True,
        help=(
            "Path to the results CSV produced after inference. "
            "Must contain sequence, smiles, the prediction column, the target "
            "column, and the cluster columns."
        ),
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="Path to the training CSV used to extract known cluster IDs.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        nargs="+",
        default=[40],
        choices=[20, 40, 60, 80, 99],
        metavar="THRESHOLD",
        help=(
            "CD-HIT identity threshold(s) to evaluate. "
            "One or more of: 20, 40, 60, 80, 99. Default: 40."
        ),
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="log10_value",
        help="Column name for ground-truth values. Default: log10_value.",
    )
    parser.add_argument(
        "--prediction_col",
        type=str,
        default="prediction",
        help="Column name for model predictions. Default: prediction.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help=(
            "Optional path to save the OOD-filtered rows to a CSV. "
            "When multiple thresholds are given, the threshold value is "
            "appended to the filename stem (e.g. ood_40_results.csv)."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core evaluation function (importable)
# ---------------------------------------------------------------------------


def run_ood_evaluation(
    results_csv: str | Path,
    train_csv: str | Path,
    threshold: int = 40,
    target_col: str = "log10_value",
    prediction_col: str = "prediction",
    output_csv: str | Path | None = None,
) -> dict[str, dict[str, float]]:
    """Load results and training CSVs, then compute OOD metrics.

    Prints a formatted table to stdout and optionally saves the OOD-filtered
    rows to ``output_csv``.

    Args:
        results_csv: Path to inference results CSV.
        train_csv: Path to training CSV (for cluster ID reference).
        threshold: CD-HIT identity threshold (20 | 40 | 60 | 80 | 99).
        target_col: Column name for ground-truth values.
        prediction_col: Column name for model predictions.
        output_csv: Optional path to write OOD-filtered rows as CSV.

    Returns:
        Dict with keys ``"full"``, ``"iid"``, ``"ood"`` → metric dicts.

    Raises:
        ValueError: If ``threshold`` is not in ``THRESHOLD_TO_COLUMN``.
        FileNotFoundError: If either CSV path does not exist.
    """
    if threshold not in THRESHOLD_TO_COLUMN:
        raise ValueError(
            f"Unsupported threshold {threshold}. "
            f"Supported: {sorted(THRESHOLD_TO_COLUMN.keys())}"
        )

    cluster_col = THRESHOLD_TO_COLUMN[threshold]

    results_path = Path(results_csv)
    if not results_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_path}")

    logger.info("[evaluate_ood] Loading results from %s", results_path)
    results_df = pd.read_csv(results_path)

    # Validate required columns
    missing: list[str] = []
    for col in (target_col, prediction_col, cluster_col):
        if col not in results_df.columns:
            missing.append(col)
    if missing:
        raise ValueError(
            f"Results CSV is missing required columns: {missing}. "
            f"Available: {list(results_df.columns)}"
        )

    evaluator = OODEvaluator(train_csv=train_csv, cluster_col=cluster_col)
    metrics = evaluator.evaluate(
        results_df,
        target_col=target_col,
        prediction_col=prediction_col,
    )

    # Print formatted table
    rows: list[tuple[str, dict[str, float]]] = [
        ("Full Test", metrics["full"]),
        ("IID (in-cluster)", metrics["iid"]),
        (f"OOD-{threshold}", metrics["ood"]),
    ]
    _print_table(rows, threshold=threshold)

    # Optionally save OOD-filtered rows
    if output_csv is not None:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ood_df = evaluator.filter_ood(results_df)
        ood_df.to_csv(out_path, index=False)
        logger.info(
            "[evaluate_ood] Saved %d OOD rows to %s",
            len(ood_df),
            out_path,
        )

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    )

    args = _parse_args()
    thresholds: list[int] = args.threshold

    any_failure = False
    for threshold in thresholds:
        # When multiple thresholds are requested and an output_csv is provided,
        # append the threshold to the filename stem to avoid collisions.
        out_csv: str | None = None
        if args.output_csv:
            if len(thresholds) == 1:
                out_csv = args.output_csv
            else:
                p = Path(args.output_csv)
                out_csv = str(p.parent / f"{p.stem}_{threshold}{p.suffix}")

        try:
            run_ood_evaluation(
                results_csv=args.results_csv,
                train_csv=args.train_csv,
                threshold=threshold,
                target_col=args.target_col,
                prediction_col=args.prediction_col,
                output_csv=out_csv,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[evaluate_ood] Failed for threshold=%d: %s", threshold, exc
            )
            any_failure = True

    if any_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
