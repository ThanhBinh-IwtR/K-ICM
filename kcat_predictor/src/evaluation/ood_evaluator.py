"""Post-hoc OOD evaluation at various CD-HIT identity thresholds.

Usage example::

    from src.evaluation import OODEvaluator

    evaluator = OODEvaluator("data/train.csv", cluster_col="sequence_40cluster")
    results = evaluator.evaluate(
        results_df,
        target_col="log10_value",
        prediction_col="prediction",
    )
    # results["full"] — metrics on all test samples
    # results["iid"]  — metrics on in-distribution samples
    # results["ood"]  — metrics on OOD samples
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..training.metrics import compute_metrics

logger = logging.getLogger(__name__)

# Mapping from threshold integer to the expected cluster column name.
THRESHOLD_TO_COLUMN: dict[int, str] = {
    20: "sequence_20cluster",
    40: "sequence_40cluster",
    60: "sequence_60cluster",
    80: "sequence_80cluster",
    99: "sequence_99cluster",
}


def _safe_compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    subset_name: str,
) -> dict[str, float]:
    """Compute metrics; return NaN dict and warn when sample count < 2.

    Args:
        preds: Predicted values array.
        targets: Ground-truth values array.
        subset_name: Human-readable label used in warning messages.

    Returns:
        Dict with keys ``r2``, ``rmse``, ``mae``, ``pearson``, ``spearman``.
        All values are ``float("nan")`` when fewer than 2 samples are present.
    """
    nan_result = {
        "r2": float("nan"),
        "rmse": float("nan"),
        "mae": float("nan"),
        "pearson": float("nan"),
        "spearman": float("nan"),
        "n_samples": len(preds),
    }

    if len(preds) < 2:
        logger.warning(
            "[OODEvaluator] Subset '%s' has %d sample(s) — "
            "need at least 2 to compute metrics. Returning NaN.",
            subset_name,
            len(preds),
        )
        return nan_result

    metrics = compute_metrics(preds, targets)
    metrics["n_samples"] = len(preds)
    return metrics


class OODEvaluator:
    """Post-hoc OOD evaluator based on CD-HIT sequence cluster columns.

    The evaluator loads the training CSV once at construction time, extracts
    the set of cluster IDs present in training, then uses that set to partition
    any results DataFrame into IID (in-cluster) and OOD (unseen-cluster) subsets.

    Args:
        train_csv: Path to the training CSV. Must contain the ``cluster_col``
            column.
        cluster_col: Name of the CD-HIT cluster column to use for filtering.
            Defaults to ``"sequence_40cluster"`` (40 % identity threshold).
    """

    def __init__(
        self,
        train_csv: str | Path,
        cluster_col: str = "sequence_40cluster",
    ) -> None:
        train_csv = Path(train_csv)
        if not train_csv.exists():
            raise FileNotFoundError(f"Training CSV not found: {train_csv}")

        df_train = pd.read_csv(train_csv, usecols=[cluster_col])
        if cluster_col not in df_train.columns:
            raise ValueError(
                f"Column '{cluster_col}' not found in {train_csv}. "
                f"Available columns: {list(df_train.columns)}"
            )

        self.cluster_col = cluster_col
        self.train_clusters: frozenset[Any] = frozenset(
            df_train[cluster_col].dropna().unique()
        )

        logger.info(
            "[OODEvaluator] Loaded %d unique training clusters from '%s' (column='%s').",
            len(self.train_clusters),
            train_csv.name,
            cluster_col,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter_ood(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Return only OOD samples (cluster ID not seen during training).

        A sample is OOD when its ``cluster_col`` value does **not** appear in
        the set of training cluster IDs.

        Args:
            results_df: DataFrame that must contain the ``cluster_col`` column.

        Returns:
            Filtered DataFrame with only OOD rows; index is preserved.

        Raises:
            ValueError: If ``cluster_col`` is missing from ``results_df``.
        """
        self._check_cluster_col(results_df)
        is_ood = ~results_df[self.cluster_col].isin(self.train_clusters)
        return results_df[is_ood]

    def filter_iid(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Return only IID samples (cluster ID seen during training).

        Args:
            results_df: DataFrame that must contain the ``cluster_col`` column.

        Returns:
            Filtered DataFrame with only IID rows; index is preserved.

        Raises:
            ValueError: If ``cluster_col`` is missing from ``results_df``.
        """
        self._check_cluster_col(results_df)
        is_iid = results_df[self.cluster_col].isin(self.train_clusters)
        return results_df[is_iid]

    def evaluate(
        self,
        results_df: pd.DataFrame,
        target_col: str = "log10_value",
        prediction_col: str = "prediction",
    ) -> dict[str, dict[str, float]]:
        """Compute metrics on full test, IID, and OOD subsets.

        Args:
            results_df: DataFrame with at minimum ``target_col``,
                ``prediction_col``, and ``cluster_col`` columns.
            target_col: Column with ground-truth values.
            prediction_col: Column with model predictions.

        Returns:
            Dict with keys ``"full"``, ``"iid"``, ``"ood"``, each mapping to
            a metric dict (``r2``, ``rmse``, ``mae``, ``pearson``, ``spearman``,
            ``n_samples``).

        Raises:
            ValueError: If required columns are missing.
        """
        self._check_cluster_col(results_df)
        for col in (target_col, prediction_col):
            if col not in results_df.columns:
                raise ValueError(
                    f"Column '{col}' not found in results DataFrame. "
                    f"Available: {list(results_df.columns)}"
                )

        targets_full = results_df[target_col].to_numpy(dtype=np.float64)
        preds_full = results_df[prediction_col].to_numpy(dtype=np.float64)

        iid_df = self.filter_iid(results_df)
        ood_df = self.filter_ood(results_df)

        logger.info(
            "[OODEvaluator] Partition sizes — full=%d  iid=%d  ood=%d  "
            "(ood_fraction=%.1f%%)",
            len(results_df),
            len(iid_df),
            len(ood_df),
            100.0 * len(ood_df) / max(len(results_df), 1),
        )

        full_metrics = _safe_compute_metrics(preds_full, targets_full, "full")
        iid_metrics = _safe_compute_metrics(
            iid_df[prediction_col].to_numpy(dtype=np.float64),
            iid_df[target_col].to_numpy(dtype=np.float64),
            "iid",
        )
        ood_metrics = _safe_compute_metrics(
            ood_df[prediction_col].to_numpy(dtype=np.float64),
            ood_df[target_col].to_numpy(dtype=np.float64),
            "ood",
        )

        return {
            "full": full_metrics,
            "iid": iid_metrics,
            "ood": ood_metrics,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_cluster_col(self, df: pd.DataFrame) -> None:
        if self.cluster_col not in df.columns:
            raise ValueError(
                f"Column '{self.cluster_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
