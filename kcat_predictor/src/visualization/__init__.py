"""Visualization package for KicmBaseModel.

Exports all six plot functions for training diagnostics, prediction quality,
and attention weight inspection.
"""

from __future__ import annotations

from src.visualization.plots import (
    plot_attention_weights,
    plot_error_distribution,
    plot_metric_summary_table,
    plot_prediction_scatter,
    plot_residuals,
    plot_training_curves,
)

__all__ = [
    "plot_training_curves",
    "plot_prediction_scatter",
    "plot_residuals",
    "plot_error_distribution",
    "plot_metric_summary_table",
    "plot_attention_weights",
]
