"""Standalone evaluation script for a trained KICMModel checkpoint.

Usage::

    python evaluate.py \\
        --config configs/train_config.yaml \\
        --checkpoint outputs/checkpoints/epoch010-r20.7123.pt \\
        --test-csv data/test.csv \\
        [--split-name test] \\
        [--train-csv data/train.csv --ood-threshold 40] \\
        [--save-predictions outputs/test_predictions.csv]

Pass ``--train-csv`` and ``--ood-threshold`` to automatically run post-hoc OOD
evaluation after computing full-test metrics.  ``--save-predictions`` writes the
raw per-sample predictions to a CSV (required by ``evaluate_ood.py``).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
import yaml

try:
    from setproctitle import setproctitle
    setproctitle("kicm evaluate")
except ImportError:
    pass


logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained KICMModel on a CSV dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training config YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pt checkpoint file.",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        required=True,
        help="Path to evaluation CSV file.",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="test",
        help="Name used as prefix in metric keys (e.g. test_iid, test_ood_40).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation (single GPU).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write JSON metrics (e.g. metrics.json).",
    )
    # --- OOD evaluation options ---
    parser.add_argument(
        "--train-csv",
        type=str,
        default=None,
        help=(
            "Path to the training CSV. When provided together with "
            "--ood-threshold, post-hoc OOD evaluation is run automatically."
        ),
    )
    parser.add_argument(
        "--ood-threshold",
        type=int,
        default=40,
        choices=[20, 40, 60, 80, 99],
        metavar="THRESHOLD",
        help=(
            "CD-HIT identity threshold for OOD filtering "
            "(20 | 40 | 60 | 80 | 99). Default: 40. "
            "Only used when --train-csv is also provided."
        ),
    )
    parser.add_argument(
        "--save-predictions",
        type=str,
        default=None,
        help=(
            "Optional path to save per-sample predictions as a CSV. "
            "The CSV will include all original columns from the test CSV "
            "plus a 'prediction' column. Required by evaluate_ood.py."
        ),
    )
    parser.add_argument(
        "--lightning",
        action="store_true",
        help=(
            "Load checkpoint as a Lightning .ckpt file using "
            "KICMLightningModule.load_from_checkpoint() and run trainer.test(). "
            "Use this for checkpoints saved by the new Lightning train.py."
        ),
    )
    return parser.parse_args()


def save_predictions(
    test_csv: str | Path,
    preds: list[float],
    output_path: str | Path,
    target_col: str = "log10_value",
    prediction_col: str = "prediction",
) -> pd.DataFrame:
    """Merge model predictions with the original test CSV and save to disk.

    The output CSV contains all columns from ``test_csv`` plus a
    ``prediction_col`` column with the model's output values.  Rows that were
    dropped during dataset loading (NaN / invalid SMILES / long sequences) are
    **not** present — the caller is responsible for ensuring ``preds`` aligns
    with the filtered dataset rows, not the raw CSV.

    Args:
        test_csv: Path to the original test CSV.
        preds: List of predicted values aligned with the filtered dataset rows.
        output_path: Where to write the merged CSV.
        target_col: Ground-truth column name (for reference, not modified).
        prediction_col: Name for the new predictions column.

    Returns:
        The merged DataFrame that was saved.
    """
    # Read raw CSV (no filtering) to preserve all metadata columns including
    # cluster columns.  We only attach predictions to the rows that the
    # KICMDataset would have kept, so we replicate the same filtering here.
    from src.data import KICMDataset  # noqa: PLC0415

    df_raw = pd.read_csv(test_csv)

    # Replicate dataset filtering to build the aligned index.
    # We rely on the dataset's __init__ filtering logic: drop NaN, long seqs,
    # invalid SMILES.  The simplest safe approach is to iterate the dataset and
    # use the raw df index via a boolean mask.
    sequence_col = "sequence"
    smiles_col = "reactant_smiles"

    # Drop NaN rows (same three columns the dataset checks)
    valid_mask = df_raw[[sequence_col, smiles_col, target_col]].notna().all(axis=1)
    df_filtered = df_raw[valid_mask].copy()

    # Drop rows with SMILES that RDKit cannot parse
    from rdkit import Chem  # noqa: PLC0415
    valid_smiles = df_filtered[smiles_col].apply(
        lambda s: isinstance(s, str) and s.strip() != "" and Chem.MolFromSmiles(s) is not None
    )
    df_filtered = df_filtered[valid_smiles].reset_index(drop=True)

    if len(df_filtered) != len(preds):
        raise ValueError(
            f"Prediction count ({len(preds)}) does not match the number of "
            f"filtered test rows ({len(df_filtered)}). "
            "Ensure preds come from the same KICMDataset instance."
        )

    df_filtered[prediction_col] = preds

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(out_path, index=False)
    logger.info(
        "[evaluate] Saved %d predictions to %s",
        len(df_filtered),
        out_path,
    )
    return df_filtered


def evaluate(
    config: dict,
    checkpoint_path: str | Path,
    test_csv: str | Path,
    split_name: str = "test",
    batch_size: int = 32,
) -> tuple[dict[str, float], list[float]]:
    """Load a checkpoint and evaluate on a test CSV.

    Args:
        config: Full training config dict.
        checkpoint_path: Path to .pt checkpoint.
        test_csv: Path to evaluation CSV.
        split_name: Metric key prefix.
        batch_size: Evaluation batch size.

    Returns:
        Tuple of:
            - Dict of ``{split_name}_{metric}`` → float.
            - List of raw per-sample predictions (float, aligned with filtered
              dataset rows in the same order as KICMDataset loads them).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[evaluate] Using device: %s", device)

    # Load model
    from src.models import KICMModel  # noqa: PLC0415
    from src.data import KICMDataset, KICMCollator  # noqa: PLC0415
    from src.training.metrics import compute_metrics  # noqa: PLC0415

    model = KICMModel(config)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(
        "[evaluate] Loaded checkpoint from epoch %d  "
        "(checkpoint metrics: %s)",
        ckpt.get("epoch", -1),
        ckpt.get("metrics", {}),
    )

    # DataLoader
    data_cfg = config["data"]
    # ESM3Encoder has no alphabet attribute; the collator owns tokenisation.
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer  # noqa: PLC0415
    tokenizer = EsmSequenceTokenizer()
    collator = KICMCollator(tokenizer)

    dataset = KICMDataset(
        test_csv,
        max_seq_len=data_cfg["max_seq_len"],
        sequence_col=data_cfg["sequence_col"],
        smiles_col=data_cfg["smiles_col"],
        target_col=data_cfg["target_col"],
    )

    from torch.utils.data import DataLoader  # noqa: PLC0415

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=False,
        collate_fn=collator,
    )

    # Inference
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            tokens = batch["sequence_tokens"].to(device)
            esm_pad_mask = batch["esm_pad_mask"].to(device)
            smiles = batch["smiles"]
            targets = batch["targets"]  # [B] — collator now produces 1-D tensors

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                preds = model(tokens, esm_pad_mask, smiles)

            all_preds.append(preds.detach().cpu())
            all_targets.append(targets)

    preds_np = torch.cat(all_preds).float().numpy()
    targets_np = torch.cat(all_targets).float().numpy()

    metrics = compute_metrics(preds_np, targets_np)
    labeled = {f"{split_name}_{k}": v for k, v in metrics.items()}

    for k, v in labeled.items():
        logger.info("[evaluate]  %s = %.4f", k, v)

    raw_preds: list[float] = preds_np.tolist()
    return labeled, raw_preds


def evaluate_lightning(
    config: dict,
    checkpoint_path: str | Path,
    test_csv: str | Path,
    batch_size: int = 32,
) -> dict[str, float]:
    """Evaluate a Lightning checkpoint using trainer.test().

    Loads the model via KICMLightningModule.load_from_checkpoint(), overrides
    the test_csv in the config with the provided path, then runs trainer.test().

    Args:
        config: Full training config dict.
        checkpoint_path: Path to a Lightning .ckpt file.
        test_csv: Path to evaluation CSV.
        batch_size: Evaluation batch size (overrides config per-GPU batch size).

    Returns:
        Dict of metric names to float values as logged by KICMLightningModule.
    """
    import pytorch_lightning as pl  # noqa: PLC0415
    from src.models import KICMLightningModule  # noqa: PLC0415
    from src.data import KICMDataModule  # noqa: PLC0415

    # Patch config to point at the requested test CSV and batch size
    eval_config = {**config}
    eval_config["data"] = {**config["data"], "test_csv": str(test_csv)}
    eval_config["training"] = {
        **config["training"],
        "batch_size_per_gpu": batch_size,
    }

    model = KICMLightningModule.load_from_checkpoint(
        str(checkpoint_path),
        config=eval_config,
    )
    datamodule = KICMDataModule(eval_config)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_progress_bar=True,
    )
    results = trainer.test(model, datamodule=datamodule)
    # trainer.test returns a list with one dict per dataloader
    return results[0] if results else {}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    )

    args = _parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.lightning:
        # Lightning checkpoint path — use trainer.test()
        metrics = evaluate_lightning(
            config=config,
            checkpoint_path=args.checkpoint,
            test_csv=args.test_csv,
            batch_size=args.batch_size,
        )
        raw_preds = []  # Lightning aggregation does not return raw preds here
    else:
        metrics, raw_preds = evaluate(
            config=config,
            checkpoint_path=args.checkpoint,
            test_csv=args.test_csv,
            split_name=args.split_name,
            batch_size=args.batch_size,
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("[evaluate] Metrics written to %s", out_path)

    print("\nEvaluation results:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")

    # ------------------------------------------------------------------
    # Optional: save predictions CSV
    # ------------------------------------------------------------------
    predictions_csv_path: str | None = args.save_predictions

    if predictions_csv_path is not None and raw_preds:
        data_cfg = config["data"]
        merged_df = save_predictions(
            test_csv=args.test_csv,
            preds=raw_preds,
            output_path=predictions_csv_path,
            target_col=data_cfg["target_col"],
        )
    elif predictions_csv_path is not None and not raw_preds:
        logger.warning(
            "[evaluate] --save-predictions is not supported with --lightning "
            "(raw predictions are not returned by trainer.test). Skipping."
        )
        merged_df = None
    else:
        merged_df = None

    # ------------------------------------------------------------------
    # Optional: post-hoc OOD evaluation
    # ------------------------------------------------------------------
    if args.train_csv is not None and not args.lightning:
        # Build the results DataFrame if we have not already saved predictions.
        # If save_predictions was requested we can reuse the merged_df.
        if merged_df is None:
            import os  # noqa: PLC0415
            import tempfile  # noqa: PLC0415

            data_cfg = config["data"]

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as tmp:
                tmp_path = tmp.name

            try:
                merged_df = save_predictions(
                    test_csv=args.test_csv,
                    preds=raw_preds,
                    output_path=tmp_path,
                    target_col=data_cfg["target_col"],
                )
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        from evaluate_ood import _print_table  # noqa: PLC0415
        from src.evaluation.ood_evaluator import THRESHOLD_TO_COLUMN, OODEvaluator  # noqa: PLC0415

        cluster_col = THRESHOLD_TO_COLUMN[args.ood_threshold]
        if cluster_col not in merged_df.columns:
            logger.warning(
                "[evaluate] Cluster column '%s' not found in test CSV — "
                "skipping OOD evaluation. "
                "Ensure the test CSV contains CD-HIT cluster columns.",
                cluster_col,
            )
        else:
            evaluator = OODEvaluator(
                train_csv=args.train_csv,
                cluster_col=cluster_col,
            )
            ood_metrics = evaluator.evaluate(
                merged_df,
                target_col=config["data"]["target_col"],
                prediction_col="prediction",
            )
            rows = [
                ("Full Test", ood_metrics["full"]),
                ("IID (in-cluster)", ood_metrics["iid"]),
                (f"OOD-{args.ood_threshold}", ood_metrics["ood"]),
            ]
            _print_table(rows, threshold=args.ood_threshold)


if __name__ == "__main__":
    main()
