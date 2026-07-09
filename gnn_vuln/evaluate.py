"""
evaluate.py — Evaluation entry point.

Usage:
    uv run evaluate --checkpoint checkpoints/<run_id>/best_*.pt
    uv run evaluate --checkpoint checkpoints/<run_id>/best_*.pt --config configs/...yaml
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from torch_geometric.loader import DataLoader

from gnn_vuln.config import Config, load_default_config
from gnn_vuln.data.dataset_lm import CodeBERTGraphDataset
from gnn_vuln.evaluation.localize import LocalizationExtractor
from gnn_vuln.evaluation.plots import ResultPlotter
from gnn_vuln.metrics import LocalizationMetrics
from gnn_vuln.models.registry import build_model
from gnn_vuln.utils import get_device, load_checkpoint, setup_logging


# ---------------------------------------------------------------------------
# Eval result — pure compute output; persistence is the caller's choice
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Everything `Evaluator.compute()` produces, in memory. The API consumes
    `.summary` directly (no disk); the research/CLI path persists it via
    `Evaluator.save_artifacts()`."""
    summary: dict
    y_true: "np.ndarray"
    y_pred: "np.ndarray"
    y_prob: "np.ndarray"
    confidence: "np.ndarray"
    correct_mask: "np.ndarray"
    target_names: list
    loc_results: list
    func_metrics: dict
    loc_metrics: LocalizationMetrics


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------

class Evaluator:
    """
    End-to-end evaluation: runs inference, computes metrics, writes all outputs.

    Outputs saved to results_dir/:
      predictions.csv, localization_scores.csv, metrics_summary.json,
      roc_curve.png, confusion_matrix.png, pr_curve.png,
      recall_at_loc_curve.png, ifa_distribution.png
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: CodeBERTGraphDataset,
        test_idx: list[int],
        device: torch.device,
        results_dir: Path,
        batch_size: int = 16,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.test_idx = test_idx
        self.device = device
        self.results_dir = Path(results_dir)
        self.batch_size = batch_size
        self.checkpoint_path: str = ""

        self.class_names: list[str] | None = getattr(dataset, "class_names", None)
        self.raw_funcs = getattr(dataset, "raw_funcs", None)
        self._loader = DataLoader(dataset[test_idx], batch_size=batch_size)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def compute(self) -> EvalResult:
        """Run inference + metrics. Returns an EvalResult in memory and writes
        NOTHING — the caller decides whether/where to persist (API: consume
        `.summary`; research: `save_artifacts`)."""
        logger.info("Running inference…")
        extractor = LocalizationExtractor(self.model, self._loader, self.device)
        y_true, y_pred, y_prob, confidence, loc_results = extractor.run()

        target_names = self.class_names or [str(i) for i in range(y_prob.shape[1])]
        correct_mask = y_true == y_pred

        func_metrics = self._function_level(y_true, y_pred, y_prob, confidence,
                                             correct_mask, target_names)
        loc_metrics  = LocalizationMetrics(loc_results)

        self._print_report(y_true, y_pred, target_names, func_metrics, loc_metrics,
                           loc_results, y_true, confidence, correct_mask)
        summary = self._build_summary(func_metrics, loc_metrics)
        return EvalResult(summary, y_true, y_pred, y_prob, confidence, correct_mask,
                          target_names, loc_results, func_metrics, loc_metrics)

    def run(self) -> dict:
        """Full evaluation + persist all research artifacts to results_dir
        (research/CLI behavior). Returns metrics_summary dict."""
        r = self.compute()
        self.save_artifacts(r)
        return r.summary

    # ------------------------------------------------------------------
    # Function-level metrics
    # ------------------------------------------------------------------

    def _function_level(self, y_true, y_pred, y_prob, confidence, correct_mask, target_names) -> dict:
        n_classes = y_prob.shape[1]
        # macro metrics average over classes PRESENT in y_true (F1 is undefined for a class with
        # no test samples). Passing labels= makes the denominator independent of what the model
        # predicts — without it sklearn averages over unique(y_true ∪ y_pred), which silently
        # includes a 0-support class as F1=0 the moment the model emits one stray prediction for it.
        present = np.unique(y_true)
        try:
            if n_classes == 2:
                auc_roc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                y_p = y_prob[:, present]
                y_p = y_p / y_p.sum(axis=1, keepdims=True)
                auc_roc = roc_auc_score(y_true, y_p, multi_class="ovr",
                                        average="macro", labels=present)
        except ValueError:
            auc_roc = float("nan")

        return {
            "accuracy": float((y_true == y_pred).mean()),
            "f1_macro": f1_score(y_true, y_pred, average="macro", labels=present, zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "auc_roc_macro_ovr": auc_roc,
            "confidence_mean":    float(confidence.mean()),
            "confidence_correct": float(confidence[correct_mask].mean()) if correct_mask.any() else None,
            "confidence_wrong":   float(confidence[~correct_mask].mean()) if (~correct_mask).any() else None,
            "num_classes": n_classes,
            "num_test_samples": int(len(y_true)),
        }

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------

    def _print_report(self, y_true, y_pred, target_names, func_metrics,
                      loc_metrics: LocalizationMetrics, loc_results, yt_arr,
                      confidence, correct_mask) -> None:
        n = len(target_names)
        print("\n" + "=" * 65)
        print("Function-Level Classification Report")
        print("=" * 65)
        print(classification_report(y_true, y_pred, labels=list(range(n)),
                                    target_names=target_names, zero_division=0))
        print(f"AUC-ROC (macro OvR) : {func_metrics['auc_roc_macro_ovr']:.4f}")
        print(f"F1 Score (macro)    : {func_metrics['f1_macro']:.4f}")
        print(f"F1 Score (weighted) : {func_metrics['f1_weighted']:.4f}")
        print(f"Accuracy            : {func_metrics['accuracy']:.4f}")
        print("=" * 65)

        n_gt = loc_metrics.num_funcs_with_flaw_gt
        print(f"\n{'=' * 65}")
        print(f"Statement-Level Localization  (functions with flaw GT: {n_gt})")
        print("=" * 65)
        if n_gt == 0:
            print("  No flaw-line ground truth found.")
        else:
            d = loc_metrics.to_dict()
            print(f"  Top-1  Accuracy    : {d['top_1_accuracy']:.4f}")
            print(f"  Top-3  Accuracy    : {d['top_3_accuracy']:.4f}")
            print(f"  Top-5  Accuracy    : {d['top_5_accuracy']:.4f}")
            print(f"  Top-10 Accuracy    : {d['top_10_accuracy']:.4f}")
            print(f"  IFA (mean)         : {d['ifa_mean']:.2f}")
            print(f"  Effort@20%Recall   : {d['effort_at_20pct_recall']:.4f}")
            print(f"  Recall@1%LOC       : {d['recall_at_1pct_loc']:.4f}")
            print(f"  Recall@5%LOC       : {d['recall_at_5pct_loc']:.4f}")
            print(f"  Recall@20%LOC      : {d['recall_at_20pct_loc']:.4f}")
            self._print_sample_lines(loc_results, yt_arr)
        print("=" * 65 + "\n")

    def _print_sample_lines(self, loc_results, y_true) -> None:
        print()
        print("  Sample — top-3 suspicious lines (first 3 vulnerable functions):")
        shown = 0
        for func_idx, (r, yt) in enumerate(zip(loc_results, y_true)):
            if int(yt) == 0 or shown >= 3:
                continue
            src_lines = self._get_src_lines(func_idx)
            print(f"  func {func_idx} (class={int(yt)}):")
            for ln, sc, lab in zip(r["ranked_line_numbers"][:3],
                                   r["ranked_scores"][:3], r["ranked_labels"][:3]):
                code   = src_lines[ln - 1].strip() if 0 < ln <= len(src_lines) else "<no code>"
                marker = "FLAW" if lab else "    "
                print(f"    [{marker}] line {ln:4d} score={sc:.3f}  {code[:60]}")
            shown += 1

    def _get_src(self, func_idx: int) -> tuple[list[str], int]:
        """Source lines + parquet_id for a test function. Reads raw_func/parquet_id
        per-sample off the graph so it works for lazy storage too — the inmem-only
        raw_funcs list is used as a fallback when the graph lacks raw_func."""
        ds_idx = self.test_idx[func_idx]
        g = self.dataset[ds_idx]
        raw = getattr(g, "raw_func", "") or ""
        if not raw and self.raw_funcs is not None:
            raw = self.raw_funcs[ds_idx] if ds_idx < len(self.raw_funcs) else ""
        pid = getattr(g, "parquet_id", None)
        try:
            pid = int(pid.item()) if hasattr(pid, "item") else int(pid)
        except (TypeError, ValueError):
            pid = -1
        return (raw.splitlines() if raw else []), pid

    def _get_src_lines(self, func_idx: int) -> list[str]:
        return self._get_src(func_idx)[0]

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    def _safe(self, v) -> object:
        if isinstance(v, float) and np.isnan(v):
            return None
        return v

    def _build_summary(self, func_metrics, loc_metrics: LocalizationMetrics) -> dict:
        """Build the metrics_summary dict — pure, no I/O. Used by compute()."""
        d = loc_metrics.to_dict()
        n_gt = loc_metrics.num_funcs_with_flaw_gt
        return {
            "function_level": {k: self._safe(v) for k, v in func_metrics.items()},
            "localization": {
                "top_1_accuracy":         self._safe(d["top_1_accuracy"]),
                "top_3_accuracy":         self._safe(d["top_3_accuracy"]),
                "top_5_accuracy":         self._safe(d["top_5_accuracy"]),
                "top_10_accuracy":        self._safe(d["top_10_accuracy"]),
                "ifa_mean":               self._safe(d["ifa_mean"]),
                "effort_at_20pct_recall": self._safe(d["effort_at_20pct_recall"]),
                "recall_at_1pct_loc":     self._safe(d["recall_at_1pct_loc"]),
                "recall_at_5pct_loc":     self._safe(d["recall_at_5pct_loc"]),
                "recall_at_20pct_loc":    self._safe(d["recall_at_20pct_loc"]),
                "num_funcs_with_flaw_gt": n_gt,
            },
            "localization_curve": {
                "k_values":     d["recall_at_loc_curve_k"],
                "recall_values":[self._safe(v) for v in d["recall_at_loc_curve_v"]],
            },
            "ifa_distribution": d["ifa_per_func"],
        }

    def save_artifacts(self, res: EvalResult) -> dict:
        """Persist research artifacts (predictions.csv, localization_scores.csv,
        metrics_summary.json, plots, copied config) to results_dir. Research/CLI
        ONLY — the API consumes res.summary directly and never calls this, so the
        bulky per-sample CSVs and plots are simply never written in the API path."""
        rd = self.results_dir
        rd.mkdir(parents=True, exist_ok=True)
        y_true, y_pred, y_prob, target_names = res.y_true, res.y_pred, res.y_prob, res.target_names

        # per-test-func (src_lines, parquet_id) — one graph load each, reused by both CSVs so
        # every row traces back to the exact source (code) + parquet row (parquet_id).
        src_cache = [self._get_src(i) for i in range(len(y_true))]

        # predictions.csv
        pred_df = pd.DataFrame({"parquet_id": [c[1] for c in src_cache],
                                 "y_true": y_true, "y_pred": y_pred,
                                 "confidence": res.confidence, "correct": res.correct_mask})
        for i, name in enumerate(target_names):
            pred_df[f"prob_{name}"] = y_prob[:, i]
        pred_df.to_csv(rd / "predictions.csv", index=False)
        logger.info(f"predictions.csv → {rd/'predictions.csv'}")

        # localization_scores.csv
        loc_rows: list[dict] = []
        for func_idx, (r, yt, yp) in enumerate(zip(res.loc_results, y_true, y_pred)):
            src_lines, pid = src_cache[func_idx]
            for ln, sc, lab in zip(r["line_numbers"], r["line_scores"], r["line_labels"]):
                code = src_lines[ln - 1].strip() if 0 < ln <= len(src_lines) else ""
                loc_rows.append({"func_idx": func_idx, "parquet_id": pid,
                                  "y_true": int(yt), "y_pred": int(yp),
                                  "line_number": int(ln), "score": round(float(sc), 6),
                                  "is_flaw_line": int(lab), "code": code})
        if loc_rows:
            pd.DataFrame(loc_rows).to_csv(rd / "localization_scores.csv", index=False)
            logger.info(f"localization_scores.csv → {rd/'localization_scores.csv'}")
        else:
            logger.warning("No localization data collected (node_line not in dataset).")

        # metrics_summary.json
        with open(rd / "metrics_summary.json", "w") as f:
            json.dump(res.summary, f, indent=2)
        logger.info(f"metrics_summary.json → {rd/'metrics_summary.json'}")

        # Plots
        plotter = ResultPlotter(rd)
        plotter.plot_roc_curve(y_true, y_prob, self.class_names or target_names)
        plotter.plot_confusion_matrix(y_true, y_pred, target_names)
        plotter.plot_pr_curve(y_true, y_prob, self.class_names or target_names)
        if res.loc_metrics.num_funcs_with_flaw_gt > 0:
            k_vals, recall_vals = res.loc_metrics.recall_at_loc_curve
            plotter.plot_recall_at_loc_curve(k_vals, recall_vals)
            plotter.plot_ifa_distribution(res.loc_metrics.to_dict()["ifa_per_func"])

        # Copy config + training files from checkpoint dir to results dir
        ckpt_dir = Path(self.checkpoint_path).parent
        for fname in ("config.yaml", "training_log.csv", "training_summary.json", "training_curves.png"):
            src = ckpt_dir / fname
            if src.exists():
                import shutil as _shutil
                _shutil.copy2(src, rd / fname)
                logger.info(f"Copied {fname} → {rd/fname}")

        logger.info(f"All results saved to {rd}/")
        return res.summary

    def save_summary(self, res: EvalResult) -> dict:
        """API path: write ONLY metrics_summary.json (the tiny handoff the caller
        reads back). No per-sample CSVs, no plots — those are research artifacts the
        API never uses, so nothing bulky lands on disk."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.results_dir / "metrics_summary.json", "w") as f:
            json.dump(res.summary, f, indent=2)
        logger.info(f"metrics_summary.json → {self.results_dir/'metrics_summary.json'} (metrics-only)")
        return res.summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained vulnerability detector")
    parser.add_argument("--checkpoint", required=True, help="Path to best_*.pt checkpoint")
    parser.add_argument("--config", default=None, help="Config YAML (auto-detected if omitted)")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Write only metrics_summary.json, skip per-sample CSVs + plots (API path).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override cfg.train.seed to match a multi-seed run's split.")
    parser.add_argument("--split-seed", type=int, default=None,
                        help="Override the split seed to match the model's training split.")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else Path(args.checkpoint).parent / "config.yaml"
    cfg = Config.from_yaml(config_path) if config_path.exists() else load_default_config()
    if not (args.config or config_path.exists()):
        logger.warning("No config.yaml found, using defaults.")
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.split_seed is not None:
        cfg.data.split_seed = args.split_seed

    setup_logging(cfg.train.log_dir)
    device = get_device(cfg.train.device)

    pretrained_lm    = getattr(cfg.model, "pretrained_lm", "microsoft/codebert-base")
    func_lm          = getattr(cfg.model, "func_lm", "") or pretrained_lm
    add_func_tokens  = getattr(cfg.model, "add_func_tokens", False)
    func_lm_source   = getattr(cfg.model, "func_lm_source", "raw")

    logger.info("Loading dataset…")
    dataset = CodeBERTGraphDataset(
        root=str(cfg.data.processed_dir.parent),
        max_nodes=cfg.data.max_nodes,
        embedder_device=cfg.train.device,
        mode=cfg.data.mode,
        source=getattr(cfg.data, "source", "bigvul"),
        target_vocab=getattr(cfg.data, "target_vocab", None),
        pretrained_lm=pretrained_lm,
        func_lm=func_lm,
        add_func_tokens=add_func_tokens,
        func_lm_source=func_lm_source,
        top_cwe=getattr(cfg.data, "top_cwe", 0),
        cwe_list=getattr(cfg.data, "cwe_list", None),
        cwe_groups=getattr(cfg.data, "cwe_groups", None),
        filter_owasp=getattr(cfg.data, "filter_owasp", False),
        filter_top25_dangerous=getattr(cfg.data, "filter_top25_dangerous", False),
        max_per_class=getattr(cfg.data, "max_per_class", 0),
        resample_seed=getattr(cfg.data, "resample_seed", 42),
        func_max_length=getattr(cfg.model, "func_max_length", 512),
        storage=getattr(cfg.data, "storage", "inmemory"),
        ds_name_suffix=getattr(cfg.data, "ds_name_suffix", ""),
    )
    _, _, test_idx = dataset.get_splits(
        train_ratio=getattr(cfg.data, "train_ratio", 0.8),
        val_ratio=getattr(cfg.data, "val_ratio", 0.1),
        seed=getattr(cfg.data, "split_seed", None) or cfg.train.seed,
    )
    if not test_idx:
        logger.info("Empty test split (train_ratio + val_ratio = 1.0) — evaluation skipped (prod 90/10/0).")
        return

    in_channels = dataset[0].x.size(1)
    model = build_model(cfg, in_channels).to(device)
    load_checkpoint(model, args.checkpoint, device=str(device))
    logger.info(f"Model loaded from {args.checkpoint}")
    from gnn_vuln.models.heads import StmtHead
    for m in model.modules():
        if isinstance(m, StmtHead):
            m._vectorized = True

    run_id = Path(args.checkpoint).parent.name
    results_dir = cfg.train.results_dir / run_id

    evaluator = Evaluator(
        model=model,
        dataset=dataset,
        test_idx=test_idx,
        device=device,
        results_dir=results_dir,
        batch_size=cfg.train.batch_size,
    )
    evaluator.checkpoint_path = args.checkpoint
    if args.metrics_only or os.environ.get("GNN_VULN_API_MODE") == "1":
        evaluator.save_summary(evaluator.compute())
    else:
        evaluator.run()


if __name__ == "__main__":
    main()
