"""
metrics.py — Vulnerability localization metrics.

Adapted from LineVul (Fu & Tantithamthavorn, MSR 2022), WAVES, VulLMGNN, VulChecker.

Main class: LocalizationMetrics — compute all localization metrics from per-function results.
Helper: make_func_loc_result — build canonical per-function result dict.
"""

from __future__ import annotations

import numpy as np


class LocalizationMetrics:
    """
    Computes all localization metrics from per-function result dicts.

    Only functions with flaw-line ground truth (num_flaw_lines > 0) are scored.
    NaN is returned when no such functions exist.

    Usage
    -----
        m = LocalizationMetrics(loc_results)
        print(m.top_1_accuracy, m.ifa_mean, m.effort_at_20pct_recall)
        summary = m.to_dict()
    """

    def __init__(self, func_results: list[dict]) -> None:
        self.func_results = func_results
        self.gt_results   = [r for r in func_results if r.get("num_flaw_lines", 0) > 0]
        self._ifa_mean, self._ifa_per_func = self._ifa_metric(self.gt_results)
        self._k_vals, self._recall_curve   = self._recall_at_loc_curve(self.gt_results)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def num_funcs_with_flaw_gt(self) -> int:
        return len(self.gt_results)

    @property
    def top_1_accuracy(self) -> float:
        return self._top_k_accuracy(self.gt_results, k=1)

    @property
    def top_3_accuracy(self) -> float:
        return self._top_k_accuracy(self.gt_results, k=3)

    @property
    def top_5_accuracy(self) -> float:
        return self._top_k_accuracy(self.gt_results, k=5)

    @property
    def top_10_accuracy(self) -> float:
        return self._top_k_accuracy(self.gt_results, k=10)

    @property
    def ifa_mean(self) -> float:
        return self._ifa_mean

    @property
    def ifa_per_func(self) -> list[float]:
        return self._ifa_per_func

    @property
    def effort_at_20pct_recall(self) -> float:
        return self._effort_at_k_recall(self.gt_results, k=0.20)

    @property
    def recall_at_1pct_loc(self) -> float:
        return self._recall_at_k_loc(self.gt_results, k=0.01)

    @property
    def recall_at_5pct_loc(self) -> float:
        return self._recall_at_k_loc(self.gt_results, k=0.05)

    @property
    def recall_at_20pct_loc(self) -> float:
        return self._recall_at_k_loc(self.gt_results, k=0.20)

    @property
    def recall_at_loc_curve(self) -> tuple[list[float], list[float]]:
        return self._k_vals, self._recall_curve

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        k_vals, recall_curve = self.recall_at_loc_curve
        return {
            "top_1_accuracy":         self.top_1_accuracy,
            "top_3_accuracy":         self.top_3_accuracy,
            "top_5_accuracy":         self.top_5_accuracy,
            "top_10_accuracy":        self.top_10_accuracy,
            "ifa_mean":               self.ifa_mean,
            "effort_at_20pct_recall": self.effort_at_20pct_recall,
            "recall_at_1pct_loc":     self.recall_at_1pct_loc,
            "recall_at_5pct_loc":     self.recall_at_5pct_loc,
            "recall_at_20pct_loc":    self.recall_at_20pct_loc,
            "recall_at_loc_curve_k":  k_vals,
            "recall_at_loc_curve_v":  recall_curve,
            "ifa_per_func":           self.ifa_per_func,
            "num_funcs_with_flaw_gt": self.num_funcs_with_flaw_gt,
        }

    # ------------------------------------------------------------------
    # Static methods
    # ------------------------------------------------------------------

    @staticmethod
    def make_result(
        line_numbers: list[int],
        line_scores: list[float],
        line_labels: list[int],
    ) -> dict:
        """Build canonical per-function localization result dict."""
        assert len(line_numbers) == len(line_scores) == len(line_labels)
        order = sorted(range(len(line_scores)), key=lambda i: line_scores[i], reverse=True)
        return {
            "line_numbers":        line_numbers,
            "line_scores":         line_scores,
            "line_labels":         line_labels,
            "ranked_scores":       [line_scores[i] for i in order],
            "ranked_labels":       [line_labels[i] for i in order],
            "ranked_line_numbers": [line_numbers[i] for i in order],
            "num_lines":           len(line_scores),
            "num_flaw_lines":      sum(line_labels),
        }

    @staticmethod
    def _top_k_accuracy(results: list[dict], k: int) -> float:
        """Fraction of functions where ≥1 flaw line is in top-k ranked lines."""
        correct = total = 0
        for r in results:
            if r["num_flaw_lines"] == 0:
                continue
            total += 1
            if 1 in r["ranked_labels"][:k]:
                correct += 1
        return correct / total if total > 0 else float("nan")

    @staticmethod
    def _ifa_metric(results: list[dict]) -> tuple[float, list[float]]:
        """Initial False Alarms: clean lines inspected before first flaw line."""
        values: list[float] = []
        for r in results:
            if r["num_flaw_lines"] == 0:
                continue
            clean = 0
            for label in r["ranked_labels"]:
                if label == 1:
                    break
                clean += 1
            values.append(float(clean))
        mean = float(np.mean(values)) if values else float("nan")
        return mean, values

    @staticmethod
    def _effort_at_k_recall(results: list[dict], k: float = 0.20) -> float:
        """Fraction of total lines to inspect to catch k-fraction of all flaw lines (global ranking)."""
        all_scored: list[tuple[float, int]] = []
        total_lines = total_flaw = 0
        for r in results:
            if r["num_flaw_lines"] == 0:
                continue
            for sc, lab in zip(r["ranked_scores"], r["ranked_labels"]):
                all_scored.append((sc, lab))
            total_lines += r["num_lines"]
            total_flaw  += r["num_flaw_lines"]
        if total_flaw == 0 or total_lines == 0:
            return float("nan")
        all_scored.sort(key=lambda x: x[0], reverse=True)
        target = max(1, int(total_flaw * k))
        caught = inspected = 0
        for _, lab in all_scored:
            inspected += 1
            caught    += lab
            if caught >= target:
                break
        return round(inspected / total_lines, 4)

    @staticmethod
    def _recall_at_k_loc(results: list[dict], k: float = 0.01) -> float:
        """Fraction of flaw lines caught when inspecting top-K% of all lines (global ranking)."""
        all_scored: list[tuple[float, int]] = []
        total_lines = total_flaw = 0
        for r in results:
            if r["num_flaw_lines"] == 0:
                continue
            for sc, lab in zip(r["ranked_scores"], r["ranked_labels"]):
                all_scored.append((sc, lab))
            total_lines += r["num_lines"]
            total_flaw  += r["num_flaw_lines"]
        if total_flaw == 0 or total_lines == 0:
            return float("nan")
        all_scored.sort(key=lambda x: x[0], reverse=True)
        target_inspect = max(1, int(total_lines * k))
        caught = sum(lab for _, lab in all_scored[:target_inspect])
        return round(caught / total_flaw, 4)

    @staticmethod
    def _recall_at_loc_curve(
        results: list[dict],
        k_values: list[float] | None = None,
    ) -> tuple[list[float], list[float]]:
        """Recall@K%LOC for a range of K values."""
        if k_values is None:
            k_values = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
        recalls = [LocalizationMetrics._recall_at_k_loc(results, k) for k in k_values]
        return k_values, recalls


# ---------------------------------------------------------------------------
# Backward-compat module-level wrappers
# ---------------------------------------------------------------------------

def make_func_loc_result(line_numbers, line_scores, line_labels) -> dict:
    return LocalizationMetrics.make_result(line_numbers, line_scores, line_labels)

def compute_all_localization_metrics(func_results: list[dict]) -> dict:
    return LocalizationMetrics(func_results).to_dict()

def top_k_accuracy(func_results, k):
    return LocalizationMetrics._top_k_accuracy(func_results, k)

def top_10_accuracy(func_results):
    return LocalizationMetrics._top_k_accuracy(func_results, k=10)

def ifa_metric(func_results):
    return LocalizationMetrics._ifa_metric(func_results)

def effort_at_k_recall(func_results, k=0.20):
    return LocalizationMetrics._effort_at_k_recall(func_results, k)

def recall_at_k_loc(func_results, k=0.01):
    return LocalizationMetrics._recall_at_k_loc(func_results, k)

def recall_at_k_loc_curve(func_results, k_values=None):
    return LocalizationMetrics._recall_at_loc_curve(func_results, k_values)
