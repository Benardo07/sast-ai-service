"""ResultPlotter — generates all evaluation plots to a results directory."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class ResultPlotter:
    """Generates and saves evaluation plots to a results directory."""

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = Path(results_dir)

    def plot_roc_curve(self, y_true, y_prob, class_names: list[str]) -> None:
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        n_classes = y_prob.shape[1]
        fig, ax = plt.subplots(figsize=(8, 6))
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            ax.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}")
        else:
            y_bin = label_binarize(y_true, classes=list(range(n_classes)))
            for i, name in enumerate(class_names):
                if y_bin[:, i].sum() == 0:
                    continue
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                ax.plot(fpr, tpr, label=f"{name} AUC={auc(fpr, tpr):.3f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
        ax.legend(loc="lower right", fontsize=7)
        fig.tight_layout()
        fig.savefig(self.results_dir / "roc_curve.png", dpi=150)
        plt.close(fig)

    def plot_confusion_matrix(self, y_true, y_pred, class_names: list[str]) -> None:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ticks = list(range(len(class_names)))
        ax.set(xticks=ticks, yticks=ticks, xticklabels=class_names, yticklabels=class_names,
               xlabel="Predicted", ylabel="True", title="Confusion Matrix")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        plt.setp(ax.get_yticklabels(), fontsize=7)
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=7)
        fig.tight_layout()
        fig.savefig(self.results_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)

    def plot_pr_curve(self, y_true, y_prob, class_names: list[str]) -> None:
        from sklearn.metrics import precision_recall_curve, auc
        from sklearn.preprocessing import label_binarize

        n_classes = y_prob.shape[1]
        fig, ax = plt.subplots(figsize=(8, 6))
        if n_classes == 2:
            prec, rec, _ = precision_recall_curve(y_true, y_prob[:, 1])
            ax.plot(rec, prec, label=f"AUC={auc(rec, prec):.3f}")
        else:
            y_bin = label_binarize(y_true, classes=list(range(n_classes)))
            for i, name in enumerate(class_names):
                if y_bin[:, i].sum() == 0:
                    continue
                prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
                ax.plot(rec, prec, label=f"{name} AUC={auc(rec, prec):.3f}")
        ax.set(xlabel="Recall", ylabel="Precision", title="PR Curve")
        ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout()
        fig.savefig(self.results_dir / "pr_curve.png", dpi=150)
        plt.close(fig)

    def plot_recall_at_loc_curve(self, k_values: list[float], recall_values: list[float]) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        k_pct = [k * 100 for k in k_values]
        r_pct = [r * 100 for r in recall_values]
        ax.plot(k_pct, r_pct, marker="o")
        ax.set(xlabel="Top-K% LOC inspected", ylabel="Flaw Recall (%)",
               title="Recall@K%LOC Curve")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.results_dir / "recall_at_loc_curve.png", dpi=150)
        plt.close(fig)

    def plot_ifa_distribution(self, ifa_values: list[float]) -> None:
        if not ifa_values:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(ifa_values, bins=30, edgecolor="black", alpha=0.75)
        ax.axvline(np.mean(ifa_values), color="red", linestyle="--",
                   label=f"Mean IFA = {np.mean(ifa_values):.2f}")
        ax.set(xlabel="Initial False Alarms", ylabel="Count", title="IFA Distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.results_dir / "ifa_distribution.png", dpi=150)
        plt.close(fig)
