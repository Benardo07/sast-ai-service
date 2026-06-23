"""
baselines.py - Published baseline results for comparison at evaluation time.

Sources (all numbers extracted directly from the PDFs)
------------------------------------------------------
LineVul  - Fu & Tantithamthavorn, MSR 2022  (BigVul / Fan et al. dataset)
           Function-level: README Appendix Table (RQ1)
           Localization:   README Appendix Tables (RQ2 / RQ3)
VulLMGNN - Cao et al., ICSE 2023
           Accuracy on DiverseVul/Devign/VDSIC/ReVeal; no BigVul numbers
WAVES    - Gu et al., 2023
           Function-level: Table 2 (Fan et al. dataset)
           Localization:   Table 3 (Fan et al. dataset)
           Per-CWE TPR:    Table 9

Dataset notes
-------------
LineVul RQ1/RQ2/RQ3 use BigVul (Fan et al.) - full dataset, no class cap.
WAVES Table 2/3 use BigVul (Fan et al.) subset.
Our evaluation uses MegaVul subsampled to 1600/class with a 80/10/10 split.
=> Function-level numbers are NOT perfectly comparable (different splits + class
   balance), but localization metrics share the same definition and can be compared.

Metric definitions
------------------
f1_binary : sklearn f1_score(average="binary") - positive class only (vulnerable=1)
f1_macro  : sklearn f1_score(average="macro")
top_N_accuracy : at least one flaw line in top-N ranked statements
ifa_mean       : mean clean lines inspected before first flaw line (lower = better)
mfr            : mean first ranking (rank of first flaw line, 1-indexed)
mar            : mean average ranking across all flaw lines
effort_at_20pct_recall : fraction of code to inspect for 20% flaw recall (lower = better)
recall_at_1pct_loc     : flaw recall at top 1% of lines (higher = better)

None = metric not reported in the paper for that dataset.

WARNING — numbers here are NOT directly comparable to our evaluate.py output:
  1. Dataset split: baselines use full BigVul; we use 2000/class subsample.
  2. Localization denominator: baselines may include all vulnerable functions;
     we compute only over functions with flaw-line ground truth (flaw_line_mask > 0).
  3. Task: baselines are binary; our primary task is 11-class multiclass.
These numbers are stored for reference only. Fair comparison requires running
LineVul / VulLMGNN on our exact dataset split with matching evaluation code.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# BigVul binary -- primary comparison dataset
# ---------------------------------------------------------------------------

# LineVul README (RQ1) -- function-level
# Note: F1/Precision/Recall use average="binary" (vulnerable-class only), not macro.
LINEVUL_FUNC: dict[str, float | None] = {
    "f1_binary": 0.91,       # binary F1 for vulnerable class (NOT macro)
    "precision": 0.97,
    "recall": 0.86,
    "f1_macro": None,        # not reported
    "accuracy": None,        # not reported (only F1/P/R in RQ1)
}

# LineVul README (RQ2) -- localization, best method = Self-Attention
LINEVUL_LOC_ATTENTION: dict[str, float | None] = {
    "top_1_accuracy": 0.10,
    "top_3_accuracy": 0.31,
    "top_5_accuracy": 0.46,
    "top_10_accuracy": 0.65,
    "ifa_mean": 4.56,
    "mfr": None,
    "mar": None,
    "effort_at_20pct_recall": 0.0075,   # RQ3 table
    "recall_at_1pct_loc": 0.24,          # RQ3 table
}

# LineVul other localization methods (from README RQ2/RQ3)
LINEVUL_LOC_OTHER: dict[str, dict] = {
    "LIG": {
        "top_10_accuracy": 0.53, "ifa_mean": 8.31,
        "effort_at_20pct_recall": 0.0106, "recall_at_1pct_loc": 0.19,
    },
    "Saliency": {
        "top_10_accuracy": 0.58, "ifa_mean": 6.93,
        "effort_at_20pct_recall": 0.0151, "recall_at_1pct_loc": 0.13,
    },
    "DeepLift": {
        "top_10_accuracy": 0.57, "ifa_mean": 6.27,
        "effort_at_20pct_recall": 0.0151, "recall_at_1pct_loc": 0.13,
    },
    "CppCheck": {
        "top_10_accuracy": 0.15, "ifa_mean": 21.60,
        "effort_at_20pct_recall": 0.13, "recall_at_1pct_loc": 0.04,
    },
}

# LineVul README (RQ1) -- other function-level baselines on BigVul
OTHER_FUNC_BASELINES: dict[str, dict] = {
    "IVDetect":     {"f1_binary": 0.35, "precision": 0.23, "recall": 0.72},
    "Reveal":       {"f1_binary": 0.30, "precision": 0.19, "recall": 0.74},
    "SySeVR":       {"f1_binary": 0.27, "precision": 0.15, "recall": 0.74},
    "Devign":       {"f1_binary": 0.26, "precision": 0.18, "recall": 0.52},
    "BoW+RF":       {"f1_binary": 0.25, "precision": 0.48, "recall": 0.17},
}

# WAVES Table 2 -- function-level on Fan et al. (BigVul subset)
# Note: different split than LineVul RQ1 -> lower numbers for LineVul here
WAVES_FUNC: dict[str, float | None] = {
    "accuracy": 0.977, "f1_binary": 0.607, "precision": 0.724, "recall": 0.522,
}
LINEVUL_WAVES_FUNC: dict[str, float | None] = {
    "accuracy": 0.972, "f1_binary": 0.516, "precision": 0.632, "recall": 0.436,
}

# WAVES Table 3 -- statement-level localization on Fan et al. (BigVul subset)
WAVES_LOC: dict[str, float | None] = {
    "top_1_accuracy": 0.283,
    "top_3_accuracy": 0.484,
    "top_5_accuracy": 0.609,
    "top_10_accuracy": None,           # not reported in WAVES
    "ifa_mean": 5.46,
    "mfr": 6.46,
    "mar": 9.08,
    "effort_at_20pct_recall": None,    # not reported in WAVES
    "recall_at_1pct_loc": None,        # not reported in WAVES
}
LINEVUL_WAVES_LOC: dict[str, float | None] = {
    "top_1_accuracy": 0.005,
    "top_3_accuracy": 0.252,
    "top_5_accuracy": 0.375,
    "top_10_accuracy": None,
    "ifa_mean": 6.17,
    "mfr": 7.17,
    "mar": 9.49,
}

# ---------------------------------------------------------------------------
# VulLMGNN -- accuracy on other datasets (no BigVul, no localization)
# ---------------------------------------------------------------------------

VULLMGNN_ACCURACY: dict[str, dict] = {
    "DiverseVul": {
        "VulLMGNN": 0.9306, "GraphCodeBERT": 0.9296,
        "CodeBERT": 0.9240, "BERT": 0.9199,
    },
    "Devign": {
        "VulLMGNN": 0.6570, "GraphCodeBERT": 0.6480,
        "CodeBERT": 0.6480, "BERT": 0.6058,
    },
    "VDSIC": {
        "VulLMGNN": 0.8438, "GraphCodeBERT": 0.8398,
        "CodeBERT": 0.8313, "BERT": 0.7941,
    },
    "ReVeal": {
        "VulLMGNN": 0.9080, "GraphCodeBERT": 0.8925,
        "CodeBERT": 0.8864, "BERT": 0.8688,
    },
}

# ---------------------------------------------------------------------------
# WAVES per-CWE TPR (Table 9, Fan et al. dataset)
# ---------------------------------------------------------------------------

WAVES_PER_CWE_TPR: dict[str, float] = {
    "CWE-787": 0.500,   # 7/14
    "CWE-416": 0.615,   # 8/13
    "CWE-20":  0.566,   # 61/108
    "CWE-125": 0.542,   # 13/24
    "CWE-476": 0.556,   # 5/9
    "CWE-190": 0.778,   # 14/18
    "CWE-119": 0.560,   # 70/125
    "CWE-362": 0.524,   # 11/21
    "CWE-284": 0.375,   # 3/8
    "CWE-189": 0.526,   # 10/19
    "CWE-732": 0.571,   # 4/7
    "CWE-254": 0.444,   # 4/9
    "CWE-200": 0.429,   # 12/28
    "CWE-415": 0.714,   # 5/7
    "CWE-399": 0.487,   # 19/39
}

# LineVul per-CWE TPR (README table, BigVul full)
LINEVUL_PER_CWE_TPR: dict[str, float] = {
    "CWE-787": 0.75,   # 18/24
    "CWE-20":  0.86,   # 98/114
    "CWE-22":  1.00,   # 4/4
    "CWE-190": 0.90,   # 27/30
    "CWE-119": 0.88,   # 173/197
    "CWE-200": 0.85,   # 45/53
    "CWE-77":  1.00,   # 2/2
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sign(ours: float, theirs: float, higher_better: bool) -> str:
    if higher_better:
        return " [+]" if ours > theirs else (" [-]" if ours < theirs else " [=]")
    else:
        return " [+]" if ours < theirs else (" [-]" if ours > theirs else " [=]")


def print_binary_comparison(our_metrics: dict) -> None:
    """
    Print a comparison table of our binary results vs published baselines.

    our_metrics keys (all optional -- missing values shown as ---):
      f1_macro, f1_binary, precision, recall, accuracy,
      top_1_accuracy, top_3_accuracy, top_5_accuracy, top_10_accuracy,
      ifa_mean, effort_at_20pct_recall, recall_at_1pct_loc
    """
    o = our_metrics

    print("\n" + "=" * 72)
    print("Comparison vs Published Baselines  (dataset: BigVul binary)")
    print("=" * 72)
    print("Notes:")
    print("  - LineVul F1/P/R: average='binary' (vulnerable-class only), not macro.")
    print("  - Our F1: average='macro' -- directly comparable to classification report.")
    print("  - Localization metrics share identical definitions -> directly comparable.")
    print("  - WAVES/LineVul(WAVES) use a different BigVul split -> approx. only.")
    print("-" * 72)

    # Function-level
    print(f"\n{'Model':<30} {'F1':>8} {'Prec':>8} {'Recall':>8}  Note")
    print("-" * 65)

    our_f1b = o.get("f1_binary")
    our_f1m = o.get("f1_macro")
    our_p   = o.get("precision")
    our_r   = o.get("recall")

    f1_str = (f"{our_f1m:.4f}(m)" if our_f1m is not None
              else (f"{our_f1b:.4f}(b)" if our_f1b is not None else "   --- "))
    p_str  = f"{our_p:.4f}" if our_p  is not None else "   ---"
    r_str  = f"{our_r:.4f}" if our_r  is not None else "   ---"
    print(f"{'Ours (LM-GCN)':<30} {f1_str:>8} {p_str:>8} {r_str:>8}  macro F1")

    for label, b in [
        ("LineVul (BigVul full)",       LINEVUL_FUNC),
        ("LineVul (WAVES split)",       LINEVUL_WAVES_FUNC),
        ("WAVES   (WAVES split)",       WAVES_FUNC),
        *[(f"  {n}", v) for n, v in OTHER_FUNC_BASELINES.items()],
    ]:
        bf1 = b.get("f1_binary")
        bp  = b.get("precision")
        br  = b.get("recall")
        bf1_str = f"{bf1:.4f}" if bf1 is not None else "   ---"
        bp_str  = f"{bp:.4f}"  if bp  is not None else "   ---"
        br_str  = f"{br:.4f}"  if br  is not None else "   ---"
        print(f"{label:<30} {bf1_str:>8} {bp_str:>8} {br_str:>8}  binary F1")

    # Localization
    print(f"\n{'Model':<30} {'Top-1':>7} {'Top-3':>7} {'Top-5':>7} {'Top-10':>8} "
          f"{'IFA':>6} {'Eff@20':>8} {'R@1%':>7}")
    print("-" * 78)

    def _fmt(val: float | None, ours: float | None, hb: bool, w: int = 5) -> str:
        if val is None:
            return "  ---"
        s = f"{val:.3f}"
        if ours is not None:
            s += _sign(ours, val, hb)
        return s

    t1  = o.get("top_1_accuracy")
    t3  = o.get("top_3_accuracy")
    t5  = o.get("top_5_accuracy")
    t10 = o.get("top_10_accuracy")
    ifa = o.get("ifa_mean")
    eff = o.get("effort_at_20pct_recall")
    r1  = o.get("recall_at_1pct_loc")

    def _v(v, fmt=".3f"):
        return f"{v:{fmt}}" if v is not None else "  ---"

    print(f"{'Ours (LM-GCN)':<30} "
          f"{_v(t1):>7} {_v(t3):>7} {_v(t5):>7} "
          f"{_v(t10):>8} {_v(ifa, '.2f'):>6} "
          f"{_v(eff):>8} {_v(r1):>7}")

    rows = [
        ("LineVul-Attn (BigVul)",  LINEVUL_LOC_ATTENTION),
        ("LineVul (WAVES split)",  LINEVUL_WAVES_LOC),
        ("WAVES   (WAVES split)",  WAVES_LOC),
        *[(f"LineVul-{k}", v) for k, v in LINEVUL_LOC_OTHER.items()],
    ]
    for label, b in rows:
        bt1  = b.get("top_1_accuracy")
        bt3  = b.get("top_3_accuracy")
        bt5  = b.get("top_5_accuracy")
        bt10 = b.get("top_10_accuracy")
        bifa = b.get("ifa_mean")
        beff = b.get("effort_at_20pct_recall")
        br1  = b.get("recall_at_1pct_loc")
        print(f"{label:<30} "
              f"{_fmt(bt1,  t1,  True):>7} "
              f"{_fmt(bt3,  t3,  True):>7} "
              f"{_fmt(bt5,  t5,  True):>7} "
              f"{_fmt(bt10, t10, True):>8} "
              f"{_fmt(bifa, ifa, False):>6} "
              f"{_fmt(beff, eff, False):>8} "
              f"{_fmt(br1,  r1,  True):>7}")

    print("\n[+] = our model is better   [-] = baseline is better   --- = not reported")
    print("(m) = macro F1   (b) = binary/class-1 F1 -- not directly comparable")
    print("=" * 72 + "\n")


def print_multiclass_comparison(our_metrics: dict, class_names: list[str] | None = None) -> None:
    """
    Print comparison for multiclass CWE mode.
    No published baseline does 10-CWE multiclass on BigVul, so we show
    per-CWE recall vs WAVES / LineVul binary per-CWE TPR.
    """
    print("\n" + "=" * 72)
    print("Comparison vs Published Baselines  (dataset: BigVul multiclass)")
    print("=" * 72)
    print("No published baseline performs 10-CWE multiclass classification on")
    print("BigVul. Closest comparison: per-CWE recall (TPR) from WAVES / LineVul.")
    print("(Both report binary per-CWE recall on Fan et al. dataset.)\n")

    our_per_class = our_metrics.get("per_class_recall", {})

    cwes_shown = sorted(
        set(WAVES_PER_CWE_TPR) | set(LINEVUL_PER_CWE_TPR) | set(our_per_class)
    )

    print(f"{'CWE':<12} {'Ours':>7} {'WAVES':>7} {'LineVul':>9}")
    print("-" * 40)
    for cwe in cwes_shown:
        ours_v    = our_per_class.get(cwe)
        waves_v   = WAVES_PER_CWE_TPR.get(cwe)
        linevul_v = LINEVUL_PER_CWE_TPR.get(cwe)
        o_str = f"{ours_v:.3f}" if ours_v    is not None else "  ---"
        w_str = f"{waves_v:.3f}" if waves_v  is not None else "  ---"
        l_str = f"{linevul_v:.3f}" if linevul_v is not None else "  ---"
        print(f"{cwe:<12} {o_str:>7} {w_str:>7} {l_str:>9}")

    print("\nWAVES / LineVul: binary recall per CWE (vulnerable vs benign).")
    print("Ours: multiclass recall (correct CWE class vs all others).")
    print("=> Ours is a harder task -- multiclass recall expected to be lower.")
    print("=" * 72 + "\n")
