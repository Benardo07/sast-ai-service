"""
prepare_dataset.py
~~~~~~~~~~~~~~~~~~
End-to-end CPG generation: read a vulnerability dataset → run Joern →
write GraphML CPG files into data/raw/<class>/.

Dataset support
---------------
devign     Binary (0/1). Flaw lines extracted from vul_lines column.
bigvul     Multi-class by default (CWE ID → class index). Binary with --binary.
           Flaw lines extracted by diffing func_before vs func_after.
diversevul Binary only. No localization ground truth available.
csv        Binary. Supply --code-col and --label-col.

Multi-class (BigVul)
--------------------
The top --top-cwe CWE categories are mapped to class indices 1..K.
Benign samples are always class 0.
Rare CWEs (outside top-K) are dropped.

A cwe_vocab.json file is saved to --out-dir so that val/test splits use the
same class mapping. Pass --cwe-vocab path/to/cwe_vocab.json when processing
val/test to reuse an existing vocabulary.

Localization metadata
---------------------
For datasets that have line-level ground truth (BigVul via diff, Devign via
vul_lines), a sidecar <cpg_name>.meta.json is saved next to each CPG file:

    func_012345.json       ← Joern CPG
    func_012345.meta.json  ← {"class_id": 2, "cwe": "CWE-20", "flaw_lines": [14, 17]}

CodeBERTGraphDataset reads these automatically during preprocessing and stores
the result as a flaw_line_mask node tensor in each PyG Data object.

Usage examples
--------------
    # BigVul multi-class (default), saves cwe_vocab.json to out-dir
    uv run python scripts/prepare_dataset.py \\
        --input data/datasets/bigvul/train.parquet \\
        --format bigvul \\
        --joern-cli C:/joern/joern-cli \\
        --out-dir data/raw \\
        --top-cwe 10 \\
        --sample-per-class 2000 \\
        --workers 4

    # BigVul val split — reuse vocab from train
    uv run python scripts/prepare_dataset.py \\
        --input data/datasets/bigvul/validation.parquet \\
        --format bigvul \\
        --joern-cli C:/joern/joern-cli \\
        --out-dir data/raw_val \\
        --cwe-vocab data/raw/cwe_vocab.json \\
        --workers 4

    # BigVul binary (collapse CWE → 0/1, still saves flaw lines)
    uv run python scripts/prepare_dataset.py \\
        --input data/datasets/bigvul/train.parquet \\
        --format bigvul --binary ...

    # Devign (binary + localization)
    uv run python scripts/prepare_dataset.py \\
        --input data/datasets/devign/train.parquet \\
        --format devign \\
        --joern-cli C:/joern/joern-cli \\
        --out-dir data/raw \\
        --sample-per-class 2000 \\
        --workers 4

Key flags
---------
--top-cwe N        Top-N CWE classes for BigVul multi-class (default 10).
--binary           Collapse all vulnerable labels to 1, regardless of CWE.
--cwe-vocab PATH   Load existing CWE vocab instead of building from data.
                   Use this for val/test splits to keep class indices stable.
--sample-per-class N   Sample N rows per class (balanced). Recommended.
--idx-offset N     Add N to every file index to avoid collisions when merging
                   multiple datasets into the same --out-dir.
--workers N        Parallel Joern workers (threads). Each uses one JVM.
--resume           Skip functions whose CPG file already exists.
"""

from __future__ import annotations

import argparse
import difflib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# package module: data/ -> gnn_vuln/ -> src/ -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

import pandas as pd
from loguru import logger

from gnn_vuln.data.joern_runner import process_function
from gnn_vuln.data.preprocess import preprocess


# ---------------------------------------------------------------------------
# Flaw line helpers
# ---------------------------------------------------------------------------

def _diff_flaw_lines(func_before: str, func_after: str) -> list[int]:
    """
    Return 1-indexed line numbers in func_before that were changed or removed
    when the vulnerability was patched (func_after).
    """
    if not func_before or not func_after or func_before == func_after:
        return []
    before = func_before.splitlines(keepends=True)
    after = func_after.splitlines(keepends=True)
    flaw_lines: list[int] = []
    for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(None, before, after).get_opcodes():
        if tag in ("replace", "delete"):
            flaw_lines.extend(range(i1 + 1, i2 + 1))
    return flaw_lines


def _extract_devign_flaw_lines(vul_lines) -> list[int]:
    """Extract sorted unique line numbers from Devign's vul_lines dict."""
    if not isinstance(vul_lines, dict):
        return []
    line_no = vul_lines.get("line_no", [])
    if hasattr(line_no, "tolist"):
        line_no = line_no.tolist()
    return sorted({int(ln) for ln in line_no if ln is not None})


# ---------------------------------------------------------------------------
# File reader
# ---------------------------------------------------------------------------

def _read_file(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    elif suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in (".json", ".jsonl"):
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported file format: {suffix}. Use .parquet, .csv, or .json")


# ---------------------------------------------------------------------------
# CWE vocabulary builder (BigVul only)
# ---------------------------------------------------------------------------

_CWE_JUNK = {"CWE-unknown", "CWE-Other", "CWE-other", "unknown", "other",
             "NVD-CWE-Other", "NVD-CWE-noinfo", ""}

def _build_cwe_vocab(df: pd.DataFrame, top_k: int) -> dict[str, int]:
    """
    Map top-k most common CWE IDs to class indices 1..K.
    top_k=0 means all CWEs (no limit).
    "benign" is always 0. Rows with missing/empty/unknown CWE are excluded.
    """
    vul_cwe = df[df["vul"] == 1]["CWE ID"].fillna("").astype(str).str.strip()
    vul_cwe = vul_cwe[~vul_cwe.isin(_CWE_JUNK)]
    counts = vul_cwe.value_counts()
    top_cwes = counts.index.tolist() if top_k == 0 else counts.head(top_k).index.tolist()
    vocab: dict[str, int] = {"benign": 0}
    for i, cwe in enumerate(top_cwes, start=1):
        vocab[cwe] = i
    return vocab


# ---------------------------------------------------------------------------
# Dataset loaders
#
# All loaders return a DataFrame with columns:
#   code        str        function source code
#   label       int        class index (0 = benign)
#   flaw_lines  list[int]  1-indexed vulnerable line numbers (empty if unavailable)
#   cwe         str        CWE ID string (empty for non-BigVul datasets)
# ---------------------------------------------------------------------------

def load_devign(path: Path) -> pd.DataFrame:
    """
    Binary classification (label 0/1).
    Flaw lines are read from the vul_lines column when available.
    """
    df = _read_file(path)
    rows = []
    for tup in df.itertuples(index=True):
        row_id = tup.Index
        code = getattr(tup, "func", None)
        target = int(getattr(tup, "target", 0))
        if code is None or (isinstance(code, float) and pd.isna(code)):
            continue
        vl = getattr(tup, "vul_lines", None)
        flaw_lines = _extract_devign_flaw_lines(vl) if target else []
        rows.append({"id": row_id, "code": str(code), "label": target, "flaw_lines": flaw_lines, "cwe": ""})
    return pd.DataFrame(rows)


def load_bigvul(
    path: Path,
    top_k_cwe: int = 10,
    binary: bool = False,
    cwe_vocab: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Multi-class by default: top_k_cwe CWE categories become class indices 1..K.
    Benign = 0. Samples with rare CWEs (outside top-K) are dropped.

    Pass binary=True to collapse all vulnerable labels to 1.
    Pass cwe_vocab to reuse an existing vocabulary (val/test splits).

    Flaw lines are computed by diffing func_before vs func_after.

    Returns (df, cwe_vocab).  cwe_vocab is {} when binary=True.
    """
    df = _read_file(path)

    code_col = next((c for c in ["func_before", "Function", "func"] if c in df.columns), None)
    after_col = "func_after" if "func_after" in df.columns else None
    cwe_col = "CWE ID" if "CWE ID" in df.columns else None
    label_col = next((c for c in ["vul", "Vulnerable", "target"] if c in df.columns), None)

    if not code_col or not label_col:
        raise ValueError(
            f"Cannot find code/label columns in BigVul. Got: {list(df.columns)}"
        )

    df = df.rename(columns={code_col: "code", label_col: "vul"})
    df = df.dropna(subset=["code"])
    df["vul"] = df["vul"].astype(int)

    if binary:
        vocab: dict[str, int] = {}
        df["label"] = df["vul"]
        df["cwe_str"] = ""
    else:
        if cwe_vocab is None:
            vocab = _build_cwe_vocab(df, top_k_cwe) if cwe_col else {}
        else:
            vocab = cwe_vocab

        if cwe_col:
            df["cwe_str"] = df[cwe_col].fillna("").astype(str)
        else:
            df["cwe_str"] = ""
            logger.warning("No CWE ID column found, falling back to binary labels.")
            df["label"] = df["vul"]

        if vocab and cwe_col:
            def _assign_label(row: pd.Series) -> int:
                if row["vul"] == 0:
                    return 0
                return vocab.get(row["cwe_str"], -1)

            df["label"] = df.apply(_assign_label, axis=1)
            before = len(df)
            df = df[df["label"] >= 0].copy()
            dropped = before - len(df)
            if dropped:
                logger.info(f"Dropped {dropped} rows with unknown/junk CWE IDs (NVD-CWE-Other, CWE-unknown, etc.)")

    # Compute flaw lines from func_before → func_after diff
    if after_col and after_col in df.columns:
        def _flaw(row: pd.Series) -> list[int]:
            if row["vul"] == 0:
                return []
            fa = row.get(after_col, "")
            if not fa or (isinstance(fa, float) and pd.isna(fa)):
                return []
            return _diff_flaw_lines(str(row["code"]), str(fa))

        logger.info("Computing flaw lines from func_before/func_after diff…")
        df["flaw_lines"] = df.apply(_flaw, axis=1)
    else:
        logger.warning("func_after column not found — flaw_lines will be empty.")
        df["flaw_lines"] = [[] for _ in range(len(df))]

    df = df.rename(columns={"cwe_str": "cwe"})
    df["id"] = df.index
    return df[["id", "code", "label", "flaw_lines", "cwe"]], vocab


def load_api(
    path: Path,
    top_k_cwe: int = 0,
    binary: bool = False,
    cwe_vocab: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """BigVul schema plus an optional per-row flaw_lines annotation, for corpora ingested
    through the API rather than mined from patches.

    Each vulnerable row localizes itself either by a hand-written flaw_lines list or by a
    func_after to diff. Resolution is per row, so one file may mix both. A row whose
    flaw_lines is absent or empty falls back to the diff.

    Kept apart from load_bigvul on purpose: research parquets (MegaVul, relearn) carry a
    flaw_lines column whose empty lists are meaningful, and must keep diffing.
    """
    df, vocab = load_bigvul(path, top_k_cwe=top_k_cwe, binary=binary, cwe_vocab=cwe_vocab)
    raw = _read_file(path)
    if "flaw_lines" not in raw.columns:
        return df, vocab

    def _annotated(row_id: int) -> list[int] | None:
        if not (0 <= row_id < len(raw)):
            return None
        fl = raw["flaw_lines"].iloc[row_id]
        if fl is None or (isinstance(fl, float) and pd.isna(fl)) or len(fl) == 0:
            return None
        return sorted({int(i) for i in fl})

    n = 0
    for pos, row in enumerate(df.itertuples(index=False)):
        given = _annotated(int(row.id)) if row.label != 0 else None
        if given is not None:
            df.iat[pos, df.columns.get_loc("flaw_lines")] = given
            n += 1
    logger.info(f"Flaw lines: {n} row(s) from the flaw_lines column, rest from the diff")
    return df, vocab


def load_diversevul(path: Path) -> pd.DataFrame:
    """
    Binary only (target column). No localization ground truth.
    flaw_lines will be empty for all samples.
    """
    logger.warning(
        "DiverseVul: no line-level ground truth available. "
        "flaw_line_mask will be all-zero. Multi-class CWE labels are not "
        "extracted (multi-label, complex). Use --binary behavior is implicit."
    )
    df = _read_file(path)
    df = df[["func", "target"]].rename(
        columns={"func": "code", "target": "label"}
    ).dropna(subset=["code"])
    df["label"] = df["label"].astype(int)
    df["flaw_lines"] = [[] for _ in range(len(df))]
    df["cwe"] = ""
    df["id"] = df.index
    return df


def load_csv(path: Path, code_col: str, label_col: str) -> pd.DataFrame:
    df = _read_file(path)
    if code_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Columns '{code_col}' / '{label_col}' not found. "
            f"Available: {list(df.columns)}"
        )
    df = df.rename(columns={code_col: "code", label_col: "label"}).dropna(subset=["code"])
    df["label"] = df["label"].astype(int)
    df["flaw_lines"] = [[] for _ in range(len(df))]
    df["cwe"] = ""
    df["id"] = df.index
    return df


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

_EXT_TO_LANG = {
    # C family
    "c": "C", "h": "C",
    "cpp": "C++", "cc": "C++", "cxx": "C++", "hpp": "C++",
    # JVM
    "java": "Java", "kt": "Kotlin", "scala": "Scala",
    # Scripting
    "js": "JavaScript", "jsx": "JavaScript", "ts": "TypeScript",
    "py": "Python", "rb": "Ruby", "php": "PHP",
    "lua": "Lua", "m": "Objective-C",
    # Systems
    "go": "Go", "rs": "Rust", "swift": "Swift",
    # .NET
    "cs": "C#",
}

# Full language name OR raw extension → Joern file extension.
# Covers all Joern-supported languages. Unsupported (Rust, Lua, Scala, ObjC)
# are omitted — they fall back to detect_language() → "c".
_LANG_TO_EXT: dict[str, str] = {
    # full names (from download parquet language column)
    "c": "c", "c++": "cpp", "java": "java", "kotlin": "kt",
    "javascript": "js", "typescript": "js", "python": "py", "ruby": "rb",
    "php": "php", "go": "go", "swift": "swift", "c#": "cs",
    # raw extensions (from parquet extension column)
    "cpp": "cpp", "cc": "cpp", "cxx": "cpp",
    "h": "c", "hpp": "cpp",
    "js": "js", "jsx": "js", "ts": "js",
    "py": "py", "rb": "rb",
    "kt": "kt", "kts": "kt",
    "go": "go", "swift": "swift", "cs": "cs", "php": "php",
}


def _run_one(
    idx: int,
    code: str,
    out_dir: Path,
    joern_cli_dir: Path | None,
    java_home: str | None,
    normalize: bool,
    resume: bool,
    flaw_lines: list[int] | None = None,
    class_id: int | None = None,
    cwe: str = "",
    is_multi_class: bool = False,
    row_id: int = -1,
    language: str = "",    # ground-truth language name (e.g. "C", "Java"); empty = auto-detect
) -> tuple[int, Path | None, str | None]:
    """
    Parse one function with Joern, save CPG JSON, and optionally write a
    .meta.json sidecar with class_id (multi-class) and flaw_lines.
    """
    if resume:
        cpg_existing = [
            f for f in out_dir.glob(f"func_{idx}.*")
            if ".meta." not in f.name
        ]
        if cpg_existing:
            return idx, cpg_existing[0], None

    raw_func = code  # capture original before any normalization

    from gnn_vuln.data.joern_runner import detect_language as _detect_lang

    # Resolve Joern extension
    lang_lower = language.lower() if language else ""
    is_unknown = lang_lower in ("", "unknown")

    if not is_unknown:
        joern_ext = _LANG_TO_EXT.get(lang_lower)
        if joern_ext is None:
            # Known language, not supported by Joern (Rust, Scala, Lua, ObjC…) → skip
            return idx, None, f"skipped: '{language}' not supported by Joern"
    else:
        # Empty/unknown → detect from code; detect_language always returns a supported ext
        joern_ext = _detect_lang(code)

    lang_name = (
        language                                         # parquet ground truth
        or _EXT_TO_LANG.get(joern_ext, joern_ext.upper())  # heuristic
        or "C"                                           # absolute fallback
    )

    if normalize:
        if joern_ext in ("c", "cpp"):
            code = preprocess(code, lang="c", normalize=True)

    try:
        dest = process_function(
            code=code,
            idx=idx,
            out_dir=out_dir,
            joern_cli_dir=joern_cli_dir,
            java_home=java_home,
            lang=joern_ext,
        )
        if dest is None:
            return idx, None, "process_function returned None (Joern may have failed silently)"

        # Build sidecar metadata dict
        meta: dict = {"id": row_id, "raw_func": raw_func, "language": lang_name}
        if is_multi_class and class_id is not None:
            meta["class_id"] = class_id
        if cwe:
            meta["cwe"] = cwe
        if flaw_lines:
            meta["flaw_lines"] = flaw_lines
        (out_dir / f"func_{idx}.meta.json").write_text(json.dumps(meta))

        return idx, dest, None
    except Exception:
        import traceback
        return idx, None, traceback.format_exc()


# ---------------------------------------------------------------------------
# MegaVul hybrid worker: copy pre-built graph OR run Joern fallback
# ---------------------------------------------------------------------------

def _run_megavul_one(
    idx: int,
    code: str,
    out_dir: Path,
    joern_cli_dir: Path | None,
    java_home: str | None,
    normalize: bool,
    resume: bool,
    flaw_lines: list[int] | None = None,
    class_id: int | None = None,
    cwe: str = "",
    is_multi_class: bool = False,
    row_id: int = -1,
    language: str = "",
    graph_path: str = "",     # pre-built graph path from MegaVul (may be empty)
) -> tuple[int, Path | None, str | None]:
    """
    MegaVul hybrid: if graph_path exists, copy it; otherwise fall back to Joern.
    Pre-built graphs are MegaVul's JSON format — handled by _parse_megavul_json().
    """
    import shutil as _shutil

    if resume:
        existing = [f for f in out_dir.glob(f"func_{idx}.*") if ".meta." not in f.name]
        if existing:
            return idx, existing[0], None

    # Try pre-built graph first
    if graph_path and Path(graph_path).exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"func_{idx}.json"
        _shutil.copy2(graph_path, dest)
        raw_func = code
        lang_name = language or "C"
        meta: dict = {"id": row_id, "raw_func": raw_func, "language": lang_name}
        if is_multi_class and class_id is not None:
            meta["class_id"] = class_id
        if cwe:
            meta["cwe"] = cwe
        if flaw_lines:
            meta["flaw_lines"] = flaw_lines
        (out_dir / f"func_{idx}.meta.json").write_text(json.dumps(meta))
        return idx, dest, None

    # No pre-built graph — fall back to Joern
    return _run_one(
        idx=idx, code=code, out_dir=out_dir, joern_cli_dir=joern_cli_dir,
        java_home=java_home, normalize=normalize, resume=resume,
        flaw_lines=flaw_lines, class_id=class_id, cwe=cwe,
        is_multi_class=is_multi_class, row_id=row_id, language=language,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Joern CPG files from a vulnerability dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--format", default="devign",
        choices=["devign", "bigvul", "api", "megavul", "diversevul", "csv", "merged", "titanvul", "relearn", "megavul_cil"],
    )
    parser.add_argument("--code-col", default="func")
    parser.add_argument("--label-col", default="target")
    parser.add_argument("--joern-cli", type=Path, default=None)
    parser.add_argument("--java-home", type=str, default=None)
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "data" / "raw",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-per-class", type=int, default=None)
    parser.add_argument("--idx-offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--resume", action="store_true")
    # Multi-class options
    parser.add_argument(
        "--top-cwe", type=int, default=0,
        help="Number of top CWE categories to use as classes (0 = all CWEs, default 0).",
    )
    parser.add_argument(
        "--binary", action="store_true",
        help="Collapse all vulnerable labels to 1 regardless of CWE (binary mode).",
    )
    parser.add_argument(
        "--cwe-vocab", type=Path, default=None,
        help="Path to an existing cwe_vocab.json to reuse (for val/test splits). "
             "If omitted, vocab is auto-saved to <out-dir>/cwe_vocab.json.",
    )
    parser.add_argument(
        "--split", default="train", choices=["train", "val", "test"],
        help="Dataset split — appends _val/_test to output subdir (e.g. bigvul_val/).",
    )
    args = parser.parse_args()

    # Auto-subdir: data/raw/bigvul/, data/raw/bigvul_val/, data/raw/bigvul_test/
    split_suffix = f"_{args.split}" if args.split != "train" else ""
    args.out_dir = args.out_dir / f"{args.format}{split_suffix}"

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    logger.info(f"Loading dataset: {args.input}")
    cwe_vocab: dict[str, int] = {}

    if args.format == "devign":
        df = load_devign(args.input)
        is_multi_class = False

    elif args.format in ("bigvul", "api"):
        # api = bigvul schema + optional per-row flaw_lines annotation (API ingest)
        # Determine if multi-class
        is_multi_class = not args.binary

        # Load or build CWE vocab
        vocab_path = args.cwe_vocab or (args.out_dir / "cwe_vocab.json")
        existing_vocab: dict[str, int] | None = None
        if vocab_path.exists():
            with open(vocab_path) as f:
                existing_vocab = json.load(f)
            logger.info(
                f"Loaded existing CWE vocab ({len(existing_vocab)} classes) from {vocab_path}"
            )

        loader = load_api if args.format == "api" else load_bigvul
        df, cwe_vocab = loader(
            args.input,
            top_k_cwe=args.top_cwe,
            binary=args.binary,
            cwe_vocab=existing_vocab,
        )

        # Always save vocab to output dir so val/test splits can find it
        out_vocab = args.out_dir / "cwe_vocab.json"
        if is_multi_class and cwe_vocab and not out_vocab.exists():
            args.out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_vocab, "w") as f:
                json.dump(cwe_vocab, f, indent=2)
            logger.info(f"CWE vocabulary ({len(cwe_vocab)} classes) saved → {out_vocab}")

    elif args.format == "diversevul":
        df = load_diversevul(args.input)
        is_multi_class = False

    elif args.format in ("megavul", "megavul_cil"):
        # MegaVul: has pre-built graphs in func_graph_path column
        # megavul_cil = class-incremental task-B subset (non-top25 CWE), same prebuilt-graph path
        # Uses load_bigvul for CWE vocab + label assignment, then joins graph paths
        is_multi_class = not args.binary
        vocab_path = args.cwe_vocab or (args.out_dir / "cwe_vocab.json")
        existing_vocab = None
        if vocab_path.exists():
            with open(vocab_path) as f:
                existing_vocab = json.load(f)
            logger.info(f"Loaded existing CWE vocab ({len(existing_vocab)} classes) from {vocab_path}")
        df, cwe_vocab = load_bigvul(
            args.input, top_k_cwe=args.top_cwe, binary=args.binary, cwe_vocab=existing_vocab,
        )
        out_vocab = args.out_dir / "cwe_vocab.json"
        if is_multi_class and cwe_vocab and not out_vocab.exists():
            args.out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_vocab, "w") as f:
                json.dump(cwe_vocab, f, indent=2)
            logger.info(f"CWE vocabulary ({len(cwe_vocab)} classes) saved → {out_vocab}")
        # Join pre-built graph paths from original parquet by original row id
        raw_df = pd.read_parquet(args.input, columns=["func_graph_path"])
        df["func_graph_path"] = df["id"].map(lambda i: raw_df["func_graph_path"].iloc[i] if 0 <= i < len(raw_df) else "")
        df["func_graph_path"] = df["func_graph_path"].fillna("")

    elif args.format in ("merged", "titanvul", "relearn"):
        # merged/titanvul/relearn: same schema as BigVul (func_before, func_after, vul, CWE ID)
        is_multi_class = not args.binary
        vocab_path = args.cwe_vocab or (args.out_dir / "cwe_vocab.json")
        existing_vocab = None
        if vocab_path.exists():
            with open(vocab_path) as f:
                existing_vocab = json.load(f)
            logger.info(f"Loaded existing CWE vocab ({len(existing_vocab)} classes) from {vocab_path}")
        df, cwe_vocab = load_bigvul(
            args.input, top_k_cwe=args.top_cwe, binary=args.binary, cwe_vocab=existing_vocab,
        )
        out_vocab = args.out_dir / "cwe_vocab.json"
        if is_multi_class and cwe_vocab and not out_vocab.exists():
            args.out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_vocab, "w") as f:
                json.dump(cwe_vocab, f, indent=2)
            logger.info(f"CWE vocabulary ({len(cwe_vocab)} classes) saved → {out_vocab}")

    else:
        df = load_csv(args.input, args.code_col, args.label_col)
        is_multi_class = False

    df["label"] = df["label"].astype(int)

    if args.limit:
        df = df.head(args.limit)

    # Balanced per-class sampling (works for binary and multi-class)
    if args.sample_per_class:
        n = args.sample_per_class
        df = pd.concat(
            [g.sample(min(len(g), n), random_state=42) for _, g in df.groupby("label")],
            ignore_index=True,
        )

    logger.info(
        f"Functions to process: {len(df)}\n"
        f"{df['label'].value_counts().rename('count').to_frame().to_string()}"
    )

    # -----------------------------------------------------------------------
    # Output directories (always benign/ + vulnerable/)
    # Multi-class class_id is recorded in .meta.json, not in dir names.
    # -----------------------------------------------------------------------
    benign_dir = args.out_dir / "benign"
    vuln_dir = args.out_dir / "vulnerable"
    benign_dir.mkdir(parents=True, exist_ok=True)
    vuln_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Build work list
    # -----------------------------------------------------------------------
    # Resolve language column — iterate priority tuple (not df.columns order)
    # so "language" (full names, already mapped by download) beats "extension"
    _LANG_COL = next((c for c in ("lang", "language", "extension") if c in df.columns), None)
    _EXT_NORM = {"C": "C", "CPP": "C++", "C++": "C++"}  # BigVul lang column values

    is_megavul = args.format in ("megavul", "megavul_cil")
    work: list[tuple] = []
    for local_idx, row in enumerate(df.itertuples(index=False)):
        class_id = int(row.label)
        phys_dir = benign_dir if class_id == 0 else vuln_dir
        flaw_lines = list(row.flaw_lines) if hasattr(row, "flaw_lines") else []
        cwe = str(row.cwe) if hasattr(row, "cwe") else ""
        row_id = int(row.id) if hasattr(row, "id") else -1
        lang_raw = str(getattr(row, _LANG_COL, "") or "") if _LANG_COL else ""
        lang_gt  = _EXT_NORM.get(lang_raw.upper(), lang_raw) if lang_raw else ""
        # For megavul: include pre-built graph path (empty = must run Joern)
        graph_path = str(getattr(row, "func_graph_path", "") or "") if is_megavul else ""
        work.append((local_idx + args.idx_offset, str(row.code), phys_dir, class_id,
                     flaw_lines, cwe, row_id, lang_gt, graph_path))

    logger.info(f"Workers: {args.workers}  |  resume: {args.resume}  |  jobs: {len(work)}")
    if args.workers > 1:
        est_secs = len(work) * 6 / args.workers
        logger.info(
            f"Estimated time: ~{est_secs / 60:.0f} min  "
            f"({len(work)} jobs × ~6s / {args.workers} workers)"
        )

    # -----------------------------------------------------------------------
    # Run Joern
    # -----------------------------------------------------------------------
    success = 0
    skipped = 0
    start = time.monotonic()

    def submit(idx: int, code: str, phys_dir: Path, class_id: int, flaw_lines: list[int],
               cwe: str, row_id: int = -1, language: str = "", graph_path: str = ""):
        if is_megavul:
            return _run_megavul_one(
                idx=idx, code=code, out_dir=phys_dir,
                joern_cli_dir=args.joern_cli, java_home=args.java_home,
                normalize=args.normalize, resume=args.resume,
                flaw_lines=flaw_lines, class_id=class_id, cwe=cwe,
                is_multi_class=is_multi_class, row_id=row_id,
                language=language, graph_path=graph_path,
            )
        return _run_one(
            idx=idx, code=code, out_dir=phys_dir,
            joern_cli_dir=args.joern_cli, java_home=args.java_home,
            normalize=args.normalize, resume=args.resume,
            flaw_lines=flaw_lines, class_id=class_id, cwe=cwe,
            is_multi_class=is_multi_class, row_id=row_id, language=language,
        )

    if args.workers <= 1:
        for i, item in enumerate(work):
            _, dest, err = submit(*item)
            if dest:
                success += 1
                if success % 50 == 0:
                    elapsed = time.monotonic() - start
                    rate = success / elapsed
                    remaining = len(work) - (i + 1)
                    logger.info(
                        f"  [{i+1}/{len(work)}] {success} done  "
                        f"({rate:.1f}/s, ~{remaining / rate / 60:.0f} min left)"
                    )
            else:
                skipped += 1
                logger.warning(f"  [{item[0]}] FAILED: {(err or 'unknown')[:300]}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(submit, *item): item[0]
                for item in work
            }
            completed = 0
            for future in as_completed(futures):
                completed += 1
                _, dest, err = future.result()
                if dest:
                    success += 1
                else:
                    skipped += 1
                    if err:
                        orig_idx = futures[future]
                        logger.warning(f"  [{orig_idx}] {err[:200]}")
                if completed % 50 == 0:
                    elapsed = time.monotonic() - start
                    rate = completed / elapsed
                    logger.info(
                        f"  [{completed}/{len(work)}] {success} done, {skipped} failed  "
                        f"({rate:.1f}/s, ~{(len(work) - completed) / rate / 60:.0f} min left)"
                    )

    elapsed = time.monotonic() - start
    logger.info(
        f"\nDone in {elapsed / 60:.1f} min. "
        f"{success} CPG files written, {skipped} skipped/failed.\n"
        f"Next step: uv run train --config configs/lmgcn/binary.yaml"
    )


if __name__ == "__main__":
    main()
