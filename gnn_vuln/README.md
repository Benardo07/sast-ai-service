# `gnn_vuln` — Library API Reference

The installable model library behind the vulnerability-detection service. This is the
complete public surface: what to import, the inputs, and the outputs.

**Not everything is file-based.** You pass a function **source string** and get a **result
dict** back. The only files involved are the model checkpoint + config (normal — weights and
config live on disk) and the Joern CPG, which is created in a private temp dir and hidden
from you. In-memory in, in-memory out.

---

## Install

```bash
# 1. torch + PyG sparse ext from their own indexes (PyPI can't resolve these alone)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu     # or cu124
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
# 2. the library
pip install gnn-vuln
```

Plus **Joern** (CPG generation) + a **JDK 21** on the host. Point the predictor at the
`joern-cli` directory.

---

## Inference — `gnn_vuln.inference`

### `VulnPredictor` (high-level, recommended)

```python
from gnn_vuln.inference import VulnPredictor

predictor = VulnPredictor.from_checkpoint(
    checkpoint="checkpoints/<run>/best_model.pt",   # trained weights (.pt file)
    config="configs/<arch>/config.yaml",            # its config (file, or pass a list)
    device="cuda",                                  # "cpu" | "cuda"
)
predictor.class_names = ["benign", "CWE-787", ...]  # optional: override label names
```

| Method | Input | Output |
| --- | --- | --- |
| `predict_code(code, joern_cli, max_nodes=2500, top_k_lines=None)` | function **source string** | result `dict`, or `None` if Joern produced no CPG |
| `predict_codes(codes, joern_cli, max_nodes=2500, top_k_lines=None)` | `list[str]` | list of result dicts (`None` per entry on Joern failure) |
| `predict(data, top_k_lines=None)` | a PyG `Data` object (already built) | result `dict` |
| `predict_from_file(cpg_path, max_nodes=1000, top_k_lines=None)` | path to a Joern CPG file | result `dict`, or `None` |

```python
# the everyday call — string in, dict out (Joern handled internally)
result = predictor.predict_code(
    "void f(char *s){ char b[8]; strcpy(b, s); }",
    joern_cli="C:/joern/joern-cli",
    top_k_lines=5,
)
```

### Result dict (schema)

```python
{
  "prediction":          "CWE-120",          # predicted class name
  "class_id":            7,                   # predicted class index
  "is_vulnerable":       True,                # class_id > 0
  "confidence":          0.87,                # softmax prob of the predicted class [0,1]
  "class_probabilities": {"benign": 0.01, "CWE-120": 0.87, ...},
  "suspicious_lines":    [{"line": 3, "score": 0.92, "code": "strcpy(b, s);"}, ...],  # score-desc
  "cls_embedding":       [0.013, -0.44, ...], # pre-head function vector (for search/drift)
}
```

`suspicious_lines` may also carry `predicted_cwe` + per-line `class_probabilities` for the
multiclass statement head. `cls_embedding` is the representation fed to the output head.

### Module functions (lower-level)

```python
from gnn_vuln.inference import load_model, predict, predict_from_file

model, class_names = load_model(checkpoint, config, device="cpu")   # -> (nn.Module, list[str])
result = predict(model, data, class_names, device=None, top_k_lines=None)   # PyG Data -> dict
result = predict_from_file(model, cpg_path, class_names, pretrained_lm=..., ...)  # file -> dict
```

---

## CPG generation — `gnn_vuln.data.joern_runner`

Only needed if you want the CPG file yourself; `predict_code` calls this for you.

```python
from gnn_vuln.data.joern_runner import process_function
from pathlib import Path

cpg_path = process_function(
    code="int add(int a,int b){return a+b;}",  # source string
    idx=0,
    out_dir=Path("./out"),
    joern_cli_dir=Path("C:/joern/joern-cli"),
    fmt="graphml",         # "graphml" | "json"
    lang=None,             # None = auto-detect (c/cpp/java/js/py)
)   # -> Path to the written CPG, or None on failure
```

---

## Config — `gnn_vuln.config`

```python
from gnn_vuln.config import Config

cfg = Config.from_yaml("N48.yaml")                              # one monolithic file
cfg = Config.from_yamls(["data.yaml", "model.yaml", "train.yaml"])  # split, merged in order
# cfg.data, cfg.model, cfg.train, cfg.ewc, cfg.replay  — dataclasses
cfg.data.mode          # "binary" | "multiclass"
cfg.model.architecture # "lmgat_codebert" | "lmgat_seqgnn"
cfg.train.epochs       # 100
```

`from_yamls` lets you split data / model / train configs into separate files; a single file
is just the one-element case (identical behaviour).

### Train/val/test split

The split (`dataset.get_splits`, used by both train + evaluate) is seeded + deterministic.
Control it via config:

```python
cfg.data.train_ratio   # 0.8  — seeded split; test ratio = 1 - train - val
cfg.data.val_ratio     # 0.1  — e.g. 0.9 / 0.1 → 90/10/0 (no test holdout, prod)
cfg.train.seed         # 42   — shuffle seed (reproducible across runs/Python versions)
cfg.data.split_file    # ""   — path to {"train":[id],"val":[],"test":[]} keyed on parquet_id;
                       #        OVERRIDES the ratios (bring-your-own / match-a-baseline split)
```

`python -m gnn_vuln.train` writes `<results_dir>/<run>/split.json` — the realized train/val/test
parquet_ids — next to `training_summary.json`, so the exact split is always recoverable.

A **0-ratio test split** (e.g. `0.9 / 0.1` → no test) is supported: training + validation run
as usual and the end-of-training **test evaluation is skipped** (no crash, no test metrics).
Use it for a production model that should train on all labelled data without a holdout.

---

## Data pipeline & training — module CLIs (`python -m`)

Each step is a runnable module. All accept **one** config file or **several** split files
(merged section-by-section).

| Command                                                                                                 | In                 | Out                                       |
| ------------------------------------------------------------------------------------------------------- | ------------------ | ----------------------------------------- |
| `python -m gnn_vuln.data.prepare --input <parquet> --format bigvul --out-dir <dir> --joern-cli <joern>` | raw rows (parquet) | per-function CPGs + `cwe_vocab.json`      |
| `python -m gnn_vuln.data.build_pt --config <yaml…> --split train`                                       | CPG dir            | processed `.pt` (UniXcoder node features) |
| `python -m gnn_vuln.data.merge --config <yaml…> --sources <s1> <s2> … --out-source <name> [--dedup]`     | built `.pt`s       | one merged `.pt` (label space unified)    |
| `python -m gnn_vuln.train --config <yaml…>`                                                             | `.pt` + config     | checkpoint + training_summary + split.json |

`prepare` flags: `--binary`, `--top-cwe N`, `--sample-per-class N`, `--workers N`.
Installed console scripts: `train`, `evaluate` (= `python -m gnn_vuln.train` / `.evaluate`).

The whole raw→pt→train flow:

```bash
python -m gnn_vuln.data.prepare  --input data.parquet --format bigvul --out-dir data/raw --joern-cli <joern>
python -m gnn_vuln.data.build_pt --config config.yaml --split train
python -m gnn_vuln.train         --config config.yaml
```

---

## Evaluation outputs & `GNN_VULN_API_MODE`

`Evaluator` separates **compute** from **persistence** so a caller can decide what hits disk:

- `Evaluator.compute() -> EvalResult` — runs inference + metrics, returns everything in memory,
  writes **nothing**.
- `Evaluator.save_artifacts(res)` — research persistence: `predictions.csv`,
  `localization_scores.csv`, `metrics_summary.json`, ROC / confusion / PR plots.
- `Evaluator.save_summary(res)` — writes **only** `metrics_summary.json` (the small handoff).
- `Evaluator.run()` = `compute()` + `save_artifacts()` (the research/CLI default).

`python -m gnn_vuln.evaluate --checkpoint <pt>` runs the full research path. Pass
`--metrics-only` (or set `GNN_VULN_API_MODE=1`) to write just `metrics_summary.json` — for a
service that reads the metrics back and persists them elsewhere, with no bulky per-sample CSVs
or plots on disk.

`GNN_VULN_API_MODE=1` also tells **the trainer** to skip research-only outputs
(`training_log.csv`, `training_curves.png`); the small handoffs `split.json` +
`training_summary.json` are still written. Set it when embedding the library in a service; leave
it unset for research runs that want the full artifacts for analysis.

---

## Package layout

```
gnn_vuln/
  inference.py            VulnPredictor, load_model, predict, predict_from_file
  config.py               Config (data/model/train/ewc/replay), from_yaml / from_yamls
  train.py                trainer  (python -m gnn_vuln.train)
  evaluate.py             evaluation (python -m gnn_vuln.evaluate)
  models/                 lmgat_codebert, lmgat_seqgnn — the architectures (built via config)
  data/
    prepare.py            raw rows → Joern CPG            (python -m)
    build_pt.py           CPG → .pt                       (python -m)
    joern_runner.py       process_function — Joern wrapper
    dataset_lm.py         CodeBERTGraphDataset (PyG InMemoryDataset, UniXcoder features)
    node_embedder.py      frozen LM per-node embeddings
```

The library resolves its data/checkpoint root from `$GNN_VULN_ROOT` (else the current working
directory), so it behaves the same installed-from-PyPI as in a source checkout.
