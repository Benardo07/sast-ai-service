"""
process_dataset.py
~~~~~~~~~~~~~~~~~~
Generate processed .pt dataset cache without starting training.
Useful for pre-computing embeddings locally before transferring to cloud.

Usage:
    uv run python scripts/process_dataset.py --config configs/lmgat/top1_cwe.yaml
    uv run python scripts/process_dataset.py --config configs/lmgat/top1_cwe.yaml --device cuda
    uv run python scripts/process_dataset.py --config configs/lmgat/top1_cwe.yaml --split val
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from gnn_vuln.config import Config, load_default_config
from gnn_vuln.data.dataset_lm import CodeBERTGraphDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute dataset .pt cache from config.")
    parser.add_argument("--config", required=True, nargs="+",
                        help="One YAML config (monolithic) or several split files merged in order.")
    parser.add_argument("--device", default=None, help="Device for embedding (cpu/cuda). Defaults to config value.")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Which source split to build (default: train)",
    )
    parser.add_argument(
        "--max-length", type=int, default=None,
        help="Tokenizer max sequence length for func tokens. Defaults to config func_max_length (512 if unset).",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete existing .pt and rebuild from scratch (skip patch fast-path)",
    )
    args = parser.parse_args()

    cfg_paths = args.config if isinstance(args.config, (list, tuple)) else [args.config]
    cfg = (Config.from_yamls(cfg_paths)
           if all(Path(p).exists() for p in cfg_paths) else load_default_config())

    pretrained_lm   = getattr(cfg.model, "pretrained_lm", "microsoft/unixcoder-base")
    func_lm         = getattr(cfg.model, "func_lm", "microsoft/unixcoder-base")
    add_func_tokens = getattr(cfg.model, "add_func_tokens", False)
    func_lm_source  = getattr(cfg.model, "func_lm_source", "raw")
    func_max_length = args.max_length if args.max_length is not None else getattr(cfg.model, "func_max_length", 512)
    device          = args.device or str(cfg.train.device)
    top_cwe        = getattr(cfg.data, "top_cwe", 0)
    cwe_list       = getattr(cfg.data, "cwe_list", None)
    cwe_groups     = getattr(cfg.data, "cwe_groups", None)
    filter_owasp = getattr(cfg.data, "filter_owasp", False)
    filter_top25_dangerous   = getattr(cfg.data, "filter_top25_dangerous", False)
    max_per_class  = getattr(cfg.data, "max_per_class", 0)
    resample_seed  = getattr(cfg.data, "resample_seed", 42)
    storage        = getattr(cfg.data, "storage", "inmemory")

    source_map = {
        "train": getattr(cfg.data, "source",      "bigvul"),
        "val":   getattr(cfg.data, "source_val",  ""),
        "test":  getattr(cfg.data, "source_test", ""),
    }
    source = source_map[args.split]
    if not source:
        print(f"No source configured for split='{args.split}' in {args.config}")
        raise SystemExit(1)

    print(f"Config        : {args.config}")
    print(f"Split         : {args.split}  source={source}")
    print(f"Mode          : {cfg.data.mode}")
    print(f"Pretrained LM : {pretrained_lm}")
    print(f"Function LM   : {func_lm}")
    print(f"func_tokens   : {add_func_tokens}")
    if add_func_tokens:
        print(f"func_max_len  : {func_max_length}")
    print(f"Device        : {device}")
    print(f"Max nodes     : {cfg.data.max_nodes}")
    print(f"Storage       : {storage}")
    print(f"top_cwe       : {top_cwe if top_cwe > 0 else 'all'}")
    print(f"cwe_list      : {cwe_list}")
    print(f"cwe_groups    : {cwe_groups}")
    print(f"filter_owasp  : {filter_owasp}")
    print(f"filter_top25_dangerous  : {filter_top25_dangerous}")
    print(f"max_per_class : {max_per_class if max_per_class > 0 else 'unlimited'}")
    if max_per_class > 0:
        print(f"resample_seed : {resample_seed}")
    print()

    dataset = CodeBERTGraphDataset(
        root=str(cfg.data.processed_dir.parent),
        source=source,
        max_nodes=cfg.data.max_nodes,
        embedder_device=device,
        mode=cfg.data.mode,
        pretrained_lm=pretrained_lm,
        func_lm=func_lm,
        add_func_tokens=add_func_tokens,
        top_cwe=top_cwe,
        cwe_list=cwe_list,
        cwe_groups=cwe_groups,
        filter_owasp=filter_owasp,
        filter_top25_dangerous=filter_top25_dangerous,
        max_per_class=max_per_class,
        resample_seed=resample_seed,
        func_lm_source=func_lm_source,
        func_max_length=func_max_length,
        force_rebuild=args.force_rebuild,
        storage=storage,
        use_flash_attention=getattr(cfg.train, "use_flash_attention", False),
        embedder_use_amp=getattr(cfg.train, "use_amp", True),
    )

    n_graphs    = len(dataset)
    num_classes = dataset.num_classes
    class_names = dataset.class_names
    pt_file     = dataset.processed_paths[0]

    # Explicit cleanup before interpreter shutdown to avoid OOM during GC
    del dataset
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    print(f"\nDone: {n_graphs} graphs cached.")
    print(f"num_classes   : {num_classes}")
    if class_names:
        print(f"class_names   : {class_names}")
    print(f"File          : {pt_file}")

    # Skip Python GC/finalizers entirely — .pt already saved, nothing left to do.
    # Prevents OOM during interpreter shutdown when GPU tensors are freed by GC.
    import os as _os
    _os._exit(0)


if __name__ == "__main__":
    main()
