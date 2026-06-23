"""
Central configuration for the gnn_vuln project.

Edit configs under configs/<model>/ to override defaults without changing code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Resolve the project root robustly for BOTH layouts:
    - editable/src checkout: <root>/src/gnn_vuln/config.py → parents[2] has configs/.
    - installed wheel (site-packages): __file__-relative is bogus, so honor
      $GNN_VULN_ROOT, else fall back to the current working directory (the API/CLI
      sets cwd to its app root). Prevents data/checkpoints resolving under site-packages."""
    env = os.environ.get("GNN_VULN_ROOT")
    if env:
        return Path(env).resolve()
    src_guess = Path(__file__).resolve().parents[2]
    if (src_guess / "configs").is_dir() or (src_guess / "src").is_dir():
        return src_guess
    return Path.cwd()


PROJECT_ROOT = _project_root()
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    raw_dir: Path = DATA_DIR / "raw"
    processed_dir: Path = DATA_DIR / "processed"
    splits_dir: Path = DATA_DIR / "splits"
    # Graph building
    max_nodes: int = 500        # drop graphs larger than this
    edge_types: list[str] = field(
        default_factory=lambda: ["AST", "CFG", "PDG", "CDG", "DDG"]
    )
    # Dataset mode: "binary" (benign/vuln) or "multiclass" (per-CWE).
    # Controls which processed cache file is used so both can coexist.
    mode: str = "binary"
    # Source subdirectory under data/raw/ — isolates datasets and appears in
    # the processed .pt filename so bigvul and merged never collide.
    source: str = "bigvul"
    # Optional separate val/test source dirs (e.g. bigvul_val, bigvul_test).
    # When both are set, official splits are used instead of internal 70/15/15.
    # Leave empty for datasets without separate val/test parquets.
    source_val: str = ""
    source_test: str = ""
    # Filter vocab to top-K CWE classes at .pt build time (0 = use all in vocab).
    # Raw data can be generated with --top-cwe 999; this narrows it at processed stage.
    top_cwe: int = 0
    # Explicit CWE whitelist, e.g. ["CWE-119", "CWE-787"]. Unioned with cwe_groups.
    cwe_list: list | None = None
    # Group whitelist, e.g. ["memory_safety", "injection"]. Expanded via CWE_GROUP_MAP.
    cwe_groups: list | None = None
    # Automatically filter to OWASP Top 10 (2025) CWEs (unioned with cwe_list/cwe_groups)
    filter_owasp: bool = False
    # Automatically filter to MITRE Top 25 CWEs (unioned with cwe_list/cwe_groups)
    filter_top25_dangerous: bool = False
    # Max graphs per class/CWE/group bucket during .pt build (0 = no limit).
    max_per_class: int = 0
    # Random seed for max_per_class sampling. Change for a different sample.
    resample_seed: int = 42
    # Explicit split file (json {"train":[parquet_id...],"val":[...],"test":[...]}). When set,
    # overrides seeded get_splits — used to match a baseline's exact split (e.g. LIVABLE Big-Vul
    # survivors). parquet_id == the "id" column of the source parquet. Empty = default 80/10/10.
    split_file: str = ""
    # train/val ratios fed to get_splits (test = 1 - train - val). split_file (explicit) overrides them.
    train_ratio: float = 0.8
    val_ratio: float = 0.1


@dataclass
class ModelConfig:
    architecture: str = "lmgat_codebert"  # lmgat_codebert | lmgat_codebert_mtl | ...
    # GNN encoder backbone (lmgat_codebert). Replaces the old standalone
    # lmgcn / lmgin / lmrgcn / lmrgcn_codebert / lmggnn architectures.
    #   gat  — GATv2Conv, edge-feature attention (uses heads, edge_dim)
    #   gcn  — GCNConv, edge-agnostic
    #   gin  — GINEConv, per-layer edge projection (uses edge_dim)
    #   rgcn — RGCNConv, one weight per edge type (uses num_relations, num_bases)
    #   ggnn — GatedGraphConv, edge-agnostic
    gnn_model: str = "gat"
    pretrained_lm: str = "microsoft/codebert-base"  # HuggingFace model ID for node embeddings (frozen)
    func_lm: str = ""               # live LM for function branch; if empty falls back to pretrained_lm
    add_func_tokens: bool = False   # tokenize full function text → stored in Data for live LM
    func_lm_source: str = "raw"    # source for func_lm text: "raw" | "normalized" (cpg reconstruct kept for compat only)
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.3
    # Graph-level pooling for the function classification representation.
    #   mean      — global mean pool over nodes (default)
    #   meanmax   — 0.8*max + 0.6*mean (parameter-free, peak + context)
    #   attention — gated attention pool: per-node score → softmax → weighted sum
    #   dualflow  — focal (per-node suspicion-weighted pool) + context (mean);
    #               single-encoder suspicion head, no two-stage GNN
    # Support: lmgat_codebert
    graph_pool: str = "mean"
    heads: int = 4              # number of GAT attention heads (gnn_model=gat)
    edge_dim: int = 7          # edge feature dimension (gnn_model=gat injects into GATv2 attention; gin projects it)
    num_relations: int = 7     # RGCN: number of edge-type relations (gnn_model=rgcn)
    num_bases: int | None = None  # RGCN: basis-decomposition count, None = no decomposition (gnn_model=rgcn)
    num_classes: int = 2        # 2 = binary; set higher for multi-class
    # Statement-level MIL head
    mil_weight: float = 0.5     # λ: weight of stmt MIL loss vs function loss
    mil_k: int = 3              # top-k statements used for pseudo-label assignment
    rank_loss_weight: float = 0.0  # pairwise ranking loss weight (0 = disabled)
    # Sliding-window encoding for long functions (live LM branch only).
    # func_chunk_size: tokens per window; should match the model's trained max length
    #   (512 for UniXcoder/CodeBERT, 512 for codet5p-110m-embedding, 1024 for codet5p-220m).
    # func_chunk_stride: step between windows (< chunk_size → overlapping windows).
    #   0 = disabled (single forward pass, truncates at func_max_length as before).
    #   Recommended: chunk_size // 2 for 50% overlap, chunk_size for non-overlapping.
    func_chunk_size: int = 0    # 0 = disabled
    func_chunk_stride: int = 0  # 0 = defaults to chunk_size // 2 when chunking is enabled
    # Max token length stored per function in the .pt cache.
    # When func_chunk_size > 0, set this to func_chunk_size * N_chunks you want to cover.
    # E.g. func_chunk_size=512, func_max_length=2048 → up to 4 windows per function.
    func_max_length: int = 512  # default matches model trained length
    # Live LM branch mode (lmgat_codebert).
    #   none          — no live LM. GNN-only, fused = h_graph (no LM concat).
    #                   localization_encoder MUST be "gnn" (lm/both need LM).
    #                   Replaces the old standalone "lmgat" architecture.
    #   func          — function-level forward (single [CLS] over full function,
    #                   sliding window via func_chunk_size if set). Default.
    #   func_and_line — func-level [CLS] for classification + per-line LM forward
    #                   for localization. EDAT-style line isolation: each source
    #                   line forwarded through LM independently → per-line [CLS]
    #                   used as synthetic hidden for localization. Pair with
    #                   mmoe_loc_transformer to recover cross-line context.
    #   line          — fully hierarchical, NO whole-function forward. Each line
    #                   forwarded through the LM independently → per-line [CLS];
    #                   a line-level transformer adds cross-line context.
    #                   Classification = meanmax pool of the transformer output;
    #                   localization = its per-line output. Function length is
    #                   unbounded (no func_max_length truncation). BERT-family
    #                   func_lm only (needs last_hidden_state per line).
    live_lm: str = "func"
    # ── Bidirectional cross-task (Phase 2, lmgat_codebert) ────────────────────
    # Makes localization (stmt_head) and classification (func_head) inform each
    # other.
    #   none            — independent heads (Phase 1 baseline)
    #   cross_attention — Q from one task, K/V from the other task's encoder
    #                     units (decoder-style cross-attention)
    #   self_attention  — EDAT-style: self-attention over a task's own encoder
    #                     units, query biased by the other task's signal
    #   mmoe            — Multi-gate Mixture-of-Experts (Ma et al. 2018): shared
    #                     expert pool + per-task gates (EDAT's released code)
    cross_task_method: str = "none"
    # MMOE only: replace the single Linear task projections with a per-task MLP
    # encoder (Linear→LN→ReLU→Dropout→Linear) — EDAT's TaskSpecificEncoder, light
    # variant. Gives each task a private adapter before the shared experts.
    # General encoder: cls + loc both use MLP when true; loc can be overridden
    # by mmoe_loc_transformer (line-level transformer for localization).
    mmoe_task_encoder: bool = False
    # MMOE only: override the localization task encoder with a transformer over
    # per-statement features (EDAT line_level_encoder pattern). Statements
    # within each graph self-attend → cross-statement context recovered at line
    # granularity. Useful when per-line LM embedding is used (lines isolated by
    # per-line CodeBERT need a downstream encoder to recover cross-line
    # context). Cls path always uses the general encoder.
    mmoe_loc_transformer: bool = False
    # Cross-task fusion mode (ablation):
    #   true  — gated residual side-branch: fused_mod = fused + γ·cross
    #           (γ zero-init, baseline-safe); func_head stays the fat MLP.
    #   false — in-path replace: fused_mod = cross (no residual, no gate);
    #           func_head simplified to LayerNorm+Linear (EDAT-style thin head).
    cross_task_residual: bool = True
    # ── Statement localization "both" mode ────────────────────────────────────
    # Only used when localization_encoder="both". Controls how GNN + LM features combine.
    #   concat   — torch.cat([gnn, lm]) (legacy, LM dim dominates GNN by 3:1 on UniXcoder)
    #   weighted — (1-α)*gnn + α*lm_proj, α fixed by stmt_lm_alpha
    #   gated    — per-statement learnable gate σ(W·[gnn;lm_proj]), no manual α
    stmt_both_mode: str = "concat"
    stmt_lm_alpha: float = 0.5   # only for stmt_both_mode="weighted"
    # Use raw CodeT5+ encoder hidden states (d_model dim, no proj/L2-norm) instead
    # of the model's projected 256-dim output. <s> token used for classification,
    # full hidden for localization. Only active when func_lm is a codet5p-*-embedding.
    codet5p_raw_encoder: bool = False
    # Apply F.normalize(dim=-1) to CodeT5+ per-token projected vectors so they
    # match the unit-norm scale of the pooled embedding. F3 leaves per_token
    # unnormalized; F6 tests whether normalizing per-token improves localization.
    codet5p_normalize_per_token: bool = False
    # Apply F.normalize(dim=-1) to GNN h_graph (classification) and per-node h
    # (localization) before concat/stmt_head. Symmetric to codet5p_normalize_per_token.
    normalize_gnn_output: bool = False
    # ── Structural graph augmentation (training-time, lmgat_codebert GNN-only) ──
    # Resampled fresh every forward pass (Bernoulli mask, like DropEdge's M ~ Bern(1-p)).
    # Targets the 26-class long-tail: multiplies effective views per sparse-class sample
    # without modeling its distribution (ruled out G-Mixup/GraphSMOTE — too few real
    # samples per tail class to estimate a graphon or train an edge generator).
    #   graph_aug_drop_edge    — DropEdge (Rong et al. 2020, ICLR): probability of
    #                            dropping each edge; edge_attr re-indexed in sync.
    #                            Their reference sampling-percent 0.05-0.5 (deep GCNs);
    #                            for our 4-layer GATv2 a milder 0.1-0.2 is the sane start.
    #   graph_aug_drop_node    — NodeDropping (graph aug survey): probability of
    #                            dropping each node; dropped nodes lose incident edges
    #                            AND have features zeroed (can't leak into pooling).
    #   graph_aug_mask_feature — FeatureMasking (graph aug survey): probability of
    #                            masking each feature column ("col") or entry ("all").
    #   graph_aug_mask_mode    — "col" (mask whole channels, GraphCL-style) | "all"
    #                            (mask random entries) | "row" (≈ node feature drop).
    # 0.0 = disabled (default, matches all prior ablation runs).
    graph_aug_drop_edge: float = 0.0
    graph_aug_drop_node: float = 0.0
    graph_aug_mask_feature: float = 0.0
    graph_aug_mask_mode: str = "col"


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    lm_lr: float = 2e-5         # CodeBERT learning rate for lmgat_ft / lmgat_mc
    # ULMFiT-style fine-tuning (Howard & Ruder 2018). Both work independently or together.
    # LLRD: lr for transformer layer i = lm_lr * lm_llrd_decay^(N-1-i) where N = #layers
    # → top layer ≈ lm_lr, bottom layer ≈ lm_lr * decay^(N-1). 1.0 = disabled (uniform).
    lm_llrd_decay: float = 1.0
    # Gradual unfreezing: schedule = [[start_epoch, n_top_layers_unfrozen | "all"], ...]
    # Below first scheduled epoch: head only. Pass [] to disable.
    lm_unfreeze_schedule: list = field(default_factory=list)
    warmup_ratio: float = 0.0   # fraction of total steps for linear warmup (0 = disabled)
    grad_clip: float = 0.0      # gradient clipping max norm (0 = disabled)
    weight_decay: float = 1e-4
    patience: int = 10          # early stopping patience
    early_stop_metric: str = "f1"  # "f1" (macro, maximize) or "loss" (minimize)
    checkpoint_dir: Path = CHECKPOINT_DIR
    results_dir: Path = RESULTS_DIR
    log_dir: Path = LOG_DIR
    device: str = "cpu"         # set to "cuda" if GPU available
    use_class_weights: bool = True  # inverse-frequency weighting for imbalanced classes
    focal_loss_gamma: float = 0.0  # focal loss gamma; 0 = standard CE, 2.0 recommended for imbalanced
    # Epoch-adaptive inverse-frequency class weights (NOT the real LIVABLE paper loss).
    # Ramps class weights from uniform → inverse-frequency over training epochs.
    # Use this as a simple baseline rebalancing strategy.
    epoch_adaptive_weights: bool = False
    # Real LIVABLE two-branch loss (arXiv:2306.06935, Eq. 11-12).
    # L = T * focal + (1-T) * label_smooth_CE, T = 1 - (epoch/max_epoch)^2
    # focal_gamma and label_smoothing are reused for the two branches.
    # Mutually exclusive with epoch_adaptive_weights — use one or the other.
    livable_loss: bool = False
    # Label smoothing for cross-entropy loss (0.0 = disabled, 0.1 recommended).
    # Prevents overconfidence by replacing hard one-hot targets with soft targets.
    # Helps reduce the loss-F1 gap on imbalanced multiclass (see LOSS_F1_GAP.md §3.2).
    label_smoothing: float = 0.0
    # LR scheduler type: "plateau" (ReduceLROnPlateau, default) or "cosine" (CosineAnnealingLR).
    # "cosine" smoothly decays LR to 0 over all epochs — reduces overfitting in later epochs.
    # "plateau" reduces LR when val_loss stops improving (legacy behavior).
    lr_scheduler: str = "plateau"
    # Bit-exact determinism across runs (CUDA atomics, FlashAttention-2 backward, cuBLAS).
    # Enable only for replication studies — costs 20-40% training speed.
    # For ablation statistical comparisons, prefer multi-seed runs with this flag off.
    deterministic: bool = False
    # ── EDAT adversarial training (Embedding-layer Driven Adversarial Training) ─
    # Perturbs identifier token embeddings (variable/function names) during training
    # to improve robustness and generalisation on rare CWE types.
    # Requires live_lm != 'none' and compile_model=false.
    # Doubles effective LM forward passes per batch — expect ~40-60% slower training.
    use_edat: bool = False
    edat_epsilon: float = 0.02   # L∞ perturbation bound on embedding table
    edat_alpha:   float = 1e-2   # FGSM step size per ascent step
    edat_steps:   int   = 3      # number of FGSM-sign ascent steps K
    # ── Decoupled cRT (Classifier Re-training, Kang et al. 2020, ICLR) ─────────
    # Stage 2 of decoupling for long-tail: load a frozen backbone checkpoint,
    # freeze everything except func_head, re-init the classifier, and re-train it
    # alone with class-balanced sampling. Backbone (incl. BatchNorm running stats)
    # is kept in eval() so representations stay fixed. "" disables (normal run).
    crt_init_checkpoint: str = ""
    crt_reinit_head: bool = True   # randomly re-initialize func_head (paper: re-init W,b)
    # Class-balanced sampler for cRT (Eq. 1, q=0): p_j = 1/C then uniform instance.
    # Independent of supcon_balanced_sampling (that forces N distinct classes per
    # batch for positive pairs — different goal). Replaces shuffle when true.
    class_balanced_sampling: bool = False
    # ── Balanced-Mixup / Remix (Chou et al. 2020) for long-tail ───────────────
    # Manifold mixup on the pooled graph embedding (h_graph): h~ = lam*h_i + (1-lam)*h_j,
    # lam ~ Beta(alpha, alpha). Remix decouples the LABEL mix ratio from the feature
    # ratio so the minority class in each pair keeps more label weight (kappa, tau).
    # 0.0 = disabled. Full-training-time method (not a frozen-head retrain).
    mixup_alpha: float = 0.0       # Beta(alpha, alpha); 0 disables mixup
    mixup_remix: bool = True       # True = Remix imbalance-aware label mixing; False = vanilla
    mixup_remix_kappa: float = 3.0 # n_i/n_j ratio threshold to reassign label to minority
    mixup_remix_tau: float = 0.5   # feature-ratio guard so reassignment only when lam extreme
    # ── Logit Adjustment loss (Menon et al. 2021) for long-tail ───────────────
    # Train-time logit offset: CE is computed on z_y + tau*log(pi_y), pi = class prior.
    # Tail classes (small pi -> very negative log pi) get a negative offset, forcing a
    # larger logit/margin; at inference raw logits are used so the tail is favoured.
    # Imbalance handler — use INSTEAD of class weights, not on top. 0/false disables.
    logit_adjustment: bool = False
    logit_adjustment_tau: float = 1.0
    # ── FLAG adversarial node-feature augmentation (Kong et al. 2020) ──────────
    # GNN analog of EDAT, matching the reference flag() (src/FLAG attacks.py): perturb
    # the GNN INPUT node features, delta ~ U(-step_size, step_size), then M unbounded
    # ascent steps delta += step_size*sign(grad), accumulating param grads (loss/=M)
    # over the M perturbed forwards, one optimizer step. No epsilon ball / no clamp.
    # gnn_only compatible. Requires use_amp=false. 0/false disables.
    use_flag: bool = False
    flag_step_size: float = 1e-3   # init range AND ascent step (single param, = paper step_size)
    flag_steps:     int   = 3      # M ascent steps (= grad-accumulation count)
    # ── Node-masked JEPA downstream init (Q-series) ───────────────────────────
    # Load a JEPA-SSL-pretrained GNN encoder (gnn_vuln.pretrain_jepa) into
    # model.encoder. freeze_gnn=true → frozen linear probe (encoder frozen + kept
    # eval, only func_head trains — mirrors cRT, the canonical JEPA SSL-quality
    # measure); false → finetune from the SSL init (all params trainable).
    # The JEPA SSL hyperparams (jepa_mask_ratio, jepa_ema_start/end, jepa_loss,
    # jepa_predictor_layers, jepa_epochs, jepa_lr, jepa_out_dir) are read directly
    # by pretrain_jepa.py via getattr and need no declared field here.
    gnn_init_checkpoint: str = ""
    freeze_gnn: bool = False


@dataclass
class EWCConfig:
    """EWC-DR continual learning. Read by train.py:_setup_ewc."""
    enabled: bool = False
    weight: float = 1000.0
    scope: str = "all"               # all | gnn | lm
    importance_cache: str = ""       # precomputed Fisher/theta* (computed on task-A)
    source_checkpoint: str = ""      # task-A trained weights
    n_batches: int = 0               # FIM batches (0 = all)
    compute_only: bool = False       # compute+save importance cache then exit (no training)


@dataclass
class ReplayConfig:
    """Experience Replay (Chaudhry et al. 2019). Read by train.py:_setup_replay."""
    enabled: bool = False
    source: str = ""                 # task-A dataset to replay (e.g. megavul)
    ds_name_suffix: str = ""         # task-A .pt suffix (e.g. _vulnonly)
    buffer_per_class: int = 0        # samples per class in the memory buffer (0 = all train)
    weight: float = 1.0              # replay loss weight
    buffer_seed: int = 42
    # Optional overrides for the replay (task-A) dataset params. None = inherit cfg.data.
    # Needed when task-B's data block differs from task-A — e.g. CIL: task-B is
    # megavul_cil (filter off, no cap) but the replay buffer must use task-A's megavul
    # subset (filter_top25 + max_per_class). Set these to the task-A values.
    top_cwe: int | None = None
    filter_top25_dangerous: bool | None = None
    max_per_class: int | None = None
    resample_seed: int | None = None


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ewc: EWCConfig = field(default_factory=EWCConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load config from a single YAML file, merging with defaults.

        Backward compatible: the classic monolithic config (data/model/train in one
        file) is just the one-file case. For split configs, see `from_yamls`."""
        return cls.from_yamls([path])

    @classmethod
    def from_yamls(cls, paths) -> "Config":
        """Compose a Config from one OR MORE YAML files. Each file may carry any subset
        of the sections (data / model / train / ewc / replay); later files override
        earlier ones, section by section.

        This enables split configs — e.g. `from_yamls([data.yaml, model.yaml,
        train.yaml])` — while a single merged file stays the one-element case, so all
        existing callers (CLI training, train_cloud.sh) behave exactly as before."""
        if isinstance(paths, (str, Path)):
            paths = [paths]
        merged: dict = {}
        for p in paths:
            with open(p, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            for section, vals in raw.items():
                if isinstance(vals, dict) and isinstance(merged.get(section), dict):
                    merged[section].update(vals)
                elif isinstance(vals, dict):
                    merged[section] = dict(vals)
                else:
                    merged[section] = vals
        return cls._from_raw(merged)

    @classmethod
    def _from_raw(cls, raw: dict) -> "Config":
        """Apply a merged raw-dict (sections → field maps) onto the dataclass defaults."""
        def _coerce(current, value):
            # honor Path-typed fields when a YAML supplies them as strings
            if isinstance(current, Path) and value is not None and not isinstance(value, Path):
                return Path(value)
            return value

        cfg = cls()
        for section in ("data", "model", "train", "ewc", "replay"):
            if section in raw and isinstance(raw[section], dict):
                target = getattr(cfg, section)
                for k, v in raw[section].items():
                    setattr(target, k, _coerce(getattr(target, k, None), v))
        return cfg


def load_default_config() -> Config:
    default_yaml = PROJECT_ROOT / "configs" / "lmgcn" / "binary.yaml"
    if default_yaml.exists():
        return Config.from_yaml(default_yaml)
    return Config()
