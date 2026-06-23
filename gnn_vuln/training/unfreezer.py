"""ULMFiT-style gradual unfreezing for pretrained LM fine-tuning.

Reference: Howard & Ruder, "Universal Language Model Fine-tuning for Text
Classification" (ACL 2018, arXiv:1801.06146).

Initial state: all LM layers frozen (only classification head + GNN trainable).
At each scheduled epoch, top-N transformer layers are unfrozen; embeddings stay
frozen unless the schedule reaches "all".

Schedule format (from config):
    lm_unfreeze_schedule:
      - [1, 0]      # epoch 1: head only (0 LM layers trainable)
      - [3, 4]      # epoch 3: unfreeze top 4 LM layers
      - [6, 8]      # epoch 6: unfreeze top 8 LM layers
      - [10, "all"] # epoch 10: unfreeze everything (incl. embeddings)
"""
from __future__ import annotations

import torch.nn as nn


class GradualUnfreezer:
    def __init__(self, lm_module: nn.Module, schedule: list) -> None:
        self.lm = lm_module
        # Normalize: sort by epoch ascending, keep last value if duplicates.
        self.schedule = sorted(
            [(int(e), n) for e, n in schedule], key=lambda x: x[0]
        )
        self.layers = self._find_layers(lm_module)
        self.n_layers = len(self.layers)
        for p in lm_module.parameters():
            p.requires_grad = False
        self._current_state: int | str | None = None

    @staticmethod
    def _find_layers(lm_module: nn.Module) -> list[nn.Module]:
        # RoBERTa / BERT / UniXcoder / GraphCodeBERT
        if hasattr(lm_module, "encoder") and hasattr(lm_module.encoder, "layer"):
            return list(lm_module.encoder.layer)
        # CodeT5+ embedding model — encoder.block (T5 stack)
        if hasattr(lm_module, "encoder") and hasattr(lm_module.encoder, "block"):
            return list(lm_module.encoder.block)
        return []

    def step(self, epoch: int) -> str | None:
        """Apply unfreezing rule for `epoch`. Returns a log message if state changed."""
        target = None
        for ep, n in self.schedule:
            if epoch >= ep:
                target = n
        if target is None or target == self._current_state:
            return None
        if self.n_layers == 0:
            return None

        for p in self.lm.parameters():
            p.requires_grad = False

        is_all = isinstance(target, str) and target.lower() == "all"
        n_unfreeze = self.n_layers if is_all else min(int(target), self.n_layers)

        for layer in self.layers[self.n_layers - n_unfreeze:]:
            for p in layer.parameters():
                p.requires_grad = True

        for name, p in self.lm.named_parameters():
            if any(f"layer.{i}." in name or f"block.{i}." in name for i in range(self.n_layers)):
                continue
            if "embeddings" in name and not is_all:
                continue
            p.requires_grad = True

        self._current_state = target
        emb_state = "trainable" if is_all else "frozen"
        return (
            f"GradualUnfreezer epoch {epoch}: top {n_unfreeze}/{self.n_layers} "
            f"LM layers trainable, embeddings {emb_state}"
        )
