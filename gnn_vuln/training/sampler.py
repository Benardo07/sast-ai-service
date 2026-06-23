"""Class-balanced batch sampler for SupCon training."""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler


def class_balanced_sampler(
    labels: list[int] | torch.Tensor,
    seed: int = 42,
) -> WeightedRandomSampler:
    """cRT class-balanced sampler (Kang et al. 2020, ICLR — Eq. 1 with q=0).

    Decoupled stage-2 sampling: each draw picks a class uniformly (p_j = 1/C),
    then an instance from that class uniformly. Implemented as the equivalent
    per-sample weight w_i = 1 / (C * n_{y_i}) with replacement, so tail classes
    are oversampled and head classes undersampled. ``num_samples`` equals the
    dataset size, so one epoch keeps the same length as instance-balanced.

    Use as ``sampler=`` (not ``batch_sampler=``) in a plain DataLoader; it
    replaces ``shuffle=True``. Distinct from SupConBalancedSampler, which forces
    N distinct classes per batch for positive pairs — a different objective.
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
    labels = [int(y) for y in labels]
    counts = Counter(labels)
    n_classes = len(counts)
    weights = [1.0 / (n_classes * counts[y]) for y in labels]
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
        generator=g,
    )


class SupConBalancedSampler(Sampler):
    """
    Batch sampler that guarantees positive pairs in every batch.

    Each batch = classes_per_batch distinct classes × samples_per_class samples.
    Use as `batch_sampler` in DataLoader (replaces batch_size + shuffle).

    RNG is stateful across epochs so each epoch sees different batches.
    """

    def __init__(
        self,
        labels: list[int] | torch.Tensor,
        batch_size: int,
        classes_per_batch: int,
        seed: int = 42,
    ) -> None:
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        self.labels = labels
        self.batch_size = batch_size

        n_classes = len(set(labels))
        self.classes_per_batch = min(classes_per_batch, n_classes)
        self.samples_per_class = max(1, batch_size // self.classes_per_batch)
        self._rng = np.random.default_rng(seed)

        self.class_to_indices: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            self.class_to_indices.setdefault(int(label), []).append(idx)
        self.classes = sorted(self.class_to_indices.keys())
        self.n_batches = len(labels) // batch_size

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            chosen_classes = self._rng.choice(
                self.classes, size=self.classes_per_batch, replace=False
            )
            batch: list[int] = []
            for cls in chosen_classes:
                idxs = self.class_to_indices[cls]
                sampled = self._rng.choice(
                    idxs,
                    size=self.samples_per_class,
                    replace=len(idxs) < self.samples_per_class,
                )
                batch.extend(sampled.tolist())
            yield batch
