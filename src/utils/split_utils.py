"""Split helpers used by training scripts to keep split behavior consistent."""

from typing import Dict, Optional, Tuple

from src.config import BATCHES
from src.experiments.domain_split import DomainSplitter


def build_split_tag(split_mode: str, test_batch: Optional[int] = None) -> str:
    """Return a stable tag for checkpoint naming."""
    if split_mode == "random":
        return "random"
    if split_mode == "batch":
        batch_id = 3 if test_batch is None else test_batch
        return f"batch_test{batch_id}"
    raise ValueError(f"未知划分方式: {split_mode}")


def resolve_split(
    labels: Dict,
    n_samples: int,
    split_mode: str,
    test_batch: Optional[int],
    random_seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[Dict, str, Optional[list], Optional[list]]:
    """
    Resolve split indices and a split tag.

    Returns:
        split: dict with train/val/test indices
        split_tag: stable name suffix for artifacts
        train_batches: list for batch split, else None
        test_batches: list for batch split, else None
    """
    splitter = DomainSplitter(random_seed)
    split_tag = build_split_tag(split_mode, test_batch)

    if split_mode == "random":
        split = splitter.random_split(
            n_samples=n_samples,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        return split, split_tag, None, None

    if split_mode == "batch":
        batch_id = 3 if test_batch is None else test_batch
        if batch_id < 0 or batch_id >= len(BATCHES):
            raise ValueError(f"--test-batch 必须在 [0, {len(BATCHES) - 1}] 范围内")

        test_batches = [batch_id]
        train_batches = [i for i in range(len(BATCHES)) if i != batch_id]
        split = splitter.batch_split(
            batch_labels=labels["batch"],
            train_batches=train_batches,
            test_batches=test_batches,
            val_ratio=val_ratio,
        )
        return split, split_tag, train_batches, test_batches

    raise ValueError(f"未知划分方式: {split_mode}")
