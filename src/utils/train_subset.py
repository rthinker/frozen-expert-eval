from pathlib import Path

import numpy as np

from src.utils.cache_ids import load_sample_ids


def sample_source(sample_id):
    """Top-level source folder in a portable DF40 sample identifier."""
    path = str(sample_id).replace("\\", "/").removeprefix("DF40/")
    source, separator, _ = path.partition("/")
    return source if separator else ""


def balanced_subset_indices(labels, max_samples, sample_ids_path=None):
    """Sample equal class counts, proportionally across known source folders."""
    if max_samples is None or max_samples <= 0 or max_samples >= len(labels):
        return np.arange(len(labels), dtype=np.int64)

    labels = np.asarray(labels)
    per_class = max_samples // 2

    if sample_ids_path is not None and Path(sample_ids_path).exists():
        sample_ids = load_sample_ids(sample_ids_path)
        if len(sample_ids) != len(labels):
            raise ValueError(
                f"sample_ids length mismatch: {len(sample_ids)} ids vs "
                f"{len(labels)} labels"
            )
    else:
        sample_ids = None

    real, fake = (int(np.count_nonzero(labels == label)) for label in (0, 1))
    if min(real, fake) < per_class:
        raise ValueError(
            f"Could not build balanced subset of {max_samples}: "
            f"available real={real}, fake={fake}"
        )

    rng = np.random.default_rng(1024)
    order = np.arange(len(labels)) if sample_ids is None else np.argsort(sample_ids, kind="stable")
    order = rng.permutation(order)
    chosen = []
    for label in (0, 1):
        indices = order[labels[order] == label]
        if sample_ids is None:
            chosen.extend(indices[:per_class])
            continue
        groups = {}
        for index in indices:
            groups.setdefault(sample_source(sample_ids[index]), []).append(index)
        # Allocate fractional shares by largest remainder to keep exact class counts.
        shares = {source: divmod(per_class * len(group), len(indices)) for source, group in groups.items()}
        spare = per_class - sum(count for count, _ in shares.values())
        ranked = sorted(shares, key=lambda source: (-shares[source][1], source))
        for rank, source in enumerate(ranked):
            count = shares[source][0] + (rank < spare)
            chosen.extend(groups[source][:count])

    return np.array(sorted(chosen), dtype=np.int64)
