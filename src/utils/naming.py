"""Centralised artifact naming for checkpoints, results, and figures.

Convention:
    {NN}_{category}_{pool}[_{variant}][_d{dim}][_{tag}].json

Where:
    NN       = zero-padded script number (02, 03, 04, ...)
    category = ds, fs, pca, disjoint, bootstrap, ci, disagreement, loo, ...
    pool     = expert pool name (clip4, arch6, ...)
    variant  = optional qualifier (mlp, linear, rawfeat, ...)
    dim      = projection dimension, only when result depends on it
    tag      = train_tag suffix, ONLY for non-default tags

Tag suffix is empty for the default train tag. This avoids cluttering
filenames with '_CUSTOM_TAG' when there is only one training set.
"""

from src.utils.config import DEFAULT_TRAIN_TAG


def tag_suffix(train_tag: str = DEFAULT_TRAIN_TAG) -> str:
    """Return filename suffix for a given train_tag.

    Returns empty string for the default tag, '_{tag}' otherwise.
    """
    if train_tag == DEFAULT_TRAIN_TAG:
        return ""
    return f"_{train_tag}"


def result_name(script_num: str, category: str, pool: str = "",
                variant: str = "", dim: int = None,
                train_tag: str = DEFAULT_TRAIN_TAG) -> str:
    """Build a standardised result JSON filename.

    Args:
        script_num: Zero-padded script number, e.g. "02", "08a".
        category: Result category, e.g. "ds", "fs", "pca".
        pool: Expert pool name. Empty for pool-independent results.
        variant: Optional qualifier, e.g. "mlp", "rawfeat".
        dim: Projection dimension (omit if not applicable).
        train_tag: Training data tag.

    Returns:
        Filename string (no directory), e.g. "05_fs_clip4_mlp_d32.json"
    """
    parts = [script_num, category]
    if pool:
        parts.append(pool)
    if variant:
        parts.append(variant)
    if dim is not None:
        parts.append(f"d{dim}")
    ts = tag_suffix(train_tag)
    return "_".join(parts) + ts + ".json"


def checkpoint_name(pool: str, combiner_type: str, proj_dim: int,
                    seed: int, train_tag: str = DEFAULT_TRAIN_TAG) -> str:
    """Checkpoint filename (no directory)."""
    ts = tag_suffix(train_tag)
    return f"{pool}_{combiner_type}_d{proj_dim}_seed{seed}_best{ts}.pt"
