"""Cache per-expert features + logits.

Dataset-outer, expert-inner loop so each expert sees identical pixels.
Per-expert resolution via F.interpolate (CLIP 224, Xception family 256;
loader uses 256 max). Outputs the raw DF40_all train cache (consumed by
02_resplit_fit_select.py) and one TEST cache per held-out generator.

Flags: --pool (single pool), --expert (single expert).
"""

import _bootstrap  # noqa: F401  # MUST be first: binds CUBLAS_WORKSPACE_CONFIG before torch.

import argparse
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
import yaml
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.config import (
    LOGS_ROOT, CACHE_ROOT,
    DATASET_ROOT, DATASET_JSON_FOLDER,
    EXPERTS_CFG, RAW_TRAIN_TAG, get_pool_experts,
)

from src.df40_adapter import init_df40, df40_context, get_dataset_class
from src.utils.expert_loader import load_expert
from src.utils.seed import set_seed
from src.utils.logging_setup import setup_logging
from src.utils.train_subset import sample_source
from src.utils.dataset_helpers import (
    TEST_GENERATORS, build_label_dict,
)

MAX_RESOLUTION = 256  # Collect at max; resize per expert


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=str, default=None,
                   help="Named pool from experts.yaml (default: union of all pools)")
    p.add_argument("--expert", nargs="+", default=None,
                   help="Specific expert names (overrides --pool)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32,
                   help="Batch size for inference (32 for 8GB, 64 for 16GB)")
    p.add_argument("--train_samples", type=int, default=0,
                   help="Training subset size (0 = use the full DF40_all train split)")
    p.add_argument("--chunk_size", type=int, default=10000,
                   help="Max images in RAM at once for chunked extraction")
    p.add_argument("--skip_train", action="store_true",
                   help="Skip raw-train caching (only re-extract TEST generators)")
    p.add_argument("--skip_report", action="store_true",
                   help="Skip TEST (8 held-out generators) caching")
    return p.parse_args()


def _load_configs():
    """Build base DF40 config from central config."""
    # Build base config for DF40 dataset class
    base = {
        "dataset_root_rgb": DATASET_ROOT,
        "dataset_json_folder": str(Path(DATASET_JSON_FOLDER).resolve()),
        "compression": "c23",          # Required by DF40 dataset class; c23 is standard
        "resolution": MAX_RESOLUTION,
        "with_mask": False,
        "with_landmark": False,
        "mean": [0.5, 0.5, 0.5],           # Common DF40 input normalization for all experts
        "std": [0.5, 0.5, 0.5],
        "model_name": "clip",               # Placeholder; only matters for SBI-specific code paths
        "frame_num": {"train": 1, "test": 1},   # Image-level: 1 frame per video
        "use_data_augmentation": False,
        "data_aug": {
            "flip_prob": 0.0, "rotate_prob": 0.0, "rotate_limit": [0, 0],
            "blur_prob": 0.0, "blur_limit": [3, 7],
            "brightness_limit": [0, 0], "contrast_limit": [0, 0],
            "quality_lower": 100, "quality_upper": 100,
        },
    }
    return base


def build_loader(base_config, dataset_name, mode, batch_size):
    """Build a DF40 DataLoader for the given dataset and mode.

    Args:
        base_config: Base config dict.
        dataset_name: e.g., "whichisreal", "DF40_all"
        mode: "train" or "test"
        batch_size: Batch size.

    Returns:
        (DataLoader, dataset) tuple. Dataset exposed for frame path extraction.
    """
    config = deepcopy(base_config)
    config["mode"] = mode

    # Set dataset key based on mode
    label_dict = build_label_dict(dataset_name)
    if mode == "train":
        config["train_dataset"] = [dataset_name]
        config["train_batchSize"] = batch_size
    else:
        config["test_dataset"] = dataset_name
        config["test_batchSize"] = batch_size

    # Merge label dict
    if "label_dict" not in config:
        config["label_dict"] = {}
    config["label_dict"].update(label_dict)

    # DF40_all has many label prefixes -- load them from DF40's train_config
    if dataset_name == "DF40_all":
        from src.df40_adapter import DF40_TRAINING
        train_cfg_path = DF40_TRAINING / "config" / "train_config.yaml"
        with open(train_cfg_path, encoding="utf-8") as f:
            df40_cfg = yaml.safe_load(f)
        if "label_dict" in df40_cfg:
            config["label_dict"].update(df40_cfg["label_dict"])

    DatasetClass = get_dataset_class()
    with df40_context():
        dataset = DatasetClass(config=config, mode=mode)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=False,
            drop_last=False, collate_fn=dataset.collate_fn,
        )
    return loader, dataset


def _cache_fingerprint(labels, sample_ids=None):
    """Compute a stable fingerprint from labels + sample IDs for cache verification.

    Includes sample IDs (frame paths) hash when available so the fingerprint
    detects reordering, not just label distribution.
    """
    import hashlib
    h_labels = hashlib.sha256(labels.tobytes()).hexdigest()[:16]
    if sample_ids is not None:
        h_ids = hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()[:16]
    else:
        h_ids = "no_ids"
    return f"n={len(labels)}_r={int((labels==0).sum())}_f={int((labels==1).sum())}_{h_labels}_{h_ids}"


_DF40_MARKER = "/DF40/"


def _canonicalize_sample_id(raw) -> str:
    """Canonical relative DF40 path; strips local absolute prefix.

    The DF40 loader resolves frame paths to absolute filesystem locations
    (e.g. `<DF40_ROOT>/DF40/<gen>/<class>/<file>`).
    For a portable, anonymity-clean cache we want only the relative
    `<gen>/<class>/<file>` part, with forward slashes. Idempotent: a path
    that does not contain `/DF40/` is returned unchanged (already
    relative or non-DF40 source).
    """
    if not isinstance(raw, str):
        raw = str(raw)
    forward = raw.replace("\\", "/")
    idx = forward.rfind(_DF40_MARKER)
    if idx == -1:
        return forward
    return forward[idx + len(_DF40_MARKER):]


def collect_all(loader, desc="collecting"):
    """Iterate loader once, collect all images and labels into memory.
    Returns (images, labels, kept_indices) where kept_indices tracks
    which global sample indices were actually emitted.
    """
    all_images = []
    all_labels = []
    kept_indices = []
    global_idx = 0
    for batch in tqdm(loader, desc=desc):
        if batch is None:
            continue
        imgs = batch["image"]
        labs = batch["label"]
        if imgs.numel() == 0:
            continue
        bs = imgs.shape[0]
        all_images.append(imgs.cpu())
        all_labels.append(labs.cpu())
        kept_indices.extend(range(global_idx, global_idx + bs))
        global_idx += bs
    images = torch.cat(all_images, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy().astype(np.int32)
    return images, labels, kept_indices


def collect_balanced(loader, max_samples, desc="collecting balanced",
                     target_indices=None):
    """Collect balanced real/fake subset from loader.

    Stops each class bucket at max_samples/2. Returns early once full.
    Also tracks global dataset indices for sample ID extraction.
    If target_indices is provided, only keeps samples in that set
    (makes selection deterministic regardless of loader order).
    """
    per_class = max_samples // 2
    real_imgs, fake_imgs = [], []
    real_labs, fake_labs = [], []
    real_idx, fake_idx = [], []
    global_idx = 0

    for batch in tqdm(loader, desc=desc):
        if batch is None:
            continue
        imgs = batch["image"]
        labs = batch["label"]
        if imgs.numel() == 0:
            continue

        for i in range(imgs.shape[0]):
            sample_idx = global_idx + i
            if target_indices is not None and sample_idx not in target_indices:
                continue
            lab = int(labs[i].item())
            if lab == 0 and len(real_imgs) < per_class:
                real_imgs.append(imgs[i:i + 1].half())  # fp16 to halve RAM
                real_labs.append(0)
                real_idx.append(global_idx + i)
            elif lab == 1 and len(fake_imgs) < per_class:
                fake_imgs.append(imgs[i:i + 1].half())  # fp16 to halve RAM
                fake_labs.append(1)
                fake_idx.append(global_idx + i)
            if len(real_imgs) >= per_class and len(fake_imgs) >= per_class:
                break
        global_idx += imgs.shape[0]
        if len(real_imgs) >= per_class and len(fake_imgs) >= per_class:
            break

    images = torch.cat(real_imgs + fake_imgs, dim=0)
    labels = np.array(real_labs + fake_labs, dtype=np.int32)
    indices = real_idx + fake_idx
    print(f"  Collected {len(real_imgs)} real + {len(fake_imgs)} fake = {len(labels)}")
    return images, labels, indices


@torch.no_grad()
def run_expert_on_collected(model, images, expert_info, device, batch_size=32):
    """Run a frozen expert on collected images, return features + logits.

    Handles per-expert resolution via F.interpolate.
    """
    target_res = expert_info["resolution"]
    all_features = []
    all_logits = []

    for start in range(0, len(images), batch_size):
        batch_imgs = images[start:start + batch_size].to(device, dtype=torch.float32)

        # Resize to expert's expected resolution if needed
        if batch_imgs.shape[-1] != target_res:
            batch_imgs = F.interpolate(
                batch_imgs, size=(target_res, target_res),
                mode="bilinear", align_corners=False,
            )

        data_dict = {"image": batch_imgs}
        out = model(data_dict, inference=True)

        logits = out["cls"].cpu().numpy()
        feat = out.get("feat", None)
        if feat is not None:
            # Global average pool if spatial (xception returns [B,D,H,W])
            if feat.dim() > 2:
                feat = feat.mean(dim=list(range(2, feat.dim())))
            feat = feat.cpu().numpy()

        all_logits.append(logits)
        if feat is not None:
            all_features.append(feat)

    logits = np.concatenate(all_logits, axis=0)
    features = np.concatenate(all_features, axis=0) if all_features else None
    return features, logits


@torch.no_grad()
def extract_expert_streaming(model, loader, expert_info, device, batch_size=32,
                             desc="streaming"):
    """Run a frozen expert by streaming through a dataloader.

    Never holds more than one batch of images in RAM.
    Returns features [N, D], logits [N, 2], labels [N], sample_indices [N].
    """
    target_res = expert_info["resolution"]
    all_features = []
    all_logits = []
    all_labels = []
    global_idx = 0
    all_indices = []

    for batch in tqdm(loader, desc=desc):
        if batch is None:
            continue
        imgs = batch["image"]
        labs = batch["label"]
        if imgs.numel() == 0:
            continue

        bs = imgs.shape[0]
        all_indices.extend(range(global_idx, global_idx + bs))
        global_idx += bs

        imgs = imgs.to(device)
        if imgs.shape[-1] != target_res:
            imgs = F.interpolate(imgs, size=(target_res, target_res),
                                 mode="bilinear", align_corners=False)

        out = model({"image": imgs}, inference=True)

        logits = out["cls"].cpu().numpy()
        feat = out.get("feat", None)
        if feat is not None:
            if feat.dim() > 2:
                feat = feat.mean(dim=list(range(2, feat.dim())))
            feat = feat.cpu().numpy()

        all_logits.append(logits)
        if feat is not None:
            all_features.append(feat)
        all_labels.append(labs.cpu().numpy().astype(np.int32))

    logits = np.concatenate(all_logits, axis=0)
    features = np.concatenate(all_features, axis=0) if all_features else None
    labels = np.concatenate(all_labels, axis=0)
    return features, logits, labels, all_indices


def main():
    args = parse_args()
    set_seed(1024)
    log = setup_logging("01_extract", LOGS_ROOT)
    init_df40()

    base_config = _load_configs()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if torch.cuda.is_available():
        log.info(f"  GPU: {torch.cuda.get_device_name(args.gpu)}")
        log.info(f"  VRAM: {torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3:.1f} GB")

    # Determine experts
    if args.expert:
        expert_names = args.expert
    elif args.pool:
        expert_names = get_pool_experts(args.pool)
    else:
        # Default: union of all named pools (extracts everything needed
        # for all paper experiments in one pass)
        all_needed = []
        seen = set()
        for pool_name, pool_experts in EXPERTS_CFG["pools"].items():
            for e in pool_experts:
                if e not in seen:
                    all_needed.append(e)
                    seen.add(e)
        expert_names = all_needed
        log.info(f"No --pool specified: extracting union of all pools "
                 f"({len(expert_names)} experts)")
    log.info(f"Experts ({len(expert_names)}): {expert_names}")

    out_root = CACHE_ROOT
    out_root.mkdir(parents=True, exist_ok=True)

    # Build dataset list: (name, mode, max_samples_or_None, tag).
    #
    # The raw DF40_all train cache lives under RAW_TRAIN_TAG (default
    # `_raw_df40_train/`). The FIT and SELECT caches are produced downstream
    # by 02_resplit_fit_select.py, which reads from RAW_TRAIN_TAG,
    # deduplicates by sample_id, and stratified-splits 80/20. We no longer
    # emit a `DF40_all_test/` split here -- the old "SELECT from per-regime
    # test union" construction was the cause of the 140-real corrupt SELECT
    # and is removed entirely.
    datasets_to_cache = []
    if not args.skip_train:
        datasets_to_cache.append(
            ("DF40_all", "train", args.train_samples, RAW_TRAIN_TAG))
    if not args.skip_report:
        for gen in TEST_GENERATORS:
            datasets_to_cache.append((gen, "test", None, gen))

    log.info(f"Datasets to cache: {[d[3] for d in datasets_to_cache]}")
    total_start = time.time()

    for ds_name, mode, max_samples, tag in datasets_to_cache:
        log.info(f"\n{'='*60}")
        log.info(f"Dataset: {tag} (mode={mode})")
        log.info(f"{'='*60}")

        ds_dir = out_root / tag

        # Check if already fully cached for all experts
        labels_path = ds_dir / "labels.npy"
        fp_path = ds_dir / "fingerprint.txt"
        all_experts_cached = all(
            (ds_dir / f"{name}_features.npy").exists()
            and (ds_dir / f"{name}_logits.npy").exists()
            for name in expert_names
        )
        if all_experts_cached and labels_path.exists() and fp_path.exists():
            log.info("  All experts cached. Skipping.")
            continue

        # If labels exist from a previous partial run, verify fingerprint
        # before adding more experts (ensures sample alignment)
        if labels_path.exists() and fp_path.exists():
            existing_fp = fp_path.read_text(encoding="utf-8").strip()
            log.info(f"  Partial cache found (fingerprint: {existing_fp})")
            log.info("  Will add missing experts to existing cache.")

        # Build loader and collect images into memory. (The old
        # `DF40_all_test` regime-union branch was removed: it produced a
        # 140-real corrupt SELECT which has been superseded by the
        # 02_resplit_fit_select.py deduplicate + stratified-split path.)
        if True:
            log.info("  Building loader...")
            loader, dataset = build_loader(base_config, ds_name, mode, args.batch_size)
            # Canonicalize once: every downstream sample_ids list reads
            # from this array, so normalizing here means every cached
            # sample_id is a relative DF40 path with forward slashes,
            # never an absolute author-machine path.
            all_frame_paths = [
                _canonicalize_sample_id(p)
                for p in dataset.data_dict["image"]
            ]
            if max_samples and max_samples > 0 and mode == "train":
                # Pre-select balanced subset via STRATIFIED sampling
                # across manipulation types. Within each type, paths are
                # sorted alphabetically for determinism.
                all_data_labels = dataset.data_dict["label"]
                per_class = max_samples // 2

                # Group indices by (manipulation_type, label)
                from collections import defaultdict
                type_label_indices = defaultdict(list)
                for idx in range(len(all_frame_paths)):
                    mtype = sample_source(all_frame_paths[idx])
                    lab = int(all_data_labels[idx])
                    type_label_indices[(mtype, lab)].append(idx)

                # Sort indices within each group by path (deterministic)
                for key in type_label_indices:
                    type_label_indices[key].sort(
                        key=lambda i: all_frame_paths[i])

                # Proportional allocation per type, per class
                target_indices = set()
                for lab_val, budget in [(0, per_class), (1, per_class)]:
                    groups = {k: v for k, v in type_label_indices.items()
                              if k[1] == lab_val}
                    total_available = sum(len(v) for v in groups.values())
                    selected = 0
                    for (mtype, _), indices in sorted(groups.items()):
                        # Proportional share, at least 1 if available
                        share = max(1, int(len(indices) / total_available * budget))
                        share = min(share, len(indices), budget - selected)
                        target_indices.update(indices[:share])
                        selected += share
                    # Fill remainder from largest groups
                    if selected < budget:
                        remaining = budget - selected
                        for (mtype, _), indices in sorted(
                                groups.items(),
                                key=lambda x: len(x[1]), reverse=True):
                            unused = [i for i in indices
                                      if i not in target_indices]
                            take = min(len(unused), remaining)
                            target_indices.update(unused[:take])
                            remaining -= take
                            if remaining <= 0:
                                break

                # Log distribution
                from collections import Counter
                type_counts = Counter()
                for idx in target_indices:
                    type_counts[sample_source(all_frame_paths[idx])] += 1
                dist_str = ", ".join(f"{k}:{v}" for k, v in
                                     sorted(type_counts.items()))
                log.info(f"  Pre-selected {len(target_indices)} samples "
                         f"(r={per_class}, f={per_class}) stratified: "
                         f"{dist_str}")
                images, labels, kept_indices = collect_balanced(
                    loader, max_samples, desc=f"  {tag} (balanced)",
                    target_indices=target_indices)
                sample_ids = [all_frame_paths[i] for i in kept_indices]
            elif max_samples == 0 and mode == "train":
                # Chunked mode: too large for RAM all at once.
                # Collect images in chunks of chunk_size, run ALL experts
                # per chunk (preserves sample alignment), free images, repeat.
                chunk_size = args.chunk_size
                log.info(f"  CHUNKED MODE (train_samples=0): "
                         f"chunk_size={chunk_size}")

                ds_dir.mkdir(parents=True, exist_ok=True)

                # Accumulators: per-expert feature/logit lists
                expert_feats_acc = {n: [] for n in expert_names}
                expert_logits_acc = {n: [] for n in expert_names}
                all_labels_acc = []
                all_indices_acc = []
                global_idx = 0

                chunk_imgs = []
                chunk_labs = []
                chunk_idx = []
                chunk_count = 0

                def _process_chunk(chunk_imgs_list, chunk_labs_list,
                                   chunk_idx_list, chunk_num):
                    """Run all experts on one chunk of images."""
                    imgs_t = torch.cat(chunk_imgs_list, dim=0)
                    labs = np.concatenate(chunk_labs_list, axis=0)
                    log.info(f"    Chunk {chunk_num}: {len(labs)} images")

                    for ei, name in enumerate(expert_names):
                        # Check if this expert already done for all data
                        # (can't check per-chunk, just skip if fully done)
                        model, info = load_expert(name, device=device)
                        features, logits = run_expert_on_collected(
                            model, imgs_t, info, device,
                            batch_size=args.batch_size)
                        expert_feats_acc[name].append(features)
                        expert_logits_acc[name].append(logits)
                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    all_labels_acc.extend(chunk_labs_list)
                    all_indices_acc.extend(chunk_idx_list)
                    del imgs_t

                for batch in tqdm(loader, desc=f"  {tag} (chunked)"):
                    if batch is None:
                        continue
                    imgs = batch["image"]
                    labs = batch["label"]
                    if imgs.numel() == 0:
                        continue

                    bs = imgs.shape[0]
                    chunk_imgs.append(imgs.cpu())
                    chunk_labs.append(labs.cpu().numpy().astype(np.int32))
                    chunk_idx.extend(range(global_idx, global_idx + bs))
                    global_idx += bs

                    n_in_chunk = sum(c.shape[0] for c in chunk_imgs)
                    if n_in_chunk >= chunk_size:
                        chunk_count += 1
                        _process_chunk(chunk_imgs, chunk_labs,
                                       chunk_idx, chunk_count)
                        chunk_imgs, chunk_labs, chunk_idx = [], [], []

                # Process remaining
                if chunk_imgs:
                    chunk_count += 1
                    _process_chunk(chunk_imgs, chunk_labs,
                                   chunk_idx, chunk_count)

                # Concatenate all chunks
                labels = np.concatenate(all_labels_acc, axis=0)
                sample_ids = [all_frame_paths[i]
                              for i in all_indices_acc
                              if i < len(all_frame_paths)]
                fingerprint = _cache_fingerprint(labels, sample_ids)

                # Save labels + fingerprint
                np.save(ds_dir / "labels.npy", labels)
                np.save(ds_dir / "sample_ids.npy",
                        np.array(sample_ids, dtype=str), allow_pickle=False)
                with open(ds_dir / "fingerprint.txt", "w", encoding="utf-8") as fout:
                    fout.write(fingerprint)
                log.info(f"  Total: {len(labels)} samples, "
                         f"real={int((labels==0).sum())}, "
                         f"fake={int((labels==1).sum())}")
                log.info(f"  Fingerprint: {fingerprint}")

                # Save per-expert concatenated features
                for name in expert_names:
                    feats = np.concatenate(expert_feats_acc[name], axis=0)
                    logits = np.concatenate(expert_logits_acc[name], axis=0)
                    np.save(ds_dir / f"{name}_features.npy",
                            feats.astype(np.float16))
                    np.save(ds_dir / f"{name}_logits.npy",
                            logits.astype(np.float32))
                    log.info(f"    {name}: feat={feats.shape}, "
                             f"logits={logits.shape}")

                del expert_feats_acc, expert_logits_acc
                log.info(f"  Dataset {tag} complete (chunked): "
                         f"{len(expert_names)} experts, "
                         f"{chunk_count} chunks")
                continue  # skip the normal collect-then-extract path
            else:
                images, labels, kept_idx = collect_all(loader, desc=f"  {tag}")
                sample_ids = [all_frame_paths[i] for i in kept_idx
                              if i < len(all_frame_paths)]

        log.info(f"  Collected: {len(labels)} samples, "
                 f"images={images.shape}, "
                 f"real={int((labels == 0).sum())}, "
                 f"fake={int((labels == 1).sum())}")

        # Save labels, sample_ids, fingerprint, or verify existing cache
        ds_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = _cache_fingerprint(labels, sample_ids)
        fp_path = ds_dir / "fingerprint.txt"
        if fp_path.exists():
            existing_fp = fp_path.read_text(encoding="utf-8").strip()
            if existing_fp != fingerprint:
                log.error(
                    f"  FINGERPRINT MISMATCH on resume!\n"
                    f"  Cached:  {existing_fp}\n"
                    f"  Current: {fingerprint}\n"
                    f"  Delete {ds_dir} and re-run from scratch.")
                raise RuntimeError(f"Cache fingerprint mismatch for {tag}")
            log.info(f"  Fingerprint verified: {fingerprint}")
        else:
            np.save(ds_dir / "labels.npy", labels)
            np.save(ds_dir / "sample_ids.npy", np.array(sample_ids, dtype=str), allow_pickle=False)
            with open(fp_path, "w", encoding="utf-8") as fout:
                fout.write(fingerprint)
            log.info(f"  Saved labels ({len(labels)}), "
                     f"sample_ids ({len(sample_ids)}), "
                     f"fingerprint: {fingerprint}")

        # Run each expert on the SAME collected images
        for ei, name in enumerate(expert_names):
            feat_path = ds_dir / f"{name}_features.npy"
            logit_path = ds_dir / f"{name}_logits.npy"

            if feat_path.exists() and logit_path.exists():
                log.info(f"  [{ei+1}/{len(expert_names)}] "
                         f"{name}: already cached, skipping")
                continue

            log.info(f"  [{ei+1}/{len(expert_names)}] {name}: loading...")
            t0 = time.time()
            try:
                model, info = load_expert(name, device=device)
            except Exception as e:
                log.error(f"  FAILED to load {name}: {e}")
                continue

            log.info(f"    Loaded in {time.time()-t0:.1f}s "
                     f"(feat_dim={info['feature_dim']}, "
                     f"res={info['resolution']})")

            # Extract features + logits
            t1 = time.time()
            features, logits = run_expert_on_collected(
                model, images, info, device, batch_size=args.batch_size)
            elapsed = time.time() - t1

            log.info(f"    Extracted in {elapsed:.1f}s: "
                     f"feat={'x'.join(str(s) for s in features.shape) if features is not None else 'None'}, "
                     f"logits={'x'.join(str(s) for s in logits.shape)}")

            # Verify feature dimension matches config
            if features is not None:
                expected_dim = info["feature_dim"]
                actual_dim = features.shape[1]
                if actual_dim != expected_dim:
                    log.warning(
                        f"    Feature dim mismatch: expected {expected_dim}, "
                        f"got {actual_dim}")

            # Save (features as fp16 for storage, logits as fp32)
            if features is not None:
                np.save(feat_path, features.astype(np.float16))
            np.save(logit_path, logits.astype(np.float32))

            feat_mb = features.nbytes / 1024 / 1024 if features is not None else 0
            log.info(f"    Saved: features={feat_mb:.1f}MB (fp16), "
                     f"logits={logits.nbytes/1024:.1f}KB")

            # Unload expert to free GPU memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Free collected images for this dataset
        del images
        log.info(f"  Dataset {tag} complete: "
                 f"{len(expert_names)} experts cached")

    # ── Summary ──
    total_time = time.time() - total_start
    log.info(f"\n{'='*60}")
    log.info("FEATURE EXTRACTION COMPLETE")
    log.info(f"{'='*60}")
    log.info(f"Total time: {total_time/60:.1f} minutes")
    log.info(f"Output: {out_root}")

    # Report what was cached
    for _, _, _, tag in datasets_to_cache:
        ds_dir = out_root / tag
        if not ds_dir.exists():
            continue
        n_feat = len(list(ds_dir.glob("*_features.npy")))
        n_logit = len(list(ds_dir.glob("*_logits.npy")))
        labels_path = ds_dir / "labels.npy"
        n_samples = len(np.load(labels_path)) if labels_path.exists() else 0
        log.info(f"  {tag}: {n_samples} samples, "
                 f"{n_feat} feature files, {n_logit} logit files")


if __name__ == "__main__":
    main()
