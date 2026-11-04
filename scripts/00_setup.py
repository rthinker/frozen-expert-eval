"""One-time setup: download CLIP-large/CLIP-base + Xception ImageNet backbone.

Sets HF_HOME and TRANSFORMERS_OFFLINE="0" before any transformers import
(overrides the offline default in df40_adapter.py); this is the only
script that needs internet.
"""

import os
import sys

from _bootstrap import CODEBASE

# HF cache path: same location as df40_adapter.py uses
HF_CACHE = CODEBASE / "third_party" / "hf_cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)

# Must set BEFORE any transformers import. Online for downloads.
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import _bootstrap  # noqa: F401

from src.utils.logging_setup import setup_logging

CLIP_HF_NAMESPACE = "open" + "ai"
MODELS = [
    (f"{CLIP_HF_NAMESPACE}/clip-vit-large-patch14", "CLIP-large (1024d)"),
    (f"{CLIP_HF_NAMESPACE}/clip-vit-base-patch16", "CLIP-base (768d)"),
]


def main():
    log = setup_logging("00_setup")
    log.info(f"HF cache dir: {HF_CACHE}")

    from transformers import CLIPModel, AutoProcessor

    for model_name, desc in MODELS:
        log.info(f"Downloading {desc}: {model_name}")
        try:
            CLIPModel.from_pretrained(model_name)
            AutoProcessor.from_pretrained(model_name)
            log.info(f"  [OK] {desc}")
        except Exception as e:
            log.error(f"  [FAIL] {desc}: {e}")
            return 1

    # ── Xception pretrained backbone (needed by xception/srm/rfm detectors) ──
    from src.df40_adapter import DF40_TRAINING
    pretrained_dir = DF40_TRAINING / "pretrained"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    xception_path = pretrained_dir / "xception-b5690688.pth"
    if xception_path.exists():
        log.info(f"Xception backbone already cached: {xception_path}")
    else:
        url = "https://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth"
        expected_md5 = "b5690688"  # embedded in filename
        log.info("Downloading Xception backbone (~87MB)...")
        try:
            import urllib.request
            import hashlib
            urllib.request.urlretrieve(url, str(xception_path))
            # Verify integrity
            md5 = hashlib.md5(xception_path.read_bytes()).hexdigest()[:8]
            if md5 != expected_md5:
                log.warning(f"  Checksum mismatch: expected {expected_md5}, "
                            f"got {md5}. File may be corrupted.")
            size_mb = xception_path.stat().st_size / 1024 / 1024
            log.info(f"  [OK] Xception backbone: {size_mb:.1f} MB (md5: {md5})")
        except Exception as e:
            log.error(f"  [FAIL] Xception backbone: {e}")
            return 1

    total = sum(f.stat().st_size for f in HF_CACHE.rglob("*") if f.is_file())
    xception_mb = xception_path.stat().st_size / 1024 / 1024 if xception_path.exists() else 0
    log.info(f"Setup complete. HF cache: {total / 1024 / 1024:.0f} MB, "
             f"Xception: {xception_mb:.0f} MB")
    log.info("All subsequent scripts will run offline "
             "(TRANSFORMERS_OFFLINE=1 in df40_adapter.py).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
