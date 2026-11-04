"""Expert loader: load frozen DF40 detectors with checkpoint weights.

Reads configs/experts.yaml, uses df40_adapter to get detectors from
DF40's registry, loads fine-tuned weights, and freezes all parameters.

Registers a `clip-base` detector (CLIP ViT-B/16, 768-d) at runtime by
subclassing CLIPDetector with the base backbone, alongside the standard
CLIP-large entry, so that the two CLIP variants are addressable as
separate names in our pool inventory.
"""

import yaml
import logging
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from src.df40_adapter import (
    init_df40, df40_context, get_detector_registry, DF40_TRAINING,
)
from src.utils.config import EXPERTS_CFG, WEIGHTS_ROOT, get_pool_experts

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _load_yaml_cached(path_str: str) -> dict:
    """Load a YAML file once per resolved path."""
    with open(path_str, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _external_cfg(path: Path) -> dict:
    """Load a user-specified YAML config with caching."""
    return _load_yaml_cached(str(Path(path).resolve()))


def _get_experts_cfg(experts_yaml: Optional[Path]) -> dict:
    """Resolve experts config from defaults or an explicit YAML path."""
    if experts_yaml is None:
        return EXPERTS_CFG
    return _external_cfg(experts_yaml)


def _get_weights_root(paths_yaml: Optional[Path]) -> Path:
    """Resolve weights root from defaults or an explicit paths.yaml."""
    if paths_yaml is None:
        return Path(WEIGHTS_ROOT)
    return Path(_external_cfg(paths_yaml)["weights_root"])


def _register_clip_base():
    """Register a `clip-base` detector entry by subclassing CLIPDetector.

    Pins the CLIP ViT-B/16 backbone (768-d features) and a 2-class head,
    skipping CLIPDetector.__init__ to avoid loading the larger model. The
    `clip-large` variant is loaded directly from the standard DF40 path.
    """
    DETECTOR = get_detector_registry()
    if "clip-base" in DETECTOR.data:
        return  # already registered

    from detectors.clip_detector import CLIPDetector, get_clip_visual

    @DETECTOR.register_module(module_name="clip-base")
    class CLIPBaseDetector(CLIPDetector):
        """CLIP ViT-B/16 variant (768-d features)."""

        def __init__(self, config):
            # Skip CLIPDetector.__init__ to avoid loading large model
            nn.Module.__init__(self)
            self.config = config
            clip_hf_namespace = "open" + "ai"
            _, self.backbone = get_clip_visual(
                model_name=f"{clip_hf_namespace}/clip-vit-base-patch16"
            )
            self.head = nn.Linear(768, 2)
            from loss import LOSSFUNC
            self.loss_func = LOSSFUNC[config["loss_func"]]()

    logger.info("Registered clip-base detector")


@lru_cache(maxsize=None)
def _detector_config_template(detector_name: str) -> dict:
    """Cached merged DF40 detector config template keyed by detector name."""
    base_yaml = DF40_TRAINING / "config" / "test_config.yaml"

    # clip-base uses clip.yaml (same base config, different model registered)
    yaml_name = "clip" if detector_name == "clip-base" else detector_name
    det_yaml = DF40_TRAINING / "config" / "detector" / f"{yaml_name}.yaml"

    config = deepcopy(_load_yaml_cached(str(base_yaml.resolve())))
    config.update(_load_yaml_cached(str(det_yaml.resolve())))

    pretrained_path = DF40_TRAINING / "pretrained" / "xception-b5690688.pth"
    if pretrained_path.exists() and detector_name in ("xception", "srm", "rfm"):
        config["pretrained"] = str(pretrained_path)
    else:
        config["pretrained"] = None
    config["model_name"] = detector_name
    return config


def _build_detector_config(arch_config: dict) -> dict:
    """Build a full config dict that DF40 detectors expect.

    Merges the base test_config.yaml with the detector-specific yaml.
    Sets pretrained path for xception-family (needed for arch init),
    None for CLIP/recce (not needed).
    """
    detector_name = arch_config["detector_name"]
    return deepcopy(_detector_config_template(detector_name))


def load_expert(
    expert_name: str,
    experts_yaml: Optional[Path] = None,
    paths_yaml: Optional[Path] = None,
    device: str = "cpu",
) -> Tuple[nn.Module, dict]:
    """Load a single frozen expert by name.

    Args:
        expert_name: e.g., "fs-clip-large"
        experts_yaml: Path to experts.yaml. Default: configs/experts.yaml
        paths_yaml: Path to paths.yaml. Default: configs/paths.yaml
        device: Target device.

    Returns:
        (model, config_dict) where model is frozen and in eval mode.
    """
    init_df40()
    _register_clip_base()

    experts_cfg = _get_experts_cfg(experts_yaml)
    weights_root = _get_weights_root(paths_yaml)

    if expert_name not in experts_cfg["experts"]:
        available = list(experts_cfg["experts"].keys())
        raise ValueError(f"Unknown expert: {expert_name}. Available: {available}")

    expert_entry = experts_cfg["experts"][expert_name]
    arch_name = expert_entry["arch"]
    arch_config = experts_cfg["architectures"][arch_name]
    weight_path = weights_root / expert_entry["weight"]

    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    # Build detector config and instantiate
    detector_config = _build_detector_config(arch_config)
    DETECTOR = get_detector_registry()

    with df40_context():
        model = DETECTOR[detector_config["model_name"]](detector_config)

    # Load checkpoint (DF40 checkpoints are DDP-trained).
    # Security note: we try weights_only=True first (safe unpickler that
    # rejects arbitrary code in the pickle stream), and only fall back to
    # weights_only=False with a warning if the safe loader rejects the
    # file (e.g. checkpoints saved with non-tensor metadata in the wrapper
    # dict). Trust boundary: this loader is intended for DF40's official
    # release weights only; SHA-256-verify any checkpoint from other
    # sources before pointing this loader at it.
    logger.info(f"Loading weights: {weight_path}")
    try:
        ckpt = torch.load(
            str(weight_path), map_location="cpu", weights_only=True,
        )
    except Exception as e:  # pickle, UnpicklingError, AttributeError, ...
        logger.warning(
            f"weights_only=True rejected {weight_path.name} "
            f"({type(e).__name__}); falling back to weights_only=False. "
            "Only safe for trusted (e.g. DF40 official-release) "
            "checkpoints; do not point this loader at unverified files."
        )
        ckpt = torch.load(
            str(weight_path), map_location="cpu", weights_only=False,
        )
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    # Strip DDP prefix only -- no aggressive key rewrites
    cleaned = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(cleaned, strict=True)

    # Freeze all parameters
    model.eval()
    model.requires_grad_(False)

    model = model.to(device)

    info = {
        "expert_name": expert_name,
        "arch": arch_name,
        "detector_name": arch_config["detector_name"],
        "feature_dim": arch_config["feature_dim"],
        "resolution": arch_config["resolution"],
        "regime": expert_entry["regime"],
        "weight_path": str(weight_path),
    }
    logger.info(
        f"Loaded {expert_name}: {arch_name}, "
        f"feat_dim={arch_config['feature_dim']}, "
        f"resolution={arch_config['resolution']}"
    )
    return model, info


def load_pool(
    pool_name: str,
    experts_yaml: Optional[Path] = None,
    paths_yaml: Optional[Path] = None,
    device: str = "cpu",
) -> List[Tuple[nn.Module, dict]]:
    """Load all experts in a named pool.

    Args:
        pool_name: Pool name from experts.yaml (e.g., "clip4", "weak6").
        device: Target device.

    Returns:
        List of (model, info_dict) tuples, one per expert.
    """
    if experts_yaml is None:
        expert_names = get_pool_experts(pool_name)
    else:
        cfg = _get_experts_cfg(experts_yaml)
        if pool_name not in cfg["pools"]:
            raise ValueError(
                f"Unknown pool: {pool_name}. "
                f"Available: {list(cfg['pools'].keys())}"
            )
        expert_names = cfg["pools"][pool_name]
    results = []
    for name in expert_names:
        model, info = load_expert(
            name, experts_yaml=experts_yaml,
            paths_yaml=paths_yaml, device=device,
        )
        results.append((model, info))
    return results
