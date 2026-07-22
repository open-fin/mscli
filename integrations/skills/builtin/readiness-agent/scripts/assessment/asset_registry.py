import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

Asset = Dict[str, object]


def asset_candidate(
    *,
    kind: str,
    source_type: str,
    status: str,
    required: bool,
    locator: Optional[Dict[str, object]] = None,
    selection_source: str = "detected",
    confidence: str = "medium",
    label: Optional[str] = None,
) -> Asset:
    payload: Asset = {
        "kind": kind,
        "source_type": source_type,
        "status": status,
        "required": required,
        "locator": locator or {},
        "selection_source": selection_source,
        "confidence": confidence,
        "label": label or kind,
    }
    if locator and "path" in locator:
        payload["path"] = str(locator["path"])
    if source_type == "local_path":
        payload["validation"] = "path-existence-only"
    return payload


def local_path_asset(kind: str, path: Path, *, required: bool, selection_source: str, confidence: str = "high") -> Asset:
    status = "present" if path.exists() else "missing"
    return asset_candidate(
        kind=kind,
        source_type="local_path",
        status=status,
        required=required,
        locator={"path": str(path)},
        selection_source=selection_source,
        confidence=confidence,
        label=f"local {kind} path: {path}",
    )


def remote_asset(kind: str, repo_id: str, *, required: bool, selection_source: str) -> Asset:
    return asset_candidate(
        kind=kind,
        source_type="hf_hub",
        status="remote_declared",
        required=required,
        locator={"repo_id": repo_id},
        selection_source=selection_source,
        confidence="high",
        label=f"Hugging Face {kind} repo: {repo_id}",
    )


def unknown_asset(kind: str, *, required: bool) -> Asset:
    return asset_candidate(
        kind=kind,
        source_type="unknown",
        status="unknown",
        required=required,
        selection_source="not_found",
        confidence="low",
        label=f"unknown {kind}",
    )


COMMON_CONFIG_NAMES = ("config.yaml", "config.yml", "config.json", "config.toml")
COMMON_MODEL_DIRS = ("model", "models")
COMMON_DATASET_DIRS = ("dataset", "data")
COMMON_CHECKPOINT_DIRS = ("checkpoints", "ckpt")
HF_MODEL_CACHE_VARS = ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE")
HF_DATASET_CACHE_VARS = ("HF_DATASETS_CACHE", "HF_HOME")


def resolve_optional_path(raw_path: Optional[str], root: Path) -> Optional[Path]:
    if not raw_path:
        return None
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def first_existing(root: Path, names: Iterable[str]) -> Optional[Path]:
    for name in names:
        path = root / name
        if path.exists():
            return path.resolve()
    return None


def hf_cache_asset(kind: str, env_names: Iterable[str], *, required: bool) -> Optional[Asset]:
    for env_name in env_names:
        raw_path = os.environ.get(env_name)
        if not raw_path:
            continue
        path = Path(raw_path).expanduser()
        if path.exists():
            return {
                "kind": kind,
                "source_type": "hf_cache",
                "status": "present",
                "required": required,
                "locator": {"path": str(path), "env": env_name},
                "path": str(path),
                "selection_source": "environment",
                "confidence": "medium",
                "label": f"Hugging Face cache from {env_name}: {path}",
                "validation": "path-existence-only",
            }
    return None


def inline_config_asset(root: Path, *, required: bool) -> Optional[Asset]:
    config_path = first_existing(root, COMMON_CONFIG_NAMES)
    if config_path:
        return local_path_asset("config", config_path, required=required, selection_source="workspace_common_file")
    return None


def detect_asset_registry(args, root: Path, target_value: Optional[str]) -> Dict[str, Asset]:
    training = target_value == "training"
    assets: Dict[str, Asset] = {}

    config_path = resolve_optional_path(getattr(args, "config_path", None), root)
    assets["config"] = (
        local_path_asset("config", config_path, required=False, selection_source="explicit_input")
        if config_path
        else inline_config_asset(root, required=False) or unknown_asset("config", required=False)
    )

    model_path = resolve_optional_path(getattr(args, "model_path", None), root)
    model_hub_id = getattr(args, "model_hub_id", None)
    if model_path:
        assets["model"] = local_path_asset("model", model_path, required=True, selection_source="explicit_input")
    elif model_hub_id:
        assets["model"] = remote_asset("model", str(model_hub_id), required=True, selection_source="explicit_input")
    else:
        common_model = first_existing(root, COMMON_MODEL_DIRS)
        assets["model"] = (
            local_path_asset("model", common_model, required=True, selection_source="workspace_common_dir")
            if common_model
            else hf_cache_asset("model", HF_MODEL_CACHE_VARS, required=True) or unknown_asset("model", required=True)
        )

    assets["tokenizer"] = unknown_asset("tokenizer", required=False)

    dataset_path = resolve_optional_path(getattr(args, "dataset_path", None), root)
    dataset_hub_id = getattr(args, "dataset_hub_id", None)
    if dataset_path:
        assets["dataset"] = local_path_asset("dataset", dataset_path, required=training, selection_source="explicit_input")
    elif dataset_hub_id:
        assets["dataset"] = remote_asset("dataset", str(dataset_hub_id), required=training, selection_source="explicit_input")
    else:
        common_dataset = first_existing(root, COMMON_DATASET_DIRS)
        assets["dataset"] = (
            local_path_asset("dataset", common_dataset, required=training, selection_source="workspace_common_dir")
            if common_dataset
            else hf_cache_asset("dataset", HF_DATASET_CACHE_VARS, required=training) or unknown_asset("dataset", required=training)
        )

    checkpoint_path = resolve_optional_path(getattr(args, "checkpoint_path", None), root)
    common_checkpoint = first_existing(root, COMMON_CHECKPOINT_DIRS)
    if checkpoint_path:
        assets["checkpoint"] = local_path_asset("checkpoint", checkpoint_path, required=False, selection_source="explicit_input")
    elif common_checkpoint:
        assets["checkpoint"] = local_path_asset("checkpoint", common_checkpoint, required=False, selection_source="workspace_common_dir")
    else:
        assets["checkpoint"] = unknown_asset("checkpoint", required=False)

    return assets


def legacy_asset_view(asset: Asset) -> Dict[str, object]:
    view: Dict[str, object] = {
        "status": asset.get("status"),
        "required": asset.get("required", False),
    }
    if asset.get("path"):
        view["path"] = asset["path"]
    if asset.get("validation"):
        view["validation"] = asset["validation"]
    if asset.get("source_type") != "unknown":
        view["source_type"] = asset.get("source_type")
    if asset.get("locator"):
        view["locator"] = asset.get("locator")
    if asset.get("selection_source") and asset.get("selection_source") != "not_found":
        view["selection_source"] = asset.get("selection_source")
    return view


def workspace_lock_asset_view(assets: Dict[str, Asset]) -> Dict[str, Dict[str, object]]:
    return {kind: legacy_asset_view(asset) for kind, asset in assets.items()}
