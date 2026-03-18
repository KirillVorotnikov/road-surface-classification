from omegaconf import OmegaConf, DictConfig
from pathlib import Path


def load_config(config_path: str) -> DictConfig:
    """Load config from YAML."""
    cfg = OmegaConf.load(config_path)

    if "defaults" in cfg:
        for default in cfg.defaults:
            base_path = _resolve_default_path(config_path, default)
            if base_path and base_path.exists():
                base = OmegaConf.load(str(base_path))
                cfg = OmegaConf.merge(base, cfg)

    return cfg


def _resolve_default_path(config_path: str, default: str) -> Path | None:
    """Resolve the path to the base config."""
    config_dir = Path(config_path).parent
    if default.startswith("/"):
        configs_root = config_dir
        while configs_root.name != "configs" and configs_root != configs_root.parent:
            configs_root = configs_root.parent
        return configs_root / f"{default.lstrip('/')}.yaml"
    return config_dir / f"{default}.yaml"