"""
config_loader.py
----------------
Loads the ICMDP master configuration from a YAML file.
Requires:  pip install pyyaml
"""
from pathlib import Path


def load_config(path: str = "config.yaml") -> dict:
    """
    Load and return the ICMDP configuration dict from *path*.

    All runtime parameters (states, actions, constraints, policies,
    probabilities, rewards) are read from this file so that no code change
    is required to modify behaviour.

    Raises FileNotFoundError if the config file does not exist.
    Raises yaml.YAMLError if the file is malformed.
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML is required: pip install pyyaml"
        ) from e

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.absolute()}\n"
            f"Copy config.yaml to the working directory or pass the correct path."
        )

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _validate(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Minimal structural validation — catches missing top-level sections early
# ---------------------------------------------------------------------------
_REQUIRED_TOP_LEVEL = [
    "state_space", "action_space", "intent_constraints",
    "policies", "probabilities", "rewards",
]

def _validate(cfg: dict) -> None:
    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in cfg]
    if missing:
        raise ValueError(
            f"config.yaml is missing required top-level sections: {missing}"
        )
