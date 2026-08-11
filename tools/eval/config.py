"""Ablation matrix and gate-threshold loading (doc 05 §2, §4).

The matrix is data, not code: each configuration is a named dict of
runtime-knob overrides applied on top of registry defaults (the mechanism
``tools/eval_prod.py`` and the sweeps already use). Phases add their
C-configurations by editing ``matrix.yaml``; the frozen C0 definition is never
edited afterwards.

``eval_thresholds.yaml`` holds the pre-registered gate thresholds (doc 05 §4):
fixed before any gated data is examined; edits require a dedicated commit with
written justification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MATRIX_PATH = Path(__file__).with_name("matrix.yaml")
THRESHOLDS_PATH = Path(__file__).with_name("eval_thresholds.yaml")


@dataclass(frozen=True, slots=True)
class MatrixConfig:
    config_id: str
    description: str
    available: bool  # False until the phase implementing the feature lands
    overrides: dict[str, Any]
    same_as: str | None = None  # CF may alias C0 while no V2 feature exists


def load_matrix(path: Path = MATRIX_PATH) -> dict[str, MatrixConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    configs: dict[str, MatrixConfig] = {}
    for entry in raw["configurations"]:
        config = MatrixConfig(
            config_id=entry["id"],
            description=entry["description"],
            available=bool(entry.get("available", True)),
            overrides=dict(entry.get("overrides") or {}),
            same_as=entry.get("same_as"),
        )
        configs[config.config_id] = config
    for config in configs.values():
        if config.same_as is not None and config.same_as not in configs:
            raise ValueError(f"{config.config_id}: same_as references unknown {config.same_as!r}")
    if "C0" not in configs:
        raise ValueError("matrix must define the frozen baseline C0")
    return configs


def resolve_overrides(config: MatrixConfig, configs: dict[str, MatrixConfig]) -> dict[str, Any]:
    if config.same_as is not None:
        base = resolve_overrides(configs[config.same_as], configs)
        base.update(config.overrides)
        return base
    return dict(config.overrides)


def load_thresholds(path: Path = THRESHOLDS_PATH) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))
