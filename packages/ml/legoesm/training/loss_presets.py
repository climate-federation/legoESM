"""Named loss presets for the unified WB+AIMIP training driver (D5).

A preset is a YAML mapping of ``LossConfig`` field overrides kept in
``config/wb/loss_presets/<name>.yaml`` — numbers live ONCE there, never
copy-pasted between suite YAMLs. A suite selects one via ``loss_preset:
<name>`` (or an explicit path); its own ``loss:`` block wins key-by-key
over the preset.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Repo layout: packages/ml/legoesm/training/loss_presets.py -> repo root is
# four levels up. The dev/test workflow runs from the checkout with
# PYTHONPATH=packages/* (editable-install doctrine), so this resolution is
# the supported path; an explicit path argument bypasses it entirely.
_PRESET_DIR = Path(__file__).resolve().parents[4] / "config" / "wb" / "loss_presets"


def available_presets() -> list[str]:
    """Names of the bundled presets (sorted, without the .yaml suffix)."""
    if not _PRESET_DIR.is_dir():
        return []
    return sorted(p.stem for p in _PRESET_DIR.glob("*.yaml"))


def load_loss_preset(name_or_path: str) -> dict:
    """Load a loss preset by bundled name or explicit YAML path.

    Returns the flat dict of ``LossConfig`` field overrides. Unknown preset
    name -> ``ValueError`` listing the available presets.
    """
    cand = Path(name_or_path)
    if cand.suffix in (".yaml", ".yml"):
        if not cand.is_file():
            raise ValueError(f"Loss preset file not found: {cand}")
        path = cand
    else:
        path = _PRESET_DIR / f"{name_or_path}.yaml"
        if not path.is_file():
            raise ValueError(
                f"Unknown loss preset {name_or_path!r}; available: "
                f"{available_presets()}."
            )
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    loss = data.get("loss", data)
    if not isinstance(loss, dict):
        raise ValueError(
            f"Loss preset {path} must be a mapping of LossConfig fields "
            f"(optionally under a top-level 'loss:' key); got {type(loss)}."
        )
    return dict(loss)


def merge_loss_preset(preset: dict, suite_loss: dict | None) -> dict:
    """Merge a preset with a suite's ``loss:`` block — suite keys win."""
    merged = dict(preset)
    merged.update(suite_loss or {})
    return merged
