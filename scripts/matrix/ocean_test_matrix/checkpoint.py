"""Simple ocean state checkpoint: save/load to NPZ."""

from pathlib import Path
import json
import numpy as np
import jax.numpy as jnp
from legoesm.core.field import Field


def save_ocean_checkpoint(path, state, step: int, day: float,
                          config_dict: dict | None = None):
    """Save ocean state to NPZ + metadata JSON.

    Works with any NamedTuple-of-Field ocean state (LatLonCGrid, MPAS, etc.).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {}
    field_meta = {}
    for name in state._fields:
        field = getattr(state, name)
        if hasattr(field, 'data'):
            arrays[name] = np.array(field.data)
            field_meta[name] = {
                "dims": field.dims if hasattr(field, 'dims') else None,
                "units": field.units if hasattr(field, 'units') else None,
                "shape": list(arrays[name].shape),
                "dtype": str(arrays[name].dtype),
            }

    np.savez_compressed(path, **arrays)

    meta = {
        "step": step,
        "day": day,
        "fields": field_meta,
    }
    if config_dict:
        meta["config"] = config_dict
    meta_path = path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_ocean_checkpoint(path, state_template):
    """Load ocean state from NPZ, using state_template for Field metadata.

    Returns (state, step, day).
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".npz")

    data = np.load(path)
    meta_path = path.with_suffix(".meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    replacements = {}
    for name in state_template._fields:
        if name in data:
            template_field = getattr(state_template, name)
            replacements[name] = Field(
                jnp.array(data[name]),
                name=name,
                dims=template_field.dims if hasattr(template_field, 'dims') else None,
                units=template_field.units if hasattr(template_field, 'units') else None,
            )

    state = state_template._replace(**replacements)
    return state, meta["step"], meta["day"]
