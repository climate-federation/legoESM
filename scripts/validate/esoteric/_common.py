"""Shared loaders for the esoteric-climate checkpoint diagnostics (MPAS lane)."""
import json
from pathlib import Path

import numpy as np

BANDS = [(-90, -60), (-60, -30), (-30, -10), (-10, 10), (10, 30), (30, 60), (60, 90)]


def find_key(d, key):
    """First value stored under ``key`` anywhere in a nested dict, else None."""
    if key in d:
        return d[key]
    for v in d.values():
        if isinstance(v, dict):
            hit = find_key(v, key)
            if hit is not None:
                return hit
    return None


def load_manifest(out_dir):
    return json.loads((Path(out_dir) / "coupled" / "run_manifest.json").read_text())


def apply_run_precision(manifest):
    """Activate the run's precision policy so rebuilt states match its dtypes."""
    from legoesm.runtime.precision import apply_precision

    mode = find_key(manifest, "precision")
    if mode is None:
        raise KeyError("run manifest records no 'precision'")
    apply_precision(mode)
    return mode


def band_means(field, lat_deg, weight, bands=BANDS):
    """Weighted mean of ``field`` (nCells,) or (nCells, nlev) per latitude band."""
    field, weight = np.asarray(field, np.float64), np.asarray(weight, np.float64)
    out = []
    for a, b in bands:
        m = (lat_deg >= a) & (lat_deg < b) & (weight > 0)
        if not m.any():
            raise ValueError(f"band {a}..{b} has no cells")
        w = weight[m]
        out.append(np.tensordot(w, field[m], axes=(0, 0)) / w.sum())
    return np.array(out)
