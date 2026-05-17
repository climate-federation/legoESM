"""Restart harness for the legoESM ocean dycores.

Saves the full ocean prognostic state to disk as a numpy ``.npz``
archive and loads it back into an empty state container of the
matching grid. Round-trip preserves every prognostic field to
bit-exact precision (``np.array_equal``) so a model started from a
restart steps to bit-identical results vs. an uninterrupted run.

Currently supports the three production grids:

* lat-lon C-grid (``LatLonCGridOceanState``)
* MPAS Voronoi (``MPASOceanState`` -- accessed via duck-typing on
  the ``state.u.dims`` shape)
* cubed-sphere (``OceanState``)

The serialised payload is a dict-of-numpy-arrays keyed by field
name; auxiliary metadata (model time in seconds, step index, source
sha) is stored under reserved underscore keys to keep field names
clean.

Usage::

    from legoesm.ocean.restart import save_restart, load_restart

    save_restart(state, "results/ocean/restart_t0.npz",
                 time_s=86400.0 * 30, step=10_000)
    state_loaded = load_restart("results/ocean/restart_t0.npz", state)
    # state_loaded is bit-identical to state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from legoesm.core.field import Field


_RESERVED_KEYS: tuple[str, ...] = ("_time_s", "_step", "_sha")


def _iter_state_fields(state) -> list[str]:
    """Return the prognostic field names of the state NamedTuple."""
    # NamedTuple subclasses expose ``_fields``; fall back to dir() filtered.
    fields = getattr(state, "_fields", None)
    if fields is not None:
        return list(fields)
    return [k for k in dir(state) if isinstance(getattr(state, k), Field)]


def save_restart(state, path: str | Path, *,
                 time_s: float | None = None,
                 step: int | None = None,
                 sha: str | None = None) -> Path:
    """Serialise every ``Field`` of ``state`` to a ``.npz`` archive.

    Returns the resolved ``Path`` of the written file.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for name in _iter_state_fields(state):
        attr = getattr(state, name)
        if attr is None:
            continue
        if isinstance(attr, Field):
            payload[name] = np.asarray(attr.data)
    if time_s is not None:
        payload["_time_s"] = np.asarray(float(time_s))
    if step is not None:
        payload["_step"] = np.asarray(int(step))
    if sha is not None:
        payload["_sha"] = np.asarray(str(sha))
    np.savez(out_path, **payload)
    return out_path


def load_restart(path: str | Path, template_state) -> tuple:
    """Read a ``.npz`` archive and return a state with the loaded data.

    ``template_state`` provides the ``Field`` ``dims`` / ``units`` /
    ``name`` metadata; the returned state has the same NamedTuple type.
    """
    in_path = Path(path)
    with np.load(in_path, allow_pickle=False) as f:
        loaded = {k: f[k] for k in f.files}

    replace_kw: dict[str, Field] = {}
    for name in _iter_state_fields(template_state):
        if name not in loaded:
            continue
        ref_field = getattr(template_state, name)
        if not isinstance(ref_field, Field):
            continue
        replace_kw[name] = Field(
            data=jnp.asarray(loaded[name]),
            name=ref_field.name,
            dims=ref_field.dims,
            units=ref_field.units,
        )
    return template_state._replace(**replace_kw)


def restart_metadata(path: str | Path) -> dict[str, Any]:
    """Return ``{time_s, step, sha}`` (any present) from the archive."""
    in_path = Path(path)
    out: dict[str, Any] = {}
    with np.load(in_path, allow_pickle=False) as f:
        for key in _RESERVED_KEYS:
            if key in f.files:
                clean = key.lstrip("_")
                val = f[key]
                # Decode scalars + bytestrings.
                if val.dtype.kind in ("U", "S"):
                    out[clean] = str(val)
                else:
                    out[clean] = val.item()
    return out


__all__ = ["save_restart", "load_restart", "restart_metadata"]
