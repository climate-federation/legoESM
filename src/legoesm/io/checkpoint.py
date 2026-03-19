"""Zarr-based checkpoint IO for legoESM.

Provides save/load of atmospheric state to/from Zarr stores with Zstd
compression. Backward-compatible with existing .npz checkpoints
via auto-detection in ``load_checkpoint_auto``.

Compatible with zarr v2 (>= 2.18) and v3 (>= 3.0).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from legoesm.forcing.amip_config import (
    config_to_dict,
    config_from_dict,
    AMIPExperimentConfig,
)


def _auto_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Choose chunk shape: per-face for 4D (6,n,n,nlev), whole-array otherwise."""
    if len(shape) == 4 and shape[0] == 6:
        # One chunk per face: (1, n, n, nlev)
        return (1,) + shape[1:]
    return shape


def save_checkpoint_zarr(
    path,
    state,
    q_v,
    step: int,
    day: float,
    config,
    *,
    q_c=None,
    q_r=None,
    diag_accumulators: dict | None = None,
    compressor=None,
) -> None:
    """Save checkpoint to Zarr store.

    Layout::

        checkpoint.zarr/
            T/          -- temperature array
            u/          -- zonal wind
            v/          -- meridional wind
            p_s/        -- surface pressure
            phis/       -- surface geopotential
            q_v/        -- specific humidity
            q_c/        -- cloud water (optional)
            q_r/        -- rain water (optional)
            diag_*/     -- diagnostic accumulators
            .zattrs     -- metadata (step, day, config JSON)

    Parameters
    ----------
    path : Path-like
        Output Zarr store path (directory).
    state : HydrostaticState
    q_v : array-like
    step : int
    day : float
    config : AMIPExperimentConfig
    q_c, q_r : array-like, optional
    diag_accumulators : dict, optional
    compressor : optional
        Zarr v3 codec. Default: Zstd(level=3).
    """
    import zarr

    path = Path(path)

    # Zarr v2/v3 compatible compressor
    if compressor is None:
        try:
            # zarr v3 API
            from zarr.codecs import ZstdCodec
            compressor = ZstdCodec(level=3)
            _use_v3 = True
        except ImportError:
            # zarr v2 API
            from numcodecs import Zstd
            compressor = Zstd(level=3)
            _use_v3 = False
    else:
        _use_v3 = hasattr(compressor, '__class__') and 'Codec' in type(compressor).__name__

    root = zarr.open_group(str(path), mode="w")

    # State arrays
    arrays = {
        "T": np.asarray(state.T.data),
        "u": np.asarray(state.u.data),
        "v": np.asarray(state.v.data),
        "p_s": np.asarray(state.p_s.data),
        "phis": np.asarray(state.phis.data),
        "q_v": np.asarray(q_v),
    }
    if q_c is not None:
        arrays["q_c"] = np.asarray(q_c)
    if q_r is not None:
        arrays["q_r"] = np.asarray(q_r)

    if diag_accumulators:
        for k, v in diag_accumulators.items():
            arrays[f"diag_{k}"] = np.asarray(v)

    for name, arr in arrays.items():
        kwargs = dict(
            name=name,
            data=arr,
            chunks=_auto_chunks(arr.shape),
        )
        if _use_v3:
            kwargs["compressors"] = compressor
        else:
            kwargs["compressor"] = compressor
        root.create_dataset(**kwargs) if not _use_v3 else root.create_array(**kwargs)

    # Metadata as root attributes
    root.attrs["step"] = int(step)
    root.attrs["day"] = float(day)
    root.attrs["config_json"] = json.dumps(config_to_dict(config))

    zarr.consolidate_metadata(root.store)


def load_checkpoint_zarr(
    path,
    grid,
    sigma,
    *,
    lazy: bool = False,
):
    """Load checkpoint from Zarr store.

    Parameters
    ----------
    path : Path-like
    grid : grid object
    sigma : vertical coordinate
    lazy : bool
        If True, return zarr arrays without materializing to numpy/jax.
        Useful for inspection.

    Returns
    -------
    tuple
        ``(state, q_v, step, day, config, diag_accumulators, q_c, q_r)``
        Same 8-tuple as ``load_checkpoint`` for drop-in compatibility.
    """
    import zarr
    import jax.numpy as jnp
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState

    path = Path(path)

    # Try consolidated metadata first, fall back to regular open
    try:
        root = zarr.open_consolidated(str(path), mode="r")
    except Exception:
        root = zarr.open_group(str(path), mode="r")

    # v2/v3 compatible key iteration
    def _array_keys(group):
        if hasattr(group, 'array_keys'):
            return list(group.array_keys())
        return [k for k in group.keys() if hasattr(group[k], 'shape')]

    array_names = _array_keys(root)

    if lazy:
        arrays = {k: root[k] for k in array_names}
        step = int(root.attrs["step"])
        day = float(root.attrs["day"])
        config_str = root.attrs.get("config_json", "{}")
        config = config_from_dict(json.loads(config_str))
        diag = {k[5:]: root[k] for k in array_names if k.startswith("diag_")}
        q_c = root["q_c"] if "q_c" in root else None
        q_r = root["q_r"] if "q_r" in root else None
        return arrays, step, day, config, diag, q_c, q_r

    T = jnp.array(root["T"][:])
    u = jnp.array(root["u"][:])
    v = jnp.array(root["v"][:])
    p_s = jnp.array(root["p_s"][:])
    phis = jnp.array(root["phis"][:])
    q_v = jnp.array(root["q_v"][:])

    step = int(root.attrs["step"])
    day = float(root.attrs["day"])

    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")

    state = HydrostaticState(
        u=Field(data=u, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=p_s, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=phis, name="phis", dims=dims_2d, units="m^2/s^2"),
    )

    config_str = root.attrs.get("config_json", "{}")
    config = config_from_dict(json.loads(config_str))

    diag_accumulators = {}
    for k in array_names:
        if k.startswith("diag_"):
            arr = root[k]
            diag_accumulators[k[5:]] = arr[()] if arr.ndim == 0 else arr[:]

    q_c = jnp.array(root["q_c"][:]) if "q_c" in root else None
    q_r = jnp.array(root["q_r"][:]) if "q_r" in root else None

    return state, q_v, step, day, config, diag_accumulators, q_c, q_r


def load_checkpoint_auto(path, grid, sigma):
    """Auto-detect checkpoint format (.npz or .zarr) and load.

    Parameters
    ----------
    path : Path-like
    grid : grid object
    sigma : vertical coordinate

    Returns
    -------
    tuple
        Same 8-tuple as ``load_checkpoint``.
    """
    from legoesm.forcing.amip_config import load_checkpoint as load_npz

    path = Path(path)
    # Zarr stores are directories; detect by presence of zarr metadata
    if path.is_dir():
        return load_checkpoint_zarr(path, grid, sigma)
    else:
        return load_npz(path, grid, sigma)
