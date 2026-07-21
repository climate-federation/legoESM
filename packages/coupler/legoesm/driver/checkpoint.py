"""Zarr-based checkpoint IO for legoESM.

Provides save/load of atmospheric state to/from Zarr stores with Zstd
compression. Backward-compatible with existing .npz checkpoints
via auto-detection in ``load_checkpoint_auto``.

Compatible with zarr v2 (>= 2.18) and v3 (>= 3.0).
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np


def wallclock_exhausted(elapsed_s: float, max_s: float, buffer_s: float) -> bool:
    """True if the run should checkpoint and exit to fit the wallclock budget.

    ``max_s <= 0`` disables the check.  Otherwise fire once the elapsed time is
    within ``buffer_s`` of the budget, leaving time to write the checkpoint
    before the scheduler kills the job (so a dependency chain can resume).
    Shared by the OMIP/LMIP run drivers (previously copy-pasted).
    """
    return max_s > 0.0 and elapsed_s >= (max_s - buffer_s)


# Lazy imports to avoid circular dependency:
#   io.checkpoint → driver.config → driver.__init__ → model_driver → io.checkpoint
_experiment_config_from_dict = None


def _get_experiment_config_from_dict():
    global _experiment_config_from_dict
    if _experiment_config_from_dict is None:
        from legoesm.driver.config import experiment_config_from_dict
        _experiment_config_from_dict = experiment_config_from_dict
    return _experiment_config_from_dict


def _config_from_dict_auto(d: dict):
    """Detect config format and deserialize.

    ExperimentConfig JSON has a ``"grid"`` key with nested sub-configs;
    legacy AMIPExperimentConfig JSON has flat fields (``"resolution"``
    at the top level).

    Returns an ``ExperimentConfig`` in both cases (upconverting legacy
    AMIP format via ``ExperimentConfig.from_amip_config``).
    """
    if "grid" in d and isinstance(d["grid"], dict):
        # New ExperimentConfig format
        ecfd = _get_experiment_config_from_dict()
        return ecfd(d)
    else:
        # Legacy AMIPExperimentConfig format — load then upconvert.
        # Deferred driver.config import: a top-level one would reintroduce the
        # io.checkpoint -> driver.config -> driver.__init__ -> model_driver cycle.
        from legoesm.driver.config import config_from_dict
        amip_cfg = config_from_dict(d)
        from legoesm.driver.config import ExperimentConfig
        return ExperimentConfig.from_amip_config(amip_cfg)


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
    config : ExperimentConfig or AMIPExperimentConfig
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

    # Pull all state arrays in a single ``jax.device_get`` call so the
    # JAX runtime can pipeline the device→host transfers in parallel.
    # The previous per-leaf ``np.asarray(...)`` chain forced the
    # transfers to serialize, blocking the GPU pipeline at every
    # checkpoint cadence.
    _names = ["T", "u", "v", "p_s", "phis", "q_v"]
    _values = [
        state.T.data, state.u.data, state.v.data,
        state.p_s.data, state.phis.data, q_v,
    ]
    if q_c is not None:
        _names.append("q_c"); _values.append(q_c)
    if q_r is not None:
        _names.append("q_r"); _values.append(q_r)
    if diag_accumulators:
        for k, v in diag_accumulators.items():
            _names.append(f"diag_{k}"); _values.append(v)
    _host = jax.device_get(_values)
    arrays = {name: np.asarray(val) for name, val in zip(_names, _host)}

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
    from legoesm.driver.config import config_to_dict
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
        config = _config_from_dict_auto(json.loads(config_str))
        diag = {k[5:]: root[k] for k in array_names if k.startswith("diag_")}
        q_c = root["q_c"] if "q_c" in root else None
        q_r = root["q_r"] if "q_r" in root else None
        return arrays, step, day, config, diag, q_c, q_r

    # Load arrays and cast to the active storage dtype so that restarts
    # are consistent with the current precision policy (e.g. a checkpoint
    # saved in fp32 can be loaded into an fp64 session without dtype drift).
    from legoesm.core.precision import get_policy
    _storage_dtype = get_policy().storage

    T = jnp.array(root["T"][:], dtype=_storage_dtype)
    u = jnp.array(root["u"][:], dtype=_storage_dtype)
    v = jnp.array(root["v"][:], dtype=_storage_dtype)
    p_s = jnp.array(root["p_s"][:], dtype=_storage_dtype)
    phis = jnp.array(root["phis"][:], dtype=_storage_dtype)
    q_v = jnp.array(root["q_v"][:], dtype=_storage_dtype)

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
    config = _config_from_dict_auto(json.loads(config_str))

    diag_accumulators = {}
    for k in array_names:
        if k.startswith("diag_"):
            arr = root[k]
            diag_accumulators[k[5:]] = arr[()] if arr.ndim == 0 else arr[:]

    q_c = jnp.array(root["q_c"][:], dtype=_storage_dtype) if "q_c" in root else None
    q_r = jnp.array(root["q_r"][:], dtype=_storage_dtype) if "q_r" in root else None

    return state, q_v, step, day, config, diag_accumulators, q_c, q_r


def _upconvert_config(config):
    """Ensure *config* is an ExperimentConfig.

    If it is already an ExperimentConfig, return as-is.
    If it is an AMIPExperimentConfig, upconvert via
    ``ExperimentConfig.from_amip_config``.
    """
    from legoesm.driver.config import ExperimentConfig
    if isinstance(config, ExperimentConfig):
        return config
    return ExperimentConfig.from_amip_config(config)


def load_checkpoint_auto(path, grid, sigma):
    """Auto-detect checkpoint format (.npz or .zarr) and load.

    Returns an ``ExperimentConfig`` regardless of whether the
    checkpoint was saved with the legacy AMIP format or the new
    ExperimentConfig format.

    Parameters
    ----------
    path : Path-like
    grid : grid object
    sigma : vertical coordinate

    Returns
    -------
    tuple
        9-tuple: (state, q_v, step, day, config, diag_accumulators,
        q_c, q_r, carry_aux).  *config* is always an
        ``ExperimentConfig``.
    """
    from legoesm.forcing.amip_config import load_checkpoint as load_npz

    path = Path(path)
    # Zarr stores are directories; detect by presence of zarr metadata
    if path.is_dir():
        result = load_checkpoint_zarr(path, grid, sigma)
        # Zarr path returns 8-tuple; pad with empty carry_aux
        if len(result) == 8:
            result = (*result, {})
    else:
        result = load_npz(path, grid, sigma)

    # Upconvert config (element [4]) to ExperimentConfig if needed
    result_list = list(result)
    result_list[4] = _upconvert_config(result_list[4])
    return tuple(result_list)
