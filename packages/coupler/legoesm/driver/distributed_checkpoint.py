"""Distributed checkpointing for multi-rank simulations.

Each rank saves its local partition independently, avoiding the rank-0
bottleneck.  Metadata (timestep, config, etc.) is saved by rank 0 only.

Layout on disk
--------------
::

    checkpoint_dir/
        metadata.json          # rank 0 only: step, day, n_ranks, config
        rank_000.npz           # rank 0 local arrays
        rank_001.npz           # rank 1 local arrays
        ...
        rank_NNN.npz
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import numpy as np

from legoesm.driver.config import config_to_dict, config_from_dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rank_filename(rank: int) -> str:
    """Return the per-rank npz filename, e.g. ``rank_003.npz``."""
    return f"rank_{rank:03d}.npz"


def _state_to_arrays(state) -> dict[str, np.ndarray]:
    """Extract numpy arrays from a HydrostaticState.

    Single batched ``jax.device_get`` instead of one ``np.asarray``
    per leaf so transfers can overlap across the 5 fields.
    """
    names = ["T", "u", "v", "p_s", "phis"]
    values = [
        state.T.data, state.u.data, state.v.data,
        state.p_s.data, state.phis.data,
    ]
    host = jax.device_get(values)
    return {n: np.asarray(v) for n, v in zip(names, host)}


# ---------------------------------------------------------------------------
# Per-rank distributed checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint_distributed(
    path,
    state,
    rank: int,
    n_ranks: int,
    step: int,
    day: float,
    config=None,
    q_v=None,
    q_c=None,
    q_r=None,
    diag_accumulators: dict | None = None,
) -> None:
    """Save checkpoint with per-rank partitioning.

    Creates::

        path/metadata.json   (rank 0 only)
        path/rank_000.npz
        path/rank_001.npz
        ...

    Parameters
    ----------
    path : Path-like
        Directory in which checkpoint files are written.  Created if it
        does not exist.
    state : HydrostaticState
        Atmospheric state (may contain only the local partition).
    rank : int
        MPI rank of the calling process.
    n_ranks : int
        Total number of MPI ranks.
    step : int
        Current time-step number.
    day : float
        Current simulation day.
    config : AMIPExperimentConfig, optional
        Experiment configuration (serialized to JSON by rank 0).
    q_v, q_c, q_r : array-like, optional
        Moisture / hydrometeor fields.
    diag_accumulators : dict, optional
        Diagnostic accumulator arrays to preserve across restart.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # -- Per-rank arrays --
    # Batch the moisture / diag arrays into one ``jax.device_get`` so
    # the GPU runtime can overlap transfers with the state arrays
    # already pulled inside ``_state_to_arrays``.  (We could pass them
    # in via ``_state_to_arrays``; keeping the helper minimal preserves
    # API for non-moisture state tests.)
    extra_names: list[str] = []
    extra_values: list = []
    if q_v is not None:
        extra_names.append("q_v"); extra_values.append(q_v)
    if q_c is not None:
        extra_names.append("q_c"); extra_values.append(q_c)
    if q_r is not None:
        extra_names.append("q_r"); extra_values.append(q_r)
    if diag_accumulators:
        for k, v in diag_accumulators.items():
            extra_names.append(f"diag_{k}"); extra_values.append(v)

    arrays: dict[str, np.ndarray] = _state_to_arrays(state)
    if extra_values:
        host_extra = jax.device_get(extra_values)
        for n, v in zip(extra_names, host_extra):
            arrays[n] = np.asarray(v)

    np.savez(str(path / _rank_filename(rank)), **arrays)

    # -- Metadata (rank 0 only) --
    if rank == 0:
        meta: dict[str, Any] = {
            "step": step,
            "day": day,
            "n_ranks": n_ranks,
        }
        if config is not None:
            meta["config"] = config_to_dict(config)

        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)


def load_checkpoint_distributed(
    path,
    rank: int,
    n_ranks: int,
):
    """Load checkpoint for a specific rank.

    Parameters
    ----------
    path : Path-like
        Directory containing the distributed checkpoint.
    rank : int
        MPI rank of the calling process.
    n_ranks : int
        Expected number of MPI ranks.

    Returns
    -------
    tuple
        ``(arrays, step, day, config, diag_accumulators)``

        * *arrays* — ``dict[str, np.ndarray]`` of the per-rank fields
          (``T``, ``u``, ``v``, ``p_s``, ``phis``, ``q_v``, and
          optionally ``q_c``, ``q_r``).
        * *step* — ``int``
        * *day* — ``float``
        * *config* — :class:`AMIPExperimentConfig` or ``None``
        * *diag_accumulators* — ``dict[str, np.ndarray]``

    Raises
    ------
    FileNotFoundError
        If the checkpoint directory or per-rank file is missing.
    ValueError
        If *n_ranks* does not match the value recorded at save time.
    """
    path = Path(path)

    # -- Load metadata --
    meta_path = path / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No metadata.json found in checkpoint directory {path}"
        )
    with open(meta_path) as f:
        meta = json.load(f)

    saved_n_ranks = meta["n_ranks"]
    if saved_n_ranks != n_ranks:
        raise ValueError(
            f"Checkpoint was saved with {saved_n_ranks} ranks, but "
            f"load_checkpoint_distributed was called with n_ranks={n_ranks}."
        )

    step = int(meta["step"])
    day = float(meta["day"])

    config = None
    if "config" in meta:
        config = config_from_dict(meta["config"])

    # -- Load per-rank arrays --
    rank_path = path / _rank_filename(rank)
    if not rank_path.exists():
        raise FileNotFoundError(
            f"Per-rank checkpoint file not found: {rank_path}"
        )
    data = np.load(str(rank_path), allow_pickle=True)

    arrays: dict[str, np.ndarray] = {}
    diag_accumulators: dict[str, np.ndarray] = {}
    for key in data.files:
        if key.startswith("diag_"):
            diag_accumulators[key[5:]] = data[key]
        else:
            arrays[key] = data[key]

    return arrays, step, day, config, diag_accumulators
