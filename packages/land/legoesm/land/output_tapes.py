"""CLM-style history tapes for offline land runs.

Each *tape* aggregates a chosen set of variables at a chosen frequency and
writes to its own NetCDF (``<output>.<tape_name>.nc``).  Tapes are declared in
YAML — see ``configs/output/lmip_biophys_default.yaml`` for the shipped
default (monthly ``h0`` time-mean fluxes + monthly ``h_state`` instantaneous
soil snapshot).

Design
------
* Slot indices per step are pre-computed on the host, so the ``lax.scan`` body
  never touches Python time arithmetic.
* Each tape carries a per-slot accumulator (``count`` scalar + per-variable
  ``(n_slots, ncol)`` array) that lives inside the scan carry.
* ``average="mean"`` -> scatter-add; divide by count at the end.
* ``average="inst"`` -> scatter-SET; the last step in a slot wins (= end-of-
  interval snapshot, matching CLM's ``I`` averaging type).

Memory per tape at 2 deg latlon, monthly, 5 years:
``60 slots x 16200 col x 8 B x n_vars`` -- a few MB total.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np


class TapeSpec(NamedTuple):
    """Static configuration for one output tape."""
    name: str                          # short label; produces "<output>.<name>.nc"
    freq: str                          # "step" | "hourly" | "daily" | "monthly" | "annual"
    average: str                       # "mean" | "inst"
    vars: tuple[str, ...]              # ordered list of variable names


_VALID_FREQ = ("step", "hourly", "daily", "monthly", "annual")
_VALID_AVG = ("mean", "inst")

# Noleap 365-day calendar; standard start-day-of-year per month (zero-indexed).
_MONTH_START_DOY = np.array(
    [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334],
    dtype=np.int32,
)
_DAYS_PER_YEAR = 365
_SEC_PER_HOUR = 3600.0
_SEC_PER_DAY = 86400.0


def default_config_path() -> Path:
    """Path to the repo's shipped default output YAML."""
    # packages/land/legoesm/land/output_tapes.py
    #   parents[0] = land/
    #   parents[1] = legoesm/
    #   parents[2] = land/
    #   parents[3] = packages/
    #   parents[4] = <repo root>
    return (Path(__file__).resolve().parents[4]
            / "configs" / "output" / "lmip_biophys_default.yaml")


def load_output_config(path: str | Path | None) -> list[TapeSpec]:
    """Load YAML → list of :class:`TapeSpec`.  ``None`` loads the shipped default."""
    import yaml
    p = Path(path) if path else default_config_path()
    with open(p) as f:
        data = yaml.safe_load(f)
    tapes: list[TapeSpec] = []
    for entry in data.get("tapes", []):
        spec = TapeSpec(
            name=str(entry["name"]),
            freq=str(entry["freq"]),
            average=str(entry.get("average", "mean")),
            vars=tuple(str(v) for v in entry["vars"]),
        )
        if spec.freq not in _VALID_FREQ:
            raise ValueError(f"tape {spec.name!r}: freq={spec.freq!r} not in {_VALID_FREQ}")
        if spec.average not in _VALID_AVG:
            raise ValueError(f"tape {spec.name!r}: average={spec.average!r} not in {_VALID_AVG}")
        if not spec.vars:
            raise ValueError(f"tape {spec.name!r}: empty vars list")
        tapes.append(spec)
    return tapes


def build_slot_indices(
    model_times_s: np.ndarray, freq: str,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Per-step slot index + slot count + per-slot midpoint time (for netCDF).

    Parameters
    ----------
    model_times_s : (n_steps,) array
        Model step times, seconds since year_start Jan 1 (noleap calendar).
    freq : str
        One of :data:`_VALID_FREQ`.

    Returns
    -------
    slot_idx : (n_steps,) int32
        For each step, which slot its value contributes to.
    n_slots : int
    slot_times : (n_slots,) float64
        Time coordinate for the tape's NetCDF ``time`` axis, in the same
        seconds-since-year_start frame as the input.
    """
    t = np.asarray(model_times_s, dtype=np.float64)
    if t.size == 0:
        raise ValueError("model_times_s is empty")

    if freq == "step":
        n = t.size
        return np.arange(n, dtype=np.int32), n, t

    # For binned frequencies the slot index is ABSOLUTE (counted from year_start
    # Jan 1), so a run that does not start at slot 0 -- any --start-doy > 0 or a
    # restart continuation -- must be OFFSET by the first occupied slot.  Without
    # this, slots [0, first) are never written, and finalize_tape divides them by
    # max(count,1)=1 -> phantom leading records of 0.0 that silently pollute any
    # time-mean over the tape.  ``_offset_slots`` returns (slot_idx-rebased,
    # n_slots, absolute-slot-ids) so slot_times stay in the true year_start frame.
    def _offset_slots(abs_idx):
        lo = int(abs_idx.min())
        rebased = (abs_idx - lo).astype(np.int32)
        n = int(rebased.max()) + 1
        return rebased, n, np.arange(lo, lo + n)

    if freq == "hourly":
        slot_idx, n_slots, abs_slots = _offset_slots((t / _SEC_PER_HOUR).astype(np.int64))
        slot_times = (abs_slots + 0.5) * _SEC_PER_HOUR
        return slot_idx, n_slots, slot_times

    if freq == "daily":
        slot_idx, n_slots, abs_slots = _offset_slots((t / _SEC_PER_DAY).astype(np.int64))
        slot_times = (abs_slots + 0.5) * _SEC_PER_DAY
        return slot_idx, n_slots, slot_times

    if freq == "monthly":
        doy_total = t / _SEC_PER_DAY
        year_offset = (doy_total / _DAYS_PER_YEAR).astype(np.int64)
        doy_in_year = doy_total - year_offset * _DAYS_PER_YEAR
        # searchsorted returns 1..12; subtract 1 for 0-indexed month.
        month = np.searchsorted(_MONTH_START_DOY, doy_in_year, side="right") - 1
        month = np.clip(month, 0, 11)
        slot_idx, n_slots, abs_slots = _offset_slots(year_offset * 12 + month)
        # midpoint doy per ABSOLUTE slot id (year*12+month)
        month_len = np.diff(np.concatenate(
            [_MONTH_START_DOY, np.array([_DAYS_PER_YEAR], dtype=np.int32)]))
        slot_month = abs_slots % 12
        slot_year = abs_slots // 12
        mid_doy = _MONTH_START_DOY[slot_month] + month_len[slot_month] / 2.0
        slot_times = (slot_year * _DAYS_PER_YEAR + mid_doy) * _SEC_PER_DAY
        return slot_idx, n_slots, slot_times

    if freq == "annual":
        slot_idx, n_slots, abs_slots = _offset_slots(
            ((t / _SEC_PER_DAY) / _DAYS_PER_YEAR).astype(np.int64))
        slot_times = (abs_slots + 0.5) * _DAYS_PER_YEAR * _SEC_PER_DAY
        return slot_idx, n_slots, slot_times

    raise ValueError(f"unknown freq {freq!r} (allowed: {_VALID_FREQ})")


#: Tape variables carrying a per-layer SOIL PROFILE rather than a single value
#: per column.  Their accumulator is ``(n_slots, ncol, n_layers)`` and they are
#: written with a trailing ``layer`` dimension.  Everything else on a tape stays
#: ``(n_slots, ncol)``, so mixing profile and scalar variables on one tape works.
PROFILE_VARS = ("T_soil", "theta_soil")


def is_profile_var(name: str) -> bool:
    """True if ``name`` is archived as a per-layer soil profile."""
    return name in PROFILE_VARS


def init_tape_accumulator(tape: TapeSpec, n_slots: int, ncol: int,
                          dtype=jnp.float64, n_layers: int | None = None) -> dict[str, Any]:
    """Zero-initialised accumulator for one tape.

    ``n_layers`` is required only when the tape requests a variable in
    :data:`PROFILE_VARS`; those get a trailing layer axis.
    """
    if n_layers is None and any(is_profile_var(v) for v in tape.vars):
        raise ValueError(
            f"tape {tape.name!r} requests profile variable(s) "
            f"{[v for v in tape.vars if is_profile_var(v)]} but n_layers was not "
            "supplied; the accumulator cannot be shaped without it")

    def _zeros(var):
        shape = ((n_slots, ncol, n_layers) if is_profile_var(var)
                 else (n_slots, ncol))
        return jnp.zeros(shape, dtype)

    return {
        "count": jnp.zeros(n_slots, dtype),
        **{var: _zeros(var) for var in tape.vars},
    }


def accumulate_tape_step(
    carry: dict[str, Any], tape: TapeSpec,
    slot: jnp.ndarray, values: dict[str, jnp.ndarray],
) -> dict[str, Any]:
    """One-step update to a tape's accumulator (JAX-traceable)."""
    out = dict(carry)
    if tape.average == "mean":
        out["count"] = out["count"].at[slot].add(1.0)
        for var in tape.vars:
            out[var] = out[var].at[slot].add(values[var])
    else:                                       # inst
        out["count"] = out["count"].at[slot].set(1.0)
        for var in tape.vars:
            out[var] = out[var].at[slot].set(values[var])
    return out


def finalize_tape(carry: dict[str, Any], tape: TapeSpec) -> dict[str, np.ndarray]:
    """Post-scan reduce.  Divides mean tapes by counts; passes through inst tapes."""
    if tape.average == "mean":
        count = np.maximum(np.asarray(carry["count"]), 1.0)
        out = {}
        for var in tape.vars:
            a = np.asarray(carry[var])
            # Broadcast the per-slot count against however many trailing axes the
            # variable has: (n_slots, ncol) for scalars, (n_slots, ncol, n_layers)
            # for profiles.  A hardcoded [:, None] silently mis-divides profiles.
            out[var] = a / count.reshape((-1,) + (1,) * (a.ndim - 1))
        return out
    return {var: np.asarray(carry[var]) for var in tape.vars}


__all__ = [
    "PROFILE_VARS",
    "TapeSpec",
    "is_profile_var",
    "default_config_path",
    "load_output_config",
    "build_slot_indices",
    "init_tape_accumulator",
    "accumulate_tape_step",
    "finalize_tape",
]
