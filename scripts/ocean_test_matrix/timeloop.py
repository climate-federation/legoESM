"""Shared time-loop utilities for the ocean test matrix."""

from __future__ import annotations

import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np


# ===========================================================================
# Shared utilities
# ===========================================================================

def check_finite(arrays: dict[str, Any]) -> bool:
    for arr in arrays.values():
        if not bool(jnp.all(jnp.isfinite(arr))):
            return False
    return True


def _snapshot_steps(n_steps: int, n_snaps: int = 10) -> set[int]:
    """Return step numbers at which to save snapshots."""
    if n_steps <= 0:
        return set()
    steps = {0, n_steps}
    for i in range(1, n_snaps):
        steps.add(max(1, int(i * n_steps / n_snaps)))
    return steps


def _compute_drift(values: list[float]) -> float:
    """Scalar drift wrapper.

    iter-91 (codex iter-90 followup audit): the previous inline
    implementation used ``max(abs(values[0]), 1e-30)`` — the iter-78
    pathology pattern that was already fixed in
    ``scripts/run_ocean_test_matrix.py:_compute_drift`` (iter-90)
    and factored into
    ``legoesm.diagnostics.conservation_drift`` (iter-88).  This
    second copy in the ``ocean_test_matrix`` package was missed in
    iter-88 and iter-90; iter-91's audit (re-greping for ``1e-30``
    in scripts/) caught it.

    Used by 10 callsites in ``experiments.py``
    (T_drift, PE_drift, S_integral_drift).  All inherit the 1.0
    floor convention via this delegation.
    """
    from legoesm.diagnostics.conservation_drift import compute_relative_drift
    return compute_relative_drift(values)


def _apply_drift_tolerance(
    ok: bool, notes: str, drift: float, tol: float,
    *, label: str, n_samples: int | None = None,
) -> tuple[bool, str]:
    """Thin wrapper that delegates to the centralized
    ``legoesm.diagnostics.conservation_drift.apply_drift_tolerance``
    helper (iter-127 codex iter-126-followup LOW-5).
    """
    from legoesm.diagnostics.conservation_drift import (
        apply_drift_tolerance,
    )
    return apply_drift_tolerance(
        ok, notes, drift, tol,
        label=label, n_samples=n_samples,
    )


# ===========================================================================
# Generic time loop
# ===========================================================================

def _run_timeloop(
    step_fn: Callable,
    state: Any,
    dt: float,
    n_steps: int,
    check_fn: Callable,
    scalar_fn: Callable,
    extract_fn: Callable,
    diag_every: int,
    key_array_fn: Callable,
    *,
    label: str = "",
    total_days: float = 0,
    blowup_threshold: float = 100.0,
    max_speed_threshold: float = 50.0,
    n_snaps: int = 10,
) -> tuple[Any, dict, dict, float, bool]:
    """Run time loop with diagnostics and runtime CFL monitoring.

    Returns (final_state, snapshots, diag, wall_time, ok).
    """
    snap_targets = _snapshot_steps(n_steps, n_snaps)
    snapshots: dict[int, dict[str, np.ndarray]] = {0: extract_fn(state)}
    scalars_0 = scalar_fn(state)
    diag: dict[str, list] = {"times": [0.0], "steps": [0]}
    for k, v in scalars_0.items():
        diag.setdefault(k, []).append(v)

    t0 = time.time()
    last_print = t0
    blown_up = False

    for i in range(n_steps):
        state = step_fn(state, dt)
        step = i + 1

        if step in snap_targets:
            snapshots[step] = extract_fn(state)

        if step % 100 == 0:
            is_finite, metric = check_fn(state)
            if not is_finite or metric > blowup_threshold:
                print(f"  BLOWUP at step {step}, metric={metric}")
                blown_up = True
                break

        if step % diag_every == 0:
            day = step * dt / 86400.0
            scalars = scalar_fn(state)
            diag["times"].append(day)
            diag["steps"].append(step)
            for k, v in scalars.items():
                diag.setdefault(k, []).append(v)

            max_spd = scalars.get("max_speed", 0.0)
            if max_spd > max_speed_threshold:
                print(f"  BLOWUP at step {step}: max_speed={max_spd:.2f} m/s "
                      f"exceeds threshold {max_speed_threshold}")
                blown_up = True
                break

            now = time.time()
            if now - last_print > 30:
                summary = " | ".join(
                    f"{k}={v:.4g}" for k, v in list(scalars.items())[:3])
                spd_str = f"max_spd={max_spd:.4f}"
                print(f"    Day {day:7.1f}/{total_days} | {summary} | {spd_str}")
                last_print = now

    jax.block_until_ready(key_array_fn(state))
    wall = time.time() - t0

    is_finite, _ = check_fn(state)
    ok = is_finite and not blown_up

    return state, snapshots, diag, wall, ok
