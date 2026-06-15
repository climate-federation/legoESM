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
    ``scripts/matrix/run_ocean_test_matrix.py:_compute_drift`` (iter-90)
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


def _apply_value_threshold(
    ok: bool, notes: str, value: float, threshold: float,
    *, label: str, op: str = "le", units: str = "",
    n_samples: int | None = None,
) -> tuple[bool, str]:
    """Thin wrapper for non-drift PASS thresholds.  Delegates
    to ``legoesm.diagnostics.conservation_drift.apply_value_threshold``
    (iter-128 codex iter-127-followup MEDIUM-2/3).

    iter-130 (codex iter-129-followup HIGH-2): added the
    ``n_samples`` kwarg.  Without it, callsites that pass
    ``n_samples=`` would raise ``TypeError``.  Mirror of the
    fix in monolithic ``run_ocean_test_matrix.py``.
    """
    from legoesm.diagnostics.conservation_drift import (
        apply_value_threshold,
    )
    return apply_value_threshold(
        ok, notes, value, threshold,
        label=label, op=op, units=units, n_samples=n_samples,
    )


# iter-154: import the centralized sentinel from
# legoesm.diagnostics so the same singleton is used here and
# in scripts/matrix/run_ocean_test_matrix.py.
from legoesm.diagnostics import DAYS_REQUIRED as _DAYS_REQUIRED

# Absolute deadband for the Overflow / Lock-Exchange RPE-sign gate (iter-156).
# 10x above the observed O(1e-6) quick-mode sign-noise, ~1000x below the 1e-2
# conservation-health scale: passes discretization noise, fails a real RPE rise.
_PE_REL_SIGN_DEADBAND = 1.0e-5


def _apply_pe_rel_sign(
    ok: bool, notes: str, pe_rel_final: float, *, label: str,
    n_samples: int | None = None, days=_DAYS_REQUIRED,
) -> tuple[bool, str]:
    """Apply the documented ``pe_rel_final < 0`` sign
    constraint (iter-128 codex iter-127-followup MEDIUM-2).

    iter-129 (codex iter-128-followup MEDIUM-1/LOW-3): switched
    from deprecated ``op="lt_zero"`` to general ``op="lt"`` with
    explicit ``threshold=0.0``.  Added ``n_samples`` kwarg.

    Used by Overflow and Lock Exchange (both have the same
    documented ``pe_rel_final < 0`` contract).

    iter-138 (iter-137 production finding FAIL-2): same
    ``op="lt" → op="le"`` loosening as monolithic; the strict
    gate broke quick mode where 28-30 timesteps weren't enough
    for measurable PE evolution.

    iter-152 (codex iter-151 review MEDIUM-2): days-aware op
    selection — strict ``< 0`` for full mode, ``≤ 0`` for
    quick (matches monolithic).

    iter-153 (codex iter-152 review MEDIUM-1): ``days`` is
    REQUIRED.

    iter-154 (codex iter-153 review MEDIUM-1): also reject
    ``days=None`` and non-finite/non-positive — see monolithic
    docstring for full rationale.

    iter-156 (smoke-sweep finding): in QUICK MODE ONLY, apply a
    small ABSOLUTE deadband instead of a strict sign-of-noise
    check. A short quick-mode gravity current (e.g. Overflow on
    cubed_sphere, 0.1 days) barely evolves the plume, so the
    diagnosed RPE change is dominated by O(1e-7) discretization
    noise that can land marginally POSITIVE (+6.5e-7 observed)
    even though the plume is not gaining available potential
    energy. Quick mode now passes while ``pe_rel_final`` stays
    below ``_PE_REL_SIGN_DEADBAND`` (10x above the observed
    O(1e-6) sign-noise, ~1000x below the 1e-2 conservation-health
    scale). FULL mode (days>=1) keeps the strict ``< 0`` contract
    (threshold 0), so a genuine RPE INCREASE still fails. Lock
    Exchange (-1e-8) and full-mode Overflow (-5.7e-8) unaffected.
    """
    if days is _DAYS_REQUIRED:
        raise TypeError(
            f"_apply_pe_rel_sign: 'days' kwarg is required "
            f"(label={label!r})."
        )
    # iter-155 (codex iter-154 review LOW-1): same numbers.Real
    # + bool reject as monolithic.
    import math as _math
    import numbers as _numbers
    if (days is None or isinstance(days, bool)
            or not isinstance(days, _numbers.Real)):
        raise ValueError(
            f"_apply_pe_rel_sign: 'days' must be a real number, "
            f"got {type(days).__name__}={days!r} "
            f"(label={label!r})."
        )
    days_f = float(days)
    if not _math.isfinite(days_f) or days_f <= 0:
        raise ValueError(
            f"_apply_pe_rel_sign: 'days' must be a finite "
            f"positive number, got {days!r} (label={label!r})."
        )
    op = "lt" if days_f >= 1.0 else "le"
    # Full mode keeps the STRICT documented RPE-decrease contract (threshold 0);
    # the deadband applies ONLY in quick mode, where a barely-evolved plume's
    # O(1e-7) discretization noise can land marginally positive.
    threshold = 0.0 if days_f >= 1.0 else _PE_REL_SIGN_DEADBAND
    return _apply_value_threshold(
        ok, notes, pe_rel_final, threshold,
        label=label, op=op, n_samples=n_samples,
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
