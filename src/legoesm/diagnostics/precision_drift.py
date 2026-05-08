"""Precision drift checker and health diagnostics for legoESM.

Compares reference (FP64) and test (FP32/mixed) runs to detect
precision-induced instability before it contaminates long climate runs.

Usage
-----
>>> checker = PrecisionDriftChecker()
>>> checker.record(step=0, ref_state=ref, test_state=test, grid=grid)
>>> checker.record(step=100, ref_state=ref, test_state=test, grid=grid)
>>> report = checker.summary()

Runtime health:
>>> health = precision_health_report(state, state_prev, grid, sigma_coord, dt)
>>> if health["status"] == "UNHEALTHY":
...     print("Precision problem:", health["warnings"])
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants


def _best_float():
    """Return float64 if x64 is enabled, else float32."""
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


# ---------------------------------------------------------------------------
# Drift checker — compare FP64 reference vs FP32/mixed test
# ---------------------------------------------------------------------------

class DriftSnapshot(NamedTuple):
    """Single-timestep comparison between reference and test runs."""
    step: int
    rms_T: float          # RMS difference in temperature [K]
    rms_u: float          # RMS difference in zonal wind [m/s]
    rms_ps: float         # RMS difference in surface pressure [Pa]
    linf_T: float         # L-infinity difference in temperature [K]
    linf_u: float         # L-infinity difference in zonal wind [m/s]
    linf_ps: float        # L-infinity difference in surface pressure [Pa]
    global_mass_ref: float
    global_mass_test: float
    global_energy_ref: float
    global_energy_test: float


def _rms(a: jax.Array, b: jax.Array) -> jax.Array:
    """Root-mean-square difference."""
    diff = a.ravel() - b.ravel()
    return jnp.sqrt(jnp.mean(diff.astype(_best_float()) ** 2))


def _linf(a: jax.Array, b: jax.Array) -> jax.Array:
    """L-infinity (max absolute) difference."""
    return jnp.max(jnp.abs(a.ravel() - b.ravel()))


def compare_states(
    ref_state,
    test_state,
    grid,
    sigma_coord=None,
    step: int = 0,
) -> DriftSnapshot:
    """Compare reference and test model states.

    Parameters
    ----------
    ref_state : HydrostaticState
        Reference run state (typically FP64).
    test_state : HydrostaticState
        Test run state (FP32 or mixed).
    grid : CubedSphereGrid or LatLonGrid
        Grid with cell areas.
    sigma_coord : SigmaCoordinate, optional
        Vertical coordinate (for energy computation).
    step : int
        Timestep index.

    Returns
    -------
    DriftSnapshot
    """
    _acc = _best_float()
    ref_T = ref_state.T.data.astype(_acc)
    test_T = test_state.T.data.astype(_acc)
    ref_u = ref_state.u.data.astype(_acc)
    test_u = test_state.u.data.astype(_acc)
    ref_ps = ref_state.p_s.data.astype(_acc)
    test_ps = test_state.p_s.data.astype(_acc)

    area = grid.area.astype(_acc)
    total_area = jnp.sum(area)

    # Global mass = ∫ p_s dA / g
    g = constants.g
    mass_ref = jnp.sum(ref_ps * area) / g
    mass_test = jnp.sum(test_ps * area) / g

    # Global energy (approximate: internal only)
    c_p = constants.c_pd
    if sigma_coord is not None:
        dsigma = jnp.asarray(sigma_coord.dsigma, dtype=_acc)
        energy_ref = jnp.sum(
            c_p * ref_T * ref_ps[..., None] * dsigma * area[..., None]
        ) / g
        energy_test = jnp.sum(
            c_p * test_T * test_ps[..., None] * dsigma * area[..., None]
        ) / g
    else:
        energy_ref = jnp.sum(c_p * ref_T * area[..., None]) / g
        energy_test = jnp.sum(c_p * test_T * area[..., None]) / g

    # Fuse the 10 ``float(...)`` calls into a single ``jnp.stack`` +
    # ``np.asarray`` device→host transfer.  The old per-scalar
    # ``device_get`` chain serialised 10 GPU stalls per snapshot;
    # ``compare_states`` is called every diagnostic interval, so this
    # adds up.
    _stats = jnp.stack([
        _rms(ref_T, test_T).astype(ref_T.dtype),
        _rms(ref_u, test_u).astype(ref_T.dtype),
        _rms(ref_ps, test_ps).astype(ref_T.dtype),
        _linf(ref_T, test_T).astype(ref_T.dtype),
        _linf(ref_u, test_u).astype(ref_T.dtype),
        _linf(ref_ps, test_ps).astype(ref_T.dtype),
        mass_ref.astype(ref_T.dtype),
        mass_test.astype(ref_T.dtype),
        energy_ref.astype(ref_T.dtype),
        energy_test.astype(ref_T.dtype),
    ])
    _h = np.asarray(_stats)
    return DriftSnapshot(
        step=step,
        rms_T=float(_h[0]),
        rms_u=float(_h[1]),
        rms_ps=float(_h[2]),
        linf_T=float(_h[3]),
        linf_u=float(_h[4]),
        linf_ps=float(_h[5]),
        global_mass_ref=float(_h[6]),
        global_mass_test=float(_h[7]),
        global_energy_ref=float(_h[8]),
        global_energy_test=float(_h[9]),
    )


class PrecisionDriftChecker:
    """Accumulate and analyze precision drift over time.

    Call ``record()`` at regular intervals during parallel ref/test runs,
    then call ``summary()`` to get aggregate statistics and pass/fail.

    Attributes
    ----------
    snapshots : list[DriftSnapshot]
        All recorded comparisons.
    """

    def __init__(
        self,
        *,
        rms_T_threshold: float = 1.0,
        rms_u_threshold: float = 5.0,
        rms_ps_threshold: float = 100.0,
        mass_rel_threshold: float = 1e-10,
        energy_rel_threshold: float = 1e-6,
    ):
        self.snapshots: list[DriftSnapshot] = []
        self.rms_T_threshold = rms_T_threshold
        self.rms_u_threshold = rms_u_threshold
        self.rms_ps_threshold = rms_ps_threshold
        self.mass_rel_threshold = mass_rel_threshold
        self.energy_rel_threshold = energy_rel_threshold

    def record(
        self,
        ref_state,
        test_state,
        grid,
        sigma_coord=None,
        step: int = 0,
    ) -> DriftSnapshot:
        """Record a comparison snapshot."""
        snap = compare_states(ref_state, test_state, grid, sigma_coord, step)
        self.snapshots.append(snap)
        return snap

    def summary(self) -> dict:
        """Compute aggregate statistics and pass/fail assessment.

        Returns
        -------
        dict with keys:
            n_snapshots, max_rms_T, max_rms_u, max_rms_ps,
            max_linf_T, max_linf_u, max_linf_ps,
            mass_drift_rel, energy_drift_rel,
            passed, warnings
        """
        if not self.snapshots:
            return {"n_snapshots": 0, "passed": True, "warnings": []}

        warnings = []

        max_rms_T = max(s.rms_T for s in self.snapshots)
        max_rms_u = max(s.rms_u for s in self.snapshots)
        max_rms_ps = max(s.rms_ps for s in self.snapshots)
        max_linf_T = max(s.linf_T for s in self.snapshots)
        max_linf_u = max(s.linf_u for s in self.snapshots)
        max_linf_ps = max(s.linf_ps for s in self.snapshots)

        # Mass drift: relative difference at final snapshot
        last = self.snapshots[-1]
        mass_ref = last.global_mass_ref
        mass_drift_rel = abs(last.global_mass_test - mass_ref) / max(
            abs(mass_ref), 1e-30
        )

        energy_ref = last.global_energy_ref
        energy_drift_rel = abs(last.global_energy_test - energy_ref) / max(
            abs(energy_ref), 1e-30
        )

        if max_rms_T > self.rms_T_threshold:
            warnings.append(
                f"RMS T difference {max_rms_T:.3f} K > "
                f"threshold {self.rms_T_threshold}"
            )
        if max_rms_u > self.rms_u_threshold:
            warnings.append(
                f"RMS u difference {max_rms_u:.3f} m/s > "
                f"threshold {self.rms_u_threshold}"
            )
        if max_rms_ps > self.rms_ps_threshold:
            warnings.append(
                f"RMS p_s difference {max_rms_ps:.1f} Pa > "
                f"threshold {self.rms_ps_threshold}"
            )
        if mass_drift_rel > self.mass_rel_threshold:
            warnings.append(
                f"Mass drift {mass_drift_rel:.2e} > "
                f"threshold {self.mass_rel_threshold}"
            )
        if energy_drift_rel > self.energy_rel_threshold:
            warnings.append(
                f"Energy drift {energy_drift_rel:.2e} > "
                f"threshold {self.energy_rel_threshold}"
            )

        return {
            "n_snapshots": len(self.snapshots),
            "max_rms_T": max_rms_T,
            "max_rms_u": max_rms_u,
            "max_rms_ps": max_rms_ps,
            "max_linf_T": max_linf_T,
            "max_linf_u": max_linf_u,
            "max_linf_ps": max_linf_ps,
            "mass_drift_rel": mass_drift_rel,
            "energy_drift_rel": energy_drift_rel,
            "passed": len(warnings) == 0,
            "warnings": warnings,
        }


# ---------------------------------------------------------------------------
# Runtime health diagnostics
# ---------------------------------------------------------------------------

def precision_health_report(
    state,
    state_prev=None,
    grid=None,
    sigma_coord=None,
    dt: float = 1.0,
    *,
    cfl_limit: float = 0.95,
    max_wind: float = 200.0,
    T_range: tuple[float, float] = (150.0, 350.0),
    ps_range: tuple[float, float] = (4e4, 1.15e5),
    energy_drift_per_day: float = 1e-3,
) -> dict:
    """Runtime precision health check.

    Detects precision-induced instability symptoms:
    - CFL violations
    - Unphysical temperature/pressure values
    - NaN/Inf contamination
    - Excessive energy drift rate
    - Tracer negativity

    Parameters
    ----------
    state : HydrostaticState
        Current model state.
    state_prev : HydrostaticState, optional
        Previous state (for drift rate computation).
    grid : grid object, optional
        Grid with dx attribute (for CFL check).
    sigma_coord : vertical coordinate, optional
    dt : float
        Timestep in seconds.

    Returns
    -------
    dict with keys:
        status ("HEALTHY" or "UNHEALTHY"),
        warnings (list of strings),
        metrics (dict of diagnostic values)
    """
    warnings_list = []
    metrics = {}

    T = state.T.data
    u = state.u.data
    v = state.v.data
    ps = state.p_s.data

    # Fuse 7 reductions into one host transfer.  ``bool(jnp.any(...) or
    # jnp.any(...))`` was triggering 6 separate device→host syncs (the
    # Python ``or`` between traced booleans calls ``__bool__`` on each
    # branch).  Plus 5 separate ``float(jnp.X(...))`` for T/ps/wind
    # bounds.
    _stats = jnp.stack([
        (jnp.any(jnp.isnan(T)) | jnp.any(jnp.isnan(u))
            | jnp.any(jnp.isnan(ps))).astype(T.dtype),
        (jnp.any(jnp.isinf(T)) | jnp.any(jnp.isinf(u))
            | jnp.any(jnp.isinf(ps))).astype(T.dtype),
        jnp.min(T).astype(T.dtype),
        jnp.max(T).astype(T.dtype),
        jnp.min(ps).astype(T.dtype),
        jnp.max(ps).astype(T.dtype),
        jnp.max(jnp.sqrt(u ** 2 + v ** 2)).astype(T.dtype),
    ])
    _h = np.asarray(_stats)
    has_nan = bool(_h[0] > 0.5)
    has_inf = bool(_h[1] > 0.5)
    T_min = float(_h[2])
    T_max = float(_h[3])
    ps_min = float(_h[4])
    ps_max = float(_h[5])
    wind_max = float(_h[6])

    metrics["has_nan"] = has_nan
    metrics["has_inf"] = has_inf
    if has_nan:
        warnings_list.append("NaN detected in state variables")
    if has_inf:
        warnings_list.append("Inf detected in state variables")

    # Temperature bounds
    metrics["T_min"] = T_min
    metrics["T_max"] = T_max
    if T_min < T_range[0]:
        warnings_list.append(f"T_min={T_min:.1f} K below {T_range[0]} K")
    if T_max > T_range[1]:
        warnings_list.append(f"T_max={T_max:.1f} K above {T_range[1]} K")

    # Surface pressure bounds
    metrics["ps_min"] = ps_min
    metrics["ps_max"] = ps_max
    if ps_min < ps_range[0]:
        warnings_list.append(f"p_s min={ps_min:.0f} Pa below {ps_range[0]:.0f} Pa")
    if ps_max > ps_range[1]:
        warnings_list.append(f"p_s max={ps_max:.0f} Pa above {ps_range[1]:.0f} Pa")

    # Wind speed
    metrics["wind_max"] = wind_max
    if wind_max > max_wind:
        warnings_list.append(f"Max wind {wind_max:.1f} m/s exceeds {max_wind} m/s")

    # CFL check
    if grid is not None and hasattr(grid, "dx"):
        dx_min = float(jnp.min(grid.dx)) if grid.dx.ndim > 0 else float(grid.dx)
        c_sound = 340.0  # approximate
        cfl = (wind_max + c_sound) * dt / dx_min
        metrics["cfl"] = cfl
        if cfl > cfl_limit:
            warnings_list.append(f"CFL={cfl:.3f} exceeds limit {cfl_limit}")

    # Energy drift rate — fuse the two energy sums into a single
    # ``jnp.stack`` + host pull so the diagnostic costs one GPU stall
    # instead of two.
    if state_prev is not None and grid is not None:
        _acc = _best_float()
        area = grid.area.astype(_acc)
        c_p = constants.c_pd
        g = constants.g
        T_prev = state_prev.T.data
        ps_prev = state_prev.p_s.data
        _energy_pair = jnp.stack([
            jnp.sum(c_p * T.astype(_acc) * ps.astype(_acc)[..., None]
                    * area[..., None]),
            jnp.sum(c_p * T_prev.astype(_acc) * ps_prev.astype(_acc)[..., None]
                    * area[..., None]),
        ])
        _eh = np.asarray(_energy_pair)
        energy_now = float(_eh[0]) / g
        energy_prev = float(_eh[1]) / g
        # iter-92 audit followup: this used to inline
        # ``max(abs(energy_prev), 1e-30)`` — the iter-78/80
        # pathology pattern.  In production atmospheric energy
        # ~5e+24 J so the 1e-30 floor never bit, but for
        # consistency with the iter-88 ``conservation_drift``
        # helper (and to keep the ``legoesm.diagnostics``
        # package's drift-normalization conventions in one
        # place), delegate to the canonical helper.
        from legoesm.diagnostics.conservation_drift import (
            compute_relative_drift,
        )
        dE_rel = compute_relative_drift([energy_prev, energy_now])
        # Scale to per-day rate
        steps_per_day = 86400.0 / dt
        dE_per_day = dE_rel * steps_per_day
        metrics["energy_drift_per_day"] = dE_per_day
        if dE_per_day > energy_drift_per_day:
            warnings_list.append(
                f"Energy drift rate {dE_per_day:.2e}/day exceeds "
                f"threshold {energy_drift_per_day:.2e}/day"
            )

    status = "UNHEALTHY" if warnings_list else "HEALTHY"
    return {
        "status": status,
        "warnings": warnings_list,
        "metrics": metrics,
    }


def check_tracer_negativity(
    tracers: dict[str, jax.Array],
    *,
    threshold: float = -1e-12,
) -> dict[str, float]:
    """Check for negative tracer values.

    Parameters
    ----------
    tracers : dict of name -> array
        Tracer fields to check.
    threshold : float
        Values below this are flagged.

    Returns
    -------
    dict : tracer_name -> minimum value (only for those below threshold).
    """
    if not tracers:
        return {}
    # Stack the per-tracer mins into one ``jnp.stack`` and pull host
    # in one transfer.  The previous per-tracer ``float(jnp.min(...))``
    # serialised one device→host stall per tracer (typically 6-9
    # tracers under full microphysics).
    names = list(tracers.keys())
    mins_host = np.asarray(jnp.stack([jnp.min(tracers[n]) for n in names]))
    violations = {}
    for name, arr_min in zip(names, mins_host):
        v = float(arr_min)
        if v < threshold:
            violations[name] = v
    return violations
