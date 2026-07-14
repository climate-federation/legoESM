"""Validate the float32 shallow-water dycore on the current JAX backend.

Runs Williamson Test 2 (global steady-state nonlinear zonal geostrophic flow)
on the lat-lon C-grid in **float32** and checks three physical-sanity metrics:

* finiteness of the integrated state (no NaN/Inf),
* steady-state drift — TC2 is an exact steady solution, so ``max|h-h0|`` must
  stay tiny relative to the height scale, and
* mass conservation — the column-integrated mass must be held by the fixer.

Purpose: exercise the real C-grid SW solver (RK3 integrator, Sadourny
energy-conserving Coriolis, conservative mass-flux divergence, polar wall BC,
mass fixer) end-to-end in single precision.  This is the committed form of the
probe used to confirm that the Apple-Silicon GPU backend ``jax-mps`` (MLX,
float32-only) runs the dycore correctly — run it under ``JAX_PLATFORMS=mps`` to
validate the Apple GPU path, or under ``JAX_PLATFORMS=cpu`` for a portable
reference (CI).  Backend-agnostic: it asserts physical sanity on whatever
backend JAX selects, so it is NOT specific to mps.

Run::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 .venv/bin/python \
        scripts/validate/validate_mps_sw_float32.py
    JAX_PLATFORMS=mps JAX_ENABLE_X64=0 .venv/bin/python \
        scripts/validate/validate_mps_sw_float32.py
"""

from __future__ import annotations

import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np

# Default physical-sanity tolerances.  The CPU reference run drifts ~6e-5
# (steady-state) and holds mass to ~2e-5 over 50 steps (mps holds it tighter,
# ~2e-7).  These bounds keep ~3x/~5x headroom over the measured reference — tight
# enough that an order-of-magnitude regression (a broken flux form, a mass leak,
# or a growing instability) trips the gate, loose enough to absorb cross-backend
# float32 rounding and step-count variation.
_DEFAULT_DRIFT_TOL = 2.0e-4
_DEFAULT_MASS_TOL = 1.0e-4


class X64EnabledError(RuntimeError):
    """Raised when float32 validation is attempted with jax_enable_x64 on.

    A dedicated type (not a bare ``RuntimeError``) so the CLI can distinguish
    this expected precondition from a genuine dycore/JIT/backend failure and
    report ``SKIP`` only for the former — a real crash still surfaces.
    """


def run_sw_float32_validation(
    *,
    n_lat: int = 32,
    n_lon: int = 64,
    dt: float = 600.0,
    n_steps: int = 50,
) -> dict:
    """Integrate Williamson TC2 in float32 and return physical-sanity metrics.

    Pure (no file I/O).  Runs in the genuine single-precision regime: it
    REQUIRES ``jax_enable_x64`` to be OFF and raises ``RuntimeError`` otherwise.
    Under x64 the conservation accumulator and the mass fixer compute in float64
    and re-promote the state, so the run would no longer be single precision and
    mixing float32/float64 in the scan carry diverges to NaN.  That is not a
    real restriction: the Apple GPU backend (``mps`` / MLX) has no float64 at
    all, so x64-off is its native — and only — regime.

    Returns
    -------
    dict with keys: ``backend``, ``dtype``, ``n_steps``, ``finite``,
    ``h_min``, ``h_max``, ``h_drift_rel``, ``mass_rel``.  ``dtype`` is
    ``"float32"`` only when EVERY floating leaf of the integrated state is
    float32 (otherwise the offending dtype set, so the gate fails loudly);
    ``finite`` covers EVERY floating leaf (h, u, v, h_s), not just h.

    Raises
    ------
    X64EnabledError
        If ``jax.config.jax_enable_x64`` is set (run with ``JAX_ENABLE_X64=0``).
    """
    if jax.config.jax_enable_x64:
        raise X64EnabledError(
            "float32 shallow-water validation requires JAX_ENABLE_X64=0 — under "
            "x64 the conservation accumulator and mass fixer run in float64 and "
            "re-promote the state (mixing float32/float64 in the scan carry then "
            "diverges to NaN).  The Apple GPU backend (mps) has no float64, so "
            "this is its native regime: re-run with JAX_ENABLE_X64=0."
        )

    # Deferred imports: keep the heavy dycore graph off module import so the
    # CLI ``--help`` and the threshold-only unit test stay light.
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
        CGridLatLonShallowWaterConfig,
        CGridLatLonShallowWaterModel,
        williamson_test2_cgrid,
    )

    def _is_float(x):
        return hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)

    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon, dtype=jnp.float32)
    state0 = williamson_test2_cgrid(grid)
    # Belt-and-suspenders: cast any stray float leaf to float32.
    state0 = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float32) if _is_float(x) else x, state0,
    )

    config = CGridLatLonShallowWaterConfig(fix_mass=True)
    model = CGridLatLonShallowWaterModel(grid, config, dt=dt)

    mass0 = float(model.compute_mass(state0))
    state = state0
    for _ in range(n_steps):
        state = model.step(state, dt)
    jax.block_until_ready(state)
    mass_n = float(model.compute_mass(state))

    # Inspect EVERY floating leaf for finiteness and dtype (a NaN in v or h_s
    # must fail, and any non-float32 leaf must trip the dtype gate).
    leaves = [l for l in jax.tree_util.tree_leaves(state) if _is_float(l)]
    leaves_np = [np.asarray(jax.device_get(l)) for l in leaves]
    finite = bool(all(np.isfinite(arr).all() for arr in leaves_np))
    dtypes = sorted({str(l.dtype) for l in leaves})
    dtype = "float32" if dtypes == ["float32"] else "+".join(dtypes)

    h = np.asarray(jax.device_get(state.h))
    h0 = np.asarray(jax.device_get(state0.h))
    h_scale = float(np.max(np.abs(h0)))
    h_drift_rel = float(np.max(np.abs(h - h0)) / h_scale)
    mass_rel = float(abs(mass_n - mass0) / abs(mass0))

    return {
        "backend": jax.default_backend(),
        "dtype": dtype,
        "n_steps": n_steps,
        "finite": finite,
        "h_min": float(h.min()),
        "h_max": float(h.max()),
        "h_drift_rel": h_drift_rel,
        "mass_rel": mass_rel,
    }


def check_validation(
    metrics: dict,
    *,
    drift_tol: float = _DEFAULT_DRIFT_TOL,
    mass_tol: float = _DEFAULT_MASS_TOL,
) -> list[str]:
    """Return a list of failure reasons for *metrics* (empty list == pass).

    Pure threshold logic — no JAX — so it is unit-testable in-process under any
    precision policy.
    """
    failures: list[str] = []
    if metrics.get("dtype") != "float32":
        failures.append(f"state dtype is {metrics.get('dtype')!r}, expected 'float32'")
    if not metrics.get("finite", False):
        failures.append("integrated state contains NaN/Inf")
    drift = metrics.get("h_drift_rel", float("inf"))
    if not (drift <= drift_tol):
        failures.append(
            f"steady-state drift {drift:.3e} exceeds tol {drift_tol:.1e}"
        )
    mass = metrics.get("mass_rel", float("inf"))
    if not (mass <= mass_tol):
        failures.append(
            f"mass drift {mass:.3e} exceeds tol {mass_tol:.1e}"
        )
    return failures


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-lat", type=int, default=32)
    p.add_argument("--n-lon", type=int, default=64)
    p.add_argument("--dt", type=float, default=600.0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--drift-tol", type=float, default=_DEFAULT_DRIFT_TOL)
    p.add_argument("--mass-tol", type=float, default=_DEFAULT_MASS_TOL)
    args = p.parse_args(argv)

    try:
        metrics = run_sw_float32_validation(
            n_lat=args.n_lat, n_lon=args.n_lon, dt=args.dt, n_steps=args.steps,
        )
    except X64EnabledError as exc:
        # Precondition not met (x64 on) — report cleanly, no traceback.  Only
        # this specific error is swallowed; a genuine dycore/JIT failure still
        # propagates so it is not silently mislabeled as a skip.
        print(f"SKIP: {exc}")
        return 2
    print(
        f"backend={metrics['backend']} dtype={metrics['dtype']} "
        f"steps={metrics['n_steps']}"
    )
    print(
        f"h_range=[{metrics['h_min']:.2f}, {metrics['h_max']:.2f}]  "
        f"drift_rel={metrics['h_drift_rel']:.3e}  "
        f"mass_rel={metrics['mass_rel']:.3e}  finite={metrics['finite']}"
    )

    failures = check_validation(
        metrics, drift_tol=args.drift_tol, mass_tol=args.mass_tol,
    )
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("PASS: float32 shallow-water dycore validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
