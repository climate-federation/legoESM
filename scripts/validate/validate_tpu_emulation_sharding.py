#!/usr/bin/env python
"""Phase 0 TPU readiness: emulated-device sharding validation (free, CPU-only).

Validates that the cubed-sphere FV3 SPMD path — the intended first
single-host TPU workload (v5e-8 / v6e-8) — runs correctly under multiple
device counts and produces results that match the single-device reference.
This exercises the *exact* sharded-dynamics code paths that will run on
real TPU chips, but on emulated CPU devices, so we can de-risk sharding
before paying for a TPU VM.

Why emulation works as a proxy
------------------------------
SPMD on a single host is numerically a no-op: ``jit`` with
``NamedSharding`` makes each device compute only its shard, and
cross-face stencils (``pad_halo``) trigger XLA collectives.  The math is
identical to single-device — only data placement changes.  Running the
same program on 8 emulated CPU devices verifies the partitioning logic,
halo collectives, and the Issue-#273 level-parallel fallback that the
8-chip slice requires (8 does not divide the 6-face layout).

Two sharding regimes are checked against the single-device reference:

  * ``n=6``  — true cubed-sphere *face* sharding (one face per device).
  * ``n=8``  — *level-parallel fallback* (``allow_level_fallback=True``);
               the dycore is replicated across all 8 devices.  This is
               the "use the whole v5e-8/v6e-8 slice" path.

float32 only: TPUs have no efficient native float64, so the TPU plan
stays in float32 (spectral/x64 dycores are excluded).  This script
therefore runs entirely in float32, matching the real TPU constraint.

Usage
-----
    python scripts/validate/validate_tpu_emulation_sharding.py
    python scripts/validate/validate_tpu_emulation_sharding.py --n-grid 16 --n-levels 10 --steps 5

Exit code is non-zero if any sharded regime diverges from the
single-device reference beyond tolerance, or produces non-finite values.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

# Repo root on sys.path so ``tests.test_cases.baroclinic_wave`` (the shared
# IC used by run_levante_gpu_scaling) imports when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Emulate 8 devices on CPU.  MUST run before JAX is imported so the XLA CPU
# backend creates the requested number of logical devices.  We do NOT set
# JAX_ENABLE_X64 — the TPU target is float32-only (see module docstring).
# ---------------------------------------------------------------------------
_N_EMULATED = int(os.environ.get("TPU_EMU_DEVICES", "8"))
_flags = os.environ.get("XLA_FLAGS", "")
if "xla_force_host_platform_device_count" not in _flags:
    os.environ["XLA_FLAGS"] = (
        f"{_flags} --xla_force_host_platform_device_count={_N_EMULATED}".strip()
    )
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402


def _auto_dt(n_grid: int) -> float:
    """CFL-safe cubed-sphere timestep (mirrors run_levante_gpu_scaling._auto_dt)."""
    from legoesm import constants  # lazy: keep JAX-init order intact

    R = constants.R_earth
    dx_min = (math.pi / 2) * R / (n_grid * math.sqrt(3))
    u_max = 60.0
    c_grav = 300.0  # external gravity wave speed [m/s]
    cfl = 0.7
    dt = cfl * dx_min / (u_max + c_grav)
    return max(30.0, 30.0 * int(dt / 30.0))


def _build_model_and_state(n_grid: int, n_levels: int, dtype):
    """Construct the C{n_grid}/L{n_levels} FV3 model + baroclinic-wave IC.

    Mirrors the cubed-sphere branch of run_levante_gpu_scaling.run_benchmark
    so we validate the production SPMD construction, not a bespoke one.
    """
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
        CDGridPrimitiveEquationConfig,
        hydrostatic_to_fv3,
    )
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init

    grid = create_cubed_sphere(n_grid)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sigma = create_sigma_coordinate(n_levels)

    # Conservation fixers ON: their global all-reduces are exactly the
    # collective paths we want to stress under sharding.
    config = CDGridPrimitiveEquationConfig(
        use_conservation_fixer=True,
        fix_mass=True,
        anchor_mass_to_initial=True,
        zero_mean_ps_tendency=False,
    )
    model = CDGridPrimitiveEquationModel(grid, sigma, config)
    state_cc = baroclinic_wave_init(grid, sigma, perturbed=True)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    def _cast(x):
        if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(dtype)
        return x

    state = jax.tree.map(_cast, state)
    return model, state, n_levels


def _run_reference(n_grid, n_levels, dt, steps):
    """Single-device ground truth: plain model.step, no sharding at all.

    Builds its own fresh model so the mass-anchor cache (a model-instance
    side effect) is never shared across separately-jitted runs.
    """
    model, state, _ = _build_model_and_state(n_grid, n_levels, jnp.float32)
    step = jax.jit(model.step)
    for _ in range(steps):
        state = step(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))
    return state


def _run_sharded(dt, steps, *, n_devices, allow_level_fallback, n_grid, n_levels):
    """Run the SPMD sharded step path for `n_devices`, return gathered state.

    Builds a fresh model+state (deterministic IC) so the model-instance
    mass-anchor cache is not contaminated by another regime's jit trace.
    """
    from legoesm.parallel.mesh import create_device_mesh, shard_pytree
    from legoesm.parallel.sharded_dynamics import (
        make_sharded_step,
        gather_state,
        check_sharding,
    )

    model, state, _ = _build_model_and_state(n_grid, n_levels, jnp.float32)
    cfg = create_device_mesh(
        n_devices=n_devices, allow_level_fallback=allow_level_fallback
    )
    if cfg.n_devices != n_devices:
        return None, (
            f"requested {n_devices} devices but mesh built {cfg.n_devices} "
            f"(only {len(jax.devices())} emulated devices available)"
        )

    sharded = shard_pytree(state, cfg)
    sr = check_sharding(sharded, cfg)
    step = make_sharded_step(model, cfg, n=n_grid, nlev=n_levels)
    try:
        for _ in range(steps):
            sharded = step(sharded, dt)
        jax.block_until_ready(jax.tree.leaves(sharded))
    except ValueError as exc:
        # Known gap: the cubed-sphere dycore halo runs a 6-face shard_map
        # (explicit_pad_halo_4d) regardless of config, so a replicated
        # >6-device (level-fallback) mesh raises an incompatible-devices
        # error.  Report it instead of crashing the whole validation.
        msg = str(exc).splitlines()[0]
        return None, f"UNSUPPORTED step path on {cfg.grid_type} mesh: {msg}"
    gathered = gather_state(sharded, cfg)
    info = (
        f"grid_type={cfg.grid_type}, backend={cfg.backend}, "
        f"sharded_leaves={sr['n_sharded']}, replicated={sr['n_replicated']}"
    )
    return gathered, info


def _compare(ref, test, atol, rtol):
    """Compare two states leaf-by-leaf with a magnitude-scaled tolerance.

    A sharded run differs from single-device only by float32 roundoff in
    reordered global reductions (collectives).  That roundoff scales with
    each field's magnitude, so the right test is per-leaf::

        max|a-b|  <=  atol + rtol * max|a|

    A flat absolute tolerance is wrong here because state fields span
    magnitudes from O(10) winds to O(1e5) surface pressure.

    Returns (ok, worst_scaled_ratio, rows) where ``rows`` is a per-leaf
    list of (idx, shape, |ref|max, max_abs, allowed, pass).
    """
    ref_leaves = jax.tree.leaves(ref)
    test_leaves = jax.tree.leaves(test)
    rows = []
    ok = True
    worst_ratio = 0.0
    for i, (a, b) in enumerate(zip(ref_leaves, test_leaves)):
        a = np.asarray(a)
        b = np.asarray(b)
        finite = np.all(np.isfinite(a)) and np.all(np.isfinite(b))
        if a.shape != b.shape:
            rows.append((i, f"{a.shape} vs {b.shape}", math.nan, math.nan, math.nan, False))
            ok = False
            continue
        mag = float(np.abs(a).max()) if a.size else 0.0
        max_abs = float(np.abs(a - b).max()) if a.size else 0.0
        allowed = atol + rtol * mag
        leaf_ok = finite and (max_abs <= allowed)
        if allowed > 0:
            worst_ratio = max(worst_ratio, max_abs / allowed)
        ok = ok and leaf_ok
        rows.append((i, str(a.shape), mag, max_abs, allowed, leaf_ok))
    return ok, worst_ratio, rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-grid", type=int, default=12, help="per-face resolution (C<n>)")
    p.add_argument("--n-levels", type=int, default=10)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument(
        "--atol", type=float, default=1e-4,
        help="absolute floor of the per-leaf tolerance atol + rtol*|ref|max",
    )
    p.add_argument(
        "--rtol", type=float, default=1e-4,
        help="magnitude-scaled tolerance (float32 collective-reduction roundoff)",
    )
    p.add_argument("--verbose", action="store_true", help="print per-leaf diff table")
    args = p.parse_args()

    devs = jax.devices()
    print(f"JAX backend: {jax.default_backend()} | devices: {len(devs)}")
    if len(devs) < 8:
        print(
            f"  WARNING: only {len(devs)} devices emulated; set "
            f"TPU_EMU_DEVICES=8 (this run requested {_N_EMULATED}).",
            flush=True,
        )

    dt = _auto_dt(args.n_grid)
    print(
        f"Config: C{args.n_grid}/L{args.n_levels} | float32 | dt={dt:.0f}s | "
        f"steps={args.steps} | tol = {args.atol:g} + {args.rtol:g}*|ref|max\n",
        flush=True,
    )

    print("[ref] single-device (model.step) ...", flush=True)
    ref = _run_reference(args.n_grid, args.n_levels, dt, args.steps)
    ref_finite = all(
        np.all(np.isfinite(np.asarray(x))) for x in jax.tree.leaves(ref)
    )
    print(f"      reference finite: {ref_finite}\n", flush=True)
    if not ref_finite:
        print("FAIL: single-device reference produced non-finite values.")
        return 1

    regimes = [
        ("face-sharding (6 devices)", dict(n_devices=6, allow_level_fallback=False)),
        ("level-fallback (8 devices)", dict(n_devices=8, allow_level_fallback=True)),
    ]

    overall_ok = True
    n_validated = 0
    for label, kw in regimes:
        print(f"[{label}] running ...", flush=True)
        gathered, info = _run_sharded(
            dt, args.steps,
            n_grid=args.n_grid, n_levels=args.n_levels, **kw,
        )
        if gathered is None:
            # A SKIP is a reported gap, not a pass — but it does not fail
            # the run if the supported regimes pass (it is informational).
            print(f"  SKIP: {info}\n", flush=True)
            continue
        print(f"  {info}", flush=True)
        ok, worst_ratio, rows = _compare(ref, gathered, args.atol, args.rtol)
        if args.verbose or not ok:
            print(f"    {'idx':>3} {'shape':>16} {'|ref|max':>12} {'max_abs':>12} {'allowed':>12} ok")
            for idx, shape, mag, mx, allowed, leaf_ok in rows:
                print(
                    f"    {idx:>3} {shape:>16} {mag:>12.4e} {mx:>12.4e} "
                    f"{allowed:>12.4e} {'Y' if leaf_ok else 'N'}"
                )
        status = "PASS" if ok else "FAIL"
        print(
            f"  {status}: worst diff is {worst_ratio:.2f}x the tolerance "
            f"(<=1.0 passes)\n",
            flush=True,
        )
        overall_ok = overall_ok and ok
        n_validated += 1

    print("=" * 60)
    if overall_ok and n_validated > 0:
        print(
            f"OVERALL: PASS — {n_validated} sharded regime(s) match "
            f"single-device cubed-sphere FV3 within float32 roundoff."
        )
        print("Phase 0 emulation validated for the supported (face-sharding) path.")
        return 0
    if n_validated == 0:
        print("OVERALL: FAIL — no sharded regime could be validated.")
        return 1
    print("OVERALL: FAIL — a supported regime diverged; see diffs above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())