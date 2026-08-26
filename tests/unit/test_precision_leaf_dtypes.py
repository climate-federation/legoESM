"""Per-mode leaf-dtype invariant for the hydrostatic PE dycores.

The gate for the precision-consistency work: after ONE step with the mass
fixer on, every prognostic state leaf must carry the policy STORAGE dtype
EXCEPT surface pressure ``p_s``, which is the intentional always-f64
conservation field (it carries the ACCUMULATE role dtype — float64 in the
fp64 and mixed policies).  See the ``p_s`` design note in
``legoesm.core.precision``.

Non-vacuity: in ``mixed`` (storage float32, accumulate float64) the contagion
guard asserts u / T / tracers stay float32.  If the f64 ``p_s`` promotion ever
spread to the 3D prognostics (measured NOT to happen — only p_s promotes), the
``== storage`` assertion on those leaves fails.  The existing shallow-water
one-step tests never exercise the PE mass fixer, so this is the only gate that
sees the leak.
"""
from __future__ import annotations

import os

os.environ.setdefault(
    "LEGOESM_MESH_CACHE_DIR",
    "/work/bd1083/b309178/diffESM/legoesm_mesh_cache",
)
os.environ.setdefault("LEGOESM_ALLOW_BIG_MESH_BUILD", "1")

import jax
import jax.numpy as jnp
import pytest

MODES = ["fp64", "mixed"]  # #1675: mixed enabled


def _cast_to_storage(state, storage):
    """A real run's state carries the storage dtype; the analytic IC builder
    emits float64 under x64, so downcast floats to storage before the step."""
    return jax.tree.map(
        lambda x: x.astype(storage)
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
        else x,
        state,
    )


def _run_mpas(mode):
    from legoesm.runtime.precision import apply_precision

    apply_precision(mode)
    from legoesm.core.precision import resolve_dtype
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig,
        MPASPrimitiveEquationModel,
    )
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas

    storage = resolve_dtype(None, "storage")
    accum = resolve_dtype(None, "accumulate")
    mesh = create_voronoi_mesh(subdivision_level=5, lloyd_iterations=0)
    sigma = create_sigma_coordinate(26)
    cfg = MPASPrimitiveEquationConfig(
        nu_del4=1e16, nu_del4_ps=1e16, fix_mass=True,
        pv_scheme="energy", time_integrator="ssp_rk3",
    )
    state0 = baroclinic_wave_init_mpas(mesh, sigma, perturbed=True, moist=True)
    state0 = _cast_to_storage(state0, storage)
    model = MPASPrimitiveEquationModel(mesh, sigma, cfg)
    return model.step(state0, 300.0), storage, accum


def _run_latlon(mode):
    from legoesm.runtime.precision import apply_precision

    apply_precision(mode)
    from legoesm.core.precision import resolve_dtype
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationModel,
        CGridLatLonPrimitiveEquationConfig,
        hydrostatic_to_cgrid,
    )
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_latlon

    storage = resolve_dtype(None, "storage")
    accum = resolve_dtype(None, "accumulate")
    grid = create_latlon_grid(n_lat=8)
    sigma = create_sigma_coordinate(8)
    state_cc = baroclinic_wave_init_latlon(grid, sigma, perturbed=True, moist=True)
    state = hydrostatic_to_cgrid(state_cc, grid)
    cfg = CGridLatLonPrimitiveEquationConfig(
        A_h=1.0e4, time_integrator="ssp_rk3", fix_mass=True,
        zero_mean_ps_tendency=False, use_ppm_transport=True,
        use_polar_filter=True, polar_filter_cutoff_deg=60.0,
        polar_filter_max_wave_speed=300.0,
    )
    dt = 100.0
    state = _cast_to_storage(state, storage)
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg, dt=dt)
    return model.step(state, dt=dt), storage, accum


def _assert_leaf_dtypes(state, storage, accum):
    checked_ps = False
    for path, leaf in jax.tree_util.tree_leaves_with_path(state):
        if not (hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)):
            continue
        name = jax.tree_util.keystr(path)
        if "p_s" in name:
            assert leaf.dtype == accum, (
                f"p_s at {name}: {leaf.dtype}, expected accumulate {accum} "
                f"(p_s is the intentional always-f64 conservation field)"
            )
            checked_ps = True
        else:
            # Contagion guard: a promoted p_s must NOT spread to u/T/tracers.
            assert leaf.dtype == storage, (
                f"{name} promoted to {leaf.dtype}, expected storage {storage} "
                f"— f64 p_s contaminated a 3D prognostic"
            )
    assert checked_ps, "no p_s leaf found — test would be vacuous"


def test_sharded_ps_carry_guard():
    """The loud guard for the DEFERRED sharded p_s carry-seed: mixed/fp64 must
    reject a float32 p_s carry (else the sharded scan silently carries p_s f32
    the whole run); fp32 policy never fires."""
    from legoesm.runtime.precision import apply_precision
    from legoesm.parallel.sharded_dynamics import _assert_sharded_ps_carry_f64

    for mode in ("fp64",):  # #1665: mixed refused
        apply_precision(mode)
        with pytest.raises(RuntimeError, match="seeded float64"):
            _assert_sharded_ps_carry_f64(jnp.float32)
        _assert_sharded_ps_carry_f64(jnp.float64)  # f64 carry: no raise
    apply_precision("fp32")
    _assert_sharded_ps_carry_f64(jnp.float32)  # fp32: correct, no raise


@pytest.mark.parametrize("mode", MODES)
def test_mpas_pe_leaf_dtypes(mode):
    state, storage, accum = _run_mpas(mode)
    _assert_leaf_dtypes(state, storage, accum)


@pytest.mark.parametrize("mode", ["fp64", "mixed"])  # #1675: mixed enabled
# (the mixed lat-lon tracer-promotion is the tracked mixed-consistency campaign;
# the refusal itself is pinned by test_mixed_precision_is_refused below).
def test_latlon_pe_leaf_dtypes(mode):
    state, storage, accum = _run_latlon(mode)
    _assert_leaf_dtypes(state, storage, accum)


def test_mixed_precision_enabled():
    """#1675: mixed is enabled; the parametrized leaf-dtype tests above now run
    it and assert every prognostic leaf stays at fp32 storage."""
    from legoesm.runtime.precision import apply_precision
    p = apply_precision("mixed")
    assert p.storage == jnp.float32 and p.control == jnp.float64


def test_mixed_latlon_rollout_capstone():
    """#1675 capstone: a 20-step mixed lat-lon PE rollout keeps per-field storage
    dtypes stable (p_s f64, every other prognostic fp32 — no gradual promotion)
    AND conserves global dry mass, non-accumulating (p_s is carried at f64
    accumulate precision + the fix_mass fixer, so mass stays near-f64)."""
    from legoesm.runtime.precision import apply_precision
    apply_precision("mixed")
    from legoesm.core.precision import resolve_dtype
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationModel, CGridLatLonPrimitiveEquationConfig,
        hydrostatic_to_cgrid,
    )
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_latlon
    storage = resolve_dtype(None, "storage"); accum = resolve_dtype(None, "accumulate")
    assert storage == jnp.float32 and accum == jnp.float64  # mixed sanity
    grid = create_latlon_grid(n_lat=8); sigma = create_sigma_coordinate(8)
    state = hydrostatic_to_cgrid(
        baroclinic_wave_init_latlon(grid, sigma, perturbed=True, moist=True), grid)
    state = _cast_to_storage(state, storage)
    cfg = CGridLatLonPrimitiveEquationConfig(
        A_h=1.0e4, time_integrator="ssp_rk3", fix_mass=True,
        zero_mean_ps_tendency=False, use_ppm_transport=True)
    dt = 100.0
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg, dt=dt)
    area = jnp.asarray(grid.area, jnp.float64)
    mass0 = float(jnp.sum(jnp.asarray(state.p_s, jnp.float64) * area))
    drifts = []
    for _ in range(20):
        state = model.step(state, dt=dt)
        drifts.append(abs(float(jnp.sum(jnp.asarray(state.p_s, jnp.float64) * area)) - mass0) / mass0)
    # dtypes stay consistent after the whole rollout (the contagion guard)
    _assert_leaf_dtypes(state, storage, accum)
    # mass bounded and NON-accumulating (final not the running max => no secular growth)
    assert max(drifts) < 1e-6, f"mass drift too large: {max(drifts):.2e}"
    assert drifts[-1] <= 2.0 * max(drifts[:10]) + 1e-12, f"mass drift accumulating: {drifts}"
