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

# Mesh cache: honour whatever the caller set; otherwise fall back to a path
# that exists on THIS machine.  The previous default was one cluster's absolute
# scratch path, so everywhere else the level-5 mesh build died in ``makedirs``
# with ``PermissionError: /work`` before a single assertion ran.
if not os.environ.get("LEGOESM_MESH_CACHE_DIR"):
    import tempfile as _tf
    for _cand in ("/work/bd1083/b309178/diffESM/legoesm_mesh_cache",):
        if os.path.isdir(os.path.dirname(_cand)):
            os.environ["LEGOESM_MESH_CACHE_DIR"] = _cand
            break
    else:
        os.environ["LEGOESM_MESH_CACHE_DIR"] = os.path.join(
            _tf.gettempdir(), "legoesm_mesh_cache")
os.environ.setdefault("LEGOESM_ALLOW_BIG_MESH_BUILD", "1")

import jax
import jax.numpy as jnp
import pytest

MODES = ["fp64", "mixed"]  # #1675 lifted the #1665 interim refusal

#: TWO steps, not one.  Adversarial review on #1675 traced the cube (CD-grid)
#: PE core keeping its bulk state at storage dtype after ONE step and promoting
#: it on the SECOND — the first step's promoted values only reach the bulk
#: state through the next step's tracer rescale.  A one-step gate is blind to
#: exactly the core it most needs to watch.
_N_STEPS = 2


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
    """MPAS PE — the NEGATIVE control, and it is labelled as one.

    Measured: this core does not promote in mixed even without the
    step-boundary re-cast (its tendencies are cast explicitly, the mass fixer
    touches only ``p_s``, and the positivity stage casts its weights to the
    tracer dtype).  So this row cannot go red by removing the MPAS
    finalization — it is here to show the re-cast does not BREAK a core that
    was already clean, and the discriminating rows are the lat-lon and cube
    ones.  Recorded so nobody reads a green MPAS row as coverage it is not.
    """
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
    state = state0
    for _ in range(_N_STEPS):
        state = model.step(state, 300.0)
    return state, storage, accum


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
    for _ in range(_N_STEPS):
        state = model.step(state, dt=dt)
    return state, storage, accum


def _run_cube(mode):
    """Cube (CD-grid) PE — the core a one-step gate missed entirely."""
    from legoesm.runtime.precision import apply_precision

    apply_precision(mode)
    from legoesm.core.precision import resolve_dtype
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    from legoesm.grids.vertical import standard_hybrid_levels
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationConfig,
        CDGridPrimitiveEquationModel,
        hydrostatic_to_fv3,
    )
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init

    storage = resolve_dtype(None, "storage")
    accum = resolve_dtype(None, "accumulate")
    grid = create_cubed_sphere(6)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(6)
    cfg = CDGridPrimitiveEquationConfig(fix_mass=True)
    state = _cast_to_storage(
        hydrostatic_to_fv3(held_suarez_init(grid, coord), cdgrid), storage)
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)
    for _ in range(_N_STEPS):
        state = model.step(state, 150.0)
    return state, storage, accum


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

    for mode in ("fp64", "mixed"):
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


@pytest.mark.parametrize("mode", MODES)
def test_cube_pe_leaf_dtypes(mode):
    state, storage, accum = _run_cube(mode)
    _assert_leaf_dtypes(state, storage, accum)


@pytest.mark.parametrize("mode", MODES)
# The lat-lon tracer promotion this used to be waived for is the #1675 defect:
# the mass fixer's float64 ``p_s`` flowed into the tracer mass rescale, so
# q_v/q_c/q_r all came back float64 from the eager step.  Fixed by the
# step-boundary ``finalize_to_storage`` re-cast; this parametrisation is the
# gate that keeps it fixed.
def test_latlon_pe_leaf_dtypes(mode):
    state, storage, accum = _run_latlon(mode)
    _assert_leaf_dtypes(state, storage, accum)


def test_mixed_requires_x64():
    """#1675: 'mixed' is selectable again, but only with x64 on.

    Its accumulate/control roles are float64; with x64 off JAX demotes them to
    float32, so the mode would silently run all-fp32 while reporting mixed.
    """
    import jax
    from legoesm.runtime.precision import apply_precision

    if jax.config.read("jax_enable_x64"):
        apply_precision("mixed")  # the supported combination: no raise
        from legoesm.core.precision import resolve_dtype
        assert resolve_dtype(None, "storage") == jnp.float32
        assert resolve_dtype(None, "accumulate") == jnp.float64
        apply_precision("fp64")
        return
    with pytest.raises(RuntimeError, match="x64"):
        apply_precision("mixed")
