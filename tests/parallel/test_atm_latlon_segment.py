"""M2b parity gate — compiled multi-step lat-band SPMD segments + band-SHARDED
geometry for the lat-lon C-grid hydrostatic atm.

Three merge-bar gates (2 virtual CPU devices, x64):

(i)   ``make_sharded_atm_latlon_segment(n_steps=10)`` — ONE compiled
      ``lax.scan`` — matches 10 sequential ``make_sharded_atm_latlon_step``
      calls to 1e-12, and 10 SERIAL single-device steps: to 1e-12 with
      centered transport (the band decomposition is exact there), and to the
      documented Stage-5 limited-FV-PPM cut-truncation bound with PPM on
      (``tests/parallel/test_atm_latlon_spmd_step.py`` module docstring — the
      2nd-order halo edge at cut rows is a boundary-order property of
      band-decomposed limited PPM, not an SPMD bug).

(ii)  the segment's IN-GRAPH finite scalar (``state_finite_scalar``: psum of
      per-band non-finite presence over ALL state leaves) agrees with a
      host-side isfinite of the gathered state — on a healthy run AND under
      synthetic NaN injections in T (band 0) and u-only (band 1), so the gate
      is provably non-vacuous and the cross-band psum is exercised.

(iii) the geometry-SHARDED step (``shard_geometry=True``: per-device band
      slice, ``P("lat")`` stacks) BIT-matches the replicated-geometry step
      (the historical default) — with and without the polar-filter mask
      stacks — and the layouts are REALLY different on device (sharding
      introspection), so the bit-match cannot pass vacuously.

Plus the production-lane wiring gate: ``run_atm_latlon_spmd(...,
compiled_segments=True)`` matches the per-step path (final state + status,
including the remainder segment and the BLOWUP status contract).

Runs on host CPU devices (``XLA_FLAGS=--xla_force_host_platform_device_count>=2``).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
    CGridLatLonHydrostaticState,
    CGridLatLonPrimitiveEquationConfig,
    CGridLatLonPrimitiveEquationModel,
    cgrid_to_hydrostatic,
)
from legoesm.atmosphere.dynamics.sharded_atm_latlon_step import (
    atm_latlon_geometry_bytes,
    gather_state_atm_latlon,
    make_sharded_atm_latlon_segment,
    make_sharded_atm_latlon_step,
    run_atm_latlon_spmd,
    shard_state_atm_latlon,
    state_finite_scalar,
)
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate

N_DEV = 2
N_LAT = 16        # divisible by N_DEV
N_LON = 16
NLEV = 4
DT = 100.0


@pytest.fixture(autouse=True)
def _restore_halo_backend():
    yield
    from legoesm.grids.halo import set_halo_backend
    set_halo_backend("local")


def _mesh():
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")
    return jax.sharding.Mesh(np.array(jax.devices()[:N_DEV]),
                             axis_names=("lat",))


def _model_and_state(use_ppm_transport=True, use_polar_filter=False):
    grid = create_latlon_grid(
        n_lat=N_LAT, n_lon=N_LON, radius=constants.R_earth,
        omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=NLEV)
    cfg = CGridLatLonPrimitiveEquationConfig(
        fix_mass=True,                 # exercises the psum'd mass fixer
        use_polar_filter=use_polar_filter,
        use_ppm_transport=use_ppm_transport,
        time_integrator="ssp_rk3",
    )
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg)
    rng = np.random.default_rng(31337)
    eps = 1.0e-3
    state = CGridLatLonHydrostaticState(
        u=jnp.asarray(eps * rng.standard_normal((N_LAT, N_LON + 1, NLEV))),
        v=jnp.asarray(eps * rng.standard_normal((N_LAT + 1, N_LON, NLEV))),
        T=jnp.asarray(300.0 + eps * rng.standard_normal((N_LAT, N_LON, NLEV))),
        p_s=jnp.asarray(1.0e5 + 10.0 * rng.standard_normal((N_LAT, N_LON))),
        phis=jnp.zeros((N_LAT, N_LON)),
    )
    # Pole-wall v (the state the model maintains: v == 0 at both poles), so
    # the v_lower drop + gather re-append round-trips.
    v0 = np.array(state.v)
    v0[0] = 0.0
    v0[-1] = 0.0
    return model, state._replace(v=jnp.asarray(v0))


def _fields(state):
    return {f: np.asarray(getattr(state, f)) for f in ("u", "v", "T", "p_s")}


# ==============================================================================
# (i) segment == sequential sharded steps == serial
# ==============================================================================

@pytest.mark.parametrize("use_ppm", [False, True])
def test_segment_matches_sequential_and_serial(use_ppm):
    mesh = _mesh()
    n_steps = 10
    seg_model, c0 = _model_and_state(use_ppm_transport=use_ppm)
    seq_model, _ = _model_and_state(use_ppm_transport=use_ppm)
    ser_model, _ = _model_and_state(use_ppm_transport=use_ppm)

    # (a) 10 SERIAL single-device steps (the truth trajectory).
    s = c0
    for _ in range(n_steps):
        s, _ = ser_model._step_cgrid(s, DT)
    serial_out = _fields(s)

    # (b) 10 sequential per-step sharded calls (the historical SPMD path,
    # replicated geometry).
    step = make_sharded_atm_latlon_step(seq_model, mesh)
    sc = shard_state_atm_latlon(c0, mesh)
    for _ in range(n_steps):
        sc = step(sc, DT)
    seq_out = _fields(gather_state_atm_latlon(sc, mesh))

    # (c) ONE compiled segment of 10 steps (scan inside one jitted shard_map,
    # band-SHARDED geometry).
    seg = make_sharded_atm_latlon_segment(seg_model, mesh, n_steps)
    c_seg, ok = seg(shard_state_atm_latlon(c0, mesh), DT)
    assert bool(ok), "segment finite scalar is False on a healthy run"
    # Non-vacuity: the segment output is genuinely lat-sharded across the mesh.
    assert c_seg.T.sharding.num_devices == N_DEV
    seg_out = _fields(gather_state_atm_latlon(c_seg, mesh))

    # segment vs sequential sharded: same band numerics + decomposition; only
    # scan-vs-unrolled compilation differs -> 1e-12.
    for f in ("u", "v", "T", "p_s"):
        np.testing.assert_allclose(
            seg_out[f], seq_out[f], rtol=1e-12, atol=1e-12,
            err_msg=(f"compiled segment diverged from sequential sharded "
                     f"steps in '{f}' — the scanned band body is not the "
                     f"per-step body"))

    # segment vs SERIAL: exact decomposition without PPM; the documented
    # Stage-5 limited-FV-PPM cut-row truncation bound with PPM on.
    rtol, atol = ((1e-6, 1e-9) if use_ppm else (1e-12, 1e-12))
    for f in ("u", "v", "T", "p_s"):
        np.testing.assert_allclose(
            seg_out[f], serial_out[f], rtol=rtol, atol=atol,
            err_msg=(f"compiled segment (ppm={use_ppm}) diverged from the "
                     f"serial single-device trajectory in '{f}' beyond the "
                     f"documented bound — a real decomposition/scan bug"))


def test_segment_single_device_matches_serial_loop():
    """mesh=None twin: jit(scan) over the serial C-grid step == the per-step
    _step_cgrid loop."""
    model, c0 = _model_and_state()
    ser_model, _ = _model_and_state()
    seg = make_sharded_atm_latlon_segment(model, None, 5)
    out, ok = seg(c0, DT)
    assert bool(ok)
    s = c0
    for _ in range(5):
        s, _ = ser_model._step_cgrid(s, DT)
    a, b = _fields(out), _fields(s)
    for f in ("u", "v", "T", "p_s"):
        np.testing.assert_allclose(
            a[f], b[f], rtol=1e-12, atol=1e-13,
            err_msg=f"mesh=None segment diverged from the serial loop in '{f}'")


def test_segment_with_held_suarez_matches_sequential():
    """The production physics envelope (stateless Held-Suarez) threads through
    the compiled scan identically to the per-step path."""
    from legoesm.atmosphere.held_suarez import held_suarez_forcing_latlon
    mesh = _mesh()
    seg_model, c0 = _model_and_state()
    seq_model, _ = _model_and_state()
    n_steps = 5

    seg = make_sharded_atm_latlon_segment(
        seg_model, mesh, n_steps, physics_fn=held_suarez_forcing_latlon)
    c_seg, ok = seg(shard_state_atm_latlon(c0, mesh), DT)
    assert bool(ok)

    step = make_sharded_atm_latlon_step(
        seq_model, mesh, physics_fn=held_suarez_forcing_latlon)
    sc = shard_state_atm_latlon(c0, mesh)
    for _ in range(n_steps):
        sc = step(sc, DT)

    a = _fields(gather_state_atm_latlon(c_seg, mesh))
    b = _fields(gather_state_atm_latlon(sc, mesh))
    for f in ("u", "v", "T", "p_s"):
        np.testing.assert_allclose(
            a[f], b[f], rtol=1e-12, atol=1e-12,
            err_msg=f"HS segment diverged from HS per-step path in '{f}'")


# ==============================================================================
# (ii) in-graph finite scalar == host-side isfinite of the gathered state
# ==============================================================================

def _host_all_finite(c_gathered) -> bool:
    return all(bool(np.isfinite(np.asarray(leaf)).all())
               for leaf in jax.tree.leaves(c_gathered))


def test_segment_finite_scalar_matches_host_isfinite():
    mesh = _mesh()
    model, c0 = _model_and_state()
    seg = make_sharded_atm_latlon_segment(model, mesh, 3)

    # Healthy run: scalar True, host agrees.
    c_out, ok = seg(shard_state_atm_latlon(c0, mesh), DT)
    assert bool(ok) is True
    assert _host_all_finite(gather_state_atm_latlon(c_out, mesh)) is True

    # Synthetic violations (non-vacuous gate): NaN seeded in a band-0 T cell
    # and in a band-1 u cell (u-only exercises the all-leaves superset AND the
    # cross-band psum — the bad band is NOT the one whose scalar the host
    # would trivially see).
    for field, idx in (("T", (3, 5, 0)), ("u", (9, 2, 1))):
        arr = np.array(np.asarray(getattr(c0, field)))
        arr[idx] = np.nan
        c_bad = c0._replace(**{field: jnp.asarray(arr)})
        c_out_b, ok_b = seg(shard_state_atm_latlon(c_bad, mesh), DT)
        host_ok = _host_all_finite(gather_state_atm_latlon(c_out_b, mesh))
        assert bool(ok_b) is False, (
            f"in-graph finite scalar missed a NaN seeded in {field}[{idx}]")
        assert host_ok is False
        assert bool(ok_b) == host_ok


def test_state_finite_scalar_serial_and_inf():
    """The scalar reducer itself (no mesh): catches Inf as well as NaN."""
    _, c0 = _model_and_state()
    assert bool(state_finite_scalar(c0)) is True
    bad = c0._replace(p_s=c0.p_s.at[0, 0].set(jnp.inf))
    assert bool(state_finite_scalar(bad)) is False


# ==============================================================================
# (iii) geometry-sharded step bit-matches the replicated-geometry step
# ==============================================================================

@pytest.mark.parametrize("use_polar_filter", [False, True])
def test_geometry_sharded_step_bitmatches_replicated(use_polar_filter):
    mesh = _mesh()
    model_r, c0 = _model_and_state(use_polar_filter=use_polar_filter)
    model_s, _ = _model_and_state(use_polar_filter=use_polar_filter)

    step_rep = make_sharded_atm_latlon_step(model_r, mesh)  # default: replicated
    step_shd = make_sharded_atm_latlon_step(model_s, mesh, shard_geometry=True)

    # Non-vacuity: the two layouts are REALLY different on device.
    assert step_rep._geom_stacks["area"].sharding.is_fully_replicated
    assert not step_shd._geom_stacks["area"].sharding.is_fully_replicated
    if use_polar_filter:
        assert "__polar_mask" in step_shd._geom_stacks
        assert not (step_shd._geom_stacks["__polar_mask"]
                    .sharding.is_fully_replicated)

    sc_r = shard_state_atm_latlon(c0, mesh)
    sc_s = shard_state_atm_latlon(c0, mesh)
    for _ in range(3):
        sc_r = step_rep(sc_r, DT)
        sc_s = step_shd(sc_s, DT)
    a = _fields(gather_state_atm_latlon(sc_r, mesh))
    b = _fields(gather_state_atm_latlon(sc_s, mesh))
    for f in ("u", "v", "T", "p_s"):
        np.testing.assert_array_equal(
            a[f], b[f],
            err_msg=(f"geometry-sharded step is not BIT-identical to the "
                     f"replicated-geometry step in '{f}' "
                     f"(polar_filter={use_polar_filter})"))


def test_atm_latlon_geometry_bytes_shrinks_by_ndev():
    """The honest residency numbers: band-sharding is exactly 1/n_dev of the
    replicated layout (uniform bands), and the 2-D fields dominate."""
    grid = create_latlon_grid(
        n_lat=N_LAT, n_lon=N_LON, radius=constants.R_earth,
        omega=constants.Omega)
    geo = atm_latlon_geometry_bytes(grid, N_DEV)
    assert geo["sharded_per_device_bytes"] * N_DEV == \
        geo["replicated_per_device_bytes"]
    assert geo["sharded_per_device_bytes"] < geo["replicated_per_device_bytes"]
    # At least the five (n_lat, n_lon) f64 fields (lat2d, lon2d, f, dx, area).
    assert geo["replicated_per_device_bytes"] >= N_LAT * N_LON * 8 * 5


# ==============================================================================
# production-lane wiring: run_atm_latlon_spmd(compiled_segments=True)
# ==============================================================================

def test_run_atm_latlon_spmd_compiled_segments_matches_per_step():
    """7 steps in 3-step segments (3+3+1: exercises the remainder-length
    compile) — final state matches the per-step path to 1e-12, same status."""
    mesh = _mesh()
    model_a, c0 = _model_and_state()
    model_b, _ = _model_and_state()
    hs0 = cgrid_to_hydrostatic(c0, model_a.grid)

    hs_ref, st_ref = run_atm_latlon_spmd(
        model_a, mesh, hs0, DT, 7, segment_steps=3)
    hs_new, st_new = run_atm_latlon_spmd(
        model_b, mesh, hs0, DT, 7, segment_steps=3, compiled_segments=True)

    assert st_ref == "COMPLETED"
    assert st_new == "COMPLETED"
    for f in ("u", "v", "T", "p_s"):
        np.testing.assert_allclose(
            np.asarray(getattr(hs_new, f).data),
            np.asarray(getattr(hs_ref, f).data),
            rtol=1e-12, atol=1e-12,
            err_msg=f"compiled_segments run diverged from per-step in '{f}'")


def test_run_atm_latlon_spmd_compiled_segments_blowup_status():
    """A NaN'd IC must report the SAME 'BLOWUP at step N' boundary on both
    paths (the in-graph scalar is a superset of the gathered p_s/T check)."""
    mesh = _mesh()
    model_a, c0 = _model_and_state()
    model_b, _ = _model_and_state()
    arr = np.array(np.asarray(c0.T))
    arr[0, 0, 0] = np.nan
    hs_bad = cgrid_to_hydrostatic(c0._replace(T=jnp.asarray(arr)),
                                  model_a.grid)

    _, st_old = run_atm_latlon_spmd(model_a, mesh, hs_bad, DT, 6,
                                    segment_steps=3)
    _, st_new = run_atm_latlon_spmd(model_b, mesh, hs_bad, DT, 6,
                                    segment_steps=3, compiled_segments=True)
    assert st_old == "BLOWUP at step 3"
    assert st_new == "BLOWUP at step 3"


def test_run_atm_latlon_spmd_compiled_segments_on_segment_cadence():
    """The on_segment callback still fires at every boundary with gathered
    cell-centered states (output contract preserved)."""
    mesh = _mesh()
    model, c0 = _model_and_state()
    hs0 = cgrid_to_hydrostatic(c0, model.grid)
    seen = []
    _, status = run_atm_latlon_spmd(
        model, mesh, hs0, DT, 6, segment_steps=2, compiled_segments=True,
        on_segment=lambda hs, done: seen.append(
            (done, bool(np.isfinite(np.asarray(hs.T.data)).all()))))
    assert status == "COMPLETED"
    assert seen == [(2, True), (4, True), (6, True)]


# ==============================================================================
# dispatch hardening
# ==============================================================================

def test_segment_rejects_bad_nsteps_and_stateful_carry():
    mesh = _mesh()
    model, c0 = _model_and_state()
    with pytest.raises(ValueError, match="n_steps"):
        make_sharded_atm_latlon_segment(model, mesh, 0)
    with pytest.raises(ValueError, match="n_steps"):
        make_sharded_atm_latlon_segment(model, None, 0)

    seg = make_sharded_atm_latlon_segment(model, mesh, 2)
    with pytest.raises(NotImplementedError, match="PhysicsState"):
        seg(shard_state_atm_latlon(c0, mesh), DT, phys_state=object())
    seg0 = make_sharded_atm_latlon_segment(model, None, 2)
    with pytest.raises(NotImplementedError, match="PhysicsState"):
        seg0(c0, DT, phys_state=object())


def test_segment_factory_rejects_anchor_mass():
    """The shared guard also protects the segment factory (a band-local mass
    anchor would silently mis-anchor under SPMD)."""
    mesh = _mesh()
    grid = create_latlon_grid(
        n_lat=N_LAT, n_lon=N_LON, radius=constants.R_earth,
        omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=NLEV)
    cfg = CGridLatLonPrimitiveEquationConfig(anchor_mass_to_initial=True)
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg)
    with pytest.raises(NotImplementedError, match="anchor_mass_to_initial"):
        make_sharded_atm_latlon_segment(model, mesh, 2)
