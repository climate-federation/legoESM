"""M3a parity gates — native 2-D ("lat", "lon") tiling for the atmosphere
lat-lon SPMD step.

The 1-D lat-band decomposition's halo perimeter is the CONSTANT n_lon per cut
(independent of the device count) — the term that caps band scaling.  M3a
adds a native 2-D tiling: periodic longitude as a cyclic ring ppermute,
staggered ownership for BOTH staggers (v_lower rows + u_left columns), the
EXACT serial 180-deg pole fold under a lon split (lon-ring all_gather at the
pole tiles), and topology-aware (p_lat, p_lon) selection.  The 1-D band path
stays the default and byte-identical.

Merge-bar gates (4 virtual CPU devices, x64):

(1)  PAD PARITY — ``make_latlon_2d_pad_body`` reproduces, per tile, the
     SERIAL ``pad_halo_latlon_local`` window BIT-exactly ((2,2) and (4,1);
     halo 1 and 2; scalar and vector fold).  This is the direct proof of the
     lon-ring exchange, the two-pass corner composition, and the all_gather
     pole fold.

(2)  TENDENCY — the (2,2)-tile tendency is BIT-tight vs serial for momentum
     + continuity; dT carries only the documented Stage-5 limited-FV-PPM
     boundary-order truncation, now at cut-adjacent lat ROWS and lon COLUMNS
     (the same halo-2 edge-order property, rotated; the lon flux metric
     ``hy`` is per-lat so a lon cut adds no metric error).

(3)  STEP — the integrated (2,2) 2-D step matches serial at the Stage-5
     bound, with SEAM-SPECIFIC assertions: the u periodic-seam column, the
     u/v values along every tile cut, and the v pole rows.

(4)  DEGENERACY — the (4,1) 2-D step is BIT-IDENTICAL to the existing 1-D
     band step (every lon-ring op takes its static local branch).

(5)  SEGMENT — the compiled 10-step 2-D ``lax.scan`` matches 10 sequential
     2-D steps to 1e-12 and serial at the Stage-5 bound; the in-graph finite
     scalar (psum over BOTH axes) is exercised healthy + NaN-injected; a
     mixed-precision (f32) IC exercises the dtype-fixed-point unroll.

(6)  CHOOSER — perimeter minimization, the p_lon == 1 band preference,
     divisibility/min-tile feasibility, and the loud no-factorization error.

Runs on host CPU devices (``XLA_FLAGS=--xla_force_host_platform_device_count=4``).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from legoesm import constants
from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
    CGridLatLonHydrostaticState,
    CGridLatLonPrimitiveEquationConfig,
    CGridLatLonPrimitiveEquationModel,
)
from legoesm.atmosphere.dynamics.sharded_atm_latlon_step import (
    build_tile_grids_atm_2d,
    gather_state_atm_latlon,
    gather_state_atm_latlon_2d,
    make_sharded_atm_latlon_segment_2d,
    make_sharded_atm_latlon_step,
    make_sharded_atm_latlon_step_2d,
    shard_state_atm_latlon,
    shard_state_atm_latlon_2d,
    tile_spec,
)
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.parallel.latlon_spmd import choose_latlon_2d_topology
from legoesm.parallel.shard_map_compat import shard_map

N_DEV = 4
N_LAT = 16
N_LON = 16
NLEV = 4
DT = 100.0


@pytest.fixture(autouse=True)
def _restore_halo_backend():
    yield
    from legoesm.grids.halo import set_halo_backend, set_spmd_mesh
    set_spmd_mesh(None)
    set_halo_backend("local")


def _mesh2d(p_lat: int, p_lon: int):
    if len(jax.devices()) < p_lat * p_lon:
        pytest.skip(
            f"needs --xla_force_host_platform_device_count={p_lat * p_lon}")
    devs = np.array(jax.devices()[:p_lat * p_lon]).reshape(p_lat, p_lon)
    return jax.sharding.Mesh(devs, axis_names=("lat", "lon"))


def _mesh1d(n: int = N_DEV):
    if len(jax.devices()) < n:
        pytest.skip(f"needs --xla_force_host_platform_device_count={n}")
    return jax.sharding.Mesh(np.array(jax.devices()[:n]),
                             axis_names=("lat",))


def _model_and_state(use_polar_filter=False, use_ppm_transport=True):
    grid = create_latlon_grid(
        n_lat=N_LAT, n_lon=N_LON, radius=constants.R_earth,
        omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=NLEV)
    cfg = CGridLatLonPrimitiveEquationConfig(
        fix_mass=True,                 # exercises the ("lat","lon") psum
        use_polar_filter=use_polar_filter,
        use_ppm_transport=use_ppm_transport,
        time_integrator="ssp_rk3",
    )
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg)
    rng = np.random.default_rng(31337)
    eps = 1.0e-3
    u0 = eps * rng.standard_normal((N_LAT, N_LON + 1, NLEV))
    v0 = eps * rng.standard_normal((N_LAT + 1, N_LON, NLEV))
    # State invariants the model maintains (and the sharded layouts drop +
    # reconstruct): v == 0 at both pole walls; u's last column is the
    # periodic seam closure (u[:, n_lon] == u[:, 0]).
    v0[0] = 0.0
    v0[-1] = 0.0
    u0[:, -1] = u0[:, 0]
    state = CGridLatLonHydrostaticState(
        u=jnp.asarray(u0),
        v=jnp.asarray(v0),
        T=jnp.asarray(300.0 + eps * rng.standard_normal((N_LAT, N_LON, NLEV))),
        p_s=jnp.asarray(1.0e5 + 10.0 * rng.standard_normal((N_LAT, N_LON))),
        phis=jnp.zeros((N_LAT, N_LON)),
    )
    return model, state


def _serial_steps(model, state, n_steps, dt=DT, physics_fn=None):
    s = state
    for _ in range(n_steps):
        s, _ = model._step_cgrid(s, dt, target_mass=None,
                                 physics_fn=physics_fn)
    return s


def _fields(state):
    return {f: np.asarray(getattr(state, f)) for f in ("u", "v", "T", "p_s")}


# ==============================================================================
# (1) pad-body parity: tile block == serial padded window (fold/corners/ring)
# ==============================================================================

@pytest.mark.parametrize("p_lat,p_lon", [(2, 2), (4, 1), (1, 4)])
@pytest.mark.parametrize("halo", [1, 2])
@pytest.mark.parametrize("negate", [False, True])
def test_2d_pad_body_matches_serial_window(p_lat, p_lon, halo, negate):
    """Every tile's padded block from ``make_latlon_2d_pad_body`` must equal
    the SERIAL local pad's window (rows [r*nl, r*nl+nl+2h), cols
    [c*w, c*w+w+2h)) BIT-exactly — the direct proof of the lon ring exchange,
    the lat-then-lon corner composition, and the all_gather'd 180-deg pole
    fold (including the serial fold's padded-roll quirks)."""
    from legoesm.grids.halo_latlon import (
        pad_halo_latlon_3d_local, pad_halo_latlon_vector_3d_local)
    from legoesm.parallel.latlon_spmd import make_latlon_2d_pad_body

    mesh = _mesh2d(p_lat, p_lon)
    rng = np.random.default_rng(97)
    field = jnp.asarray(rng.standard_normal((N_LAT, N_LON, NLEV)))
    serial_pad = np.asarray(
        pad_halo_latlon_vector_3d_local(field, halo) if negate
        else pad_halo_latlon_3d_local(field, halo))

    body = make_latlon_2d_pad_body(mesh, halo=halo, negate=negate)
    fn = jax.jit(shard_map(
        body, mesh=mesh,
        in_specs=P("lat", "lon", None),
        out_specs=P("lat", "lon", None), check_vma=False))
    sharded = jax.device_put(
        field, NamedSharding(mesh, P("lat", "lon", None)))
    out = np.asarray(jax.device_put(fn(sharded), NamedSharding(mesh, P())))

    nl, w = N_LAT // p_lat, N_LON // p_lon
    hl, hw = nl + 2 * halo, w + 2 * halo
    for r in range(p_lat):
        for c in range(p_lon):
            tile_block = out[r * hl:(r + 1) * hl, c * hw:(c + 1) * hw]
            window = serial_pad[r * nl:r * nl + hl, c * w:c * w + hw]
            np.testing.assert_array_equal(
                tile_block, window,
                err_msg=(
                    f"2-D pad tile ({r},{c}) [{p_lat}x{p_lon}, halo={halo}, "
                    f"negate={negate}] != serial pad window — lon ring / "
                    f"corner composition / pole fold defect."))


# ==============================================================================
# (2) tendency parity on the (2,2) tiling
# ==============================================================================

def _serial_and_tile_tendency():
    """Serial tendencies and the (2,2)-tile shard_map body's tendencies,
    gathered to global (u as u_left, v as v_lower)."""
    from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
        cgrid_latlon_hydrostatic_tendencies)
    from legoesm.atmosphere.dynamics.sharded_atm_latlon_step import (
        _build_geometry_stacks_2d, atm_grid_array_field_names)
    from legoesm.parallel.latlon_spmd import (
        activate_latlon_spmd_halo, deactivate_latlon_spmd_halo,
        latlon_band_perms, reconstruct_uface_left, reconstruct_vface_lower,
        to_uface_left, to_vface_lower)

    p_lat, p_lon = 2, 2
    mesh = _mesh2d(p_lat, p_lon)
    model, state = _model_and_state()
    grid, sigma, cfg = model.grid, model.sigma_coord, model.config

    du_s, dv_s, dT_s, dps_s, _ = cgrid_latlon_hydrostatic_tendencies(
        state, grid, sigma, cfg)
    serial = tuple(np.asarray(x) for x in (du_s, dv_s, dT_s, dps_s))

    template, afn, stacks, stacks_spec = _build_geometry_stacks_2d(
        model, mesh, p_lat, p_lon, shard_geometry=True)
    perm_north, _ = latlon_band_perms(p_lat)

    def _body(sl, st):
        tg = template._replace(**{n: st[n][0, 0] for n in afn})
        vf = reconstruct_vface_lower(sl.v, "lat", perm_north)
        uf = reconstruct_uface_left(sl.u, "lon", p_lon)
        du, dv, dT, dps, _ = cgrid_latlon_hydrostatic_tendencies(
            sl._replace(u=uf, v=vf), tg, sigma, cfg)
        return to_uface_left(du), to_vface_lower(dv), dT, dps

    sc = shard_state_atm_latlon_2d(state, mesh)
    in_spec = jax.tree.map(tile_spec, sc)
    sp3 = P("lat", "lon", None)
    sp2 = P("lat", "lon")
    fn = shard_map(
        _body, mesh=mesh, in_specs=(in_spec, stacks_spec),
        out_specs=(sp3, sp3, sp3, sp2), check_vma=False)
    activate_latlon_spmd_halo(mesh)
    try:
        out = fn(sc, stacks)
    finally:
        deactivate_latlon_spmd_halo()
    g = lambda x: np.asarray(jax.device_put(x, NamedSharding(mesh, P())))
    dul_b, dvl_b, dT_b, dps_b = (g(x) for x in out)
    return serial, (dul_b, dvl_b, dT_b, dps_b), (p_lat, p_lon)


def test_2d_tendency_matches_serial():
    """Momentum + continuity BIT-tight on the (2,2) tiling; dT carries only
    the documented limited-FV-PPM halo-2 boundary-order truncation at
    cut-adjacent lat rows AND lon columns."""
    (du_s, dv_s, dT_s, dps_s), (dul_b, dvl_b, dT_b, dps_b), (p_lat, p_lon) = (
        _serial_and_tile_tendency())

    # Layout invariants that justify the dropped stagger slots: the serial
    # tendency preserves the u periodic seam and the v pole walls.
    np.testing.assert_array_equal(
        du_s[:, -1], du_s[:, 0],
        err_msg="serial du seam identity broke — u_left carry is unsound.")
    np.testing.assert_array_equal(dv_s[0], np.zeros_like(dv_s[0]))
    np.testing.assert_array_equal(dv_s[-1], np.zeros_like(dv_s[-1]))

    for name, a, b in (("du_dt", dul_b, du_s[:, :N_LON]),
                       ("dv_dt", dvl_b, dv_s[:N_LAT]),
                       ("dp_s_dt", dps_b, dps_s)):
        np.testing.assert_allclose(
            a, b, rtol=1e-10, atol=1e-12,
            err_msg=(
                f"atm 2-D tile tendency '{name}' diverged from serial beyond "
                f"fp64 precision — a real tile-decomposition bug (lon ring, "
                f"u seam, Coriolis/vertex at cuts, geometry, or the "
                f"('lat','lon') mass psum)."))

    # dT: bit-tight away from cuts; bounded truncation at cut-adjacent lat
    # rows and lon columns (the wrap cut cols {n_lon-1, 0} included — the
    # tile reconstruction is halo-2 there while serial sees the full circle).
    nl, w = N_LAT // p_lat, N_LON // p_lon
    cut_rows = sorted({r * nl - 1 for r in range(1, p_lat)}
                      | {r * nl for r in range(1, p_lat)})
    cut_cols = sorted({(c * w - 1) % N_LON for c in range(p_lon)}
                      | {c * w for c in range(p_lon)})
    int_rows = [j for j in range(N_LAT) if j not in cut_rows]
    int_cols = [j for j in range(N_LON) if j not in cut_cols]
    np.testing.assert_allclose(
        dT_b[np.ix_(int_rows, int_cols)], dT_s[np.ix_(int_rows, int_cols)],
        rtol=1e-10, atol=1e-12,
        err_msg="dT_dt diverged at an INTERIOR (non-cut) cell — real bug.")
    resid = float(np.max(np.abs(dT_b - dT_s)))
    assert resid < 1e-9, (
        f"cut-adjacent dT_dt residual {resid:.2e} exceeds the FV-PPM halo-2 "
        f"truncation floor by too much — not the boundary-order reduction.")


# ==============================================================================
# (3) integrated step parity + explicit staggered-seam checks
# ==============================================================================

def test_2d_step_matches_serial_with_seam_checks():
    mesh = _mesh2d(2, 2)
    model, state = _model_and_state()
    n_steps = 3

    serial_out = _serial_steps(model, state, n_steps)

    step2d = make_sharded_atm_latlon_step_2d(model, mesh)
    sc = shard_state_atm_latlon_2d(state, mesh)
    for _ in range(n_steps):
        sc = step2d(sc, DT)
    out = gather_state_atm_latlon_2d(sc, mesh)

    a, b = _fields(out), _fields(serial_out)
    for f in ("u", "v", "T", "p_s"):
        assert a[f].shape == b[f].shape, f"{f} {a[f].shape} vs {b[f].shape}"
        np.testing.assert_allclose(
            a[f], b[f], rtol=1e-6, atol=1e-9,
            err_msg=f"2-D (2,2) step diverged from serial in '{f}' beyond "
                    f"the FV-PPM cut-truncation bound.")

    # Staggered-seam gate: the u periodic-seam column, the u columns at the
    # lon tile cut, the v rows at the lat tile cut, and the v pole rows —
    # compared against serial AT THE SEAMS SPECIFICALLY.
    w, nl = N_LON // 2, N_LAT // 2
    np.testing.assert_array_equal(
        a["u"][:, -1], a["u"][:, 0],
        err_msg="gathered u broke the periodic seam identity "
                "u[:, n_lon] == u[:, 0].")
    for col in (0, w, N_LON):
        np.testing.assert_allclose(
            a["u"][:, col], b["u"][:, col], rtol=1e-6, atol=1e-9,
            err_msg=f"u seam/cut column {col} diverged — u_left ownership / "
                    f"lon-ring reconstruction defect.")
    for row in (0, nl, N_LAT):
        np.testing.assert_allclose(
            a["v"][row], b["v"][row], rtol=1e-6, atol=1e-9,
            err_msg=f"v cut/pole row {row} diverged — v_lower ownership / "
                    f"lat reconstruction defect.")
    np.testing.assert_array_equal(
        a["v"][0], np.zeros_like(a["v"][0]),
        err_msg="south pole v wall is nonzero after the 2-D step.")
    np.testing.assert_array_equal(
        a["v"][-1], np.zeros_like(a["v"][-1]),
        err_msg="north pole v wall is nonzero after the 2-D step.")

    # Non-vacuity: the tiling genuinely ran (the PPM cut truncation makes the
    # trajectory differ from serial above the fp64 floor).
    assert float(np.max(np.abs(a["T"] - b["T"]))) > 1e-13, (
        "2-D T is bit-identical to serial — sharding collapsed to "
        "single-device; this gate would be vacuous.")


def test_2d_state_layout_roundtrip():
    """shard_2d -> gather_2d is bit-exact given the layout invariants
    (v pole walls zero, u seam column == column 0)."""
    mesh = _mesh2d(2, 2)
    _, state = _model_and_state()
    back = gather_state_atm_latlon_2d(shard_state_atm_latlon_2d(state, mesh),
                                      mesh)
    for f in ("u", "v", "T", "p_s", "phis"):
        np.testing.assert_array_equal(
            np.asarray(getattr(back, f)), np.asarray(getattr(state, f)),
            err_msg=f"2-D shard/gather round-trip changed '{f}'.")


def test_2d_step_with_held_suarez_matches_serial():
    """Stateless column-local physics threads through the 2-D step (evaluated
    on the TILE geometry) and matches the serial physics loop."""
    from legoesm.atmosphere.held_suarez import held_suarez_forcing_latlon
    mesh = _mesh2d(2, 2)
    model, c_state = _model_and_state()
    n_steps = 3
    phys = held_suarez_forcing_latlon

    serial_out = _serial_steps(model, c_state, n_steps, physics_fn=phys)

    step2d = make_sharded_atm_latlon_step_2d(model, mesh, physics_fn=phys)
    sc = shard_state_atm_latlon_2d(c_state, mesh)
    for _ in range(n_steps):
        sc = step2d(sc, DT)
    out = gather_state_atm_latlon_2d(sc, mesh)

    # Non-vacuity: physics actually altered the trajectory.
    dyn_only = _serial_steps(model, c_state, n_steps)
    assert float(np.max(np.abs(
        np.asarray(serial_out.T) - np.asarray(dyn_only.T)))) > 1e-6

    for f in ("u", "v", "T", "p_s"):
        np.testing.assert_allclose(
            np.asarray(getattr(out, f)), np.asarray(getattr(serial_out, f)),
            rtol=1e-6, atol=1e-9,
            err_msg=f"2-D Held-Suarez step diverged from serial in '{f}' — "
                    f"tile physics coupling bug.")


# ==============================================================================
# (4) (4,1) degenerate mesh == the existing 1-D band step, BIT-IDENTICAL
# ==============================================================================

def test_2d_degenerate_41_bitmatches_1d_band():
    """On a (4,1) mesh every lon-ring op takes its static local branch, so
    the 2-D step must reproduce the 1-D band step BIT-FOR-BIT (same ops, same
    order — pure data-movement differences only).  Both factories run with
    shard_geometry=True (the geometry VALUES are identical either way, gated
    by the M2b bit-match)."""
    mesh1 = _mesh1d(N_DEV)
    mesh2 = _mesh2d(N_DEV, 1)
    model, state = _model_and_state()
    n_steps = 3

    step1d = make_sharded_atm_latlon_step(model, mesh1, shard_geometry=True)
    s1 = shard_state_atm_latlon(state, mesh1)
    for _ in range(n_steps):
        s1 = step1d(s1, DT)
    out1 = gather_state_atm_latlon(s1, mesh1)

    step2d = make_sharded_atm_latlon_step_2d(model, mesh2,
                                             shard_geometry=True)
    s2 = shard_state_atm_latlon_2d(state, mesh2)
    for _ in range(n_steps):
        s2 = step2d(s2, DT)
    out2 = gather_state_atm_latlon_2d(s2, mesh2)

    for f in ("u", "v", "T", "p_s"):
        np.testing.assert_array_equal(
            np.asarray(getattr(out2, f)), np.asarray(getattr(out1, f)),
            err_msg=(
                f"(4,1) 2-D step is not BIT-identical to the 1-D band step "
                f"in '{f}' — the degenerate lon axis changed the numerics "
                f"(a static-branch or exchange-order defect)."))


# ==============================================================================
# (5) compiled 10-step segment: scan stability, finite scalar, f32 unroll
# ==============================================================================

def test_2d_segment_matches_sequential_and_serial():
    mesh = _mesh2d(2, 2)
    model, state = _model_and_state()
    n_steps = 10

    seg = make_sharded_atm_latlon_segment_2d(model, mesh, n_steps)
    sc0 = shard_state_atm_latlon_2d(state, mesh)
    out_seg, ok = seg(sc0, DT)
    assert bool(ok), "healthy 10-step 2-D segment reported non-finite."
    seg_g = _fields(gather_state_atm_latlon_2d(out_seg, mesh))

    step2d = make_sharded_atm_latlon_step_2d(model, mesh)
    sc = sc0
    for _ in range(n_steps):
        sc = step2d(sc, DT)
    seq_g = _fields(gather_state_atm_latlon_2d(sc, mesh))

    serial_g = _fields(_serial_steps(model, state, n_steps))

    for f in ("u", "v", "T", "p_s"):
        np.testing.assert_allclose(
            seg_g[f], seq_g[f], rtol=1e-12, atol=1e-12,
            err_msg=f"compiled 2-D segment != sequential 2-D steps in '{f}'.")
        np.testing.assert_allclose(
            seg_g[f], serial_g[f], rtol=1e-6, atol=1e-8,
            err_msg=f"compiled 2-D segment diverged from serial in '{f}' "
                    f"beyond the integrated FV-PPM cut-truncation bound.")


def test_2d_segment_finite_scalar_detects_nan():
    """The psum-over-('lat','lon') finite scalar: True on a healthy state,
    False when a NaN is injected into a SINGLE tile's interior (non-vacuous:
    the cross-tile psum must carry the presence to every device)."""
    mesh = _mesh2d(2, 2)
    model, state = _model_and_state()
    seg = make_sharded_atm_latlon_segment_2d(model, mesh, 1)

    _, ok = seg(shard_state_atm_latlon_2d(state, mesh), DT)
    assert bool(ok)

    T_bad = np.asarray(state.T).copy()
    T_bad[N_LAT - 2, N_LON - 2, 0] = np.nan   # north-east tile interior
    bad = state._replace(T=jnp.asarray(T_bad))
    _, ok_bad = seg(shard_state_atm_latlon_2d(bad, mesh), DT)
    assert not bool(ok_bad), (
        "NaN in one tile was not reported by the ('lat','lon') psum'd "
        "finite scalar.")


def test_2d_segment_f32_ic_dtype_fixed_point():
    """A mixed-precision (f32) IC promotes over the first step(s) under x64;
    the segment's dtype-fixed-point unroll must absorb it exactly as the
    per-step Python loop does (M2b `_unroll_to_dtype_fixed_point`, reused)."""
    mesh = _mesh2d(2, 2)
    model, state = _model_and_state()
    state32 = jax.tree.map(
        lambda x: x.astype(jnp.float32) if hasattr(x, "astype") else x,
        state)
    n_steps = 4

    seg = make_sharded_atm_latlon_segment_2d(model, mesh, n_steps)
    out_seg, ok = seg(shard_state_atm_latlon_2d(state32, mesh), DT)
    assert bool(ok)
    seg_g = _fields(gather_state_atm_latlon_2d(out_seg, mesh))

    step2d = make_sharded_atm_latlon_step_2d(model, mesh)
    sc = shard_state_atm_latlon_2d(state32, mesh)
    for _ in range(n_steps):
        sc = step2d(sc, DT)
    seq_g = _fields(gather_state_atm_latlon_2d(sc, mesh))

    for f in ("u", "v", "T", "p_s"):
        assert seg_g[f].dtype == seq_g[f].dtype, (
            f"'{f}' dtype diverged: segment {seg_g[f].dtype} vs per-step "
            f"{seq_g[f].dtype}")
        np.testing.assert_allclose(
            seg_g[f], seq_g[f], rtol=1e-12, atol=1e-12,
            err_msg=f"f32-IC 2-D segment != per-step loop in '{f}' — the "
                    f"dtype fixed-point unroll changed the numerics.")


# ==============================================================================
# (6) topology chooser
# ==============================================================================

def test_chooser_prefers_band_at_low_counts_and_ties():
    # Square grid, 4 devices: (2,2) ties the band's perimeter -> band wins
    # (fewer collectives; selects the byte-identical 1-D production path).
    assert choose_latlon_2d_topology(4, 16, 16) == (4, 1)
    # 2 devices: band always.
    assert choose_latlon_2d_topology(2, 16, 16) == (2, 1)
    # Single device: trivially (1, 1).
    assert choose_latlon_2d_topology(1, 16, 16) == (1, 1)


def test_chooser_minimizes_perimeter():
    # Wide grid: the pure lon split owns both poles locally and moves only
    # 2h*nl per exchange — beats the band's 2h*n_lon by 4x.
    assert choose_latlon_2d_topology(4, 8, 32) == (1, 4)
    # Tall grid: the band is optimal.
    assert choose_latlon_2d_topology(4, 32, 8) == (4, 1)
    # Square grid, 8 devices: 2-D (12) beats band (16) beyond the 1.25x
    # preference; ties among 2-D candidates break toward smaller p_lon.
    assert choose_latlon_2d_topology(8, 16, 16) == (4, 2)


def test_chooser_feasibility_and_errors():
    # Indivisible longitude forces p_lon == 1.
    assert choose_latlon_2d_topology(4, 16, 17) == (4, 1)
    # min_tile: n_lat=8 over p_lat=8 leaves 1-row tiles -> infeasible; the
    # 2-D factorizations remain.
    assert choose_latlon_2d_topology(8, 8, 8) == (4, 2)
    # No factorization at all -> loud error, never a silent fallback.
    with pytest.raises(ValueError, match="no feasible"):
        choose_latlon_2d_topology(3, 16, 16)
    with pytest.raises(ValueError, match="n_devices"):
        choose_latlon_2d_topology(0, 16, 16)


# ==============================================================================
# dispatch-hardening refusals
# ==============================================================================

def test_2d_factory_refusals():
    mesh = _mesh2d(2, 2)
    model, state = _model_and_state()

    # Polar filter under a genuine lon split: loud refusal (lon-global FFT).
    pf_model, _ = _model_and_state(use_polar_filter=True)
    with pytest.raises(NotImplementedError, match="polar filter|use_polar_filter"):
        make_sharded_atm_latlon_step_2d(pf_model, mesh)

    # anchor_mass_to_initial: shared band refusal applies.
    grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON,
                              radius=constants.R_earth, omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=NLEV)
    anchor_model = CGridLatLonPrimitiveEquationModel(
        grid, sigma,
        CGridLatLonPrimitiveEquationConfig(anchor_mass_to_initial=True))
    with pytest.raises(NotImplementedError, match="anchor_mass_to_initial"):
        make_sharded_atm_latlon_step_2d(anchor_model, mesh)

    # Stateful PhysicsState carry: not 2-D-tile-routed -> loud refusal.
    step2d = make_sharded_atm_latlon_step_2d(model, mesh)
    sc = shard_state_atm_latlon_2d(state, mesh)
    with pytest.raises(NotImplementedError, match="PhysicsState"):
        step2d(sc, DT, phys_state=object())
    seg = make_sharded_atm_latlon_segment_2d(model, mesh, 2)
    with pytest.raises(NotImplementedError, match="PhysicsState"):
        seg(sc, DT, phys_state=object())

    # Wrong mesh axes: loud.
    bad_mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:N_DEV]).reshape(2, 2),
        axis_names=("lat", "x"))
    with pytest.raises(ValueError, match="axes"):
        make_sharded_atm_latlon_step_2d(model, bad_mesh)

    # Indivisible tiling / too-thin tiles: loud.
    with pytest.raises(ValueError, match="% p_lat|% p_lon|uniform"):
        build_tile_grids_atm_2d(model.grid, 3, 1)
    with pytest.raises(ValueError, match=">= 2 cells"):
        build_tile_grids_atm_2d(model.grid, 16, 1)   # 1-row tiles

    # n_steps hardening on the segment factory.
    with pytest.raises(ValueError, match="n_steps"):
        make_sharded_atm_latlon_segment_2d(model, mesh, 0)
