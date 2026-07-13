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
     + continuity; dT carries only the genuine limited-FV-PPM halo-2
     truncation at cut-adjacent lat ROWS and lon COLUMNS: the outermost
     ``ppm_edge_values`` edges on a halo-2 padded tile are 2nd-order and
     couple into the ghost-cell parabolas' LIMITED face states (CW84), so
     bit parity at a cut is impossible at halo 2.  The flux METRIC at cuts
     is exact (``lat_v_interfaces == grid.lat_v``, gated by the geometry
     test + the realistic-amplitude tendency gate — codex M3a findings 2/6:
     the pre-fix operator fabricated ±π/2 pole faces at every tile cut).

(3)  STEP — the integrated (2,2) 2-D step matches serial at the Stage-5
     bound, with SEAM-SPECIFIC assertions: the u periodic-seam column, the
     u/v values along every tile cut, and the v pole rows.

(4)  DEGENERACY — the (4,1) 2-D step is BIT-IDENTICAL to the existing 1-D
     band step (every lon-ring op takes its static local branch).

(5)  SEGMENT — the compiled 10-step 2-D ``lax.scan`` matches 10 sequential
     2-D steps at measured per-field ABSOLUTE caps (rtol=0 — honest gates,
     codex M3a finding 7) and serial at the Stage-5 bound; the in-graph
     finite scalar (psum over BOTH axes) is exercised healthy + NaN-injected;
     a mixed-precision (f32) IC exercises the dtype-fixed-point unroll, also
     gated at measured absolute caps with an injected-defect tripwire.

(6)  CHOOSER — modeled pad communication volume (ppermute perimeter PLUS the
     two pole-fold lon-all_gathers every lon split pays on every tile —
     codex M3a finding 4), band selection whenever feasible,
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

def _realistic_amplitude_state(model):
    """A production-amplitude state (O(10 m/s) winds, O(40 K) meridional T
    contrast, O(1e3 Pa) p_s structure) satisfying the layout invariants.

    The small-amplitude ``_model_and_state`` (1e-3 winds, 1e-3 T noise)
    suppresses any CUT-FACE METRIC defect to ~4e-12 in dT because the
    advective form's corrupted-face terms cancel to
    ``v*hx*(q_face - q_c)/area`` — both factors 1e-3-scale there.  At
    realistic amplitude the pre-fix fabricated-pole metric (codex M3a
    findings 2/6) measures 8.5e-5 K/s = 7 K/day at cut rows (job 8970803),
    20x the genuine halo-2 limiter truncation (4.3e-6)."""
    grid = model.grid
    rng = np.random.default_rng(4242)
    lat = np.asarray(grid.lat)                      # (n_lat,)
    u0 = 10.0 * rng.standard_normal((N_LAT, N_LON + 1, NLEV))
    v0 = 10.0 * rng.standard_normal((N_LAT + 1, N_LON, NLEV))
    v0[0] = 0.0
    v0[-1] = 0.0
    u0[:, -1] = u0[:, 0]
    # sin(lat): MAXIMUM meridional T gradient and ZERO curvature at the
    # equator — i.e. at the (2,2) lat-cut face — so the fabricated-pole
    # metric defect signal (~ v*hx*(q_face - q_c)) is maximal there while
    # the genuine PPM edge-order truncation (~ curvature) is minimal:
    # the sharpest discrimination between the two.
    T0 = (250.0 + 40.0 * np.sin(lat)[:, None, None]
          + 0.5 * rng.standard_normal((N_LAT, N_LON, NLEV)))
    ps0 = 1.0e5 + 1.0e3 * rng.standard_normal((N_LAT, N_LON))
    return CGridLatLonHydrostaticState(
        u=jnp.asarray(u0),
        v=jnp.asarray(v0),
        T=jnp.asarray(T0),
        p_s=jnp.asarray(ps0),
        phis=jnp.zeros((N_LAT, N_LON)),
    )


def _serial_and_tile_tendency(state=None):
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
    model, default_state = _model_and_state()
    state = default_state if state is None else state
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
    #
    # WHY a residual is genuinely allowed at cuts (and only there): on the
    # halo-2 padded tile the outermost ``ppm_edge_values`` edges are
    # 2ND-order (a 4th-order edge needs 2 cells each side) and feed the
    # GHOST-cell parabolas, whose CW84-limited a_R/a_L ARE the owned
    # cut-face upwind states — a real one-sided-stencil truncation needing
    # halo 3 to vanish, not a decomposition bug.  The cut-face flux METRIC
    # carries NO residual: ``lat_v_interfaces`` returns the tile's owned
    # ``grid.lat_v`` (codex M3a findings 2/6 — the pre-fix operator
    # fabricated ±π/2 pole faces at every cut; gated bitwise below by
    # test_lat_v_interfaces_owned_faces and at amplitude by
    # test_2d_tendency_realistic_amplitude_cut_gate).
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
    assert resid < _DT_CUT_TRUNC_CAP_SMALL_AMP, (
        f"cut-adjacent dT_dt residual {resid:.2e} exceeds the measured "
        f"halo-2 limiter one-sided-edge truncation cap "
        f"{_DT_CUT_TRUNC_CAP_SMALL_AMP:.1e} — a real cut defect (metric, "
        f"halo, or reconstruction), not the boundary-order reduction.")


# Measured post-fix (2,2) cut-adjacent dT truncation maxima on the CI CPU
# lane (x64, 4 host devices), capped at ~4x:
#   small-amp 1.399e-12 (job 8970746; PRE-fix fabricated-pole metric
#     measured 4.13e-12 there — inside this cap, so the small-amp gate
#     alone does NOT discriminate the metric defect: that is the realistic
#     gate's job);
#   realistic-amp 4.32e-6 with the sin(lat) profile (job 8970803; PRE-fix
#     metric measured 8.51e-5 there — 4.7x above this cap and 20x the
#     genuine truncation, so the realistic gate fails loudly on any
#     pole-fabrication regression while tolerating the truncation).
_DT_CUT_TRUNC_CAP_SMALL_AMP = 6e-12
_DT_CUT_TRUNC_CAP_REALISTIC = 1.8e-5


def test_2d_tendency_realistic_amplitude_cut_gate():
    """(2,2) dT parity at PRODUCTION amplitude — the finding-2/6 pin.

    Every (2,2) tile touches a physical pole row AND an interior equator
    cut.  Pre-fix, ``lat_v_interfaces`` fabricated a ±π/2 pole at the cut
    face of every tile, collapsing the PPM meridional flux metric
    ``hx = R*dlon*cos(lat_v)`` to the 1e-10 clamp there — a spurious
    transport wall.  The small-amplitude gate cannot see it (advective-form
    cancellation leaves ~4e-12, inside that gate's cap); at realistic
    amplitude it is an 8.5e-5 K/s = 7 K/day dT error at cut rows (measured
    by re-injecting the fabricated metric, job 8970803) vs a genuine halo-2
    limiter truncation of 4.3e-6 K/s.  The cap sits between them (4.2x the
    truncation): a pole-fabrication regression fails it by ~5x."""
    model, _ = _model_and_state()
    state = _realistic_amplitude_state(model)
    (du_s, dv_s, dT_s, dps_s), (dul_b, dvl_b, dT_b, dps_b), _ = (
        _serial_and_tile_tendency(state=state))

    # Momentum + continuity stay decomposition-tight at amplitude.
    for name, a, b in (("du_dt", dul_b, du_s[:, :N_LON]),
                       ("dv_dt", dvl_b, dv_s[:N_LAT]),
                       ("dp_s_dt", dps_b, dps_s)):
        np.testing.assert_allclose(
            a, b, rtol=1e-10, atol=1e-10,
            err_msg=f"realistic-amplitude '{name}' diverged from serial — "
                    f"a real tile-decomposition bug.")

    resid = float(np.max(np.abs(dT_b - dT_s)))
    assert resid < _DT_CUT_TRUNC_CAP_REALISTIC, (
        f"realistic-amplitude cut-adjacent dT_dt residual {resid:.2e} "
        f"exceeds the measured halo-2 truncation cap "
        f"{_DT_CUT_TRUNC_CAP_REALISTIC:.1e}.  A fabricated-pole cut-face "
        f"metric (codex M3a findings 2/6) measures 8.5e-5 here (job "
        f"8970803) — this gate exists to fail on any such metric "
        f"regression.")


def test_lat_v_interfaces_owned_faces_never_fabricated_poles():
    """PPM v-face latitudes == the grid's OWNED ``lat_v`` on every
    decomposition — bitwise (single-device: always enforced, no 4-device
    skip).

    Pins codex M3a findings 2/6: a tile/band's end faces are INTERIOR
    partition cuts, never fabricated ±π/2 poles.  Also pins the shared-face
    identity the flux-form conservation argument rests on: adjacent tiles
    hold the SAME bitwise latitude for their shared cut face."""
    from legoesm.core.operators_fv_latlon import lat_v_interfaces
    from legoesm.parallel.latlon_mpi import (
        make_latlon_2d_layout, make_latlon_band_layout,
        slice_latlon_grid_to_band, slice_latlon_grid_to_block_2d)

    grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON,
                              radius=constants.R_earth, omega=constants.Omega)
    lat_v_g = np.asarray(grid.lat_v)

    # Global grid: owned faces; ends ARE the poles (bitwise at the grid's
    # STORAGE dtype — the default precision policy stores f32); interior =
    # cell-center midpoints to storage precision (stored values are
    # f64-computed then cast, a recompute from cast lat is <= ~1 ulp off).
    out_g = np.asarray(lat_v_interfaces(grid))
    np.testing.assert_array_equal(out_g, lat_v_g)
    half_pi = np.asarray(np.pi / 2, dtype=out_g.dtype)
    np.testing.assert_array_equal(out_g[[0, -1]],
                                  np.asarray([-half_pi, half_pi]))
    np.testing.assert_allclose(
        out_g[1:-1], 0.5 * (np.asarray(grid.lat)[:-1]
                            + np.asarray(grid.lat)[1:]),
        rtol=0, atol=4 * np.finfo(out_g.dtype).eps)

    # (2,2) tiles: every tile's faces == the global lat_v window [s:e+1].
    p_lat, p_lon = 2, 2
    nl = N_LAT // p_lat
    tiles = {}
    for r in range(p_lat):
        for c in range(p_lon):
            layout = make_latlon_2d_layout(
                r * p_lon + c, p_lat, p_lon, N_LAT, N_LON, None)
            tile = slice_latlon_grid_to_block_2d(
                grid, layout, skip_total_area_reduce=True)
            tiles[(r, c)] = np.asarray(lat_v_interfaces(tile))
            np.testing.assert_array_equal(
                tiles[(r, c)], lat_v_g[r * nl:r * nl + nl + 1],
                err_msg=f"tile ({r},{c}) v-face latitudes != owned global "
                        f"window — fabricated faces at a partition cut.")

    # The tripwire that fails on the pre-fix fabrication: the SOUTH tile's
    # NORTH end face is the equator-adjacent interface, nowhere near +π/2.
    equator_face = tiles[(0, 0)][-1]
    assert abs(equator_face) < 0.2, (
        f"south tile's north cut face at lat {equator_face:.3f} rad — "
        f"expected the equator interface (~0), got a fabricated pole?")
    # Shared-face identity across the cut (flux-consistency anchor).
    np.testing.assert_array_equal(tiles[(0, 0)][-1:], tiles[(1, 0)][:1])
    np.testing.assert_array_equal(tiles[(0, 1)][-1:], tiles[(1, 1)][:1])

    # 1-D band slicer: same contract.
    for rank in range(4):
        layout = make_latlon_band_layout(rank, 4, N_LAT, N_LON)
        band = slice_latlon_grid_to_band(grid, layout,
                                         skip_total_area_reduce=True)
        s = rank * (N_LAT // 4)
        np.testing.assert_array_equal(
            np.asarray(lat_v_interfaces(band)),
            lat_v_g[s:s + N_LAT // 4 + 1],
            err_msg=f"band {rank} v-face latitudes != owned global window.")


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

# Honest segment gates (codex M3a finding 7): rtol=0 with per-field
# MAXIMUM-ABSOLUTE-error caps.  A relative tolerance on p_s ~ 1e5 Pa
# (rtol=1e-2 -> ~1e3 Pa; the "1e-12" f64 gate -> ~1e-7 Pa) masks defects
# orders of magnitude above both the measured reassociation envelope and
# the O(1e-3) Stage-5 defect scale.  Caps are ~4x the measured envelope on
# the CI CPU lane (x64, 4 host devices, job 8970746):
#   f64 segment-vs-sequential (10 steps):
#     u 1.83e-13, v 5.60e-14, T 2.27e-13, p_s 7.28e-11
#     (the fused scan vs n separate programs reassociates even in f64);
#   f32-IC segment-vs-sequential (4 steps):
#     u 1.101e-5, v 3.20e-6, T 4.07e-5, p_s 5.29e-4
#     (u bit-reproduces job 8942341's 1.10110183e-05 on a different node —
#     the envelope is deterministic on this lane, so 4x is real headroom).
# The tripwire below proves every cap sits BELOW the per-field defect
# scale, so an injected decomposition bug fails every field's gate.
_F64_SEG_SEQ_ABS_CAPS = {"u": 8e-13, "v": 3e-13, "T": 1e-12, "p_s": 3e-10}
_F32_SEG_SEQ_ABS_CAPS = {"u": 4.5e-5, "v": 1.3e-5, "T": 1.7e-4, "p_s": 2.2e-3}
# What a REAL decomposition bug looks like per field, in field units
# (Stage-5 doctrine: O(1e-3) abs on the eps=1e-3 wind/T-noise scale; p_s
# carries 10 Pa perturbations on 1e5 Pa, so a real p_s defect is O(1) Pa).
_SEG_DEFECT_SCALES = {"u": 1e-3, "v": 1e-3, "T": 1e-3, "p_s": 1.0}


def _assert_fields_within_abs_caps(actual, desired, caps, context):
    """rtol=0 per-field max-abs gate shared by the real segment gates and
    the injected-defect tripwire (so the tripwire exercises the EXACT
    assertion path the gates use — non-vacuous by construction)."""
    for f, cap in caps.items():
        np.testing.assert_allclose(
            actual[f], desired[f], rtol=0, atol=cap,
            err_msg=(
                f"'{f}' exceeded its measured maximum-absolute-error cap "
                f"{cap:.2e} ({context}) — a real schedule/decomposition "
                f"defect, not reassociation noise."))


def test_segment_abs_cap_tripwire_catches_injected_defect():
    """Mutation tripwire for the finding-7 gates: an injected per-field
    defect at the documented Stage-5 scale must FAIL that field's gate,
    for BOTH cap sets — and every cap must sit strictly below the defect
    scale it claims to detect (else the gate is vacuous)."""
    rng = np.random.default_rng(7)
    base = {f: rng.standard_normal((3, 4, 2)) for f in ("u", "v", "T", "p_s")}
    for caps, label in ((_F64_SEG_SEQ_ABS_CAPS, "f64"),
                        (_F32_SEG_SEQ_ABS_CAPS, "f32")):
        # Sanity: the healthy path passes.
        _assert_fields_within_abs_caps(base, base, caps, f"{label} healthy")
        for f in ("u", "v", "T", "p_s"):
            defect = _SEG_DEFECT_SCALES[f]
            assert caps[f] < defect, (
                f"{label} cap for '{f}' ({caps[f]:.2e}) is not below the "
                f"defect scale ({defect:.2e}) — the gate could not detect "
                f"a real bug in that field.")
            bad = dict(base)
            bad[f] = base[f] + defect
            with pytest.raises(AssertionError):
                _assert_fields_within_abs_caps(
                    bad, base, caps, f"{label} tripwire '{f}'")


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

    # rtol=0, measured per-field absolute caps (codex M3a finding 7 — a
    # relative gate on p_s ~ 1e5 was ~1e-7 Pa, not the "1e-12" it claimed).
    _assert_fields_within_abs_caps(
        seg_g, seq_g, _F64_SEG_SEQ_ABS_CAPS,
        "f64 compiled 2-D segment vs sequential 2-D steps")
    for f in ("u", "v", "T", "p_s"):
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
    per-step Python loop does (M2b `_unroll_to_dtype_fixed_point`, reused).

    Contract gated here: (a) the segment COMPILES and runs (without the
    unroll the ``lax.scan`` carry dtype mismatch is a trace-time error);
    (b) the OUTPUT dtype signature matches the per-step lane exactly (the
    unroll's actual guarantee — same promotion schedule, no silent cast);
    (c) the promotion genuinely engaged (non-vacuity — the f32 IC leaves
    promote under x64, the M2b-documented behaviour of this model);
    (d) values match the per-step lane within MEASURED per-field
    maximum-absolute caps at rtol=0 (``_F32_SEG_SEQ_ABS_CAPS`` — codex M3a
    finding 7: the previous rtol=1e-2 permitted ~1e3 Pa in p_s, masking
    defects far above both the measured ~1.1e-5 u envelope, job 8942341,
    and the O(1e-3) Stage-5 defect scale).  The segment is ONE fused XLA
    program while the per-step lane is n separate programs, so f32
    intermediates reassociate within the caps; anything above them is a
    real schedule/decomposition defect.  Detection is proven non-vacuous
    by test_segment_abs_cap_tripwire_catches_injected_defect; the f64 lane
    is gated at its own measured caps in
    test_2d_segment_matches_sequential_and_serial."""
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

    # (c) non-vacuity: the promotion the unroll exists for actually
    # happened (else the fixed point is trivial and this gate is vacuous).
    assert any(seg_g[f].dtype == np.float64 for f in ("u", "v", "T", "p_s")), (
        "f32 IC did not promote under x64 — the dtype-fixed-point unroll "
        "was never engaged; this gate is vacuous, revisit it.")

    for f in ("u", "v", "T", "p_s"):
        # (b) exact dtype-signature parity with the per-step lane.
        assert seg_g[f].dtype == seq_g[f].dtype, (
            f"'{f}' dtype diverged: segment {seg_g[f].dtype} vs per-step "
            f"{seq_g[f].dtype}")
    # (d) rtol=0, measured per-field absolute caps (see docstring).
    _assert_fields_within_abs_caps(
        seg_g, seq_g, _F32_SEG_SEQ_ABS_CAPS,
        "f32-IC compiled 2-D segment vs per-step loop")


# ==============================================================================
# (6) topology chooser
# ==============================================================================

def test_chooser_prefers_band_at_low_counts_and_ties():
    # Square grid, 4 devices: band 16 vs (2,2) = 8+8+16 = 32 (the lon split
    # pays the pole-fold all_gathers) -> band wins outright (selects the
    # byte-identical 1-D production path).
    assert choose_latlon_2d_topology(4, 16, 16) == (4, 1)
    # 2 devices: band always.
    assert choose_latlon_2d_topology(2, 16, 16) == (2, 1)
    # Single device: trivially (1, 1).
    assert choose_latlon_2d_topology(1, 16, 16) == (1, 1)


def test_chooser_counts_pole_fold_all_gathers():
    """Codex M3a finding 4: every p_lon > 1 pad executes TWO lon
    all_gathers of the (h, n_lon) pole edge rows on EVERY tile (both
    ``jnp.where`` fold operands evaluate), so a perimeter-only score
    mis-ranks lon splits.  Wide grid, 4 devices: perimeter-only scored
    (1,4) at nl=8 "beating" the band's 32 by 4x; the honest volume is
    8 + 32 (gathers) = 40 > 32 -> the band wins whenever feasible."""
    assert choose_latlon_2d_topology(4, 8, 32) == (4, 1)
    # Tall grid: the band is optimal under both models.
    assert choose_latlon_2d_topology(4, 32, 8) == (4, 1)
    # Square grid, 8 devices: perimeter-only claimed a crossover to (4,2)
    # (12 vs 16); honestly (4,2) = 8+4+16 = 28 > 16 -> band.
    assert choose_latlon_2d_topology(8, 16, 16) == (8, 1)
    # General property: with the gather term, ANY p_lon > 1 candidate
    # scores >= nl + n_lon > n_lon = the band -> band whenever feasible.
    for n_dev, n_lat, n_lon in ((4, 16, 16), (8, 32, 32), (16, 64, 32)):
        assert choose_latlon_2d_topology(n_dev, n_lat, n_lon) == (n_dev, 1)


def test_chooser_2d_when_band_infeasible():
    """The 2-D lane is selected exactly in its raison-d'etre regime: the
    band infeasible (indivisible n_lat or sub-min_tile bands).  Ranking
    among p_lon > 1 candidates stays perimeter-based (the gather term is
    common to all of them)."""
    # n_lat=6 % 4 != 0 -> band infeasible; (1,4)=6+32=38 beats
    # (2,2)=16+3+32=51.
    assert choose_latlon_2d_topology(4, 6, 32) == (1, 4)
    # min_tile: p_lat=8 leaves 1-row bands -> infeasible; the gather term
    # is common to the surviving 2-D candidates, so the perimeter tie
    # (4,2)=4+2+8=14 vs (2,4)=2+4+8=14 breaks toward smaller p_lon.
    assert choose_latlon_2d_topology(8, 8, 8) == (4, 2)


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
