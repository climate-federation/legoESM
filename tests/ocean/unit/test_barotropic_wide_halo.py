"""Wide-halo split-explicit barotropic (opt-in, scaling-audit item 3).

Gates the wide-halo path's THREE contracts:

1. **Serial value-parity**: on a single band the wide path (extended array,
   forced-local pads, chunked loop) reproduces the standard local-clamp path
   on the owned rows — any reach under-estimate, staggered off-by-one in the
   v widening, or pole-extension defect shows up here at O(1), far above the
   re-association floor.
2. **Stencil-reach pin (NaN sentinel)**: garbage planted in the wide halo
   must NOT reach the owned rows within the budgeted number of substeps —
   and MUST reach them when run past the budget, proving the sentinel test
   is non-vacuous.  Trust this test, not the comment in
   ``_substep_stencil_reach``, when editing the substep body.
3. **Conservation + dispatch**: the model-level flag routes to the wide
   path, conserves volume like the standard path, and is refused for
   non-explicit barotropic solvers.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    _run_substep_loop,
    _compute_weights,
    _dissipation_coeffs,
    _substep_stencil_reach,
    barotropic_substeps_latlon_cgrid,
    barotropic_substeps_wide_halo_latlon_cgrid,
    estimate_barotropic_halo_messages,
)

N_LAT, N_LON, NLEV = 24, 32, 4


def _land_mask(n_lat, n_lon):
    """Ocean with a continent blob + one land row touching the south pole
    (exercises masked coastlines AND the pole extension)."""
    mask = np.ones((n_lat, n_lon))
    mask[n_lat // 3: n_lat // 2, n_lon // 4: n_lon // 2] = 0.0
    mask[0, :] = 0.0
    return jnp.asarray(mask)


@pytest.fixture()
def setup():
    grid = ensure_geometry(create_latlon_grid(n_lat=N_LAT, n_lon=N_LON))
    z_coord = create_ocean_z_star(n_levels=NLEV, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0, land_mask_override=_land_mask(N_LAT, N_LON))
    rng = np.random.default_rng(3)
    u = 0.05 * rng.standard_normal((N_LAT, N_LON + 1, NLEV))
    v = 0.05 * rng.standard_normal((N_LAT + 1, N_LON, NLEV))
    eta = 0.01 * rng.standard_normal((N_LAT, N_LON))
    u = u * np.asarray(state.u_mask.data)[..., None]
    v = v * np.asarray(state.v_mask.data)[..., None]
    eta = eta * np.asarray(state.land_mask.data)
    state = state._replace(
        u=state.u.replace(data=jnp.asarray(u)),
        v=state.v.replace(data=jnp.asarray(v)),
        eta=state.eta.replace(data=jnp.asarray(eta)))
    return grid, z_coord, state


def _cfg(**flat):
    # local clamp on BOTH paths so the schemes match exactly (the wide path
    # forces local per-substep clamping by contract).
    flat.setdefault("barotropic_local_subcycle_clamp", True)
    flat.setdefault("bottom_drag_r", 0.0)
    return LatLonCGridOceanConfig.from_flat(**flat)


def _assert_close(a, b, atol, what):
    a = np.asarray(a)
    b = np.asarray(b)
    mx = float(np.max(np.abs(a - b))) if a.size else 0.0
    assert np.allclose(a, b, rtol=0.0, atol=atol), f"{what}: max|diff|={mx:.3e}"


@pytest.mark.parametrize("flat", [
    dict(),                                     # default: diffusion on
    dict(barotropic_div_damp=0.02),             # + divergence damping
    dict(barotropic_time_filter="power_law"),   # n_loop > n_substeps
    dict(barotropic_wide_halo_chunk=3),         # multi-chunk (re-exchange)
    dict(                                        # round-59 raw recurrence
        barotropic_time_filter="nemo_boxcar_centred",
        barotropic_transport_accumulation_evaluation="nemo_literal",
        barotropic_wide_halo_chunk=3),
])
def test_serial_wide_matches_standard(setup, flat):
    grid, z_coord, state = setup
    cfg = _cfg(**flat)
    n_sub = 12
    s_std, (hu_s, hv_s) = barotropic_substeps_latlon_cgrid(
        state, 30.0, n_sub, grid, z_coord, cfg)
    s_wide, (hu_w, hv_w) = barotropic_substeps_wide_halo_latlon_cgrid(
        state, 30.0, n_sub, grid, z_coord, cfg)
    # f64, ~12 substeps: identical op sequence on a larger array — only
    # XLA re-association separates the paths.
    atol = 1e-12
    _assert_close(s_wide.eta.data, s_std.eta.data, atol, "eta")
    _assert_close(s_wide.u.data, s_std.u.data, atol, "u")
    _assert_close(s_wide.v.data, s_std.v.data, atol, "v")
    _assert_close(hu_w, hu_s, atol, "Hu_avg")
    _assert_close(hv_w, hv_s, atol, "Hv_avg")


@pytest.mark.parametrize("filt", ["nemo_ab3am4", "nemo_boxcar_ab3"])
def test_wide_halo_refuses_ab3_time_filters(setup, filt):
    """The AB3 filters must be REFUSED, not silently mis-run.

    The wide entry point builds none of the AB3 machinery (predictor
    coefficients, cross-window bt_hist carry, final-substep state selection)
    and passes no ``ab3_*`` argument to the shared substep loop.  Before the
    refusal, running them gave a wrong answer with no error: measured at 12
    substeps in fp64, ``nemo_ab3am4`` returned sea surface height IDENTICALLY
    ZERO (standard max|eta| = 4.377e-01 m) because it zeroes the filter
    weights while leaving the normaliser at 1, and ``nemo_boxcar_ab3`` --
    DINO's own filter -- diverged from the standard path by 2.445e-03 m,
    nine orders above this suite's 1e-12 parity gate.
    """
    grid, z_coord, state = setup
    cfg = _cfg(barotropic_time_filter=filt)
    # The standard path still runs it -- the refusal is specific to the twin.
    barotropic_substeps_latlon_cgrid(state, 30.0, 12, grid, z_coord, cfg)
    with pytest.raises(NotImplementedError, match="barotropic_time_filter"):
        barotropic_substeps_wide_halo_latlon_cgrid(
            state, 30.0, 12, grid, z_coord, cfg)


def test_wide_halo_honours_reconcile_target(setup):
    """The wide-halo path must READ ``barotropic_reconcile_target``, not
    silently reconcile onto the velocity mean.

    Two claims, both needed for non-vacuity:

    1. the option is LIVE on this state (the two targets give genuinely
       different 3-D velocities on the wide path), and
    2. the wide path's ``transport_avg`` result matches the standard path's
       ``transport_avg`` result to re-association tolerance.

    Before the fix the wide path ignored the option entirely, so (2) failed
    at O(1) (it reproduced the standard path's velocity_avg arm instead).
    """
    grid, z_coord, state = setup
    n_sub = 12
    cfg_v = _cfg(barotropic_reconcile_target="velocity_avg")
    cfg_t = _cfg(barotropic_reconcile_target="transport_avg")

    s_wide_v, _ = barotropic_substeps_wide_halo_latlon_cgrid(
        state, 30.0, n_sub, grid, z_coord, cfg_v)
    s_wide_t, _ = barotropic_substeps_wide_halo_latlon_cgrid(
        state, 30.0, n_sub, grid, z_coord, cfg_t)
    s_std_t, _ = barotropic_substeps_latlon_cgrid(
        state, 30.0, n_sub, grid, z_coord, cfg_t)

    # (1) the two kernels do not coincide on this state.
    spread = float(np.max(np.abs(
        np.asarray(s_wide_t.u.data) - np.asarray(s_wide_v.u.data))))
    assert spread > 1e-9, (
        f"velocity_avg and transport_avg give the SAME wide-halo velocity "
        f"(max diff {spread:.2e}) — the test cannot detect an ignored option")

    # (2) wide == standard for the non-default target.
    _assert_close(s_wide_t.u.data, s_std_t.u.data, 1e-12, "u (transport_avg)")
    _assert_close(s_wide_t.v.data, s_std_t.v.data, 1e-12, "v (transport_avg)")


def test_wide_volume_drift_matches_standard(setup):
    """The wide path introduces NO conservation change: its area-weighted
    eta drift over the subcycle equals the standard path's to round-off.
    (The subcycle's absolute drift itself is a pre-existing property of the
    time-filtered scheme on a transient IC; the OUTER step owns absolute
    conservation via the #271 fix_eta_drift projection.)"""
    grid, z_coord, state = setup
    cfg = _cfg()
    area = np.asarray(grid.area)
    mask = np.asarray(state.land_mask.data)
    s_std, _ = barotropic_substeps_latlon_cgrid(
        state, 30.0, 12, grid, z_coord, cfg)
    s_wide, _ = barotropic_substeps_wide_halo_latlon_cgrid(
        state, 30.0, 12, grid, z_coord, cfg)
    vol_std = float(np.sum(np.asarray(s_std.eta.data) * area * mask))
    vol_wide = float(np.sum(np.asarray(s_wide.eta.data) * area * mask))
    scale = float(np.sum(area * mask)) * 4000.0
    assert abs(vol_wide - vol_std) / scale < 1e-15, (vol_std, vol_wide)


@pytest.mark.parametrize("flat", [dict(), dict(barotropic_div_damp=0.02)])
def test_nan_sentinel_pins_stencil_reach(setup, flat):
    """Mechanical pin of the per-substep stencil reach.

    Model: after ONE wide exchange every extended row is valid; garbage then
    enters from the extended EDGES (the local zero-pads the operators apply
    once the exchange stops) and penetrates inward at the substep body's true
    stencil rate.  A NaN band planted at the array edge is a conservative
    tracer of that front (NaN survives every arithmetic channel real garbage
    can take, including ``min``).  The budget ``_substep_stencil_reach``
    must bound the measured per-substep penetration RATE — and the sentinel
    must actually move (non-vacuity).
    """
    from legoesm.grids.halo import local_halo_pads
    from legoesm.grids.halo_latlon import (
        widen_band_cell_fields,
        widen_band_vface_fields,
        widen_cgrid_geometry_band,
    )
    from legoesm.ocean.dynamics.barotropic_common import coriolis_at_faces

    grid, z_coord, _state = setup
    cfg = _cfg(**flat)
    reach = _substep_stencil_reach(cfg)
    W = 12  # deep sentinel band: room to measure 3 substeps of penetration

    dtype = jnp.float64
    eta = jnp.zeros((N_LAT, N_LON), dtype)
    U = jnp.full((N_LAT, N_LON + 1), 0.01, dtype)
    V = jnp.full((N_LAT + 1, N_LON), 0.01, dtype)
    mask = jnp.ones((N_LAT, N_LON), dtype)
    u_mask = jnp.ones((N_LAT, N_LON + 1), dtype)
    v_mask = jnp.ones((N_LAT + 1, N_LON), dtype)
    H_bathy = jnp.full((N_LAT, N_LON), 4000.0, dtype)

    grid_ext = widen_cgrid_geometry_band(grid, W)
    (eta_x, U_x, H_x, m_x, um_x) = widen_band_cell_fields(
        (eta, U, H_bathy, mask, u_mask), W)
    (V_x, vm_x) = widen_band_vface_fields((V, v_mask), W)

    # Treat BOTH extended edges as INTERIOR cuts (open ocean), not poles —
    # zero-masked rows would hold NaN in place (0*NaN=NaN) while absorbing
    # the finite garbage the real path sees; the open-ocean picture is the
    # one the reach budget must bound.
    m_x = m_x.at[:W].set(1.0).at[-W:].set(1.0)
    um_x = um_x.at[:W].set(1.0).at[-W:].set(1.0)
    vm_x = vm_x.at[:W].set(1.0).at[-W:].set(1.0)
    H_x = H_x.at[:W].set(4000.0).at[-W:].set(4000.0)

    # Sentinel: NaN-fill the SOUTH halo band of the carry (edge garbage).
    eta_x = eta_x.at[:W].set(jnp.nan)
    U_x = U_x.at[:W].set(jnp.nan)
    V_x = V_x.at[:W].set(jnp.nan)

    dt_s = jnp.asarray(30.0, dtype)
    minw = jnp.asarray(cfg.min_water_column_m, dtype)
    g = jnp.asarray(cfg.g, dtype)
    area_ext = grid_ext.area.astype(dtype)
    eta_floor_ext = minw - H_x
    zeros_c = jnp.zeros_like(eta_x)
    zeros_u = jnp.zeros_like(U_x)
    zeros_v = jnp.zeros_like(V_x)
    w_f, _w_tot, w_tr, _n_loop = _compute_weights(cfg, 12, dtype)

    def deepest_nan_row(n_steps):
        with local_halo_pads():
            f_u, f_v = coriolis_at_faces(grid_ext, dtype)
            coeffs = _dissipation_coeffs(
                cfg, grid_ext, area_ext, dt_s, dtype, m_x)
            finals = _run_substep_loop(
                eta_x, U_x, V_x,
                dt_s=dt_s, n_loop=n_steps,
                w_filter=w_f[:n_steps], w_transport=w_tr[:n_steps],
                grid=grid_ext, config=cfg, g=g, H_bathy=H_x, mask=m_x,
                u_mask=um_x, v_mask=vm_x, min_water_col=minw,
                eta_floor=eta_floor_ext, area=area_ext,
                F_slow_eta=zeros_c, F_slow_u=zeros_u, F_slow_v=zeros_v,
                f_u=f_u, f_v=f_v, add_barotropic_coriolis=True,
                coeffs=coeffs, local_subcycle_clamp=True,
            )
        deepest = -1
        for arr in finals:
            a = np.asarray(arr)
            bad = np.where(~np.isfinite(a).all(axis=tuple(range(1, a.ndim))))[0]
            if bad.size:
                deepest = max(deepest, int(bad.max()))
        return deepest

    base = W - 1  # deepest initially-NaN row
    for k in (1, 2, 3):
        deepest = deepest_nan_row(k)
        rate_rows = deepest - base
        assert rate_rows <= k * reach, (
            f"substep stencil reach UNDER-budgeted: NaN advanced "
            f"{rate_rows} rows in {k} substeps but "
            f"_substep_stencil_reach={reach} budgets {k * reach}.")
        assert rate_rows >= k, (
            "NaN sentinel barely moved — the test lost its teeth "
            f"(advanced {rate_rows} rows in {k} substeps).")


def test_model_level_dispatch_and_validation(setup):
    grid, z_coord, state = setup
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    cfg_std = _cfg()
    cfg_wide = _cfg(barotropic_wide_halo=True)
    m_std = LatLonCGridOceanModel(grid, z_coord, cfg_std)
    m_wide = LatLonCGridOceanModel(grid, z_coord, cfg_wide)
    s1 = m_std.step(state, 600.0)
    s2 = m_wide.step(state, 600.0)
    for name in ("eta", "u", "v", "T", "S"):
        _assert_close(getattr(s2, name).data, getattr(s1, name).data,
                      1e-10, f"model-step {name}")

    with pytest.raises(ValueError, match="wide_halo"):
        LatLonCGridOceanModel(
            grid, z_coord,
            _cfg(barotropic_wide_halo=True, barotropic_solver="implicit_cn"))
    with pytest.raises(ValueError, match="wide_halo_chunk"):
        LatLonCGridOceanModel(
            grid, z_coord, _cfg(barotropic_wide_halo_chunk=-1))
    # The local-clamp scheme must be the EXPLICIT choice (the wide path's
    # per-substep clamp is local by construction — codex finding 4).
    with pytest.raises(ValueError, match="local_subcycle_clamp"):
        LatLonCGridOceanModel(
            grid, z_coord,
            LatLonCGridOceanConfig.from_flat(
                barotropic_wide_halo=True,
                barotropic_local_subcycle_clamp=False))


def test_2d_pencil_layout_refused(setup):
    """local_halo_pads would also localize the 2-D pencil's ZONAL exchanges
    (silent E/W wrap inside the lon block) — the wide path must refuse the
    layout loudly (codex finding 2)."""
    # The simulated 2-D pencil layout routes setup pads through the mpi4jax
    # backend; envs without it (plain conda) die in the pad before reaching
    # the refusal under test — skip there (MPI envs run it).
    pytest.importorskip("mpi4jax")
    from legoesm.grids.halo import set_halo_backend
    from legoesm.parallel.latlon_mpi import make_latlon_2d_layout

    grid, z_coord, state = setup
    layout = make_latlon_2d_layout(0, 2, 2, N_LAT, N_LON)
    set_halo_backend("mpi", topology=layout)
    try:
        with pytest.raises(ValueError, match="2-D lat-lon pencil"):
            barotropic_substeps_wide_halo_latlon_cgrid(
                state, 30.0, 6, grid, z_coord, _cfg(barotropic_wide_halo=True))
    finally:
        set_halo_backend("local")


def test_tripole_fold_refused_at_construction():
    """Wide-halo + active fold must fail at MODEL CONSTRUCTION with the
    config knob named — not mid-run inside the widen helper (codex 3)."""
    from legoesm.grids.tripole import create_synthetic_tripole
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    tri = create_synthetic_tripole(n_lat=12, n_lon=16)
    z_coord = create_ocean_z_star(n_levels=3, H_max=2000.0)
    with pytest.raises(ValueError, match="tripolar"):
        LatLonCGridOceanModel(tri, z_coord, _cfg(barotropic_wide_halo=True))


def test_chunk_exceeding_band_height_raises(setup):
    grid, z_coord, state = setup
    cfg = _cfg(barotropic_wide_halo_chunk=N_LAT)  # W = chunk*reach > n_lat
    with pytest.raises(ValueError, match="exceeds the band-height budget"):
        barotropic_substeps_wide_halo_latlon_cgrid(
            state, 30.0, N_LAT, grid, z_coord, cfg)


def test_estimate_halo_messages():
    # #1609 centred the box/cosine averaging window on t+dt, so the substep
    # loop runs n_loop = 2n-1 times, not n.  The message count scales with the
    # substeps actually run, so derive it from the reported n_loop instead of
    # re-hardcoding a window length that a future filter change would stale out
    # again (this assertion still read 4*30 -- the pre-#1609 half window).
    cfg = _cfg()
    est = estimate_barotropic_halo_messages(cfg, 30)
    assert est["n_loop"] == 2 * 30 - 1
    assert est["standard_messages"] == 4 * est["n_loop"]  # diffusion on by default
    assert est["wide_messages_fixed"] == 4
    assert est["wide_messages_per_chunk"] == 2
    est_dd = estimate_barotropic_halo_messages(
        _cfg(barotropic_div_damp=0.02), 30)
    assert est_dd["n_loop"] == est["n_loop"]
    assert est_dd["standard_messages"] == 6 * est_dd["n_loop"]
    assert est_dd["stencil_reach"] == est["stencil_reach"] + 2
