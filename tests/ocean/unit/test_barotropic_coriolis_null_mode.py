"""The C-grid barotropic Coriolis 2Δx rotational null mode and its faithful cure.

Root cause of the §5 eddy-resolving turbulent blow-up (docs/issues/
barotropic_mode_noise.md §A): the in-substep barotropic Coriolis interpolates V
to u-points with the 4-point average ``0.25·(V[i]+V[i+1] + V_west[i]+V_west[i+1])``
(``barotropic_latlon_cgrid.py`` L377-379), where ``V_west = roll(V, 1, axis=lon)``.
A 2Δx ZONAL checkerboard, V[:,j] = (-1)^j, makes V_west = −V, so the average
vanishes and the discrete Coriolis operator exerts NO restoring on it — the
divergent high-zonal-wavenumber Arakawa-Lamb (1977) null mode, free to grow under
the eddy field until the run blows up.

The faithful cure (``coriolis_scheme="explicit_ab2"``) routes the planetary
Coriolis to the barotropic mode through the AB2-extrapolated slow forcing F_slow
(Oceananigans split-explicit convention: ∂_tU = −gH∇η + G^U, NO in-substep
Coriolis) → ``add_barotropic_coriolis=False`` → the null-mode operator is never
applied.  No dissipation backstop.

These tests drive the REAL ``barotropic_substeps_latlon_cgrid`` (no duplicated
stencil numerics) to pin both halves: (1) the checkerboard is a Coriolis null
mode — it produces no U response while a smooth V does; (2) ``add_barotropic_
coriolis=False`` (the cure) removes the in-substep Coriolis coupling entirely.

The full 160×128×50 80-day survival of the cured stack is validated offline by
``scripts/tmp/_silvestri_eps_validate.py`` (too expensive for CI); these unit
tests pin the *mechanism* that makes that survival hold.
"""

import jax.numpy as jnp
import numpy as np
from legoesm.core.field import Field
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import barotropic_substeps_latlon_cgrid
from legoesm.ocean.experiments.silvestri_baroclinic_jet import (
    SilvestriJetConfig,
    build_silvestri_baroclinic_jet_setup,
)

set_policy(PrecisionPolicy.fp64())


def _rest_setup():
    """Small §5 setup, then quiesce: u=v=eta=0, alpha=0 (isolate Coriolis)."""
    r = build_silvestri_baroclinic_jet_setup(
        n_lat=24, n_lon=16, scheme="W9V", nlev=4,
        config=SilvestriJetConfig(), stabilize=False)
    cfg = r.model_config._replace(barotropic=r.model_config.barotropic._replace(barotropic_diffusion_alpha=0.0))
    s = r.initial_state
    z_u = jnp.zeros_like(s.u.data)
    z_v = jnp.zeros_like(s.v.data)
    z_e = jnp.zeros_like(s.eta.data)
    s = s._replace(
        u=Field(data=z_u, name="u", dims=s.u.dims, units=s.u.units),
        v=Field(data=z_v, name="v", dims=s.v.dims, units=s.v.units),
        eta=Field(data=z_e, name="eta", dims=s.eta.dims, units=s.eta.units))
    return r, cfg, s


def _set_v(s, v3d):
    return s._replace(v=Field(data=v3d, name="v", dims=s.v.dims, units=s.v.units))


def _checkerboard_v(s):
    """2Δx zonal checkerboard: v[:,j,:] = (-1)^j (the Coriolis null mode).

    The 4-point V→u average includes ``V_west = roll(V, 1, axis=lon)``
    (``barotropic_latlon_cgrid.py`` L377): for a zonal checkerboard
    V_west = −V, so ``V_bar_c + V_west = 0`` and V_at_u vanishes — the
    divergent high-zonal-wavenumber Arakawa-Lamb mode the Coriolis
    operator cannot see (docs/issues/barotropic_mode_noise.md §A)."""
    nlat_v, nlon, nlev = s.v.data.shape
    sign = ((-1.0) ** jnp.arange(nlon))[None, :, None]
    return jnp.broadcast_to(sign, (nlat_v, nlon, nlev)).astype(s.v.data.dtype)


def _smooth_v(s):
    """A smooth (DC) v field of the same amplitude for the control."""
    return jnp.ones_like(s.v.data)


def _u_field(r, cfg, s, add_cor):
    """One barotropic step from rest with the given v; return the u field."""
    out, _ = barotropic_substeps_latlon_cgrid(
        s, dt_s=30.0, n_substeps=10, grid=r.grid, z_coord=r.z_coord,
        config=cfg, add_barotropic_coriolis=add_cor)
    return np.asarray(out.u.data)


def _coriolis_u(r, cfg, s):
    """Coriolis-ONLY U contribution = |u(cor on) − u(cor off)| (isolates the
    Coriolis term from the continuity→PGF gravity-wave response, which both
    paths share)."""
    return float(np.max(np.abs(
        _u_field(r, cfg, s, add_cor=True) - _u_field(r, cfg, s, add_cor=False))))


def test_checkerboard_is_coriolis_null_mode():
    """A 2Δx zonal checkerboard V drives ~no Coriolis U; a smooth V drives a lot."""
    r, cfg, s = _rest_setup()
    cor_checker = _coriolis_u(r, cfg, _set_v(s, _checkerboard_v(s)))
    cor_smooth = _coriolis_u(r, cfg, _set_v(s, _smooth_v(s)))
    # The smooth field feels the Coriolis operator...
    assert cor_smooth > 1e-4, f"smooth V felt no Coriolis: {cor_smooth:.2e}"
    # ...the checkerboard is annihilated by the 4-point average → null mode:
    # the discrete Coriolis exerts no restoring on it.
    assert cor_checker < 1e-3 * cor_smooth, (
        f"checkerboard is NOT a Coriolis null mode: cor_checker={cor_checker:.2e} "
        f"vs cor_smooth={cor_smooth:.2e}")


def test_cure_removes_in_substep_coriolis():
    """add_barotropic_coriolis=False (the explicit_ab2 cure) drops the term."""
    r, cfg, s = _rest_setup()
    s_smooth = _set_v(s, _smooth_v(s))
    u_on = _u_field(r, cfg, s_smooth, add_cor=True)
    u_off = _u_field(r, cfg, s_smooth, add_cor=False)
    # With Coriolis ON a smooth (divergence-free) V drives U purely via Coriolis;
    # with it OFF (the cure) there is no in-substep Coriolis and no U at all.
    assert float(np.max(np.abs(u_on))) > 1e-4, (
        f"control: Coriolis-on must drive U, got {np.max(np.abs(u_on)):.2e}")
    assert float(np.max(np.abs(u_off))) < 1e-12, (
        f"cure: add_barotropic_coriolis=False must remove the in-substep "
        f"Coriolis (no U from a divergence-free V), got {np.max(np.abs(u_off)):.2e}")


def _een_cfg(cfg):
    return cfg._replace(barotropic=cfg.barotropic._replace(
        barotropic_coriolis="een"))


def test_een_restores_the_checkerboard_null_mode():
    """NEMO EEN (``barotropic_coriolis="een"``) EXERTS a restoring on the 2Δx
    checkerboard the 4-pt average annihilates — the node-16 fix.

    The 4-pt-avg Coriolis gives ~0 U response to the checkerboard (null mode);
    the enstrophy-conserving EEN gives a response COMPARABLE to a smooth V
    (the mode is no longer invisible to the discrete Coriolis).
    """
    r, cfg, s = _rest_setup()
    een = _een_cfg(cfg)
    ck = _set_v(s, _checkerboard_v(s))
    sm = _set_v(s, _smooth_v(s))
    cor_ck_avg = _coriolis_u(r, cfg, ck)
    cor_ck_een = _coriolis_u(r, een, ck)
    cor_sm_een = _coriolis_u(r, een, sm)
    # avg annihilates the checkerboard...
    assert cor_ck_avg < 1e-3 * cor_sm_een, (
        f"control: avg should annihilate the checkerboard, got {cor_ck_avg:.2e}")
    # ...EEN restores it: the checkerboard now drives a Coriolis U of the same
    # order as a smooth field (no longer a null mode).
    assert cor_ck_een > 0.1 * cor_sm_een, (
        f"EEN failed to restore the null mode: cor_ck_een={cor_ck_een:.2e} "
        f"vs cor_sm_een={cor_sm_een:.2e}")
    # and EEN is a genuine change vs avg on the checkerboard.
    assert cor_ck_een > 100.0 * max(cor_ck_avg, 1e-30)


def test_een_barotropic_coriolis_conserves_energy():
    """The EEN barotropic Coriolis does ~no work (Σ hu·A·U·cor_u + hv·A·V·cor_v
    ≈ 0), the defining property of the enstrophy-conserving triad.  Residual is
    limited by the coastal Neumann fill / partial cells (same character as the
    baroclinic AL81 operator), not machine precision on a walled basin."""
    from legoesm.grids.latlon import ensure_geometry
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _build_een_barotropic_inputs,
        een_barotropic_coriolis,
    )
    from legoesm.ocean.vertical import compute_layer_thickness
    setup = build_silvestri_baroclinic_jet_setup(
        n_lat=24, n_lon=16, scheme="W9V", nlev=4,
        config=SilvestriJetConfig(), stabilize=False)
    grid = ensure_geometry(setup.grid)
    st = setup.initial_state
    h_bathy = st.H_bathy.data.astype(jnp.float64)
    mask = st.land_mask.data.astype(jnp.float64)
    um = st.u_mask.data.astype(jnp.float64)
    vm = st.v_mask.data.astype(jnp.float64)
    h_k = compute_layer_thickness(
        jnp.zeros_like(h_bathy), h_bathy, setup.z_coord,
        min_water_column_m=setup.model_config.min_water_column_m)
    pre = _build_een_barotropic_inputs(h_k, grid, mask, um, vm, jnp.float64)
    nlat, nlon = mask.shape
    rng = np.random.default_rng(0)
    u_r = jnp.asarray(rng.standard_normal((nlat, nlon + 1))) * um
    v_r = jnp.asarray(rng.standard_normal((nlat + 1, nlon))) * vm
    cu, cv = een_barotropic_coriolis(u_r, v_r, pre)
    area = grid.area
    a_u = 0.5 * (jnp.roll(area, 1, 1) + area)
    a_u = jnp.concatenate([a_u, a_u[:, :1]], 1)
    a_v = jnp.concatenate([area[:1], 0.5 * (area[:-1] + area[1:]), area[-1:]], 0)
    work = float(jnp.sum(pre["hu"] * a_u * u_r * cu)
                 + jnp.sum(pre["hv"] * a_v * v_r * cv))
    scale = float(jnp.sum(jnp.abs(pre["hu"] * a_u * u_r * cu))
                  + jnp.sum(jnp.abs(pre["hv"] * a_v * v_r * cv)))
    assert abs(work) / scale < 1e-2, (
        f"EEN Coriolis does spurious work: rel={work / scale:.2e}")


def test_een_pre_step_matches_substep_zero_live_term():
    """``barotropic_coriolis_een_pre_step`` (the live-split F_slow subtraction) IS
    exactly the substep-0 live EEN Coriolis the loop applies — so subtracting it
    from F_slow cancels the double-count at substep 0 (the node-16 LIVE cure).

    Pins the argument wiring: the helper must compose the SAME
    ``_depth_average_to_faces`` + ``_build_een_barotropic_inputs`` +
    ``een_barotropic_coriolis`` the substep loop calls internally (a swapped u/v
    or wrong thickness would break this equality)."""
    from legoesm.grids.latlon import ensure_geometry
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _build_een_barotropic_inputs,
        _depth_average_to_faces,
        barotropic_coriolis_een_pre_step,
        een_barotropic_coriolis,
    )
    from legoesm.ocean.vertical import compute_layer_thickness
    setup = build_silvestri_baroclinic_jet_setup(
        n_lat=24, n_lon=16, scheme="W9V", nlev=4,
        config=SilvestriJetConfig(), stabilize=False)
    grid = ensure_geometry(setup.grid)
    st = setup.initial_state
    h_bathy = st.H_bathy.data.astype(jnp.float64)
    mask = st.land_mask.data.astype(jnp.float64)
    um = st.u_mask.data.astype(jnp.float64)
    vm = st.v_mask.data.astype(jnp.float64)
    h_k = compute_layer_thickness(
        jnp.zeros_like(h_bathy), h_bathy, setup.z_coord,
        min_water_column_m=setup.model_config.min_water_column_m)
    nlat, nlon, nlev = h_k.shape
    rng = np.random.default_rng(1)
    u3 = jnp.asarray(rng.standard_normal((nlat, nlon + 1, nlev))) * um[..., None]
    v3 = jnp.asarray(rng.standard_normal((nlat + 1, nlon, nlev))) * vm[..., None]
    mwc = jnp.asarray(setup.model_config.min_water_column_m, dtype=jnp.float64)
    cu, cv = barotropic_coriolis_een_pre_step(
        u3, v3, h_k, grid, mask, um, vm, mwc, jnp.float64)
    # independent reconstruction of the substep-0 live term
    pre = _build_een_barotropic_inputs(h_k, grid, mask, um, vm, jnp.float64)
    U, V = _depth_average_to_faces(u3, v3, h_k, mwc, mask, um, vm, grid)
    cu_ref, cv_ref = een_barotropic_coriolis(U, V, pre)
    np.testing.assert_allclose(np.asarray(cu), np.asarray(cu_ref), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(cv), np.asarray(cv_ref), rtol=0, atol=0)
    # non-vacuous: the term is actually non-trivial
    assert float(np.max(np.abs(np.asarray(cu)))) > 0.0


def test_explicit_ab2_config_gates_in_substep_coriolis():
    """The §5 faithful stack wires coriolis_scheme=explicit_ab2 (=> term off)."""
    r = build_silvestri_baroclinic_jet_setup(
        n_lat=24, n_lon=16, scheme="W9V", nlev=4,
        config=SilvestriJetConfig(), stabilize=False)
    mc = r.model_config
    assert mc.coriolis_scheme == "explicit_ab2"
    assert mc.barotropic.barotropic_solver == "explicit_substep"
    assert mc.barotropic.barotropic_slow_forcing_ab2 is True
    assert mc.barotropic.barotropic_diffusion_alpha == 0.0
