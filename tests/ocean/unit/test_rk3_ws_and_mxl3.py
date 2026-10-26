"""Direct unit gates for the NEMO WS-RK3 integrator (rk3_ws), the
skip_lateral_viscosity stage gate, and the nn_mxl=3 mixing-length scans
(consolidated-review findings 2+3, 2026-07-16)."""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import numpy as np
import pytest


def test_ws_rk3_linear_stability_polynomial():
    """The 3-stage WS recurrence u_s = u0 + c_s*h*F(u_{s-1}) with
    c=(1/3,1/2,1) and F(u)=z*u yields R(z) = 1 + z + z^2/2 + z^3/6 —
    identical to SSP-RK3 (the explicit_ab2 Coriolis bound carries over)."""
    for z in (0.1, -0.3, 0.5j, 1.0j, -0.2 + 0.7j):
        u0 = 1.0
        u1 = u0 + (1.0 / 3.0) * z * u0
        u2 = u0 + (1.0 / 2.0) * z * u1
        u3 = u0 + z * u2
        R = 1.0 + z + z**2 / 2.0 + z**3 / 6.0
        assert abs(u3 - R) < 1e-14


def _gyre_model(**cfg_over):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_recipe import build_nemo_gyre_recipe

    r = build_nemo_gyre_recipe()
    mc = r.model_config._replace(**cfg_over) if cfg_over else r.model_config
    return r, LatLonCGridOceanModel(r.grid, r.z_coord, mc)


def test_skip_lateral_viscosity_gate_nonvacuous_and_closure():
    """skip_lateral_viscosity=True removes exactly the lateral-viscosity
    tendency (nonzero difference == the Ah term) and keeps the
    sum-of-components == du_dt closure with zeroed viscosity diags."""
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        latlon_cgrid_ocean_baroclinic_tendencies,
    )
    from legoesm.ocean.fidelity.nemo_recipe import (
        _NEMO_GYRE_DT_S,
        build_nemo_gyre_recipe,
    )

    r = build_nemo_gyre_recipe()
    st = r.initial_state
    # seed a sheared flow so the viscosity term is nonzero
    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.standard_normal(st.u.data.shape) * 1e-2)
    v = jnp.asarray(rng.standard_normal(st.v.data.shape) * 1e-2)
    st = st._replace(u=st.u.replace(data=u * st.u_mask.data[..., None]),
                     v=st.v.replace(data=v * st.v_mask.data[..., None]))
    out = {}
    for skip in (False, True):
        td, diag = latlon_cgrid_ocean_baroclinic_tendencies(
            st, r.grid, r.z_coord, r.model_config, dt=_NEMO_GYRE_DT_S,
            diagnose_momentum=True, skip_lateral_viscosity=skip)
        out[skip] = (np.asarray(td.du_dt.data), diag)
    dd = out[False][0] - out[True][0]
    ah = np.asarray(out[False][1].Ah_lap_u.data)
    # the difference IS the viscosity term
    np.testing.assert_allclose(dd, ah, rtol=0, atol=1e-15)
    assert float(np.max(np.abs(ah))) > 0.0
    # skipped run reports zero viscosity diags
    assert float(np.max(np.abs(np.asarray(out[True][1].Ah_lap_u.data)))) == 0.0


def test_rk3_ws_differs_from_rk3_and_is_finite():
    """Non-vacuity + byte-identity guard: rk3_ws changes the step vs rk3
    (different stage structure), and the rk3 path is untouched by the new
    branch (steps finite either way)."""
    from legoesm.ocean.fidelity.nemo_recipe import (
        _NEMO_GYRE_DT_S,
        nemo_gyre_wind_forcing,
    )

    r, m_ws = _gyre_model()
    assert r.model_config.momentum_time_integrator == "rk3_ws"
    _, m_ssp = _gyre_model(
        momentum_time_integrator="rk3", tracer_time_integrator="rk3")
    st = r.initial_state
    n_lat, n_lon = st.T.data.shape[0], st.T.data.shape[1]
    sf = nemo_gyre_wind_forcing(n_lat, n_lon, 0.0)
    a = m_ws.step(st, dt=_NEMO_GYRE_DT_S, surface_forcing=sf)
    b = m_ssp.step(st, dt=_NEMO_GYRE_DT_S, surface_forcing=sf)
    for f in (a.u.data, a.v.data, b.u.data, b.v.data):
        assert bool(jnp.all(jnp.isfinite(f)))
    a2 = m_ws.step(a, dt=_NEMO_GYRE_DT_S, surface_forcing=sf)
    b2 = m_ssp.step(b, dt=_NEMO_GYRE_DT_S, surface_forcing=sf)
    assert float(jnp.max(jnp.abs(a2.u.data - b2.u.data))) > 0.0


def test_ene_total_weno_momentum_rejected():
    """Review finding 1: ene_total + WENO momentum would silently drop f x u
    (the WENO branch never receives f_vtx) — must raise."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_recipe import build_nemo_gyre_recipe

    r = build_nemo_gyre_recipe()
    with pytest.raises(ValueError, match="ene_total"):
        LatLonCGridOceanModel(
            r.grid, r.z_coord,
            r.model_config._replace(momentum_advection="weno5"))


def test_nn_mxl3_scans_match_direct_loop():
    """The lax.scan lup/ldown implementation == a direct NEMO-style index loop
    (zdftke.F90:692-707), including the asymmetric e3t(k-1)/e3t(k+1) bounds,
    l_k=min, l_eps=sqrt(lup*ldown), and the ln_mxl0 anchor at the surface."""
    from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
    from legoesm.ocean.physics.vertical_mixing.tke import (
        _mixing_length_floor, compute_mixing_lengths,
    )

    rng = np.random.default_rng(3)
    n = 9
    e = jnp.asarray(np.abs(rng.standard_normal((2, 2, n - 1))) * 1e-3)
    N2 = jnp.asarray(np.abs(rng.standard_normal((2, 2, n - 1))) * 1e-6)
    dz_half = jnp.asarray(10.0 + np.abs(rng.standard_normal((2, 2, n - 1))))
    dz_cell = jnp.asarray(10.0 + np.abs(rng.standard_normal((2, 2, n))))
    anchor = jnp.asarray(np.abs(rng.standard_normal((2, 2))) * 0.5 + 0.04)
    cfg = TKEConfig(tke_mxl_choice=3)
    mxl_min = _mixing_length_floor(cfg)
    l_k, l_eps = compute_mixing_lengths(
        e, N2, dz_half, cfg, signed_n2=True, dz_cell=dz_cell,
        l_surface_anchor=anchor)

    # direct loop reference
    sqrt2e = np.sqrt(2.0) * np.sqrt(np.maximum(np.asarray(e), 0.0))
    l_int = np.maximum(sqrt2e / np.sqrt(np.maximum(np.asarray(N2), 1e-12)),
                       mxl_min)
    l_w = np.concatenate([np.asarray(anchor)[..., None], l_int], axis=-1)
    e3 = np.asarray(dz_cell)
    lup = l_w.copy()
    for k in range(1, n):
        lup[..., k] = np.minimum(lup[..., k - 1] + e3[..., k - 1],
                                 l_w[..., k])
    ldn = l_w.copy()
    # #1226 ldown-seed fix (zdftke.F90:678 + 786-789): the deepest carried
    # row is itself BOUNDED, seeded from NEMO's untouched zmxlm(jpk) =
    # rmxl_min; under the legacy (+1-row) dz_cell contract the e3t paired
    # with that seed step is dz_cell's own last row (the e3t(jpk) proxy).
    ldn[..., n - 1] = np.minimum(mxl_min + e3[..., n - 1],
                                 l_w[..., n - 1])
    for k in range(n - 2, 0, -1):
        ldn[..., k] = np.minimum(ldn[..., k + 1] + e3[..., k + 1],
                                 l_w[..., k])
    ref_k = np.maximum(np.minimum(lup, ldn), mxl_min)[..., 1:]
    ref_e = np.maximum(np.sqrt(lup * ldn), mxl_min)[..., 1:]
    np.testing.assert_allclose(np.asarray(l_k), ref_k, rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.asarray(l_eps), ref_e, rtol=0, atol=1e-14)

    # --- NEMO nn_mxl=2 (tke_mxl_choice=4) on the SAME batch --------------
    # CASE(2) applies both sweeps sequentially in place and sets
    # zmxld = zmxlm, so the reference is ref_k for BOTH lengths.  Reuses this
    # test's direct-loop machinery rather than standing up a second oracle.
    cfg2 = TKEConfig(tke_mxl_choice=4)
    l_k2, l_eps2 = compute_mixing_lengths(
        e, N2, dz_half, cfg2, signed_n2=True, dz_cell=dz_cell,
        l_surface_anchor=anchor)
    np.testing.assert_allclose(np.asarray(l_k2), ref_k, rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.asarray(l_eps2), ref_k, rtol=0, atol=1e-14)
    # non-vacuity: the two schemes must actually DIFFER on this batch, else
    # "choice 4 collapses l_eps" would be untestable here.
    assert np.max(np.abs(ref_e - ref_k)) > 1e-6


def test_dissipation_discretization_dispatch_and_forms():
    """NEMO 1.5/0.5 dissipation split: unknown value raises (dispatch
    hardening); split vs backward-Euler agree at small dt*diss (both are
    first-order consistent) and differ at large dt*diss (non-vacuity)."""
    import jax.numpy as jnp
    import numpy as np
    import pytest

    from legoesm.ocean.fidelity.nemo_recipe import build_nemo_gyre_recipe
    from legoesm.ocean.physics.vertical_mixing.tke import (
        tke_integrate_post_mixing, tke_set_diffusivities)
    from legoesm.ocean.eos import nemo_roquet_eos

    r = build_nemo_gyre_recipe()
    cfg = r.model_config.physics.vertical_mixing.tke
    assert cfg.dissipation_discretization == "nemo_1p5_split"

    NL = 12
    z = np.linspace(5, 500, NL)
    T = jnp.asarray(20.0 - 0.01 * z)[None, None, :]
    S = jnp.full((1, 1, NL), 35.0)
    p = jnp.asarray(1026.0 * 9.80665 * z)
    rho = nemo_roquet_eos(T[0, 0], S[0, 0], p, rho0=1026.0)[None, None, :]
    dzh = jnp.asarray(np.diff(z))[None, None, :]
    dz_ref = jnp.asarray(np.gradient(z))
    tke = jnp.full((1, 1, NL - 1), 1.0e-3)
    common = dict(
        taum_surface=jnp.asarray([[0.05]]), p_cell=p[None, None, :],
        dz_ref=dz_ref, jacobian=jnp.ones((1, 1)),
        eos_fn=lambda TT, SS, pp: nemo_roquet_eos(TT, SS, pp, rho0=1026.0),
        z_interface=jnp.asarray(-0.5 * (z[:-1] + z[1:]))[: NL - 1],
        dz_surface=jnp.asarray([[z[0]]]),
        surface_tmask=jnp.ones((1, 1)))
    zeros = jnp.zeros((1, 1, NL - 1))

    u3 = jnp.zeros((1, 1, NL))

    def solve(disc, dt):
        c = cfg._replace(dissipation_discretization=disc)
        _, _, ctx = tke_set_diffusivities(
            u3, u3,
            T, S, rho, dzh, tke,
            jnp.asarray([[0.05]]), jnp.asarray([[0.0]]), c, 1026.0, 9.80665,
            **common)
        return np.asarray(tke_integrate_post_mixing(
            ctx, zeros, zeros, jnp.zeros((1, 1)), dt, c))

    with pytest.raises(ValueError, match="dissipation_discretization"):
        solve("typo_scheme", 100.0)

    e_be_small = solve("backward_euler", 1.0)
    e_sp_small = solve("nemo_1p5_split", 1.0)
    np.testing.assert_allclose(e_be_small, e_sp_small, rtol=1e-3)

    e_be_big = solve("backward_euler", 1.0e5)
    e_sp_big = solve("nemo_1p5_split", 1.0e5)
    assert float(np.max(np.abs(e_be_big - e_sp_big))) > 0.0


def test_evd_two_level_trigger():
    """NEMO zdfevd MIN(rn2,rn2b): with two_level_trigger, a column unstable at
    EITHER time level fires EVD (max of the two single-level coefficient
    fields); flag off or before_tracers=None => single-level (bit-identical
    legacy)."""
    import jax.numpy as jnp
    import numpy as np

    from legoesm.ocean.fidelity.nemo_recipe import build_nemo_gyre_recipe
    from legoesm.ocean.physics.vertical_mixing.k_profiles import (
        _enhanced_diffusion_K,
    )

    r = build_nemo_gyre_recipe()
    conv = r.model_config.physics.convection
    assert conv.enhanced_diffusion.two_level_trigger is True
    st = r.initial_state  # stably stratified IC
    nlat, nlon, nlev = st.T.data.shape

    # "before" tracers with an UNSTABLE inversion at one interior cell
    T_b = st.T.data.at[5, 5, 3].add(-2.0)   # cold above warm -> unstable below
    S_b = st.S.data
    K_now, A_now = _enhanced_diffusion_K(st, r.z_coord, conv)
    K_2lv, A_2lv = _enhanced_diffusion_K(
        st, r.z_coord, conv, before_tracers=(T_b, S_b))
    # the before-level instability must fire in the two-level result
    assert float(jnp.max(K_2lv - K_now)) > 50.0   # K_conv=100 appears
    # and two-level == max(now, before-only) pointwise
    st_b = st._replace(T=st.T.replace(data=T_b))
    K_bef, A_bef = _enhanced_diffusion_K(st_b, r.z_coord, conv)
    np.testing.assert_allclose(np.asarray(K_2lv),
                               np.maximum(np.asarray(K_now), np.asarray(K_bef)),
                               atol=1e-12)
    # None before_tracers => single-level (legacy)
    K_none, _ = _enhanced_diffusion_K(st, r.z_coord, conv, before_tracers=None)
    np.testing.assert_array_equal(np.asarray(K_none), np.asarray(K_now))


# ---------------------------------------------------------------------------
# EnhancedDiffusionConfig.evd_n2_time_level (#1317 S17)
#
# NEMO's zdfevd trigger arms (src/OCE/ZDF/zdfevd.F90:93-94 avt, :119-120 avm)
#   IF( MIN( rn2(ji,jj,jk), rn2b(ji,jj,jk) ) <= -1.e-12 )
# are built at cfgs/DINO/MY_SRC/stpmlf.F90:186-187
#   CALL bn2( ts(:,:,:,:,Nbb), rab_b, rn2b, Nnn )   ! BEFORE T/S, NOW geometry
#   CALL bn2( ts(:,:,:,:,Nnn), rab_n, rn2 , Nnn )   ! NOW    T/S, NOW geometry
# i.e. Nnn and Nbb tracers, Nnn geometry for BOTH arms. legoESM's leap-frog
# hands compute_vertical_K_profiles the POST-EXPLICIT (Kaa) state, so the
# default "solver_state" arms are (Kaa, Nnn) on Kaa geometry.
#
# These tests drive the REAL production entry point
# (compute_vertical_K_profiles), NOT the private _enhanced_diffusion_K helper
# and NOT a hand-written time-level selection -- reverting the selection block
# in k_profiles.py makes test_evd_n2_time_level_selects_nemo_arms fail.
# ---------------------------------------------------------------------------
def _evd_tl_fixture(evd_n2_time_level, partial_cells=False):
    """(state_kaa, z_coord, physics_config, nn, bb, eta_nn) for the arm tests.

    ``state_kaa`` carries a cold spike at [5,5,3] that exists ONLY at the Kaa
    level; ``bb`` carries a different cold spike at [6,6,3] that exists ONLY at
    the Nbb level. NEMO's pair must see the second and not the first.

    ``partial_cells=True`` swaps the flat-bottom z* coord for a STEPPED
    ``OceanPartialCellCoordinate`` (which carries ``is_active``) and rock-fills
    T=S=0 below the seafloor in BOTH time levels -- the only configuration
    that reaches the sub-seafloor-extrapolation branch of the NEMO arm.
    """
    import numpy as np

    from legoesm.ocean.fidelity.nemo_recipe import build_nemo_gyre_recipe
    from legoesm.ocean.physics.vertical_mixing.config import (
        VerticalMixingConfig,
    )
    from legoesm.ocean.vertical import create_partial_cell_coordinate

    r = build_nemo_gyre_recipe()
    phys = r.model_config.physics
    ed = phys.convection.enhanced_diffusion._replace(
        evd_n2_time_level=evd_n2_time_level)
    phys = phys._replace(
        # scheme="none" isolates the convection branch -- the closure is not
        # what is under test and would need surface forcing.
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        convection=phys.convection._replace(enhanced_diffusion=ed),
    )
    st = r.initial_state
    z_coord = r.z_coord
    T_nn, S_nn = st.T.data, st.S.data
    T_kaa = T_nn.at[5, 5, 3].add(-2.0)      # Kaa-only inversion
    T_bb = T_nn.at[6, 6, 3].add(-2.0)       # Nbb-only inversion
    eta_nn = st.eta.data
    if partial_cells:
        ny, nx, nz = T_nn.shape
        dz = np.asarray(z_coord.dz_ref)
        gdepw = np.concatenate([[0.0], np.cumsum(dz)])
        # A different bottom level per column so the sub-seafloor row is a
        # genuine staircase, not one flat slab.
        kbot = 5 + ((np.arange(ny)[:, None] + np.arange(nx)[None, :])
                    % (nz - 7))
        H = jnp.asarray(gdepw[kbot + 1] - 0.4 * dz[kbot])
        z_coord = create_partial_cell_coordinate(z_coord, H)
        rock = z_coord.is_active
        T_nn, S_nn = jnp.where(rock, T_nn, 0.0), jnp.where(rock, S_nn, 0.0)
        T_kaa, T_bb = jnp.where(rock, T_kaa, 0.0), jnp.where(rock, T_bb, 0.0)
        st = st._replace(H_bathy=st.H_bathy.replace(data=H))
    state_kaa = st._replace(
        T=st.T.replace(data=T_kaa),
        S=st.S.replace(data=S_nn),
        # Kaa eta differs from Nnn eta -> Kaa gdept / z* jacobian.
        eta=st.eta.replace(data=eta_nn + 0.5),
    )
    return (state_kaa, z_coord, phys, (T_nn, S_nn), (T_bb, S_nn), eta_nn)


def test_evd_n2_time_level_selects_nemo_arms():
    """"nemo_now_before" fires on the Nbb inversion and NOT on the Kaa one;
    "solver_state" (default) does the opposite -- proving the option changes
    WHICH cells fire, through the production call path."""
    import numpy as np

    from legoesm.ocean.physics.vertical_mixing.k_profiles import (
        compute_vertical_K_profiles,
    )

    out = {}
    for tl in ("solver_state", "nemo_now_before"):
        state, z, phys, nn, bb, eta_nn = _evd_tl_fixture(tl)
        K, _ = compute_vertical_K_profiles(
            state, z, None, phys,
            n2_tracers=nn, n2_tracers_before=bb, eta_now=eta_nn)
        out[tl] = np.asarray(K)

    fire = 50.0                              # K_conv=100 on the gyre card
    legacy, nemo = out["solver_state"], out["nemo_now_before"]
    assert legacy.shape == nemo.shape
    # The two masks are NOT the same -- the option has teeth.
    assert not np.array_equal(legacy > fire, nemo > fire)
    # Kaa-only inversion at [5,5]: fires under solver_state, not under NEMO's.
    assert (legacy[5, 5] > fire).any(), "Kaa arm should fire the [5,5] spike"
    assert not (nemo[5, 5] > fire).any(), (
        "NEMO's arms are Nnn/Nbb -- the Kaa-only spike must NOT fire")
    # Nbb-only inversion at [6,6]: fires under NEMO's pair, not under legacy
    # (whose second arm is n2_tracers = the Nnn level).
    assert (nemo[6, 6] > fire).any(), "Nbb arm should fire the [6,6] spike"
    assert not (legacy[6, 6] > fire).any(), (
        "solver_state's arms are Kaa/Nnn -- the Nbb-only spike must NOT fire")


def test_evd_n2_time_level_consumes_eta_now():
    """The NEMO path takes its GEOMETRY (gdept / z* jacobian) from ``eta_now``,
    not from the solver state's eta: perturbing eta_now moves the answer.

    Run on the SMOOTH branch: with the hard 0/100 flag a small geometric
    change cannot flip a cell's N² sign, so the hard branch cannot see
    (and therefore cannot certify) the geometry substitution.
    """
    import numpy as np

    from legoesm.ocean.physics.vertical_mixing.k_profiles import (
        compute_vertical_K_profiles,
    )

    state, z, phys, nn, bb, eta_nn = _evd_tl_fixture("nemo_now_before")
    ed = phys.convection.enhanced_diffusion._replace(smooth_transition=True)
    phys = phys._replace(
        convection=phys.convection._replace(enhanced_diffusion=ed))

    def K(eta):
        out, _ = compute_vertical_K_profiles(
            state, z, None, phys,
            n2_tracers=nn, n2_tracers_before=bb, eta_now=eta)
        return np.asarray(out)

    assert np.array_equal(K(eta_nn), K(eta_nn)), "not deterministic"
    assert not np.array_equal(K(eta_nn), K(eta_nn + 5.0))
    # ... and it is eta_now, not the solver state's eta, that is in force:
    # feeding eta_now = the Kaa eta reproduces the solver-state geometry.
    np.testing.assert_array_equal(K(state.eta.data),
                                  K(np.asarray(eta_nn) + 0.5))


def test_evd_n2_time_level_default_is_bit_identical():
    """Omitting the new kwargs entirely reproduces the legacy result exactly."""
    import numpy as np

    from legoesm.ocean.physics.vertical_mixing.k_profiles import (
        compute_vertical_K_profiles,
    )

    state, z, phys, nn, bb, eta_nn = _evd_tl_fixture("solver_state")
    assert phys.convection.enhanced_diffusion.evd_n2_time_level == \
        "solver_state"
    K_legacy, A_legacy = compute_vertical_K_profiles(state, z, None, phys,
                                                     n2_tracers=nn)
    K_thread, A_thread = compute_vertical_K_profiles(
        state, z, None, phys,
        n2_tracers=nn, n2_tracers_before=bb, eta_now=eta_nn)
    assert np.array_equal(np.asarray(K_legacy), np.asarray(K_thread))
    assert np.array_equal(np.asarray(A_legacy), np.asarray(A_thread))


def test_evd_n2_time_level_dispatch_hardening():
    """Unknown value raises; the NEMO path raises rather than silently
    degrading when a required input is missing."""
    import pytest

    from legoesm.ocean.physics.vertical_mixing.k_profiles import (
        compute_vertical_K_profiles,
    )

    state, z, phys, nn, bb, eta_nn = _evd_tl_fixture("typo_level")
    with pytest.raises(ValueError, match="evd_n2_time_level"):
        compute_vertical_K_profiles(state, z, None, phys, n2_tracers=nn,
                                    n2_tracers_before=bb, eta_now=eta_nn)

    state, z, phys, nn, bb, eta_nn = _evd_tl_fixture("nemo_now_before")
    for kw in ({"n2_tracers": None}, {"n2_tracers_before": None},
               {"eta_now": None}):
        call = dict(n2_tracers=nn, n2_tracers_before=bb, eta_now=eta_nn)
        call.update(kw)
        with pytest.raises(ValueError, match=list(kw)[0]):
            compute_vertical_K_profiles(state, z, None, phys, **call)

    # two_level_trigger=False would silently drop the Nbb arm -> raise.
    state, z, phys, nn, bb, eta_nn = _evd_tl_fixture("nemo_now_before")
    ed = phys.convection.enhanced_diffusion._replace(two_level_trigger=False)
    phys = phys._replace(
        convection=phys.convection._replace(enhanced_diffusion=ed))
    with pytest.raises(ValueError, match="two_level_trigger"):
        compute_vertical_K_profiles(state, z, None, phys, n2_tracers=nn,
                                    n2_tracers_before=bb, eta_now=eta_nn)


def test_evd_nemo_arms_extrapolate_nbb_below_seafloor(monkeypatch):
    """On a PARTIAL-CELL coord the Nbb arm's rock fill is extrapolated away
    before it reaches the trigger (the ``_is_active`` block inside the
    ``nemo_now_before`` branch of ``compute_vertical_K_profiles``).

    WHY THIS IS A WHITE-BOX TEST, stated so nobody "strengthens" it into a
    return-value assertion that can only ever pass vacuously: MEASURED on the
    production DINO topo bridge (fp64, 199x52x36, 30394 sub-seafloor cells,
    the extrapolation moves 29966 of them by up to 26.39 degC), the guard
    changes EXACTLY ZERO wet interfaces -- ``max|dA_v| = 0.0`` and fired-mask
    ``XOR = 0`` after ``_wet_interface_mask``. That is structural, not luck:
    interior interface ``j`` is wet iff T-cell ``j+1`` is active, so a wet
    interface's N^2 only ever reads active cells and the rock fill cannot
    reach it. The branch is defence-in-depth on that masking invariant, so
    the ONLY non-vacuous observable is the tracer pair the trigger is handed.
    (A companion "perturb the rock fill, assert K unchanged" test was written
    and DELETED as structurally vacuous: the guard overwrites every dry cell
    with the deepest-active value, so the two arms become bit-identical
    BEFORE the masking invariant is ever exercised.)

    Deleting the ``_is_active`` block in the NEMO arm makes this test fail.
    """
    import numpy as np

    from legoesm.ocean.physics.vertical_mixing import k_profiles as KP
    from legoesm.ocean.vertical import extrapolate_below_seafloor

    state, z, phys, nn, bb, eta_nn = _evd_tl_fixture("nemo_now_before",
                                                     partial_cells=True)
    assert getattr(z, "is_active", None) is not None, "fixture is not partial"
    dry = ~np.asarray(z.is_active)
    assert dry.any(), "fixture has no sub-seafloor cells"
    assert np.asarray(bb[0])[dry].max() == 0.0, "Nbb is not rock-filled"

    seen = {}
    real = KP._enhanced_diffusion_K

    def spy(state_, z_, conv, **kw):
        seen["before"] = kw["before_tracers"]
        seen["T_arm0"] = np.asarray(state_.T.data)
        return real(state_, z_, conv, **kw)

    monkeypatch.setattr(KP, "_enhanced_diffusion_K", spy)
    KP.compute_vertical_K_profiles(state, z, None, phys, n2_tracers=nn,
                                   n2_tracers_before=bb, eta_now=eta_nn)
    assert "before" in seen, "the EVD branch never ran"

    T_bb_seen = np.asarray(seen["before"][0])
    S_bb_seen = np.asarray(seen["before"][1])
    # (a) the rock fill is gone from the arm the trigger actually sees ...
    assert not np.array_equal(T_bb_seen[dry], np.asarray(bb[0])[dry]), (
        "the Nbb arm still carries the raw T=0 rock fill -- the sub-seafloor "
        "extrapolation in the nemo_now_before branch was removed")
    # (b) ... and it is exactly the deepest-active fill, not some other value.
    np.testing.assert_array_equal(
        T_bb_seen, np.asarray(extrapolate_below_seafloor(bb[0], z)))
    np.testing.assert_array_equal(
        S_bb_seen, np.asarray(extrapolate_below_seafloor(bb[1], z)))
    # (c) both arms of the one MIN() share the convention (the NOW arm gets
    #     the same treatment at the top of compute_vertical_K_profiles).
    np.testing.assert_array_equal(
        np.asarray(seen["T_arm0"]),
        np.asarray(extrapolate_below_seafloor(nn[0], z)))
    # (d) active cells are untouched by either extrapolation.
    act = ~dry
    np.testing.assert_array_equal(T_bb_seen[act], np.asarray(bb[0])[act])
