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
    _, m_ssp = _gyre_model(momentum_time_integrator="rk3")
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
        compute_mixing_lengths,
    )

    rng = np.random.default_rng(3)
    n = 9
    e = jnp.asarray(np.abs(rng.standard_normal((2, 2, n - 1))) * 1e-3)
    N2 = jnp.asarray(np.abs(rng.standard_normal((2, 2, n - 1))) * 1e-6)
    dz_half = jnp.asarray(10.0 + np.abs(rng.standard_normal((2, 2, n - 1))))
    dz_cell = jnp.asarray(10.0 + np.abs(rng.standard_normal((2, 2, n))))
    anchor = jnp.asarray(np.abs(rng.standard_normal((2, 2))) * 0.5 + 0.04)
    cfg = TKEConfig(tke_mxl_choice=3)
    l_k, l_eps = compute_mixing_lengths(
        e, N2, dz_half, cfg, signed_n2=True, dz_cell=dz_cell,
        l_surface_anchor=anchor)

    # direct loop reference
    sqrt2e = np.sqrt(2.0) * np.sqrt(np.maximum(np.asarray(e), 0.0))
    l_int = np.maximum(sqrt2e / np.sqrt(np.maximum(np.asarray(N2), 1e-12)),
                       cfg.mxl_min)
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
    ldn[..., n - 1] = np.minimum(cfg.mxl_min + e3[..., n - 1],
                                 l_w[..., n - 1])
    for k in range(n - 2, 0, -1):
        ldn[..., k] = np.minimum(ldn[..., k + 1] + e3[..., k + 1],
                                 l_w[..., k])
    ref_k = np.maximum(np.minimum(lup, ldn), cfg.mxl_min)[..., 1:]
    ref_e = np.maximum(np.sqrt(lup * ldn), cfg.mxl_min)[..., 1:]
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
        dz_surface=jnp.asarray([[z[0]]]))
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
