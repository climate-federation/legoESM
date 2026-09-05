"""The real (volume) freshwater closure must DILUTE the top cell like NEMO's
non-linear free surface: dS_1 = -S_1 F dt/(rho h_1), S and T below unchanged,
column salt conserved.

Before 2026-09-05 the closure only stretched the column uniformly (both
reviewers confirmed), so a 180-day arm carried no surface dilution at all:
freezing regions went fresh, melting regions salty, Amazon plume too salty.
These tests are the exact one-column certificate the fix was pre-registered
on (harmonization_plan_2026-09-05.md), and they FAIL on the pre-fix code.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.ocean.freshwater import real_freshwater_dilution_tendencies


def _column(nlev=6, S0=35.0, T0=10.0, h=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0)):
    h = np.asarray(h, dtype=np.float64)[None, :]
    S = np.full((1, nlev), S0)
    T = np.full((1, nlev), T0)
    return S, T, h


@pytest.mark.parametrize("F", [3.0e-8, -3.0e-8])
def test_helper_transport_matches_the_exact_layer_algebra(F):
    """Transport term alone: top -S F (H-h1)/(H h1), every layer below
    +S F/H (which the z-star stretch -S F/H cancels), T untouched."""
    S, T, h = _column()
    mask = np.ones(1)
    dS, dT = real_freshwater_dilution_tendencies(np.array([F]), S, T, h, mask)
    dS, dT = np.asarray(dS), np.asarray(dT)
    H = h.sum(); h1 = h[0, 0]
    assert dS[0, 0] == pytest.approx(-35.0 * F * (H - h1) / (H * h1), rel=1e-12)
    assert np.allclose(dS[0, 1:], 35.0 * F / H, rtol=1e-12)
    # T: the surface carries T_1, so the transport alone gives +T F/H in
    # every layer -- exactly what the z-star stretch (-T F/H) removes
    assert np.allclose(dT, 10.0 * F / H, rtol=1e-12)
    # with the uniform z-star stretch (-C F/H per layer) the sum is NEMO's
    assert dS[0, 0] - 35.0 * F / H == pytest.approx(-35.0 * F / h1, rel=1e-12)
    assert np.allclose(dS[0, 1:] - 35.0 * F / H, 0.0, atol=1e-15)
    assert np.allclose(dT - 10.0 * F / H, 0.0, atol=1e-15)


def test_helper_conserves_column_salt_for_any_profile():
    rng = np.random.default_rng(0)
    S = 30.0 + 5.0 * rng.random((4, 7))
    T = 2.0 + 20.0 * rng.random((4, 7))
    h = 1.0 + 9.0 * rng.random((4, 7))
    F = 1e-7 * rng.standard_normal(4)
    mask = np.array([1.0, 1.0, 0.0, 1.0])
    dS, dT = real_freshwater_dilution_tendencies(F, S, T, h, mask)
    col = np.sum(h * np.asarray(dS), axis=-1)
    assert np.allclose(col, 0.0, atol=1e-12 * np.max(np.abs(S * F[:, None])))
    assert np.allclose(np.asarray(dS)[2], 0.0) and np.allclose(np.asarray(dT)[2], 0.0)
    # heat: the surface carries T_1, so the column heat changes only by the
    # water added at T_1 -- i.e. sum h dT = F T_1 (exactly, no other source)
    heat = np.sum(h * np.asarray(dT), axis=-1)
    assert np.allclose(heat, F * T[:, 0] * mask, rtol=1e-10, atol=1e-18)


def test_dry_layers_below_the_seafloor_carry_nothing():
    S, T, h = _column()
    h = h.copy(); h[0, 4:] = 0.0                 # partial column, 4 wet layers
    dS, dT = real_freshwater_dilution_tendencies(np.array([2e-8]), S, T, h, np.ones(1))
    dS = np.asarray(dS)
    assert np.all(dS[0, 4:] == 0.0)
    H = h[0, :4].sum()
    assert dS[0, 0] == pytest.approx(-35.0 * 2e-8 * (H - h[0, 0]) / (H * h[0, 0]), rel=1e-12)


class TestOneColumnExactStep:
    """Full lat-lon model step, rest state, uniform S/T, uniform pure-water
    flux: the ONLY changes are the volume rise and the dilution, so the top
    cell must land on NEMO's exact S_new = S h1/(h1 + F dt/rho) and the
    layers below must not move."""

    @staticmethod
    def _run(closure, F_kg=3.0e-5, dt=300.0):
        import jax.numpy as jnp
        from legoesm.core.precision import PrecisionPolicy, set_policy
        set_policy(PrecisionPolicy.fp64())
        from legoesm.grids.latlon import create_beta_plane_cgrid_geometry
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
        from legoesm.ocean.freshwater import FreshwaterForcing
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        from legoesm.ocean.state import LatLonCGridOceanConfig
        from legoesm.ocean.vertical import compute_layer_thickness, create_ocean_z_star

        g = create_beta_plane_cgrid_geometry(8, 8, dx_m=50e3, f0=1e-4, beta=0.0)
        z = create_ocean_z_star(3, H_max=300.0)
        st = rest_state_latlon_cgrid_ocean(g, z, land_lat_threshold=90.0)
        # uniform tracers: the only tendencies left are volume + dilution
        st = st._replace(T=st.T.replace(data=jnp.full_like(st.T.data, 10.0)),
                         S=st.S.replace(data=jnp.full_like(st.S.data, 35.0)))
        cfg = LatLonCGridOceanConfig.from_flat(
            freshwater_closure=closure, normalize_freshwater=False,
            use_conservation_fixer=False, fix_salt=False, fix_volume=False,
            fix_eta_drift=True, enable_runtime_checks=False,
            n_barotropic_substeps=20)
        model = LatLonCGridOceanModel(g, z, cfg)
        shp = st.eta.data.shape
        fw = FreshwaterForcing(precip=jnp.full(shp, F_kg), evap=jnp.zeros(shp),
                               runoff=jnp.zeros(shp), ice_fw=jnp.zeros(shp))
        h0 = compute_layer_thickness(st.eta.data, st.H_bathy.data, z,
                                     min_water_column_m=cfg.min_water_column_m)
        s1 = model.step(st, dt, freshwater=fw)
        h1 = compute_layer_thickness(s1.eta.data, s1.H_bathy.data, z,
                                     min_water_column_m=cfg.min_water_column_m)
        wet = np.asarray(st.land_mask.data) > 0.5
        return (np.asarray(st.S.data), np.asarray(s1.S.data), np.asarray(st.T.data),
                np.asarray(s1.T.data), np.asarray(h0), np.asarray(h1), wet, cfg.rho_0)

    def test_real_closure_dilutes_the_top_cell_exactly(self):
        S0, S1, T0, T1, h0, h1, wet, rho0 = self._run("real_freshwater")
        F_dt = 3.0e-5 * 300.0 / rho0                       # metres of water added
        # interior columns only (the same 6x6 block the domain's walls leave
        # untouched by any pressure-gradient noise)
        sl = (slice(1, -1), slice(1, -1))
        top0, top1 = S0[sl][..., 0], S1[sl][..., 0]
        exact = top0 * h0[sl][..., 0] / (h0[sl][..., 0] + F_dt)
        first_order = top0 * (1.0 - F_dt / h0[sl][..., 0])
        assert np.all(top1 < top0), "the top cell did not freshen"
        # exact to second order in F dt/h (the stretch is applied by the
        # z-star step, the transport by the helper; both first-order)
        # rtol=0: numpy's default 1e-5 relative tolerance is 3.5e-4 PSU on S=35,
        # 200x the signal -- with it this test passed on the pre-fix code
        assert np.allclose(top1, exact, rtol=0.0, atol=2.0 * abs(exact - first_order).max() + 1e-12)
        assert np.allclose(S1[sl][..., 1:], S0[sl][..., 1:], rtol=0.0, atol=1e-10), \
            "layers below the top cell must keep their salinity"
        # rain enters at 0 degC in the tracer equation (its heat content is
        # carried by the surface heat flux, NEMO blk_oce_2 -- not applied in
        # this bare step), so the top cell cools by T F dt/h1 and nothing else
        # moves
        exact_T = T0[sl][..., 0] * h0[sl][..., 0] / (h0[sl][..., 0] + F_dt)
        assert np.allclose(T1[sl][..., 0], exact_T, rtol=0.0, atol=2.0 * abs(exact - first_order).max() + 1e-12)
        assert np.allclose(T1[sl][..., 1:], T0[sl][..., 1:], rtol=0.0, atol=1e-10)
        # column salt is conserved: sum h S before == after
        m0 = np.sum(h0[sl] * S0[sl], axis=-1); m1 = np.sum(h1[sl] * S1[sl], axis=-1)
        assert np.allclose(m1, m0, rtol=1e-12)

    def test_the_pre_fix_behaviour_is_what_the_test_rejects(self):
        """Non-vacuity: the uniform stretch alone (what the closure did before)
        changes the top cell by S F dt/(rho H), ~1/100 of the exact
        dilution here (h1 = 100 m of 300 m); the assertion above cannot pass
        on it."""
        S0, S1, T0, T1, h0, h1, wet, rho0 = self._run("real_freshwater")
        F_dt = 3.0e-5 * 300.0 / rho0
        sl = (slice(1, -1), slice(1, -1))
        H = h0[sl].sum(-1); h_top = h0[sl][..., 0]
        stretch_only = S0[sl][..., 0] * (1.0 - F_dt / H)
        exact = S0[sl][..., 0] * h_top / (h_top + F_dt)
        assert np.all(np.abs(S1[sl][..., 0] - exact) < 0.1 * np.abs(stretch_only - exact))


@pytest.mark.parametrize("F", [4.0e-8, -4.0e-8])
def test_stratified_partial_column_both_signs_match_the_upwind_algebra(F):
    """GLM review 2026-09-05: uniform profiles hide an off-by-one in
    H_below (only the column total leaks) and the upwind choice (only a
    stratified column tells cell-above from cell-below).  Non-uniform h with
    a partial bottom cell, stratified S and T, both flux signs."""
    h = np.array([[1.0, 3.0, 9.0, 27.0, 13.5, 0.0]])          # partial 5th cell, dry 6th
    S = np.array([[33.0, 34.0, 34.6, 34.9, 35.0, 35.0]])
    T = np.array([[25.0, 20.0, 12.0, 6.0, 3.0, 3.0]])
    mask = np.ones(1)
    dS, dT = (np.asarray(v) for v in real_freshwater_dilution_tendencies(np.array([F]), S, T, h, mask))
    wet = h[0] > 0
    H = h[0, wet].sum()
    Hb = np.array([h[0, wet][k + 1:].sum() for k in range(wet.sum())])
    W = F * Hb / H                                              # interface under cell k
    Wa = np.concatenate([[F], W[:-1]])                          # interface above cell k
    Sw, Tw = S[0, wet], T[0, wet]
    up = lambda C, k: C[k] if W[k] >= 0 else C[min(k + 1, wet.sum() - 1)]
    ref_S = np.array([((0.0 if k == 0 else W[k - 1] * up(Sw, k - 1)) - W[k] * up(Sw, k)) / h[0, k]
                      for k in range(wet.sum())])
    ref_T = np.array([((F * Tw[0] if k == 0 else W[k - 1] * up(Tw, k - 1)) - W[k] * up(Tw, k)) / h[0, k]
                      for k in range(wet.sum())])
    assert np.allclose(dS[0, wet], ref_S, rtol=1e-12, atol=1e-18)
    assert np.allclose(dT[0, wet], ref_T, rtol=1e-12, atol=1e-18)
    assert dS[0, ~wet].tolist() == [0.0] and dT[0, ~wet].tolist() == [0.0]
    # (a) column salt conserved to round-off; (b) heat change = F T_1
    assert abs(np.sum(h[0] * dS[0])) < 1e-15 * np.abs(F) * 35.0 * H
    assert np.sum(h[0] * dT[0]) == pytest.approx(F * T[0, 0], rel=1e-12)
    # (c) top cell with the z-star stretch (-S_1 F/H) = NEMO's -S_1 F/h_1 for
    #     water entering; for water leaving the upwind value is S_2, so the
    #     top cell salinifies by |F| (S_2 (H-h1)/H + S_1 h1/H)/h1 (mixing in
    #     the water that replaces it), not by the local S_1 alone
    top = dS[0, 0] - S[0, 0] * F / H
    if F > 0:
        assert top == pytest.approx(-S[0, 0] * F / h[0, 0], rel=1e-12)
    else:
        assert top == pytest.approx(-F * (S[0, 1] * (H - h[0, 0]) / H + S[0, 0] * h[0, 0] / H) / h[0, 0], rel=1e-12)
        assert top > 0.0


def test_runoff_entry_profile_dilutes_each_level_by_its_own_share():
    """codex review 2026-09-05: runoff must enter over NEMO's h_rnf, not at the
    surface.  A river of rate R spread over 8 m into levels of 1, 3, 9 m
    (fractions 1/8, 3/8, 4/8 of the third level -> 4 m of it): with the
    z-star stretch each level's dilution is -S R frac_k / h_k, the layers
    below h_rnf are untouched, column salt conserved, heat = sum entry T."""
    from legoesm.ocean.freshwater import FreshwaterForcing, runoff_entry_profile
    # uniform S so "each level diluted by its own share" is the whole answer
    # (a stratified S adds the resident-water transport between levels)
    h = np.array([[1.0, 3.0, 9.0, 27.0]]); S = np.full((1, 4), 34.0)
    T = np.array([[20.0, 15.0, 10.0, 5.0]]); mask = np.ones(1)
    rho0 = 1025.0; R_kg = 2.0e-4
    fw = FreshwaterForcing(precip=np.zeros(1), evap=np.zeros(1), runoff=np.array([R_kg]), ice_fw=np.zeros(1))
    entry = np.asarray(runoff_entry_profile(fw, h, mask, rho0, 8.0))
    R = R_kg / rho0
    assert np.allclose(entry, R * np.array([[1, 3, 4, 0]]) / 8.0, rtol=1e-12)
    dS, dT = (np.asarray(v) for v in real_freshwater_dilution_tendencies(
        np.array([R]), S, T, h, mask, F_entry=entry))
    H = h.sum()
    total = dS[0] - S[0] * R / H                       # + z-star stretch
    assert np.allclose(total, -S[0] * entry[0] / h[0], rtol=1e-12, atol=1e-18)
    assert total[3] == pytest.approx(0.0, abs=1e-18)
    assert abs(np.sum(h * dS)) < 1e-15 * R * 35.0 * H
    assert np.sum(h * dT) == pytest.approx(np.sum(entry * T), rel=1e-12)
    # T: water at the local temperature (default) -> only the resident-water
    # transport between stratified layers remains; column heat = sum entry T
    assert np.sum(h * dT) == pytest.approx(np.sum(entry * T), rel=1e-12)


def test_entry_builder_channel_heat_conventions():
    """Rain/evaporation/restoring water carries zero tracer temperature (heat
    already in the surface heat flux); runoff and ice melt water enter at the
    local temperature; the normalisation residual too."""
    from legoesm.ocean.freshwater import FreshwaterForcing, real_freshwater_entry
    h = np.array([[1.0, 3.0, 9.0]]); T = np.array([[20.0, 15.0, 10.0]]); mask = np.ones(1)
    rho0 = 1025.0
    def fw(**kw):
        d = dict(precip=0.0, evap=0.0, runoff=0.0, ice_fw=0.0)
        d.update(kw)
        return FreshwaterForcing(**{k: np.array([v]) for k, v in d.items()})
    # rain only: enters at the top, zero heat
    F = 2e-4 / rho0
    Fe, He = (np.asarray(v) for v in real_freshwater_entry(fw(precip=2e-4), F, h, mask, rho0, T))
    assert np.allclose(Fe, [[F, 0, 0]]) and np.allclose(He, 0.0)
    # evaporation only: negative entry at the top, zero heat
    Fe, He = (np.asarray(v) for v in real_freshwater_entry(fw(evap=2e-4), -F, h, mask, rho0, T))
    assert np.allclose(Fe, [[-F, 0, 0]]) and np.allclose(He, 0.0)
    # ice melt: enters at the top at T_1
    Fe, He = (np.asarray(v) for v in real_freshwater_entry(fw(ice_fw=2e-4), F, h, mask, rho0, T))
    assert np.allclose(He, [[F * 20.0, 0, 0]])
    # runoff spread over 4 m (1 m + 3 m): enters both levels at SST (NEMO rnf_tsc)
    Fe, He = (np.asarray(v) for v in real_freshwater_entry(fw(runoff=2e-4), F, h, mask, rho0, T, runoff_spread_m=4.0))
    assert np.allclose(Fe, [[F / 4, 3 * F / 4, 0]]) and np.allclose(He, [[F / 4 * 20.0, 3 * F / 4 * 20.0, 0]])
    # normalisation residual (F_rate differs from the channel sum) enters at T_1
    Fe, He = (np.asarray(v) for v in real_freshwater_entry(fw(precip=2e-4), 0.5 * F, h, mask, rho0, T))
    assert np.allclose(Fe, [[0.5 * F, 0, 0]]) and np.allclose(He, [[-0.5 * F * 20.0, 0, 0]])


def test_subsurface_runoff_under_net_evaporation_changes_sign_inside_the_column():
    """GLM round 2: with runoff entering at depth under a surface that
    evaporates, the transport is upward above the river level and downward
    below it; the per-interface upwind donor must still conserve column
    salt exactly and the heat budget must close on the entry heat."""
    from legoesm.ocean.freshwater import FreshwaterForcing, real_freshwater_entry
    h = np.array([[1.0, 3.0, 9.0, 27.0]]); S = np.array([[36.0, 35.5, 35.0, 34.8]])
    T = np.array([[26.0, 22.0, 15.0, 8.0]]); mask = np.ones(1); rho0 = 1025.0
    # evaporation at the surface, a larger river spread over the top 4 m:
    # net water IN, so the transport is upward above the river level and
    # downward below it (with net evaporation it is upward everywhere)
    fw = FreshwaterForcing(precip=np.zeros(1), evap=np.array([1.0e-4]),
                           runoff=np.array([3.0e-4]), ice_fw=np.zeros(1))
    F = (-1.0e-4 + 3.0e-4) / rho0
    Fe, He = (np.asarray(v) for v in real_freshwater_entry(fw, F, h, mask, rho0, T, runoff_spread_m=4.0))
    assert Fe.sum() == pytest.approx(F, rel=1e-12)
    dS, dT = (np.asarray(v) for v in real_freshwater_dilution_tendencies(
        np.array([F]), S, T, h, mask, F_entry=Fe, entry_heat=He))
    H = h.sum()
    W = np.cumsum(Fe[0]) - F * np.cumsum(h[0]) / H          # interface under cell k
    assert W[0] < 0.0 and W[1] > 0.0 and abs(W[-1]) < 1e-20, W
    assert abs(np.sum(h * dS)) < 1e-15 * 36.0 * abs(F) * H
    assert np.sum(h * dT) == pytest.approx(He.sum(), rel=1e-12)
    # no new extrema from the transport alone: every layer stays within the
    # column's [min, max] after one large step
    dt = 1800.0
    S1 = S[0] + dt * dS[0]
    assert S1.min() >= S[0].min() - 1e-12 and S1.max() <= S[0].max() + 1e-12 + 36.0 * 1e-4 / rho0 * dt
