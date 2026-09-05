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
        assert np.allclose(top1, exact, atol=2.0 * abs(exact - first_order).max() + 1e-12)
        assert np.allclose(S1[sl][..., 1:], S0[sl][..., 1:], atol=1e-10), \
            "layers below the top cell must keep their salinity"
        assert np.allclose(T1[sl], T0[sl], atol=1e-10), "temperature must not change"
        # column salt is conserved: sum h S before == after
        m0 = np.sum(h0[sl] * S0[sl], axis=-1); m1 = np.sum(h1[sl] * S1[sl], axis=-1)
        assert np.allclose(m1, m0, rtol=1e-10)

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
