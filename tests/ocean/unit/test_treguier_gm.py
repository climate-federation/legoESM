"""Direct unit tests for the Treguier-1997 adaptive GM coefficient
(NEMO 5.0.1 ldftra.F90::ldf_eiv, nn_aei_ijk_t=21 — the DINO/ORCA1 oracle
scaling): κ = min( min(1,|f/f20|)·Ro²·T⁻¹, aei0 ).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
    _TREGUIER_RO_FACTOR,
    _TREGUIER_RO_MAX_M,
    _TREGUIER_RO_MIN_M,
    _TREGUIER_ZHW_OFFSET_M,
    compute_treguier_kappa_gm,
)
from legoesm.ocean.physics.lateral_mixing.config import (
    GMRediConfig,
    TreguierConfig,
    VisbeckConfig,
)
from legoesm.ocean.vertical import create_ocean_z_star

jax.config.update("jax_enable_x64", True)


def _setup(nlev=10, H=1000.0, n=3, slope=1e-4, jac=1.0):
    z = create_ocean_z_star(n_levels=nlev, H_max=H)
    shape = (n, n, nlev)
    rho = jnp.broadcast_to(
        jnp.linspace(constants.rho_ocean, constants.rho_ocean + 2.0, nlev),
        shape)
    S_x = jnp.full(shape[:-1] + (nlev - 1,), slope)
    S_y = jnp.zeros_like(S_x)
    jacobian = jnp.full(shape[:-1], jac)
    return rho, S_x, S_y, z, jacobian


def _expected_kappa(rho, S_x, S_y, z, jacobian, f, aei0):
    """Independently reassemble the NEMO ldf_eiv formula using the model's
    own shared N² (the exact quantity the implementation consumes)."""
    from legoesm.ocean.eos import compute_buoyancy_frequency
    dz_actual = np.asarray(z.dz_ref) * np.asarray(jacobian)[..., None]
    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
    N2 = np.asarray(compute_buoyancy_frequency(
        rho, z.dz_ref, jacobian, rho_ref=constants.rho_ocean, g=constants.g))
    N = np.sqrt(np.maximum(N2, 1e-30))
    S = np.sqrt(np.asarray(S_x) ** 2 + np.asarray(S_y) ** 2 + 1e-30)
    int_N_dz = np.sum(N * dz_half, axis=-1)
    ro = np.clip(_TREGUIER_RO_FACTOR * int_N_dz / np.maximum(np.abs(f), 1e-10),
                 _TREGUIER_RO_MIN_M, _TREGUIER_RO_MAX_M)
    zah = np.sum((N * S) ** 2 * dz_half, axis=-1)
    zhw = _TREGUIER_ZHW_OFFSET_M + np.sum(dz_half, axis=-1)
    t_inv = np.sqrt(zah / zhw)
    f20 = 2.0 * constants.Omega * np.sin(np.deg2rad(20.0))
    taper = np.minimum(1.0, np.abs(f) / f20)
    return np.minimum(taper * ro ** 2 * t_inv, aei0)


class TestTreguierKappa:
    def test_matches_independent_formula(self):
        rho, S_x, S_y, z, jac = _setup()
        f = jnp.full((3, 3), 1.0e-4)                 # midlatitude
        cfg = TreguierConfig(enabled=True, aei0=1.0e9)   # cap inert
        got = np.asarray(compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, f, cfg))
        want = _expected_kappa(rho, S_x, S_y, z, jac, np.asarray(f), 1.0e9)
        np.testing.assert_allclose(got, want, rtol=1e-6)
        assert (got > 0.0).all()

    def test_rossby_radius_clamps(self):
        rho, S_x, S_y, z, jac = _setup()
        cfg = TreguierConfig(enabled=True, aei0=1.0e12)
        # Tiny |f| -> Ro hits the 40 km cap; huge |f| -> the 2 km floor.
        k_lo_f = np.asarray(compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, jnp.full((3, 3), 1.0e-9), cfg))
        k_hi_f = np.asarray(compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, jnp.full((3, 3), 1.0), cfg))
        want_lo = _expected_kappa(rho, S_x, S_y, z, jac,
                                  np.full((3, 3), 1.0e-9), 1.0e12)
        want_hi = _expected_kappa(rho, S_x, S_y, z, jac,
                                  np.full((3, 3), 1.0), 1.0e12)
        np.testing.assert_allclose(k_lo_f, want_lo, rtol=1e-6)
        np.testing.assert_allclose(k_hi_f, want_hi, rtol=1e-6)
        # The clamps genuinely BIND in these regimes (non-vacuous): the raw
        # (unclamped) Ro = 0.4·∫N dz/|f| straddles the [2 km, 40 km] bounds.
        # (No directional κ assert: the tropical taper ∝|f| dominates, so the
        # tiny-f κ is SMALLER despite its 40 km radius.)
        from legoesm.ocean.eos import compute_buoyancy_frequency
        dz_actual = np.asarray(z.dz_ref) * np.asarray(jac)[..., None]
        dzh = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
        N = np.sqrt(np.maximum(np.asarray(compute_buoyancy_frequency(
            rho, z.dz_ref, jac, rho_ref=constants.rho_ocean,
            g=constants.g)), 1e-30))
        int_N = np.sum(N * dzh, axis=-1)
        assert (_TREGUIER_RO_FACTOR * int_N / 1.0e-9 > _TREGUIER_RO_MAX_M).all()
        assert (_TREGUIER_RO_FACTOR * int_N / 1.0 < _TREGUIER_RO_MIN_M).all()

    def test_tropical_taper(self):
        """At |f| = ½f₂₀ the taper halves κ relative to the untapered value
        at the SAME f (cap inert, Ro un-clamped regime)."""
        rho, S_x, S_y, z, jac = _setup(slope=1e-5)
        f20 = 2.0 * constants.Omega * np.sin(np.deg2rad(20.0))
        f_half = jnp.full((3, 3), 0.5 * f20)
        cfg = TreguierConfig(enabled=True, aei0=1.0e12)
        got = np.asarray(compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, f_half, cfg))
        want = _expected_kappa(rho, S_x, S_y, z, jac,
                               np.asarray(f_half), 1.0e12)
        np.testing.assert_allclose(got, want, rtol=1e-6)
        # taper factor is exactly 0.5 in the expected formula — assert the
        # implementation reproduces it (ratio vs the taper-free value).
        untapered = want / 0.5
        np.testing.assert_allclose(got * 2.0, untapered, rtol=1e-6)

    def test_aei0_cap_binds(self):
        rho, S_x, S_y, z, jac = _setup()
        f = jnp.full((3, 3), 1.0e-4)
        k_capped = np.asarray(compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, f, TreguierConfig(enabled=True, aei0=100.0)))
        k_free = np.asarray(compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, f, TreguierConfig(enabled=True, aei0=1e12)))
        assert (k_free > 100.0).all()           # cap is genuinely binding
        np.testing.assert_allclose(k_capped, 100.0, rtol=1e-12)

    def test_dry_column_zero(self):
        rho, S_x, S_y, z, _ = _setup()
        jac = jnp.zeros((3, 3))                  # all-dry
        k = np.asarray(compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, jnp.full((3, 3), 1e-4),
            TreguierConfig(enabled=True)))
        assert np.allclose(k, 0.0)

    def test_grad_finite(self):
        rho, S_x, S_y, z, jac = _setup()
        f = jnp.full((3, 3), 1e-4)
        cfg = TreguierConfig(enabled=True)

        def total(r):
            return jnp.sum(compute_treguier_kappa_gm(
                r, S_x, S_y, z, jac, f, cfg))

        g = jax.grad(total)(rho)
        assert bool(jnp.isfinite(g).all())


class TestTreguierKappaNemoNative:
    """#1317: compute_treguier_kappa_gm_nemo_native must consume the SAME
    wslpi/wslpj (compute_nemo_native_slopes) NEMO's own ldf_eiv sums over
    (ldftra.F90:664-706) — not the simplified cell-centred S_x/S_y the
    generic compute_treguier_kappa_gm uses. Verified (day-0 DINO twin) the
    two formulations disagree materially (corr=0.28, mean 646 vs 196 m^2/s)
    and the fix raises the ADVECTION-bucket (bolus-inclusive) tracer-tendency
    corr vs the NEMO oracle from 0.9347 to 0.9889 (full3D)."""

    def _dino_fixture(self, n_lon=50):
        from legoesm.ocean.experiments.dino import (
            DINOConfig, create_dino_z_star, dino_lat_lon_grid,
            dino_lat_lon_state,
        )
        dcfg = DINOConfig()
        z = create_dino_z_star(dcfg)
        g = dino_lat_lon_grid(dcfg, n_lon=n_lon)
        st = dino_lat_lon_state(g, z, dcfg)
        return dcfg, z, g, st

    def _independent_kappa(self, rho, T, S, mask, wslpi, wslpj, z, g_grid,
                            f, cfg, rho_0, g, eos_fn, act):
        """Reassemble ldftra.F90:664-706 straight from the definition
        (full jk=1..jpk column sum incl. the surface w-level e3w(1)),
        independent of the shared helper the implementation calls. Uses
        the SAME eos_fn AND the SAME topography-aware 3-D active mask
        (``act`` — DINO has a variable bathymetry, so a column's active
        depth is shallower than nlev below the shelf/ridge) as the
        implementation for the adiabatic N^2 (that part is validated on
        its own by test_mixed_layer_depth.py / test_isoneutral_slope_
        density.py); this test's job is the ldf_eiv REDUCTION (full-column
        sum, zhw offset, Ro clamp, tropical taper, aei0 cap), not
        re-deriving N^2/the wet-column mask from scratch.
        """
        dz = np.asarray(z.dz_ref)
        gdept = np.cumsum(dz) - 0.5 * dz
        e3w = np.concatenate([dz[:1], gdept[1:] - gdept[:-1]])
        nlat, nlon, nlev = np.asarray(rho).shape
        e3w_3d = np.broadcast_to(e3w, (nlat, nlon, nlev))
        m = np.asarray(mask)
        act = np.asarray(act)
        wmask3 = act * np.roll(act, 1, axis=2)
        wmask3[:, :, 0] = act[:, :, 0]

        from legoesm.ocean.eos import compute_buoyancy_frequency_adiabatic
        p_cell = (rho_0 * g * gdept)[None, None, :] * np.ones_like(np.asarray(rho))
        J1 = np.ones((nlat, nlon))
        n2_int = np.asarray(compute_buoyancy_frequency_adiabatic(
            T, S, jnp.asarray(p_cell), z.dz_ref, jnp.asarray(J1), eos_fn=eos_fn))
        pn2 = np.concatenate([np.zeros((nlat, nlon, 1)), n2_int], axis=-1) * wmask3

        zn2 = np.maximum(pn2, 0.0)
        zn = np.sum(np.sqrt(zn2) * e3w_3d, axis=-1)
        ze3w = e3w_3d * wmask3
        wi = np.asarray(wslpi)
        wj = np.asarray(wslpj)
        zah = np.sum(zn2 * (wi ** 2 + wj ** 2) * ze3w, axis=-1)
        zhw = _TREGUIER_ZHW_OFFSET_M + np.sum(ze3w, axis=-1)
        f_abs = np.maximum(np.abs(np.asarray(f)), 1e-10)
        ro = np.clip(_TREGUIER_RO_FACTOR * zn / f_abs,
                     _TREGUIER_RO_MIN_M, _TREGUIER_RO_MAX_M)
        t_inv = np.sqrt(zah / np.maximum(zhw, 1e-10))
        f20 = 2.0 * constants.Omega * np.sin(np.deg2rad(20.0))
        taper = np.minimum(1.0, np.abs(np.asarray(f)) / f20)
        return np.where(m > 0.5, np.minimum(taper * ro ** 2 * t_inv, cfg.aei0), 0.0)

    def test_matches_ldf_eiv_full_column_formula(self):
        from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
            compute_nemo_native_slopes, compute_treguier_kappa_gm_nemo_native,
            gm_redi_density_and_jacobian,
        )
        from legoesm.ocean.eos import make_eos_fn
        dcfg, z, g, st = self._dino_fixture()
        gm_cfg = GMRediConfig(slope_scheme="nemo_iso_lap", slope_positions="nemo_native")
        eos_fn = make_eos_fn("nemo_seos")
        mask = st.land_mask.data
        rho, jacobian = gm_redi_density_and_jacobian(
            st.T.data, st.S.data, st.eta.data, st.H_bathy.data, g, z,
            eos="nemo_seos", mask=mask)
        f = jnp.broadcast_to(g.f, mask.shape)
        _z_top = jnp.cumsum(z.dz_ref) - z.dz_ref
        act = ((mask[:, :, None] > 0.5)
               & (_z_top[None, None, :] < st.H_bathy.data[:, :, None])).astype(st.T.data.dtype)
        uslp, vslp, wslpi, wslpj = compute_nemo_native_slopes(
            rho, st.T.data, st.S.data, mask, st.u_mask.data, st.v_mask.data,
            z, g, gm_cfg, eos_fn, active_3d=act)
        cfg = TreguierConfig(enabled=True, aei0=1500.0)
        got = np.asarray(compute_treguier_kappa_gm_nemo_native(
            rho, st.T.data, st.S.data, wslpi, wslpj, mask, z, g, f, cfg,
            eos_fn, active_3d=act))
        want = self._independent_kappa(
            rho, st.T.data, st.S.data, mask, wslpi, wslpj, z, g, f, cfg,
            rho_0=constants.rho_ocean, g=constants.g, eos_fn=eos_fn, act=act)
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-8)
        assert (got[np.asarray(mask) > 0.5] >= 0.0).all()

    def test_dry_column_zero(self):
        from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
            compute_nemo_native_slopes, compute_treguier_kappa_gm_nemo_native,
            gm_redi_density_and_jacobian,
        )
        from legoesm.ocean.eos import make_eos_fn
        dcfg, z, g, st = self._dino_fixture()
        gm_cfg = GMRediConfig(slope_scheme="nemo_iso_lap", slope_positions="nemo_native")
        eos_fn = make_eos_fn("nemo_seos")
        mask = jnp.zeros_like(st.land_mask.data)  # all-dry
        rho, jacobian = gm_redi_density_and_jacobian(
            st.T.data, st.S.data, st.eta.data, st.H_bathy.data, g, z,
            eos="nemo_seos", mask=mask)
        f = jnp.broadcast_to(g.f, mask.shape)
        u_mask = jnp.zeros_like(st.u_mask.data)
        v_mask = jnp.zeros_like(st.v_mask.data)
        act = jnp.zeros_like(rho)
        uslp, vslp, wslpi, wslpj = compute_nemo_native_slopes(
            rho, st.T.data, st.S.data, mask, u_mask, v_mask, z, g, gm_cfg,
            eos_fn, active_3d=act)
        cfg = TreguierConfig(enabled=True, aei0=1500.0)
        got = np.asarray(compute_treguier_kappa_gm_nemo_native(
            rho, st.T.data, st.S.data, wslpi, wslpj, mask, z, g, f, cfg,
            eos_fn, active_3d=act))
        assert np.allclose(got, 0.0)

    def test_grad_finite(self):
        from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
            compute_nemo_native_slopes, compute_treguier_kappa_gm_nemo_native,
            gm_redi_density_and_jacobian,
        )
        from legoesm.ocean.eos import make_eos_fn
        dcfg, z, g, st = self._dino_fixture()
        gm_cfg = GMRediConfig(slope_scheme="nemo_iso_lap", slope_positions="nemo_native")
        eos_fn = make_eos_fn("nemo_seos")
        mask = st.land_mask.data
        f = jnp.broadcast_to(g.f, mask.shape)
        _z_top = jnp.cumsum(z.dz_ref) - z.dz_ref
        act = ((mask[:, :, None] > 0.5)
               & (_z_top[None, None, :] < st.H_bathy.data[:, :, None])).astype(st.T.data.dtype)
        cfg = TreguierConfig(enabled=True, aei0=1500.0)

        def total(T):
            rho, _ = gm_redi_density_and_jacobian(
                T, st.S.data, st.eta.data, st.H_bathy.data, g, z,
                eos="nemo_seos", mask=mask)
            uslp, vslp, wslpi, wslpj = compute_nemo_native_slopes(
                rho, T, st.S.data, mask, st.u_mask.data, st.v_mask.data,
                z, g, gm_cfg, eos_fn, active_3d=act)
            return jnp.sum(compute_treguier_kappa_gm_nemo_native(
                rho, T, st.S.data, wslpi, wslpj, mask, z, g, f, cfg,
                eos_fn, active_3d=act))

        grad = jax.grad(total)(st.T.data)
        # Finite on cells with a fully-WET 8-neighbourhood, EXCLUDING the
        # channel's non-periodic meridional (row) boundary: the grid is
        # periodic in longitude only, so row 0 / row -1 have no real
        # north/south neighbour: pad those with "dry" (False) rather than
        # wrapping, matching the domain's actual (non-periodic-in-lat)
        # topology (a plain np.roll on axis 0 would incorrectly treat row
        # 0's neighbour as row -1, an unrelated part of the channel).
        m = np.asarray(mask)
        wet_interior = np.ones_like(m, dtype=bool)
        for di in (-1, 0, 1):
            row_shifted = np.roll(m, di, axis=0) > 0.5
            if di == -1:
                row_shifted[-1, :] = False
            elif di == 1:
                row_shifted[0, :] = False
            for dj in (-1, 0, 1):
                wet_interior &= np.roll(row_shifted, dj, axis=1)
        wet3d = jnp.broadcast_to(jnp.asarray(wet_interior)[:, :, None], grad.shape)
        assert bool(wet_interior.any())  # non-vacuous
        assert bool(jnp.isfinite(jnp.where(wet3d, grad, 0.0)).all())

    def test_dispatch_prefers_nemo_native_over_generic_treguier(self):
        """gm_redi_tracer_tendency_latlon must route through the
        nemo_native-consistent kappa_GM (not the generic simplified-slope
        path) when slope_scheme='nemo_iso_lap' + slope_positions='nemo_native'
        + treguier.enabled — the #1317 wiring fix."""
        from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
            gm_redi_tracer_tendency_latlon,
        )
        dcfg, z, g, st = self._dino_fixture()
        mask = st.land_mask.data
        common = dict(
            kappa_Redi=100.0, slope_scheme="nemo_iso_lap",
            slope_density="neutral", treguier=TreguierConfig(enabled=True, aei0=1500.0),
        )
        cfg_native = GMRediConfig(slope_positions="nemo_native", **common)
        cfg_mode_b = GMRediConfig(slope_positions="mode_b", **common)
        kwargs = dict(
            eos="nemo_seos", mask=mask, u_mask=st.u_mask.data, v_mask=st.v_mask.data,
        )
        dT_native, dS_native = gm_redi_tracer_tendency_latlon(
            st.T.data, st.S.data, st.eta.data, st.H_bathy.data, g, z,
            cfg_native, **kwargs)
        dT_modeb, dS_modeb = gm_redi_tracer_tendency_latlon(
            st.T.data, st.S.data, st.eta.data, st.H_bathy.data, g, z,
            cfg_mode_b, **kwargs)
        # Different slope_positions -> different operator AND (post-fix)
        # different kappa_GM -> tendencies must differ (non-vacuous: the
        # pre-fix code would still differ here via the operator alone, but
        # a regression that silently drops the native-kappa branch would
        # only be caught by the corr-vs-NEMO oracle check, which is exactly
        # what this dispatch test is a cheap proxy for).
        wet3d = jnp.broadcast_to(mask[:, :, None] > 0.5, dT_native.shape)
        assert not np.allclose(
            np.asarray(jnp.where(wet3d, dT_native, 0.0)),
            np.asarray(jnp.where(wet3d, dT_modeb, 0.0)))
        assert bool(jnp.isfinite(jnp.where(wet3d, dT_native, 0.0)).all())


class TestDispatchAndWiring:
    def test_gm_redi_config_carries_treguier_default_off(self):
        cfg = GMRediConfig()
        assert cfg.treguier.enabled is False
        assert cfg.treguier.aei0 == 3000.0       # DINO rn_Ue*rn_Le

    def test_mutual_exclusion_raises_on_latlon_path(self):
        from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
            gm_redi_tracer_tendency_latlon,
        )
        from legoesm.ocean.experiments.dino import (
            DINOConfig, create_dino_z_star, dino_lat_lon_grid,
            dino_lat_lon_state,
        )
        dcfg = DINOConfig()
        z = create_dino_z_star(dcfg)
        g = dino_lat_lon_grid(dcfg, n_lon=50)
        st = dino_lat_lon_state(g, z, dcfg)
        bad = GMRediConfig(
            visbeck=VisbeckConfig(enabled=True),
            treguier=TreguierConfig(enabled=True))
        with pytest.raises(ValueError, match="mutually exclusive"):
            gm_redi_tracer_tendency_latlon(
                st.T.data, st.S.data, st.eta.data, st.H_bathy.data,
                g, z, bad, mask=st.land_mask.data,
                u_mask=st.u_mask.data, v_mask=st.v_mask.data)

    def test_dino_gm_kappa_scheme_wiring(self):
        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_grid, dino_lat_lon_model_config,
        )
        base = DINOConfig()
        assert base.gm_kappa_scheme == "visbeck"     # historical default
        g = dino_lat_lon_grid(base, n_lon=50)
        mc, _ = dino_lat_lon_model_config(g, base, physics=True)
        assert mc.gm_redi.visbeck.enabled is True
        assert mc.gm_redi.treguier.enabled is False
        treg = dataclasses.replace(base, gm_kappa_scheme="treguier")
        mc2, _ = dino_lat_lon_model_config(g, treg, physics=True)
        assert mc2.gm_redi.visbeck.enabled is False
        assert mc2.gm_redi.treguier.enabled is True
        # DINOConfig.treguier_aei0 default = 0.5*rn_Ue*rn_Le (ldftra.F90:332
        # explicit 1/2 factor) = 1500, not the un-halved rn_Ue*rn_Le = 3000.
        assert mc2.gm_redi.treguier.aei0 == pytest.approx(1500.0)

    def test_dino_unknown_scheme_raises(self):
        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_grid, dino_lat_lon_model_config,
        )
        base = dataclasses.replace(DINOConfig(), gm_kappa_scheme="bogus")
        g = dino_lat_lon_grid(base, n_lon=50)
        with pytest.raises(ValueError, match="gm_kappa_scheme"):
            dino_lat_lon_model_config(g, base, physics=True)
