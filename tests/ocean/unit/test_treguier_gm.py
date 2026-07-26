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
