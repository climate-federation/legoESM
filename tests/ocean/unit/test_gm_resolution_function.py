"""Direct unit tests for the Hallberg (2013) resolution function that tapers
the GM coefficient by resolution (MED-3).

    f_res = 1 / (1 + (L_d / (gamma * dx))**2),   L_d = c_bcl / |f|

Covers the two shared pure helpers in ``_gm_redi_common`` and the wiring into
the lat-lon C-grid GM/Redi tendency (default off => byte-identical).

Run with:

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/ocean/unit/test_gm_resolution_function.py -v
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
    _RESFN_F_FLOOR_S,
    gm_resolution_function,
    gm_resolution_scaled_kappa,
)
from legoesm.ocean.physics.lateral_mixing.config import (
    GMRediConfig,
    VisbeckConfig,
    __param_spec__ as _LATMIX_PARAM_SPEC,
)

jax.config.update("jax_enable_x64", True)


def _f_res_ref(L_d, dx, gamma):
    """Independent numpy reference for the resolution function."""
    L_d = np.asarray(L_d, dtype=np.float64)
    dx = np.asarray(dx, dtype=np.float64)
    ratio = L_d / (gamma * dx)
    return 1.0 / (1.0 + ratio * ratio)


class TestResolutionFunctionFormula:
    def test_coarse_limit_full_gm(self):
        """dx >> L_d (coarse, eddies unresolved) -> f_res -> 1 (full GM)."""
        f_res = float(gm_resolution_function(1.0e4, 1.0e7, 2.0))
        assert f_res > 0.999
        assert f_res <= 1.0

    def test_fine_limit_gm_off(self):
        """dx << L_d (eddy-resolving) -> f_res -> ~0 (GM off)."""
        f_res = float(gm_resolution_function(1.0e6, 1.0e3, 2.0))
        assert 0.0 < f_res < 1.0e-4          # strictly positive, ~0

    def test_one_deg_vs_twelfth_deg_same_Ld(self):
        """Spec item 4: for the SAME L_d, f_res -> 1 at 1deg (dx >> L_d) and
        -> ~0 at 1/12deg (dx << L_d)."""
        L_d = 60.0e3            # 60 km deformation radius (subtropical scale)
        gamma = 2.0
        dx_1deg = 100.0e3       # ~1deg cell
        dx_1_12 = dx_1deg / 12.0
        f_coarse = float(gm_resolution_function(L_d, dx_1deg, gamma))
        f_fine = float(gm_resolution_function(L_d, dx_1_12, gamma))
        assert f_coarse > 0.9          # -> 1 on the coarse grid
        assert f_fine < 0.12           # -> ~0 on the eddy-resolving grid
        assert f_coarse > f_fine       # GM suppressed as the grid refines
        np.testing.assert_allclose(f_coarse, _f_res_ref(L_d, dx_1deg, gamma),
                                   rtol=1e-12)
        np.testing.assert_allclose(f_fine, _f_res_ref(L_d, dx_1_12, gamma),
                                   rtol=1e-12)

    def test_half_point_exact(self):
        """gamma*dx == L_d -> ratio == 1 -> f_res == 0.5 exactly."""
        L_d, dx, gamma = 2.0e4, 1.0e4, 2.0     # gamma*dx = 2e4 = L_d
        np.testing.assert_allclose(
            float(gm_resolution_function(L_d, dx, gamma)), 0.5, rtol=1e-12)

    def test_bounded_0_1_over_random_inputs(self):
        rng = np.random.default_rng(0)
        L_d = jnp.asarray(rng.uniform(0.0, 3.0e5, size=500))
        dx = jnp.asarray(rng.uniform(1.0e2, 2.0e5, size=500))
        gamma = 2.0
        f_res = np.asarray(gm_resolution_function(L_d, dx, gamma))
        assert np.all(np.isfinite(f_res))
        assert np.all(f_res > 0.0)
        assert np.all(f_res <= 1.0)

    def test_monotone_increasing_in_dx(self):
        """Coarser grid (larger dx) -> larger f_res (more GM)."""
        L_d, gamma = 5.0e4, 2.0
        dxs = jnp.asarray([5.0e3, 1.0e4, 3.0e4, 1.0e5, 3.0e5])
        f_res = np.asarray(gm_resolution_function(L_d, dxs, gamma))
        assert np.all(np.diff(f_res) > 0.0)

    def test_land_cell_zero_dx_no_nan(self):
        """Degenerate zero-area (land) cell: denom floored at EPS, f_res -> 0,
        finite (no division-by-zero NaN)."""
        f_res = float(gm_resolution_function(5.0e4, 0.0, 2.0))
        assert np.isfinite(f_res)
        assert 0.0 <= f_res < 1.0e-6

    def test_differentiable_finite_grad(self):
        def total(L_d):
            return jnp.sum(gm_resolution_function(
                L_d, jnp.full_like(L_d, 5.0e4), 2.0))
        g = jax.grad(total)(jnp.asarray([1.0e4, 5.0e4, 2.0e5]))
        assert bool(jnp.isfinite(g).all())


class TestResolutionScaledKappa:
    def test_scalar_kappa_becomes_scaled_field(self):
        kappa0 = 1000.0
        f = jnp.full((4,), 1.0e-4)
        dx = jnp.full((4,), 1.0e5)
        gamma, c = 2.0, 2.0
        out = np.asarray(gm_resolution_scaled_kappa(kappa0, f, dx, gamma, c))
        assert out.shape == (4,)
        L_d = c / np.maximum(np.abs(np.asarray(f)), _RESFN_F_FLOOR_S)
        want = kappa0 * _f_res_ref(L_d, dx, gamma)
        np.testing.assert_allclose(out, want, rtol=1e-12)
        assert np.all(out <= kappa0)      # GM only ever reduced, never amplified

    def test_field_kappa_elementwise(self):
        kappa = jnp.asarray([500.0, 1000.0, 2000.0])
        f = jnp.asarray([1.0e-4, 5.0e-5, 3.0e-4])
        dx = jnp.asarray([1.0e5, 8.0e4, 1.2e5])
        gamma, c = 2.0, 2.0
        out = np.asarray(gm_resolution_scaled_kappa(kappa, f, dx, gamma, c))
        L_d = c / np.maximum(np.abs(np.asarray(f)), _RESFN_F_FLOOR_S)
        want = np.asarray(kappa) * _f_res_ref(L_d, dx, gamma)
        np.testing.assert_allclose(out, want, rtol=1e-12)

    def test_equator_f_floor_no_nan(self):
        """|f| = 0 at the equator: floored so L_d is capped, f_res stays in
        (0, 1] with NO NaN (does not collapse to 0)."""
        f = jnp.asarray([0.0, 1.0e-4])
        dx = jnp.full((2,), 1.0e5)
        out = np.asarray(gm_resolution_scaled_kappa(1000.0, f, dx, 2.0, 2.0))
        assert np.all(np.isfinite(out))
        assert np.all(out > 0.0)
        assert np.all(out <= 1000.0)
        # Equator cell: L_d = c / f_floor = 2 / 1e-5 = 2e5 m; with dx=1e5,
        # gamma=2 -> ratio = 2e5/2e5 = 1 -> f_res = 0.5 (physical, not 0).
        np.testing.assert_allclose(out[0], 500.0, rtol=1e-9)

    def test_depth_resolved_kappa_broadcasts_over_levels(self):
        """3-D (prognostic-EKE) kappa (..., nlev-1): f_res broadcasts over the
        trailing level axis it lacks."""
        nlat, nlon, nlevm1 = 3, 4, 5
        kappa = jnp.ones((nlat, nlon, nlevm1)) * 1000.0
        f = jnp.full((nlat, nlon), 1.0e-4)
        dx = jnp.full((nlat, nlon), 1.0e5)
        out = np.asarray(gm_resolution_scaled_kappa(kappa, f, dx, 2.0, 2.0))
        assert out.shape == (nlat, nlon, nlevm1)
        L_d = 2.0 / np.maximum(1.0e-4, _RESFN_F_FLOOR_S)
        want_level = 1000.0 * _f_res_ref(L_d, 1.0e5, 2.0)
        np.testing.assert_allclose(out, want_level, rtol=1e-9)
        # every level identical (2-D f_res broadcast)
        np.testing.assert_allclose(out[..., 0], out[..., -1], rtol=1e-12)


class TestConfigSurface:
    def test_defaults_off_and_byte_identical_intent(self):
        cfg = GMRediConfig()
        assert cfg.resolution_function is False
        assert cfg.resfn_gamma == 2.0
        assert cfg.resfn_cbcl_ms == 2.0

    def test_param_spec_declares_new_floats(self):
        params = _LATMIX_PARAM_SPEC["GMRediConfig"]["params"]
        for name in ("resfn_gamma", "resfn_cbcl_ms"):
            assert name in params, f"{name} missing from GMRediConfig __param_spec__"
            spec = params[name]
            assert "units" in spec and "bounds" in spec
            assert spec["tunable_tier"] in (0, 1, 2, 3)
        # bool field is not float-spec-eligible -> must NOT appear in params
        assert "resolution_function" not in params


class TestLatLonWiring:
    """Exercise the REAL wired lat-lon GM/Redi path on a DINO state."""

    def _dino_call(self, resfn_cfg_kwargs, **call_kwargs):
        from legoesm.ocean.experiments.dino import (
            DINOConfig, create_dino_z_star, dino_lat_lon_grid,
            dino_lat_lon_state,
        )
        from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
            gm_redi_tracer_tendency_latlon,
        )
        dcfg = DINOConfig()
        z = create_dino_z_star(dcfg)
        g = dino_lat_lon_grid(dcfg, n_lon=50)
        st = dino_lat_lon_state(g, z, dcfg)
        cfg = GMRediConfig(visbeck=VisbeckConfig(enabled=True), **resfn_cfg_kwargs)
        dT, dS = gm_redi_tracer_tendency_latlon(
            st.T.data, st.S.data, st.eta.data, st.H_bathy.data, g, z, cfg,
            mask=st.land_mask.data, u_mask=st.u_mask.data, v_mask=st.v_mask.data,
            **call_kwargs,
        )
        return np.asarray(dT), np.asarray(dS)

    def test_off_byte_identical_to_default(self):
        dT_default, dS_default = self._dino_call({})
        dT_off, dS_off = self._dino_call(
            {"resolution_function": False, "resfn_gamma": 2.0,
             "resfn_cbcl_ms": 2.0})
        np.testing.assert_array_equal(dT_default, dT_off)
        np.testing.assert_array_equal(dS_default, dS_off)

    def test_on_changes_tendency_non_vacuous(self):
        """resolution_function=True must non-vacuously scale the GM (bolus) flux.

        The Hallberg taper scales the GM (skew/bolus) coefficient ONLY --
        ``kappa_Redi`` is deliberately left unscaled (NEMO ldf_eiv / MOM6
        RESOLN_SCALED_KHTH scale the eddy-transport coefficient, not the
        along-isopycnal diffusion).  On the coarse DINO grid the FULL tracer
        tendency is Redi/diagonal-dominated -- Visbeck's ``kappa_GM`` floors at
        ``kappa_min = 100 m^2/s`` on the quiescent initial state, an order of
        magnitude below ``kappa_Redi = 1000`` -- so the GM skew flux is only
        ~2% of the total and a GM-only taper is invisible under a full-tendency
        ``np.allclose`` (its absolute change ~1e-9 sits below the default
        ``atol = 1e-8``).  So isolate the GM skew flux by zeroing the Redi
        diffusivity (``kappa_redi_override = 0``): that tendency is linear in
        ``kappa_GM``, so applying ``f_res < 1`` (``gamma = 0.1`` =>
        ``f_res ~ 0.003-0.14`` across the basin) must suppress it by the same
        factor.  This exercises the SAME wired ``kappa_GM`` the full tendency
        consumes -- a genuine, tolerance-independent wiring check.
        """
        dT_gm_off, dS_gm_off = self._dino_call({}, kappa_redi_override=0.0)
        dT_gm_on, dS_gm_on = self._dino_call(
            {"resolution_function": True, "resfn_gamma": 0.1,
             "resfn_cbcl_ms": 2.0}, kappa_redi_override=0.0)
        assert np.all(np.isfinite(dT_gm_on)) and np.all(np.isfinite(dS_gm_on))
        off_mag = float(np.max(np.abs(dT_gm_off)))
        on_mag = float(np.max(np.abs(dT_gm_on)))
        assert off_mag > 0.0, (
            "GM-only tendency is identically zero -- test is vacuous")
        # f_res < 1 across the basin => the GM bolus tendency is strongly
        # suppressed (measured max ratio ~0.13; assert a robust < 0.5).
        assert on_mag < 0.5 * off_mag, (
            f"resolution_function=True did not suppress the GM bolus tendency "
            f"(max on={on_mag:.3e} vs off={off_mag:.3e})")
        # ... and it is a genuine, tolerance-independent relative change (the
        # unscaled Redi part is absent here, so atol=0 is the right comparison).
        assert not np.allclose(dT_gm_on, dT_gm_off, atol=0.0), (
            "resolution_function=True did not change the GM/Redi tendency")
