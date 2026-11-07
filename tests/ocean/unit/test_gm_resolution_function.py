"""Direct unit tests for the Hallberg (2013) resolution function that tapers
the GM coefficient by resolution (MED-3).

    f_res = 1 / (1 + (L_d / (gamma * dx))**2),   L_d = c_bcl / |f|

Covers the shared pure helpers in ``_gm_redi_common`` and the wiring into ALL
three GM/Redi grid paths (lat-lon C-grid, cubed-sphere, MPAS Voronoi), the
constant-kappa closure, the Redi invariance (GM-only taper), and the
prognostic-EKE / GEOMETRIC / 3-D-realized eddy-energy-budget coupling
(codex MED-3 r2 coverage matrix).  Default off => byte-identical.

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
    EPS,
    gm_resolution_factor,
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
        """Degenerate zero-area (land) cell: denom floored at EPS, f_res -> 0+,
        finite AND STRICTLY POSITIVE (the documented (0, 1] codomain holds
        exactly; codex MED-3 r2 tightened this from the old ``0.0 <=``)."""
        f_res = float(gm_resolution_function(5.0e4, 0.0, 2.0))
        assert np.isfinite(f_res)
        assert 0.0 < f_res < 1.0e-6

    def test_differentiable_finite_grad(self):
        def total(L_d):
            return jnp.sum(gm_resolution_function(
                L_d, jnp.full_like(L_d, 5.0e4), 2.0))
        g = jax.grad(total)(jnp.asarray([1.0e4, 5.0e4, 2.0e5]))
        assert bool(jnp.isfinite(g).all())


class TestFloatExactBounds:
    """Codex MED-3 r2: naive ``1/(1+ratio*ratio)`` collapsed to exactly 0.0
    once ``ratio**2`` overflowed — violating the documented strict (0, 1]
    codomain.  The hypot form + tiny-clamp make the bound float-exact."""

    def test_overflow_ratio_clamps_to_exact_tiny_f64(self):
        # ratio = 5e199: the naive square hit inf; the hypot form underflows
        # s*s to 0.0 instead — either way the result must be EXACTLY the
        # documented finfo.tiny clamp (codex r2 NIT: positivity alone would
        # let an arbitrary epsilon floor pass).
        f_res = gm_resolution_function(
            jnp.asarray(1.0e200), jnp.asarray(1.0), 2.0)
        assert float(f_res) == float(jnp.finfo(jnp.float64).tiny)

    def test_overflow_ratio_clamps_to_exact_tiny_f32(self):
        # (5e29)**2 = 2.5e59 > f32 max (3.4e38). jnp arrays (not np.float32
        # scalars) so weak-typed Python-float factors keep the math in f32.
        f_res = gm_resolution_function(
            jnp.asarray(1.0e30, dtype=jnp.float32),
            jnp.asarray(1.0, dtype=jnp.float32), 2.0)
        assert f_res.dtype == jnp.float32
        assert float(f_res) == float(jnp.finfo(jnp.float32).tiny)

    def test_bounds_hold_across_extreme_random_inputs(self):
        rng = np.random.default_rng(7)
        # log-uniform L_d over 60 decades incl. overflow-triggering values
        L_d = jnp.asarray(10.0 ** rng.uniform(-30.0, 300.0, size=300))
        dx = jnp.asarray(10.0 ** rng.uniform(-10.0, 8.0, size=300))
        f_res = np.asarray(gm_resolution_function(L_d, dx, 2.0))
        assert np.all(np.isfinite(f_res))
        assert np.all(f_res > 0.0)          # STRICT — the fixed contract
        assert np.all(f_res <= 1.0)


class TestPiecewiseGradientsFinite:
    """Codex MED-3 r2: the function is PIECEWISE-smooth (EPS denominator
    floor, |f| equatorial floor, tiny clamp are hard kinks — now documented
    as such instead of the old 'smooth and differentiable' overclaim).
    Gradients must be FINITE everywhere, including at/near every threshold
    and at zero inputs."""

    def test_grad_wrt_dx_at_zero_and_eps_floor(self):
        gamma = 2.0
        dx_kink = EPS / gamma          # where gamma*dx == EPS (the floor kink)
        dxs = jnp.asarray([0.0, 0.5 * dx_kink, dx_kink, 2.0 * dx_kink, 1.0e5])

        def total(d):
            return jnp.sum(gm_resolution_function(5.0e4, d, gamma))

        g = jax.grad(total)(dxs)
        assert bool(jnp.isfinite(g).all()), f"non-finite d/d(dx): {g}"

    def test_grad_wrt_f_at_equator_and_floor(self):
        dx = jnp.full((5,), 1.0e5)
        fs = jnp.asarray([0.0, 0.5 * _RESFN_F_FLOOR_S, _RESFN_F_FLOOR_S,
                          -_RESFN_F_FLOOR_S, 1.0e-4])

        def total(f):
            return jnp.sum(gm_resolution_scaled_kappa(1000.0, f, dx, 2.0, 2.0))

        g = jax.grad(total)(fs)
        assert bool(jnp.isfinite(g).all()), f"non-finite d/df: {g}"
        # Inside the floored band the kappa is |f|-independent -> exact zero.
        np.testing.assert_allclose(np.asarray(g)[:2], 0.0, atol=0.0)
        # Away from the floor the sensitivity is genuinely nonzero.
        assert abs(float(g[-1])) > 0.0

    def test_grad_wrt_gamma_and_c_finite_nonzero(self):
        f = jnp.asarray([0.0, 1.0e-5, 1.0e-4])
        dx = jnp.full((3,), 1.0e5)

        def total(gamma, c):
            return jnp.sum(gm_resolution_scaled_kappa(1000.0, f, dx, gamma, c))

        g_gamma = jax.grad(total, argnums=0)(2.0, 2.0)
        g_c = jax.grad(total, argnums=1)(2.0, 2.0)
        assert np.isfinite(float(g_gamma)) and np.isfinite(float(g_c))
        assert abs(float(g_gamma)) > 0.0
        assert abs(float(g_c)) > 0.0

    def test_grad_wrt_Ld_at_overflow_finite(self):
        def total(L):
            return jnp.sum(gm_resolution_function(L, jnp.asarray([1.0]), 2.0))

        g = jax.grad(total)(jnp.asarray([1.0e200]))
        assert bool(jnp.isfinite(g).all())

    def test_grad_and_jvp_at_float_max_Ld_and_zero_dx(self):
        """Codex MED-3 r2: the NAIVE form's VJP hit 0*inf = NaN once
        ``L_d/denom`` itself overflowed (L_d near float_max with the dx=0
        EPS-floored denominator). The hypot form must give FINITE reverse-
        AND forward-mode derivatives there, and a finite strictly-positive
        primal."""
        L_max = jnp.asarray([float(jnp.finfo(jnp.float64).max)])
        dx0 = jnp.asarray([0.0])

        def total(L, d):
            return jnp.sum(gm_resolution_function(L, d, 2.0))

        val = total(L_max, dx0)
        assert np.isfinite(float(val)) and float(val) > 0.0
        g_L, g_dx = jax.grad(total, argnums=(0, 1))(L_max, dx0)
        assert bool(jnp.isfinite(g_L).all()), f"NaN/Inf d/dL at float_max: {g_L}"
        assert bool(jnp.isfinite(g_dx).all()), f"NaN/Inf d/d(dx) at 0: {g_dx}"
        _, jvp_out = jax.jvp(
            lambda L: total(L, dx0), (L_max,), (jnp.ones_like(L_max),))
        assert np.isfinite(float(jvp_out))

    def test_factor_is_single_definition_of_scaled_kappa(self):
        """gm_resolution_scaled_kappa == kappa * gm_resolution_factor —
        one f_res definition shared with the model step's EKE coupling."""
        f = jnp.asarray([0.0, 3.0e-5, 1.0e-4, -8.0e-5])
        dx = jnp.asarray([5.0e4, 1.0e5, 2.0e5, 1.5e5])
        kappa = jnp.asarray([200.0, 500.0, 1000.0, 800.0])
        factor = gm_resolution_factor(f, dx, 2.0, 2.0)
        np.testing.assert_array_equal(
            np.asarray(gm_resolution_scaled_kappa(kappa, f, dx, 2.0, 2.0)),
            np.asarray(kappa * factor))
        assert np.all(np.asarray(factor) > 0.0)
        assert np.all(np.asarray(factor) <= 1.0)


class TestBroadcast3DRobust:
    """Codex MED-3 r2: the depth-resolved broadcast used
    ``isinstance(kappa_GM, jnp.ndarray)``, which silently SKIPPED the
    level-axis reshape for NumPy inputs (np.ndarray is not a jnp.ndarray)
    and is not a reliable tracer check — ``(lat,lon,lev) * (lat,lon)`` then
    failed to broadcast.  The fix is rank-based (``jnp.ndim``)."""

    def test_numpy_3d_kappa_input(self):
        kappa = np.ones((3, 4, 5)) * 1000.0        # NumPy, NOT jax
        f = jnp.full((3, 4), 1.0e-4)
        dx = jnp.full((3, 4), 1.0e5)
        out = np.asarray(gm_resolution_scaled_kappa(kappa, f, dx, 2.0, 2.0))
        assert out.shape == (3, 4, 5)
        np.testing.assert_allclose(out[..., 0], out[..., -1], rtol=1e-12)
        L_d = 2.0 / 1.0e-4
        np.testing.assert_allclose(
            out[..., 0], 1000.0 * _f_res_ref(L_d, 1.0e5, 2.0), rtol=1e-12)

    def test_3d_kappa_under_jit_matches_eager(self):
        kappa = jnp.ones((3, 4, 5)) * 1000.0
        f = jnp.full((3, 4), 1.0e-4)
        dx = jnp.full((3, 4), 1.0e5)
        eager = gm_resolution_scaled_kappa(kappa, f, dx, 2.0, 2.0)
        jitted = jax.jit(
            lambda k, ff, dd: gm_resolution_scaled_kappa(k, ff, dd, 2.0, 2.0)
        )(kappa, f, dx)
        assert jitted.shape == (3, 4, 5)
        np.testing.assert_allclose(
            np.asarray(jitted), np.asarray(eager), rtol=1e-14)

    def test_3d_kappa_grad_and_jit_grad(self):
        kappa = jnp.ones((2, 3, 4)) * 500.0
        f = jnp.full((2, 3), 5.0e-5)
        dx = jnp.full((2, 3), 2.0e5)

        def loss(k):
            return jnp.sum(gm_resolution_scaled_kappa(k, f, dx, 2.0, 2.0))

        g = jax.grad(loss)(kappa)
        assert g.shape == kappa.shape
        assert bool(jnp.isfinite(g).all())
        # dLoss/dkappa = f_res broadcast over levels — strictly in (0, 1].
        assert float(jnp.min(g)) > 0.0
        assert float(jnp.max(g)) <= 1.0
        g_jit = jax.jit(jax.grad(loss))(kappa)
        np.testing.assert_allclose(np.asarray(g_jit), np.asarray(g),
                                   rtol=1e-14)

    def test_grad_through_f_and_dx_with_3d_kappa(self):
        kappa = jnp.ones((2, 3, 4)) * 500.0

        def loss(f, dx):
            return jnp.sum(gm_resolution_scaled_kappa(kappa, f, dx, 2.0, 2.0))

        g_f, g_dx = jax.grad(loss, argnums=(0, 1))(
            jnp.full((2, 3), 5.0e-5), jnp.full((2, 3), 2.0e5))
        assert bool(jnp.isfinite(g_f).all())
        assert bool(jnp.isfinite(g_dx).all())
        assert float(jnp.max(jnp.abs(g_dx))) > 0.0


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

    def _dino_call(self, cfg_kwargs, **call_kwargs):
        """``cfg_kwargs`` are the FULL GMRediConfig kwargs (closure selection
        included — pass ``visbeck=VisbeckConfig(enabled=True)`` for the
        adaptive case, nothing for the constant-kappa case)."""
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
        cfg = GMRediConfig(**cfg_kwargs)
        dT, dS = gm_redi_tracer_tendency_latlon(
            st.T.data, st.S.data, st.eta.data, st.H_bathy.data, g, z, cfg,
            mask=st.land_mask.data, u_mask=st.u_mask.data, v_mask=st.v_mask.data,
            **call_kwargs,
        )
        return np.asarray(dT), np.asarray(dS)

    _VISBECK_ON = {"visbeck": VisbeckConfig(enabled=True)}

    def test_off_byte_identical_to_default(self):
        dT_default, dS_default = self._dino_call(dict(self._VISBECK_ON))
        dT_off, dS_off = self._dino_call(
            {**self._VISBECK_ON,
             "resolution_function": False, "resfn_gamma": 2.0,
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
        dT_gm_off, dS_gm_off = self._dino_call(
            dict(self._VISBECK_ON), kappa_redi_override=0.0)
        dT_gm_on, dS_gm_on = self._dino_call(
            {**self._VISBECK_ON,
             "resolution_function": True, "resfn_gamma": 0.1,
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

    def test_constant_kappa_closure_scaled(self):
        """CONSTANT-kappa closure (visbeck disabled — the path that never
        computes int N dz): the taper must scale the scalar cfg.kappa_GM
        into a suppressed field (codex MED-3 r2 coverage matrix)."""
        base = {"kappa_GM": 1000.0}
        dT_off, _ = self._dino_call(dict(base), kappa_redi_override=0.0)
        dT_on, _ = self._dino_call(
            {**base, "resolution_function": True, "resfn_gamma": 0.1,
             "resfn_cbcl_ms": 2.0}, kappa_redi_override=0.0)
        assert np.all(np.isfinite(dT_on))
        off_mag = float(np.max(np.abs(dT_off)))
        on_mag = float(np.max(np.abs(dT_on)))
        assert off_mag > 0.0, "constant-kappa GM tendency vacuously zero"
        assert on_mag < 0.5 * off_mag, (
            f"constant-kappa GM not suppressed (on={on_mag:.3e} vs "
            f"off={off_mag:.3e})")

    def test_redi_invariant_when_gm_zero(self):
        """Redi-invariance (codex MED-3 r2): with the GM coefficient zeroed
        (kappa_gm_override=0.0) the tendency is PURE Redi — and the taper is
        GM-only, so resolution_function=True must be an exact no-op."""
        base = dict(self._VISBECK_ON)
        dT_off, dS_off = self._dino_call(dict(base), kappa_gm_override=0.0)
        dT_on, dS_on = self._dino_call(
            {**base, "resolution_function": True, "resfn_gamma": 0.1,
             "resfn_cbcl_ms": 2.0}, kappa_gm_override=0.0)
        assert float(np.max(np.abs(dT_off))) > 0.0, (
            "pure-Redi tendency vacuously zero")
        np.testing.assert_array_equal(dT_off, dT_on)
        np.testing.assert_array_equal(dS_off, dS_on)


# ---------------------------------------------------------------------------
# Cubed-sphere wiring (gm_redi.gm_redi_lateral_mixing)
# ---------------------------------------------------------------------------


def _cube_setup(n=4, nlev=6, slope=1.0e-4):
    """Tilted stratified density on a C-n cube (same construction family as
    tests/ocean/unit/test_visbeck_gm.py's fixture — fixture only, no model
    numerics)."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_cubed_sphere(n)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=1000.0)
    shape = (6, n, n, nlev)
    jacobian = jnp.ones((6, n, n))
    rho_z = jnp.linspace(constants.rho_ocean, constants.rho_ocean + 2.0, nlev)
    H_total = float(jnp.sum(z_coord.dz_ref))
    grad_h = slope * (2.0 / H_total) * jnp.arange(n, dtype=jnp.float64)
    rho = (rho_z[None, None, None, :] + grad_h[None, :, None, None]
           + jnp.zeros(shape, dtype=jnp.float64))
    # Vertical + a mild horizontal T gradient so BOTH the GM off-diagonal
    # (S * dT/dz) and the isoneutral horizontal fluxes are exercised.
    T = (jnp.linspace(20.0, 5.0, nlev)[None, None, None, :]
         + 0.1 * jnp.arange(n, dtype=jnp.float64)[None, :, None, None]
         + jnp.zeros(shape, dtype=jnp.float64))
    S = jnp.full(shape, 35.0)
    u = jnp.zeros(shape)
    v = jnp.zeros(shape)
    return grid, z_coord, jacobian, u, v, T, S, rho


class TestCubeWiring:
    """Codex MED-3 r2 coverage matrix: the cubed-sphere entry point."""

    def test_cube_scales_gm_only_tendency(self):
        from legoesm.ocean.physics.lateral_mixing.gm_redi import (
            gm_redi_lateral_mixing,
        )
        grid, z, jac, u, v, T, S, rho = _cube_setup()
        base = dict(kappa_GM=1000.0, kappa_Redi=0.0)   # isolate the GM flux
        out_off = gm_redi_lateral_mixing(
            u, v, T, S, rho, z, jac, grid, GMRediConfig(**base))
        # C4 cells are huge (dx ~ 2.5e6 m); gamma must be tiny to push
        # f_res << 1 everywhere incl. the poles (ratio >~ 5 -> f_res <~ 0.04).
        out_on = gm_redi_lateral_mixing(
            u, v, T, S, rho, z, jac, grid,
            GMRediConfig(**base, resolution_function=True,
                         resfn_gamma=1.0e-3, resfn_cbcl_ms=2.0))
        off_mag = float(jnp.max(jnp.abs(out_off.dT_dt)))
        on_mag = float(jnp.max(jnp.abs(out_on.dT_dt)))
        assert np.all(np.isfinite(np.asarray(out_on.dT_dt)))
        assert off_mag > 0.0, "cube GM-only tendency vacuously zero"
        assert on_mag < 0.5 * off_mag, (
            f"cube GM tendency not suppressed (on={on_mag:.3e} vs "
            f"off={off_mag:.3e})")

    def test_cube_redi_invariant_when_gm_zero(self):
        from legoesm.ocean.physics.lateral_mixing.gm_redi import (
            gm_redi_lateral_mixing,
        )
        grid, z, jac, u, v, T, S, rho = _cube_setup()
        base = dict(kappa_GM=0.0, kappa_Redi=1000.0)   # pure Redi
        out_off = gm_redi_lateral_mixing(
            u, v, T, S, rho, z, jac, grid, GMRediConfig(**base))
        out_on = gm_redi_lateral_mixing(
            u, v, T, S, rho, z, jac, grid,
            GMRediConfig(**base, resolution_function=True,
                         resfn_gamma=1.0e-3, resfn_cbcl_ms=2.0))
        assert float(jnp.max(jnp.abs(out_off.dT_dt))) > 0.0, (
            "cube pure-Redi tendency vacuously zero")
        np.testing.assert_array_equal(
            np.asarray(out_off.dT_dt), np.asarray(out_on.dT_dt))
        np.testing.assert_array_equal(
            np.asarray(out_off.dS_dt), np.asarray(out_on.dS_dt))


# ---------------------------------------------------------------------------
# MPAS Voronoi wiring (gm_redi_mpas.gm_redi_tracer_tendency_mpas)
# ---------------------------------------------------------------------------


def _mpas_setup():
    """Level-2 icosahedral mesh + stably-stratified T with a meridional
    front (same construction family as tests/ocean/unit/test_gm_redi_mpas.py
    — fixture only, no model numerics)."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.vertical import create_ocean_z_star

    mesh = create_voronoi_mesh(subdivision_level=2)
    z_coord = create_ocean_z_star(
        n_levels=5, H_max=500.0, dz_surface=20.0, dz_deep=200.0)
    nlev = 5
    dz = np.asarray(z_coord.dz_ref)
    edges = np.concatenate([[0.0], -np.cumsum(dz)])
    zc = jnp.asarray(0.5 * (edges[:-1] + edges[1:]))
    T = jnp.broadcast_to(
        (5.0 + 15.0 * jnp.cos(mesh.latCell))[:, None] + (-0.005 * zc)[None, :],
        (mesh.nCells, nlev))
    S = jnp.full((mesh.nCells, nlev), 35.0, dtype=jnp.float64)
    eta = jnp.zeros((mesh.nCells,), dtype=jnp.float64)
    H_bathy = jnp.full((mesh.nCells,), 500.0, dtype=jnp.float64)
    return mesh, z_coord, T, S, eta, H_bathy


class TestMPASWiring:
    """Codex MED-3 r2 coverage matrix: the MPAS Voronoi entry point."""

    def test_mpas_scales_gm_only_tendency(self):
        from legoesm.ocean.physics.lateral_mixing.gm_redi_mpas import (
            gm_redi_tracer_tendency_mpas,
        )
        mesh, z, T, S, eta, H = _mpas_setup()
        base = dict(kappa_GM=1000.0, kappa_Redi=0.0, S_max=1e-2,
                    slope_scheme="centered")
        dT_off, _ = gm_redi_tracer_tendency_mpas(
            T, S, eta, H, mesh, z, GMRediConfig(**base), eos="linear")
        dT_on, _ = gm_redi_tracer_tendency_mpas(
            T, S, eta, H, mesh, z,
            GMRediConfig(**base, resolution_function=True,
                         resfn_gamma=1.0e-3, resfn_cbcl_ms=2.0),
            eos="linear")
        off_mag = float(jnp.max(jnp.abs(dT_off)))
        on_mag = float(jnp.max(jnp.abs(dT_on)))
        assert np.all(np.isfinite(np.asarray(dT_on)))
        assert off_mag > 0.0, "MPAS GM-only tendency vacuously zero"
        assert on_mag < 0.5 * off_mag, (
            f"MPAS GM tendency not suppressed (on={on_mag:.3e} vs "
            f"off={off_mag:.3e})")

    def test_mpas_redi_invariant_when_gm_zero(self):
        from legoesm.ocean.physics.lateral_mixing.gm_redi_mpas import (
            gm_redi_tracer_tendency_mpas,
        )
        mesh, z, T, S, eta, H = _mpas_setup()
        base = dict(kappa_GM=0.0, kappa_Redi=1000.0, S_max=1e-2,
                    slope_scheme="centered")
        dT_off, dS_off = gm_redi_tracer_tendency_mpas(
            T, S, eta, H, mesh, z, GMRediConfig(**base), eos="linear")
        dT_on, dS_on = gm_redi_tracer_tendency_mpas(
            T, S, eta, H, mesh, z,
            GMRediConfig(**base, resolution_function=True,
                         resfn_gamma=1.0e-3, resfn_cbcl_ms=2.0),
            eos="linear")
        assert float(jnp.max(jnp.abs(dT_off))) > 0.0, (
            "MPAS pure-Redi tendency vacuously zero")
        np.testing.assert_array_equal(np.asarray(dT_off), np.asarray(dT_on))
        np.testing.assert_array_equal(np.asarray(dS_off), np.asarray(dS_on))


# ---------------------------------------------------------------------------
# EKE-budget coupling (codex MED-3 r2): the SAME f_res that tapers the GM
# tracer flux must scale the GM-derived eddy-energy production, so the E
# budget receives the conversion the APPLIED coefficient performs.
# ---------------------------------------------------------------------------


class TestEKEProductionScaleHelper:
    """Direct unit tests of eke_apply_local_source(production_scale=...)."""

    def _inputs(self, shape=(3,)):
        E = jnp.full(shape, 0.5)
        sigma = jnp.full(shape, 5.0e-6)
        L = jnp.full(shape, 5.0e4)
        return E, sigma, L

    def test_affine_in_scale_and_none_means_unscaled(self):
        from legoesm.ocean.physics.lateral_mixing.eke import (
            EKEConfig, eke_apply_local_source,
        )
        cfg = EKEConfig()
        E, sigma, L = self._inputs()
        dt = 1800.0
        s = jnp.asarray([0.25, 0.5, 0.9])
        out_none = eke_apply_local_source(E, sigma, L, cfg, dt)
        out_kw_none = eke_apply_local_source(E, sigma, L, cfg, dt,
                                             production_scale=None)
        # None (default) is bit-identical to omitting the kwarg.
        np.testing.assert_array_equal(np.asarray(out_none),
                                      np.asarray(out_kw_none))
        out_zero = eke_apply_local_source(E, sigma, L, cfg, dt,
                                          production_scale=jnp.zeros_like(E))
        out_one = eke_apply_local_source(E, sigma, L, cfg, dt,
                                         production_scale=jnp.ones_like(E))
        # scale=1 reproduces the unscaled update exactly.
        np.testing.assert_allclose(np.asarray(out_one), np.asarray(out_none),
                                   rtol=1e-15)
        out_s = eke_apply_local_source(E, sigma, L, cfg, dt,
                                       production_scale=s)
        # The semi-implicit update is AFFINE in the production:
        # E_new(s) = E_new(0) + s * (E_new(1) - E_new(0)) exactly.
        np.testing.assert_allclose(
            np.asarray(out_s),
            np.asarray(out_zero + s * (out_one - out_zero)), rtol=1e-13)
        # s < 1 with positive production strictly reduces E_new,
        # and never below the production-free floor.
        assert np.all(np.asarray(out_s) < np.asarray(out_none))
        assert np.all(np.asarray(out_s) >= np.asarray(out_zero))

    def test_3d_broadcast_shape(self):
        from legoesm.ocean.physics.lateral_mixing.eke import (
            EKEConfig, eke_apply_local_source,
        )
        cfg = EKEConfig()
        E, sigma, L = self._inputs(shape=(2, 3, 4))
        scale_2d = jnp.full((2, 3, 1), 0.25)   # f_res[..., None], W-grid
        out = eke_apply_local_source(E, sigma, L, cfg, 1800.0,
                                     production_scale=scale_2d)
        assert out.shape == (2, 3, 4)
        assert bool(jnp.isfinite(out).all())
        assert bool((out >= 0.0).all())

    def test_ignored_when_production_override_supplied(self):
        """Documented contract: overrides carry their own scaling — the
        kwarg must NOT double-scale them."""
        from legoesm.ocean.physics.lateral_mixing.eke import (
            EKEConfig, eke_apply_local_source,
        )
        cfg = EKEConfig()
        E, sigma, L = self._inputs()
        P = jnp.full(E.shape, 3.0e-9)
        out_a = eke_apply_local_source(E, sigma, L, cfg, 1800.0,
                                       production_override=P)
        out_b = eke_apply_local_source(E, sigma, L, cfg, 1800.0,
                                       production_override=P,
                                       production_scale=jnp.full(E.shape, 0.1))
        np.testing.assert_array_equal(np.asarray(out_a), np.asarray(out_b))


def _channel_runner(eke_cfg, *, seed_eke=None, seed_velocity=False,
                    with_front=True, dt=1800.0):
    """Coarse-channel one-step runner for the EKE budget-coupling tests.

    Same fixture family as tests/ocean/unit/test_eke_geometric.py::_channel
    (12x24x4 lat-lon C-grid; optional meridional T front; optional sheared
    zonal velocity so the GEOMETRIC barotropic production B_T is nonzero).
    Returns ``(grid, run)`` where ``run(resfn_kwargs_or_None)`` steps the SAME
    initial state once under the given resolution-function config.  ONE step
    from an identical initial state isolates the production coupling exactly:
    transport and dissipation see the same state, so the E update is AFFINE
    in the production scale, ``E(f) = A + f*B`` pointwise.
    """
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    grid = create_latlon_grid(12, 24)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    state0 = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=4000.0, land_lat_threshold=85.0)
    if with_front:
        T = np.asarray(state0.T.data)
        lat = np.degrees(np.asarray(grid.lat))
        T = T + 2.0 * np.tanh(lat / 20.0)[:, None, None]
        state0 = state0._replace(T=state0.T.replace(data=jnp.asarray(T)))
    if seed_velocity:
        # Meridionally-sheared, ZONALLY-UNIFORM zonal jet (0.3 m/s scale):
        # |grad_h u|^2 > 0 so the GEOMETRIC B_T = int kappa_u*|grad u|^2 dz
        # is genuinely nonzero (codex MED-3 r2: a rest state made the "B_T
        # untouched" claim vacuous), while du/dx = 0 and v = 0 keep the
        # depth-mean flow NON-DIVERGENT — the transport of a uniform E field
        # is then exactly zero, so any E growth above the uniform cold start
        # is attributable to production alone.
        u0 = np.asarray(state0.u.data)
        cosl = np.cos(np.asarray(grid.lat))[:, None, None]
        u_seed = 0.3 * cosl * np.ones_like(u0)
        u_seed = u_seed * np.asarray(state0.u_mask.data)[:, :, None]
        state0 = state0._replace(u=state0.u.replace(data=jnp.asarray(u_seed)))
    if seed_eke is not None:
        state0 = state0._replace(eke=seed_eke(grid, z_coord))

    def run(resfn_kwargs, *, eager=False):
        from legoesm.core.precision import (
            PrecisionPolicy, get_policy, set_policy,
        )
        gm = GMRediConfig(kappa_GM=0.0, kappa_Redi=1.0e3, eke=eke_cfg,
                          **(resfn_kwargs or {}))
        cfg = LatLonCGridOceanConfig.from_flat(
            A_h=2.0e4, bottom_drag_r=1.0e-3, implicit_vertical_mixing=True,
            n_barotropic_substeps=8, enable_runtime_checks=False, gm_redi=gm)
        # FLOAT64 storage policy for the oracle runs (restored in finally):
        # the DEFAULT policy stores the stepped state in float32, whose
        # ~1e-9 ulp at E~1e-2 quantizes the recovered one-step production
        # term (only ~8 f32 ulps here) and broke the exact affine oracle
        # with a spurious few-percent mismatch (v3 job 8919883 adjudication:
        # fix-test — the coupling itself was live and correct; the
        # suppression inequalities passed even at f32).
        old_policy = get_policy()
        set_policy(PrecisionPolicy.fp64())
        try:
            model = LatLonCGridOceanModel(grid, z_coord, cfg)
            step = model._step_impl if eager else model.step
            return step(state0, dt=dt)
        finally:
            set_policy(old_policy)

    return grid, run


def _resfn_on(gamma):
    return {"resolution_function": True, "resfn_gamma": gamma,
            "resfn_cbcl_ms": 2.0}


# gamma=1e-3 on the 12x24 global grid (dx ~ 1.5e6 m) drives f_res << 1
# everywhere (<~ 0.03 at the poles), so the production suppression is a
# first-order effect, not a tolerance game.
_RESFN_ON = _resfn_on(1.0e-3)


def _f_res_field(grid, gamma, c=2.0):
    """Independent f_res oracle: the SAME expression the model step and the
    tracer tendency use (broadcast grid.f, sqrt(cell area)) — via the
    unit-tested shared helper."""
    f2d = jnp.broadcast_to(grid.f, (grid.n_lat, grid.n_lon))
    f_res = gm_resolution_factor(f2d, jnp.sqrt(grid.area), gamma, c)
    return np.broadcast_to(np.asarray(f_res), (grid.n_lat, grid.n_lon))


def _recovered_production_term(E_off, E_g, f_res):
    """Invert the affine one-step update E(f) = A + f*B pointwise:
    B = (E(1) - E(f)) / (1 - f) — the (dissipation-folded) GM production
    term.  Must be IDENTICAL for every gamma if and only if the production
    multiplier is exactly f_res (codex MED-3 r2: an inequality alone cannot
    distinguish f_res from f_res**2 or a double-scaling)."""
    return (E_off - E_g) / (1.0 - f_res)


class TestEKEBudgetCoupling:
    """Step-level wiring: prognostic closures' E budget consumes the SAME
    f_res-scaled GM conversion the tracer flux applies — pinned EXACTLY via
    the affine-in-scale one-step oracle, not just an inequality."""

    def _affine_oracle_check(self, grid, E_off, E_1, E_2, g1, g2):
        f1 = _f_res_field(grid, g1)
        f2 = _f_res_field(grid, g2)
        assert float(np.max(f1)) < 0.9 and float(np.max(f2)) < 0.9  # safe 1-f
        B1 = _recovered_production_term(E_off, E_1, f1)
        B2 = _recovered_production_term(E_off, E_2, f2)
        assert np.all(np.isfinite(B1)) and np.all(np.isfinite(B2))
        # Affine consistency: the same B field from BOTH gammas.  Tolerance
        # rationale (v4 job 8920136 adjudication): the model multiplies
        # f_res IN-GRAPH (fused with the flux/production ops) while this
        # oracle rescales the stepped OUTPUTS post hoc, so float
        # reassociation between the two differently-compiled programs
        # leaves ~1-2 ulps of E (~2.7e-18) on recovered terms as small as
        # ~8e-13 — observed rel ~1e-6.  rtol=1e-5 absorbs that while
        # keeping full discriminating power: an f_res**2 / double-scaling
        # defect errs at the few-PERCENT level and a dropped f_res by
        # O(10) — both still fail by 3+ orders.  Do NOT tighten toward
        # bit-exactness; that would test XLA fusion, not the physics.
        np.testing.assert_allclose(B1, B2, rtol=1e-5, atol=1e-17)
        # The recovered GM production term is >= 0 and non-vacuous.
        assert float(np.min(B1)) >= -1e-17
        assert float(np.max(B1)) > 0.0, "GM production vacuously zero"

    def test_eg2d_production_scaled_by_exactly_f_res(self):
        from legoesm.ocean.physics.lateral_mixing.eke import EKEConfig
        from legoesm.ocean.state import Field

        def seed(grid, z):
            return Field(data=jnp.full((grid.n_lat, grid.n_lon), 1.0e-2),
                         name="eke", dims=("lat", "lon"), units="m^2/s^2")

        grid, run = _channel_runner(EKEConfig(), seed_eke=seed)
        g1, g2 = 1.0e-3, 3.0e-3
        s_off = run(None)
        s_1 = run(_resfn_on(g1))
        s_2 = run(_resfn_on(g2))
        E_off = np.asarray(s_off.eke.data)
        E_1 = np.asarray(s_1.eke.data)
        E_2 = np.asarray(s_2.eke.data)
        assert np.all(np.isfinite(E_1)) and np.all(E_1 >= 0.0)
        # One step from an identical state: pointwise E_1 <= E_off, strict
        # somewhere (production > 0 on the front), never trivially equal.
        assert np.all(E_1 <= E_off + 1e-30)
        assert float(np.sum(E_off - E_1)) > 0.0, (
            "resolution_function did not reach the EG-2D EKE production")
        # EXACT oracle: production multiplier is f_res itself.
        self._affine_oracle_check(grid, E_off, E_1, E_2, g1, g2)
        # ... and the tracer path is ALSO suppressed in the same step (the
        # two consumers see one effective kappa).
        assert not np.array_equal(np.asarray(s_off.T.data),
                                  np.asarray(s_1.T.data))

    def test_geometric_bc_scaled_by_exactly_f_res(self):
        from legoesm.ocean.physics.lateral_mixing.eke import (
            EKEConfig, GeometricConfig,
        )
        eke_cfg = EKEConfig(closure="geometric", geometric=GeometricConfig())
        # Cold start (state.eke=None): the step builds e0_per_depth*H itself.
        # Seeded velocity => B_T > 0 sits in the affine intercept A; if B_T
        # were wrongly tapered the recovered-B fields would still agree, so
        # the dedicated zero-B_C invariance test below is the discriminator —
        # here B_T>0 exercises the full override sum under scaling.
        grid, run = _channel_runner(eke_cfg, seed_velocity=True)
        g1, g2 = 1.0e-3, 3.0e-3
        s_off = run(None)
        s_1 = run(_resfn_on(g1))
        s_2 = run(_resfn_on(g2))
        E_off = np.asarray(s_off.eke.data)
        E_1 = np.asarray(s_1.eke.data)
        E_2 = np.asarray(s_2.eke.data)
        assert np.all(np.isfinite(E_1)) and np.all(E_1 >= 0.0)
        assert np.all(E_1 <= E_off + 1e-30)
        assert float(np.sum(E_off - E_1)) > 0.0, (
            "resolution_function did not reach the GEOMETRIC B_C production")
        self._affine_oracle_check(grid, E_off, E_1, E_2, g1, g2)

    def test_geometric_bt_only_state_invariant_under_taper(self):
        """B_T-untouched discriminator (codex MED-3 r2): horizontally-uniform
        T/S => isopycnal slopes ~ 0 => B_C ~ 0, while the seeded sheared jet
        keeps B_T > 0.  The taper scales ONLY the GM-derived B_C, so the EKE
        after one step must be UNCHANGED by resolution_function to far below
        the dt*(1-f_res)*B_T signal a wrongly-tapered B_T would leave
        (~1e-5 here; tolerance 1e-12; slope-floor noise ~1e-24)."""
        from legoesm.ocean.physics.lateral_mixing.eke import (
            EKEConfig, GeometricConfig,
        )
        eke_cfg = EKEConfig(closure="geometric", geometric=GeometricConfig())
        grid, run = _channel_runner(eke_cfg, seed_velocity=True,
                                    with_front=False)
        s_off = run(None)
        s_on = run(_RESFN_ON)
        E_off = np.asarray(s_off.eke.data)
        E_on = np.asarray(s_on.eke.data)
        # Non-vacuity: some cell GAINED energy over the uniform cold start.
        # The seeded flow is non-divergent (see _channel_runner) so the
        # transport of the uniform E is exactly zero and dissipation only
        # shrinks E — with B_C ~ 0 (flat isopycnals) the ONLY lift is B_T.
        e0 = float(eke_cfg.geometric.e0_per_depth) * 4000.0
        assert float(np.max(E_off)) > e0, (
            "B_T vacuously zero — seeded velocity produced no production")
        np.testing.assert_allclose(E_on, E_off, rtol=0.0, atol=1e-12)

    def test_eke3d_realized_conversion_suppressed_one_step(self):
        from legoesm.ocean.physics.lateral_mixing.eke import EKEConfig
        from legoesm.ocean.state import Field

        def seed(grid, z):
            nlevm1 = z.dz_ref.shape[0] - 1
            return Field(
                data=jnp.full((grid.n_lat, grid.n_lon, nlevm1), 1.0e-3),
                name="eke", dims=("lat", "lon", "level"), units="m^2/s^2")

        eke_cfg = EKEConfig(eke_3d=True, gm_source_mode="realized")
        grid, run = _channel_runner(eke_cfg, seed_eke=seed)
        s_off = run(None)
        s_on = run(_RESFN_ON)
        E_off = np.asarray(s_off.eke.data)
        E_on = np.asarray(s_on.eke.data)
        assert E_on.ndim == 3
        assert np.all(np.isfinite(E_on)) and np.all(E_on >= 0.0)
        assert np.all(E_on <= E_off + 1e-30)
        assert float(np.sum(E_off - E_on)) > 0.0, (
            "resolution_function did not reach the 3-D realized EKE source")

    def test_signed_iso_sink_gets_raw_kappa_split_call(self, monkeypatch):
        """Codex MED-3 r2 finding 1 regression: in the realized_signed +
        source_p_diss_iso + isopycnal_diffusion=False corner, the callee's
        kappa_redi_w=None fallback aliases its kappa_gm_w argument — the
        step must therefore SPLIT the call: skew conversion from the SCALED
        kappa, iso conversion from the RAW kappa.  Recorded via a delegating
        wrapper on the model module's imported symbol; with the taper OFF
        the legacy single call is preserved."""
        import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as mod
        from legoesm.ocean.physics.lateral_mixing.eke import EKEConfig
        from legoesm.ocean.state import Field

        real = mod.compute_realized_signed_conversions
        calls = []

        def recorder(*args, **kwargs):
            calls.append((args, kwargs))
            return real(*args, **kwargs)

        monkeypatch.setattr(
            mod, "compute_realized_signed_conversions", recorder)

        def seed(grid, z):
            nlevm1 = z.dz_ref.shape[0] - 1
            return Field(
                data=jnp.full((grid.n_lat, grid.n_lon, nlevm1), 1.0e-3),
                name="eke", dims=("lat", "lon", "level"), units="m^2/s^2")

        eke_cfg = EKEConfig(eke_3d=True, gm_source_mode="realized_signed",
                            source_p_diss_iso=True,
                            isopycnal_diffusion=False)
        grid, run = _channel_runner(eke_cfg, seed_eke=seed)

        # disable_jit: model.step is jitted, so the recorder would otherwise
        # capture TRACERS (np.asarray on them raises
        # TracerArrayConversionError — v3 job 8919883). Eager mode hands it
        # the concrete kappa arrays; the tiny channel keeps this cheap.
        # Taper OFF: the legacy SINGLE call (byte-identity of the corner).
        calls.clear()
        with jax.disable_jit():
            run(None, eager=True)
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert kwargs.get("want_skew", True) is True
        assert kwargs.get("want_iso") is True
        # The RAW kappa reference: gamma feeds only f_res (never the kappa
        # chain), and eager execution is per-op deterministic, so the ON
        # run's raw kappa must be BITWISE this one.
        kappa_off = np.asarray(args[7])

        # Taper ON: split — skew from the SCALED kappa, iso from the RAW.
        gamma = 1.0e-3
        calls.clear()
        with jax.disable_jit():
            run(_resfn_on(gamma), eager=True)
        assert len(calls) == 2, (
            "expected the split skew/iso calls in the uncoupled corner")
        skew_calls = [c for c in calls if c[1].get("want_skew") is True]
        iso_calls = [c for c in calls if c[1].get("want_iso") is True]
        assert len(skew_calls) == 1 and len(iso_calls) == 1
        (skew_args, skew_kwargs) = skew_calls[0]
        (iso_args, iso_kwargs) = iso_calls[0]
        assert skew_kwargs["want_iso"] is False
        assert iso_kwargs["want_skew"] is False
        kappa_skew = np.asarray(skew_args[7])   # kappa_gm_w positional slot
        kappa_iso = np.asarray(iso_args[7])
        assert kappa_skew.shape == kappa_iso.shape
        # THE invariant of codex finding 1: the iso conversion receives the
        # RAW kappa — bitwise equal to the taper-OFF run's kappa (same
        # eager computation; gamma never feeds the kappa chain).
        np.testing.assert_array_equal(kappa_iso, kappa_off)
        # ... and the skew kappa is the f_res-scaled one: elementwise ratio
        # strictly inside (0, 1) and ~f_res.  Loose rtol=1e-5 on the f_res
        # comparison (v4 job 8920136 adjudication): the model computes
        # f_res from its own (policy-cast) grid arrays while this oracle
        # uses the raw host grid — a ~5e-7 relative precision-chain delta,
        # NOT a wrong factor (a dropped/squared f_res errs by O(10) / percent-level).
        wet = kappa_iso > 0.0
        assert wet.any()
        ratio = kappa_skew[wet] / kappa_iso[wet]
        assert np.all(ratio > 0.0) and np.all(ratio < 1.0)
        f_res_w = np.broadcast_to(
            _f_res_field(grid, gamma)[..., None], kappa_iso.shape)
        np.testing.assert_allclose(ratio, f_res_w[wet], rtol=1e-5, atol=0.0)
        assert not np.array_equal(kappa_skew, kappa_iso)
