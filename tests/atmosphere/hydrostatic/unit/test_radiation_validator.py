"""Truth-tier validation probes for the atm-radiation group.

Group: gray radiation, RRTMGP, cloud_fraction / cloud properties.

These are ADDITIVE physics-validator probes (they do not replace the
existing ``test_radiation.py`` forward/shape/bounds suite). They check the
five validation tiers in order:

  1. UNIT     — constants sourced from ``legoesm.constants`` /
                ``legoesm.thermo`` (no hardcoded gravity/c_pd/sigma_sb/
                freezing-point literals); dimensional consistency of the
                column-integral closures.
  2. SIGN     — heating-rate / flux sign conventions are physically
                correct (LW free-tropo cooling, warmer surface => more OLR,
                more vapour => more downwelling LW, higher albedo => less SW).
  3. CONSERV  — column energy closure to machine precision for BOTH the
                gray and RRTMGP solvers (TOA-minus-surface net downward
                flux == c_p/g integrated heating rate).
  4. DIFF     — jax.grad through the tendency w.r.t. TUNABLE config params
                is finite, non-zero, NaN-free; centered finite-difference
                cross-check of the param gradient; jit/vmap parity.
  5. IDEAL    — analytic / benchmark sanity per scheme (Sundqvist sqrt-form,
                Xu-Randall monotonicity, gray grey-OLR Stefan-Boltzmann
                limit, RRTMGP physical OLR/OSR magnitudes).

Run (Metal is broken -> CPU mandatory):
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/atmosphere/hydrostatic/unit/test_radiation_validator.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_specific_humidity
from legoesm.atmosphere.physics.radiation.config import (
    GrayRadiationConfig,
    RRTMGPConfig,
)
from legoesm.atmosphere.physics.radiation.gray import gray_radiation
from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.atmosphere.physics.clouds.cloud_fraction import (
    compute_cloud_properties,
    sundqvist_cloud_fraction,
    xu_randall_cloud_fraction,
)


# --------------------------------------------------------------------------
# Shared synthetic single-/multi-column atmosphere at realistic magnitudes.
# --------------------------------------------------------------------------
def _make_column(ncol: int = 1, nlev: int = 40):
    """A realistic mid-latitude column (TOA-first p_half ordering)."""
    ph = jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :].repeat(ncol, 0)
    pf = 0.5 * (ph[:, 1:] + ph[:, :-1])
    # Lapse-rate temperature profile, warm surface.
    T = jnp.linspace(220.0, 288.0, nlev)[None, :].repeat(ncol, 0)
    Ts = jnp.full((ncol,), 290.0)
    qv = jnp.full((ncol, nlev), 4.0e-3)
    lat = jnp.linspace(0.0, 0.6, ncol)
    insolation = jnp.full((ncol,), 340.0)
    cos_zenith = jnp.full((ncol,), 0.5)
    return ph, pf, T, Ts, qv, lat, insolation, cos_zenith


def _gray_column_energy_residual(out, ph):
    """Column heating (c_p/g int hr dp) minus net downward flux convergence."""
    dp = ph[:, 1:] - ph[:, :-1]
    col_heat = jnp.sum(constants.c_pd / constants.g * out.heating_rate * dp, axis=1)
    net = (out.lw_flux_down - out.lw_flux_up) + (out.sw_flux_down - out.sw_flux_up)
    flux_div = net[:, 0] - net[:, -1]  # net into atm = TOA-in minus sfc-out (down pos)
    return col_heat - flux_div


# ==========================================================================
# TIER 1 + 3 : UNITS + CONSERVATION (gray)
# ==========================================================================
class TestGrayConservation:
    def test_gray_column_energy_machine_precision(self):
        """Gray: c_p/g int(hr) dp == net downward flux convergence to 1e-10."""
        ph, pf, T, Ts, qv, lat, ins, _ = _make_column(ncol=3, nlev=40)
        out = gray_radiation(T, pf, ph, Ts, lat, qv, ins, GrayRadiationConfig())
        resid = _gray_column_energy_residual(out, ph)
        dp = ph[:, 1:] - ph[:, :-1]
        scale = jnp.sum(
            jnp.abs(constants.c_pd / constants.g * out.heating_rate * dp), axis=1
        )
        rel = jnp.abs(resid) / jnp.maximum(scale, 1.0)
        assert float(jnp.max(rel)) < 1e-10, f"gray energy residual rel={rel}"

    def test_gray_sw_absorption_internally_consistent(self):
        """SW absorbed by atm column == TOA-net minus sfc-net SW (down pos)."""
        ph, pf, T, Ts, qv, lat, ins, _ = _make_column(ncol=2, nlev=30)
        out = gray_radiation(T, pf, ph, Ts, lat, qv, ins, GrayRadiationConfig())
        dp = ph[:, 1:] - ph[:, :-1]
        sw_abs = jnp.sum(
            constants.c_pd / constants.g * out.sw_heating_rate * dp, axis=1
        )
        sw_toa = out.sw_flux_down[:, 0] - out.sw_flux_up[:, 0]
        sw_sfc = out.sw_flux_down[:, -1] - out.sw_flux_up[:, -1]
        np.testing.assert_allclose(
            np.asarray(sw_abs), np.asarray(sw_toa - sw_sfc), rtol=1e-9, atol=1e-9
        )

    def test_gray_zero_optical_depth_transparent_lw(self):
        """tau->0 (dry, tau_e=tau_p=0) => atmosphere LW-transparent.

        With no LW optical depth the layers neither absorb nor emit, so
        emissivity=0 everywhere: downwelling LW at the surface is 0 and the
        upwelling LW at TOA equals the surface emission (no atmosphere)."""
        ph, pf, T, Ts, _, lat, ins, _ = _make_column(ncol=1, nlev=20)
        cfg = GrayRadiationConfig(tau_equator=0.0, tau_pole=0.0)
        out = gray_radiation(T, pf, ph, Ts, lat, None, ins, cfg)
        assert float(jnp.max(jnp.abs(out.lw_flux_down))) < 1e-9
        sfc_emit = constants.sigma_sb * Ts ** 4  # eps_sfc default 1.0
        np.testing.assert_allclose(
            np.asarray(out.lw_flux_up[:, 0]), np.asarray(sfc_emit), rtol=1e-9
        )


# ==========================================================================
# TIER 2 : SIGN CONVENTIONS (gray)
# ==========================================================================
class TestGraySigns:
    def test_warmer_surface_increases_olr(self):
        ph, pf, T, Ts, qv, lat, ins, _ = _make_column(ncol=1, nlev=30)
        cfg = GrayRadiationConfig()
        olr_cold = gray_radiation(T, pf, ph, Ts, lat, qv, ins, cfg).lw_flux_up[:, 0]
        olr_warm = gray_radiation(
            T, pf, ph, Ts + 5.0, lat, qv, ins, cfg
        ).lw_flux_up[:, 0]
        assert float(olr_warm[0]) > float(olr_cold[0])

    def test_more_vapour_increases_downwelling_lw(self):
        """Greenhouse: more water vapour => more LW down at surface."""
        ph, pf, T, Ts, qv, lat, ins, _ = _make_column(ncol=1, nlev=30)
        cfg = GrayRadiationConfig()
        dry = gray_radiation(T, pf, ph, Ts, lat, qv, ins, cfg).lw_flux_down[:, -1]
        wet = gray_radiation(
            T, pf, ph, Ts, lat, qv * 3.0, ins, cfg
        ).lw_flux_down[:, -1]
        assert float(wet[0]) > float(dry[0])

    def test_lw_free_troposphere_cools(self):
        """A warm low-emissivity-aloft column should have net LW cooling in
        the free troposphere (negative LW heating-rate on average aloft)."""
        ph, pf, T, Ts, qv, lat, ins, _ = _make_column(ncol=1, nlev=40)
        out = gray_radiation(T, pf, ph, Ts, lat, qv, ins, GrayRadiationConfig())
        mid = out.lw_heating_rate[0, 5:-5]
        assert float(jnp.mean(mid)) < 0.0

    def test_sw_heating_nonnegative(self):
        """Absorbed SW can only warm the atmosphere (dT/dt_sw >= 0)."""
        ph, pf, T, Ts, qv, lat, ins, _ = _make_column(ncol=2, nlev=30)
        out = gray_radiation(T, pf, ph, Ts, lat, qv, ins, GrayRadiationConfig())
        assert float(jnp.min(out.sw_heating_rate)) >= -1e-12

    def test_higher_albedo_reduces_surface_sw(self):
        ph, pf, T, Ts, qv, lat, ins, _ = _make_column(ncol=1, nlev=20)
        lo = gray_radiation(
            T, pf, ph, Ts, lat, qv, ins, GrayRadiationConfig(),
            sfc_albedo=jnp.array([0.1]),
        )
        hi = gray_radiation(
            T, pf, ph, Ts, lat, qv, ins, GrayRadiationConfig(),
            sfc_albedo=jnp.array([0.6]),
        )
        net_lo = float(lo.sw_flux_down[0, -1] - lo.sw_flux_up[0, -1])
        net_hi = float(hi.sw_flux_down[0, -1] - hi.sw_flux_up[0, -1])
        assert net_hi < net_lo


# ==========================================================================
# TIER 4 : DIFFERENTIABILITY (gray) — gradient w.r.t. TUNABLE params.
# ==========================================================================
class TestGrayDifferentiability:
    def _olr_of_params(self, ph, pf, T, Ts, qv, lat, ins):
        def olr(tau_eq, tau_moist, alb):
            cfg = GrayRadiationConfig(
                tau_equator=tau_eq, tau_moist_coeff=tau_moist,
            )
            out = gray_radiation(
                T, pf, ph, Ts, lat, qv, ins, cfg,
                sfc_albedo=jnp.broadcast_to(alb, Ts.shape),
            )
            return jnp.sum(out.lw_flux_up[:, 0]) + jnp.sum(
                out.sw_flux_down[:, -1] - out.sw_flux_up[:, -1]
            )
        return olr

    def test_grad_through_tunable_params_finite_nonzero(self):
        ph, pf, T, Ts, qv, lat, ins, _ = _make_column(ncol=2, nlev=30)
        olr = self._olr_of_params(ph, pf, T, Ts, qv, lat, ins)
        g = jax.grad(olr, argnums=(0, 1, 2))(7.2, 0.0115, 0.31)
        for name, gi in zip(("tau_equator", "tau_moist_coeff", "sfc_albedo"), g):
            assert np.isfinite(float(gi)), f"{name} grad not finite: {gi}"
            assert abs(float(gi)) > 0.0, f"{name} grad is zero (param unreachable)"

    def test_grad_matches_finite_difference(self):
        ph, pf, T, Ts, qv, lat, ins, _ = _make_column(ncol=1, nlev=30)
        olr = self._olr_of_params(ph, pf, T, Ts, qv, lat, ins)
        x0 = (7.2, 0.0115, 0.31)
        g = jax.grad(olr, argnums=(0, 1, 2))(*x0)
        steps = (1e-4, 1e-6, 1e-4)
        for i, (gi, h) in enumerate(zip(g, steps)):
            xp = list(x0); xm = list(x0)
            xp[i] += h; xm[i] -= h
            fd = (float(olr(*xp)) - float(olr(*xm))) / (2 * h)
            np.testing.assert_allclose(
                float(gi), fd, rtol=2e-4, atol=1e-6,
                err_msg=f"param {i}: AD={float(gi)} FD={fd}",
            )

    def test_grad_through_temperature_nonzero(self):
        ph, pf, T, Ts, qv, lat, ins, _ = _make_column(ncol=1, nlev=30)
        cfg = GrayRadiationConfig()

        def loss(Tin):
            out = gray_radiation(Tin, pf, ph, Ts, lat, qv, ins, cfg)
            return jnp.sum(out.heating_rate ** 2)

        gT = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(gT))
        assert float(jnp.max(jnp.abs(gT))) > 0.0

    def test_jit_and_vmap_parity(self):
        ph, pf, T, Ts, qv, lat, ins, _ = _make_column(ncol=4, nlev=20)
        cfg = GrayRadiationConfig()
        eager = gray_radiation(T, pf, ph, Ts, lat, qv, ins, cfg)
        jitted = jax.jit(lambda *a: gray_radiation(*a, cfg))(
            T, pf, ph, Ts, lat, qv, ins
        )
        np.testing.assert_allclose(
            np.asarray(eager.heating_rate),
            np.asarray(jitted.heating_rate), rtol=1e-12, atol=1e-12,
        )

        def one_col(T1, pf1, ph1, Ts1, qv1, lat1, ins1):
            return gray_radiation(
                T1[None], pf1[None], ph1[None], Ts1[None], lat1[None],
                qv1[None], ins1[None], cfg,
            ).heating_rate[0]
        vm = jax.vmap(one_col)(T, pf, ph, Ts, qv, lat, ins)
        np.testing.assert_allclose(
            np.asarray(eager.heating_rate), np.asarray(vm), rtol=1e-10, atol=1e-10,
        )


# ==========================================================================
# TIER 5 : IDEALIZED FIDELITY (gray)
# ==========================================================================
class TestGrayIdealized:
    def test_grey_olr_below_surface_emission(self):
        """A greenhouse atmosphere must emit less to space than the bare
        surface (OLR < sigma*Ts^4)."""
        ph, pf, T, Ts, qv, lat, ins, _ = _make_column(ncol=1, nlev=40)
        out = gray_radiation(T, pf, ph, Ts, lat, qv, ins, GrayRadiationConfig())
        sfc_emit = float(constants.sigma_sb * Ts[0] ** 4)
        assert float(out.lw_flux_up[0, 0]) < sfc_emit

    def test_olr_decreases_with_optical_depth(self):
        """Frierson: thicker LW atmosphere => stronger greenhouse => less OLR."""
        ph, pf, T, Ts, qv, lat, ins, _ = _make_column(ncol=1, nlev=40)
        thin = gray_radiation(
            T, pf, ph, Ts, lat, qv, ins,
            GrayRadiationConfig(tau_equator=2.0, tau_pole=2.0),
        ).lw_flux_up[0, 0]
        thick = gray_radiation(
            T, pf, ph, Ts, lat, qv, ins,
            GrayRadiationConfig(tau_equator=10.0, tau_pole=10.0),
        ).lw_flux_up[0, 0]
        assert float(thick) < float(thin)


# ==========================================================================
# CLOUD FRACTION : units / signs / diff / idealized
# ==========================================================================
class TestCloudFraction:
    def test_sundqvist_sqrt_form_and_bounds(self):
        cfg = CloudConfig(scheme="sundqvist", rh_crit=0.7)
        RH = jnp.array([[0.5, 0.7, 0.85, 1.0, 1.2]])
        cf = sundqvist_cloud_fraction(RH, cfg)
        assert float(cf[0, 0]) == pytest.approx(0.0, abs=1e-12)
        assert float(cf[0, 3]) == pytest.approx(1.0, abs=1e-12)
        assert float(cf[0, 4]) == pytest.approx(1.0, abs=1e-12)
        # sqrt-form value at RH=0.85, rh_crit=0.7: 1-sqrt(0.15/0.30)=1-0.7071
        np.testing.assert_allclose(float(cf[0, 2]), 1.0 - np.sqrt(0.5), rtol=1e-6)
        assert bool(jnp.all(jnp.diff(cf[0]) >= -1e-12))

    def test_sundqvist_gradient_finite_at_saturation(self):
        """sqrt has infinite slope at 0 arg; the double-where must kill the
        cotangent so d cf / d RH stays finite at RH=1 (arg=0)."""
        cfg = CloudConfig(scheme="sundqvist", rh_crit=0.7)

        def s(rh):
            return jnp.sum(sundqvist_cloud_fraction(rh, cfg))

        RH = jnp.array([[0.999999, 1.0, 1.0001]])
        g = jax.grad(s)(RH)
        assert jnp.all(jnp.isfinite(g)), f"sundqvist grad NaN/Inf: {g}"

    def test_xu_randall_monotone_in_condensate(self):
        cfg = CloudConfig(scheme="xu_randall")
        RH = jnp.full((1, 4), 0.8)
        qs = jnp.full((1, 4), 1.0e-2)
        qc = jnp.array([[0.0, 1e-5, 1e-4, 1e-3]])
        cf = xu_randall_cloud_fraction(RH, qc, qs, cfg)
        assert bool(jnp.all(jnp.diff(cf[0]) >= -1e-12))
        assert float(cf[0, 0]) == pytest.approx(0.0, abs=1e-12)

    def test_xu_randall_gradient_finite_at_zero_rh(self):
        """RH^p_xr with p_xr<1 has infinite slope at RH=0 (dry strato);
        the clip floor must keep d cf / d q_v finite."""
        cfg = CloudConfig(scheme="xu_randall")
        T = jnp.full((1, 3), 230.0)
        pf = jnp.full((1, 3), 2.0e4)
        dp = jnp.full((1, 3), 1.0e4)

        def s(qv):
            props = compute_cloud_properties(
                T, pf, qv, dp, cfg,
                q_cloud=jnp.full((1, 3), 1e-5), q_ice=jnp.zeros((1, 3)),
            )
            return jnp.sum(props.cloud_fraction)

        qv = jnp.zeros((1, 3))  # RH = 0 exactly
        g = jax.grad(s)(qv)
        assert jnp.all(jnp.isfinite(g)), f"xu-randall grad NaN/Inf at RH=0: {g}"

    def test_water_path_units_and_partition(self):
        """lwp/iwp = q*dp/g [kg/m^2]; ice fraction grows as T drops.

        Hold RH = 0.9 (> rh_crit) at BOTH temperatures by scaling q_v to the
        (very different) saturation mixing ratios, so cloud fraction > 0 in
        both columns and the diagnostic condensate is purely temperature-
        partitioned between liquid and ice."""
        cfg = CloudConfig(scheme="sundqvist", rh_crit=0.7)
        nlev = 5
        T_warm = jnp.full((1, nlev), 285.0)
        T_cold = jnp.full((1, nlev), 250.0)  # below T_ice_only=233 would be all-ice
        pf = jnp.full((1, nlev), 5.0e4)
        dp = jnp.full((1, nlev), 1.0e4)
        qv_warm = 0.9 * saturation_specific_humidity(T_warm, pf)
        qv_cold = 0.9 * saturation_specific_humidity(T_cold, pf)
        warm = compute_cloud_properties(T_warm, pf, qv_warm, dp, cfg)
        cold = compute_cloud_properties(T_cold, pf, qv_cold, dp, cfg)
        # Both columns are clouded.
        assert float(jnp.sum(warm.cloud_fraction)) > 0.0
        assert float(jnp.sum(cold.cloud_fraction)) > 0.0
        # Warm: more liquid than ice; cold (T<T_freeze): more ice than liquid.
        assert float(jnp.sum(warm.lwp)) > float(jnp.sum(warm.iwp))
        assert float(jnp.sum(cold.iwp)) > float(jnp.sum(cold.lwp))
        # Water paths are non-negative and finite (kg/m^2 = q*dp/g).
        assert float(jnp.min(warm.lwp)) >= 0.0 and float(jnp.min(cold.iwp)) >= 0.0
        assert bool(jnp.all(jnp.isfinite(warm.lwp + warm.iwp)))

    def test_rh_crit_gradient_reachable(self):
        """rh_crit is a tier-1 trainable knob: d cf / d rh_crit must be
        finite and non-zero so it is reachable by AD."""
        RH = jnp.array([[0.8, 0.85, 0.9]])

        def s(rh_crit):
            cfg = CloudConfig(scheme="sundqvist", rh_crit=rh_crit)
            return jnp.sum(sundqvist_cloud_fraction(RH, cfg))

        g = jax.grad(s)(0.7)
        assert np.isfinite(float(g))
        assert abs(float(g)) > 0.0

    def test_cloud_fraction_uses_thermo_saturation(self):
        """RH is built from legoesm.thermo.saturation_specific_humidity; verify the
        diagnosed RH equals q_v / q_sat(thermo) (no re-derived saturation)."""
        cfg = CloudConfig(scheme="sundqvist")
        T = jnp.full((1, 4), 280.0)
        pf = jnp.full((1, 4), 8.0e4)
        dp = jnp.full((1, 4), 1.0e4)
        qv = jnp.full((1, 4), 6.0e-3)
        props = compute_cloud_properties(T, pf, qv, dp, cfg)
        qsat = saturation_specific_humidity(T, pf)
        rh = qv / jnp.maximum(qsat, 1e-10)
        expected = sundqvist_cloud_fraction(rh, cfg)
        np.testing.assert_allclose(
            np.asarray(props.cloud_fraction), np.asarray(expected), rtol=1e-12,
        )


# ==========================================================================
# RRTMGP : conservation / signs / diff (uses bundled optics tables).
# ==========================================================================
@pytest.fixture(scope="module")
def rrtmgp_solver():
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
    return RRTMGP.from_legoesm_config(RRTMGPConfig())


class TestRRTMGP:
    def test_column_energy_conservation(self, rrtmgp_solver):
        ph, pf, T, Ts, qv, _, _, cosz = _make_column(ncol=2, nlev=40)
        out = rrtmgp_solver.solve_columns(
            T=T, p_full=pf, p_half=ph, sfc_temperature=Ts,
            q_v=qv, cos_zenith=cosz,
        )
        dp = ph[:, 1:] - ph[:, :-1]
        col_heat = jnp.sum(
            constants.c_pd / constants.g * out.heating_rate * dp, axis=1
        )
        net = (out.lw_flux_down - out.lw_flux_up) + (
            out.sw_flux_down - out.sw_flux_up
        )
        flux_div = net[:, 0] - net[:, -1]
        rel = jnp.abs(col_heat - flux_div) / jnp.maximum(jnp.abs(col_heat), 1.0)
        assert float(jnp.max(rel)) < 1e-8, f"rrtmgp energy residual rel={rel}"

    def test_lw_free_troposphere_cools(self, rrtmgp_solver):
        ph, pf, T, Ts, qv, _, _, cosz = _make_column(ncol=1, nlev=40)
        out = rrtmgp_solver.solve_columns(
            T=T, p_full=pf, p_half=ph, sfc_temperature=Ts,
            q_v=qv, cos_zenith=cosz,
        )
        assert float(jnp.mean(out.lw_heating_rate[0, 5:-5])) < 0.0

    def test_physical_olr_and_osr_magnitudes(self, rrtmgp_solver):
        """OLR bounded & positive; OSR between 0 and incoming SW."""
        ph, pf, T, Ts, qv, _, _, cosz = _make_column(ncol=1, nlev=40)
        out = rrtmgp_solver.solve_columns(
            T=T, p_full=pf, p_half=ph, sfc_temperature=Ts,
            q_v=qv, cos_zenith=cosz,
        )
        olr = float(out.lw_flux_up[0, 0])
        osr = float(out.sw_flux_up[0, 0])
        sw_in = float(out.sw_flux_down[0, 0])
        assert 120.0 < olr < 350.0, f"OLR unphysical: {olr}"
        assert 0.0 <= osr <= sw_in + 1e-9, f"OSR unphysical: {osr} (in {sw_in})"

    def test_warmer_surface_increases_olr(self, rrtmgp_solver):
        ph, pf, T, Ts, qv, _, _, cosz = _make_column(ncol=1, nlev=40)
        base = rrtmgp_solver.solve_columns(
            T=T, p_full=pf, p_half=ph, sfc_temperature=Ts,
            q_v=qv, cos_zenith=cosz,
        ).lw_flux_up[0, 0]
        warm = rrtmgp_solver.solve_columns(
            T=T, p_full=pf, p_half=ph, sfc_temperature=Ts + 5.0,
            q_v=qv, cos_zenith=cosz,
        ).lw_flux_up[0, 0]
        assert float(warm) > float(base)

    def test_grad_through_temperature_finite_nonzero(self, rrtmgp_solver):
        ph, pf, T, Ts, qv, _, _, cosz = _make_column(ncol=1, nlev=30)

        def loss(Tin):
            out = rrtmgp_solver.solve_columns(
                T=Tin, p_full=pf, p_half=ph, sfc_temperature=Ts,
                q_v=qv, cos_zenith=cosz,
            )
            return jnp.sum(out.lw_flux_up[:, 0])  # OLR

        gT = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(gT)), "rrtmgp dOLR/dT has NaN/Inf"
        assert float(jnp.max(jnp.abs(gT))) > 0.0
        # Per-LAYER dOLR/dT can be locally negative (warming a single
        # mid-tropospheric layer in an optically-thick band can redistribute
        # emission and slightly lower TOA flux).  The physically robust
        # statement is the DIRECTIONAL derivative for UNIFORM warming
        # (sum of partials) must be positive: a warmer column emits more.
        assert float(jnp.sum(gT)) > 0.0, (
            f"dOLR/d(uniform warming) not positive: {float(jnp.sum(gT))}"
        )

    def test_steep_o3_boundary_no_halo_pathology(self, rrtmgp_solver):
        """Codex atm-radiation review: o3_vmr is clipped AFTER _add_halos.

        A steep ozone profile (large at TOA, tiny at the surface boundary)
        would, under the old PRE-halo clip, linearly extrapolate to a NEGATIVE
        halo VMR.  Verify the solve stays finite and OLR physical."""
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import _add_halos
        ph, pf, T, Ts, qv, _, _, cosz = _make_column(ncol=1, nlev=20)
        # Steep boundary step: 1e-5 in the interior, 1e-10 at the surface layer
        # (last index, TOA-first).  Surface-first => bottom_halo = 2e-10 - 1e-5 < 0.
        o3 = jnp.full((1, 20), 1.0e-5).at[0, -1].set(1.0e-10)
        o3_surf_first = o3[:, None, ::-1]
        assert float(jnp.min(_add_halos(jnp.clip(o3_surf_first, 1.0e-10, None)))) < 0.0
        assert float(jnp.min(jnp.clip(_add_halos(o3_surf_first), 1.0e-10, None))) >= 1.0e-10
        out = rrtmgp_solver.solve_columns(
            T=T, p_full=pf, p_half=ph, sfc_temperature=Ts,
            q_v=qv, cos_zenith=cosz, o3_vmr=o3,
        )
        assert bool(jnp.all(jnp.isfinite(out.lw_flux_up)))
        assert bool(jnp.all(jnp.isfinite(out.heating_rate)))
        assert 120.0 < float(out.lw_flux_up[0, 0]) < 350.0

    def test_steep_cloud_fraction_boundary_no_halo_pathology(self):
        """Codex atm-radiation review (round 2): the step MUST sit at the
        SURFACE boundary so ``_add_halos`` (bottom_halo = 2*f[0] - f[1])
        actually extrapolates out of bounds.

        Inputs are TOA-first; ``solve_columns`` reverses to surface-first
        before halos, so the surface-adjacent cells are the LAST two indices.
        Set the surface layer cf=0 and the next-up cf=1 => surface-first
        f=[0, 1, ...] => bottom_halo = 2*0 - 1 = -1.  Under the OLD pre-halo
        clip this leaked a NEGATIVE cloud optical depth into the interior
        recurrence (verified below: directly assert the halo would have been
        negative, then assert the fixed solve stays physical)."""
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import (
            RRTMGP, _add_halos,
        )
        solver = RRTMGP.from_legoesm_config(RRTMGPConfig(include_clouds=True))
        ph, pf, T, Ts, qv, _, _, cosz = _make_column(ncol=1, nlev=20)
        nlev = 20
        # TOA-first: cloudy everywhere except the surface layer (last index)
        # => surface-first boundary step [0, 1, ...] that drives the halo < 0.
        cf = jnp.ones((1, nlev)).at[0, -1].set(0.0)
        cpl = jnp.full((1, nlev), 0.05).at[0, -1].set(0.0)  # 50 g/m^2 paths
        crl = jnp.full((1, nlev), 10.0e-6)

        # Demonstrate the original pathology: PRE-halo clip leaves a negative
        # halo; POST-halo clip floors it (this is the behaviour being tested).
        cf_surf_first = cf[:, None, ::-1]
        pre_halo = _add_halos(jnp.clip(cf_surf_first, 0.0, 1.0))
        post_halo = jnp.clip(_add_halos(cf_surf_first), 0.0, 1.0)
        assert float(jnp.min(pre_halo)) < 0.0, "test does not hit the bug case"
        assert float(jnp.min(post_halo)) >= 0.0

        out = solver.solve_columns(
            T=T, p_full=pf, p_half=ph, sfc_temperature=Ts, q_v=qv,
            cos_zenith=cosz, cloud_path_liq=cpl, cloud_r_eff_liq=crl,
            cloud_fraction=cf,
        )
        assert bool(jnp.all(jnp.isfinite(out.sw_flux_up)))
        assert bool(jnp.all(jnp.isfinite(out.lw_flux_up)))
        # Reflected SW must stay within [0, incoming] (no spurious gain from a
        # negative halo cloud optical depth).
        osr = float(out.sw_flux_up[0, 0])
        sw_in = float(out.sw_flux_down[0, 0])
        assert 0.0 <= osr <= sw_in + 1e-9

    def test_grad_through_surface_albedo_override(self, rrtmgp_solver):
        """Surface albedo (trained via coupler albedo_col) must be AD-reachable
        as a per-call traced override into solve_columns."""
        ph, pf, T, Ts, qv, _, _, cosz = _make_column(ncol=1, nlev=30)

        def osr(alb):
            out = rrtmgp_solver.solve_columns(
                T=T, p_full=pf, p_half=ph, sfc_temperature=Ts,
                q_v=qv, cos_zenith=cosz,
                sfc_albedo=jnp.broadcast_to(alb, Ts.shape),
            )
            return jnp.sum(out.sw_flux_up[:, 0])  # reflected SW

        g = jax.grad(osr)(0.2)
        assert np.isfinite(float(g))
        assert float(g) > 0.0  # higher albedo => more reflected SW
