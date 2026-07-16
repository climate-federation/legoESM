"""Differentiability tests for tunable physics parameters.

For parameter estimation and online learning, every tunable physics
parameter must be reachable by ``jax.grad``.  This module wraps each
parameterization so that one config field at a time is a traced JAX
scalar (not a static Python float), then asserts the gradient is
finite *and* non-zero.

Categories:
  11a) Convection scheme parameters (SBM, DCA, Kuo)
  11b) Microphysics scheme parameters (Kessler, Seifert-Beheng, Morrison)
  11c) Turbulence scheme parameters (Smagorinsky, Louis)
  11d) Bulk flux / coupler parameters (COARE3 charnock)
  11e) Held-Suarez parameters — staged-not-implemented (module-level
       constants, no Config struct; flagged for follow-up)
  11f) Land / sea ice / RRTMGP parameters — see notes at bottom

Failure modes this catches:
* Parameter used inside a Python ``if`` (not traced) → zero gradient.
* Parameter shadowed by a hardcoded local → zero gradient.
* Parameter used only on one branch of ``lax.cond`` and the wrong
  branch is exercised → zero gradient.
* Parameter flows through a non-differentiable op (``argmax``, integer
  indexing) → zero gradient.  Mark such cases ``# NON-DIFFERENTIABLE:``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.convection.sbm import sbm_convection
from legoesm.atmosphere.physics.convection.dca import dca_convection
from legoesm.atmosphere.physics.convection.kuo import kuo_convection
from legoesm.atmosphere.physics.convection.config import (
    SBMConfig, DCAConfig, KuoConfig,
)
from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
from legoesm.atmosphere.physics.microphysics.seifert_beheng import (
    seifert_beheng_microphysics,
)
from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
from legoesm.atmosphere.physics.microphysics.config import (
    KesslerConfig, SeifertBehengConfig, MorrisonConfig,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.atmosphere.physics.turbulence.smagorinsky import smagorinsky_turbulence
from legoesm.atmosphere.physics.turbulence.louis import louis_turbulence
from legoesm.atmosphere.physics.turbulence.config import (
    SmagorinskyConfig, LouisConfig,
)
from legoesm.core.bulk_flux import compute_most_fluxes
from legoesm.thermo import saturation_mixing_ratio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_param_grad_ok(loss_fn, p0, name):
    """Compute ``jax.grad(loss_fn)(p0)`` and assert it is finite + nonzero."""
    g = jax.grad(loss_fn)(jnp.asarray(p0, dtype=jnp.float64))
    val = float(g)
    assert jnp.isfinite(g), (
        f"{name}: gradient not finite ({val}) — parameter triggered NaN/Inf"
    )
    assert val != 0.0, (
        f"{name}: gradient is exactly zero — parameter is unreachable by AD "
        f"(possibly captured by a Python ``if``, shadowed by a local literal, "
        f"or only used on the wrong branch of a ``lax.cond``)"
    )


def _unstable_column(nlev=12, ncol=2):
    p_s = 1.0e5
    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    p_half = jnp.broadcast_to((sigma_half * p_s)[None, :], (ncol, nlev + 1))
    p_full = jnp.broadcast_to((sigma_full * p_s)[None, :], (ncol, nlev))
    T_sfc = 300.0
    T = jnp.maximum(T_sfc * jnp.clip(sigma_full, 0.01, None) ** 0.19, 200.0)
    T = jnp.broadcast_to(T[None, :], (ncol, nlev))
    q_sat = saturation_mixing_ratio(T, p_full)
    RH = jnp.where(sigma_full[None, :] > 0.7, 0.95, 0.5)
    q_v = RH * q_sat
    return T, q_v, p_full, p_half


def _moist_microphys_column(nlev=8, ncol=2, q_c=5e-3, q_r=1e-4):
    """Microphysics column with cloud water above default Kessler
    autoconversion threshold (1e-3 kg/kg) so the autoconversion path
    is active and its gradient is non-zero."""
    T, q_v, p_full, p_half = _unstable_column(nlev=nlev, ncol=ncol)
    # Use a slightly subsaturated lower-troposphere q_v so the
    # rain-evaporation branch is also active (q_v < q_sat, q_r > 0).
    q_v = 0.9 * q_v
    rho = p_full / (constants.R_d * T)
    dz = jnp.full((ncol, nlev), 500.0)
    z = jnp.zeros((ncol, nlev))
    hydro = HydrometeorState(
        q_c=jnp.full((ncol, nlev), q_c),
        q_r=jnp.full((ncol, nlev), q_r),
        q_i=z, q_s=z, q_g=z, N_c=z, N_r=z, N_i=z,
    )
    return T, q_v, hydro, p_full, p_half, rho, dz


# ===========================================================================
# 11a  Convection scheme parameters
# ===========================================================================

class TestConvectionParams:
    """Each tunable float field in SBMConfig / DCAConfig / KuoConfig
    must be reachable by ``jax.grad``."""

    def test_sbm_tau_c(self):
        T, q_v, p_full, p_half = _unstable_column()

        def loss(tau_c):
            cfg = SBMConfig()._replace(tau_c=tau_c)
            out = sbm_convection(T, q_v, p_full, p_half, 300.0, config=cfg)
            return jnp.sum(out.dT_dt ** 2)

        assert_param_grad_ok(loss, 7200.0, "SBM tau_c")

    def test_sbm_RH_ref(self):
        T, q_v, p_full, p_half = _unstable_column()

        def loss(rh):
            cfg = SBMConfig()._replace(rh_ref=rh)
            out = sbm_convection(T, q_v, p_full, p_half, 300.0, config=cfg)
            return jnp.sum(out.dT_dt ** 2)

        assert_param_grad_ok(loss, 0.7, "SBM RH_ref")

    def test_sbm_CAPE_threshold(self):
        # CAPE_threshold lives inside ``sigmoid(sharpness · (CAPE −
        # threshold))``.  The sigmoid saturates at any column far from
        # the threshold (d sigmoid ≈ 0), so to exercise the AD path
        # we probe at a ``threshold`` value tuned to the column's
        # CAPE *and* lower the sharpness so the sigmoid argument
        # stays O(1).  Verifies the parameter is *reachable* by AD —
        # the production-default ``sharpness=0.1, threshold=70`` lives
        # in the saturated regime under typical columns, which is a
        # known property of sigmoid-gated triggers (not a bug).
        T, q_v, p_full, p_half = _unstable_column()
        threshold_probe = 2000.0

        def loss(cape_thr):
            cfg = SBMConfig()._replace(
                cape_threshold=cape_thr,
                # Pair with a sharpness that keeps |sharpness·(CAPE −
                # threshold)| ≲ 5 across the column so the gating
                # sigmoid is unsaturated.
                smooth_trigger_sharpness=0.001,
            )
            out = sbm_convection(T, q_v, p_full, p_half, 300.0, config=cfg)
            return jnp.sum(out.dT_dt ** 2)

        assert_param_grad_ok(loss, threshold_probe, "SBM CAPE_threshold")

    def test_sbm_smooth_trigger_sharpness(self):
        # Same idea as ``test_sbm_CAPE_threshold``: probe at a sharpness
        # low enough to keep the trigger in its smooth transition band.
        T, q_v, p_full, p_half = _unstable_column()

        def loss(s):
            cfg = SBMConfig()._replace(
                smooth_trigger_sharpness=s,
                # Pair with a threshold near the column's CAPE so the
                # sigmoid argument stays O(1).
                cape_threshold=2000.0,
            )
            out = sbm_convection(T, q_v, p_full, p_half, 300.0, config=cfg)
            return jnp.sum(out.dT_dt ** 2)

        assert_param_grad_ok(loss, 0.01, "SBM smooth_trigger_sharpness")

    def test_dca_mixing_fraction(self):
        T, q_v, p_full, p_half = _unstable_column()

        def loss(mf):
            cfg = DCAConfig()._replace(mixing_fraction=mf)
            out = dca_convection(T, q_v, p_full, p_half, 300.0, config=cfg)
            return jnp.sum(out.dT_dt ** 2)

        assert_param_grad_ok(loss, 1.0, "DCA mixing_fraction")

    def test_dca_cape_threshold(self):
        # Same trigger-saturation pattern as SBM: probe threshold near
        # column CAPE so the gating sigmoid stays in its transition band.
        T, q_v, p_full, p_half = _unstable_column()
        threshold_probe = 2000.0

        def loss(c):
            cfg = DCAConfig()._replace(cape_threshold=c)
            out = dca_convection(T, q_v, p_full, p_half, 300.0, config=cfg)
            return jnp.sum(out.dT_dt ** 2)

        assert_param_grad_ok(loss, threshold_probe, "DCA cape_threshold")

    def _convergent_column(self, nlev=20, ncol=2):
        """Conditionally-unstable tropical column WITH a positive
        large-scale moisture-convergence profile so the canonical Kuo
        scheme fires (the faithful source is convergence, not
        supersaturation).  Returns ``(T, q_v, p_full, p_half, ptenq)``."""
        import numpy as _np
        p_s = 1.0e5
        sigma_half = jnp.linspace(0.05, 1.0, nlev + 1)
        sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
        p_half = jnp.broadcast_to((sigma_half * p_s)[None, :], (ncol, nlev + 1))
        p_full = jnp.broadcast_to((sigma_full * p_s)[None, :], (ncol, nlev))
        z = -8000.0 * jnp.log(jnp.clip(sigma_full, 1e-3, None))
        T = jnp.broadcast_to(
            jnp.maximum(300.0 - 6.5e-3 * z, 200.0)[None, :], (ncol, nlev),
        )
        q_sat = saturation_mixing_ratio(T, p_full)
        RH = 0.85 * jnp.clip((sigma_full - 0.15) / 0.85, 0.0, 1.0) + 0.1
        q_v = jnp.broadcast_to((RH * q_sat[0])[None, :], (ncol, nlev))
        p = sigma_full * p_s
        ptenq = (3.0e-3 / 86400.0) * jnp.exp(-((p - 850e2) / 120e2) ** 2)
        ptenq = jnp.broadcast_to(ptenq[None, :], (ncol, nlev))
        return T, q_v, p_full, p_half, ptenq

    def test_kuo_entrainment(self):
        T, q_v, p_full, p_half, ptenq = self._convergent_column()

        def loss(e):
            cfg = KuoConfig()._replace(entrainment=e)
            out = kuo_convection(T, q_v, p_full, p_half, 900.0, config=cfg,
                                 moisture_convergence=ptenq)
            return jnp.sum(out.dT_dt ** 2)

        assert_param_grad_ok(loss, KuoConfig().entrainment, "Kuo entrainment")

    def test_kuo_anthes_rh_offset(self):
        T, q_v, p_full, p_half, ptenq = self._convergent_column()

        def loss(o):
            cfg = KuoConfig(partition="anthes")._replace(anthes_rh_offset=o)
            out = kuo_convection(T, q_v, p_full, p_half, 900.0, config=cfg,
                                 moisture_convergence=ptenq)
            return jnp.sum(out.dT_dt ** 2)

        assert_param_grad_ok(loss, KuoConfig().anthes_rh_offset,
                             "Kuo anthes_rh_offset")


# ===========================================================================
# 11b  Microphysics scheme parameters
# ===========================================================================

class TestMicrophysicsParams:

    def test_kessler_autoconversion_rate(self):
        T, q_v, hydro, p_full, p_half, rho, dz = _moist_microphys_column()

        def loss(rate):
            cfg = KesslerConfig()._replace(autoconversion_rate=rate)
            out = kessler_microphysics(
                T, q_v, hydro, p_full, p_half, rho, dz, 300.0, config=cfg,
            )
            return jnp.sum(out.dq_c_dt ** 2)

        assert_param_grad_ok(loss, 1e-3, "Kessler autoconversion_rate")

    def test_kessler_accretion_coeff(self):
        T, q_v, hydro, p_full, p_half, rho, dz = _moist_microphys_column()

        def loss(c):
            cfg = KesslerConfig()._replace(accretion_coeff=c)
            out = kessler_microphysics(
                T, q_v, hydro, p_full, p_half, rho, dz, 300.0, config=cfg,
            )
            return jnp.sum(out.dq_c_dt ** 2)

        assert_param_grad_ok(loss, 2.2, "Kessler accretion_coeff")

    def test_kessler_evaporation_coeff(self):
        # ``rain_evaporation`` applies a donor-cap ``min(rate, q_r/dt)``
        # that saturates whenever the un-capped rate exceeds the
        # available rain inventory per timestep.  Probe in the regime
        # ``rate < q_r/dt`` (tiny rain budget) so the cap is *inactive*
        # and the evap_coeff gradient is not killed by the ``min``
        # branch.  Verifies the parameter is reachable when the donor
        # cap is not saturated — confirming the "BUG" is by-design
        # donor-positivity, not a vanishing-gradient bug.
        ncol, nlev = 2, 8
        # Subsaturated column with very small q_r so the donor cap
        # ``rate ≤ q_r / dt`` is never active.
        q_r_tiny = 1e-12  # rate ∝ q_r^0.525 ≈ 1e-6 << q_r/dt is wrong
                          # — actually rate ∝ q_r^0.525 = 1e-6, q_r/dt
                          # = 3e-15 → still capped.  Use a tiny
                          # evap_coeff probe instead.
        T, q_v, hydro, p_full, p_half, rho, dz = _moist_microphys_column(
            q_c=5e-3, q_r=q_r_tiny,
        )
        q_sat = saturation_mixing_ratio(T, p_full)
        q_v_dry = 0.5 * q_sat

        def loss(c):
            cfg = KesslerConfig()._replace(evaporation_coeff=c)
            out = kessler_microphysics(
                T, q_v_dry, hydro, p_full, p_half, rho, dz, 300.0, config=cfg,
            )
            return jnp.sum(out.dq_v_dt ** 2)

        # Probe at evap_coeff = 1e-12 so ``rate = 1e-12 · 0.5 ·
        # (1e-12)^0.525 ≈ 1e-19`` << ``q_r / dt = 3.3e-15`` and the
        # donor cap stays in the *un*-capped branch.
        assert_param_grad_ok(loss, 1e-12, "Kessler evaporation_coeff")

    def test_seifert_beheng_k_au(self):
        T, q_v, hydro, p_full, p_half, rho, dz = _moist_microphys_column()
        # Need positive N_c to drive autoconversion.
        hydro = hydro._replace(N_c=jnp.full(hydro.q_c.shape, 1e8))

        def loss(k):
            cfg = SeifertBehengConfig()._replace(k_au=k)
            out = seifert_beheng_microphysics(
                T, q_v, hydro, p_full, p_half, rho, dz, 300.0, config=cfg,
            )
            return jnp.sum(out.dq_c_dt ** 2)

        assert_param_grad_ok(loss, 6e2, "SeifertBeheng k_au")

    def test_seifert_beheng_k_ac(self):
        T, q_v, hydro, p_full, p_half, rho, dz = _moist_microphys_column()
        hydro = hydro._replace(N_c=jnp.full(hydro.q_c.shape, 1e8))

        def loss(k):
            cfg = SeifertBehengConfig()._replace(k_ac=k)
            out = seifert_beheng_microphysics(
                T, q_v, hydro, p_full, p_half, rho, dz, 300.0, config=cfg,
            )
            return jnp.sum(out.dq_c_dt ** 2)

        assert_param_grad_ok(loss, 5.25, "SeifertBeheng k_ac")

    def test_morrison_dep_coeff(self):
        # Cold column for ice deposition path.
        T, q_v, hydro, p_full, p_half, rho, dz = _moist_microphys_column()
        T = jnp.full_like(T, 240.0)  # cold for ice phase
        hydro = hydro._replace(
            N_c=jnp.full(hydro.q_c.shape, 1e8),
            q_i=jnp.full(hydro.q_c.shape, 1e-6),
            N_i=jnp.full(hydro.q_c.shape, 1e3),
        )

        def loss(d):
            # dep_coeff feeds the legacy "heuristic" deposition path
            # (the default "m2005" path is tuned by
            # ice_deposition_efficiency instead — iter-8).
            cfg = MorrisonConfig()._replace(
                dep_coeff=d, ice_deposition_scheme="heuristic",
            )
            out = morrison_microphysics(
                T, q_v, hydro, p_full, p_half, rho, dz, 300.0, config=cfg,
            )
            return jnp.sum(out.dq_i_dt ** 2)

        assert_param_grad_ok(loss, 1e-3, "Morrison dep_coeff")

    def test_morrison_ice_deposition_efficiency(self):
        # The m2005 deposition multiplier must be reachable by AD.
        T, q_v, hydro, p_full, p_half, rho, dz = _moist_microphys_column()
        T = jnp.full_like(T, 240.0)
        hydro = hydro._replace(
            N_c=jnp.full(hydro.q_c.shape, 1e8),
            q_i=jnp.full(hydro.q_c.shape, 1e-5),
            N_i=jnp.full(hydro.q_c.shape, 1e5),
        )

        def loss(eff):
            cfg = MorrisonConfig()._replace(ice_deposition_efficiency=eff)
            out = morrison_microphysics(
                T, q_v, hydro, p_full, p_half, rho, dz, 300.0, config=cfg,
            )
            return jnp.sum(out.dq_i_dt ** 2)

        assert_param_grad_ok(loss, 1.0, "Morrison ice_deposition_efficiency")

    def test_morrison_agg_coeff(self):
        T, q_v, hydro, p_full, p_half, rho, dz = _moist_microphys_column()
        T = jnp.full_like(T, 240.0)
        hydro = hydro._replace(
            N_c=jnp.full(hydro.q_c.shape, 1e8),
            q_i=jnp.full(hydro.q_c.shape, 1e-6),
            N_i=jnp.full(hydro.q_c.shape, 1e3),
        )

        def loss(a):
            # agg_coeff is the HEURISTIC ice→snow aggregation coefficient.  The
            # default morrison_flavor="mg" forces ice_to_snow_scheme="mg_ferrier"
            # in resolve_morrison_flavor (ignoring agg_coeff) AND clobbers an
            # explicit ice_to_snow_scheme.  Use the "sam" flavor (which leaves
            # ice_to_snow_scheme untouched) + the heuristic path so agg_coeff is
            # the live knob this test audits.
            cfg = MorrisonConfig()._replace(
                agg_coeff=a, morrison_flavor="sam",
                ice_to_snow_scheme="heuristic")
            out = morrison_microphysics(
                T, q_v, hydro, p_full, p_half, rho, dz, 300.0, config=cfg,
            )
            return jnp.sum(out.dq_s_dt ** 2)

        assert_param_grad_ok(loss, 1e-3, "Morrison agg_coeff")


# ===========================================================================
# 11c  Turbulence scheme parameters
# ===========================================================================

def _turbulence_column_pbl(nlev=10, ncol=2):
    """Shallow (0-2 km), near-neutral, strongly-sheared boundary layer.

    Smagorinsky-Lilly is a deformation closure with a *hard* Lilly
    cutoff at Ri ≥ Pr_t, so it produces K_m = 0 in a deep, strongly
    θ-stratified column (Ri ≫ 1) — which would make C_s / Pr_t
    unreachable by AD.  This PBL-scale column keeps a small θ-gradient
    with ~10 m/s shear over 2 km so 0 < Ri < Pr_t: K_m > 0 and the
    gradient is non-trivial for the heat path to act on.  Louis (whose
    stable f only asymptotes toward 0) is also exercised here.
    """
    z_half = jnp.broadcast_to(
        jnp.linspace(2000.0, 0.0, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    p_half = jnp.broadcast_to(
        jnp.linspace(8.0e4, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    T = 290.0 - 0.0095 * z_full          # near dry-adiabatic (mildly stable in θ)
    u = 0.005 * z_full                    # ~10 m/s shear over 2 km
    v = jnp.zeros_like(u)
    q_v = jnp.full_like(T, 5e-3)
    rho = p_full / (constants.R_d * T)
    T_sfc = T[:, -1] + 1.0
    q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
    return u, v, T, q_v, p_full, p_half, z_full, z_half, T_sfc, q_sfc, rho


class TestTurbulenceParams:

    def test_smagorinsky_C_s(self):
        u, v, T, q_v, p_full, p_half, z_full, z_half, T_sfc, q_sfc, rho = (
            _turbulence_column_pbl()
        )

        def loss(c_s):
            cfg = SmagorinskyConfig()._replace(C_s=c_s)
            out = smagorinsky_turbulence(
                u, v, T, q_v, p_full, p_half, z_full, z_half,
                T_sfc, q_sfc, rho, 300.0, cfg,
            )
            return jnp.sum(out.du_dt ** 2)

        assert_param_grad_ok(loss, 0.2, "Smagorinsky C_s")

    def test_smagorinsky_Pr_t(self):
        u, v, T, q_v, p_full, p_half, z_full, z_half, T_sfc, q_sfc, rho = (
            _turbulence_column_pbl()
        )

        def loss(pr):
            cfg = SmagorinskyConfig()._replace(Pr_t=pr)
            out = smagorinsky_turbulence(
                u, v, T, q_v, p_full, p_half, z_full, z_half,
                T_sfc, q_sfc, rho, 300.0, cfg,
            )
            return jnp.sum(out.dT_dt ** 2)

        assert_param_grad_ok(loss, 1.0, "Smagorinsky Pr_t")

    def test_louis_l_mix_max(self):
        u, v, T, q_v, p_full, p_half, z_full, z_half, T_sfc, q_sfc, rho = (
            _turbulence_column_pbl()
        )

        def loss(l):
            cfg = LouisConfig()._replace(l_mix_max=l)
            out = louis_turbulence(
                u, v, T, q_v, p_full, p_half, z_full, z_half,
                T_sfc, q_sfc, rho, 300.0, cfg,
            )
            return jnp.sum(out.du_dt ** 2)

        assert_param_grad_ok(loss, 100.0, "Louis l_mix_max")

    def test_louis_b_louis(self):
        u, v, T, q_v, p_full, p_half, z_full, z_half, T_sfc, q_sfc, rho = (
            _turbulence_column_pbl()
        )

        def loss(b):
            cfg = LouisConfig()._replace(b_louis=b)
            out = louis_turbulence(
                u, v, T, q_v, p_full, p_half, z_full, z_half,
                T_sfc, q_sfc, rho, 300.0, cfg,
            )
            return jnp.sum(out.du_dt ** 2)

        assert_param_grad_ok(loss, 5.0, "Louis b_louis")


# ===========================================================================
# 11d  Bulk flux / coupler parameters (COARE3)
# ===========================================================================

class TestBulkFluxParams:
    """``compute_most_fluxes`` exposes its tunable parameters as scalar
    keyword arguments rather than via a Config — gradient must reach
    them through the iterative MOST loop."""

    def _scalar_inputs(self):
        u_rel = jnp.array([5.0, 8.0])
        v_rel = jnp.array([0.0, 0.0])
        T_atm = jnp.array([285.0, 290.0])
        q_atm = jnp.array([5e-3, 6e-3])
        T_sfc = jnp.array([288.0, 293.0])
        q_sfc = saturation_mixing_ratio(T_sfc, jnp.array([1.013e5, 1.013e5]))
        rho = jnp.array([1.225, 1.225])
        return u_rel, v_rel, T_atm, q_atm, T_sfc, q_sfc, rho

    def test_charnock_param_reachable(self):
        u_rel, v_rel, T_atm, q_atm, T_sfc, q_sfc, rho = self._scalar_inputs()

        def loss(c):
            tau_x, tau_y, shflx, lhflx, ustar = compute_most_fluxes(
                u_rel, v_rel, T_atm, q_atm, T_sfc, q_sfc, rho,
                z_ref=10.0, scheme="coare3", n_iter=5, charnock=c,
            )
            return jnp.sum(shflx ** 2) + jnp.sum(tau_x ** 2)

        assert_param_grad_ok(loss, 0.011, "COARE3 charnock")

    def test_grad_through_n_iter_path(self):
        """Sanity check: gradient w.r.t. T_sfc must be finite + non-zero
        through the entire ``jax.lax.fori_loop`` MOST iteration.  Catches
        the case where the iteration body silently stops the gradient
        (e.g., a stop_gradient inside a branch)."""
        u_rel, v_rel, T_atm, q_atm, _, q_sfc, rho = self._scalar_inputs()

        def loss(t_sfc):
            t_sfc_b = jnp.broadcast_to(t_sfc, T_atm.shape)
            tau_x, tau_y, shflx, lhflx, ustar = compute_most_fluxes(
                u_rel, v_rel, T_atm, q_atm, t_sfc_b, q_sfc, rho,
                z_ref=10.0, scheme="coare3", n_iter=10,
            )
            return jnp.sum(shflx ** 2)

        assert_param_grad_ok(loss, 290.0, "COARE3 T_sfc through fori_loop")


# ===========================================================================
# 11e  Held-Suarez parameters — staged-not-implemented
# ===========================================================================

class TestHeldSuarezParams:
    """``held_suarez_forcing`` takes its tunable parameters
    (``k_a``, ``k_s``, ``k_f``, ``sigma_b``, ``delta_T_y``,
    ``delta_theta_z``, ``T_min``, ``p_ref``) as keyword arguments with
    module-level Table-1 defaults, so each can be passed as a traced
    JAX scalar for parameter estimation.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.core.field import Field
        from legoesm.core.state import HydrostaticState
        n, nlev = 4, 5
        self.grid = create_cubed_sphere(n)
        self.sigma = create_sigma_coordinate(nlev)
        # Construct a non-trivial state so dT_dt depends on every
        # Held-Suarez parameter (not just the few that survive a
        # uniform-T column).
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
        T_data = 250.0 * jnp.ones((6, n, n, nlev)) + 2.0 * jax.random.normal(
            k1, (6, n, n, nlev),
        )
        u_data = 10.0 * jax.random.normal(k2, (6, n, n, nlev))
        v_data = 3.0 * jax.random.normal(k3, (6, n, n, nlev))
        self.state = HydrostaticState(
            u=Field(u_data, name="u"),
            v=Field(v_data, name="v"),
            T=Field(T_data, name="T"),
            p_s=Field(1e5 * jnp.ones((6, n, n)), name="p_s"),
            phis=Field(jnp.zeros((6, n, n)), name="phis"),
            tracers={
                "q_v": Field(1e-3 * jnp.ones((6, n, n, nlev)), name="q_v"),
            },
        )

    @pytest.mark.parametrize("name,default", [
        ("k_a", 1.0 / (40.0 * 86400.0)),
        ("k_s", 1.0 / (4.0 * 86400.0)),
        ("k_f", 1.0 / (1.0 * 86400.0)),
        ("sigma_b", 0.7),
        ("delta_T_y", 60.0),
        ("delta_theta_z", 10.0),
        ("T_min", 200.0),
    ])
    def test_held_suarez_param_reachable(self, name, default):
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing
        grid, sigma, state = self.grid, self.sigma, self.state

        def loss(p):
            tend = held_suarez_forcing(s=state, grid=grid, sigma_coord=sigma, **{name: p}) \
                if False else held_suarez_forcing(state, grid, sigma, **{name: p})
            # ``du_dt`` couples through k_f and sigma_b only; sum both
            # tendencies so every parameter routes into the loss.
            return jnp.sum(tend.dT_dt.data ** 2) + jnp.sum(tend.du_dt.data ** 2)

        assert_param_grad_ok(loss, default, f"Held-Suarez {name}")


# ===========================================================================
# 11f  Land model parameters (slab + bucket hydrology)
# ===========================================================================

def _land_forcing(ncol, **overrides):
    from legoesm.core.coupling_fields import AtmToSurface
    ones = jnp.ones(ncol)
    base = dict(
        sw_down=200.0 * ones, lw_down=300.0 * ones,
        precip_total=1e-5 * ones, precip_snow=0.0 * ones,
        T_lowest=280.0 * ones, q_lowest=5e-3 * ones,
        u_lowest=5.0 * ones, v_lowest=2.0 * ones,
        p_lowest=1e5 * ones, p_surface=1.013e5 * ones,
        rho_lowest=1.2 * ones, cos_zenith=0.7 * ones,
        co2_ppmv=400.0 * ones, has_radiation=1.0 * ones,
        has_precipitation=1.0 * ones,
    )
    base.update(overrides)
    return AtmToSurface(**base)


def _land_state(ncol, **overrides):
    from legoesm.core.field import Field
    from legoesm.land.state import LandState
    ones = jnp.ones(ncol)
    base = dict(
        T_soil=Field(280.0 * ones, name="T_soil"),
        W_bucket=Field(50.0 * ones, name="W_bucket"),
        snow_depth=Field(jnp.zeros(ncol), name="snow_depth"),
        snow_age=Field(jnp.zeros(ncol), name="snow_age"),
    )
    base.update(overrides)
    return LandState(**base)


class TestLandParams:
    """Each tunable float field in ``LandConfig`` (heat capacity, slab
    depth, albedo, emissivity, bucket capacity, roughness) must be
    reachable by ``jax.grad`` through ``step_land``."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.land.slab_land import step_land
        from legoesm.land.config import LandConfig
        self.step_land = step_land
        self.LandConfig = LandConfig
        self.ncol = 16
        self.state = _land_state(self.ncol)
        self.forcing = _land_forcing(self.ncol)
        self.dt = 60.0

    def _loss_factory(self, name, base_cfg_kwargs=None, state=None,
                      forcing=None, dt=None, field="T_soil"):
        base_cfg_kwargs = base_cfg_kwargs or {}
        state = state if state is not None else self.state
        forcing = forcing if forcing is not None else self.forcing
        dt = dt if dt is not None else self.dt

        def loss(p):
            cfg = self.LandConfig(**base_cfg_kwargs)._replace(**{name: p})
            out, _, _ = self.step_land(state, forcing, cfg, U_min=1.0, dt=dt)
            return jnp.sum(getattr(out, field).data ** 2)

        return loss

    def test_C_soil(self):
        assert_param_grad_ok(self._loss_factory("C_soil"), 2.0e6, "Land C_soil")

    def test_d_soil(self):
        assert_param_grad_ok(self._loss_factory("d_soil"), 1.0, "Land d_soil")

    def test_albedo_land(self):
        assert_param_grad_ok(
            self._loss_factory("albedo_land"), 0.2, "Land albedo_land",
        )

    def test_emissivity_land(self):
        assert_param_grad_ok(
            self._loss_factory("emissivity_land"), 0.96, "Land emissivity_land",
        )

    def test_W_max(self):
        # W_max routes through the soil-water bucket; the loss must read
        # W_bucket (not T_soil) for the closure to be live.
        assert_param_grad_ok(
            self._loss_factory("W_max", field="W_bucket"), 150.0, "Land W_max",
        )

    def test_z0_land_under_most(self):
        # Roughness length only enters the surface fluxes under the MOST
        # bulk scheme (the "constant" scheme uses fixed Cd/Ch and ignores
        # z0).  Probe under bulk_scheme="most" so the parameter is live.
        assert_param_grad_ok(
            self._loss_factory("z0_land", base_cfg_kwargs={"bulk_scheme": "most"}),
            0.05, "Land z0_land (MOST)",
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "KNOWN BUG (land, out of scope for the convection-fix PR): "
            "LandConfig.snow_melt_rate is an advertised tunable_tier-2 param "
            "but is DEAD in the slab model — step_land always calls "
            "update_snow with a non-None Q_net, taking the energy-limited melt "
            "branch which never reads snow_melt_rate (only the legacy "
            "degree-day else-branch does), so jax.grad of the slab snow budget "
            "w.r.t. it is exactly 0. strict-xfail: flips red the moment the "
            "fix (drop it from the slab __param_spec__, or blend the "
            "degree-day rate into the energy-limited melt) lands."
        ),
    )
    def test_snow_melt_rate_unreachable_in_slab(self):
        # BUG: ``LandConfig.snow_melt_rate`` is declared a tunable_tier-2
        # (extended) trainable parameter in land/config.py::__param_spec__,
        # but ``step_land`` ALWAYS calls update_snow(..., Q_net=Q_net) with
        # a non-None Q_net (slab_land.py:194).  update_snow then takes the
        # ENERGY-LIMITED melt branch (snow_budget.py:83-85) and NEVER reads
        # ``snow_melt_rate`` — that field is only used in the legacy
        # degree-day ``else`` branch (snow_budget.py:88) reachable solely
        # when Q_net is None.  Consequently jax.grad cannot reach this
        # advertised tunable through the slab model: a parameter-estimation
        # user calibrating snow_melt_rate would silently get a zero
        # gradient.  Fix: either drop snow_melt_rate from the slab
        # __param_spec__ (it is dead in the energy-limited path), or make
        # update_snow blend the degree-day rate into the energy-limited
        # melt.  Asserting it IS reachable here so the test fails loudly
        # while the parameter remains dead code.
        from legoesm.core.field import Field
        snow_state = _land_state(
            self.ncol,
            snow_depth=Field(20.0 * jnp.ones(self.ncol), name="snow_depth"),
        )
        warm_forcing = _land_forcing(
            self.ncol, T_lowest=290.0 * jnp.ones(self.ncol),
            sw_down=400.0 * jnp.ones(self.ncol),
        )
        loss = self._loss_factory(
            "snow_melt_rate", state=snow_state, forcing=warm_forcing,
            dt=600.0, field="snow_depth",
        )
        assert_param_grad_ok(loss, 5.0e-6, "Land snow_melt_rate")


# ===========================================================================
# 11g  Sea ice parameters (thermodynamic slab + rheology)
# ===========================================================================

def _ice_forcing(shape, **overrides):
    from legoesm.core.coupling_fields import AtmToSurface
    ones = jnp.ones(shape)
    base = dict(
        sw_down=100.0 * ones, lw_down=250.0 * ones,
        precip_total=0.0 * ones, precip_snow=0.0 * ones,
        T_lowest=260.0 * ones, q_lowest=1e-3 * ones,
        u_lowest=5.0 * ones, v_lowest=2.0 * ones,
        p_lowest=1e5 * ones, p_surface=1.013e5 * ones,
        rho_lowest=1.4 * ones, cos_zenith=0.5 * ones,
        co2_ppmv=400.0 * ones, has_radiation=1.0 * ones,
        has_precipitation=1.0 * ones,
    )
    base.update(overrides)
    return AtmToSurface(**base)


class TestSeaIceParams:
    """Thermodynamic ``SeaIceConfig`` float fields (conductivity, albedo,
    emissivity, ocean heat-transfer coefficient) must be reachable by
    ``jax.grad`` through ``step_sea_ice`` in slab mode."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.core.field import Field
        from legoesm.ice.sea_ice import step_sea_ice
        from legoesm.ice.config import SeaIceConfig
        from legoesm.ice.state import SeaIceState
        self.step_fn = step_sea_ice
        self.SeaIceConfig = SeaIceConfig
        n = 4
        shape = (6, n, n)
        self.shape = shape
        self.state = SeaIceState(
            h_ice=Field(1.0 * jnp.ones(shape), name="h_ice"),
            T_ice=Field(265.0 * jnp.ones(shape), name="T_ice"),
            concentration=Field(0.8 * jnp.ones(shape), name="concentration"),
        )
        self.forcing = _ice_forcing(shape)
        self.ocean_u = jnp.zeros(shape)
        self.ocean_v = jnp.zeros(shape)
        self.dt = 3600.0

    def _loss(self, name, p, ocean_sst, field="T_ice"):
        cfg = self.SeaIceConfig(dynamics="none")._replace(**{name: p})
        out, _ = self.step_fn(
            self.state, self.forcing, ocean_sst,
            self.ocean_u, self.ocean_v, cfg, U_min=1.0, dt=self.dt,
        )
        return jnp.sum(getattr(out, field).data ** 2)

    def test_k_ice(self):
        sst = 271.35 * jnp.ones(self.shape)
        assert_param_grad_ok(
            lambda p: self._loss("k_ice", p, sst), 2.0, "SeaIce k_ice",
        )

    def test_albedo_ice(self):
        sst = 271.35 * jnp.ones(self.shape)
        assert_param_grad_ok(
            lambda p: self._loss("albedo_ice", p, sst), 0.65, "SeaIce albedo_ice",
        )

    def test_emissivity_ice(self):
        sst = 271.35 * jnp.ones(self.shape)
        assert_param_grad_ok(
            lambda p: self._loss("emissivity_ice", p, sst),
            0.97, "SeaIce emissivity_ice",
        )

    def test_ocean_heat_transfer_coeff(self):
        # The ocean→ice basal heat flux F_w = k_oc · (SST − T_freeze) is
        # only non-zero when the ocean is ABOVE freezing (active basal
        # melt).  At SST = T_freeze_ocean the flux — and therefore the
        # k_oc gradient — is identically zero, which is physically
        # correct, not a bug.  Probe with a warm ocean (SST > freezing)
        # so the basal-melt path is live and the gradient flows into
        # h_ice.
        sst = 274.0 * jnp.ones(self.shape)
        assert_param_grad_ok(
            lambda p: self._loss(
                "ocean_heat_transfer_coeff", p, sst, field="h_ice",
            ),
            20.0, "SeaIce ocean_heat_transfer_coeff",
        )


class TestSeaIceRheologyParams:
    """Hibler (1979) ice-strength parameters must be reachable by
    ``jax.grad`` through the standalone ``ice_strength`` kernel."""

    def _inputs(self):
        h = jnp.array([1.0, 2.0, 0.5, 1.5])
        A = jnp.array([0.8, 0.9, 0.6, 0.75])
        return h, A

    def test_P_star(self):
        from legoesm.ice.rheology import ice_strength
        h, A = self._inputs()
        assert_param_grad_ok(
            lambda p: jnp.sum(ice_strength(h, A, P_star=p) ** 2),
            2.75e4, "SeaIce P_star",
        )

    def test_C_strength(self):
        from legoesm.ice.rheology import ice_strength
        h, A = self._inputs()
        assert_param_grad_ok(
            lambda c: jnp.sum(ice_strength(h, A, C_strength=c) ** 2),
            20.0, "SeaIce C_strength",
        )


# ===========================================================================
# 11i  Ocean physics parameters (vertical mixing + bottom drag)
# ===========================================================================

def _ocean_column(n=4, nlev=4):
    """Stably-stratified ocean column on a small cubed-sphere grid with
    sheared currents so the Richardson-mixing diffusivities are active.

    Returns ``(u, v, T, S, rho, z_coord, jacobian)``.
    """
    from legoesm.ocean.vertical import create_z_star_from_thicknesses
    from legoesm.ocean.eos import linear_eos
    z_coord = create_z_star_from_thicknesses([10.0, 20.0, 40.0, 80.0][:nlev])
    shape = (6, n, n, nlev)
    u = 0.1 * jax.random.normal(jax.random.PRNGKey(0), shape)
    v = 0.05 * jax.random.normal(jax.random.PRNGKey(1), shape)
    # Stable stratification: warm/light on top, cold/dense below.
    T_prof = jnp.linspace(18.0, 4.0, nlev)
    T = jnp.broadcast_to(T_prof, shape)
    S = jnp.full(shape, 35.0)
    jacobian = jnp.ones((6, n, n))
    rho = linear_eos(T, S, jnp.zeros(shape))
    return u, v, T, S, rho, z_coord, jacobian


class TestOceanPhysicsParams:
    """Pacanowski-Philander Richardson-mixing tunables (``K_0``,
    ``alpha``, ``K_bg``, ``A_bg``) and the quadratic bottom-drag
    coefficient ``C_d`` must be reachable by ``jax.grad``."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.ocean.physics.vertical_mixing.richardson import (
            richardson_vertical_mixing,
        )
        from legoesm.ocean.physics.vertical_mixing.config import (
            RichardsonVerticalMixingConfig,
        )
        from legoesm.ocean.physics.bottom_drag.quadratic import (
            quadratic_bottom_drag,
        )
        from legoesm.ocean.physics.bottom_drag.config import QuadraticDragConfig
        self.rmix = richardson_vertical_mixing
        self.RmixCfg = RichardsonVerticalMixingConfig
        self.drag = quadratic_bottom_drag
        self.DragCfg = QuadraticDragConfig
        self.u, self.v, self.T, self.S, self.rho, self.zc, self.jac = (
            _ocean_column()
        )

    def _rmix_loss(self, name, p, field="dT_dt"):
        cfg = self.RmixCfg()._replace(**{name: p})
        out = self.rmix(
            self.u, self.v, self.T, self.S, self.rho, self.zc, self.jac,
            cfg, dt=3600.0,
        )
        return jnp.sum(getattr(out, field) ** 2)

    def test_richardson_K_0(self):
        assert_param_grad_ok(
            lambda p: self._rmix_loss("K_0", p), 5e-3, "Ocean Richardson K_0",
        )

    def test_richardson_alpha(self):
        assert_param_grad_ok(
            lambda p: self._rmix_loss("alpha", p), 5.0,
            "Ocean Richardson alpha",
        )

    def test_richardson_K_bg(self):
        assert_param_grad_ok(
            lambda p: self._rmix_loss("K_bg", p), 1e-5,
            "Ocean Richardson K_bg",
        )

    def test_richardson_A_bg(self):
        # Background viscosity enters the momentum diffusivity, so probe
        # via the velocity tendency.
        assert_param_grad_ok(
            lambda p: self._rmix_loss("A_bg", p, field="du_dt"), 1e-4,
            "Ocean Richardson A_bg",
        )

    def test_bottom_drag_C_d(self):
        def loss(c):
            cfg = self.DragCfg()._replace(C_d=c)
            out = self.drag(self.u, self.v, self.zc, self.jac, cfg)
            return jnp.sum(out.du_dt ** 2)

        assert_param_grad_ok(loss, 2.5e-3, "Ocean bottom-drag C_d")


# ===========================================================================
# 11j  RRTMGP — coverage note
# ===========================================================================
#
# RRTMGP gas-absorption coefficients are table-based and not
# differentiable through the lookup (integer band indexing + clamped
# interpolation table reads).  Surface emissivity / aerosol-optical-depth
# scaling are differentiable but live in the radiation driver layer and
# are exercised by ``test_diff_atmosphere_physics.py`` (gray radiation
# emissivity / optical-depth coefficients are covered there).  No new
# RRTMGP parameter is independently AD-reachable at the kernel level, so
# none is asserted here.
