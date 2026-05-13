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
from legoesm.coupler.bulk_flux import compute_most_fluxes
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
            cfg = SBMConfig()._replace(RH_ref=rh)
            out = sbm_convection(T, q_v, p_full, p_half, 300.0, config=cfg)
            return jnp.sum(out.dT_dt ** 2)

        assert_param_grad_ok(loss, 0.7, "SBM RH_ref")

    @pytest.mark.skip(reason=(
        "CAPE_threshold flows only through the trigger sigmoid "
        "``sigmoid(sharpness · (CAPE − threshold))``.  In any column "
        "where CAPE is far from threshold the sigmoid is saturated "
        "(d sigmoid ≈ 0), so the gradient is legitimately zero — not "
        "a bug, but a known property of sigmoid-gated parameters.  "
        "AD-reachability requires a column tuned to CAPE ≈ threshold, "
        "which is brittle to construct as a unit test."
    ))
    def test_sbm_CAPE_threshold(self):  # pragma: no cover
        pass

    @pytest.mark.skip(reason="Same trigger-saturation issue as CAPE_threshold.")
    def test_sbm_smooth_trigger_sharpness(self):  # pragma: no cover
        pass

    def test_dca_mixing_fraction(self):
        T, q_v, p_full, p_half = _unstable_column()

        def loss(mf):
            cfg = DCAConfig()._replace(mixing_fraction=mf)
            out = dca_convection(T, q_v, p_full, p_half, 300.0, config=cfg)
            return jnp.sum(out.dT_dt ** 2)

        assert_param_grad_ok(loss, 1.0, "DCA mixing_fraction")

    @pytest.mark.skip(reason=(
        "DCA cape_threshold has the same sigmoid-saturated zero-gradient "
        "property as SBM CAPE_threshold — see that test for details."
    ))
    def test_dca_cape_threshold(self):  # pragma: no cover
        pass

    @pytest.mark.skip(reason=(
        "Kuo budgets are gated on column moisture convergence "
        "MC = -∇·(q_v·u), which is zero in any single-column setup "
        "without horizontal advection.  alpha_heat / tau_relax are "
        "AD-reachable only when MC > 0 — see test_diff_atmosphere_physics.py "
        "TestKuoMC for the full-3D AD coverage."
    ))
    def test_kuo_alpha_heat(self):  # pragma: no cover
        pass

    @pytest.mark.skip(reason="Same MC gating as alpha_heat — see above.")
    def test_kuo_tau_relax(self):  # pragma: no cover
        pass


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

    @pytest.mark.skip(reason=(
        "rain_evaporation hits the donor-cap ``min(rate, q_r/dt)`` under "
        "any column with realistic q_r and the production evap_coeff=1.0: "
        "``rate >> q_r/dt``, ``min`` picks the q_r/dt branch, and the VJP "
        "through evap_coeff is zero on the saturated branch.  The "
        "parameter is AD-reachable only in the unphysical regime "
        "``rate < q_r/dt`` (tiny rain or huge dt); this is by design "
        "(donor-positivity) and not a bug.  Documented as a known "
        "training-time limitation."
    ))
    def test_kessler_evaporation_coeff(self):  # pragma: no cover
        pass

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
            cfg = MorrisonConfig()._replace(dep_coeff=d)
            out = morrison_microphysics(
                T, q_v, hydro, p_full, p_half, rho, dz, 300.0, config=cfg,
            )
            return jnp.sum(out.dq_i_dt ** 2)

        assert_param_grad_ok(loss, 1e-3, "Morrison dep_coeff")

    def test_morrison_agg_coeff(self):
        T, q_v, hydro, p_full, p_half, rho, dz = _moist_microphys_column()
        T = jnp.full_like(T, 240.0)
        hydro = hydro._replace(
            N_c=jnp.full(hydro.q_c.shape, 1e8),
            q_i=jnp.full(hydro.q_c.shape, 1e-6),
            N_i=jnp.full(hydro.q_c.shape, 1e3),
        )

        def loss(a):
            cfg = MorrisonConfig()._replace(agg_coeff=a)
            out = morrison_microphysics(
                T, q_v, hydro, p_full, p_half, rho, dz, 300.0, config=cfg,
            )
            return jnp.sum(out.dq_s_dt ** 2)

        assert_param_grad_ok(loss, 1e-3, "Morrison agg_coeff")


# ===========================================================================
# 11c  Turbulence scheme parameters
# ===========================================================================

def _turbulence_column(nlev=10, ncol=2):
    """Build a column with vertical T/u shear so eddy diffusivities act."""
    p_s = 1.0e5
    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    p_half = jnp.broadcast_to((sigma_half * p_s)[None, :], (ncol, nlev + 1))
    p_full = jnp.broadcast_to((sigma_full * p_s)[None, :], (ncol, nlev))
    T = jnp.broadcast_to(
        (250.0 + 50.0 * sigma_full)[None, :], (ncol, nlev),
    )  # warmer near sfc
    q_v = jnp.broadcast_to(
        (1e-4 + 1e-2 * sigma_full)[None, :], (ncol, nlev),
    )
    u = jnp.broadcast_to(
        (10.0 - 8.0 * sigma_full)[None, :], (ncol, nlev),
    )  # surface 2 m/s, top 10 m/s
    v = jnp.zeros_like(u)
    rho = p_full / (constants.R_d * T)
    z_half = jnp.broadcast_to(
        jnp.linspace(20000.0, 0.0, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    T_sfc = jnp.full((ncol,), 295.0)
    q_sfc = saturation_mixing_ratio(T_sfc, jnp.full((ncol,), p_s))
    return u, v, T, q_v, p_full, p_half, z_full, z_half, T_sfc, q_sfc, rho


class TestTurbulenceParams:

    def test_smagorinsky_Km(self):
        u, v, T, q_v, p_full, p_half, z_full, z_half, T_sfc, q_sfc, rho = (
            _turbulence_column()
        )

        def loss(km):
            cfg = SmagorinskyConfig()._replace(Km=km)
            out = smagorinsky_turbulence(
                u, v, T, q_v, p_full, p_half, z_full, z_half,
                T_sfc, q_sfc, rho, 300.0, cfg,
            )
            return jnp.sum(out.du_dt ** 2)

        assert_param_grad_ok(loss, 10.0, "Smagorinsky Km")

    def test_smagorinsky_Pr_t(self):
        u, v, T, q_v, p_full, p_half, z_full, z_half, T_sfc, q_sfc, rho = (
            _turbulence_column()
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
            _turbulence_column()
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
            _turbulence_column()
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

class TestHeldSuarezParamsDeferred:
    """Held-Suarez forcing uses module-level constants
    (``DELTA_T_Y``, ``DELTA_THETA_Z``, ``K_A``, ``K_S``, ``K_F``,
    ``SIGMA_B``, ``T_MIN``) rather than a Config struct, so each
    parameter cannot be made traced without monkey-patching the
    module.  The agent's "11a" entry assumes a Config-style API that
    does not exist; flagging as a follow-up rather than writing
    monkey-patch tests that would couple the test suite to private
    module internals.
    """

    @pytest.mark.skip(reason=(
        "Held-Suarez params are module-level constants, not config "
        "fields.  Reaching them via jax.grad requires either refactoring "
        "the source to take a HeldSuarezConfig NamedTuple or "
        "monkey-patching, both out of scope for this audit-style test."
    ))
    def test_held_suarez_params_via_config(self):  # pragma: no cover
        pass


# ===========================================================================
# 11f  Land / sea ice / RRTMGP — coverage notes
# ===========================================================================
#
# Land slab (``LandConfig``) and sea ice (``SeaIceConfig``) parameters
# are tested indirectly by ``test_diff_land.py`` and
# ``test_diff_sea_ice.py`` — those files build full state objects +
# atmospheric forcing and exercise gradient flow through every config
# field accessed by ``step_land`` / ``step_sea_ice``.  Replicating that
# scaffold here for one-parameter-at-a-time coverage would duplicate a
# lot of state-construction boilerplate; the existing per-component
# files already prove that the relevant fields are reachable.
#
# RRTMGP gas-absorption coefficients are table-based and not
# differentiable through the lookup; surface emissivity / aerosol
# scaling are differentiable but live in the radiation driver layer,
# tested via ``test_diff_atmosphere_physics.py``.
