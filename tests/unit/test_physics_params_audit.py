"""Differentiability audit: every physics leaf scheme is callable and at least
one representative tunable float reaches ``jax.grad`` (finite + nonzero).

Companion to ``test_diff_physics_params.py`` (which covers SBM/DCA/Kuo,
Kessler/SeifertBeheng/Morrison, Smagorinsky/Louis, COARE3).  This module
extends coverage to the remaining atmosphere convection / microphysics /
turbulence / gravity-wave-drag / radiation / cloud schemes, plus the ocean
physics schemes reached through the production ``make_ocean_physics`` entry.

Each parameter is probed under a column/state that *activates its process*
(generic columns leave sigmoid-gated triggers saturated or donor-limiters
binding, which legitimately zero the gradient — see the per-scheme comments in
``test_diff_physics_params.py``).  The schemes flagged by the 2026-06
differentiability audit are regression-guarded here:

* ocean ``plume`` — ``epsilon`` / ``T_excess`` / ``active_sigmoid_sharpness``
  feed the ``lax.scan`` carry; a float64-traced param used to promote the
  float32 carry mid-scan and crash.  Fixed by pinning the carry to the state
  dtype.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio, saturation_mixing_ratio_ice

jax.config.update("jax_enable_x64", True)


def assert_grad_ok(loss_fn, p0, name):
    g = jax.grad(loss_fn)(jnp.asarray(p0, dtype=jnp.float64))
    val = float(g)
    assert jnp.isfinite(g), f"{name}: gradient not finite ({val})"
    assert val != 0.0, (
        f"{name}: gradient is exactly zero — tunable parameter unreachable "
        f"by AD (Python if / shadowed local / wrong lax.cond branch)"
    )


# ===========================================================================
# Atmosphere column helpers
# ===========================================================================
_NCOL, _NLEV = 2, 12


def _column():
    p_s = 1.0e5
    sh = jnp.linspace(0.0, 1.0, _NLEV + 1)
    sf = 0.5 * (sh[:-1] + sh[1:])
    p_half = jnp.broadcast_to((sh * p_s)[None, :], (_NCOL, _NLEV + 1))
    p_full = jnp.broadcast_to((sf * p_s)[None, :], (_NCOL, _NLEV))
    T = jnp.maximum(300.0 * jnp.clip(sf, 0.01, None) ** 0.19, 200.0)
    T = jnp.broadcast_to(T[None, :], (_NCOL, _NLEV))
    q_sat = saturation_mixing_ratio(T, p_full)
    q_v = jnp.where(sf[None, :] > 0.7, 0.95, 0.5) * q_sat
    return T, q_v, p_full, p_half


def _aux():
    T, q_v, p_full, p_half = _column()
    rho = p_full / (constants.R_d * T)
    dz = jnp.full((_NCOL, _NLEV), 500.0)
    z_full = jnp.broadcast_to(
        jnp.cumsum(dz[0][::-1])[::-1][None, :], (_NCOL, _NLEV))
    z_half = jnp.concatenate([z_full + 250.0, z_full[:, -1:] - 250.0], axis=1)
    u = jnp.broadcast_to(jnp.linspace(10.0, 2.0, _NLEV)[None, :], (_NCOL, _NLEV))
    v = jnp.full((_NCOL, _NLEV), 1.0)
    w = jnp.full((_NCOL, _NLEV), 0.1)
    lat = jnp.full((_NCOL,), 0.5)
    T_sfc = jnp.full((_NCOL,), 301.0)
    q_sfc = saturation_mixing_ratio(T_sfc, p_half[:, -1])
    prog = jnp.zeros((_NCOL, _NLEV))
    return locals()


# ===========================================================================
# Convection
# ===========================================================================
class TestConvectionAudit:
    def test_bechtold_epsilon_deep(self):
        from legoesm.atmosphere.physics.convection.bechtold import bechtold_convection
        from legoesm.atmosphere.physics.convection.config import BechtoldConfig
        T, q_v, p_full, p_half = _column()
        A = _aux()

        def loss(x):
            cfg = BechtoldConfig()._replace(epsilon_deep=x)
            out = bechtold_convection(T, q_v, p_full, p_half, A["u"], A["v"],
                A["prog"], jnp.zeros((_NCOL,)), jax.random.PRNGKey(0), 600.0,
                config=cfg)[0]
            return jnp.sum(out.dT_dt ** 2)
        assert_grad_ok(loss, 1.75e-3, "bechtold.epsilon_deep")

    def test_emanuel_cu_coefficient(self):
        from legoesm.atmosphere.physics.convection.emanuel import emanuel_convection
        from legoesm.atmosphere.physics.convection.config import EmanuelConfig
        # ``cu_coefficient`` is the sub-cloud buoyancy-sort multiplier that
        # scales detrainment (``delta_0 * sort_multiplier``) ONLY in the
        # mass-flux-kernel branch.  The default ``use_genuine_mixing=True``
        # path computes detrainment via the buoyancy-sorted mixer and never
        # reads ``cu_coefficient`` (its gradient is then exactly zero — the
        # parameter is simply inactive in that mode, not a wiring bug).
        # Audit AD-reachability in the mode where the knob is live, on a
        # destabilised column so the buoyancy sort yields a nonzero
        # negatively-buoyant share for ``cu_coefficient`` to act on.
        T, q_v, p_full, p_half = _column()
        T = T.at[:, -1].add(20.0)  # hot, convecting boundary layer

        def loss(x):
            cfg = EmanuelConfig()._replace(
                use_genuine_mixing=False, cu_coefficient=x,
            )
            out = emanuel_convection(T, q_v, p_full, p_half,
                jnp.zeros((_NCOL, _NLEV)), 600.0, config=cfg)[0]
            return jnp.sum(out.dT_dt ** 2)
        assert_grad_ok(loss, 0.7, "emanuel.cu_coefficient")

    def test_kain_fritsch_cape_consumption_time(self):
        from legoesm.atmosphere.physics.convection.kain_fritsch import kain_fritsch_convection
        from legoesm.atmosphere.physics.convection.config import KainFritschConfig
        # ``cape_consumption_time`` (TIMEC) sets the closure cloud-base mass
        # flux ``M_b = rho_BL * CAPE / (g * TIMEC)``.  In a very high-CAPE
        # column ``M_b`` saturates at the literature cap ``M_b_max`` (clip),
        # which makes the *applied* mass flux — and hence ``dT_dt`` —
        # TIMEC-independent (gradient exactly zero).  The default ``_column``
        # is past that cap, so audit reachability on a moderate-CAPE column
        # (stabilised + slightly dried) where the closure is below the cap
        # and TIMEC genuinely flows into the tendency.
        T, q_v, p_full, p_half = _column()
        T = T.at[:, -1].add(-2.0)   # gentler boundary-layer instability
        q_v = q_v * 0.85            # below the M_b_max saturation regime
        A = _aux()

        def loss(x):
            cfg = KainFritschConfig()._replace(cape_consumption_time=x)
            out = kain_fritsch_convection(T, q_v, p_full, p_half, A["w"],
                A["prog"], 600.0, config=cfg)[0]
            return jnp.sum(out.dT_dt ** 2)
        assert_grad_ok(loss, 1800.0, "kain_fritsch.cape_consumption_time")

    def test_tiedtke_epsilon_deep(self):
        from legoesm.atmosphere.physics.convection.tiedtke import tiedtke_convection
        from legoesm.atmosphere.physics.convection.config import TiedtkeConfig
        T, q_v, p_full, p_half = _column()
        A = _aux()

        def loss(x):
            cfg = TiedtkeConfig()._replace(epsilon_deep=x)
            out = tiedtke_convection(T, q_v, p_full, p_half, A["u"], A["v"],
                A["prog"], 600.0, config=cfg)[0]
            return jnp.sum(out.dT_dt ** 2)
        assert_grad_ok(loss, 1.0e-4, "tiedtke.epsilon_deep")

    def test_zhang_mcfarlane_tau_cape(self):
        from legoesm.atmosphere.physics.convection.zhang_mcfarlane import zhang_mcfarlane_convection
        from legoesm.atmosphere.physics.convection.config import ZhangMcFarlaneConfig
        T, q_v, p_full, p_half = _column()
        A = _aux()

        def loss(x):
            cfg = ZhangMcFarlaneConfig()._replace(tau_cape=x)
            out = zhang_mcfarlane_convection(T, q_v, p_full, p_half, A["u"],
                A["v"], A["prog"], 600.0, config=cfg)[0]
            return jnp.sum(out.dT_dt ** 2)
        assert_grad_ok(loss, 3600.0, "zhang_mcfarlane.tau_cape")

    def test_mass_flux_M_scale(self):
        from legoesm.atmosphere.physics.convection.mass_flux import mass_flux_convection
        from legoesm.atmosphere.physics.convection.config import MassFluxConfig
        T, q_v, p_full, p_half = _column()

        def loss(x):
            cfg = MassFluxConfig()._replace(M_scale=x)
            out = mass_flux_convection(T, q_v, p_full, p_half,
                jnp.zeros((_NCOL,)), 600.0, config=cfg)[0]
            return jnp.sum(out.dT_dt ** 2)
        assert_grad_ok(loss, 0.01, "mass_flux.M_scale")

    def test_edmf_convection_a_u_init(self):
        from legoesm.atmosphere.physics.convection.mass_flux import edmf_convection
        from legoesm.atmosphere.physics.convection.config import ConvectiveEDMFConfig
        T, q_v, p_full, p_half = _column()

        def loss(x):
            cfg = ConvectiveEDMFConfig()._replace(a_u_init=x)
            out = edmf_convection(T, q_v, p_full, p_half,
                jnp.full((_NCOL,), 0.1), 600.0, config=cfg)[0]
            return jnp.sum(out.dT_dt ** 2)
        assert_grad_ok(loss, 0.1, "edmf_convection.a_u_init")

    def test_bechtold_tiedtke_no_dead_precip_efficiency(self):
        """``precip_efficiency`` was a dead tunable (declared, never used) and
        an inert AIMIP training knob — removed in the 2026-06 audit.  Guard
        against re-introduction."""
        from legoesm.atmosphere.physics.convection.config import (
            BechtoldConfig, TiedtkeConfig)
        assert not hasattr(BechtoldConfig(), "precip_efficiency")
        assert not hasattr(TiedtkeConfig(), "precip_efficiency")
        # Also guard the AIMIP trainable set against re-introduction of the
        # inert knob (codex review: defaults-preservation test alone would
        # not catch a re-add).
        from legoesm.training.aimip_params import AIMIPClassicalParams
        names = set(AIMIPClassicalParams.from_defaults().as_dict())
        assert "tiedtke_precip_efficiency" not in names


class TestAIMIPDefaultsInteriorization:
    """``from_defaults`` interiorizes edge-of-range knobs for trainability (v7).

    A canonical default sitting at a sigmoid bound maps to a saturated raw
    value whose initial gradient is ~0, freezing the knob.  ``from_defaults``
    nudges such knobs ``sigmoid_margin`` (5%) inside the bound; mid-range knobs
    stay exactly at their canonical default.  Regression guard: ``default_clamped``
    was computed but not passed to ``range_to_sigmoid``, so edge knobs
    initialized saturated/untrainable (and a default run silently used a
    near-bound emissivity with a frozen gradient).
    """

    def test_edge_knob_interiorized_and_trainable(self):
        from legoesm.training.aimip_params import AIMIPClassicalParams
        params = AIMIPClassicalParams.from_defaults()
        vals = params.as_dict()
        # gray_sfc_emissivity: canonical 1.0 in [0.5, 1.0] -> margin 0.025 -> 0.975
        assert abs(float(vals["gray_sfc_emissivity"]) - 0.975) < 1e-5
        # rrtmgp_sfc_emissivity: canonical 0.98 within margin of 1.0 -> 0.975
        assert abs(float(vals["rrtmgp_sfc_emissivity"]) - 0.975) < 1e-5
        # raw is far from saturation -> non-trivial inverse-sigmoid gradient.
        # logit(0.95) ~= 2.94; a saturated edge default would give ~6.9.
        raw = float(params.raw_values["gray_sfc_emissivity"])
        assert abs(raw) < 4.0

    def test_midrange_knob_exact_canonical(self):
        from legoesm.training.aimip_params import (
            AIMIPClassicalParams, _canonical_scheme_defaults)
        defaults = _canonical_scheme_defaults()
        vals = AIMIPClassicalParams.from_defaults().as_dict()
        # tiedtke_cape_threshold canonical 70 in [10, 500] is mid-range:
        # the clamp is a no-op and init must equal the canonical default.
        name = "tiedtke_cape_threshold"
        assert abs(float(vals[name]) - defaults[name]) < 1e-4 * defaults[name]

    def test_to_rrtmgp_config_freezes_raw_hashed_cache_key_fields(self):
        """RRTMGP folds gas concentrations + aerosol_ssa/aerosol_g into its
        solver-cache key as RAW (non-_hashable) tuple elements, so a traced
        trainable leaf there is unhashable under jax.grad. to_rrtmgp_config()
        must freeze ALL of them to the RRTMGPConfig defaults (codex review
        2026-06-13: the gas-only freeze missed aerosol). Only sfc_emissivity/
        sfc_albedo (routed through _hashable + the per-call spatial override)
        may carry trained values."""
        from legoesm.training.aimip_params import AIMIPClassicalParams
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        base = RRTMGPConfig()
        cfg = AIMIPClassicalParams.from_defaults().to_rrtmgp_config()
        for field in ("co2_ppmv", "ch4_ppbv", "n2o_ppbv", "aerosol_ssa", "aerosol_g"):
            assert float(getattr(cfg, field)) == float(getattr(base, field)), (
                f"RRTMGPConfig.{field} is a raw cache-key tuple element and must "
                f"stay frozen at the default {getattr(base, field)}, got {getattr(cfg, field)}"
            )

    def test_instance_cache_key_hashable_under_traced_surface_scalar(self):
        """sfc_emissivity / sfc_albedo ARE wired into RRTMGPConfig from trained
        AIMIP leaves (and routed through _hashable). _instance_cache_key must
        therefore produce a HASHABLE key even when those fields are traced 0-D
        scalars under jax.grad -- the _hashable float(x) coercion is guarded to
        fall back to id() instead of raising ConcretizationTypeError. Regression
        for the codex r6 RRTMGP fix."""
        import jax
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        def f(e):
            cfg = RRTMGPConfig()._replace(sfc_emissivity=e, sfc_albedo=e * 0.5)
            hash(RRTMGP._instance_cache_key(cfg))  # must not raise
            return e * 1.0

        jax.grad(f)(jnp.asarray(0.95))  # raises if the cache key crashes on a tracer


# ===========================================================================
# Microphysics
# ===========================================================================
def _ice_column():
    """Cold, ice-bearing, mixed-phase column that activates warm + ice
    process rates (autoconversion, accretion, deposition, melting)."""
    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
    p_s = 1.0e5
    sh = jnp.linspace(0.0, 1.0, _NLEV + 1)
    sf = 0.5 * (sh[:-1] + sh[1:])
    p_full = jnp.broadcast_to((sf * p_s)[None, :], (_NCOL, _NLEV))
    p_half = jnp.broadcast_to((sh * p_s)[None, :], (_NCOL, _NLEV + 1))
    T = jnp.broadcast_to(jnp.linspace(258.0, 268.0, _NLEV)[None, :], (_NCOL, _NLEV))
    q_si = saturation_mixing_ratio_ice(T, p_full)
    q_v = 1.05 * q_si
    rho = p_full / (constants.R_d * T)
    dz = jnp.full((_NCOL, _NLEV), 500.0)
    z = jnp.zeros((_NCOL, _NLEV))
    hydro = HydrometeorState(
        q_c=jnp.full((_NCOL, _NLEV), 2e-3), q_r=jnp.full((_NCOL, _NLEV), 2e-4),
        q_i=jnp.full((_NCOL, _NLEV), 3e-4), q_s=jnp.full((_NCOL, _NLEV), 1e-4),
        q_g=z, N_c=jnp.full((_NCOL, _NLEV), 1e8),
        N_r=jnp.full((_NCOL, _NLEV), 1e3), N_i=jnp.full((_NCOL, _NLEV), 1e4))
    return T, q_v, hydro, p_full, p_half, rho, dz


class TestMicrophysicsAudit:
    def test_thompson_k_ac(self):
        from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
        from legoesm.atmosphere.physics.microphysics.config import ThompsonConfig
        T, q_v, hydro, p_full, p_half, rho, dz = _ice_column()

        def loss(x):
            cfg = ThompsonConfig()._replace(k_ac=x)
            out = thompson_microphysics(T, q_v, hydro, p_full, p_half, rho, dz,
                60.0, config=cfg)
            return jnp.sum(out.dq_r_dt ** 2)
        assert_grad_ok(loss, 5.25, "thompson.k_ac")

    def test_p3_dep_coeff(self):
        from legoesm.atmosphere.physics.microphysics.p3 import p3_microphysics
        from legoesm.atmosphere.physics.microphysics.config import P3Config
        T, q_v, hydro, p_full, p_half, rho, dz = _ice_column()

        def loss(x):
            cfg = P3Config()._replace(dep_coeff=x)
            out = p3_microphysics(T, q_v, hydro, p_full, p_half, rho, dz,
                60.0, config=cfg)
            return jnp.sum(out.dT_dt ** 2) + jnp.sum(out.dq_i_dt ** 2)
        assert_grad_ok(loss, 1e-3, "p3.dep_coeff")

    def test_sundqvist_RH_crit(self):
        from legoesm.atmosphere.physics.microphysics.sundqvist import sundqvist_microphysics
        from legoesm.atmosphere.physics.microphysics.config import SundqvistConfig
        from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
        T, q_v, p_full, p_half = _column()
        rho = p_full / (constants.R_d * T)
        dz = jnp.full((_NCOL, _NLEV), 500.0)
        z = jnp.zeros((_NCOL, _NLEV))
        hydro = HydrometeorState(q_c=jnp.full((_NCOL, _NLEV), 3e-3),
            q_r=z, q_i=z, q_s=z, q_g=z, N_c=z, N_r=z, N_i=z)

        def loss(x):
            cfg = SundqvistConfig()._replace(rh_crit=x)
            out = sundqvist_microphysics(T, 0.9 * q_v, hydro, p_full, p_half,
                rho, dz, 60.0, config=cfg)
            return jnp.sum(out.dT_dt ** 2)
        assert_grad_ok(loss, 0.8, "sundqvist.RH_crit")


# ===========================================================================
# Turbulence
# ===========================================================================
def _turb_call(fn, cfg, second):
    A = _aux()
    T, q_v, p_full, p_half = _column()
    tke0 = jnp.full((_NCOL, _NLEV), 0.5)
    args = (A["u"], A["v"], T, q_v)
    if second:
        args = args + (tke0,)
    args = args + (p_full, p_half, A["z_full"], A["z_half"], A["T_sfc"],
                   A["q_sfc"], A["rho"], 60.0)
    r = fn(*args, cfg)
    out = r if hasattr(r, "du_dt") else r[0]
    return jnp.sum(out.du_dt ** 2 + out.Kh ** 2)


class TestTurbulenceAudit:
    def test_tke_Ck(self):
        from legoesm.atmosphere.physics.turbulence.tke import tke_turbulence
        from legoesm.atmosphere.physics.turbulence.config import TKEConfig
        assert_grad_ok(lambda x: _turb_call(tke_turbulence,
            TKEConfig()._replace(Ck=x), True), 0.1, "tke.Ck")

    def test_mynn25_A1(self):
        from legoesm.atmosphere.physics.turbulence.mynn25 import mynn25_turbulence
        from legoesm.atmosphere.physics.turbulence.config import MYNN25Config
        assert_grad_ok(lambda x: _turb_call(mynn25_turbulence,
            MYNN25Config()._replace(A1=x), True), 1.18, "mynn25.A1")

    def test_clubb_lite_C_K(self):
        from legoesm.atmosphere.physics.turbulence.clubb_lite import clubb_lite_turbulence
        from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
        assert_grad_ok(lambda x: _turb_call(clubb_lite_turbulence,
            CLUBBLiteConfig()._replace(C_K=x), True), 0.4, "clubb_lite.C_K")

    def test_holtslag_boville_fakn(self):
        # The faithful HB scheme controls the nonlocal countergradient
        # strength via ``fakn`` (fak3 = fakn*wstar/wm); verify grad flow.
        from legoesm.atmosphere.physics.turbulence.holtslag_boville import holtslag_boville_turbulence
        from legoesm.atmosphere.physics.turbulence.config import HoltslagBovilleConfig
        assert_grad_ok(lambda x: _turb_call(holtslag_boville_turbulence,
            HoltslagBovilleConfig()._replace(fakn=x), False), 7.2,
            "holtslag_boville.fakn")

    def test_ysu_entrainment_coeff(self):
        from legoesm.atmosphere.physics.turbulence.ysu import ysu_turbulence
        from legoesm.atmosphere.physics.turbulence.config import YSUConfig
        assert_grad_ok(lambda x: _turb_call(ysu_turbulence,
            YSUConfig()._replace(entrainment_coeff=x), False), 0.2,
            "ysu.entrainment_coeff")

    def test_edmf_turbulence_Ck(self):
        from legoesm.atmosphere.physics.turbulence.edmf import edmf_turbulence
        from legoesm.atmosphere.physics.turbulence.config import TurbulentEDMFConfig
        assert_grad_ok(lambda x: _turb_call(edmf_turbulence,
            TurbulentEDMFConfig()._replace(Ck=x), True), 0.1, "turb_edmf.Ck")


# ===========================================================================
# Gravity-wave drag + radiation + clouds
# ===========================================================================
def _gwd_call(fn, cfg, extra=()):
    A = _aux()
    T, q_v, p_full, p_half = _column()
    out = fn(A["u"], A["v"], T, p_full, p_half, A["z_full"], A["z_half"],
             A["rho"], A["lat"], 60.0, cfg, *extra)
    # GWDOutput is itself a NamedTuple (==tuple); only unwrap the
    # (output, spectrum) 2-tuple, detected by absence of du_dt.
    out = out if hasattr(out, "du_dt") else out[0]
    return jnp.sum(out.du_dt ** 2)


class TestGWDRadiationCloudsAudit:
    def test_rayleigh_k_max(self):
        from legoesm.atmosphere.physics.gravity_wave_drag.rayleigh import rayleigh_gwd
        from legoesm.atmosphere.physics.gravity_wave_drag.config import RayleighConfig
        assert_grad_ok(lambda x: _gwd_call(rayleigh_gwd,
            RayleighConfig()._replace(k_max=x)), 1.0 / 86400.0, "rayleigh.k_max")

    def test_lindzen_h_topo(self):
        from legoesm.atmosphere.physics.gravity_wave_drag.lindzen import lindzen_gwd
        from legoesm.atmosphere.physics.gravity_wave_drag.config import LindzenConfig
        assert_grad_ok(lambda x: _gwd_call(lindzen_gwd,
            LindzenConfig()._replace(h_topo=x)), 500.0, "lindzen.h_topo")

    def test_mcfarlane_G_0(self):
        from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd
        from legoesm.atmosphere.physics.gravity_wave_drag.config import McFarlaneConfig
        assert_grad_ok(lambda x: _gwd_call(mcfarlane_gwd,
            McFarlaneConfig()._replace(G_0=x)), 0.5, "mcfarlane.G_0")

    def test_hines_total_rms_wind(self):
        from legoesm.atmosphere.physics.gravity_wave_drag.hines import hines_gwd
        from legoesm.atmosphere.physics.gravity_wave_drag.config import HinesConfig
        assert_grad_ok(lambda x: _gwd_call(hines_gwd,
            HinesConfig()._replace(total_rms_wind=x)), 2.0, "hines.total_rms_wind")

    def test_prognostic_spectral_launch_flux(self):
        from legoesm.atmosphere.physics.gravity_wave_drag.prognostic_spectral import prognostic_spectral_gwd
        from legoesm.atmosphere.physics.gravity_wave_drag.config import PrognosticSpectralConfig
        A = _aux()
        T, q_v, p_full, p_half = _column()

        def loss(x):
            cfg = PrognosticSpectralConfig()._replace(launch_flux=x)
            spec = jnp.full((_NCOL, cfg.n_azimuths, cfg.n_wavenumbers), 1e-4)
            out, snew = prognostic_spectral_gwd(A["u"], A["v"], T, p_full,
                p_half, A["z_full"], A["z_half"], A["rho"], A["lat"], 60.0,
                cfg, spec)
            return jnp.sum(out.du_dt ** 2) + jnp.sum(snew ** 2)
        assert_grad_ok(loss, 1e-3, "prognostic_spectral.launch_flux")

    def test_gray_tau_equator(self):
        from legoesm.atmosphere.physics.radiation.gray import gray_radiation
        from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
        A = _aux()
        T, q_v, p_full, p_half = _column()

        def loss(x):
            cfg = GrayRadiationConfig()._replace(tau_equator=x)
            out = gray_radiation(T, p_full, p_half, A["T_sfc"], A["lat"], q_v,
                jnp.full((_NCOL,), 400.0), cfg)
            return jnp.sum(out.heating_rate ** 2)
        assert_grad_ok(loss, 7.2, "gray.tau_equator")

    def test_clouds_rh_crit(self):
        from legoesm.atmosphere.physics.clouds.cloud_fraction import compute_cloud_properties
        from legoesm.atmosphere.physics.clouds.config import CloudConfig
        T, q_v, p_full, p_half = _column()
        dp = p_half[:, 1:] - p_half[:, :-1]

        def loss(x):
            cfg = CloudConfig()._replace(rh_crit=x, scheme="sundqvist")
            out = compute_cloud_properties(T, p_full, q_v, dp, cfg)
            return jnp.sum(out.cloud_fraction ** 2)
        assert_grad_ok(loss, 0.7, "clouds.rh_crit")


# ===========================================================================
# Ocean physics (through production make_ocean_physics)
# ===========================================================================
@pytest.fixture(scope="module")
def ocean_setup():
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init import rest_state_ocean
    from legoesm.ocean.state import OceanSurfaceForcing
    n, nlev = 4, 12
    grid = create_cubed_sphere(n)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    state = rest_state_ocean(grid, z_coord, T_water_init_C=18.0, T_deep=2.0,
                             S_uniform=35.0)
    rng = np.random.default_rng(0)
    noise = jnp.asarray(rng.normal(size=state.T.data.shape),
                        state.T.data.dtype) * 0.8
    state = state._replace(T=state.T.replace(data=state.T.data + noise))
    # depth-varying u: vertical shear so momentum mixing/viscosity is active
    prof = jnp.linspace(0.5, -0.3, nlev).astype(state.u.data.dtype)
    state = state._replace(u=state.u.replace(data=state.u.data + prof[None, None, None, :]))
    forcing = OceanSurfaceForcing(sw_down=jnp.full((6, n, n), 200.0))
    return state, grid, z_coord, forcing


def _ocean_base():
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
    return OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None)


def _ocean_loss(cfg, setup):
    from legoesm.ocean.physics.combined import make_ocean_physics
    state, grid, z_coord, forcing = setup
    out = make_ocean_physics(cfg)(state, grid, z_coord, forcing)
    s = 0.0
    for f in ("du_dt", "dv_dt", "dT_dt", "dS_dt"):
        v = getattr(out, f, None)
        if v is not None:
            d = v.data if hasattr(v, "data") else v
            s = s + jnp.sum(d ** 2)
    return s


class TestOceanPhysicsAudit:
    def test_vmix_kpp_K_max(self, ocean_setup):
        from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig, KPPConfig

        def loss(x):
            cfg = _ocean_base()._replace(vertical_mixing=VerticalMixingConfig(
                scheme="kpp", kpp=KPPConfig(K_max=x)))
            return _ocean_loss(cfg, ocean_setup)
        assert_grad_ok(loss, 0.05, "ocean.kpp.K_max")

    def test_vmix_constant_A_v(self, ocean_setup):
        from legoesm.ocean.physics.vertical_mixing.config import (
            VerticalMixingConfig, ConstantVerticalMixingConfig)

        def loss(x):
            cfg = _ocean_base()._replace(vertical_mixing=VerticalMixingConfig(
                scheme="constant", constant=ConstantVerticalMixingConfig(A_v=x)))
            return _ocean_loss(cfg, ocean_setup)
        assert_grad_ok(loss, 1e-3, "ocean.constant.A_v")

    def test_lateral_gm_redi_kappa_GM(self, ocean_setup):
        from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig, GMRediConfig

        def loss(x):
            cfg = _ocean_base()._replace(lateral_mixing=LateralMixingConfig(
                scheme="gm_redi", gm_redi=GMRediConfig(kappa_GM=x)))
            return _ocean_loss(cfg, ocean_setup)
        assert_grad_ok(loss, 1e3, "ocean.gm_redi.kappa_GM")

    def test_lateral_harmonic_A_h(self, ocean_setup):
        from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig, HarmonicConfig

        def loss(x):
            cfg = _ocean_base()._replace(lateral_mixing=LateralMixingConfig(
                scheme="harmonic", harmonic=HarmonicConfig(A_h=x)))
            return _ocean_loss(cfg, ocean_setup)
        assert_grad_ok(loss, 1e4, "ocean.harmonic.A_h")

    def test_convection_enhanced_K_conv(self, ocean_setup):
        from legoesm.ocean.physics.convection.config import OceanConvectionConfig, EnhancedDiffusionConfig

        def loss(x):
            cfg = _ocean_base()._replace(convection=OceanConvectionConfig(
                scheme="enhanced_diffusion",
                enhanced_diffusion=EnhancedDiffusionConfig(K_conv=x)))
            return _ocean_loss(cfg, ocean_setup)
        assert_grad_ok(loss, 1.0, "ocean.enhanced_diffusion.K_conv")

    @pytest.mark.parametrize("field,val", [
        ("epsilon", 1e-3), ("T_excess", 0.05),
        # probe sharpness in its unsaturated band: the production default
        # 1e4 saturates the activity sigmoid (zero grad by construction);
        # the regression target here is the carry-dtype crash, which the
        # carry still exercises at any sharpness.
        ("active_sigmoid_sharpness", 1e-2)])
    def test_convection_plume_carry_params(self, ocean_setup, field, val):
        """Regression for the 2026-06 plume scan-carry dtype crash: epsilon /
        T_excess / active_sigmoid_sharpness feed the lax.scan carry, so a
        float64-traced value used to promote the float32 carry mid-scan and
        raise a carry-type mismatch.  The carry is now pinned to the state
        dtype."""
        from legoesm.ocean.physics.convection.config import OceanConvectionConfig, PlumeConfig

        def loss(x):
            cfg = _ocean_base()._replace(convection=OceanConvectionConfig(
                scheme="plume", plume=PlumeConfig()._replace(**{field: x})))
            return _ocean_loss(cfg, ocean_setup)
        assert_grad_ok(loss, val, f"ocean.plume.{field}")
