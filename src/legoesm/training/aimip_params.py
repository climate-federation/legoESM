"""AIMIP classical-physics trainable parameters and physics builder.

Exposes the tunable scheme knobs for the AIMIP "classical" variant
(Tiedtke convection, Louis turbulence, surface bulk fluxes, McFarlane
gravity wave drag, Xu-Randall cloud fraction) as a single Equinox
module so that ``eqx.filter_value_and_grad`` propagates gradients
through legoESM's differentiable spectral PE rollout into each scheme
parameter.

The companion :func:`make_aimip_classical_spectral_physics` builder
mirrors :func:`legoesm.training.neural_gcm_spectral.make_physics_params_spectral_physics`
but injects the full AIMIP scheme set (Tiedtke / Louis / McFarlane /
Xu-Randall + gray radiation) through ``combined.make_physics`` with
``model_type='spectral_pe'`` — which fully supports the
profile-prognostic Tiedtke carry path (unlike the cubed-sphere
``PhysicsPipeline``; see ``physics_pipeline.py:897``).

Pattern follows :class:`legoesm.training.trainable_params.TrainablePhysicsParams`:
dict-of-raw-values Equinox module + ``ParamConstraint`` list with
sigmoid transforms.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.atmosphere.physics.convection.config import (
    ConvectionConfig,
    TiedtkeConfig,
)
from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    GravityWaveDragConfig,
    McFarlaneConfig,
)
from legoesm.atmosphere.physics.turbulence.config import (
    LouisConfig,
    SurfaceLayerConfig,
    TurbulenceConfig,
)
from legoesm.training.trainable_params import (
    ParamConstraint,
    _range_to_sigmoid,
    _sigmoid_to_range,
)
from legoesm.training.aimip_spatial import (
    AIMIPSpatialSurfaceParams,
    SPATIAL_FIELD_NAMES,
)


# ----------------------------------------------------------------------
# Tunable parameter groups
# ----------------------------------------------------------------------
# Bounds chosen to span physically defensible ranges around each
# scheme's published defaults.  All transforms are 'sigmoid' so the
# unconstrained optimizer-space leaves are unbounded reals.

_TIEDTKE_TRAINABLE: list[ParamConstraint] = [
    ParamConstraint("tiedtke_tau_M_u_relax", 600.0, 14400.0, "sigmoid"),
    ParamConstraint("tiedtke_tau_MC_proxy", 1800.0, 28800.0, "sigmoid"),
    ParamConstraint("tiedtke_cape_threshold", 10.0, 500.0, "sigmoid"),
    ParamConstraint("tiedtke_precip_efficiency", 0.2, 0.95, "sigmoid"),
    ParamConstraint("tiedtke_downdraft_alpha", 0.05, 0.6, "sigmoid"),
    ParamConstraint("tiedtke_downdraft_RH_min", 0.1, 0.6, "sigmoid"),
    # Extended: entrainment / detrainment + closure knobs (audit pass).
    ParamConstraint("tiedtke_epsilon_deep", 2.0e-5, 5.0e-4, "sigmoid"),
    ParamConstraint("tiedtke_delta_deep", 2.0e-5, 5.0e-4, "sigmoid"),
    ParamConstraint("tiedtke_epsilon_shallow", 1.0e-4, 1.0e-3, "sigmoid"),
    ParamConstraint("tiedtke_delta_shallow", 1.0e-4, 1.0e-3, "sigmoid"),
    ParamConstraint("tiedtke_epsilon_midlevel", 2.0e-5, 5.0e-4, "sigmoid"),
    ParamConstraint("tiedtke_delta_midlevel", 2.0e-5, 5.0e-4, "sigmoid"),
    ParamConstraint("tiedtke_cmt_c_u", 0.3, 1.0, "sigmoid"),
    ParamConstraint("tiedtke_cmt_c_d", 0.3, 1.0, "sigmoid"),
    ParamConstraint("tiedtke_parcel_dT", 0.1, 1.5, "sigmoid"),
    ParamConstraint("tiedtke_parcel_dq", 1.0e-4, 5.0e-3, "sigmoid"),
    ParamConstraint("tiedtke_cloud_depth_deep", 1500.0, 5000.0, "sigmoid"),
    ParamConstraint("tiedtke_cloud_depth_shallow_max", 500.0, 3000.0, "sigmoid"),
    ParamConstraint("tiedtke_mc_proxy_RH_crit", 0.4, 0.9, "sigmoid"),
]

_LOUIS_TRAINABLE: list[ParamConstraint] = [
    ParamConstraint("louis_l_mix_max", 20.0, 400.0, "sigmoid"),
    ParamConstraint("louis_Ck", 0.1, 0.6, "sigmoid"),
    ParamConstraint("louis_Ri_crit", 0.1, 0.6, "sigmoid"),
    ParamConstraint("louis_b_louis", 2.0, 10.0, "sigmoid"),
    ParamConstraint("louis_c_louis", 5.0, 30.0, "sigmoid"),
    ParamConstraint("louis_d_louis", 2.0, 15.0, "sigmoid"),
]

_SURFACE_TRAINABLE: list[ParamConstraint] = [
    ParamConstraint("surface_Cd_neutral", 5.0e-4, 3.0e-3, "sigmoid"),
    ParamConstraint("surface_Ch_neutral", 5.0e-4, 3.0e-3, "sigmoid"),
    ParamConstraint("surface_z0", 1.0e-5, 1.0e-3, "sigmoid"),
]

_MCFARLANE_TRAINABLE: list[ParamConstraint] = [
    ParamConstraint("mcfarlane_h_topo", 100.0, 2000.0, "sigmoid"),
    ParamConstraint("mcfarlane_G_0", 0.1, 1.0, "sigmoid"),
    ParamConstraint("mcfarlane_efficiency", 0.1, 1.0, "sigmoid"),
    ParamConstraint("mcfarlane_min_wind", 0.5, 5.0, "sigmoid"),
    ParamConstraint("mcfarlane_envelope_scale", 0.5, 2.0, "sigmoid"),
    # Extended: orographic wavenumber + reference BV + spread + tau cap.
    ParamConstraint("mcfarlane_k_wave", 1.0e-5, 5.0e-4, "sigmoid"),
    ParamConstraint("mcfarlane_N_ref", 5.0e-3, 2.0e-2, "sigmoid"),
    ParamConstraint("mcfarlane_directional_spread", 0.5, 2.0, "sigmoid"),
    ParamConstraint("mcfarlane_tau_max", 1.0, 50.0, "sigmoid"),
]

# Xu-Randall cloud fraction.  ``T_freeze`` is NOT trainable per
# ``CLAUDE.md`` constants discipline; it lives in ``constants.py``.
_XU_RANDALL_TRAINABLE: list[ParamConstraint] = [
    ParamConstraint("cloud_rh_crit", 0.5, 0.95, "sigmoid"),
    ParamConstraint("cloud_alpha_xr", 25.0, 400.0, "sigmoid"),
    ParamConstraint("cloud_p_xr", 0.1, 1.0, "sigmoid"),
    ParamConstraint("cloud_q_c_diagnostic", 5.0e-5, 5.0e-4, "sigmoid"),
    # Cloud particle effective radii (drive RRTMGP cloud optics).
    ParamConstraint("cloud_r_eff_liq", 5.0e-6, 30.0e-6, "sigmoid"),
    ParamConstraint("cloud_r_eff_ice", 10.0e-6, 100.0e-6, "sigmoid"),
]

# Gray two-stream radiation knobs (Frierson et al. 2006).  The user
# audit identified this as the biggest gap — radiation is the
# dominant lever on the residual T bias.  ``tau_equator`` and
# ``tau_pole`` were already exposed via ``_AIMIP_COMMON_TRAINABLE``.
_GRAY_RAD_TRAINABLE: list[ParamConstraint] = [
    ParamConstraint("gray_linear_frac", 0.0, 0.6, "sigmoid"),
    ParamConstraint("gray_tau_moist_coeff", 5.0e-3, 2.5e-2, "sigmoid"),
    ParamConstraint("gray_lw_diff_factor", 1.2, 2.0, "sigmoid"),
    ParamConstraint("gray_sfc_emissivity", 0.85, 1.0, "sigmoid"),
    ParamConstraint("gray_sw_tau_0", 0.0, 0.5, "sigmoid"),
    ParamConstraint("gray_sw_exponent", 1.0, 4.0, "sigmoid"),
    ParamConstraint("gray_sfc_albedo", 0.05, 0.4, "sigmoid"),
]

# Sundqvist large-scale condensation (now the AIMIP-winning
# microphysics scheme; previously had zero trained knobs).
_SUNDQVIST_TRAINABLE: list[ParamConstraint] = [
    ParamConstraint("sundqvist_RH_crit", 0.5, 0.95, "sigmoid"),
    ParamConstraint("sundqvist_sigmoid_sharpness", 5.0, 50.0, "sigmoid"),
    ParamConstraint("sundqvist_auto_rate", 1.0e-4, 1.0e-2, "sigmoid"),
    ParamConstraint("sundqvist_evap_coeff", 1.0e-5, 5.0e-3, "sigmoid"),
]

# Simplified Betts-Miller convection (now the AIMIP-winning
# convection scheme).
_SBM_TRAINABLE: list[ParamConstraint] = [
    ParamConstraint("sbm_tau_c", 1800.0, 21600.0, "sigmoid"),
    ParamConstraint("sbm_RH_ref", 0.5, 0.9, "sigmoid"),
    ParamConstraint("sbm_CAPE_threshold", 10.0, 500.0, "sigmoid"),
    ParamConstraint("sbm_T_min_convect", 180.0, 220.0, "sigmoid"),
]

# RRTMGP knobs (parked; active when ``aimip_radiation=rrtmgp``).
_RRTMGP_TRAINABLE: list[ParamConstraint] = [
    ParamConstraint("rrtmgp_co2_ppmv", 200.0, 800.0, "sigmoid"),
    ParamConstraint("rrtmgp_ch4_ppbv", 700.0, 3000.0, "sigmoid"),
    ParamConstraint("rrtmgp_n2o_ppbv", 250.0, 400.0, "sigmoid"),
    ParamConstraint("rrtmgp_sfc_emissivity", 0.85, 1.0, "sigmoid"),
    ParamConstraint("rrtmgp_sfc_albedo", 0.03, 0.4, "sigmoid"),
    ParamConstraint("rrtmgp_aerosol_ssa", 0.8, 1.0, "sigmoid"),
    ParamConstraint("rrtmgp_aerosol_g", 0.5, 0.9, "sigmoid"),
]

# Shared surface-energy-balance knobs (also feed gray radiation
# ``tau_equator`` / ``tau_pole``).
_AIMIP_COMMON_TRAINABLE: list[ParamConstraint] = [
    ParamConstraint("tau_equator", 3.0, 12.0, "sigmoid"),
    ParamConstraint("tau_pole", 0.5, 4.0, "sigmoid"),
    ParamConstraint("albedo_ice", 0.4, 0.8, "sigmoid"),
    ParamConstraint("albedo_ocean", 0.03, 0.10, "sigmoid"),
]


AIMIP_CLASSICAL_CONSTRAINTS: list[ParamConstraint] = (
    _TIEDTKE_TRAINABLE
    + _LOUIS_TRAINABLE
    + _SURFACE_TRAINABLE
    + _MCFARLANE_TRAINABLE
    + _XU_RANDALL_TRAINABLE
    + _GRAY_RAD_TRAINABLE
    + _SUNDQVIST_TRAINABLE
    + _SBM_TRAINABLE
    + _RRTMGP_TRAINABLE
    + _AIMIP_COMMON_TRAINABLE
)


# ----------------------------------------------------------------------
# Equinox container for the AIMIP classical variant
# ----------------------------------------------------------------------

class AIMIPClassicalParams(eqx.Module):
    """Trainable scheme tunables for the AIMIP classical variant.

    Stores unconstrained raw values; ``as_dict`` returns physically
    constrained values via per-knob sigmoid transforms.  ``to_*_config``
    methods build the corresponding scheme ``*Config`` NamedTuples
    with the current learnable values substituted into the
    scheme-tunable fields (all other fields take their scheme defaults
    so that future scheme changes do not silently mutate AIMIP
    behavior).

    When ``spatial_surface`` is non-None, the surface-aerodynamic and
    surface-radiation fields (``Cd_neutral``, ``Ch_neutral``, ``z0``,
    ``sfc_emissivity``, ``sfc_albedo``, ``albedo_ocean``,
    ``albedo_ice``) become low-rank learnable lat-lon fields gated by
    a land mask (see :mod:`legoesm.training.aimip_spatial`).  The
    corresponding scalar knobs in ``raw_values`` are still trained
    and used as the spatial-field baselines for ocean columns; the
    spatial perturbation only takes effect where ``land_mask > 0``.
    """
    raw_values: dict[str, jax.Array]
    constraints: list[ParamConstraint] = eqx.field(static=True)
    spatial_surface: AIMIPSpatialSurfaceParams | None = None

    @staticmethod
    def from_defaults(
        spatial_surface: bool = False,
    ) -> "AIMIPClassicalParams":
        """Initialize all knobs at their canonical scheme defaults.

        Parameters
        ----------
        spatial_surface : bool
            If True, also initialize a :class:`AIMIPSpatialSurfaceParams`
            bundle for the surface and surface-radiation knobs at
            zero perturbation (i.e. equal to the scalar baseline).
        """
        scheme_defaults = _canonical_scheme_defaults()
        try:
            from legoesm.core.precision import get_policy

            param_dtype = get_policy().compute
        except Exception:
            param_dtype = jnp.float32

        raw: dict[str, jax.Array] = {}
        for c in AIMIP_CLASSICAL_CONSTRAINTS:
            default = scheme_defaults.get(c.name, 0.5 * (c.min_val + c.max_val))
            raw[c.name] = jnp.asarray(
                _range_to_sigmoid(default, c.min_val, c.max_val),
                dtype=param_dtype,
            )
        spatial = (
            AIMIPSpatialSurfaceParams.from_defaults(dtype=param_dtype)
            if spatial_surface else None
        )
        return AIMIPClassicalParams(
            raw_values=raw,
            constraints=AIMIP_CLASSICAL_CONSTRAINTS,
            spatial_surface=spatial,
        )

    def as_dict(self) -> dict[str, jax.Array]:
        """Return constrained physical values for every knob."""
        return {
            c.name: _sigmoid_to_range(self.raw_values[c.name], c.min_val, c.max_val)
            for c in self.constraints
        }

    # ------------------------------------------------------------------
    # Per-scheme config builders.  These are pure: they only read
    # ``as_dict()`` and produce NamedTuples, so
    # ``eqx.filter_value_and_grad`` traces gradients through each
    # constructed config field.
    # ------------------------------------------------------------------

    def to_tiedtke_config(self) -> TiedtkeConfig:
        d = self.as_dict()
        base = TiedtkeConfig()
        return base._replace(
            tau_M_u_relax=d["tiedtke_tau_M_u_relax"],
            tau_MC_proxy=d["tiedtke_tau_MC_proxy"],
            cape_threshold=d["tiedtke_cape_threshold"],
            precip_efficiency=d["tiedtke_precip_efficiency"],
            downdraft_alpha=d["tiedtke_downdraft_alpha"],
            downdraft_RH_min=d["tiedtke_downdraft_RH_min"],
            epsilon_deep=d["tiedtke_epsilon_deep"],
            delta_deep=d["tiedtke_delta_deep"],
            epsilon_shallow=d["tiedtke_epsilon_shallow"],
            delta_shallow=d["tiedtke_delta_shallow"],
            epsilon_midlevel=d["tiedtke_epsilon_midlevel"],
            delta_midlevel=d["tiedtke_delta_midlevel"],
            cmt_c_u=d["tiedtke_cmt_c_u"],
            cmt_c_d=d["tiedtke_cmt_c_d"],
            parcel_dT=d["tiedtke_parcel_dT"],
            parcel_dq=d["tiedtke_parcel_dq"],
            cloud_depth_deep=d["tiedtke_cloud_depth_deep"],
            cloud_depth_shallow_max=d["tiedtke_cloud_depth_shallow_max"],
            mc_proxy_RH_crit=d["tiedtke_mc_proxy_RH_crit"],
        )

    def to_surface_config(self) -> SurfaceLayerConfig:
        d = self.as_dict()
        base = SurfaceLayerConfig()
        return base._replace(
            Cd_neutral=d["surface_Cd_neutral"],
            Ch_neutral=d["surface_Ch_neutral"],
            z0=d["surface_z0"],
        )

    def to_louis_config(self) -> LouisConfig:
        d = self.as_dict()
        base = LouisConfig(surface=self.to_surface_config())
        return base._replace(
            l_mix_max=d["louis_l_mix_max"],
            Ck=d["louis_Ck"],
            Ri_crit=d["louis_Ri_crit"],
            b_louis=d["louis_b_louis"],
            c_louis=d["louis_c_louis"],
            d_louis=d["louis_d_louis"],
        )

    def to_mcfarlane_config(self) -> McFarlaneConfig:
        d = self.as_dict()
        base = McFarlaneConfig()
        return base._replace(
            h_topo=d["mcfarlane_h_topo"],
            G_0=d["mcfarlane_G_0"],
            efficiency=d["mcfarlane_efficiency"],
            min_wind=d["mcfarlane_min_wind"],
            envelope_scale=d["mcfarlane_envelope_scale"],
            k_wave=d["mcfarlane_k_wave"],
            N_ref=d["mcfarlane_N_ref"],
            directional_spread=d["mcfarlane_directional_spread"],
            tau_max=d["mcfarlane_tau_max"],
        )

    def to_cloud_config(self) -> CloudConfig:
        d = self.as_dict()
        base = CloudConfig(scheme="xu_randall")
        return base._replace(
            rh_crit=d["cloud_rh_crit"],
            alpha_xr=d["cloud_alpha_xr"],
            p_xr=d["cloud_p_xr"],
            q_c_diagnostic=d["cloud_q_c_diagnostic"],
            r_eff_liq=d["cloud_r_eff_liq"],
            r_eff_ice=d["cloud_r_eff_ice"],
        )

    def to_sundqvist_config(self):
        """Build a SundqvistConfig (microphysics) with trained leaves."""
        from legoesm.atmosphere.physics.microphysics.config import (
            SundqvistConfig,
        )
        d = self.as_dict()
        base = SundqvistConfig()
        return base._replace(
            RH_crit=d["sundqvist_RH_crit"],
            sigmoid_sharpness=d["sundqvist_sigmoid_sharpness"],
            auto_rate=d["sundqvist_auto_rate"],
            evap_coeff=d["sundqvist_evap_coeff"],
        )

    def to_sbm_config(self):
        """Build an SBMConfig (convection) with trained leaves."""
        from legoesm.atmosphere.physics.convection.config import SBMConfig
        d = self.as_dict()
        base = SBMConfig()
        return base._replace(
            tau_c=d["sbm_tau_c"],
            RH_ref=d["sbm_RH_ref"],
            CAPE_threshold=d["sbm_CAPE_threshold"],
            T_min_convect=d["sbm_T_min_convect"],
        )

    def to_gray_radiation_config(self):
        """Build a GrayRadiationConfig with all 9 traced fields."""
        from legoesm.atmosphere.physics.radiation.config import (
            GrayRadiationConfig,
        )
        d = self.as_dict()
        base = GrayRadiationConfig()
        return base._replace(
            tau_equator=d["tau_equator"],
            tau_pole=d["tau_pole"],
            linear_frac=d["gray_linear_frac"],
            tau_moist_coeff=d["gray_tau_moist_coeff"],
            lw_diff_factor=d["gray_lw_diff_factor"],
            sfc_emissivity=d["gray_sfc_emissivity"],
            sw_tau_0=d["gray_sw_tau_0"],
            sw_exponent=d["gray_sw_exponent"],
            sfc_albedo=d["gray_sfc_albedo"],
        )

    def to_rrtmgp_config(self):
        """Build a RRTMGPConfig with trained surface + aerosol knobs.

        Gas concentrations (CO2, CH4, N2O) are intentionally NOT
        pulled from the trained sigmoid leaves: the RRTMGP optics
        cache (``rrtmgp.RRTMGP._cache_key``) hashes them, and a
        traced JAX array is unhashable under
        ``eqx.filter_value_and_grad``.  Cold-bias closure under
        AIMIP is driven by cloud-LW coupling + surface
        emissivity/albedo, not by the modest gas-absorption
        perturbations the sigmoid bounds would allow, so we freeze
        gas concentrations to the canonical RRTMGP defaults and
        keep surface + aerosol knobs trainable.  The corresponding
        ``rrtmgp_co2_ppmv`` / ``ch4_ppbv`` / ``n2o_ppbv`` raw
        leaves still exist for forward-compatibility but are not
        wired into the radiation config until the cache-key bug is
        addressed in ``radiation/rrtmgp/rrtmgp.py``.
        """
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        d = self.as_dict()
        base = RRTMGPConfig()
        return base._replace(
            # Gas concentrations frozen to scheme defaults (see docstring).
            sfc_emissivity=d["rrtmgp_sfc_emissivity"],
            sfc_albedo=d["rrtmgp_sfc_albedo"],
            aerosol_ssa=d["rrtmgp_aerosol_ssa"],
            aerosol_g=d["rrtmgp_aerosol_g"],
        )


def _canonical_scheme_defaults() -> dict[str, float]:
    """Look up the canonical scheme-default for each AIMIP knob."""
    from legoesm.atmosphere.physics.convection.config import SBMConfig
    from legoesm.atmosphere.physics.microphysics.config import SundqvistConfig
    from legoesm.atmosphere.physics.radiation.config import (
        GrayRadiationConfig,
        RRTMGPConfig,
    )

    t = TiedtkeConfig()
    lo = LouisConfig()
    su = SurfaceLayerConfig()
    mc = McFarlaneConfig()
    cl = CloudConfig()
    g = GrayRadiationConfig()
    sq = SundqvistConfig()
    sbm = SBMConfig()
    rr = RRTMGPConfig()
    return {
        # Tiedtke (base + extended)
        "tiedtke_tau_M_u_relax": float(t.tau_M_u_relax),
        "tiedtke_tau_MC_proxy": float(t.tau_MC_proxy),
        "tiedtke_cape_threshold": float(t.cape_threshold),
        "tiedtke_precip_efficiency": float(t.precip_efficiency),
        "tiedtke_downdraft_alpha": float(t.downdraft_alpha),
        "tiedtke_downdraft_RH_min": float(t.downdraft_RH_min),
        "tiedtke_epsilon_deep": float(t.epsilon_deep),
        "tiedtke_delta_deep": float(t.delta_deep),
        "tiedtke_epsilon_shallow": float(t.epsilon_shallow),
        "tiedtke_delta_shallow": float(t.delta_shallow),
        "tiedtke_epsilon_midlevel": float(t.epsilon_midlevel),
        "tiedtke_delta_midlevel": float(t.delta_midlevel),
        "tiedtke_cmt_c_u": float(t.cmt_c_u),
        "tiedtke_cmt_c_d": float(t.cmt_c_d),
        "tiedtke_parcel_dT": float(t.parcel_dT),
        "tiedtke_parcel_dq": float(t.parcel_dq),
        "tiedtke_cloud_depth_deep": float(t.cloud_depth_deep),
        "tiedtke_cloud_depth_shallow_max": float(t.cloud_depth_shallow_max),
        "tiedtke_mc_proxy_RH_crit": float(t.mc_proxy_RH_crit),
        # Louis
        "louis_l_mix_max": float(lo.l_mix_max),
        "louis_Ck": float(lo.Ck),
        "louis_Ri_crit": float(lo.Ri_crit),
        "louis_b_louis": float(lo.b_louis),
        "louis_c_louis": float(lo.c_louis),
        "louis_d_louis": float(lo.d_louis),
        # Surface
        "surface_Cd_neutral": float(su.Cd_neutral),
        "surface_Ch_neutral": float(su.Ch_neutral),
        "surface_z0": float(su.z0),
        # McFarlane
        "mcfarlane_h_topo": float(mc.h_topo),
        "mcfarlane_G_0": float(mc.G_0),
        "mcfarlane_efficiency": float(mc.efficiency),
        "mcfarlane_min_wind": float(mc.min_wind),
        "mcfarlane_envelope_scale": float(mc.envelope_scale),
        "mcfarlane_k_wave": float(mc.k_wave),
        "mcfarlane_N_ref": float(mc.N_ref),
        "mcfarlane_directional_spread": float(mc.directional_spread),
        "mcfarlane_tau_max": float(mc.tau_max),
        # Cloud (Xu-Randall)
        "cloud_rh_crit": float(cl.rh_crit),
        "cloud_alpha_xr": float(cl.alpha_xr),
        "cloud_p_xr": float(cl.p_xr),
        "cloud_q_c_diagnostic": float(cl.q_c_diagnostic),
        "cloud_r_eff_liq": float(cl.r_eff_liq),
        "cloud_r_eff_ice": float(cl.r_eff_ice),
        # Gray radiation (the audit's biggest gap)
        "gray_linear_frac": float(g.linear_frac),
        "gray_tau_moist_coeff": float(g.tau_moist_coeff),
        "gray_lw_diff_factor": float(g.lw_diff_factor),
        "gray_sfc_emissivity": float(g.sfc_emissivity),
        "gray_sw_tau_0": float(g.sw_tau_0),
        "gray_sw_exponent": float(g.sw_exponent),
        "gray_sfc_albedo": float(g.sfc_albedo),
        # Sundqvist microphysics
        "sundqvist_RH_crit": float(sq.RH_crit),
        "sundqvist_sigmoid_sharpness": float(sq.sigmoid_sharpness),
        "sundqvist_auto_rate": float(sq.auto_rate),
        "sundqvist_evap_coeff": float(sq.evap_coeff),
        # SBM convection
        "sbm_tau_c": float(sbm.tau_c),
        "sbm_RH_ref": float(sbm.RH_ref),
        "sbm_CAPE_threshold": float(sbm.CAPE_threshold),
        "sbm_T_min_convect": float(sbm.T_min_convect),
        # RRTMGP
        "rrtmgp_co2_ppmv": float(rr.co2_ppmv),
        "rrtmgp_ch4_ppbv": float(rr.ch4_ppbv),
        "rrtmgp_n2o_ppbv": float(rr.n2o_ppbv),
        "rrtmgp_sfc_emissivity": float(rr.sfc_emissivity),
        "rrtmgp_sfc_albedo": float(rr.sfc_albedo),
        "rrtmgp_aerosol_ssa": float(rr.aerosol_ssa),
        "rrtmgp_aerosol_g": float(rr.aerosol_g),
        # Shared
        "tau_equator": 7.2,
        "tau_pole": 1.8,
        "albedo_ice": 0.65,
        "albedo_ocean": 0.06,
    }


# ----------------------------------------------------------------------
# Spectral-PE physics builder for AIMIP classical
# ----------------------------------------------------------------------

def make_aimip_classical_spectral_physics(
    params: AIMIPClassicalParams,
    grid,
    dt: float,
    *,
    radiation: str = "gray",
    rad_update_interval_steps: int = 6,
    convection_scheme: str = "tiedtke",
    turbulence_scheme: str = "louis",
    gwd_scheme: str = "mcfarlane",
    microphysics_scheme: str = "none",
    cloud_scheme: str = "xu_randall",
    land_mask: "jax.Array | None" = None,
):
    """Build a SpectralPE physics function for the AIMIP classical variant.

    Constructs a ``PhysicsConfig`` by reading the current ``params``
    Equinox leaves and assembling Tiedtke / Louis / McFarlane scheme
    configs from them, then dispatches through ``combined.make_physics``
    with ``model_type='spectral_pe'``.  The returned callable matches
    the signature consumed by :func:`spectral_rollout`:

        physics_fn(state, grid, sigma_coord) -> SpectralHydrostaticState

    Gray radiation is used (tunable ``tau_equator`` / ``tau_pole``) so
    surface-energy-balance gradients flow back through radiation as
    well as through the dynamic schemes.

    When ``params.spatial_surface`` is set and ``land_mask`` is
    provided, the surface (``Cd_neutral``, ``Ch_neutral``, ``z0``) and
    surface-radiation (``sfc_emissivity``, ``sfc_albedo``) knobs are
    replaced with column-flattened lat-lon fields produced by the
    spatial-parameter bundle.  Over-ocean columns fall back to the
    global scalar baseline (per the user-facing semantics in
    ``aimip_spatial.SpatialField.evaluate``).

    Mirrors the role of :func:`make_physics_params_spectral_physics`
    in ``training.neural_gcm_spectral`` but with the full AIMIP scheme
    set, including the profile-prognostic Tiedtke path that
    ``PhysicsPipeline`` cannot host (see
    ``physics_pipeline.py:897`` ``_PIPELINE_UNSUPPORTED_CONVECTION``).
    """
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.radiation.config import (
        GrayRadiationConfig,
        RadiationConfig,
        RRTMGPConfig,
    )

    p = params.as_dict()

    # ---- Spatial surface fields (optional) ----
    # When ``params.spatial_surface`` is non-None, each spatial field
    # is evaluated on the grid (with optional land-mask gating) and
    # flattened to (ncol,) so the downstream surface_layer / gray-
    # radiation code paths receive arrays that broadcast against the
    # column-wise prognostic fields.
    spatial_fields_col: dict[str, jax.Array] = {}
    if params.spatial_surface is not None:
        fields_2d = params.spatial_surface.evaluate(grid, land_mask=land_mask)
        n_lat, n_lon = grid.n_lat, grid.n_lon
        for name, arr in fields_2d.items():
            spatial_fields_col[name] = arr.reshape(n_lat * n_lon)

    # Radiation backend toggle.  ``rrtmgp`` is the production
    # correlated-k path: it explicitly couples Xu-Randall cloud
    # fraction into shortwave + longwave fluxes and makes the
    # cloud knobs trainable end-to-end (via
    # ``RadiationConfig.cloud_config`` -> ``radiation/integration.py``).
    # ``gray`` is the Frierson-style two-stream analytic path:
    # cheap, no cloud coupling, only ``tau_equator`` /
    # ``tau_pole`` are differentiated.  Default ``gray`` keeps the
    # AIMIP harness tractable on a single GPU; bump to ``rrtmgp``
    # for production-grade physics realism.
    # ---- Cloud config (trained when xu_randall, defaults otherwise) ----
    if cloud_scheme == "xu_randall":
        cloud_cfg_trained = params.to_cloud_config()
    elif cloud_scheme == "none":
        cloud_cfg_trained = None
    else:
        from legoesm.atmosphere.physics.clouds.config import CloudConfig as _CC
        cloud_cfg_trained = _CC(scheme=cloud_scheme)

    # ---- Radiation ----
    if radiation == "rrtmgp":
        rrtmgp_cfg = params.to_rrtmgp_config()
        # Substitute spatial sfc_emissivity / sfc_albedo when present.
        # ``rrtmgp_radiation`` accepts array overrides via the
        # integration bridge (rrtmgp_radiation.py: sfc_albedo_override /
        # sfc_emissivity_override), and ``RRTMGPConfig.sfc_*`` fields
        # broadcast naturally over the column dimension when set to
        # ``(ncol,)`` arrays here.
        if "sfc_emissivity" in spatial_fields_col:
            rrtmgp_cfg = rrtmgp_cfg._replace(
                sfc_emissivity=spatial_fields_col["sfc_emissivity"],
            )
        if "sfc_albedo" in spatial_fields_col:
            rrtmgp_cfg = rrtmgp_cfg._replace(
                sfc_albedo=spatial_fields_col["sfc_albedo"],
            )
        rad_cfg = RadiationConfig(
            scheme="rrtmgp",
            rrtmgp=rrtmgp_cfg,
            cloud_scheme=cloud_scheme,
            cloud_config=cloud_cfg_trained,
            update_interval_steps=rad_update_interval_steps,
        )
    elif radiation == "gray":
        # Full 9-knob gray radiation (audit pass).  Was previously
        # only ``tau_equator`` / ``tau_pole`` — the residual T bias
        # was traced to fixed-default ``tau_moist_coeff``,
        # ``lw_diff_factor``, ``sfc_emissivity`` etc.
        gray_cfg = params.to_gray_radiation_config()
        # Substitute spatial sfc_emissivity / sfc_albedo when present.
        # ``gray.py`` lines 175-176 and 250 use these as scalars that
        # broadcast against column-shaped arrays — passing (ncol,)
        # arrays substitutes pointwise without code changes.
        if "sfc_emissivity" in spatial_fields_col:
            gray_cfg = gray_cfg._replace(
                sfc_emissivity=spatial_fields_col["sfc_emissivity"],
            )
        if "sfc_albedo" in spatial_fields_col:
            gray_cfg = gray_cfg._replace(
                sfc_albedo=spatial_fields_col["sfc_albedo"],
            )
        rad_cfg = RadiationConfig(
            scheme="gray",
            gray=gray_cfg,
        )
    else:
        raise ValueError(
            f"Unknown AIMIP radiation backend: {radiation!r} "
            f"(expected 'gray' or 'rrtmgp')."
        )

    # ---- Convection ----
    if convection_scheme == "tiedtke":
        conv_cfg = ConvectionConfig(
            scheme="tiedtke", tiedtke=params.to_tiedtke_config(),
        )
    elif convection_scheme == "sbm":
        conv_cfg = ConvectionConfig(
            scheme="sbm", sbm=params.to_sbm_config(),
        )
    else:
        conv_cfg = ConvectionConfig(scheme=convection_scheme)

    # ---- Turbulence (with optional spatial surface params) ----
    # When the surface knobs (``Cd_neutral``, ``Ch_neutral``, ``z0``)
    # are spatial, replace the corresponding ``SurfaceLayerConfig``
    # fields with (ncol,) arrays.  ``compute_surface_fluxes`` in
    # ``turbulence/surface_layer.py`` already treats these fields as
    # broadcastable scalars (lines 83-84, 90-100), so no scheme-side
    # code change is required.
    def _spatial_surface_override(inner_cfg):
        if not spatial_fields_col or not hasattr(inner_cfg, "surface"):
            return inner_cfg
        new_surface = inner_cfg.surface._replace(
            Cd_neutral=spatial_fields_col.get(
                "Cd_neutral", inner_cfg.surface.Cd_neutral,
            ),
            Ch_neutral=spatial_fields_col.get(
                "Ch_neutral", inner_cfg.surface.Ch_neutral,
            ),
            z0=spatial_fields_col.get("z0", inner_cfg.surface.z0),
        )
        return inner_cfg._replace(surface=new_surface)

    if turbulence_scheme == "louis":
        louis_cfg = _spatial_surface_override(params.to_louis_config())
        turb_cfg = TurbulenceConfig(scheme="louis", louis=louis_cfg)
    elif turbulence_scheme == "tke":
        from legoesm.atmosphere.physics.turbulence.config import TKEConfig
        turb_cfg = TurbulenceConfig(
            scheme="tke",
            tke=_spatial_surface_override(TKEConfig()),
        )
    elif turbulence_scheme == "smagorinsky":
        from legoesm.atmosphere.physics.turbulence.config import (
            SmagorinskyConfig,
        )
        turb_cfg = TurbulenceConfig(
            scheme="smagorinsky",
            smagorinsky=_spatial_surface_override(SmagorinskyConfig()),
        )
    else:
        # Other schemes (holtslag_boville, ysu, edmf, clubb_lite, none).
        # Spatial surface override silently skipped — extend this
        # branch when those become AIMIP ablation candidates.
        turb_cfg = TurbulenceConfig(scheme=turbulence_scheme)

    # ---- Gravity wave drag ----
    if gwd_scheme == "mcfarlane":
        gwd_cfg = GravityWaveDragConfig(
            scheme="mcfarlane", mcfarlane=params.to_mcfarlane_config(),
        )
    elif gwd_scheme == "none":
        gwd_cfg = GravityWaveDragConfig(scheme="none")
    else:
        gwd_cfg = GravityWaveDragConfig(scheme=gwd_scheme)

    # ---- Microphysics ----
    from legoesm.atmosphere.physics.microphysics.config import (
        MicrophysicsConfig,
    )
    if microphysics_scheme == "sundqvist":
        micro_cfg = MicrophysicsConfig(
            scheme="sundqvist", sundqvist=params.to_sundqvist_config(),
        )
    else:
        micro_cfg = MicrophysicsConfig(scheme=microphysics_scheme)

    # Note: ``p = params.as_dict()`` is already computed above for
    # gray-radiation tau knobs; reused here only when gray is active.
    del p  # avoid leaking variable into nested closure
    physics_config = PhysicsConfig(
        radiation=rad_cfg,
        convection=conv_cfg,
        turbulence=turb_cfg,
        microphysics=micro_cfg,
        gravity_wave_drag=gwd_cfg,
    )
    raw_fn = make_physics(physics_config, model_type="spectral_pe", dt=dt)

    def physics_fn(state, grid_, sigma_coord):
        result = raw_fn(state, grid_, sigma_coord)
        return result[0] if isinstance(result, tuple) else result

    return physics_fn
