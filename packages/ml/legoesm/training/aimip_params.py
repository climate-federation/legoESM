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
    range_to_sigmoid,
    sigmoid_to_range,
)
from legoesm.training.aimip_spatial import AIMIPSpatialSurfaceParams


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
    ParamConstraint("louis_Ri_crit", 0.1, 0.6, "sigmoid"),
    ParamConstraint("louis_b_louis", 2.0, 10.0, "sigmoid"),
    ParamConstraint("louis_c_louis", 5.0, 30.0, "sigmoid"),
    ParamConstraint("louis_d_louis", 2.0, 15.0, "sigmoid"),
]

_SURFACE_TRAINABLE: list[ParamConstraint] = [
    ParamConstraint("surface_Cd_neutral", 5.0e-4, 3.0e-3, "sigmoid"),
    ParamConstraint("surface_Ch_neutral", 5.0e-4, 3.0e-3, "sigmoid"),
    ParamConstraint("surface_z0", 1.0e-5, 1.0e-3, "sigmoid"),
    # Global Monin-Obukhov (MOST) stability-function coefficients (Businger-Dyer
    # / Dyer 1974). Only bite when the classical curriculum runs a
    # stability-dependent bulk_scheme (large_yeager); the constant-coefficient
    # default surface path ignores them. Bounds match SurfaceLayerConfig's
    # __param_spec__ (most_unstable_gamma / most_stable_beta).
    ParamConstraint("surface_most_unstable_gamma", 8.0, 28.0, "sigmoid"),
    ParamConstraint("surface_most_stable_beta", 2.0, 10.0, "sigmoid"),
    # Thermal/momentum roughness ratio z0h/z0 (Garratt 1992). Only bites on a
    # stability-dependent bulk_scheme's fixed-roughness log-law (constant/most)
    # branch; the constant-Cd default path ignores it. Bounds match
    # SurfaceLayerConfig's __param_spec__ (z0h_z0_ratio).
    ParamConstraint("surface_z0h_z0_ratio", 0.01, 1.0, "sigmoid"),
]

_MCFARLANE_TRAINABLE: list[ParamConstraint] = [
    ParamConstraint("mcfarlane_h_topo", 100.0, 2000.0, "sigmoid"),
    ParamConstraint("mcfarlane_G_0", 0.1, 1.0, "sigmoid"),
    ParamConstraint("mcfarlane_efficiency", 0.1, 1.0, "sigmoid"),
    ParamConstraint("mcfarlane_min_wind", 0.5, 5.0, "sigmoid"),
    ParamConstraint("mcfarlane_envelope_scale", 0.5, 2.0, "sigmoid"),
    # Extended: orographic wavenumber + spread + tau cap.
    ParamConstraint("mcfarlane_k_wave", 1.0e-5, 5.0e-4, "sigmoid"),
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
#
# v7 (2026-05-19): widened sfc_emissivity / sfc_albedo bounds.
# v5+v6 produced a +1.07 K warm T bias that the optimizer could
# not close because the scalar trained leaves were saturated near
# their published defaults (sfc_emissivity bound 0.85-1.0, default
# 1.0 -> sigmoid pinned at upper edge, gradient ~0; sfc_albedo
# bound 0.05-0.4 with default 0.31 = 77 % of range).  Wider bounds
# put the defaults closer to the sigmoid interior so the
# bias-penalty gradient can move the knobs.  Centering the defaults
# inside the new range is left to a follow-up that adjusts
# ``_canonical_scheme_defaults`` consistently.
_GRAY_RAD_TRAINABLE: list[ParamConstraint] = [
    ParamConstraint("gray_linear_frac", 0.0, 0.6, "sigmoid"),
    ParamConstraint("gray_tau_moist_coeff", 5.0e-3, 2.5e-2, "sigmoid"),
    ParamConstraint("gray_lw_diff_factor", 1.2, 2.0, "sigmoid"),
    ParamConstraint("gray_sfc_emissivity", 0.5, 1.0, "sigmoid"),
    ParamConstraint("gray_sw_tau_0", 0.0, 0.5, "sigmoid"),
    ParamConstraint("gray_sw_exponent", 1.0, 4.0, "sigmoid"),
    ParamConstraint("gray_sfc_albedo", 0.03, 0.6, "sigmoid"),
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
]

# RRTMGP knobs (active when ``aimip_radiation=rrtmgp``).
# TRULY-TUNABLE SET (2026-06-13): only sfc_emissivity / sfc_albedo are trainable
# here. They reach the radiation solve as PER-CALL overrides
# (make_aimip_classical_spectral_physics -> make_physics(sfc_*_override=...) ->
# _resolve_surface_field), so the trained spatial field is differentiable AND
# never written into RRTMGPConfig.sfc_* (which RRTMGP folds into its Python
# solver-cache key). The CO2/CH4/N2O gas concentrations and aerosol_ssa/aerosol_g
# are RAW elements of that cache-key tuple (unhashable when traced) and are NOT
# trainable until RRTMGP consumes them as per-call traced inputs — they were
# previously dead leaves (``to_rrtmgp_config`` froze them to defaults, zero
# gradient), so they are removed from the trainable set rather than kept as dead
# degrees of freedom. (sfc_emissivity bound 0.5-1.0 / sfc_albedo 0.03-0.6; v7
# widened these, see _GRAY_RAD_TRAINABLE.)
_RRTMGP_TRAINABLE: list[ParamConstraint] = [
    ParamConstraint("rrtmgp_sfc_emissivity", 0.5, 1.0, "sigmoid"),
    ParamConstraint("rrtmgp_sfc_albedo", 0.03, 0.6, "sigmoid"),
]

# Shared gray-radiation optical-depth knobs (``tau_equator`` /
# ``tau_pole``).  ``albedo_ice`` / ``albedo_ocean`` were removed in the
# 2026-06 dead-code audit: no scalar or spatial consumer in
# ``make_aimip_classical_spectral_physics`` ever injected them (the
# Mode-1 RRTMGP knob set in ``trainable_params.py`` is separate and
# still carries live albedo knobs).
_AIMIP_COMMON_TRAINABLE: list[ParamConstraint] = [
    ParamConstraint("tau_equator", 3.0, 12.0, "sigmoid"),
    ParamConstraint("tau_pole", 0.5, 4.0, "sigmoid"),
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
    ``sfc_emissivity``, ``sfc_albedo``) become low-rank learnable
    lat-lon fields gated by
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
        spatial_init_std: float = 0.0,
        spatial_seed: int = 0,
    ) -> "AIMIPClassicalParams":
        """Initialize all knobs at (or just inside) their canonical defaults.

        Edge-of-range knobs whose canonical default sits within
        ``sigmoid_margin`` (5% of the range) of a bound are interiorized to
        that margin, so the initial inverse-sigmoid gradient is non-trivial and
        the optimizer can actually move them (a default exactly at a bound maps
        to a saturated logit with ~0 gradient).  Example: ``gray_sfc_emissivity``
        canonical 1.0 initializes at 0.975.  Mid-range knobs initialize exactly
        at their canonical default.  This trades a small physical perturbation
        at the edges for trainability — see the ``sigmoid_margin`` note below.

        Parameters
        ----------
        spatial_surface : bool
            If True, initialize a :class:`AIMIPSpatialSurfaceParams`
            bundle for the surface and surface-radiation knobs.
        spatial_init_std : float
            Standard deviation of the Gaussian initialization in
            spatial-coefficient space.  ``0.0`` keeps the spatial
            field equal to the scalar baseline at step zero (useful
            for sanity-checking that ``spatial_surface=True`` reduces
            to the scalar mode at init); a small positive value
            (e.g. 0.02) breaks the zero-gradient symmetry and lets
            the optimizer explore the spatial degrees of freedom
            immediately.  Only used when ``spatial_surface=True``.
        spatial_seed : int
            PRNG seed for the spatial-coefficient initialization.
            Only used when ``spatial_surface=True`` and
            ``spatial_init_std > 0``.
        """
        scheme_defaults = _canonical_scheme_defaults()
        try:
            from legoesm.core.precision import get_policy

            param_dtype = get_policy().compute
        except Exception:
            param_dtype = jnp.float32

        # v7: clamp the inverse-sigmoid input away from the bounds so
        # the initial gradient is non-trivial even when the canonical
        # default sits at the saturation edge.  Without this,
        # ``sfc_emissivity`` (canonical default 0.98 or 1.0) maps to
        # raw values where ``sigmoid'`` is ~1e-2 or less and the
        # bias-penalty loss cannot move the knob.  5% of the range
        # is a small physical perturbation (emissivity 0.98 -> 0.975
        # in [0.5, 1.0]) but bumps the sigmoid gradient by ~3x.
        sigmoid_margin = 0.05
        raw: dict[str, jax.Array] = {}
        for c in AIMIP_CLASSICAL_CONSTRAINTS:
            default = scheme_defaults.get(c.name, 0.5 * (c.min_val + c.max_val))
            margin = sigmoid_margin * (c.max_val - c.min_val)
            default_clamped = min(
                c.max_val - margin, max(c.min_val + margin, default),
            )
            raw[c.name] = jnp.asarray(
                range_to_sigmoid(default_clamped, c.min_val, c.max_val),
                dtype=param_dtype,
            )
        spatial = None
        if spatial_surface:
            spatial = AIMIPSpatialSurfaceParams.from_defaults(
                dtype=param_dtype,
                init_std=spatial_init_std,
                key=jax.random.PRNGKey(int(spatial_seed)),
            )
        return AIMIPClassicalParams(
            raw_values=raw,
            constraints=AIMIP_CLASSICAL_CONSTRAINTS,
            spatial_surface=spatial,
        )

    def as_dict(self) -> dict[str, jax.Array]:
        """Return constrained physical values for every knob."""
        return {
            c.name: sigmoid_to_range(self.raw_values[c.name], c.min_val, c.max_val)
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

    def to_surface_config(
        self, bulk_scheme: str = "constant",
    ) -> SurfaceLayerConfig:
        """Assemble the trained SurfaceLayerConfig.

        Parameters
        ----------
        bulk_scheme : str
            Surface bulk-flux scheme routed into
            ``SurfaceLayerConfig.bulk_scheme`` (``constant`` /
            ``most`` / ``coare3`` / ``large_yeager``).  The default
            ``constant`` reproduces the legacy AIMIP surface path
            (constant-Cd bulk aerodynamics), for which the trained
            ``most_unstable_gamma`` / ``most_stable_beta`` /
            ``z0h_z0_ratio`` leaves are DEAD (never read).  A
            stability-dependent scheme (``most`` etc.) routes through
            ``compute_most_fluxes`` in ``turbulence/surface_layer.py``,
            which consumes those leaves — making them live, trainable
            gradients.  Validated at the builder entry via
            ``legoesm.core.bulk_flux.validate_bulk_scheme``.
        """
        d = self.as_dict()
        base = SurfaceLayerConfig()
        return base._replace(
            Cd_neutral=d["surface_Cd_neutral"],
            Ch_neutral=d["surface_Ch_neutral"],
            z0=d["surface_z0"],
            bulk_scheme=bulk_scheme,
            most_unstable_gamma=d["surface_most_unstable_gamma"],
            most_stable_beta=d["surface_most_stable_beta"],
            z0h_z0_ratio=d["surface_z0h_z0_ratio"],
        )

    def to_louis_config(self, bulk_scheme: str = "constant") -> LouisConfig:
        d = self.as_dict()
        base = LouisConfig(surface=self.to_surface_config(bulk_scheme=bulk_scheme))
        return base._replace(
            l_mix_max=d["louis_l_mix_max"],
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
            rh_crit=d["sundqvist_RH_crit"],
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
            rh_ref=d["sbm_RH_ref"],
            cape_threshold=d["sbm_CAPE_threshold"],
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
        """Build a RRTMGPConfig at the canonical defaults (no trained values).

        EVERY RRTMGPConfig field is folded into the RRTMGP solver-cache key
        (``_instance_cache_key`` / ``_optics_cache_key``) and so cannot safely
        carry a traced JAX leaf under ``eqx.filter_value_and_grad``: gas
        concentrations (CO2/CH4/N2O) and aerosol optics (aerosol_ssa/aerosol_g)
        are RAW tuple elements (unhashable when traced), and sfc_emissivity/
        sfc_albedo route through ``_hashable`` (a traced 0-D scalar is not
        concretizable; an array id-fallback would balloon the global instance
        cache by trace identity). So this returns the unmodified defaults.

        The trained surface knobs ``rrtmgp_sfc_emissivity`` / ``rrtmgp_sfc_albedo``
        reach the radiation solve as PER-CALL OVERRIDES — never through this
        config: ``make_aimip_classical_spectral_physics`` passes the spatial
        ``(ncol,)`` fields (expanded from these leaves via ``as_dict()``
        baselines) to ``make_physics(sfc_albedo_override=, sfc_emissivity_override=)``
        -> ``make_radiation_physics`` -> ``_call_radiation_backend`` ->
        ``_resolve_surface_field``. They stay differentiable while
        ``RRTMGPConfig.sfc_*`` remains the concrete default, so the cache key is
        never keyed on a tracer. The gas/aerosol fields are NOT trainable (they
        would need RRTMGP to consume them as per-call traced inputs too) and were
        dropped from the AIMIP trainable set rather than kept as dead leaves.
        """
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        return RRTMGPConfig()


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
        "louis_Ri_crit": float(lo.Ri_crit),
        "louis_b_louis": float(lo.b_louis),
        "louis_c_louis": float(lo.c_louis),
        "louis_d_louis": float(lo.d_louis),
        # Surface
        "surface_Cd_neutral": float(su.Cd_neutral),
        "surface_Ch_neutral": float(su.Ch_neutral),
        "surface_z0": float(su.z0),
        "surface_most_unstable_gamma": float(su.most_unstable_gamma),
        "surface_most_stable_beta": float(su.most_stable_beta),
        "surface_z0h_z0_ratio": float(su.z0h_z0_ratio),
        # McFarlane
        "mcfarlane_h_topo": float(mc.h_topo),
        "mcfarlane_G_0": float(mc.G_0),
        "mcfarlane_efficiency": float(mc.efficiency),
        "mcfarlane_min_wind": float(mc.min_wind),
        "mcfarlane_envelope_scale": float(mc.envelope_scale),
        "mcfarlane_k_wave": float(mc.k_wave),
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
        "sundqvist_RH_crit": float(sq.rh_crit),
        "sundqvist_sigmoid_sharpness": float(sq.sigmoid_sharpness),
        "sundqvist_auto_rate": float(sq.auto_rate),
        "sundqvist_evap_coeff": float(sq.evap_coeff),
        # SBM convection
        "sbm_tau_c": float(sbm.tau_c),
        "sbm_RH_ref": float(sbm.rh_ref),
        "sbm_CAPE_threshold": float(sbm.cape_threshold),
        # RRTMGP — only the surface knobs are trainable (routed as per-call
        # overrides, not config cache-key fields); gas/aerosol cache-key fields
        # are not trainable, so they carry no canonical-default entry here.
        "rrtmgp_sfc_emissivity": float(rr.sfc_emissivity),
        "rrtmgp_sfc_albedo": float(rr.sfc_albedo),
        # Shared
        "tau_equator": 7.2,
        "tau_pole": 1.8,
    }


# ----------------------------------------------------------------------
# Spectral-PE physics builder for AIMIP classical
# ----------------------------------------------------------------------

def spatial_baselines_from_params(d: dict, radiation: str) -> dict:
    """Map trained-scalar keys onto the spatial-surface FIELD names.

    ``AIMIPSpatialSurfaceParams.evaluate(baselines=...)`` expects the field
    names (``Cd_neutral``, ``Ch_neutral``, ``z0``, ``sfc_emissivity``,
    ``sfc_albedo``); the trained scalars live under ``surface_*`` and the
    radiation-scheme-specific ``{rrtmgp,gray}_sfc_*`` keys. The values are the
    sigmoid-bounded TRACED leaves, so the scalar knobs receive gradient through
    the spatial fields — and, because ocean columns fall back to the baseline,
    they are the only trainable ocean-surface levers.
    """
    rad = "rrtmgp" if radiation == "rrtmgp" else "gray"
    return {
        "Cd_neutral": d["surface_Cd_neutral"],
        "Ch_neutral": d["surface_Ch_neutral"],
        "z0": d["surface_z0"],
        "sfc_emissivity": d[f"{rad}_sfc_emissivity"],
        "sfc_albedo": d[f"{rad}_sfc_albedo"],
    }


def make_aimip_classical_spectral_physics(
    params: AIMIPClassicalParams,
    grid,
    dt: float,
    *,
    radiation: str = "gray",
    rad_update_interval_steps: int = 6,
    convection_scheme: str = "tiedtke",
    turbulence_scheme: str = "louis",
    surface_bulk_scheme: str = "constant",
    gwd_scheme: str = "mcfarlane",
    microphysics_scheme: str = "none",
    cloud_scheme: str = "xu_randall",
    land_mask: "jax.Array | None" = None,
    split_rad: bool = False,
    rrtmgp_gpoint_checkpoint: bool = True,
    rrtmgp_gpoint_batch_size: int = 16,
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
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.core.bulk_flux import validate_bulk_scheme

    # Dispatch hardening: reject an unknown surface bulk scheme at builder
    # entry on the static Python value (raises ValueError) so a YAML typo
    # cannot silently fall back to constant-Cd bulk aerodynamics — which
    # would leave the trained MOST stability / z0h-ratio leaves dead with no
    # error.  Mirrors compute_surface_fluxes' own validate_bulk_scheme gate.
    validate_bulk_scheme(surface_bulk_scheme)

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

    # ---- Spatial surface field columns (optional) ----
    # When the trainable params bundle carries an
    # :class:`AIMIPSpatialSurfaceParams` and a ``land_mask`` is
    # provided, evaluate each spatial field on the Gaussian grid and
    # flatten to (ncol,) so the column-shaped physics configs
    # (``RRTMGPConfig``, ``GrayRadiationConfig``, ``SurfaceLayerConfig``)
    # can accept them as broadcastable arrays.  Falls back to an
    # empty dict when the spatial-surface mode is off — every
    # downstream ``"<name>" in spatial_fields_col`` check then
    # short-circuits to False, preserving the global-scalar path.
    spatial_fields_col: dict[str, jax.Array] = {}
    if getattr(params, "spatial_surface", None) is not None and land_mask is not None:
        # Alias-map the trained scalar keys onto the SPATIAL FIELD names
        # (codex review): ``evaluate`` looks up ``Cd_neutral``/``sfc_albedo``
        # etc. while ``as_dict`` carries ``surface_Cd_neutral`` /
        # ``{rrtmgp,gray}_sfc_albedo``. Passing the raw dict left every
        # baseline at the STATIC f_0 -> the trained scalars were DEAD under
        # ``spatial_surface=True`` and (since ocean columns fall back to the
        # baseline) the ocean surface exchange/albedo had NO trainable lever.
        baselines = spatial_baselines_from_params(params.as_dict(), radiation)
        fields_2d = params.spatial_surface.evaluate(
            grid, land_mask=land_mask, baselines=baselines,
        )
        spatial_fields_col = {
            name: arr.reshape(-1) for name, arr in fields_2d.items()
        }

    # ---- Radiation ----
    # Surface overrides routed to the radiation solve as PER-CALL inputs
    # (make_physics -> make_radiation_physics -> _call_radiation_backend ->
    # _resolve_surface_field) rather than written into the radiation config.
    # For RRTMGP this is REQUIRED: RRTMGPConfig.sfc_* are folded into RRTMGP's
    # Python solver-cache key, so a trained value there would key the instance
    # cache by tracer/array identity. Routing as an override keeps the trained
    # surface field traceable+differentiable AND off the cache key.
    _sfc_albedo_override = None
    _sfc_emissivity_override = None
    if radiation == "rrtmgp":
        rrtmgp_cfg = params.to_rrtmgp_config()
        # Spatial (ncol,) field if present, else the trained SCALAR knob — both
        # routed as overrides so ``rrtmgp_sfc_albedo`` / ``rrtmgp_sfc_emissivity``
        # are genuinely trainable in BOTH the spatial AND the default non-spatial
        # path (without the scalar they would be dead leaves: to_rrtmgp_config
        # returns defaults and nothing else consumes them).
        _rr_d = params.as_dict()
        _sfc_albedo_override = spatial_fields_col.get(
            "sfc_albedo", _rr_d["rrtmgp_sfc_albedo"]
        )
        _sfc_emissivity_override = spatial_fields_col.get(
            "sfc_emissivity", _rr_d["rrtmgp_sfc_emissivity"]
        )
        # Derive the RRTMGP cloud gate from the selected cloud scheme.
        # ``include_clouds`` and ``cloud_scheme`` are independent knobs;
        # ``to_rrtmgp_config`` leaves ``include_clouds`` at its False default,
        # so without this a default AIMIP run (cloud_scheme='xu_randall')
        # would build cloud optics that the RRTMGP solver SILENTLY discards
        # → clear-sky radiation and a dead gradient through the trained
        # Xu-Randall cloud knobs (the cloud_config below exists precisely to
        # make cloud-radiation coupling trainable end-to-end).  Mirrors
        # ``physics_pipeline._build_rrtmgp_radiation_fn`` and is required by
        # the ``make_radiation_physics`` gate-consistency check.
        # G-point compile/memory tradeoff (exploit g-point sparsity):
        #  * gpoint_checkpoint=True, batch=0 (legacy): scan+checkpoint per
        #    g-point — memory-frugal but prevent_cse=True emits a distinct body
        #    per g-point (~Ng-fold code) => multi-hour GPU compile.
        #  * gpoint_checkpoint=False, batch=0: one reused body (fast compile)
        #    but backward holds ALL g-points' activations => OOM (113 GiB).
        #  * gpoint_batch_size>0 (e.g. 16): vmap g-points in blocks — ONE
        #    compiled block body (fast compile) holding only block_size
        #    g-points' activations (bounded memory). The middle ground that
        #    trains: fast compile AND fits memory.
        rrtmgp_cfg = rrtmgp_cfg._replace(
            include_clouds=(cloud_scheme != "none"),
            gpoint_checkpoint=rrtmgp_gpoint_checkpoint,
            gpoint_batch_size=rrtmgp_gpoint_batch_size,
        )
        rad_cfg = RadiationConfig(
            scheme="rrtmgp",
            rrtmgp=rrtmgp_cfg,
            cloud_scheme=cloud_scheme,
            cloud_config=cloud_cfg_trained,
            update_interval_steps=rad_update_interval_steps,
            diurnal_cycle=True,
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
            diurnal_cycle=True,
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
    # Thread the selected surface bulk scheme onto the surface config of
    # EVERY turbulence scheme (scalar AND spatial paths).  The override
    # INSTALLS the AIMIP-trained surface config
    # (``to_surface_config(bulk_scheme=...)``) onto ``inner_cfg.surface`` — so
    # every scheme carries the TRAINED Cd/Ch/z0 AND the trained MOST leaves
    # (most_unstable_gamma / most_stable_beta / z0h_z0_ratio), not the scheme
    # ``*Config`` DEFAULT surface (codex iter-2 finding #4: a non-Louis scheme
    # under "most" previously ran the default, non-trainable MOST coefficients).
    # Then any spatial (ncol,) Cd/Ch/z0 fields overwrite the scalar baselines.
    # For ``louis`` this is idempotent with ``to_louis_config`` (which already
    # installed the trained surface).  Schemes without a ``surface`` member
    # ("none") are returned unchanged.
    trained_surface = params.to_surface_config(bulk_scheme=surface_bulk_scheme)

    def _spatial_surface_override(inner_cfg):
        if not hasattr(inner_cfg, "surface"):
            return inner_cfg
        new_surface = trained_surface
        if spatial_fields_col:
            new_surface = new_surface._replace(
                Cd_neutral=spatial_fields_col.get(
                    "Cd_neutral", new_surface.Cd_neutral,
                ),
                Ch_neutral=spatial_fields_col.get(
                    "Ch_neutral", new_surface.Ch_neutral,
                ),
                z0=spatial_fields_col.get("z0", new_surface.z0),
            )
        return inner_cfg._replace(surface=new_surface)

    if turbulence_scheme == "louis":
        louis_cfg = _spatial_surface_override(
            params.to_louis_config(bulk_scheme=surface_bulk_scheme)
        )
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
        # Other schemes (mynn25, holtslag_boville, ysu, edmf, clubb_lite,
        # clubb, none).  Build the scheme's default sub-config, install the
        # trained surface (+ selected bulk scheme + any spatial fields) onto
        # its ``surface`` member, and place it in the matching TurbulenceConfig
        # field.  Without this, a non-Louis scheme selected together with
        # ``surface_bulk_scheme="most"`` would SILENTLY stay on the constant
        # surface path — the exact dead-knob failure this change fixes for
        # Louis (codex review, iter-1/iter-2).  ``clubb`` is materialised from
        # ``TurbulenceConfig.clubb`` (default None -> CLUBBConfig() with a
        # settable ``surface``) in the dispatcher, so it MUST be threaded here
        # too (iter-2 finding #6).  ``"none"`` carries no surface sub-config and
        # falls through to a bare TurbulenceConfig (documented no-op).
        from legoesm.atmosphere.physics.turbulence.config import (
            CLUBBLiteConfig,
            HoltslagBovilleConfig,
            MYNN25Config,
            TurbulentEDMFConfig,
            YSUConfig,
        )
        # CLUBBConfig lives in clubb.py (config.py only imports it under
        # TYPE_CHECKING to avoid a config<->clubb import cycle).
        from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
        _surface_scheme_configs = {
            "mynn25": MYNN25Config,
            "clubb_lite": CLUBBLiteConfig,
            "clubb": CLUBBConfig,
            "holtslag_boville": HoltslagBovilleConfig,
            "ysu": YSUConfig,
            "edmf": TurbulentEDMFConfig,
        }
        factory = _surface_scheme_configs.get(turbulence_scheme)
        if factory is not None:
            turb_cfg = TurbulenceConfig(
                scheme=turbulence_scheme,
                **{turbulence_scheme: _spatial_surface_override(factory())},
            )
        else:
            # "none" — no surface sub-config to thread the bulk scheme onto.
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

    physics_config = PhysicsConfig(
        radiation=rad_cfg,
        convection=conv_cfg,
        turbulence=turb_cfg,
        microphysics=micro_cfg,
        gravity_wave_drag=gwd_cfg,
    )

    # Combined (legacy) path: a single callable computes every-step
    # physics including radiation.  Returned when ``split_rad=False``
    # so the caller can dispatch through the simple
    # ``spectral_rollout(state, physics_fn, ...)`` branch.
    if not split_rad:
        combined_raw = make_physics(
            physics_config, model_type="spectral_pe", dt=dt,
            sfc_albedo_override=_sfc_albedo_override,
            sfc_emissivity_override=_sfc_emissivity_override,
        )

        def combined_fn(state, grid_, sigma_coord):
            result = combined_raw(state, grid_, sigma_coord)
            return result[0] if isinstance(result, tuple) else result

        return combined_fn

    # Rad-split path: separate non-radiative and radiative callables.
    # ``spectral_rollout`` gates the rad fn via ``lax.cond`` on the
    # scan step index so RRTMGP only fires every
    # ``rad_update_interval_steps`` steps.
    #
    # NOTE on the rad branch -- the combined-physics wrapper
    # (``make_physics`` -> ``_make_spectral_pe_combined``) does NOT
    # forward ``sim_time_seconds`` to the inner radiation callable, so
    # going through it would silently strip the diurnal-cycle time
    # offset that ``spectral_rollout`` and the multi-step rollout pass
    # in.  Call the spectral-PE radiation builder directly here so the
    # kwarg reaches ``_make_spectral_pe_radiation._physics_fn_core``.
    from legoesm.atmosphere.physics.microphysics.config import (
        MicrophysicsConfig,
    )
    from legoesm.atmosphere.physics.radiation.integration import (
        make_radiation_physics,
    )
    non_rad_cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=conv_cfg,
        turbulence=turb_cfg,
        microphysics=micro_cfg,
        gravity_wave_drag=gwd_cfg,
    )
    non_rad_raw = make_physics(non_rad_cfg, model_type="spectral_pe", dt=dt)
    rad_only_raw = make_radiation_physics(
        rad_cfg, "spectral_pe",
        sfc_albedo_override=_sfc_albedo_override,
        sfc_emissivity_override=_sfc_emissivity_override,
    )

    def non_rad_fn(state, grid_, sigma_coord, phys_state=None, forcing=None):
        # ``phys_state`` carries the prescribed-SST anchor for the
        # surface-flux / turbulence scheme via ``surface_T_sfc_override``
        # (the AMIP-inference path threads a per-month ERA5 SST here);
        # ``forcing`` is forwarded for any non-rad scheme that consumes it.
        # Both default ``None`` -> the free-running training/eval path
        # (``spectral_rollout`` calls ``non_rad_fn(state, grid, sigma)``),
        # which is byte-for-byte unchanged.
        result = non_rad_raw(
            state, grid_, sigma_coord, phys_state=phys_state, forcing=forcing,
        )
        return result[0] if isinstance(result, tuple) else result

    def rad_fn(state, grid_, sigma_coord, *, sim_time_seconds=0.0, forcing=None):
        # ``make_radiation_physics`` returns the per-module physics_fn
        # with signature
        # ``(state, grid, sigma_coord, grid_fields=None, sim_time_seconds=0.0)``
        # -> tendency-only.  The radiation module does not touch the
        # tracer pytree, so its output carries ``tracers=None``; the
        # non-rad combined wrapper attaches a tracer dict from Tiedtke
        # + microphysics.  ``_add_phys_tendencies`` performs a
        # tree_map that requires identical pytree structure -- so we
        # synthesise a zero-tendency tracer dict here when the input
        # state carries tracers, keeping the rad and non-rad
        # tendency pytrees structurally identical.
        # Always forward BOTH the in-rollout elapsed time and the traced
        # forcing dict.  The spectral radiation kernel resolves precedence
        # per key: forcing['day_of_year'/'seconds_of_day'] (AMIP-inference
        # path) override the sim_time thread ONLY when present, so a
        # forcing dict carrying just ghg_vmr (the classical-training GHG
        # pin) keeps the diurnal/seasonal sim_time cycle intact.  The old
        # either/or branch dropped sim_time_seconds whenever any forcing
        # arrived, freezing radiation time at the closure default.
        rad_out = rad_only_raw(
            state, grid_, sigma_coord,
            sim_time_seconds=sim_time_seconds, forcing=forcing,
        )
        if rad_out.tracers is None and state.tracers is not None:
            zero_tracers = {}
            for k, f in state.tracers.items():
                if hasattr(f, "data") and hasattr(f, "replace"):
                    zero_tracers[k] = f.replace(data=jnp.zeros_like(f.data))
                else:
                    zero_tracers[k] = jnp.zeros_like(f)
            rad_out = rad_out._replace(tracers=zero_tracers)
        return rad_out

    return non_rad_fn, rad_fn
