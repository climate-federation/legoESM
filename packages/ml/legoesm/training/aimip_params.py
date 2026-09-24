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

from legoesm import constants

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
    # tiedtke_cape_threshold REMOVED (#1417): the trigger sigmoid saturates
    # to exactly 1.0 in a convecting column and CAPE clamps to exactly 0 in a
    # stable one, so its gradient is EXACTLY zero in both regimes. It was
    # being optimised with no signal; its spec is tier 0 for the same reason.
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
    ParamConstraint("cloud_q_c_diagnostic", 1.0e-6, 5.0e-4, "sigmoid"),  # lower bound = spec/driver (1e-6)
    # Cloud particle effective radii (drive RRTMGP cloud optics).
    ParamConstraint("cloud_r_eff_liq", 5.0e-6, 30.0e-6, "sigmoid"),
    ParamConstraint("cloud_r_eff_ice", 10.0e-6, 100.0e-6, "sigmoid"),
]

# Gray two-stream radiation knobs (Frierson et al. 2006) — HISTORICAL.
# These were trained until 2026-08-11, when the directive "remove gray
# radiation, we will never train it" retired the whole set (the seven gray_*
# plus ``tau_equator``/``tau_pole``). Radiation is now trained through RRTMGP's
# surface albedo/emissivity instead.
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
# Gray radiation carries no trainable OPTICAL knob (user directive 2026-08-11:
# "remove gray radiation, we will never train it"): no optical depth, moisture
# coefficient, diffusivity factor or shortwave knob is trained. The SURFACE
# albedo / emissivity remain trainable — they are properties of the surface,
# not of the radiation scheme, and they are what the directive asked to keep
# ("add surface albedo and emissivity in addition to surface roughness").
# Precisely, under GRAY: no scalar gray leaf is trained (the AIMIP surface
# scalars are RRTMGP-scoped, so the baselines fall back to
# GrayRadiationConfig's defaults), but the learned SPATIAL field coefficients
# do replace gray's sfc_emissivity / sfc_albedo, and the flat lat-lon set
# (trainable_params.DEFAULT_TRAINABLE) still trains scalar albedo_ice /
# albedo_ocean, which gray's shortwave consumes. Frozen means gray's OPTICAL
# knobs, not every number the scheme reads. It stays available as
# the cheap fixed backend for smokes, at its documented defaults;
# the classical model trains RRTMGP instead, where the only trainable
# radiative knobs are the surface albedo and emissivity below. The nine
# former knobs (7 gray_* + tau_equator/tau_pole, which ARE gray optical
# depths) sit at tunable_tier 0 in GrayRadiationConfig.__param_spec__, which
# build_trainable_params never selects (it takes 1 <= tier <= level), so the
# spec-driven collector cannot re-expose them either. Their BOUNDS stay, because
# the LES feedback loop promotes gray_tau_* to per-column fields and clamps that
# diagnosis to exactly those ranges.
_GRAY_RAD_TRAINABLE: list[ParamConstraint] = []

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
    # sbm_CAPE_threshold REMOVED (#1417): same AD-unreachable trigger as
    # tiedtke_cape_threshold — see the note there.
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
_AIMIP_COMMON_TRAINABLE: list[ParamConstraint] = []   # see _GRAY_RAD_TRAINABLE


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
            # cape_threshold is NOT overridden: #1417 dropped it from
            # _TIEDTKE_TRAINABLE (AD-unreachable trigger), so it is absent
            # from as_dict() and TiedtkeConfig's published default stands.
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
        # The AIMIP lanes substitute the lowest air temperature for the
        # surface wherever no surface is supplied (free-running spectral
        # physics; NaN over land in the SST-anchored forcing), and the MOST
        # height adjustment against such a "surface" invents an air-surface
        # contrast and a downward sensible heat flux out of nothing.  Keep it
        # off here (the scheme default is True) until this lane carries a
        # real surface temperature everywhere.
        base = SurfaceLayerConfig(z_ref_model_level=False)
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
            # cape_threshold is NOT overridden: #1417 dropped it from
            # _SBM_TRAINABLE (AD-unreachable trigger), so it is absent from
            # as_dict() and SBMConfig's published default stands.
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
        RRTMGPConfig,
    )

    t = TiedtkeConfig()
    lo = LouisConfig()
    su = SurfaceLayerConfig()
    mc = McFarlaneConfig()
    cl = CloudConfig()
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
    base = {
        "Cd_neutral": d["surface_Cd_neutral"],
        "Ch_neutral": d["surface_Ch_neutral"],
        "z0": d["surface_z0"],
    }
    if radiation == "rrtmgp":
        base["sfc_emissivity"] = d["rrtmgp_sfc_emissivity"]
        base["sfc_albedo"] = d["rrtmgp_sfc_albedo"]
        return base
    # Gray carries no trained surface knobs since 2026-08-11 (it is not
    # trained at all), so the spatial field is anchored on the scheme's own
    # published defaults instead of on a trained scalar. Reading the removed
    # ``gray_sfc_*`` leaves here raised KeyError for every
    # spatial_surface=True gray run (codex).
    from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
    gray = GrayRadiationConfig()
    base["sfc_emissivity"] = gray.sfc_emissivity
    base["sfc_albedo"] = gray.sfc_albedo
    return base


# Family field on PhysicsConfig -> registry namespace prefix. The prefix is a
# GUARD, not a lookup: class names are not unique across components (the ocean
# registers a TKEConfig too, and matching on the bare name once collected the
# ocean's parameters and tried to splice them into the atmosphere's config).
# Same lesson as run_scm_les_turbulence_tuning._scheme_keys_for.
# A family may span SEVERAL namespaces: the two bin/particle microphysics
# schemes publish their specs under ``atm.sdm.`` / ``atm.fastsbm.`` rather than
# ``atm.micro.``, so a single-prefix guard silently derived NO key for them and
# they trained nothing.
_AIMIP_FAMILY_NAMESPACES: dict[str, tuple[str, ...]] = {
    "convection": ("atm.conv.",),
    "turbulence": ("atm.turb.",),
    "gravity_wave_drag": ("atm.gwd.",),
    "microphysics": ("atm.micro.", "atm.sdm.", "atm.fastsbm."),
    "radiation": ("atm.rad.",),
}


def aimip_active_scheme_keys(physics_config) -> set[str]:
    """Registry ``scheme_key`` set for the schemes a BUILT config actually runs.

    Derived from the config TREE, never from a hardcoded scheme list, so ANY
    parameterization — including one added after this function was written —
    is trainable the moment it carries a ``__param_spec__``. Generalises
    ``run_scm_les_turbulence_tuning._scheme_keys_for`` from turbulence-only to
    every physics family.

    For each family the ACTIVE sub-config is ``getattr(family, family.scheme)``
    — the union config holds every scheme's sub-config, so only the selected
    one may contribute — and a ``.params`` member is followed one level down
    (full CLUBB nests its closure coefficients in a ``CLUBBParams``, which is
    what carries the spec).
    """
    from legoesm.training.param_collector import build_registry

    registry = build_registry()
    keys: set[str] = set()
    for family, prefixes in _AIMIP_FAMILY_NAMESPACES.items():
        fam_cfg = getattr(physics_config, family, None)
        scheme = getattr(fam_cfg, "scheme", None)
        if fam_cfg is None or not scheme or scheme == "none":
            continue
        # A COMPOSITE scheme runs every part: the GWD executor splits
        # ``"mcfarlane+hines"`` on '+' (gravity_wave_drag/integration.py:167,
        # 199), so a whole-string getattr resolves to None and the arm would
        # derive no GWD key at all — training nothing, silently.
        for part in str(scheme).split("+"):
            active = getattr(fam_cfg, part, None)
            if active is None:
                continue
            # Descend into a nested ``params`` holder when that is what carries
            # the spec (CLUBBConfig.params -> CLUBBParams).
            target = getattr(active, "params", None)
            if target is None or not hasattr(target, "_fields"):
                target = active
            cls = type(target).__name__
            keys |= {m.scheme_key for m in registry
                     if m.config_class == cls
                     and m.scheme_key.startswith(prefixes)}
    return keys


# The six parameterization families a CLASSICAL model must fill. Composability
# is the point of the classical variant: any scheme may be swapped for another
# of the same family, but a family may never be EMPTY — a run missing (say)
# microphysics is not a cheaper classical model, it is a different and
# incomparable one, and it silently invalidates a scheme-swap comparison
# against runs that have it. User directive, 2026-08-11.
CLASSICAL_SCHEME_FAMILIES = (
    "convection", "turbulence", "cloud", "microphysics", "radiation", "gwd",
)

# The default scheme per family, in ONE place. Training and evaluation used to
# carry their own copies of these strings and had already drifted: microphysics
# defaulted to "none" on the eval side while training used a real scheme, so an
# omitted key trained one model and scored another (codex).
CLASSICAL_DEFAULT_SCHEMES = {
    "convection": "tiedtke",
    "turbulence": "louis",
    "cloud": "xu_randall",
    "microphysics": "sundqvist",
    "radiation": "rrtmgp",
    "gwd": "mcfarlane",
}

_UNFILLED = ("", "none", "off", "false")

# --- Per-rollout TKE seed (training-lane seeding POLICY, not scheme
# physics; GLM 2026-08-17 review, option c) -------------------------------
# ``init_physics_state`` seeds the TKE slot at the scheme floor
# (~1e-6 m^2/s^2). Turbulence production scales with sqrt(TKE), so a floor
# seed cannot spin up inside a 6-h training window (measured: threading the
# memory moved the 12-step loss by only 5e-4 relative) — the mixing stays
# effectively absent. The seed below evaluates the NEUTRAL
# production-dissipation balance on the initial state,
#     c_K * L * sqrt(w) * S^2 = w^(3/2) / L   =>   w = c_K * L^2 * S^2,
# with fixed policy constants: TKE is quasi-equilibrium (tau ~ minutes to
# tens of minutes), so any physically-scaled seed relaxes to the scheme's
# own balance within a few steps — the seed's job is escaping the sqrt
# bottleneck, not being exact. The active scheme's own c_K IS used when
# available — DETACHED via stop_gradient, so the seed tracks the trained
# equilibrium without the trained value retro-coupling into its own
# initial condition (GLM round 2); the constants below are the fallback
# policy values and the fixed length/clip band, deliberately not tunables.
_TKE_SEED_CK = 0.5          # representative eddy-diffusivity coefficient
_TKE_SEED_LENGTH_M = 100.0  # neutral-BL mixing-length scale [m]
_TKE_SEED_FLOOR = 1.0e-6    # scheme tke_min class floor [m^2/s^2]
_TKE_SEED_CAP = 10.0        # sanity cap [m^2/s^2] (jet shear layers)


def _apply_wp2_seed(ps, state, grid_, sigma_coord, c_k=None):
    """Replace ``ps.tke`` with the shear-equilibrium wp2 seed (see above).

    ``c_k``: the ACTIVE scheme's eddy coefficient, DETACHED
    (``lax.stop_gradient``) by the caller — the seed then matches the
    scheme's own equilibrium as training moves c_K, without the trained
    value retro-coupling into its own initial condition (GLM: a fixed
    policy c_K guarantees a per-window adjustment transient the loss
    would mis-attribute to the sink terms). ``None`` falls back to the
    fixed policy constant. Bechtold's organization profile and the GWD
    spectrum keep their cold defaults — no diagnostic exists for them
    (named open item).
    """
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        spectral_pe_to_grid,
    )

    if ps.tke is None:
        return ps
    ncol = int(grid_.n_lat) * int(grid_.n_lon)
    nlev = int(jnp.shape(sigma_coord.sigma_full)[0])
    fields = spectral_pe_to_grid(state, grid_, sigma_coord)
    u = fields["u"].reshape(ncol, nlev)
    v = fields["v"].reshape(ncol, nlev)
    T = fields["T"].reshape(ncol, nlev)
    # Layer separation dz ~ (R_d T / g) * dln(p): the grid converter exposes
    # no heights, and a seed needs only order-of-magnitude shear. sigma_full
    # is static, so dln p is a trace-time constant.
    sig = jnp.asarray(sigma_coord.sigma_full)
    dlnp = jnp.abs(jnp.log(sig[1:]) - jnp.log(sig[:-1]))      # (nlev-1,)
    T_half = 0.5 * (T[:, :-1] + T[:, 1:])
    dz = jnp.clip((constants.R_d / constants.g) * T_half * dlnp, 1.0, None)
    s2_half = (((u[:, :-1] - u[:, 1:]) / dz) ** 2
               + ((v[:, :-1] - v[:, 1:]) / dz) ** 2)
    # Interior interfaces -> full levels by edge-replicated averaging (the
    # same half->full stencil the diagnostic scheme itself uses).
    s2 = jnp.concatenate(
        [s2_half[:, :1],
         0.5 * (s2_half[:, :-1] + s2_half[:, 1:]),
         s2_half[:, -1:]], axis=1)
    # 1-2-1 vertical smoothing: raw per-layer S^2 from analysis winds is
    # grid-noisy, and a noisy seed imprints spurious layer-scale K at step 1
    # (GLM). Replicated-edge convolution: the endpoints blend 3/4-1/4 with
    # their neighbour (leaving them raw skips the smoothing exactly at the
    # surface, where the seed matters most — codex round 2).
    s2 = jnp.concatenate(
        [0.75 * s2[:, :1] + 0.25 * s2[:, 1:2],
         0.25 * s2[:, :-2] + 0.5 * s2[:, 1:-1] + 0.25 * s2[:, 2:],
         0.25 * s2[:, -2:-1] + 0.75 * s2[:, -1:]], axis=1)
    _ck = _TKE_SEED_CK if c_k is None else jax.lax.stop_gradient(c_k)
    wp2_seed = jnp.clip(_ck * _TKE_SEED_LENGTH_M ** 2 * s2,
                        _TKE_SEED_FLOOR, _TKE_SEED_CAP)
    return ps._replace(tke=wp2_seed.astype(ps.tke.dtype))


def validate_classical_scheme_set(
    *, convection: str, turbulence: str, cloud: str, microphysics: str,
    radiation: str, gwd: str, allow_unfilled: bool = False,
) -> dict[str, str]:
    """Require one active scheme in EVERY classical family; return the set.

    Parameters are the scheme NAMES as the runner resolves them (the
    ``aimip_<family>`` config keys). ``"none"`` / ``""`` count as unfilled.

    ``allow_unfilled=True`` is the ABLATION escape: a study whose whole point is
    "run without gravity-wave drag" is legitimate, but it must say so
    explicitly, because the resulting model is not comparable to a complete
    one. Production training leaves it False.

    Raises
    ------
    ValueError
        Naming EVERY unfilled family at once — fixing them one error at a time
        would cost one job submission per family.

    Returns
    -------
    dict
        ``{family: scheme}``, so a caller can log exactly what it validated.
    """
    selected = {
        "convection": convection, "turbulence": turbulence, "cloud": cloud,
        "microphysics": microphysics, "radiation": radiation, "gwd": gwd,
    }
    missing = [f for f in CLASSICAL_SCHEME_FAMILIES
               if str(selected[f]).strip().lower() in _UNFILLED]
    if missing and not allow_unfilled:
        raise ValueError(
            "a classical model needs one parameterization of EVERY family "
            f"{list(CLASSICAL_SCHEME_FAMILIES)}; unfilled: {missing}. "
            f"Selected: {selected}. Set the aimip_<family> key for each "
            "(scheme swaps are what the classical variant is for; an empty "
            "family is a different model, not a smaller one)."
        )
    return selected


def aimip_scheme_keys_for(
    *, convection: str = "none", turbulence: str = "none",
    gwd: str = "none", microphysics: str = "none", radiation: str = "none",
    cloud: str = "none",
) -> set[str]:
    """``aimip_active_scheme_keys`` from SCHEME NAMES rather than a built tree.

    The runner knows the names before any physics is built, so it needs this
    entry point. A minimal probe ``PhysicsConfig`` is assembled from the real
    config classes here — in the SAME module as the factory — so the knowledge
    of which sub-config a scheme name materialises lives in one place instead
    of being duplicated into the run script.

    Only the ``scheme`` selector and the active sub-config's TYPE are read, so
    the probe needs no tuned values.
    """
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.microphysics.config import (
        MicrophysicsConfig,
    )
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig

    turb_kw = {}
    if turbulence not in ("none", ""):
        # Sub-configs that default to None must be materialised or the family
        # contributes nothing (full CLUBB is the case that matters).
        from legoesm.atmosphere.physics.turbulence import config as _tc
        if turbulence == "clubb":
            from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
            turb_kw["clubb"] = CLUBBConfig()
        else:
            factory = {
                "louis": _tc.LouisConfig, "tke": _tc.TKEConfig,
                "mynn25": _tc.MYNN25Config,
                "smagorinsky": _tc.SmagorinskyConfig,
                "clubb_lite": _tc.CLUBBLiteConfig,
                "holtslag_boville": _tc.HoltslagBovilleConfig,
                "ysu": _tc.YSUConfig, "edmf": _tc.TurbulentEDMFConfig,
            }.get(turbulence)
            if factory is not None:
                turb_kw[turbulence] = factory()
    probe = PhysicsConfig(
        radiation=RadiationConfig(scheme=radiation),
        convection=ConvectionConfig(scheme=convection),
        turbulence=TurbulenceConfig(scheme=turbulence, **turb_kw),
        microphysics=MicrophysicsConfig(scheme=microphysics),
        gravity_wave_drag=GravityWaveDragConfig(scheme=gwd),
    )
    keys = aimip_active_scheme_keys(probe)
    # Cloud fraction is not a PhysicsConfig family — the factory places the
    # (single, scheme-shared) CloudConfig under RadiationConfig.cloud_config,
    # and ONLY on the rrtmgp path (gray radiation carries no CloudConfig in
    # its tree, so a cloud override there raises 'matched no config' — caught
    # by the six-suite smoke, 2026-08-10). Route it explicitly so cloud spec
    # parameters are collectable whenever they have somewhere to land.
    if cloud not in ("none", "") and radiation == "rrtmgp":
        from legoesm.training.param_collector import build_registry
        keys |= {m.scheme_key for m in build_registry()
                 if m.scheme_key.startswith("atm.clouds.")}
    # NOT ROUTED: atm.rad.OzoneProfileConfig. It was added here on 2026-08-10
    # and REVERTED the same day, for two independent reasons — leave it out
    # until BOTH are addressed:
    #   1. INERT. The AIMIP factory builds RadiationConfig without setting an
    #      ozone source, so it keeps OzoneProfileConfig(source="standard");
    #      the analytic branch in radiation/integration.py returns None for
    #      that source and RRTMGP falls back to its own built-in
    #      climatological profile. Training p_peak_hPa / o3_max_vmr /
    #      sigma_logp would have had ZERO forward effect — a parameter that
    #      reaches the config but not the model. Routing it needs
    #      source="analytical" wired first.
    #   2. IMPLAUSIBLE AT THIS RESOLUTION. The 8-level sigma grid's highest
    #      FULL level is ~109 hPa (top interface 50 hPa) while the spec's
    #      p_peak_hPa bounds are (1, 100) hPa — the trainable ozone peak sits
    #      above the model's entire domain, leaving only the tail of the
    #      Gaussian inside. There is ~1 level of stratosphere to heat, and no
    #      credible path from it to an 850 hPa temperature drift.
    return keys


def aimip_legacy_owned_scheme_keys() -> set[str]:
    """Registry ``scheme_key``s the LEGACY hand-written params already train.

    OWNERSHIP RULE (design call, 2026-08-06): a config field has exactly ONE
    trainer.  ``AIMIPClassicalParams.to_<x>_config`` writes these classes'
    fields from legacy leaves and ``_splice_scheme_overrides`` runs AFTER it,
    so letting the spec-driven collector also cover such a class would
    overwrite the legacy value and silently zero those gradients — trading the
    46 leaves that train today for new ones, with no error.  The spec route
    therefore covers only the schemes the legacy route CANNOT reach (Bechtold,
    CLUBB, Thompson, ...), which is why the opt-in exists.

    Legacy wins the overlap rather than the spec because the alternative —
    dropping the colliding legacy ``ParamConstraint``s — changes the
    ``AIMIPClassicalParams`` pytree layout and invalidates every existing
    classical checkpoint, for no additional trained parameter.

    The owned set is derived by CALLING every ``to_*_config`` method and
    reading the returned TYPE, so a method added later is owned automatically
    (a hardcoded class list would rot into a silent gradient loss).
    """
    from legoesm.training.param_collector import build_registry

    probe = AIMIPClassicalParams.from_defaults()
    owned_classes = {
        type(getattr(probe, name)()).__name__
        for name in dir(type(probe))
        if name.startswith("to_") and name.endswith("_config")
    }
    return {m.scheme_key for m in build_registry()
            if m.config_class in owned_classes}


def aimip_legacy_owned_fields(*, cloud_scheme: str = "xu_randall") -> set[str]:
    """Qualified ``scheme_key.field`` names the legacy leaves actually WRITE.

    Field-level refinement of :func:`aimip_legacy_owned_scheme_keys`: the
    class-level subtraction excluded EVERY spec parameter of a class the
    legacy route touches, which suppressed spec-only fields the legacy never
    writes (Sundqvist ``qc_crit``, McFarlane ``fcrit2``, most of
    ``CloudConfig``) — they trained nowhere. The one-trainer ownership rule
    only requires excluding the FIELDS the legacy ``to_*_config`` methods
    populate, because ``_splice_scheme_overrides`` runs after them and would
    overwrite exactly those.

    Derived mechanically (no hardcoded list to rot): every raw leaf is
    perturbed in unconstrained space and each ``to_*_config`` output is
    diffed field-by-field against the unperturbed build. A field that moves
    is legacy-written; one that stays at its default is free for the spec
    route. Monotonicity alone does not survive float32 (GLM review): a
    deeply saturated logit could absorb the perturbation below
    representation. ``from_defaults`` interiorizes every leaf to >=5% of its
    sigmoid range (|raw| <= logit(0.95) ~ 2.94), where +0.37 moves the
    constrained value by >~1% of the range — far above float32 resolution —
    and the assertion below turns any future saturated default into a loud
    failure instead of a silent misclassification.
    """
    import numpy as np

    from legoesm.training.param_collector import build_registry

    probe = AIMIPClassicalParams.from_defaults()
    for k, v in probe.raw_values.items():
        if float(jnp.max(jnp.abs(v))) > 6.0:   # sigmoid slope ~2.5e-3 there
            raise AssertionError(
                f"raw leaf {k!r} is saturated (|raw|>6); the perturb-and-diff "
                "ownership derivation would silently misclassify it — "
                "interiorize the default (see from_defaults sigmoid_margin).")
    perturbed = eqx.tree_at(
        lambda p: p.raw_values, probe,
        {k: v + 0.37 for k, v in probe.raw_values.items()},
    )
    registry = build_registry()
    known = {m.qualified_name for m in registry}
    key_by_class: dict[str, list[str]] = {}
    for m in registry:
        key_by_class.setdefault(m.config_class, []).append(m.scheme_key)

    owned: set[str] = set()
    for name in dir(type(probe)):
        if not (name.startswith("to_") and name.endswith("_config")):
            continue
        # The factory routes to_cloud_config's leaves ONLY under
        # cloud_scheme == "xu_randall"; every other cloud scheme starts from
        # CloudConfig defaults, so its spec fields (rh_crit above all) must
        # stay collectable there (codex: the sundqvist-cloud arm would
        # otherwise lose its primary control). The other conditional routes
        # (tiedtke/sbm conv, gray radiation, louis/mcfarlane/sundqvist-micro)
        # have their class active only when they are also legacy-routed, so
        # an unconditional exclusion is harmless for them.
        if name == "to_cloud_config" and cloud_scheme != "xu_randall":
            continue
        cfg_a = getattr(probe, name)()
        cfg_b = getattr(perturbed, name)()
        for field in cfg_a._fields:
            va, vb = getattr(cfg_a, field), getattr(cfg_b, field)
            try:
                same = bool(np.array_equal(np.asarray(va), np.asarray(vb)))
            except (TypeError, ValueError):
                same = va == vb
            if not same:
                for scheme_key in key_by_class.get(type(cfg_a).__name__, []):
                    q = f"{scheme_key}.{field}"
                    # Only registry-known names: a legacy-written field with
                    # no __param_spec__ entry cannot collide with the spec
                    # route, and build_trainable_params(exclude=...) raises
                    # on unknown names.
                    if q in known:
                        owned.add(q)
    return owned


class AIMIPTrainableBundle(eqx.Module):
    """The classical arm's trained model when generic scheme params are on.

    Holds the legacy hand-written ``AIMIPClassicalParams`` AND a spec-driven
    ``TrainablePhysicsParams`` for the ACTIVE schemes, so
    ``eqx.filter_value_and_grad`` differentiates BOTH. Used only when a suite
    opts in (``aimip_trainable_schemes``); without it the trained model stays a
    bare ``AIMIPClassicalParams`` and every existing checkpoint keeps its
    layout.
    """
    classical: "AIMIPClassicalParams"
    schemes: object


def unpack_aimip_params(p):
    """``p -> (AIMIPClassicalParams, overrides_or_None)``.

    One place decides how the trained model is shaped, so the factory call
    site cannot drift from the runner's construction.
    """
    if isinstance(p, AIMIPTrainableBundle):
        return p.classical, p.schemes.to_overrides()
    return p, None


def _splice_scheme_overrides(node, overrides: dict, _ctx=None):
    """Splice ``{scheme_key: {field: value}}`` into a config NamedTuple tree.

    ``scheme_key`` is the param registry's ``<module>.<ClassName>`` (what
    ``TrainablePhysicsParams.to_overrides()`` returns). Each key is resolved
    through ``build_registry`` to the DEFINING ``(module, class)`` pair and
    matched on both — never on the bare class name, because the same name
    exists in different components (an atmosphere ``TKEConfig`` and an ocean
    ``TKEConfig``), and name-only matching would cross-route between them.
    This mirrors ``driver.run_config_yaml._route_overrides_by_class``, which is
    private and in another package (cross-package private imports are
    forbidden); each splice delegates to the shared
    ``core.param_overrides.apply_param_overrides``, which raises on an unknown
    field, rather than re-deriving ``_replace``.

    Two refusals, both because a trained parameter that fails to reach the
    model — or reaches the wrong copy of it — is the failure this exists to
    remove:

    * a key matching NO config in the tree raises;
    * a key matching the same ``(module, class)`` MORE THAN ONCE raises as
      AMBIGUOUS. ``TurbulenceConfig`` instantiates every scheme's sub-config
      and each carries its own ``SurfaceLayerConfig``, so a bare-name walk
      silently spliced all of them; only the active scheme's copy is read, so
      "applied" would not have meant "used".
    """
    from legoesm.core.param_overrides import apply_param_overrides

    top = _ctx is None
    if top:
        from legoesm.training.param_collector import build_registry
        by_key = {}
        for m in build_registry():
            if m.scheme_key:
                by_key.setdefault(m.scheme_key, (m.module, m.config_class))
        unknown = sorted(set(overrides) - set(by_key))
        if unknown:
            raise ValueError(
                f"override key(s) {unknown} are not in the parameter registry."
            )
        _ctx = {"targets": {k: by_key[k] for k in overrides}, "hits": {}}

    fields = getattr(node, "_fields", None)
    if fields is not None and isinstance(node, tuple):
        repl = {}
        for f in fields:
            child = getattr(node, f)
            new_child = _splice_scheme_overrides(child, overrides, _ctx)
            if new_child is not child:
                repl[f] = new_child
        if repl:
            node = node._replace(**repl)
        ident = (type(node).__module__, type(node).__name__)
        for key, target in _ctx["targets"].items():
            if ident == target:
                _ctx["hits"][key] = _ctx["hits"].get(key, 0) + 1
                node = apply_param_overrides(node, overrides[key])

    if top:
        missing = sorted(k for k in overrides if not _ctx["hits"].get(k))
        if missing:
            raise ValueError(
                f"trained parameter override(s) for {missing} matched no "
                f"config in the built physics tree. The class is absent "
                f"entirely — a stale/typo'd key, an unsupported component, or "
                f"a sub-config this scheme does not materialise. (An INACTIVE "
                f"but instantiated scheme would have matched.)"
            )
        dup = sorted(k for k, n in _ctx["hits"].items() if n > 1)
        if dup:
            raise ValueError(
                f"override key(s) {dup} matched MORE THAN ONE instance of "
                f"their config class in the physics tree; only the active "
                f"scheme's copy is read, so the splice is ambiguous. Route "
                f"them through the owning scheme's config instead."
            )
    return node


def make_aimip_classical_spectral_physics(
    params,
    grid,
    dt: float,
    *,
    param_overrides: dict | None = None,
    # WB/AIMIP always run rrtmgp (2026-08-12). Gray stays reachable for a
    # --smoke wiring check and for non-campaign callers, but it is no longer
    # what you get by omission.
    radiation: str = "rrtmgp",
    rad_update_interval_steps: int = 6,
    convection_scheme: str = "tiedtke",
    turbulence_scheme: str = "louis",
    surface_bulk_scheme: str = "constant",
    gwd_scheme: str = "mcfarlane",
    microphysics_scheme: str = "sundqvist",
    cloud_scheme: str = "xu_randall",
    # CAM ``trop_cloud_top_press`` [Pa] for turbulence_scheme="clubb": the
    # pressure above which CLUBB's mixing tapers to zero. None keeps
    # CLUBBConfig's default 0.0 = OFF (CAM's own code default; the taper is a
    # no-op branch); the 32-level WB arm sets 5000 Pa (see CLUBBConfig
    # docstring).
    clubb_top_press: float | None = None,
    # Ablations that deliberately drop a family set this True; production
    # training does not (see validate_classical_scheme_set).
    allow_unfilled_families: bool = False,
    land_mask: "jax.Array | None" = None,
    # Production AMIP settings that live on the FAMILY config rather than on a
    # scheme's own -- so the registry-keyed override route cannot address them,
    # and they would otherwise sit silently at their library defaults while a
    # deck claimed to match production.
    orbital_insolation: bool = False,
    convective_rain_to_surface: bool = False,
    split_rad: bool = False,
    rrtmgp_gpoint_checkpoint: bool = True,
    rrtmgp_gpoint_batch_size: int = 16,
    rrtmgp_column_chunk_size: int = 0,
):
    """Build a SpectralPE physics function for the AIMIP classical variant.

    Constructs a ``PhysicsConfig`` by reading the current ``params``
    Equinox leaves and assembling Tiedtke / Louis / McFarlane scheme
    configs from them, then dispatches through ``combined.make_physics``
    with ``model_type='spectral_pe'``.  The returned callable matches
    the signature consumed by :func:`spectral_rollout`:

        physics_fn(state, grid, sigma_coord) -> SpectralHydrostaticState

    Radiation is RRTMGP in production; gray remains selectable as the cheap
    backend but carries NO trainable optical knob since 2026-08-11. The
    radiative gradients a classical model gets come from the surface albedo /
    emissivity leaves, which reach the solver as per-call overrides.

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
    # Enforced HERE, not in a single runner: the ablation drivers, the AMIP
    # finetune and the WB spectral lane all build their classical physics
    # through this factory, and a runner-only gate left every one of them
    # unchecked (codex). NOT universal: the lat-lon carry lane
    # (run_aimip_latlon -> training_driver) builds physics through
    # physics_pipeline and never reaches here.
    validate_classical_scheme_set(
        convection=convection_scheme, turbulence=turbulence_scheme,
        cloud=cloud_scheme, microphysics=microphysics_scheme,
        radiation=radiation, gwd=gwd_scheme,
        allow_unfilled=allow_unfilled_families,
    )
    # ``params`` may be a bare AIMIPClassicalParams (legacy) or an
    # AIMIPTrainableBundle carrying spec-driven scheme params too. Unpacking
    # HERE means every caller — runner, tests, eval — gets the same behaviour
    # without knowing which shape it holds.
    params, _bundle_overrides = unpack_aimip_params(params)
    if _bundle_overrides is not None:
        # MERGED PER FIELD, not per scheme_key: both sides address the same
        # config by key, so a shallow merge silently DROPPED every trained
        # field of any config the caller also pinned a fixed value on -- the
        # cloud config being exactly that case (tuned overlap/sub-columns
        # alongside trained rh_crit). The caller's value wins field by field.
        merged = {k: dict(v) for k, v in _bundle_overrides.items()}
        for key, fields in (param_overrides or {}).items():
            merged.setdefault(key, {}).update(fields)
        param_overrides = merged

    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.radiation.solar import earth_orbit
    from legoesm.atmosphere.physics.radiation.config import (
        GrayRadiationConfig, RadiationConfig,
    )
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
    # cheap, no cloud coupling, and NOT trained (2026-08-11) — no
    # gradient reaches any of its OPTICAL knobs. (Under
    # spatial_surface=True the learned surface albedo/emissivity FIELD
    # is still substituted into its config; that is a surface
    # property, not a radiation-scheme knob.)  Default ``gray`` keeps the
    # AIMIP harness tractable on a single GPU; bump to ``rrtmgp``
    # for production-grade physics realism.
    # Realistic (elliptical) orbit, as the production AMIP deck runs; None
    # keeps the circular-orbit default every earlier caller had.
    _orbit = earth_orbit() if orbital_insolation else None

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
            column_chunk_size=rrtmgp_column_chunk_size,
        )
        rad_cfg = RadiationConfig(
            scheme="rrtmgp",
            rrtmgp=rrtmgp_cfg,
            cloud_scheme=cloud_scheme,
            cloud_config=cloud_cfg_trained,
            # cam6_clubb has exactly one legal value here (the radiation
            # builder refuses the scheme without it), so this is derived,
            # not a knob.  Every other cloud scheme keeps the RH path.
            use_clubb_cloud_fraction=(cloud_scheme == "cam6_clubb"),
            update_interval_steps=rad_update_interval_steps,
            diurnal_cycle=True,
            orbit=_orbit,
        )
    elif radiation == "gray":
        # Gray radiation runs at its published defaults — it is NOT
        # trained (2026-08-11 directive).  Only the SPATIAL surface fields
        # below are substituted, and those come from the surface knobs, not
        # from any gray-specific one.
        gray_cfg = GrayRadiationConfig()
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
            orbit=_orbit,
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
    if convective_rain_to_surface:
        # Production routes convective rain straight to the surface instead of
        # detraining it into the resolved rain field for the microphysics to
        # re-handle; the two give different surface precipitation and
        # different re-evaporation.
        conv_cfg = conv_cfg._replace(rain_to_surface=True)

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
            _turb_sub = factory()
            if turbulence_scheme == "clubb" and clubb_top_press is not None:
                # CAM trop_cloud_top_press override (see the kwarg above).
                _turb_sub = _turb_sub._replace(
                    trop_cloud_top_press=float(clubb_top_press))
            turb_cfg = TurbulenceConfig(
                scheme=turbulence_scheme,
                **{turbulence_scheme: _spatial_surface_override(_turb_sub)},
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

    # SCHEME-AGNOSTIC trained parameters (the generic route).
    #
    # The ``to_*_config`` methods above are hand-written per scheme, so a
    # scheme with no method — Bechtold, CLUBB, Thompson, ... — silently ran at
    # its defaults and received no gradient. ``TrainablePhysicsParams`` is the
    # spec-driven alternative: ``build_trainable_params(active_scheme_keys=...)``
    # collects every ``__param_spec__`` parameter of whatever schemes are
    # ACTIVE, and ``to_overrides()`` returns them keyed by scheme. Measured at
    # tier "extended": edmf+louis = 46 trainable leaves, bechtold+clubb = 102.
    #
    # Applied HERE, after the tree is assembled and BEFORE make_physics, so the
    # spliced leaves are traced inside the loss (SegmentForcing doctrine) and
    # both the combined and rad-split paths below inherit them.
    if param_overrides:
        physics_config = _splice_scheme_overrides(
            physics_config, param_overrides)
        # The rad-split branch rebuilds a PhysicsConfig from the ORIGINAL
        # sub-configs, so re-read the spliced ones rather than letting that
        # path silently keep untrained values.
        rad_cfg = physics_config.radiation
        conv_cfg = physics_config.convection
        turb_cfg = physics_config.turbulence
        micro_cfg = physics_config.microphysics
        gwd_cfg = physics_config.gravity_wave_drag

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

        def combined_fn_with_phys_state(state, grid_, sigma_coord, phys_state,
                                        forcing=None):
            # Same stateful contract as the split-rad entry below (codex P1:
            # the non-split path must not stay silently memoryless).
            result = combined_raw(
                state, grid_, sigma_coord, phys_state=phys_state,
                forcing=forcing,
            )
            if not (isinstance(result, tuple) and len(result) == 2):
                raise TypeError(
                    "stateful classical physics expected (tendencies, "
                    f"phys_state), got {type(result)!r}")
            return result

        def _init_phys_state_combined(ncol: int, nlev: int, dtype=None):
            from legoesm.atmosphere.physics.physics_state import (
                init_physics_state,
            )
            return init_physics_state(ncol, nlev, physics_config, dtype=dtype)

        def _seed_phys_state_combined(state, grid_, sigma_coord):
            ps = _init_phys_state_combined(
                int(grid_.n_lat) * int(grid_.n_lon),
                int(jnp.shape(sigma_coord.sigma_full)[0]))
            _ck = None
            if turb_cfg.scheme == "clubb" and turb_cfg.clubb is not None:
                _ck = turb_cfg.clubb.params.c_K
            return _apply_wp2_seed(ps, state, grid_, sigma_coord, c_k=_ck)

        combined_fn.with_phys_state = combined_fn_with_phys_state
        combined_fn.init_phys_state = _init_phys_state_combined
        combined_fn.seed_phys_state = _seed_phys_state_combined
        combined_fn.physics_config = physics_config
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
    # The split builds radiation and the non-radiative physics SEPARATELY, so
    # neither sees the other and the combined wrapper's producer gate never
    # runs: cam6_clubb + a non-CLUBB closure would build with the routing on
    # and read a cloud fraction nobody writes (zero => clouds cleared).  Same
    # gate as combined.py, applied to the RESOLVED pair before splitting.
    if rad_cfg.use_clubb_cloud_fraction and turb_cfg.scheme != "clubb":
        raise ValueError(
            "RadiationConfig.use_clubb_cloud_fraction=True requires a "
            "cloud-fraction-producing turbulence closure "
            "(turbulence.scheme='clubb', diagnostic or prognostic); got "
            f"turbulence.scheme={turb_cfg.scheme!r} (cloud_scheme={cloud_scheme!r})."
        )
    non_rad_raw = make_physics(non_rad_cfg, model_type="spectral_pe", dt=dt)
    rad_only_raw = make_radiation_physics(
        rad_cfg, "spectral_pe",
        sfc_albedo_override=_sfc_albedo_override,
        sfc_emissivity_override=_sfc_emissivity_override,
        use_clubb_cloud_fraction=rad_cfg.use_clubb_cloud_fraction,
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

    def non_rad_fn_with_phys_state(state, grid_, sigma_coord, phys_state,
                                   forcing=None):
        """``(tendencies, phys_state_out)`` — the STATEFUL entry.

        The combined spectral wrapper has always produced the updated
        prognostic physics state (CLUBB's wp2/TKE, Bechtold's
        ``conv_prog_profile`` + stochastic state, the GWD spectrum, the
        PDF cloud fraction) and ``non_rad_fn`` above DISCARDS it — so the
        WB/AIMIP spectral training rollout ran every stateful scheme
        MEMORYLESS: CLUBB's turbulence energy sat at its floor forever,
        i.e. effectively NO boundary-layer mixing (2026-08-17 scene-17
        dissection: four in-scheme probes were bit-flat because they
        patched outputs this lane threw away). ``spectral_rollout``
        threads this entry's state through its scan carry when the
        marker below is present.
        """
        result = non_rad_raw(
            state, grid_, sigma_coord, phys_state=phys_state, forcing=forcing,
        )
        if not (isinstance(result, tuple) and len(result) == 2):
            # A silent fallback here would freeze the physics memory and
            # reintroduce the memoryless pathology this entry exists to fix
            # (codex P3) — fail loudly instead.
            raise TypeError(
                "stateful classical physics expected (tendencies, "
                f"phys_state) from the combined wrapper, got {type(result)!r}")
        return result

    def _init_phys_state(ncol: int, nlev: int, dtype=None):
        from legoesm.atmosphere.physics.physics_state import (
            init_physics_state,
        )
        return init_physics_state(ncol, nlev, non_rad_cfg, dtype=dtype)

    def _seed_phys_state(state, grid_, sigma_coord):
        ps = _init_phys_state(
            int(grid_.n_lat) * int(grid_.n_lon),
            int(jnp.shape(sigma_coord.sigma_full)[0]))
        # The active scheme's own (possibly trained/traced) c_K, detached
        # in _apply_wp2_seed; None -> fixed policy constant.
        _ck = None
        if turb_cfg.scheme == "clubb" and turb_cfg.clubb is not None:
            _ck = turb_cfg.clubb.params.c_K
        return _apply_wp2_seed(ps, state, grid_, sigma_coord, c_k=_ck)

    # Markers consumed by ``spectral_rollout``: their ABSENCE selects the
    # legacy stateless path (learned arms, older callers) byte-identically.
    non_rad_fn.with_phys_state = non_rad_fn_with_phys_state
    non_rad_fn.init_phys_state = _init_phys_state
    non_rad_fn.seed_phys_state = _seed_phys_state
    # What each callable ACTUALLY runs, so a caller can assert that the deck's
    # trained values, pinned settings, orbit and rain routing reached the model
    # instead of trusting that they did. On this path the two halves carry
    # DIFFERENT configurations -- the non-radiative families here, radiation on
    # the rad callable below -- and exposing the pre-split whole would let a
    # test pass on a config this lane never evaluates.
    non_rad_fn.physics_config = non_rad_cfg

    def rad_fn(state, grid_, sigma_coord, *, sim_time_seconds=0.0, forcing=None,
               phys_state=None):
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
        # ``phys_state`` is the LAGGED carry the rollout threads (CLUBB's
        # PDF cloud fraction, ZM's mass flux / in-cloud water); the
        # spectral radiation builder reads it only when the CLUBB
        # cloud-fraction routing is on, else it is ignored.
        rad_out = rad_only_raw(
            state, grid_, sigma_coord,
            sim_time_seconds=sim_time_seconds, forcing=forcing,
            phys_state=phys_state,
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

    rad_fn.radiation_config = rad_cfg
    # The rollout decides whether to fill the carry before the first radiation
    # call by this marker; it lives on the raw builder output, so re-export it
    # on the wrapper the rollout actually receives.
    if getattr(rad_only_raw, "_wants_phys_state_ro", False):
        rad_fn._wants_phys_state_ro = True
    return non_rad_fn, rad_fn
