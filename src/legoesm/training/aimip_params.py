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
]

_LOUIS_TRAINABLE: list[ParamConstraint] = [
    ParamConstraint("louis_l_mix_max", 20.0, 400.0, "sigmoid"),
    ParamConstraint("louis_Ck", 0.1, 0.6, "sigmoid"),
    ParamConstraint("louis_Ri_crit", 0.1, 0.6, "sigmoid"),
    ParamConstraint("louis_b_louis", 2.0, 10.0, "sigmoid"),
    ParamConstraint("louis_c_louis", 5.0, 30.0, "sigmoid"),
]

_SURFACE_TRAINABLE: list[ParamConstraint] = [
    ParamConstraint("surface_Cd_neutral", 5.0e-4, 3.0e-3, "sigmoid"),
    ParamConstraint("surface_Ch_neutral", 5.0e-4, 3.0e-3, "sigmoid"),
]

_MCFARLANE_TRAINABLE: list[ParamConstraint] = [
    ParamConstraint("mcfarlane_h_topo", 100.0, 2000.0, "sigmoid"),
    ParamConstraint("mcfarlane_G_0", 0.1, 1.0, "sigmoid"),
    ParamConstraint("mcfarlane_efficiency", 0.1, 1.0, "sigmoid"),
    ParamConstraint("mcfarlane_min_wind", 0.5, 5.0, "sigmoid"),
    ParamConstraint("mcfarlane_envelope_scale", 0.5, 2.0, "sigmoid"),
]

# Xu-Randall cloud fraction.  ``T_freeze`` is NOT trainable per
# ``CLAUDE.md`` constants discipline; it lives in ``constants.py``.
_XU_RANDALL_TRAINABLE: list[ParamConstraint] = [
    ParamConstraint("cloud_rh_crit", 0.5, 0.95, "sigmoid"),
    ParamConstraint("cloud_alpha_xr", 25.0, 400.0, "sigmoid"),
    ParamConstraint("cloud_p_xr", 0.1, 1.0, "sigmoid"),
    ParamConstraint("cloud_q_c_diagnostic", 5.0e-5, 5.0e-4, "sigmoid"),
]

# Shared surface-energy-balance knobs (also feed gray radiation).
_AIMIP_COMMON_TRAINABLE: list[ParamConstraint] = [
    ParamConstraint("tau_equator", 5.0, 10.0, "sigmoid"),
    ParamConstraint("tau_pole", 1.0, 3.0, "sigmoid"),
    ParamConstraint("albedo_ice", 0.4, 0.8, "sigmoid"),
    ParamConstraint("albedo_ocean", 0.03, 0.10, "sigmoid"),
]


AIMIP_CLASSICAL_CONSTRAINTS: list[ParamConstraint] = (
    _TIEDTKE_TRAINABLE
    + _LOUIS_TRAINABLE
    + _SURFACE_TRAINABLE
    + _MCFARLANE_TRAINABLE
    + _XU_RANDALL_TRAINABLE
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
    """
    raw_values: dict[str, jax.Array]
    constraints: list[ParamConstraint] = eqx.field(static=True)

    @staticmethod
    def from_defaults() -> "AIMIPClassicalParams":
        """Initialize all knobs at their canonical scheme defaults."""
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
        return AIMIPClassicalParams(
            raw_values=raw,
            constraints=AIMIP_CLASSICAL_CONSTRAINTS,
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
        )

    def to_surface_config(self) -> SurfaceLayerConfig:
        d = self.as_dict()
        base = SurfaceLayerConfig()
        return base._replace(
            Cd_neutral=d["surface_Cd_neutral"],
            Ch_neutral=d["surface_Ch_neutral"],
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
        )

    def to_cloud_config(self) -> CloudConfig:
        d = self.as_dict()
        base = CloudConfig(scheme="xu_randall")
        return base._replace(
            rh_crit=d["cloud_rh_crit"],
            alpha_xr=d["cloud_alpha_xr"],
            p_xr=d["cloud_p_xr"],
            q_c_diagnostic=d["cloud_q_c_diagnostic"],
        )


def _canonical_scheme_defaults() -> dict[str, float]:
    """Look up the canonical scheme-default for each AIMIP knob."""
    t = TiedtkeConfig()
    lo = LouisConfig()
    su = SurfaceLayerConfig()
    mc = McFarlaneConfig()
    cl = CloudConfig()
    return {
        "tiedtke_tau_M_u_relax": float(t.tau_M_u_relax),
        "tiedtke_tau_MC_proxy": float(t.tau_MC_proxy),
        "tiedtke_cape_threshold": float(t.cape_threshold),
        "tiedtke_precip_efficiency": float(t.precip_efficiency),
        "tiedtke_downdraft_alpha": float(t.downdraft_alpha),
        "tiedtke_downdraft_RH_min": float(t.downdraft_RH_min),
        "louis_l_mix_max": float(lo.l_mix_max),
        "louis_Ck": float(lo.Ck),
        "louis_Ri_crit": float(lo.Ri_crit),
        "louis_b_louis": float(lo.b_louis),
        "louis_c_louis": float(lo.c_louis),
        "surface_Cd_neutral": float(su.Cd_neutral),
        "surface_Ch_neutral": float(su.Ch_neutral),
        "mcfarlane_h_topo": float(mc.h_topo),
        "mcfarlane_G_0": float(mc.G_0),
        "mcfarlane_efficiency": float(mc.efficiency),
        "mcfarlane_min_wind": float(mc.min_wind),
        "mcfarlane_envelope_scale": float(mc.envelope_scale),
        "cloud_rh_crit": float(cl.rh_crit),
        "cloud_alpha_xr": float(cl.alpha_xr),
        "cloud_p_xr": float(cl.p_xr),
        "cloud_q_c_diagnostic": float(cl.q_c_diagnostic),
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
    )

    p = params.as_dict()

    rad_cfg = RadiationConfig(
        scheme="gray",
        gray=GrayRadiationConfig(
            tau_equator=p["tau_equator"],
            tau_pole=p["tau_pole"],
        ),
    )
    conv_cfg = ConvectionConfig(
        scheme="tiedtke", tiedtke=params.to_tiedtke_config(),
    )
    turb_cfg = TurbulenceConfig(
        scheme="louis", louis=params.to_louis_config(),
    )
    gwd_cfg = GravityWaveDragConfig(
        scheme="mcfarlane", mcfarlane=params.to_mcfarlane_config(),
    )

    physics_config = PhysicsConfig(
        radiation=rad_cfg,
        convection=conv_cfg,
        turbulence=turb_cfg,
        gravity_wave_drag=gwd_cfg,
    )
    raw_fn = make_physics(physics_config, model_type="spectral_pe", dt=dt)

    def physics_fn(state, grid_, sigma_coord):
        result = raw_fn(state, grid_, sigma_coord)
        return result[0] if isinstance(result, tuple) else result

    return physics_fn
