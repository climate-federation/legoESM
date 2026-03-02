"""Combined physics orchestrator for legoESM.

Provides `PhysicsConfig` and `make_physics()`, which create a single
physics function that combines radiation, convection, turbulence,
and microphysics tendencies.  Each sub-module can be independently
enabled/disabled via its ``scheme`` field (set to ``"none"`` to disable).

Example
-------
>>> from legoesm.atmosphere.physics import (
...     PhysicsConfig, make_physics,
...     RadiationConfig, ConvectionConfig, TurbulenceConfig,
...     MicrophysicsConfig,
... )
>>> config = PhysicsConfig(
...     radiation=RadiationConfig(scheme="gray"),
...     convection=ConvectionConfig(scheme="sbm"),
...     turbulence=TurbulenceConfig(scheme="louis"),
...     microphysics=MicrophysicsConfig(scheme="kessler"),
... )
>>> physics_fn = make_physics(config, model_type="hydrostatic", dt=300.0)
>>> state = model.step_with_physics(state, dt, physics_fn)
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import (
    HydrostaticTendencies,
    NonHydrostaticTendencies,
)

from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.convection.config import ConvectionConfig
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig

from legoesm.atmosphere.physics.radiation.integration import (
    make_radiation_physics,
)
from legoesm.atmosphere.physics.convection.integration import (
    make_convection_physics,
)
from legoesm.atmosphere.physics.turbulence.integration import (
    make_turbulence_physics,
)
from legoesm.atmosphere.physics.microphysics.integration import (
    make_microphysics_physics,
)
from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
    make_gwd_physics,
)


class PhysicsConfig(NamedTuple):
    """Unified physics configuration.

    Holds sub-configurations for every physics module.  Set the
    ``scheme`` field of any sub-config to ``"none"`` to disable that
    module entirely.

    Fields
    ------
    radiation : RadiationConfig
        Radiation configuration (schemes: "gray", "rrtmgp").
    convection : ConvectionConfig
        Convection configuration (schemes: "sbm", "dca", "none").
    turbulence : TurbulenceConfig
        Turbulence configuration (schemes: "smagorinsky", "louis",
        "tke", "clubb_lite", "none").
    microphysics : MicrophysicsConfig
        Microphysics configuration (schemes: "kessler", "sundqvist",
        "seifert_beheng", "morrison", "thompson", "ml_emulator", "none").
    gravity_wave_drag : GravityWaveDragConfig
        Gravity wave drag configuration (schemes: "rayleigh", "lindzen",
        "mcfarlane", "hines", "prognostic_spectral", "ml_emulator", "none").
    """
    radiation: RadiationConfig = RadiationConfig()
    convection: ConvectionConfig = ConvectionConfig()
    turbulence: TurbulenceConfig = TurbulenceConfig()
    microphysics: MicrophysicsConfig = MicrophysicsConfig()
    gravity_wave_drag: GravityWaveDragConfig = GravityWaveDragConfig()


def make_physics(
    config: PhysicsConfig,
    model_type: str = "hydrostatic",
    dt: float = 300.0,
) -> Callable:
    """Create a combined physics function for a dynamical core.

    The returned function calls each enabled physics module and sums
    their tendencies.

    Parameters
    ----------
    config : PhysicsConfig
        Unified physics configuration.
    model_type : str
        One of ``"hydrostatic"``, ``"nonhydrostatic"``, ``"spectral_pe"``.
    dt : float
        Model time step [s].

    Returns
    -------
    Callable
        Physics function with the correct signature for *model_type*.
    """
    if model_type == "hydrostatic":
        return _make_hydrostatic_combined(config, dt)
    elif model_type == "nonhydrostatic":
        return _make_nonhydrostatic_combined(config, dt)
    elif model_type == "spectral_pe":
        return _make_spectral_pe_combined(config, dt)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Choose from 'hydrostatic', 'nonhydrostatic', 'spectral_pe'."
        )


# ======================================================================
# Hydrostatic
# ======================================================================

def _make_hydrostatic_combined(config: PhysicsConfig, dt: float) -> Callable:
    fns = []
    # Radiation has no dt parameter
    if config.radiation.scheme != "none":
        fns.append(make_radiation_physics(config.radiation, "hydrostatic"))
    if config.convection.scheme != "none":
        fns.append(make_convection_physics(config.convection, "hydrostatic", dt))
    if config.turbulence.scheme != "none":
        fns.append(make_turbulence_physics(config.turbulence, "hydrostatic", dt))
    if config.microphysics.scheme != "none":
        fns.append(make_microphysics_physics(config.microphysics, "hydrostatic", dt))
    if config.gravity_wave_drag.scheme != "none":
        fns.append(make_gwd_physics(config.gravity_wave_drag, "hydrostatic", dt))

    def physics_fn(state, grid, sigma_coord):
        if not fns:
            shape_3d = state.T.data.shape
            shape_2d = state.p_s.data.shape
            dims_3d = ("face", "x", "y", "level")
            dims_2d = ("face", "x", "y")
            return HydrostaticTendencies(
                du_dt=Field(data=jnp.zeros(shape_3d), name="du_dt_phys", dims=dims_3d, units="m/s^2"),
                dv_dt=Field(data=jnp.zeros(shape_3d), name="dv_dt_phys", dims=dims_3d, units="m/s^2"),
                dT_dt=Field(data=jnp.zeros(shape_3d), name="dT_dt_phys", dims=dims_3d, units="K/s"),
                dp_s_dt=Field(data=jnp.zeros(shape_2d), name="dp_s_dt_phys", dims=dims_2d, units="Pa/s"),
                dphis_dt=Field(data=jnp.zeros(shape_2d), name="dphis_dt_phys", dims=dims_2d, units="m^2/s^3"),
            )

        first = fns[0](state, grid, sigma_coord)
        du_dt = first.du_dt.data
        dv_dt = first.dv_dt.data
        dT_dt = first.dT_dt.data
        dp_s_dt = first.dp_s_dt.data
        dphis_dt = first.dphis_dt.data

        for fn in fns[1:]:
            t = fn(state, grid, sigma_coord)
            du_dt = du_dt + t.du_dt.data
            dv_dt = dv_dt + t.dv_dt.data
            dT_dt = dT_dt + t.dT_dt.data
            dp_s_dt = dp_s_dt + t.dp_s_dt.data
            dphis_dt = dphis_dt + t.dphis_dt.data

        return HydrostaticTendencies(
            du_dt=first.du_dt.replace(data=du_dt),
            dv_dt=first.dv_dt.replace(data=dv_dt),
            dT_dt=first.dT_dt.replace(data=dT_dt),
            dp_s_dt=first.dp_s_dt.replace(data=dp_s_dt),
            dphis_dt=first.dphis_dt.replace(data=dphis_dt),
        )

    def reset_state():
        for fn in fns:
            reset_fn = getattr(fn, "reset_state", None)
            if callable(reset_fn):
                reset_fn()

    physics_fn.reset_state = reset_state
    return physics_fn


# ======================================================================
# Non-hydrostatic
# ======================================================================

def _make_nonhydrostatic_combined(config: PhysicsConfig, dt: float) -> Callable:
    fns = []
    if config.radiation.scheme != "none":
        fns.append(make_radiation_physics(config.radiation, "nonhydrostatic"))
    if config.convection.scheme != "none":
        fns.append(make_convection_physics(config.convection, "nonhydrostatic", dt))
    if config.turbulence.scheme != "none":
        fns.append(make_turbulence_physics(config.turbulence, "nonhydrostatic", dt))
    if config.microphysics.scheme != "none":
        fns.append(make_microphysics_physics(config.microphysics, "nonhydrostatic", dt))
    if config.gravity_wave_drag.scheme != "none":
        fns.append(make_gwd_physics(config.gravity_wave_drag, "nonhydrostatic", dt))

    def physics_fn(state, grid, height_coord, terrain_metric):
        if not fns:
            shape_3d = state.theta_prime.data.shape
            shape_w = state.w.data.shape
            shape_2d = state.phis.data.shape
            dims_3d = ("face", "x", "y", "level")
            dims_w = ("face", "x", "y", "level_half")
            dims_2d = ("face", "x", "y")
            dims_tr = ("face", "x", "y", "level", "tracer")
            return NonHydrostaticTendencies(
                du_dt=Field(data=jnp.zeros(shape_3d), name="du_dt_phys", dims=dims_3d, units="m/s^2"),
                dv_dt=Field(data=jnp.zeros(shape_3d), name="dv_dt_phys", dims=dims_3d, units="m/s^2"),
                dw_dt=Field(data=jnp.zeros(shape_w), name="dw_dt_phys", dims=dims_w, units="m/s^2"),
                dtheta_prime_dt=Field(data=jnp.zeros(shape_3d), name="dtheta_prime_dt_phys", dims=dims_3d, units="K/s"),
                drho_prime_dt=Field(data=jnp.zeros(shape_3d), name="drho_prime_dt_phys", dims=dims_3d, units="kg/m^3/s"),
                dphis_dt=Field(data=jnp.zeros(shape_2d), name="dphis_dt_phys", dims=dims_2d, units="m^2/s^3"),
                dtracers_dt=Field(data=jnp.zeros_like(state.tracers.data), name="dtracers_dt_phys", dims=dims_tr, units="1/s"),
            )

        first = fns[0](state, grid, height_coord, terrain_metric)
        du_dt = first.du_dt.data
        dv_dt = first.dv_dt.data
        dw_dt = first.dw_dt.data
        dtheta_prime_dt = first.dtheta_prime_dt.data
        drho_prime_dt = first.drho_prime_dt.data
        dphis_dt = first.dphis_dt.data
        dtracers_dt = first.dtracers_dt.data

        for fn in fns[1:]:
            t = fn(state, grid, height_coord, terrain_metric)
            du_dt = du_dt + t.du_dt.data
            dv_dt = dv_dt + t.dv_dt.data
            dw_dt = dw_dt + t.dw_dt.data
            dtheta_prime_dt = dtheta_prime_dt + t.dtheta_prime_dt.data
            drho_prime_dt = drho_prime_dt + t.drho_prime_dt.data
            dphis_dt = dphis_dt + t.dphis_dt.data
            dtracers_dt = dtracers_dt + t.dtracers_dt.data

        return NonHydrostaticTendencies(
            du_dt=first.du_dt.replace(data=du_dt),
            dv_dt=first.dv_dt.replace(data=dv_dt),
            dw_dt=first.dw_dt.replace(data=dw_dt),
            dtheta_prime_dt=first.dtheta_prime_dt.replace(data=dtheta_prime_dt),
            drho_prime_dt=first.drho_prime_dt.replace(data=drho_prime_dt),
            dphis_dt=first.dphis_dt.replace(data=dphis_dt),
            dtracers_dt=first.dtracers_dt.replace(data=dtracers_dt),
        )

    def reset_state():
        for fn in fns:
            reset_fn = getattr(fn, "reset_state", None)
            if callable(reset_fn):
                reset_fn()

    physics_fn.reset_state = reset_state
    return physics_fn


# ======================================================================
# Spectral PE
# ======================================================================

def _make_spectral_pe_combined(config: PhysicsConfig, dt: float) -> Callable:
    fns = []
    if config.radiation.scheme != "none":
        fns.append(make_radiation_physics(config.radiation, "spectral_pe"))
    if config.convection.scheme != "none":
        fns.append(make_convection_physics(config.convection, "spectral_pe", dt))
    if config.turbulence.scheme != "none":
        fns.append(make_turbulence_physics(config.turbulence, "spectral_pe", dt))
    if config.microphysics.scheme != "none":
        fns.append(make_microphysics_physics(config.microphysics, "spectral_pe", dt))
    if config.gravity_wave_drag.scheme != "none":
        fns.append(make_gwd_physics(config.gravity_wave_drag, "spectral_pe", dt))

    def physics_fn(state, grid, sigma_coord):
        from legoesm.atmosphere.dynamics.spectral_pe import (
            SpectralHydrostaticState,
            spectral_pe_to_grid,
        )

        if not fns:
            zero_3d = jnp.zeros_like(state.vor_hat.data)
            zero_2d = jnp.zeros_like(state.lnps_hat.data)
            return SpectralHydrostaticState(
                vor_hat=state.vor_hat.replace(data=zero_3d),
                div_hat=state.div_hat.replace(data=zero_3d),
                T_hat=state.T_hat.replace(data=jnp.zeros_like(state.T_hat.data)),
                lnps_hat=state.lnps_hat.replace(data=zero_2d),
                phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
            )

        # Compute grid-space diagnostics once and reuse across all active modules.
        shared_fields = spectral_pe_to_grid(state, grid, sigma_coord)

        first = fns[0](state, grid, sigma_coord, grid_fields=shared_fields)
        vor_hat = first.vor_hat.data
        div_hat = first.div_hat.data
        T_hat = first.T_hat.data
        lnps_hat = first.lnps_hat.data
        phis_hat = first.phis_hat.data

        for fn in fns[1:]:
            t = fn(state, grid, sigma_coord, grid_fields=shared_fields)
            vor_hat = vor_hat + t.vor_hat.data
            div_hat = div_hat + t.div_hat.data
            T_hat = T_hat + t.T_hat.data
            lnps_hat = lnps_hat + t.lnps_hat.data
            phis_hat = phis_hat + t.phis_hat.data

        return SpectralHydrostaticState(
            vor_hat=first.vor_hat.replace(data=vor_hat),
            div_hat=first.div_hat.replace(data=div_hat),
            T_hat=first.T_hat.replace(data=T_hat),
            lnps_hat=first.lnps_hat.replace(data=lnps_hat),
            phis_hat=first.phis_hat.replace(data=phis_hat),
        )

    def reset_state():
        for fn in fns:
            reset_fn = getattr(fn, "reset_state", None)
            if callable(reset_fn):
                reset_fn()

    physics_fn.reset_state = reset_state
    return physics_fn
