"""Combined physics orchestrator for legoESM.

Provides `PhysicsConfig` and `make_physics()`, which create a single
physics function that combines radiation, convection, turbulence,
and microphysics tendencies.  Each sub-module can be independently
enabled/disabled via its ``scheme`` field (set to ``"none"`` to disable).

The combined function accepts an optional ``phys_state`` (``PhysicsState``)
argument.  When provided, prognostic physics variables (TKE, convective
mass flux, GWD wave action) are read from and written to the state,
enabling checkpoint/restart, ensemble ``vmap``, and clean JIT tracing.

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


def _build_updated_phys_state(phys_state, updates):
    """Build updated PhysicsState from explicitly returned field updates.

    Parameters
    ----------
    phys_state : PhysicsState or None
        Input physics state.  When ``None``, returns ``None``.
    updates : dict
        Mapping from field name ('tke', 'conv_prog', 'gwd_spectrum')
        to the updated JAX array returned by the sub-physics function.

    Returns
    -------
    PhysicsState or None
    """
    if phys_state is None:
        return None
    from legoesm.atmosphere.physics.physics_state import PhysicsState

    return PhysicsState(
        tke=updates.get('tke', phys_state.tke),
        conv_prog=updates.get('conv_prog', phys_state.conv_prog),
        gwd_spectrum=updates.get('gwd_spectrum', phys_state.gwd_spectrum),
    )


# ======================================================================
# Hydrostatic
# ======================================================================

def _make_hydrostatic_combined(config: PhysicsConfig, dt: float) -> Callable:
    # Tagged list: (fn, accepts_phys_state, updated_field_name)
    # Turbulence, convection, GWD accept phys_state and return updates;
    # radiation, microphysics don't.
    tagged_fns = []
    if config.radiation.scheme != "none":
        tagged_fns.append((make_radiation_physics(config.radiation, "hydrostatic"), False, None))
    if config.convection.scheme != "none":
        tagged_fns.append((make_convection_physics(config.convection, "hydrostatic", dt), True, "conv_prog"))
    if config.turbulence.scheme != "none":
        tagged_fns.append((make_turbulence_physics(config.turbulence, "hydrostatic", dt), True, "tke"))
    if config.microphysics.scheme != "none":
        tagged_fns.append((make_microphysics_physics(config.microphysics, "hydrostatic", dt), False, None))
    if config.gravity_wave_drag.scheme != "none":
        tagged_fns.append((make_gwd_physics(config.gravity_wave_drag, "hydrostatic", dt), True, "gwd_spectrum"))

    def physics_fn(state, grid, sigma_coord, phys_state=None):
        if not tagged_fns:
            shape_3d = state.T.data.shape
            shape_2d = state.p_s.data.shape
            dims_3d = ("face", "x", "y", "level")
            dims_2d = ("face", "x", "y")
            _sd = state.T.data.dtype
            zero_tend = HydrostaticTendencies(
                du_dt=Field(data=jnp.zeros(shape_3d, dtype=_sd), name="du_dt_phys", dims=dims_3d, units="m/s^2"),
                dv_dt=Field(data=jnp.zeros(shape_3d, dtype=_sd), name="dv_dt_phys", dims=dims_3d, units="m/s^2"),
                dT_dt=Field(data=jnp.zeros(shape_3d, dtype=_sd), name="dT_dt_phys", dims=dims_3d, units="K/s"),
                dp_s_dt=Field(data=jnp.zeros(shape_2d, dtype=_sd), name="dp_s_dt_phys", dims=dims_2d, units="Pa/s"),
                dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_sd), name="dphis_dt_phys", dims=dims_2d, units="m^2/s^3"),
            )
            return zero_tend, None

        phys_updates = {}

        fn0, accepts_ps, field_name = tagged_fns[0]
        if accepts_ps:
            first, field_val = fn0(state, grid, sigma_coord, phys_state=phys_state)
            if field_val is not None and field_name is not None:
                phys_updates[field_name] = field_val
        else:
            first = fn0(state, grid, sigma_coord)
        du_dt = first.du_dt.data
        dv_dt = first.dv_dt.data
        dT_dt = first.dT_dt.data
        dp_s_dt = first.dp_s_dt.data
        dphis_dt = first.dphis_dt.data

        # Accumulate tracer tendencies from all physics modules
        combined_tracer_tends = {}
        if first.tracer_tendencies is not None:
            for k, v in first.tracer_tendencies.items():
                combined_tracer_tends[k] = v.data

        for fn, accepts_ps, field_name in tagged_fns[1:]:
            if accepts_ps:
                t, field_val = fn(state, grid, sigma_coord, phys_state=phys_state)
                if field_val is not None and field_name is not None:
                    phys_updates[field_name] = field_val
            else:
                t = fn(state, grid, sigma_coord)
            du_dt = du_dt + t.du_dt.data
            dv_dt = dv_dt + t.dv_dt.data
            dT_dt = dT_dt + t.dT_dt.data
            dp_s_dt = dp_s_dt + t.dp_s_dt.data
            dphis_dt = dphis_dt + t.dphis_dt.data

            if t.tracer_tendencies is not None:
                for k, v in t.tracer_tendencies.items():
                    if k in combined_tracer_tends:
                        combined_tracer_tends[k] = combined_tracer_tends[k] + v.data
                    else:
                        combined_tracer_tends[k] = v.data

        # Build tracer_tendencies dict with Field wrappers
        dims_3d = ("face", "x", "y", "level")
        tracer_tends_out = None
        if combined_tracer_tends:
            tracer_tends_out = {
                k: Field(data=v, name=f"d{k}_dt_phys", dims=dims_3d, units="kg/kg/s")
                for k, v in combined_tracer_tends.items()
            }

        combined = HydrostaticTendencies(
            du_dt=first.du_dt.replace(data=du_dt),
            dv_dt=first.dv_dt.replace(data=dv_dt),
            dT_dt=first.dT_dt.replace(data=dT_dt),
            dp_s_dt=first.dp_s_dt.replace(data=dp_s_dt),
            dphis_dt=first.dphis_dt.replace(data=dphis_dt),
            tracer_tendencies=tracer_tends_out,
        )
        phys_state_out = _build_updated_phys_state(phys_state, phys_updates)
        return combined, phys_state_out

    def reset_state():
        for fn, _, _ in tagged_fns:
            reset_fn = getattr(fn, "reset_state", None)
            if callable(reset_fn):
                reset_fn()

    def set_time(day_of_year: float, seconds_of_day: float):
        """Propagate time to all sub-physics modules (e.g. radiation)."""
        for fn, _, _ in tagged_fns:
            st = getattr(fn, "set_time", None)
            if callable(st):
                st(day_of_year, seconds_of_day)

    physics_fn.reset_state = reset_state
    physics_fn.set_time = set_time
    return physics_fn


# ======================================================================
# Non-hydrostatic
# ======================================================================

def _make_nonhydrostatic_combined(config: PhysicsConfig, dt: float) -> Callable:
    tagged_fns = []
    if config.radiation.scheme != "none":
        tagged_fns.append((make_radiation_physics(config.radiation, "nonhydrostatic"), False, None))
    if config.convection.scheme != "none":
        tagged_fns.append((make_convection_physics(config.convection, "nonhydrostatic", dt), True, "conv_prog"))
    if config.turbulence.scheme != "none":
        tagged_fns.append((make_turbulence_physics(config.turbulence, "nonhydrostatic", dt), True, "tke"))
    if config.microphysics.scheme != "none":
        tagged_fns.append((make_microphysics_physics(config.microphysics, "nonhydrostatic", dt), False, None))
    if config.gravity_wave_drag.scheme != "none":
        tagged_fns.append((make_gwd_physics(config.gravity_wave_drag, "nonhydrostatic", dt), True, "gwd_spectrum"))

    def physics_fn(state, grid, height_coord, terrain_metric, phys_state=None):
        if not tagged_fns:
            shape_3d = state.theta_prime.data.shape
            shape_w = state.w.data.shape
            shape_2d = state.phis.data.shape
            dims_3d = ("face", "x", "y", "level")
            dims_w = ("face", "x", "y", "level_half")
            dims_2d = ("face", "x", "y")
            dims_tr = ("face", "x", "y", "level", "tracer")
            _sd = state.theta_prime.data.dtype
            zero_tend = NonHydrostaticTendencies(
                du_dt=Field(data=jnp.zeros(shape_3d, dtype=_sd), name="du_dt_phys", dims=dims_3d, units="m/s^2"),
                dv_dt=Field(data=jnp.zeros(shape_3d, dtype=_sd), name="dv_dt_phys", dims=dims_3d, units="m/s^2"),
                dw_dt=Field(data=jnp.zeros(shape_w, dtype=_sd), name="dw_dt_phys", dims=dims_w, units="m/s^2"),
                dtheta_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_sd), name="dtheta_prime_dt_phys", dims=dims_3d, units="K/s"),
                drho_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_sd), name="drho_prime_dt_phys", dims=dims_3d, units="kg/m^3/s"),
                dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_sd), name="dphis_dt_phys", dims=dims_2d, units="m^2/s^3"),
                dtracers_dt=Field(data=jnp.zeros_like(state.tracers.data), name="dtracers_dt_phys", dims=dims_tr, units="1/s"),
            )
            return zero_tend, None

        phys_updates = {}

        fn0, accepts_ps, field_name = tagged_fns[0]
        if accepts_ps:
            first, field_val = fn0(state, grid, height_coord, terrain_metric, phys_state=phys_state)
            if field_val is not None and field_name is not None:
                phys_updates[field_name] = field_val
        else:
            first = fn0(state, grid, height_coord, terrain_metric)
        du_dt = first.du_dt.data
        dv_dt = first.dv_dt.data
        dw_dt = first.dw_dt.data
        dtheta_prime_dt = first.dtheta_prime_dt.data
        drho_prime_dt = first.drho_prime_dt.data
        dphis_dt = first.dphis_dt.data
        dtracers_dt = first.dtracers_dt.data

        for fn, accepts_ps, field_name in tagged_fns[1:]:
            if accepts_ps:
                t, field_val = fn(state, grid, height_coord, terrain_metric, phys_state=phys_state)
                if field_val is not None and field_name is not None:
                    phys_updates[field_name] = field_val
            else:
                t = fn(state, grid, height_coord, terrain_metric)
            du_dt = du_dt + t.du_dt.data
            dv_dt = dv_dt + t.dv_dt.data
            dw_dt = dw_dt + t.dw_dt.data
            dtheta_prime_dt = dtheta_prime_dt + t.dtheta_prime_dt.data
            drho_prime_dt = drho_prime_dt + t.drho_prime_dt.data
            dphis_dt = dphis_dt + t.dphis_dt.data
            dtracers_dt = dtracers_dt + t.dtracers_dt.data

        combined = NonHydrostaticTendencies(
            du_dt=first.du_dt.replace(data=du_dt),
            dv_dt=first.dv_dt.replace(data=dv_dt),
            dw_dt=first.dw_dt.replace(data=dw_dt),
            dtheta_prime_dt=first.dtheta_prime_dt.replace(data=dtheta_prime_dt),
            drho_prime_dt=first.drho_prime_dt.replace(data=drho_prime_dt),
            dphis_dt=first.dphis_dt.replace(data=dphis_dt),
            dtracers_dt=first.dtracers_dt.replace(data=dtracers_dt),
        )
        phys_state_out = _build_updated_phys_state(phys_state, phys_updates)
        return combined, phys_state_out

    def reset_state():
        for fn, _, _ in tagged_fns:
            reset_fn = getattr(fn, "reset_state", None)
            if callable(reset_fn):
                reset_fn()

    def set_time(day_of_year: float, seconds_of_day: float):
        for fn, _, _ in tagged_fns:
            st = getattr(fn, "set_time", None)
            if callable(st):
                st(day_of_year, seconds_of_day)

    physics_fn.reset_state = reset_state
    physics_fn.set_time = set_time
    return physics_fn


# ======================================================================
# Spectral PE
# ======================================================================

def _make_spectral_pe_combined(config: PhysicsConfig, dt: float) -> Callable:
    tagged_fns = []
    if config.radiation.scheme != "none":
        tagged_fns.append((make_radiation_physics(config.radiation, "spectral_pe"), False, None))
    if config.convection.scheme != "none":
        tagged_fns.append((make_convection_physics(config.convection, "spectral_pe", dt), True, "conv_prog"))
    if config.turbulence.scheme != "none":
        tagged_fns.append((make_turbulence_physics(config.turbulence, "spectral_pe", dt), True, "tke"))
    if config.microphysics.scheme != "none":
        tagged_fns.append((make_microphysics_physics(config.microphysics, "spectral_pe", dt), False, None))
    if config.gravity_wave_drag.scheme != "none":
        tagged_fns.append((make_gwd_physics(config.gravity_wave_drag, "spectral_pe", dt), True, "gwd_spectrum"))

    def physics_fn(state, grid, sigma_coord, phys_state=None):
        from legoesm.atmosphere.dynamics.spectral_pe import (
            SpectralHydrostaticState,
            spectral_pe_to_grid,
        )

        if not tagged_fns:
            zero_3d = jnp.zeros_like(state.vor_hat.data)
            zero_2d = jnp.zeros_like(state.lnps_hat.data)
            zero_tend = SpectralHydrostaticState(
                vor_hat=state.vor_hat.replace(data=zero_3d),
                div_hat=state.div_hat.replace(data=zero_3d),
                T_hat=state.T_hat.replace(data=jnp.zeros_like(state.T_hat.data)),
                lnps_hat=state.lnps_hat.replace(data=zero_2d),
                phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
            )
            return zero_tend, None

        # Compute grid-space diagnostics once and reuse across all active modules.
        shared_fields = spectral_pe_to_grid(state, grid, sigma_coord)
        phys_updates = {}

        fn0, accepts_ps, field_name = tagged_fns[0]
        if accepts_ps:
            first, field_val = fn0(state, grid, sigma_coord, grid_fields=shared_fields, phys_state=phys_state)
            if field_val is not None and field_name is not None:
                phys_updates[field_name] = field_val
        else:
            first = fn0(state, grid, sigma_coord, grid_fields=shared_fields)
        vor_hat = first.vor_hat.data
        div_hat = first.div_hat.data
        T_hat = first.T_hat.data
        lnps_hat = first.lnps_hat.data
        phis_hat = first.phis_hat.data

        for fn, accepts_ps, field_name in tagged_fns[1:]:
            if accepts_ps:
                t, field_val = fn(state, grid, sigma_coord, grid_fields=shared_fields, phys_state=phys_state)
                if field_val is not None and field_name is not None:
                    phys_updates[field_name] = field_val
            else:
                t = fn(state, grid, sigma_coord, grid_fields=shared_fields)
            vor_hat = vor_hat + t.vor_hat.data
            div_hat = div_hat + t.div_hat.data
            T_hat = T_hat + t.T_hat.data
            lnps_hat = lnps_hat + t.lnps_hat.data
            phis_hat = phis_hat + t.phis_hat.data

        combined = SpectralHydrostaticState(
            vor_hat=first.vor_hat.replace(data=vor_hat),
            div_hat=first.div_hat.replace(data=div_hat),
            T_hat=first.T_hat.replace(data=T_hat),
            lnps_hat=first.lnps_hat.replace(data=lnps_hat),
            phis_hat=first.phis_hat.replace(data=phis_hat),
        )
        phys_state_out = _build_updated_phys_state(phys_state, phys_updates)
        return combined, phys_state_out

    def reset_state():
        for fn, _, _ in tagged_fns:
            reset_fn = getattr(fn, "reset_state", None)
            if callable(reset_fn):
                reset_fn()

    def set_time(day_of_year: float, seconds_of_day: float):
        for fn, _, _ in tagged_fns:
            st = getattr(fn, "set_time", None)
            if callable(st):
                st(day_of_year, seconds_of_day)

    physics_fn.reset_state = reset_state
    physics_fn.set_time = set_time
    return physics_fn
