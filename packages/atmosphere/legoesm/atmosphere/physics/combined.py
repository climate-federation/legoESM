"""Combined physics orchestrator for legoESM.

Provides `PhysicsConfig` and `make_physics()`, which create a single
physics function that combines radiation, convection, turbulence,
microphysics, and **gravity wave drag** tendencies. Each sub-module
can be independently enabled/disabled via its ``scheme`` field (set to
``"none"`` to disable).

Gravity wave drag is included as a first-class component on equal
footing with the other parameterizations; the supported schemes are
``"rayleigh"``, ``"lindzen"``, ``"mcfarlane"``, ``"hines"``,
``"prognostic_spectral"``, ``"ml_emulator"``, and ``"none"`` (see
``GravityWaveDragConfig``).

The combined function accepts an optional ``phys_state`` (``PhysicsState``)
argument.  When provided, prognostic physics variables (TKE, convective
mass flux, GWD wave action) are read from and written to the state,
enabling checkpoint/restart, ensemble ``vmap``, and clean JIT tracing.

Example
-------
>>> from legoesm.atmosphere.physics import (
...     PhysicsConfig, make_physics,
...     RadiationConfig, ConvectionConfig, TurbulenceConfig,
...     MicrophysicsConfig, GravityWaveDragConfig,
... )
>>> config = PhysicsConfig(
...     radiation=RadiationConfig(scheme="gray"),
...     convection=ConvectionConfig(scheme="sbm"),
...     turbulence=TurbulenceConfig(scheme="louis"),
...     microphysics=MicrophysicsConfig(scheme="kessler"),
...     gravity_wave_drag=GravityWaveDragConfig(scheme="lindzen"),
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
    get_turbulence_fn,
    make_turbulence_physics,
    turbulence_carry_field,
)
from legoesm.atmosphere.physics.microphysics.integration import (
    make_microphysics_physics,
)
from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
    make_gwd_physics,
)
from legoesm.atmosphere.physics.physics_state import update_physics_state
from legoesm.atmosphere.physics._shared import zero_like_tracers
from legoesm.atmosphere.dynamics.spectral_pe import (
    SpectralHydrostaticState,
    spectral_pe_to_grid,
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
    dt: float = 300.0,  # coeff-ok: default physics timestep [s]
    column_mesh=None,
    sfc_albedo_override=None,
    sfc_emissivity_override=None,
) -> Callable:
    """Create a combined physics function for a dynamical core.

    The returned function calls each enabled physics module and sums
    their tendencies.

    Parameters
    ----------
    config : PhysicsConfig
        Unified physics configuration.
    model_type : str
        One of ``"hydrostatic"``, ``"nonhydrostatic"``, ``"spectral_pe"``,
        ``"mpas"``.
    dt : float
        Model time step [s].
    column_mesh : jax.sharding.Mesh or None, optional
        Issue #273 follow-up.  When supplied (build via
        ``legoesm.parallel.column_shard.create_column_mesh``), the
        per-column radiation kernel is sharded across the mesh's
        ``'col'`` axis so a device count that fails cubed-sphere
        face-divisibility (e.g. 4-GPU node) still keeps every device
        busy on the radiation hot path.  Default ``None`` preserves
        bit-exact behavior.

    Returns
    -------
    Callable
        Physics function with the correct signature for *model_type*.
        Carries a ``_requires_phys_state`` attribute: ``True`` when the
        configured schemes are stateful (prognostic TKE-family /
        MYNN-2.5 turbulence, prognostic-spectral GWD, prognostic
        convection), so step wrappers that cannot thread the
        ``PhysicsState`` carry can refuse loudly instead of silently
        reseeding every step (issue #405/#413).
    """
    # Build-time surface overrides (e.g. AIMIP's trained spatial sfc_albedo /
    # sfc_emissivity) are routed to the radiation solve as per-call overrides so
    # a trained value never touches RRTMGPConfig.sfc_* (RRTMGP's solver-cache
    # key). Only the spectral_pe combined path threads them today; reject loudly
    # elsewhere rather than silently dropping a trained surface field.
    if (sfc_albedo_override is not None or sfc_emissivity_override is not None) \
            and model_type != "spectral_pe":
        raise ValueError(
            "sfc_albedo_override / sfc_emissivity_override are only wired for "
            f"model_type='spectral_pe', got {model_type!r}."
        )
    if model_type == "hydrostatic":
        fn = _make_hydrostatic_combined(config, dt, column_mesh=column_mesh)
    elif model_type == "nonhydrostatic":
        fn = _make_nonhydrostatic_combined(config, dt)
    elif model_type == "spectral_pe":
        fn = _make_spectral_pe_combined(
            config, dt,
            sfc_albedo_override=sfc_albedo_override,
            sfc_emissivity_override=sfc_emissivity_override,
        )
    elif model_type == "mpas":
        fn = _make_mpas_combined(config, dt, column_mesh=column_mesh)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Choose from 'hydrostatic', 'nonhydrostatic', 'spectral_pe', 'mpas'."
        )
    fn._requires_phys_state = physics_config_requires_phys_state(config)
    return fn


def physics_config_requires_phys_state(config: PhysicsConfig) -> bool:
    """True when *config* selects any scheme with a prognostic carry.

    Single predicate for step wrappers (sharded dynamics, lat-lon MPI)
    that cannot thread ``PhysicsState`` and must refuse loudly rather
    than silently reseed (issue #405/#413).  Mirrors the driver guard:
    energy-carrying turbulence (shared traits), prognostic/stochastic
    convection, prognostic-spectral GWD.
    """
    from legoesm.atmosphere.physics.turbulence.integration import (
        turbulence_scheme_traits,
    )
    from legoesm.atmosphere.physics.convection.integration import (
        convection_scheme_traits,
    )
    # Profile-prognostic convection counts as stateful (codex round 5):
    # the bridge reads phys_state.conv_prog_profile for ZM/KF/Emanuel/
    # Tiedtke/Bechtold and falls back to zeros when the carry is absent
    # — Tiedtke concretely relaxes the previous profile into M_u_new,
    # so a dropped carry silently erases that memory every step.
    conv = convection_scheme_traits(config.convection.scheme)
    return bool(
        turbulence_scheme_traits(config.turbulence.scheme).carries_energy
        or conv.is_scalar_prognostic
        or conv.is_profile_prognostic
        or conv.is_stochastic
        or config.gravity_wave_drag.scheme == "prognostic_spectral"
    )


# ======================================================================
# Hydrostatic
# ======================================================================

def _make_hydrostatic_combined(config: PhysicsConfig, dt: float,
                               model_type: str = "hydrostatic",
                               column_mesh=None) -> Callable:
    """Combined physics for any hydrostatic model (cubed-sphere, lat-lon, MPAS).

    Uses the unified ``HydrostaticTendencies`` with optional ``dv_dt``.
    When *model_type* is ``"mpas"``, the radiation factory is called
    with ``"mpas"`` so that lat/lon extraction uses mesh.latCell/lonCell.

    Issue #273 follow-up: when ``column_mesh`` is supplied, the
    per-column radiation kernel runs sharded across the mesh.
    """
    tagged_fns = []
    if config.radiation.scheme != "none":
        tagged_fns.append((
            make_radiation_physics(
                config.radiation, model_type, column_mesh=column_mesh,
            ),
            False,
            None,
        ))
    if config.convection.scheme != "none":
        tagged_fns.append((make_convection_physics(config.convection, model_type, dt), True, "conv_prog_profile"))
    if config.turbulence.scheme != "none":
        # MYNN-2.5 writes its prognostic ``qke = 2·TKE`` into a
        # dedicated PhysicsState field so a restart-time scheme switch
        # cannot silently feed the wrong moment as energy (Phase C
        # codex iter-1 medium finding).
        # qke (MYNN-2.5) / clubb_moments (prognostic CLUBB) / tke — must match the
        # slot the per-model physics_fn reads (turbulence_carry_field, the
        # config-aware refinement of turbulence_scheme_traits.energy_field).
        _turb_sn, _, _turb_sc = get_turbulence_fn(config.turbulence)
        _turb_field = turbulence_carry_field(_turb_sn, _turb_sc)
        tagged_fns.append((
            make_turbulence_physics(config.turbulence, model_type, dt),
            True,
            _turb_field,
        ))
    if config.microphysics.scheme != "none":
        tagged_fns.append((make_microphysics_physics(config.microphysics, model_type, dt), False, None))
    if config.gravity_wave_drag.scheme != "none":
        tagged_fns.append((make_gwd_physics(config.gravity_wave_drag, model_type, dt), True, "gwd_spectrum"))

    def physics_fn(state, grid, sigma_coord, phys_state=None, forcing=None):
        has_v = state.v is not None

        if not tagged_fns:
            _sd = state.T.data.dtype
            dims_T = state.T.dims
            dims_ps = state.p_s.dims
            dv_dt_zero = None
            if has_v:
                dv_dt_zero = Field(
                    data=jnp.zeros_like(state.v.data), name="dv_dt_phys",
                    dims=state.v.dims, units="m/s^2",
                )
            zero_tend = HydrostaticTendencies(
                du_dt=Field(data=jnp.zeros_like(state.u.data), name="du_dt_phys",
                            dims=state.u.dims, units="m/s^2"),
                dT_dt=Field(data=jnp.zeros_like(state.T.data), name="dT_dt_phys",
                            dims=dims_T, units="K/s"),
                dp_s_dt=Field(data=jnp.zeros_like(state.p_s.data), name="dp_s_dt_phys",
                              dims=dims_ps, units="Pa/s"),
                dphis_dt=Field(data=jnp.zeros_like(state.p_s.data), name="dphis_dt_phys",
                               dims=dims_ps, units="m^2/s^3"),
                dv_dt=dv_dt_zero,
            )
            return zero_tend, None

        phys_updates = {}

        fn0, accepts_ps, field_name = tagged_fns[0]
        if accepts_ps:
            if getattr(fn0, "_wants_forcing", False):
                first, field_val = fn0(state, grid, sigma_coord,
                                       phys_state=phys_state, forcing=forcing)
            else:
                first, field_val = fn0(state, grid, sigma_coord, phys_state=phys_state)
            if field_val is not None and field_name is not None:
                # Multi-field updates (e.g., Bechtold's conv_prog_profile
                # and conv_stoch_state) are returned as a dict, which
                # we merge into ``phys_updates``.
                if isinstance(field_val, dict):
                    phys_updates.update(field_val)
                else:
                    phys_updates[field_name] = field_val
        else:
            if getattr(fn0, "_wants_forcing", False):
                first = fn0(state, grid, sigma_coord, forcing=forcing)
            else:
                first = fn0(state, grid, sigma_coord)
        du_dt = first.du_dt.data
        dv_dt = first.dv_dt.data if first.dv_dt is not None else None
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
                if getattr(fn, "_wants_forcing", False):
                    t, field_val = fn(state, grid, sigma_coord,
                                      phys_state=phys_state, forcing=forcing)
                else:
                    t, field_val = fn(state, grid, sigma_coord, phys_state=phys_state)
                if field_val is not None and field_name is not None:
                    # Match the first-iteration branch: dict updates
                    # (e.g., Bechtold's conv_prog_profile +
                    # conv_stoch_state + prng_key) must MERGE into
                    # phys_updates, not be assigned as a single
                    # blob under field_name.
                    if isinstance(field_val, dict):
                        phys_updates.update(field_val)
                    else:
                        phys_updates[field_name] = field_val
            else:
                if getattr(fn, "_wants_forcing", False):
                    t = fn(state, grid, sigma_coord, forcing=forcing)
                else:
                    t = fn(state, grid, sigma_coord)
            du_dt = du_dt + t.du_dt.data
            if dv_dt is not None and t.dv_dt is not None:
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
        tracer_tends_out = None
        if combined_tracer_tends:
            tracer_tends_out = {
                k: Field(data=v, name=f"d{k}_dt_phys",
                         dims=first.dT_dt.dims, units="kg/kg/s")
                for k, v in combined_tracer_tends.items()
            }

        combined_dv_dt = None
        if first.dv_dt is not None:
            combined_dv_dt = first.dv_dt.replace(data=dv_dt)

        combined = HydrostaticTendencies(
            du_dt=first.du_dt.replace(data=du_dt),
            dT_dt=first.dT_dt.replace(data=dT_dt),
            dp_s_dt=first.dp_s_dt.replace(data=dp_s_dt),
            dphis_dt=first.dphis_dt.replace(data=dphis_dt),
            dv_dt=combined_dv_dt,
            tracer_tendencies=tracer_tends_out,
        )
        phys_state_out = update_physics_state(phys_state, phys_updates)
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

    def set_T_sfc_override(value):
        """Propagate prescribed-T_sfc override to sub-physics that honour
        the hook (currently radiation; turbulence reads from PhysicsState).

        Called by the single-column driver when
        ``SCMForcing(prescribe="T_s")`` so the radiative surface
        boundary stays in sync with turbulence's bulk-flux boundary
        (Phase B v2 codex iter-2 finding).
        """
        for fn, _, _ in tagged_fns:
            st = getattr(fn, "set_T_sfc_override", None)
            if callable(st):
                st(value)

    physics_fn.reset_state = reset_state
    physics_fn.set_time = set_time
    physics_fn.set_T_sfc_override = set_T_sfc_override
    return physics_fn


# ======================================================================
# Non-hydrostatic
# ======================================================================

def _make_nonhydrostatic_combined(config: PhysicsConfig, dt: float) -> Callable:
    tagged_fns = []
    if config.radiation.scheme != "none":
        tagged_fns.append((make_radiation_physics(config.radiation, "nonhydrostatic"), False, None))
    if config.convection.scheme != "none":
        tagged_fns.append((make_convection_physics(config.convection, "nonhydrostatic", dt), True, "conv_prog_profile"))
    if config.turbulence.scheme != "none":
        # Phase C codex iter-2 high: route MYNN-2.5 to ``qke``
        # (PhysicsState) so a non-SCM nonhydrostatic run also persists
        # qke across steps.
        # qke (MYNN-2.5) / clubb_moments (prognostic CLUBB) / tke — must match the
        # slot the per-model physics_fn reads (turbulence_carry_field).
        _turb_sn, _, _turb_sc = get_turbulence_fn(config.turbulence)
        _turb_field = turbulence_carry_field(_turb_sn, _turb_sc)
        tagged_fns.append((
            make_turbulence_physics(config.turbulence, "nonhydrostatic", dt),
            True,
            _turb_field,
        ))
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
                # Multi-field updates (e.g., Bechtold's conv_prog_profile
                # and conv_stoch_state) are returned as a dict, which
                # we merge into ``phys_updates``.
                if isinstance(field_val, dict):
                    phys_updates.update(field_val)
                else:
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
                    # Match the first-iteration branch: dict updates
                    # (e.g., Bechtold's conv_prog_profile +
                    # conv_stoch_state + prng_key) must MERGE into
                    # phys_updates, not be assigned as a single
                    # blob under field_name.
                    if isinstance(field_val, dict):
                        phys_updates.update(field_val)
                    else:
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
        phys_state_out = update_physics_state(phys_state, phys_updates)
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

    def set_T_sfc_override(value):
        """Propagate prescribed-T_sfc override to sub-physics radiation
        modules (SCM Phase B v2)."""
        for fn, _, _ in tagged_fns:
            st = getattr(fn, "set_T_sfc_override", None)
            if callable(st):
                st(value)

    physics_fn.reset_state = reset_state
    physics_fn.set_time = set_time
    physics_fn.set_T_sfc_override = set_T_sfc_override
    return physics_fn


# ======================================================================
# Spectral PE
# ======================================================================

def _make_spectral_pe_combined(
    config: PhysicsConfig, dt: float,
    sfc_albedo_override=None, sfc_emissivity_override=None,
) -> Callable:
    tagged_fns = []
    if config.radiation.scheme != "none":
        tagged_fns.append((make_radiation_physics(
            config.radiation, "spectral_pe",
            sfc_albedo_override=sfc_albedo_override,
            sfc_emissivity_override=sfc_emissivity_override,
        ), False, None))
    if config.convection.scheme != "none":
        tagged_fns.append((make_convection_physics(config.convection, "spectral_pe", dt), True, "conv_prog_profile"))
    if config.turbulence.scheme != "none":
        # Phase C codex iter-2 high: route MYNN-2.5 to ``qke`` on the
        # spectral PE combined path too.
        # qke (MYNN-2.5) / clubb_moments (prognostic CLUBB) / tke — must match the
        # slot the per-model physics_fn reads (turbulence_carry_field).
        _turb_sn, _, _turb_sc = get_turbulence_fn(config.turbulence)
        _turb_field = turbulence_carry_field(_turb_sn, _turb_sc)
        tagged_fns.append((
            make_turbulence_physics(config.turbulence, "spectral_pe", dt),
            True,
            _turb_field,
        ))
    if config.microphysics.scheme != "none":
        tagged_fns.append((make_microphysics_physics(config.microphysics, "spectral_pe", dt), False, None))
    if config.gravity_wave_drag.scheme != "none":
        tagged_fns.append((make_gwd_physics(config.gravity_wave_drag, "spectral_pe", dt), True, "gwd_spectrum"))

    def physics_fn(state, grid, sigma_coord, phys_state=None, forcing=None):
        if not tagged_fns:
            zero_3d = jnp.zeros_like(state.vor_hat.data)
            zero_2d = jnp.zeros_like(state.lnps_hat.data)
            # Preserve the input state's tracer pytree structure as a
            # zero tendency so downstream tree.map(state, tendency)
            # works.  ``zero_like_tracers`` duck-types Field vs raw-array.
            zero_tracers = zero_like_tracers(state.tracers)
            zero_tend = SpectralHydrostaticState(
                vor_hat=state.vor_hat.replace(data=zero_3d),
                div_hat=state.div_hat.replace(data=zero_3d),
                T_hat=state.T_hat.replace(data=jnp.zeros_like(state.T_hat.data)),
                lnps_hat=state.lnps_hat.replace(data=zero_2d),
                phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
                tracers=zero_tracers,
            )
            return zero_tend, None

        # Compute grid-space diagnostics once and reuse across all active modules.
        shared_fields = spectral_pe_to_grid(state, grid, sigma_coord)
        phys_updates = {}

        fn0, accepts_ps, field_name = tagged_fns[0]
        _fwd0 = ({"forcing": forcing}
                 if getattr(fn0, "_wants_forcing", False) else {})
        if accepts_ps:
            first, field_val = fn0(state, grid, sigma_coord, grid_fields=shared_fields, phys_state=phys_state, **_fwd0)
            if field_val is not None and field_name is not None:
                # Multi-field updates (e.g., Bechtold's conv_prog_profile
                # and conv_stoch_state) are returned as a dict, which
                # we merge into ``phys_updates``.
                if isinstance(field_val, dict):
                    phys_updates.update(field_val)
                else:
                    phys_updates[field_name] = field_val
        else:
            first = fn0(state, grid, sigma_coord, grid_fields=shared_fields, **_fwd0)
        vor_hat = first.vor_hat.data
        div_hat = first.div_hat.data
        T_hat = first.T_hat.data
        lnps_hat = first.lnps_hat.data
        phis_hat = first.phis_hat.data

        # Tracer accumulation: start from the first module's
        # contribution if it has any, otherwise None.  Subsequent
        # modules are summed in the loop below.  Each value is kept
        # as a raw grid-space jax array (dropping the Field container)
        # so the per-key sum is a plain ``+``.  We re-wrap with the
        # state's container at the end for pytree-leaf consistency.
        def _grid_data(value):
            return value.data if hasattr(value, "data") else value
        accumulated_tracers = None
        if first.tracers is not None:
            accumulated_tracers = {
                k: _grid_data(v) for k, v in first.tracers.items()
            }

        for fn, accepts_ps, field_name in tagged_fns[1:]:
            _fwd = ({"forcing": forcing}
                    if getattr(fn, "_wants_forcing", False) else {})
            if accepts_ps:
                t, field_val = fn(state, grid, sigma_coord, grid_fields=shared_fields, phys_state=phys_state, **_fwd)
                if field_val is not None and field_name is not None:
                    # Match the first-iteration branch: dict updates
                    # (e.g., Bechtold's conv_prog_profile +
                    # conv_stoch_state + prng_key) must MERGE into
                    # phys_updates, not be assigned as a single
                    # blob under field_name.
                    if isinstance(field_val, dict):
                        phys_updates.update(field_val)
                    else:
                        phys_updates[field_name] = field_val
            else:
                t = fn(state, grid, sigma_coord, grid_fields=shared_fields, **_fwd)
            vor_hat = vor_hat + t.vor_hat.data
            div_hat = div_hat + t.div_hat.data
            T_hat = T_hat + t.T_hat.data
            lnps_hat = lnps_hat + t.lnps_hat.data
            phis_hat = phis_hat + t.phis_hat.data
            # Per-tracer accumulation across modules.
            if t.tracers is not None:
                if accumulated_tracers is None:
                    accumulated_tracers = {
                        k: _grid_data(v) for k, v in t.tracers.items()
                    }
                else:
                    for k, v in t.tracers.items():
                        v_data = _grid_data(v)
                        if k in accumulated_tracers:
                            accumulated_tracers[k] = (
                                accumulated_tracers[k] + v_data
                            )
                        else:
                            accumulated_tracers[k] = v_data

        # Build the final tracers dict for the combined tendency,
        # mirroring the input state's container types so pytree leaves
        # match downstream tree.map(state, tendency) calls.  When no
        # physics module touched tracers we emit ``zero_like_tracers``
        # of the input state's tracers (the original safe default).
        if state.tracers is None:
            tracers_combined = None
        elif accumulated_tracers is None:
            tracers_combined = zero_like_tracers(state.tracers)
        else:
            tracers_combined = {}
            for k, template in state.tracers.items():
                if k in accumulated_tracers:
                    arr = accumulated_tracers[k]
                else:
                    arr = (
                        jnp.zeros_like(template.data)
                        if hasattr(template, "data")
                        else jnp.zeros_like(template)
                    )
                if hasattr(template, "data") and hasattr(template, "replace"):
                    tracers_combined[k] = template.replace(data=arr)
                else:
                    tracers_combined[k] = arr
        combined = SpectralHydrostaticState(
            vor_hat=first.vor_hat.replace(data=vor_hat),
            div_hat=first.div_hat.replace(data=div_hat),
            T_hat=first.T_hat.replace(data=T_hat),
            lnps_hat=first.lnps_hat.replace(data=lnps_hat),
            phis_hat=first.phis_hat.replace(data=phis_hat),
            tracers=tracers_combined,
        )
        phys_state_out = update_physics_state(phys_state, phys_updates)
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

    def set_T_sfc_override(value):
        """Propagate prescribed-T_sfc override to sub-physics radiation
        modules (SCM Phase B v2)."""
        for fn, _, _ in tagged_fns:
            st = getattr(fn, "set_T_sfc_override", None)
            if callable(st):
                st(value)

    physics_fn.reset_state = reset_state
    physics_fn.set_time = set_time
    physics_fn.set_T_sfc_override = set_T_sfc_override
    # Marker: the combined fn consumes a per-step traced ``forcing``
    # dict when any sub-physics advertises ``_wants_forcing`` (currently
    # the spectral_pe radiation factory: T_sfc/o3_vmr/aerosol_od/ghg_vmr).
    physics_fn._wants_forcing = any(
        getattr(fn, "_wants_forcing", False) for fn, _, _ in tagged_fns
    )
    return physics_fn


# ======================================================================
# MPAS (Voronoi mesh) — uses unified hydrostatic combined path
# ======================================================================

def _make_mpas_combined(config: PhysicsConfig, dt: float,
                        column_mesh=None) -> Callable:
    return _make_hydrostatic_combined(
        config, dt, model_type="mpas", column_mesh=column_mesh,
    )
