"""Combined physics orchestrator for legoESM.

Provides `PhysicsConfig` and `make_physics()`, which create a single
physics function that combines radiation, convection, turbulence,
microphysics, and **gravity wave drag** tendencies. Each sub-module
can be independently enabled/disabled via its ``scheme`` field (set to
``"none"`` to disable).

Gravity wave drag is included as a first-class component on equal
footing with the other parameterizations; the supported schemes are
``"rayleigh"``, ``"lindzen"``, ``"mcfarlane"``, ``"hines"``,
``"prognostic_spectral"``, ``"e3sm_cam"``, ``"ml_emulator"``, and
``"none"`` (see ``GravityWaveDragConfig``).

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
from legoesm.atmosphere.physics.physics_state import (
    PHYSSTATE_PER_CALL_INPUTS,
    update_physics_state,
)
from legoesm.atmosphere.physics._shared import zero_like_tracers
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
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
        Radiation configuration (schemes: "gray", "rrtmgp", "none").
    convection : ConvectionConfig
        Convection configuration (schemes: "sbm", "dca", "kuo",
        "mass_flux", "edmf", "zhang_mcfarlane", "kain_fritsch",
        "emanuel", "tiedtke", "bechtold", "none").
    turbulence : TurbulenceConfig
        Turbulence configuration (schemes: "smagorinsky", "louis",
        "tke", "mynn25", "clubb_lite", "clubb", "holtslag_boville",
        "ysu", "edmf", "none").
    microphysics : MicrophysicsConfig
        Microphysics configuration (schemes: "kessler", "sundqvist",
        "seifert_beheng", "morrison", "thompson", "p3", "sdm",
        "fast_sbm", "ml_emulator", "none").
    gravity_wave_drag : GravityWaveDragConfig
        Gravity wave drag configuration (schemes: "rayleigh", "lindzen",
        "mcfarlane", "hines", "prognostic_spectral", "e3sm_cam",
        "ml_emulator", "none").
    """
    radiation: RadiationConfig = RadiationConfig()
    convection: ConvectionConfig = ConvectionConfig()
    turbulence: TurbulenceConfig = TurbulenceConfig()
    microphysics: MicrophysicsConfig = MicrophysicsConfig()
    gravity_wave_drag: GravityWaveDragConfig = GravityWaveDragConfig()


def _micro_liquid_from_closure_on(microphysics_config) -> bool:
    """True when the RESOLVED microphysics sub-config hands liquid to the closure.

    Reads the materialized sub-config, not the top-level switch, so a
    hand-built ``MicrophysicsConfig(morrison=MorrisonConfig(...))`` is seen.
    """
    if microphysics_config is None:
        return False
    # ``or ""`` because getattr(obj, None, default) raises TypeError rather
    # than returning the default, and a None scheme would crash the guard
    # itself (GLM).
    sub = getattr(microphysics_config,
                  getattr(microphysics_config, "scheme", "") or "", None)
    return bool(getattr(sub, "liquid_from_closure", False))


def _clubb_liquid_partition_on(turbulence_config) -> bool:
    """Is CLUBB's cloud-liquid exchange selected on this RESOLVED config?

    Reads the sub-config dispatch will actually run, not the experiment-level
    flag: ``TurbulenceConfig.clubb`` defaults to None and the factory
    substitutes a fresh ``CLUBBConfig()``, and an authoritative
    ``turbulence_override`` can carry the lever without the flag ever being set.
    """
    if getattr(turbulence_config, "scheme", None) != "clubb":
        return False
    from legoesm.atmosphere.physics.turbulence.integration import (
        materialize_sub_config,
    )
    sub = getattr(materialize_sub_config(turbulence_config), "clubb", None)
    return bool(getattr(sub, "liquid_partition", False))


def make_physics(
    config: PhysicsConfig,
    model_type: str = "hydrostatic",
    dt: float = 300.0,  # coeff-ok: default physics timestep [s]
    column_mesh=None,
    sfc_albedo_override=None,
    sfc_emissivity_override=None,
    need_rad: bool = True,
    f_land=None,
    land_beta: float = 1.0,
    budget_ledger: bool = False,
    budget_ledger_level_weight=None,
    physics_cadence: str = "off",
    physics_cadence_steps: int = 1,
    cld_macmic_num_steps: int = 1,
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
    need_rad : bool, optional
        Radiation sub-cycle selector (hydrostatic / MPAS paths only).
        ``True`` (default) builds the full-physics variant that solves
        radiation every call AND caches the resulting heating tendency in
        ``PhysicsState.rad_heating``.  ``False`` builds the held-radiation
        variant that SKIPS the RRTMGP/gray solve and re-uses the cached
        ``rad_heating`` instead — the driver alternates the two by
        ``step % rad_update_steps`` to run radiation on a coarse (e.g.
        1-hour) cadence (CESM/E3SM standard) while every other module runs
        every step.  When radiation is ``scheme="none"`` the flag is inert.

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
    # MPAS land surface boundary knobs (f_land + land_beta): consumed by the
    # MPAS turbulence factory only.  Reject loudly elsewhere — the FV/spectral
    # pipelines have their own land tile, so accepting the args there would be
    # a silently-inert configuration (the 2026-07-23 --slab-land-active lesson).
    if model_type != "mpas" and (f_land is not None or land_beta != 1.0):
        raise ValueError(
            "f_land/land_beta are MPAS-only land surface boundary knobs "
            f"(model_type={model_type!r} would silently ignore them)."
        )
    if (sfc_albedo_override is not None or sfc_emissivity_override is not None) \
            and model_type != "spectral_pe":
        raise ValueError(
            "sfc_albedo_override / sfc_emissivity_override are only wired for "
            f"model_type='spectral_pe', got {model_type!r}."
        )
    if physics_cadence not in ("off", "write"):
        raise ValueError(
            f"physics_cadence must be 'off' or 'write'; got {physics_cadence!r}")
    if physics_cadence != "off" and model_type != "mpas":
        raise ValueError(
            "physics_cadence (physics_update_steps > 1) is wired into the MPAS "
            f"lane only; model_type={model_type!r} would silently ignore it")
    if (not isinstance(physics_cadence_steps, int)
            or isinstance(physics_cadence_steps, bool) or physics_cadence_steps < 1):
        raise ValueError(
            f"physics_cadence_steps must be an int >= 1, got {physics_cadence_steps!r}")
    if physics_cadence == "off" and physics_cadence_steps != 1:
        raise ValueError(
            f"physics_cadence_steps={physics_cadence_steps} with physics_cadence="
            "'off' would be silently ignored; pass physics_cadence='write'")
    if (not isinstance(cld_macmic_num_steps, int)
            or isinstance(cld_macmic_num_steps, bool) or cld_macmic_num_steps < 1):
        raise ValueError(
            f"cld_macmic_num_steps must be an int >= 1, got {cld_macmic_num_steps!r}")
    if cld_macmic_num_steps != 1 and model_type not in ("hydrostatic", "mpas"):
        raise ValueError(
            "cld_macmic_num_steps > 1 (CAM macrophysics/microphysics sub-cycle) is "
            f"wired into the hydrostatic/MPAS combined physics only; got {model_type!r}")
    # CLUBB's liquid exchange REPLACES the host's cloud water, so the
    # microphysics has to read the replaced value.  That ordering exists only in
    # the macmic branch of ``_accumulate_step`` below: at N=1 it returns the
    # PARALLEL ``_accumulate``, where every module is evaluated on the same
    # start-of-step state and the tendencies are summed, leaving the
    # microphysical sinks computed from the pre-exchange liquid.  Autoconversion
    # goes as roughly the 2.5th power of cloud water, so that sink is wrong by
    # nearly an order of magnitude -- and water is still conserved, so nothing
    # fails.  Checked HERE, against the RESOLVED turbulence config, because the
    # experiment-level flag is not the only way in: an authoritative
    # turbulence_override carrying CLUBBConfig(liquid_partition=True), or a
    # direct make_physics call, both reach this factory without it (codex).
    # The MICROPHYSICS half alone deletes the model's only liquid source, so it
    # is the more dangerous half to reach by itself -- the mirror of the
    # turbulence-side guard below, and the same defect class both reviewers
    # found on that side.  A direct ``make_physics`` call bypasses every
    # ExperimentConfig guard, and they all key off the turbulence side.
    if (_micro_liquid_from_closure_on(config.microphysics)
            and not _clubb_liquid_partition_on(config.turbulence)):
        raise ValueError(
            "MorrisonConfig.liquid_from_closure=True without the CLUBB liquid "
            "partition delivering liquid: the microphysics would stop "
            "condensing and nothing would replace it, leaving the model with "
            "NO cloud-liquid source at all. Enable both halves together "
            "(ExperimentConfig.clubb_liquid_partition drives them), or "
            "neither.")
    if _clubb_liquid_partition_on(config.turbulence) and cld_macmic_num_steps < 2:
        raise ValueError(
            "CLUBBConfig.liquid_partition needs cld_macmic_num_steps>=2: the "
            "closure REPLACES the host cloud water, so the microphysics must "
            "run on the replaced value, and only the macro/micro sub-cycle "
            f"applies the modules in sequence. At cld_macmic_num_steps="
            f"{cld_macmic_num_steps} they are evaluated in parallel on the same "
            "state and summed, leaving the microphysical sinks evaluated on the "
            "pre-exchange liquid.")
    if model_type == "hydrostatic":
        fn = _make_hydrostatic_combined(
            config, dt, column_mesh=column_mesh, need_rad=need_rad,
            cld_macmic_num_steps=cld_macmic_num_steps)
    elif model_type == "nonhydrostatic":
        fn = _make_nonhydrostatic_combined(config, dt)
    elif model_type == "spectral_pe":
        fn = _make_spectral_pe_combined(
            config, dt,
            sfc_albedo_override=sfc_albedo_override,
            sfc_emissivity_override=sfc_emissivity_override,
        )
    elif model_type == "mpas":
        # MPAS (Voronoi mesh) uses the unified hydrostatic combined path.
        fn = _make_hydrostatic_combined(
            config, dt, model_type="mpas", column_mesh=column_mesh,
            need_rad=need_rad, f_land=f_land, land_beta=land_beta,
            physics_cadence=physics_cadence,
            physics_cadence_steps=physics_cadence_steps,
            cld_macmic_num_steps=cld_macmic_num_steps,
            budget_ledger=budget_ledger,
            budget_ledger_level_weight=budget_ledger_level_weight)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Choose from 'hydrostatic', 'nonhydrostatic', 'spectral_pe', 'mpas'."
        )
    fn._requires_phys_state = (physics_config_requires_phys_state(config)
                               or physics_cadence != "off")
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
    from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
        gwd_carries_spectrum,
        gwd_reads_conv_heating,
    )
    # Profile-prognostic convection counts as stateful (codex round 5):
    # the bridge reads phys_state.conv_prog_profile for ZM/KF/Emanuel/
    # Tiedtke/Bechtold and falls back to zeros when the carry is absent
    # — Tiedtke concretely relaxes the previous profile into M_u_new,
    # so a dropped carry silently erases that memory every step.
    conv = convection_scheme_traits(config.convection.scheme)
    # A '+'-composite containing prognostic_spectral carries the wave-action
    # spectrum too (issue #834), so it also requires a threaded PhysicsState.
    return bool(
        turbulence_scheme_traits(config.turbulence.scheme).carries_energy
        or conv.is_scalar_prognostic
        or conv.is_profile_prognostic
        or conv.is_stochastic
        or gwd_carries_spectrum(config.gravity_wave_drag.scheme)
        # The E3SM/CAM Beres convective source reads the lagged
        # ``conv_heating`` carry; unthreaded it would launch nothing.
        or gwd_reads_conv_heating(config.gravity_wave_drag)
    )


def _aerosol_ccn_active(config: PhysicsConfig) -> bool:
    """True when the microphysics selects the aerosol-CCN specified-Nc fill.

    The ``nc_from_aerosol`` switch lives on the per-scheme microphysics
    sub-config (currently Morrison) and is only meaningful in specified-Nc
    mode (``predict_Nc=False``).  The combined-physics builder reads it here
    to keep the radiation cloud-optics droplet number consistent with the
    microphysics fill (both diagnose N_c from ``forcing["aerosol_od"]``).
    """
    mc = config.microphysics
    scheme = getattr(mc, "scheme", "none")
    if scheme == "none":
        return False
    sc = getattr(mc, scheme, None)
    return bool(
        getattr(sc, "nc_from_aerosol", False)
        and not getattr(sc, "predict_Nc", False)
    )


def _aerosol_activation_config(config: PhysicsConfig):
    """The selected microphysics scheme's ``ActivationConfig`` (or None).

    Threaded into the radiation factory alongside ``nc_from_aerosol`` so the
    radiation cloud-optics droplet number runs the SAME proxy|arg activation
    dispatch as the microphysics N_c fill — one droplet number per step for
    both indirect effects.  None (schemes without the field / default) keeps
    the radiation fill on the default proxy, byte-identical to before.
    """
    if not _aerosol_ccn_active(config):
        return None
    mc = config.microphysics
    sc = getattr(mc, getattr(mc, "scheme", "none"), None)
    return getattr(sc, "activation", None)


def _attach_lifecycle_hooks(physics_fn, tagged_fns):
    """Attach reset_state / set_time / set_T_sfc_override propagation hooks.

    Each hook forwards to every sub-physics fn in ``tagged_fns`` that
    advertises the matching attribute (radiation honours all three;
    turbulence reads T_sfc from PhysicsState).  Shared by the hydrostatic,
    non-hydrostatic and spectral combined builders — pure Python attribute
    wiring that runs outside JIT/AD.  Returns ``physics_fn`` for chaining.
    """
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
        for fn, _, _ in tagged_fns:
            st = getattr(fn, "set_T_sfc_override", None)
            if callable(st):
                st(value)

    physics_fn.reset_state = reset_state
    physics_fn.set_time = set_time
    physics_fn.set_T_sfc_override = set_T_sfc_override
    return physics_fn


# ======================================================================
# Hydrostatic
# ======================================================================

def _copy_physics_fn_attrs(dst, src):
    # every lifecycle hook / marker the driver probes on a physics fn
    dst.__dict__.update(getattr(src, "__dict__", {}))
    dst._requires_phys_state = True
    return dst


def merge_physics_cache(prev, new):
    """The new tendency pytree in the STRUCTURE of the previous cache.

    Every field the new call produced replaces the cached one, re-wrapped in
    the cached Field (metadata and dtype from the seed, so the held-radiation
    variant's differently-named first module cannot change the carry
    structure); a field the new call did not produce (radiation-only
    diagnostics on a held-radiation step) keeps its cached value -- the same
    keep-last-per-slot rule the dycore's surface-diagnostic bundle applies.
    Any other structural difference raises.
    """
    if type(new) is not type(prev):
        raise ValueError(
            f"physics cache: seed is {type(prev).__name__}, step produced "
            f"{type(new).__name__}")
    out = {}
    for f in type(prev)._fields:
        vp, vn = getattr(prev, f), getattr(new, f, None)
        if f == "tracer_tendencies":
            if vp is None:
                if vn is not None:
                    raise ValueError("physics cache: the seed has no tracer "
                                     "tendencies but this step produced some")
                out[f] = None
            elif vn is None:
                out[f] = vp
            else:
                if set(vn) != set(vp):
                    raise ValueError(
                        "physics cache: tracer set changed between the seed "
                        f"{sorted(vp)} and this step {sorted(vn)}")
                out[f] = {k: vp[k].replace(data=_same_shape(f, vp[k].data, vn[k].data))
                          for k in vp}
            continue
        elif vp is None:
            if vn is not None:
                raise ValueError(
                    f"physics cache: the seed has no {f!r} but this step produced "
                    "one; seed the cache from the full variant")
            out[f] = None
        elif vn is None:
            out[f] = vp
        elif isinstance(vp, Field):
            out[f] = vp.replace(data=_same_shape(f, vp.data, vn.data))
        else:
            out[f] = _same_shape(f, jnp.asarray(vp), jnp.asarray(vn))
    return type(prev)(**out)


def _same_shape(name, cached, new):
    if new.shape != cached.shape:
        raise ValueError(
            f"physics cache: {name} shape changed from the seed {cached.shape} "
            f"to {new.shape}")
    return new.astype(cached.dtype)


# leafless marker `physics_cache_seed` puts in ``held_physics`` while tracing
# (a bare object is not a pytree leaf jax.eval_shape can abstract)
_SEEDING = ()


# The tendency fields the dycore integrates into the STATE (``state + dt *
# tend``); everything else on the tendency pytree (precip, surface / TOA
# fluxes, ledger rows) is a diagnostic RATE the driver accumulates per step.
_STATE_TENDENCY_FIELDS = ("du_dt", "dv_dt", "dT_dt", "dp_s_dt", "dphis_dt",
                          "tracer_tendencies")


def _scale_state_tendencies(tend, factor):
    """``tend`` with the state-changing tendencies times ``factor`` (a static
    Python number; ``0`` gives exact zeros, never ``0 * NaN``) and every
    diagnostic field untouched."""
    def _one(f):
        if factor == 0:
            return f.replace(data=jnp.zeros_like(f.data))
        return f.replace(data=f.data * factor)
    upd = {}
    for name in _STATE_TENDENCY_FIELDS:
        v = getattr(tend, name, None)
        if v is None:
            continue
        upd[name] = ({k: _one(f) for k, f in v.items()} if isinstance(v, dict)
                     else _one(v))
    return tend._replace(**upd)


def _with_physics_cache(physics_fn, n_steps=1):
    """The physics ("write") step of the cadence.

    CAM-FV sequence (``physics_update`` applies ``ptend * ztodt`` to the state
    once per physics step, ``uv3s_update`` likewise for the winds, and the
    dynamics substeps run unforced): the state tendencies are returned times
    ``n_steps`` so the dycore's ``state + dt * tend`` on THIS step is the whole
    ``dt * n_steps`` physics increment, and the held steps in between apply
    zero (:func:`held_physics_variant`).  The cache keeps the UNSCALED rates
    (window-mean precip / fluxes / ledger rows for the accumulators).
    ``n_steps == 1`` returns the tendency unchanged.
    """
    def cached_physics_fn(state, grid, sigma_coord, phys_state=None, forcing=None):
        out = physics_fn(state, grid, sigma_coord, phys_state=phys_state, forcing=forcing)
        tend, ps_out = (out if type(out) is tuple else (out, None))
        if ps_out is None:
            ps_out = phys_state
        if ps_out is None:
            raise ValueError(
                "physics cadence 'write' needs a PhysicsState carry to hold the "
                "cache; step was called with phys_state=None")
        prev = phys_state.held_physics
        applied = tend if n_steps == 1 else _scale_state_tendencies(tend, n_steps)
        if prev == _SEEDING:
            return applied, ps_out
        if prev is None:
            raise ValueError(
                "physics cadence 'write' with an unseeded PhysicsState.held_physics: "
                "seed it with physics_cache_seed() so the carry structure is fixed "
                "before the first step")
        ps_out = update_physics_state(ps_out, {"held_physics": merge_physics_cache(prev, tend)})
        return applied, ps_out
    return _copy_physics_fn_attrs(cached_physics_fn, physics_fn)


def held_physics_variant(physics_fn):
    """The held-physics step: ``PhysicsState.held_physics`` with the STATE
    tendencies zeroed (the whole physics increment was applied on the physics
    step, CAM-FV style) and the diagnostic rates (precip, surface / TOA
    fluxes, ledger rows) re-published unchanged for the per-step accumulators.

    Derived from the built ``"write"`` physics fn (no second construction of
    the module chain); carries every hook/marker of the source fn.  Apply any
    outer wrapper (Held-Suarez) to the derived fn separately.
    """
    def held_physics_fn(state, grid, sigma_coord, phys_state=None, forcing=None):
        cache = getattr(phys_state, "held_physics", None) if phys_state is not None else None
        if cache is None:
            raise ValueError(
                "held-physics step with an empty PhysicsState.held_physics: seed it with "
                "physics_cache_seed() and run a physics ('write') step first")
        # per-step INPUT fields are consumed (reset) like on any other call
        return _scale_state_tendencies(cache, 0), update_physics_state(phys_state, {})
    held_physics_fn = _copy_physics_fn_attrs(held_physics_fn, physics_fn)
    # reads no neighbour cell (no state at all): the MPI step skips the halo
    # exchange before it
    held_physics_fn._column_local = True
    return held_physics_fn


def physics_cache_seed(physics_fn, state, grid, sigma_coord, phys_state, forcing=None):
    """Zero-filled ``held_physics`` with the full variant's output structure.

    Traced with ``jax.eval_shape`` (no compute, no compile), so the carry has
    its final structure BEFORE the first jitted step and the full variant is
    compiled once.  Pass the state in the precision the step hands the
    physics (the dycore casts to the compute policy); the merge also casts
    every write to the seed's dtype, so the carry's dtypes are fixed here.
    """
    import jax
    def _f(s, ps, f):
        out = physics_fn(s, grid, sigma_coord, phys_state=ps, forcing=f)
        return out[0] if type(out) is tuple else out
    shapes = jax.eval_shape(
        _f, state, phys_state._replace(held_physics=_SEEDING), forcing)
    return jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), shapes)


def _make_hydrostatic_combined(config: PhysicsConfig, dt: float,
                               model_type: str = "hydrostatic",
                               column_mesh=None,
                               need_rad: bool = True,
                               f_land=None,
                               land_beta: float = 1.0,
                               budget_ledger: bool = False,
                               budget_ledger_level_weight=None,
                               physics_cadence: str = "off",
                               physics_cadence_steps: int = 1,
                               cld_macmic_num_steps: int = 1) -> Callable:
    """Combined physics for any hydrostatic model (cubed-sphere, lat-lon, MPAS).

    ``cld_macmic_num_steps`` (CAM6 namelist ``cld_macmic_num_steps``, 3 for
    CLUBB+MG2 at dtime=1800): 1 (default) is the byte-identical parallel
    split of every module on the pre-physics state.  N > 1 ports the CAM
    ``tphysbc`` macro/micro loop: deep convection is applied to an
    intermediate state first (CAM ``physics_update`` after
    ``convect_deep_tend``), then turbulence (macrophysics) and microphysics
    run N times sequentially at ``dt/N`` on that state, each sub-step's
    increment applied before the next (``cld_macmic_ztodt``); their
    tendencies, precipitation and ledger rows are averaged over the window
    (CAM's ``1/N`` scaling).  Departures from CAM: radiation and gravity-wave
    drag see the PRE-physics state (CAM: post-macmic / post-coupler) exactly
    as the N=1 parallel split already does, and the whole convective
    increment (including detrained condensate) is applied before sub-step 1
    rather than spread over the sub-steps (CAM hands ``dlf`` per sub-step).

    ``physics_cadence``: ``"off"`` (default, byte-identical to before);
    ``"write"`` additionally stores the full tendency pytree of every call in
    ``PhysicsState.held_physics`` (seed it with :func:`physics_cache_seed`)
    and returns the state tendencies times ``physics_cadence_steps`` so the
    dycore applies the whole physics increment on this step;
    :func:`held_physics_variant` derives the step that applies zero state
    tendency and re-publishes the cached diagnostic rates in between.

    Uses the unified ``HydrostaticTendencies`` with optional ``dv_dt``.
    When *model_type* is ``"mpas"``, the radiation factory is called
    with ``"mpas"`` so that lat/lon extraction uses mesh.latCell/lonCell.

    Issue #273 follow-up: when ``column_mesh`` is supplied, the
    per-column radiation kernel runs sharded across the mesh.
    """
    # Static Python gate (a closure constant, never a traced value): False
    # keeps every ledger branch out of the graph entirely, so the default path
    # is byte-identical.  This is the documented feature-gating exception --
    # jnp.where would trace BOTH branches.
    _budget_ledger = bool(budget_ledger)
    # Optional (nlev,) per-layer weight restricting every ledger row to a
    # vertical band.  The column ledger cannot see a vertical-REDISTRIBUTION
    # bias: convection's column water row is exactly zero by construction, so a
    # scheme that moves water from the boundary layer into the mid-troposphere
    # is invisible in it.  None = full column, byte-identical.
    _ledger_level_weight = budget_ledger_level_weight
    tagged_fns = []
    # Aerosol-CCN specified-Nc coupling: the microphysics factory self-
    # detects the switch from its own sub-config, but the radiation factory
    # has no microphysics config, so the flag is threaded explicitly so its
    # cloud-optics droplet number (Twomey r_eff) matches the microphysics
    # fill.  False (default) keeps both paths byte-identical.
    _nc_from_aerosol = _aerosol_ccn_active(config)
    _activation_cfg = _aerosol_activation_config(config)
    # CLUBB sub-grid cloud fraction -> radiation routing (marine-Sc albedo lever).
    # A moist higher-order closure diagnoses a less-overcast cloud fraction than
    # the RH grid-scale scheme; route it to the cloud optics when the user opts in
    # AND a cf-producing closure is active.  ONLY diagnostic CLUBB
    # (turbulence.scheme='clubb', CLUBBConfig.prognostic=False) writes the
    # ``PhysicsState.cloud_fraction`` carry today (clubb_lite dropped its PDF
    # moments; prognostic CLUBB carries packed moments, not a diagnosed cf).
    # Requiring the producer keeps the radiation override off the zero-init carry
    # (which would spuriously clear clouds).  Misconfiguration is LOUD, never a
    # silent no-op (dispatch-hardening).
    # BOTH CLUBB paths now publish a PDF cloud fraction.  The prognostic one
    # used to be refused here on the grounds that it emitted "packed moments, no
    # diagnosed cloud fraction" -- true of the OUTPUT, never of the closure: the
    # moment advance computes the post-advance PDF cloud fraction and simply did
    # not hand it out.  It does now, so the prognostic path is the BETTER source,
    # not an unsupported one: its variance is carried state rather than a
    # mixing-length estimate re-derived each step.
    _turb_produces_cf = config.turbulence.scheme == "clubb"
    _n_macmic = int(cld_macmic_num_steps)
    if _n_macmic > 1 and (config.turbulence.scheme == "none"
                          and config.microphysics.scheme == "none"):
        raise ValueError(
            f"cld_macmic_num_steps={_n_macmic} sub-cycles turbulence and "
            "microphysics, but both are 'none'; set it to 1")
    # Sub-cycled modules are BUILT with the sub-step length: implicit
    # diffusion, CLUBB's moment advance and Morrison's process/sedimentation
    # integration all take their own ``dt``.
    _dt_sub = dt / _n_macmic
    if config.radiation.use_clubb_cloud_fraction and not _turb_produces_cf:
        raise ValueError(
            "RadiationConfig.use_clubb_cloud_fraction=True requires a "
            "cloud-fraction-producing turbulence closure "
            "(turbulence.scheme='clubb', diagnostic or prognostic); got "
            f"turbulence.scheme={config.turbulence.scheme!r}."
        )
    _use_clubb_cf = config.radiation.use_clubb_cloud_fraction
    # MG2 in-cloud warm rain reads the cloud fraction CLUBB wrote in the SAME
    # macmic sub-step (CAM order); at N=1 the modules run in parallel and it
    # would read the previous step's value.
    if (getattr(getattr(config.microphysics, config.microphysics.scheme, None),
                "warm_rain_incloud", False)
            and (not _turb_produces_cf or _n_macmic < 2)):
        raise ValueError(
            "MorrisonConfig.warm_rain_incloud=True needs CLUBB turbulence and "
            "cld_macmic_num_steps>=2 (the microphysics reads the cloud "
            "fraction CLUBB diagnosed in the same sub-step); got "
            f"turbulence.scheme={config.turbulence.scheme!r}, "
            f"cld_macmic_num_steps={_n_macmic}.")
    if config.radiation.scheme != "none":
        tagged_fns.append((
            make_radiation_physics(
                config.radiation, model_type, column_mesh=column_mesh,
                nc_from_aerosol=_nc_from_aerosol,
                activation_config=_activation_cfg,
                use_clubb_cloud_fraction=_use_clubb_cf,
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
            make_turbulence_physics(config.turbulence, model_type, _dt_sub,
                                    f_land=f_land, land_beta=land_beta),
            True,
            _turb_field,
        ))
    if config.microphysics.scheme != "none":
        # The run's BUILT cloud config (MPAS lane: _standalone_cloud_config),
        # never a default one: the aist rh ramps are run parameters.
        tagged_fns.append((
            make_microphysics_physics(
                config.microphysics, model_type, _dt_sub,
                cloud_config=config.radiation.cloud_config),
            False, None))
    if config.gravity_wave_drag.scheme != "none":
        tagged_fns.append((make_gwd_physics(config.gravity_wave_drag, model_type, dt), True, "gwd_spectrum"))
    # Macmic group of each module, PARALLEL to ``tagged_fns`` (same append
    # order): "pre" = applied to the intermediate state before the loop
    # (deep convection), "sub" = sub-cycled (turbulence, microphysics),
    # "once" = evaluated once on the pre-physics state (radiation, GWD).
    _macmic_group = []
    if config.radiation.scheme != "none":
        _macmic_group.append("once")
    if config.convection.scheme != "none":
        _macmic_group.append("pre")
    if config.turbulence.scheme != "none":
        _macmic_group.append("sub")
    if config.microphysics.scheme != "none":
        _macmic_group.append("sub")
    if config.gravity_wave_drag.scheme != "none":
        _macmic_group.append("once")
    if len(_macmic_group) != len(tagged_fns):
        raise AssertionError(
            f"macmic group map has {len(_macmic_group)} entries for "
            f"{len(tagged_fns)} physics modules — the two append sequences "
            "have drifted; every module needs exactly one group.")

    # Radiation sub-cycle plumbing.  Radiation is always ``tagged_fns[0]``
    # when configured (appended first above), so in the accumulator below
    # ``first.dT_dt`` IS the radiative heating contribution — that is what
    # the full-physics variant caches into ``PhysicsState.rad_heating`` and
    # the held-radiation variant (``need_rad=False``) re-uses.
    # ``_non_rad_fns`` is the module list with radiation removed.
    _has_rad = config.radiation.scheme != "none"
    _non_rad_fns = tagged_fns[1:] if _has_rad else tagged_fns
    _non_rad_groups = _macmic_group[1:] if _has_rad else _macmic_group

    # --- Per-process budget-ledger row map (#1311 MPAS attribution) --------
    # A list PARALLEL to ``tagged_fns`` giving each module's ledger row.  Kept
    # parallel rather than widening the (fn, accepts_ps, field_name) tuple:
    # that tuple is unpacked in three lifecycle-hook loops and in two OTHER
    # model-type factories, none of which need the row.
    # GWD maps to ``other_physics``, matching the FV pipeline, where GWD is
    # part of the other_physics residual.
    _ledger_row_of = []
    if _budget_ledger:
        from legoesm.diagnostics.process_ledger import (
            ROW_CONVECTION, ROW_MICROPHYSICS, ROW_OTHER, ROW_RADIATION,
            ROW_TURBULENCE,
        )
        if config.radiation.scheme != "none":
            _ledger_row_of.append(ROW_RADIATION)
        if config.convection.scheme != "none":
            _ledger_row_of.append(ROW_CONVECTION)
        if config.turbulence.scheme != "none":
            _ledger_row_of.append(ROW_TURBULENCE)
        if config.microphysics.scheme != "none":
            _ledger_row_of.append(ROW_MICROPHYSICS)
        if config.gravity_wave_drag.scheme != "none":
            _ledger_row_of.append(ROW_OTHER)
        # Drift guard: the row list is built by REPEATING the append order
        # above, so a module added to tagged_fns without a matching row here
        # would silently mis-attribute every later module's tendencies.
        if len(_ledger_row_of) != len(tagged_fns):
            raise AssertionError(
                f"budget-ledger row map has {len(_ledger_row_of)} entries for "
                f"{len(tagged_fns)} physics modules — the two append sequences "
                f"have drifted; every module needs exactly one ledger row."
            )
    _non_rad_rows = _ledger_row_of[1:] if _has_rad else _ledger_row_of

    # Water MASS species only — the canonical shared tuple (number
    # concentrations deliberately excluded; see its docstring).
    from legoesm.diagnostics.process_ledger import LEDGER_WATER_SPECIES
    _LEDGER_WATER = LEDGER_WATER_SPECIES

    def _module_ledger_row(t, p_s, dsigma):
        """One module's per-column ``[water, energy]`` from its OWN tendency.

        Simpler than the FV capture, which must reach into intermediate
        variables because its rain bypasses the column store.  Here every
        module returns its complete tendency, so the module's column-store
        contribution IS the mass integral of its water-species and dT
        tendencies — no per-scheme special-casing, and the module rows sum to
        the combined total by construction (pinned by a test).
        """
        from legoesm.diagnostics.process_ledger import ledger_entry_column
        dq = None
        tt = t.tracer_tendencies
        if tt is not None:
            for _k in _LEDGER_WATER:
                if _k in tt:
                    dq = tt[_k].data if dq is None else dq + tt[_k].data
        return ledger_entry_column(dq, t.dT_dt.data, p_s, dsigma,
                                   level_weight=_ledger_level_weight)

    def _zero_tendencies(state, has_v):
        dims_T = state.T.dims
        dims_ps = state.p_s.dims
        dv_dt_zero = None
        if has_v:
            dv_dt_zero = Field(
                data=jnp.zeros_like(state.v.data), name="dv_dt_phys",
                dims=state.v.dims, units="m/s^2",
            )
        return HydrostaticTendencies(
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

    def _accumulate(fns, state, grid, sigma_coord, phys_state, forcing,
                    ledger_rows=None):
        """Sum the tendencies of every module in ``fns`` (non-empty list).

        Returns ``(du_dt, dv_dt, dT_dt, dp_s_dt, dphis_dt,
        combined_tracer_tends, phys_updates, first)``.  ``first`` is the
        first module's tendency object — used both for ``Field`` templates
        and (when ``fns`` leads with radiation) to read the radiative
        heating contribution ``first.dT_dt``.
        """
        phys_updates = {}
        fn0, accepts_ps, field_name = fns[0]
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
            _kw0 = {}
            if getattr(fn0, "_wants_forcing", False):
                _kw0["forcing"] = forcing
            if getattr(fn0, "_wants_phys_state_ro", False):
                # Read-only phys_state consumer (radiation reading the CLUBB
                # sub-grid cloud-fraction carry): forward phys_state but keep the
                # single-return contract — accepts_ps=False, so no carry is
                # written back.  Byte-identical when unset (empty kwargs).
                _kw0["phys_state"] = phys_state
            first = fn0(state, grid, sigma_coord, **_kw0)
        du_dt = first.du_dt.data
        dv_dt = first.dv_dt.data if first.dv_dt is not None else None
        dT_dt = first.dT_dt.data
        dp_s_dt = first.dp_s_dt.data
        dphis_dt = first.dphis_dt.data
        # Surface precip [kg/m^2/s]: summed across whichever modules produce it
        # (microphysics does; radiation/turbulence/GWD do not -> None). Kept as a
        # diagnostic (not a tendency) so the lean MPAS loop can export it.
        precip_accum = (first.precip.data
                        if getattr(first, "precip", None) is not None else None)
        # Per-module surface/TOA diagnostic fields for the lean-loop CMOR
        # feed: each comes from exactly ONE module (TOA trio + clear-sky TOA
        # pair from radiation, shflx/lhflx from turbulence), so
        # first-non-None across modules is the correct combine (no summing).
        # The ``*_clr`` pair is None unless RadiationConfig.clear_sky_diag is
        # on (#843 lean-lane port) — a None extra is never attached.
        _DIAG_FIELDS = ("sw_up_toa", "lw_up_toa", "sw_down_toa",
                        "shflx_sfc", "lhflx_sfc",
                        "sw_up_toa_clr", "lw_up_toa_clr",
                        "sed_substeps_required", "evap_sfc")
        sfc_diag_extras = {k: getattr(first, k, None) for k in _DIAG_FIELDS}

        # Per-process ledger: capture each module's row from its OWN complete
        # tendency, before the sum below fuses them.  ``ledger_rows`` is the
        # parallel row-index list; None (default) keeps this out of the graph.
        _led = None
        if _budget_ledger and ledger_rows is not None:
            from legoesm.diagnostics.process_ledger import zero_ledger_column
            _bl_dsigma = sigma_coord.dsigma
            _bl_ps = state.p_s.data
            # Budget accumulator in the WIDEST available float (f64 under
            # x64, f32 otherwise — canonicalize_dtype respects the setting):
            # per-module rows arrive in MIXED precision (f32 radiation, f64
            # Louis), and seeding from any single row scatters the wider rows
            # into a narrower ledger — a silent downcast, and a JAX
            # FutureWarning slated to become an error.  Every add casts
            # explicitly.
            from jax.dtypes import canonicalize_dtype
            _led_dtype = canonicalize_dtype(jnp.float64)
            _led = zero_ledger_column(_bl_ps.shape[0], dtype=_led_dtype)
            _led = _led.at[:, ledger_rows[0], :].add(
                _module_ledger_row(first, _bl_ps, _bl_dsigma)
                .astype(_led_dtype))

        # Accumulate tracer tendencies from all physics modules
        combined_tracer_tends = {}
        if first.tracer_tendencies is not None:
            for k, v in first.tracer_tendencies.items():
                combined_tracer_tends[k] = v.data

        for _i_mod, (fn, accepts_ps, field_name) in enumerate(fns[1:], start=1):
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
                _kw = {}
                if getattr(fn, "_wants_forcing", False):
                    _kw["forcing"] = forcing
                if getattr(fn, "_wants_phys_state_ro", False):
                    _kw["phys_state"] = phys_state
                t = fn(state, grid, sigma_coord, **_kw)
            if _led is not None:
                # ``.add`` not ``.set``: two modules can legitimately map to
                # the same row (e.g. GWD and any future module on
                # other_physics), and a set would silently discard the first.
                _led = _led.at[:, ledger_rows[_i_mod], :].add(
                    _module_ledger_row(t, _bl_ps, _bl_dsigma)
                    .astype(_led.dtype))
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

            if getattr(t, "precip", None) is not None:
                precip_accum = (t.precip.data if precip_accum is None
                                else precip_accum + t.precip.data)

            for _k in _DIAG_FIELDS:
                if sfc_diag_extras[_k] is None:
                    sfc_diag_extras[_k] = getattr(t, _k, None)

        return (du_dt, dv_dt, dT_dt, dp_s_dt, dphis_dt,
                combined_tracer_tends, phys_updates, first, precip_accum,
                sfc_diag_extras, _led)

    def _build_combined(first, du_dt, dv_dt, dT_dt, dp_s_dt, dphis_dt,
                        combined_tracer_tends):
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
        return HydrostaticTendencies(
            du_dt=first.du_dt.replace(data=du_dt),
            dT_dt=first.dT_dt.replace(data=dT_dt),
            dp_s_dt=first.dp_s_dt.replace(data=dp_s_dt),
            dphis_dt=first.dphis_dt.replace(data=dphis_dt),
            dv_dt=combined_dv_dt,
            tracer_tendencies=tracer_tends_out,
        )

    def _attach_sfc_precip(combined, first, precip_accum):
        """Carry surface precip [kg/m^2/s] on the combined tendency (dropped by
        _build_combined) so the lean MPAS loop can export it. No-op / byte-
        identical when microphysics produced no precip (precip_accum is None)."""
        if precip_accum is None:
            return combined
        return combined._replace(precip=Field(
            data=precip_accum, name="precip",
            dims=first.dp_s_dt.dims, units="kg/m^2/s"))

    def _attach_sfc_diag_extras(combined, extras):
        """Carry the per-module surface/TOA diagnostic Fields (TOA trio from
        radiation, shflx/lhflx from turbulence) on the combined tendency —
        _build_combined constructs a fresh tendency that drops them. No-op /
        byte-identical when every extra is None (radiation+turbulence off)."""
        _set = {k: v for k, v in extras.items() if v is not None}
        return combined._replace(**_set) if _set else combined

    # Diagnostic extras that are COUNTS (int): combined across macmic
    # sub-steps as the maximum, never weighted-averaged like the fluxes.
    _COUNT_DIAG_FIELDS = frozenset({"sed_substeps_required"})
    assert _COUNT_DIAG_FIELDS <= {"sw_up_toa", "lw_up_toa", "sw_down_toa",
                                  "shflx_sfc", "lhflx_sfc", "sw_up_toa_clr",
                                  "lw_up_toa_clr", "sed_substeps_required",
                                  "evap_sfc"}

    def _advance_state(state, du_dt, dv_dt, dT_dt, dp_s_dt, tracer_tends, dt_x):
        """``state + dt_x * tendency`` (CAM ``physics_update``); ``phis`` fixed."""
        kw = dict(
            u=state.u.replace(data=state.u.data + dt_x * du_dt),
            T=state.T.replace(data=state.T.data + dt_x * dT_dt),
            p_s=state.p_s.replace(data=state.p_s.data + dt_x * dp_s_dt),
        )
        if state.v is not None and dv_dt is not None:
            kw["v"] = state.v.replace(data=state.v.data + dt_x * dv_dt)
        if state.tracers is not None and tracer_tends:
            kw["tracers"] = {
                k: (f.replace(data=f.data + dt_x * tracer_tends[k])
                    if k in tracer_tends else f)
                for k, f in state.tracers.items()
            }
        return state._replace(**kw)

    def _add_scaled(acc, r, w):
        """``acc + w * r`` over the ``_accumulate`` result tuple (None-aware).

        ``first`` keeps the first non-None; ``phys_updates`` merge (later
        wins, so a sub-cycled carry leaves the loop at its final value).
        """
        if acc is None:
            acc = (None, None, None, None, None, {}, {}, None, None,
                   {k: None for k in r[9]}, None)
        (du, dv, dT, dps, dphis, tr, upd, first, pr, ex, led) = acc
        (du_r, dv_r, dT_r, dps_r, dphis_r, tr_r, upd_r, first_r, pr_r,
         ex_r, led_r) = r

        def _ws(a, b):
            if b is None:
                return a
            return w * b if a is None else a + w * b

        tr = dict(tr)
        for k, v in tr_r.items():
            tr[k] = _ws(tr.get(k), v)
        upd = dict(upd)
        upd.update(upd_r)
        ex = dict(ex)
        for k, v in ex_r.items():
            if v is None:
                continue
            if k in _COUNT_DIAG_FIELDS:
                # a COUNT: the window's worst sub-step, not an average
                ex[k] = (v if ex.get(k) is None
                         else ex[k].replace(data=jnp.maximum(ex[k].data, v.data)))
                continue
            ex[k] = (v.replace(data=w * v.data) if ex.get(k) is None
                     else ex[k].replace(data=ex[k].data + w * v.data))
        return (_ws(du, du_r), _ws(dv, dv_r), _ws(dT, dT_r), _ws(dps, dps_r),
                _ws(dphis, dphis_r), tr, upd,
                first if first is not None else first_r,
                _ws(pr, pr_r), ex, _ws(led, led_r))

    def _carry_within_call(ps, upd):
        """Intermediate carry update INSIDE one physics call: the per-call
        inputs (dynamics tendencies, prescribed surface fluxes) stay visible
        to every later sub-module; the call's final update clears them."""
        if ps is None:
            return None
        keep = {k: getattr(ps, k) for k in PHYSSTATE_PER_CALL_INPUTS}
        return update_physics_state(ps, {**keep, **upd})

    def _accumulate_step(fns, groups, state, grid, sigma_coord, phys_state,
                         forcing, ledger_rows=None):
        """``_accumulate`` (N=1, byte-identical) or the CAM macmic loop."""
        if _n_macmic == 1:
            return _accumulate(fns, state, grid, sigma_coord, phys_state,
                               forcing, ledger_rows=ledger_rows)
        rows = ledger_rows if ledger_rows is not None else [None] * len(fns)
        sel = {g: [(f, r) for f, gg, r in zip(fns, groups, rows) if gg == g]
               for g in ("once", "pre", "sub")}

        def _run(fns_rows, st, ps):
            return _accumulate([f for f, _ in fns_rows], st, grid, sigma_coord,
                               ps, forcing,
                               ledger_rows=([r for _, r in fns_rows]
                                            if ledger_rows is not None else None))
        # Evaluation order: the group holding ``fns[0]`` first, so ``first``
        # (radiation when configured) is the template exactly as at N=1.
        # ``first`` is only ever a Field TEMPLATE and, when radiation leads,
        # the radiative heating; on a sub-led list it is sub-step 1's dt/N
        # tendency and must not be read quantitatively.
        acc = None
        state_m = state
        ps_m = phys_state
        if groups[0] == "once" and sel["once"]:
            acc = _add_scaled(acc, _run(sel["once"], state, phys_state), 1.0)
        if sel["pre"]:
            r_pre = _run(sel["pre"], state, phys_state)
            acc = _add_scaled(acc, r_pre, 1.0)
            state_m = _advance_state(state_m, r_pre[0], r_pre[1], r_pre[2],
                                     r_pre[3], r_pre[5], dt)
            # CAM order: clubb_tend_cam reads THIS step's convect_deep_tend
            # output (cmfmc, dlf), so the convective carries are visible to
            # the sub-cycle (the N=1 split hands them over one step late).
            ps_m = _carry_within_call(ps_m, r_pre[6])
        # Each sub-cycled module is applied (state AND carry) before the next
        # one runs: CAM physics_update after clubb_tend_cam, then
        # microp_driver_tend on the updated state (physpkg.F90:2097-2101).
        for _i in range(_n_macmic):
            for _fr in sel["sub"]:
                r_sub = _run([_fr], state_m, ps_m)
                state_m = _advance_state(state_m, r_sub[0], r_sub[1], r_sub[2],
                                         r_sub[3], r_sub[5], _dt_sub)
                ps_m = _carry_within_call(ps_m, r_sub[6])
                acc = _add_scaled(acc, r_sub, 1.0 / _n_macmic)
        if groups[0] != "once" and sel["once"]:
            acc = _add_scaled(acc, _run(sel["once"], state, phys_state), 1.0)
        return acc

    def physics_fn(state, grid, sigma_coord, phys_state=None, forcing=None):
        has_v = state.v is not None

        # ---- Held-radiation sub-cycle variant (need_rad=False) ----
        # Skip the RRTMGP/gray solve; add the cached heating from the most
        # recent radiation step (``phys_state.rad_heating``).  The driver
        # alternates this variant with the full one by ``step %
        # rad_update_steps``; step 0 of every sub-cycle is a full step, so
        # the cache is always populated before a held step reads it.
        if (not need_rad) and _has_rad:
            cached_rad = (phys_state.rad_heating
                          if phys_state is not None else None)
            if not _non_rad_fns:
                # Radiation was the only active module: held heating only.
                zt = _zero_tendencies(state, has_v)
                dT = zt.dT_dt.data
                if cached_rad is not None:
                    # cached_rad is column-shaped (ncol, nlev); restore the
                    # native layout to add to the native tendency data.
                    dT = dT + cached_rad.reshape(dT.shape)
                return zt._replace(dT_dt=zt.dT_dt.replace(data=dT)), phys_state
            (du_dt, dv_dt, dT_dt, dp_s_dt, dphis_dt,
             combined_tracer_tends, phys_updates, first,
             precip_accum, sfc_diag_extras, _led) = _accumulate_step(
                _non_rad_fns, _non_rad_groups, state, grid, sigma_coord,
                phys_state, forcing,
                ledger_rows=(_non_rad_rows if _budget_ledger else None))
            if cached_rad is not None:
                # cached_rad is column-shaped (ncol, nlev); restore native layout.
                dT_dt = dT_dt + cached_rad.reshape(dT_dt.shape)
                if _led is not None:
                    # The held sub-step adds the CACHED radiative heating here,
                    # after _accumulate (radiation is not in _non_rad_fns), so
                    # without this the radiation row would read zero on every
                    # held step and the ledger would not sum to the applied
                    # total.  Pinned by a held-step closure test.
                    from legoesm.diagnostics.process_ledger import (
                        ROW_RADIATION, ledger_entry_column,
                    )
                    _led = _led.at[:, ROW_RADIATION, :].add(
                        ledger_entry_column(
                            None, cached_rad.reshape(dT_dt.shape),
                            state.p_s.data, sigma_coord.dsigma,
                            level_weight=_ledger_level_weight)
                        .astype(_led.dtype))
            combined = _build_combined(
                first, du_dt, dv_dt, dT_dt, dp_s_dt, dphis_dt,
                combined_tracer_tends)
            # Held-radiation sub-step: no fresh sw/lw solve, but precip (from
            # microphysics, which runs every step) is still exported — same
            # for the turbulence shflx/lhflx extras (radiation's TOA extras
            # are None here; the consumer keeps the last radiation-step
            # value slot-wise, exactly like sw/lw net).
            combined = _attach_sfc_precip(combined, first, precip_accum)
            combined = _attach_sfc_diag_extras(combined, sfc_diag_extras)
            if _led is not None:
                combined = combined._replace(ledger_rows=_led)
            # rad_heating is carried UNCHANGED (not in phys_updates).
            phys_state_out = update_physics_state(phys_state, phys_updates)
            return combined, phys_state_out

        # ---- Full-physics variant (radiation solved this step) ----
        if not tagged_fns:
            return _zero_tendencies(state, has_v), None
        (du_dt, dv_dt, dT_dt, dp_s_dt, dphis_dt,
         combined_tracer_tends, phys_updates, first,
         precip_accum, sfc_diag_extras, _led) = _accumulate_step(
            tagged_fns, _macmic_group, state, grid, sigma_coord, phys_state,
            forcing, ledger_rows=(_ledger_row_of if _budget_ledger else None))
        # Cache the radiative heating contribution for the held sub-cycle
        # steps.  Radiation is tagged_fns[0], so ``first.dT_dt`` is exactly
        # its contribution before any other module is summed.
        if _has_rad and phys_state is not None:
            # PhysicsState carries are COLUMN-shaped (ncol, nlev) — that is how
            # init_physics_state seeds rad_heating (and every other carry).  The
            # radiation tendency Field data is in the model's NATIVE layout: 2D
            # (ncol, nlev) for MPAS but 4D (face, x, y, nlev) for hydrostatic/SCM.
            # Storing the native array would flip the lax.scan carry shape
            # 2D->4D after step 0 (the SCM/hydrostatic radiation-substep bug);
            # flatten the horizontal axes to the canonical column shape so the
            # carry type is stable for ALL models (no-op for MPAS).  The held
            # consumers below reshape it back to native before adding.
            _rad = first.dT_dt.data
            phys_updates["rad_heating"] = _rad.reshape(-1, _rad.shape[-1])
        combined = _build_combined(
            first, du_dt, dv_dt, dT_dt, dp_s_dt, dphis_dt,
            combined_tracer_tends)
        if _has_rad:
            # Carry the surface net radiative fluxes (radiation is tagged_fns[0],
            # so ``first`` holds them) on the combined tendency, so the lean MPAS
            # coupled loop can export sw/lw net to the coupler (_build_combined
            # constructs a fresh tendency that drops these diagnostic fields).
            combined = combined._replace(
                sw_net_sfc=first.sw_net_sfc, lw_net_sfc=first.lw_net_sfc,
                sw_down_sfc=first.sw_down_sfc, lw_down_sfc=first.lw_down_sfc)
        # Same for surface precip (from microphysics; _build_combined drops it)
        # and the TOA/turbulent-flux diagnostic extras.
        combined = _attach_sfc_precip(combined, first, precip_accum)
        combined = _attach_sfc_diag_extras(combined, sfc_diag_extras)
        if _led is not None:
            combined = combined._replace(ledger_rows=_led)
        phys_state_out = update_physics_state(phys_state, phys_updates)
        return combined, phys_state_out

    physics_fn = _attach_lifecycle_hooks(physics_fn, tagged_fns)
    if physics_cadence == "write":
        physics_fn = _with_physics_cache(physics_fn, physics_cadence_steps)
    return physics_fn


# ======================================================================
# Non-hydrostatic
# ======================================================================

def _make_nonhydrostatic_combined(config: PhysicsConfig, dt: float) -> Callable:
    tagged_fns = []
    if config.radiation.scheme != "none":
        # CLUBB-cf routing is only wired for hydrostatic today; passing the flag
        # (rather than dropping it) makes make_radiation_physics raise loudly if a
        # user set use_clubb_cloud_fraction on the nonhydrostatic path — never a
        # silent no-op.
        tagged_fns.append((make_radiation_physics(
            config.radiation, "nonhydrostatic",
            use_clubb_cloud_fraction=config.radiation.use_clubb_cloud_fraction,
        ), False, None))
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

    return _attach_lifecycle_hooks(physics_fn, tagged_fns)


# ======================================================================
# Spectral PE
# ======================================================================

def _make_spectral_pe_combined(
    config: PhysicsConfig, dt: float,
    sfc_albedo_override=None, sfc_emissivity_override=None,
) -> Callable:
    tagged_fns = []
    # Same producer gate as the hydrostatic combined: routing the CLUBB
    # cloud fraction to radiation needs a closure that writes it, else the
    # override would read the zero-initialised carry and clear every cloud.
    if (config.radiation.use_clubb_cloud_fraction
            and config.turbulence.scheme != "clubb"):
        raise ValueError(
            "RadiationConfig.use_clubb_cloud_fraction=True requires a "
            "cloud-fraction-producing turbulence closure "
            "(turbulence.scheme='clubb', diagnostic or prognostic); got "
            f"turbulence.scheme={config.turbulence.scheme!r}."
        )
    if config.radiation.scheme != "none":
        # The spectral radiation builder has a READ side for the CLUBB
        # cloud-fraction carry (and the CAM6 deepcu carries); the flag
        # selects it, and the dispatcher below forwards ``phys_state`` to
        # the fn that advertises ``_wants_phys_state_ro``.
        tagged_fns.append((make_radiation_physics(
            config.radiation, "spectral_pe",
            sfc_albedo_override=sfc_albedo_override,
            sfc_emissivity_override=sfc_emissivity_override,
            use_clubb_cloud_fraction=config.radiation.use_clubb_cloud_fraction,
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
            if getattr(fn0, "_wants_phys_state_ro", False):
                # Read-only phys_state consumer (radiation reading the CLUBB
                # cloud-fraction / CAM6 deepcu carries): forward it but keep
                # the single-return contract -- no carry is written back.
                _fwd0 = {**_fwd0, "phys_state": phys_state}
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
                if getattr(fn, "_wants_phys_state_ro", False):
                    _fwd = {**_fwd, "phys_state": phys_state}
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

    physics_fn = _attach_lifecycle_hooks(physics_fn, tagged_fns)
    # Marker: the combined fn consumes a per-step traced ``forcing``
    # dict when any sub-physics advertises ``_wants_forcing`` (currently
    # the spectral_pe radiation factory: T_sfc/o3_vmr/aerosol_od/ghg_vmr).
    physics_fn._wants_forcing = any(
        getattr(fn, "_wants_forcing", False) for fn, _, _ in tagged_fns
    )
    return physics_fn
