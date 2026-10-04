"""Factory for ocean convection physics."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.eos import (
    compute_ocean_rho as _compute_rho,
    compute_ocean_rho_and_pressure as _compute_rho_and_pressure,
)
from legoesm.ocean.constants_config import ConstantsConfig
from legoesm.ocean.state import OceanState, OceanTendencies
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_ocean_jacobian
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.physics.convection.enhanced_diffusion import enhanced_diffusion_convection
from legoesm.ocean.physics.convection.plume import plume_convection
from legoesm.ocean.physics.tendencies import make_none_physics_fn, wrap_ocean_tendencies


def make_convection_physics(
    config: OceanConvectionConfig,
    apply_diffusion: bool = True,
    emit_momentum_viscosity: bool = True,
    eos_fn: Callable | None = None,
    constants_config: ConstantsConfig = ConstantsConfig(),
    seos_cfg=None,
) -> Callable:
    """Create an ocean convection physics function.

    Parameters
    ----------
    config : OceanConvectionConfig
    apply_diffusion : bool
        If False, the ``enhanced_diffusion`` scheme returns zero tendency
        but still produces the K_v profile so the dynamics step can apply
        it via an implicit backward-Euler solve (combined with KPP /
        background diffusivities).  The ``plume`` scheme ignores this
        flag (it is not a diffusion).
    emit_momentum_viscosity : bool
        If False, the ``enhanced_diffusion`` scheme suppresses the
        convective **momentum** viscosity (both the explicit du/dv
        tendency and the implicit ``A_v`` profile) while still mixing
        tracers.  The combiner sets this to False when the vertical-
        mixing scheme is KPP, because KPP already enhances interior
        momentum for the same ``N² < 0`` instability — emitting ``A_v``
        here too would double-count.  This makes the combiner fast path
        consistent with the ``vmix.scheme != "kpp"`` gate in
        ``compute_vertical_K_profiles`` (the implicit fallback path).

    Returns
    -------
    Callable : physics_fn(state, grid, z_coord) -> OceanTendencies
    """
    scheme = config.scheme

    if scheme == "none":
        return make_none_physics_fn()
    elif scheme == "enhanced_diffusion":
        return _make_enhanced_diffusion(
            config, apply_diffusion=apply_diffusion,
            emit_momentum_viscosity=emit_momentum_viscosity,
            eos_fn=eos_fn, constants_config=constants_config,
            seos_cfg=seos_cfg,
        )
    elif scheme == "plume":
        return _make_plume(config, eos_fn=eos_fn,
                           constants_config=constants_config)
    else:
        raise ValueError(f"Unknown ocean convection scheme: {scheme!r}")


def _make_enhanced_diffusion(
        config: OceanConvectionConfig,
        apply_diffusion: bool = True,
        emit_momentum_viscosity: bool = True,
        eos_fn: Callable | None = None,
        constants_config: ConstantsConfig = ConstantsConfig(),
        seos_cfg=None) -> Callable:
    cfg = config.enhanced_diffusion

    # Decision 94, at CONSTRUCTION on the static config: a card whose
    # convective trigger is NEMO's own must state how the coefficient
    # composes, and the EXPLICIT branch cannot express NEMO's replacement --
    # it adds a tendency, where zdfevd.f90:107-110 overwrites the assembled
    # coefficient.  Refuse rather than silently sum.
    from legoesm.ocean.physics.convection.enhanced_diffusion import (
        resolve_evd_composition,
    )
    if resolve_evd_composition(cfg) == "nemo_replace" and apply_diffusion:
        raise ValueError(
            'EnhancedDiffusionConfig.evd_composition="nemo_replace" '
            "transcribes NEMO's zdfevd, which OVERWRITES the vertical "
            "diffusivity (zdfevd.f90:107-110); the explicit convection "
            "branch only adds a tendency and cannot express it. Select "
            "implicit_vertical_mixing=True (the coefficient is then "
            "composed in the implicit solve) or state "
            'evd_composition="additive".')

    # Fail closed at construction: suppression (emit_momentum_viscosity=False)
    # with a nonzero convective momentum viscosity is contradictory — the
    # configured nu_conv/nu_bg would be silently dropped.  The only internal
    # caller that suppresses is the KPP composition, which is required to set
    # nu_*=0 (and make_ocean_physics enforces that before reaching here), so
    # this guard only catches direct callers passing the inconsistent combo.
    if not emit_momentum_viscosity and (cfg.nu_conv != 0.0 or cfg.nu_bg != 0.0):
        raise ValueError(
            "enhanced_diffusion convective momentum viscosity (nu_conv/nu_bg) "
            "is nonzero but emit_momentum_viscosity=False requests its "
            "suppression — the value would be silently dropped. Suppression is "
            "only valid for nu_conv=nu_bg=0 (the KPP tracer-only path). Set "
            "EnhancedDiffusionConfig(nu_conv=0.0, nu_bg=0.0)."
        )

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        # #518: use the threaded recipe EOS (None → wright, byte-identical)
        # so the convective trigger/K profiles match the dynamics EOS.
        # The adiabatic N² trigger (cfg.n2_mode == "adiabatic") also needs the
        # cell-centre hydrostatic pressure; compute it only when opted in so
        # the default in-situ path stays bit-identical.
        if cfg.n2_mode == "adiabatic":
            rho, p_cell = _compute_rho_and_pressure(
                state, z_coord, J, eos_fn=eos_fn, g=constants_config.g,
                rho0=constants_config.rho_0)
        else:
            rho = _compute_rho(state, z_coord, J, eos_fn=eos_fn,
                               g=constants_config.g,
                               rho0=constants_config.rho_0)
            p_cell = None
        # Suppress momentum mixing (no u/v) when KPP owns interior momentum
        # convection — avoids the A_v double-count flagged in the combiner
        # fast path.  Tracers (K_v) are unaffected.
        #
        # The EXPLICIT momentum operator (vertical_diffusion_variable_K)
        # builds its coefficient / jacobian / dry mask on tracer-cell
        # columns, so it is only valid when u/v are *cell-centred* (cubed-
        # sphere / A-grid: u.shape == T.shape).  On the lat-lon C-grid u/v
        # are face-staggered (n_lon+1 / n_lat+1) and the explicit operator
        # would shape-mismatch; that grid runs vertical mixing implicitly
        # and applies the convective A_v to face momentum through its own
        # face-aware backward-Euler solve (A_v is still plumbed below).
        cell_centred = state.u.data.shape == state.T.data.shape
        # Don't SILENTLY drop the configured convective momentum viscosity on
        # a staggered grid in explicit mode: there, A_v is neither applied by
        # this cell-centred explicit operator nor plumbed for a face-aware
        # implicit solve (A_v is only attached when apply_diffusion=False).
        # Fail loudly so the user picks implicit mode or disables nu_*.
        # The guard depends only on the static config + grid stagger + mode —
        # NOT on emit_momentum_viscosity — so the KPP composition (which sets
        # emit_momentum_viscosity=False) cannot bypass it.
        nu_active = (cfg.nu_conv != 0.0) or (cfg.nu_bg != 0.0)
        if apply_diffusion and not cell_centred and nu_active:
            raise ValueError(
                "enhanced_diffusion convective momentum viscosity "
                "(nu_conv/nu_bg) is unsupported on staggered (C-grid) "
                "velocities in EXPLICIT vertical-mixing mode: the explicit "
                "operator is cell-centred only, and with apply_diffusion=True "
                "there is no backward-Euler A_v solve to apply it to the u/v "
                "faces. Set implicit_vertical_mixing=True (routes A_v through "
                "the face-aware implicit solve) or "
                "EnhancedDiffusionConfig(nu_conv=0.0, nu_bg=0.0) to mix "
                "tracers only."
            )
        pass_velocity = emit_momentum_viscosity and cell_centred
        u_in = state.u.data if pass_velocity else None
        v_in = state.v.data if pass_velocity else None
        out = enhanced_diffusion_convection(
            state.T.data, state.S.data, rho, z_coord, J, cfg,
            apply_diffusion=apply_diffusion,
            u=u_in, v=v_in,
            p_cell=p_cell, eos_fn=eos_fn,
            eta=state.eta.data, H_bathy=state.H_bathy.data,
            # The CARD's NEMO &nameos coefficients for the nemo_bn2 trigger
            # (decision 94); None keeps NemoSEOSConfig()'s defaults, which is
            # what the card's own density EOS resolves to when it states none.
            seos_cfg=seos_cfg,
            # Recipe-pinned constants (the N^2 trigger's g / reference
            # density); defaults reproduce legoesm.constants exactly.
            g=constants_config.g, rho_ref=constants_config.rho_0,
        )
        du = out.du_dt if out.du_dt is not None else None
        dv = out.dv_dt if out.dv_dt is not None else None
        t = wrap_ocean_tendencies(du, dv, out.dT_dt, out.dS_dt, state)
        # When implicit, pass convection K_v / A_v through for downstream
        # use by the tridiagonal solve (avoids re-running EOS/N² in
        # compute_vertical_K_profiles).  A_v is withheld when momentum
        # viscosity is suppressed (KPP composition) so the combiner does
        # not sum it on top of KPP's A_v.
        if not apply_diffusion:
            if out.K_v is not None:
                t = t._replace(K_v=out.K_v)
            if emit_momentum_viscosity and out.A_v is not None:
                t = t._replace(A_v=out.A_v)
        return t
    return physics_fn


def _make_plume(config: OceanConvectionConfig, eos_fn: Callable | None = None,
                constants_config: ConstantsConfig = ConstantsConfig()) -> Callable:
    cfg = config.plume

    def physics_fn(state: OceanState, grid: CubedSphereGrid,
                   z_coord: OceanZStarCoordinate,
                   surface_forcing=None) -> OceanTendencies:
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
        # Ambient rho and plume-parcel rho MUST share one EOS (#518): pass
        # the SAME eos_fn to both so the buoyancy comparison is consistent.
        # eos_fn=None → wright in both, byte-identical to the legacy path.
        rho, p_hydro = _compute_rho_and_pressure(
            state, z_coord, J, eos_fn=eos_fn, g=constants_config.g,
                rho0=constants_config.rho_0)
        out = plume_convection(
            state.T.data, state.S.data, rho, p_hydro, z_coord, J, cfg,
            eos_fn=eos_fn,
        )
        z3 = jnp.zeros_like(state.u.data)
        return wrap_ocean_tendencies(z3, z3, out.dT_dt, out.dS_dt, state)
    return physics_fn



