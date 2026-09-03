"""fp64 legoESM card for NEMO/SI3 ``ICE_ADV2D_RHG_OMIP_L3``.

This is the one-category, landfast-off rung-3.3 composition preregistered in
``docs/ocean/fidelity/testcases/nemo_testcases_l3dyn_phase2_rung33_preregister.md``.
It extends the existing ICE_ADV2D Prather card with the selectable C-grid aEVP
arm; it does not introduce another sea-ice model or another transport kernel.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ice.dynamics import (
    SI3CGridAEVPConfig,
    SI3CGridAEVPForcing,
    SI3CGridAEVPState,
    SI3CGridMetrics,
    si3_cgrid_aevp_solver,
)
from legoesm.ice.fidelity.nemo_adv2d_testcase_recipe import (
    ICE_ADV2D_TRACERS,
    ICEAdv2DCard,
    apply_ice_adv2d_source_corrections,
    apply_ice_adv2d_zapsmall,
    build_ice_adv2d_card,
)
from legoesm.ice.transport import (
    SI3PratherMoments,
    advect_si3_prather_2d,
)

from legoesm import constants

# Resolved NEMO/ORCA1 values.  Provenance is output.namelist.ice:219-297,
# output.namelist.dyn:25-33,67, and ORCA1 namelist_ice_ref:57,108-116,136 plus
# namelist_ice_cfg:52-55, as enumerated in the preregistration.
_RHG_SCHEME = "si3_aevp"
_RHG_STAGGERING = "si3_c_grid"
_RHG_N_SUBCYCLES = 100
_RHG_ECCENTRICITY = 2.0
_RHG_CREEP_LIMIT_S_INV = 2.0e-9
_RHG_STRENGTH_PARAMETER_PA = 2.0e4
_RHG_STRENGTH_DECAY = 20.0
_RHG_DRAG_IO = 5.0e-3
_RHG_AIR_STRESS_U = 1.3  # ICE_ADV2D/MY_SRC/usrdef_sbc.F90:83-96
_RHG_AIR_STRESS_V = 0.0  # ICE_ADV2D/MY_SRC/usrdef_sbc.F90:83-96
_RHG_REQUIRED_SHAPE = (103, 103)  # ocean.output:64-65, two-cell halo
_RHG_REQUIRED_JPL = 1
_RHG_REQUIRED_NLAY_I = 3
_RHG_REQUIRED_NLAY_S = 3


class ICEAdv2DRHGState(NamedTuple):
    """Rung-3.3 transported and rheology prognostic state."""

    contents: jnp.ndarray
    bulk_salt_diagnostic: jnp.ndarray
    moments: SI3PratherMoments
    dynamics: SI3CGridAEVPState
    t_surface: jnp.ndarray


class ICEAdv2DRHGCard(NamedTuple):
    """Only the resolved ICE_ADV2D_RHG selector composition."""

    case: str
    transport_scheme: str
    dynamics_scheme: str
    rheology_staggering: str
    base: ICEAdv2DCard
    metrics: SI3CGridMetrics
    forcing_template: SI3CGridAEVPForcing
    dynamics_config: SI3CGridAEVPConfig
    initial_state: ICEAdv2DRHGState


def _full_metric(value: np.ndarray, halo_width: int) -> jnp.ndarray:
    """Pad a physical oracle metric with the rung's periodic two-cell halo."""

    return jnp.asarray(np.pad(value, halo_width, mode="wrap"), dtype=jnp.float64)


def _build_metrics(base: ICEAdv2DCard) -> SI3CGridMetrics:
    """Build same-index SI3 metrics from the already geometry-gated base card."""

    size = base.wet_global_xy.shape
    dx = np.full(size, base.dx_m, dtype=np.float64)
    dy = np.full(size, base.dy_m, dtype=np.float64)
    area = dx * dy
    full_dx = _full_metric(dx, base.halo_width)
    full_dy = _full_metric(dy, base.halo_width)
    full_area = _full_metric(area, base.halo_width)
    return SI3CGridMetrics(
        e1t=full_dx,
        e2t=full_dy,
        e1u=full_dx,
        e2u=full_dy,
        e1v=full_dx,
        e2v=full_dy,
        e1f=full_dx,
        e2f=full_dy,
        area_t=full_area,
        area_u=full_area,
        area_v=full_area,
        area_f=full_area,
    )


def _intensive(state: ICEAdv2DRHGState, base: ICEAdv2DCard, name: str) -> jnp.ndarray:
    area = base.dx_m * base.dy_m
    return state.contents[..., ICE_ADV2D_TRACERS.index(name)] / area


def _forcing_for_state(
    template: SI3CGridAEVPForcing,
    state: ICEAdv2DRHGState,
    base: ICEAdv2DCard,
) -> SI3CGridAEVPForcing:
    return template._replace(
        concentration_t=_intensive(state, base, "a_i"),
        ice_volume_t=_intensive(state, base, "v_i"),
        snow_volume_t=_intensive(state, base, "v_s"),
        pond_volume_t=_intensive(state, base, "v_ip"),
        lid_volume_t=_intensive(state, base, "v_il"),
    )


def build_ice_adv2d_rhg_card(
    ocean_surface_temperature_c: np.ndarray,
) -> ICEAdv2DRHGCard:
    """Construct the source-pinned, fp64, CPU-compatible rung-3.3 card."""

    set_policy(PrecisionPolicy.fp64())
    if get_policy() != PrecisionPolicy.fp64():
        raise RuntimeError("ICE_ADV2D_RHG card requires PrecisionPolicy.fp64()")
    base = build_ice_adv2d_card(ocean_surface_temperature_c)
    shape = base.initial_state.u_ice.shape
    zero = jnp.zeros(shape, dtype=jnp.float64)
    one = jnp.ones(shape, dtype=jnp.float64)
    dynamics = SI3CGridAEVPState(zero, zero, zero, zero, zero)
    forcing = SI3CGridAEVPForcing(
        concentration_t=zero,
        ice_volume_t=zero,
        snow_volume_t=zero,
        pond_volume_t=zero,
        lid_volume_t=zero,
        air_stress_u_t=jnp.full(shape, _RHG_AIR_STRESS_U, dtype=jnp.float64),
        air_stress_v_t=jnp.full(shape, _RHG_AIR_STRESS_V, dtype=jnp.float64),
        ocean_u_u=zero,
        ocean_v_v=zero,
        ssh_t=zero,
        coriolis_t=zero,
        tmask_t=one,
        umask_u=one,
        vmask_v=one,
        # icedyn.F90:113-118 reads this file field independently of landfast;
        # the gate verifies that the pinned NOT-USED field resolves to zero.
        fast_tmask=zero,
    )
    initial = ICEAdv2DRHGState(
        contents=base.initial_state.contents,
        bulk_salt_diagnostic=base.initial_state.bulk_salt_diagnostic,
        moments=base.initial_state.moments,
        dynamics=dynamics,
        t_surface=base.initial_state.t_surface,
    )
    config = SI3CGridAEVPConfig(
        scheme=_RHG_SCHEME,
        staggering=_RHG_STAGGERING,
        dt_s=base.dt_s,
        n_subcycles=_RHG_N_SUBCYCLES,
        eccentricity=_RHG_ECCENTRICITY,
        creep_limit_s_inv=_RHG_CREEP_LIMIT_S_INV,
        strength_parameter_pa=_RHG_STRENGTH_PARAMETER_PA,
        strength_decay=_RHG_STRENGTH_DECAY,
        rho_snow=constants.rho_snow,
        rho_ice=constants.rho_ice,
        rho_water=constants.rho_water,
        rho_ocean=constants.rho_ocean_nemo,
        gravity=constants.g_nemo,
        drag_io=_RHG_DRAG_IO,
        halo_width=base.halo_width,
        category_count=base.jpl,
        landfast=False,
        convergence_check=0,
    )
    card = ICEAdv2DRHGCard(
        case="ICE_ADV2D_RHG_OMIP_L3",
        transport_scheme="si3_prather_xy_alternating",
        dynamics_scheme=_RHG_SCHEME,
        rheology_staggering=_RHG_STAGGERING,
        base=base,
        metrics=_build_metrics(base),
        forcing_template=forcing,
        dynamics_config=config,
        initial_state=initial,
    )
    validate_ice_adv2d_rhg_card(card)
    return card


def validate_ice_adv2d_rhg_card(
    card: ICEAdv2DRHGCard,
    state: ICEAdv2DRHGState | None = None,
) -> None:
    """Fail closed on unsupported selectors, shapes, masks, or dtypes."""

    expected = (
        "ICE_ADV2D_RHG_OMIP_L3",
        "si3_prather_xy_alternating",
        _RHG_SCHEME,
        _RHG_STAGGERING,
        _RHG_REQUIRED_JPL,
        _RHG_REQUIRED_NLAY_I,
        _RHG_REQUIRED_NLAY_S,
        False,
        _RHG_N_SUBCYCLES,
    )
    actual = (
        card.case,
        card.transport_scheme,
        card.dynamics_scheme,
        card.rheology_staggering,
        card.base.jpl,
        card.base.nlay_i,
        card.base.nlay_s,
        card.base.landfast,
        card.dynamics_config.n_subcycles,
    )
    if actual != expected:
        raise ValueError(
            f"ICE_ADV2D_RHG selector composition {actual!r} != {expected!r}"
        )
    state = card.initial_state if state is None else state
    array_leaves = (
        state.contents,
        state.bulk_salt_diagnostic,
        *state.moments,
        *state.dynamics,
        state.t_surface,
        *card.metrics,
        *card.forcing_template,
    )
    if any(value.shape[:2] != _RHG_REQUIRED_SHAPE for value in array_leaves):
        raise ValueError("ICE_ADV2D_RHG requires the pinned 103x103 same-index geometry")
    if any(value.dtype != jnp.float64 for value in array_leaves):
        raise ValueError("ICE_ADV2D_RHG requires fp64 state, forcing, and geometry")
    if bool(np.any(np.asarray(card.forcing_template.coriolis_t))):
        raise ValueError("ICE_ADV2D_RHG card requires zero Coriolis")
    if bool(np.any(np.asarray(card.forcing_template.ocean_u_u))) or bool(
        np.any(np.asarray(card.forcing_template.ocean_v_v))
    ):
        raise ValueError("ICE_ADV2D_RHG card requires a resting ocean")
    if bool(np.any(np.asarray(card.forcing_template.ssh_t))):
        raise ValueError("ICE_ADV2D_RHG card requires zero SSH")
    if bool(np.any(np.asarray(card.forcing_template.fast_tmask))):
        raise ValueError("ICE_ADV2D_RHG pinned fast-mask input is not zero")


def step_ice_adv2d_rhg_card(
    card: ICEAdv2DRHGCard,
    state: ICEAdv2DRHGState | None = None,
    *,
    completed_steps: int,
    differentiable: bool = False,
) -> ICEAdv2DRHGState:
    """Advance dynamics, Prather transport, Hbig/Hsnow, then zapsmall."""

    state = card.initial_state if state is None else state
    forcing = _forcing_for_state(card.forcing_template, state, card.base)
    dynamics = si3_cgrid_aevp_solver(
        state.dynamics,
        forcing,
        card.metrics,
        card.dynamics_config,
        differentiable=differentiable,
    )
    area = card.base.dx_m * card.base.dy_m
    cell_area = jnp.full(dynamics.u_ice_u.shape, area, dtype=jnp.float64)
    wet = jnp.ones(dynamics.u_ice_u.shape, dtype=bool)
    contents, moments, ignored = advect_si3_prather_2d(
        state.contents,
        state.moments,
        dynamics.u_ice_u,
        dynamics.v_ice_v,
        cell_area,
        wet,
        card.base.dt_s,
        dx=card.base.dx_m,
        dy=card.base.dy_m,
        ice_step_index=completed_steps + 1,
        nn_fsbc=card.base.nn_fsbc,
        halo_width=card.base.halo_width,
        ice_volume_index=ICE_ADV2D_TRACERS.index("v_i"),
        concentration_index=ICE_ADV2D_TRACERS.index("a_i"),
        subcycles=card.base.subcycles,
    )
    del ignored
    contents = apply_ice_adv2d_source_corrections(
        card.base,
        state.contents,
        contents,
    )
    base_state = card.base.initial_state._replace(
        contents=state.contents,
        bulk_salt_diagnostic=state.bulk_salt_diagnostic,
        moments=state.moments,
        u_ice=state.dynamics.u_ice_u,
        v_ice=state.dynamics.v_ice_v,
        t_surface=state.t_surface,
    )
    corrected = apply_ice_adv2d_zapsmall(card.base, base_state, contents, moments)
    return ICEAdv2DRHGState(
        contents=corrected.contents,
        bulk_salt_diagnostic=corrected.bulk_salt_diagnostic,
        moments=corrected.moments,
        dynamics=dynamics,
        t_surface=corrected.t_surface,
    )
