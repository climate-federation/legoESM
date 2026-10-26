"""fp64 legoESM card for NEMO/SI3 ``ICE_ADV2D_RHG_OMIP_L3``.

This is the one-category, landfast-off rung-3.3 composition preregistered in
``docs/ocean/fidelity/testcases/nemo_testcases_l3dyn_phase2_rung33_preregister.md``.
It extends the existing ICE_ADV2D Prather card with the selectable C-grid aEVP
arm; it does not introduce another sea-ice model or another transport kernel.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple, cast

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
    ice_adv2d_card_contract_sha256,
)
from legoesm.ice.transport import (
    SI3PratherMoments,
    advect_si3_prather_2d,
    si3_prather_pack_intensives,
    si3_prather_unpack_intensives,
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
_RHG_RN_ISHLAT = 2.0  # output.namelist.ice:225; repair at icedyn_rhg_evp.F90:215-224
_RHG_LF_DEPFRA = 0.125  # ORCA1 namelist_ice_ref:59; inactive on this card
_RHG_LF_BFR_N_M3 = 15.0  # ORCA1 namelist_ice_ref:61; inactive on this card
_RHG_LF_RELAX_S_INV = 1.0e-5  # ORCA1 namelist_ice_ref:62; inactive on this card
_RHG_LF_TENSILE = 0.05  # ORCA1 namelist_ice_ref:63; inactive on this card
_RHG_STRESS_DIVERGENCE_WEIGHT = 0.5  # icedyn_rhg_evp.F90:497,505
_RHG_AIR_STRESS_U = 1.3  # ICE_ADV2D/MY_SRC/usrdef_sbc.F90:83-96
_RHG_AIR_STRESS_V = 0.0  # ICE_ADV2D/MY_SRC/usrdef_sbc.F90:83-96
_RHG_REQUIRED_SHAPE = (103, 103)  # ocean.output:64-65, two-cell halo
_RHG_REQUIRED_JPL = 1
_RHG_REQUIRED_NLAY_I = 3
_RHG_REQUIRED_NLAY_S = 3
ICE_ADV2D_RHG_RESTART_FORMAT = "legoesm-ice-adv2d-rhg-state-v2"


class ICEAdv2DRHGState(NamedTuple):
    """Rung-3.3 intensive tracers, moments, and rheology prognostics."""

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
    del base
    return state.contents[..., ICE_ADV2D_TRACERS.index(name)]


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
    entry_frame: Mapping[str, np.ndarray],
) -> ICEAdv2DRHGCard:
    """Construct the source-pinned, fp64, CPU-compatible rung-3.3 card."""

    base = build_ice_adv2d_card(ocean_surface_temperature_c)
    oracle_policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(oracle_policy)
    if get_policy() != oracle_policy:
        raise RuntimeError("ICE_ADV2D_RHG card requires fp64/scalar-libm policy")
    shape = base.initial_state.u_ice.shape
    zero = jnp.zeros(shape, dtype=jnp.float64)
    one = jnp.ones(shape, dtype=jnp.float64)

    def entry_field(name: str) -> jnp.ndarray:
        value = np.asarray(entry_frame[name], dtype=np.float64)
        if value.ndim == 3:
            value = value[..., 0]
        return jnp.asarray(value, dtype=jnp.float64)

    packed: list[jnp.ndarray] = []
    for name in ICE_ADV2D_TRACERS:
        if name.startswith(("e_s_l", "e_i_l", "szv_i_l")):
            family, level_text = name.rsplit("_l", 1)
            value = np.asarray(entry_frame[family], dtype=np.float64)[
                ..., int(level_text) - 1, 0
            ]
            packed.append(jnp.asarray(value, dtype=jnp.float64))
        else:
            packed.append(entry_field(name))
    entry_intensives = jnp.stack(packed, axis=-1)
    bulk_salt = entry_field("sv_i")
    t_surface = entry_field("t_su")
    dynamics = SI3CGridAEVPState(
        *(entry_field(name) for name in (
            "u_ice", "v_ice", "stress1_i", "stress2_i", "stress12_i"
        ))
    )
    forcing = SI3CGridAEVPForcing(
        concentration_t=zero,
        ice_volume_t=zero,
        snow_volume_t=zero,
        pond_volume_t=zero,
        lid_volume_t=zero,
        air_stress_u_t=jnp.full(shape, _RHG_AIR_STRESS_U, dtype=jnp.float64),
        air_stress_v_t=jnp.full(shape, _RHG_AIR_STRESS_V, dtype=jnp.float64),
        # icestp.F90:332 initializes the T-point field; icedyn_rhg_evp.F90:310-313
        # performs the directional neighbor average used by the solver.
        drag_io_t=jnp.full(shape, _RHG_DRAG_IO, dtype=jnp.float64),
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
        depth_t=zero,
        depth_u=zero,
        depth_v=zero,
        iceberg_tmask=zero,
        iceberg_umask=zero,
        iceberg_vmask=zero,
    )
    initial = ICEAdv2DRHGState(
        contents=entry_intensives,
        bulk_salt_diagnostic=bulk_salt,
        moments=base.initial_state.moments,
        dynamics=dynamics,
        t_surface=t_surface,
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
        rn_ishlat=_RHG_RN_ISHLAT,
        halo_width=base.halo_width,
        category_count=base.jpl,
        landfast=False,
        landfast_depth_fraction=_RHG_LF_DEPFRA,
        landfast_basal_friction_n_m3=_RHG_LF_BFR_N_M3,
        landfast_relaxation_s_inv=_RHG_LF_RELAX_S_INV,
        landfast_tensile_fraction=_RHG_LF_TENSILE,
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
        raise ValueError(f"ICE_ADV2D_RHG selector composition {actual!r} != {expected!r}")
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
    stress_divergence_outer_weight: float = _RHG_STRESS_DIVERGENCE_WEIGHT,
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
        stress_divergence_outer_weight=stress_divergence_outer_weight,
    )
    area = card.base.dx_m * card.base.dy_m
    cell_area = jnp.full(dynamics.u_ice_u.shape, area, dtype=jnp.float64)
    wet = jnp.ones(dynamics.u_ice_u.shape, dtype=bool)
    entry_contents = si3_prather_pack_intensives(state.contents, cell_area)
    contents, moments, ignored = advect_si3_prather_2d(
        entry_contents,
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
    intensives = si3_prather_unpack_intensives(contents, cell_area, wet)
    intensives = apply_ice_adv2d_source_corrections(
        card.base,
        state.contents,
        intensives,
        entry_intensive_contents=state.contents,
        contents_are_intensive=True,
    )
    base_state = card.base.initial_state._replace(
        contents=entry_contents,
        bulk_salt_diagnostic=state.bulk_salt_diagnostic,
        moments=state.moments,
        u_ice=state.dynamics.u_ice_u,
        v_ice=state.dynamics.v_ice_v,
        t_surface=state.t_surface,
    )
    corrected = apply_ice_adv2d_zapsmall(
        card.base,
        base_state,
        intensives,
        moments,
        contents_are_intensive=True,
    )
    return ICEAdv2DRHGState(
        contents=corrected.contents,
        bulk_salt_diagnostic=corrected.bulk_salt_diagnostic,
        moments=corrected.moments,
        dynamics=dynamics,
        t_surface=corrected.t_surface,
    )


def ice_adv2d_rhg_card_contract_sha256(card: ICEAdv2DRHGCard) -> str:
    """Stable selector/parameter identity for restart compatibility."""

    cfg = card.dynamics_config
    payload = {
        "base": ice_adv2d_card_contract_sha256(card.base),
        "case": card.case,
        "transport_scheme": card.transport_scheme,
        "dynamics_scheme": card.dynamics_scheme,
        "rheology_staggering": card.rheology_staggering,
        "dynamics": {name: getattr(cfg, name) for name in cfg._fields},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def save_ice_adv2d_rhg_restart(
    path: Path,
    card: ICEAdv2DRHGCard,
    state: ICEAdv2DRHGState,
    *,
    completed_steps: int,
) -> None:
    """Write all rung-3.3 prognostics, including stresses and moments."""

    validate_ice_adv2d_rhg_card(card, state)
    if not 0 <= completed_steps <= card.base.n_steps:
        raise ValueError("ICE_ADV2D_RHG completed_steps out of range")
    payload: dict[str, np.ndarray] = {
        "format": np.asarray(ICE_ADV2D_RHG_RESTART_FORMAT),
        "completed_steps": np.asarray(completed_steps, dtype=np.int64),
        "card_contract_sha256": np.asarray(ice_adv2d_rhg_card_contract_sha256(card)),
        "contents": np.asarray(state.contents),
        "bulk_salt_diagnostic": np.asarray(state.bulk_salt_diagnostic),
        "t_surface": np.asarray(state.t_surface),
        "u_ice": np.asarray(state.dynamics.u_ice_u),
        "v_ice": np.asarray(state.dynamics.v_ice_v),
        "stress1_i": np.asarray(state.dynamics.stress1_t),
        "stress2_i": np.asarray(state.dynamics.stress2_t),
        "stress12_i": np.asarray(state.dynamics.stress12_f),
    }
    payload.update(
        {f"moment_{index}": np.asarray(value) for index, value in enumerate(state.moments)}
    )
    np.savez(path, **payload)  # type: ignore[arg-type]


def load_ice_adv2d_rhg_restart(path: Path, card: ICEAdv2DRHGCard) -> tuple[ICEAdv2DRHGState, int]:
    """Load a fail-closed rung-3.3 restart without defaulting any carry."""

    validate_ice_adv2d_rhg_card(card)
    state_names = {
        "contents",
        "bulk_salt_diagnostic",
        "t_surface",
        "u_ice",
        "v_ice",
        "stress1_i",
        "stress2_i",
        "stress12_i",
    }
    moment_names = {f"moment_{index}" for index in range(5)}
    metadata = {"format", "completed_steps", "card_contract_sha256"}
    expected = state_names | moment_names | metadata
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            missing = sorted(expected - set(archive.files))
            extra = sorted(set(archive.files) - expected)
            raise ValueError(f"ICE_ADV2D_RHG restart keys missing={missing}, extra={extra}")
        if archive["format"].item() != ICE_ADV2D_RHG_RESTART_FORMAT:
            raise ValueError("ICE_ADV2D_RHG restart format mismatch")
        if archive["card_contract_sha256"].item() != ice_adv2d_rhg_card_contract_sha256(card):
            raise ValueError("ICE_ADV2D_RHG restart card contract mismatch")
        if archive["completed_steps"].dtype.kind not in "iu":
            raise ValueError("ICE_ADV2D_RHG restart clock is not integer")
        completed_steps = int(archive["completed_steps"].item())
        arrays = {name: archive[name].copy() for name in state_names | moment_names}
    if not 0 <= completed_steps <= card.base.n_steps:
        raise ValueError("ICE_ADV2D_RHG restart clock out of range")
    template = card.initial_state
    templates = {
        "contents": template.contents,
        "bulk_salt_diagnostic": template.bulk_salt_diagnostic,
        "t_surface": template.t_surface,
        "u_ice": template.dynamics.u_ice_u,
        "v_ice": template.dynamics.v_ice_v,
        "stress1_i": template.dynamics.stress1_t,
        "stress2_i": template.dynamics.stress2_t,
        "stress12_i": template.dynamics.stress12_f,
    }
    for name, reference in templates.items():
        value = arrays[name]
        reference_array = np.asarray(reference)
        if value.shape != reference_array.shape or value.dtype != reference_array.dtype:
            raise ValueError(f"ICE_ADV2D_RHG restart {name} shape/dtype mismatch")
    for index, reference in enumerate(template.moments):
        value = arrays[f"moment_{index}"]
        reference_array = np.asarray(reference)
        if value.shape != reference_array.shape or value.dtype != reference_array.dtype:
            raise ValueError(f"ICE_ADV2D_RHG restart moment_{index} shape/dtype mismatch")
    dynamics = SI3CGridAEVPState(
        u_ice_u=jnp.asarray(arrays["u_ice"]),
        v_ice_v=jnp.asarray(arrays["v_ice"]),
        stress1_t=jnp.asarray(arrays["stress1_i"]),
        stress2_t=jnp.asarray(arrays["stress2_i"]),
        stress12_f=jnp.asarray(arrays["stress12_i"]),
    )
    state = ICEAdv2DRHGState(
        contents=jnp.asarray(arrays["contents"]),
        bulk_salt_diagnostic=jnp.asarray(arrays["bulk_salt_diagnostic"]),
        moments=cast(
            SI3PratherMoments,
            tuple(jnp.asarray(arrays[f"moment_{index}"]) for index in range(5)),
        ),
        dynamics=dynamics,
        t_surface=jnp.asarray(arrays["t_surface"]),
    )
    validate_ice_adv2d_rhg_card(card, state)
    return state, completed_steps


__all__ = (
    "ICE_ADV2D_RHG_RESTART_FORMAT",
    "ICEAdv2DRHGCard",
    "ICEAdv2DRHGState",
    "build_ice_adv2d_rhg_card",
    "ice_adv2d_rhg_card_contract_sha256",
    "load_ice_adv2d_rhg_restart",
    "save_ice_adv2d_rhg_restart",
    "step_ice_adv2d_rhg_card",
    "validate_ice_adv2d_rhg_card",
)
