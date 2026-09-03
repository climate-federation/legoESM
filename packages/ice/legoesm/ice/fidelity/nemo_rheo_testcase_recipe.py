"""fp64 legoESM card boundary for NEMO/SI3 ``ICE_RHEO_OMIP_L3``.

The card composes the existing C-grid aEVP and Prather implementations with
the selectable SI3 jpl=1 ridge/raft arm in :mod:`legoesm.ice.ridging`.  This
file is card/state glue, not a second ice model.  Construction consumes the
pinned oracle's entry frame and mesh arrays so no analytic geometry or initial
condition can silently replace the shipped case.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple, cast

import jax.numpy as jnp
import numpy as np
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ice.dynamics import (
    SI3CGridAEVPConfig,
    SI3CGridAEVPForcing,
    SI3CGridAEVPState,
    SI3CGridMetrics,
)
from legoesm.ice.ridging import SI3JPL1RidgingConfig
from legoesm.ice.transport import SI3PratherMoments, zero_si3_prather_moments

from legoesm import constants

ICE_RHEO_TRACERS = (
    "v_i",
    "v_s",
    "a_i",
    "oa_i",
    *(f"e_s_l{level:02d}" for level in range(1, 6)),
    *(f"e_i_l{level:02d}" for level in range(1, 11)),
    *(f"szv_i_l{level:02d}" for level in range(1, 11)),
    "a_ip",
    "v_ip",
    "v_il",
)

# Shipped-case values: ICE_RHEO namelist_cfg:19-20,29-37,76 and the frame
# header.  The rheology constants are the cited ORCA1 overlay in
# nemo_testcases_l3dyn_phase2_rung34_preregister.md.
_ICE_RHEO_CASE = "ICE_RHEO_OMIP_L3"
_ICE_RHEO_DYNAMICS = "si3_aevp"
_ICE_RHEO_STAGGERING = "si3_c_grid"
_ICE_RHEO_TRANSPORT = "si3_prather_xy_alternating"
_ICE_RHEO_RIDGING = "si3_orca1_jpl1"
_ICE_RHEO_GRID_SIZE = 1000
_ICE_RHEO_HALO_WIDTH = 2
_ICE_RHEO_ALLOCATED_SIZE = _ICE_RHEO_GRID_SIZE + 2 * _ICE_RHEO_HALO_WIDTH
_ICE_RHEO_SPACING_M = 2000.0
_ICE_RHEO_CELL_AREA_M2 = _ICE_RHEO_SPACING_M * _ICE_RHEO_SPACING_M
_ICE_RHEO_DT_S = 30.0
_ICE_RHEO_N_STEPS = 720
_ICE_RHEO_JPL = 1
_ICE_RHEO_NLAY_I = 10
_ICE_RHEO_NLAY_S = 5
_ICE_RHEO_NN_ICESAL = 4
_ICE_RHEO_NN_FSBC = 1
_ICE_RHEO_SUBCYCLES = 1
_ICE_RHEO_AEVP_SUBCYCLES = 100
_ICE_RHEO_ECCENTRICITY = 2.0
_ICE_RHEO_CREEP_LIMIT_S_INV = 2.0e-9
_ICE_RHEO_STRENGTH_PA = 2.0e4
_ICE_RHEO_STRENGTH_DECAY = 20.0
_ICE_RHEO_DRAG_IO = 5.0e-3
_ICE_RHEO_RN_ISHLAT = 2.0

# Exact shipped usrdef_sbc.F90:111-127.  The source spells the radial power as
# ``**(1/4)`` with integer operands, so Fortran evaluates the exponent as zero;
# the card preserves that executed branch rather than substituting a quarter.
_ICE_RHEO_RELATIVE_WIND = 1.0
_ICE_RHEO_WIND_RATIO = -0.8
_ICE_RHEO_WIND_MAX_M_S = 15.0
_ICE_RHEO_DOMAIN_KM = 2000.0
_ICE_RHEO_RESOLUTION_KM = 2.0
_ICE_RHEO_KM_TO_M = 1000.0
_ICE_RHEO_AIR_DENSITY_KG_M3 = 1.22
_ICE_RHEO_AIR_DRAG = 1.4e-3
_ICE_RHEO_SPINUP_S = 21600.0
_ICE_RHEO_ONE = 1.0
_ICE_RHEO_HALF = 0.5
_ICE_RHEO_ZERO = 0.0
_ICE_RHEO_TWO = 2.0


class ICERheoState(NamedTuple):
    """Rung-3.4 dynamics, transported inventories, and open-water carry."""

    contents: jnp.ndarray
    bulk_salt_diagnostic: jnp.ndarray
    moments: SI3PratherMoments
    dynamics: SI3CGridAEVPState
    t_surface: jnp.ndarray
    open_water_area: jnp.ndarray


class ICERheoCard(NamedTuple):
    """Only the measured ICE_RHEO/ORCA1 selector composition."""

    case: str
    dynamics_scheme: str
    rheology_staggering: str
    transport_scheme: str
    ridging_scheme: str
    metrics: SI3CGridMetrics
    forcing_template: SI3CGridAEVPForcing
    dynamics_config: SI3CGridAEVPConfig
    ridging_config: SI3JPL1RidgingConfig
    initial_state: ICERheoState
    dt_s: float
    n_steps: int
    nn_fsbc: int
    subcycles: int
    halo_width: int
    jpl: int
    nlay_i: int
    nlay_s: int
    nn_icesal: int
    thermodynamics: bool
    ponds: bool
    landfast: bool


def _periodic_halo(value: np.ndarray) -> jnp.ndarray:
    tail = ((0, 0),) * (value.ndim - 2)
    padding = (
        (_ICE_RHEO_HALO_WIDTH, _ICE_RHEO_HALO_WIDTH),
        (_ICE_RHEO_HALO_WIDTH, _ICE_RHEO_HALO_WIDTH),
    ) + tail
    return jnp.pad(jnp.asarray(value, dtype=jnp.float64), padding, mode="wrap")


def _frame_field(frame: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    value = np.asarray(frame[name], dtype=np.float64)
    if value.ndim == 3:
        if value.shape[-1] != _ICE_RHEO_JPL:
            raise ValueError(f"ICE_RHEO frame {name} category shape changed")
        value = value[..., 0]
    return cast(np.ndarray, value)


def _layer_fields(
    frame: Mapping[str, np.ndarray], name: str, levels: int
) -> dict[str, np.ndarray]:
    value = np.asarray(frame[name], dtype=np.float64)
    expected = (
        _ICE_RHEO_ALLOCATED_SIZE,
        _ICE_RHEO_ALLOCATED_SIZE,
        levels,
        _ICE_RHEO_JPL,
    )
    if value.shape != expected:
        raise ValueError(f"ICE_RHEO frame {name} shape {value.shape} != {expected}")
    return {
        f"{name}_l{level + 1:02d}": value[..., level, 0]
        for level in range(levels)
    }


def _mesh_xy(mesh: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    value = np.asarray(mesh[name])
    while value.ndim > 2:
        value = value[0]
    expected = (_ICE_RHEO_GRID_SIZE, _ICE_RHEO_GRID_SIZE)
    if value.shape != expected:
        raise ValueError(f"ICE_RHEO mesh {name} shape {value.shape} != {expected}")
    return cast(np.ndarray, value.T)


def build_ice_rheo_card(
    entry_frame: Mapping[str, np.ndarray],
    mesh: Mapping[str, np.ndarray],
) -> ICERheoCard:
    """Build the pinned fp64 card from oracle arrays, never analytic stand-ins."""

    set_policy(PrecisionPolicy.fp64())
    if get_policy() != PrecisionPolicy.fp64():
        raise RuntimeError("ICE_RHEO card requires PrecisionPolicy.fp64()")
    fields = {
        name: _frame_field(entry_frame, name)
        for name in ("v_i", "v_s", "a_i", "oa_i", "a_ip", "v_ip", "v_il")
    }
    fields.update(_layer_fields(entry_frame, "e_s", _ICE_RHEO_NLAY_S))
    fields.update(_layer_fields(entry_frame, "e_i", _ICE_RHEO_NLAY_I))
    fields.update(_layer_fields(entry_frame, "szv_i", _ICE_RHEO_NLAY_I))
    contents = jnp.asarray(
        np.stack(
            [fields[name] * _ICE_RHEO_CELL_AREA_M2 for name in ICE_RHEO_TRACERS],
            axis=-1,
        ),
        dtype=jnp.float64,
    )
    tmask = _periodic_halo(_mesh_xy(mesh, "tmask").astype(np.float64))
    umask = _periodic_halo(_mesh_xy(mesh, "umask").astype(np.float64))
    vmask = _periodic_halo(_mesh_xy(mesh, "vmask").astype(np.float64))
    zero = jnp.zeros(contents.shape[:2], dtype=jnp.float64)
    metrics = SI3CGridMetrics(
        e1t=_periodic_halo(_mesh_xy(mesh, "e1t")),
        e2t=_periodic_halo(_mesh_xy(mesh, "e2t")),
        e1u=_periodic_halo(_mesh_xy(mesh, "e1u")),
        e2u=_periodic_halo(_mesh_xy(mesh, "e2u")),
        e1v=_periodic_halo(_mesh_xy(mesh, "e1v")),
        e2v=_periodic_halo(_mesh_xy(mesh, "e2v")),
        e1f=_periodic_halo(_mesh_xy(mesh, "e1f")),
        e2f=_periodic_halo(_mesh_xy(mesh, "e2f")),
        area_t=_periodic_halo(_mesh_xy(mesh, "e1t") * _mesh_xy(mesh, "e2t")),
        area_u=_periodic_halo(_mesh_xy(mesh, "e1u") * _mesh_xy(mesh, "e2u")),
        area_v=_periodic_halo(_mesh_xy(mesh, "e1v") * _mesh_xy(mesh, "e2v")),
        area_f=_periodic_halo(_mesh_xy(mesh, "e1f") * _mesh_xy(mesh, "e2f")),
    )
    forcing = SI3CGridAEVPForcing(
        concentration_t=jnp.asarray(fields["a_i"]),
        ice_volume_t=jnp.asarray(fields["v_i"]),
        snow_volume_t=jnp.asarray(fields["v_s"]),
        pond_volume_t=jnp.asarray(fields["v_ip"]),
        lid_volume_t=jnp.asarray(fields["v_il"]),
        air_stress_u_t=zero,
        air_stress_v_t=zero,
        drag_io_t=jnp.full_like(zero, _ICE_RHEO_DRAG_IO),
        ocean_u_u=zero,
        ocean_v_v=zero,
        ssh_t=zero,
        coriolis_t=zero,
        tmask_t=tmask,
        umask_u=umask,
        vmask_v=vmask,
        # icedyn.F90:115-117 reads this independently; the oracle contract
        # verifies its file-provided field is zero on this landfast-off card.
        fast_tmask=zero,
    )
    config = SI3CGridAEVPConfig(
        scheme=_ICE_RHEO_DYNAMICS,
        staggering=_ICE_RHEO_STAGGERING,
        dt_s=_ICE_RHEO_DT_S,
        n_subcycles=_ICE_RHEO_AEVP_SUBCYCLES,
        eccentricity=_ICE_RHEO_ECCENTRICITY,
        creep_limit_s_inv=_ICE_RHEO_CREEP_LIMIT_S_INV,
        strength_parameter_pa=_ICE_RHEO_STRENGTH_PA,
        strength_decay=_ICE_RHEO_STRENGTH_DECAY,
        rho_snow=constants.rho_snow,
        rho_ice=constants.rho_ice,
        rho_water=constants.rho_water,
        rho_ocean=constants.rho_ocean_nemo,
        gravity=constants.g_nemo,
        rn_ishlat=_ICE_RHEO_RN_ISHLAT,
        halo_width=_ICE_RHEO_HALO_WIDTH,
        category_count=_ICE_RHEO_JPL,
        landfast=False,
        convergence_check=0,
    )
    dynamics = SI3CGridAEVPState(
        u_ice_u=jnp.asarray(entry_frame["u_ice"], dtype=jnp.float64),
        v_ice_v=jnp.asarray(entry_frame["v_ice"], dtype=jnp.float64),
        stress1_t=jnp.asarray(entry_frame["stress1_i"], dtype=jnp.float64),
        stress2_t=jnp.asarray(entry_frame["stress2_i"], dtype=jnp.float64),
        stress12_f=jnp.asarray(entry_frame["stress12_i"], dtype=jnp.float64),
    )
    initial = ICERheoState(
        contents=contents,
        bulk_salt_diagnostic=jnp.asarray(entry_frame["sv_i"][..., 0]),
        moments=zero_si3_prather_moments(contents),
        dynamics=dynamics,
        t_surface=jnp.asarray(entry_frame["t_su"][..., 0]),
        open_water_area=_ICE_RHEO_ONE - jnp.asarray(fields["a_i"]),
    )
    card = ICERheoCard(
        case=_ICE_RHEO_CASE,
        dynamics_scheme=_ICE_RHEO_DYNAMICS,
        rheology_staggering=_ICE_RHEO_STAGGERING,
        transport_scheme=_ICE_RHEO_TRANSPORT,
        ridging_scheme=_ICE_RHEO_RIDGING,
        metrics=metrics,
        forcing_template=forcing,
        dynamics_config=config,
        ridging_config=SI3JPL1RidgingConfig(),
        initial_state=initial,
        dt_s=_ICE_RHEO_DT_S,
        n_steps=_ICE_RHEO_N_STEPS,
        nn_fsbc=_ICE_RHEO_NN_FSBC,
        subcycles=_ICE_RHEO_SUBCYCLES,
        halo_width=_ICE_RHEO_HALO_WIDTH,
        jpl=_ICE_RHEO_JPL,
        nlay_i=_ICE_RHEO_NLAY_I,
        nlay_s=_ICE_RHEO_NLAY_S,
        nn_icesal=_ICE_RHEO_NN_ICESAL,
        thermodynamics=False,
        ponds=True,
        landfast=False,
    )
    validate_ice_rheo_card(card)
    return card


def validate_ice_rheo_card(
    card: ICERheoCard, state: ICERheoState | None = None
) -> None:
    """Fail closed on every selector or layout outside the measured rung."""

    expected = (
        _ICE_RHEO_CASE,
        _ICE_RHEO_DYNAMICS,
        _ICE_RHEO_STAGGERING,
        _ICE_RHEO_TRANSPORT,
        _ICE_RHEO_RIDGING,
        _ICE_RHEO_DT_S,
        _ICE_RHEO_N_STEPS,
        _ICE_RHEO_JPL,
        _ICE_RHEO_NLAY_I,
        _ICE_RHEO_NLAY_S,
        _ICE_RHEO_NN_ICESAL,
        False,
        True,
        False,
    )
    actual = (
        card.case,
        card.dynamics_scheme,
        card.rheology_staggering,
        card.transport_scheme,
        card.ridging_scheme,
        card.dt_s,
        card.n_steps,
        card.jpl,
        card.nlay_i,
        card.nlay_s,
        card.nn_icesal,
        card.thermodynamics,
        card.ponds,
        card.landfast,
    )
    if actual != expected:
        raise ValueError(f"ICE_RHEO selector composition {actual!r} != {expected!r}")
    state = card.initial_state if state is None else state
    shape = (_ICE_RHEO_ALLOCATED_SIZE, _ICE_RHEO_ALLOCATED_SIZE)
    expected_contents = shape + (len(ICE_RHEO_TRACERS),)
    if state.contents.shape != expected_contents:
        raise ValueError(f"ICE_RHEO contents shape {state.contents.shape} != {expected_contents}")
    leaves = (
        state.contents,
        state.bulk_salt_diagnostic,
        *state.moments,
        *state.dynamics,
        state.t_surface,
        state.open_water_area,
        *card.metrics,
        *card.forcing_template,
    )
    if any(value.shape[:2] != shape for value in leaves):
        raise ValueError("ICE_RHEO card requires pinned 1004x1004 same-index storage")
    if any(value.dtype != jnp.float64 for value in leaves):
        raise ValueError("ICE_RHEO card requires fp64 state, forcing, and geometry")


def _replace_periodic_halo(value: jnp.ndarray) -> jnp.ndarray:
    interior = value[
        _ICE_RHEO_HALO_WIDTH : -_ICE_RHEO_HALO_WIDTH,
        _ICE_RHEO_HALO_WIDTH : -_ICE_RHEO_HALO_WIDTH,
    ]
    padding = (
        (_ICE_RHEO_HALO_WIDTH, _ICE_RHEO_HALO_WIDTH),
        (_ICE_RHEO_HALO_WIDTH, _ICE_RHEO_HALO_WIDTH),
    )
    return jnp.pad(interior, padding, mode="wrap")


def ice_rheo_air_stress(
    state: ICERheoState, ice_step_index: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate the shipped ice-relative wind stress for one outer step."""

    if ice_step_index < 1:
        raise ValueError("ICE_RHEO ice_step_index must be positive")
    physical_indices = jnp.arange(
        _ICE_RHEO_HALO_WIDTH + 1,
        _ICE_RHEO_HALO_WIDTH + _ICE_RHEO_GRID_SIZE + 1,
        dtype=jnp.float64,
    )
    x = _ICE_RHEO_DOMAIN_KM - (
        _ICE_RHEO_TWO * physical_indices * _ICE_RHEO_RESOLUTION_KM
    )
    y = x
    spinup = jnp.minimum(
        ice_step_index * _ICE_RHEO_DT_S / _ICE_RHEO_SPINUP_S,
        _ICE_RHEO_ONE,
    )
    # usrdef_sbc.F90:124-127 executes integer (1/4)==0, making the radial
    # denominator exactly one.  The remaining normalization is retained.
    normalization = _ICE_RHEO_WIND_MAX_M_S / jnp.sqrt(
        _ICE_RHEO_DOMAIN_KM * _ICE_RHEO_KM_TO_M
    )
    wind_u_physical = jnp.broadcast_to(
        normalization * x[:, None] * spinup,
        (_ICE_RHEO_GRID_SIZE, _ICE_RHEO_GRID_SIZE),
    )
    wind_v_physical = (
        jnp.broadcast_to(
            normalization * y[None, :] * spinup,
            (_ICE_RHEO_GRID_SIZE, _ICE_RHEO_GRID_SIZE),
        )
        * _ICE_RHEO_WIND_RATIO
    )
    wind_u = _periodic_halo(wind_u_physical)
    wind_v = _periodic_halo(wind_v_physical)
    relative_u = wind_u - _ICE_RHEO_RELATIVE_WIND * _ICE_RHEO_HALF * (
        jnp.roll(state.dynamics.u_ice_u, 1, axis=0) + state.dynamics.u_ice_u
    )
    relative_v = wind_v - _ICE_RHEO_RELATIVE_WIND * _ICE_RHEO_HALF * (
        jnp.roll(state.dynamics.v_ice_v, 1, axis=1) + state.dynamics.v_ice_v
    )
    magnitude = jnp.sqrt(relative_u * relative_u + relative_v * relative_v)
    stress_u = (
        _ICE_RHEO_AIR_DENSITY_KG_M3
        * _ICE_RHEO_AIR_DRAG
        * magnitude
        * relative_u
    )
    stress_v = (
        _ICE_RHEO_AIR_DENSITY_KG_M3
        * _ICE_RHEO_AIR_DRAG
        * magnitude
        * relative_v
    )
    return _replace_periodic_halo(stress_u), _replace_periodic_halo(stress_v)


__all__ = (
    "ICE_RHEO_TRACERS",
    "ICERheoCard",
    "ICERheoState",
    "build_ice_rheo_card",
    "ice_rheo_air_stress",
    "validate_ice_rheo_card",
)
