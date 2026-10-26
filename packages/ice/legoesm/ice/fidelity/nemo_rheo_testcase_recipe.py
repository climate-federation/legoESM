"""fp64 legoESM card boundary for NEMO/SI3 ``ICE_RHEO_OMIP_L3``.

The card composes the existing C-grid aEVP and Prather implementations with
the NEMO-literal SI3 ridging (:func:`legoesm.ice.ridging.si3_ice_dyn_rdgrft`)
and ``ice_cor`` (:func:`legoesm.ice.itd.si3_ice_cor_dyn`) kernels at jpl=1.  This
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
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG
from legoesm.ice.dynamics import (
    SI3CGridAEVPConfig,
    SI3CGridAEVPForcing,
    SI3CGridAEVPState,
    SI3CGridMetrics,
    si3_cgrid_aevp_solver,
    si3_cgrid_deformation,
)
from legoesm.ice.fidelity.nemo_adv2d_testcase_recipe import (
    apply_si3_prather_source_corrections,
)
from legoesm.ice.itd import SI3IceCorConfig, SI3IceCorState, si3_ice_cor_dyn
from legoesm.ice.ridging import (
    SI3RdgrftConfig,
    SI3RdgrftResult,
    SI3RdgrftState,
    si3_ice_dyn_rdgrft,
    si3_rdgrft_config_after_init,
)
from legoesm.ice.transport import (
    SI3PratherMoments,
    advect_si3_prather_2d,
    si3_prather_pack_intensives,
    si3_prather_unpack_intensives,
    zero_si3_prather_moments,
)

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
_ICE_RHEO_LF_DEPFRA = 0.125  # ORCA1 namelist_ice_ref:59; inactive on this card
_ICE_RHEO_LF_BFR_N_M3 = 15.0  # ORCA1 namelist_ice_ref:61; inactive on this card
_ICE_RHEO_LF_RELAX_S_INV = 1.0e-5  # ORCA1 namelist_ice_ref:62; inactive here
_ICE_RHEO_LF_TENSILE = 0.05  # ORCA1 namelist_ice_ref:63; inactive on this card
_ICE_RHEO_MINIMUM_THICKNESS_M = 0.1  # rn_himin
_ICE_RHEO_MAXIMUM_CONCENTRATION = 0.997  # rn_amax_n = rn_amax_s
_ICE_RHEO_MAXIMUM_THICKNESS_M = 99.0  # rn_himax = hi_max(jpl), iceitd.F90 ice_itd_init
_ICE_RHEO_MINIMUM_SALINITY_G_KG = 0.1  # rn_simin
_ICE_RHEO_NEW_ICE_SALINITY_FRACTION = 0.75  # rn_sinew
_ICE_RHEO_OCEAN_SALINITY_G_KG = 35.0
# Resolved namdyn_rdgrft (final/output.namelist.ice), before ice_dyn_rdgrft_init's
# ln_icethd=F override (si3_rdgrft_config_after_init) sets porosity 0 and retention 1.
_ICE_RHEO_RN_CSRDG = 0.5
_ICE_RHEO_RN_ASTAR = 0.03
_ICE_RHEO_RN_MURDG = 3.0
_ICE_RHEO_RN_HSTAR = 25.0
_ICE_RHEO_RN_PORORDG = 0.0
_ICE_RHEO_RN_FSNWRDG = 0.5
_ICE_RHEO_RN_FPNDRDG = 0.5
_ICE_RHEO_RN_HRAFT = 0.75
_ICE_RHEO_RN_CRAFT = 5.0
_ICE_RHEO_RN_FSNWRFT = 0.5
_ICE_RHEO_RN_FPNDRFT = 0.5

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
    sst_m: jnp.ndarray
    forcing_template: SI3CGridAEVPForcing
    dynamics_config: SI3CGridAEVPConfig
    ridging_config: SI3RdgrftConfig
    cor_config: SI3IceCorConfig
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


def _layer_fields(frame: Mapping[str, np.ndarray], name: str, levels: int) -> dict[str, np.ndarray]:
    value = np.asarray(frame[name], dtype=np.float64)
    expected = (
        _ICE_RHEO_ALLOCATED_SIZE,
        _ICE_RHEO_ALLOCATED_SIZE,
        levels,
        _ICE_RHEO_JPL,
    )
    if value.shape != expected:
        raise ValueError(f"ICE_RHEO frame {name} shape {value.shape} != {expected}")
    return {f"{name}_l{level + 1:02d}": value[..., level, 0] for level in range(levels)}


def _mesh_xy(mesh: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    value = np.asarray(mesh[name])
    while value.ndim > 2:
        value = value[0]
    expected = (_ICE_RHEO_GRID_SIZE, _ICE_RHEO_GRID_SIZE)
    if value.shape != expected:
        raise ValueError(f"ICE_RHEO mesh {name} shape {value.shape} != {expected}")
    return cast(np.ndarray, value.T)


def _ice_rheo_ridging_config() -> SI3RdgrftConfig:
    """ICE_RHEO's resolved namdyn_rdgrft through ice_dyn_rdgrft_init's override."""

    return si3_rdgrft_config_after_init(
        SI3RdgrftConfig(
            rn_csrdg=_ICE_RHEO_RN_CSRDG,
            ln_partf_exp=True,
            rn_astar=_ICE_RHEO_RN_ASTAR,
            ln_distf_lin=False,
            ln_distf_exp=True,
            rn_murdg=_ICE_RHEO_RN_MURDG,
            ln_ridging=True,
            rn_hstar=_ICE_RHEO_RN_HSTAR,
            rn_porordg=_ICE_RHEO_RN_PORORDG,
            rn_fsnwrdg=_ICE_RHEO_RN_FSNWRDG,
            rn_fpndrdg=_ICE_RHEO_RN_FPNDRDG,
            ln_rafting=True,
            rn_hraft=_ICE_RHEO_RN_HRAFT,
            rn_craft=_ICE_RHEO_RN_CRAFT,
            rn_fsnwrft=_ICE_RHEO_RN_FSNWRFT,
            rn_fpndrft=_ICE_RHEO_RN_FPNDRFT,
            ln_rhg_evp_or_vp=True,
            ln_pnd=True,
            ln_pnd_lids=True,
            nn_icesal=_ICE_RHEO_NN_ICESAL,
            rdt_ice=_ICE_RHEO_DT_S,
            rhoi=NEMO_SI3_CONSTANTS_CONFIG.rho_ice,
            rhos=NEMO_SI3_CONSTANTS_CONFIG.rho_snow,
            rhow=constants.rho_water,
            rcp=NEMO_SI3_CONSTANTS_CONFIG.c_ocean,
        ),
        ln_icethd=False,
    )


def _ice_rheo_cor_config() -> SI3IceCorConfig:
    """ICE_RHEO's resolved ice_cor constants."""

    return SI3IceCorConfig(
        rn_himin=_ICE_RHEO_MINIMUM_THICKNESS_M,
        rn_simin=_ICE_RHEO_MINIMUM_SALINITY_G_KG,
        rn_sinew=_ICE_RHEO_NEW_ICE_SALINITY_FRACTION,
        nn_icesal=_ICE_RHEO_NN_ICESAL,
        ln_pnd=True,
        ln_pnd_lids=True,
        rdt_ice=_ICE_RHEO_DT_S,
        rhoi=NEMO_SI3_CONSTANTS_CONFIG.rho_ice,
        rhos=NEMO_SI3_CONSTANTS_CONFIG.rho_snow,
        rhow=constants.rho_water,
        rt0=constants.T_freeze,
    )


def ice_rheo_frame_contents(entry_frame: Mapping[str, np.ndarray]) -> jnp.ndarray:
    """An oracle frame's inventories in the card's ``ICE_RHEO_TRACERS`` layout."""

    fields = {
        name: _frame_field(entry_frame, name)
        for name in ("v_i", "v_s", "a_i", "oa_i", "a_ip", "v_ip", "v_il")
    }
    fields.update(_layer_fields(entry_frame, "e_s", _ICE_RHEO_NLAY_S))
    fields.update(_layer_fields(entry_frame, "e_i", _ICE_RHEO_NLAY_I))
    fields.update(_layer_fields(entry_frame, "szv_i", _ICE_RHEO_NLAY_I))
    return jnp.asarray(
        np.stack([fields[name] for name in ICE_RHEO_TRACERS], axis=-1),
        dtype=jnp.float64,
    )


def build_ice_rheo_card(
    entry_frame: Mapping[str, np.ndarray],
    mesh: Mapping[str, np.ndarray],
    sst_m: np.ndarray,
) -> ICERheoCard:
    """Build the pinned fp64 card from oracle arrays, never analytic stand-ins.

    ``sst_m`` is NEMO's ocean surface temperature in degC (``output.init_ice.nc``).
    """

    oracle_policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(oracle_policy)
    if get_policy() != oracle_policy:
        raise RuntimeError("ICE_RHEO card requires fp64/scalar-libm policy")
    contents = ice_rheo_frame_contents(entry_frame)
    a_i = np.asarray(contents[..., ICE_RHEO_TRACERS.index("a_i")])
    tmask = _periodic_halo(_mesh_xy(mesh, "tmask").astype(np.float64))
    umask = _periodic_halo(_mesh_xy(mesh, "umask").astype(np.float64))
    vmask = _periodic_halo(_mesh_xy(mesh, "vmask").astype(np.float64))
    zero = jnp.zeros(contents.shape[:2], dtype=jnp.float64)
    ocean_temperature = np.asarray(sst_m, dtype=np.float64)
    expected_surface = (_ICE_RHEO_GRID_SIZE, _ICE_RHEO_GRID_SIZE)
    if ocean_temperature.shape != expected_surface:
        raise ValueError(
            f"ICE_RHEO ocean temperature shape {ocean_temperature.shape} != {expected_surface}"
        )
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
        concentration_t=_intensive(contents, "a_i"),
        ice_volume_t=_intensive(contents, "v_i"),
        snow_volume_t=_intensive(contents, "v_s"),
        pond_volume_t=_intensive(contents, "v_ip"),
        lid_volume_t=_intensive(contents, "v_il"),
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
        depth_t=zero,
        depth_u=zero,
        depth_v=zero,
        iceberg_tmask=zero,
        iceberg_umask=zero,
        iceberg_vmask=zero,
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
        landfast_depth_fraction=_ICE_RHEO_LF_DEPFRA,
        landfast_basal_friction_n_m3=_ICE_RHEO_LF_BFR_N_M3,
        landfast_relaxation_s_inv=_ICE_RHEO_LF_RELAX_S_INV,
        landfast_tensile_fraction=_ICE_RHEO_LF_TENSILE,
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
        open_water_area=_ICE_RHEO_ONE - jnp.asarray(a_i),
    )
    card = ICERheoCard(
        case=_ICE_RHEO_CASE,
        dynamics_scheme=_ICE_RHEO_DYNAMICS,
        rheology_staggering=_ICE_RHEO_STAGGERING,
        transport_scheme=_ICE_RHEO_TRANSPORT,
        ridging_scheme=_ICE_RHEO_RIDGING,
        metrics=metrics,
        sst_m=_periodic_halo(ocean_temperature.T),
        forcing_template=forcing,
        dynamics_config=config,
        ridging_config=_ice_rheo_ridging_config(),
        cor_config=_ice_rheo_cor_config(),
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


def validate_ice_rheo_card(card: ICERheoCard, state: ICERheoState | None = None) -> None:
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
    if card.ridging_config != _ice_rheo_ridging_config():
        raise ValueError("ICE_RHEO ridging namelist differs from the resolved deck")
    if card.cor_config != _ice_rheo_cor_config():
        raise ValueError("ICE_RHEO ice_cor constants differ from the resolved deck")
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
        card.sst_m,
        *card.metrics,
        *card.forcing_template,
    )
    if any(value.shape[:2] != shape for value in leaves):
        raise ValueError("ICE_RHEO card requires pinned 1004x1004 same-index storage")
    if any(value.dtype != jnp.float64 for value in leaves):
        raise ValueError("ICE_RHEO card requires fp64 state, forcing, and geometry")


def _replace_periodic_halo(value: jnp.ndarray) -> jnp.ndarray:
    interior = value[
        _ICE_RHEO_HALO_WIDTH:-_ICE_RHEO_HALO_WIDTH,
        _ICE_RHEO_HALO_WIDTH:-_ICE_RHEO_HALO_WIDTH,
    ]
    padding = (
        (_ICE_RHEO_HALO_WIDTH, _ICE_RHEO_HALO_WIDTH),
        (_ICE_RHEO_HALO_WIDTH, _ICE_RHEO_HALO_WIDTH),
    ) + ((0, 0),) * (value.ndim - 2)
    return jnp.pad(interior, padding, mode="wrap")


def ice_rheo_air_stress(
    state: ICERheoState, ice_step_index: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate the shipped ice-relative wind stress for one outer step."""

    if ice_step_index < 1:
        raise ValueError("ICE_RHEO ice_step_index must be positive")
    return _ice_rheo_air_stress_impl(state, ice_step_index)


def _ice_rheo_air_stress_impl(
    state: ICERheoState, ice_step_index: int | jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Traceable air-stress body after the public clock validation."""

    physical_indices = jnp.arange(
        _ICE_RHEO_HALO_WIDTH + 1,
        _ICE_RHEO_HALO_WIDTH + _ICE_RHEO_GRID_SIZE + 1,
        dtype=jnp.float64,
    )
    x = _ICE_RHEO_DOMAIN_KM - (_ICE_RHEO_TWO * physical_indices * _ICE_RHEO_RESOLUTION_KM)
    y = x
    spinup = jnp.minimum(
        ice_step_index * _ICE_RHEO_DT_S / _ICE_RHEO_SPINUP_S,
        _ICE_RHEO_ONE,
    )
    # usrdef_sbc.F90:124-127 executes integer (1/4)==0, making the radial
    # denominator exactly one.  The remaining normalization is retained.
    normalization = _ICE_RHEO_WIND_MAX_M_S / jnp.sqrt(_ICE_RHEO_DOMAIN_KM * _ICE_RHEO_KM_TO_M)
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
    stress_u = _ICE_RHEO_AIR_DENSITY_KG_M3 * _ICE_RHEO_AIR_DRAG * magnitude * relative_u
    stress_v = _ICE_RHEO_AIR_DENSITY_KG_M3 * _ICE_RHEO_AIR_DRAG * magnitude * relative_v
    return _replace_periodic_halo(stress_u), _replace_periodic_halo(stress_v)


def _intensive(contents: jnp.ndarray, name: str) -> jnp.ndarray:
    return contents[..., ICE_RHEO_TRACERS.index(name)]


def _forcing_for_state(
    card: ICERheoCard, state: ICERheoState, ice_step_index: int | jnp.ndarray
) -> SI3CGridAEVPForcing:
    stress_u, stress_v = _ice_rheo_air_stress_impl(state, ice_step_index)
    return card.forcing_template._replace(
        concentration_t=_intensive(state.contents, "a_i"),
        ice_volume_t=_intensive(state.contents, "v_i"),
        snow_volume_t=_intensive(state.contents, "v_s"),
        pond_volume_t=_intensive(state.contents, "v_ip"),
        lid_volume_t=_intensive(state.contents, "v_il"),
        air_stress_u_t=stress_u,
        air_stress_v_t=stress_v,
    )


def _layers(contents: jnp.ndarray, name: str, levels: int) -> jnp.ndarray:
    return jnp.stack(
        [_intensive(contents, f"{name}_l{level:02d}") for level in range(1, levels + 1)],
        axis=-1,
    )


def _set_category_fields(contents: jnp.ndarray, state) -> jnp.ndarray:
    """Write a jpl=1 SI3 kernel state (category axis last) back into the card layout."""

    for name in ("a_i", "v_i", "v_s", "oa_i", "a_ip", "v_ip", "v_il"):
        contents = contents.at[..., ICE_RHEO_TRACERS.index(name)].set(getattr(state, name)[..., 0])
    layered = (("e_s", _ICE_RHEO_NLAY_S), ("e_i", _ICE_RHEO_NLAY_I), ("szv_i", _ICE_RHEO_NLAY_I))
    for name, levels in layered:
        for level in range(1, levels + 1):
            contents = contents.at[..., ICE_RHEO_TRACERS.index(f"{name}_l{level:02d}")].set(
                getattr(state, name)[..., level - 1, 0]
            )
    return contents


def ridge_ice_rheo(
    card: ICERheoCard,
    contents: jnp.ndarray,
    open_water: jnp.ndarray,
    divergence: jnp.ndarray,
    deformation: jnp.ndarray,
) -> SI3RdgrftResult:
    """``ice_dyn_rdgrft`` on the card's intensive inventories, at ``icedyn.F90:193``.

    The dynamics fluxes start at zero: this card carries none of them.
    """

    zero = jnp.zeros_like(open_water)

    def cat(name: str) -> jnp.ndarray:
        return _intensive(contents, name)[..., None]

    state = SI3RdgrftState(
        a_i=cat("a_i"),
        v_i=cat("v_i"),
        v_s=cat("v_s"),
        oa_i=cat("oa_i"),
        a_ip=cat("a_ip"),
        v_ip=cat("v_ip"),
        v_il=cat("v_il"),
        e_s=_layers(contents, "e_s", _ICE_RHEO_NLAY_S)[..., None],
        e_i=_layers(contents, "e_i", _ICE_RHEO_NLAY_I)[..., None],
        szv_i=_layers(contents, "szv_i", _ICE_RHEO_NLAY_I)[..., None],
        at_i=_intensive(contents, "a_i"),
        ato_i=open_water,
        wfx_dyn=zero,
        sfx_dyn=zero,
        hfx_dyn=zero,
        wfx_snw_dyn=zero,
        wfx_pnd=zero,
    )
    return si3_ice_dyn_rdgrft(
        state,
        divu_i=divergence,
        delta_i=deformation,
        sst_m=card.sst_m,
        sss_m=jnp.full_like(open_water, _ICE_RHEO_OCEAN_SALINITY_G_KG),
        hi_max=jnp.asarray([_ICE_RHEO_ZERO, _ICE_RHEO_MAXIMUM_THICKNESS_M]),
        nn_hls=card.halo_width,
        config=card.ridging_config,
    )


def _ice_cor(
    card: ICERheoCard,
    contents: jnp.ndarray,
    t_surface: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``ice_cor(kt, 1)`` at ``icedyn.F90:194``; only the inventories and ``t_su`` are carried."""

    zero = jnp.zeros_like(t_surface)
    e_s = _layers(contents, "e_s", _ICE_RHEO_NLAY_S)[..., None]
    e_i = _layers(contents, "e_i", _ICE_RHEO_NLAY_I)[..., None]

    def cat(name: str) -> jnp.ndarray:
        return _intensive(contents, name)[..., None]

    state = SI3IceCorState(
        a_i=cat("a_i"),
        v_i=cat("v_i"),
        v_s=cat("v_s"),
        oa_i=cat("oa_i"),
        a_ip=cat("a_ip"),
        v_ip=cat("v_ip"),
        v_il=cat("v_il"),
        e_s=e_s,
        e_i=e_i,
        szv_i=_layers(contents, "szv_i", _ICE_RHEO_NLAY_I)[..., None],
        sz_i=jnp.zeros_like(e_i),
        t_su=t_surface[..., None],
        t_i=jnp.zeros_like(e_i),
        t_s=jnp.zeros_like(e_s),
        h_i=zero[..., None],
        h_s=zero[..., None],
        h_ip=zero[..., None],
        h_il=zero[..., None],
        at_i=zero,
        vt_i=zero,
        ato_i=zero,
        sfx_res=zero,
        hfx_res=zero,
        wfx_res=zero,
    )
    out = si3_ice_cor_dyn(
        state,
        rn_amax_2d=jnp.full_like(zero, _ICE_RHEO_MAXIMUM_CONCENTRATION),
        sss_m=jnp.full_like(zero, _ICE_RHEO_OCEAN_SALINITY_G_KG),
        sst_m=card.sst_m,
        hi_max=jnp.asarray([_ICE_RHEO_ZERO, _ICE_RHEO_MAXIMUM_THICKNESS_M]),
        # ice_itd_reb is not called at jpl=1 (icecor.F90:99): NaN so any use is loud.
        hi_mean=jnp.asarray([jnp.nan]),
        nn_hls=card.halo_width,
        config=card.cor_config,
    )
    return (
        _replace_periodic_halo(_set_category_fields(contents, out)),
        _replace_periodic_halo(out.t_su[..., 0]),
    )


def step_ice_rheo_card(
    card: ICERheoCard,
    state: ICERheoState | None = None,
    *,
    completed_steps: int,
) -> ICERheoState:
    """Advance the resolved ``rhg -> adv -> rdgrft -> cor`` outer step.

    The order is the shipped dynALL dispatcher at ``icedyn.F90:130-135``.
    Thermodynamics, landfast, multi-category redistribution, and every other
    scheme cross-product are rejected by :func:`validate_ice_rheo_card`.
    """

    state = card.initial_state if state is None else state
    validate_ice_rheo_card(card, state)
    if not 0 <= completed_steps < card.n_steps:
        raise ValueError("ICE_RHEO completed_steps out of range")
    ice_step_index = completed_steps + 1
    return _step_ice_rheo_card_impl(
        card,
        state,
        ice_step_index=ice_step_index,
        transport_step_index=ice_step_index,
    )


def _step_ice_rheo_card_impl(
    card: ICERheoCard,
    state: ICERheoState,
    *,
    ice_step_index: int | jnp.ndarray,
    transport_step_index: int,
) -> ICERheoState:
    """Traceable step body with a static NEMO sweep-parity selector."""

    forcing = _forcing_for_state(card, state, ice_step_index)
    dynamics = si3_cgrid_aevp_solver(state.dynamics, forcing, card.metrics, card.dynamics_config)
    divergence, deformation = si3_cgrid_deformation(
        dynamics, forcing, card.metrics, card.dynamics_config
    )
    cell_area = jnp.full_like(dynamics.u_ice_u, _ICE_RHEO_CELL_AREA_M2, dtype=jnp.float64)
    wet = card.forcing_template.tmask_t.astype(bool)
    entry_contents = si3_prather_pack_intensives(state.contents, cell_area)
    contents, moments, ignored_subcycles = advect_si3_prather_2d(
        entry_contents,
        state.moments,
        dynamics.u_ice_u,
        dynamics.v_ice_v,
        cell_area,
        wet,
        card.dt_s,
        dx=_ICE_RHEO_SPACING_M,
        dy=_ICE_RHEO_SPACING_M,
        ice_step_index=transport_step_index,
        nn_fsbc=card.nn_fsbc,
        halo_width=card.halo_width,
        ice_volume_index=ICE_RHEO_TRACERS.index("v_i"),
        concentration_index=ICE_RHEO_TRACERS.index("a_i"),
        subcycles=card.subcycles,
    )
    del ignored_subcycles
    contents = si3_prather_unpack_intensives(contents, cell_area, wet)
    contents = apply_si3_prather_source_corrections(
        state.contents,
        contents,
        tracer_names=ICE_RHEO_TRACERS,
        nlay_i=card.nlay_i,
        nlay_s=card.nlay_s,
        cell_area_m2=_ICE_RHEO_CELL_AREA_M2,
        halo_width=card.halo_width,
        entry_intensive_contents=state.contents,
        contents_are_intensive=True,
    )

    entry_area = _intensive(state.contents, "a_i")
    transported_area = _intensive(contents, "a_i")
    u_transport = dynamics.u_ice_u * card.metrics.e2u
    v_transport = dynamics.v_ice_v * card.metrics.e1v
    flux_divergence = (
        (u_transport - jnp.roll(u_transport, 1, axis=0))
        + (v_transport - jnp.roll(v_transport, 1, axis=1))
    ) / card.metrics.area_t
    open_water = jnp.maximum(
        _ICE_RHEO_ZERO,
        state.open_water_area - (transported_area - entry_area) - flux_divergence * card.dt_s,
    )
    ridged = ridge_ice_rheo(card, contents, open_water, divergence, deformation).state
    contents = _set_category_fields(contents, ridged)
    contents, t_surface = _ice_cor(card, contents, state.t_surface)
    final_area = _intensive(contents, "a_i")
    return ICERheoState(
        contents=contents,
        bulk_salt_diagnostic=_replace_periodic_halo(state.bulk_salt_diagnostic),
        moments=moments,
        dynamics=dynamics,
        t_surface=t_surface,
        # icestp.F90:182-184 calls ice_var_agg after ice_dyn, so the next
        # entry's open-water state is exactly one minus the corrected area.
        open_water_area=_ICE_RHEO_ONE - final_area,
    )


__all__ = (
    "ICE_RHEO_TRACERS",
    "ICERheoCard",
    "ICERheoState",
    "build_ice_rheo_card",
    "ice_rheo_air_stress",
    "ice_rheo_frame_contents",
    "ridge_ice_rheo",
    "step_ice_rheo_card",
    "validate_ice_rheo_card",
)
