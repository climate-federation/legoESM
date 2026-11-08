"""Pure fp64 legoESM card for the shipped NEMO/SI3 ``ICE_ADV2D`` case."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import NamedTuple, cast

import jax.numpy as jnp
import numpy as np
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.core.smooth import positive_part_c1
from legoesm.grids.latlon import create_beta_plane_cgrid_geometry
from legoesm.ice.transport import (
    SI3_PRATHER_MOMENT_NAMES,
    SI3PratherMoments,
    advect_si3_prather_2d,
    si3_hbig_pra,
    si3_hsnow_pra,
    si3_icemax2d_pra,
    si3_prather_pack_intensives,
    si3_prather_unpack_intensives,
    si3_var_zapneg,
    si3_zapsmall_mask,
    zero_si3_prather_moments,
)

from legoesm import constants

ICE_ADV2D_TRACERS = (
    "v_i",
    "v_s",
    "a_i",
    "oa_i",
    "e_s_l01",
    "e_s_l02",
    "e_s_l03",
    "e_i_l01",
    "e_i_l02",
    "e_i_l03",
    "szv_i_l01",
    "szv_i_l02",
    "szv_i_l03",
    "a_ip",
    "v_ip",
    "v_il",
)
ICE_ADV2D_RESTART_FORMAT = "legoesm-ice-adv2d-state-v1"

# Values resolved by the pinned shipped-case oracle. Sources are the immutable
# case, shared-reference, and ORCA1 files named in the receipt.
_ICE_ADV2D_GRID_SIZE = 99  # usrdef_nam.F90:82-87 with namelist_cfg:19-20
_ICE_ADV2D_SPACING_M = 3000.0  # ICE_ADV2D/EXPREF/namelist_cfg:19-20
_ICE_ADV2D_ORIGIN_M = -148500.0  # usrdef_hgr.F90:77-103
_ICE_ADV2D_HALO_WIDTH = 2  # built domain/frame header: 103 = 99 + 2*2
_ICE_ADV2D_N_STEPS = 485  # ICE_ADV2D/EXPREF/namelist_cfg:29-30
_ICE_ADV2D_DT_S = 1200.0  # ICE_ADV2D/EXPREF/namelist_cfg:37
_ICE_ADV2D_JPL = 1  # ORCA1 namelist_ice_cfg:24
_ICE_ADV2D_NLAY_I = 3  # ORCA1 namelist_ice_cfg:25
_ICE_ADV2D_NLAY_S = 3  # ORCA1 namelist_ice_cfg:26
_ICE_ADV2D_NN_ICESAL = 4  # SHARED/namelist_ice_ref:199; resolved oracle
_ICE_ADV2D_NN_FSBC = 1  # ICE_ADV2D/EXPREF/namelist_cfg:76
_ICE_ADV2D_SUBCYCLES = 1  # icedyn_adv_pra.F90:116-132; case CFL=0.2
_ICE_ADV2D_U_M_S = 0.5  # ICE_ADV2D/EXPREF/namelist_ice_cfg:38
_ICE_ADV2D_V_M_S = 0.5  # ICE_ADV2D/EXPREF/namelist_ice_cfg:39
_ICE_ADV2D_SIGMA_X = -0.04  # make_INITICE.py:101
_ICE_ADV2D_SIGMA_Y = -0.04  # make_INITICE.py:102
_ICE_ADV2D_SHIFT_X = 49.0  # make_INITICE.py:103
_ICE_ADV2D_SHIFT_Y = 49.0  # make_INITICE.py:104
_ICE_ADV2D_PATCH_MARGIN_X = 21  # make_INITICE.py:105-110
_ICE_ADV2D_PATCH_MARGIN_Y = 21  # make_INITICE.py:105-110
_ICE_ADV2D_INITIAL_THICKNESS_M = 2.0  # make_INITICE.py:49,111
_ICE_ADV2D_INITIAL_SNOW_M = 0.2  # make_INITICE.py:50,112
_ICE_ADV2D_INITIAL_CONCENTRATION = 0.9  # make_INITICE.py:51,113
_ICE_ADV2D_INITIAL_SALINITY_G_KG = 6.3  # make_INITICE.py:52,114
_ICE_ADV2D_INITIAL_TEMPERATURE_K = 270.0  # make_INITICE.py:53-54,87-88
_ICE_ADV2D_INITIAL_POND_FRACTION = 0.2  # SHARED/namelist_ice_ref:281-282
_ICE_ADV2D_INITIAL_POND_DEPTH_M = 0.05  # SHARED/namelist_ice_ref:283-284
_ICE_ADV2D_INITIAL_LID_DEPTH_M = 0.0  # SHARED/namelist_ice_ref:285-286
_ICE_ADV2D_SMALL = 1.0e-10  # icevar.F90:611-706 (`epsi10`)
_ICE_ADV2D_MAXIMUM_FLOOR = 1.0e-20  # icedyn_adv_pra.F90:1519-1534,1561-1571 (`epsi20`)
_ICE_ADV2D_HBIG_CONCENTRATION = 0.15  # icedyn_adv_pra.F90:992,1000,1007
# Width of the C1 one-sided ramp that replaces zapneg's final hard
# WHERE(pa_i < 0) pa_i = 0 (icevar.F90:836): 0.01 x epsi10, the port's
# concentration-presence threshold (vote W2, me/codex/GLM).  0.0 = hard NEMO.
_SI3_NEGATIVE_AREA_RAMP_WIDTH = 1.0e-12


class SI3PratherResidualFluxes(NamedTuple):
    """Ocean residual ledgers of the Prather corrections (per unit cell area)."""

    wfx_res: jnp.ndarray  # [kg m-2 s-1], + into ocean
    hfx_res: jnp.ndarray  # [W m-2], -(removed ice/snow enthalpy)/dt
    sfx_res: jnp.ndarray  # [g m-2 s-1], + into ocean


class ICEAdv2DState(NamedTuple):
    contents: jnp.ndarray
    bulk_salt_diagnostic: jnp.ndarray
    moments: SI3PratherMoments
    u_ice: jnp.ndarray
    v_ice: jnp.ndarray
    t_surface: jnp.ndarray


class ICEAdv2DCard(NamedTuple):
    case: str
    transport_scheme: str
    grid: object
    wet_global_xy: jnp.ndarray
    prescribed_u_ice: jnp.ndarray
    prescribed_v_ice: jnp.ndarray
    empty_surface_temperature: jnp.ndarray
    initial_state: ICEAdv2DState
    dt_s: float
    dx_m: float
    dy_m: float
    halo_width: int
    n_steps: int
    nn_fsbc: int
    subcycles: int
    jpl: int
    nlay_i: int
    nlay_s: int
    nn_icesal: int
    thermodynamics: bool
    ponds: bool
    landfast: bool


def _initial_fields_xy(size: int) -> dict[str, np.ndarray]:
    """Transcribe the float32 assignments in ``make_INITICE.py:82-114``."""

    h_i: np.ndarray = np.zeros((size, size), dtype=np.float32)
    h_s = np.zeros_like(h_i)
    a_i = np.zeros_like(h_i)
    s_i = np.zeros_like(h_i)
    for y in range(_ICE_ADV2D_PATCH_MARGIN_Y, size - _ICE_ADV2D_PATCH_MARGIN_Y):
        for x in range(_ICE_ADV2D_PATCH_MARGIN_X, size - _ICE_ADV2D_PATCH_MARGIN_X):
            gaussian = np.exp(_ICE_ADV2D_SIGMA_X * (x - _ICE_ADV2D_SHIFT_X) ** 2)
            gaussian *= np.exp(_ICE_ADV2D_SIGMA_Y * (y - _ICE_ADV2D_SHIFT_Y) ** 2)
            h_i[x, y] = np.float32(_ICE_ADV2D_INITIAL_THICKNESS_M * gaussian)
            h_s[x, y] = np.float32(_ICE_ADV2D_INITIAL_SNOW_M * gaussian)
            a_i[x, y] = np.float32(_ICE_ADV2D_INITIAL_CONCENTRATION)
            s_i[x, y] = np.float32(_ICE_ADV2D_INITIAL_SALINITY_G_KG)
    h_i = h_i.astype(np.float64)
    h_s = h_s.astype(np.float64)
    a_i = a_i.astype(np.float64)
    s_i = s_i.astype(np.float64)
    v_i = h_i * a_i
    v_s = h_s * a_i
    sv_i = s_i * v_i
    melt_temperature = constants.T_freeze - constants.mu_ice_freeze * s_i
    temperature = np.float64(_ICE_ADV2D_INITIAL_TEMPERATURE_K)
    phase_fraction = np.maximum(
        0.0,
        1.0
        - (melt_temperature - constants.T_freeze)
        / np.minimum(temperature - constants.T_freeze, -1.0e-10),
    )
    ice_energy = (
        v_i
        * constants.rho_ice
        / _ICE_ADV2D_NLAY_I
        * (
            constants.c_p_ice_nemo * (melt_temperature - temperature)
            + constants.L_fus_nemo * phase_fraction
            - constants.c_p_seawater * (melt_temperature - constants.T_freeze)
        )
    )
    snow_energy = (
        v_s
        * constants.rho_snow
        / _ICE_ADV2D_NLAY_S
        * (constants.c_p_ice_nemo * (constants.T_freeze - temperature) + constants.L_fus_nemo)
    )
    zero = np.zeros_like(v_i)
    return {
        "v_i": v_i,
        "v_s": v_s,
        "a_i": a_i,
        "oa_i": zero,
        "e_s_l01": snow_energy,
        "e_s_l02": snow_energy,
        "e_s_l03": snow_energy,
        "e_i_l01": ice_energy,
        "e_i_l02": ice_energy,
        "e_i_l03": ice_energy,
        "szv_i_l01": sv_i / _ICE_ADV2D_NLAY_I,
        "szv_i_l02": sv_i / _ICE_ADV2D_NLAY_I,
        "szv_i_l03": sv_i / _ICE_ADV2D_NLAY_I,
        "a_ip": _ICE_ADV2D_INITIAL_POND_FRACTION * a_i,
        "v_ip": (_ICE_ADV2D_INITIAL_POND_DEPTH_M * _ICE_ADV2D_INITIAL_POND_FRACTION * a_i),
        "v_il": _ICE_ADV2D_INITIAL_LID_DEPTH_M * a_i,
        "sv_i": sv_i,
    }


def ice_adv2d_card_contract_sha256(card: ICEAdv2DCard) -> str:
    selectors = {
        name: getattr(card, name)
        for name in (
            "case",
            "transport_scheme",
            "dt_s",
            "dx_m",
            "dy_m",
            "halo_width",
            "n_steps",
            "nn_fsbc",
            "subcycles",
            "jpl",
            "nlay_i",
            "nlay_s",
            "nn_icesal",
            "thermodynamics",
            "ponds",
            "landfast",
        )
    }
    schema = {
        "selectors": selectors,
        "ordered_tracers": ICE_ADV2D_TRACERS,
        "ordered_moments": SI3_PRATHER_MOMENT_NAMES,
    }
    digest = hashlib.sha256(json.dumps(schema, sort_keys=True).encode())
    for name in (
        "wet_global_xy",
        "prescribed_u_ice",
        "prescribed_v_ice",
        "empty_surface_temperature",
    ):
        value = np.ascontiguousarray(np.asarray(getattr(card, name)))
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def build_ice_adv2d_card(ocean_surface_temperature_c: np.ndarray) -> ICEAdv2DCard:
    set_policy(PrecisionPolicy.fp64())
    if get_policy() != PrecisionPolicy.fp64():
        raise RuntimeError("ICE_ADV2D card requires PrecisionPolicy.fp64()")
    size = _ICE_ADV2D_GRID_SIZE
    halo = _ICE_ADV2D_HALO_WIDTH
    grid = create_beta_plane_cgrid_geometry(
        size,
        size,
        dx_m=_ICE_ADV2D_SPACING_M,
        dy_m=_ICE_ADV2D_SPACING_M,
        f0=0.0,
        beta=0.0,
        x_origin_m=_ICE_ADV2D_ORIGIN_M,
        y_origin_m=_ICE_ADV2D_ORIGIN_M,
        cartesian_pseudo_lat=False,
        dtype=jnp.float64,
    )
    sst = np.asarray(ocean_surface_temperature_c, dtype=np.float64)
    if sst.shape != (size, size):
        raise ValueError("ICE_ADV2D ocean surface temperature shape mismatch")
    fields = _initial_fields_xy(size)
    area = _ICE_ADV2D_SPACING_M * _ICE_ADV2D_SPACING_M
    contents_global = np.stack([fields[name] * area for name in ICE_ADV2D_TRACERS], axis=-1)
    pad3 = ((halo, halo), (halo, halo), (0, 0))
    contents = jnp.asarray(np.pad(contents_global, pad3, mode="wrap"), dtype=jnp.float64)
    full_shape = contents.shape[:2]
    prescribed_u = jnp.full(full_shape, _ICE_ADV2D_U_M_S, dtype=jnp.float64)
    prescribed_v = jnp.full(full_shape, _ICE_ADV2D_V_M_S, dtype=jnp.float64)
    t_surface_global: np.ndarray = np.full(
        (size, size), _ICE_ADV2D_INITIAL_TEMPERATURE_K, dtype=np.float64
    )
    t_surface = jnp.asarray(np.pad(t_surface_global, halo, mode="wrap"))
    empty_temperature = jnp.asarray(
        np.pad(sst + constants.T_freeze, halo, mode="wrap"), dtype=jnp.float64
    )
    zero_velocity = jnp.zeros(full_shape, dtype=jnp.float64)
    wet_global = jnp.ones((size, size), dtype=bool)
    return ICEAdv2DCard(
        case="ICE_ADV2D_OMIP_L3",
        transport_scheme="si3_prather_xy_alternating",
        grid=grid,
        wet_global_xy=wet_global,
        prescribed_u_ice=prescribed_u,
        prescribed_v_ice=prescribed_v,
        empty_surface_temperature=empty_temperature,
        initial_state=ICEAdv2DState(
            contents=contents,
            # With nn_icesal=4, psv_i is a stale carried diagnostic: active
            # transport uses pszv_i (`icedyn_adv_pra.F90:232-240,368-375`).
            bulk_salt_diagnostic=jnp.asarray(
                np.pad(fields["sv_i"], halo, mode="wrap"), dtype=jnp.float64
            ),
            moments=zero_si3_prather_moments(contents),
            u_ice=zero_velocity,
            v_ice=zero_velocity,
            t_surface=t_surface,
        ),
        dt_s=_ICE_ADV2D_DT_S,
        dx_m=_ICE_ADV2D_SPACING_M,
        dy_m=_ICE_ADV2D_SPACING_M,
        halo_width=halo,
        n_steps=_ICE_ADV2D_N_STEPS,
        nn_fsbc=_ICE_ADV2D_NN_FSBC,
        subcycles=_ICE_ADV2D_SUBCYCLES,
        jpl=_ICE_ADV2D_JPL,
        nlay_i=_ICE_ADV2D_NLAY_I,
        nlay_s=_ICE_ADV2D_NLAY_S,
        nn_icesal=_ICE_ADV2D_NN_ICESAL,
        thermodynamics=False,
        ponds=True,
        landfast=False,
    )


def validate_ice_adv2d_card(card: ICEAdv2DCard, state: ICEAdv2DState | None = None) -> None:
    expected = (
        "ICE_ADV2D_OMIP_L3",
        "si3_prather_xy_alternating",
        _ICE_ADV2D_JPL,
        _ICE_ADV2D_NLAY_I,
        _ICE_ADV2D_NLAY_S,
        _ICE_ADV2D_NN_ICESAL,
        _ICE_ADV2D_DT_S,
        _ICE_ADV2D_SPACING_M,
        _ICE_ADV2D_SPACING_M,
        _ICE_ADV2D_HALO_WIDTH,
        _ICE_ADV2D_N_STEPS,
        _ICE_ADV2D_NN_FSBC,
        _ICE_ADV2D_SUBCYCLES,
        False,
        True,
        False,
    )
    actual = (
        card.case,
        card.transport_scheme,
        card.jpl,
        card.nlay_i,
        card.nlay_s,
        card.nn_icesal,
        card.dt_s,
        card.dx_m,
        card.dy_m,
        card.halo_width,
        card.n_steps,
        card.nn_fsbc,
        card.subcycles,
        card.thermodynamics,
        card.ponds,
        card.landfast,
    )
    if actual != expected:
        raise ValueError(f"ICE_ADV2D selector composition {actual!r} != {expected!r}")
    state = card.initial_state if state is None else state
    if state.contents.shape[:2] != card.prescribed_u_ice.shape:
        raise ValueError("ICE_ADV2D state shape mismatch")
    if state.bulk_salt_diagnostic.shape != card.prescribed_u_ice.shape:
        raise ValueError("ICE_ADV2D bulk salt diagnostic shape mismatch")


def apply_ice_adv2d_zapsmall(
    card: ICEAdv2DCard,
    state: ICEAdv2DState,
    contents: jnp.ndarray,
    moments: SI3PratherMoments,
    *,
    contents_are_intensive: bool = False,
) -> ICEAdv2DState:
    # NEMO restores the intensive fields at icedyn_adv_pra.F90:355-381 before
    # calling Hbig/Hsnow/zapneg at :405-421.  The rung-3.2 card retains its
    # historical extensive outer-state contract; the dynamics cards use the
    # literal intensive call order and therefore must not cross the area
    # bridge a second time here.
    area = 1.0 if contents_are_intensive else card.dx_m * card.dy_m
    v_i = contents[..., ICE_ADV2D_TRACERS.index("v_i")] / area
    a_i = contents[..., ICE_ADV2D_TRACERS.index("a_i")] / area
    small, _ = si3_zapsmall_mask(v_i, a_i, nn_hls=0)
    contents = jnp.where(small[..., None], 0.0, contents)
    return state._replace(
        contents=contents,
        moments=moments,
        u_ice=card.prescribed_u_ice,
        v_ice=card.prescribed_v_ice,
        t_surface=jnp.where(small, card.empty_surface_temperature, state.t_surface),
    )


def _max9(value: jnp.ndarray) -> jnp.ndarray:
    """SI3's floored surrounding maximum (`icedyn_adv_pra.F90:1519-1571`)."""

    neighbors = (
        jnp.roll(jnp.roll(value, di, axis=0), dj, axis=1) for di in (-1, 0, 1) for dj in (-1, 0, 1)
    )
    return jnp.maximum(
        _ICE_ADV2D_MAXIMUM_FLOOR,
        jnp.max(jnp.stack(tuple(neighbors)), axis=0),
    )



def _si3_lane_max9(value: jnp.ndarray) -> jnp.ndarray:
    """SI3's floored surrounding maximum (`icedyn_adv_pra.F90:1519-1571`)."""
    return si3_icemax2d_pra(0, value, nn_hls=0, mask_inactive=False)


def _periodic_halo(value: jnp.ndarray, halo: int) -> jnp.ndarray:
    interior = value[halo:-halo, halo:-halo, ...]
    padding = ((halo, halo), (halo, halo)) + ((0, 0),) * (value.ndim - 2)
    return jnp.pad(interior, padding, mode="wrap")


def apply_si3_prather_source_corrections(
    entry_contents: jnp.ndarray,
    transported_contents: jnp.ndarray,
    *,
    tracer_names: tuple[str, ...],
    nlay_i: int,
    nlay_s: int,
    cell_area_m2: float,
    halo_width: int,
    ponds_active: bool,
    entry_intensive_contents: jnp.ndarray | None = None,
    contents_are_intensive: bool = False,
    negative_area_ramp_width: float = _SI3_NEGATIVE_AREA_RAMP_WIDTH,
    dt_s: float | None = None,
    return_residual_fluxes: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, SI3PratherResidualFluxes]:
    """Apply SI3's active Hbig/Hsnow/zapneg Prather corrections.

    This shared implementation is the single-category, option-4-salinity
    composition in ``icedyn_adv_pra.F90:156-180,405-421,946-1142`` and
    ``icevar.F90:726-839``.  Rung-specific wrappers supply only their resolved
    layer roster, area, and halo.  ``ponds_active`` is the resolved
    ``ln_pnd_LEV`` selector (required, no default): ``False`` (ORCA1,
    ``ln_pnd=.false.``) skips the pond-thickness branch of Hbig_pra
    (``icedyn_adv_pra.F90:1033``); ``ln_pnd_TOPO`` is not ported.

    ``negative_area_ramp_width`` [concentration] replaces zapneg's final hard
    ``WHERE(pa_i < 0) pa_i = 0`` (``icevar.F90:836``) with the C1 one-sided ramp
    :func:`legoesm.core.smooth.positive_part_c1`: exactly 0 for ``a_i <= 0``
    (ice-free cells unchanged), exactly ``a_i`` for ``a_i >= width``, and up to
    ``4 width / 27`` less in between.  It changes only ``a_i`` (no volume, heat
    or salt).  ``0.0`` selects the hard NEMO form (oracle tests).

    ``return_residual_fluxes=True`` also returns the ocean residual ledgers
    the corrections feed (``wfx_res`` [kg m-2 s-1], ``hfx_res`` [W m-2],
    ``sfx_res`` [g m-2 s-1 = salt content * rhoi / dt]) as an
    :class:`SI3PratherResidualFluxes`, accumulated per branch in NEMO 5.0.1
    order: ``Hbig_pra`` (icedyn_adv_pra.F90:1053-1054,1071,1083,1097,1109),
    ``Hsnow_pra`` (:1167-1168), ``ice_var_zapneg`` (icevar.F90 zapneg salt,
    heat and volume branches).  Sign: positive wfx/sfx = mass/salt INTO the
    ocean; hfx = -(removed enthalpy content)/dt (NEMO "W.m-2 <0").  ``dt_s``
    is NEMO's ``pdt = rDt_ice`` (the full ice step, also under subcycling).
    The ledgers are zero outside the inner domain (NEMO adds DO_2D(0,0,0,0)
    only).  Default output (corrected contents only) is unchanged.
    """
    if negative_area_ramp_width < 0.0:
        raise ValueError("negative_area_ramp_width must be >= 0 (0 = hard NEMO form)")

    if return_residual_fluxes and dt_s is None:
        raise ValueError("return_residual_fluxes requires dt_s (NEMO rDt_ice)")
    z1_dt = 0.0 if dt_s is None else 1.0 / dt_s

    # Hbig_pra receives the already-restored intensive pv_i/pv_s/pa_i/... at
    # icedyn_adv_pra.F90:405-408.  Keep the extensive mode for the rung-3.2
    # card's established state contract, but let dynamics cards reproduce the
    # literal NEMO ordering without an extensive->intensive->extensive detour.
    area = 1.0 if contents_are_intensive else cell_area_m2
    index = tracer_names.index
    snow_layers = tuple(f"e_s_l{level:02d}" for level in range(1, nlay_s + 1))
    ice_layers = tuple(f"e_i_l{level:02d}" for level in range(1, nlay_i + 1))
    salt_layers = tuple(f"szv_i_l{level:02d}" for level in range(1, nlay_i + 1))

    def intensive(contents: jnp.ndarray, name: str) -> jnp.ndarray:
        return contents[..., index(name)] / area

    def entry_intensive(name: str) -> jnp.ndarray:
        if entry_intensive_contents is None:
            return intensive(entry_contents, name)
        if entry_intensive_contents.shape != entry_contents.shape:
            raise ValueError("Prather entry intensive/extensive shapes differ")
        # NEMO recovers the transported fields at icedyn_adv_pra.F90:355-381,
        # but its pre-advection maxima were formed directly from the carried
        # intensive fields at :156-180.  Cards whose outer state is intensive
        # pass that exact carry here instead of re-deriving it through area.
        return entry_intensive_contents[..., index(name)]

    def ratio(numerator: jnp.ndarray, denominator: jnp.ndarray, valid: jnp.ndarray) -> jnp.ndarray:
        safe_denominator = jnp.where(valid, denominator, 1.0)
        return jnp.where(valid, numerator / safe_denominator, 0.0)

    entry_v_i = entry_intensive("v_i")
    entry_v_s = entry_intensive("v_s")
    entry_a_i = entry_intensive("a_i")
    entry_a_ip = entry_intensive("a_ip")
    entry_v_ip = entry_intensive("v_ip")
    entry_h_s = ratio(entry_v_s, entry_a_i, entry_a_i > _ICE_ADV2D_SMALL)
    entry_h_ip = ratio(
        entry_v_ip,
        entry_a_ip,
        entry_a_ip > _ICE_ADV2D_MAXIMUM_FLOOR,
    )
    h_s_max = _max9(entry_h_s)
    h_ip_max = _max9(entry_h_ip)

    contents = transported_contents
    v_i = intensive(contents, "v_i")
    wfx_res = jnp.zeros(contents.shape[:2], dtype=contents.dtype)
    hfx_res = jnp.zeros_like(wfx_res)
    sfx_res = jnp.zeros_like(wfx_res)
    v_s = intensive(contents, "v_s")
    a_i = intensive(contents, "a_i")
    a_ip = intensive(contents, "a_ip")
    v_ip = intensive(contents, "v_ip")

    # Hbig_pra outer gate (icedyn_adv_pra.F90:1029): the pond, ice and snow
    # thickness branches all need v_i > 0 and a_i > 0.  The ice-thickness
    # branch (:1042-1045) already ran in advect_si3_prather_2d; it rescales
    # a_i to a positive value only inside this same gate, so the gate is
    # unchanged by it.
    has_ice = (v_i > 0.0) & (a_i > 0.0)

    # Hbig_pra pond concentration (:1033-1038), only under ln_pnd_LEV; the
    # level-pond arm is active even when a cell's pond volume is zero.  With
    # ponds off NEMO never forms zhip_max (:163-165) nor enters this branch.
    if ponds_active:
        h_ip = v_ip / jnp.maximum(_ICE_ADV2D_MAXIMUM_FLOOR, a_ip)
        pond_correct = (
            has_ice & (h_ip > h_ip_max) & (a_ip < _ICE_ADV2D_HBIG_CONCENTRATION)
        )
        corrected_a_ip = ratio(v_ip, h_ip_max, pond_correct)
        contents = contents.at[..., index("a_ip")].set(
            jnp.where(
                pond_correct,
                corrected_a_ip * area,
                contents[..., index("a_ip")],
            )
        )

    # The ice-thickness part of Hbig_pra (:997-1002) is already applied by
    # advect_si3_prather_2d because its two indices are dispatch-mandatory.
    a_i = intensive(contents, "a_i")

    # Hbig_pra snow volume and heat correction (:1047-1058), inside the
    # outer v_i > 0 .AND. a_i > 0 gate of :1029 (``has_ice``).
    h_s = ratio(v_s, a_i, a_i > 0.0)
    snow_correct = (
        has_ice
        & (v_s > 0.0)
        & (h_s > h_s_max)
        & (a_i < _ICE_ADV2D_HBIG_CONCENTRATION)
    )
    snow_fraction = jnp.where(snow_correct, ratio(h_s_max, h_s, snow_correct), 1.0)
    corrected_v_s = a_i * h_s_max
    snow_enthalpy = sum(intensive(contents, name) for name in snow_layers)
    wfx_res = wfx_res + jnp.where(
        snow_correct, (v_s - corrected_v_s) * constants.rho_snow * z1_dt, 0.0
    )
    hfx_res = hfx_res - jnp.where(
        snow_correct, snow_enthalpy * (1.0 - snow_fraction) * z1_dt, 0.0
    )
    contents = contents.at[..., index("v_s")].set(
        jnp.where(
            snow_correct,
            corrected_v_s * area,
            contents[..., index("v_s")],
        )
    )
    v_s = intensive(contents, "v_s")
    for name in snow_layers:
        value = intensive(contents, name) * snow_fraction
        contents = contents.at[..., index(name)].set(
            jnp.where(snow_correct, value * area, contents[..., index(name)])
        )

    # Option-4 layer salt and all layer enthalpy maxima (:1020-1069).
    low_concentration = a_i < _ICE_ADV2D_HBIG_CONCENTRATION
    for name in salt_layers:
        entry_specific = ratio(
            intensive(entry_contents, name), entry_v_i, entry_v_i >= _ICE_ADV2D_SMALL
        )
        value = intensive(contents, name)
        specific = ratio(value, v_i, v_i > 0.0)
        cap = _max9(entry_specific)
        correct = (v_i > 0.0) & (a_i > 0.0) & (specific > cap) & low_concentration
        corrected = ratio(value * cap, specific, correct)
        contents = contents.at[..., index(name)].set(
            jnp.where(correct, corrected * area, contents[..., index(name)])
        )
        sfx_res = sfx_res + jnp.where(
            correct,
            value * (1.0 - ratio(cap, specific, correct)) * constants.rho_ice * z1_dt,
            0.0,
        )
    for name in ice_layers:
        entry_specific = ratio(
            intensive(entry_contents, name), entry_v_i, entry_v_i >= _ICE_ADV2D_SMALL
        )
        value = intensive(contents, name)
        specific = ratio(value, v_i, v_i > 0.0)
        cap = _max9(entry_specific)
        correct = (v_i > 0.0) & (a_i > 0.0) & (specific > cap) & low_concentration
        corrected = ratio(value * cap, specific, correct)
        contents = contents.at[..., index(name)].set(
            jnp.where(correct, corrected * area, contents[..., index(name)])
        )
        hfx_res = hfx_res - jnp.where(
            correct, value * (1.0 - ratio(cap, specific, correct)) * z1_dt, 0.0
        )
    for name in snow_layers:
        entry_specific = ratio(
            intensive(entry_contents, name), entry_v_s, entry_v_s >= _ICE_ADV2D_SMALL
        )
        value = intensive(contents, name)
        specific = ratio(value, v_s, v_s > 0.0)
        cap = _max9(entry_specific)
        correct = (v_s > 0.0) & (a_i > 0.0) & (specific > cap) & low_concentration
        corrected = ratio(value * cap, specific, correct)
        contents = contents.at[..., index(name)].set(
            jnp.where(correct, corrected * area, contents[..., index(name)])
        )
        hfx_res = hfx_res - jnp.where(
            correct, value * (1.0 - ratio(cap, specific, correct)) * z1_dt, 0.0
        )

    # Hsnow_pra (:1115-1133): bound snow loading and pond area.
    snow_capacity = v_i * (constants.rho_ocean_nemo - constants.rho_ice) / constants.rho_snow
    excess = jnp.where(v_i > 0.0, jnp.maximum(0.0, v_s - snow_capacity), 0.0)
    has_excess = excess > 0.0
    fraction = jnp.where(v_s > 0.0, ratio(v_s - excess, v_s, v_s > 0.0), 1.0)
    corrected_v_s = v_s - excess
    snow_enthalpy = sum(intensive(contents, name) for name in snow_layers)
    wfx_res = wfx_res + jnp.where(has_excess, excess * constants.rho_snow * z1_dt, 0.0)
    hfx_res = hfx_res - jnp.where(
        has_excess, snow_enthalpy * (1.0 - fraction) * z1_dt, 0.0
    )
    contents = contents.at[..., index("v_s")].set(
        jnp.where(
            has_excess,
            corrected_v_s * area,
            contents[..., index("v_s")],
        )
    )
    v_s = intensive(contents, "v_s")
    for name in snow_layers:
        value = intensive(contents, name) * fraction
        contents = contents.at[..., index(name)].set(
            jnp.where(has_excess, value * area, contents[..., index(name)])
        )
    old_a_ip = intensive(contents, "a_ip")
    cap_pond = old_a_ip > a_i
    contents = contents.at[..., index("a_ip")].set(
        jnp.where(
            cap_pond,
            contents[..., index("a_i")],
            contents[..., index("a_ip")],
        )
    )

    # ice_var_zapneg (:759-837).  The positive-flow testcase normally makes
    # this a no-op, but retaining it makes the selector composition explicit.
    # `ice_var_zapneg` first updates pa_i (:759-760); every later test sees
    # that updated value, notably the snow-volume removal at :807-816.
    invalid_area = v_i <= 0.0
    a_i = jnp.where(invalid_area, 0.0, a_i)
    invalid_ice = (v_i <= 0.0) | (a_i <= 0.0)
    contents = contents.at[..., index("a_i")].set(
        jnp.where(invalid_area, 0.0, contents[..., index("a_i")])
    )
    for name in salt_layers + ice_layers:
        value = intensive(contents, name)
        invalid = (value < 0.0) | invalid_ice
        if name in salt_layers:
            sfx_res = sfx_res + jnp.where(invalid, value * constants.rho_ice * z1_dt, 0.0)
        else:
            hfx_res = hfx_res - jnp.where(invalid, value * z1_dt, 0.0)
        contents = contents.at[..., index(name)].set(
            jnp.where(invalid, 0.0, contents[..., index(name)])
        )
    for name in snow_layers:
        value = intensive(contents, name)
        invalid = (value < 0.0) | (a_i <= 0.0) | (v_s <= 0.0)
        hfx_res = hfx_res - jnp.where(invalid, value * z1_dt, 0.0)
        contents = contents.at[..., index(name)].set(
            jnp.where(invalid, 0.0, contents[..., index(name)])
        )
    invalid_v_i = (v_i < 0.0) | (a_i <= 0.0)
    invalid_v_s = (v_s < 0.0) | (a_i <= 0.0)
    oa_i = intensive(contents, "oa_i")
    invalid_oa_i = oa_i < 0.0
    a_ip = intensive(contents, "a_ip")
    invalid_a_ip = a_ip < 0.0
    v_ip = intensive(contents, "v_ip")
    v_il = intensive(contents, "v_il")
    invalid_pond = (v_ip < 0.0) | (a_ip <= 0.0)
    # icevar.F90 zapneg volume branch order: v_i, v_s, v_il, v_ip.
    for value, invalid, density in (
        (v_i, invalid_v_i, constants.rho_ice),
        (v_s, invalid_v_s, constants.rho_snow),
        (v_il, invalid_pond | (v_il < 0.0), constants.rho_water),
        (v_ip, invalid_pond, constants.rho_water),
    ):
        wfx_res = wfx_res + jnp.where(invalid, value * density * z1_dt, 0.0)
    for name, invalid in (
        ("v_i", invalid_v_i),
        ("v_s", invalid_v_s),
        ("oa_i", invalid_oa_i),
        ("a_ip", invalid_a_ip),
        ("v_ip", invalid_pond),
        ("v_il", invalid_pond | (v_il < 0.0)),
    ):
        contents = contents.at[..., index(name)].set(
            jnp.where(invalid, 0.0, contents[..., index(name)])
        )
    # icevar.F90:835-837 final WHEREs: oa_i and a_ip were zeroed above; a
    # negative a_i with v_i > 0 survives the :760 reset and is floored here.
    # Width 0 = NEMO's hard WHERE; > 0 = C1 one-sided ramp (default).
    # Cells at or above the width keep their stored value bitwise (no
    # extensive/intensive round trip).
    a_i_stored = contents[..., index("a_i")]
    if negative_area_ramp_width == 0.0:
        floored = jnp.where(a_i_stored < 0.0, 0.0, a_i_stored)
    else:
        a_i_now = a_i_stored / area
        ramp = positive_part_c1(a_i_now, 1.0 / negative_area_ramp_width) * area
        floored = jnp.where(a_i_now < negative_area_ramp_width, ramp, a_i_stored)
    contents = contents.at[..., index("a_i")].set(floored)
    # NEMO corrects physical cells, then lbc_lnk refreshes the bi-periodic
    # halos (`icedyn_adv_pra.F90:405-479`).  Never feed independently corrected
    # halo values into the next split step.
    corrected = _periodic_halo(contents, halo_width)
    if not return_residual_fluxes:
        return corrected
    inner = jnp.zeros(wfx_res.shape, dtype=bool)
    inner = inner.at[halo_width:-halo_width, halo_width:-halo_width].set(True)
    return corrected, SI3PratherResidualFluxes(
        wfx_res=jnp.where(inner, wfx_res, 0.0),
        hfx_res=jnp.where(inner, hfx_res, 0.0),
        sfx_res=jnp.where(inner, sfx_res, 0.0),
    )



def apply_si3_prather_source_corrections_lane(
    entry_contents: jnp.ndarray,
    transported_contents: jnp.ndarray,
    *,
    tracer_names: tuple[str, ...],
    nlay_i: int,
    nlay_s: int,
    cell_area_m2: float,
    halo_width: int,
    entry_intensive_contents: jnp.ndarray | None = None,
    contents_are_intensive: bool = False,
) -> jnp.ndarray:
    """Apply SI3's active Hbig/Hsnow/zapneg Prather corrections.

    Delegates to the shared `si3_icemax2d_pra`, `si3_hbig_pra`, `si3_hsnow_pra`,
    and `si3_var_zapneg` kernels in `legoesm.ice.transport`
    (`icedyn_adv_pra.F90:156-180,405-421,946-1142`).
    """
    area = 1.0 if contents_are_intensive else cell_area_m2
    index = tracer_names.index
    snow_layers = tuple(f"e_s_l{level:02d}" for level in range(1, nlay_s + 1))
    ice_layers = tuple(f"e_i_l{level:02d}" for level in range(1, nlay_i + 1))
    salt_layers = tuple(f"szv_i_l{level:02d}" for level in range(1, nlay_i + 1))

    def intensive(contents: jnp.ndarray, name: str) -> jnp.ndarray:
        return contents[..., index(name)] / area

    def entry_intensive(name: str) -> jnp.ndarray:
        if entry_intensive_contents is None:
            return intensive(entry_contents, name)
        if entry_intensive_contents.shape != entry_contents.shape:
            raise ValueError("Prather entry intensive/extensive shapes differ")
        return entry_intensive_contents[..., index(name)]

    def ratio(numerator: jnp.ndarray, denominator: jnp.ndarray, valid: jnp.ndarray) -> jnp.ndarray:
        safe_denominator = jnp.where(valid, denominator, 1.0)
        return jnp.where(valid, numerator / safe_denominator, 0.0)

    entry_v_i = entry_intensive("v_i")
    entry_v_s = entry_intensive("v_s")
    entry_a_i = entry_intensive("a_i")
    entry_a_ip = entry_intensive("a_ip")
    entry_v_ip = entry_intensive("v_ip")
    entry_h_s = ratio(entry_v_s, entry_a_i, entry_a_i > _ICE_ADV2D_SMALL)
    entry_h_ip = ratio(
        entry_v_ip,
        entry_a_ip,
        entry_a_ip > _ICE_ADV2D_MAXIMUM_FLOOR,
    )
    h_s_max = _si3_lane_max9(entry_h_s)
    h_ip_max = _si3_lane_max9(entry_h_ip)
    pszi_max = jnp.stack(
        [
            _si3_lane_max9(
                ratio(
                    intensive(entry_contents, name),
                    entry_v_i,
                    entry_v_i >= _ICE_ADV2D_SMALL,
                )
            )
            for name in salt_layers
        ],
        axis=2,
    )
    pei_max = jnp.stack(
        [
            _si3_lane_max9(
                ratio(
                    intensive(entry_contents, name),
                    entry_v_i,
                    entry_v_i >= _ICE_ADV2D_SMALL,
                )
            )
            for name in ice_layers
        ],
        axis=2,
    )
    pes_max = jnp.stack(
        [
            _si3_lane_max9(
                ratio(
                    intensive(entry_contents, name),
                    entry_v_s,
                    entry_v_s >= _ICE_ADV2D_SMALL,
                )
            )
            for name in snow_layers
        ],
        axis=2,
    )

    contents = transported_contents
    zero_2d = jnp.zeros(contents.shape[:2], dtype=contents.dtype)
    v_i = contents[..., index("v_i")]
    v_s = contents[..., index("v_s")]
    a_i = contents[..., index("a_i")]
    oa_i = contents[..., index("oa_i")]
    a_ip = contents[..., index("a_ip")]
    v_ip = contents[..., index("v_ip")]
    v_il = contents[..., index("v_il")]
    e_s = jnp.stack([contents[..., index(name)] for name in snow_layers], axis=2)
    e_i = jnp.stack([contents[..., index(name)] for name in ice_layers], axis=2)
    szv_i = jnp.stack([contents[..., index(name)] for name in salt_layers], axis=2)

    (
        v_i,
        v_s,
        a_i,
        a_ip,
        v_ip,
        _,
        e_s,
        e_i,
        szv_i,
        _,
        _,
        _,
    ) = si3_hbig_pra(
        0,
        1.0,
        zero_2d,
        h_s_max,
        h_ip_max,
        zero_2d,
        pes_max,
        pei_max,
        pszi_max,
        v_i,
        v_s,
        a_i,
        a_ip,
        v_ip,
        zero_2d,
        e_s,
        e_i,
        szv_i,
        zero_2d,
        zero_2d,
        zero_2d,
        nn_hls=0,
        nn_icesal=4,
        ln_pnd_LEV=True,
        ln_pnd_TOPO=False,
        hi_max_top=None,
        rhoi=constants.rho_ice,
        rhos=constants.rho_snow,
        apply_hi_cap=False,
        multiply_before_divide=True,
        extensive_area=None if contents_are_intensive else cell_area_m2,
    )
    v_i, v_s, a_i, a_ip, e_s, _, _ = si3_hsnow_pra(
        0,
        1.0,
        v_i,
        v_s,
        a_i,
        a_ip,
        e_s,
        zero_2d,
        zero_2d,
        nn_hls=0,
        rho0=constants.rho_ocean_nemo,
        rhoi=constants.rho_ice,
        rhos=constants.rho_snow,
        use_reciprocal_rhos=False,
        extensive_area=None if contents_are_intensive else cell_area_m2,
    )
    (
        v_i,
        v_s,
        _,
        oa_i,
        a_i,
        a_ip,
        v_ip,
        v_il,
        e_s,
        e_i,
        szv_i,
        _,
        _,
        _,
    ) = si3_var_zapneg(
        0,
        1.0,
        v_i,
        v_s,
        zero_2d,
        oa_i,
        a_i,
        a_ip,
        v_ip,
        v_il,
        e_s,
        e_i,
        szv_i,
        zero_2d,
        zero_2d,
        zero_2d,
        nn_hls=0,
        nn_icesal=4,
        rhoi=constants.rho_ice,
        rhos=constants.rho_snow,
        rhow=constants.rho_water,
    )

    updated_by_name: dict[str, jnp.ndarray] = {
        "v_i": v_i,
        "v_s": v_s,
        "a_i": a_i,
        "oa_i": oa_i,
        "a_ip": a_ip,
        "v_ip": v_ip,
        "v_il": v_il,
        **{name: e_s[:, :, jk] for jk, name in enumerate(snow_layers)},
        **{name: e_i[:, :, jk] for jk, name in enumerate(ice_layers)},
        **{name: szv_i[:, :, jk] for jk, name in enumerate(salt_layers)},
    }
    contents = jnp.stack(
        [
            updated_by_name.get(name, contents[..., idx])
            for idx, name in enumerate(tracer_names)
        ],
        axis=-1,
    )
    return _periodic_halo(contents, halo_width)


def apply_ice_adv2d_source_corrections(
    card: ICEAdv2DCard,
    entry_contents: jnp.ndarray,
    transported_contents: jnp.ndarray,
    *,
    entry_intensive_contents: jnp.ndarray | None = None,
    contents_are_intensive: bool = False,
    _use_legacy_extensive_order: bool = False,
) -> jnp.ndarray:
    """Rung-3.2 wrapper over the shared SI3 Prather correction ledger.

    NEMO recovers intensives before corrections
    (``icedyn_adv_pra.F90:355-381,405-421``).  The old extensive-ledger
    association is retained only by the private one-variable ablation.
    """

    if not contents_are_intensive and not _use_legacy_extensive_order:
        shape = transported_contents.shape[:2]
        cell_area = jnp.full(shape, card.dx_m * card.dy_m, dtype=transported_contents.dtype)
        wet = jnp.ones(shape, dtype=bool)
        entry_intensives = si3_prather_unpack_intensives(entry_contents, cell_area, wet)
        transported_intensives = si3_prather_unpack_intensives(
            transported_contents, cell_area, wet
        )
        corrected = apply_si3_prather_source_corrections(
            entry_intensives,
            transported_intensives,
            tracer_names=ICE_ADV2D_TRACERS,
            nlay_i=card.nlay_i,
            nlay_s=card.nlay_s,
            cell_area_m2=card.dx_m * card.dy_m,
            halo_width=card.halo_width,
            ponds_active=card.ponds,
            entry_intensive_contents=entry_intensives,
            contents_are_intensive=True,
        )
        return si3_prather_pack_intensives(corrected, cell_area)

    return apply_si3_prather_source_corrections(
        entry_contents,
        transported_contents,
        tracer_names=ICE_ADV2D_TRACERS,
        nlay_i=card.nlay_i,
        nlay_s=card.nlay_s,
        cell_area_m2=card.dx_m * card.dy_m,
        halo_width=card.halo_width,
        ponds_active=card.ponds,
        entry_intensive_contents=entry_intensive_contents,
        contents_are_intensive=contents_are_intensive,
    )


def step_ice_adv2d_card(
    card: ICEAdv2DCard,
    state: ICEAdv2DState | None = None,
    *,
    completed_steps: int,
    _use_legacy_extensive_correction_order: bool = False,
) -> ICEAdv2DState:
    state = card.initial_state if state is None else state
    area = card.dx_m * card.dy_m
    cell_area = jnp.full(state.u_ice.shape, area, dtype=jnp.float64)
    wet = jnp.ones(state.u_ice.shape, dtype=bool)
    contents, moments, _ = advect_si3_prather_2d(
        state.contents,
        state.moments,
        card.prescribed_u_ice,
        card.prescribed_v_ice,
        cell_area,
        wet,
        card.dt_s,
        dx=card.dx_m,
        dy=card.dy_m,
        ice_step_index=completed_steps + 1,
        nn_fsbc=card.nn_fsbc,
        halo_width=card.halo_width,
        ice_volume_index=ICE_ADV2D_TRACERS.index("v_i"),
        concentration_index=ICE_ADV2D_TRACERS.index("a_i"),
        subcycles=card.subcycles,
    )
    if _use_legacy_extensive_correction_order:
        contents = apply_ice_adv2d_source_corrections(
            card,
            state.contents,
            contents,
            _use_legacy_extensive_order=True,
        )
        return apply_ice_adv2d_zapsmall(card, state, contents, moments)

    # Default: reproduce NEMO's single extensive-to-intensive recovery before
    # every post-advection correction (`icedyn_adv_pra.F90:355-421`).
    intensives = si3_prather_unpack_intensives(contents, cell_area, wet)
    entry_intensives = si3_prather_unpack_intensives(state.contents, cell_area, wet)
    intensives = apply_ice_adv2d_source_corrections(
        card,
        entry_intensives,
        intensives,
        entry_intensive_contents=entry_intensives,
        contents_are_intensive=True,
    )
    corrected = apply_ice_adv2d_zapsmall(
        card,
        state,
        intensives,
        moments,
        contents_are_intensive=True,
    )
    return corrected._replace(
        contents=si3_prather_pack_intensives(corrected.contents, cell_area)
    )


def save_ice_adv2d_restart(
    path: Path, card: ICEAdv2DCard, state: ICEAdv2DState, *, completed_steps: int
) -> None:
    validate_ice_adv2d_card(card, state)
    if not 0 <= completed_steps <= card.n_steps:
        raise ValueError("ICE_ADV2D completed_steps out of range")
    payload: dict[str, np.ndarray] = {
        "format": np.asarray(ICE_ADV2D_RESTART_FORMAT),
        "completed_steps": np.asarray(completed_steps, dtype=np.int64),
        "card_contract_sha256": np.asarray(ice_adv2d_card_contract_sha256(card)),
        "contents": np.asarray(state.contents),
        "bulk_salt_diagnostic": np.asarray(state.bulk_salt_diagnostic),
        "u_ice": np.asarray(state.u_ice),
        "v_ice": np.asarray(state.v_ice),
        "t_surface": np.asarray(state.t_surface),
    }
    payload.update(
        {f"moment_{index}": np.asarray(value) for index, value in enumerate(state.moments)}
    )
    np.savez(path, **payload)  # type: ignore[arg-type]


def load_ice_adv2d_restart(path: Path, card: ICEAdv2DCard) -> tuple[ICEAdv2DState, int]:
    validate_ice_adv2d_card(card)
    array_names = {"contents", "bulk_salt_diagnostic", "u_ice", "v_ice", "t_surface"} | {
        f"moment_{index}" for index in range(5)
    }
    metadata = {"format", "completed_steps", "card_contract_sha256"}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != array_names | metadata:
            missing = sorted((array_names | metadata) - set(archive.files))
            extra = sorted(set(archive.files) - (array_names | metadata))
            raise ValueError(f"ICE_ADV2D restart keys missing={missing}, extra={extra}")
        if archive["format"].item() != ICE_ADV2D_RESTART_FORMAT:
            raise ValueError("ICE_ADV2D restart format mismatch")
        if archive["card_contract_sha256"].item() != ice_adv2d_card_contract_sha256(card):
            raise ValueError("ICE_ADV2D restart card contract mismatch")
        if archive["completed_steps"].dtype.kind not in "iu":
            raise ValueError("ICE_ADV2D restart clock is not integer")
        completed_steps = int(archive["completed_steps"].item())
        arrays = {name: archive[name].copy() for name in array_names}
    if not 0 <= completed_steps <= card.n_steps:
        raise ValueError("ICE_ADV2D restart clock out of range")
    template = card.initial_state
    for name in ("contents", "bulk_salt_diagnostic", "u_ice", "v_ice", "t_surface"):
        reference = np.asarray(getattr(template, name))
        if arrays[name].shape != reference.shape or arrays[name].dtype != reference.dtype:
            raise ValueError(f"ICE_ADV2D restart {name} shape/dtype mismatch")
    for index, reference in enumerate(template.moments):
        value = arrays[f"moment_{index}"]
        reference_array = np.asarray(reference)
        if value.shape != reference_array.shape or value.dtype != reference_array.dtype:
            raise ValueError(f"ICE_ADV2D restart moment_{index} shape/dtype mismatch")
    state = ICEAdv2DState(
        contents=jnp.asarray(arrays["contents"]),
        bulk_salt_diagnostic=jnp.asarray(arrays["bulk_salt_diagnostic"]),
        moments=cast(
            SI3PratherMoments,
            tuple(jnp.asarray(arrays[f"moment_{index}"]) for index in range(5)),
        ),
        u_ice=jnp.asarray(arrays["u_ice"]),
        v_ice=jnp.asarray(arrays["v_ice"]),
        t_surface=jnp.asarray(arrays["t_surface"]),
    )
    validate_ice_adv2d_card(card, state)
    return state, completed_steps


__all__ = (
    "ICE_ADV2D_RESTART_FORMAT",
    "ICE_ADV2D_TRACERS",
    "ICEAdv2DCard",
    "ICEAdv2DState",
    "build_ice_adv2d_card",
    "apply_ice_adv2d_source_corrections",
    "apply_si3_prather_source_corrections",
    "apply_si3_prather_source_corrections_lane",
    "ice_adv2d_card_contract_sha256",
    "load_ice_adv2d_restart",
    "save_ice_adv2d_restart",
    "step_ice_adv2d_card",
    "validate_ice_adv2d_card",
)
