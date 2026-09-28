"""Pure-config legoESM card for NEMO/SI3 ``ICE_ADV1D``.

The card contains no alternate ice model.  It selects the opt-in SI3 Prather
program in :mod:`legoesm.ice.transport` and reuses the canonical Cartesian
C-grid geometry.  Its host-side initial-condition arithmetic mirrors the
float32 NetCDF input shipped by NEMO before promoting the live state to fp64.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import NamedTuple, cast

import jax.numpy as jnp
import numpy as np
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.grids.latlon import create_beta_plane_cgrid_geometry
from legoesm.ice.transport import (
    SI3PratherMoments,
    advect_si3_prather_1d,
    zero_si3_prather_moments,
)

from legoesm import constants

ICE_ADV1D_TRACERS = (
    "v_i",
    "v_s",
    "a_i",
    "oa_i",
    "sv_i",
    "e_s_l01",
    "e_s_l02",
    "e_s_l03",
    "e_i_l01",
    "e_i_l02",
    "e_i_l03",
)
ICE_ADV1D_RESTART_FORMAT = "legoesm-ice-adv1d-state-v1"

# Shipped ICE_ADV1D/ORCA1 card values.  The profile is intentionally a fixed
# oracle testcase transcription, not a tunable production parameterization.
_ICE_ADV1D_GRID_SIZE = 59  # usrdef_nam.F90:73-75 with namelist_cfg:19-20
_ICE_ADV1D_SPACING_M = 4.0  # ICE_ADV1D/EXPREF/namelist_cfg:19-20
_ICE_ADV1D_ORIGIN_M = -118.0  # usrdef_hgr.F90:76-89; phase-1 mesh gate
_ICE_ADV1D_HALO_WIDTH = 2  # built cpp/domain halo; phase-1 frame registry
_ICE_ADV1D_N_STEPS = 40  # ICE_ADV1D/EXPREF/namelist_cfg:29-30
_ICE_ADV1D_DT_S = 2.0  # ICE_ADV1D/EXPREF/namelist_cfg:37
_ICE_ADV1D_JPL = 1  # ORCA1 namelist_ice_cfg:24
_ICE_ADV1D_NLAY_I = 3  # ORCA1 namelist_ice_cfg:25
_ICE_ADV1D_NLAY_S = 3  # ORCA1 namelist_ice_cfg:26
_ICE_ADV1D_NN_ICESAL = 2  # ORCA1 namelist_ice_cfg:108
_ICE_ADV1D_U_PROFILE_FACTOR = 1.5  # icedyn.F90:149-153
_ICE_ADV1D_PRATHER_SUBCYCLES = 2  # icedyn_adv_pra.F90:119-131 for this card CFL
_ICE_ADV1D_BASE_THICKNESS_M = 1.0  # make_initice.py:97
_ICE_ADV1D_NOTCH_THICKNESS_M = 0.2  # make_initice.py:101-102
_ICE_ADV1D_NOTCH_START = 15  # make_initice.py:101
_ICE_ADV1D_NOTCH_STOP = 44  # make_initice.py:101 (inclusive 43)
_ICE_ADV1D_BACKGROUND_CONCENTRATION = 0.001  # make_initice.py:106
_ICE_ADV1D_MAX_CONCENTRATION = 0.9  # make_initice.py:110-113
_ICE_ADV1D_RAMP_START = 10  # make_initice.py:110
_ICE_ADV1D_RAMP_PEAK = 29  # make_initice.py:110-112
_ICE_ADV1D_RAMP_STOP = 48  # make_initice.py:112
_ICE_ADV1D_RAMP_OFFSET = 9.0  # make_initice.py:111
_ICE_ADV1D_RAMP_WIDTH = 20.0  # make_initice.py:111
_ICE_ADV1D_INITIAL_TEMPERATURE_K = 270.0  # make_initice.py:56-57,90-91
_ICE_ADV1D_INITIAL_SALINITY_G_KG = 0.1  # ORCA1 namelist_ice_cfg:115


class ICEAdv1DState(NamedTuple):
    """Prognostic and prescribed-at-step-boundary state for ICE_ADV1D."""

    contents: jnp.ndarray
    moments: SI3PratherMoments
    u_ice: jnp.ndarray
    v_ice: jnp.ndarray
    t_surface: jnp.ndarray


class ICEAdv1DCard(NamedTuple):
    """Immutable selectors and initial state for the certified 1-D rung."""

    case: str
    transport_scheme: str
    grid: object
    wet_global_xy: jnp.ndarray
    prescribed_u_ice: jnp.ndarray
    empty_surface_temperature: jnp.ndarray
    initial_state: ICEAdv1DState
    dt_s: float
    dx_m: float
    dy_m: float
    halo_width: int
    n_steps: int
    jpl: int
    nlay_i: int
    nlay_s: int
    nn_icesal: int
    thermodynamics: bool
    ponds: bool
    landfast: bool


def ice_adv1d_card_contract_sha256(card: ICEAdv1DCard) -> str:
    """Fingerprint every selector/array that can change card continuation."""

    selectors = {
        "case": card.case,
        "transport_scheme": card.transport_scheme,
        "dt_s": card.dt_s,
        "dx_m": card.dx_m,
        "dy_m": card.dy_m,
        "halo_width": card.halo_width,
        "n_steps": card.n_steps,
        "jpl": card.jpl,
        "nlay_i": card.nlay_i,
        "nlay_s": card.nlay_s,
        "nn_icesal": card.nn_icesal,
        "thermodynamics": card.thermodynamics,
        "ponds": card.ponds,
        "landfast": card.landfast,
    }
    digest = hashlib.sha256(
        json.dumps(selectors, sort_keys=True, separators=(",", ":")).encode()
    )
    for name in (
        "wet_global_xy",
        "prescribed_u_ice",
        "empty_surface_temperature",
    ):
        value = np.ascontiguousarray(np.asarray(getattr(card, name)))
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _closed_box_mask_xy(size: int) -> np.ndarray:
    wet: np.ndarray = np.ones((size, size), dtype=bool)
    wet[(0, -1), :] = False
    wet[:, (0, -1)] = False
    return wet


def _initial_fields_xy(size: int, wet: np.ndarray) -> dict[str, np.ndarray]:
    """Transcribe ``ICE_ADV1D/EXPREF/make_initice.py:85-113``."""

    # The shipped NetCDF variables are `f` (float32).  Build in that storage
    # type first, exactly as the oracle input file, then promote to fp64.
    h: np.ndarray = np.full(
        (size, size), np.float32(_ICE_ADV1D_BASE_THICKNESS_M), dtype=np.float32
    )
    h[_ICE_ADV1D_NOTCH_START:_ICE_ADV1D_NOTCH_STOP, :] = np.float32(
        _ICE_ADV1D_NOTCH_THICKNESS_M
    )
    a: np.ndarray = np.full(
        (size, size),
        np.float32(_ICE_ADV1D_BACKGROUND_CONCENTRATION),
        dtype=np.float32,
    )
    for x in range(size):
        if _ICE_ADV1D_RAMP_START <= x <= _ICE_ADV1D_RAMP_PEAK:
            a[x, :] = np.float32(
                _ICE_ADV1D_MAX_CONCENTRATION
                * (x - _ICE_ADV1D_RAMP_OFFSET)
                / _ICE_ADV1D_RAMP_WIDTH
            )
        elif _ICE_ADV1D_RAMP_PEAK < x <= _ICE_ADV1D_RAMP_STOP:
            a[x, :] = np.float32(_ICE_ADV1D_MAX_CONCENTRATION)
    h = h.astype(np.float64) * wet
    a = a.astype(np.float64) * wet
    v_i = h * a
    v_s = np.zeros_like(v_i)
    salinity = np.float64(_ICE_ADV1D_INITIAL_SALINITY_G_KG)
    sv_i = salinity * v_i

    # iceistate.F90:357-367 with t_i=270 K and a uniform 0.1 g/kg layer
    # salinity.  Every constant is the named NEMO-parity constant from the
    # canonical constants module; no card-local physical literal is used.
    melt_temperature = constants.T_freeze - constants.mu_ice_freeze * salinity
    temperature = np.float64(_ICE_ADV1D_INITIAL_TEMPERATURE_K)
    phase_fraction = max(
        0.0,
        1.0
        - (melt_temperature - constants.T_freeze)
        / min(temperature - constants.T_freeze, -1.0e-10),
    )
    ice_energy_per_layer_volume = (
        constants.rho_ice
        / _ICE_ADV1D_NLAY_I
        * (
            constants.c_p_ice_nemo * (melt_temperature - temperature)
            + constants.L_fus_nemo * phase_fraction
            - constants.c_p_seawater
            * (melt_temperature - constants.T_freeze)
        )
    )
    e_i = v_i * ice_energy_per_layer_volume
    zero = np.zeros_like(v_i)
    return {
        "v_i": v_i,
        "v_s": v_s,
        "a_i": a,
        "oa_i": zero,
        "sv_i": sv_i,
        "e_s_l01": zero,
        "e_s_l02": zero,
        "e_s_l03": zero,
        "e_i_l01": e_i,
        "e_i_l02": e_i,
        "e_i_l03": e_i,
    }


def build_ice_adv1d_card(ocean_surface_temperature_c: np.ndarray) -> ICEAdv1DCard:
    """Build the sole phase-2 card, rejecting any implicit selector choice."""

    set_policy(PrecisionPolicy.fp64())
    if get_policy() != PrecisionPolicy.fp64():
        raise RuntimeError("ICE_ADV1D card requires PrecisionPolicy.fp64()")
    size = _ICE_ADV1D_GRID_SIZE
    spacing = _ICE_ADV1D_SPACING_M
    halo = _ICE_ADV1D_HALO_WIDTH
    grid = create_beta_plane_cgrid_geometry(
        size,
        size,
        dx_m=spacing,
        dy_m=spacing,
        f0=0.0,
        beta=0.0,
        x_origin_m=_ICE_ADV1D_ORIGIN_M,
        y_origin_m=_ICE_ADV1D_ORIGIN_M,
        cartesian_pseudo_lat=False,
        dtype=jnp.float64,
    )
    wet_global = _closed_box_mask_xy(size)
    ocean_surface_temperature_c = np.asarray(
        ocean_surface_temperature_c, dtype=np.float64
    )
    if ocean_surface_temperature_c.shape != (size, size):
        raise ValueError(
            "ICE_ADV1D ocean surface temperature must have shape (59, 59)"
        )
    fields = _initial_fields_xy(size, wet_global)
    contents_global = np.stack(
        [fields[name] * spacing * spacing for name in ICE_ADV1D_TRACERS], axis=-1
    )
    pad = ((halo, halo), (halo, halo), (0, 0))
    contents = jnp.asarray(np.pad(contents_global, pad), dtype=jnp.float64)
    # icedyn.F90:149-153 evaluates the profile over `jpiglo`, which includes
    # the configured two-cell halos in this serial case (63, not mesh x=59).
    # Build it on that native frame before applying the U mask.
    frame_size = size + 2 * halo
    frame_i: np.ndarray = np.arange(1, frame_size + 1, dtype=np.float64)
    coefficient = (
        (0.5 * (frame_size + 1) - frame_i)
        / (0.5 * (frame_size + 1) - 1.0)
    )
    u_mask = wet_global & np.roll(wet_global, -1, axis=0)
    u_mask[-1, :] = False
    u_mask_full = np.pad(u_mask, halo, constant_values=False)
    prescribed_u_ice = jnp.asarray(
        _ICE_ADV1D_U_PROFILE_FACTOR * coefficient[:, None] * u_mask_full,
        dtype=jnp.float64,
    )
    u_ice = jnp.zeros_like(prescribed_u_ice)
    v_ice = jnp.zeros_like(prescribed_u_ice)
    t_surface_global = np.where(
        wet_global, np.float64(_ICE_ADV1D_INITIAL_TEMPERATURE_K), 0.0
    )
    t_surface = jnp.asarray(np.pad(t_surface_global, halo), dtype=jnp.float64)
    empty_surface_temperature = jnp.asarray(
        np.pad(ocean_surface_temperature_c + constants.T_freeze, halo, mode="edge"),
        dtype=jnp.float64,
    )
    return ICEAdv1DCard(
        case="ICE_ADV1D_OMIP_L3",
        transport_scheme="si3_prather",
        grid=grid,
        wet_global_xy=jnp.asarray(wet_global),
        prescribed_u_ice=prescribed_u_ice,
        empty_surface_temperature=empty_surface_temperature,
        initial_state=ICEAdv1DState(
            contents=contents,
            moments=zero_si3_prather_moments(contents),
            u_ice=u_ice,
            v_ice=v_ice,
            t_surface=t_surface,
        ),
        dt_s=_ICE_ADV1D_DT_S,
        dx_m=spacing,
        dy_m=spacing,
        halo_width=halo,
        n_steps=_ICE_ADV1D_N_STEPS,
        jpl=_ICE_ADV1D_JPL,
        nlay_i=_ICE_ADV1D_NLAY_I,
        nlay_s=_ICE_ADV1D_NLAY_S,
        nn_icesal=_ICE_ADV1D_NN_ICESAL,
        thermodynamics=False,
        ponds=False,
        landfast=False,
    )


def validate_ice_adv1d_card(
    card: ICEAdv1DCard, state: ICEAdv1DState | None = None
) -> None:
    expected = (
        "ICE_ADV1D_OMIP_L3",
        "si3_prather",
        _ICE_ADV1D_JPL,
        _ICE_ADV1D_NLAY_I,
        _ICE_ADV1D_NLAY_S,
        _ICE_ADV1D_NN_ICESAL,
        _ICE_ADV1D_DT_S,
        _ICE_ADV1D_SPACING_M,
        _ICE_ADV1D_SPACING_M,
        _ICE_ADV1D_HALO_WIDTH,
        _ICE_ADV1D_N_STEPS,
        False,
        False,
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
        card.thermodynamics,
        card.ponds,
        card.landfast,
    )
    if actual != expected:
        raise ValueError(
            f"ICE_ADV1D selector composition {actual!r} != certified {expected!r}"
        )
    state = card.initial_state if state is None else state
    if bool(np.any(np.asarray(state.v_ice) != 0.0)):
        raise ValueError("ICE_ADV1D requires v_ice == 0 exactly")


def apply_ice_adv1d_zapsmall(
    card: ICEAdv1DCard,
    state: ICEAdv1DState,
    contents: jnp.ndarray,
    moments: SI3PratherMoments,
) -> ICEAdv1DState:
    """Apply the executed option-2 arm of ``icevar.F90:611-708``."""

    # The packed fields are extensive, so the
    # common area factor cancels from h=v/a and only affects the threshold's
    # v/a values through the conversion below.  Moments are deliberately not
    # zeroed: SI3 leaves them as restart state until the next limiter call.
    area = card.dx_m * card.dy_m
    v_i = contents[..., ICE_ADV1D_TRACERS.index("v_i")] / area
    a_i = contents[..., ICE_ADV1D_TRACERS.index("a_i")] / area
    has_ice = a_i > 1.0e-10
    safe_a_i = jnp.where(has_ice, a_i, 1.0)
    h_i = jnp.where(has_ice, v_i / safe_a_i, 0.0)
    small = jnp.minimum(jnp.minimum(a_i, v_i), h_i) < 1.0e-10
    contents = jnp.where(small[..., None], 0.0, contents)
    t_surface = jnp.where(
        small, card.empty_surface_temperature, state.t_surface
    )
    return state._replace(
        contents=contents,
        moments=moments,
        u_ice=card.prescribed_u_ice,
        t_surface=t_surface,
    )


def step_ice_adv1d_card(
    card: ICEAdv1DCard, state: ICEAdv1DState | None = None
) -> ICEAdv1DState:
    """Advance one pure-JAX SI3 advection step and its rung-local cleanup."""

    state = card.initial_state if state is None else state
    halo = card.halo_width
    wet_full = jnp.pad(card.wet_global_xy, halo)
    cell_area = jnp.full(state.u_ice.shape, card.dx_m * card.dy_m, jnp.float64)
    contents, moments, _ = advect_si3_prather_1d(
        state.contents,
        state.moments,
        card.prescribed_u_ice,
        state.v_ice,
        cell_area,
        wet_full,
        card.dt_s,
        dx=card.dx_m,
        dy=card.dy_m,
        halo_width=halo,
        ice_volume_index=ICE_ADV1D_TRACERS.index("v_i"),
        concentration_index=ICE_ADV1D_TRACERS.index("a_i"),
        subcycles=_ICE_ADV1D_PRATHER_SUBCYCLES,
    )
    return apply_ice_adv1d_zapsmall(card, state, contents, moments)


def save_ice_adv1d_restart(
    path: Path, card: ICEAdv1DCard, state: ICEAdv1DState, *, completed_steps: int
) -> None:
    """Persist the complete opt-in card state and selector/clock contract."""

    validate_ice_adv1d_card(card, state)
    if not 0 <= completed_steps <= card.n_steps:
        raise ValueError(f"ICE_ADV1D completed_steps {completed_steps} is out of range")
    payload: dict[str, np.ndarray] = {
        "format": np.asarray(ICE_ADV1D_RESTART_FORMAT),
        "case": np.asarray(card.case),
        "transport_scheme": np.asarray(card.transport_scheme),
        "completed_steps": np.asarray(completed_steps, dtype=np.int64),
        "jpl": np.asarray(card.jpl, dtype=np.int64),
        "nlay_i": np.asarray(card.nlay_i, dtype=np.int64),
        "nlay_s": np.asarray(card.nlay_s, dtype=np.int64),
        "nn_icesal": np.asarray(card.nn_icesal, dtype=np.int64),
        "card_contract_sha256": np.asarray(ice_adv1d_card_contract_sha256(card)),
        "contents": np.asarray(state.contents),
        "u_ice": np.asarray(state.u_ice),
        "v_ice": np.asarray(state.v_ice),
        "t_surface": np.asarray(state.t_surface),
    }
    payload.update(
        {f"moment_{index}": np.asarray(value) for index, value in enumerate(state.moments)}
    )
    # NumPy's stub treats arbitrary named arrays as the reserved boolean
    # `allow_pickle` keyword; runtime `savez` preserves these exact names.
    np.savez(path, **payload)  # type: ignore[arg-type]


def load_ice_adv1d_restart(
    path: Path, card: ICEAdv1DCard
) -> tuple[ICEAdv1DState, int]:
    """Load a complete card restart, refusing missing leaves/selectors."""

    validate_ice_adv1d_card(card)
    array_names = {"contents", "u_ice", "v_ice", "t_surface"} | {
        f"moment_{index}" for index in range(5)
    }
    metadata = {
        "format",
        "case",
        "transport_scheme",
        "completed_steps",
        "jpl",
        "nlay_i",
        "nlay_s",
        "nn_icesal",
        "card_contract_sha256",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = (array_names | metadata) - set(archive.files)
        extra = set(archive.files) - (array_names | metadata)
        if missing:
            raise ValueError(f"ICE_ADV1D restart missing {sorted(missing)}")
        if extra:
            raise ValueError(f"ICE_ADV1D restart has extra {sorted(extra)}")
        expected = {
            "format": ICE_ADV1D_RESTART_FORMAT,
            "case": card.case,
            "transport_scheme": card.transport_scheme,
            "jpl": card.jpl,
            "nlay_i": card.nlay_i,
            "nlay_s": card.nlay_s,
            "nn_icesal": card.nn_icesal,
            "card_contract_sha256": ice_adv1d_card_contract_sha256(card),
        }
        for name, value in expected.items():
            if archive[name].item() != value:
                raise ValueError(
                    f"ICE_ADV1D restart {name}={archive[name].item()!r} != {value!r}"
                )
        arrays = {name: archive[name].copy() for name in array_names}
        if archive["completed_steps"].dtype.kind not in "iu":
            raise ValueError("ICE_ADV1D restart completed_steps is not an integer")
        completed_steps = int(archive["completed_steps"].item())
        if not 0 <= completed_steps <= card.n_steps:
            raise ValueError(
                f"ICE_ADV1D restart completed_steps {completed_steps} is out of range"
            )
    template = card.initial_state
    for name in ("contents", "u_ice", "v_ice", "t_surface"):
        value = arrays[name]
        reference = np.asarray(getattr(template, name))
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f"ICE_ADV1D restart {name} shape/dtype mismatch")
    for index, reference in enumerate(template.moments):
        value = arrays[f"moment_{index}"]
        reference_array = np.asarray(reference)
        if value.shape != reference_array.shape or value.dtype != reference_array.dtype:
            raise ValueError(
                f"ICE_ADV1D restart moment_{index} shape/dtype mismatch"
            )
    moments = cast(
        SI3PratherMoments,
        tuple(jnp.asarray(arrays[f"moment_{index}"]) for index in range(5)),
    )
    state = ICEAdv1DState(
        contents=jnp.asarray(arrays["contents"]),
        moments=moments,
        u_ice=jnp.asarray(arrays["u_ice"]),
        v_ice=jnp.asarray(arrays["v_ice"]),
        t_surface=jnp.asarray(arrays["t_surface"]),
    )
    validate_ice_adv1d_card(card, state)
    return state, completed_steps


__all__ = (
    "ICE_ADV1D_TRACERS",
    "ICE_ADV1D_RESTART_FORMAT",
    "ICEAdv1DCard",
    "ICEAdv1DState",
    "apply_ice_adv1d_zapsmall",
    "build_ice_adv1d_card",
    "ice_adv1d_card_contract_sha256",
    "load_ice_adv1d_restart",
    "save_ice_adv1d_restart",
    "step_ice_adv1d_card",
    "validate_ice_adv1d_card",
)
