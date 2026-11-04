"""SI3 ``ice_cor(kt, kn=1)``: out-of-bounds corrections after the ice dynamics.

Port of NEMO 5.0.1 ``src/ICE/icecor.F90:64-116`` for one ice category
(``jpl = 1``, so ``ice_itd_reb`` is never called) together with the
``ice_var_zapsmall`` it calls (``src/ICE/icevar.F90``), in NEMO statement order:

1. thin ice: ``h_i = v_i / a_i`` (``a_i >= epsi20``); where ``h_i < rn_himin``
   the concentration (and the pond area when level/topographic ponds are on) is
   rescaled by ``h_i / rn_himin`` so the thickness never falls below
   ``rn_himin``;
2. concentration cap: where ``at_i > rn_amax`` the area is rescaled to
   ``rn_amax``;
3. salinity bounds (``nn_icesal = 2`` bulk ``sv_i`` or ``nn_icesal = 4`` layer
   ``szv_i``) with the associated salt flux booked into ``sfx_res``;
4. ``ice_var_zapsmall``: every cell with ``min(a_i, v_i, h_i) < epsi10`` is
   emptied; its salt, enthalpy and water go to the ocean through ``sfx_res``,
   ``hfx_res`` and ``wfx_res``; ``t_su = sst_m + rt0``, ``ato_i = 1`` where
   the cell becomes ice-free.

Residual-flux convention (NEMO): ``sfx_res`` [kg m-2 s-1 of salt, positive
into the ocean], ``wfx_res`` [kg m-2 s-1 of water, positive into the ocean],
``hfx_res`` [W m-2, positive into the ocean; zapping ice of negative enthalpy
gives a negative flux].  The routine ADDS to the ledgers it receives, exactly as
NEMO accumulates them; the per-call increment is ``out - in``.

Limits.  ``limit_sharpness_scale = math.inf`` is the literal NEMO
``WHERE``/``MIN``/``MAX``.  A finite value replaces every limit (thin-ice
rescale, concentration cap, salinity clamp, zap) with the ONE-SIDED C1 ramp
:func:`legoesm.core.smooth.positive_part_c1` (``0`` for ``x <= 0``, ``x`` for
``x >= w``, ``x^2 (2w - x)/w^2`` between), so the result equals NEMO exactly
outside a band of width ``w = limit_scale / limit_sharpness_scale`` on one side of
each threshold: ``rn_himin`` for the thickness,
``max(1 - rn_amax, epsi10)`` for the cap (the open-water margin, so
concentrations below the cap are not biased; NEMO itself caps ``rn_amax`` at
``1 - epsi10``), ``rn_simin`` / ``rn_sinew * sss_m`` for the two salinity
bounds and ``epsi10`` for the zap.  Inside a band the shortfall is at most
``4/27 x w``.  The zap becomes a share ``z`` in ``min(a_i, v_i, h_i)`` (1 below
``epsi10 - w``, 0 at or above ``epsi10``): a fraction ``z`` of every content goes
to the ocean ledgers and ``1 - z`` stays, so water, salt and heat close exactly
in both modes.  The ``a_i >= epsi20`` / ``> epsi10`` divide guards of the hard
thickness are NEMO's own.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp
from legoesm.core.smooth import positive_part_c1, smoothstep_cubic
from legoesm.core.source_rounding import nemo_source_round as _sr

# NEMO ice.F90 / icevar.F90 numerical floors (par_ice epsi10/epsi20).
_SI3_ICE_COR_EPSI10 = 1.0e-10
_SI3_ICE_COR_EPSI20 = 1.0e-20
_SI3_ICE_COR_ZERO = 0.0
_SI3_ICE_COR_ONE = 1.0
# Smooth-limit width = (limit scale) / this; math.inf = literal NEMO hard limits.
SI3_ICE_COR_DEFAULT_LIMIT_SHARPNESS_SCALE = 100.0
_SI3_ICE_COR_SALINITY_OPTIONS = (2, 4)


class SI3IceCorConfig(NamedTuple):
    """Resolved namelist / physical constants of one run (all required).

    ``nn_icesal`` 2 = bulk salt content ``sv_i`` (state ``salt_content`` shape
    ``(...)``), 4 = layer salt content ``szv_i`` (shape ``(..., nlay_i)``).
    ``ponds`` = ``ln_pnd_LEV .OR. ln_pnd_TOPO``.
    """

    rn_himin_m: float
    rn_simin_g_kg: float
    rn_sinew: float
    nn_icesal: int
    ponds: bool
    rho_ice_kg_m3: float
    rho_snow_kg_m3: float
    rho_water_kg_m3: float
    rt0_k: float
    dt_s: float
    limit_sharpness_scale: float = SI3_ICE_COR_DEFAULT_LIMIT_SHARPNESS_SCALE


class SI3IceCorState(NamedTuple):
    """One-category (``jpl = 1``) SI3 state, category axis squeezed.

    Layer arrays carry the layer axis last.  Optional fields (``None`` = not
    carried) are reset on zapped cells exactly as NEMO resets them.
    """

    ice_area: jnp.ndarray
    ice_volume: jnp.ndarray
    snow_volume: jnp.ndarray
    salt_content: jnp.ndarray
    age_content: jnp.ndarray
    pond_area: jnp.ndarray
    pond_volume: jnp.ndarray
    pond_lid_volume: jnp.ndarray
    snow_enthalpy: jnp.ndarray
    ice_enthalpy: jnp.ndarray
    surface_temperature: jnp.ndarray
    open_water_area: jnp.ndarray | None = None
    snow_thickness: jnp.ndarray | None = None
    snow_temperature: jnp.ndarray | None = None
    ice_temperature: jnp.ndarray | None = None
    ice_layer_salinity: jnp.ndarray | None = None


class SI3IceCorFluxes(NamedTuple):
    """NEMO residual ice-ocean ledgers (see module docstring for units/signs)."""

    sfx_res: jnp.ndarray
    wfx_res: jnp.ndarray
    hfx_res: jnp.ndarray


class SI3IceCorOutput(NamedTuple):
    state: SI3IceCorState
    fluxes: SI3IceCorFluxes
    ice_thickness: jnp.ndarray  # h_i after ice_var_zapsmall
    total_ice_area: jnp.ndarray  # at_i
    total_ice_volume: jnp.ndarray  # vt_i
    zap_fraction: jnp.ndarray  # share of the cell emptied by ice_var_zapsmall (hard: 0 or 1)


def _validate(config: SI3IceCorConfig, state: SI3IceCorState) -> None:
    if config.nn_icesal not in _SI3_ICE_COR_SALINITY_OPTIONS:
        raise ValueError(
            f"SI3 ice_cor nn_icesal={config.nn_icesal!r} not ported "
            f"(supported: {_SI3_ICE_COR_SALINITY_OPTIONS})"
        )
    if not config.limit_sharpness_scale > _SI3_ICE_COR_ZERO:
        raise ValueError("SI3 ice_cor limit_sharpness_scale must be > 0 (math.inf = hard)")
    if not config.dt_s > _SI3_ICE_COR_ZERO:
        raise ValueError("SI3 ice_cor dt_s must be > 0")
    layered = config.nn_icesal == 4
    salt_ndim = state.salt_content.ndim
    if layered and salt_ndim != state.ice_enthalpy.ndim:
        raise ValueError("nn_icesal=4 needs salt_content with the ice layer axis last")
    if not layered and salt_ndim != state.ice_area.ndim:
        raise ValueError("nn_icesal=2 needs bulk salt_content shaped like ice_area")


def apply_si3_jpl1_ice_cor(
    state: SI3IceCorState,
    fluxes: SI3IceCorFluxes,
    rn_amax: jnp.ndarray,
    sss_m: jnp.ndarray,
    sst_m: jnp.ndarray,
    config: SI3IceCorConfig,
) -> SI3IceCorOutput:
    """NEMO ``ice_cor(kt, kn=1)`` for ``jpl = 1`` (see module docstring).

    ``rn_amax`` is NEMO's ``rn_amax_2d`` (per cell or scalar), ``sss_m`` /
    ``sst_m`` the sea-surface salinity [g/kg] / temperature [degC] means.
    """

    _validate(config, state)
    hard = config.limit_sharpness_scale == math.inf
    r1_dt = _sr(_SI3_ICE_COR_ONE / config.dt_s)
    himin = config.rn_himin_m
    a_i = state.ice_area
    v_i = state.ice_volume
    a_ip = state.pond_area

    # --- icecor.F90:66-73: thickness must exceed rn_himin ---------------------
    has_area = a_i >= _SI3_ICE_COR_EPSI20
    h_i = jnp.where(has_area, _sr(v_i / jnp.where(has_area, a_i, _SI3_ICE_COR_ONE)), _SI3_ICE_COR_ZERO)
    # Every limit = NEMO's hard statement + a band correction ``x^+ - ramp(x)`` that is
    # exactly 0 outside the one-sided ramp band, so the default equals NEMO bit for bit there.
    a_in = a_i
    thin = h_i < himin
    if config.ponds:
        a_ip_hard = jnp.where(thin, _sr(_sr(a_ip * h_i) / himin), a_ip)
    a_i = jnp.where(thin, _sr(_sr(a_i * h_i) / himin), a_i)
    if not hard:
        # min(h, himin) = himin - (himin - h)^+ -> himin - ramp(himin - h)
        thin_corr = _band_correction(himin - h_i, config.limit_sharpness_scale / himin) / himin
        if config.ponds:
            a_ip_hard = a_ip_hard + a_ip * thin_corr
        a_i = a_i + a_in * thin_corr
    if config.ponds:
        a_ip = a_ip_hard

    # --- :77-80: concentration must not exceed rn_amax (jpl = 1: at_i = a_i) --
    at_i = a_i
    over = at_i > rn_amax
    a_cap = jnp.where(over, _sr(_sr(a_i * rn_amax) / jnp.where(over, at_i, _SI3_ICE_COR_ONE)), a_i)
    if not hard:
        # icestp.F90:361-362 caps rn_amax at 1 - epsi10, so the open-water margin
        # (the cap's ramp width scale) is never below epsi10.
        margin = jnp.maximum(_SI3_ICE_COR_ONE - rn_amax, _SI3_ICE_COR_EPSI10)
        a_cap = a_cap + _band_correction(at_i - rn_amax, config.limit_sharpness_scale / margin)
    a_i = a_cap

    # --- :90-107: salinity bounds + salt flux ----------------------------------
    sfx = fluxes.sfx_res
    rhoi = config.rho_ice_kg_m3
    salt = state.salt_content
    if config.nn_icesal == 2:
        lower = _sr(config.rn_simin_g_kg * v_i)
        upper = _sr(_sr(config.rn_sinew * sss_m) * v_i)
        new_salt = _clamp(salt, lower, upper, v_i, config, sss_m, hard)
        sfx = _sr(sfx - _sr(_sr(_sr(new_salt - salt) * rhoi) * r1_dt))
    else:
        nlay_i = salt.shape[-1]
        r1_nlay_i = _sr(_SI3_ICE_COR_ONE / float(nlay_i))
        lower = _sr(_sr(config.rn_simin_g_kg * v_i) * r1_nlay_i)
        upper = _sr(_sr(_sr(config.rn_sinew * sss_m) * v_i) * r1_nlay_i)
        layers = []
        for k in range(nlay_i):
            old = salt[..., k]
            new = _clamp(old, lower, upper, v_i * r1_nlay_i, config, sss_m, hard)
            sfx = _sr(sfx - _sr(_sr(_sr(new - old) * rhoi) * r1_dt))
            layers.append(new)
        new_salt = jnp.stack(layers, axis=-1)

    # --- icevar.F90 ice_var_zapsmall ------------------------------------------
    sz_i = state.ice_layer_salinity
    layered = config.nn_icesal == 4
    if hard:
        has_ice = a_i > _SI3_ICE_COR_EPSI10
        h_i = jnp.where(has_ice, _sr(v_i / jnp.where(has_ice, a_i, _SI3_ICE_COR_ONE)), _SI3_ICE_COR_ZERO)
        small = jnp.minimum(jnp.minimum(a_i, v_i), h_i) < _SI3_ICE_COR_EPSI10
        zap_fraction = small.astype(a_i.dtype)
    else:
        # One-sided C1 zap share: z = 0 for min(a, v, h) >= epsi10 (NEMO keeps the cell),
        # z = 1 below epsi10 - width (NEMO zaps), C1 cubic step in between; width
        # = epsi10 / limit_sharpness_scale.  h = v/a for a > 0 keeps z continuous (a = 0
        # lies inside the z = 1 region).
        width = _SI3_ICE_COR_EPSI10 / config.limit_sharpness_scale
        positive = a_i > _SI3_ICE_COR_ZERO
        h_i = jnp.where(positive, v_i / jnp.where(positive, a_i, _SI3_ICE_COR_ONE), _SI3_ICE_COR_ZERO)
        deficit = _SI3_ICE_COR_EPSI10 - jnp.minimum(jnp.minimum(a_i, v_i), h_i)
        zap_fraction = smoothstep_cubic(deficit / width)  # C1, exactly 0 / 1 outside the band
        small = zap_fraction
    z = zap_fraction
    zl = z[..., None]

    def book(ledger, amount):  # ledger + z * amount (hard: NEMO's conditional update)
        if hard:
            return jnp.where(small, _sr(ledger + amount), ledger)
        return ledger + z * amount

    def zap(x, fill=_SI3_ICE_COR_ZERO, layers=False):
        if hard:
            return jnp.where(small[..., None] if layers else small, fill, x)
        w = zl if layers else z
        return w * fill + (_SI3_ICE_COR_ONE - w) * x

    if layered:
        for k in range(new_salt.shape[-1]):
            sfx = book(sfx, _sr(_sr(new_salt[..., k] * rhoi) * r1_dt))
        new_salt = zap(new_salt, layers=True)
        if sz_i is not None:
            sz_i = zap(sz_i, config.rn_simin_g_kg, layers=True)
    else:
        sfx = book(sfx, _sr(_sr(new_salt * rhoi) * r1_dt))
        new_salt = zap(new_salt)

    hfx = fluxes.hfx_res
    for k in range(state.ice_enthalpy.shape[-1]):
        hfx = book(hfx, -_sr(state.ice_enthalpy[..., k] * r1_dt))
    for k in range(state.snow_enthalpy.shape[-1]):
        hfx = book(hfx, -_sr(state.snow_enthalpy[..., k] * r1_dt))

    wfx = fluxes.wfx_res
    wfx = book(wfx, _sr(_sr(v_i * rhoi) * r1_dt))
    wfx = book(wfx, _sr(_sr(state.snow_volume * config.rho_snow_kg_m3) * r1_dt))
    pond_water = _sr(state.pond_volume + state.pond_lid_volume)
    wfx = book(wfx, _sr(_sr(pond_water * config.rho_water_kg_m3) * r1_dt))

    a_i = zap(a_i)
    t_su = zap(state.surface_temperature, _sr(sst_m + config.rt0_k))
    open_water = state.open_water_area
    if open_water is not None:
        if hard:
            open_water = jnp.where(a_i == _SI3_ICE_COR_ZERO, _SI3_ICE_COR_ONE, open_water)
        else:
            open_water = zap(open_water, _SI3_ICE_COR_ONE)
    out_state = SI3IceCorState(
        ice_area=a_i,
        ice_volume=zap(v_i),
        snow_volume=zap(state.snow_volume),
        salt_content=new_salt,
        age_content=zap(state.age_content),
        pond_area=zap(a_ip),
        pond_volume=zap(state.pond_volume),
        pond_lid_volume=zap(state.pond_lid_volume),
        snow_enthalpy=zap(state.snow_enthalpy, layers=True),
        ice_enthalpy=zap(state.ice_enthalpy, layers=True),
        surface_temperature=t_su,
        open_water_area=open_water,
        snow_thickness=None if state.snow_thickness is None else zap(state.snow_thickness),
        snow_temperature=(
            None if state.snow_temperature is None
            else zap(state.snow_temperature, config.rt0_k, layers=True)
        ),
        ice_temperature=(
            None if state.ice_temperature is None
            else zap(state.ice_temperature, config.rt0_k, layers=True)
        ),
        ice_layer_salinity=sz_i,
    )
    return SI3IceCorOutput(
        state=out_state,
        fluxes=SI3IceCorFluxes(sfx_res=sfx, wfx_res=wfx, hfx_res=hfx),
        ice_thickness=zap(h_i),
        total_ice_area=a_i,
        total_ice_volume=out_state.ice_volume,
        zap_fraction=zap_fraction,
    )


def _band_correction(x, sharpness):
    """``max(x, 0) - positive_part_c1(x)``: exactly 0 outside ``(0, 1/sharpness)``.

    Added to NEMO's hard ``MIN``/``MAX`` it turns the limit into the one-sided C1 ramp
    (``min(y, b) = y - (y - b)^+``) without touching any cell outside the band.
    """
    return jnp.maximum(x, _SI3_ICE_COR_ZERO) - positive_part_c1(x, sharpness)


def _clamp(value, lower, upper, volume, config, sss_m, hard):
    """``MIN(MAX(lower, value), upper)`` (NEMO); smooth default adds the one-sided ramp
    corrections on the specific salinity ``S = value/volume`` (widths 0.01 x each bound).
    """
    clamped = jnp.minimum(jnp.maximum(lower, value), upper)
    if hard:
        return clamped
    has_volume = volume > _SI3_ICE_COR_ZERO
    safe = jnp.where(has_volume, volume, _SI3_ICE_COR_ONE)
    specific = value / safe
    simin = config.rn_simin_g_kg
    simax = config.rn_sinew * sss_m
    # S + ramp(simin - S) - ramp(S - simax) = clamped/v - bc(simin - S) + bc(S - simax)
    corr = _band_correction(specific - simax, config.limit_sharpness_scale / simax) - _band_correction(
        simin - specific, config.limit_sharpness_scale / simin
    )
    return jnp.where(has_volume, clamped + volume * corr, clamped)


__all__ = (
    "SI3_ICE_COR_DEFAULT_LIMIT_SHARPNESS_SCALE",
    "SI3IceCorConfig",
    "SI3IceCorFluxes",
    "SI3IceCorOutput",
    "SI3IceCorState",
    "apply_si3_jpl1_ice_cor",
)
