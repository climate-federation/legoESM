"""RRTM_PREPARE_GASES as ecRad (OpenIFS 49r1, 8f6f722) calls it from
``radiation_ifs_rrtm.F90`` gas_optics.

Line-by-line port of ``bld49/src/rrtm_prepare_gases.F90``: ecRad gas mass
mixing ratios on model levels (top-down, ecRad order) -> RRTM column amounts on
layers counted from the SURFACE up (RRTM order, index 0 = lowest layer).
Shared by the LW (:mod:`.rrtmg_lw`) and SW (:mod:`.rrtmg_sw`) gas optics.

Outputs keep the Fortran names and the oracle dump layout, column first:
``coldry, wbrodl, pavel, tavel (ncol, nlev)``, ``pz, tz (ncol, nlev+1)`` (index
0 = surface half level), ``wkl (ncol, 35, nlev)``, ``wx (ncol, 4, nlev)``,
``ireflect (ncol,)`` (always 0).

Gravity is the caller's (vote V6-lw: required argument; the 49r1 oracle runs
with ``RG/RPLRG = 9.80665`` from yomcst/yomdyncore). The molecular weights and
Avogadro number are the oracle's own DATA statements (they differ from
``legoesm.constants``: e.g. Avogadro 6.02214e23 vs 6.02214076e23), so they are
ported here as a provenance block. No switches: nothing to smooth.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

__physics_contract__ = {
    "summary": (
        "RRTM_PREPARE_GASES (ecRad 49r1): mass mixing ratios on top-down model "
        "levels -> RRTM bottom-up layer pressures/temperatures, dry-air column "
        "and gas column amounts (molec/cm^2) for the RRTMG LW/SW gas optics."
    ),
    "inputs": {
        "pressure_hl": "Pa", "pressure_fl": "Pa", "temperature_hl": "K",
        "temperature_fl": "K", "gas_mmr": "kg/kg (ecRad gas order)",
        "gravity": "m/s^2",
    },
    "outputs": {
        "coldry": "molec/cm^2", "wbrodl": "molec/cm^2", "wkl": "molec/cm^2",
        "wx": "1e20 molec/cm^2", "pavel": "hPa", "tavel": "K", "pz": "hPa",
        "tz": "K", "ireflect": "1",
    },
    "sign_convention": (
        "RRTM layer index 0 is the lowest layer (input level order reversed); "
        "column amounts are positive for positive pressure thickness."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "OpenIFS 49r1 (8f6f722) arpifs/phys_radi/rrtm_prepare_gases.F90; "
        "Mlawer et al. 1997 JGR 102:16663"
    ),
    "idealized_test": (
        "tests/atmosphere/test_ecrad_ifs_lw_gas_oracle.py: element match to the "
        "49r1 stage dump lw_prep_* at 1e-12 relative"
    ),
}

# --- ecRad gas indices (radiation_gas_constants.F90:27-38), zero-based ---
IH2O, ICO2, IO3, IN2O, ICO, ICH4, IO2 = 0, 1, 2, 3, 4, 5, 6
ICFC11, ICFC12, IHCFC22, ICCL4, INO2 = 7, 8, 9, 10, 11

# --- array sizes (parrrtm.F90:35-36) ---
JPXSEC = 4
JPINPX = 35

# --- molecular weights g/mol and Avogadro (rrtm_prepare_gases.F90:94-104) ---
_ZAMD = 28.970
_ZAMW = 18.0154
_ZAMCO2 = 44.011
_ZAMO = 47.9982
_ZAMCH4 = 16.043
_ZAMN2O = 44.013
_ZAMC11 = 137.3686
_ZAMC12 = 120.9140
_ZAMC22 = 86.4690
_ZAMCL4 = 153.8230
_ZAVGDRO = 6.02214e23
# --- hard-coded O2 vmr, unit conversions (rrtm_prepare_gases.F90:144,176,182,184) ---
_O2_VMR = 0.209488
_PA_PER_HPA = 100.0
_CM_PER_M = 1.0e2
_G_PER_KG = 1.0e3
_XSEC_SCALE = 1.0e-20
_ITMOL = 7   # rrtm_prepare_gases.F90:166


class PreparedGases(NamedTuple):
    coldry: jnp.ndarray
    wbrodl: jnp.ndarray
    wkl: jnp.ndarray
    wx: jnp.ndarray
    pavel: jnp.ndarray
    tavel: jnp.ndarray
    pz: jnp.ndarray
    tz: jnp.ndarray
    ireflect: jnp.ndarray


def const_operand(v):
    """A constant divisor XLA may not fold. Under jit, XLA rewrites x / c into
    x * (1/c), which is not the oracle's IEEE division (measured: SETCOEF
    selffrac 1.9e-12 relative off the 49r1 dump on the 32-column file, eager
    bitwise exact). The barrier keeps the true division; values and gradients
    are otherwise unchanged."""
    return jax.lax.optimization_barrier(jnp.asarray(v))


def full_levels(pressure_hl, temperature_hl):
    """radiation_ifs_rrtm.F90:356-366: full-level p, T as half-level means."""
    pfl = 0.5 * (pressure_hl[:, :-1] + pressure_hl[:, 1:])
    tfl = 0.5 * (temperature_hl[:, :-1] + temperature_hl[:, 1:])
    return pfl, tfl


def rrtm_prepare_gases(pressure_hl, pressure_fl, temperature_hl, temperature_fl,
                       gas_mmr, gravity) -> PreparedGases:
    """Port of RRTM_PREPARE_GASES.

    pressure_hl, temperature_hl: (ncol, nlev+1), top-down (ecRad order).
    pressure_fl, temperature_fl: (ncol, nlev). gas_mmr: (ncol, nlev, 12) in the
    ecRad gas order (IH2O..INO2). gravity: m/s^2 (RG/RPLRG of the caller).
    """

    def rev(a):                                      # JK -> KLEV-JK+1
        return a[:, ::-1]

    zgravit = gravity * _CM_PER_M                    # (RG/RPLRG)*1.E2
    pz = (jnp.concatenate([pressure_hl[:, -1:], rev(pressure_hl[:, :-1])], axis=1)
          / const_operand(_PA_PER_HPA))
    tz = jnp.concatenate([temperature_hl[:, -1:], rev(temperature_hl[:, :-1])], axis=1)
    pavel = rev(pressure_fl) / const_operand(_PA_PER_HPA)
    tavel = rev(temperature_fl)
    g = gas_mmr[:, ::-1, :]
    v1 = g[..., IH2O] * _ZAMD / const_operand(_ZAMW)
    v2 = g[..., ICO2] * _ZAMD / const_operand(_ZAMCO2)
    v3 = g[..., IO3] * _ZAMD / const_operand(_ZAMO)
    v4 = g[..., IN2O] * _ZAMD / const_operand(_ZAMN2O)
    v6 = g[..., ICH4] * _ZAMD / const_operand(_ZAMCH4)
    v7 = jnp.full_like(v1, _O2_VMR)
    zero = jnp.zeros_like(v1)
    zamm = (1.0 - v1) * _ZAMD + v1 * _ZAMW
    coldry = ((pz[:, :-1] - pz[:, 1:]) * _G_PER_KG * _ZAVGDRO
              / (zgravit * zamm * (1.0 + v1)))
    wx = jnp.stack([
        coldry * (g[..., ICCL4] * _ZAMD / const_operand(_ZAMCL4)) * _XSEC_SCALE,
        coldry * (g[..., ICFC11] * _ZAMD / const_operand(_ZAMC11)) * _XSEC_SCALE,
        coldry * (g[..., ICFC12] * _ZAMD / const_operand(_ZAMC12)) * _XSEC_SCALE,
        coldry * (g[..., IHCFC22] * _ZAMD / const_operand(_ZAMC22)) * _XSEC_SCALE,
    ], axis=1)
    vmr = [v1, v2, v3, v4, zero, v6, v7]             # WKL(1..ITMOL); CO (5) stays 0
    summol = zero
    for v in vmr[1:_ITMOL]:                          # DO JMOL = 2, ITMOL (in order)
        summol = summol + v
    wbrodl = coldry * (1.0 - summol)
    wkl = jnp.stack([coldry * v for v in vmr]
                    + [zero] * (JPINPX - _ITMOL), axis=1)
    ireflect = jnp.zeros(pressure_hl.shape[:1], dtype=jnp.int32)
    return PreparedGases(coldry, wbrodl, wkl, wx, pavel, tavel, pz, tz, ireflect)
