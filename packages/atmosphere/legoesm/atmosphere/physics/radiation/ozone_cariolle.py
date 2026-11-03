"""IFS 49r1 prognostic ozone: Cariolle & Deque (1986) linear chemistry.

ORACLE.  OpenIFS 49r1 (8f6f722) ``arpifs/phys_ec/o3chem.F90`` on its 49r1
default path (su0phy.F90:930-933: ``LO3CH_SAFE = .TRUE.``, hybrid linear
ozone ``LO3CH_HLO = .TRUE.``; ``LO3CH_OLDVER``/``LO3CH_BMS`` false).  The
eight coefficients per level are the zonal fields GPINO3CH (adiab/
gpino3ch.F90) copies per Gaussian latitude:

  k1 r_clim, k2 (P-L)_clim, k3 d(P-L)/dr, k4 T_clim, k5 d(P-L)/dT,
  k6 Sigma_clim (overhead O3 column), k7 d(P-L)/dSigma, k8 (unused at HLO).

Each level is solved implicitly in its own mixing ratio and in the overhead
column (built top-down from the already-updated levels above):

  r* = [r + dt (k2 + k5 (T'-k4) + k7 (Sigma-k6) - k1 k3)]
       / [1 - dt (k3 + k7 (p_full - p_half_above)/g)],
  r_new = max(1e-9, r*),  T' = clamp(T, k4 - 15 K, k4 + 15 K).

RULE 5 (vote V7, oracle_audit/ifs_port/wms_ozone/VOTES.md): the 15 K
temperature clamp is a smooth_clamp of width ``t_clamp_width_K`` and the
1e-9 floor a linear smooth_max of width ``floor_rel_width * 1e-9``
(0 = O3CHEM's hard forms).  Checked against the running oracle in
``tests/atmosphere/test_ifs_wms_ozone_oracle.py``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.smooth import smooth_clamp, smooth_max

__physics_contract__ = {
    "summary": (
        "IFS 49r1 O3CHEM (hybrid linear ozone, LO3CH_SAFE): Cariolle linear "
        "ozone chemistry tendency, implicit in the local mixing ratio and in "
        "the overhead ozone column."
    ),
    "inputs": {
        "o3": "kg/kg (ozone mass mixing ratio)", "T": "K",
        "p_full": "Pa", "p_half": "Pa (index 0 = model top)",
        "coef": "8 Cariolle coefficients per level (see module docstring)",
    },
    "outputs": {"do3_dt": "kg/kg/s (ozone tendency)"},
    "sign_convention": "do3_dt = (r_new - r)/dt; r_new >= 1e-9 kg/kg (hard form).",
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Cariolle and Deque 1986 (JGR 91D10); Cariolle and Teyssedre 2007 "
        "(ACP 7); OpenIFS 49r1 phys_ec/o3chem.F90, adiab/gpino3ch.F90."
    ),
    "idealized_test": (
        "tests/atmosphere/test_ifs_wms_ozone_oracle.py: matches the running "
        "49r1 O3CHEM at rel 1e-12 (fp64) on the tier-2 audit columns."
    ),
}

__param_spec__ = {
    "OzoneCariolleConfig": {
        "scheme_key": "atm.chem.OzoneCariolleConfig",
        "excluded": {
            "t_clamp_K": "oracle safety constant o3chem.F90 ZDTSAFE (fit-validity guard)",
            "o3_floor_kg_kg": "oracle positivity floor o3chem.F90 MAX(1.E-9, ZANEX)",
            "t_clamp_width_K": "numerics: RULE 5 smoothing width (vote V7), 0 = oracle hard form",
            "floor_rel_width": "numerics: RULE 5 smoothing width (vote V7), 0 = oracle hard form",
        },
        "params": {},
    },
}

N_CARIOLLE_COEF = 8


class OzoneCariolleConfig(NamedTuple):
    # --- O3CHEM safety settings (LO3CH_SAFE) ---
    t_clamp_K: float = 15.0
    o3_floor_kg_kg: float = 1.0e-9
    # --- numerics (RULE 5) ---
    t_clamp_width_K: float = 0.01
    floor_rel_width: float = 0.01


def _sharp(w):
    return math.inf if w == 0.0 else 1.0 / w


def cariolle_tendency(o3, T, p_full, p_half, coef, dt,
                      cfg: OzoneCariolleConfig = OzoneCariolleConfig(),
                      g: float = constants.g):
    """O3CHEM ``PTENO3`` [kg/kg/s] (o3chem.F90:226-307, HLO + SAFE path).

    ``o3``, ``T``, ``p_full``: ``(ncol, nlev)``; ``p_half``: ``(ncol, nlev+1)``;
    ``coef``: ``(ncol, nlev, 8)``; levels top to bottom; ``dt`` = TSPHY [s].
    """
    o3 = jnp.asarray(o3)
    k = jnp.asarray(coef)
    delps = p_full - p_half[:, :-1]                     # ZDELPS
    delp = p_half[:, 1:] - p_half[:, :-1]               # PDELP
    tc = cfg.t_clamp_K
    lo, hi = k[..., 3] - tc, k[..., 3] + tc
    tt = smooth_clamp(T, lo, hi, _sharp(cfg.t_clamp_width_K))
    rapp = 1.0 - dt * (k[..., 2] + k[..., 6] * delps / g)
    floor = cfg.o3_floor_kg_kg
    s_floor = _sharp(cfg.floor_rel_width * floor)

    def level(co3, xs):
        r, t, kk, rp, dp = xs
        anex = r + dt * (kk[:, 1] + kk[:, 4] * (t - kk[:, 3])
                         + kk[:, 6] * (co3 - kk[:, 5]) - kk[:, 0] * kk[:, 2])
        anex = smooth_max(jnp.full_like(anex, floor), anex / rp, s_floor)
        return co3 + dp * anex / g, anex

    xs = (o3.T, tt.T, jnp.moveaxis(k, 1, 0), rapp.T, delp.T)
    _, anex = jax.lax.scan(level, jnp.zeros(o3.shape[0], o3.dtype), xs)
    return (anex.T - o3) / dt
