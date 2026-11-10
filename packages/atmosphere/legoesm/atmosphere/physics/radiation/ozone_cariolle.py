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
    # --- model top (vote Y1a): start the overhead column at the climatological
    # column above the top half-level, k6 - k1*(p_full - p_half)/g, not at 0.
    # KNOWN DIFFERENCE: O3CHEM starts at 0 at IFS's ~0.01 hPa top, where that
    # column is ~0; above our 3.6 hPa top it is 4e-5..1.2e-4 kg/m2 (2-6 DU),
    # and 0 biased top-level ozone by ~8%. False = the oracle's literal form.
    seed_overhead_column: bool = True
    # --- switch: carry the o3_ifs tracer (MPAS column lane only) ---
    enabled: bool = False


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
    co3_top = (k[:, 0, 5] - k[:, 0, 0] * delps[:, 0] / g
               if cfg.seed_overhead_column else jnp.zeros(o3.shape[0], o3.dtype))
    _, anex = jax.lax.scan(level, co3_top.astype(o3.dtype), xs)
    return (anex.T - o3) / dt


# ---------------------------------------------------------------------------
# Run-time coefficients and the MPAS-lane tracer module (votes C1-C5, O1-O6,
# P1-P9).  The prognostic ozone is a TRACER ONLY: radiation keeps its own
# ozone (user decision 2026-11-10).
# ---------------------------------------------------------------------------

import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402

CARIOLLE_TRACER = "o3_ifs"
CARIOLLE_FILE = os.path.join(os.path.dirname(__file__), "data",
                             "cariolle_48r1_t63_cam_l32.npz")
# Votes P6a/X1a: the npz holds the deck's standard pressures built in fp32;
# an fp64 run rebuilds them 5.5e-8 apart (measured), a different coordinate
# differs by >= 1e-3, so 1e-6 accepts both precisions and refuses the rest.
_P_STD_RTOL = 1.0e-6


def load_cariolle_coefficients(p_std_full, lat_deg, path: str = CARIOLLE_FILE):
    """Monthly coefficients on the run's columns: ``(12, ncol, nlev, 7)``.

    Verifies the npz sha256 against its json record, refuses a deck whose
    standard full-level pressures (``p_s = p_atm_std``) are not the ones the
    file was built for, and interpolates linearly in latitude to each
    column, constant beyond the outermost Gaussian rows (votes O2a, P7a)."""
    raw = open(path, "rb").read()
    meta = json.load(open(os.path.splitext(path)[0] + ".json"))
    if hashlib.sha256(raw).hexdigest() != meta["npz_sha256"]:
        raise ValueError(f"{path}: sha256 does not match its json record")
    d = np.load(path)
    p_file = np.asarray(d["p_std_full"], np.float64)
    p_run = np.asarray(p_std_full, np.float64).reshape(-1)
    if p_run.shape != p_file.shape or not np.allclose(
            p_run, p_file, rtol=_P_STD_RTOL, atol=0.0):
        raise ValueError(
            f"Cariolle coefficients were built for {str(d['vertical_coord'])} "
            f"({p_file.size} levels); this run's standard pressures differ. "
            "Rebuild with scripts/data/build_ifs_cariolle_coefficients.py.")
    lat_f = np.asarray(d["lat_deg"], np.float64)[::-1]         # south to north
    coef = np.asarray(d["coef"], np.float64)[..., ::-1]         # (12, 7, nlev, nlat)
    lat = np.asarray(lat_deg, np.float64).reshape(-1)
    out = np.empty((12, lat.size, p_file.size, coef.shape[1]))
    for m in range(12):
        for j in range(coef.shape[1]):
            for k in range(p_file.size):
                out[m, :, k, j] = np.interp(lat, lat_f, coef[m, j, k])
    return out


def cariolle_month_weights(day_of_year: float):
    """UPDO3CH time weights (updo3ch.F90:175-197) on the noleap calendar:
    ``(month1, month2, w1)`` with 0-based months; the monthly fields sit on
    day 15, ``w1 = (IJT2 - IJOUR) / (IJT2 - IJT1)`` with IJOUR the integer
    day of month of 1-based ``day_of_year``."""
    from legoesm.forcing.time_utils import (
        NOLEAP_DAYS_PER_MONTH, NOLEAP_MONTH_STARTS)
    doy = int(math.floor(day_of_year)) - 1                    # 0-based
    imois = max(m for m in range(12) if NOLEAP_MONTH_STARTS[m] <= doy)
    ijour = doy - NOLEAP_MONTH_STARTS[imois] + 1
    if ijour >= 15:
        m1, m2 = imois, (imois + 1) % 12
        ijt1, ijt2 = 15, 15 + NOLEAP_DAYS_PER_MONTH[m1]
    else:
        m1, m2 = (imois - 1) % 12, imois
        ijt1, ijt2 = 15 - NOLEAP_DAYS_PER_MONTH[m1], 15
    return m1, m2, (ijt2 - ijour) / (ijt2 - ijt1)


def cariolle_daily_coefficients(monthly, day_of_year: float):
    """``(ncol, nlev, 7)`` coefficients for one model day (vote C3a)."""
    m1, m2, w1 = cariolle_month_weights(day_of_year)
    return w1 * monthly[m1] + (1.0 - w1) * monthly[m2]


def make_cariolle_physics(cfg: OzoneCariolleConfig, dt: float):
    """MPAS-lane module: O3CHEM tendency of the advected ``o3_ifs`` tracer on
    the pre-physics state (parallel split, like METHOX), TSPHY = the physics
    dt (vote P4a), coefficients from ``forcing["cariolle_coef"]``."""
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticTendencies

    def physics_fn(state, mesh, sigma_coord, forcing=None):
        coef = None if forcing is None else forcing.get("cariolle_coef")
        if coef is None:
            raise ValueError("ifs_ozone_chem: no cariolle_coef in the forcing")
        o3 = state.tracers[CARIOLLE_TRACER].data
        T = state.T.data
        p_s = state.p_s.data
        p_full = sigma_coord.pressure_at_full(p_s).reshape(T.shape)
        p_half = sigma_coord.pressure_at_half(p_s).reshape(
            T.shape[0], T.shape[1] + 1)
        do3 = cariolle_tendency(o3, T, p_full, p_half,
                                jnp.asarray(coef, T.dtype), dt, cfg)
        zero = jnp.zeros_like
        return HydrostaticTendencies(
            du_dt=state.u.replace(data=zero(state.u.data), name="du_dt_o3chem"),
            dv_dt=(None if state.v is None
                   else state.v.replace(data=zero(state.v.data), name="dv_dt_o3chem")),
            dT_dt=state.T.replace(data=zero(T), name="dT_dt_o3chem"),
            dp_s_dt=state.p_s.replace(data=zero(p_s), name="dp_s_dt_o3chem"),
            dphis_dt=state.phis.replace(data=zero(state.phis.data), name="dphis_dt_o3chem"),
            tracer_tendencies={CARIOLLE_TRACER: Field(
                data=do3, name="do3_ifs_dt", dims=state.T.dims, units="kg/kg/s")},
        )

    physics_fn._wants_forcing = True
    physics_fn.reset_state = lambda: None
    return physics_fn
