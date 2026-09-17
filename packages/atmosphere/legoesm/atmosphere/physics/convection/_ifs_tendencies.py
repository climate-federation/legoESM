"""Convective temperature/humidity tendencies from updraught/downdraught fluxes.

Port of OpenIFS ``cudtdqn.F90`` (implicit mass-flux solver branch, RMFSOLTQ=1,
RMFSOLRHS=0, LDTDKMF=.FALSE., LPHYLIN=.FALSE., LMFENTHCONS=.FALSE.) together
with the ``cubidiag.F90`` forward-substitution bidiagonal solver.

Main 8f6f722.

Declared departures from the general source (fixed by the porting contract):
  * Liquid-only microphysics: PLGLAC = PDPMEL = PSNDE = 0, PLUDELI = (PLUDE, 0),
    RLMLT terms vanish, FOELHMCU -> constants.L_v.
  * ZINT = 0 (LMFENTHCONS = .FALSE., sucumf.F90:208).
  * PTENT/PTENQ accumulation becomes a pure return of the increments.
  * LDTDKMF = .FALSE., so the advective correction (RMFADVW/RMFADVWDD) branch
    is the active one.

Index conventions (surface-last arrays, as in ``_ifs_ascent.py``):
  * our full level j = 0..nlev-1, top first; IFS full JK <-> our j = JK-1.
  * our half level j = 0..nlev, surface = nlev; IFS half JK <-> our j = JK-1.
  * ``k_ctop``/``k_dtop`` are carried such that the IFS conditions
    ``JK >= KCTOP-1`` / ``JK >= KDTOP`` translate to ``j >= k_ctop - 1`` /
    ``j >= k_dtop`` in our indexing.
"""

from __future__ import annotations

from typing import Dict, NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants


# ---------------------------------------------------------------------------
# Module constants (provenance)
# ---------------------------------------------------------------------------

# cubidiag.F90:98,102 -- ZBET = 1/(PB + 1.E-35). The additive floor keeps the
# division finite (and AD-safe w.r.t. 1/0 NaNs) for degenerate pivots; it is
# negligible for any physically meaningful PB (=1+ZZP*(...) >= 1 here).
_BIDIAG_FLOOR: float = 1.0e-35

# Floor for the ZGS denominator ZGH = constants.c_pd*T + geo (cudtdqn.F90:249). ZGH is a
# dry static energy (~ constants.c_pd*T) and never approaches zero physically; the floor
# only guards the autodiff of the division.
_ZGH_FLOOR: float = 1.0e-10


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class IFSTendencyConfig(NamedTuple):
    """yoecumf switches/values entering cudtdqn.F90 (frozen to this branch).

    All defaults are the operational defaults of sucumf.F90; in particular
    RMFADVW = 0 and RMFADVWDD = 0 disable the advective correction entirely
    (sucumf.F90:228-229, verified against the provided excerpt).
    """

    # sucumf.F90:225 -- mass-flux solver switch for T and q (implicit branch,
    # value 1.0; this module implements RMFSOLTQ > 0 only).
    rmfsoltq: float = 1.0
    # sucumf.F90:227 -- include (1) or not (0) model tendencies in the implicit
    # RHS; fixed to 0 for this branch.
    rmfsolrhs: float = 0.0
    # sucumf.F90:228 -- fraction [0-1] of convective subsidence handed to the
    # dynamics (cudtdqn.F90:195-196 builds ZADVW from this for KTYPE == 1).
    rmfadvw: float = 0.0
    # sucumf.F90:229 -- if RMFADVW > 0, keep (0) the downdraught mass flux in
    # convection or include it in dynamics (1).
    rmfadvwdd: float = 0.0



__param_spec__: Dict[str, Dict[str, object]] = {
    "IFSTendencyConfig": {
        "scheme_key": "atm.conv.IFSTendencyConfig",
        # Source switches of a faithful port, not calibration knobs: every
        # field is frozen to the operational branch this module implements.
        "excluded": {},
        "params": {
            "rmfsoltq": {
                "units": "1", "bounds": (0.0, 1.0), "tunable_tier": 0,
                "transform": "none", "category": "convection", "shape": None,
                "reference": "sucumf.F90:225 RMFSOLTQ=1.0; implicit solver "
                             "branch, source switch of a faithful port",
            },
            "rmfsolrhs": {
                "units": "1", "bounds": (0.0, 1.0), "tunable_tier": 0,
                "transform": "none", "category": "convection", "shape": None,
                "reference": "sucumf.F90:227 RMFSOLRHS=0.0; model tendencies "
                             "excluded from the implicit RHS",
            },
            "rmfadvw": {
                "units": "1", "bounds": (0.0, 1.0), "tunable_tier": 0,
                "transform": "none", "category": "convection", "shape": None,
                "reference": "sucumf.F90:228 RMFADVW=0.0; fraction of the "
                             "subsidence done by the dynamics (KTYPE==1)",
            },
            "rmfadvwdd": {
                "units": "1", "bounds": (0.0, 1.0), "tunable_tier": 0,
                "transform": "none", "category": "convection", "shape": None,
                "reference": "sucumf.F90:229 RMFADVWDD=0.0; downdraught "
                             "mass-flux handling when RMFADVW > 0",
            },
        },
    }
}



__physics_contract__: Dict[str, object] = {
    "summary": (
        "Implicit (RMFSOLTQ=1) convective T/q tendencies from the updraught "
        "and downdraught mass-flux fluxes, with the CUBIDIAG forward "
        "substitution; liquid-only, no enthalpy-conservation correction "
        "(LMFENTHCONS=.FALSE.), no RHS model tendencies (RMFSOLRHS=0). "
        "Surface-last arrays; IFS level JK maps to our j = JK-1."
    ),
    "inputs": {
        "T": "K", "q": "kg kg-1", "qs": "kg kg-1",
        "p_full": "Pa", "p_half": "Pa",
        "geo_full": "m2 s-2", "geo_half": "m2 s-2",
        "T_h": "K", "q_h": "kg kg-1",
        "M_u": "kg m-2 s-1", "M_d": "kg m-2 s-1",
        "PMFUS": "W m-2", "PMFDS": "W m-2",
        "PMFUQ": "kg m-2 s-1", "PMFDQ": "kg m-2 s-1",
        "PMFUL": "kg m-2 s-1", "PLUDE": "kg m-2 s-1", "PDMFUP": "kg m-2 s-1",
        "ldcum": "1 (bool)", "lddraf": "1 (bool)", "ktype": "1",
        "k_ctop": "1 (level index)", "k_dtop": "1 (level index)", "dt": "s",
    },
    "outputs": {
        "dT_dt": "K s-1", "dq_dt": "kg kg-1 s-1",
        "PENTH": "K s-1 (cudtdqn.F90:445 defines it as a temperature "
                 "tendency, not an enthalpy flux)",
    },
    "sign_convention": (
        "fluxes positive upward; pressure increases with the array index "
        "(surface last); the increments are added to the state, so a "
        "positive dT_dt warms the layer."
    ),
    "conserves": ["moisture", "energy"],
    "differentiable": True,
    "reference": (
        "OpenIFS cudtdqn.F90 + cubidiag.F90 (main 8f6f722), implicit branch "
        "RMFSOLTQ=1, RMFSOLRHS=0, LDTDKMF=.FALSE., LPHYLIN=.FALSE."
    ),
    "idealized_test": "tests/unit/test_ifs_tendencies.py (pending)",
}


# ---------------------------------------------------------------------------
# Bidiagonal solver (cubidiag.F90)
# ---------------------------------------------------------------------------


def bidiagonal_forward(
    A: jnp.ndarray,
    B: jnp.ndarray,
    R: jnp.ndarray,
    mask: jnp.ndarray,
    k_ctop: jnp.ndarray,
) -> jnp.ndarray:
    """Forward substitution of cubidiag.F90:87-105 as a top-down lax.scan.

    Solves the bidiagonal system ``B[j] U[j] + A[j] U[j-1] = R[j]`` per column,
    with the pivot ``U = R/B`` at the column's cloud-top level ``j = k_ctop - 1``
    (our indexing, IFS JK = KCTOP-1) and
    ``U[j] = (R[j] - A[j] U[j-1]) / B[j]`` below, only where ``mask[j]``
    (LLCUMBAS = LDCUM & j >= k_ctop-1) holds; elsewhere U = 0.

    Args:
        A: (ncol, nlev) sub-diagonal coefficients (IFS PA).
        B: (ncol, nlev) diagonal coefficients (IFS PB).
        R: (ncol, nlev) right-hand sides.
        mask: (ncol, nlev) LLCUMBAS mask.
        k_ctop: (ncol,) cloud-top full-level index, our indexing.

    Returns:
        U: (ncol, nlev) solution, zero outside the mask.
    """
    nlev = R.shape[1]
    levels = jnp.arange(nlev)

    def body(u_prev: jnp.ndarray, j: jnp.ndarray):
        is_pivot = levels[j] == (k_ctop - 1)  # (ncol,)
        denom = B[:, j] + _BIDIAG_FLOOR  # cubidiag.F90:98/102 additive floor
        u_new = jnp.where(
            is_pivot,
            R[:, j] / denom,
            (R[:, j] - A[:, j] * u_prev) / denom,
        )
        u_new = jnp.where(mask[:, j], u_new, jnp.zeros_like(u_new))
        return u_new, u_new

    _, u_stacked = jax.lax.scan(
        body, jnp.zeros(R.shape[0], dtype=R.dtype), levels
    )
    return u_stacked.T  # (nlev, ncol) -> (ncol, nlev)


# ---------------------------------------------------------------------------
# cudtdqn.F90 port
# ---------------------------------------------------------------------------


def ifs_convective_tendencies(
    T: jnp.ndarray,
    q: jnp.ndarray,
    qs: jnp.ndarray,
    p_full: jnp.ndarray,
    p_half: jnp.ndarray,
    geo_full: jnp.ndarray,
    geo_half: jnp.ndarray,
    T_h: jnp.ndarray,
    q_h: jnp.ndarray,
    ldcum: jnp.ndarray,
    ktype: jnp.ndarray,
    k_ctop: jnp.ndarray,
    k_dtop: jnp.ndarray,
    lddraf: jnp.ndarray,
    M_u: jnp.ndarray,
    M_d: jnp.ndarray,
    PMFUS: jnp.ndarray,
    PMFDS: jnp.ndarray,
    PMFUQ: jnp.ndarray,
    PMFDQ: jnp.ndarray,
    PMFUL: jnp.ndarray,
    PLUDE: jnp.ndarray,
    PDMFUP: jnp.ndarray,
    dt: jnp.ndarray,
    cfg: IFSTendencyConfig,
):
    """Implicit convective T/q tendencies (cudtdqn.F90, RMFSOLTQ branch).

    Returns:
        (dT_dt, dq_dt, PENTH): K/s, kg/kg/s, and PENTH in the source's own
        (temperature-tendency) unit, all zero where LDCUM is false.
    """
    # dtype pinning: everything derived from T's dtype.
    dtype = T.dtype
    ncol, nlev = T.shape
    # half-level inputs carry nlev+1 interfaces (index nlev = surface); the
    # IFS half arrays have KLEV entries (JK <-> our j = JK-1): drop the surface
    T_h = T_h[:, :nlev]
    q_h = q_h[:, :nlev]
    geo_half = geo_half[:, :nlev]
    jlev = jnp.arange(nlev)[None, :]  # our full-level index, top first

    # --- 1.0 setup (cudtdqn.F90:231-238) ---------------------------------
    # ZDP(JK) = RG/(PAPH(JK+1)-PAPH(JK)) -- IFS JK <-> our j+1, so our
    # ZDP[j] = constants.g/(p_half[j+1]-p_half[j]) (surface-last half levels).
    zdp = constants.g / (p_half[:, 1:] - p_half[:, :-1])

    # "level above" and "level below" shifted full-level fields.
    def shift_up(x):  # x[j-1] with a safe (masked) duplicate at j = 0
        return jnp.concatenate([x[:, :1], x[:, :-1]], axis=1)

    def shift_dn(x):  # x[j+1] with a safe duplicate at j = nlev-1
        return jnp.concatenate([x[:, 1:], x[:, -1:]], axis=1)

    T_a, q_a, geo_a = shift_up(T), shift_up(q), shift_up(geo_full)

    # ZADVW: RMFADVW for KTYPE == 1 columns, else 0 (cudtdqn.F90:193-196).
    zadvw = jnp.where(ktype == 1, cfg.rmfadvw, 0.0).astype(dtype)

    # The IFS tendency/matrix loops run JK = KTOPM2..KLEV (KTOPM2 >= 2), i.e.
    # our j >= 1: the topmost model level is never updated. We include the
    # j >= 1 restriction in every active mask below.
    base_mask = ldcum[:, None] & (jlev >= 1)

    # --- 2.0 implicit flux recomputation (:244-272, RMFSOLTQ > 0) --------
    # ZIMP = 1 - RMFSOLTQ = 0 for this branch, so the IK (level-above)
    # contributions to ZS/ZQ drop out except through the ZGS/ZGQ
    # interpolations; we keep the ZIMP form written out with ZIMP = 0 to
    # mirror the source, floored where needed for AD-safety.
    zimp = 1.0 - cfg.rmfsoltq
    flux_window = base_mask & (jlev >= (k_ctop[:, None] - 1))

    zgq = (q_h - q_a) / qs
    zgh = constants.c_pd * T + geo_full
    zgs = (constants.c_pd * (T_h - T_a) + geo_half - geo_a) / (zgh + _ZGH_FLOOR)
    zs = constants.c_pd * (zimp * T_a + zgs * T) + geo_a + zgs * geo_full
    zq = zimp * q_a + zgq * qs

    zmfus = jnp.where(flux_window, PMFUS - M_u * zs, PMFUS)
    zmfuq = jnp.where(flux_window, PMFUQ - M_u * zq, PMFUQ)
    ddraf_window = lddraf[:, None] & (jlev >= k_dtop[:, None])
    zmfds = jnp.where(
        flux_window & ddraf_window, PMFDS - M_d * zs, PMFDS
    )
    zmfdq = jnp.where(
        flux_window & ddraf_window, PMFDQ - M_d * zq, PMFDQ
    )

    # --- 3.0 RHS tendencies (:305-353, non-LDTDKMF, ZINT=0, liquid-only) --
    tend_mask = base_mask & (jlev >= (k_ctop[:, None] - 1))
    is_bot = jlev == (nlev - 1)  # IFS JK = KLEV

    d_mfus = shift_dn(zmfus) - zmfus
    d_mfds = shift_dn(zmfds) - zmfds
    d_mfuq = shift_dn(zmfuq) - zmfuq
    d_mfdq = shift_dn(zmfdq) - zmfdq
    d_mful = shift_dn(PMFUL) - PMFUL

    # interior (JK < KLEV): the RLMLT/PLGLAC/PDPMEL terms vanish and, with
    # ZINT = 0 and liquid-only detrainment, the latent-heat bracket reduces
    # to -constants.L_v*(PMFUL[j+1]-PMFUL[j]) + constants.L_v*PDMFUP[j] + constants.L_v*PLUDE[j]
    # (PLUDELI(:,1) = PLUDE at RLVTT, PLUDELI(:,2) = PSNDE = 0).
    zdtdt_int = (zdp / constants.c_pd) * (
        d_mfus + d_mfds - constants.L_v * d_mful + constants.L_v * PDMFUP + constants.L_v * PLUDE
    )
    zdqdt_int = zdp * (
        d_mfuq + d_mfdq + d_mful - PLUDE - PDMFUP
    )
    # surface (JK = KLEV): -(ZMFUS+ZMFDS) + constants.L_v*PMFUL + constants.L_v*PDMFUP etc.
    zdtdt_bot = -(zdp[:, -1] / constants.c_pd) * (
        zmfus[:, -1] + zmfds[:, -1] - constants.L_v * PMFUL[:, -1] - constants.L_v * PDMFUP[:, -1]
    )
    zdqdt_bot = -zdp[:, -1] * (
        zmfuq[:, -1] + zmfdq[:, -1] + PMFUL[:, -1] + PDMFUP[:, -1]
    )

    zdtdt = jnp.where(
        is_bot,
        jnp.broadcast_to(zdtdt_bot[:, None], (ncol, nlev)).astype(dtype),
        zdtdt_int,
    )
    zdqdt = jnp.where(
        is_bot,
        jnp.broadcast_to(zdqdt_bot[:, None], (ncol, nlev)).astype(dtype),
        zdqdt_int,
    )
    zdtdt = jnp.where(tend_mask, zdtdt, jnp.zeros_like(zdtdt))
    zdqdt = jnp.where(tend_mask, zdqdt, jnp.zeros_like(zdqdt))

    # --- 3.2 implicit solution (:369-428) --------------------------------
    # LLCUMBAS = LDCUM & JK >= KCTOP-1 (plus the KTOPM2 j>=1 restriction).
    llcumbas = tend_mask

    # ZZP = RMFSOLTQ*ZDP*PTSPHY; A = -ZZP*(PMFU+PMFD) (reusing ZMFUS in IFS);
    # B = 1 + ZZP*(PMFU[j+1]+PMFD[j+1]) for JK < KLEV else 1.
    zzp = cfg.rmfsoltq * zdp * dt
    a_mat = -zzp * (M_u + M_d)
    b_mat = jnp.where(
        ~is_bot, 1.0 + zzp * (shift_dn(M_u) + shift_dn(M_d)), jnp.ones_like(zzp)
    )

    # Advective correction (non-LDTDKMF): RMFSOLRHS = 0 kills the PTENT/PTENQ
    # RHS terms, leaving ZS2/ZQ2 from RMFADVW/RMFADVWDD. PAP(JK)-PAP(JK-1)
    # is the full-level pressure difference to the level above (floored).
    dp_full = jnp.abs(p_full - shift_up(p_full)) + _BIDIAG_FLOOR
    zzp2 = (
        constants.g * (M_u + cfg.rmfadvwdd * M_d) / dp_full * dt * zadvw[:, None]
    )
    zs2 = zzp2 * (T_a - T + (geo_a - geo_full) / constants.c_pd)
    zq2 = zzp2 * (q_a - q)

    rhs_t = zdtdt * dt + T - zs2
    rhs_q = zdqdt * dt + q - zq2

    # Mask A/B/RHS exactly as IFS (A = 0, B = 1, RHS = 0 outside LLCUMBAS).
    a_mat = jnp.where(llcumbas, a_mat, jnp.zeros_like(a_mat))
    b_mat = jnp.where(llcumbas, b_mat, jnp.ones_like(b_mat))
    rhs_t = jnp.where(llcumbas, rhs_t, jnp.zeros_like(rhs_t))
    rhs_q = jnp.where(llcumbas, rhs_q, jnp.zeros_like(rhs_q))

    zr1 = bidiagonal_forward(a_mat, b_mat, rhs_t, llcumbas, k_ctop)
    zr2 = bidiagonal_forward(a_mat, b_mat, rhs_q, llcumbas, k_ctop)

    # Tendencies (:425-434, RMFSOLRHS = 0): pure increments of PTENT/PTENQ.
    dT_dt = jnp.where(llcumbas, (zr1 - T) / dt, jnp.zeros_like(T))
    dq_dt = jnp.where(llcumbas, (zr2 - q) / dt, jnp.zeros_like(q))
    # PENTH as written at cudtdqn.F90:445: a temperature tendency (see
    # __physics_contract__ note on the unit).
    penth = dT_dt

    return dT_dt, dq_dt, penth
