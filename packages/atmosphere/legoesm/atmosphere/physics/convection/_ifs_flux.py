"""IFS convective flux construction, ported from OpenIFS cuflxn.F90
(source main 8f6f722), branch Earth, deterministic configuration
(LPHYLIN=.FALSE., liquid-only, no downdraught).

Ported sections
---------------
* environmental_subtraction   cuflxn.F90:240-284  ("SECTION 1.4": absolute
  plume fluxes -> difference fluxes w.r.t. the environment; out-of-cloud
  zeroing of PMFU/PMFUS/PMFUQ/PMFUL/PDMFUP/PLUDE)
* below_cloud_base_scaling    cuflxn.F90:292-345  ("SECTION 1.5": linear
  decrease of the fluxes between cloud base and the surface, driven by the
  half-level pressure profile)

Scope boundary (DECLARED)
-------------------------
The precipitation-flux, melting and rain-evaporation parts of cuflxn
(PMFLXR / PMFLXS / PDPMEL / PSNDE / PLUDELI and the RHEBC evaporation
loop) are NOT ported here: the repository already carries a separate IFS
sub-cloud evaporation leaf, and wiring that coupling is a separate step.
Likewise the downdraught (PMFD/PMFDS/PMFDQ/IDBAS) is identically zero in
this reduced branch (cudlfsn/cuddrafn out of scope, as in the sibling
modules), and KTYPE is restricted to {1,2} so the KTYPE==3 ZZP squaring
never applies.  PLGLAC is zero (liquid-only).

Conventions (as _ifs_ascent.py / _ifs_closure.py / _ifs_tendencies.py)
---------------------------------------------------------------------
* surface-last; plume flux arrays are half-level, shape (ncol, nlev+1),
  IFS half level JK <-> our index j = JK-1, so IFS PAPH(KLEV+1) (surface)
  is p_half[:, nlev] and IFS array index JK is our column JK-1.
* k_cbot / k_ctop are OUR 0-based surface-last indices; per the closure
  module, IFS KCBOT == our k_cbot numerically, i.e. PAPH(KCBOT) is
  p_half[:, k_cbot] and source "IK = KCBOT+1" is our column k_cbot+1.
* specific humidity throughout; T_h/q_h are the half-level environment
  (IFS PTENH/PQENH) on the same (ncol, nlev+1) grid; geo_half is
  surface-relative, geo_half[:, nlev] = 0.
* pure JAX, no python loops over levels; jnp.where masks, not branching.
"""

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants as C

# ---------------------------------------------------------------------------
# Fixed source constants -- do not tune here.
# ---------------------------------------------------------------------------
RCPD = C.c_pd      # RCPD, cuflxn.F90:249 (RCPD*PTENH term in PMFUS)
L_VAPOR = C.L_v    # FOELHMCU in the liquid-only, LPHYLIN=.FALSE. branch is
                   # simply the latent heat of vaporisation (cuflxn.F90:311)

_EPS = 1.0e-20     # coeff-ok: AD-safe denominator floor, matches sibling modules


class IFSFluxConfig(NamedTuple):
    """Switches of cuflxn relevant to the ported sections.

    Both ported sections carry no tunables in the reduced branch
    (LPHYLIN=.FALSE. removes ZOEALFA/ZOELHM; KTYPE in {1,2} removes the
    ZZP squaring; no downdraught removes the SQRT(ZZP) branch), so this
    config is deliberately empty and kept only for call-signature
    symmetry with the sibling modules.
    """


__physics_contract__ = {'summary': "Port of cuflxn sections 1.4-1.5: converts the ascent's absolute plume "
            'fluxes (M_u, PMFUS, PMFUQ, PMFUL) into environment-difference fluxes '
            'for cudtdqn, zeroes fluxes at and above cloud top and for '
            'non-convective columns, and applies the linear (in pressure) '
            'below-cloud-base taper of the mass flux and its fluxes down to the '
            'surface.',
 'inputs': {'M_u': 'kg m-2 s-1',
            'PMFUS': 'W m-2',
            'PMFUQ': 'kg m-2 s-1',
            'PMFUL': 'kg m-2 s-1',
            'PLUDE': 'kg m-2 s-1',
            'PDMFUP': 'kg m-2 s-1',
            'T_h': 'K',
            'q_h': 'kg kg-1',
            'geo_half': 'm2 s-2',
            'p_half': 'Pa',
            'ldcum': '1 (bool)',
            'ktype': '1 (1 deep, 2 shallow)',
            'k_cbot': '1 (level index)',
            'k_ctop': '1 (level index)'},
 'outputs': {'M_u': 'kg m-2 s-1',
             'PMFUS': 'W m-2 (difference flux)',
             'PMFUQ': 'kg m-2 s-1 (difference flux)',
             'PMFUL': 'kg m-2 s-1',
             'PLUDE': 'kg m-2 s-1',
             'PDMFUP': 'kg m-2 s-1'},
 'sign_convention': 'After section 1.4 the fluxes are difference fluxes: PMFUS -= '
                    'M_u*(cpd*T_h + geo_half), PMFUQ -= M_u*q_h, i.e. positive '
                    'upward flux of (dry static energy, moisture) excess over the '
                    'environment; cudtdqn divides their vertical divergence by the '
                    'full-level pressure thickness.',
 'conserves': ['energy', 'moisture'],
 'differentiable': True,
 'reference': 'OpenIFS cuflxn.F90 (main 8f6f722), sections 1.4 (lines 240-284) and '
              '1.5 (lines 292-345); FOELHMCU -> latent heat of vaporisation in the '
              'liquid-only LPHYLIN=.FALSE. branch.',
 'idealized_test': 'Single deep convecting column: fluxes above k_ctop and in '
                   'non-convective columns are exactly zero; below cloud base M_u '
                   'decreases linearly in (p_surf - p) to zero at the surface; '
                   'PMFUL vanishes below k_cbot+1.'}


def _f(x) -> jnp.ndarray:
    """Guard zero denominators (AD-safe floor); keeps the input dtype."""
    x = jnp.asarray(x)
    return jnp.maximum(x, jnp.asarray(_EPS, x.dtype))


def _gather(a: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    """Per-column single-level gather a[:, idx[:]] -> (ncol,)."""
    return jnp.take_along_axis(a, idx[:, None], axis=1)[:, 0]


def ifs_convective_fluxes(
    M_u, PMFUS, PMFUQ, PMFUL, PLUDE, PDMFUP,
    T_h, q_h, geo_half, p_half,
    ldcum, ktype, k_cbot, k_ctop,
    cfg: IFSFluxConfig = IFSFluxConfig(),
):
    """Sections 1.4 + 1.5 of cuflxn (see module docstring for scope)."""
    del cfg, ktype  # no tunables in the reduced branch; KTYPE!=3 guaranteed

    # The flux arrays are (ncol, nlev): one entry per IFS half level
    # JK = 1..KLEV, our index j = JK-1.  Only p_half carries the extra
    # surface interface PAPH(KLEV+1) at index nlev.
    nlev = M_u.shape[1]
    nlev1 = nlev
    j = jnp.arange(nlev1)[None, :]

    # ------------------------------------------------------------------
    # SECTION 1.4 (cuflxn.F90:240-284)
    # environmental subtraction + out-of-cloud zeroing.
    # In-cloud (IFS JK >= KCTOP) <-> our j >= k_ctop.  The downdraught
    # sub-branch (PMFD*/IDBAS, LLDDRAF) is identically zero here and is
    # skipped; PLGLAC is zero in the liquid-only branch.
    # ------------------------------------------------------------------
    incloud = ldcum[:, None] & (j >= k_ctop[:, None])

    # PMFUS = PMFUS - PMFU*(RCPD*PTENH + PGEOH)   (cuflxn.F90:250)
    # The source loop runs JK = KTOPM2..KLEV with KTOPM2 forced to 2, so our
    # index 0 (IFS JK = 1) is never written by this section.
    active = j >= 1
    PMFUS = jnp.where(active,
                      jnp.where(incloud,
                                PMFUS - M_u * (RCPD * T_h + geo_half), 0.0),
                      PMFUS)
    # PMFUQ = PMFUQ - PMFU*PQENH                  (cuflxn.F90:251)
    PMFUQ = jnp.where(active,
                      jnp.where(incloud, PMFUQ - M_u * q_h, 0.0), PMFUQ)
    M_u = jnp.where(active, jnp.where(incloud, M_u, 0.0), M_u)
    PMFUL = jnp.where(active, jnp.where(incloud, PMFUL, 0.0), PMFUL)

    # The source's ELSE branch also zeroes the "JK-1" quantities
    # PDMFUP(JL,JK-1) and PLUDE(JL,JK-1) (cuflxn.F90:271-273), i.e. on OUR
    # index grid column k is zeroed when level k+1 is out of cloud.
    # Pad on the left with True so column 0 (IFS index 0, never written by
    # the source because its level loop starts at JK>=2) is zeroed too --
    # consistent and harmless.
    # The ELSE branch also clears the JK-1 quantities (cuflxn.F90:271-273).
    # Fortran level JK is our j, so its JK-1 DESTINATION is j-1: destination k
    # is cleared when level k+1 is out of cloud.  Source JK = 2..KLEV maps onto
    # destinations 0..nlev-2, so the last destination is left untouched.
    outcloud_shift = jnp.pad(
        (~incloud)[:, 1:], ((0, 0), (0, 1)), constant_values=False)
    PDMFUP = jnp.where(outcloud_shift, 0.0, PDMFUP)
    PLUDE = jnp.where(outcloud_shift, 0.0, PLUDE)

    # ------------------------------------------------------------------
    # SECTION 1.5 (cuflxn.F90:292-345): linear decrease below cloud base.
    # IFS IKB = KCBOT  <-> our k_cbot (numerically equal, see docstring),
    # IFS IK  = IKB+1  <-> our column ikb1 = k_cbot + 1.
    # ------------------------------------------------------------------
    p_surf = p_half[:, nlev]                       # PAPH(JL,KLEV+1)
    ikb = k_cbot
    ikb1 = jnp.minimum(k_cbot + 1, nlev)           # gather-safe clamp

    # ---- step 1 (cuflxn.F90:297-322): write level IK from level IKB ----
    dp_base = _f(p_surf - _gather(p_half, ikb))    # PAPH(KLEV+1)-PAPH(IKB)
    zzp1 = ((p_surf - _gather(p_half, ikb1)) / dp_base)[:, None]

    mu_ikb = _gather(M_u, ikb)[:, None]
    s_ikb = _gather(PMFUS, ikb)[:, None]
    q_ikb = _gather(PMFUQ, ikb)[:, None]
    l_ikb = _gather(PMFUL, ikb)[:, None]

    mask1 = ldcum[:, None] & (j == ikb1[:, None])
    M_u = jnp.where(mask1, mu_ikb * zzp1, M_u)
    # LPHYLIN=.FALSE. branch (cuflxn.F90:314):
    # PMFUS(IK) = (PMFUS(IKB) - FOELHMCU(PTENH(IKB))*PMFUL(IKB))*ZZP
    PMFUS = jnp.where(mask1, (s_ikb - L_VAPOR * l_ikb) * zzp1, PMFUS)
    PMFUQ = jnp.where(mask1, (q_ikb + l_ikb) * zzp1, PMFUQ)
    PMFUL = jnp.where(mask1, 0.0, PMFUL)

    # ---- step 2 (cuflxn.F90:323-345): levels below IK from level IK ----
    # Vectorisation note: the Fortran loop runs JK = KTOPM2..KLEV and, for
    # JK > KCBOT+1, sets each level from the FIXED level IKB = KCBOT+1
    # (PMFU(JL,IKB) with IKB constant inside the loop), never from
    # previously written loop levels.  The level IKB = KCBOT+1 itself was
    # already finalised by step 1 above and is excluded from this loop
    # (condition JK > KCBOT+1), and this loop never writes it either.
    # Hence every level in the loop depends only on the post-step-1 state,
    # and a single jnp.where over all levels is exactly equivalent -- no
    # sequential dependency survives, so no python level loop is needed.
    dp_ikb1 = _f(p_surf - _gather(p_half, ikb1))   # PAPH(KLEV+1)-PAPH(IKB=KCBOT+1)
    # PAPH(JK) at our index j is p_half[:, j]; the surface interface
    # PAPH(KLEV+1) = p_half[:, nlev] is not one of the flux levels.
    zzp2 = ((p_surf[:, None] - p_half[:, :nlev]) / dp_ikb1[:, None])

    mask2 = (
        ldcum[:, None]
        & (j > ikb1[:, None])                      # JK > KCBOT+1
        & (j <= nlev - 1)                          # JK <= KLEV (source loop bound)
    )

    mu_ik = _gather(M_u, ikb1)[:, None]
    s_ik = _gather(PMFUS, ikb1)[:, None]
    q_ik = _gather(PMFUQ, ikb1)[:, None]

    M_u = jnp.where(mask2, mu_ik * zzp2, M_u)
    PMFUS = jnp.where(mask2, s_ik * zzp2, PMFUS)
    PMFUQ = jnp.where(mask2, q_ik * zzp2, PMFUQ)
    PMFUL = jnp.where(mask2, 0.0, PMFUL)

    return M_u, PMFUS, PMFUQ, PMFUL, PLUDE, PDMFUP
