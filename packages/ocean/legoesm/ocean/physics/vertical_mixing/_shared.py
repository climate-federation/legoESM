"""Shared vertical-mixing kernels.

Factored to remove the identical stack -> vmap -> unstack block that the
constant- and Richardson-coefficient schemes duplicated (they differ only in
the per-field diffusion function and the viscosity/diffusivity coefficients).
"""

from __future__ import annotations

import math
from typing import Callable

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.eos import (
    thermal_expansion_coeff,
    haline_contraction_coeff,
    eos_density_derivatives,
)

# EOS salinity floor [PSU] for the generic-EOS surface α/β autodiff path.
# sqrt(S)/S^1.5 EOS terms (veros_gsw, unesco80) have an INFINITE ∂ρ/∂S at S=0
# (land/edge fill cells), which a later 0·mask multiply would poison; floor S
# before differentiating.  Matches tke.compute_surface_buoyancy_P_diss_v.  Ocean
# S ≫ floor, so this is a no-op on wet cells.
_EOS_SALINITY_FLOOR = 1.0e-3

# Shared float32-eps floor for the vertical-mixing kernels.
_EPS = float(jnp.finfo(jnp.float32).eps)

# --- Latitude-dependent internal-wave background (Gregg et al. 2003 / CVMix) ---
# Reference latitude at which the Henyey-Wright-Flatte scaling is normalised to
# 1 (Gregg et al. 2003; MOM6 MOM_bkgnd_mixing.F90 Henyey_IGW_background).
_GREGG_REF_LAT_DEG = 30.0
# |f_30| = |2 Omega sin(30 deg)| — the Coriolis magnitude at the reference lat.
_F_CORIOLIS_REF = 2.0 * constants.Omega * math.sin(math.radians(_GREGG_REF_LAT_DEG))
# Floor on |sin(lat)| so |f| never vanishes at the equator (avoids 0/0 and an
# unbounded acosh argument); tiny, so the equatorial value is set by K_bg_eq.
_SIN_LAT_FLOOR = 1.0e-6
# acosh(x) has an INFINITE slope at x = 1; clamp its argument to 1 + this eps so
# the domain guard (x >= 1) stays autodiff-safe (finite reverse-mode gradient).
_ACOSH_ARG_EPS = 1.0e-6
# N^2 [1/s^2] floor before sqrt: keeps the sqrt gradient finite at N2 -> 0 while
# staying far below any real oceanic N^2 (so N -> ~0 in an unstratified column,
# driving the Gregg scaling L -> 0 => K_bg -> K_bg_eq there).
_N2_FLOOR = 1.0e-12


def vertical_shear_squared(
    u_cell: jnp.ndarray, v_cell: jnp.ndarray, dz_half: jnp.ndarray,
) -> jnp.ndarray:
    """Compute ``|du/dz|^2 + |dv/dz|^2`` at interfaces.

    Shared by the TKE (Gaspar/Burchard) and CATKE prognostic closures.

    Parameters
    ----------
    u_cell, v_cell : (..., nlev) — cell-centre velocities.
    dz_half : (..., nlev-1) — distance between cell centres.

    Returns
    -------
    S2 : (..., nlev-1) — squared vertical shear at interfaces.
    """
    dz_safe = jnp.maximum(dz_half, _EPS)
    du = (u_cell[..., 1:] - u_cell[..., :-1]) / dz_safe
    dv = (v_cell[..., 1:] - v_cell[..., :-1]) / dz_safe
    return du * du + dv * dv


def vertical_shear_burchard(
    u_now: jnp.ndarray, v_now: jnp.ndarray,
    u_before: jnp.ndarray, v_before: jnp.ndarray,
    dz_half: jnp.ndarray,
) -> jnp.ndarray:
    r"""Burchard (2002) energy-conserving now×before shear cross term.

    Faithful port of the TIME discretization NEMO's ``zdf_sh2`` uses
    (zdfsh2.F90:44-92, the no-Stokes-drift branch):

    .. math::

        sh2 = \left(\frac{du_{now}}{dz}\right)\!
              \left(\frac{du_{before}}{dz}\right)
            + \left(\frac{dv_{now}}{dz}\right)\!
              \left(\frac{dv_{before}}{dz}\right)

    i.e. the shear production entering the TKE budget is linear in the
    NOW gradient and linear in the BEFORE (leap-frog) gradient of the
    SAME velocity component — not the squared now-only form
    (:func:`vertical_shear_squared`). This is the energy-conserving
    discretization consistent with the implicit-vertical-friction/
    leap-frog time stepping (Burchard 2002): the shear production
    entering the TKE budget is built from the SAME du/dz, dv/dz the
    implicit friction solve actually applied between the before and now
    states, so the KE removed by friction and the TKE produced by shear
    balance exactly (to the implicit solve's own truncation).

    NEMO additionally averages ``avm`` at u/v-points then face-interpolates
    to the T-point before applying this shear (zdfsh2.F90:80-90, wet-only
    2-2 coast masking) — a staggered-C-grid detail with no analog on
    legoESM's cell-centred TKE closure (which already applies a SINGLE
    per-interface ``K_M`` uniformly, exactly as
    :func:`vertical_shear_squared`'s caller does); only the TIME
    discretization (now×before vs now²) is transcribed here.

    Parameters
    ----------
    u_now, v_now, u_before, v_before : (..., nlev) — cell-centre velocities
        at the NOW and (leap-frog) BEFORE time levels.
    dz_half : (..., nlev-1) — distance between cell centres.

    Returns
    -------
    sh2 : (..., nlev-1) — Burchard shear production at interfaces. May be
        NEGATIVE where the now/before gradients have opposite sign (a
        genuine feature of the energy-conserving form, unlike the
        squared-now form which is always >= 0); the caller (P_s = K_M·sh2)
        should treat this as a signed production term, matching NEMO's
        ``en += rn_Dt·p_sh2`` (zdftke.F90:414, no clipping).
    """
    dz_safe = jnp.maximum(dz_half, _EPS)
    du_now = (u_now[..., 1:] - u_now[..., :-1]) / dz_safe
    dv_now = (v_now[..., 1:] - v_now[..., :-1]) / dz_safe
    du_before = (u_before[..., 1:] - u_before[..., :-1]) / dz_safe
    dv_before = (v_before[..., 1:] - v_before[..., :-1]) / dz_safe
    return du_now * du_before + dv_now * dv_before


def vertical_shear_face_native(
    u_face_now: jnp.ndarray, v_face_now: jnp.ndarray,
    u_face_before: jnp.ndarray, v_face_before: jnp.ndarray,
    dz_half: jnp.ndarray,
    u_mask: jnp.ndarray, v_mask: jnp.ndarray,
) -> jnp.ndarray:
    r"""NEMO ``zdf_sh2`` face-native shear "equivalent shear-squared" — the
    T-point-collapse-before-differencing fix (#1226 ``sh2_walk.py`` Candidate
    E/F; see ``sh2_walk.py`` header for the A/B evidence: corr 0.982 / ratio
    0.975 unrestricted, ratio 0.912 restricted-to-signal, vs the naive
    T-collapse-first approximation's corr 0.963 / ratio 0.556).

    NEMO differences the RAW u-/v-FACE velocities first, one vertical
    difference PER FACE, and combines the two adjacent faces' shears onto
    the T-point ONLY AFTER squaring/cross-multiplying (zdfsh2.F90:78-94,
    no-Stokes-drift branch, ``ln_stshear=.false.``/``ln_wave=.false.`` —
    namelist_ref:597,212, confirmed live for DINO):

    .. code-block:: fortran

        ! zdfsh2.F90:78-91 (DO_2D(1,0,1,0), jk=2,jpkm1)
        zsh2u(ji,jj) = ( avm(ji+1,jj,jk) + avm(ji,jj,jk) )              &
           &         * ( uu(ji,jj,jk-1,Kmm) - uu(ji,jj,jk,Kmm) )        &
           &         * ( uu(ji,jj,jk-1,Kbb) - uu(ji,jj,jk,Kbb) )        &
           &         / ( e3uw(ji,jj,jk,Kmm) * e3uw(ji,jj,jk,Kbb) ) * wumask(ji,jj,jk)
        zsh2v(ji,jj) = analogous at v-faces                              ! :85-91
        ! zdfsh2.F90:92-94 (DO_2D(0,0,0,0)) -- T-point combination, coast
        ! doubling weight "2 - mask*mask" (=2 at a coast, 1 in the open ocean;
        ! NEMO's own comment: "wmask useless as zsh2 are masked")
        p_sh2(ji,jj,jk) = 0.25 * (   ( zsh2u(ji-1,jj) + zsh2u(ji,jj) )        &
           &                       * ( 2. - umask(ji-1,jj,jk)*umask(ji,jj,jk) )   &
           &                     + ( zsh2v(ji,jj-1) + zsh2v(ji,jj) )          &
           &                       * ( 2. - vmask(ji,jj-1,jk)*vmask(ji,jj,jk) )   )

    legoESM's u-/v-face arrays use the WEST-face convention (``u[..., i]`` is
    the west face of ``T[..., i]``, ``u[..., i+1]`` its east face —
    :func:`legoesm.grids.operators_latlon_cgrid.interp_cell_to_uface`), the
    mirror image of NEMO's east-face-of-T(i) indexing; the T-point
    combination below uses the matching pair ``(zsh2u[i], zsh2u[i+1])``
    (NOT NEMO's literal ``(i-1, i)``) so the SAME two faces bracketing T(i)
    are combined either way.

    Two deliberate scope limits, both already documented at the same class
    of simplification by :func:`vertical_shear_burchard` /
    ``tke._prandtl_number``'s ``"nemo_ri"`` docstring, kept here rather than
    re-derived:

    * ``avm`` is NOT face-averaged — this function does not carry a
      viscosity at all (mirrors :func:`vertical_shear_squared`'s contract:
      it returns a bare shear-production-EQUIVALENT quantity that the caller
      multiplies by its own single per-interface ``K_M``, exactly as
      ``P_s = K_M · shear_sq`` already does downstream). NEMO's own
      face-averaged-``avm`` complicates unit tracking without changing the
      dominant effect the #1226 walk isolated (Candidate B: static-vs-live
      metric was negligible; the avm face-averaging was never isolated as
      its own candidate because ``_vertical_shear_squared`` already shares
      the single-``K_M`` simplification with every other scheme on this
      C-grid).
    * The vertical metric ``dz_half`` is the caller's STATIC reference
      spacing (T-point ``dz_half_ref·J``), not NEMO's LIVE
      ``e3uw(Kmm)·e3uw(Kbb)`` QCO-stretched product — Candidate B measured
      this substitution as negligible for DINO (corr 1.000, ratio 0.9999;
      SSH-driven r3u/r3v ~ 1e-4..1e-3), so using the static metric here is
      an oracle-supported approximation, not an untested guess.

    Parameters
    ----------
    u_face_now, v_face_now, u_face_before, v_face_before : arrays.
        ``u_face`` : ``(n_lat, n_lon+1, nlev)``, ``v_face`` :
        ``(n_lat+1, n_lon, nlev)`` — RAW (uncollapsed) C-grid face
        velocities at the NOW and (leap-frog) BEFORE time levels,
        west-face-of-T(i) / south-face-of-T(j) convention.
    dz_half : ``(n_lat, n_lon, nlev-1)`` — T-point distance between cell
        centres (static reference metric; see scope limit above).
    u_mask, v_mask : ``(n_lat, n_lon+1, nlev)`` / ``(n_lat+1, n_lon, nlev)``
        — per-LEVEL wet face masks (``umask``/``vmask`` in NEMO's sense:
        product of the two adjacent T-cells' wet flags at that level, e.g.
        :func:`compute_face_masks_3d`, or a 2-D static mask broadcast to
        every level by the caller). Used BOTH for the coast-doubling
        weight (matching NEMO's ``umask``/``vmask``) and, after this
        function derives the vertical-adjacency product internally
        (``wumask``/``wvmask``, dommsk.F90:176-182), for the interior
        zero-out at the seafloor transition.

    Returns
    -------
    shear_sq_equivalent : ``(n_lat, n_lon, nlev-1)`` — face-native
        shear-production equivalent at T-point interfaces, SIGNED (the
        now×before cross term can be negative, mirroring
        :func:`vertical_shear_burchard`). The caller forms
        ``P_s = K_M · shear_sq_equivalent`` exactly as the squared/Burchard
        forms do.
    """
    dz_safe = jnp.maximum(dz_half, _EPS)
    dz_sq = dz_safe * dz_safe

    # dommsk.F90:176-182 wumask/wvmask: interior jk = mask(jk)*mask(jk-1)
    # (vertical adjacency of the SAME face across consecutive levels).
    # Interfaces here (index k, k=0..nlev-2) sit BETWEEN T-levels k and
    # k+1, so the two mask levels bracketing interface k are (k, k+1) —
    # matching zdf_sh2's jk-1/jk pair (its jk indexes the UPPER level of
    # the pair, one-based; our 0-based interface k plays the same role).
    wumask = u_mask[..., :-1] * u_mask[..., 1:]
    wvmask = v_mask[..., :-1] * v_mask[..., 1:]

    def _face_shear_over_dzsq(u_face_n, u_face_b, dz_sq_face):
        # Vertical difference AT the face, one difference per face level —
        # NOT collapsed to the T-point first (zdfsh2.F90:80-84/85-90).
        du_now = u_face_n[..., :-1] - u_face_n[..., 1:]
        du_bef = u_face_b[..., :-1] - u_face_b[..., 1:]
        return du_now * du_bef / dz_sq_face

    # u-faces share the T grid's lat axis (0) and vertical axis (-1); only
    # the lon axis (-2) is n_lon+1 instead of n_lon. dz_sq is T-shaped
    # (n_lat, n_lon, nlev-1); Candidate B (static-vs-live metric) measured
    # this substitution as negligible for DINO (corr 1.000, ratio 0.9999),
    # so reusing each T column's own dz at its adjacent face (no separate
    # u/v-face metric) is an oracle-supported approximation. Pad by
    # repeating the last column/row (the periodic-fold / wall face has no
    # T-neighbour on that side; its shear is masked out by wumask/wvmask
    # there in every recipe this option targets, so the padded dz value is
    # never selected).
    dz_sq_u = jnp.concatenate([dz_sq, dz_sq[:, -1:, :]], axis=1)
    dz_sq_v = jnp.concatenate([dz_sq, dz_sq[-1:, :, :]], axis=0)

    zsh2u = _face_shear_over_dzsq(u_face_now, u_face_before, dz_sq_u) * wumask
    zsh2v = _face_shear_over_dzsq(v_face_now, v_face_before, dz_sq_v) * wvmask

    # Coast-doubling weight "2 - mask*mask" (zdfsh2.F90:92-94): =2 at a
    # coast (one neighbouring face dry), =1 in the open ocean (both wet).
    # umask/vmask here are the RAW per-level face masks (not wumask/wvmask
    # — matching NEMO's literal ``umask(ji-1,jj,jk)*umask(ji,jj,jk)``,
    # zdfsh2.F90:93), evaluated at the interior interface's UPPER T-level
    # (index k+1, matching zdf_sh2's jk) since NEMO's own comment notes
    # "wmask useless as zsh2 are masked" — the coast weight only needs to
    # know which NEIGHBOUR is dry, not repeat the vertical wet test.
    coast_u = 2.0 - u_mask[:, :-1, 1:] * u_mask[:, 1:, 1:]
    coast_v = 2.0 - v_mask[:-1, :, 1:] * v_mask[1:, :, 1:]

    # T-point combination: T(i) is bracketed by faces i (west) and i+1
    # (east) under legoESM's convention (the mirror of NEMO's (i-1, i)
    # east-face-of-T(i) pairing — see docstring).
    p_sh2 = 0.25 * (
        (zsh2u[:, :-1, :] + zsh2u[:, 1:, :]) * coast_u
        + (zsh2v[:-1, :, :] + zsh2v[1:, :, :]) * coast_v
    )
    return p_sh2


def avm_weighted_shear_production(
    u_face_now: jnp.ndarray, v_face_now: jnp.ndarray,
    u_face_before: jnp.ndarray, v_face_before: jnp.ndarray,
    dz_half: jnp.ndarray,
    u_mask: jnp.ndarray, v_mask: jnp.ndarray,
    kappaM_T: jnp.ndarray,
) -> jnp.ndarray:
    r"""NEMO ``zdf_sh2`` shear-PRODUCTION term ``p_sh2`` with the viscosity
    face-averaged INSIDE the face sum, exactly as ``zdfsh2.F90:80-94``
    (#1455 sh2 chain-walk avm-weighting gap — see the module docstring on
    :func:`vertical_shear_face_native`, which explicitly scopes OUT this
    avm face-averaging as "no face-averaging analog"; this function is that
    analog, using array rolls on legoESM's already-existing 3-D ``K_M``, not
    new staggered state).

    .. code-block:: fortran

        ! zdfsh2.F90:80-94 (no-Stokes-drift branch)
        zsh2u(ji,jj) = ( avm(ji+1,jj,jk) + avm(ji,jj,jk) )   &   ! avm INSIDE
           &         * ( uu(jk-1,Kmm)-uu(jk,Kmm) ) * ( uu(jk-1,Kbb)-uu(jk,Kbb) ) &
           &         / ( e3uw(jk,Kmm)*e3uw(jk,Kbb) ) * wumask(jk)
        zsh2v analogous at v-faces
        p_sh2(ji,jj,jk) = 0.25 * ( (zsh2u(ji-1,jj)+zsh2u(ji,jj))*(2-umask*umask)
                                  + (zsh2v(ji,jj-1)+zsh2v(ji,jj))*(2-vmask*vmask) )

    NB (verified by the synthetic self-consistency control, #1455 step 2):
    NEMO's own comment at :79 reads "2 x shear production ... (energy
    conserving form)" — ``avm(ji+1)+avm(ji)`` is a face SUM, not an average,
    so for spatially UNIFORM ``kappaM_T = K0`` this function returns EXACTLY
    ``2 * K0 * shear_sq_tpoint`` (``shear_sq_tpoint`` = the T-collapsed
    output of :func:`vertical_shear_face_native`), NOT
    ``K0 * shear_sq_tpoint``. This is NOT a bug — it is NEMO's own
    documented "energy conserving form" convention (verified against the
    literal Fortran, not inferred) — so this function returns the full
    ``p_sh2`` PRODUCT directly rather than a ``shear_sq`` the caller
    multiplies by a bare ``K_M``, precisely because that external multiply
    cannot reproduce the avm-INSIDE-the-face-sum structure.

    Parameters
    ----------
    u_face_now, v_face_now, u_face_before, v_face_before, dz_half, u_mask,
    v_mask : as :func:`vertical_shear_face_native`.
    kappaM_T : ``(n_lat, n_lon, nlev-1)`` — T-point (single per-interface)
        viscosity — face-summed here via array rolls, matching NEMO's
        ``avm(ji+1,jj,jk)+avm(ji,jj,jk)`` exactly; NOT new staggered state.
        NB time level: NEMO computes ``zdf_sh2`` ONCE per step from the
        previous-step ``p_avm``; legoESM's orchestrator wires the CURRENT
        sub-iteration ``K_M_curr`` (same convention its pre-existing tpoint
        path uses) — a documented deviation of the sub-iteration loop
        structure, not of this function.

    Returns
    -------
    p_sh2 : ``(n_lat, n_lon, nlev-1)`` — avm-weighted shear production
        [m^2/s^3] at T-point interfaces (matches ``tke_dump_sh2.bin``
        units), SIGNED.
    """
    dz_safe = jnp.maximum(dz_half, _EPS)
    dz_sq = dz_safe * dz_safe
    wumask = u_mask[..., :-1] * u_mask[..., 1:]
    wvmask = v_mask[..., :-1] * v_mask[..., 1:]

    def _face_shear_over_dzsq(u_face_n, u_face_b, dz_sq_face):
        du_now = u_face_n[..., :-1] - u_face_n[..., 1:]
        du_bef = u_face_b[..., :-1] - u_face_b[..., 1:]
        return du_now * du_bef / dz_sq_face

    dz_sq_u = jnp.concatenate([dz_sq, dz_sq[:, -1:, :]], axis=1)
    dz_sq_v = jnp.concatenate([dz_sq, dz_sq[-1:, :, :]], axis=0)

    zsh2u_bare = _face_shear_over_dzsq(u_face_now, u_face_before, dz_sq_u) * wumask
    zsh2v_bare = _face_shear_over_dzsq(v_face_now, v_face_before, dz_sq_v) * wvmask

    coast_u = 2.0 - u_mask[:, :-1, 1:] * u_mask[:, 1:, 1:]
    coast_v = 2.0 - v_mask[:-1, :, 1:] * v_mask[1:, :, 1:]

    # Face-sum kappaM_T (T-point) onto each u-/v-face via array rolls —
    # NEMO's avm(ji+1,jj,jk)+avm(ji,jj,jk), a SUM not a mean (matches the
    # "2 x" comment above). Zonal (u-face) boundary pads WRAP periodically:
    # the seam face between T[-1] and T[0] gets kappa[-1]+kappa[0] — correct
    # under BOTH seam conventions (a land-column seam masks it via wumask
    # anyway; the nemo_faithful all-wet seam needs the true wrapped sum;
    # #1455 review N1). Meridional (v-face) boundaries are walls in every
    # recipe this option targets — edge-repeat there is never selected
    # (masked by wvmask/coast_v; same idiom as dz_sq_v above).
    kM_left_u = jnp.concatenate([kappaM_T[:, -1:, :], kappaM_T], axis=1)
    kM_right_u = jnp.concatenate([kappaM_T, kappaM_T[:, :1, :]], axis=1)
    kM_face_u = kM_left_u + kM_right_u          # (n_lat, n_lon+1, nlev-1)
    kM_left_v = jnp.concatenate([kappaM_T[:1, :, :], kappaM_T], axis=0)
    kM_right_v = jnp.concatenate([kappaM_T, kappaM_T[-1:, :, :]], axis=0)
    kM_face_v = kM_left_v + kM_right_v          # (n_lat+1, n_lon, nlev-1)

    zsh2u = kM_face_u * zsh2u_bare
    zsh2v = kM_face_v * zsh2v_bare

    p_sh2 = 0.25 * (
        (zsh2u[:, :-1, :] + zsh2u[:, 1:, :]) * coast_u
        + (zsh2v[:-1, :, :] + zsh2v[1:, :, :]) * coast_v
    )
    return p_sh2


def richardson_number(
    N2: jnp.ndarray,
    u_cell: jnp.ndarray,
    v_cell: jnp.ndarray,
    dz_actual: jnp.ndarray,
    *,
    eps: float,
    clip_negative: bool = False,
) -> jnp.ndarray:
    """Gradient Richardson number ``Ri = N^2 / S^2`` at interfaces.

    Single source for the Richardson-scheme (``richardson.py``) and KPP
    interior-shear (``kpp.py``) blocks, which computed this from byte-identical
    inline code (#518 item 5).  ``N2`` is supplied by the caller (both already
    call ``eos.compute_buoyancy_frequency``; KPP also reuses ``N2`` for its
    static-instability term).

    The squared shear uses the floor-on-the-SQUARED-denominator form
    ``S^2 = (du^2 + dv^2) / max(dz_half^2, eps)`` with ``du = u[k] - u[k+1]``
    — preserved bit-for-bit from both call sites.  NOTE: this floor placement
    differs from :func:`vertical_shear_squared` (which floors the LINEAR
    ``dz_half`` before dividing); the two are equal away from vanishing
    ``dz_half`` but diverge in sub-eps-thin layers, so they are intentionally
    NOT merged here — reconciling the floor convention is a numerics-affecting
    change for a separate PR.

    Parameters
    ----------
    N2 : (..., nlev-1) — buoyancy frequency squared at interfaces.
    u_cell, v_cell : (..., nlev) — cell-centre velocities.
    dz_actual : (..., nlev) — actual layer thickness (z* Jacobian applied).
    eps : float — denominator floor (caller's scheme eps; applied to both
        ``dz_half**2`` and ``S2``).
    clip_negative : bool — when True, clip ``Ri`` to ``>= 0`` (the Richardson
        scheme's "unstable -> max mixing" convention); KPP passes False and
        clamps downstream via ``Ri / Ri_0``.

    Returns
    -------
    Ri : (..., nlev-1)
    """
    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
    du = u_cell[..., :-1] - u_cell[..., 1:]
    dv = v_cell[..., :-1] - v_cell[..., 1:]
    S2 = (du**2 + dv**2) / jnp.maximum(dz_half**2, eps)
    Ri = N2 / jnp.maximum(S2, eps)
    if clip_negative:
        Ri = jnp.maximum(Ri, 0.0)
    return Ri


def compute_N2(
    rho_cell: jnp.ndarray, dz_half: jnp.ndarray, rho_0: float,
    g: float = constants.g,
    *,
    T_cell: jnp.ndarray | None = None,
    S_cell: jnp.ndarray | None = None,
    p_cell: jnp.ndarray | None = None,
    dz_ref: jnp.ndarray | None = None,
    jacobian: jnp.ndarray | None = None,
    eos_fn=None,
    n2_mode: str = "insitu",
    adiabatic_over_dz_half: bool = False,
    t_depth: jnp.ndarray | None = None,
    w_depth: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """N^2 at interfaces (shared by the TKE and CATKE closures).

    ``n2_mode="insitu"`` (default): from cell-centre *in-situ* density,
    ``N^2 = -(g/rho_0) drho/dz`` with z positive upward, clipped >= 0 (the
    in-situ contrast carries compressibility and is biased too stable, so its
    sign is not a reliable convection trigger).  ``n2_mode="adiabatic"``: the
    true static stability via adiabatic parcel displacement to the upper cell's
    pressure (delegating to ``eos.compute_buoyancy_frequency_adiabatic``),
    SIGNED (N^2 < 0 marks a statically unstable interface — the convection
    trigger CATKE / signed-N2 TKE need); requires ``T_cell``/``S_cell``/
    ``p_cell``/``dz_ref``/``jacobian``.  ``adiabatic_over_dz_half=True`` divides
    the adiabatic contrast by the caller's ``dz_half`` (the Veros ``dzw`` slot).

    Returns
    -------
    N2 : (..., nlev-1). Clipped >= 0 for ``"insitu"``; signed for ``"adiabatic"``.
    """
    if n2_mode in ("insitu", "insitu_signed"):
        dz_safe = jnp.maximum(dz_half, _EPS)
        # drho/dz with z positive upward — negative for stable stratification.
        drho_dz = (rho_cell[..., :-1] - rho_cell[..., 1:]) / dz_safe
        N2 = -g / rho_0 * drho_dz
        # "insitu": clipped >= 0 (in-situ contrast is biased too stable; sign is
        # not a reliable convection trigger — TKE's default).  "insitu_signed":
        # SIGNED (N2<0 marks static instability), the cheap convection trigger
        # for convection-aware closures (CATKE) without the adiabatic EOS call.
        return N2 if n2_mode == "insitu_signed" else jnp.maximum(N2, 0.0)
    if n2_mode == "adiabatic":
        if (T_cell is None or S_cell is None or p_cell is None
                or dz_ref is None or jacobian is None):
            raise ValueError(
                "n2_mode='adiabatic' requires T_cell, S_cell, p_cell "
                "(cell-centre pressure [Pa]), dz_ref and jacobian to "
                "displace parcels through the EOS."
            )
        from legoesm.ocean.eos import compute_buoyancy_frequency_adiabatic
        return compute_buoyancy_frequency_adiabatic(
            T_cell, S_cell, p_cell, dz_ref, jacobian,
            eos_fn=eos_fn, rho_ref=rho_0, g=g,
            dz_half=dz_half if adiabatic_over_dz_half else None,
        )
    if n2_mode == "nemo_bn2":
        # NEMO eosbn2 bn2 (S-EOS): local alpha,beta at each cell's own gdept,
        # interpolated to the w-point by the geometric zrw weight (SIGNED).
        # NEMO's rn2 feeds BOTH zdfevd and zdftke, so the TKE closure consumes
        # the same trigger as convection.
        if (T_cell is None or S_cell is None
                or t_depth is None or w_depth is None):
            raise ValueError(
                "n2_mode='nemo_bn2' requires T_cell, S_cell and the geometric "
                "depth ladders t_depth (gdept) / w_depth (interior gdepw) — "
                "the k_profiles TKE caller threads them from "
                "eos.nemo_bn2_depth_ladders(z_coord)."
            )
        # NemoSEOSConfig() defaults (the DINO/Kamm set) — matching the density
        # path, where make_eos_fn's "nemo_seos" branch also has no custom-
        # coefficient threading from any recipe. Thread a cfg through here the
        # day a recipe carries non-default S-EOS coefficients.
        # NEMO evaluates alpha/beta at the LIVE gdept(Kmm) = gdept_0*(1+r3t)
        # and divides by the LIVE e3w(Kmm) = e3w_0*(1+r3t), r3t = eta/ht_0
        # (domzgr_substitute.h90:131,139; eosbn2.F90 rab_3d_t/bn2_t).  The
        # caller (k_profiles.py / enhanced_diffusion.py) applies that stretch
        # via eos.nemo_bn2_live_ladders BEFORE this call, and
        # compute_buoyancy_frequency_nemo_bn2 derives e3w as diff(t_depth)
        # internally (NEMO's e3w_0 IS the gdept_0 centre-difference, verified
        # exactly against mesh_mask), so the stretch propagates through
        # alpha/beta AND e3w -- no separate division here (that would
        # double-count it).  NB the stretch factor is the LOCAL 1+eta/H_bathy,
        # NOT legoESM's z* Jacobian (eta+H)/H_max -- see the warning in
        # eos.nemo_bn2_live_ladders (#1226).
        from legoesm.ocean.eos import compute_buoyancy_frequency_nemo_bn2
        return compute_buoyancy_frequency_nemo_bn2(
            T_cell, S_cell, t_depth, w_depth, g=g,
        )
    raise ValueError(
        f"Unknown n2_mode={n2_mode!r}; expected 'insitu', 'insitu_signed', "
        f"'adiabatic' or 'nemo_bn2'."
    )


def _safe_acosh(x: jnp.ndarray) -> jnp.ndarray:
    """``arccosh`` clamped into its domain with a finite edge slope.

    ``acosh`` is defined for ``x >= 1`` and has an INFINITE derivative at
    ``x = 1``.  Clamp the argument to ``1 + _ACOSH_ARG_EPS`` so the reverse-mode
    gradient stays finite (no ``0*inf`` NaN) while preserving the physical floor
    ``acosh -> 0`` as ``x -> 1`` (weak stratification / near the equator).
    """
    return jnp.arccosh(jnp.maximum(x, 1.0 + _ACOSH_ARG_EPS))


def latitude_background_diffusivity(lat_deg, N2, cfg):
    r"""Latitude / stratification-scaled background diffusivity ``K_bg(lat, N)``.

    Gregg et al. (2003) Henyey-Wright-Flatte internal-wave scaling, as used by
    CVMix ``bkgnd`` (``lvary_horizontal``) and MOM6 ``Henyey_IGW_background``:
    the diapycnal background diffusivity is REDUCED toward the equator, where
    the Coriolis parameter vanishes and internal-wave breaking is suppressed::

        f          = 2 Omega sin(theta)                 (Coriolis parameter)
        L(theta,N) = |f| acosh(N/|f|)
                     / ( |f_30| acosh(N_ref/|f_30|) )    (Gregg 2003 shape)
        K_bg       = K_bg_eq + (K_bg_pole - K_bg_eq) * clip(L, 0, 1)

    with ``|f_30| = |2 Omega sin(30 deg)|`` the Coriolis magnitude at the 30-deg
    normalisation latitude, ``N = sqrt(max(N2, 0))`` the local buoyancy
    frequency and ``N_ref = cfg.N_ref`` the Gregg reference stratification.
    ``L`` -> 0 at the equator (``|f| acosh(N/|f|) -> 0`` as ``|f| -> 0``), ``~1``
    near +/-30 deg for ``N ~ N_ref`` and ``> 1`` poleward (clipped), so ``K_bg``
    rises from ``K_bg_eq`` (equator) toward ``K_bg_pole``.  In the WELL-
    STRATIFIED regime ``|f| << N`` (typical ocean) the shape ``|f| acosh(N/|f|)``
    is monotone increasing in ``|f|`` (hence non-decreasing in ``|lat|`` up to
    the ``K_bg_pole`` cap).  For WEAK stratification ``N ~ |f|`` the raw
    Henyey shape turns over at high latitude (a known property of the Gregg
    formula), so ``K_bg`` is not strictly monotone there — but it always stays
    BOUNDED in ``[K_bg_eq, K_bg_pole]`` and finite.

    Sign / range: ``K_bg >= 0`` (``0 <= clip(L) <= 1``, ``K_bg_eq, K_bg_pole >
    0``).  A scalar diffusivity magnitude — no directional sign convention.

    Guards (traced-safe, differentiable):
      * ``|sin(lat)|`` floored at ``_SIN_LAT_FLOOR`` so ``|f|`` never vanishes at
        the equator (no ``0/0``, no unbounded acosh argument);
      * both ``acosh`` arguments clamped to ``>= 1 + _ACOSH_ARG_EPS`` (domain +
        finite gradient at the ``x = 1`` cusp, via :func:`_safe_acosh`);
      * ``N2`` floored at ``_N2_FLOOR`` before the sqrt (finite sqrt gradient in
        ``N2 -> 0`` convecting columns; the floor is far below any real oceanic
        ``N^2`` so an unstratified column returns ``K_bg_eq``).

    Parameters
    ----------
    lat_deg : (col...,) column latitudes [deg].  Aligned with the LEADING
        column axes of ``N2`` (latitude varies over axis 0), so both a 1-D
        ``(n_lat,)`` and a full 2-D ``(n_lat, n_lon)`` field broadcast correctly.
    N2 : (col..., nlev-1) buoyancy frequency squared at interior interfaces
        [1/s^2].
    cfg : ``ConstantVerticalMixingConfig`` — reads ``K_bg_eq``, ``K_bg_pole``
        [m^2/s] and ``N_ref`` [1/s].

    Returns
    -------
    K_bg : (col..., nlev-1) latitude/N-scaled background diffusivity [m^2/s].
    """
    dtype = N2.dtype
    ncol = N2.ndim - 1
    lat = jnp.asarray(lat_deg, dtype)
    # Align latitude with the LEADING column axes, then add the interface axis
    # so it broadcasts against N2 (col..., nlev-1) for a 1-D or 2-D lat input.
    lat = lat.reshape(lat.shape + (1,) * (ncol - lat.ndim))[..., None]
    sin_abs = jnp.maximum(jnp.abs(jnp.sin(jnp.deg2rad(lat))), _SIN_LAT_FLOOR)
    f_abs = 2.0 * constants.Omega * sin_abs
    # Local buoyancy frequency; N2 may be <= 0 in convecting columns -> the
    # acosh domain clamp then sends L -> 0 (so K_bg -> K_bg_eq).
    N_local = jnp.sqrt(jnp.maximum(N2, _N2_FLOOR))
    num = f_abs * _safe_acosh(N_local / f_abs)
    # Normalisation at (30 deg, N_ref).  N_ref is a fixed reference (excluded
    # from training) so this folds to a compile-time scalar.
    denom = _F_CORIOLIS_REF * _safe_acosh(
        jnp.asarray(cfg.N_ref, dtype) / _F_CORIOLIS_REF)
    s = jnp.clip(num / denom, 0.0, 1.0)
    K_bg = cfg.K_bg_eq + (cfg.K_bg_pole - cfg.K_bg_eq) * s
    # Exact physical bound: the affine interpolation leaves a ~5e-13 float
    # residual above K_bg_pole at a saturated column even in x64 (clip(s)
    # bounds s, not the rounded product+sum).  K_bg_pole IS the maximum
    # background, so clamp K itself — the test asserts the bound to 1e-15
    # (codex batch2 catch: the clip was lost to a worktree race and the
    # merged #926 carried the strict test WITHOUT it).
    K_bg = jnp.clip(K_bg, cfg.K_bg_eq, cfg.K_bg_pole)
    return K_bg.astype(dtype)


def surface_buoyancy_flux(
    q_net,
    fw,
    salt,
    T_sfc: jnp.ndarray,
    S_sfc: jnp.ndarray,
    *,
    g: float,
    rho_0: float,
    c_sw: float,
    real_salt_in_qs: bool,
    eos_fn=None,
):
    """Surface buoyancy flux ``B_f`` [m^2/s^3] (>0 destabilising) + the kinematic
    surface heat / salt fluxes for the KPP / CATKE boundary-layer closures.

    Single grid-agnostic source for the surface-forcing buoyancy block
    (#518 item 1).  Previously three byte-identical-in-logic copies existed:
    the lat-lon ``integration.py`` inline, ``k_profiles._surface_buoyancy_flux``,
    and ``mpas_integration._mpas_surface_buoyancy_flux``.

    Sign convention (KPP / CATKE: ``B_f > 0`` = unstable = convection):
    ``B_f = -g*alpha*Q_T + g*beta*Q_S`` where ``Q_T = q_net/(rho_0*c_sw)`` and
    ``Q_S`` is the kinematic salt flux.  ``alpha``/``beta`` are the EOS
    thermal-expansion / haline-contraction coefficients at the surface
    (``p = 0``).

    ``real_salt_in_qs`` (the ONE deliberate grid difference):

    * ``True`` (lat-lon): the real brine salt-mass flux feeds BOTH the surface
      buoyancy AND the returned non-local salt flux ``Q_sfc_S`` — they are
      accumulated into a single ``Q_sfc_S`` and ``B_salt = g*beta*Q_sfc_S``.
    * ``False`` (MPAS): the real salt feeds the surface BUOYANCY ONLY; the
      returned ``Q_sfc_S`` carries the freshwater (virtual-salt) term ONLY,
      starting from an explicit ``0`` baseline (NOT ``None``) so the KPP caller
      does not gradient-diagnose the non-local salt flux.  The real-salt
      injection is instead the mass-exact floored-h_k source in
      ``mpas_ocean_baroclinic_tendencies`` (adding it to the non-local term too
      would double-count and break partial-cell mass conservation).

    The float-operation ORDER per branch is preserved bit-for-bit from each
    original site.

    Parameters
    ----------
    q_net : surface net heat flux [W/m^2] (>0 into ocean) or ``None``.
    fw : surface freshwater flux [kg/m^2/s] (>0 P-E into ocean) or ``None``.
    salt : real surface salt-mass flux [kg/m^2/s] (>0 brine into ocean) or
        ``None``.
    T_sfc, S_sfc : (...,) — top-layer temperature [degC] and salinity [PSU].
    g, rho_0, c_sw : gravity [m/s^2], Boussinesq reference density [kg/m^3],
        seawater specific heat [J/(kg K)] — passed by the caller from its own
        constant source (``constants_config`` for lat-lon, module constants for
        MPAS; both equal the canonical values by default).
    real_salt_in_qs : see above.

    Returns
    -------
    (B_f, Q_sfc_T, Q_sfc_S) : each ``None`` when its forcing channel is absent
    (``B_f`` is ``None`` only when BOTH heat and freshwater/salt are absent).
    """
    p_sfc = jnp.zeros_like(T_sfc)

    # Surface α/β from the MODEL-selected EOS (``eos_fn``).  ``None`` ⇒ the
    # Wright-specific helpers (bit-identical legacy: those helpers ARE autodiff
    # of ``wright_eos``, verified == ``eos_density_derivatives(wright_eos)`` to
    # the bit).  A non-Wright EOS — e.g. the thermobaric ``nemo_seos`` — has
    # different surface α/β, so the KPP/CATKE surface buoyancy forcing must use
    # it (else the boundary layer mixes against Wright while the interior
    # ρ/N²/Ri use S-EOS — a half-applied closure).
    if eos_fn is None:
        def _alpha_sfc():
            return thermal_expansion_coeff(T_sfc, S_sfc, p_sfc)

        def _beta_sfc():
            return haline_contraction_coeff(T_sfc, S_sfc, p_sfc)
    else:
        # Floor salinity before the EOS-generic autodiff (sqrt(S) EOSes diverge
        # at S=0 land cells; later masking is a multiply ⇒ 0·NaN unsafe).
        S_eval = jnp.maximum(S_sfc, _EOS_SALINITY_FLOOR)

        def _alpha_sfc():
            drho_dT, _ = eos_density_derivatives(eos_fn, T_sfc, S_eval, p_sfc)
            return -drho_dT / eos_fn(T_sfc, S_eval, p_sfc)

        def _beta_sfc():
            _, drho_dS = eos_density_derivatives(eos_fn, T_sfc, S_eval, p_sfc)
            return drho_dS / eos_fn(T_sfc, S_eval, p_sfc)

    Q_sfc_T = None
    B_f = None
    if q_net is not None:
        Q_sfc_T = q_net / (rho_0 * c_sw)
        alpha = _alpha_sfc()
        B_f = -g * alpha * Q_sfc_T

    Q_sfc_S = None
    if fw is not None or salt is not None:
        beta = _beta_sfc()
        Q_sfc_S = jnp.zeros_like(S_sfc)
        if real_salt_in_qs:
            # Lat-lon: freshwater dilution PLUS real salt both accumulate into
            # one Q_sfc_S; the buoyancy uses that single combined flux.
            if fw is not None:
                Q_sfc_S = Q_sfc_S - S_sfc * fw / rho_0
            if salt is not None:
                Q_sfc_S = Q_sfc_S + salt * 1.0e3 / rho_0
            B_salt = g * beta * Q_sfc_S
        else:
            # MPAS: Q_sfc_S carries the freshwater term only; the real salt
            # adds to the surface buoyancy alone (separate B_salt term).
            B_salt = jnp.zeros_like(S_sfc)
            if fw is not None:
                Q_sfc_S = -S_sfc * jnp.asarray(fw, S_sfc.dtype) / rho_0
                B_salt = B_salt + g * beta * Q_sfc_S
            if salt is not None:
                B_salt = B_salt + g * beta * (
                    jnp.asarray(salt, S_sfc.dtype) * 1.0e3 / rho_0)
        B_f = B_salt if B_f is None else (B_f + B_salt)

    return B_f, Q_sfc_T, Q_sfc_S


def tridiag_thomas(a, b, c, d):
    """Solve a tridiagonal system A x = d via the Thomas algorithm.

    a, b, c, d each have shape ``(..., N)`` and ``a[..., 0]``, ``c[..., -1]``
    are unused (left as zero by the caller). Returns ``x`` of shape ``(..., N)``.
    Shared by the TKE and CATKE backward-Euler vertical solves.
    """
    # Mixed-precision guard: callers may assemble rows from float32 state
    # fields and float64 scalars (JAX_ENABLE_X64 promotes constants). The
    # lax.scan carries are seeded from `b`/`d` while the bodies compute in
    # the promoted dtype — a mismatch is a hard TypeError at trace time.
    # Unify once at entry; no-op when dtypes already agree.
    common = jnp.result_type(a, b, c, d)
    a, b, c, d = (x.astype(common) for x in (a, b, c, d))
    N = b.shape[-1]

    def step(carry, k):
        c_prev, d_prev = carry
        denom = b[..., k] - a[..., k] * c_prev
        denom_safe = jnp.where(jnp.abs(denom) > _EPS, denom, _EPS)
        cp = c[..., k] / denom_safe
        dp = (d[..., k] - a[..., k] * d_prev) / denom_safe
        return (cp, dp), (cp, dp)

    # Forward sweep
    init_c = jnp.zeros_like(b[..., 0])
    init_d = jnp.zeros_like(d[..., 0])
    _, (cp_all, dp_all) = jax.lax.scan(
        step, (init_c, init_d), jnp.arange(N),
    )
    # cp_all, dp_all have shape (N, ...); transpose so trailing axis is N.
    cp_all = jnp.moveaxis(cp_all, 0, -1)
    dp_all = jnp.moveaxis(dp_all, 0, -1)

    # Back substitution
    def back(carry, k_rev):
        x_next = carry
        k = N - 1 - k_rev
        x = jnp.where(
            k_rev == 0, dp_all[..., k],
            dp_all[..., k] - cp_all[..., k] * x_next,
        )
        return x, x

    x_init = jnp.zeros_like(b[..., 0])
    _, x_rev = jax.lax.scan(back, x_init, jnp.arange(N))
    x_rev = jnp.moveaxis(x_rev, 0, -1)
    # Reverse the back-sub output to get x in natural index order.
    return x_rev[..., ::-1]


def vmap_vertical_diffusion(
    u: jnp.ndarray,
    v: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    vel_coeff,
    tracer_coeff,
    diffuse_fn: Callable,
    apply_diffusion: bool,
):
    """Apply a per-field vertical-diffusion function to the [u, v] and [T, S]
    pairs via a single vmap each, returning stacked tendencies.

    The velocity pair uses ``vel_coeff`` (viscosity), the tracer pair uses
    ``tracer_coeff`` (diffusivity); ``diffuse_fn(q, coeff)`` returns the
    per-field tendency (it captures z_coord / jacobian / dt). Shared by the
    constant and Richardson schemes, which differ only in ``diffuse_fn`` and the
    coefficients.

    When ``apply_diffusion`` is False, returns zero-tendency stacks (the
    diffusion is deferred to the implicit backward-Euler solve in the dynamics
    step).

    Returns
    -------
    vel_tend, tr_tend : array ``(2, ...)`` each — ``[du_dt, dv_dt]`` and
        ``[dT_dt, dS_dt]``.
    """
    if apply_diffusion:
        vel_tend = jax.vmap(
            lambda q: diffuse_fn(q, vel_coeff), in_axes=0, out_axes=0,
        )(jnp.stack([u, v], axis=0))
        tr_tend = jax.vmap(
            lambda q: diffuse_fn(q, tracer_coeff), in_axes=0, out_axes=0,
        )(jnp.stack([T, S], axis=0))
    else:
        zero_uv = jnp.zeros_like(u)
        vel_tend = jnp.stack([zero_uv, zero_uv], axis=0)
        zero_T = jnp.zeros_like(T)
        tr_tend = jnp.stack([zero_T, zero_T], axis=0)
    return vel_tend, tr_tend
