"""Fox-Kemper mixed-layer-eddy (MLE) restratification — shared, grid-agnostic core.

Faithful port of NEMO 5.0.1 ``src/OCE/TRA/tramle.F90`` for the ORCA1 setting
``ln_mle=.true., nn_mle=1`` (the "new formulation").  The submesoscale mixed-
layer eddies are represented by an eddy-induced (bolus) overturning streamfunction
that restratifies the mixed layer — flattening ML isopycnals and SHOALING the
mixed-layer depth, especially in the subtropical-gyre mode-water regions south of
the western boundary currents where coarse models lack the resolved eddies.

This module holds the grid-AGNOSTIC pieces (config, the MLE coefficient, the
MLE mixed-layer depth + mean buoyancy, the vertical structure function).  The
C-grid (lat-lon / tripole) and MPAS adapters build the face/edge streamfunction
and the bolus tracer-flux divergence on top of these.

NEMO nn_mle=1 streamfunction (per u-face):
    Psi_u = rc_f * H_u^2 * (e2u/e1u) * (bm_E - bm_W) * min(111 km, e1u) * mu(z)
with rc_f = rn_ce / (5 km * f0),  f0 = 2*Omega*sin(rn_lat),  rn_lat = 20 deg
(constant -> no equatorial singularity, no hemisphere sign flip), and bm the
vertically-averaged mixed-layer buoyancy.  The bolus transport dk[Psi] is added
to the tracer advective transport (here: a conservative bolus-flux divergence).

References
----------
Fox-Kemper, Ferrari & Hallberg (2008), JPO 38, 1145-1165.
Fox-Kemper & Ferrari (2008), JPO 38, 1166-1179.
NEMO 5.0.1 TRA/tramle.F90 (ORCA1 RUN_REF: ln_mle, nn_mle=1, rn_ce=0.06, rn_lat=20).
"""
from __future__ import annotations

from typing import NamedTuple

import math

import jax
import jax.numpy as jnp

from legoesm import constants

# --- Fox-Kemper MLE fixed constants (NEMO 5.0.1 TRA/tramle.F90) ---
# Vertical-structure factor 5/21 in mu(z) = (1-zeta^2)(1 + 5/21 zeta^2).
_R5_21 = 5.0 / 21.0
# Reference horizontal scale in the nn_mle=1 coefficient
# rc_f = rn_ce / (RC_F_LENGTH_SCALE_M * f0); NEMO hard-codes 5 km (= "5.e3").
# Not a tunable knob (it is the fixed normalising length of the FK closure); the
# efficiency rn_ce (MLEConfig.ce) is the tunable scaling.
_RC_F_LENGTH_SCALE_M = 5.0e3
# Default MLE-MLD density threshold rn_rho_c_mle [kg/m^3] (ORCA1 RUN_REF: 0.01).
# Mirrors MLEConfig.rho_c_mle so the standalone helper's signature default is a
# named reference (the Config field is the single source of truth at call sites).
_RHO_C_MLE_DEFAULT = 0.01

__param_spec__ = {
    "MLEConfig": {
        "scheme_key": "ocean.lat.mle",
        "excluded": {
            # Measurement/closure conventions of the FK MLD criterion (a fixed
            # reference latitude, density-threshold and reference depth define the
            # diagnostic — not free training knobs; changing them changes WHICH
            # mixed layer is detected, not the eddy efficiency).
            "lat_ref_deg": "convention: fixed reference latitude for constant f0 (nn_mle=1)",
            "rho_c_mle": "convention: MLE-MLD density criterion (defines the diagnosed ML)",
            "ref_depth_m": "convention: reference depth for the MLD density criterion",
            # Numerics guards: grid-scale cap and the optional bolus-CFL clamp.
            "max_grid_scale_m": "numerics: min(111 km, e1u) grid-scale cap",
            "bolus_cfl_cap": "numerics: optional |w_mle| Courant clamp (0 = off = oracle)",
        },
        "params": {
            # The single FK efficiency knob (NEMO rn_ce); typical 0.06-0.08.
            "ce": {"units": "1", "bounds": (0.033, 0.3), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "Fox-Kemper, Ferrari & Hallberg (2008)", "shape": None},
        },
    },
}


class MLEConfig(NamedTuple):
    """Fox-Kemper MLE configuration (NEMO namtra_mle, nn_mle=1 defaults).

    Parameters
    ----------
    ce : float
        MLE efficiency coefficient ``rn_ce`` (typical 0.06-0.08; ORCA1 0.06).
        The tunable knob.
    lat_ref_deg : float
        Reference latitude ``rn_lat`` [deg] setting the constant Coriolis f0 in
        ``rc_f`` (nn_mle=1).  ORCA1 20 deg.
    rho_c_mle : float
        Density threshold ``rn_rho_c_mle`` [kg/m^3] for the MLE mixed-layer depth
        (referenced to ``ref_depth_m``).  ORCA1 0.01 -- DISTINCT from the 0.03
        de Boyer Montegut MLD diagnostic; uses IN-SITU density.
    ref_depth_m : float
        Reference depth [m] for the MLE-MLD density criterion (NEMO ~10 m).
    mld_uv : str
        Face mixed-layer depth from the two neighbour cells (NEMO nn_mld_uv):
        "min" (0), "avg" (1) or "max" (2).  ORCA1 default "min".
    no_mle_in_convection : bool
        NEMO nn_conv=1: zero the streamfunction at faces where a neighbour is
        statically unstable (N^2 < 0).
    max_grid_scale_m : float
        The ``min(111 km, e1u)`` grid-scale cap [m] (NEMO 111.e3).
    bolus_cfl_cap : float
        If > 0, cap |w_mle| so the bolus vertical Courant number stays below this
        fraction (stability guard; 0 = off = oracle behaviour).
    """
    ce: float = 0.06
    lat_ref_deg: float = 20.0
    rho_c_mle: float = 0.01
    ref_depth_m: float = 10.0
    mld_uv: str = "min"
    no_mle_in_convection: bool = True
    max_grid_scale_m: float = 111.0e3
    bolus_cfl_cap: float = 0.0


def mle_coefficient(ce: float, lat_ref_deg: float) -> float | jnp.ndarray:
    """NEMO nn_mle=1 coefficient ``rc_f = rn_ce / (5 km * 2*Omega*sin(rn_lat))``.

    Constant (uses the reference latitude, not local f) so the streamfunction is
    finite at the equator and does not flip sign across hemispheres.  Guarded
    against ``lat_ref_deg`` ~ 0.
    """
    lat = float(lat_ref_deg)
    # Guard the f0 = 2*Omega*sin(lat) blow-up at ANY zero-Coriolis reference
    # latitude (0, +/-180, ...), not just |lat|<1 deg: test |sin(lat)| directly
    # so an out-of-range lat_ref like 180 deg (sin -> 0) also raises instead of
    # producing a huge finite rc_f artefact from floating-point sin(pi).
    if abs(math.sin(math.radians(lat))) < math.sin(math.radians(1.0)):
        raise ValueError(
            f"MLE lat_ref_deg={lat} has |sin(lat)| too small: f0 -> 0 makes rc_f "
            "blow up. Use a mid-latitude reference (NEMO default 20 deg).")
    # ``lat_ref_deg`` is a FIXED reference latitude (excluded-tier convention,
    # never traced), so f0 is built with pure-Python ``math`` and stays a plain
    # Python float — ``jnp.sin`` of a traced angle would be unnecessary here.
    f0 = 2.0 * float(constants.Omega) * math.sin(math.radians(lat))
    # Do NOT cast ``ce``: it is the registered tunable (MLEConfig.ce, tier-2,
    # SPEC_MODULES, transform=sigmoid).  A prior ``float(ce)`` raised
    # ConcretizationTypeError when a traced override was spliced into the config
    # during extended-tier training, violating the differentiable=True contract.
    # A Python-float ``ce`` still yields a Python float (production constant-
    # folding preserved); a traced ``ce`` yields a differentiable traced scalar.
    return ce / (_RC_F_LENGTH_SCALE_M * f0)


def mle_streamfunction_magnitude(
    rc_f: float,
    H_face: jnp.ndarray,
    face_width: jnp.ndarray,
    dbm_face: jnp.ndarray,
    cap: jnp.ndarray,
) -> jnp.ndarray:
    """NEMO nn_mle=1 face streamfunction MAGNITUDE [m^3/s] (#518 item 9).

    ``psim = rc_f · H_face² · face_width · dbm_face · cap``

    The single grid-agnostic product shared by the lat-lon ``psim_u`` / ``psim_v``
    and the MPAS ``psim_e``.  Each grid passes its OWN face operands:

    * lat-lon u-face: ``face_width = e2u``, ``dbm_face = (bm_E-bm_W)/e1u``,
      ``cap = min(max_grid_scale_m, e1u)`` (and the v-face analogue).
    * MPAS edge:     ``face_width = dvEdge``, ``dbm_face = (bm[c2]-bm[c1])/dcEdge``,
      ``cap = min(max_grid_scale_m, dcEdge)``.

    The multiply is the same left-to-right ``rc_f * H * H * width * dbm * cap``
    order as every original site, so the result is byte-identical.  ``H_face`` is
    the face mixed-layer depth [m]; the square keeps the FK scaling.
    """
    return rc_f * H_face * H_face * face_width * dbm_face * cap


def mle_vertical_structure(gdepw_over_H: jnp.ndarray) -> jnp.ndarray:
    """FK vertical structure mu(z) on w-interfaces (NEMO tramle.F90).

    ``mu = max(0, (1 - zeta^2)(1 + 5/21 zeta^2))`` with ``zeta = 1 - 2*gdepw/H``.
    Zero at the surface (gdepw=0 -> zeta=1) and at the ML base (gdepw=H ->
    zeta=-1), peaking mid-mixed-layer.  ``gdepw_over_H`` = w-interface depth / H.
    """
    zeta = 1.0 - 2.0 * gdepw_over_H
    z2 = zeta * zeta
    return jnp.maximum(0.0, (1.0 - z2) * (1.0 + _R5_21 * z2))


def mle_mld_and_buoyancy(
    rho_insitu: jnp.ndarray,
    dz_live: jnp.ndarray,
    wet_cell: jnp.ndarray,
    *,
    z_centers: jnp.ndarray,
    rho_c_mle: float = _RHO_C_MLE_DEFAULT,
    ref_depth_m: float = 10.0,
    rho0: float = constants.rho_ocean,
    grav: float = constants.g,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """MLE mixed-layer depth + ML-mean buoyancy (NEMO tramle.F90, integer level).

    Faithful to NEMO: the mixed layer is the levels SHALLOWER than the first level
    whose in-situ density exceeds the reference-depth density by ``rho_c_mle``
    (Delta-rho criterion on IN-SITU rho, referenced to ``ref_depth_m``); the ML
    depth ``zmld`` is the sum of their live thicknesses, and the ML-mean buoyancy
    is ``bm = grav * sum_ML[(rho0 - rho)/rho0 * dz] / max(dz_top, zmld)``.

    Parameters
    ----------
    rho_insitu : array (..., nlev)  in-situ density [kg/m^3].
    dz_live    : array (..., nlev)  live layer thickness [m] (dry cells -> 0).
    wet_cell   : array (..., nlev)  ocean mask {0,1}.
    z_centers  : array (nlev,)      level-centre reference depths [m, positive].
    Returns
    -------
    zmld : array (...)        MLE mixed-layer depth [m].
    bm   : array (...)        ML-mean buoyancy [m/s^2].
    in_ml : array (..., nlev) mixed-layer membership mask {0,1} (for the
                              ML-integrated convection-gate N^2, NEMO zn2).
    """
    nlev = z_centers.shape[0]
    wet = wet_cell.astype(rho_insitu.dtype)
    # Reference density at ~ref_depth_m: nearest level centre at/above the ref
    # depth (NEMO uses the level value rhop(nla10), not an interpolation).  The
    # index is derived from z_centers via jnp ops (NO Python int()) so the whole
    # routine stays jit-safe when z_centers is a traced coordinate.
    iref = jnp.clip(
        jnp.searchsorted(z_centers, jnp.asarray(ref_depth_m, dtype=z_centers.dtype)),
        0, nlev - 1)                                                 # traced scalar
    rho_ref = jnp.take(rho_insitu, iref, axis=-1)                    # (...)
    z_ref = jnp.take(z_centers, iref)                               # scalar depth
    excess = rho_insitu - (rho_ref[..., jnp.newaxis] + rho_c_mle)    # (..., nlev)
    # First level (from the surface) denser than the threshold, among WET levels
    # below the reference level; the ML is the levels above it.
    below_ref = z_centers > z_ref                                    # (nlev,)
    below_ref = below_ref.reshape((1,) * (rho_insitu.ndim - 1) + (nlev,))
    exceed = (excess > 0.0) & (wet > 0.5) & jnp.broadcast_to(below_ref, excess.shape)
    has = jnp.any(exceed, axis=-1)
    first = jnp.argmax(exceed.astype(jnp.int32), axis=-1)            # (...)
    # If no level exceeds, the whole wet column is "mixed".
    first = jnp.where(has, first, jnp.sum((wet > 0.5).astype(jnp.int32), axis=-1))
    lvl = jnp.arange(nlev)
    in_ml = (lvl.reshape((1,) * (rho_insitu.ndim - 1) + (nlev,)) < first[..., jnp.newaxis]) \
        & (wet > 0.5)
    in_ml = in_ml.astype(rho_insitu.dtype)
    zmld = jnp.sum(in_ml * dz_live, axis=-1)                          # (...)
    dz_top = dz_live[..., 0]
    b_int = grav * jnp.sum(in_ml * ((rho0 - rho_insitu) / rho0) * dz_live, axis=-1)
    bm = b_int / jnp.maximum(jnp.maximum(dz_top, zmld), 1e-10)        # (...)
    # MLD/buoyancy are diagnostic gates -> stop gradient so AD users do not treat
    # threshold motion as a smooth control path (the bolus magnitude through bm/H
    # stays differentiable in T,S elsewhere).  ``in_ml`` is returned so the
    # convection gate can ML-integrate N^2 over the SAME mixed layer (NEMO zn2).
    return jax.lax.stop_gradient(zmld), bm, jax.lax.stop_gradient(in_ml)


def face_mld(h_a: jnp.ndarray, h_b: jnp.ndarray, mode: str) -> jnp.ndarray:
    """Face mixed-layer depth from the two neighbour MLDs (NEMO nn_mld_uv)."""
    if mode == "min":
        return jnp.minimum(h_a, h_b)
    if mode == "avg":
        return 0.5 * (h_a + h_b)
    if mode == "max":
        return jnp.maximum(h_a, h_b)
    raise ValueError(f"unknown mld_uv {mode!r} (expected 'min'|'avg'|'max')")
