"""LES → reference-artifact emission: horizontal-mean + flux diagnostics.

``run_les_suite.py`` turns a registry :class:`LESCase` into a finished LES run and
must write the self-describing :class:`~...bridge.LESReferenceArtifact` the SCM
tuner consumes. This module holds the *diagnostic* core of that emission — the
horizontal-mean profile and turbulent-flux reductions — so it is unit-testable on
CPU without a GPU LES integration (the driver is then a thin wrapper).

Sign conventions (match ``bridge`` / ``spectral_les_plane``): heights increase with
index (surface-first); vertical turbulent flux positive **upward**; the SGS scalar
flux is down-gradient ``<w'φ'>_sgs = -<K_h ∂φ/∂z>`` with ``K_h = ν_t / Pr >= 0`` — so
a stable layer (``∂θ/∂z > 0``) gives a **downward** (negative) SGS heat flux and a
super-adiabatic surface layer (``∂θ/∂z < 0``) gives an **upward** (positive) one,
which is exactly the near-surface flux the resolved field misses.

All reductions are plain array math over the horizontal ``(y, x)`` axes; nothing
here integrates the LES or touches the closure internals beyond the ``ν_t`` field
the caller supplies (from ``spectral_les_plane.eddy_viscosity``).
"""
from __future__ import annotations

import numpy as np

from .bridge import LESReferenceArtifact

Array = np.ndarray

# Horizontal axes of the plane LES state arrays ``(ny, nx, nz)``.
_HORIZ = (0, 1)


def horizontal_mean(field: Array) -> Array:
    """Horizontal mean over ``(y, x)`` of a ``(ny, nx, nz)`` field → ``(nz,)``."""
    f = np.asarray(field)
    if f.ndim != 3:
        raise ValueError(f"expected a (ny,nx,nz) field, got shape {f.shape}")
    return f.mean(axis=_HORIZ)


def resolved_vertical_flux(w_center: Array, phi: Array) -> Array:
    """Resolved horizontal-mean vertical flux ``<w'φ'>`` → ``(nz,)``.

    ``w_center`` and ``phi`` are ``(ny, nx, nz)`` on cell centres (use
    ``spectral_les_plane.f2c`` to bring ``w`` from faces to centres first). The
    prime is the deviation from the horizontal mean at each level; the covariance
    is the resolved turbulent flux, positive upward.
    """
    w = np.asarray(w_center)
    phi = np.asarray(phi)
    if w.shape != phi.shape or w.ndim != 3:
        raise ValueError("w_center and phi must be matching (ny,nx,nz) fields")
    w_prime = w - w.mean(axis=_HORIZ, keepdims=True)
    phi_prime = phi - phi.mean(axis=_HORIZ, keepdims=True)
    return (w_prime * phi_prime).mean(axis=_HORIZ)


def sgs_vertical_scalar_flux_mean(
    phi: Array, nu_t: Array, dz: float, *, pr_sgs: float = 1.0,
    surface_flux: float = 0.0,
) -> Array:
    """SGS horizontal-mean vertical scalar flux, matching the core's discretization.

    Reproduces EXACTLY the vertical SGS flux ``spectral_les_plane.scalar_rhs``
    integrates (not a centred approximation), so the emitted ``<w'φ'>_sgs`` equals
    the flux the model actually applied. ``phi`` and ``nu_t`` are ``(ny, nx, nz)``
    cell-centre fields (``nu_t`` from ``eddy_viscosity``); ``dz`` is the uniform
    layer thickness; ``surface_flux`` is the prescribed surface kinematic flux
    ``<w'φ'>_0`` (``Q0`` for θ, ``0`` for a scalar with no surface source).

    The core forms the DIFFUSIVE flux ``J = K_h ∂φ/∂z`` on interior faces
    (``K_h = ν_t/Pr``, ``K_h`` averaged centre→face by ``c2f``, gradient by
    ``ddz_c2f``), imposes ``J = -surface_flux`` on the surface face and ``J = 0`` on
    the lid, then takes ``∂J/∂z`` at centres. The turbulent flux is ``<w'φ'> = -J``;
    this returns the horizontal mean of ``-J`` mapped to centres (``f2c``), positive
    upward. Sign check: stable (``∂θ/∂z>0``) ⇒ negative (downward) interior SGS heat
    flux; the surface carries exactly ``+surface_flux`` (upward for CBL heating).
    """
    phi = np.asarray(phi)
    nu_t = np.asarray(nu_t)
    if phi.shape != nu_t.shape or phi.ndim != 3:
        raise ValueError("phi and nu_t must be matching (ny,nx,nz) fields")
    if not (pr_sgs > 0):
        raise ValueError(f"pr_sgs must be > 0, got {pr_sgs}")
    if not (dz > 0):
        raise ValueError(f"dz must be > 0, got {dz}")
    k_h = nu_t / pr_sgs
    # interior faces (nz-1): K_h centre→face (c2f), gradient centre→face (ddz_c2f)
    k_h_face = 0.5 * (k_h[..., :-1] + k_h[..., 1:])
    dphi_dz_face = (phi[..., 1:] - phi[..., :-1]) / dz
    # TURBULENT flux <w'φ'> = -J on interior faces (down-gradient), surface = the
    # prescribed +surface_flux (upward), lid = 0 (no-flux).
    turb_int = -k_h_face * dphi_dz_face                              # (ny,nx,nz-1)
    surf = np.full(phi[..., :1].shape, float(surface_flux), dtype=phi.dtype)
    lid = np.zeros(phi[..., :1].shape, dtype=phi.dtype)
    turb_face = np.concatenate([surf, turb_int, lid], axis=-1)      # (ny,nx,nz+1)
    turb_center = 0.5 * (turb_face[..., :-1] + turb_face[..., 1:])  # f2c -> (nz)
    return turb_center.mean(axis=_HORIZ)


def build_reference_artifact(
    *,
    case_name: str,
    sgs: str,
    z: Array,
    times_s: Array,
    theta: Array,
    u: Array,
    v: Array,
    wtheta_resolved: Array,
    wtheta_sgs: Array | None,
    qt: Array | None = None,
    wqt_resolved: Array | None = None,
    wqt_sgs: Array | None = None,
    f_c: float = 0.0,
    u_geo: Array | None = None,
    v_geo: Array | None = None,
    subsidence_w: Array | None = None,
    theta_adv: Array | None = None,
    qv_adv: Array | None = None,
    prescribe: str = "fluxes",
    T_s: Array | None = None,
    w_theta_s: Array | None = None,
    w_qv_s: Array | None = None,
) -> LESReferenceArtifact:
    """Assemble a validated :class:`LESReferenceArtifact` from stacked mean profiles.

    The profile time-series (``theta``/``u``/``v``/fluxes) are ``(nt, nz)``; the
    forcing profiles are ``(nz,)`` and the surface series ``(nt,)``. This is a thin,
    validated constructor (``LESReferenceArtifact.validate`` runs on build) so the
    GPU driver's only job is to run the LES and stack these reductions per output
    time — the contract lives in one place.
    """
    return LESReferenceArtifact(
        case_name=case_name,
        sgs=sgs,
        heights_m=np.asarray(z),
        times_s=np.asarray(times_s),
        theta=np.asarray(theta),
        u=np.asarray(u),
        v=np.asarray(v),
        wtheta_resolved=np.asarray(wtheta_resolved),
        wtheta_sgs=None if wtheta_sgs is None else np.asarray(wtheta_sgs),
        qt=None if qt is None else np.asarray(qt),
        wqt_resolved=None if wqt_resolved is None else np.asarray(wqt_resolved),
        wqt_sgs=None if wqt_sgs is None else np.asarray(wqt_sgs),
        f_c=float(f_c),
        u_geo=None if u_geo is None else np.asarray(u_geo),
        v_geo=None if v_geo is None else np.asarray(v_geo),
        subsidence_w=None if subsidence_w is None else np.asarray(subsidence_w),
        theta_adv=None if theta_adv is None else np.asarray(theta_adv),
        qv_adv=None if qv_adv is None else np.asarray(qv_adv),
        prescribe=prescribe,
        T_s=None if T_s is None else np.asarray(T_s),
        w_theta_s=None if w_theta_s is None else np.asarray(w_theta_s),
        w_qv_s=None if w_qv_s is None else np.asarray(w_qv_s),
    )
