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


def covariance(a: Array, b: Array) -> Array:
    """Horizontal-mean covariance ``<a'b'>`` of two ``(ny, nx, nz)`` fields → ``(nz,)``
    (primes = deviation from the horizontal mean at each level)."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError("a and b must be matching (ny,nx,nz) fields")
    return ((a - a.mean(axis=_HORIZ, keepdims=True))
            * (b - b.mean(axis=_HORIZ, keepdims=True))).mean(axis=_HORIZ)


def third_moment(a: Array, b: Array, c: Array) -> Array:
    """Horizontal-mean triple correlation ``<a'b'c'>`` → ``(nz,)``."""
    fields = [np.asarray(f) for f in (a, b, c)]
    if any(f.shape != fields[0].shape or f.ndim != 3 for f in fields):
        raise ValueError("a, b, c must be matching (ny,nx,nz) fields")
    prod = np.ones_like(fields[0])
    for f in fields:
        prod = prod * (f - f.mean(axis=_HORIZ, keepdims=True))
    return prod.mean(axis=_HORIZ)


def sgs_flux_from_face_stress(tau_faces: Array) -> Array:
    """Horizontal-mean SGS momentum flux ``<u_i'w'>_sgs`` at centres → ``(nz,)``.

    ``tau_faces`` is the core's vertical SGS stress on all ``nz+1`` faces
    (``spectral_les_plane.sgs_vertical_momentum_flux_faces``: ``∂u_i/∂t ∋ ∂τ_i3/∂z``,
    surface face = minus the wall stress, lid = 0). The turbulent flux is ``−τ_i3``,
    averaged face→centre exactly as :func:`sgs_vertical_scalar_flux_mean` does for
    heat, so a decelerating wall gives a NEGATIVE (downward) near-surface flux."""
    t = np.asarray(tau_faces)
    if t.ndim != 3:
        raise ValueError(f"expected a (ny,nx,nz+1) face field, got shape {t.shape}")
    turb = -t
    return (0.5 * (turb[..., :-1] + turb[..., 1:])).mean(axis=_HORIZ)


def column_share(part: Array, total: Array) -> float:
    """``Σ|part| / Σ|total|`` over levels — the fraction of a column flux profile
    carried by one component (e.g. the background-viscosity floor). An L1 ratio,
    not a partition: with sign changes in the total it can exceed 1. ``nan`` when
    the total vanishes identically."""
    den = float(np.sum(np.abs(np.asarray(total))))
    if den == 0.0:
        return float("nan")
    return float(np.sum(np.abs(np.asarray(part)))) / den


def sgs_vertical_scalar_flux_faces(
    phi: Array, nu_t: Array, dz: float, *, pr_sgs: float = 1.0,
    surface_flux: float = 0.0, top_flux: float = 0.0,
) -> Array:
    """Horizontal-mean SGS vertical scalar flux on all ``nz+1`` faces, exactly as
    ``spectral_les_plane.scalar_rhs`` (and ``scalar_rhs_monotone``) apply it.

    ``phi`` and ``nu_t`` are ``(ny, nx, nz)`` cell-centre fields (``nu_t`` the
    viscosity the core applied, ``effective_eddy_viscosity``); ``dz`` is the
    uniform layer thickness; ``surface_flux`` is the surface kinematic flux
    ``<w'φ'>_0`` the core imposed (``0`` for a scalar with no surface source).

    The core forms the DIFFUSIVE flux ``J = K_h ∂φ/∂z`` on interior faces
    (``K_h = ν_t/Pr``, ``K_h`` averaged centre→face by ``c2f``, gradient by
    ``ddz_c2f``), imposes ``J = -surface_flux`` on the surface face and
    ``J = -top_flux`` on the lid (0 = no-flux lid; a lid inversion Γ gives
    ``top_flux = -⟨K_h⟩Γ``), then takes ``∂J/∂z`` at centres. The turbulent flux is
    ``<w'φ'> = -J``, positive upward; this returns its horizontal mean per face.
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
    turb_int = (-k_h_face * dphi_dz_face).mean(axis=_HORIZ)        # (nz-1,)
    return np.concatenate([np.asarray([float(surface_flux)], dtype=turb_int.dtype),
                           turb_int, np.asarray([float(top_flux)], dtype=turb_int.dtype)])


def faces_to_centres(face_profile: Array) -> Array:
    """Average a ``(nz+1,)`` face profile to the ``(nz,)`` centres (the core's
    ``f2c``)."""
    f = np.asarray(face_profile)
    return 0.5 * (f[:-1] + f[1:])


def sgs_vertical_scalar_flux_mean(
    phi: Array, nu_t: Array, dz: float, *, pr_sgs: float = 1.0,
    surface_flux: float = 0.0,
) -> Array:
    """SGS horizontal-mean vertical scalar flux at the centres, matching the core's
    discretization (:func:`sgs_vertical_scalar_flux_faces` mapped face→centre).

    Sign check: stable (``∂θ/∂z>0``) ⇒ negative (downward) interior SGS heat flux;
    the surface face carries exactly ``+surface_flux`` (upward for CBL heating).
    """
    return faces_to_centres(sgs_vertical_scalar_flux_faces(
        phi, nu_t, dz, pr_sgs=pr_sgs, surface_flux=surface_flux))


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
    les_setup: dict | None = None,
    **extra: Array | None,
) -> LESReferenceArtifact:
    """Assemble a validated :class:`LESReferenceArtifact` from stacked mean profiles.

    The profile time-series (``theta``/``u``/``v``/fluxes) are ``(nt, nz)``; the
    forcing profiles are ``(nz,)`` and the surface series ``(nt,)``. This is a thin,
    validated constructor (``LESReferenceArtifact.validate`` runs on build) so the
    GPU driver's only job is to run the LES and stack these reductions per output
    time — the contract lives in one place. ``extra`` passes the optional
    turbulence-statistics / realised-surface channels
    (``bridge.TURBULENCE_STAT_FIELDS`` + ``bridge.SURFACE_SERIES_FIELDS``) through
    by name; an unknown name is a ``TypeError`` from the dataclass.
    """
    return LESReferenceArtifact(
        **{k: (None if a is None else np.asarray(a)) for k, a in extra.items()},
        les_setup=les_setup,
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
