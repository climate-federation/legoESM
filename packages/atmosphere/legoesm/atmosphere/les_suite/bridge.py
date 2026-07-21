"""LES→SCM bridge: reference-artifact schema + forcing/truth extraction.

The LES suite (``docs/atmosphere/les_suite/LES_SUITE.md``) consumes LES output as
*cached truth*, never re-integrating the LES during SCM tuning (the "consumption
rule", §5). This module is the contract between the two sides:

* :class:`LESReferenceArtifact` — the **self-describing** record one LES case
  emits. It carries BOTH the horizontal-mean truth profiles over height×time AND
  the exact large-scale forcing the LES received. The SCM bridge reconstructs its
  forcing FROM the artifact, so the controlled-comparison rule (CLAUDE.md: "hold
  the eval protocol byte-identical to baseline") is structural, not a convention
  the caller must remember.
* :func:`artifact_to_scm_forcing` — build an :class:`SCMForcing` reproducing the
  LES forcing (D6 prognostic path: the SCM is driven by the *same* forcing).
* :func:`diagnostic_truth` / :func:`prognostic_truth` — extract the LES truth the
  two D6 scores compare against (a single snapshot vs the full time series).
* :func:`total_turbulent_flux` — resolved + SGS turbulent flux, the quantity Q1's
  counter-gradient diagnostic and the flux score consume.

Sign conventions (match ``forcing/scm/scm_forcing.py`` verbatim):
``subsidence_w`` positive **upward**; surface kinematic heat flux ``w_th_s``
[K m/s] positive **upward** (into the BL); ``theta_adv`` [K/s], ``qv_adv``
[(kg/kg)/s]. A profile stored ``None`` in the artifact disables that SCM channel.

Scope boundary: pure data + array assembly. No LES integration, no SCM stepping,
no scoring arithmetic (that is :mod:`~legoesm.atmosphere.les_suite.score`). I/O
(npz save/load) is at the module edge so the core stays JAX-safe.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from legoesm.atmosphere.forcing.scm.scm_forcing import SCMForcing

Array = jnp.ndarray


class BridgeError(ValueError):
    """Raised when an artifact violates the LES→SCM contract."""


@dataclass(frozen=True)
class LESReferenceArtifact:
    """Horizontal-mean LES truth + applied forcing for one reference run.

    All profiles are on the LES full-level height grid ``heights_m`` (shape
    ``(nz,)``, monotonically increasing from the surface). Time series are indexed
    by ``times_s`` (shape ``(nt,)``); profile fields are ``(nt, nz)``. A ``None``
    field means the LES did not carry that channel (e.g. ``qt`` for a dry case).

    Truth profiles (horizontal means, resolved by the LES):
        theta : (nt, nz) potential temperature [K] (dry) or liquid-water potential
            temperature θ_l for moist cases (D9 uses θ_l as the conserved variable).
        qt : (nt, nz) total-water mixing ratio [kg/kg] or ``None`` (dry).
        u, v : (nt, nz) horizontal wind [m/s].

    Turbulent fluxes (horizontal-mean covariances; positive upward):
        wtheta_resolved, wtheta_sgs : (nt, nz) resolved + SGS ``<w'θ'>`` [K m/s].
        wqt_resolved, wqt_sgs : (nt, nz) moisture flux ``<w'q_t'>`` [(kg/kg) m/s]
            or ``None`` (dry).

    Applied forcing (exactly what the LES received — the bridge rebuilds SCMForcing
    from these so SCM and LES share byte-identical forcing):
        f_c : Coriolis parameter [1/s].
        u_geo, v_geo : (nz,) geostrophic wind [m/s] or ``None``.
        subsidence_w : (nz,) large-scale vertical velocity [m/s], positive up, or ``None``.
        theta_adv, qv_adv : (nz,) prescribed horizontal-advection tendencies
            [K/s] / [(kg/kg)/s] or ``None``.
        prescribe : "none" | "T_s" | "fluxes" — surface boundary treatment.
        T_s : (nt,) prescribed surface temperature [K] or ``None``.
        w_theta_s, w_qv_s : (nt,) surface kinematic heat/moisture fluxes
            [K m/s] / [(kg/kg) m/s], positive up, or ``None``.

    Provenance:
        case_name : the :class:`LESCase` this run realizes.
        sgs : SGS closure used (one of the case's ``sgs_variants``).
    """

    case_name: str
    sgs: str
    heights_m: Array
    times_s: Array
    theta: Array
    u: Array
    v: Array
    wtheta_resolved: Array
    wtheta_sgs: Array
    qt: Array | None = None
    wqt_resolved: Array | None = None
    wqt_sgs: Array | None = None
    f_c: float = 0.0
    u_geo: Array | None = None
    v_geo: Array | None = None
    subsidence_w: Array | None = None
    theta_adv: Array | None = None
    qv_adv: Array | None = None
    prescribe: str = "fluxes"
    T_s: Array | None = None
    w_theta_s: Array | None = None
    w_qv_s: Array | None = None

    def __post_init__(self) -> None:
        self.validate()

    @property
    def nz(self) -> int:
        return int(jnp.asarray(self.heights_m).shape[0])

    @property
    def nt(self) -> int:
        return int(jnp.asarray(self.times_s).shape[0])

    @property
    def is_moist(self) -> bool:
        return self.qt is not None

    def validate(self) -> None:
        if not self.case_name:
            raise BridgeError("LESReferenceArtifact.case_name must be non-empty")
        if self.prescribe not in ("none", "T_s", "fluxes"):
            raise BridgeError(
                f"{self.case_name}: prescribe must be one of none/T_s/fluxes, "
                f"got {self.prescribe!r}"
            )
        heights = jnp.asarray(self.heights_m)
        times = jnp.asarray(self.times_s)
        if heights.ndim != 1 or heights.shape[0] < 2:
            raise BridgeError(f"{self.case_name}: heights_m must be 1-D with >=2 levels")
        # monotone-increasing height (surface first) — donor-side upwind + flux
        # gradients assume this ordering.
        dz = jnp.diff(heights)
        if not bool(jnp.all(dz > 0)):
            raise BridgeError(f"{self.case_name}: heights_m must be strictly increasing")
        if times.ndim != 1 or times.shape[0] < 1:
            raise BridgeError(f"{self.case_name}: times_s must be 1-D with >=1 sample")
        if not bool(jnp.all(jnp.diff(times) > 0)) and times.shape[0] > 1:
            raise BridgeError(f"{self.case_name}: times_s must be strictly increasing")
        nt, nz = self.nt, self.nz
        # profile time series: (nt, nz)
        for name in ("theta", "u", "v", "wtheta_resolved", "wtheta_sgs",
                     "qt", "wqt_resolved", "wqt_sgs"):
            arr = getattr(self, name)
            if arr is None:
                continue
            a = jnp.asarray(arr)
            if a.shape != (nt, nz):
                raise BridgeError(
                    f"{self.case_name}: {name} must have shape {(nt, nz)}, got {a.shape}"
                )
        # forcing profiles: (nz,)
        for name in ("u_geo", "v_geo", "subsidence_w", "theta_adv", "qv_adv"):
            arr = getattr(self, name)
            if arr is None:
                continue
            a = jnp.asarray(arr)
            if a.shape != (nz,):
                raise BridgeError(
                    f"{self.case_name}: {name} must have shape {(nz,)}, got {a.shape}"
                )
        # surface time series: (nt,)
        for name in ("T_s", "w_theta_s", "w_qv_s"):
            arr = getattr(self, name)
            if arr is None:
                continue
            a = jnp.asarray(arr)
            if a.shape != (nt,):
                raise BridgeError(
                    f"{self.case_name}: {name} must have shape {(nt,)}, got {a.shape}"
                )
        # moisture consistency: a moist artifact (qt set) MUST carry a resolved
        # moisture flux; a dry artifact (qt None) must carry NO moisture fields at
        # all — otherwise a populated moisture field is silently discarded
        # downstream (diagnostic_truth/forcing extraction key off qt), which would
        # violate the self-describing contract.
        if self.qt is not None:
            if self.wqt_resolved is None:
                raise BridgeError(
                    f"{self.case_name}: moist artifact (qt set) must carry wqt_resolved"
                )
        else:
            dry_forbidden = [
                n for n in ("wqt_resolved", "wqt_sgs", "qv_adv", "w_qv_s")
                if getattr(self, n) is not None
            ]
            if dry_forbidden:
                raise BridgeError(
                    f"{self.case_name}: dry artifact (qt is None) must not carry "
                    f"moisture field(s) {dry_forbidden}"
                )
        # A SGS moisture flux without its resolved counterpart is inconsistent.
        if self.wqt_sgs is not None and self.wqt_resolved is None:
            raise BridgeError(
                f"{self.case_name}: wqt_sgs set without wqt_resolved"
            )
        # Surface-boundary consistency: each prescribe mode requires exactly its
        # own field(s) and FORBIDS the others, so a stored-but-ignored surface
        # forcing (e.g. w_theta_s under prescribe='T_s') can never silently drop.
        if self.prescribe == "fluxes":
            if self.w_theta_s is None:
                raise BridgeError(
                    f"{self.case_name}: prescribe='fluxes' requires w_theta_s"
                )
            if self.T_s is not None:
                raise BridgeError(
                    f"{self.case_name}: prescribe='fluxes' forbids T_s"
                )
        elif self.prescribe == "T_s":
            if self.T_s is None:
                raise BridgeError(f"{self.case_name}: prescribe='T_s' requires T_s")
            surf_forbidden = [
                n for n in ("w_theta_s", "w_qv_s") if getattr(self, n) is not None
            ]
            if surf_forbidden:
                raise BridgeError(
                    f"{self.case_name}: prescribe='T_s' forbids {surf_forbidden}"
                )
        else:  # "none"
            surf_forbidden = [
                n for n in ("T_s", "w_theta_s", "w_qv_s")
                if getattr(self, n) is not None
            ]
            if surf_forbidden:
                raise BridgeError(
                    f"{self.case_name}: prescribe='none' forbids surface field(s) "
                    f"{surf_forbidden}"
                )


# Array-valued fields serialized to/from npz (order-independent; a missing key on
# load ⇒ that optional channel was None). Scalar/string fields ride in a metadata
# key so an int/str never becomes a 0-d array on round-trip.
_ARTIFACT_ARRAY_FIELDS = (
    "heights_m", "times_s", "theta", "u", "v",
    "wtheta_resolved", "wtheta_sgs", "qt", "wqt_resolved", "wqt_sgs",
    "u_geo", "v_geo", "subsidence_w", "theta_adv", "qv_adv",
    "T_s", "w_theta_s", "w_qv_s",
)


def save_artifact(artifact: LESReferenceArtifact, path) -> None:
    """Write an artifact to a compressed ``.npz`` (I/O at the module edge).

    Optional (``None``) channels are simply omitted; scalars/strings (``case_name``,
    ``sgs``, ``f_c``, ``prescribe``) go in a JSON ``__meta__`` key so they survive
    the round-trip as their native types rather than 0-d arrays.
    """
    import json

    import numpy as np

    arrays = {}
    for name in _ARTIFACT_ARRAY_FIELDS:
        val = getattr(artifact, name)
        if val is not None:
            arrays[name] = np.asarray(val)
    meta = {
        "case_name": artifact.case_name,
        "sgs": artifact.sgs,
        "f_c": float(artifact.f_c),
        "prescribe": artifact.prescribe,
    }
    arrays["__meta__"] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8
    )
    np.savez_compressed(path, **arrays)


def load_artifact(path) -> LESReferenceArtifact:
    """Read an artifact written by :func:`save_artifact` (validates on construct)."""
    import json

    import numpy as np

    with np.load(path, allow_pickle=False) as npz:
        meta = json.loads(bytes(npz["__meta__"]).decode("utf-8"))
        kwargs = {
            name: jnp.asarray(npz[name])
            for name in _ARTIFACT_ARRAY_FIELDS
            if name in npz.files
        }
    kwargs.update(meta)
    return LESReferenceArtifact(**kwargs)


def total_turbulent_flux(
    resolved: Array, sgs: Array | None
) -> Array:
    """Total horizontal-mean turbulent flux ``<w'φ'> = resolved + SGS``.

    In the spectral truth core the SGS closure is the *only* subgrid dissipation
    (LES_SUITE.md §2), so the physically comparable flux the SCM closure must
    reproduce is resolved + SGS. ``sgs=None`` (an idealized resolved-only artifact)
    returns the resolved part unchanged.
    """
    resolved = jnp.asarray(resolved)
    if sgs is None:
        return resolved
    return resolved + jnp.asarray(sgs)


@dataclass(frozen=True)
class LESTruth:
    """Extracted LES truth for one score protocol (D6).

    ``heights_m`` (nz,); every profile field is (nz,) for a single-time diagnostic
    snapshot or (nt, nz) for the full prognostic time series (``times_s`` (nt,)).
    ``wtheta``/``wqt`` are the *total* (resolved+SGS) turbulent fluxes.
    """

    case_name: str
    heights_m: Array
    times_s: Array
    theta: Array
    u: Array
    v: Array
    wtheta: Array
    qt: Array | None = None
    wqt: Array | None = None


def _time_index(artifact: LESReferenceArtifact, time_s: float | None) -> int:
    """Index of the artifact snapshot nearest ``time_s`` (default: last)."""
    times = jnp.asarray(artifact.times_s)
    if time_s is None:
        return int(times.shape[0] - 1)
    return int(jnp.argmin(jnp.abs(times - time_s)))


def diagnostic_truth(
    artifact: LESReferenceArtifact, *, time_s: float | None = None
) -> LESTruth:
    """LES truth at a single output time (D6 diagnostic score).

    The diagnostic score sets the SCM mean state to this snapshot and asks each
    closure for its flux; the comparison target is the LES total turbulent flux at
    the same instant. ``time_s`` selects the nearest snapshot (default: the last,
    i.e. the most-equilibrated profile).
    """
    k = _time_index(artifact, time_s)
    wtheta = total_turbulent_flux(artifact.wtheta_resolved, artifact.wtheta_sgs)[k]
    wqt = None
    if artifact.qt is not None:
        wqt = total_turbulent_flux(artifact.wqt_resolved, artifact.wqt_sgs)[k]
    return LESTruth(
        case_name=artifact.case_name,
        heights_m=jnp.asarray(artifact.heights_m),
        times_s=jnp.asarray(artifact.times_s)[k][None],
        theta=jnp.asarray(artifact.theta)[k],
        u=jnp.asarray(artifact.u)[k],
        v=jnp.asarray(artifact.v)[k],
        wtheta=wtheta,
        qt=None if artifact.qt is None else jnp.asarray(artifact.qt)[k],
        wqt=wqt,
    )


def prognostic_truth(
    artifact: LESReferenceArtifact, *, t0_s: float | None = None
) -> LESTruth:
    """LES truth as the full time series (D6 prognostic score).

    The prognostic score initializes the SCM to LES(t0) and integrates freely,
    comparing to the LES mean profiles at matched times. ``t0_s`` (default: the
    first sample) trims the leading spin-up; samples at or after it are kept.
    """
    times = jnp.asarray(artifact.times_s)
    if t0_s is None:
        start = 0
    else:
        # first sample AT OR AFTER t0 (not the nearest — argmin would keep an
        # earlier sample for a t0 between samples, contradicting "trim the spin-up").
        start = int(jnp.searchsorted(times, jnp.asarray(t0_s, dtype=times.dtype),
                                     side="left"))
        if start >= times.shape[0]:
            raise BridgeError(
                f"{artifact.case_name}: t0_s={t0_s} is after the last sample "
                f"{float(times[-1])}; no truth remains"
            )
    sl = slice(start, None)
    wtheta = total_turbulent_flux(artifact.wtheta_resolved, artifact.wtheta_sgs)[sl]
    wqt = None
    if artifact.qt is not None:
        wqt = total_turbulent_flux(artifact.wqt_resolved, artifact.wqt_sgs)[sl]
    return LESTruth(
        case_name=artifact.case_name,
        heights_m=jnp.asarray(artifact.heights_m),
        times_s=times[sl],
        theta=jnp.asarray(artifact.theta)[sl],
        u=jnp.asarray(artifact.u)[sl],
        v=jnp.asarray(artifact.v)[sl],
        wtheta=wtheta,
        qt=None if artifact.qt is None else jnp.asarray(artifact.qt)[sl],
        wqt=wqt,
    )


def _const_profile_fn(profile: Array):
    """Wrap a static ``(nz,)`` profile as a time-independent SCMForcing channel."""
    arr = jnp.asarray(profile)

    def _fn(_t_seconds: float) -> Array:
        return arr

    return _fn


def _interp_scalar_fn(times_s: Array, values: Array):
    """Wrap a ``(nt,)`` surface series as a time-interpolated scalar channel."""
    t = jnp.asarray(times_s)
    v = jnp.asarray(values)

    def _fn(t_seconds: float) -> Array:
        return jnp.interp(jnp.asarray(t_seconds, dtype=v.dtype), t, v)

    return _fn


def artifact_to_scm_forcing(
    artifact: LESReferenceArtifact, *, scm_top_to_bottom: bool = True
) -> SCMForcing:
    """Reconstruct the exact large-scale forcing the LES received (D6 prognostic).

    Builds an :class:`SCMForcing` whose channels reproduce the artifact's stored
    forcing. Geostrophic wind, subsidence, and advective tendencies are
    time-independent profiles (the idealized LES cases hold them fixed); surface
    channels (``T_s`` / ``w_theta_s`` / ``w_qv_s``) are linearly interpolated in
    time from the stored series so a variable surface forcing (e.g. GABLS1's
    prescribed cooling) is honored. Disabled (``None``) channels stay ``None``.

    Because the forcing comes from the artifact — not re-derived from the registry
    case — the SCM is driven by byte-identical forcing to the LES, satisfying the
    controlled-comparison rule structurally.

    Orientation: artifact profiles are stored **surface-first** (heights increasing
    with index), the LES-native ordering the counter-gradient diagnostic assumes.
    The SCM state is stored **top-to-bottom** (``large_scale_forcing.py``: index 0 =
    model top). With ``scm_top_to_bottom=True`` (default) every vertical forcing
    profile is reversed so the returned forcing is directly usable by the SCM —
    NOT a silently mis-oriented pass-through. Set it ``False`` to keep the artifact
    (surface-first) ordering (e.g. to drive a surface-first consumer). NOTE: this
    reversal assumes the SCM runs on the artifact's own vertical grid (levels
    reversed, same count); interpolating onto a *different* SCM grid is a separate
    regridding step the coupling layer must do explicitly.
    """
    def _orient(prof: Array) -> Array:
        arr = jnp.asarray(prof)
        return arr[::-1] if scm_top_to_bottom else arr

    kwargs: dict = {"f_c": float(artifact.f_c), "prescribe": artifact.prescribe}
    for name, target in (
        ("u_geo", "u_geo"),
        ("v_geo", "v_geo"),
        ("subsidence_w", "subsidence_w"),
        ("theta_adv", "theta_adv"),
        ("qv_adv", "qv_adv"),
    ):
        prof = getattr(artifact, name)
        if prof is not None:
            kwargs[target] = _const_profile_fn(_orient(prof))
    if artifact.prescribe == "T_s" and artifact.T_s is not None:
        kwargs["T_s"] = _interp_scalar_fn(artifact.times_s, artifact.T_s)
    if artifact.prescribe == "fluxes":
        if artifact.w_theta_s is not None:
            kwargs["w_th_s"] = _interp_scalar_fn(artifact.times_s, artifact.w_theta_s)
        if artifact.w_qv_s is not None:
            kwargs["w_qv_s"] = _interp_scalar_fn(artifact.times_s, artifact.w_qv_s)
    return SCMForcing(**kwargs)
