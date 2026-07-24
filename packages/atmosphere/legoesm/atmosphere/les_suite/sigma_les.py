"""σ_LES — the LES's OWN spread (D7), on the SCM tuner's exact loss scale.

LES_SUITE.md D7: the Q3 inter-regime coefficient spread and the Q1b local→nonlocal
skill margin are results ONLY where they exceed σ_LES. A difference smaller than the
LES's own SGS-model (+ resolution) spread is not a result — it is within the LES's own
pairwise SGS/resolution spread (a comparison SCALE, not a strict statistical floor; see
the calibration caveat below).

This module aggregates a set of SAME-CASE LES artifacts — the ``{lasd, smagorinsky,
vreman}`` SGS spread emitted by ``run_les_suite --sgs`` (± 2×-resolution runs) — into a
σ_LES magnitude. It scores each variant PAIR with the SAME objective the derivative-free
tuner minimises: :func:`scm_runner.final_prognostic_truth` at the LAST output time fed to
:func:`score.prognostic_profile_score` — i.e. the FINAL-snapshot, std-normalized,
mass-weighted θ/u/v RMSE. Sharing that exact objective is what makes the number
comparable to a closure's tuned loss and to a Q1b margin (both also final-snapshot).

Definition (be precise about what this is): ``sigma_combined`` is the RMS over ORDERED
variant pairs ``(A→B)`` of ``prognostic_profile_score(truth=A_final, B_final).combined``
— a **bidirectional pairwise RMS distance** between LES runs of the same case with
IDENTICAL forcing that differ only in the SGS/resolution choice. Ordered pairs are used
because the score's normalization is truth-dependent (asymmetric); both directions are
kept. This is a comparison SCALE, NOT a population standard deviation (for a symmetric
metric it would be √2× the population RMS deviation), and a variant with an unusually
small truth-profile std can dominate it — so treat the exact D7 threshold as needing
calibration, not as a strict statistical σ. Pure assembly over cached artifacts + the
existing tuner objective (no LES, no SCM, no new numerics).
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from legoesm.atmosphere.les_suite.bridge import LESReferenceArtifact
from legoesm.atmosphere.les_suite.scm_coupling import interp_profile
from legoesm.atmosphere.les_suite.scm_runner import final_prognostic_truth
from legoesm.atmosphere.les_suite.score import prognostic_profile_score

_EXTENT_TOL_M = 1.0  # same-case artifacts must share surface+top height to this tol [m]


class SigmaLESError(ValueError):
    """Raised when the artifact set is unfit for a σ_LES estimate."""


@dataclass(frozen=True)
class SigmaLES:
    """σ_LES for one case, on the tuner's (dimensionless, final-snapshot normalized-RMSE)
    loss scale. See the module docstring: a bidirectional pairwise RMS distance, not a
    population standard deviation."""

    case_name: str
    variants: tuple            # the artifacts' sgs labels, in input order
    n_pairs: int               # ordered variant pairs scored (N*(N-1))
    sigma_combined: float      # RMS over pairs of the combined final-snapshot score
    sigma_theta: float         # per-variable RMS over pairs (θ_l)
    sigma_u: float
    sigma_v: float
    per_pair: tuple            # ((truth_label, candidate_label, combined), ...)


def _rms(xs: list[float]) -> float:
    if not xs:
        return 0.0
    arr = jnp.asarray(xs)
    return float(jnp.sqrt(jnp.mean(arr ** 2)))


def _require_finite(a: LESReferenceArtifact) -> None:
    # These are TRUTH artifacts; a non-finite value is corruption. safe_sqrt inside the
    # score maps NaN→0 (a "perfect" fit), so a corrupt variant would SILENTLY LOWER
    # σ_LES — reject loudly instead (mirrors scm_les_final_loss's finite guard intent).
    for name in ("heights_m", "times_s", "theta", "u", "v"):
        arr = jnp.asarray(getattr(a, name))
        if not bool(jnp.all(jnp.isfinite(arr))):
            raise SigmaLESError(f"{a.case_name}: artifact {a.sgs!r} has non-finite {name}")


def _times_match(a: LESReferenceArtifact, b: LESReferenceArtifact) -> bool:
    ta = jnp.asarray(a.times_s)
    tb = jnp.asarray(b.times_s)
    if ta.shape != tb.shape:
        return False
    scale = float(jnp.maximum(jnp.max(jnp.abs(ta)), 1.0))
    return bool(jnp.all(jnp.abs(ta - tb) <= 1e-6 * scale))


def _extents_match(a: LESReferenceArtifact, b: LESReferenceArtifact) -> bool:
    za = jnp.asarray(a.heights_m)
    zb = jnp.asarray(b.heights_m)
    return (abs(float(za[0]) - float(zb[0])) <= _EXTENT_TOL_M
            and abs(float(za[-1]) - float(zb[-1])) <= _EXTENT_TOL_M)


def sigma_les_prognostic(
    artifacts: list[LESReferenceArtifact], *, weights=None
) -> SigmaLES:
    """σ_LES on the tuner's loss scale from ≥2 same-case LES artifacts.

    Each ordered pair ``(A→B)`` contributes ``prognostic_profile_score(truth=A_final,
    B_final regridded to A's heights).combined`` (the tuner's final-snapshot objective);
    σ_LES is their RMS (and per-variable RMS). Artifacts must be the SAME case, share the
    output-time schedule, and share vertical extent (so a 2×-resolution run — finer nz,
    same domain — compares to a coarse one WITHOUT extrapolating past either top). Rejects
    <2 / mixed-case / moist / non-finite / mismatched-time / mismatched-extent inputs.
    """
    if len(artifacts) < 2:
        raise SigmaLESError(
            f"σ_LES needs >=2 artifacts (the SGS/resolution spread); got {len(artifacts)}")
    cases = {a.case_name for a in artifacts}
    if len(cases) != 1:
        raise SigmaLESError(f"all artifacts must be the SAME case; got {sorted(cases)}")
    if any(a.is_moist for a in artifacts):
        raise SigmaLESError(
            "sigma_les_prognostic is dry-only (θ,u,v); a moist artifact was passed")
    for a in artifacts:
        _require_finite(a)

    comb, th_terms, u_terms, v_terms = [], [], [], []
    per_pair: list[tuple] = []
    for i, a in enumerate(artifacts):
        z_a = jnp.asarray(a.heights_m)
        truth = final_prognostic_truth(a, z_a)  # A's FINAL snapshot, tuner objective
        for j, b in enumerate(artifacts):
            if i == j:
                continue
            if not _times_match(a, b):
                raise SigmaLESError(
                    f"{a.case_name}: artifacts {a.sgs!r} and {b.sgs!r} do not share the "
                    "output-time schedule; σ_LES needs matched snapshot times")
            if not _extents_match(a, b):
                raise SigmaLESError(
                    f"{a.case_name}: artifacts {a.sgs!r} and {b.sgs!r} span different "
                    "vertical extents; σ_LES needs the same domain (no extrapolation)")
            z_b = jnp.asarray(b.heights_m)
            scm_theta = interp_profile(jnp.asarray(b.theta)[-1], z_b, z_a)
            scm_u = interp_profile(jnp.asarray(b.u)[-1], z_b, z_a)
            scm_v = interp_profile(jnp.asarray(b.v)[-1], z_b, z_a)
            s = prognostic_profile_score(truth, scm_theta, scm_u, scm_v, weights=weights)
            comb.append(float(s.combined))
            th_terms.append(float(s.theta_rmse))
            u_terms.append(float(s.u_rmse))
            v_terms.append(float(s.v_rmse))
            per_pair.append((a.sgs, b.sgs, float(s.combined)))  # (truth, candidate, d)
    return SigmaLES(
        case_name=next(iter(cases)),
        variants=tuple(a.sgs for a in artifacts),
        n_pairs=len(comb),
        sigma_combined=_rms(comb),
        sigma_theta=_rms(th_terms),
        sigma_u=_rms(u_terms),
        sigma_v=_rms(v_terms),
        per_pair=tuple(per_pair),
    )
