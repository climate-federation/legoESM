"""LES intercomparison gate (D7 / gate-0): LES output vs published envelopes.

LES_SUITE.md D7 makes LES credibility a **hard gate**: the truth core must
reproduce a published intercomparison before any downstream tuning result is
trusted (and D10 puts the buoyant-path gate-0 — the Nieuwstadt 1993 dry CBL —
*before* the suite build). This module scores a finished LES run's horizontal-mean
profiles against the **universal convective-scaling envelope** every faithful dry
CBL LES reproduces, and returns a per-metric pass/fail + an overall gate verdict.

It is deliberately tolerance-*banded*, not point-matching: the reference is a range
across the published LES codes, so a value inside the band passes and a value
outside is a real failure of the buoyant path. Pure array math on already-computed
profiles — no LES integration, no plotting.

Reference envelope (dry, free-convective CBL; prescribed surface heat flux + capping
inversion):
  * ``max σ_w / w_*``            ≈ 0.6   — the canonical peak of the vertical-velocity
    variance normalized by the convective velocity ``w_* = (g/θ · Q0 · z_i)^{1/3}``.
  * mixed-layer ``∂⟨θ⟩/∂z``      ≈ 0     — a WELL-MIXED bulk layer.
  * surface ``⟨w'θ'⟩ / Q0``       ≈ 1     — the resolved+SGS heat flux recovers the
    prescribed surface flux near the ground.
  * entrainment ``min ⟨w'θ'⟩ / Q0`` ≈ −0.2 — the negative flux MINIMUM at the base
    of the inversion (entrainment ratio ``A_R = −min⟨w'θ'⟩/Q0 ≈ 0.1–0.2``). NB this
    minimum sits *below* the max-∂θ/∂z height, so it is taken as the flux minimum
    through the inversion band, not the flux at the θ-gradient peak.

References: Nieuwstadt et al. (1993) convective-BL LES intercomparison; the ``A_R``
entrainment-ratio range from the mixed-layer / LES literature (Moeng & Sullivan
1994; Sullivan & Patton 2011). Bands widened to the reported inter-code spread.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class MetricBand:
    """One reference metric: a target and an accept band ``[lo, hi]``."""

    name: str
    target: float
    lo: float
    hi: float
    units: str
    description: str

    def check(self, value: float) -> MetricResult:
        ok = bool(np.isfinite(value) and self.lo <= value <= self.hi)
        return MetricResult(band=self, value=float(value), passed=ok)


@dataclass(frozen=True)
class MetricResult:
    band: MetricBand
    value: float
    passed: bool


# --- the published dry-CBL convective-scaling envelope (see module docstring) ---
# Bands span the reported inter-code spread; the target is the canonical value.
CBL_ENVELOPE: tuple[MetricBand, ...] = (
    MetricBand(
        "sigma_w_over_wstar_max", target=0.60, lo=0.50, hi=0.72,
        units="1", description="peak σ_w/w_* (vertical-velocity variance scaling)",
    ),
    MetricBand(
        "mixed_layer_dtheta_dz_mK_m", target=0.0, lo=-0.6, hi=0.6,
        units="mK/m", description="bulk mixed-layer ∂⟨θ⟩/∂z (well-mixed ⇒ ~0)",
    ),
    MetricBand(
        "surface_flux_ratio", target=1.0, lo=0.85, hi=1.08,
        units="1", description="⟨w'θ'⟩/Q0 near the surface",
    ),
    MetricBand(
        "entrainment_flux_ratio", target=-0.20, lo=-0.40, hi=-0.05,
        units="1", description="min ⟨w'θ'⟩/Q0 through inversion (entrainment, A_R≈0.1-0.2)",
    ),
)


@dataclass(frozen=True)
class CBLDiagnostics:
    """Convective-scaling diagnostics extracted from a CBL LES mean profile."""

    z_i_m: float
    w_star_m_s: float
    sigma_w_over_wstar_max: float
    mixed_layer_dtheta_dz_mK_m: float
    surface_flux_ratio: float
    entrainment_flux_ratio: float


def cbl_diagnostics(
    z: Array,
    theta_mean: Array,
    w_variance: Array,
    wtheta_flux: Array,
    *,
    Q0: float,
    theta0: float = 300.0,
    g: float | None = None,
    z_i_m: float | None = None,
) -> CBLDiagnostics:
    """Extract the CBL convective-scaling diagnostics from mean profiles.

    Parameters
    ----------
    z : (nz,) heights [m], surface-first (increasing).
    theta_mean : (nz,) horizontal-mean potential temperature [K].
    w_variance : (nz,) ⟨w'w'⟩ [m^2/s^2].
    wtheta_flux : (nz,) total ⟨w'θ'⟩ [K m/s] (resolved+SGS ideally; resolved-only
        near the surface underestimates the ratio — pass the total when available).
    Q0 : prescribed surface kinematic heat flux [K m/s].
    theta0 : reference potential temperature [K] for w_*.
    g : gravity [m/s^2]; defaults to ``legoesm.constants.g``.
    z_i_m : inversion height [m]; if ``None`` it is diagnosed as the height of
        maximum ∂⟨θ⟩/∂z above the surface layer.
    """
    g_override = None if g is None else float(g)
    z = np.asarray(z, dtype=float)
    theta_mean = np.asarray(theta_mean, dtype=float)
    w_variance = np.asarray(w_variance, dtype=float)
    wtheta_flux = np.asarray(wtheta_flux, dtype=float)
    if not (z.shape == theta_mean.shape == w_variance.shape == wtheta_flux.shape):
        raise ValueError("z, theta_mean, w_variance, wtheta_flux must share shape")
    if z.ndim != 1 or z.shape[0] < 4:
        raise ValueError("need a 1-D profile with >=4 levels")
    if not (np.isfinite(Q0) and Q0 > 0):
        raise ValueError(f"Q0 must be a positive heat flux, got {Q0!r}")

    dtheta_dz = np.gradient(theta_mean, z)
    # inversion height = height of max ∂θ/∂z above the near-surface layer (skip the
    # first few points to avoid the surface-layer gradient).
    skip = 3
    if z_i_m is None:
        k_inv = skip + int(np.argmax(dtheta_dz[skip:]))
        z_i = float(z[k_inv])
    else:
        z_i = float(z_i_m)
        k_inv = int(np.argmin(np.abs(z - z_i)))

    # g/θ₀ from the canonical helper (physics._shared.buoyancy_coefficient),
    # which takes the gravity so an explicit override stays bit-exact.
    from legoesm.atmosphere.physics._shared import buoyancy_coefficient
    g_over_theta0 = float(buoyancy_coefficient(theta0, gravity=g_override))
    w_star = (g_over_theta0 * Q0 * z_i) ** (1.0 / 3.0)
    sigma_w = np.sqrt(np.maximum(w_variance, 0.0))
    sigma_w_over_wstar_max = float(np.max(sigma_w) / w_star)

    # surface flux ratio: use the first interior level (skip the very bottom point,
    # whose resolved flux is near zero by the no-penetration boundary).
    surface_flux_ratio = float(wtheta_flux[1] / Q0)

    # Entrainment flux ratio = the MINIMUM of ⟨w'θ'⟩/Q0 through the inversion — the
    # standard entrainment-ratio definition A_R = -min⟨w'θ'⟩/Q0. This minimum sits
    # at the *base* of the inversion, which is BELOW the height of maximum ∂θ/∂z
    # (the flux has already recovered toward 0 in the mid-inversion): evaluating at
    # the θ-gradient max would spuriously report ~0. Search a band around z_i
    # (surface layer .. 1.3 z_i) so a far-aloft flux wiggle cannot be picked up.
    k_top = int(np.searchsorted(z, 1.3 * z_i, side="right"))
    k_top = min(max(k_top, k_inv + 1), z.shape[0])
    entrainment_flux_ratio = float(np.min(wtheta_flux[skip:k_top]) / Q0)

    # bulk mixed-layer ∂θ/∂z averaged over 0.2 z_i .. 0.8 z_i (mK/m).
    lo, hi = int(0.2 * k_inv), int(0.8 * k_inv)
    if hi <= lo:
        lo, hi = 1, max(2, k_inv)
    ml_dtheta_dz = float(np.mean(dtheta_dz[lo:hi]) * 1000.0)

    return CBLDiagnostics(
        z_i_m=z_i,
        w_star_m_s=float(w_star),
        sigma_w_over_wstar_max=sigma_w_over_wstar_max,
        mixed_layer_dtheta_dz_mK_m=ml_dtheta_dz,
        surface_flux_ratio=surface_flux_ratio,
        entrainment_flux_ratio=entrainment_flux_ratio,
    )


@dataclass(frozen=True)
class GateResult:
    """Overall intercomparison-gate verdict + per-metric results."""

    passed: bool
    results: tuple[MetricResult, ...]
    diagnostics: CBLDiagnostics

    def report(self) -> str:
        lines = [
            f"CBL intercomparison gate: {'PASS' if self.passed else 'FAIL'}",
            f"  z_i = {self.diagnostics.z_i_m:.0f} m   "
            f"w_* = {self.diagnostics.w_star_m_s:.2f} m/s",
        ]
        for r in self.results:
            flag = "ok " if r.passed else "XX "
            lines.append(
                f"  [{flag}] {r.band.name:<28} = {r.value:+.3f} {r.band.units:<5} "
                f"target {r.band.target:+.2f} band [{r.band.lo:+.2f},{r.band.hi:+.2f}]"
            )
        return "\n".join(lines)


def evaluate_cbl_gate(diagnostics: CBLDiagnostics) -> GateResult:
    """Score CBL diagnostics against :data:`CBL_ENVELOPE`; all bands must pass."""
    value_by_name = {
        "sigma_w_over_wstar_max": diagnostics.sigma_w_over_wstar_max,
        "mixed_layer_dtheta_dz_mK_m": diagnostics.mixed_layer_dtheta_dz_mK_m,
        "surface_flux_ratio": diagnostics.surface_flux_ratio,
        "entrainment_flux_ratio": diagnostics.entrainment_flux_ratio,
    }
    results = tuple(band.check(value_by_name[band.name]) for band in CBL_ENVELOPE)
    return GateResult(
        passed=all(r.passed for r in results),
        results=results,
        diagnostics=diagnostics,
    )


def gate_from_cbl_profiles(
    z: Array,
    theta_mean: Array,
    w_variance: Array,
    wtheta_flux: Array,
    *,
    Q0: float,
    theta0: float = 300.0,
    z_i_m: float | None = None,
) -> GateResult:
    """Convenience: diagnostics + gate evaluation from raw CBL mean profiles."""
    diag = cbl_diagnostics(
        z, theta_mean, w_variance, wtheta_flux,
        Q0=Q0, theta0=theta0, z_i_m=z_i_m,
    )
    return evaluate_cbl_gate(diag)
