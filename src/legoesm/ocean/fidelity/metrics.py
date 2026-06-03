"""Tier-level metric reductions for ocean fidelity assessment.

Every function operates on saved snapshot arrays (numpy), not on the live
JAX state. They are designed to be called by Layer A pytest fixtures and
by Layer B report builder. None of them re-integrate the model: they
extract scalars and 1-D / 2-D fields from outputs that
``scripts/run_ocean_test_matrix.py`` has already produced.

All physical constants come from :mod:`legoesm.constants`. Pure numpy
keeps the metric layer free of JAX/Metal complications and trivially
testable in isolation.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from legoesm import constants

_EPS = 1e-30


def measured_phase_speed_2d(field_x_t: np.ndarray, dx: float, dt: float) -> float:
    """Dominant phase speed from a 2-D FFT of an ``(x, t)`` Hovmöller field.

    Parameters
    ----------
    field_x_t : np.ndarray, shape ``(n_x, n_t)``
        Real-valued snapshot of one variable along ``x`` as a function of time.
    dx : float
        Sample spacing in ``x`` (m).
    dt : float
        Sample spacing in ``t`` (s).

    Returns
    -------
    float
        Phase speed ``ω / k`` (m/s) at the spectral peak. Sign tracks the
        dominant peak — westward propagation returns negative speed.
    """
    if field_x_t.ndim != 2:
        raise ValueError(f"field_x_t must be 2-D, got shape {field_x_t.shape}")
    if dx <= 0 or dt <= 0:
        raise ValueError("dx and dt must be positive")
    n_x, n_t = field_x_t.shape
    centred = field_x_t - field_x_t.mean()
    spectrum = np.fft.fftshift(np.fft.fft2(centred))
    k = 2.0 * math.pi * np.fft.fftshift(np.fft.fftfreq(n_x, dx))
    omega = 2.0 * math.pi * np.fft.fftshift(np.fft.fftfreq(n_t, dt))
    power = np.abs(spectrum)
    # Zero out the k = 0 column so the peak search never hits divide-by-zero.
    k_zero_idx = n_x // 2
    power[k_zero_idx, :] = 0.0
    j, i = np.unravel_index(int(np.argmax(power)), power.shape)
    k_peak = k[j]
    omega_peak = omega[i]
    # numpy's forward FFT uses exp(-i(kx + ωt)). A rightward-propagating
    # cosine cos(k0·x − ω0·t) puts its spectral peak at (+k0, −ω0), so the
    # physical phase speed ``c = ω/k`` is recovered as ``−ω_peak/k_peak``.
    return float(-omega_peak / k_peak)


def adjustment_timescale(
    ke_t: np.ndarray, t: np.ndarray, plateau_frac: float = 0.9
) -> float:
    """First time at which KE crosses ``plateau_frac · max(KE)`` (s).

    If the threshold is never reached, the last time sample is returned so
    the caller still gets a finite number to compare against tolerance.
    """
    ke = np.asarray(ke_t, dtype=float)
    tt = np.asarray(t, dtype=float)
    if ke.shape != tt.shape:
        raise ValueError(f"ke and t must share shape, got {ke.shape} vs {tt.shape}")
    if ke.ndim != 1:
        raise ValueError("ke_t and t must be 1-D")
    if not (0.0 < plateau_frac < 1.0):
        raise ValueError(f"plateau_frac must lie in (0, 1), got {plateau_frac}")
    threshold = plateau_frac * float(ke.max())
    crossed = np.where(ke >= threshold)[0]
    if crossed.size == 0:
        return float(tt[-1])
    return float(tt[int(crossed[0])])


def thermal_wind_residual(
    rho: np.ndarray,
    u: np.ndarray,
    f: float,
    dy: float,
    dz: float,
    rho_0: float = constants.rho_water,
) -> np.ndarray:
    """Pointwise thermal-wind residual ``|∂u/∂z + g/(ρ_0 f) · ∂ρ/∂y|`` (1/s).

    Centred differences in z (axis 0) and y (axis 1). The residual is
    reported on the inner grid (z and y boundary rows are dropped).
    Inputs must share shape ``(nz, ny[, nx])``.
    """
    if f == 0:
        raise ValueError("thermal_wind_residual requires f != 0")
    if rho.shape != u.shape:
        raise ValueError(f"shape mismatch: rho {rho.shape} vs u {u.shape}")
    if rho.ndim < 2:
        raise ValueError(f"rho must have at least 2 dims, got {rho.ndim}")
    if dy <= 0 or dz <= 0:
        raise ValueError("dy and dz must be positive")
    if rho_0 <= 0:
        raise ValueError("rho_0 must be positive")
    du_dz = (u[2:, ...] - u[:-2, ...]) / (2.0 * dz)
    drho_dy = (rho[:, 2:, ...] - rho[:, :-2, ...]) / (2.0 * dy)
    du_dz_inner = du_dz[:, 1:-1, ...]
    drho_dy_inner = drho_dy[1:-1, :, ...]
    return np.abs(du_dz_inner + constants.g * drho_dy_inner / (rho_0 * f))


def overflow_nose_descent(
    rho_anom: np.ndarray,
    z_centers: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Maximum depth at which the density anomaly exceeds ``threshold``.

    Parameters
    ----------
    rho_anom : np.ndarray, shape ``(n_t, n_z, ...)``
        Density anomaly. Axis 0 is time, axis 1 is vertical, remaining axes
        are horizontal (any layout).
    z_centers : np.ndarray, shape ``(n_z,)``
        Vertical cell centres (m), positive downward.
    threshold : float
        Anomaly magnitude (kg/m^3) defining the nose front.

    Returns
    -------
    np.ndarray, shape ``(n_t,)``
        For each time, the maximum depth at which any horizontal cell
        carries ``rho_anom >= threshold``. Times that never reach the
        threshold return ``0.0``.
    """
    if rho_anom.ndim < 2:
        raise ValueError(f"rho_anom needs time + z axes, got {rho_anom.ndim}")
    if z_centers.ndim != 1 or z_centers.size != rho_anom.shape[1]:
        raise ValueError("z_centers must be 1-D and match rho_anom axis 1")
    n_t, n_z = rho_anom.shape[:2]
    flat = rho_anom.reshape(n_t, n_z, -1)
    nose = np.zeros(n_t, dtype=float)
    for it in range(n_t):
        any_above = (flat[it] >= threshold).any(axis=1)
        if any_above.any():
            nose[it] = float(z_centers[int(np.where(any_above)[0].max())])
    return nose


def boundary_layer_width(u_midline: np.ndarray, x: np.ndarray) -> float:
    """Western-boundary-current full width at half maximum of ``|u|`` (m)."""
    u = np.abs(np.asarray(u_midline, dtype=float))
    xx = np.asarray(x, dtype=float)
    if u.shape != xx.shape:
        raise ValueError(f"shape mismatch: u {u.shape} vs x {xx.shape}")
    if u.ndim != 1 or u.size < 3:
        raise ValueError("need a 1-D u with at least 3 samples")
    peak_idx = int(np.argmax(u))
    peak = float(u[peak_idx])
    if peak <= 0:
        return 0.0
    above = u >= 0.5 * peak
    if not above.any():
        return 0.0
    left = peak_idx
    while left > 0 and above[left - 1]:
        left -= 1
    right = peak_idx
    while right < u.size - 1 and above[right + 1]:
        right += 1
    return float(xx[right] - xx[left])


def sverdrup_residual(
    v_integrated: np.ndarray,
    curl_tau: np.ndarray,
    beta: float,
    rho_0: float = constants.rho_water,
) -> np.ndarray:
    """Sverdrup-balance residual ``|β · V − curl(τ)/ρ_0|`` (units: 1/s)."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    if rho_0 <= 0:
        raise ValueError("rho_0 must be positive")
    if v_integrated.shape != curl_tau.shape:
        raise ValueError(
            f"shape mismatch: v_integrated {v_integrated.shape} vs "
            f"curl_tau {curl_tau.shape}"
        )
    return np.abs(beta * np.asarray(v_integrated, dtype=float)
                  - np.asarray(curl_tau, dtype=float) / rho_0)


def linear_growth_rate(
    ke_pert_t: np.ndarray,
    t: np.ndarray,
    t_window: Tuple[float, float],
) -> float:
    """Exponential growth-rate fit ``ln(KE) ≈ 2σ·t + const`` over a window.

    Returns ``σ`` (1/s). Raises if the window contains fewer than 3 samples
    or any non-positive KE value.
    """
    ke = np.asarray(ke_pert_t, dtype=float)
    tt = np.asarray(t, dtype=float)
    if ke.shape != tt.shape:
        raise ValueError(f"shape mismatch: ke {ke.shape} vs t {tt.shape}")
    if ke.ndim != 1:
        raise ValueError("ke_pert_t and t must be 1-D")
    t_start, t_end = t_window
    if not (t_end > t_start):
        raise ValueError(f"t_window must have t_end > t_start, got {t_window}")
    mask = (tt >= t_start) & (tt <= t_end)
    if int(mask.sum()) < 3:
        raise ValueError(f"t_window {t_window} selects < 3 samples")
    if np.any(ke[mask] <= 0):
        raise ValueError("ke_pert_t must be strictly positive in fit window")
    slope, _intercept = np.polyfit(tt[mask], np.log(ke[mask]), 1)
    return float(slope / 2.0)


def bottom_form_stress(
    p_b: np.ndarray,
    dh_b_dx: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Area-weighted mean bottom form stress ``⟨p_b · ∂h_b/∂x⟩`` (Pa)."""
    pb = np.asarray(p_b, dtype=float)
    dh = np.asarray(dh_b_dx, dtype=float)
    if pb.shape != dh.shape:
        raise ValueError(f"shape mismatch: p_b {pb.shape} vs dh_b_dx {dh.shape}")
    if weights is None:
        w = np.ones_like(pb)
    else:
        if weights.shape != pb.shape:
            raise ValueError(
                f"weights shape {weights.shape} must match p_b {pb.shape}"
            )
        w = np.asarray(weights, dtype=float)
    w_sum = float(w.sum())
    return float(np.sum(w * pb * dh) / max(w_sum, _EPS))
