"""Ensemble Transform Kalman Inversion (ETKI) — minimal in-repo optimizer.

The gradient-free calibration method of the GEOMETRIC campaign
(``docs/planning/geometric_calibration_campaign.md`` §5.2): Iglesias et al.
(2013) Ensemble Kalman Inversion in the transform variant of Huang et al.
(2022), exactly as used by Perezhogin, Adcroft & Zanna (arXiv:2604.06398,
their Eqs. 4-7). API mirrors ``EnsembleKalmanProcesses.jl`` (ensemble in,
forward evaluations in, updated ensemble out; explicit scheduler step) so a
swap to that package stays trivial.

Loss being minimized: ``L(θ) = || R^{-1/2} (y - G(θ)) ||²``. One iteration:

    θ̄⁺ = θ̄ + δt Θ (I + δt Gᵀ R⁻¹ G)⁻¹ Gᵀ R⁻¹ (y - ḡ)        (mean update)
    Θ⁺ = Θ (I + δt Gᵀ R⁻¹ G)^{-1/2}                          (shrink)

with Θ (n_p × n_e) and G (n_o × n_e) the 1/√(n_e-1)-normalized perturbation
matrices. All linear algebra is in ensemble space (n_e × n_e), so n_o can be
large (full observation maps). Members that returned non-finite forward
evaluations (blow-ups) are dropped from the statistics for that iteration —
the Perezhogin NW2 lesson (size the ensemble up rather than crash).

fp64 throughout; value-pure functions of arrays with HOST-side control
flow (Python raise/loop on member validity) — for the outer calibration
loop, deliberately NOT jit-traceable. No model coupling here.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class ETKIStep(NamedTuple):
    """Result of one ETKI iteration."""
    theta: jnp.ndarray        # (n_e, n_p) updated ensemble
    misfit: float             # ||R^{-1/2}(y - g_mean)||^2 BEFORE the update
    n_valid: int              # members with finite forward evaluations


def etki_update(
    theta: jnp.ndarray,        # (n_e, n_p) parameter ensemble
    g: jnp.ndarray,            # (n_e, n_o) forward evaluations G(theta_i)
    y: jnp.ndarray,            # (n_o,) observations
    *,
    dt: float = 1.0,           # scheduler step (Iglesias & Yang 2021)
    r_diag: jnp.ndarray | None = None,  # (n_o,) diag of R; None -> identity
) -> ETKIStep:
    """One ETKI iteration (mean update + ensemble shrink to consensus)."""
    theta = jnp.asarray(theta, jnp.float64)
    g = jnp.asarray(g, jnp.float64)
    y = jnp.asarray(y, jnp.float64)
    n_e = theta.shape[0]
    if g.shape[0] != n_e:
        raise ValueError(f"theta has {n_e} members but g has {g.shape[0]}")
    if dt <= 0.0:
        raise ValueError(f"scheduler step dt must be > 0, got {dt!r}")

    # Drop blown-up members from the statistics for this iteration.
    valid = jnp.all(jnp.isfinite(g), axis=1)
    n_valid = int(jnp.sum(valid))
    if n_valid < 2:
        raise RuntimeError(
            f"ETKI: only {n_valid} ensemble member(s) returned finite "
            "forward evaluations — increase the ensemble size or shrink "
            "the prior spread (blow-up handling per Perezhogin et al.)")
    w = valid.astype(jnp.float64)
    n_eff = jnp.sum(w)

    r_inv = (jnp.ones(y.shape[0], jnp.float64) if r_diag is None
             else 1.0 / jnp.asarray(r_diag, jnp.float64))

    # Sanitize BEFORE any arithmetic with the invalid rows: masking by
    # multiplication after subtraction would give NaN*0 = NaN.
    g_safe = jnp.where(valid[:, None], g, 0.0)
    theta_mean = jnp.sum(theta * w[:, None], axis=0) / n_eff
    g_mean = jnp.sum(g_safe * w[:, None], axis=0) / n_eff
    norm = 1.0 / jnp.sqrt(n_eff - 1.0)
    # Invalid members contribute ZERO perturbation columns; on output they
    # are RESET to exactly the updated ensemble mean (their prior values
    # are discarded) — the ensemble re-seeds the blown-up member at the
    # consensus point.
    Theta = (theta - theta_mean[None, :]) * (w * norm)[:, None]   # (n_e,n_p)
    G = (g_safe - g_mean[None, :]) * (w * norm)[:, None]          # (n_e,n_o)

    misfit = float(jnp.sum(r_inv * (y - g_mean) ** 2))

    # Ensemble-space system A = I + dt * G R^-1 G^T  (n_e x n_e, symmetric PSD)
    GR = G * r_inv[None, :]                                        # (n_e,n_o)
    A = jnp.eye(n_e, dtype=jnp.float64) + dt * (GR @ G.T)

    # Mean update: theta_mean += dt * Theta^T A^{-1} G R^{-1} (y - g_mean)
    rhs = GR @ (y - g_mean)                                        # (n_e,)
    mean_inc = dt * (Theta.T @ jnp.linalg.solve(A, rhs))           # (n_p,)
    theta_mean_new = theta_mean + mean_inc

    # Shrink: Theta_new^T = Theta^T A^{-1/2}  (symmetric inverse sqrt by eigh)
    evals, evecs = jnp.linalg.eigh(A)
    a_inv_sqrt = (evecs * (1.0 / jnp.sqrt(jnp.maximum(evals, 1e-300)))) @ evecs.T
    Theta_new = a_inv_sqrt @ Theta                                  # (n_e,n_p)

    theta_new = theta_mean_new[None, :] + Theta_new / norm
    return ETKIStep(theta=theta_new, misfit=misfit, n_valid=n_valid)


def run_etki(
    forward_fn,                # theta_batch (n_e,n_p) -> g_batch (n_e,n_o)
    theta0: jnp.ndarray,       # (n_e, n_p) initial ensemble
    y: jnp.ndarray,
    *,
    n_iterations: int = 5,
    dt: float = 1.0,
    r_diag: jnp.ndarray | None = None,
    callback=None,             # (iteration, ETKIStep) -> None
) -> tuple[jnp.ndarray, list[float]]:
    """Driver loop: returns (final ensemble, per-iteration misfits).

    ``forward_fn`` receives the UNCONSTRAINED ensemble; parameter
    transforms (``trainable_ocean_params.constrain``) belong inside it.
    Early stopping is the CALLER's job (monitor held-out metrics in the
    callback — the structural-error plateau lesson).
    """
    theta = jnp.asarray(theta0, jnp.float64)
    misfits: list[float] = []
    for j in range(n_iterations):
        g = forward_fn(theta)
        step = etki_update(theta, g, y, dt=dt, r_diag=r_diag)
        theta = step.theta
        misfits.append(step.misfit)
        if callback is not None:
            callback(j, step)
    return theta, misfits
