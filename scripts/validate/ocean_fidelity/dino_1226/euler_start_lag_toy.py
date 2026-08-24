#!/usr/bin/env python
"""Why a forward-Euler FIRST step costs a leap-frog half a step of phase.

This is the mechanism behind the retraction in
``docs/ocean/fidelity/dino_eta_wave_field_result.md``: the DINO eta-wave twin
ran legoESM without ``--bridge-before``, so legoESM took a forward-Euler first
step while NEMO leap-frogged from step 1, and the measured impulse-response lag
was 0.47 of a baroclinic step.  A committed toy is the difference between an
asserted mechanism and a checkable one.

THE IDENTITY (gamma = 0, no Asselin filter).  Both arms start from the same two
levels -- the impulse was written into ``sshb`` AND ``sshn``, and the Euler
branch sets ``before := now`` -- so for a mode ``deta/dt = i*omega*eta`` with
``s = omega*dt`` the only difference is the first step::

    NEMO     d1 = d_before + 2is*d_now = d(1 + 2is)     leap-frog over 2dt
    legoESM  d1 = d_now    +  is*d_now = d(1 +  is)     forward Euler over dt

Both then obey the SAME three-term recurrence, and the two-point running mean
of the leap-frog trajectory reproduces the Euler trajectory's first two levels,
so by induction::

    eta_euler[k] == ( eta_lf[k-1] + eta_lf[k] ) / 2      EXACTLY

A two-point running mean is a HALF-SAMPLE DELAY at every frequency (its phase
is exactly -pi*f, linear through the origin).  That single line is the whole
mechanism, and it explains why the lag is broadband and frequency-flat rather
than a propagation-speed error.

WHAT THE MODEL ACTUALLY RUNS is gamma = 0.1 (``rn_atfp``), and NEMO filters
from step 1 while legoESM's Euler branch does not
(``ocean_model_latlon_cgrid.py:8058`` -- "no RA filter").  That asymmetry tilts
the answer slightly UPWARD and makes it period-dependent, rising from ~0.504 at
a 6-step period to ~0.555 at 80 steps.  Both numbers bracket the two estimators
the twin reported (+0.470 backward, +0.551 centred).

STABILITY CAVEAT, which bounds what this toy may be asked.  The filtered
leap-frog is unstable at short periods: at gamma=0.1 the P=4 and P=5 columns
reach 1e25 and 1e16 over 159 samples, so any broadband number that includes
them is meaningless.  P=4 sits exactly at the scheme's own limit (s = sin(pi/2)
= 1).  The toy is quoted for periods >= 6 only.

Usage
-----
    python euler_start_lag_toy.py
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

# Robert-Asselin coefficient on the DINO card (NEMO rn_atfp).
GAMMA_DINO = 0.1
# Periods (in baroclinic steps) the toy is quoted at.  The filtered leap-frog
# is unstable below ~6, so STABLE_MIN_PERIOD bounds every aggregate.
TOY_PERIODS = (4, 5, 6, 8, 12, 20, 40, 80)
STABLE_MIN_PERIOD = 6
N_SAMPLES = 159


def run_mode(period_steps: float, gamma: float, euler_start: bool,
             n_samples: int = N_SAMPLES) -> np.ndarray:
    """One oscillatory mode through a leap-frog, with or without an Euler start.

    ``euler_start`` reproduces legoESM's un-bridged entry
    (``ocean_model_latlon_cgrid.py:8496-8520``): a forward-Euler step over
    ``dt`` from ``before := now``, and NO Asselin filter on that step.  The
    ``False`` branch is NEMO's: leap-frog over ``2dt`` from a real before level,
    filtered from step 1.
    """
    if period_steps <= 2.0:
        raise ValueError(
            f"period_steps must exceed the 2-step Nyquist, got {period_steps!r}")
    s = np.sin(2.0 * np.pi / period_steps)          # omega*dt
    before = now = 1.0 + 0j
    out = np.empty(n_samples)
    for k in range(n_samples):
        if k == 0 and euler_start:
            after = now + 1j * s * now              # Euler over dt, unfiltered
            filtered = now
        else:
            after = before + 2j * s * now           # leap-frog over 2dt
            filtered = now + gamma * (before - 2.0 * now + after)
        before, now = filtered, after
        out[k] = after.real
    return out


def running_mean_residual(period_steps: float,
                          n_samples: int = N_SAMPLES) -> float:
    """max|euler_trajectory - 2-point running mean of the leap-frog one|, gamma=0.

    Zero (to roundoff) is the identity that makes the half-step exact.
    """
    lf = run_mode(period_steps, 0.0, False, n_samples)
    eu = run_mode(period_steps, 0.0, True, n_samples)
    seeded = np.concatenate([[1.0], lf])            # level 0 is the shared seed
    return float(np.abs(eu - 0.5 * (seeded[:-1] + seeded[1:])).max())


def fit_alpha(periods, gamma: float, n_samples: int = N_SAMPLES):
    """Score the toy with the twin's OWN estimator, not a re-derived one."""
    import importlib.util  # noqa: PLC0415
    import os  # noqa: PLC0415
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "eta_wave_twin.py")
    spec = importlib.util.spec_from_file_location("eta_wave_twin", path)
    ewt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ewt)
    periods = list(periods)
    lego = np.stack([run_mode(p, gamma, True, n_samples) for p in periods], 1)
    nemo = np.stack([run_mode(p, gamma, False, n_samples) for p in periods], 1)
    wet = np.ones((1, len(periods)), dtype=bool)
    return ewt.substep_lag_fit(lego[:, None, :], nemo[:, None, :], wet), ewt


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gamma", type=float, default=GAMMA_DINO)
    a = p.parse_args(argv)

    print("1. RUNNING-MEAN IDENTITY (gamma = 0): euler == 2-point mean of leapfrog")
    for period in TOY_PERIODS:
        if period < STABLE_MIN_PERIOD:
            continue
        print(f"   period {period:3d} steps: max residual = "
              f"{running_mean_residual(period):.2e}")
    print("   -> a 2-point running mean is a HALF-SAMPLE delay at every frequency\n")

    stable = [q for q in TOY_PERIODS if q >= STABLE_MIN_PERIOD]
    for gamma in (0.0, a.gamma):
        res, ewt = fit_alpha(stable, gamma)
        print(f"2. alpha from the SHIPPED substep_lag_fit, gamma = {gamma}")
        print(f"   broadband (periods >= {STABLE_MIN_PERIOD}): "
              f"alpha = {res['alpha_steps']:+.5f}  R2 = {res['r_squared']:.5f}")
        for period in stable:
            r1, _ = fit_alpha([period], gamma)
            peak = float(np.abs(run_mode(period, gamma, False)).max())
            print(f"     period {period:3d} -> alpha {r1['alpha_steps']:+.5f}"
                  f"   (max|nemo| over the run = {peak:.3e})")
        print()

    print("3. CONTROL: identical starts must give exactly zero")
    lego = np.stack([run_mode(q, a.gamma, False) for q in stable], 1)
    _, ewt = fit_alpha(stable, a.gamma)
    ctrl = ewt.substep_lag_fit(lego[:, None, :], lego[:, None, :],
                               np.ones((1, len(stable)), dtype=bool))
    print(f"   alpha = {ctrl['alpha_steps']:+.6f}  (r_squared {ctrl['r_squared']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
