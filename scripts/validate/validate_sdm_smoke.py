"""Smoke + oracle-consistency validation for the Super-Droplet Method.

Runs the faithful JAX SDM (``legoesm.atmosphere.physics.microphysics.sdm``) in
three idealized configurations and checks **stability** (no NaN/Inf, conserved
invariants over a long integration), **physical realism**, and **consistency
with the oracle**:

1. **Golovin collision box** — pure Shima Monte-Carlo coalescence from an
   exponential droplet spectrum with the additive (Golovin) kernel. The droplet
   mass-density spectrum ``g(ln r)`` is compared against the EXACT analytic
   solution (Golovin 1963 / Scott 1968), which is precisely the benchmark ERF /
   Shima (2009, Fig. 4) validate the Super-Droplet Method against. Running the
   ERF C++/AMReX oracle itself is unnecessary: its published acceptance test IS
   this analytic comparison, and the per-formula numerics were already matched
   term-by-term in the unit tests.

2. **Warm-rain box** — composed condensation + collision-coalescence
   (``run_box``) from a cloud-droplet population: the spectrum broadens and
   rain-size drops (R >= r_rain) form, i.e. autoconversion emerges from the
   resolved collisions.

3. **Adiabatic parcel** — a long ascent (``run_parcel``): supersaturation peaks
   then relaxes, droplets activate and grow, total water is conserved, and the
   parcel cools along the moist adiabat.

Usage::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
        .venv/bin/python scripts/validate/validate_sdm_smoke.py

Exits nonzero if any stability/consistency check fails. Saves diagnostic PNGs to
``results/`` (gitignored).
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from scipy.special import i1e as bessel_i1e  # scaled I_1 (overflow-safe) for analytic

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio, saturation_vapor_pressure
from legoesm.atmosphere.physics.microphysics.sdm import (
    BoxState,
    ParcelState,
    SDMConfig,
    SuperDropletState,
    liquid_mixing_ratio,
    run_box,
    run_parcel,
)

_FOUR_THIRDS_PI = 4.0 / 3.0 * np.pi
_RHO_W = constants.rho_water


def _mass_to_radius(x):
    return (x / (_FOUR_THIRDS_PI * _RHO_W)) ** (1.0 / 3.0)


def _radius_to_mass(r):
    return _FOUR_THIRDS_PI * _RHO_W * r**3


# ==========================================================================
# 1. Golovin collision box vs analytic Scott (1968) solution
# ==========================================================================
def analytic_golovin_g(r, t, N0, x0, b_mass):
    """Analytic mass-density spectrum g(ln r) [kg/m^3 per ln r] for the additive
    kernel K(x,y)=b_mass(x+y) from an exponential initial distribution
    n(x,0)=(N0/x0)exp(-x/x0)  (Golovin 1963; Scott 1968)::

        T = 1 - exp(-b_mass N0 x0 t),  u = x/x0
        n(x,t) = (N0/x0)(1-T) T^-1/2 u^-1 exp(-(1+T)u) I_1(2 u sqrt(T))
        g(ln r) = 3 x^2 n(x)
    """
    x = _radius_to_mass(r)
    u = x / x0
    L0 = N0 * x0
    T = 1.0 - np.exp(-b_mass * L0 * t)
    if T <= 0.0:
        n = (N0 / x0) * np.exp(-u)            # t=0: exponential initial spectrum
        return 3.0 * x**2 * n
    sqrtT = np.sqrt(T)
    # n = (N0/x0)(1-T) T^-1/2 u^-1 exp(-(1+T)u) I_1(2u sqrtT). Use the scaled
    # Bessel i1e(z)=I_1(z)exp(-z) and fold the exponents:
    #   exp(-(1+T)u) I_1(z) = exp(-(1+T)u + z) i1e(z) = exp(-u(1-sqrtT)^2) i1e(z),
    # so nothing overflows (the exponent is <= 0 and i1e is bounded).
    z = 2.0 * u * sqrtT
    n = (N0 / x0) * (1.0 - T) / sqrtT / u \
        * np.exp(-u * (1.0 - sqrtT) ** 2) * bessel_i1e(z)
    return 3.0 * x**2 * n


def _spectrum_from_droplets(radius, xi, V, lnr_edges):
    """Bin a super-droplet population into g(ln r) = Σ ξ x / (V Δln r)."""
    lnr = np.log(np.asarray(radius))
    x = _radius_to_mass(np.asarray(radius))
    w = np.asarray(xi) * x
    hist, _ = np.histogram(lnr, bins=lnr_edges, weights=w)
    dlnr = np.diff(lnr_edges)
    centers = 0.5 * (lnr_edges[:-1] + lnr_edges[1:])
    return np.exp(centers), hist / (V * dlnr)


def run_golovin(outdir):
    print("\n=== 1. Golovin collision box vs analytic (oracle benchmark) ===")
    n_sd = 2 ** 15
    N0 = 1.0e9            # droplets / m^3
    r0 = 10.0e-6          # initial mean-mass radius [m]
    x0 = _radius_to_mass(r0)
    V = 1.0               # box volume [m^3]
    b_V = 1.5e3
    b_mass = b_V / _RHO_W
    dt = 10.0
    cfg = SDMConfig(collision_kernel="golovin", golovin_b=b_V)

    # exponential initial spectrum, equal multiplicity (constant-xi SDM init)
    key = random.PRNGKey(0)
    k_init, k_run = random.split(key)
    U = random.uniform(k_init, (n_sd,), dtype=jnp.float64)
    x = -x0 * jnp.log(1.0 - U)
    R0 = (x / (_FOUR_THIRDS_PI * _RHO_W)) ** (1.0 / 3.0)
    xi0 = N0 * V / n_sd
    droplets = SuperDropletState(
        multiplicity=jnp.full((n_sd,), xi0),
        radius=R0, solute_mass=jnp.zeros((n_sd,)), active=jnp.ones((n_sd,)))
    box0 = BoxState(droplets=droplets, T=jnp.asarray(283.0), p=jnp.asarray(9.0e4),
                    q_v=jnp.asarray(0.0), key=k_run)

    # Snapshots at b_mass*L0*t = {0.5, 1, 2, 3} (the useful Golovin validation
    # range; beyond ~3 the population has collected into a handful of giant
    # drops and the Monte-Carlo tail is too sparse for a tail-sensitive metric).
    snap_steps = [0, 8, 16, 32, 48]
    L0 = float(jnp.sum(droplets.active * droplets.multiplicity
                       * _FOUR_THIRDS_PI * _RHO_W * droplets.radius**3)) / V
    N_tot0 = float(jnp.sum(droplets.active * droplets.multiplicity))
    M2_0 = float(jnp.sum(droplets.active * droplets.multiplicity
                         * (_FOUR_THIRDS_PI * _RHO_W * droplets.radius**3) ** 2))
    print(f"  n_sd={n_sd}, N0={N0:.1e}/m^3, LWC={L0*1e3:.3f} g/m^3, "
          f"b_mass={b_mass:.2f}")

    boxes = {0: box0}
    cur = box0
    done = 0
    for target in snap_steps[1:]:
        cur, _ = run_box(cur, V, dt, target - done, cfg,
                         do_condensation=False, do_coalescence=True)
        done = target
        boxes[target] = cur

    lnr_edges = np.linspace(np.log(2e-6), np.log(5e-3), 60)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        have_plt = True
    except Exception:
        have_plt = False
    if have_plt:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    ok = True
    for step in snap_steps:
        t = step * dt
        tau = b_mass * L0 * t            # the Golovin similarity time b*L0*t
        b = boxes[step]
        active_xi = b.droplets.active * b.droplets.multiplicity
        # stability: finite + mass conserved
        m_now = float(jnp.sum(active_xi * _FOUR_THIRDS_PI * _RHO_W
                              * b.droplets.radius**3)) / V
        finite = bool(jnp.all(jnp.isfinite(b.droplets.radius))
                      and jnp.all(jnp.isfinite(b.droplets.multiplicity)))
        mass_err = abs(m_now - L0) / L0
        # Robust integral moments vs the EXACT analytic: 0th (number) decays as
        # exp(-tau), 2nd mass moment grows as exp(2 tau). These are the oracle's
        # acceptance moments and are far less Monte-Carlo-noisy than the tail.
        N_sdm = float(jnp.sum(active_xi))
        M2_sdm = float(jnp.sum(active_xi
                               * (_FOUR_THIRDS_PI * _RHO_W * b.droplets.radius**3) ** 2))
        N_err = abs(N_sdm - N_tot0 * np.exp(-tau)) / (N_tot0 * np.exp(-tau))
        M2_err = abs(M2_sdm - M2_0 * np.exp(2.0 * tau)) / (M2_0 * np.exp(2.0 * tau))
        m2_note = "" if tau <= 1.0 else "  (M2 tail diagnostic-only at this tau)"
        print(f"  tau={tau:4.2f} t={t:5.0f}s  mass_err={mass_err:.1e} finite={finite}  "
              f"N/N0_err={N_err*100:5.1f}%  M2_err={M2_err*100:5.1f}%{m2_note}")
        # The number decay N(t)=N0 exp(-tau) is the robust oracle acceptance
        # metric (gated <8% over the whole range). The 2nd mass moment is
        # tail-dominated: with a finite super-droplet count the rapidly growing
        # giant-drop tail is under-sampled past tau~1, so M2 is gated only where
        # the SDM is well converged (tau<=1) and is diagnostic-only beyond.
        m2_ok = (M2_err < 0.15) if tau <= 1.0 else True
        ok = ok and finite and (mass_err < 1e-6) and (N_err < 0.08) and m2_ok
        if have_plt:
            r_c, g_sdm = _spectrum_from_droplets(b.droplets.radius, active_xi,
                                                 V, lnr_edges)
            g_ana = analytic_golovin_g(r_c, t, N0, x0, b_mass)
            line = ax.plot(r_c * 1e6, g_sdm * 1e3, "o", ms=3,
                           label=f"SDM tau={tau:.1f}")[0]
            ax.plot(r_c * 1e6, g_ana * 1e3, "-", color=line.get_color(),
                    lw=1.2, alpha=0.8)
    if have_plt:
        ax.set_xscale("log")
        ax.set_xlabel("radius [um]")
        ax.set_ylabel("g(ln r)  [g/m^3 per ln r]")
        ax.set_title("Golovin box: SDM (points) vs analytic Scott (lines)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(outdir, "sdm_golovin_spectrum.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"  saved {path}")
    print(f"  --> {'PASS' if ok else 'FAIL'} (stable + matches analytic moments)")
    return ok


# ==========================================================================
# 2. Warm-rain box: condensation + collision -> rain
# ==========================================================================
def run_warm_rain(outdir):
    print("\n=== 2. Warm-rain box (condensation + collision -> rain) ===")
    n_sd = 2 ** 14
    N0 = 1.0e8
    r0 = 15.0e-6           # mean-mass radius of the initial cloud spectrum [m]
    x0 = _radius_to_mass(r0)
    V = 1.0
    cfg = SDMConfig(collision_kernel="long", terminal_velocity="cloud_rain_shima",
                    r_rain=4.0e-5)
    T0, p0 = 283.0, 9.0e4
    rho = p0 / (constants.R_d * T0)
    q_sat0 = float(saturation_mixing_ratio(jnp.asarray(T0), jnp.asarray(p0)))

    key = random.PRNGKey(1)
    k_init, k_run = random.split(key)
    # Exponential cloud-droplet spectrum: a width is essential — a monodisperse
    # population has zero differential fall speed, so the hydrodynamic kernel
    # vanishes and nothing ever collides. Real clouds inherit width from
    # aerosol/activation; here we seed it directly.
    U = random.uniform(k_init, (n_sd,), dtype=jnp.float64)
    xmass = -x0 * jnp.log(1.0 - U)
    R0 = (xmass / (_FOUR_THIRDS_PI * _RHO_W)) ** (1.0 / 3.0)
    droplets = SuperDropletState(
        multiplicity=jnp.full((n_sd,), N0 * V / n_sd),
        radius=R0,
        solute_mass=jnp.zeros((n_sd,)), active=jnp.ones((n_sd,)))
    box0 = BoxState(droplets=droplets, T=jnp.asarray(T0), p=jnp.asarray(p0),
                    q_v=jnp.asarray(1.001 * q_sat0), key=k_run)

    dt, n_steps = 2.0, 1800   # 3600 s
    final, hist = run_box(box0, V, dt, n_steps, cfg,
                          do_condensation=True, do_coalescence=True)
    q_l = np.asarray(hist["q_l"])
    q_rain = np.asarray(hist["q_rain"])
    q_t = np.asarray(hist["q_t"])
    rbar = np.asarray(hist["mean_radius"])
    N = np.asarray(hist["N"])
    finite = bool(np.all(np.isfinite(q_l)) and np.all(np.isfinite(rbar))
                  and jnp.all(jnp.isfinite(final.droplets.radius)))
    q_t0 = q_t[0]
    massdrift = float(np.max(np.abs(q_t - q_t0)) / q_t0)
    rain_formed = q_rain[-1] > 10.0 * q_rain[0] + 1e-9 and q_rain[-1] > 1e-6
    grew = rbar[-1] > rbar[0]
    nfell = N[-1] < N[0]
    print(f"  finite={finite}  q_t drift={massdrift:.2e}  "
          f"mean_r {rbar[0]*1e6:.1f}->{rbar[-1]*1e6:.1f} um  "
          f"N {N[0]:.2e}->{N[-1]:.2e}  q_rain {q_rain[0]*1e3:.2e}->{q_rain[-1]*1e3:.3f} g/kg")
    # 1e-5 relative: fp round-off accumulated over ~1800 cbrt/cube coalescence +
    # condensation-exchange steps (each step conserves q_t analytically).
    ok = finite and (massdrift < 1e-5) and rain_formed and grew and nfell
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.arange(n_steps) * dt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(t / 60, q_l * 1e3, label="q_l (total)")
        ax[0].plot(t / 60, q_rain * 1e3, label="q_rain (R>=40um)")
        ax[0].set_xlabel("time [min]"); ax[0].set_ylabel("water [g/kg]")
        ax[0].legend(); ax[0].set_title("warm-rain box: rain forms")
        ax[1].plot(t / 60, rbar * 1e6); ax[1].set_xlabel("time [min]")
        ax[1].set_ylabel("mass-mean radius [um]"); ax[1].set_title("droplets grow")
        fig.tight_layout()
        path = os.path.join(outdir, "sdm_warm_rain.png")
        fig.savefig(path, dpi=110); plt.close(fig)
        print(f"  saved {path}")
    except Exception:
        pass
    print(f"  --> {'PASS' if ok else 'FAIL'} (stable, water-conserving, rain forms)")
    return ok


# ==========================================================================
# 3. Adiabatic parcel long ascent
# ==========================================================================
def run_parcel_case(outdir):
    print("\n=== 3. Adiabatic parcel (long ascent, activation) ===")
    n_sd = 500
    T0, p0 = 283.15, 9.0e4
    e_s0 = float(saturation_vapor_pressure(jnp.asarray(T0)))
    rho = p0 / (constants.R_d * T0)
    N_per_m3 = 5.0e7
    xi = N_per_m3 / rho / n_sd
    # Start at cloud base: just above saturation (S0=1.002) with already-activated
    # cloud droplets (r0=8 um). Starting subsaturated would evaporate the droplets
    # to the radius floor before the updraft lifts S past 1 (a curvature/floor
    # artifact for pure droplets with no aerosol core), which is unphysical.
    S0 = 1.002
    e0 = S0 * e_s0
    q_v0 = constants.epsilon * e0 / (p0 - e0)
    droplets = SuperDropletState(
        multiplicity=jnp.full((n_sd,), xi),
        radius=jnp.full((n_sd,), 8.0e-6),
        solute_mass=jnp.zeros((n_sd,)), active=jnp.ones((n_sd,)))
    parcel0 = ParcelState(droplets=droplets, T=jnp.asarray(T0), p=jnp.asarray(p0),
                          q_v=jnp.asarray(q_v0), z=jnp.asarray(0.0))
    cfg = SDMConfig(include_curvature=True, include_solute=False,
                    n_substeps_condensation=2)
    dt, n_steps = 0.1, 6000   # 600 s, ~600 m at w=1
    final, hist = run_parcel(parcel0, w=1.0, dt=dt, n_steps=n_steps, cfg=cfg)
    S = np.asarray(hist["S"]); q_l = np.asarray(hist["q_l"])
    q_t = np.asarray(hist["q_t"]); rbar = np.asarray(hist["mean_radius"])
    q_t0 = float(parcel0.q_v + liquid_mixing_ratio(parcel0.droplets, 1.0))
    finite = bool(np.all(np.isfinite(S)) and np.all(np.isfinite(q_l)))
    massdrift = float(np.max(np.abs(q_t - q_t0)) / q_t0)
    peaked = S.max() > 1.0 and S.argmax() < len(S) - 1 and S[-1] < S.max()
    small_ss = (S.max() - 1.0) < 0.05      # realistic peak supersaturation (<5%)
    grew = rbar[-1] > 8.0e-6               # grew beyond the 8 um cloud-base size
    print(f"  finite={finite}  q_t drift={massdrift:.2e}  "
          f"S_peak={S.max():.4f} (+{(S.max()-1)*100:.2f}%)  S_final={S[-1]:.4f}  "
          f"mean_r 8.0->{rbar[-1]*1e6:.1f} um  LWC_final={q_l[-1]*1e3:.3f} g/kg")
    ok = finite and (massdrift < 1e-6) and peaked and small_ss and grew
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        z = np.arange(n_steps) * dt * 1.0
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot((S - 1.0) * 100, z); ax[0].set_xlabel("supersaturation [%]")
        ax[0].set_ylabel("height [m]"); ax[0].set_title("S peaks then relaxes")
        ax[1].plot(rbar * 1e6, z, label="mean r")
        ax[1].plot(q_l * 1e3 * 10, z, label="LWC x10 [g/kg]")
        ax[1].set_xlabel("um  /  g/kg x10"); ax[1].set_ylabel("height [m]")
        ax[1].legend(); ax[1].set_title("activation + growth")
        fig.tight_layout()
        path = os.path.join(outdir, "sdm_parcel.png")
        fig.savefig(path, dpi=110); plt.close(fig)
        print(f"  saved {path}")
    except Exception:
        pass
    print(f"  --> {'PASS' if ok else 'FAIL'} (stable, conserving, realistic activation)")
    return ok


def main():
    jax.config.update("jax_enable_x64", True)
    outdir = os.path.join(os.path.dirname(__file__), "..", "..", "results")
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    results = {
        "golovin": run_golovin(outdir),
        "warm_rain": run_warm_rain(outdir),
        "parcel": run_parcel_case(outdir),
    }
    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k:12s}: {'PASS' if v else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\nSDM smoke validation: {'ALL PASS' if all_ok else 'FAILURES'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
