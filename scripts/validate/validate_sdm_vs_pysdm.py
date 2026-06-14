"""Cross-validate legoESM SDM coalescence against PySDM (independent oracle).

PySDM (https://open-atmos.github.io/PySDM/) is the reference open-source
implementation of Shima's Super-Droplet Method. Both codes run the SAME
canonical Shima-2009 Golovin box (exponential spectrum, N0 = 2^23 m^-3,
mean volume 1.19e5 µm³, b = 1500/s, dt = 1 s) with independent Monte-Carlo
machinery; their ensemble statistics must agree with each other and with the
exact Golovin/Scott analytic — a far stronger consistency check than analytic
moments alone.

Two-process design (PySDM's numba stack must not enter the repo's jax venv):

1. ``/tmp/pysdm_venv/bin/python scripts/validate/pysdm_golovin_reference.py ref.npz``
2. ``JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python \
       scripts/validate/validate_sdm_vs_pysdm.py ref.npz``

Compares, at t = 0/1200/2400/3600 s: total number density N(t), LWC (must be
conserved by both), the 2nd mass moment M2, and the mass-density spectra
g(ln r). Exits nonzero on disagreement beyond Monte-Carlo tolerance.
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.sdm import (
    BoxState,
    SDMConfig,
    exponential_water_droplets,
    run_box,
)

_FOUR_THIRDS_PI = 4.0 / 3.0 * np.pi
_RHO_W = constants.rho_water

# Canonical Shima-2009 box (must match pysdm_golovin_reference.py)
N0_PER_M3 = 2.0 ** 23
XBAR_M3 = 1.19e5 * 1.0e-18         # mean droplet volume [m^3]
B_GOLOVIN = 1.5e3
DT = 1.0
SNAPS = (0, 1200, 2400, 3600)
N_SD = 2 ** 15
V_CELL = 1.0                        # per-m^3 box on our side (scale-invariant)
ENSEMBLE = 8                        # average our MC over a few seeds


def run_legoesm_ensemble():
    cfg = SDMConfig(collision_kernel="golovin", golovin_b=B_GOLOVIN)
    x0 = _RHO_W * XBAR_M3           # mean droplet MASS [kg]

    results = {t: [] for t in SNAPS}
    for seed in range(ENSEMBLE):
        k_init, k_run = random.split(random.PRNGKey(seed))
        droplets = exponential_water_droplets(k_init, N_SD, N0_PER_M3 * V_CELL, x0)
        box = BoxState(droplets=droplets, T=jnp.asarray(283.0),
                       p=jnp.asarray(9.0e4), q_v=jnp.asarray(0.0), key=k_run)
        results[0].append((np.asarray(droplets.radius),
                           np.asarray(droplets.multiplicity)))
        t_prev = 0
        for t in SNAPS[1:]:
            box, _ = run_box(box, V_CELL, DT, int(round((t - t_prev) / DT)),
                             cfg, do_condensation=False, do_coalescence=True)
            t_prev = t
            d = box.droplets
            results[t].append((np.asarray(d.radius),
                               np.asarray(d.active * d.multiplicity)))
    return results


def moments(radius, mult, V):
    m = mult * _FOUR_THIRDS_PI * _RHO_W * radius**3
    N = mult.sum() / V
    LWC = m.sum() / V
    M2 = (mult * (_FOUR_THIRDS_PI * _RHO_W * radius**3) ** 2).sum() / V
    return N, LWC, M2


def spectrum(radius, mult, V, lnr_edges):
    w = mult * _FOUR_THIRDS_PI * _RHO_W * radius**3
    valid = mult > 0
    hist, _ = np.histogram(np.log(radius[valid]), bins=lnr_edges,
                           weights=w[valid])
    return hist / (V * np.diff(lnr_edges))


def main(ref_path):
    ref = np.load(ref_path)
    dv = float(ref["dv"])
    # Handshake: the reference must have been generated with EXACTLY this
    # setup — a stale/mismatched .npz must fail loudly, not validate silently.
    assert abs(float(ref["n0_per_m3"]) - N0_PER_M3) / N0_PER_M3 < 1e-12, "N0 mismatch"
    assert abs(float(ref["xbar_m3"]) - XBAR_M3) / XBAR_M3 < 1e-12, "Xbar mismatch"
    assert abs(float(ref["b_golovin"]) - B_GOLOVIN) / B_GOLOVIN < 1e-12, "b mismatch"
    assert tuple(int(t) for t in ref["snap_times"]) == SNAPS, "snapshot times mismatch"
    for t in SNAPS:
        assert ref[f"radius_{t}"].shape == ref[f"multiplicity_{t}"].shape, \
            f"ragged reference arrays at t={t}"
        assert np.all(np.isfinite(ref[f"radius_{t}"])), f"non-finite radii at t={t}"
    ours = run_legoesm_ensemble()

    lnr_edges = np.linspace(np.log(5e-6), np.log(5e-3), 50)
    r_c = np.exp(0.5 * (lnr_edges[:-1] + lnr_edges[1:]))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        have_plt = True
    except Exception:
        have_plt = False

    ok = True
    lwc0 = None
    print(f"{'t[s]':>6} {'N_lego':>11} {'N_pysdm':>11} {'dN%':>6} "
          f"{'M2_lego':>11} {'M2_pysdm':>11} {'dM2%':>7} {'spec_err%':>9}")
    for t in SNAPS:
        Ns, LWCs, M2s, specs = [], [], [], []
        for radius, mult in ours[t]:
            N, LWC, M2 = moments(radius, mult, V_CELL)
            Ns.append(N); LWCs.append(LWC); M2s.append(M2)
            specs.append(spectrum(radius, mult, V_CELL, lnr_edges))
        N_l, LWC_l, M2_l = np.mean(Ns), np.mean(LWCs), np.mean(M2s)
        g_l = np.mean(specs, axis=0)

        r_p = ref[f"radius_{t}"]
        xi_p = ref[f"multiplicity_{t}"]
        N_p, LWC_p, M2_p = moments(r_p, xi_p, dv)
        g_p = spectrum(r_p, xi_p, dv, lnr_edges)

        dN = abs(N_l - N_p) / N_p
        dM2 = abs(M2_l - M2_p) / M2_p
        # spectrum L1 relative error over bins carrying >1% of the mass peak
        sig = (g_p > 0.01 * g_p.max()) | (g_l > 0.01 * g_l.max())
        spec_err = (np.abs(g_l - g_p)[sig].sum()
                    / np.maximum(g_p[sig].sum(), 1e-300))
        print(f"{t:6d} {N_l:11.4e} {N_p:11.4e} {dN*100:6.2f} "
              f"{M2_l:11.4e} {M2_p:11.4e} {dM2*100:7.2f} {spec_err*100:9.2f}")

        if lwc0 is None:
            lwc0 = (LWC_l, LWC_p)
        ok = ok and np.isfinite(N_l) and np.isfinite(M2_l)
        ok = ok and abs(LWC_l - lwc0[0]) / lwc0[0] < 1e-9   # ours conserves
        ok = ok and abs(LWC_p - lwc0[1]) / lwc0[1] < 1e-6   # PySDM conserves
        # initial conditions must match closely; evolved moments within MC noise
        tol_N = 0.02 if t == 0 else 0.10
        tol_M2 = 0.05 if t == 0 else 0.35   # M2 is giant-drop-tail dominated
        ok = ok and (dN < tol_N) and (dM2 < tol_M2)
        # spectrum gate: the distributions themselves must agree, not just low
        # moments (t=0 tight — same IC; evolved within MC tail noise).
        tol_spec = 0.05 if t == 0 else 0.30
        ok = ok and np.isfinite(spec_err) and sig.any() and (spec_err < tol_spec)

        if have_plt:
            line = ax.plot(r_c * 1e6, g_l * 1e3, "o", ms=3,
                           label=f"legoESM t={t}s")[0]
            ax.plot(r_c * 1e6, g_p * 1e3, "-", color=line.get_color(),
                    lw=1.4, alpha=0.85)

    # cross-check LWC scale: both should be ~1 g/m^3
    print(f"LWC: legoESM={lwc0[0]*1e3:.4f} g/m^3, PySDM={lwc0[1]*1e3:.4f} g/m^3")
    ok = ok and abs(lwc0[0] - lwc0[1]) / lwc0[1] < 0.05

    if have_plt:
        ax.set_xscale("log")
        ax.set_xlabel("radius [um]")
        ax.set_ylabel("g(ln r) [g/m^3 per ln r]")
        ax.set_title("Shima-2009 Golovin box: legoESM SDM (points) vs PySDM (lines)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        outdir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              "..", "..", "results"))
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "sdm_vs_pysdm_golovin.png")
        fig.savefig(path, dpi=110)
        print(f"saved {path}")

    print(f"\nlegoESM vs PySDM cross-validation: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/pysdm_golovin_ref.npz")
