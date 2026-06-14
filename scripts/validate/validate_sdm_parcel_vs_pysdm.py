"""Cross-validate the legoESM SDM activation parcel against PySDM.

Companion to ``pysdm_parcel_reference.py`` (run that first, in the PySDM venv).
Both codes integrate the SAME monodisperse-ammonium-sulfate parcel
(Arabas & Shima 2017-style: T0=283.15 K, p0=900 hPa, RH0=95%, w=1 m/s,
N=5e7/kg, r_dry=50 nm) with independent thermodynamics and condensation
solvers:

* PySDM: kappa-Köhler (κ=0.72), implicit per-droplet condensation, its own
  saturation/latent-heat formulae;
* legoESM: ideal van't Hoff Raoult (i=3 — the dilute-limit κ≈0.72 of ammonium
  sulfate), explicit adaptive RK4 growth, Tetens saturation, c_pd heating.

Exact agreement is NOT expected (different Köhler forms + thermo formulae);
the gates assert the activation physics agrees: peak supersaturation height
and magnitude, droplet growth, and liquid water within documented tolerances.

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python \
        scripts/validate/validate_sdm_parcel_vs_pysdm.py /tmp/pysdm_parcel_ref.npz
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.thermo import saturation_vapor_pressure
from legoesm.atmosphere.physics.microphysics.sdm import (
    ParcelState,
    SDMConfig,
    SuperDropletState,
    drsq_dt,
    run_parcel,
)

# --- must mirror pysdm_parcel_reference.py ---
N_SD = 64
T0, P0, RH0, W, DT, N_STEPS = 283.15, 9.0e4, 0.95, 1.0, 0.1, 6000
N_AEROSOL = 5.0e7          # per kg dry air
R_DRY = 5.0e-8
RHO_AS = 1770.0
ION_AS = 3.0               # van't Hoff i for (NH4)2SO4 (kappa_ideal ~ 0.72)
M_AS = 0.13214             # [kg/mol]


def _equilibrium_wet_radius(S0, T, m_s):
    """Haze equilibrium radius: root of drsq_dt(R; S0, T) via bisection."""
    e_s = float(saturation_vapor_pressure(jnp.asarray(T)))
    N_s = m_s * ION_AS / M_AS

    def rate(r):
        return float(drsq_dt(jnp.asarray(r) ** 2, S0, T, e_s, jnp.asarray(N_s),
                             include_curvature=True, include_solute=True))

    lo, hi = R_DRY, 5.0e-6
    assert rate(lo) > 0.0 and rate(hi) < 0.0, "equilibrium not bracketed"
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if rate(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main(ref_path):
    ref = np.load(ref_path)
    # Handshake: the reference must match this script's setup exactly.
    # The PySDM kappa must equal the ideal van't Hoff kappa implied by OUR
    # solute parameters: kappa = i·M_w·rho_s/(M_s·rho_w) — this IS the
    # cross-model solute-equivalence claim, so a reference generated with a
    # different kappa must fail here, not drift through the loose gates.
    kappa_expected = (ION_AS * (constants.M_H2O * 1e-3) * RHO_AS
                      / (M_AS * constants.rho_water))
    for name, val in (("T0", T0), ("p0", P0), ("rh0", RH0), ("w", W),
                      ("dt", DT), ("n_steps", N_STEPS),
                      ("n_aerosol", N_AEROSOL), ("r_dry", R_DRY),
                      ("rho_as", RHO_AS)):
        assert abs(float(ref[name]) - val) / max(abs(val), 1e-300) < 1e-12, \
            f"reference setup mismatch: {name}"
    assert abs(float(ref["kappa"]) - kappa_expected) / kappa_expected < 0.01, \
        (f"reference kappa {float(ref['kappa'])} != ideal van't Hoff "
         f"{kappa_expected:.4f} implied by i={ION_AS}, M_s={M_AS}")

    # initial vapour from OUR saturation curve (each side uses its own thermo,
    # consistently — that IS part of the formulae difference under test)
    e0 = RH0 * float(saturation_vapor_pressure(jnp.asarray(T0)))
    q_v0 = constants.epsilon * e0 / (P0 - e0)

    m_s = 4.0 / 3.0 * np.pi * RHO_AS * R_DRY**3
    r_wet0 = _equilibrium_wet_radius(RH0, T0, m_s)
    print(f"legoESM init: q_v0={q_v0:.5f}  haze r_eq={r_wet0*1e6:.3f} um "
          f"(PySDM: {float(ref['rbar'][0])*1e6:.3f} um)")

    xi = N_AEROSOL / N_SD          # per kg dry air; M_air = 1 kg parcel
    o = jnp.ones((N_SD,))
    droplets = SuperDropletState(
        multiplicity=o * xi, radius=o * r_wet0,
        solute_mass=o * m_s, active=o)
    parcel0 = ParcelState(droplets=droplets, T=jnp.asarray(T0),
                          p=jnp.asarray(P0), q_v=jnp.asarray(q_v0),
                          z=jnp.asarray(0.0))
    cfg = SDMConfig(include_curvature=True, include_solute=True,
                    solute_ionization=ION_AS, solute_molar_mass=M_AS,
                    condensation_integrator="rk4_adaptive")
    run = jax.jit(run_parcel, static_argnames=("n_steps", "cfg"))
    final, hist = run(parcel0, W, DT, N_STEPS, cfg, 1.0)

    S_l = np.asarray(hist["S"]); S_p = np.asarray(ref["S"])
    r_l = np.asarray(hist["mean_radius"]); r_p = np.asarray(ref["rbar"])
    ql_l = np.asarray(hist["q_l"]); ql_p = np.asarray(ref["lwc"])
    T_l = np.asarray(hist["T"]); T_p = np.asarray(ref["T"])

    ss_l, ss_p = S_l.max() - 1.0, S_p.max() - 1.0
    t_peak_l, t_peak_p = (S_l.argmax() + 1) * DT, (S_p.argmax() + 1) * DT
    print(f"{'':14}{'legoESM':>12}{'PySDM':>12}{'rel diff':>10}")
    rows = [
        ("S_max - 1", ss_l, ss_p),
        ("t(S_max) [s]", t_peak_l, t_peak_p),
        ("r_bar final", r_l[-1], r_p[-1]),
        ("LWC final", ql_l[-1], ql_p[-1]),
        ("T final", T_l[-1], T_p[-1]),
        ("S final", S_l[-1], S_p[-1]),
    ]
    ok = True
    for name, a, b in rows:
        rd = abs(a - b) / max(abs(b), 1e-300)
        print(f"{name:14}{a:12.5g}{b:12.5g}{rd*100:9.2f}%")
    # Gates (documented): activation physics must agree across the two
    # independent Köhler forms + thermo formulae.
    ok = ok and np.all(np.isfinite(S_l)) and np.all(np.isfinite(ql_l))
    ok = ok and abs(ss_l - ss_p) / ss_p < 0.30          # peak supersaturation
    ok = ok and abs(t_peak_l - t_peak_p) / t_peak_p < 0.30
    ok = ok and abs(r_l[-1] - r_p[-1]) / r_p[-1] < 0.10  # final droplet size
    ok = ok and abs(ql_l[-1] - ql_p[-1]) / ql_p[-1] < 0.10
    ok = ok and abs(T_l[-1] - T_p[-1]) < 1.0             # [K]
    # both must show the activation signature: peak then relaxation above 1
    ok = ok and (S_l.max() > 1.0) and (S_l.argmax() < len(S_l) - 1)
    ok = ok and (S_l[-1] > 1.0) and (S_l[-1] < S_l.max())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = (np.arange(N_STEPS) + 1) * DT
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        ax[0].plot(t, (S_l - 1) * 100, label="legoESM")
        ax[0].plot(t, (S_p - 1) * 100, "--", label="PySDM")
        ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("supersaturation [%]")
        ax[0].legend(); ax[0].set_title("S - 1")
        ax[1].plot(t, r_l * 1e6); ax[1].plot(t, r_p * 1e6, "--")
        ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("mean radius [um]")
        ax[1].set_yscale("log"); ax[1].set_title("droplet growth")
        ax[2].plot(t, ql_l * 1e3); ax[2].plot(t, ql_p * 1e3, "--")
        ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("LWC [g/kg]")
        ax[2].set_title("liquid water")
        fig.suptitle("Activation parcel: legoESM SDM vs PySDM (independent solvers)")
        fig.tight_layout()
        outdir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              "..", "..", "results"))
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "sdm_parcel_vs_pysdm.png")
        fig.savefig(path, dpi=110)
        print(f"saved {path}")
    except Exception:
        pass

    print(f"\nlegoESM vs PySDM parcel cross-validation: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/pysdm_parcel_ref.npz")
