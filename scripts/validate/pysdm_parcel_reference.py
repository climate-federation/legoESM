"""Generate a PySDM reference for a monodisperse-aerosol activation parcel.

Run with the isolated PySDM venv (NOT the repo venv)::

    /tmp/pysdm_venv/bin/python scripts/validate/pysdm_parcel_reference.py \
        /tmp/pysdm_parcel_ref.npz

Arabas & Shima (2017)-style setup: a parcel rising at constant ``w`` from a
subsaturated state, carrying a single monodisperse ammonium-sulfate aerosol
mode that activates into cloud droplets. PySDM solves kappa-Köhler
condensation with its own implicit solver — an independent reference for the
legoESM SDM parcel (van't Hoff Raoult + explicit growth). Dumps per-step
S, T, p, LWC, and the mean wet radius.
"""

import sys

import numpy as np

from PySDM import Builder, Formulae
from PySDM.backends import CPU
from PySDM.dynamics import AmbientThermodynamics, Condensation
from PySDM.environments import Parcel
from PySDM.initialisation import equilibrate_wet_radii
from PySDM.physics import si

# --- shared setup (mirrored by the legoESM validator) ---
N_SD = 64                      # monodisperse: a handful of identical SDs
T0 = 283.15 * si.K
P0 = 9.0e4 * si.Pa
RH0 = 0.95
W = 1.0 * si.m / si.s
DT = 0.1 * si.s
N_STEPS = 6000                  # 600 s, ~600 m
N_AEROSOL = 5.0e7 / si.kg       # aerosol number per kg of dry air
R_DRY = 5.0e-8 * si.m           # 50 nm dry radius
KAPPA = 0.72                    # ideal van't Hoff kappa for (NH4)2SO4 (i=3)
RHO_AS = 1770.0                 # ammonium sulfate density [kg/m^3]


def main(out_path):
    formulae = Formulae()
    const = formulae.constants
    pv0 = RH0 * formulae.saturation_vapour_pressure.pvs_water(T0)
    q0 = const.eps * pv0 / (P0 - pv0)   # initial vapour mixing ratio

    env = Parcel(
        dt=DT, mass_of_dry_air=1.0 * si.kg, p0=P0, T0=T0, w=W,
        initial_water_vapour_mixing_ratio=q0,
    )
    builder = Builder(n_sd=N_SD, backend=CPU(formulae=formulae), environment=env)
    builder.add_dynamic(AmbientThermodynamics())
    builder.add_dynamic(Condensation())

    r_dry = np.full(N_SD, R_DRY)
    kappa_times_dry_volume = KAPPA * (4.0 / 3.0) * np.pi * r_dry**3
    multiplicity = np.full(N_SD, N_AEROSOL / N_SD)  # per kg dry air
    r_wet = equilibrate_wet_radii(
        r_dry=r_dry, environment=builder.particulator.environment,
        kappa_times_dry_volume=kappa_times_dry_volume,
    )
    attributes = {
        "multiplicity": multiplicity,
        "dry volume": (4.0 / 3.0) * np.pi * r_dry**3,
        "kappa times dry volume": kappa_times_dry_volume,
        "volume": (4.0 / 3.0) * np.pi * r_wet**3,
    }
    particulator = builder.build(attributes, products=())

    n_out = N_STEPS
    S = np.empty(n_out); T = np.empty(n_out); P = np.empty(n_out)
    LWC = np.empty(n_out); RBAR = np.empty(n_out); Z = np.empty(n_out)
    rho_w = const.rho_w
    for i in range(n_out):
        particulator.run(steps=1)
        env_ = particulator.environment
        S[i] = env_["RH"][0]
        T[i] = env_["T"][0]
        P[i] = env_["p"][0]
        Z[i] = env_["z"][0]
        vol = particulator.attributes["volume"].to_ndarray()
        mult = particulator.attributes["multiplicity"].to_ndarray().astype(float)
        # parcel has 1 kg dry air -> q_l = sum(mult * vol * rho_w) [kg/kg]
        LWC[i] = (mult * vol).sum() * rho_w
        r = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)
        RBAR[i] = (mult * r).sum() / mult.sum()

    print(f"S: start={S[0]:.4f} peak={S.max():.4f} (+{(S.max()-1)*100:.3f}%) "
          f"@step {S.argmax()} final={S[-1]:.4f}")
    print(f"r_bar: {RBAR[0]*1e6:.3f} -> {RBAR[-1]*1e6:.2f} um   "
          f"LWC final={LWC[-1]*1e3:.3f} g/kg   T {T[0]:.2f}->{T[-1]:.2f} K")

    np.savez(out_path, S=S, T=T, p=P, z=Z, lwc=LWC, rbar=RBAR,
             dt=DT, w=W, T0=T0, p0=P0, rh0=RH0, n_aerosol=N_AEROSOL,
             r_dry=R_DRY, kappa=KAPPA, rho_as=RHO_AS, n_steps=N_STEPS)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/pysdm_parcel_ref.npz")
