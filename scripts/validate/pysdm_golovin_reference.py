"""Generate a PySDM reference solution for the Shima-2009 Golovin box.

Run with a PySDM-equipped python (NOT the repo venv — PySDM/numba pins clash
with the repo's jax stack)::

    /tmp/pysdm_venv/bin/python scripts/validate/pysdm_golovin_reference.py \
        /tmp/pysdm_golovin_ref.npz

Canonical Shima et al. (2009) §5.1.4 box-coalescence setup (also the PySDM
README example): exponential volume spectrum with ``N0 = 2^23 m^-3``, mean
volume ``X̄ = 1.19e5 µm³`` (mean-mass radius ≈ 30.5 µm, LWC ≈ 1 g/m³),
Golovin additive kernel ``b = 1500 s^-1``, ``dt = 1 s``. Dumps the
super-droplet (multiplicity, radius) snapshots at t = 0/1200/2400/3600 s to an
``.npz`` the legoESM cross-validator loads — PySDM is the independent
reference implementation of the same Monte-Carlo algorithm (open-atmos,
Shima lineage).
"""

import sys

import numpy as np

from PySDM import Builder
from PySDM.backends import CPU
from PySDM.dynamics import Coalescence
from PySDM.dynamics.collisions.collision_kernels import Golovin
from PySDM.environments import Box
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.initialisation.spectra import Exponential
from PySDM.physics import si

N_SD = 2 ** 15
DV = 1.0e6 * si.m ** 3                    # box volume (PySDM README value)
N0_PER_M3 = 2.0 ** 23                     # Shima 2009 droplet number density
XBAR = 1.19e5 * si.um ** 3                # mean droplet volume
B_GOLOVIN = 1.5e3 / si.s
DT = 1.0 * si.s
SNAP_TIMES_S = (0, 1200, 2400, 3600)


def main(out_path):
    spectrum = Exponential(norm_factor=N0_PER_M3 * DV, scale=XBAR)
    builder = Builder(n_sd=N_SD, backend=CPU(), environment=Box(dt=DT, dv=DV))
    # adaptive=False: force the plain fixed-dt Shima update so the reference
    # uses EXACTLY the same stepping as the legoESM side (PySDM defaults to
    # adaptive coalescence sub-stepping, which would compare different
    # algorithm variants).
    builder.add_dynamic(Coalescence(collision_kernel=Golovin(b=B_GOLOVIN),
                                    adaptive=False))
    attributes = {}
    sampled = ConstantMultiplicity(spectrum).sample(N_SD)
    attributes["volume"], attributes["multiplicity"] = sampled
    particulator = builder.build(attributes, products=())

    out = {}
    t_prev = 0
    for t in SNAP_TIMES_S:
        if t > t_prev:
            particulator.run(steps=int(round((t - t_prev) / DT)))
            t_prev = t
        vol = particulator.attributes["volume"].to_ndarray().copy()
        mult = particulator.attributes["multiplicity"].to_ndarray().astype(float).copy()
        radius = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)
        out[f"radius_{t}"] = radius
        out[f"multiplicity_{t}"] = mult
        n_tot = mult.sum() / float(DV)
        lwc = (mult * vol).sum() * 1000.0 / float(DV)  # rho_w = 1000 kg/m^3
        print(f"t={t:5d}s  N={n_tot:.4e}/m^3  LWC={lwc*1e3:.4f} g/m^3  "
              f"n_active_sd={(mult > 0).sum()}")

    out["dv"] = np.asarray(float(DV))
    out["n0_per_m3"] = np.asarray(N0_PER_M3)
    out["xbar_m3"] = np.asarray(float(XBAR))
    out["b_golovin"] = np.asarray(float(B_GOLOVIN))
    out["snap_times"] = np.asarray(SNAP_TIMES_S, dtype=float)
    np.savez(out_path, **out)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/pysdm_golovin_ref.npz")
