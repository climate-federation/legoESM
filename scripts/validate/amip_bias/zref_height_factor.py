#!/usr/bin/env python3
"""How much the surface fluxes change when the solver is told the true height.

The MPAS surface path hands the MOST solver the lowest full level's wind,
temperature and humidity.  Whether it also tells the solver how HIGH that
level is depends on ``SurfaceLayerConfig.z_ref_model_level`` (True by default
since 2026-09-21) -- with it False the solver divides by ``z_ref``, 10 m, while
the values came from roughly 150 m.  A log profile does not care much about the
roughness length but it cares a great deal about that ratio.

This calls the SAME routine the model calls, twice, on identical soundings:
once labelled 10 m and once labelled the true height.  It is an instrument
test of the code path, not a statement about any run's state -- the soundings
span the regimes rather than reproducing one.

Usage: zref_height_factor.py [z_level_m ...]
"""
from __future__ import annotations

import sys

import numpy as np

from legoesm import constants
from legoesm.core.bulk_flux import compute_most_fluxes
from legoesm.thermo import saturation_specific_humidity

# (label, wind [m/s], SST [K], air-sea temperature difference [K], RH of the air)
REGIMES = (
    ("trades",        8.0, 300.0, -1.5, 0.78),
    ("trades moist",  8.0, 300.0, -1.5, 0.90),
    ("ITCZ light",    3.0, 301.0, -0.5, 0.85),
    ("stratocumulus", 6.0, 291.0, -1.0, 0.82),
    ("midlat storm", 14.0, 283.0, -3.0, 0.80),
)
P_SFC = 101300.0


def _flux(u, T_air, q_air, T_sfc, q_sfc, rho, z_ref):
    _tx, _ty, shf, lhf, ustar = compute_most_fluxes(
        np.asarray([u]), np.asarray([0.0]), np.asarray([T_air]),
        np.asarray([q_air]), np.asarray([T_sfc]), np.asarray([q_sfc]),
        np.asarray([rho]), z_ref=z_ref, scheme="coare3",
        stability_scheme="dyer1974",
    )
    return float(lhf[0]), float(shf[0]), float(ustar[0])


def main(heights):
    print(f"{'regime':<16}{'z[m]':>6}{'LH(10m)':>10}{'LH(true)':>10}"
          f"{'ratio':>8}{'SH ratio':>10}{'stress':>9}")
    for z in heights:
        for name, u, sst, dT, rh in REGIMES:
            T_air = sst + dT
            q_air = rh * float(saturation_specific_humidity(
                np.asarray(T_air), P_SFC))
            q_sfc = 0.98 * float(saturation_specific_humidity(
                np.asarray(sst), P_SFC))
            rho = P_SFC / (constants.R_d * T_air)
            lh10, sh10, us10 = _flux(u, T_air, q_air, sst, q_sfc, rho, 10.0)
            lhz, shz, usz = _flux(u, T_air, q_air, sst, q_sfc, rho, z)
            if abs(lhz) < 1e-9:
                raise SystemExit("FATAL: zero reference flux -- the control "
                                 "would be a no-op and the ratio meaningless")
            print(f"{name:<16}{z:6.0f}{lh10:10.2f}{lhz:10.2f}"
                  f"{lh10 / lhz:8.3f}{sh10 / shz:10.3f}"
                  f"{(us10 / usz) ** 2:9.3f}")
    print("ratio > 1 means labelling the level 10 m makes the model evaporate "
          "that much faster than the same air would at its true height. The "
          "stress column is the same error on momentum.")


if __name__ == "__main__":
    hs = [float(a) for a in sys.argv[1:]] or [60.0, 100.0, 150.0]
    main(hs)
