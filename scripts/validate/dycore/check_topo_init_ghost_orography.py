"""#1029 tripwire: does h_0 = 0 really remove the mountain from the INITIAL STATE?

Committed because the flat-mountain arm's whole claim rests on it: an arm that
still carries orography is not a control, and the number it produces is not
evidence.  Run it before believing any h_0 = 0 result.

GLM's objection to the flat-mountain arm: the terrain pressure-gradient forcing
acts through grad(ln p_s), not through h directly.  If the surface pressure is
built from anything other than the h_0 the override sets -- a stashed field, a
topography file, a restart -- then a "zero-height" arm still carries
grad(p_s) != 0 and exonerates nothing.

So: build the topo initial state at h_0 = 2000 and at h_0 = 0 and compare.
The treatment arm is only a control if p_s is UNIFORM at h_0 = 0.
"""
import numpy as np
import jax.numpy as jnp

from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import standard_hybrid_levels
from legoesm.atmosphere.idealized.held_suarez_topo import (
    held_suarez_topo_init_latlon)
from legoesm.atmosphere.forcing.idealized.held_suarez import (
    held_suarez_init_latlon)

grid = create_latlon_grid(72, 144)
sigma = standard_hybrid_levels(40)

s2000 = held_suarez_topo_init_latlon(grid, sigma, h_0=2000.0)
s0 = held_suarez_topo_init_latlon(grid, sigma, h_0=0.0)
sflat = held_suarez_init_latlon(grid, sigma)

for name, st in (("h_0=2000", s2000), ("h_0=0", s0), ("held_suarez", sflat)):
    ps = np.asarray(st.p_s.data)
    phis = np.asarray(getattr(st, "phis").data) if hasattr(st, "phis") else None
    print(f"{name:12s} p_s min={ps.min():.6f} max={ps.max():.6f} "
          f"ptp={np.ptp(ps):.6e} Pa"
          + (f" | phis ptp={np.ptp(phis):.6e} m2/s2" if phis is not None else ""))

ps0 = np.asarray(s0.p_s.data)
ps2 = np.asarray(s2000.p_s.data)
psf = np.asarray(sflat.p_s.data)
print()
print(f"GHOST OROGRAPHY CHECK: p_s spread at h_0=0 is {np.ptp(ps0):.3e} Pa "
      f"({'UNIFORM -> the arm is a real control' if np.ptp(ps0) < 1e-6 else 'NOT uniform -> the mountain survives the override'})")
print(f"control has a real mountain: p_s spread at h_0=2000 is {np.ptp(ps2):.3e} Pa")
print(f"h_0=0 vs held_suarez: max|dp_s| = {np.abs(ps0 - psf).max():.3e} Pa, "
      f"max|dT| = {np.abs(np.asarray(s0.T.data) - np.asarray(sflat.T.data)).max():.3e} K, "
      f"max|du| = {np.abs(np.asarray(s0.u.data) - np.asarray(sflat.u.data)).max():.3e} m/s")

# Measured 2026-09-14, latlon 72x144 / L40 hybrid:
#   h_0=2000     p_s 80363.26 .. 100000.00 Pa   (ptp 1.96e+04), phis ptp 1.88e+04
#   h_0=0        p_s uniform 100000.00 Pa        (ptp 0), phis 0
#   h_0=0 vs held_suarez: max|dp_s| = 0, max|dT| = 0, max|du| = 0  -- IDENTICAL
# So zeroing the height reduces the topo initial state EXACTLY to the flat one,
# and the arm is a real control.
