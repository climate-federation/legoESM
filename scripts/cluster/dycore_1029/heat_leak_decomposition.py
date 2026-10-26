"""#1029: split the global heat leak of the lat-lon del-4 T term.

From a saved state (npz with T, p_s, phis, u, v), evaluates the del-4 part of
dT/dt (tendency with nu_del4 minus without) and integrates
  L_mass  = sum c_p dT4 dp/g dA / A          (what the model's energy sees)
  L_level = same but with the reference layer mass (p_s = p_ref) everywhere
            (removes the terrain-varying layer-mass weighting)
for the production per-row cap and with the cap lifted (cfl_frac huge; one
tendency evaluation only, so stability is irrelevant).  Measurement only.
"""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonHydrostaticState, CGridLatLonPrimitiveEquationConfig,
    cgrid_latlon_hydrostatic_tendencies)
from legoesm.grids.latlon import create_latlon_grid

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/matrix"))
import run_atmosphere_test_matrix as runner  # noqa: E402

z = np.load(sys.argv[1])
grid = create_latlon_grid(72, 144)
sigma = runner._create_vertical(40, "hybrid")
st = CGridLatLonHydrostaticState(**{k: jnp.asarray(z[k]) for k in ("u", "v", "T", "p_s", "phis")})
nu = runner._biharmonic_visc_latlon(72)
dt = 200.0
dp = jnp.diff(sigma.pressure_at_half(st.p_s), axis=-1)
dp_ref = jnp.diff(sigma.pressure_at_half(jnp.full_like(st.p_s, constants.p_ref)), axis=-1)
A = jnp.sum(grid.area)
off = cgrid_latlon_hydrostatic_tendencies(st, grid, sigma, CGridLatLonPrimitiveEquationConfig(), dt=dt)[2]
for label, frac in (("capped (production)", 0.25), ("cap lifted", 1e30)):
    cfg = CGridLatLonPrimitiveEquationConfig(nu_del4=nu, nu_del4_cfl_frac=frac)
    d4 = cgrid_latlon_hydrostatic_tendencies(st, grid, sigma, cfg, dt=dt)[2] - off
    lm = float(jnp.sum(jnp.sum(constants.c_pd * d4 * dp, -1) / constants.g * grid.area) / A)
    ll = float(jnp.sum(jnp.sum(constants.c_pd * d4 * dp_ref, -1) / constants.g * grid.area) / A)
    print(f"{label:22s} L_mass {lm:+.4e} W/m2   L_level(ref layer mass) {ll:+.4e} W/m2")
