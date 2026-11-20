"""float32 serial dry-mass fixer holds the anchored dry mass to round-off.

The fixer target used to be ``target_mass - B`` (two ~1e14 float32 totals),
which carried the float32 rounding of the total into the correction: a
one-time ~6e-6 relative dry-mass jump at the first fixer call on this state.
Runs in a subprocess with x64 OFF (the production float32 configuration).
"""
from __future__ import annotations

import os
import subprocess
import sys

_SCRIPT = r"""
import jax, jax.numpy as jnp, numpy as np
assert not jax.config.jax_enable_x64
from legoesm.atmosphere.dynamics.gcm.compressible_euler import CompressibleEulerConfig
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel, make_flat_plane_terrain_metric, make_rest_state)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate
N, NL = 32, 16
g = create_plane_grid(nx=N, ny=N, nlev=NL, dx=4.0e3, dy=4.0e3, dtype=jnp.float32)
hc = create_height_coordinate(NL, H=20.0e3)
tm = make_flat_plane_terrain_metric(g, hc)
rs = np.random.RandomState(7)
rest = make_rest_state(g, hc, dtype=jnp.float32)
s = rest._replace(
    u=rest.u.replace(data=jnp.asarray(0.5 * rs.standard_normal(rest.u.data.shape), jnp.float32)),
    rho_prime=rest.rho_prime.replace(
        data=jnp.asarray(1e-3 * rs.standard_normal(rest.rho_prime.data.shape), jnp.float32)))
cfg = CompressibleEulerConfig(fix_mass=True, anchor_mass_to_initial=True,
                              hyperdiff_coeff=0.0, hyperdiff_rho_coeff=0.0,
                              hyperdiff_w_coeff=0.0)
m = PlaneCompressibleEulerModel(g, hc, tm, cfg)
w = (np.asarray(g.area_T, np.float64) * np.asarray(tm.jacobian, np.float64))[:, :, None] \
    * np.asarray(hc.dz, np.float64)
def mass(st):
    assert st.rho_prime.data.dtype == jnp.float32
    return float(np.sum((np.asarray(hc.rho_ref, np.float64)
                         + np.asarray(st.rho_prime.data, np.float64)) * w))
m0 = mass(s)
drift = 0.0
for _ in range(3):
    s = m.step(s, 6.0)
    drift = max(drift, abs(mass(s) - m0) / m0)
print("REL_DRIFT", drift)
"""


def test_fp32_mass_fixer_holds_anchor_to_roundoff():
    """Max over 3 steps of |M(t) - M(0)| / M(0), masses summed in fp64."""
    env = os.environ.copy()
    env["JAX_ENABLE_X64"] = "0"
    env["JAX_PLATFORMS"] = "cpu"
    r = subprocess.run([sys.executable, "-c", _SCRIPT], env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    rel = float(r.stdout.split("REL_DRIFT")[1].split()[0])
    print("fp32 3-step dry-mass drift (relative):", rel)
    assert rel <= 1e-9, rel
