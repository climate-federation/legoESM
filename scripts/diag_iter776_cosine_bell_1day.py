"""Iter-776 diagnostic: cosine bell distortion measurement at
t=1 day on C36, canonical matrix path.

User visual inspection (iter-775 stop-hook) reported "slight
distortion near 1 day" on the cosine bell snapshot.  This
diagnostic measures the distortion quantitatively.

Cosine bell is a pure transport test: the bell should translate
with the prescribed wind without any shape change.  Any deviation
from the analytic solution is transport-truncation error.  Cosine
bell does NOT use the A-L gradient, so iter-774's c10 correction
is irrelevant here.  The transport-path suspects are:
- `transport_step` / `fv_tp_2d` (Lin-Rood operator-split PPM).
- `_d2a2c_vect` contravariant-velocity computation at cube corners.
- Halo handling in the PPM sweeps via `pad_halo` at halo=2.

Metrics reported:
- L1, L2, Linf error norms (PL07 convention) at t=1 day vs exact.
- Peak height vs exact (undershoot measure).
- Cell position of peak (drift away from exact peak position).
- Cross-sections through the peak along lat and lon directions.
"""
import os, sys
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
for _stale in ("JAX_PLATFORM_NAME", "JAX_DISABLE_JIT", "JAX_DEBUG_NANS"):
    os.environ.pop(_stale, None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel)
from legoesm.core.fv3_sw_core import _d2a2c_vect
from legoesm.core.fv_tp_2d import transport_step
from tests.test_cases.cosine_bell import (
    cosine_bell_cubesphere, cosine_bell_exact, cosine_bell_error_norms)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


n = 36
dt = 1800.0
days = 1
n_steps = int(days * 86400 / dt)
beta = jnp.pi / 4.0

grid = create_cubed_sphere(n)
cfg = CDGridShallowWaterConfig(
    hyperdiff_coeff=0.0,
    div_damp=_div_damp_cube(n),
    boundary_fix=True,
    damp_v=0.06,
    nord_v=2)
model = FV3EdgeShallowWaterModel(grid, config=cfg)
cdgrid = model.cdgrid

state = cosine_bell_cubesphere(grid, cdgrid, beta)
_ua, _va, _uc, _vc, ut, vt = _d2a2c_vect(state.u_d, state.v_d, cdgrid)
mass_init = float(jnp.sum(state.h * grid.area))

h = state.h
for _ in range(n_steps):
    h = transport_step(h, ut, vt, dt, cdgrid,
                        mass_target=mass_init)

# Compare to analytic exact at t=1 day.
t_s = days * 86400.0
h_exact = cosine_bell_exact(grid.lon, grid.lat, grid.radius, t_s, beta)
h_np = np.asarray(h)
h_ex = np.asarray(h_exact)
area = np.asarray(grid.area)

norms = cosine_bell_error_norms(h, h_exact, grid.area)
peak_h = float(np.max(h_np))
peak_h_exact = float(np.max(h_ex))
peak_undershoot = (peak_h_exact - peak_h) / peak_h_exact

idx_h = np.unravel_index(np.argmax(h_np), h_np.shape)
idx_ex = np.unravel_index(np.argmax(h_ex), h_ex.shape)
lat_h = float(np.rad2deg(np.asarray(grid.lat)[idx_h]))
lon_h = float(np.rad2deg(np.asarray(grid.lon)[idx_h]))
lat_ex = float(np.rad2deg(np.asarray(grid.lat)[idx_ex]))
lon_ex = float(np.rad2deg(np.asarray(grid.lon)[idx_ex]))
# Normalize lons to (-180, 180]
if lon_h > 180: lon_h -= 360
if lon_ex > 180: lon_ex -= 360

print(f"Iter-776 cosine bell C36 1-day transport diagnostic")
print()
print(f"Error norms (PL07, vs exact analytic solution at t=1 day):")
print(f"  L1   = {float(norms['l1']):.4e}")
print(f"  L2   = {float(norms['l2']):.4e}")
print(f"  Linf = {float(norms['linf']):.4e}")
print()
print(f"Peak amplitude:")
print(f"  analytic exact:        {peak_h_exact:.4f} m")
print(f"  simulated at t=1 day:  {peak_h:.4f} m")
print(f"  undershoot:            {peak_undershoot:.4%} of exact peak")
print()
print(f"Peak cell position:")
print(f"  analytic exact:        face={idx_ex[0]}, (i, j) = "
      f"({idx_ex[1]}, {idx_ex[2]}), lat={lat_ex:.2f}, lon={lon_ex:.2f}")
print(f"  simulated at t=1 day:  face={idx_h[0]}, (i, j) = "
      f"({idx_h[1]}, {idx_h[2]}), lat={lat_h:.2f}, lon={lon_h:.2f}")
