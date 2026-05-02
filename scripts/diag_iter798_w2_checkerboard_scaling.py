"""Iter-798 diagnostic: W2 h-checkerboard amplitude scaling with
resolution.

Per iter-797's new observation of a 2dx checkerboard in h_err at
C24 with iter-761 config, iter-798 runs W2 at C24, C36, C48 with
identical iter-761 config and computes the 2dx checkerboard
amplitude at each resolution.

The 2dx signal is extracted by:
  ch_x = 0.25 * (h_err[i-1,j] - 2*h_err[i,j] + h_err[i+1,j]
                - h_err[i,j-1] + 2*h_err[i,j] - h_err[i,j+1])
  ... i.e., a 2dx band-pass filter.

A simpler metric: compute the field
  checker[i,j] = (-1)^(i+j) * h_err[i,j]
and report its amplitude.  For a pure 2dx checkerboard, this
flips signs and averages to the checkerboard amplitude.  For
smooth fields, it averages toward zero.

If checkerboard amplitude scales as dx^2 (resolution-dependent
truncation error): the checkerboard is inherent numerical
truncation and improves with resolution.  If amplitude is
resolution-invariant: it's a structural artifact.
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
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


def _run_w2(n, hours=24.0, dt=300.0):
    n_steps = int(round(hours * 3600 / dt))
    div_damp = 8.0 * _div_damp_cube(n)
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=div_damp,
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2)
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid

    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
    model.set_initial_mass(state)

    h0 = state.h
    for _ in range(n_steps):
        state = model.step(state, dt)

    h_err = np.asarray(state.h - h0)
    return h_err


def _checkerboard_amp(h_err):
    """Compute 2dx checkerboard amplitude via sign-alternating mean.

    For each face: field with (-1)^(i+j) sign pattern applied, then
    average over interior (exclude 2-cell boundary buffer to
    isolate interior 2dx signal).  Return face-averaged absolute
    checkerboard amplitude.
    """
    n_faces, n_i, n_j = h_err.shape
    ii = np.arange(n_i)[:, None]
    jj = np.arange(n_j)[None, :]
    sign = ((-1) ** (ii + jj)).astype(np.float64)
    # Interior (exclude 3-cell boundary to avoid edge effects)
    b = 3
    amps = []
    for f in range(n_faces):
        sub = h_err[f, b:n_i-b, b:n_j-b] * sign[b:n_i-b, b:n_j-b]
        amp = float(np.abs(np.mean(sub)))
        amps.append(amp)
    return amps


def _pole_amp(h_err):
    """Amplitude of polar face (4 or 5) maximum |h_err|."""
    return float(np.max([np.abs(h_err[4]).max(), np.abs(h_err[5]).max()]))


print(f"Iter-798 W2 h_err checkerboard amplitude vs resolution")
print(f"iter-761 canonical config, dt=300s, 1 day")
print()
print(f"{'n':>4}  {'dx [km]':>10}  {'max |h_err|':>12}  "
      f"{'face-avg chk amp':>18}  {'face4 max':>12}")
print("-" * 68)

results = {}
for n in (24, 36, 48):
    dx_km = (6371229.0 * (np.pi / 2) / n) / 1000.0
    h_err = _run_w2(n)
    max_abs = float(np.max(np.abs(h_err)))
    chk = _checkerboard_amp(h_err)
    avg_chk = float(np.mean(chk))
    f4max = float(np.max(np.abs(h_err[4])))
    print(f"  {n:>3}  {dx_km:>9.1f}  {max_abs:>12.3e}  "
          f"{avg_chk:>18.3e}  {f4max:>12.3e}")
    results[n] = {'max_abs': max_abs, 'avg_chk': avg_chk, 'f4max': f4max}

print()
print("Scaling analysis:")
ns = sorted(results.keys())
for i in range(len(ns) - 1):
    n1, n2 = ns[i], ns[i+1]
    r = results[n1]; s = results[n2]
    dx_ratio = n2 / n1
    chk_ratio = s['avg_chk'] / r['avg_chk'] if r['avg_chk'] > 0 else np.nan
    print(f"  C{n1}→C{n2} (dx ratio {1/dx_ratio:.2f}×): "
          f"checkerboard amp ratio = {chk_ratio:.2f} "
          f"(expected dx^2 scaling: {(1/dx_ratio)**2:.2f})")

print()
print("Interpretation cues (observational only):")
print("- If checkerboard amp scales as dx^2: truncation-level, not")
print("  structural.  Improves naturally with resolution.")
print("- If resolution-invariant: structural artifact, needs")
print("  targeted suppression.")

print()
print("What iter-798 DOES measure:")
print("- 2dx checkerboard amplitude in h_err at C24/C36/C48.")
print("What it does NOT establish:")
print("- The origin of the checkerboard (PPM 2dx null mode,")
print("  boundary_fix cascaded smoothing, or A-L aliasing).")
