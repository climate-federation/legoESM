"""Iter-777 diagnostic: test enabling Fortran-faithful `nord` /
`damp_c` 4th-order smoother in cosine bell `transport_step`.

Fortran reference (`../atmos_cubed_sphere-symmetryclean/model/
sw_core.F90:886-887`):
    call fv_tp_2d(delp, crx_adv, cry_adv, ..., nord=nord_v,
                  damp_c=damp_v)

That is, Fortran's d_sw1 transport call passes `nord=nord_v` and
`damp_c=damp_v` — the vorticity-damping-order and coefficient
fields (NOT `d4_bg` — iter-777b correction of iter-777's initial
misread).  Our `CDGridShallowWaterConfig` has `damp_v=0.06,
nord_v=2` (Fortran defaults per iter-755b).  Our matrix cosine
bell (`scripts/run_atmosphere_test_matrix.py:1603-1606`) calls

    transport_step(s.h, ut, vt, dt_, cdgrid, mass_target=_mass_target)

with NO nord or damp_c, so the 4th-order smoother inside
`fv_tp_2d` (lines 527-530 of `src/legoesm/core/fv_tp_2d.py`) is
SKIPPED.  That is a Fortran-fidelity gap.

This diagnostic runs the cosine bell C36 1-day at three configs:
  (a) nord=None, damp_c=None   (matrix default; iter-776 baseline)
  (b) nord=2, damp_c=0.06      (Fortran-faithful per sw_core line
                                886-887 and our Config's nord_v,
                                damp_v)
  (c) nord=1, damp_c=0.16      (Fortran d4_bg — different knob, a
                                cross-check only)

For each, reports PL07 error norms (L1, L2, Linf) and peak
undershoot.  If (b) or (c) reduces the 9.5 % undershoot of (a),
the Fortran-faithful smoother is worth porting into the matrix
cosine bell call.  If it worsens, the None default is preferable
on this test.
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
t_s = days * 86400.0
h_exact = cosine_bell_exact(grid.lon, grid.lat, grid.radius, t_s, beta)


def run_and_measure(nord, damp_c, label):
    h = state.h
    for _ in range(n_steps):
        h = transport_step(h, ut, vt, dt, cdgrid,
                            mass_target=mass_init,
                            nord=nord, damp_c=damp_c)
    norms = cosine_bell_error_norms(h, h_exact, grid.area)
    h_np = np.asarray(h)
    peak_h = float(np.max(h_np))
    peak_ex = float(np.max(np.asarray(h_exact)))
    und = (peak_ex - peak_h) / peak_ex
    return float(norms['l1']), float(norms['l2']), float(norms['linf']), und


print(f"Iter-777 cosine bell C36 1-day: nord/damp_c sweep")
print(f"Matrix config baseline (iter-776): nord=None, damp_c=None")
print(f"Fortran sw_core.F90:886-887 passes nord=nord_v, damp_c=damp_v")
print(f"(Config defaults: nord_v=2, damp_v=0.06 per iter-755b).")
print()
print(f"{'nord':>6}  {'damp_c':>8}  {'L1':>10}  {'L2':>10}  "
      f"{'Linf':>10}  {'undershoot':>11}")
print("-" * 68)
for nord, damp_c, label in [
    (None, None, "matrix default"),
    (2, 0.06, "Fortran nord_v/damp_v"),
    (1, 0.16, "cross-check d4_bg"),
]:
    l1, l2, linf, und = run_and_measure(nord, damp_c, label)
    nord_str = "None" if nord is None else f"{nord}"
    dc_str = "None" if damp_c is None else f"{damp_c:.2f}"
    print(f"  {nord_str:>4}  {dc_str:>8}  {l1:>10.4e}  "
          f"{l2:>10.4e}  {linf:>10.4e}  {und:>10.4%}")
