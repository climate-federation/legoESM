"""Iter-905 W2 measurement: does split mass+momentum integration
(mass via transport_step ONCE outside RK3, momentum via RK3 with h
held fixed) reduce the W2 v-bias relative to iter-892 default?

Implements path (3) of iter-904b's outlined paths forward.  iter-906
is reserved for path (2) (sub-dt threading) and iter-907+ for path
(1) (full FB-chain integration).

Acceptance gate per user's iter-904 spec:
  - v_ll_Linf improves >= 10 % AND h_L2 worsens < 5 %.
"""
import os, sys
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import (
    cell_centre_angles_from_4edge)
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)
from legoesm.grids.regridding import (
    get_cubedsphere_to_latlon_weights, apply_cubedsphere_to_latlon)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


def run_w2(use_split: bool):
    n = 36
    dt = 300.0
    n_steps = int(86400 / dt)
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    weights = get_cubedsphere_to_latlon_weights(n, n_lon=360, n_lat=181)

    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(n),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2,
        apply_fortran_xppm_boundary=True,
        use_split_mass_momentum_integration=use_split,
    )
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    cdg = model.cdgrid
    u_d = cdg.cos_angle_edge_x * (u0 * jnp.cos(cdg.lat_edge_x))
    v_d = -cdg.sin_angle_edge_y * (u0 * jnp.cos(cdg.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
    model.set_initial_mass(state)

    h_ic = np.asarray(sw.h.data)
    area = np.asarray(grid.area)

    for _ in range(n_steps):
        state = model.step(state, dt)

    h_final = np.asarray(state.h)
    h_l2 = float(np.sqrt(np.sum((h_final - h_ic) ** 2 * area)
                          / np.sum(h_ic ** 2 * area)))
    h_linf = float(np.max(np.abs(h_final - h_ic)))

    ca, sa = cell_centre_angles_from_4edge(cdg)
    u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1]
                   + np.asarray(state.u_d)[:, :, 1:])
    v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :]
                   + np.asarray(state.v_d)[:, 1:, :])
    v_north = np.asarray(sa) * u_cc + np.asarray(ca) * v_cc
    v_ll = apply_cubedsphere_to_latlon(v_north, weights)
    v_ll_linf = float(np.max(np.abs(v_ll)))
    return h_l2, h_linf, v_ll_linf


def main():
    print("Iter-905 W2 measurement: split mass+momentum integration")
    print("vs iter-892 default at C36, dt=300s, 1-day integration.")
    print()
    print(f"{'config':<45}  {'h_L2':>10}  {'h_Linf':>10}  {'v_ll_Linf':>11}")
    print("-" * 90)
    h_l2_off, h_linf_off, v_off = run_w2(use_split=False)
    print(f"{'(A) iter-892 default (production)':<45}  "
          f"{h_l2_off:>10.3e}  {h_linf_off:>10.3e}  {v_off:>11.4e}")
    h_l2_on, h_linf_on, v_on = run_w2(use_split=True)
    print(f"{'(B) iter-905 split mass+momentum':<45}  "
          f"{h_l2_on:>10.3e}  {h_linf_on:>10.3e}  {v_on:>11.4e}")
    print()
    v_pct = 100.0 * (v_on - v_off) / v_off
    h_pct = 100.0 * (h_l2_on - h_l2_off) / h_l2_off
    print(f"v_ll_Linf change:  {v_pct:+.2f} %  (acceptance: <= -10 %)")
    print(f"h_L2 change:        {h_pct:+.2f} %  (acceptance: <= +5 %)")
    print()
    if v_pct <= -10.0 and h_pct <= 5.0:
        print("VERDICT: iter-905 PASSES the Ralph-protocol acceptance gate.")
        print("         v_ll_Linf improves >=10% AND h_L2 worsens <5%.")
        print("         Recommend default-flip in iter-905+ after broader")
        print("         (W5/cosine bell/ocean rest) confirmation.")
    elif v_pct > -10.0 and h_pct <= 5.0:
        print("VERDICT: iter-905 does not improve W2 v_ll_Linf >=10%.")
        print("         Keep flag default-OFF; document negative result.")
    elif v_pct <= -10.0 and h_pct > 5.0:
        print("VERDICT: iter-905 improves v_ll_Linf but h_L2 regresses")
        print("         beyond 5% threshold.  Mixed result; keep default-OFF.")
    else:
        print("VERDICT: iter-905 worsens both v_ll_Linf and h_L2.")
        print("         Keep flag default-OFF; document negative result.")


if __name__ == "__main__":
    main()
