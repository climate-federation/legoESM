"""Probe: SW effect of giving CMIP6 volcanic aerosol its own per-band optics.

One clear-sky equatorial column (CAM6 L32, p_s 1000 hPa, US-Std-like T,
no tropospheric aerosol), mid-month of a given year/month, RRTMGP full
tables.  Arms, same column / same placement / same tropopause weight:
  grey: OLD path -- ext_sun collapsed solar-weighted (5778 K Planck) to one
        gray tau, solved with the grey aerosol ssa/g (RRTMGPConfig 0.93/0.70);
  file: NEW path -- per-band (tau, tau*ssa, tau*ssa*g) from the file.
Prints TOA upward SW, surface downward SW and SW absorbed above 100 hPa
[W/m2] at cos(zenith) = 0.5 and the daily-mean-equivalent (x insolation
weight not applied; instantaneous).  Usage:
  python scripts/validate/probe_volcanic_sw_optics.py FILE YEAR MONTH
"""
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def main(path, year, month):
    from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import (
        RRTMGP, gas_table_files)
    from legoesm.forcing.external import (
        AerosolConfig, _SOLAR_PLANCK_TEMP_K, _planck_band_weights,
        get_aerosol_sw_volcanic_optics_at_time)
    from legoesm.forcing.surface_utils import (
        place_stratospheric_aod_profile_to_layers, stratospheric_layer_weight)
    from legoesm.grids.vertical import make_cam6_l32_levels

    sig = make_cam6_l32_levels()
    p_s = jnp.array([1.0e5])
    p_half = jnp.asarray(sig.pressure_at_half(p_s)).reshape(1, -1)
    p_full = jnp.asarray(sig.pressure_at_full(p_s)).reshape(1, -1)
    T = jnp.maximum(300.0 * (p_full / 1.0e5) ** 0.19, 200.0)
    q_v = jnp.clip(0.018 * (p_full / 1.0e5) ** 3, 3.0e-6, None)
    lat = jnp.array([0.0])
    day = 365.0 * (year - 1979) + 30.4 * (month - 1) + 15.0
    acfg = AerosolConfig(enabled=True, volcanic_enabled=True,
                         volcanic_path=path, start_year=1979)
    mom, p_edges, band_wn = get_aerosol_sw_volcanic_optics_at_time(acfg, day, lat)
    w = stratospheric_layer_weight(p_half, lat, p_full, T)
    nsrc = mom.shape[1]
    flat = jnp.moveaxis(jnp.asarray(mom, jnp.float64).reshape(1, nsrc, 42), -1, 0)
    placed = jax.vmap(place_stratospheric_aod_profile_to_layers,
                      in_axes=(0, None, None))(flat, jnp.asarray(p_edges), p_half) * w
    volc = jnp.moveaxis(placed, 0, -1).reshape(1, -1, 3, 14)
    wl = 1.0e4 / band_wn                       # cm-1 -> um, (14, 2)
    wts = _planck_band_weights(wl[:, 1], wl[:, 0], _SOLAR_PLANCK_TEMP_K)
    tau_gray = jnp.tensordot(volc[:, :, 0, :], jnp.asarray(wts / wts.sum()),
                             axes=(2, 0))

    lw, sw = gas_table_files("full")
    cfg = RRTMGPConfig(lw_gas_file=lw, sw_gas_file=sw)
    solver = RRTMGP.from_legoesm_config(cfg)
    kw = dict(T=T, p_full=p_full, p_half=p_half,
              sfc_temperature=jnp.array([300.0]), q_v=q_v,
              cos_zenith=jnp.array([0.5]), sfc_albedo=jnp.array([0.07]))
    arms = {
        "none": solver.solve_columns(**kw),
        "grey": solver.solve_columns(**kw, aerosol_optical_depth=tau_gray),
        "file": solver.solve_columns(**kw, aerosol_optical_depth=jnp.zeros_like(tau_gray),
                                     aerosol_volcanic_moments_sw=volc),
    }
    k100 = int(jnp.argmin(jnp.abs(p_half[0] - 1.0e4)))
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"file={path} {year}-{month:02d} lat=0 cosz=0.5 code={sha}")
    tau_vis = float(volc[0, :, 0, 10].sum())
    ssa_vis = float(volc[0, :, 1, 10].sum() / max(tau_vis, 1e-30))
    print(f"volcanic column tau 16000-22650 cm-1 {tau_vis:.4f} ssa {ssa_vis:.4f};"
          f" gray tau {float(tau_gray.sum()):.4f}")
    for k, o in arms.items():
        net = o.sw_flux_down[0] - o.sw_flux_up[0]
        print(f"{k:5s} toa_up {float(o.sw_flux_up[0, 0]):8.3f}  sfc_down "
              f"{float(o.sw_flux_down[0, -1]):8.3f}  abs_above_100hPa "
              f"{float(net[0] - net[k100]):7.3f}  abs_column "
              f"{float(net[0] - net[-1]):8.3f}")
    g, f = arms["grey"], arms["file"]
    print(f"file-grey: toa_up {float(f.sw_flux_up[0,0]-g.sw_flux_up[0,0]):+.3f} "
          f"sfc_down {float(f.sw_flux_down[0,-1]-g.sw_flux_down[0,-1]):+.3f} W/m2")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
