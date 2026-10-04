"""Single-column offline demo: bulk vs layered snowpack, 45-60N-like early winter.

30 days, dt = 1800 s, two-leaf canopy (LAI 1), soil freeze/thaw OFF in both runs
(the layered pack refuses it until the hydrology fusion heat is charged there),
starting from a 278 K / theta 0.25 autumn soil.  Air temperature ramps from 268
to 258 K with a +-4 K diurnal cycle; 8-hour days with 150 W/m2 peak sun;
downwelling LW = 0.75 sigma T_air^4; three 10 kg/m2 snowfalls (days 2, 10, 18,
12 h each) and a 275 K rain-on-snow day (day 25, 5 kg/m2); air at 80%
relative humidity over ice.  Prints soil
temperature interpolated to 0.3 m and 1 m, SWE and skin T for both snowpacks.

    JAX_ENABLE_X64=1 python scripts/validate/land_layered_snow_demo.py
"""
from __future__ import annotations

import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.boundary_data.gap_fill import bare_canopy_params
from legoesm.land.multilayer_land import (
    MultiLayerLandConfig, init_multilayer_land_state, step_multilayer_land_with_diagnostics)
from legoesm.land.soil_grid import make_soil_grid
from legoesm.land.surface_scheme import TwoLeafCanopyConfig
from legoesm.thermo import saturation_mixing_ratio_ice

jax.config.update("jax_enable_x64", True)
DT = 1800.0
DAYS = 30
DEPTHS = (0.3, 1.0)


def forcing_at(k):
    t = k * DT / 86400.0
    day, hour = int(t), (t % 1.0) * 24.0
    T = 268.0 - 10.0 * t / DAYS + 4.0 * np.sin(2 * np.pi * (hour - 9.0) / 24.0)
    sw = max(150.0 * np.sin(np.pi * (hour - 8.0) / 8.0), 0.0) if 8.0 <= hour <= 16.0 else 0.0
    snow = 10.0 / 43200.0 if (day in (2, 10, 18) and hour < 12.0) else 0.0
    rain = 0.0
    if day == 25:
        T, rain = 275.0, 5.0 / 86400.0
    one = jnp.ones(1)
    return AtmToSurface(
        sw_down=sw * one, lw_down=0.75 * constants.sigma_sb * T ** 4 * one,
        precip_total=(snow + rain) * one, precip_snow=snow * one,
        T_lowest=T * one,
        q_lowest=0.8 * saturation_mixing_ratio_ice(T * one, 1.0e5 * one),
        u_lowest=4.0 * one, v_lowest=0.0 * one, p_lowest=95000.0 * one,
        p_surface=1.0e5 * one, rho_lowest=1.0e5 / (constants.R_d * T) * one,
        cos_zenith=0.3 * one, co2_ppmv=410.0 * one, has_radiation=one,
        has_precipitation=one)


def run(snow_scheme, days=DAYS):
    cfg = MultiLayerLandConfig(surface_scheme=TwoLeafCanopyConfig(),
                               snow_scheme=snow_scheme, snow_albedo_feedback=True)
    # Freeze/thaw off for BOTH schemes: the layered pack refuses it, and the
    # bulk-vs-layered comparison must differ in the snowpack only.
    cfg = cfg._replace(thermal=cfg.thermal._replace(enable_freeze_thaw=False))
    st = init_multilayer_land_state(1, cfg, T_init=278.0, theta_init=0.25)
    lp = bare_canopy_params(1)._replace(LAI=jnp.asarray([1.0]))
    step = jax.jit(lambda s, f: step_multilayer_land_with_diagnostics(
        s, f, cfg, 1.0, DT, lat=jnp.full(1, np.deg2rad(52.0)), land_params=lp))
    z = np.asarray(make_soil_grid(cfg.soil_grid).z_node)
    rows = []
    held = 0
    for k in range(int(days * 86400 / DT)):
        st, resp, _, sfc = step(st, forcing_at(k))
        held += int(sfc.n_held)
        T = np.asarray(st.T_soil[0])
        rows.append((np.interp(DEPTHS[0], z, T), np.interp(DEPTHS[1], z, T),
                     float(st.snow_depth[0]), float(resp.T_sfc[0])))
    return np.array(rows), held


def main():
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    print(f"git {sha}  dt={DT:.0f}s days={DAYS}  depths={DEPTHS} m  freeze_thaw=off  two_leaf LAI=1")
    out = {s: run(s) for s in ("bulk", "layered")}
    spd = int(86400 / DT)
    print("day | T0.3m bulk  layered | T1m bulk  layered | SWE bulk  layered | Tsfc bulk  layered")
    for d in (1, 5, 10, 15, 20, 25, 30):
        i = d * spd - 1
        b, lay = out["bulk"][0][i], out["layered"][0][i]
        print(f"{d:3d} | {b[0]:7.2f}  {lay[0]:7.2f} | {b[1]:7.2f}  {lay[1]:7.2f} | "
              f"{b[2]:6.1f}  {lay[2]:6.1f} | {b[3]:7.2f}  {lay[3]:7.2f}")
    for s in ("bulk", "layered"):
        r, held = out[s]
        print(f"{s}: mean days 21-30 T0.3m = {r[20 * spd:, 0].mean():.2f} K, "
              f"T1m = {r[20 * spd:, 1].mean():.2f} K; min T0.3m = {r[:, 0].min():.2f} K, "
              f"held columns = {held}")


if __name__ == "__main__":
    main()
