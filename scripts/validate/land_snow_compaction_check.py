"""Snow densification of the layered pack on the demo forcing.

Runs the layered land column of ``land_layered_snow_demo`` (30 d early winter)
and prints, per layer, density, temperature, overburden and the densification
rate ``drho/dt = -rho * rate`` of the pack's CLM5 port
(``snow_column.clm5_compaction_rate``: destructive metamorphism, Vionnet 2012
overburden, wind drift).  The port itself is checked against line-by-line CTSM
transliterations in ``tests/land/test_snow_compaction_clm5.py``.

    JAX_ENABLE_X64=1 python scripts/validate/land_snow_compaction_check.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.land.boundary_data.gap_fill import bare_canopy_params
from legoesm.land.multilayer_land import (
    MultiLayerLandConfig, init_multilayer_land_state, step_multilayer_land_with_diagnostics)
from legoesm.land.snow_column import SnowColumnState, clm5_compaction_rate
from legoesm.land.surface_scheme import TwoLeafCanopyConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
import land_layered_snow_demo as demo  # noqa: E402

jax.config.update("jax_enable_x64", True)

def run_layered(days=demo.DAYS):
    cfg = MultiLayerLandConfig(surface_scheme=TwoLeafCanopyConfig(), snow_scheme="layered",
                               snow_albedo_feedback=True)
    # Freeze/thaw off, as the demo runs it.
    cfg = cfg._replace(thermal=cfg.thermal._replace(enable_freeze_thaw=False))
    st = init_multilayer_land_state(1, cfg, T_init=278.0, theta_init=0.25)
    lp = bare_canopy_params(1)._replace(LAI=jnp.asarray([1.0]))
    step = jax.jit(lambda s, f: step_multilayer_land_with_diagnostics(
        s, f, cfg, 1.0, demo.DT, lat=jnp.full(1, np.deg2rad(52.0)), land_params=lp))
    rec = []
    for k in range(int(days * 86400 / demo.DT)):
        st, _, _, _ = step(st, demo.forcing_at(k))
        rec.append([np.asarray(x[0]) for x in (st.snow_ice_layers, st.snow_liq_layers,
                                               st.snow_T_layers, st.snow_rho_layers)])
    return cfg, np.array(rec)            # (nstep, 4, n_layers)


def main():
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    cfg, rec = run_layered()
    ice, liq, T, rho = rec[:, 0], rec[:, 1], rec[:, 2], rec[:, 3]
    w = ice + liq
    burden = np.cumsum(w, axis=1) - w                     # mass above each layer
    spd = int(86400 / demo.DT)
    # The pack's own rate (CLM5 port, wind drift included) along the same history.
    wind = np.array([float(np.hypot(f.u_lowest[0], f.v_lowest[0]))
                     for f in (demo.forcing_at(k) for k in range(rec.shape[0]))])
    ours = -rho * np.asarray(clm5_compaction_rate(
        SnowColumnState(ice, liq, T, rho), wind))                   # drho/dt [kg/m3/s]
    print(f"git {sha}  demo forcing, layered column (CLM5 compaction port)")
    print("day layer |  rho    T    burden |  drho/dt [kg/m3/day]")
    for d in (5, 12, 20, 29):
        i = d * spd - 1
        for j in (0, 2, 4):
            if w[i, j] <= 0.0:
                continue
            print(f"{d:3d}  {j}    | {rho[i, j]:5.0f} {T[i, j]:6.1f} {burden[i, j]:6.1f} |"
                  f" {ours[i, j] * 86400:9.2f}")
    wm = w[-1] / w[-1].sum()
    print(f"day 30 pack-mean density {np.sum(wm * rho[-1]):.0f} kg/m3 "
          f"(layers {np.round(rho[-1]).astype(int)})")


if __name__ == "__main__":
    main()
