"""Snow densification: this model's snow column vs CLM5's, on the demo forcing.

Runs the layered land column of ``land_layered_snow_demo`` (30 d early winter)
and records every layer's temperature, mass, overburden and density.  Along
that trajectory it evaluates, term by term, CLM5's compaction rates
(CTSM ``SnowHydrologyMod.SnowCompaction`` with the CLM5 default Vionnet 2012
overburden; constants from the CLM5 technical note, van Kampenhout et al.
2017):

* destructive metamorphism  CR1 = -c3 * c1 * c5_liq * exp(-c4 (Tf - T))
* overburden (Vionnet 2012)  CR2 = -(P + w/2) / eta,
  eta = f1 f2 eta0 (rho/c_eta) exp(a_eta (Tf - T) + b_eta rho)
* melt                       CR3 = fractional ice loss per step (zero while cold)

as ``drho/dt = -rho * CR`` (ice mass fixed), next to this model's own rate
``(rho_max - rho) / tau``.  It also integrates CLM5's rates along each
layer's temperature/overburden history (a passive density tracer, same mass,
same fresh-snow density as this model) to the day-30 density.

    JAX_ENABLE_X64=1 python scripts/validate/land_snow_compaction_check.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.land.boundary_data.gap_fill import bare_canopy_params
from legoesm.land.multilayer_land import (
    MultiLayerLandConfig, init_multilayer_land_state, step_multilayer_land_with_diagnostics)
from legoesm.land.snow_column import SnowColumnState, clm5_compaction_rate
from legoesm.land.surface_scheme import TwoLeafCanopyConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
import land_layered_snow_demo as demo  # noqa: E402

jax.config.update("jax_enable_x64", True)

# --- CLM5 snow compaction (CTSM SnowHydrologyMod; CLM5 tech note, Snow Hydrology) ---
_C3 = 2.777e-6          # [1/s] destructive-metamorphism rate at T_freeze
_C4 = 0.04              # [1/K]
_C5 = 2.0               # liquid-water enhancement of metamorphism
_RHO_DM_UPPER = 175.0   # [kg/m3] upplim_destruct_metamorph (CLM5)
_DM_DECAY = 0.046       # [m3/kg] metamorphism decay above the upper limit
_ETA0 = 7.62237e6       # [kg s/m2] eta0_vionnet
_A_ETA = 0.1            # [1/K]
_B_ETA = 0.023          # [m3/kg]
_C_ETA = 450.0          # [kg/m3]
_F2 = 4.0               # fixed at maximum in CLM5
_F1_LIQ = 60.0          # liquid-water viscosity factor (Vionnet 2012)

TF = constants.T_freeze
RHO_W = constants.rho_water


def clm5_rates(rho, T, burden, w, liq, dz):
    """CLM5 fractional compaction rates CR1, CR2 [1/s] (negative = compaction)."""
    td = np.maximum(TF - T, 0.0)
    c1 = np.where(rho > _RHO_DM_UPPER, np.exp(-_DM_DECAY * (rho - _RHO_DM_UPPER)), 1.0)
    c5 = np.where(liq > 0.01 * dz, _C5, 1.0)
    cr1 = -_C3 * c1 * c5 * np.exp(-_C4 * td)
    f1 = 1.0 / (1.0 + _F1_LIQ * liq / (RHO_W * np.maximum(dz, 1e-9)))
    eta = f1 * _F2 * _ETA0 * (rho / _C_ETA) * np.exp(_A_ETA * td + _B_ETA * rho)
    cr2 = -(burden + 0.5 * w) / eta
    return cr1, cr2


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
    dz = w / np.maximum(rho, 1e-9)
    spd = int(86400 / demo.DT)
    # The pack's own rate (CLM5 port, wind drift included) along the same history.
    wind = np.array([float(np.hypot(f.u_lowest[0], f.v_lowest[0]))
                     for f in (demo.forcing_at(k) for k in range(rec.shape[0]))])
    ours = -rho * np.asarray(clm5_compaction_rate(
        SnowColumnState(ice, liq, T, rho), wind))                   # drho/dt [kg/m3/s]
    cr1, cr2 = clm5_rates(rho, T, burden, w, liq, dz)
    print(f"git {sha}  demo forcing, layered column, CLM5 constants (Vionnet 2012 overburden)")
    print("day layer |  rho    T    burden |  ours drho/dt | CLM5 metamorph  overburden  "
          "[kg/m3/day]")
    for d in (5, 12, 20, 29):
        i = d * spd - 1
        for j in (0, 2, 4):
            if w[i, j] <= 0.0:
                continue
            print(f"{d:3d}  {j}    | {rho[i, j]:5.0f} {T[i, j]:6.1f} {burden[i, j]:6.1f} |"
                  f" {ours[i, j] * 86400:9.2f}    | {-rho[i, j] * cr1[i, j] * 86400:9.2f}"
                  f"   {-rho[i, j] * cr2[i, j] * 86400:9.3f}")
    # Passive CLM5 density along each layer's T / overburden history.
    rc = np.array(rho[0])
    had = np.zeros(rho.shape[1], bool)
    for i in range(rho.shape[0]):
        new = (w[i] > 0.0) & ~had
        rc = np.where(new, rho[i], rc)
        had |= w[i] > 0.0
        a, b = clm5_rates(rc, T[i], burden[i], w[i], liq[i], w[i] / rc)
        rc = np.where(w[i] > 0.0, rc * (1.0 - (a + b) * demo.DT), rc)
    wm = w[-1] / w[-1].sum()
    print(f"day 30 pack-mean density: ours {np.sum(wm * rho[-1]):.0f} kg/m3, "
          f"CLM5 rates on the same history {np.sum(wm * rc):.0f} kg/m3 "
          f"(layers ours {np.round(rho[-1]).astype(int)}, CLM5 {np.round(rc).astype(int)})")


if __name__ == "__main__":
    main()
