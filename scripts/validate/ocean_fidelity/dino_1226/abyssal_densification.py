"""Southern-abyss densification rate, legoESM vs NEMO, from a MATCHED state.

Why this metric: #1226's ACC deficit was traced (memory addendum 19) to
southern abyssal dense-water formation running at ~16% of NEMO's rate.
Density change in the southern box is therefore both far upstream of ACC and
far more sensitive than ACC itself -- ACC's year-to-year growth scatters by
±40% in NEMO's own run, so a single year of ACC cannot discriminate a rate
deficit, whereas the densification it integrates can.

Protocol (matched-state, one variable = the model):
  reference  = NEMO's year-5-end restart (the state BOTH models started from)
  lego       = legoESM's year-6 annual mean, initialized from that restart
  NEMO       = NEMO's own year-6 annual mean, continued from that restart
Both deltas span the identical interval (end-y5 -> mid-y6) and carry the
identical annual-mean-vs-snapshot offset, so their RATIO is meaningful.

Two traps this script exists to avoid:
  * NEMO's ``votemper``/``vosaline`` in these XIOS outputs are THICKNESS-
    WEIGHTED (T*e3t) despite ``units='C'`` -- they are divided by ``vovvle3t``
    before use. Skipping that turns a thickness change into a fake T signal.
  * Density is evaluated with ONE EOS on both sides (NEMO's own S-EOS with the
    DINO coefficients, ``nemo_seos_eos``) at the SAME reference depth, so the
    only thing that can move rho is T/S.

Usage:
  abyssal_densification.py <lego_y6.npz> [nemo_grid_T.nc] [record_index]
"""
import os
import sys

import numpy as np
import netCDF4 as nc

from legoesm import constants
from legoesm.ocean.eos import nemo_seos_eos, NemoSEOSConfig

DINO = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO"
MESH = f"{DINO}/RUN_TRAJ/mesh_mask.nc"
REF_RESTART = f"{DINO}/RUN_Y5_REBUILD/DINO_00057600_restart.nc"

# Southern-channel boxes, verbatim from southern_heat_budget.py:73-74 (which
# took them from southern_convection_final.py) -- do not re-derive.
SEDGE = slice(12, 17)
CORE = slice(1, 11)

LEGO = sys.argv[1]
NEMO_T = sys.argv[2] if len(sys.argv) > 2 else (
    f"{DINO}/RUN_20Y_REBUILD/DINO_1y_00060101_00201230_grid_T.nc")
TIDX = int(sys.argv[3]) if len(sys.argv) > 3 else 0


def llz(a):
    """(lev, y, x) -> (y, x, lev), the legoESM field layout."""
    return np.moveaxis(np.asarray(a).squeeze(), 0, -1)


mm = nc.Dataset(MESH)
tmask = llz(mm["tmask"][0]) > 0.5
gdept = np.asarray(mm["gdept_1d"][:]).squeeze()          # [m], positive down
e3t = np.asarray(mm["e3t_1d"][:]).squeeze()              # [m]
area = np.asarray(mm["e1t"][0]).squeeze() * np.asarray(mm["e2t"][0]).squeeze()

# Pressure for the EOS: the Boussinesq reconstruction the model itself uses
# (nemo_seos_eos recovers depth as p/(rho0*g)), evaluated on the SAME
# reference depth ladder for every field so T/S is the only free variable.
_cfg = NemoSEOSConfig()
p_ref = _cfg.rho0 * constants.g * gdept[None, None, :]


def rho_of(T, S):
    return np.asarray(nemo_seos_eos(T, S, p_ref))


def load_nemo_annual(path, tidx):
    """Annual-mean T/S, de-weighted by the vvl thickness (see module docstring)."""
    d = nc.Dataset(path)
    e3 = llz(d["vovvle3t"][tidx])
    T = llz(d["votemper"][tidx]) / np.maximum(e3, 1e-6)
    S = llz(d["vosaline"][tidx]) / np.maximum(e3, 1e-6)
    return T, S


def load_restart(path):
    """Reference state: restart tn/sn are RAW (not thickness-weighted)."""
    d = nc.Dataset(path)
    return llz(d["tn"][0]), llz(d["sn"][0])


T_ref, S_ref = load_restart(REF_RESTART)
T_nemo, S_nemo = load_nemo_annual(NEMO_T, TIDX)
lg = np.load(LEGO)
T_lego, S_lego = lg["T"], lg["S"]

for name, arr in (("lego T", T_lego), ("NEMO T", T_nemo), ("ref T", T_ref)):
    if arr.shape != tmask.shape:
        raise SystemExit(f"SHAPE MISMATCH {name}: {arr.shape} vs mask {tmask.shape}")

rho_ref = rho_of(T_ref, S_ref)
rho_lego = rho_of(T_lego, S_lego)
rho_nemo = rho_of(T_nemo, S_nemo)

BANDS = [("0-200m", 0.0, 200.0), ("200-1000m", 200.0, 1000.0),
         ("1000-2500m", 1000.0, 2500.0), ("2500m+", 2500.0, 1e9)]

print(f"lego   : {LEGO}")
print(f"NEMO   : {NEMO_T} record {TIDX}")
print(f"ref    : {REF_RESTART} (the state BOTH models started from)")
print(f"EOS    : nemo_seos (DINO coefficients), one EOS both sides, "
      f"reference-depth pressure\n")

for box_name, rows in (("SEDGE", SEDGE), ("CORE", CORE)):
    print(f"--- {box_name} rows {rows.start}:{rows.stop} ---")
    print(f"{'band':<12}{'d_rho lego':>13}{'d_rho NEMO':>13}"
          f"{'lego/NEMO':>12}{'ncells':>9}")
    for band, z0, z1 in BANDS:
        lev = (gdept >= z0) & (gdept < z1)
        m = np.zeros_like(tmask)
        m[rows] = tmask[rows]
        m &= lev[None, None, :]
        if not m.any():
            continue
        # Volume-weighted so a band's thick deep cells are not out-voted by
        # thin ones; weights identical across all three fields.
        w = (area[..., None] * e3t[None, None, :])[m]
        d_lego = np.average((rho_lego - rho_ref)[m], weights=w)
        d_nemo = np.average((rho_nemo - rho_ref)[m], weights=w)
        ratio = d_lego / d_nemo if abs(d_nemo) > 1e-9 else np.nan
        print(f"{band:<12}{d_lego:>+13.5f}{d_nemo:>+13.5f}"
              f"{ratio:>12.3f}{int(m.sum()):>9}")
    print()

print("d_rho = kg/m^3 change from the common year-5 reference state over the "
      "SAME year.\nratio < 1 => legoESM densifies more slowly than NEMO from "
      "an identical start.")
