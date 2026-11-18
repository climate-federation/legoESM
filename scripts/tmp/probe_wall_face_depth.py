"""PROBE (measure only): barotropic face depth vs the sum of gated per-level face
thicknesses on the OceanBench 1-degree grid with the ORAS5 face walls.

(a) wet faces where the barotropic face depth H differs from sum_k of the gated
    min-rule layer thicknesses by more than 1e-9 relative, cut / uncut separately;
(b) cut faces whose open column is thinner than the minimum water column floor.

Usage (SLURM, CPU): python scripts/tmp/probe_wall_face_depth.py OUT_DIR
Inputs are the trainer's own defaults plus the two files named below.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/run"))
import train_ocean_wb_physics as T  # noqa: E402

WALLS = "/work/bb1596/b309178/diffESM/ocean_wb/oras5_face_walls.npz"
CAP = "/work/bb1596/b309178/diffESM/ocean_wb/oras5_depth_cap.npy"
DZ = Path("/work/bb1596/b309178/diffESM/ocean_wb/eorca1_L75_dz.txt")

jax.config.update("jax_enable_x64", True)
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
dz = T.write_dz_file(T.MESH_MASK, DZ)
argv = ["--grid", "latlon", "--resolution", "180x360", "--nlev", str(dz.size),
        "--dz-ref-file", str(DZ), "--H-max", f"{dz.sum():.9f}", "--dt", "1800.0", "--days", "1",
        "--bathymetry", str(REPO / "data/bathymetry/etopo_1deg.nc"),
        "--implicit-vertical-mixing", "--north-cap-lat", "80", "--precision", "fp64",
        "--output", str(out / "omip_setup"), "--depth-cap-file", CAP, "--face-wall-file", WALLS]
print("inputs:", WALLS, CAP, DZ, flush=True)
model, s, _, _, _ = T.capture_model(argv)

from legoesm.ocean.dynamics.barotropic_implicit_latlon_cgrid import _h_total_at_faces  # noqa: E402
from legoesm.ocean.dynamics.latlon_cgrid_operators import (  # noqa: E402
    face_wall_gates, min_cell_to_uface, min_cell_to_vface)
from legoesm.ocean.vertical import compute_layer_thickness  # noqa: E402

grid, zc, cfg = model.grid, model.z_coord, model.config
assert grid.face_nlev_u is not None
floor = float(cfg.min_water_column_m)
mask = s.land_mask.data
h = compute_layer_thickness(s.eta.data, s.H_bathy.data, zc, min_water_column_m=floor)
assert h.dtype == jnp.float64, h.dtype
H_u, H_v = (np.asarray(x) for x in _h_total_at_faces(
    h, jnp.asarray(floor, h.dtype), mask, grid))
hm = h * mask[..., None]
bare = grid._replace(face_nlev_u=None, face_nlev_v=None)
gu, gv = (np.asarray(g) for g in face_wall_gates(grid, h.shape[-1], h.dtype))
print(f"floor {floor} m; eta max |{float(jnp.abs(s.eta.data).max()):.3g}| m", flush=True)
for name, H, raw, gate, wet in (
        ("u", H_u, np.asarray(min_cell_to_uface(hm)), gu, np.asarray(s.u_mask.data) > 0),
        ("v", H_v, np.asarray(min_cell_to_vface(hm, bare)), gv, np.asarray(s.v_mask.data) > 0)):
    open_sum = (raw * gate).sum(-1)
    cut = (raw * (1 - gate) > 0).any(-1)
    bad = np.abs(H - open_sum) > 1e-9 * np.maximum(np.abs(H), 1e-30)
    for label, sel in (("cut", wet & cut), ("uncut", wet & ~cut)):
        d = np.abs(H - open_sum)[sel]
        print(f"{name}-faces {label}: {int(sel.sum())} wet; H != sum of gated layer thicknesses "
              f"(>1e-9 rel) at {int((bad & sel).sum())}; max |diff| {d.max() if d.size else 0.0:.6g} m",
              flush=True)
    thin = wet & cut & (open_sum < floor)
    print(f"{name}-faces cut with open column < floor: {int(thin.sum())} "
          f"(of which fully closed, open column 0: {int((thin & (open_sum == 0)).sum())}); "
          f"largest H/open-column ratio among 0 < open < floor: "
          f"{(H[thin & (open_sum > 0)] / open_sum[thin & (open_sum > 0)]).max() if (thin & (open_sum > 0)).any() else float('nan'):.4g}",
          flush=True)
print("PROBE_DONE", flush=True)
