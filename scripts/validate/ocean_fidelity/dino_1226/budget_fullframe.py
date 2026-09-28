"""Pointwise verdict on the two big channel-budget terms at the matched state."""
import dataclasses, numpy as np, netCDF4 as nc
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
from legoesm.ocean.fidelity.tendency_probe import probe_latlon_cgrid
from legoesm.ocean.experiments.dino import (dino_config_for_recipe,
    dino_lat_lon_model_config, dino_lat_lon_surface_forcing_arrays,
    dino_step_surface_forcing)
RUN = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_STEPDUMP"
H = 2; CH = slice(12, 47)
g = read_nemo_mesh_mask(f"{RUN}/mesh_mask.nc", nn_hls=0)
s = read_nemo_restart(f"{RUN}/DINO_00005760_restart.nc", nn_hls=0)
br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
cfg = dataclasses.replace(dino_config_for_recipe("nemo_dino_kamm_mlf"),
    lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
sf = dino_step_surface_forcing(dino_lat_lon_surface_forcing_arrays(br.geometry, cfg))
pr = probe_latlon_cgrid(
    br.state, br.geometry, br.z_coord, mc, surface_forcing=sf,
    dt=2700.0, tke_rn_dt=2700.0)
mm = nc.Dataset(f"{RUN}/mesh_mask.nc"); rst = nc.Dataset(f"{RUN}/DINO_00005761_restart.nc")
def llz(a): return np.moveaxis(np.asarray(a).squeeze(), 0, -1)
iy = ix = slice(None)
um = llz(mm["umask"][0])[iy, ix] > 0.5
e1u = np.asarray(mm["e1u"][0]).squeeze()[iy, ix]; e2u = np.asarray(mm["e2u"][0]).squeeze()[iy, ix]
e3 = np.asarray(mm["e3t_1d"][:]).squeeze()
W = (e1u * e2u)[..., None] * e3[None, None, :] * um
def nf(v): return llz(rst[v][0])[iy, ix]
# wall-adjacency on u-points: any of 4 lateral u-neighbours dry at that level
pad = np.pad(um, ((1,1),(1,1),(0,0)), constant_values=False)
wall = um & ~(pad[2:,1:-1]&pad[:-2,1:-1]&pad[1:-1,2:]&pad[1:-1,:-2])
def verdict(name, nfield, lfield_faces, shift=1):
    L = np.asarray(lfield_faces)[:, 1:53, :]   # face i+1 = east face of col i (52 cols)
    d = {}
    for tag, m in (("all", um), ("wall", wall), ("interior", um & ~wall)):
        mk = np.zeros_like(um); mk[CH] = m[CH]
        d[tag] = (float(np.sum(np.where(mk, nfield, 0)*W)), float(np.sum(np.where(mk, L, 0)*W)))
    c = np.corrcoef(nfield[CH][um[CH]], L[CH][um[CH]])[0,1]
    print(f"{name}: corr={c:.4f}")
    for tag in ("all","wall","interior"):
        n_, l_ = d[tag]
        print(f"   {tag:8s} NEMO {n_/1e9:+8.4f}  lego {l_/1e9:+8.4f}  diff {(l_-n_)/1e9:+8.4f}")
    return d
vN = nf("utrd_rvo") + nf("utrd_pvo")
verdict("vort (f+zeta)", vN, pr.vortcor_u)
pN = nf("utrd_hpg") + nf("utrd_keg")
verdict("pgf+keg", pN, pr.pgf_ke_u)
z = nf("utrd_zdf")
print(f"zdf diag: max|zdf|={np.abs(z).max():.3e} m/s2; surface tau/rho0/dz1={0.2/(1026*10):.3e}; "
      f"zdf surf-layer integral {np.sum(np.where(um[...,0], z[...,0], 0)[CH]*W[CH,:,0])/1e9:+.4f}")
