"""Channel-integrated zonal momentum budget: legoESM per-term probe at the
bridged NEMO day-180 state vs NEMO's own trddyn dump (kt 5760->5761).
Integral: sum(term * e1u*e2u*e3 * umask3d) over channel rows, ALL lon, depth."""
import dataclasses, numpy as np, netCDF4 as nc, jax.numpy as jnp
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
from legoesm.ocean.fidelity.tendency_probe import probe_latlon_cgrid
from legoesm.ocean.experiments.dino import (dino_config_for_recipe,
    dino_lat_lon_model_config, dino_lat_lon_surface_forcing_arrays,
    dino_step_surface_forcing)

RUN = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_STEPDUMP"
H = 2; CH = slice(12, 47)             # channel rows (open wrap), interior idx
g = read_nemo_mesh_mask(f"{RUN}/mesh_mask.nc", nn_hls=2)
s = read_nemo_restart(f"{RUN}/DINO_00005760_restart.nc", nn_hls=2)
br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
cfg = dataclasses.replace(dino_config_for_recipe("nemo_dino_kamm_mlf"),
    lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
sf = dino_step_surface_forcing(forcing)
pr = probe_latlon_cgrid(br.state, br.geometry, br.z_coord, mc,
                        surface_forcing=sf, dt=2700.0)

# --- weights on NEMO u-points (interior), channel rows ---
mm = nc.Dataset(f"{RUN}/mesh_mask.nc")
def llz(a): return np.moveaxis(np.asarray(a).squeeze(), 0, -1)
iy = ix = slice(H, -H)
umask = llz(mm["umask"][0])[iy, ix]                    # (195,48,36)
e1u = np.asarray(mm["e1u"][0]).squeeze()[iy, ix]
e2u = np.asarray(mm["e2u"][0]).squeeze()[iy, ix]
e3 = np.asarray(mm["e3t_1d"][:]).squeeze()
W = (e1u * e2u)[..., None] * e3[None, None, :] * umask  # (195,48,36) volumes
Wch = W[CH]

def nint(v, rst):
    a = llz(rst[v][0] if rst[v].ndim == 4 else rst[v][:])[iy, ix]
    return float(np.sum(a[CH] * Wch))
rst = nc.Dataset(f"{RUN}/DINO_00005761_restart.nc")
nemo = {k: nint(f"utrd_{k}", rst) for k in
        ("hpg","spg","keg","rvo","pvo","zad","ldf","zdf","tau","bfr","bfri","atf")}

# --- lego probe terms: u-faces (195,49,36) -> drop wrap dup face 48, map face i -> NEMO u-col i? 
# lego face i = WEST face of T-col i = NEMO u-point (i-1) in interior indexing.
# NEMO u-col j (interior 0..47) = east face of T-col j = lego face j+1.
def lint(a):
    a = np.asarray(a)[:, 1:49, :]                       # lego faces 1..48 <-> NEMO u-cols 0..47
    return float(np.sum(a[CH] * Wch))
lego = {
    "pgf+keg": lint(pr.pgf_ke_u),
    "cor(f)": lint(pr.coriolis_u),
    "vortcor": lint(pr.vortcor_u),
    "zad": lint(pr.vertadv_u),
    "ldf": lint(pr.ah_lap_u),
    "bfr": lint(pr.botdrag_u),
    "zdf_expl": lint(pr.av_vert_u),
    "phys": lint(pr.phys_u),
    "total": lint(pr.total_u),
}
# analytic wind input (implicit deposit not in probe): sum tau_x/rho0 * area over channel u-faces
_tx = np.asarray(sf.tau_x)
taux = _tx[:, 1:49] if _tx.shape[1] == 49 else _tx
area = (e1u * e2u)
lego["tau(analytic)"] = float(np.sum((taux[CH] / 1026.0) * area[CH] * umask[CH][:, :, 0]))

S = 1e9  # report in 1e9 m^4/s^2
print(f"{'term':14s} {'NEMO':>12s} {'legoESM':>12s}   (channel-integrated u-tendency, 1e9 m4/s2)")
print(f"{'hpg+keg+spg':14s} {(nemo['hpg']+nemo['keg']+nemo['spg'])/S:12.4f} {lego['pgf+keg']/S:12.4f}   [NEMO parts: hpg {nemo['hpg']/S:.4f} keg {nemo['keg']/S:.4f} spg {nemo['spg']/S:.4f}]")
print(f"{'rvo+pvo':14s} {(nemo['rvo']+nemo['pvo'])/S:12.4f} {(lego['cor(f)']+lego['vortcor'])/S:12.4f}   [NEMO rvo {nemo['rvo']/S:.4f} pvo {nemo['pvo']/S:.4f} | lego f {lego['cor(f)']/S:.4f} vort {lego['vortcor']/S:.4f}]")
print(f"{'zad':14s} {nemo['zad']/S:12.4f} {lego['zad']/S:12.4f}")
print(f"{'ldf':14s} {nemo['ldf']/S:12.4f} {lego['ldf']/S:12.4f}")
print(f"{'bfr(+i)':14s} {(nemo['bfr']+nemo['bfri'])/S:12.4f} {lego['bfr']/S:12.4f}")
print(f"{'zdf':14s} {nemo['zdf']/S:12.4f} {lego['zdf_expl']/S:12.4f}   (lego explicit-only; implicit friction not in probe)")
print(f"{'tau':14s} {nemo['tau']/S:12.4f} {lego['tau(analytic)']/S:12.4f}")
print(f"{'atf':14s} {nemo['atf']/S:12.4f} {'--':>12s}")
print(f"{'phys':14s} {'--':>12s} {lego['phys']/S:12.4f}")
