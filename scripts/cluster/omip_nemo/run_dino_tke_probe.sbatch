#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=dino_tkeprobe
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/dino_tkeprobe_%j.out

# Characterize the r1_exact TKE blowup (ablation 8818038: NaN day ~30,
# only kpp survives): per-step finite check; on first hit dump the last
# finite state + the bad state, print argwhere of the first non-finite
# entries per field, and a daily max|u| / max-TKE growth trace before it.
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_worktrees/nemo-gaps-p4p5; cd $WT
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; nvidia-smi -L | head -1; date; git -C $WT log --oneline -1
OUT=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/dino_tke_probe
mkdir -p $OUT
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 LEGO_PROBE_OUT=$OUT $PY - <<'PYEOF' 2>&1 | grep -vE "FutureWarning|warnings.warn|XLA|cuda|CUDA|Plugin"
import os
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import (
    dino_r1_exact_config, create_dino_z_star, dino_lat_lon_grid,
    dino_lat_lon_vertical,
    dino_lat_lon_state, dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays, apply_dino_lat_lon_surface_forcing,
    dino_step_surface_forcing)

out = os.environ["LEGO_PROBE_OUT"]
cfg = dino_r1_exact_config()
g = dino_lat_lon_grid(cfg, n_lon=50); z = dino_lat_lon_vertical(g, cfg)
st = dino_lat_lon_state(g, z, cfg)
model = LatLonCGridOceanModel(g, z, dino_lat_lon_model_config(g, cfg, physics=True)[0])
frc = dino_lat_lon_surface_forcing_arrays(g, cfg)
sf = dino_step_surface_forcing(frc) if cfg.wind_through_step else None
lat_deg = np.degrees(np.asarray(g.lat))

FIELDS = ("u", "v", "T", "S", "eta")
def snap(s):
    d = {f: np.asarray(getattr(s, f).data) for f in FIELDS}
    for extra in ("tke", "h"):
        val = getattr(s, extra, None)
        if val is not None:
            d[extra] = np.asarray(getattr(val, "data", val))
    return d

dt = cfg.dt; n = int(round(40*86400/dt))
prev = snap(st)
for k in range(n):
    st = apply_dino_lat_lon_surface_forcing(st, frc, z, cfg, dt,
                                            t_seconds=(k+1)*dt)
    st = model.step(st, dt=dt, surface_forcing=sf)
    cur = snap(st)
    day = (k+1)*dt/86400.0
    bad = [f for f, a in cur.items() if not np.isfinite(a).all()]
    if bad:
        print(f"FIRST NON-FINITE at step {k+1} (day {day:.3f}) fields={bad}")
        for f in bad:
            a = cur[f]
            ij = np.argwhere(~np.isfinite(a))
            print(f"  {f}: {ij.shape[0]} bad; first 8 idx (lat,lon[,lev]):")
            for row in ij[:8]:
                loc = f"lat={lat_deg[min(row[0], len(lat_deg)-1)]:+.1f}"
                print(f"    {tuple(int(x) for x in row)}  {loc}")
        np.savez_compressed(f"{out}/state_last_finite.npz", **prev)
        np.savez_compressed(f"{out}/state_first_bad.npz", **cur)
        print("dumped state_last_finite.npz + state_first_bad.npz")
        break
    if (k+1) % 32 == 0:
        msg = f"day {day:6.1f}  max|u|={np.abs(cur['u']).max():.3f}  max|v|={np.abs(cur['v']).max():.3f}"
        msg += f"  Tmin={cur['T'].min():.2f} Tmax={cur['T'].max():.2f}"
        if "tke" in cur:
            msg += f"  tke_max={cur['tke'].max():.3e}"
        # location of current max|u|
        im = np.unravel_index(np.abs(cur['u']).argmax(), cur['u'].shape)
        msg += f"  argmax_u={tuple(int(x) for x in im)} lat={lat_deg[min(im[0],len(lat_deg)-1)]:+.1f}"
        print(msg)
    prev = cur
else:
    print("survived 40 d — blowup did not reproduce")
PYEOF
echo "=== DONE ==="; date
