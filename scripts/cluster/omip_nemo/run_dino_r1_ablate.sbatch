#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=dino_ablate
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/dino_ablate_%j.out

# r1_exact blew up < day 30 (job 8816917).  Ablate each NEW lever over a
# 60-day window to isolate the destabilizer; also the full preset with
# vmix=kpp (the known-stable closure) as the comparison-run candidate.
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_worktrees/nemo-gaps-p4p5; cd $WT
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; nvidia-smi -L | head -1; date; git -C $WT log --oneline -1

JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY - <<'PYEOF' 2>&1 | grep -vE "FutureWarning|warnings.warn|XLA|cuda|CUDA|Plugin"
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import (
    dino_r1_exact_config, create_dino_z_star, dino_lat_lon_grid,
    dino_lat_lon_vertical,
    dino_lat_lon_state, dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays, apply_dino_lat_lon_surface_forcing,
    dino_step_surface_forcing)

CASES = [
    ("full_r1_exact", {}),
    ("no_wind_through_step", {"wind_through_step": False}),
    ("no_seasonal", {"forcing_annual_cycle": False}),
    ("tvd_advection", {"tracer_advection": "tvd"}),
    ("kpp_vmix", {"vmix_scheme": "kpp"}),
    ("wright_eos", {"eos": "wright"}),
    ("stab_floor_back", {"tke_momentum_visc_bg": 5.0e-4}),
]

for name, over in CASES:
    cfg = dino_r1_exact_config(**over)
    g = dino_lat_lon_grid(cfg, n_lon=50); z = dino_lat_lon_vertical(g, cfg)
    st = dino_lat_lon_state(g, z, cfg)
    model = LatLonCGridOceanModel(g, z, dino_lat_lon_model_config(g, cfg, physics=True)[0])
    frc = dino_lat_lon_surface_forcing_arrays(g, cfg)
    sf = dino_step_surface_forcing(frc) if cfg.wind_through_step else None
    dt = cfg.dt; n = int(round(60*86400/dt)); first_bad = None
    for k in range(n):
        st = apply_dino_lat_lon_surface_forcing(st, frc, z, cfg, dt,
                                                t_seconds=(k+1)*dt)
        st = model.step(st, dt=dt, surface_forcing=sf)
        if (k+1) % 32 == 0:   # daily check
            if not bool(np.isfinite(np.asarray(st.T.data)).all() and
                        np.isfinite(np.asarray(st.u.data)).all()):
                first_bad = (k+1)*dt/86400.0
                break
    umax = float(np.nanmax(np.abs(np.asarray(st.u.data))))
    print(f"CASE {name:22s} -> {'NaN day %.1f' % first_bad if first_bad else 'survived 60d'}  max|u|={umax:.2f}")
PYEOF
echo "=== DONE ==="; date
