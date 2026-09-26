#!/bin/bash
#SBATCH --job-name=tkearms
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/tkearms_%j.out
# Hour-12 split said: mixing LENGTH matches NEMO, TKE is 3x too high at 1-3 m
# and 0.15-0.45x at 6-18 m at every phase. Two questions: (1) does the
# nemo_z0 arm (surface value pinned at z=0 like NEMO) already fix the e
# profile? mxl4 / bsink arms as controls. (2) NEMO's own hourly budget terms
# (eshear_k, estrat_k, ediss_k) at recs 11 and 35 -> which term carries TKE to
# 6-18 m in the oracle.
# PRE-REGISTERED: z0 arm e(1 m)/NEMO < 1.5 and e(10 m)/NEMO > 0.7 -> pinning
# placement IS the profile lever and the earlier "null/worse" verdict was on
# the wrong metric; z0 arm still 3x/0.4x -> transport/production term.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
git -C $ROOT rev-parse --short HEAD | grep -q f595ecaac || { echo "worktree HEAD moved"; exit 2; }
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
N=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_HOURLY
NOISE="cuInit|jax_plugins|plugin_module|_check_cuda|cuda_versions|Traceback|^\s*\^|Jax plugin|RuntimeWarning|nanmedian|avm_m = |pr_o = |pr_N = "
for ARM in trp_z0_d5 trp_mxl4_d5 trp_bsink_d5; do for PAIR in "12:snapshot_day0000.500" "36:snapshot_day0001.500"; do
  H=${PAIR%%:*}; S=$R/$ARM/${PAIR#*:}.npz
  echo "##### ARM $ARM: NEMO restart step $H vs $(basename $S)"
  $PY -u scripts/validate/ocean_fidelity/equatorial_tke_length_vs_nemo_restart.py --restart-glob "$N/ORCA1_000000${H}_restart_oce_*.nc" --snapshot $S --n-levels 16 2>&1 | grep -vE "$NOISE"
done; done
echo "##### NEMO hourly TKE budget terms, box mean 220-240E |lat|<=2 (W/kg), recs 11 and 35 (+ 5, 17)"
$PY - <<PYEOF
import netCDF4 as nc, numpy as np
d = nc.Dataset("/ORCA1_1h_20000101_20000102_grid_W_1h.nc")
la = np.asarray(d.variables["nav_lat"][:], float); lo = np.asarray(d.variables["nav_lon"][:], float) % 360.0
box = (np.abs(la) <= 2.0) & (lo >= 220.0) & (lo < 240.0)
zw = np.asarray(d.variables["depthw"][:20], float)
def prof(v, r):
    x = np.ma.masked_invalid(d.variables[v][r, :20]); x = x[:, box]
    return np.ma.filled(x.mean(axis=1), np.nan)
for r in (5, 11, 17, 35):
    sh, st, di, evd, avm, avt = (prof(v, r) for v in ("eshear_k", "estrat_k", "ediss_k", "avt_evd", "avm", "avt"))
    print(f"--- rec {r} (UTC {r:02d}-{r+1:02d}, local {(r+15.3)%24:04.1f})")
    print("  k    z_w |  shear prod  buoy(-avt N2)   dissip  | resid(P+B-D) | avt_evd    avm      avt")
    for k in range(1, 18):
        print(f"{k+1:3d} {zw[k]:7.2f} | {sh[k]:10.2e} {st[k]:12.2e} {di[k]:9.2e} | {sh[k]+st[k]-di[k]:11.2e} | {evd[k]:8.2e} {avm[k]:8.2e} {avt[k]:8.2e}")
PYEOF
echo "=== card exit: $? ==="
