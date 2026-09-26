#!/bin/bash
#SBATCH --job-name=nemo1h
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nemo1h_%j.out
# The oracle's OWN first two days at hourly resolution (RUN_HOURLY: same IC,
# same forcing, avt/avm/bn2/to/so/mldkz5 hourly means) against our EVD arm's
# 6-hourly snapshots at matched UTC hours. NEMO rec h = mean over hour h..h+1,
# so rec 5 pairs with our 06 UTC snapshot (half an hour early, noted).
# PRE-REGISTERED: (a) NEMO's turbocline is already >= 30 m in hour 1-6 -> the
# 35.7 m layer is set by the IC/first-hours physics, and our 19 m from hour 6
# is a fast-response defect (TKE/length in the first day); (b) NEMO's is ~20 m
# early and deepens over day 1-2 -> a slow process (shear build-up), and our
# day-30 12.6 m says we never deepen. Hour-by-hour avt profiles decide which
# level the 10-40 m mixing switches on in NEMO and never does in ours.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
git -C $ROOT rev-parse --short HEAD | grep -q e28db8698 || { echo "worktree HEAD moved"; exit 2; }
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_evdocc_d5
N=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_HOURLY
STEM=ORCA1_1h_20000101_20000102
echo "### NEMO hourly mldkz5 box mean 220-240E |lat|<=2 (unweighted), 48 hours"
$PY - <<PYEOF
import netCDF4 as nc, numpy as np
d = nc.Dataset("$N/${STEM}_grid_T_1h.nc")
la = d.variables["nav_lat"][:]; lo = d.variables["nav_lon"][:] % 360.0
box = (np.abs(la) <= 2.0) & (lo >= 220.0) & (lo <= 240.0)
mk = d.variables["mldkz5"]; m10 = d.variables["mldr10_1"] if "mldr10_1" in d.variables else None
for h in range(mk.shape[0]):
    x = np.ma.masked_invalid(mk[h])[box]
    s = f"hour {h:2d} (UTC {h:02d}-{h+1:02d}, local {(h+15.3)%24:04.1f}) mldkz5 {float(x.mean()):6.2f} m"
    if m10 is not None: s += f"  mldr10_1 {float(np.ma.masked_invalid(m10[h])[box].mean()):6.2f} m"
    print(s)
PYEOF
for PAIR in "0:snapshot_day0000.250" "1:snapshot_day0000.250" "2:snapshot_day0000.250" "5:snapshot_day0000.250" "11:snapshot_day0000.500" "17:snapshot_day0000.750" "23:snapshot_day0001" "29:snapshot_day0001.250" "35:snapshot_day0001.500" "41:snapshot_day0001.750" "47:snapshot_day0002"; do
  REC=${PAIR%%:*}; S=$R/${PAIR#*:}.npz
  echo "##### NEMO rec $REC vs ours $(basename $S)  (recs 0-2: ours is NOT matched, read the NEMO column only)"
  if [ "$REC" = "5" ]; then FILT="."; else FILT="turbocline|interfaces with|example column|above:|HOLE|below:|tke field|unst|SystemExit|Error"; fi
  $PY -u scripts/validate/ocean_fidelity/equatorial_shear_ri_vs_nemo.py --snapshot $S --nemo-dir $N --nemo-stem $STEM --nemo-suffix _1h --rec $REC --n-levels 26 2>&1 | grep -E "$FILT"
done
echo "=== card exit: $? ==="
