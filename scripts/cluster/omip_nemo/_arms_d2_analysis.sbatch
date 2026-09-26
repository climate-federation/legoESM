#!/bin/bash
#SBATCH --job-name=armsd2
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/armsd2_%j.out
# Per 2-day arm (ARM env: trp_ctl_d2 / trp_nolc_d2 / trp_literal_d2), 2-hourly
# snapshots: (1) TKE/length vs NEMO restarts at hours 4/8/12/16/24/36/48
# (NEMO LC run; for trp_nolc also vs RUN_HOURLY_NOLC); (2) T/S profile change
# vs NEMO hourly means at hours 2/4/6/12/24 (first-hours divergence);
# (3) turbocline at all snapshot hours (Ri probe, turbocline line only).
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
git -C $ROOT rev-parse --short HEAD | grep -q 035ad86bd || { echo "worktree HEAD moved"; exit 2; }
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
ARM=${ARM:?set ARM}
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/$ARM
B=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00
NOISE="cuInit|jax_plugins|plugin_module|_check_cuda|cuda_versions|Traceback|^\s*\^|Jax plugin|RuntimeWarning|nanmedian|avm_m = |pr_o = |pr_N = "
snap() { local h=$1; local d=$(python3 -c "print(f'{$h/24:.3f}'.rstrip('0').rstrip('.'))"); ls $R/snapshot_day000${d}.npz $R/snapshot_day00$(printf '%05.3f' $(python3 -c "print($h/24)")).npz 2>/dev/null | head -1; }
ls $R | grep -c snapshot_day | sed 's/^/snapshots: /'
NEMODIRS="RUN_HOURLY"; [ "$ARM" = trp_nolc_d2 ] && NEMODIRS="RUN_HOURLY RUN_HOURLY_NOLC"
for D in $NEMODIRS; do for H in 04 08 12 16 24 36 48; do
  S=$(ls $R/snapshot_day*.npz | awk -v h=$H '{ if (match($0,/day([0-9.]+)\.npz/,m) && sprintf("%.3f", m[1]*24)+0 == h+0) print }' | head -1)
  [ -z "$S" ] && { echo "##### $D hour $H: no snapshot"; continue; }
  echo "##### $ARM vs $D restart step $H (ours $(basename $S))"
  $PY -u scripts/validate/ocean_fidelity/equatorial_tke_length_vs_nemo_restart.py --restart-glob "$B/$D/ORCA1_000000${H}_restart_oce_*.nc" --snapshot $S --n-levels 16 2>&1 | grep -vE "$NOISE"
done; done
echo "##### profile change vs NEMO hourly (first hours)"
PAIRS=""; for H in 2 4 6 12 24; do S=$(ls $R/snapshot_day*.npz | awk -v h=$H '{ if (match($0,/day([0-9.]+)\.npz/,m) && sprintf("%.3f", m[1]*24)+0 == h+0) print }' | head -1); [ -n "$S" ] && PAIRS="$PAIRS $((H-1)):$S"; done
$PY -u scripts/validate/ocean_fidelity/equatorial_box_profile_tendency_vs_nemo.py --nemo-dir $B/RUN_HOURLY --n-levels 14 --pairs $PAIRS 2>&1 | grep -vE "$NOISE"
echo "##### turbocline per snapshot (ours instantaneous vs NEMO hourly mean, rec = hour-1)"
for S in $(ls $R/snapshot_day*.npz | grep -v final); do
  H=$(python3 -c "import re,sys; m=re.search(r'day([0-9.]+)\.npz', '$S'); print(int(round(float(m.group(1))*24)))"); REC=$((H-1)); [ $REC -lt 0 ] && REC=0; [ $REC -gt 47 ] && REC=47
  echo -n "hour $H: "; $PY -u scripts/validate/ocean_fidelity/equatorial_shear_ri_vs_nemo.py --snapshot $S --nemo-dir $B/RUN_HOURLY --nemo-stem ORCA1_1h_20000101_20000102 --nemo-suffix _1h --rec $REC --n-levels 26 2>&1 | grep "^\[turbocline" | sed 's/(reduction caveat)//'
done
echo "=== card exit: $? ==="
