#!/bin/bash
#SBATCH --job-name=tkelen
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/tkelen_%j.out
# The 2-4x viscosity gap at 15-29 m at hour 5 (hourly rerun): energy or length?
# NEMO restarts every 4 h (nn_stock=4; must be a multiple of nn_fsbc=4) give instantaneous en/avm_k/dissl at
# the same clock hour as our snapshots.
# PRE-REGISTERED: (a) e ours/NEMO ~1 and l_m ours/NEMO ~0.3-0.5 at 15-29 m ->
# length formulation/bounds; (b) e ratio ~0.3-0.5 with l_m ~1 -> TKE budget
# (transport, shear production, dissipation) -- then eshear_k/ediss_k hourly
# means name the term; (c) both ~1 -> the K_M-to-avm split is elsewhere
# (Prandtl or EVD, read avt_evd).
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
git -C $ROOT rev-parse --short HEAD | grep -q f5588864b || { echo "worktree HEAD moved"; exit 2; }
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_evdocc_d5
N=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_HOURLY
NOISE="cuInit|jax_plugins|plugin_module|_check_cuda|cuda_versions|Traceback|^\s*\^|Jax plugin|RuntimeWarning|nanmedian|avm_m = |pr_o = |pr_N = "
for PAIR in "12:snapshot_day0000.500" "24:snapshot_day0001" "36:snapshot_day0001.500" "48:snapshot_day0002"; do
  H=${PAIR%%:*}; S=$R/${PAIR#*:}.npz
  echo "##### NEMO restart step $H (UTC $H:00, instantaneous) vs ours $(basename $S)"
  $PY -u scripts/validate/ocean_fidelity/equatorial_tke_length_vs_nemo_restart.py --restart-glob "$N/ORCA1_000000${H}_restart_oce_*.nc" --snapshot $S --n-levels 20 2>&1 | grep -vE "$NOISE"
done
STEM=ORCA1_1h_20000101_20000102
for PAIR in "5:snapshot_day0000.250" "17:snapshot_day0000.750" "41:snapshot_day0001.750"; do
  REC=${PAIR%%:*}; S=$R/${PAIR#*:}.npz
  echo "##### Ri/K tables: NEMO rec $REC (hourly mean, now with vo) vs ours $(basename $S)"
  $PY -u scripts/validate/ocean_fidelity/equatorial_shear_ri_vs_nemo.py --snapshot $S --nemo-dir $N --nemo-stem $STEM --nemo-suffix _1h --rec $REC --n-levels 26 2>&1 | grep -vE "$NOISE"
done
echo "=== card exit: $? ==="
