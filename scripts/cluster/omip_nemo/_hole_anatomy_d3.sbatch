#!/bin/bash
#SBATCH --job-name=hole
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/turbocl_%j.out
# Codex objection to the "solar deposited shallower" claim: equal DAILY-MEAN K
# does not give equal diffusive heat flux if the diurnal phasing differs.
# NEMO's mldkz5 (turbocline, avt < 5e-4) is a window mean of an HOURLY depth,
# so it encodes the phasing. Ours: turbocline from K_H_diag at four clock
# phases (00/06/12/18 UTC = 15/21/03/09 local at 230E) on the EVD arm's
# 6-hourly snapshots, days 1-4.75; the 4-phase mean is our daily mean.
# 2026-09-26 second pass: first pass searched from the SURFACE (16.0 m vs 35.7 m,
# not like-for-like); NEMO searches from nlb10 (>= 10 m). Rerun.
# PRE-REGISTERED: 4-phase mean shallower than NEMO's mldkz5 by > 3 m with the
# 15-local phase the shallow one -> daytime mixing is confined shallower than
# NEMO's (phasing defect, TKE under the diurnal layer); within 3 m -> mixing
# depth is matched at all phases and the solar-deposition hypothesis stands.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
N=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
for S in $(ls $R/trp_evdocc_d5/snapshot_day0003.500.npz $R/trp_mxl4_d5/snapshot_day0003.500.npz | sort); do
  echo "##### $(basename $S)"
  $PY -u scripts/validate/ocean_fidelity/equatorial_shear_ri_vs_nemo.py --snapshot $S --nemo-dir $N --rec 0 --n-levels 3 2>&1 | grep -E "turbocline|interfaces with|example column|above:|HOLE|below:|tke field|SystemExit|Error"
done
echo "=== card exit: $? ==="
