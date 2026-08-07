#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100"
#SBATCH --job-name=les_tuneall
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
# 72 h = the Ginsburg maximum. MEASURED cost: 9331231 reached only 5.7 of 9
# schemes in 17h43 (~3.1 h/scheme), and _write_outputs runs ONCE after the
# last scheme -- so a timeout loses every hour spent. Size from the measured
# rate and take the partition max whenever the only artifact lands at the end.
#SBATCH --time=72:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/les_tuneall_%j.out
set +e
# ONE parameter set per scheme, fitted jointly to every usable LES case:
#   bomex  shallow cumulus (moist)      gabls1 stable BL (dry)
#   rico   trade cumulus  (moist)       cbl    dry convective BL
#                                       ekman  neutral shear
# DYCOMS is excluded: its LES applies a Stevens longwave the SCM cannot
# reproduce, and the LES itself does not hold the cloud deck.
#
# Scored variables and timestep are PER CASE (theta/qv for the cumulus decks,
# theta+u+v for GABLS1, theta for CBL, u+v for the neutral Ekman layer where
# theta is constant by construction).
WT=/burg/glab/users/pg2328/les_scm_wt
cd $WT || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"; echo "host: $(hostname)"; nvidia-smi -L
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=/burg-archive/glab/users/pg2328/legoESM/results
OUT=$R/scm_les_turbulence/all_cases
rm -rf $OUT

$PY scripts/run/run_scm_les_turbulence_tuning.py \
  --cases "bomex:$R/les_ref/bomex,rico:$R/les_ref/rico,gabls1:$R/les_ref/gabls1,cbl:$R/les_ref/cbl,ekman:$R/les_ref/ekman" \
  --outdir "$OUT" --nlev 48 --steps 8
rc=${PIPESTATUS[0]}
echo "tuning rc=$rc"

$PY scripts/plot/plot_scm_les_tuning_summary.py "$OUT"
echo "summary plot rc=$?"
for c in bomex rico gabls1 cbl ekman; do
  [ -f "$OUT/profiles_$c.npz" ] && $PY scripts/plot/plot_scm_les_turbulence.py "$OUT" \
      --out "$OUT/profiles_$c.png" 2>/dev/null && echo "  plotted $c"
done
echo "=== ranking ==="; cat "$OUT/ranking.csv" 2>/dev/null
echo "DONE $(date +%T)"
exit $rc
