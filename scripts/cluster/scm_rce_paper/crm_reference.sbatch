#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --job-name=scmrce_crm
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/scmrce_crm_%j.out
set +e
WT=/burg/glab/users/pg2328/scmrce_wt
cd $WT
echo "PINNED sha: $(git rev-parse HEAD)"
echo "host: $(hostname)"; nvidia-smi -L
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
OUT=/burg-archive/glab/users/pg2328/legoESM/results/rcemip1_n128_ocean

# ---------------------------------------------------------------------------
# CRM reference for the SCM-RCE convection-scheme intercomparison.
#
# Recipe = the VALIDATED iter-229 production plane-CRM configuration
# (docs/science/specs/CRM_implementation.md:554-564: 132^2 x 30, dx=2 km,
# dt=20 s, van_leer TVD scalars + centered momentum, Smagorinsky c_s=0.2,
# hyperdiff 5e6, N_ACOUSTIC=12, off-centering beta=0.2, semi-implicit),
# at n=128 to match the campaign's reference-dir name.
#
# TWO deliberate deviations from iter-229, both stated so they are not
# mistaken for the validated config:
#   1. radiation = rrtmgp (iter-229 used gray).  REQUIRED: the SCM campaign
#      runs RRTMGP, and run_scm_rce_campaign.py:1934-1943 warns that a
#      gray CRM vs an RRTMGP SCM is a radiation confound.  This matches
#      the two sides.
#   2. radiation-interval = 90 steps = 1800 s, the SAM/WRF/CM1 recommended
#      maximum (the script WARNs above 1800 s; 150 steps = 3000 s would trip
#      that warning).
#
# theta' seed: REQUIRED.  A 1-day unseeded pilot (job 9248485) stayed
# laminar -- max|w| ~ 9e-4 m/s, i.e. no convection at all -- while theta'
# grew monotonically under radiative destabilisation.  The band-noise seed
# breaks the horizontal symmetry so convection can initiate.
#
# 207360 steps @ dt=20 s = 48.0 sim days.  Snapshots (3D volumes) at days
# 30..48 every 2 days = 10 volumes, so --last-reference-files 5 (days
# 40,42,44,46,48) has margin, and all are far past spin-up.
#
# WHY 48 DAYS AND NOT 80 (measured, not assumed).  Attempt 1 (job 9248515) ran
# this EXACT configuration and went non-finite at day 52.25, writing zero
# volumes because its first snapshot day was 62.  Its own log shows the run is
# quasi-stationary for 40 days -- max(theta') 55-80 K and max|w| 0.3-0.9 m/s
# from day 3 to day 40 -- and then transitions abruptly: theta' jumps 80 -> 400 K
# between days 39.9 and 43.2, reaching 640 K before NaN.  So the configuration
# is not inherently unstable; it has a long stable window followed by a
# late-time tip.  The reference is therefore taken from INSIDE the measured
# stable window rather than fighting the transition.
#
# Attempt 2 (job 9260006) tried --si-w-filter-nu 0.35, on the strength of that
# flag's help ("0.3-0.4 fully damps the structural exponential mode ...
# Recommended for RCE runs with theta_noise_amp > 0").  It made things far
# WORSE: NaN at step 360 (2 sim-hours).  The 6-arm pilot matrix (job 9262160,
# 1 sim-day per arm, one variable per arm) isolated the cause:
#     filter 0.00 / noise 0.5  -> survives      (= this config)
#     filter 0.35 / noise 0.2  -> NON-FINITE
#     filter 0.35 / noise 0.0  -> NON-FINITE    <- filter alone, no noise
#     filter 0.00 / noise 0.2  -> survives
#     filter 0.00 / noise 0.1  -> survives
#     filter 0.15 / noise 0.2  -> survives
# The w-filter at 0.35 is itself the destabiliser at this resolution (nlev=30,
# dz ~ 1100 m, 12 acoustic substeps) regardless of the noise amplitude; the
# documented 0.3-0.4 band evidently was not calibrated for this setup.  The
# filter stays OFF.  Note also that a 1-day pilot CANNOT see the day-40 tip --
# arm A survives 1 day and still failed at day 52 in the long run.
# ---------------------------------------------------------------------------
JAX_ENABLE_X64=1 $PY $WT/scripts/run/run_rcemip_plane.py \
  --nx 128 --ny 128 --nlev 30 --dx 2000 --dt 20.0 \
  --steps 207360 \
  --semi-implicit --n-acoustic-substeps 12 --off-centering 0.2 \
  --smag-cs 0.2 --hyperdiff 5e6 \
  --radiation rrtmgp --radiation-interval 90 \
  --microphysics kessler \
  --precision float64 \
  --theta-noise-amp 0.5 --seed-kind band_noise --seed-kmax 6 \
  --print-every 360 \
  --snapshot-days 0.5 \
  --snapshot3d-days "30,32,34,36,38,40,42,44,46,48" \
  --checkpoint-every 21600 \
  --restart "${RESTART:-}" \
  --output $OUT 2>&1
rc=$?
echo "run rc=$rc"
echo "--- vol files ---"; ls -la $OUT/snapshots3d/ 2>&1 | tail -12
echo "--- sfc files (count) ---"; ls $OUT/snapshots/ 2>&1 | wc -l

# FAIL LOUD. Run 9248515 NaN'd at day 52.25 and STILL exited rc=0, so SLURM
# reported COMPLETED/0:0 with zero volumes written — the downstream campaign
# chain would have consumed a reference that does not exist. The runner's own
# exit code is not trustworthy here, so gate on the ARTEFACT.
N_VOL=$(ls -1 $OUT/snapshots3d/vol_*.npz 2>/dev/null | wc -l)
echo "--- vol count: $N_VOL ---"
if grep -qiE "NON-FINITE|nan" "$SLURM_SUBMIT_DIR/logs/scmrce_crm_${SLURM_JOB_ID}.out" 2>/dev/null; then
  echo "CRM_STATUS=NONFINITE_DETECTED"
fi
if [ "$N_VOL" -lt 5 ]; then
  echo "CRM_STATUS=FAILED (need >=5 vol_*.npz, have $N_VOL)"
  echo ALLDONE
  exit 1
fi
echo "CRM_STATUS=OK ($N_VOL volumes)"
echo ALLDONE
