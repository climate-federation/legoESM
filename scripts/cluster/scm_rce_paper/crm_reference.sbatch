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
WT=${WT:-/burg/glab/users/pg2328/scmrce_wt}
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
# 345600 steps @ dt=20 s = 80.0 sim days.  3D volumes at days 60..80 every
# 2 days = 11 volumes; --last-reference-files 5 takes days 72..80.
#
# ROOT CAUSE OF THE EARLIER BLOW-UPS: THE SPONGE WAS TOO SHALLOW.
#
# Attempts 1 (job 9248515) and 3 (job 9276192) both ran quasi-stationary for
# ~40 days (theta'max 55-81 K, max|w| 0.3-0.9 m/s) and then tipped abruptly:
# attempt 3 went theta'max 81 -> 141 K (day 41.9) -> 387 K with max|w| 3.13
# (day 42.7); attempt 1 did the same between days 39.9 and 43.2 and reached
# NaN at 52.25.
#
# Comparing a CLEAN volume (day 40) against a POST-TIP one (day 44) localises
# it unambiguously to the MODEL TOP:
#     level      z        mean T clean -> post-tip
#     k=0     32.5 km      202.0 K  ->  253.9 K      (+52 K)
#     k=15    15.9 km      228.3 K  ->  228.2 K      (unchanged)
#     k=21     9.4 km      273.5 K  ->  273.9 K      (unchanged)
#     k=27     2.7 km      289.9 K  ->  289.6 K      (unchanged)
# and max|w| migrates to k=1 (31.4 km, 1.30 m/s).  The TROPOSPHERE is
# untouched to ~0.3 K.  This is gravity waves radiating up from convection,
# reflecting off the rigid lid, and accumulating with nothing to absorb them.
#
# The sponge exists and was ON, but at the driver default --sponge-width 5000
# = 5 km = 0.15*H, i.e. ~4.5 levels at dz ~ 1100 m.  compressible_euler_plane.py
# (the sponge implementation) states the intended pairing in its own comment:
#     "SAM damping.f90 damps the VERTICAL velocity ONLY ... With
#      sponge_w_only=True legoESM matches that; ... The CRM run-scripts set
#      True + sponge_width=0.4*H (SAM nub=0.6, top 40%)."
# The driver hardcodes sponge_w_only=True (run_rcemip_plane.py:1391) but the
# width default is 0.15*H, NOT the documented 0.4*H.  A w-only sponge relies on
# a DEEP absorbing layer; at 0.15*H it cannot absorb the upward wave flux, and
# because it damps only w it never removes the thermal anomaly the reflected
# waves deposit -- which is exactly the observed failure (a THERMAL runaway at
# the lid, k=0 warming 52 K, while w is the only damped field).
#
# FIX: --sponge-width 13200 = 0.4*33000 = 0.4*H, per that comment and SAM's
# nub=0.6.  Restores the full 80-day run and the original days 60-80 sampling.
#
# Attempt 2 (job 9260006) tried --si-w-filter-nu 0.35 instead, on the strength
# of that flag's help.  It made things far WORSE -- NaN at step 360 (2 sim-h).
# A 6-arm pilot matrix (job 9262160, 1 sim-day per arm, ONE variable per arm)
# isolated the cause:
#     filter 0.00 / noise 0.5  -> survives      (= this config)
#     filter 0.35 / noise 0.2  -> NON-FINITE
#     filter 0.35 / noise 0.0  -> NON-FINITE    <- filter alone, zero noise
#     filter 0.00 / noise 0.2  -> survives
#     filter 0.00 / noise 0.1  -> survives
#     filter 0.15 / noise 0.2  -> survives
# The w-filter at 0.35 is ITSELF the destabiliser at this resolution; it stays
# OFF.  Note a 1-day pilot CANNOT see the day-40 tip -- arm A survives 1 day
# and still tipped in the long run.  Short pilots falsify fast instabilities
# only.
# ---------------------------------------------------------------------------
JAX_ENABLE_X64=1 $PY $WT/scripts/run/run_rcemip_plane.py \
  --nx 128 --ny 128 --nlev 30 --dx 2000 --dt 20.0 \
  --steps 345600 \
  --semi-implicit --n-acoustic-substeps 12 --off-centering 0.2 \
  --smag-cs 0.2 --hyperdiff 5e6 \
  --sponge-width 13200 --sponge-coeff 0.05 \
  --radiation rrtmgp --radiation-interval 90 \
  --microphysics kessler \
  --precision float64 \
  --theta-noise-amp 0.5 --seed-kind band_noise --seed-kmax 6 \
  --print-every 360 \
  --snapshot-days 0.5 \
  --snapshot3d-days "60,62,64,66,68,70,72,74,76,78,80" \
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
# PHYSICS SANITY, not just file count.  Attempt 3 passed the count+NaN gate
# with "OK (10 volumes)" while FOUR of its ten volumes were post-tip garbage
# (theta'max ~390 K vs a 55-81 K stable window) -- and --last-reference-files 5
# takes the LAST five, so four of the five would have been unusable.  Counting
# files and grepping for NaN says nothing about whether the fields are
# physical.  Gate on the diagnostic that actually revealed the tip.
THETA_TIP_K=150          # stable window is 55-81 K; the tip goes 141 -> 387 K
LOG="$SLURM_SUBMIT_DIR/logs/scmrce_crm_${SLURM_JOB_ID}.out"
THETA_MAX=$(grep -E "^ *[0-9]+ +[0-9.]+e?[+-]?[0-9]* " "$LOG" 2>/dev/null \
            | awk '{if ($5+0 > m) m=$5+0} END {printf "%.1f", m}')
echo "--- max(theta prime) over the whole run: ${THETA_MAX} K (tip threshold ${THETA_TIP_K}) ---"
if [ -n "$THETA_MAX" ] && awk -v a="$THETA_MAX" -v b="$THETA_TIP_K" 'BEGIN{exit !(a>b)}'; then
  echo "CRM_STATUS=TIPPED (max theta'=${THETA_MAX} K > ${THETA_TIP_K} K)"
  echo "  The run left the stable regime. Volumes written after the tip are"
  echo "  NOT physical RCE and must not be used as reference. Inspect the"
  echo "  theta' trajectory and quarantine post-tip volumes before use."
  echo ALLDONE
  exit 1
fi
echo "CRM_STATUS=OK ($N_VOL volumes, max theta'=${THETA_MAX} K)"
echo ALLDONE
