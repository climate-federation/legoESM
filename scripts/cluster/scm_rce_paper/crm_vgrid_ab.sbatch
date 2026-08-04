#!/bin/bash
# CRM vertical-grid A/B — the ONE variable is the VERTICAL GRID.
#
# WHY THIS EXPERIMENT
# -------------------
# docs/science/specs/CRM_implementation.md F11 proposes "resolve convection
# explicitly: drop dx to 1 km or 256 m".  Every test of that fix path refined
# dx while leaving the vertical grid at the driver default:
#
#     nlev=30, H=33 km, UNIFORM  =>  dz = 1100 m at EVERY level, surface to top.
#
# So the "LES dx=500 m" runs had dz/dx = 2.2 — cells more than twice as tall as
# they are wide.  Refining dx alone makes the grid MORE anisotropic, not less;
# a convective plume was never resolved in the vertical at any dx tested, and
# the entire subcloud layer lived inside half of one grid cell.  RCEMIP (Wing
# 2018) specifies ~74 levels stretched from ~50 m at the surface — which this
# driver's own --stretched-vertical help text already documents as "RCEMIP1:
# nlev=74 ... dz_sfc=50m", and which no CRM run has used.
#
# ARMS — identical in EVERY other respect (domain, dx, dt, physics, seed,
# radiation cadence, sponge, hyperdiff, Smag, snapshot cadence):
#   V (control) : nlev=30, UNIFORM      -> dz = 1100 m everywhere
#   R (test)    : nlev=74, STRETCHED    -> dz = 50 m at the surface (RCEMIP)
#
# Both use the Wing 2018 Sect. 3.2.3 PROTOCOL seed at the protocol amplitude
# (--seed-kind wing2018 --theta-noise-amp 0.1) and NO convection
# parameterization — explicit Kessler microphysics only, so convection is
# resolved rather than parameterized.
#
# dt=10 s (not the 80-day run's 20 s) because arm R's 50 m surface layer has a
# vertical advective CFL w*dt/dz that dt=20 s would violate at modest w. dt is
# held IDENTICAL across both arms so it is not a confound BETWEEN them; it does
# differ from the older 128^2 run, which is therefore NOT a valid control here.
#
# Submit as:  ARM=V sbatch crm_vgrid_ab.sbatch   /   ARM=R sbatch ...
#SBATCH --account=glab
# burst has free GPU capacity; glab1 is non-preempt but was fully subscribed.
# Runs checkpoint every 21600 steps and accept RESTART=latest, so a preemption
# is recoverable.
#SBATCH --partition=burst
#SBATCH --gres=gpu:1
#SBATCH --job-name=crm_vgrid
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/crm_vgrid_%j.out
set +e

ARM="${ARM:?set ARM=V (uniform dz=1100m control) or ARM=R (RCEMIP stretched)}"
BRANCH="${BRANCH:-crm/rcemip-resolved-convection}"
WT="${WT:-/burg-archive/glab/users/pg2328/_wt_crmvgrid_$SLURM_JOB_ID}"
SRC=/burg-archive/glab/users/pg2328/legoESM

git -C "$SRC" worktree add --detach "$WT" "origin/$BRANCH" >/dev/null 2>&1
cd "$WT" || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"
echo "branch: origin/$BRANCH   ARM=$ARM"
echo "host: $(hostname)"; nvidia-smi -L

export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

# --- the ONLY difference between the arms -------------------------------
if [ "$ARM" = "V" ]; then
  VGRID=""                                        # uniform, dz = H/nlev
  NLEV=30
  OUT=/burg-archive/glab/users/pg2328/legoESM/results/crm_vgrid_V_uniform30
elif [ "$ARM" = "R" ]; then
  VGRID="--stretched-vertical --dz-sfc 50"        # RCEMIP1 standard
  NLEV=74
  OUT=/burg-archive/glab/users/pg2328/legoESM/results/crm_vgrid_R_stretch74
else
  echo "FATAL: ARM must be V or R, got '$ARM'"; exit 2
fi
mkdir -p "$OUT"
echo "ARM=$ARM  NLEV=$NLEV  VGRID='$VGRID'  OUT=$OUT"

# 30 sim-days at dt=10 s = 259200 steps.
JAX_ENABLE_X64=1 $PY $WT/scripts/run/run_rcemip_plane.py \
  --nx 96 --ny 96 --nlev $NLEV $VGRID --dx 2000 --dt 10.0 \
  --steps 259200 \
  --semi-implicit --n-acoustic-substeps 12 --off-centering 0.2 \
  --smag-cs 0.2 --hyperdiff 5e6 \
  --sponge-width 13200 --sponge-coeff 0.05 \
  --radiation rrtmgp --radiation-interval 180 \
  --microphysics kessler \
  --precision float64 \
  --seed-kind wing2018 --theta-noise-amp 0.1 --seed-rng 0 \
  --print-every 720 \
  --snapshot-days 0.5 \
  --snapshot3d-days "10,12,14,16,18,20,22,24,26,28,30" \
  --checkpoint-every 21600 \
  --restart "${RESTART:-}" \
  --output "$OUT" 2>&1
RC=$?
echo "run rc=$RC"

# --- artefact gate: exit code alone is NOT evidence ---------------------
LOG=/burg-archive/glab/users/pg2328/legoESM/logs/crm_vgrid_${SLURM_JOB_ID}.out
N_VOL=$(ls "$OUT"/snapshots3d/vol_*.npz 2>/dev/null | wc -l)
N_SFC=$(ls "$OUT"/snapshots/sfc_*.npz 2>/dev/null | wc -l)
echo "--- volumes: $N_VOL   surface snapshots: $N_SFC ---"

if grep -q "NON-FINITE\|EXIT 1" "$LOG" 2>/dev/null; then
  echo "CRM_STATUS=NONFINITE (arm $ARM went non-finite)"
fi

# Physics gate: read max(theta') from column 5 of the driver's per-step line.
# PROVEN NON-VACUOUS before use -- the awk below is checked against a synthetic
# known-bad line so it cannot silently report 0.0 and pass a 1282 K run (the
# failure mode that passed job 9285110 at theta'=372 K).
GATE_AWK='/^ *[0-9]+ +[0-9.]+ +/ {v=$5+0; if (v<0) v=-v; if (v>m) m=v} END {printf "%.3f", m+0}'
PROBE=$(printf ' 1 20.00 1.0e+00 -5.0e+00 1.282e+03 1.5e-02 0.0e+00\n' | awk "$GATE_AWK")
echo "GATE SELF-TEST: synthetic theta'=1282 K line parses as $PROBE (must be ~1282, NOT 0)"
case "$PROBE" in
  128[0-9].*) echo "GATE SELF-TEST OK";;
  *) echo "GATE SELF-TEST FAILED -- theta' gate is BROKEN, ignore its verdict";;
esac
THETA_MAX=$(awk "$GATE_AWK" "$LOG")
echo "max(theta') over the run: ${THETA_MAX} K (physical RCE stays well under 150)"

echo "ALLDONE arm=$ARM rc=$RC volumes=$N_VOL theta_max=$THETA_MAX"
