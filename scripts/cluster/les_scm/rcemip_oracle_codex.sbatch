#!/bin/bash
#SBATCH --job-name=rce_codex
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/gsam_rcemip/codex_%j.log
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
# Codex adversarial review of the RCEMIP gSAM-oracle change, read-only, from a
# PINNED worktree. CODEX_HOME is placed on node-local scratch (memory:
# mpas-tke-full-card trap — a shared CODEX_HOME on the network FS wedges).
set -uo pipefail

REPO=/burg-archive/glab/users/pg2328/legoESM
BRANCH="${BRANCH:-feat/rcemip-gsam-oracle-sounding}"
WT=/burg-archive/glab/users/pg2328/legoESM_wt/rce_codex_${SLURM_JOB_ID}

mkdir -p "$(dirname "$WT")" "$REPO/results/gsam_rcemip"
git -C "$REPO" worktree add --detach "$WT" "origin/$BRANCH" >/dev/null 2>&1
echo "PINNED sha $(git -C "$WT" rev-parse HEAD)  branch=$BRANCH"
trap 'git -C "$REPO" worktree remove --force "$WT" >/dev/null 2>&1 || true' EXIT

export CODEX_HOME=/tmp/codex_home_${SLURM_JOB_ID}
mkdir -p "$CODEX_HOME"
cp -a "$HOME/.codex/auth.json" "$CODEX_HOME/" 2>/dev/null || true
export PATH="$HOME/.nvm/versions/node/v24.16.0/bin:$PATH"

read -r -d '' PROMPT <<'EOF'
Adversarial review, NUMERICS AND INSTRUMENTS FIRST. Branch diff vs origin/main.
Run: git diff origin/main...HEAD --stat, then read the changed files.

Context: legoESM's RCEMIP1 plane-CRM initial condition was built from an
analytic Wing-2018 profile whose three case constants were wrong. gSAM 1.8.8's
own RCEMIP1 sounding (CASES/RCEMIP1/snd_rcemip_300s6.11.2) is now available as
an oracle. This change (a) adds --sounding to scripts/run/run_rcemip_plane.py
to initialise from the tabulated SAM sounding via the EXISTING shared
read_sam_snd + sam_case_setup path, (b) recalibrates WING_T_V0 295 -> 300.444,
WING_GAMMA 0.0067 -> 0.0074034, WING_Q_SFC_DEFAULT 0.01865 -> 0.0142014 against
that oracle, (c) adds tests/unit/test_rcemip_gsam_oracle_ic.py plus a tiny
vendored baseline tests/oracle_baselines/gsam_rcemip300_snd.json.

Attack these specifically:

1. UNIT / CONVENTION BUGS on the sounding path. The file column is POTENTIAL
   temperature [K] and MIXING RATIO [g/kg]; our state carries theta' relative
   to a reference and q_v in kg/kg. Trace read_sam_snd -> extend_sounding_to_top
   -> build_sam_case_height_coord -> build_sam_case_initial_state and check
   every conversion, including the level ORDERING (HeightCoordinate is
   TOP-DOWN, k=0 is the model top; the sounding is bottom-up) and the tracer
   slot. Is theta' = theta_snd - theta_ref actually near zero at init?

2. Is the RH computed in the tests on a CONSISTENT convention? The oracle q is
   a mixing ratio; the test divides by legoesm.thermo.saturation_mixing_ratio.
   Is that self-consistent, and is p recovered correctly from the
   HeightCoordinate (it stores exner_ref, and the test does
   p = p_ref * exner^(1/kappa))?

3. TEST VACUITY. tests/unit/test_rcemip_gsam_oracle_ic.py claims to fail for the
   pre-oracle constants. Would each assertion actually fail? Are any assertions
   trivially true (e.g. comparing a value to itself after a round trip through
   linear interpolation)? Which tests would still pass if the --sounding wiring
   were deleted entirely?

4. The CONSTANTS. Are the recorded estimators defensible? WING_Q_SFC_DEFAULT
   uses a point extrapolation of the 37 m level while a least-squares fit gives
   a 4% different answer. WING_T_V0/WING_GAMMA come from a least-squares fit of
   virtual T over z <= 15 km. Is anything double-counted or inconsistent
   (e.g. T_v0 fitted with one q_0 and then used with another)?

5. BLAST RADIUS. Other tests pinned numbers derived from the old constants
   (tests/validation/test_rcemip_plane_smoke.py,
   tests/atmosphere/hydrostatic/test_build_rcemip1_small_reference.py,
   tests/unit/test_rcemip_wing_ic.py,
   tests/unit/test_compute_reference_state_p_sfc_iter95.py). Were any anchors
   updated in a way that WEAKENS a test rather than re-measuring it? Is any
   remaining consumer of WING_* left inconsistent? Grep for other users.

6. Dispatch/failure modes: --sounding without --stretched-vertical raises; the
   p_sfc consistency check; the sounding-top coverage guard. Any silent
   coercion left?

Report findings as severity-tagged one-liners with file:line. Do not fix.
EOF

cd "$WT"
echo "=== CODEX ADVERSARIAL REVIEW ==="
codex exec --sandbox read-only -C "$WT" "$PROMPT" 2>&1
echo "=== CODEX rc=$? ==="
