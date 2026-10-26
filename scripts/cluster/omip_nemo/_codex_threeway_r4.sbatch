#!/bin/bash
#SBATCH --job-name=codex_3way4
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/codex_3way_r4_%j.out

# Round 4 of the adversarial loop on the three-way scorer, verifying the
# round-3 fixes (commit aef732023).  Round 3 left: radius validation RED,
# zonal-gate tautology RED, MLD-tails-under-smoothing RED (new), near_land
# labeling RED, main-mask test gap YELLOW, _demean orphan LOW.
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json  "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "=== CODEX_HOME=$CODEX_HOME contents: $(ls -A $CODEX_HOME) ==="
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="

codex exec --sandbox read-only -C $WT \
  "Round 4 review of scripts/validate/ocean_fidelity/compare_three_way_nemo.py
and tests/validate/test_compare_three_way_nemo.py, verifying the round-3 fix
commit aef732023 (git show aef732023 to see it).  You may read
scripts/validate/compare_omip_nemo.py for imported helpers.

Round 3 findings to verify.  For EACH: fixed, only appears fixed, or traded
for a new defect?  Cite line numbers.

R3-1 (RED) --smooth-radius-deg had no validation; 270 deg silently became a
  90-deg chord radius.  Claimed fix: hard (0,180] validation with an error
  message.
R3-2 (RED) the zonal 25% support gate was a tautology (availability derived
  from the same post-dropout mask it gated).  Claimed fix: availability from
  the pre-dropout common ocean mask + an absolute floor
  _MIN_ZONAL_CELLS_FLOOR=5.  Check the floor and fraction are actually applied
  to what gets PLOTTED and REPORTED, not just computed.
R3-3 (RED, new in r3) smoothing corrupted MLD tail statistics (deep columns
  smeared before the tail count).  Claimed fix: tails SUPPRESSED under any
  smoothing, with report flag mld_tails_suppressed_under_smoothing.  Confirm
  the suppression covers every tail consumer, including plots and JSON.
R3-4 (RED) near_land* metrics from nearest-centre masks called a 'real
  land/sea classification'.  Claimed fix: relabelled SCREENING figures with the
  strait/barrier caveat.  Judge whether the wording now matches what the code
  computes, everywhere it is surfaced.
R3-5 (YELLOW) tests never exercise main()'s mask selection; reverting
  build_ocean_mask to coverage would pass all tests.  Was a test added that
  FAILS on that revert?
R3-6 (LOW) _demean orphaned after SSH removal.  Removed or justified?

Also: 19 tests pass at HEAD -- name any test that asserts a behaviour the
current code does NOT have (stale-assertion sweep), and any NEW defect in the
round-3 diff itself.  End with a table: finding, verdict (FIXED / PARTIAL /
NOT FIXED / NEW), severity."
rc=$?
echo "=== CODEX RC=$rc ==="
exit $rc
