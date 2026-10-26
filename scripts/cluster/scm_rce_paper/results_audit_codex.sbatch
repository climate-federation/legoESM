#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=res_audit
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/res_audit_%j.out
set +e
WT=${WT:?}; PIN=${PIN:?}
for _CH in "${SLURM_TMPDIR:-}" "/tmp" "/burg-archive/glab/users/pg2328/_codexhome"; do
  [ -n "$_CH" ] || continue
  CANDIDATE="$_CH/codex_home_${SLURM_JOB_ID:-$$}"
  mkdir -p "$CANDIDATE" 2>/dev/null || continue
  cp /burg-archive/home/pg2328/.codex/auth.json "$CANDIDATE/" 2>/dev/null || continue
  [ -f "$CANDIDATE/auth.json" ] || continue
  export CODEX_HOME="$CANDIDATE"; break
done
[ -n "${CODEX_HOME:-}" ] || { echo "FATAL: no usable CODEX_HOME"; exit 90; }
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version || exit 91
cd $WT || exit 92
[ "$(git rev-parse HEAD)" = "$PIN" ] || { echo "PREFLIGHT FAIL"; exit 93; }

codex exec --sandbox read-only -C /burg-archive/glab/users/pg2328/legoESM "$(cat <<'PROMPT'
SAFETY REVIEW of a proposed DELETION list. This is irreversible disk cleanup of
simulation output; your job is to STOP me deleting anything that is still
load-bearing, not to help me free space.

CONTEXT: results/ is 132 GB across 130 directories dating to 2026-05. Nothing
in results/ is tracked by git (it is gitignored runtime output), so DELETION IS
PERMANENT - there is no recovery from version control.

PROPOSED DELETION CANDIDATES (my classification, which you should challenge):

A. SUPERSEDED-BY-CORRECTION
   results/scm_rce_tqtune/arm_thermo_physical_seed2026081[56]/  (~small)
   The SCM-RCE convection tuning campaign run under a boundary condition later
   found WRONG (a rotating/geostrophically-pinned surface wind; RCEMIP is
   nonrotating). Superseded by arm_thermo_physical_f0_seed*. CAUTION: these
   arms are the EVIDENCE for the published claim "the old ranking was a
   protocol artifact" - deleting them may destroy the only record of the
   before state.

B. KNOWN-BROKEN REFERENCE
   results/rcemip1_n128_ocean/  (970 MB)
   Our own CRM run that was once used as its own reference (a self-reference
   defect). Superseded by the external results/rcemip_ref_sam300.

C. ITERATION CHAINS where only the last matters
   results/atmosphere_swfix{,_final,_final2,_ship,_consist,_modontune,_specdealias}/
   (~4 GB total, 7 directories, all 2026-07-18)
   results/aimip_t106_headtohead_lr{1,2,3,4}/ (a learning-rate sweep)
   results/aimip_t106_headtohead_{column_nn,sfno_physics}_s{0,1,2,3}/ (seeds)

D. SMOKE / SCRATCH
   results/_smoke_dino_*/  results/*_smoke/  results/wb_campaign_smoke/
   results/les_smoke/  results/aimip_ace2loss_smoke/

E. LARGE, OLD, POSSIBLY SUPERSEDED (I am LEAST sure about these)
   results/aimip_001/ (8.7 GB, 2026-05)  results/aimip_wbcompare/ (6.7 GB)
   results/aimip_forcing/ (4.7 GB)  results/aimip_latlon_t106/ (4.0 GB)
   results/amip_mpas_100yr/ (1.3 GB)  results/aimip_amip_finetune/ (3.6 GB)
   results/omip_nemo/ (84 GB - by far the largest; ACTIVE work as of 2026-08-21)

WHAT I NEED FROM YOU, with file:line evidence:
1. For EACH candidate, is it referenced by anything that still matters -
   docs/**, *.md at repo root, scripts/ (a default --outdir, a plotting input,
   a validator's reference path), tests/, .claude/, or a config? grep the repo.
   A directory that is some script's DEFAULT input is not deletable.
2. Which candidates are FORCING/INPUT data rather than output? results/
   aimip_forcing/ smells like an input bundle that would have to be
   re-downloaded (expensive, possibly network-capped). Same question for
   les_ref/ and rcemip_ref_sam300/ (I am NOT proposing to delete those two).
3. Is anything an IRREPRODUCIBLE artifact - produced by a code path that no
   longer exists, or by a run whose config cannot be reconstructed from a
   committed manifest? Check for run_manifest/command_line files inside them.
4. Does anything back a claim in docs/ or a MERGED PR? Deleting the evidence
   for a published number is worse than keeping 84 GB.
5. Name any directory NOT on my list that is clearly safe to delete, and any
   ON my list that must be KEPT.

Output: a table of KEEP / DELETE / ASK-THE-USER with a one-line reason each.
Default to KEEP when uncertain; say so explicitly. Do not suggest deletion
commands - the user approves the list first.
PROMPT
)" 2>&1
echo "CODEX_RC=$?"
echo ALLDONE
