#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_pfix
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/cdx_pfix_%j.out
set +e
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha>}
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
echo "host: $(hostname)  CODEX_HOME=$CODEX_HOME"; codex --version || exit 91
cd $WT || exit 92
HEAD_SHA=$(git rev-parse HEAD); echo "PINNED sha: $HEAD_SHA"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 93; }

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
REVIEW ONE FIX. Answer the numbered questions, then stop.

THE FIX (commits: `git log --oneline -6`; the two that matter are the
"report the precipitation the column actually produced" and
"pin the precipitation readout" commits; see
`git show <sha> -- scripts/run/run_scm_rce_campaign.py`).

BEFORE: the SCM's per-substep surface precipitation came from
`precip_diagnostic(sub_state, grid, sigma_coord)` -- a SECOND, diagnostic-only
invocation of the microphysics, whose closure was built with the OUTER dt
(600 s) while the applied operator runs at dt/substeps (20 s).

AFTER: `applied_precip_mm_day(micro_tend, sub_state)` reads
`HydrostaticTendencies.precip` from the microphysics tendency that is APPLIED,
converting kg/m^2/s -> mm/day with a factor of 86400.

MEASURED EFFECT (jobs 9361582/9361587/9361599), E from the shipped
compute_surface_fluxes on each equilibrium column:
  config            E       storage   P before    P after   residual
  sbm               1.559   +0.072    4.1e-18     1.125     0.362
  mass_flux         0.906   -0.301    2.4e-18     0.971     0.236
  convection=none   0.933   -0.854    1.0e-07     1.352     0.435
Controls: the hard saturation adjustment moved the sink by 0.001 mm/day on sbm
and 0.042 with convection off (refuted as the sink); morrison's precipitation
sums precip_r+precip_i+precip_s+precip_g (frozen precip is NOT excluded).

Q1. IS THE READOUT CORRECT? `micro_tend.precip` is documented as
"[kg/m^2/s, +into surface] ... SURFACE precip (micro sedimentation)". Is
86400 the right and complete conversion (1 kg/m^2 == 1 mm)? Is taking
element [0] of a flattened field right for a single-column model, and would it
silently do the wrong thing for ncol > 1?

Q2. IS THE SAMPLING RIGHT? Each substep computes micro_tend on `sub_state`,
applies it over sub_dt, and accumulates `sub_weight * precip` with
sub_weight = 1/n. Is a uniform mean over substeps the correct time average of
a rate that varies within the outer step? Is there an off-by-one (the rate is
evaluated BEFORE the substep's own update)? Does the outer loop then average
correctly over the analysis window?

Q3. DOUBLE COUNTING. The score adds `micro_precip + convective_precip_for_score`,
and the convective term is zeroed whenever convection is sub-stepped and
microphysics != "none". With the fix, is the total now correct, or does an
adjustment scheme (sbm/dca/kuo) -- whose column drying the standalone bridge
routes into q_c (convection/integration.py:670) -- get counted once, twice, or
not at all?

Q4. THE RESIDUAL 0.24-0.43 mm/day. My reading is that it is the PROBE, not the
model: E is a snapshot at the final state while storage and P are window
means, and the residual is largest for the least-equilibrated configuration.
Is that reading defensible, or is there a real remaining sink? What single
measurement would settle it?

Q5. THE TESTS in tests/unit/test_scm_rce_applied_precip.py. Can they pass
while the readout is wrong? In particular
`test_a_summed_tendency_would_report_zero` asserts that add_tendencies DROPS
precip -- is asserting a dependency's current behaviour a good gate here, or
does it lock in something that should change?

Q6. Anything else wrong with the fix.
PROMPT
)" 2>&1
CODEX_RC=$?
echo "CODEX_RC=$CODEX_RC"
[ $CODEX_RC -eq 0 ] || echo "REVIEW DID NOT RUN"
echo ALLDONE
