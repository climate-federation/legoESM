#!/bin/bash
#SBATCH --job-name=codex_gwcum1
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/codex_gwcum_r1_%j.out

# ROUND 1 adversarial review of the per-cadence CUMULATIVE gateway dump.
# Runs against the PINNED worktree, not the shared checkout.
set -u
WT=${WT:?set WT to the pinned review worktree}
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "=== CODEX_HOME=$CODEX_HOME  auth=$(ls -l $CODEX_HOME/auth.json 2>&1) ==="
echo "=== reviewing pinned worktree $WT at $(git -C $WT rev-parse HEAD) ==="

codex exec --sandbox read-only -C $WT \
  "ROUND-1 adversarial review of a follow-up commit on PR #1440 (branch
ocean-arctic-gateway-transports).  Review ONLY the top commit; run
'git show --stat HEAD' and 'git show HEAD' to see it.

WHAT IT DOES.  --gateway-transports accumulates per-timestep Arctic gateway
volume and salt transports and writes a WHOLE-RUN time mean to transports.txt
at run end.  That mean cannot answer the actual question (the days 30-90 mean,
because days 0-30 are cold-start adjustment).  So the accumulator now ALSO
dumps its CUMULATIVE sums plus the cumulative step count to a new
gateway_transports.csv at the --snapshot-every-days cadence and once at run
end.  Any window is then recovered by differencing two rows:
    mean(n_a, n_b] = (cumsum_b - cumsum_a) / (n_b - n_a).

FILES: packages/ocean/legoesm/ocean/diagnostics_sections.py (new
gateway_cumulative_columns / format_gateway_cumulative_row),
scripts/run/run_omip_core2.py (_gateway_cumulative_open/_row/_close + three
call sites), and the two test files.

REQUIREMENTS IT MUST MEET — check each:
 1. CUMULATIVE, not per-window: nothing may be reset mid-run, so the run-end
    whole-run mean in transports.txt must be produced by the very same
    accumulator and be BYTE-IDENTICAL to before this commit.
 2. Differencing must be EXACT: the decimal representation must round-trip, or
    a window mean computed from a difference of two large sums is quantised.
 3. Every window must be recoverable, including the LAST one.  The cadence
    block is guarded by 'step != n_steps', so ask whether the run-end row
    really covers the final window and whether any dump point is written twice
    or zero times.
 4. It must not perturb the trajectory.  It is host-loop code (the lane guard
    validate_gateway_lanes rejects --scan-block and --spmd-persistent-state);
    confirm that from the code rather than taking my word for it, and say if
    any lane could reach the new code with a scan carry or a sharded state.
 5. A diagnostic must never abort or hang the run: check the non-fatal
    boundaries, the file handle lifetime on EVERY exit path out of main()
    (normal end, the [ABORT] non-finite return 1, and the accumulator being
    disabled mid-run by a gateway_step failure), and the process-0-only
    writing under --distributed.
 6. Column identity: a gateway's value must never land under another
    gateway's column, and the header must always match the row arity.

ALSO ATTACK THE TESTS, not just the code.  This branch has a history of
vacuous tests: five earlier tests passed while proving nothing because two
code paths were identical with GM off, and a fixture that varied a sha at
character 39 produced nine IDENTICAL 12-char tags making 'len<=cap' and
'sorted' assertions trivially true.  So for each new test ask: what value would
this assertion take if the feature were removed?  Name any assertion that
would still pass on a broken writer.  In particular scrutinise
test_differencing_two_dumps_equals_a_fresh_accumulator_over_that_window (is
its reference genuinely independent, or does it share the bug it is meant to
catch?) and test_accumulating_and_dumping_does_not_perturb_the_trajectory (is
the digest sensitive enough, and are the two arms actually comparable given
that store_mass_flux adds three state slots?).

THREAT MODEL — DO NOT REPORT THESE.  The threat model is DRIFT AND MISTAKES on
a shared filesystem (a stale artefact, an unwired lane, a silently disabled
flag, author error), NOT an adversary.  Forged CSV files, hostile environment
variables, a caller hand-editing the accumulator, concurrent writers racing
for the same output directory, and CSV-injection-style concerns are explicitly
OUT OF SCOPE and must not be reported.  Also out of scope: the pre-existing
h*u reconstruction caveat and the arithmetic-mean h_v in
diagnostics_streamfunction (issue #1441).

Reply as a numbered list, each line 'path:line: EMOJI severity: problem. fix.'
Use RED for a bug, YELLOW for a risk.  Be specific about the line.  End with a
final line 'OVERALL: CLEAN' or 'OVERALL: NOT CLEAN'."
echo "=== CODEX DONE rc=$? ==="
