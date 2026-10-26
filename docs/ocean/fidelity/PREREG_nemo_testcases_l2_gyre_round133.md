# Preregistration — NEMO testcase L2 GYRE round 133

Date: 2026-09-20

Incoming lane tip: `e9f3bb6a557581f0cc5b18a60a27757b60f49ef7`

This document is frozen before the retained Round-132 target, staged binary,
or run directory is formally admitted for reuse. Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round133/`.

The operator ran the Round-132 acquisition. The build completed, but the run
wrapper exited 127 at its `/usr/bin/time` invocation. Round 133 diagnoses that
failure and prepares a fail-closed resume of the existing target. It does not
run `makenemo` or `mpirun` in this agent round, does not replace any retained
artifact, and does not begin the four-family legoESM nudging experiment.

## P0 — frozen failure diagnosis

The operator log identifies the first failing command as
`/usr/bin/time -p ... mpirun ...`; the host has no `/usr/bin/time`. Because the
missing wrapper is the command that would have launched `mpirun`, the frozen
prediction is that NEMO never started: the retained run root contains zero
restart files, no `ocean.output`, no exact `STOP 0`, no `NEMO_DONE`, and no
`RUN_DONE`.

The prediction is refuted if any NEMO-produced state or completion marker is
present. In that case the directory is not a clean prepared run and may not be
resumed in place.

The two `REFUSE` lines at acquisition-script lines 358 and 369 are predicted
to be one underlying status-127 failure reported twice because the inherited
ERR trap fires first for the failed pipeline and again for its enclosing
subshell. The recovery path must place the MPI pipeline in a tested conditional
so one failure produces one named refusal.

## P1 — exact retained-build admission

The only reusable NEMO configuration and run root are:

* target configuration:
  `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/GYRE_OMIP_L2_P3_SM_R132DAILY`;
* staged run:
  `phase3/round132/oracle_daily_restarts`;
* frozen source configuration: `GYRE_OMIP_L2_P3_SM_R41ADVSP`.

Before a resume is exposed, a committed preflight must prove all of the
following without writing those roots:

1. the retained `binary.sha256` is a one-row manifest for the target binary;
2. the built target binary and staged `nemo` copy both match that digest;
3. the target compiled `restart`, `stprk3`, `dynspg_ts`, and `zdftke` programs
   remain byte-identical to the frozen source build;
4. the target cpp-key file, scalar architecture, frozen source binary, source
   card manifest, and copied-card manifests retain their Round-132 identities;
5. every staged input equals the dereferenced target-card input;
6. the staged namelist differs from the source card only in the already
   authorized rows `nn_itend: 10 -> 2160`, `nn_stock: 10 -> 6`, and
   `nn_write: 10 -> 2160`; and
7. the original producer commit is the Round-132 clean commit and the current
   recovery worktree is clean at its own full commit.

Frozen prediction: every identity check passes. The falsifier is any digest,
compiled-file, staged-input, namelist, producer-commit, or clean-tree mismatch.
Such a mismatch requests a new acquisition under a new target; it is never
silently repaired.

A binary-identity plant must substitute a false expected digest without
modifying the retained binary. It must print `STATUS PLANT-FIRED`, a named
`REFUSE`, and exit nonzero. A passing plant is a gate failure.

## P2 — no-rebuild resume and record admission

The committed Round-132 `run.sh` is extended rather than duplicated. Its new
default recovery mode may execute only the already staged `nemo` binary in the
already prepared run directory. The mode must contain no path to `makenemo`,
must not copy or patch NEMO source, and must not invoke `/usr/bin/time`.
Original failed logs remain untouched; the resumed run writes distinct
`run.resume.*` logs.

The operator run must complete 2,160 steps with process status zero and exact
`STOP 0`, then satisfy the unchanged Round-132 gates:

* exactly 360 restart files at `kt=6,12,...,2160`, each 1,466,328 bytes;
* all twelve monthly files byte-identical to the certified monthly trajectory;
* the day-30 file also byte-identical to the independent old daily control;
* all 16 registered family fields admitted by the Round-131 gate;
* the complete virtual control passes and both existing plants exit 1; and
* the digest manifest and audit are stamped with the clean recovery commit.

Frozen prediction: the resumed NEMO run and every admission row pass, and the
script prints `ROUND133_DAILY_RESTART_RECORD_READY`. A nonzero process result,
missing `STOP 0`, any changed trajectory byte, schema/census error, plant that
stays green, or stamp mismatch prevents the ready marker.

## P3 — outcome and scope

This agent runs only the recovery preflight and its plants. If P0 and P1 are
confirmed, Round 133 ends `STOPPED_FOR_RECORD` and names the amended `run.sh`
as `ACQUISITION_NEEDED`; the operator then runs its default no-rebuild mode.
If retained-build admission fails, Round 133 instead requests a new target.

No model physics, card choice, carried state, stabilizer, year harness,
ORCA2 integration, or NEMO source changes in this round. The Round-131
four-family attribution remains OPEN until the recovered record prints its
ready marker and is re-admitted at the next clean commit. A separate read-only
Codex pass must try to refute the no-run diagnosis, retained-build identity,
no-rebuild guarantee, plant non-vacuity, marker ordering, and handoff verdict.
The receipt citation gate and shifted-citation plant remain mandatory.
