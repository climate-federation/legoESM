# ORCA2 round 84 — hierarchy rung-0 entry/stage record gap

Base: `d061f3197c0013b333838af52e9cf9edc1c02b08`.  Preregistration:
`PREREG_nemo_testcases_l4_orca2_round84.md`.  Claim label:
**independent**.  No package file, rung-0 scientific assignment, shipped
ORCA2 card, NEMO source, CPP key, carried state, threshold, stabilizer, or
sea-ice selector changed.

## Verdict

**STOPPED_FOR_RECORD.**  The operator's repaired round-83 record is admitted:
40 two-rank ten-step restart shards, 100 array-equal T/S/u/v/ssh twin
comparisons, and a finite two-rank month terminal at step 240.  All five
record controls refuse.  Those files are terminal restart states, however;
they do not contain the pre-surface-boundary Nbb entry or any of the three RK
stage boundaries.  A first non-bit statement cannot be inferred from them.

The preregistered claim that `output.init` supplies the independent entry is
**REFUTED**.  It is a time-zero diagnostic file but contains nonzero evolved
u/v/ssh on both ranks.  Rank 0 has 226,236 nonzero u values (maximum absolute
0.5398191087580635 m/s), 226,637 v values (0.7971223222873309 m/s), and 8,794
ssh values (1.1923306147585286 m).  Rank 1 has 186,794 u values
(0.27915867769810554 m/s), 188,538 v values (0.3112244695776344 m/s), and
7,639 ssh values (1.3238811838909907 m).  Three zero-state plants fire.  The
committed probe records `NOT_A_STEP_ENTRY_OPERAND`; T/S entry identity remains
unmeasured rather than being mixed with this different boundary.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round84/`:
`record_admission.json`, `initial_output_role.json`, and the three plant logs.

## Compiled boundary and exact missing streams

The recorded build's compiled RK3 driver writes Nbb at the very top of each
step, before calendar and surface-boundary updates
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:90-104`).  It then executes and
records stages 1, 2, and 3 around the two time-level swaps
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:211-227`).  Therefore the missing
record is exactly, for each `kt=1..10` and MPI rank 0 and 1:

- one entry frame (`stage=0`) containing T, S, u, v, and ssh;
- three post-stage frames (`stage=1..3`) containing the same five fields.

That is 80 frames.  Without them the first differing boundary can only be
somewhere between entry and end-of-step, not one compiled statement.  The
linear implicit-drag prediction and every later statement remain
**UNMEASURED**.

## Acquisition issued

The committed round-84 launcher creates only the fresh target
`orca2_rung0_entry_stage_10step_np2`.  Its additions-only `stprk3` patch calls
a write-only module at the four compiled boundaries above.  Every file is
rank-tagged and self-describing: magic, 17 header integers, then five named
arrays with their own rank and dimensions followed by their payload.  The
gate walks each file to physical EOF, derives payload lengths from those
headers, requires finite T/S/u/v/ssh, checks the alternating RK level slots,
and requires all 80 rank-step-stage names.

The launcher pins the admitted rung-0 deck and scalar-math binary, compiles a
new configuration, rejects vector-math symbols, stamps every stream with the
producer commit, and compares all 20 step/rank restart shards byte-for-byte
against the admitted uninstrumented run.  Header, field-name, truncation,
non-finite, and provenance-stamp plants must all fire.  The script's preflight
applies the committed patch with zero fuzz, proves it removes zero source
lines, syntax-checks the Fortran module, and prints
`ORCA2_ROUND84_RUNG0_FRAMES_PREFLIGHT_READY`.

## Prediction ledger

| frozen prediction | result |
|---|---|
| repaired record admits with 40 shards / 100 comparisons / month step 240; five controls fire | **CONFIRMED** |
| explicit rung-0 card contains only assigned rung-0 modules | **UNMEASURED — stopped before card landing** |
| independent card entry is bit-identical to `output.init` | **REFUTED — `output.init` is not an entry operand; u/v/ssh are nonzero on both ranks** |
| first executable-card refusal is linear implicit bottom drag | **UNMEASURED — entry/stage record is earlier** |
| restart-only record cannot source-localize the first non-bit boundary | **CONFIRMED — 80 frames are absent** |
| month waits for a source-localized first statement | **CONFIRMED — no month score claimed** |

## Gates, review, and tests

The frame gate preflight passes and its committed Fortran module passes a
compiler syntax check.  Focused frame-gate tests pass **2/2**, including four
in-memory corruption controls.  The campaign citation gate passes both the
default cumulative receipt (274 citations) and this receipt (2 citations),
with zero failures and zero unmapped citations; its shifted-citation plant
fires `SYMBOL-NOT-AT-LINE`.

The prescribed `tests/ocean/fidelity -n 12` battery selected 2,209 tests and
reached 97% before repeating the established no-summary stall.  The six
standing failure files were rerun without parallelism: **6 failed / 37
passed**, exactly the known round-129 certification, round-35 stamp scope,
worktree-stamp emitter, missing case-board row, SI3 scalar-math provenance,
and round-51 private-trace registry reds.  No round-84 test failed.

The required `codex exec --sandbox read-only` review failed before reading the
diff: `failed to initialize in-process app-server client: Read-only file
system`.  Verdict: **independent review unavailable in-sandbox**.

No `packages/` file changed.  GYRE, DINO, tank cards, the shipped ORCA2 card,
and its `unmeasured_features` sea-ice tuple are unchanged by construction.

## OPEN

1. The operator runs the round-84 launcher and returns its log.
2. Admit all 80 entry/stage frames, show all five controls firing, and prove
   every restart shard byte-identical to the uninstrumented rung-0 record.
3. Build the explicit rung-0 card, compare its own entry against stage 0, and
   walk the first differing boundary in compiled order.
4. Only after naming and discharging the first statement, score both ten-step
   ladders and the independent month.  Rung 1 remains untouched.

ASKED: consume the repaired rung-0 record and name the first non-bit statement,
or issue the exact missing acquisition if the record cannot support that walk.

UNASKED: substituting diagnostic output for entry state, guessing a statement
from terminal restarts, changing hierarchy assignments, the shipped card or
its ice tuple, thresholds, carried state, or stabilizers.
