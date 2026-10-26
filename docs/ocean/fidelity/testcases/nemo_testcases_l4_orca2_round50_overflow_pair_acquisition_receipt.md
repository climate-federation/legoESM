# ORCA2 round 50 receipt — OVERFLOW kt=3 acquisition completed for operator

Date: 2026-09-27  
Base: `6a2df3730`  
Preregistration: `3d8ce5c7e`  
Disposition: **STOPPED_FOR_RECORD**  
ORCA2 claim label: **given NEMO's entry**  
OVERFLOW claim label: **independent**

## Frozen question

Round 49's operator handoff correctly exited 66 because it intentionally
contained no writer or parser.  Round 50 asked whether a committed,
additions-only instrument could expose the missing kt=3 boundaries without
changing a NEMO statement or configuration choice.

The executing compiled order is EOS, HPG, vorticity, and advection for
stages 2/3 at
`OVERFLOW_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:327-354`; stage 3 then
adds lateral mixing, integrates vertical mixing, and applies the barotropic
correction at
`OVERFLOW_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:378-438`.  Tracers are
zeroed, receive advection and the surface source, and then take either the
stage-1/2 QCO/RK assignment or stage-3 completion at
`OVERFLOW_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:480-567`.

## Instrument result

The committed recorder writes six new streams: momentum and tracer records
for each of kt=3 stages 1, 2, and 3.  Every file carries magic, version, kt,
stage, all four time-level slots, owned dimensions and origins, fp width, a
closed field count, named field dimensions, and physical EOF.  Only the owned
202 x 3 x 100 box is written.  The patch applies to the record's actual
`MY_SRC/stprk3_stg.F90`, removes/replaces zero lines, and adds 20 calls/import
lines.  Preprocessing and `gfortran -fsyntax-only` pass with this deck's
compiled `key_qco`, `key_vco_3d`, and `key_RK3` keys.

The parser rejects wrong magic, version, kt, stage, slots, dimensions, owned
origin, fp width, field count/order/rank, duplicate or missing fields,
non-finite payloads, truncation, trailing bytes, record-name stamps, producer
commit, and digest.  Nine focused tests pass.  Both end-to-end synthetic
plants fire: changing the first payload by one representable fp64 step causes
a digest refusal, and changing the producer stamp causes a commit refusal.

The final preflight report is stamped at clean producer commit `adb4f5ee6`,
with source SHA-256
`f3e19ce0e37df219040ddf6ccd50493e2bd5e045aa6eaaa2eb539efdaafb59f1`,
module SHA-256
`57fa67985abe4777cc23a7cc4a4833b16151ccd3c8a7954bc622c1eab803ab45`,
and patch SHA-256
`c96e891e290b49c36e4cca7fc4cf906c9e08c6e5e42a8abddeeca58d29d3eaf7`.

## Frozen predictions

| prediction | result |
|---|---|
| R50-P1: one additions-only patch exposes the required boundaries | **CONFIRMED** by zero removed lines, all source-order sentinels, and both Fortran syntax checks |
| R50-P2: the schema is mechanically closed | **CONFIRMED** by 9 focused tests and both firing plants |
| R50-P3: the instrument changes no consumed NEMO state | **UNMEASURED** until the operator run passes restart/inherited-stream admission |
| R50-P4: no scientific statement lands this round | **CONFIRMED**; `packages/` is unchanged and no record-backed pair was measured |

## Acquisition handoff

The operator entry point is:

`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round50/acquisition/run.sh`

It resolves to the committed producer, requires `--run`, a clean producer
tree, the pinned source binary hash and resolved configuration, a new target
configuration `OVERFLOW_OMIP_L1_P3_R50PAIR`, and a new run directory
`oracle_overflow_kt3_pair`.  It checks free space, preprocesses and syntax-
checks before creating the target, verifies the compiled writer sentinels and
scalar-math binary, runs ten steps, admits all inherited streams plus the
restart and mesh through the existing consumed-field gate, runs both plants,
and emits a final manifest.  It never edits canonical NEMO source.  Per the
withdrawn in-sandbox acquisition note, `makenemo` and `mpirun` were not invoked
by this round.

## Verification

- Focused round-50 gate: **9 passed**; Ruff clean; shell syntax clean.
- Citation gate: the 274-citation default receipt and all three round-50
  compiled-source spans pass with no unmapped citation; shifting the first
  round-50 span by two lines fires `SYMBOL-NOT-AT-LINE`.
- Shared card battery: **160 passed** in 365.13 s; tank-removal battery:
  **10 passed** in 8.21 s.
- `tests/ocean/fidelity -n 12`: 1,940 items reached 99%; all emitted nodes were
  green, then the known xdist tail stall reproduced and the run was
  interrupted.  The five known failures were rerun serially and retain their
  existing signatures: round-129 stale certification, round-51 private trace
  registry, SI3 scalar-math provenance, three pre-existing worktree-stamp
  offenders (the round-50 gate is not an offender), and the
  `hires_lane_surface` case-board ratchet.
- Separate read-only `codex exec` review: **independent review unavailable
  in-sandbox** (`failed to initialize in-process app-server client: Read-only
  file system`).
- No `packages/` file changed, so the GYRE trajectory/year gate is not
  triggered; this round changes instrumentation only.

## OPEN

1. Operator: run the round-50 acquisition path with `--run`.  R50-P3 remains
   UNMEASURED until restart, inherited-stream, schema, and plant admissions
   pass.
2. With the admitted kt=3 record, compare held QCO candidate and base in the
   compiled order, name the first unequal statement, and test it bit-exact
   before forming any cancelling pair.
3. Keep the QCO candidate held.  No model change from round 48 or 49 is carried.
4. Return afterward to ORCA2's kt=1 stage-1 T producer, the independent
   Decision-52 initial state/year, and round-20 slow forcing.
5. Sea ice remains unchanged at `STOP_SELECTOR_GAP`; the six selectors and
   `unmeasured_features` tuple are untouched.

ASKED: complete the previously missing write-only acquisition instrument.  
UNASKED: none.
