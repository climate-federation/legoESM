# NEMO testcase L4 ORCA2 round 62 — dyn_zdf internal acquisition

Date: 2026-09-28
Incoming tip: `121bcd1c3d513f6f75bd97cf3a12369e65ee0038`
Frozen preregistration: `9335a21a05b3f641b397b9b551852de9f9f014a0`
Final disposition: **STOPPED_FOR_RECORD; acquisition committed, no physics lands**
Evidence: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round62/`

## Outcome

The header-driven inventory confirms the round-61 record gap.  None of the
eight unique admitted, self-describing records in the round-50 lineage carries
the required `explicit_u/v`, `baro_subtract_u/v`, `baro_drag_u/v`, and
`implicit_solve_u/v` sequence.  The stage-3 round-50 momentum record has only
the pre-`dyn_zdf` RHS and final raw-Kaa endpoint.  R62-P1 is **CONFIRMED**.

Round 62 therefore commits a new additions-only NEMO writer, a parser/admission
gate, its non-vacuity controls, and an operator-run launcher under target
`OVERFLOW_OMIP_L1_P3_R62ZDF`.  The record remains **UNMEASURED** until the
operator runs the launcher.  No NEMO run was attempted in the sandbox, as
required by operator note B10.  No legoESM package, configuration, carried
state, or sea-ice field changed; the held UP3 and QCO/RK candidates remain
absent.

## Frozen predictions

| ID | verdict | evidence |
|---|---|---|
| R62-P1 | **CONFIRMED** | Eight schemas were parsed through physical EOF from their own headers; `complete_carriers` is empty. |
| R62-P2 | **PREFLIGHT PASS** | The final dry patch applies with zero removed/replaced source lines, both Fortran units compile, and all seven sentinels occur once.  Actual rebuilt-source verification remains for the operator run. |
| R62-P3 | **READY / UNMEASURED** | Unit controls accept the declared eight-field schema and reject wrong magic, RK slot, name, non-finite payload, truncation, and trailing bytes.  The real record does not yet exist. |
| R62-P4 | **UNMEASURED** | Restart identity, inherited-stream admission, endpoint identity, and the consumption plant execute only after acquisition. |
| R62-P5 | **CONFIRMED** | The committed launcher requires exactly `--run`; its no-argument control exits 63 with the named `REFUSE` line.  It was not invoked with `--run`. |
| R62-P6 | **CONFIRMED** | `git diff 121bcd1c3 -- packages` is empty. |

## Compiled source order and record boundary

The executing routine first forms the explicit stage-3 velocity at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynzdf.f90:139-151`.
It then removes the split-explicit barotropic velocity at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynzdf.f90:159-162`
and adds the executed bottom drag contribution at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynzdf.f90:164-170`.
The U and V tridiagonal constructions and solves end at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynzdf.f90:188-350`
and
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynzdf.f90:357-518`.

The additions-only patch observes complete owned rows immediately after each
of those boundaries.  Its module buffers eight arrays only for
`(kt,stage,Kbb,Kmm,Krhs,Kaa) = (3,3,1,2,3,3)`, refuses incomplete row counts
or non-fp64 execution, opens the output with `STATUS='NEW'`, and never exposes
a read-back path into NEMO state.

The parser derives dimensions and payload lengths from the record header,
requires the ordered field registry, reads through physical EOF, and rejects
non-finite arrays.  Admission also requires the two final solve arrays to be
bit-identical to the inherited round-50 `raw_kaa_u/v` endpoint.  Payload and
stamp plants are wired into the launcher; inherited-stream admission has its
own consumed-record plant.

## Acquisition contract

The launcher and every file it reads are committed under repository-relative
paths.  It requires a clean producer tree, the frozen parent binary digest,
the resolved active namelist rows, an absent new config and run directory,
two GB free space, additions-only patching, scalar-math linkage, six inherited
round-50 records, the new record, both local plants, and inherited restart and
mesh identity.  It creates the evidence parent before its `df` check and does
not call `/usr/bin/time`.

Operator command:

```bash
/tmp/autopilot-orca2-126085384/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_overflow_round62_dynzdf/run.sh --run
```

Expected final line:

```text
ORCA2_ROUND62_DYNZDF_INTERNALS_READY /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round62/acquisition/oracle_overflow_dynzdf_internals
```

The local preflight JSON SHA-256 is
`1bb03551db4cf9663df4ead388ccfe55e2002b5dc45fb052caf5410e1c5da261`.

## Failed attempt, review, and verification

The first acquisition commit contained a malformed unified-diff hunk after a
whitespace-only cleanup.  It was never run.  Commit `e8b5232c0` corrected the
hunk header; the final dry apply, additions-only check, Fortran syntax check,
and gate preflight all pass.  This supersedes the invalid `c7012cbc8` patch
artifact rather than hiding the failed attempt.

The separate read-only Codex review did not start a reviewer model.  Its exact
terminal verdict was:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Therefore **independent review unavailable in-sandbox**.  The artifact SHA-256
is `eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

The focused round-62 controls pass 11 / 11.  Final citation and ocean-fidelity
battery results are recorded in the closing commit.

## Scope ledger

**ASKED.** Inventory or acquire the source-ordered `dyn_zdf` internal frames
before another compensating-pair retry.

**UNASKED and unchanged.** No configuration, default, forcing, threshold,
stabiliser, carried state, physics statement, ORCA2 entry, or sea-ice field
changed.  The six ORCA2 ice selectors and `unmeasured_features` remain at
`STOP_SELECTOR_GAP`.

## OPEN

1. Operator runs the committed acquisition and returns its admission outputs.
2. Admit the record; add passive legoESM observers for the same four seams;
   compare base and held-UP3 arms from the frozen independent kt=3 entry and
   name the first strict direction change.  MIXED remains evidence, not owner.
3. Keep source-ordered UP3 and QCO/RK held until a strict compensating
   statement is named.
4. After the step walk closes, run Decision 52's independent-start ORCA2
   ladder, then rank month-scale ORCA2 magnitudes.
5. Sea ice remains out of scope at `STOP_SELECTOR_GAP`.

ASKED: inventory/acquire the `dyn_zdf` internal source-order walk.
UNASKED: configuration, state, stabiliser, physics, and sea-ice changes.
