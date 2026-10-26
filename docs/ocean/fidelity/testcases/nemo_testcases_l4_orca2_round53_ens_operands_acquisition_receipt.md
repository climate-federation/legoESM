# ORCA2 round 53 receipt — OVERFLOW ENS signed-zero operand acquisition

Date: 2026-09-27
Base: `1699ab33be3580867162d066d8b01af6dbb4fca8`
Preregistration: `841241ee564f95d4be39fd3dd319aa288c792a7c`
Instrument: `c237915731005eb26fb83fbbfe4a9207296783c5`
Disposition: **STOPPED_FOR_RECORD**
ORCA2 claim label: **given NEMO's entry** (Decision 52; no ORCA2 trajectory
was measured)
OVERFLOW claim label: **given NEMO's recorded operands**

## Answer

The additions-only acquisition needed by round 52 is committed and ready for
the operator.  It records the stage-2 ENS `zwz` values before and after their
depth division, `zuav`/`zvau`, each vorticity pair and product, and the u/v
accumulator immediately before and after the compiled addition.  It changes no
NEMO arithmetic statement and feeds no recorded value back into NEMO.

No scientific statement lands.  The held QCO arm remains removed, the round-52
signed-zero observation remains the first downstream non-bit statement, and
the frozen product-sign prediction remains **UNMEASURED_WITH_SPEC** until the
operator produces the record.

## Compiled source and acquisition boundary

The producing configuration dispatches the selected ENS routine with `ntot`
at `OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynvor.f90:242-248`, and the
executing UP3 branch defines `ntot=np_CME` at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynvor.f90:866-869`.  The observed
signed-zero transition is produced by the u accumulator statement at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynvor.f90:666`.

The committed patch adds calls around that statement and its operands.  A dry
application removes or replaces zero shipped-source lines.  Preprocessing with
the producing `key_qco`, `key_vco_3d`, and `key_RK3` set and `gfortran
-fsyntax-only` both pass.  The preflight binds to compiled `np_CME=5` and the
exact compiled accumulator statement.

## Self-describing record and controls

The writer emits one new stream for kt=3 stage 2.  Its header carries magic,
version, kt, stage, Kmm, executed vorticity branch, owned dimensions and
origins, fp width, and field count.  Each payload carries its own name, rank,
and three extents.  The checker obtains payload lengths from those declared
extents, walks physical EOF, and rejects wrong magic/version/time/branch,
invalid dimensions/origins/fp width, wrong/missing/duplicate fields, malformed
rank or shape, non-finite payloads, truncation, trailing bytes, digest drift,
producer drift, and disagreement with the admitted round-50 parent header.  It
contains no predicted total byte count or independently predicted header
tuple.

The operator script requires a clean committed producer, uses only committed
repo-relative patch/writer/gate/preregistration paths, verifies the pinned
round-50 binary and resolved ten-step configuration, builds under the new name
`OVERFLOW_OMIP_L1_P3_R53ENS`, and runs under the new directory
`round53/acquisition/oracle_overflow_ens_operands`.  It then admits all six
round-50 parent records, the new record, restart, mesh, and inherited streams.
Payload, producer-stamp, and inherited-consumed-field plants must all refuse
before it prints READY.

## Frozen predictions

| ID | verdict | evidence |
|---|---|---|
| R53-P1 | **PARTLY CONFIRMED** | The patch is additions-only, both Fortran units syntax-check, and the schema/preflight controls pass.  Restart and inherited-stream identity remain unmeasured until the operator run. |
| R53-P2 | **UNMEASURED_WITH_SPEC** | No internal ENS record exists yet; the predicted `-0.0 + +0.0 -> +0.0` chain is frozen but not claimed. |
| R53-P3 | **PARTLY CONFIRMED** | Synthetic payload and producer-stamp plants refuse; the operator script also requires those plants plus the inherited-field plant on the real record. |
| R53-P4 | **CONFIRMED** | The final `packages/` diff against the round base is empty; no model or scientific statement lands. |

## Verification

- Round-50/52/53 focused controls: **18 passed**.  Round-53 Ruff,
  `py_compile`, shell syntax, patch application, preprocessing, and both
  Fortran syntax checks pass.
- Shared-card battery: **170 passed**, 9 warnings, in 346.68 s.
- `tests/ocean/fidelity -n 12` collected 1,953 items, reached 99%, emitted the
  same five registered failures as round 52, and reproduced the documented
  xdist tail stall before interruption.  Serial rerun preserves all five
  signatures: round-129 stale certification, round-51 private trace registry,
  SI3 scalar-math provenance, three worktree-stamp offenders, and the
  `hires_lane_surface` case-board omission.
- Separate read-only `codex exec` review: **independent review unavailable
  in-sandbox** (`failed to initialize in-process app-server client: Read-only
  file system`).
- Default-receipt citation gate: **274 citations, PASS**, zero failures and
  zero unmapped; this receipt: **3 citations, PASS**.  Shifting the accumulator
  citation by two lines fires `SYMBOL-NOT-AT-LINE`.
- No `packages/` file changed, so the GYRE trajectory/year and shared-model
  landing gates are not triggered.  Sea ice, all six ORCA2 selectors, and the
  card's `unmeasured_features` tuple remain unchanged at `STOP_SELECTOR_GAP`.

## OPEN

1. Operator: run
   `/tmp/autopilot-orca2-106384614/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_overflow_round53_ens_operands/run.sh --run`.
2. Admit the new record, locate round 52's first active-u signed-zero cell, and
   walk `zwz`, `zuav`, their product, and the final addition in source order.
3. Keep the QCO arm held until that walk identifies a complete source-exact
   pair or the first non-bit internal statement.
4. Then return to ORCA2's whole-card kt=1 stage-1 T owner, the independent
   Decision-52 initial state/year, and round-20 slow forcing.
5. Sea ice remains out of scope at `STOP_SELECTOR_GAP`.

ASKED: commit the missing additions-only acquisition and continue the ordered
OVERFLOW pair walk.
UNASKED: none.
