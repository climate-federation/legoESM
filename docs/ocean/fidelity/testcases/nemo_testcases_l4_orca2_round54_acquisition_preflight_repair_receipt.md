# ORCA2 round 54 receipt — ENS acquisition free-space preflight repair

Date: 2026-09-27  
Base: `da29c80009876fc10af1d0b323f2e481501cee96`  
Preregistration: `673307e95e362dccf6821ee59f62a32a0699fd2e`  
Repair: `2d33f72a21783f38bc65b113307137bee71be6eb`  
Disposition: **STOPPED_FOR_RECORD**  
ORCA2 claim label: **given NEMO's entry** (Decision 52; no ORCA2 trajectory
was measured)  
OVERFLOW claim label: **given NEMO's recorded operands**

## Answer

The operator's round-53 acquisition reached its own free-space preflight and
exited before `makenemo` or `mpirun`: `df` was asked to inspect the absent
parent of the fresh target run directory.  The repaired launcher creates only
that parent after it has refused any existing target, then checks the now-live
filesystem.  The target configuration and target run themselves remain absent.

The repair is one launcher line plus one direct control.  No writer, parser,
NEMO patch, target name, run configuration, model package, selector, carried
state, score domain, threshold, stabiliser, or sea-ice field changed.  No ENS
record exists, so round 53's product-sign prediction remains
**UNMEASURED_WITH_SPEC** and no scientific statement lands.

## Source boundary retained

The unchanged acquisition still instruments the selected ENS accumulator at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynvor.f90:666`.  This round does
not alter or reinterpret that statement; it only makes the pre-run disk-space
check address an existing directory.

Repository search found the established guard/create/check sequence in the
round-156 GYRE acquisition.  The round-53 launcher was extended in place; no
second launcher or helper was added.  The source and target configuration
roots and the target run were checked after the operator failure: both targets
were absent.

## Control and fail-closed behavior

The new direct test reads the real launcher and requires this exact order:
fresh-target refusal, parent creation, free-space check.  It then starts with a
nested absent parent, executes the same shell operations, requires `df` to
succeed, and requires the target itself to remain absent.  Removing the single
parent-creation line makes that test fail at the missing line, proving the
repair binds.

The original acquisition safeguards remain unchanged: clean committed tree,
fresh target configuration and run, pinned source binary and resolved
namelist, additions-only source patch, compiled-branch sentinels,
self-describing record parser, restart/mesh/inherited-stream identity, and
payload/producer/inherited-field plants.  The sandbox did not run `makenemo` or
`mpirun`.

## Frozen predictions

| ID | verdict | evidence |
|---|---|---|
| R54-P1 | **CONFIRMED** | The direct shell control starts with an absent nested parent, passes `df` after creating only the parent, and leaves the target absent.  The real launcher keeps the freshness refusal before that creation. |
| R54-P2 | **CONFIRMED** | Removing the one-line repair makes the direct test fail; the committed implementation diff changes only the launcher and its direct test, and `packages/` is unchanged. |
| R54-P3 | **CONFIRMED** | Shell syntax, Ruff, Python compilation, additions-only/compiled-branch preflight, and all 19 focused round-50/52/53 controls pass.  The writer, parser, patch, target, and record names did not move. |
| R54-P4 | **CONFIRMED** | Both real targets remain absent and no ENS record exists; this round remains `STOPPED_FOR_RECORD`. |

## Verification

- Focused acquisition controls: **19 passed**; shell syntax, Ruff, and
  `py_compile` pass.  The clean committed preflight reports
  `PREFLIGHT_PASS`, compiled `np_CME=5`, zero removed source lines, and the
  unchanged source/module/patch digests.
- The parent-creation plant fails the direct test exactly at the missing
  launcher line; restoring the line returns the focused battery to green.
- Shared-card battery: **170 passed**, 9 warnings, in 345.87 s.
- `tests/ocean/fidelity -n 12` collected 1,954 items, reached 99% with no
  emitted failure before reproducing the registered xdist tail stall and being
  interrupted.  The five registered reds were rerun serially and retain their
  prior signatures: round-129 stale certification, round-51 private trace
  registry, SI3 scalar-math provenance, three worktree-stamp offenders, and
  the `hires_lane_surface` case-board omission.
- Separate read-only `codex exec` review: **independent review unavailable
  in-sandbox** (`failed to initialize in-process app-server client: Read-only
  file system`).
- Default-receipt citation gate: **274 citations, PASS**, zero failures and
  zero unmapped.  This receipt's final citation and planted-shift results are
  recorded with the final receipt commit.
- No `packages/` file changed, so the GYRE trajectory/year, DINO, tank, and
  generic-card model landing gates are not triggered.  Sea ice, all six ORCA2
  selectors, and the card's `unmeasured_features` tuple remain unchanged at
  `STOP_SELECTOR_GAP`.

## OPEN

1. Operator: run
   `/tmp/autopilot-orca2-gCIT7E/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_overflow_round53_ens_operands/run.sh --run`.
2. Admit the new record, locate round 52's first active-u signed-zero cell, and
   walk `zwz`, `zuav`, their product, and the final addition in compiled order.
3. Keep the QCO arm held until that walk identifies a complete source-exact
   pair or the first non-bit internal statement.
4. Then return to ORCA2's whole-card kt=1 stage-1 T owner, the independent
   Decision-52 initial state/year, and round-20 slow forcing.
5. Sea ice remains out of scope at `STOP_SELECTOR_GAP`.

ASKED: repair the failed round-53 acquisition and continue the ordered
OVERFLOW pair walk.  
UNASKED: none.
