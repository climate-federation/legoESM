# ORCA2 round 56 — stage-2 UP3 boundary and acquisition receipt

Date: 2026-09-27  
Base: `8bf64602ab505715d26f3b71518b2e04f7221d3c`  
Measurement commit: `f164060268a659f25c8da2bcc19ac07a038ab3e7`  
Disposition: **STOPPED_FOR_RECORD**  
Claim label: **given NEMO's recorded operands**

## Outcome

The admitted OVERFLOW kt=3 stage-2 boundary is non-bit across momentum
advection: `after_vor_u` to `after_adv_u` changes **651 / 16,900** active U
cells, with maximum absolute change `1.0727935161122679e-05`.  This one-row
card has no active V face, so V remains `UNMEASURED_NO_ACTIVE_FACE` rather than
being called exact.

That is the first result this round can establish.  It is not yet a first
non-bit *statement*: the parent record has the two endpoints but lacks the
kt=3 stage-2 transports and UP3 internal boundaries needed for a source-order
walk.  No physics or card change is eligible to land from endpoint evidence.

The committed operator acquisition is:

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_overflow_round56_up3_operands/run.sh`

It uses the fresh target `OVERFLOW_OMIP_L1_P3_R56UP3` and writes
`oracle_r56_up3_kt00000003_s2.bin`.  The agent did not invoke `makenemo` or
`mpirun` in the sandbox.

## Compiled statement order

The producing executable's compiled stage program records `after_vor`, calls
momentum advection with the explicit `zFu/zFv/zFw` operands, and then records
`after_adv` at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:327-363`.
Its compiled dispatcher selects `np_FLX_up3` at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynadv.f90:134-145`.
The executed routine forms horizontal curvatures at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynadv_up3.f90:150-158`, selected
face values, face fluxes, and the horizontal RHS at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynadv_up3.f90:174-219`, then the
vertical flux/RHS sequence and bottom level at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynadv_up3.f90:278-359`.

These are the boundaries the acquisition records.  It is a WRITE-only module;
the source patch adds calls around the compiled expressions and removes or
replaces zero source lines.

## Record inventory

| item | admitted state | round-56 disposition |
|---|---|---|
| round-50 kt=3 stage-2 momentum record | `AT_BAR`; fp64, slots/origin/shape and producer stamp accepted | parent endpoints used |
| `after_vor_u`, `after_adv_u` | both present | 651 / 16,900 active cells differ |
| active V faces | zero on OVERFLOW-zps | `UNMEASURED_NO_ACTIVE_FACE` |
| `oracle_transport_kt00000003_s2.bin` | absent | required record gap |
| `oracle_up3_internal_kt00000003_s2.bin` | absent | required record gap |
| round-56 combined self-describing record | not yet acquired | operator acquisition required |

The new schema contains 47 named fp64 arrays: stage transports, Kbb/Kmm
velocities, RHS entry, horizontal curvatures and selectors, T/F face fluxes,
horizontal scales and before/after RHS, and the corresponding vertical
curvature/transport/selector/flux/scale/before/after values.  The parser reads
magic, header integers, every `(name, rank, n1, n2, n3)` tuple, payload length,
and physical EOF from the record itself; it does not predict a whole-file byte
count.

## Frozen predictions

| ID | verdict | evidence |
|---|---|---|
| R56-P1 | **CONFIRMED** | Parent admission is `AT_BAR`; the compiled branch and running namelist select flux-form UP3 with explicit transports. |
| R56-P2 | **CONFIRMED** | The boundary changes 651 active-U cells; V has no active face. |
| R56-P3 | **CONFIRMED** | Both frozen missing stream names are absent from the admitted record. |
| R56-P4 | **CONFIRMED for preflight; acquisition result UNMEASURED** | Patch application, zero removed lines, preprocessing, Fortran syntax, header parser, fresh target, and script-order controls pass.  Payload-ULP, producer-stamp, restart/mesh, and inherited-field admission remain fail-closed checks in `run.sh` and cannot be claimed before the record exists. |
| R56-P5 | **CONFIRMED** | The final `packages/` diff is empty and no scientific statement landed. |

## Controls and verification

- Boundary baseline: exit 1, `ACQUISITION_NEEDED`, 651 unequal active-U cells.
- Endpoint plant: a real active-U cell changes the count to 652 and exits 2.
- Acquisition preflight: `PREFLIGHT_PASS`; the exact patch applies to the
  shipped source, has 11 call sentinels, and removes zero source lines.
- The patched source and writer both pass preprocessing and `gfortran
  -fsyntax-only` against the producing configuration's includes.
- Focused round-56 tests: **7 passed**.  Ruff, Python compilation, shell syntax,
  and `git diff --check` pass.
- Shared-card battery: **170 passed**, 9 warnings, in 358.49 s.
- `tests/ocean/fidelity -n 12` collected 1,964 items and reproduced the
  registered xdist tail stall at 99%; before interruption it logged 1,941
  passed, 7 skipped, and the same five registered failures.  The five IDs
  rerun serially preserve their existing signatures: round-129 stale
  certification, round-51 private trace registry, SI3 scalar-math provenance,
  three worktree-stamp offenders, and the `hires_lane_surface` case-board row.
- Separate read-only `codex exec` review: **independent review unavailable
  in-sandbox** (`failed to initialize in-process app-server client: Read-only
  file system`).
- The default and this-receipt citation gates, including a shifted-line plant,
  are recorded in the final round commit.

## Scope ledger

**ASKED.** Continue the current step-level walk, name the first non-bit
boundary available from admitted evidence, and provide a fail-closed
acquisition when its operands are absent.

**UNASKED and unchanged.** No `packages/` file, model state, recipe, selector,
threshold, mask, stabiliser, ORCA2 carried entry, or sea-ice field changed.
ORCA2 remains labelled **given NEMO's entry** where its ladder is discussed;
the OVERFLOW result above is separately labelled **given NEMO's recorded
operands**.  The card's six ice selectors and `unmeasured_features` tuple stay
at `STOP_SELECTOR_GAP`.

## OPEN

1. The operator runs the committed round-56 `run.sh` under the new target.
2. The next round admits the self-describing record and walks, in compiled
   order, transport, curvature, selected face value, face flux, divergence,
   and RHS addition to the first non-bit statement.
3. Only after that statement is measured may the held source-ordered QCO pair
   be retried.  After the step-level walk closes, return to Decision 52's
   independent ORCA2 start and month-scale magnitude ranking.
4. Sea ice remains out of scope at `STOP_SELECTOR_GAP`.
