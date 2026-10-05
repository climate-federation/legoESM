# ORCA2 round 52 receipt — OVERFLOW stage-2 signed-zero vorticity statement

Date: 2026-09-27  
Base: `67669aa6ec39355eee076a711af16e07d7fffa49`  
Preregistration: `65e02a76c642a724f4acba04a69923b20858e951`  
Instrument: `f070338a1c6360ded4f41bfad35a214793a6c919`  
Disposition: **HELD**  
ORCA2 claim label: **given NEMO's entry** (Decision 52; no ORCA2 trajectory
was measured)  
OVERFLOW claim label: **given NEMO's recorded operands**

## Answer

Stage-2 vorticity is the first non-bit statement after round 51's exact HPG,
but its entire residual is signed zero.  On the 16,900 active u cells, NEMO's
vorticity call changes 16,135 values from `-0.0` after HPG to `+0.0` after
vorticity.  The maximum arithmetic difference is exactly `0.0`; the first
bit-unequal legoESM-local index is `[1, 31, 0]`.  The v domain has zero active
faces and remains `UNMEASURED_NO_ACTIVE_FACE`.

The frozen prediction that vorticity was bit-exact is therefore **REFUTED**.
This is not tolerance debt: bit identity distinguishes the two IEEE-754 zero
encodings.  The held QCO arm remains removed, no model statement lands, and no
ORCA2/GYRE trajectory claim is made.

## Compiled source and statement ownership

The producing build executes EOS, HPG, vorticity, then flux-form advection at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:343-358`.
The compiled stage swap makes stage-1 Kaa slot 3 the stage-2 Kmm slot 3 at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3.f90:200-207`; the gate
parses both record headers and refuses if those slots or extents disagree.

The running namelist selects ENS.  Its compiled dispatch calls `vor_ens` with
`ntot` at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynvor.f90:242-248`, while the
executing UP3 flux-form branch defines `ntot=np_CME` at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynvor.f90:866-869`.
The first observed non-bit statement is the u-accumulator assignment at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynvor.f90:666`: adding the
zero-valued metric/Coriolis term canonicalizes the recorded negative-zero HPG
accumulator to positive zero.

## Record and controls

The round-50 self-describing record remains `AT_BAR` at producer
`932cbfa9ec2f2fbcbf51a03ca8e46e5b39b78c62`.  All six records parse from their
own magic, header integers, field names, ranks, extents and payload lengths;
the round-52 gate does not predict a byte count.  The prerequisite replays
remain exact: QCO T/S 0/17,000 unequal, EOS rhd 0/17,000, and HPG u 0/16,900.

The original preregistered plant expected an exact vorticity row and is
**REFUTED on premise**.  The replacement non-vacuity control selects an active
cell that is bit-equal before the plant and moves it by one representable fp64
value.  The unequal count rises from 16,135 to 16,136 and the gate exits 2.
The 16,135 pre-existing signed-zero differences remain separately counted.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round52/`:
`vorticity_final.json`, `vorticity_final.log`, `vorticity_plant.json`,
`vorticity_plant.log`, and the verification logs named below.

## Frozen predictions

| ID | verdict | evidence |
|---|---|---|
| R52-P1 | **CONFIRMED** | Record admission is `AT_BAR`; active-u HPG remains 0/16,900 unequal; v remains no-active-face. |
| R52-P2 | **REFUTED** | Vorticity changes 16,135/16,900 active-u bit patterns, all `-0.0` to `+0.0`, with arithmetic maximum 0.0. |
| R52-P3 | **REFUTED on premise; replacement control fires** | The row was not exact.  Planting a previously equal active cell increases the unequal count by exactly one and exits nonzero. |
| R52-P4 | **CONFIRMED** | Final `packages/` diff against the round base is empty; no scientific statement lands. |

## Verification

- Focused round-50/51/52 plus citation-control tests: **33 passed**; round-52
  Ruff and `py_compile` checks pass.
- Shared-card battery: **170 passed**, 9 warnings, in 350.32 s.  DINO, both
  tanks, lock-exchange, and OVERFLOW card coverage therefore retain their
  prior test results.
- `tests/ocean/fidelity -n 12` collected 1,948 items, reached 99% with all
  emitted nodes green, then reproduced the documented xdist tail stall and
  was interrupted.  The five registered reds rerun serially retain their
  previous signatures: round-129 stale certification, round-51 private trace
  registry, SI3 scalar-math provenance, three worktree-stamp offenders, and
  the `hires_lane_surface` case-board omission.
- Separate read-only `codex exec` review: **independent review unavailable
  in-sandbox** (`failed to initialize in-process app-server client: Read-only
  file system`).
- Default-receipt and round-52 citation gates pass with no unmapped citation;
  the planted shift of the vorticity accumulator citation fires.
- No `packages/` file changed, so the GYRE trajectory/year, DINO, and tank
  landing gates are not triggered.  Sea ice, all six ORCA2 selectors, and the
  card's `unmeasured_features` tuple remain unchanged at `STOP_SELECTOR_GAP`.

## OPEN

1. Keep the source-ordered QCO arm held.  Its independent OVERFLOW T row still
   worsens, and this round establishes only a signed-zero downstream statement.
2. Walk the selected compiled ENS statement internally, in source order, to
   record the signs of `zwz`, `zuav`, their product, and the final addition at
   the first active-u cell.  The admitted round-50 record lacks those internal
   streams, so the next round must preregister an additions-only acquisition
   before proposing a signed-zero association change.
3. After the OVERFLOW ordered walk reaches a complete bit-exact pair or
   exhausts the recorded sequence, return to ORCA2's whole-card kt=1 stage-1 T
   owner, the independent Decision-52 initial state/year, and round-20 slow
   forcing.
4. Sea ice remains out of scope at `STOP_SELECTOR_GAP`.

ASKED: replay the next recorded statement in round 51's OPEN order.  
UNASKED: none.
