# ORCA2 round 208 — OMT-0 midpoint-V history split

Date: 2026-10-09. Base: `a70cd7dfa`. Measurement gate commits:
`b9d751f51` and `510e1538b`. Acquisition handoff commits: `e3cc346e5` and
`6cf1b3b9f`. Status: **STOPPED_FOR_RECORD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round208/`.
All measured values below are **independent OMT-0**. Round 207 already proved
that independent and given-NEMO-entry OMT-0 have identical active entries and
row-direction maps. The shipped rung-10 card, sea ice, its selectors and its
`unmeasured_features` tuple did not change.

## Result

The substep-2 midpoint-V statement is exonerated. The compiled branch selects
Forward coefficients `(1, 0, 0)` for substeps 1 and 2 at
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:461-484`, ending in the
written V extrapolation. Both the instantiated
OMT-0 card and NEMO's recorded coefficients are bit-exact `(1,0,0)`.

The three recorded history inputs split as follows on the 8,554 wet V faces:

| input | unequal | maximum | disposition |
|---|---:|---:|---|
| current `vn_e` | 2,134 | `8.673617379884035e-19 m s-1` | first effective non-bit input |
| `vb_e` | 0 | 0 | bit-exact |
| `vbb_e` | 0 | 0 | bit-exact |

The candidate helper reproduces the candidate `va_e` bit-for-bit. Candidate
`va_e` equals current `vn_e` on every wet face; 55 dry/halo cells differ only
in zero-sign storage after the two zero-weight products. Replacing **only**
current `vn_e` by NEMO's recorded value closes `va_e` on the complete recorded
domain: 0/13,320 unequal. Replacing all three inputs also closes 0/13,320.
Therefore no coefficient, zero-weight history, multiply or add in the
midpoint statement creates the round-207 2,134-cell result.

The following history rotation is also exonerated. legoESM's substep-2 entry V
is array-identical to its substep-1 exit V on all 13,320 recorded cells, and
NEMO's corresponding arrays are also array-identical. This is the direct
assignment `va_e -> vn_e` at
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:781-789`.

The first unresolved upstream boundary is consequently substep 1's flux-form
V update. The executing OMT-0 branch is
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:681-702`, specifically
the V statement at `:698-701`, followed by its boundary association at
`:712-741`. The admitted ordered stream records `vn_e`, `zv_spg`, `zhvp2_e`,
`zv_trd`, `zv_frc`, the exit reciprocal and post-association `va_e`, but it
does **not** record three exact operands at this statement: entry `hv_e`,
back-interpolated `zhv_bck`, and `hv_0*(1+r3v(Kmm))`. Reconstructing those
would violate the one-recorded-operand rule, so the required update replay is
**UNMEASURED_WITH_SPEC**.

Artifacts:

- `round208/midpoint_v.json` — clean CPU/fp64/libm replay and closure;
- `round208/midpoint_v.log` — full gate output;
- `round208/plants.log` — five distinct firing plants;
- `round208/acquisition_preflight.log` — patch application plus Fortran syntax
  proof;
- `round208/acquisition_layout_plant.log` — missing-field plant refusal.

## Acquisition handoff

The committed launcher is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round208_midpoint_v_acquisition/run.sh`.
It uses a new `ORCA2_OMIP_L4_R208MIDV` target and a new
`orca2_omt0_midv_10step_np2` run directory. Its additions-only patch writes ten
self-describing arrays, one file per MPI rank, from substep 1: the three
missing depth operands plus every other V-update input and the pre-association
output. The writer uses the absolute, pre-created target path and refuses at
open time. The checker parses every field name and payload dimension from the
record header, proves exactly-once global rank coverage, requires all owned
payloads finite, and requires both kt=10 terminal restarts byte-identical to
the admitted round-203 baseline. Header, name, dimensions, truncation, rank,
restart-byte and non-finite plants are committed.

`run.sh --preflight-only` passes through patch application, layout checks,
preprocessing and `gfortran -fsyntax-only`. Its layout plant fires. Per the
sandbox PMIx prohibition, no `makenemo` or `mpirun` was attempted here.

## Frozen predictions

| ID | disposition |
|---|---|
| R208-P1 | **CONFIRMED**: twin streams, `(1,0,0)`, float64 and CPU/libm are bound. |
| R208-P2 | **CONFIRMED**: current `vn_e` alone carries all 2,134 wet differences; recorded `vn_e` alone closes `va_e` exactly. |
| R208-P3 | **CONFIRMED**: both candidate and oracle history rotations are 0/13,320 unequal. |
| R208-P4 | **UNMEASURED_WITH_SPEC**: the stream lacks `hv_e`, `zhv_bck` and `hv_0*(1+r3v(Kmm))`; the committed acquisition requests exactly them. |
| R208-P5 | **CONFIRMED**: five replay plants and the acquisition layout plant fire. The record-checker plants are committed and run after acquisition. |

## Validation and review

Focused round-208 tests pass 14/14. The measurement trace is passive: terminal
SSH, U/V external velocities and both transports are array-identical with and
without the returned trace. No `packages/` file changed, so no GYRE, DINO,
tank or ORCA2 trajectory moved and no shared-physics gate is invoked.

Independent review and the final citation/test battery are recorded by the
round's final validation commit.

## OPEN

1. The operator runs the reported acquisition with `--run`; the next round
   admits it and replays
   `ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:698-701` from all ten recorded arrays.
   If the source-written update closes, compare its pre-association output to
   the already-admitted post-association value and continue in source order.
2. OMT-1 remains blocked on OMT-0 reaching the bar or a complete cancelling
   unit. The pending ladder-order question from round 207 remains unanswered:
   whether vector-form momentum advection should enter before bottom drag.

No model or card change landed. No sea-ice choice was made.
