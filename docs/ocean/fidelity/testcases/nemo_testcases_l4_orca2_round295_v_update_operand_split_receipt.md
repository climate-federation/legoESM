# ORCA2 round 295 — vector-V update operand split

Date: 2026-10-16

Status: **HELD**.  On the independent OMT-4 kt=1 stage-1 external solve,
the source-ordered split of round 294's V-update boundary finds a signed-zero
bit difference in `zv_spg`, then the first valued operand in the compact
surface V mask.  Replacing only that mask with NEMO's raw `ssvmask` closes
the update from 35/8,589 unequal active faces to zero.  The maximum closed
difference is 0.00027511187960820976 m/s.  The mask remains private with raw
`hv_0` inside the complete fold unit; no package, card, configuration,
carried state, or NEMO record changes in this round.

Every scientific value below is labelled **independent OMT-4, kt=1 stage 1,
external substep 1**.  No given-entry result is mixed into the tables.

## Provenance and instrument

- Lane base: `7cf1ea9be`.
- Frozen preregistration:
  `PREREG_nemo_testcases_l4_orca2_round295.md`, commit `09d58aa0e`.
- Committed measurement gate: `5e2e78383`; non-vacuous closure-plant and
  classifier-control repairs: `b632e04fb`, `3d4f85d0a`.
- Evidence root:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round295/`.
- Inputs: round 294's immutable `measurement3.json`; the admitted round-222
  OMT-4 substep twins; and the admitted round-280 stage record.
- Execution: CPU, production JIT, fp64/x64, libm.  Every carried field in the
  observed and unobserved raw-`hv_0` completed states is array-identical.
- The OMT-4 selectors remain vector-form, with the literal fold unit and the
  completed-stage top-V exchange off.  Search-before-build reused the
  existing literal update, passive trace, record readers, mask bundle, and
  round-294 closure; it adds no executable observer.

## Source-ordered result

The compiled record evaluates
`va_e = (vn_e + rDt_e * (zv_spg + zv_trd + zv_frc)) * ssvmask` at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dynspg_ts.f90:666-678`.
The candidate replay equals its passive raw V exit on all 8,589 active
faces.  The NEMO replay equals the admitted post-association exit on all
8,589 active faces as well; its independently gated interior calibration is
0/8,489 unequal.  The two program points and both formulas are therefore
calibrated before the leaf attribution.

| operand, in written order | unequal / active support | maximum | argmax | classification |
|---|---:|---:|---|---|
| `vn_e` | 0 / 8,589 | 0 | — | exact |
| `rDt_e` | 0 / 1 | 0 | — | exact |
| `zv_spg` | 35 / 8,589 | 0 | fold band | signed-zero only |
| `zv_trd` | 0 / 8,589 | 0 | — | exact |
| `zv_frc` | 0 / 8,589 | 0 | — | exact |
| `ssvmask` | 35 / 8,589 | 1 | `[j=147,i=29]` | first valued, fold-local |

The baseline update against the source-exact replay is 35/8,589 unequal,
maximum 0.00027511187960820976 m/s and RMS
9.660386936122658e-06 m/s, with argmax `[j=147,i=54]`.  All 35 differences
are in the 100-face fold band; all 8,489 interior faces are bit-exact.

| one-variable substitution | unequal / active support | maximum (m/s) |
|---|---:|---:|
| NEMO `vn_e` only | 35 / 8,589 | 0.00027511187960820976 |
| NEMO `rDt_e` only | 35 / 8,589 | 0.00027511187960820976 |
| NEMO `zv_spg` only | 35 / 8,589 | 0.00027511187960820976 |
| NEMO `zv_trd` only | 35 / 8,589 | 0.00027511187960820976 |
| NEMO `zv_frc` only | 35 / 8,589 | 0.00027511187960820976 |
| NEMO raw `ssvmask` only | **0 / 8,589** | **0** |

Thus `zv_spg` owns only a separately registered signed-zero difference.
The first valued operand is the raw surface V mask, and its single-variable
substitution is sufficient at this statement.  That local closure is not a
trajectory landing predicate: prior rounds found compensating fold debt, so
the mask and raw `hv_0` remain one private atomic candidate.

## Frozen prediction ledger

| ID | Verdict | Mechanical result |
|---|---|---|
| R295-P1 | **CONFIRMED** | Parent and streams admit; CPU/fp64/libm, selectors, and whole-state passivity are pinned. |
| R295-P2 | **CONFIRMED** | Candidate raw replay is 0/8,589 unequal; oracle replay is 0/8,589 unequal, including 0/8,489 interior. |
| R295-P3 | **CONFIRMED** | First bit operand is signed-zero-only `zv_spg` on 35 fold faces; first valued operand is `ssvmask` on the same 35 faces. |
| R295-P4 | **CONFIRMED** | Raw `ssvmask` alone closes the update from 35 unequal active faces to zero. |
| R295-P5 | **CONFIRMED** | Nothing lands partially; both named fold operands stay private. |
| R295-P6 | **CONFIRMED** | All nine targeted plants print `STATUS PLANT-FIRED` and exit 2 at their intended invariant. |

The gate received two review fixes before final classification: the closure
plant is bound to the registered closing operand, and classified reports are
valid plant inputs.  These affect only the controls; `measurement.json` is
unchanged.

## Validation

- Accepted measurement and classification are `measurement.json` and
  `classification3.json`; SHA256
  `089e89915437678462568190d670d835ad155c93290fec5c1639bd721f46bcd6`
  and
  `b999eebaf9e91065b0b8e6759af4f21bf53582159d4507278fae3841589400ba`.
- Passivity, source-order, parent-closure, candidate-replay,
  oracle-interior-replay, signed-zero, locality, single-substitution, and
  one-bit plants each refuse at the named invariant; no plant stayed green.
- Focused round-293/294/295 and citation tests pass 26/26; log SHA256
  `4b2e930f3361689c46eef0f88662388b8314d26032bc55023ed934d9c1369172`.
- The single permitted `tests/ocean/fidelity -n 12` battery reached 99%.
  It printed the eight registered branch-tip reds: the GYRE round-129
  record hash, two round-33 stage-arm controls, the OMT-1 module-edge
  ratchet, round-228 citation mapping, dirty-escape scope, worktree stamp,
  and SI3 scalar provenance.  Its G1 stage-velocity worker then aborted in
  JAX compilation, producing two failed parameter markers and no terminal
  summary; after the remaining visible work passed, the stalled coordinator
  was interrupted.  The exact G1 node passes 3/3 in the required serial
  isolation (341.41 s).  Battery and isolation log SHA256 are
  `809f30e2f67b783e716a754532ff691ffb1a8cabe8852ae35d5652e0dab875b2`
  and
  `ab4fa41fcbfb0c8d9f8250add8deaffcf2c6de04d81ee2c3781af046d17572d5`.
- Independent review unavailable in-sandbox: `codex exec --sandbox
  read-only` exited before reading the diff because its app-server client
  attempted a write.  Independent review: operator.  Log SHA256
  `eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.
- No model file changed, so GYRE, DINO, tank, rung-0, and rung-10 trajectory
  gates are off-claim.  Their production configurations are untouched.

## Configuration choices

- **ASKED:** none.
- **UNASKED:** none.
- Decision 125's completed-stage top-V exchange, the resolved stage mask,
  raw `hv_0`, raw `ssvmask`, and the complete vector-form fold unit remain
  off in production.
- Decision 126 remains pending; shipped runoff is unchanged and is not
  re-raised.
- Sea ice and the shipped card's `unmeasured_features` tuple are unchanged.

## OPEN

1. Integrate raw `hv_0` and raw surface `ssvmask` into the complete private
   vector-form fold unit on the current tree, preserving all previously
   named parts.  Re-score it atomically on OMT-4 before any landing claim.
2. If the private unit completes its OMT-4 ladder, run the full Decision-96
   gates on OMT-1..4, independent rung 0, given-entry rung 10, GYRE, DINO,
   and tanks.  A partial depth or mask landing remains forbidden.
3. If it still refuses, name the first new passive boundary in compiled
   source order.  Keep the completed-stage top-V exchange off under
   Decision 125 and leave runoff unchanged while Decision 126 is pending.
