# ORCA2 round 279 — runoff-depth verification and top-V diagnosis

Date: 2026-10-15

Status: **HELD — configuration mismatch verified; held top-V growth traced to
an already-non-bit source row.**  No package/model configuration changed.
Every number below is **independent** (the OMT-4 card's own initial state),
unless the row explicitly says oracle/card configuration.

## Provenance

- Branch base: `4a7410281c6cec111a2f8857864cb41a68b0d2d8`.
- Preregistration: `PREREG_nemo_testcases_l4_orca2_round279.md`, committed as
  `5b086e210` before card instantiation or trajectory measurement.
- Gate: `nemo_testcase_l4_orca2_round279_runoff_topv_gate.py` at
  `bac8dab05`; CPU, JAX x64, fp64/libm.
- Oracle stage frames: the admitted OMT-4 record under
  `orca2_rounds/round222/acquisition/orca2_omt4_frames_10step_a_np2`.
- Evidence: `orca2_rounds/round279/measurement.json`, SHA256
  `0677235b1f58071364cde0036a1f2c5244adb8a2cad78a8f0197098cc7e2ba35`.
- Plants: runoff-depth plant SHA256
  `33ad4c85e91c877b835c9f1bb1020ecd67307e70c4f8ac87fb8d8277af94deb6`;
  fold-row plant SHA256
  `fb7e772813a770d2179796e45a8625904c60ff4ddc8e8e45225e11290ae02919`.
  Both exit 2 with `STATUS PLANT-FIRED`.

## Frozen predictions

| Prediction | Verdict | Measurement |
|---|---|---|
| R279-P1: NEMO resolves runoff at the surface only | **CONFIRMED** | both depth flags false; compiled false arm writes level 1 only |
| R279-P2: shipped rung 10 inherits 150 m; rung 0 and OMT-1..4 resolve no spread | **CONFIRMED** | exact card table below |
| R279-P3: first top-V-arm effect is kt=1 stage-1 V on the fold row only | **CONFIRMED** | 5,400 V values; all other fields 0 unequal |
| R279-P4: first downstream non-V effect is kt=1 stage-2 T/S on the fold | **CONFIRMED** | 1,319 T and 1,319 S values; both argmaxes at row 147 |
| R279-P5: adding NEMO's post-stage T/S association closes the tracer residual | **REFUTED** | T max 0.08246 -> 0.10322 K; S 1.52678 -> 1.93555 PSU |

The failed prediction remains part of the record; it is not rewritten into a
successful claim.

## 1. Runoff depth — verified hidden configuration choice

The ORCA2 deck does not override the reference flags.  The reference has
`ln_rnf_depth=.false.` and `ln_rnf_depth_ini=.false.` (with `rn_dep_max=150`
only for the inactive initialisation arm) at
`cfgs/SHARED/namelist_ref:474,478,480`.  The compiled executing branch tests those
two flags and, when both are false, changes only `phdivn(:,:,1)` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:259-282`.  Therefore the NEMO deck is
surface-only; the 150 m reference value is inert.

The mechanically instantiated legoESM cards are:

| Card | `runoff_depth_spread_m` | map | resolver result |
|---|---:|---|---|
| shipped rung 10 | 150.0 m | none | active 150.0 m spread |
| rung 0 | 0.0 m | none | surface-only (`None`) |
| OMT-1 | 0.0 m | none | surface-only (`None`) |
| OMT-2 | 0.0 m | none | surface-only (`None`) |
| OMT-3 | 0.0 m | none | surface-only (`None`) |
| OMT-4 | 0.0 m | none | surface-only (`None`) |

Thus the shipped ORCA2 card carries an uncited 150 m vertical runoff spread
that its NEMO deck does not execute.  It is a configuration choice, so this
round changes nothing.  Pick: set the shipped ORCA2 card to `0.0` with no map,
matching the compiled surface-only branch.

## 2. Held top-V exchange — first causal boundaries

The current OMT-4 card was cloned twice on the same tree; only
`rk3_fold_exchange_top_v_row` changed.  At kt=1:

| Boundary/field, arm ON minus OFF | unequal | max abs | argmax `[j,i,k]` |
|---|---:|---:|---|
| stage 1 T | 0 | 0 | — |
| stage 1 S | 0 | 0 | — |
| stage 1 U | 0 | 0 | — |
| stage 1 V | 5,400 | 0.0251382309206 m/s | [147,144,6] |
| stage 1 SSH | 0 | 0 | — |
| stage 2 T | 1,319 | 0.0824605139325 K | [147,131,0] |
| stage 2 S | 1,319 | 1.52675700945 PSU | [147,131,0] |
| stage 2 U | 1,294 | 2.18574339198e-4 m/s | [147,38,15] |
| stage 2 V | 6,698 | 0.0260481014549 m/s | [147,143,6] |
| stage 2 SSH | 0 | 0 | — |

This names the first arm-on/off boundary, not the later ten-step damage:
stage-1 V on the northern pivot row, followed by the stage-2 tracer consumer.

## 3. Source-ordered diagnosis

The compiled vector branch first writes the stage V velocity at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:467-470`, then applies the
barotropic column correction at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:513-537`, and finally associates
the completed U/V/T/S fields at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:795-796`.

The association statement itself is exonerated as the magnitude owner:

| Operand check, stage 1 V | unequal | max abs |
|---|---:|---:|
| candidate source row `j=146` vs NEMO | 1,440/5,400 | 0.0162073695895 m/s |
| candidate top row vs its own folded source | 0/5,400 | 0 |
| NEMO top row vs its own folded source | 0/5,400 | 0 |
| candidate top row vs NEMO | 1,440/5,400 | 0.0162073695895 m/s |

The top-row residual is therefore exactly the mirrored residual already
present in the completed source row.  Turning on the cited copy exposes that
upstream debt; it does not create it.  The missing partner is the source-row
completed-V unit (raw vector update plus barotropic correction), not another
fold copy.  This round does not claim which half of that two-statement unit
owns the 1,440 cells; that requires its operand split at the same matched
stage-1 program point.

The preregistered T/S partner was tested and refuted.  Applying the compiled
T-point association before replaying stage 2 increases the maxima:

| Field | raw arm vs NEMO | with T/S association vs NEMO |
|---|---:|---:|
| T | 0.0824624511707 K | 0.103218537088 K |
| S | 1.52677531470 PSU | 1.93555158811 PSU |

That association is a real NEMO statement, but it is not the compensating
partner for Decision 125 on this tree.

## Gates and tests

- Round-279 measurement gate: PASS; both independent plants fire.
- No model file changed, so no GYRE, DINO, tank, or ORCA2 trajectory was
  eligible to move or land in this round.
- Citation gate: PASS, 5/5 receipt citations mapped with no failure; the
  default receipt also passes 274/274 citations.  Shifting the runoff-branch
  citation by two lines makes the gate fail and exit 1.
- Focused citation/card tests: 42/42 pass in 344.92 s.
- Required `tests/ocean/fidelity -n 12` battery: reached 98%, then worker
  `gw9` died during JAX compilation of
  `test_the_step_19_tracer_cell_needs_the_stored_ratio`; xdist replaced the
  worker but did not terminate or print a failure summary, so the run was
  interrupted rather than allowed to leave stale workers.  The crashed test's
  two parameterisations both pass serially.  Serial isolation of the standing
  known-red files gives 22 pass and exactly three reproduced pre-existing
  failures: SI3 scalar-math V2 (`A MY_SRC is not verbatim`), the worktree-stamp
  grow-only ratchet (13 old unstamped emitters), and the round-35 scoped-dirty
  ratchet (14 old unscoped drivers).  None names or imports the round-279 gate.

## Independent review

Independent review unavailable in-sandbox.  The required
`codex exec --sandbox read-only` attempt returned: "failed to initialize
in-process app-server client: Read-only file system" before reading the diff.
No review verdict is claimed.

## OPEN

1. **Decision needed:** make the shipped ORCA2 ocean card surface-only for
   runoff (`runoff_depth_spread_m=0.0`, no map), matching the compiled deck.
2. Decision 125 remains held.  Next measurement: split the 1,440-cell stage-1
   V source-row debt between the vector update and its barotropic correction
   at the matched pre-`lbc_lnk` program point; do not score another copied row.
