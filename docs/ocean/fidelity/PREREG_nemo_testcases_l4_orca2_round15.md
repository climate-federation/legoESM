# NEMO testcase Lane 4 — ORCA2 card round 15 preregistration

Date: 2026-09-24

Parent: `8f58d909ecdc068412e0cf92f0ac4446ebbe8408`

Status: **PREREGISTERED BEFORE ANY ROUND-15 MEASUREMENT.**

Round 14 exonerated the recorded depth-mean momentum forcing: substituting it
over rank 0 moved the `0.2448430937728719 m` end-of-step sea-surface
disagreement by only `2.7614e-07 m`.  It left two solver inputs unchecked:
the sea-surface freshwater forcing and the barotropic drag coefficients.
Round 15 substitutes those two operands first, then walks the split-explicit
solver in the compiled order to the first non-bit statement.

Every number in this round is labelled **given NEMO's entry** unless it is a
geometry-only operand, which is labelled **independent**.  Decision 52's
recorded entry sea surface is used.  The six sea-ice selectors and the card's
`unmeasured_features` tuple are frozen.  Decisions 54, 57 and 58 remain
pending and nothing in this round acts on them.

## Record and compiled statements

The record is the admitted two-rank ORCA1-ice run at
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2`.
The compiled source is
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90`.

| boundary | compiled statement | recorded operand |
|---|---|---|
| solver copy | `:287-291` copies `sshe_rhs`, `Ue_rhs`, `Ve_rhs`, `CdU_u`, `CdU_v` | `continuity_forcing` in `oracle_bt_ordered_operands_kt00000001.bin`; `cd_u`, `cd_v` in `oracle_slow_forcing_kt00000001.bin` and the frozen coefficient header of `oracle_bt_drag_operands_kt00000001.bin` |
| predictor | `:460-493` forms the mid-step velocity and sea surface | ordered stream and `oracle_bt_substeps_kt00000001.bin` |
| face depth and flux | `:505-536` forms the mid-step face depths and transports | ordered stream |
| continuity | `:550-558` forms the divergence and next sea surface | ordered stream |
| pressure gradient | `:601-616` back-interpolates sea surface and forms its gradient | ordered stream |
| Coriolis and drag | `:618-652` adds the live EEN trend and bottom drag | ordered stream plus drag stream |
| velocity update | `:666-679` applies pressure gradient, trend and slow forcing in vector form | ordered stream and sub-step stream |

The ordered record covers sub-steps 1 and 2 and rank 0 only.  The broader
sub-step record covers all 65 sub-steps and rank 0 only.  No full-domain claim
will be made from either stream.

## One-variable substitutions

1. Baseline: the production ORCA2 step from NEMO's recorded kt=1 entry.
2. Freshwater arm: replace only the external solver's `F_slow_eta` with
   `-ssh_frc`, because legoESM stores positive freshwater convergence while
   NEMO's continuity statement subtracts `ssh_frc`.
3. Drag arm: replace only the external solver's frozen positive face drag
   rates with `-CdU_u`, `-CdU_v`; the sign conversion is required because
   NEMO stores a non-positive coefficient and legoESM stores a positive
   damping rate.
4. Combined arm: apply both replacements, leaving entry state, momentum
   forcing, geometry, masks, timestep, sub-step count and every selector
   unchanged.

Each arm must print a no-op control using legoESM's own operand and must prove
that its replacement landed at the production solver boundary.  A
one-representable-value plant in each channel must move the scored result or
the channel is deaf and its substitution is unmeasured.

## Source-ordered solver walk

After the two substitutions, compare the production-JIT private trace against
the record in this order for sub-step 1, stopping attribution at the first
non-bit row:

1. `eta_entry`, `u_entry`, `v_entry`
2. predictor weights and histories, then `eta_mid`, `u_mid`, `v_mid`
3. mid-step U/V face depths
4. metric transports and continuity differences/divergence
5. sea-surface freshwater forcing
6. `eta_exit`
7. back-interpolated `eta_pgf`
8. pressure-gradient U/V
9. Coriolis U/V, drag U/V and combined trend U/V
10. slow momentum forcing U/V
11. `u_exit`, `v_exit`

The record's line order is the order above
(`dynspg_ts.f90:755-779`).  A later row cannot own the walk while an earlier
row is non-bit.  Sub-step 2 is reported only if every sub-step-1 boundary is
bit-exact.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R15-P1 | The baseline reproduces round 14's end-of-step rank-0 sea-surface maximum. | `0.2448430937728719 m` exactly, or within round 14's already-registered printed tolerance of `5e-5` relative. | anything else; all new comparisons stop as instrument drift. |
| R15-P2 | The freshwater forcing is already numerically negligible after round 13's runoff-water landing and differs only at Decision 57's reciprocal spelling. | candidate-versus-record max `<= 1e-18 m/s`; substituting it moves the end-of-step sea surface by `< 2.4e-02 m`. | either threshold exceeded; the freshwater operand is a contributor and owns the next walk. |
| R15-P3 | The frozen drag coefficient is not a majority owner of the sea-surface disagreement. | substituting `CdU_u/CdU_v` moves the end-of-step sea surface by `< 2.4e-02 m`. | movement `>= 2.4e-02 m`; drag owns the next walk. |
| R15-P4 | The two unchecked inputs together are not a majority owner. | the combined arm moves the end-of-step sea surface by `< 2.4e-02 m`. | movement `>= 2.4e-02 m`; the interaction owns the next walk. |
| R15-P5 | With both recorded inputs substituted, the first non-bit solver statement occurs during sub-step 1, no later than the continuity update. | first non-bit row is one of boundaries 1-6 above. | every boundary through `eta_exit` is bit-exact, which moves the walk to pressure gradient or momentum. |
| R15-P6 | The first non-bit row carries at least ten percent of the end-of-step sea-surface disagreement when propagated through the recorded sub-step recurrence. | a causal substitution of that row moves the end-of-step sea surface by `>= 2.4e-02 m`. | smaller movement; the row is a mismatch but not the owner, and the walk continues without a fix. |
| R15-P7 | The existing kt=1..10 ORCA2 ladder is unchanged. | `LADDER_MEASURED`, same first statement, kt=10 entry temperature `3.9430791763114783` on `430552` cells. | a refusal, an earlier first statement, or an unregistered moved row. |
| R15-P8 | No production statement lands in this round unless one source-ordered statement is proved bit-exact given NEMO's operands and passes the full ORCA2 and GYRE gates. | either no `packages/` diff, or all required landing gates pass. | any ungated model diff. |

Failed predictions remain in the receipt as **REFUTED**.

## Stop rules

- A record reader must validate magic, version, kt, rank-local dimensions,
  sub-step sequence, payload size, finiteness and EOF before returning a row.
- A comparison uses rank 0's owned 90 longitude columns and the record's own
  staggered mask; halos and rank 1 are not silently mixed into it.
- The first non-bit statement is cited from the compiled build that produced
  the record, and the cited branch must be the executing vector-form arm.
- No stabiliser, configuration selection, carried-state change or sea-ice
  change is allowed.
- If a model file changes, the base and tip GYRE 10-step and 30-day trajectory
  proofs are mandatory before landing.
- The citation gate must pass and its plant must fire on a real citation that
  the receipt renders.

## Choices

ASKED: substitute the two named recorded inputs, then walk the compiled solver
in source order.

UNASKED: none.
