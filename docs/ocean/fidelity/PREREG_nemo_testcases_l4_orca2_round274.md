# ORCA2 round 274 preregistration — import the certified G1 statement set

Date: 2026-10-13

Lane base: `c0638af25a84dd4d522374c918dc3a967fd54f73`

Labels: **independent** and **given NEMO's entry**, always scored separately.

## Frozen unit

The geometry-ladder G1 tip `92435ddbde02dd6f6a3f44ec3ae36ba38f60b2dd`
certifies the TSUNAMI program on ORCA2's horizontal grid through 100 steps.
ORCA2 round 270 already imported G1 rounds 2--5.  This round imports and
selects the remaining cited, G1-certified statements from rounds 6--9:

1. the split-explicit EEN thickness-ratio fold exchange and its F-mask pivot
   row;
2. the literal stage-1 right-hand side and written stage-velocity update;
3. the written-order EEN vorticity trend, the stage-3 barotropic residual,
   the post-stage top-V exchange, and the interpolated stage F ratio;
4. the split-explicit thickness ratios in written order and the stored
   T-point ratio product;
5. the interpolated stage T/U/V ratios handed to their readers and the
   rounded density depth product.

The implementation is imported from the landed G1 commits, not re-derived.
Every new selector remains false by default.  The ORCA2 and OMT cards select
the statements their resolved branches execute.  Sea ice, decks, entry state,
carried state, and `unmeasured_features` do not change.

## Frozen predictions and falsifiers

### R274-P1 — import completeness

Prediction: all 13 statements above are present once, the current six
round-270 G1 selections remain selected, and every ORCA2/OMT card constructs
without a hidden default.  The card census names which statements each card
executes.

Falsifier: a landed G1 statement is absent, duplicated, silently defaulted,
selected on a branch that does not execute it, or changes an ice selector.

### R274-P2 — Decision-96 census

Prediction: for OMT-1 through OMT-4 under both labels, independent rung 0,
and given-entry rung 10, the imported unit has a strict majority of
RMS-changing rows toward NEMO, loses no exact row beyond the `2e-10` floor,
and moves the first over-bar row toward or leaves it unchanged.  Every
bit-touched/RMS-equal row is registered separately.

Falsifier: any exact-row loss, an earlier first-over-bar row, or toward rows
not exceeding away rows among RMS-changing rows.  A diagnostic maximum is
reported but is not an extra veto, per Decision 96 and round 270.

### R274-P3 — shared-card controls

Prediction: GYRE's certified ladder/year, the DINO month, and the generic,
lock-exchange, overflow, and UP3-scope gates are unchanged because those cards
leave the imported selectors off.  Their expected GYRE T RMS values are
`2.3432419318363155e-06`, `6.5816987106668941e-05`, and
`5.4077212586815052e-05 K`; DINO remains below `2.244317642e-03 K`.

Falsifier: any shared-card digest/row moves, DINO exceeds its bar, or a tank
gate changes status.

### R274-P4 — ORCA2 boundary remeasurement

Prediction: the matched stage-2 after-VOR row on the imported tree is no
worse than the landed-tree maximum `1.2863307536456779e-8 m s-2` under each
label.  After the import, the first non-bit ORCA2 boundary is re-ranked from
scratch at matched program points; no round-271--273 EEN leaf attribution is
carried forward unless the same boundary and value reproduce.

Falsifier: the after-VOR maximum grows, an earlier source-ordered boundary
appears, or the two labels disagree without a traced entry-state cause.  Any
such result is retained as a refutation rather than edited out.

### R274-P5 — mechanical gates

Prediction: the cumulative citation audit has no failure or unmapped anchor,
its real shifted-line plant fires, focused G1/ORCA2 tests pass, the 128-test
push battery passes, and the required ocean-fidelity battery introduces no
new reproducible red beyond the campaign's registered set.

Falsifier: an unmapped citation, a plant that stays green, a focused failure,
or a new reproducible battery failure owned by the import.

## Landing rule

Land the complete selected G1 import only if R274-P1, P2, P3, and the claimed
parts of P5 pass.  If the imported statements pass Decision 96, their own G1
bit-exact rows are sufficient even if the downstream ORCA2 boundary remains
non-bit.  After landing, remeasure R274-P4; acquisition is requested only if
the remeasured first boundary still needs a stream not already present.

