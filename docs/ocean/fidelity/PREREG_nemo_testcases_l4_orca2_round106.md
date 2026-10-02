# Preregistration — ORCA2 round 106 EEN accumulator discrimination

Date: 2026-10-02. Base: `edce5d5070a955c2508d7258aeacdb1994980be9`.
Scope is ocean-only measurement and, only if mechanically justified, one
source-cited EEN statement. Every ORCA2 number is **independent** because
hierarchy rung 0 starts from NEMO's own from-rest state. Sea ice, every card
selection, threshold, stabilizer, carried state, and the ORCA2
`unmeasured_features` tuple remain unchanged.

## Observed admission refusal

The operator-run round-105 target contains two rank-tagged operand records,
twenty restart shards, and a completed two-rank NEMO run. Rank 0's recorder
markers are in `ocean.output`; rank 1's markers are in
`run.user.stdout.log`. The round-105 launcher instead requires
`ocean.output_0001`, which does not exist, and refuses before the
self-describing record checker runs. Round 106 changes only that log-source
expectation and retains the recorded producer-content hashes; no NEMO rerun is
requested unless the existing payload or restart gate refuses.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R106-P1 | The existing round-105 record is admissible once markers are read from the two files NEMO actually produced. | Exactly one initialization and dump are associated with each distinct rank; both operand files parse; all eight payload plants fire; all 20 restarts are byte-identical to round 98; every recorded final coefficient is reconstructed bit-for-bit. | Duplicate/missing rank marker, parser refusal, plant failure, reconstruction mismatch, or restart difference: refuse the record and request only the newly named missing acquisition. |
| R106-P2 | Round-105 prediction R105-P5 holds: the first remaining non-fold signed-zero operand difference is in vertical accumulation, not final scaling. | At every registered non-fold final signed-zero mismatch, the paired scale bits equal legoESM's scale and the accumulator is the first unequal operand. | Any scale is first unequal: **REFUTED**; walk the exact compiled scale expression before changing production. |
| R106-P3 | Reproducing NEMO's accumulated zero signs is sufficient to close the active substep-1 U/V zero-sign rows without moving any magnitude. | A one-variable accumulator-sign substitution makes substep-1 U and V bit-exact and leaves all pre-existing coefficient magnitudes unchanged. | Any active substep-1 bit remains or any magnitude moves: **REFUTED**; retain the measurement and continue in source order without landing. |
| R106-P4 | The independent northern-fold `ffv_nw`/`ffv_ne` magnitude debt is not owned by accumulation/final scaling. | Its registered 66/67 magnitude support and values are unchanged by the sign-only accumulator substitution. | Any fold magnitude changes: stop and reconcile the association before a claim. |
| R106-P5 | The later 68-cell substep-2 U residual survives the source-exact accumulator sign substitution. | The row remains 68 active unequal cells with the registered maximum `2.9617669311254642e-8`; substep-2 V remains active-bit-exact. | Any different support/value: stop and name the moved boundary; do not combine owners. |

## Landing bar

A production statement lands only if it is the first unequal compiled NEMO
statement, is bit-exact given NEMO operands, and passes the rung-0 and rung-7
ten-step gates plus the standing GYRE, DINO, tank, citation, focused-test,
ocean-fidelity, and independent-review gates. Otherwise this round is
**HELD** with every failed prediction retained. No configuration choice is
authorized in this round.

## Post-discrimination preregistration — northern V-scale metric

Committed after R106-P1 through P5 were measured and before this arm. R106-P1,
P2, P3, and P5 are confirmed. R106-P4 is **REFUTED**: all non-fold scale bits
are exact, but `scl_v_nw` and `scl_v_ne` each differ at 68 northern-fold
cells; rebuilding with NEMO's accumulator retains exactly the registered
66/67 final-coefficient magnitude differences. Thus the first magnitude
statement is the compiled final V scale, not the recurrence.

The compiled statements read `e2u(ji-1,jj+1)` and `e2u(ji,jj+1)` for the
northwest/northeast V scales
(`ORCA2_OMIP_L4_R105EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:1282-1283`). At the
northern fold, `jj+1` is NEMO's U-stagger fold halo. The current literal
builder uses periodic `jnp.roll` for that north row. The one-variable arm
replaces only the top-row north-neighbour `e2u` with the existing grid's
U-stagger fold permutation; all other scale operands and rows remain fixed.

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R106-P6 | The U-stagger fold-halo metric is the sole owner of both northern V-scale debts. | The arm makes all eight recorded scales bit-exact, with zero non-fold or other-scale movement. | Any scale bit remains or any previously exact scale moves: **REFUTED**; hold and continue operand association. |
| R106-P7 | Exact fold scales close the 66/67 `ffv_nw/ne` coefficient magnitudes without changing their recorded accumulators or any other coefficient magnitude. | All eight final coefficients become magnitude-exact; only the already-registered accumulation zero signs remain. | Any magnitude remains or any new magnitude appears: hold without a production change. |
| R106-P8 | The independent 68-cell substep-2 U residual survives exact coefficients. | Exact NEMO coefficients still leave 68 active U cells, maximum `2.9617669311254642e-8`, while substep-2 V remains active-bit-exact. | Any different support/value: stop and name the moved consumer statement before landing. |

## Post-arm correction — measure the live accumulator directly

Committed after the production one-variable fold-scale arm. R106-P6 and P7
are **REFUTED**: the arm leaves the registered 66/67 fold coefficient
magnitudes. The offline arm that appeared exact had combined NEMO's recorded
accumulator with the fold-correct scale, violating the one-variable rule. The
production commit is retained in history and reverted; no physics lands from
that result.

The next discriminator exposes the current literal builder's accumulator and
scale without changing its default return or arithmetic, compares both against
the admitted record, and is reverted after measurement. This is
instrumentation, not a candidate landing.

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R106-P9 | The current northwest/northeast V accumulators carry a second northern-fold magnitude difference. | Direct internal comparison finds magnitude differences on the same fold support while all non-fold accumulator magnitudes are exact. | No fold accumulator magnitude difference: reconcile why the scale-only production arm failed before any new claim. |
| R106-P10 | Fold accumulation and final scale form a cancelling pair. | Substituting only NEMO's accumulator leaves the scale debt; substituting only NEMO's scale leaves the accumulator debt; substituting both makes both final coefficients magnitude-exact. | Either one-variable substitution alone closes the magnitude, or the pair remains non-exact: **REFUTED** and continue operand order. |
| R106-P11 | The first fold magnitude statement is in the vertical recurrence, before final scaling. | The first direct accumulator mismatch is mapped to the compiled `ffv_nw/ne` recurrence operands at `dynspg_ts.f90:1276-1277`. | Accumulators are exact: the final scale at `:1282-1283` remains first. |
