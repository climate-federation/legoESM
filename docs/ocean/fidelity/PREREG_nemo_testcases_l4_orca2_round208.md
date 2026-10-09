# ORCA2 round 208 preregistration — OMT-0 midpoint-V history split

Date: 2026-10-09. Frozen base: `a70cd7dfa`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round208/`.
Every trajectory number is labelled either **independent OMT-0** or **given
NEMO's entry**; this round reads only the admitted independent kt=1 operand
record because round 207 proved the two OMT-0 labels have identical active
entries and row-direction maps. The shipped rung-10 card, sea ice, its six
selectors and its `unmeasured_features` tuple remain unchanged.

## Frozen source order and evidence

The compiled record build selects Forward extrapolation for external substeps
1 and 2: `za1=1`, `za2=za3=0` at
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:461-469`, then writes
`va_e = za1*vn_e + za2*vb_e + za3*vbb_e` at `:480-485`. After the completed
substep it rotates `va_e` into `vn_e` at `:787-789`. The preceding V update
is the flux-form statement at `:681-702` because OMT-0 resolves
`ln_dynadv_vec=.false.` and `lk_linssh=.false.`.

Round 207's admitted post-landing replay measures substep-2 `va_e` unequal on
2,134 wet cells with maximum `8.673617379884035e-19 m s-1`; substituting that
single recorded product operand closes `zhV` bit-exactly. This round does not
revisit the already-exact `e1v` or `zhvp2_e` operands. It replays the midpoint
expression offline from the passive completed-state trace and the admitted
self-describing ordered record. No in-executable observer is permitted.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R208-P1 | The record and resolved branch are unchanged. | Twin ordered streams are defined-byte identical; substep 2 coefficients are bit-exact `(1,0,0)` in both NEMO and the instantiated OMT-0 card; CPU fp64/libm is printed. | Any header, twin, coefficient, dtype or resolved-card difference: **REFUTED**; read no operand result. |
| R208-P2 | The midpoint arithmetic does not create the 2,134-cell debt. | Candidate `vn_e` is the first effective non-bit input; candidate midpoint equals candidate `vn_e` on every wet cell; replacing only `vn_e` by NEMO's recorded value closes recorded `va_e` bit-exactly. | A coefficient, zero-weight history, or written add/multiply association remains non-bit after recorded `vn_e`: **REFUTED**; that first statement owns the walk. |
| R208-P3 | The history rotation carries rather than creates the debt. | Candidate substep-2 `vn_e` is array-identical to candidate substep-1 `va_e`, and NEMO's same pair is array-identical; their unequal-cell count and maximum equal the midpoint input row. | Either assignment changes bits: **REFUTED**; the association/record timing owns the walk. |
| R208-P4 | Given NEMO's recorded substep-1 operands, legoESM's literal flux-form V update reaches NEMO's recorded `va_e` at the fixed floor. | The offline update replay is at or below `2e-10 m s-1`, and the report lists every substituted operand plus any unavailable operand. | An over-floor result names `dynspg_ts.f90:698-701` as the first numerical statement; a missing operand yields **UNMEASURED_WITH_SPEC**, never a reconstructed oracle value. |
| R208-P5 | The gate binds. | Header, Forward-coefficient, candidate-rotation, all-recorded-midpoint and update-closure plants each refuse at their named predicate. | Any plant stays green: no claim is citable. |

## Landing predicate

If the midpoint expression itself is non-bit given NEMO's operands, a source-
cited one-statement correction may land only under Decision 96's OMT-0,
hierarchy rung-0, shipped rung-10, GYRE, DINO and tank gates. If the expression
is exonerated, this is a measurement-only HELD round and the next round starts
at the first upstream operand boundary. OMT-1 does not begin until OMT-0 reaches
the bar or a complete cancelling unit. The pending operator question about
reordering OMT-1 is not answered or acted on here.

ASKED choices: Decision 103's OMT-0 walk and Decision 96's landing rule.
UNASKED choices: empty.
