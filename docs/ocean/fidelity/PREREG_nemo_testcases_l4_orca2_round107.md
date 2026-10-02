# Preregistration — ORCA2 round 107 EEN U-accumulator operand split

Date: 2026-10-02. Base: `e72a4839eb6fdc1c5b531afce684240aabb9cdec`.
Scope is the first recorded non-bit rung-0 EEN statement from round 106:
`ffu_nw` accumulation. Every ORCA2 number is **independent** because hierarchy
rung 0 starts from NEMO's own from-rest state. The northern V fold pair and
the later 68-cell substep-2 U residual remain separate and are not landing
targets in this round.

## Compiled statement and one-variable arm

The compiled producer first forms `zpvo_nw` as three source-ordered
`ff_f / (e3f_0vor * (1 + r3f * fe3mask))` terms
(`ORCA2_OMIP_L4_R105EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:1231-1233`). It then
updates `ffu_nw` with live U thickness, live V thickness, the neighboring
`vmask`, and `zpvo_nw`
(`ORCA2_OMIP_L4_R105EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:1244`). NEMO applies
`umask` and `vmask` only inside each QCO stretch factor and applies the explicit
neighboring `vmask` once. It does not multiply either live thickness by its
own face mask after construction.

The current literal builder instead constructs both live face thicknesses
with an additional trailing own-face mask before the recurrence
(`barotropic_latlon_cgrid.py:986-987`). The arm removes only those two trailing
masks. The explicit neighboring mask, source association, vertical loop,
final scale, card, and every carried operand stay fixed.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R107-P1 | `zpvo_nw` is already bit-exact at every recorded recurrence point. | A source-order replay using the carried `ff_f`, `e3f_0vor`, `r3f`, and `fe3mask` has zero bit differences against a rank-complete oracle operand record, or the source-shaped arm closes the accumulator without changing this expression. | Any unequal `zpvo_nw` operand: stop at its first unequal division/addition and do not attribute the recurrence to thickness masking. |
| R107-P2 | The extra trailing own-face masks own all 438 `ffu_nw` accumulator magnitude differences. | Removing only those masks changes no already-equal magnitude and makes the admitted NEMO `acc_u_nw` magnitude-exact. | Any of the 438 magnitudes remains or any new magnitude difference appears: **REFUTED**; retain the measurement and request/dump the first missing per-level operand if the existing record cannot discriminate it. |
| R107-P3 | The arm also reproduces NEMO's `ffu_nw` signed-zero accumulation. | `acc_u_nw` becomes bit-exact, from the registered 3,953 unequal bits. | Magnitudes close but bits remain: **REFUTED AS WRITTEN**; hold and walk the first signed-zero term/accumulation boundary. |
| R107-P4 | The source-shaped thickness change is dynamically inert on GYRE because it changes only coefficients on masked faces. | Certified GYRE ten-step rows, day-30/240/360 values, and residual digests are unchanged. | Any GYRE movement invokes the full Decision 43/45/55/59 bar; a failing row holds the arm. |
| R107-P5 | No certified ORCA2 rung-0 or rung-7 state row leaves its current status. | Both kt=1..10 gates retain every AT-BAR row and no first-over-bar step moves earlier; every moved debt row is registered. | Any AT-BAR loss or earlier first-over-bar: hold without landing. |

## Landing bar

The two trailing-mask deletions are one compiled statement transcription and
land only if the production accumulator is bit-exact given NEMO's operands,
all touched-card gates pass, the GYRE year satisfies its standing mechanical
bar, DINO/tanks/generic-card tests pass, the citation gate and its plant fire,
the focused and prescribed ocean-fidelity batteries complete as required, and
the independent read-only review is clean. Otherwise the round is **HELD**.

No configuration choice, threshold, stabilizer, carried-state change, sea-ice
change, or `unmeasured_features` change is authorized.
