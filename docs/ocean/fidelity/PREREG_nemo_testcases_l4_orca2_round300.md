# ORCA2 round 300 preregistration — kt=7 W leaf split

Date: 2026-10-16. Frozen base: `deee20d23`. Scope: OMT-4, separately
labelled **independent** and **given NEMO's entry**. Round 299 named the
stage-1 vertical active-tracer transport `trp_w` as the only single-component
closer of the kt=7 fold-band growth and requested the source-complete record.
That record now exists and its committed admission reports
`PASS_R299_OMT4_KT7_W_ADMISSION`. This round admits it again from its
self-describing headers, replays the W builder offline from passive model
states, and walks the compiled statements in order. It changes no production
package, card, deck, selector, carried state, stabiliser, sea-ice field, or
`unmeasured_features` entry.

## Resolved compiled program

The record build is `ORCA2_OMIP_L4_R299W7`. Its resolved vector/C2 stage
program zeros the bottom U, V and W transports at
`BLD/ppsrc/nemo/traadv.f90:248-252`, calls transport-form `wzv` at
`:267-302`, and forms `pFw=e1e2t*ww` at `:314-315`.
`div_hor(np_transport)` evaluates the parenthesised U difference, the
parenthesised V difference, their sum, `r1_e1e2t`, live Kmm thickness and
mask at `BLD/ppsrc/nemo/divhor.f90:132-139`. The QCO bottom-up recurrence
adds `r1_Dt*e3t_3d*(r3t(Kaa)-r3t(Kbb))` at
`BLD/ppsrc/nemo/sshwzv.f90:271-299`.

The offline replay reuses the production shared implementations
`nemo_transport_wzv_divergence_level` and `nemo_qco_wzv_recurrence`; no
second W formula is introduced. The admitted fields cover the horizontal
transports, `r1_e1e2t`, `e1e2t`, reference `e3t_3d`, Kbb/Kmm/Kaa ratios,
T mask, pre-area `ww`, and final `trp_w` on both ranks.

## Frozen predictions and falsifiers

1. **R300-P1 — admission and controls.** Both rank records admit with 31
   fields, exactly-once 148x180 ownership, finite owned payloads, and
   byte-identical inherited frames/restarts. Both labels use one clean
   CPU/fp64/libm tree. A miss refuses every scientific number.
2. **R300-P2 — replay calibration.** The production helper replay from the
   candidate's passive kt=7 stage-1 operands reproduces its native `ww` and
   final `trp_w` to the campaign floor; replay from NEMO's recorded operands
   reproduces recorded `w_ww` and `trp_w` to the floor. A miss is an
   instrument failure, not a model finding.
3. **R300-P3 — first leaf.** The first non-bit boundary is predicted to be
   the transport-form horizontal divergence at `divhor.f90:132-139`, with
   the source-ordered U difference non-bit before the V difference and before
   the QCO recurrence. If U is exact, V is first, or both divergence terms
   are at the floor and the recurrence/area product is first, this prediction
   is retained as **REFUTED** and the measured statement becomes the finding.
4. **R300-P4 — causal arm.** Substituting the complete recorded divergence
   result into the candidate recurrence is predicted to take the candidate
   `ww` to the floor. If it does not, the Kbb/Kaa thickness-ratio recurrence
   is an independent member and is split in written order; no half is called
   causal merely because it moves the result.
5. **R300-P5 — labels and source order.** Both labels independently name the
   same first leaf. Bottom zero, U difference, V difference, horizontal sum
   and area division, live-thickness divide/multiply, Kaa-minus-Kbb stretch,
   bottom-up recurrence, and final `e1e2t*ww` are reported in compiled order.
   Label disagreement or a replay that skips a statement refuses the claim.
6. **R300-P6 — measurement only.** No constituent of the private fold unit
   lands in this round. Admission, label coverage, source order, passivity,
   candidate replay, oracle replay, first-leaf selection, Decision-127
   subnormal handling, and a normal one-bit perturbation each have a planted
   violation that must refuse.

## Stop conditions

A source-complete first leaf ends **HELD** with the compiled citation and its
magnitude; a missing operand ends **STOPPED_FOR_RECORD** with one fail-closed
rank-complete acquisition. The private fold unit remains atomic. Decision 126
remains pending and is not re-raised; shipped runoff and sea ice are unchanged.
