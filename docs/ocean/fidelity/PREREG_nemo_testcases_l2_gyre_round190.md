# Preregistration — round 190, production QSR source association

Committed before extending or running the production process trace.  Round 189
refuted the live-stretch attribution: exact NEMO `r3t(Kmm)` removes only
`1.3497213950746828e-05` of the maximum shortwave-row error.  Its source-order
rebuild is `99.99699%` closer to NEMO than the observed production bucket, so
this round separates the observer's process classification from the bridge
algebra.  Evidence lives under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round190/`.

No model physics, configuration, carried state, default, stabilizer, or NEMO
source is authorized to change.  The round extends the existing Round-186/188/
189 process trace and drives the same `LatLonCGridOceanModel.step ->
self._step_jitted` production closure.

## Compiled statement and model boundaries

NEMO calls the two-band shortwave routine once in the stage-3 tracer program;
that routine adds its rate directly to `pts(...,Krhs)` at
`GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/traqsr.f90:616-645`.
The corresponding legoESM bridge first replaces the Kbb shortwave component by
the Kmm component, then decomposes that complete source into surface and QSR
rates before rebuilding source-ordered process buckets.  The trace will return
the already-materialized bridge inputs and outputs, not recompute them in the
scorer.

## Frozen predictions and falsifiers

1. Before attribution, the extended production-JIT trace must reproduce Round
   189's actual-row/direct-rebuild split: `9,679/18,000` unequal cells and
   `2.467770444880557e-06 K` maximum.  Any change stops the scientific walk.
2. Register the exact production arrays `tendency_kbb`, `qsr_kbb`, `qsr_kmm`,
   `thickness_kbb`, `thickness_kmm`, `_process_qsr_kbb`,
   `_nemo_ws_process_surface_rate`, and `_nemo_ws_process_qsr_rate`.  Every
   scored array is masked to the NEMO wet extent and labelled production JIT.
3. Frozen source-reading prediction: `_process_qsr_kbb` is BIT with `qsr_kbb`
   on this resolved card, because its combined-physics function contains the
   same single shared `nemo_qsr_2bd` kernel and no explicit tracer-diffusion
   tendency.  Any unequal wet cell refutes this prediction and classifies the
   observer recomputation as the first boundary.
4. If prediction 3 holds, reconstruct `_nemo_ws_process_surface_rate` and
   `_nemo_ws_process_qsr_rate` from the returned operands using the exact bridge
   expressions.  BIT closure exonerates the bridge algebra and identifies the
   existing `Bqsr-Bsbc` observer row as a classification/association artifact;
   a non-bit reconstruction names the first failing bridge statement instead.
5. The existing production process plant must move the source-associated
   boundary, print `STATUS PLANT-FIRED`, and exit nonzero.  A plant that moves
   only an isolated scorer expression is invalid.
6. If all bridge identities are BIT, no compiled NEMO physics statement is
   named and the round is HELD with the observer correction as its OPEN item.
   If a bridge identity is non-bit, the first one in source order is named,
   cited, and remains HELD unless a one-variable NEMO-source-exact candidate
   passes the complete Decision-43/45/55/59 trajectory and card gates.
7. The exact-NEMO-ratio upstream walk remains prohibited: Round 189 measured it
   magnitude-inert.  No acquisition or configuration decision is expected.

The final diff receives a separate read-only Codex review.  A `DO NOT SHIP`
verdict blocks any landing.
