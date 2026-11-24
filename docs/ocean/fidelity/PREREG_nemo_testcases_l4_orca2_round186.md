# ORCA2 round 186 preregistration — live-thickness cell and halo re-test

Date: 2026-10-08. Base: `c57a4f96d5804e6d114f0000a922f65861f43aec`.
Every rung-0 number is **independent**: the card starts from its own corrected,
bit-exact initial state. Any rung-7 number is **given NEMO's entry** and is
reported separately.

## Frozen scope

Round 185 left the production independent month at its existing step-96
`raw-mesh e3w_int must contain only finite values > 0` refusal. This round
extends the round-177 production month instrument to report the refusing
interface and a ten-step-block growth table against NEMO's admitted rung-0
from-rest record. It then re-runs the already-defined private halo/V-transport
unit: raw reference face depth, no extra compact V mask, the seven-array
external-mode boundary association, and materialised `zhV`.

The search before implementation found the production month runner in
`nemo_testcase_l4_orca2_round177_independent_month_boundary.py`, the month
scoring helpers in `nemo_testcase_l4_orca2_round130_rung0_month_gate.py`, and
the private transport controls in the round-103/146 gates. They are extended
or called directly; no parallel trajectory harness is authorised.

No configuration value, forcing, carried state, stabiliser, public selector,
sea-ice selector, or `unmeasured_features` entry changes. No in-executable
operator observer is authorised. The ordinary returned state is the only
candidate-side measurement.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R186-P1 | Production retains the round-185 independent boundary. | Entry T/S/u/v/SSH are array-identical to NEMO; 95 steps complete; step 96 reaches the registered live-thickness refusal. | Any entry bit moves or an earlier/different terminal: **REFUTED**; name it and stop downstream claims. |
| R186-P2 | The first invalid live thickness is in the previously implicated `[86,159]` column. | The fail-closed diagnostic reports the first non-finite/non-positive `e3w_int` interface at horizontal cell `[86,159]`, with the value and vertical index. | Any other cell: **REFUTED** and retained; that measured cell owns the next walk. |
| R186-P3 | Finite error growth is already visible before the step-96 guard. | The fixed ten-step table for steps 10..90 plus step 95 reports finite max-absolute T/S/u/v errors against matching NEMO restarts, with at least one field growing between the last two available blocks. | Missing matching oracle checkpoints, a non-finite earlier row, or no late growth: **REFUTED**; report the exact record gap/result. |
| R186-P4 | The private halo/V-transport unit remains ineligible after the HPG landing. | Its independent ladder either refuses before kt=10 or fails Decision 96: no majority of RMS-moved rows toward NEMO, an exact-row loss, an earlier first debt, or a worse kt=10 stage-3 SSH maximum. | A complete ladder with majority toward, first debt toward/unchanged, no exact loss, and SSH maximum not worse: **REFUTED**; run the full shared landing gate before any production change. |
| R186-P5 | The private arm does not advance the independent month past production's boundary. | It refuses or becomes non-finite at step 96 or earlier; the failing step/cell and growth table are retained. | It advances beyond step 96 or completes: **REFUTED** and report the new boundary/score; do not infer eligibility without P4. |
| R186-P6 | The instrument is fail-closed and non-vacuous. | Plants for entry identity, boundary step, boundary cell, growth-row provenance, Decision-96 majority, exact-row loss, and SSH regression all fire. | Any plant stays green: instrument invalid; quote no scientific verdict. |

The exact guard site will be cited only after reading the executed legoESM
path. A landing, if unexpectedly eligible, still requires both ORCA2 ladders,
the independent month, GYRE year and ladder, DINO, tanks, shared-card census,
citation gates, focused tests, one fidelity battery, and separate read-only
Codex review. Failed predictions remain in the receipt.
