# ORCA2 round 75 preregistration — hybrid-correction cancelling pair

Date frozen: 2026-09-29

Base: `ef8ea59eb8f2e7004329843b875a6a1706ffe1bc`

Claim label: **independent**. legoESM starts from the ORCA2 card's own ocean
state and exact kt=1 surface operands. NEMO operands are substituted only in
the three explicitly labelled diagnostic arms below; the baseline remains the
independent production step.

Sea ice remains out of scope. The card's exact six-entry
`unmeasured_features` tuple, selectors, 10,800 s step, thresholds, and carried
state are frozen. No model, configuration, or NEMO source change is authorized.

## Existing evidence, not predictions

Round 74 found the first non-bit primitive inputs to the active hybrid
barotropic correction: `un_adv` differs on 8,568 / 8,568 active rank-0 U
columns (maximum `13.240378093773609 m3 s-1`) and inverse depth differs on
8,568 / 8,568 (maximum `1.731223240825086e-05 m-1`). Their product enters the
first non-bit arithmetic statement, `zub`, which differs on 8,568 / 8,568
(maximum `0.01834375357001168 m s-1`). The downstream `zFu` differs on
226,236 / 226,236 active 3-D U cells (maximum
`321212.60790659266 m3 s-1`). Entry velocity, mask, metric, and barotropic
velocity are bit-exact.

The search-before-build audit found and will reuse the round-74 production
runner, self-describing record reader, raw-bit scorer, support census, card
execution census, and the shared production helpers for source rounding,
corrected velocity, and metric transport. No second record parser, transport
kernel, model hook, or configuration path will be created.

## Compiled statement and arms

The compiled ORCA2 selector fixes `n_baro_upd=np_HYB` at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:45-49`.
The executed source statement is

`zub = un_adv * (r1_hu_0 / (1 + r3u(Kmm))) - uu_b(Kmm)`

at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:274-284`,
followed by the source-ordered metric transport in the same span.

The gate holds entry velocity, mask, metric, live face thickness, and
barotropic velocity fixed and scores these four arms on NEMO's recorded 3-D
activity mask:

1. `baseline`: live `un_adv`, live inverse depth;
2. `oracle_un_adv`: NEMO `un_adv`, live inverse depth;
3. `oracle_inverse_depth`: live `un_adv`, NEMO inverse depth;
4. `oracle_pair`: NEMO `un_adv`, NEMO inverse depth.

Each arm scores `zub`, corrected velocity, and `zFu` by raw fp64 bits, maximum
absolute error, ULP distance, and RMS error. The pair claim is decided at
`zub`; downstream rows show what debt remains after that statement closes.

## Frozen predictions and falsifiers

1. **Baseline reproduction.** The round-75 baseline predicts round 74's
   `zub` and `zFu` count and maximum exactly. Any changed tuple refuses the
   instrument.
2. **Pair closure.** Supplying both recorded operands predicts 0 unequal
   `zub` columns and 0 unequal corrected-velocity cells. Any unequal cell
   invalidates the transcription or the claim that the other operands are
   exact.
3. **`un_adv` single arm.** Supplying only recorded `un_adv` predicts a
   non-bit `zub` whose RMS error is larger than baseline. Exactness or an RMS
   no larger than baseline is **REFUTED** and retained.
4. **Inverse-depth single arm.** Supplying only recorded inverse depth predicts
   a non-bit `zub` whose RMS error is larger than baseline. Exactness or an RMS
   no larger than baseline is **REFUTED** and retained.
5. **Cancelling-pair verdict.** Predictions 3 and 4 together predict two-sided
   cancellation: repairing either input alone worsens the statement, while
   repairing both closes it. If either single arm does not worsen, the
   two-sided claim is **REFUTED** and the lower-residual arm owns the next
   compiled-order walk.
6. **Downstream thickness debt.** The paired `zFu` arm predicts DEBT because
   independent entry SSH already makes live `e3u(Kmm)` non-bit. Exact paired
   `zFu` is **REFUTED** and retained.
7. **Disposition.** This measurement-only round predicts **HELD**. No model,
   configuration, selector, threshold, stabiliser, carried-state, or sea-ice
   change is authorized.

Failed predictions remain **REFUTED**. Exact closure, directional RMS,
downstream debt, calibration, and execution-scope predicates are separate.

## Controls and validation

- A one-ULP mutation of the exact paired `zub` must fire.
- Relabelling an individual arm as the pair must fire the pair-closure gate.
- A changed round-74 baseline tuple and a false card execution selector must
  each fire.
- Run focused round-74/75 and citation tests, both citation gates with a real
  firing citation plant, shared-card and tank batteries, and
  `tests/ocean/fidelity -n 12` once, one pytest battery at a time.
- Request the required separate `codex exec --sandbox read-only` review and
  record its verdict or the exact in-sandbox failure.

ASKED: measure the two-input cancelling pair at the first non-bit independent
ORCA2 hybrid correction and name which operand owns the next walk.

UNASKED: model arithmetic, configuration, NEMO output, selectors, thresholds,
carried state, sea ice, and the held tracer QCO/RK statement.
