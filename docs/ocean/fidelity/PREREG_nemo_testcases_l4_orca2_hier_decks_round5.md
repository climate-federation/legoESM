# ORCA2 hierarchy decks round 5 preregistration — rung 7 admission repair and rung 6 constant mixing

Date: 2026-10-01

Base: `c3089b635` (`fidelity/nemo-testcases-orca2-decks`).

Scope is NEMO-side only.  The existing operator run completed 240 steps and
wrote the rung-7 products, but round 4's clean admission refused on its exact
decimal rendering of the computed standard TKE mixing-length minimum.  This
round preserves that failed prediction, repairs only the resolved-output
predicate, admits the existing record without rerunning NEMO, and constructs
rung 6 by replacing TKE with constant vertical mixing as required by Decision
80.  It changes no legoESM package, card, recipe, physics, configuration,
threshold, or carried state.  Rungs 1 through 5 and main-lane rung 0 remain out
of scope.

All record claims are labelled **independent**: NEMO starts from each rung's
own from-rest T/S initialization.

## Frozen oracle reading

Round 4 predicted the false-IWM branch would print `rmxl_min=1e-2`.  The
compiled statement does not assign that decimal: it evaluates
`1.e-6_wp / (rn_ediff * SQRT(rn_emin))` and prints the binary64 result
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdftke.f90:835-841`).
The completed record resolves `rn_ediff=0.10000000000000001` and
`rn_emin=9.9999999999999995E-007`, hence prints
`rmxl_min=9.9999999999999985E-003`.  Round 4's exact-decimal predicate is
therefore **REFUTED**; its branch prediction and physical value are unchanged.

For rung 6, the compiled manager reads `ln_zdfcst` and `ln_zdftke` in the same
namelist (`zdfphy.f90:140-149`), requires exactly one closure, maps constant
mixing directly to `np_CST`, and calls the TKE initializer only for `np_TKE`
(`zdfphy.f90:262-270`).  Constant mixing also disables the shear-production
path (`zdfphy.f90:275-278`).  At every step the `np_CST` select arm makes no
closure call, retaining the initialized `avm_k` and `avt_k`
(`zdfphy.f90:330-343`); those arrays were built from `rn_avm0`, `rn_avt0`,
`nn_avb`, and `nn_havtb` at `zdfphy.f90:205-228`.  TKE restart output is
guarded by `ln_zdftke` at `zdfphy.f90:378-385`.

Decision 80 fixes the rung-6 delta: `ln_zdftke=true->false` and
`ln_zdfcst=false->true`, retaining `rn_avm0=1.2e-4` and `rn_avt0=1.2e-5`.
Every unnamed assignment, including `nn_havtb=1`, remains byte-identical to
rung 7.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| HD5-P1 rung-7 admission repair | The completed run is valid; only round 4's decimal-rendering predicate is wrong. | A source-exact predicate for `9.9999999999999985E-003` admits the existing record, all ten plants still fire, and no NEMO product changes. | Any other record predicate fails, any plant stays green, or a NEMO rerun is required. |
| HD5-P2 rung-7 consequences | IWM initialization and execution are absent; retained backgrounds and the computed standard minimum are active. | Resolved output and the invalid-IWM-group sentinel pass with the corrected exact print value. | An IWM path runs, a retained background differs, or the sentinel is read. |
| HD5-P3 one-module rung-6 delta | Rung 6 differs from rung 7 on exactly `namzdf.ln_zdfcst` false to true and `namzdf.ln_zdftke` true to false. | Complete physical-line and parsed-assignment gates report only those two rows; all other deck/build/input/protocol bytes are inherited. | Any additional assignment, line, CPP key, input, or protocol difference. |
| HD5-P4 constant closure | Rung 6 selects `np_CST`, does not initialize or step TKE, does not compute shear production, and retains initialized `avm_k`/`avt_k`. | Resolved output reports constant true/TKE false; invalid `namzdf_tke` sentinel is unread; TKE init, step, and restart messages are absent; compiled consumer inventory is pinned. | TKE namelist is read, any TKE call executes, or constant closure is not uniquely selected. |
| HD5-P5 retained coefficients | Rung 6 keeps `rn_avm0=1.2e-4`, `rn_avt0=1.2e-5`, `nn_avb=0`, and `nn_havtb=1`; the equatorial tracer-background shape therefore remains. | Deck equality and resolved output show those exact values and selectors. | Any retained value or selector differs. |
| HD5-P6 rung-6 record completeness | The reused two-rank binary will produce 480 finite self-describing frames, finite T/U/V/W month products, finite fp64 step-240 ocean restarts, and a complete SHA-256 inventory. | Header-driven parsing reaches physical EOF for every frame and all product/restart/inventory checks pass. | Missing or malformed frame, bad payload, non-finite value, wrong step/dtype, or incomplete ledger. |
| HD5-P7 acquisition disposition | No rung-6 record exists before the operator run. | Preflight and every synthetic violation pass/fire; round ends `STOPPED_FOR_RECORD` with a committed launcher. | An already-existing admissible rung-6 record is found. |

Failed predictions remain in the receipt as **REFUTED**.

## Controls and landing predicate

The repaired rung-7 control must still fire on a forced resolved-consequence
failure.  The rung-6 gate must refuse an extra deck delta, either missing
selector delta, a changed retained coefficient/shape, a wrong build pin, a
readable invalid `namzdf_tke` sentinel, an unexpected TKE path, malformed or
truncated self-describing payloads, a missing/non-finite frame, a non-finite
terminal state, a wrong terminal step, and an incomplete SHA-256 inventory.
Every plant must fire.

This round lands the admission repair, rung-7 evidence, and rung-6 deck,
manifest, gate, launcher, tests, and receipt only if rung 7 admits and rung-6
preflight and controls pass.  Rung 6 remains **UNMEASURED** until its
operator-run acquisition is admitted.  No GYRE run is required because no
model file changes.
