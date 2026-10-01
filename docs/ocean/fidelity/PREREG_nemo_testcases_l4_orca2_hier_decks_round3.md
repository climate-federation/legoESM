# ORCA2 hierarchy decks round 3 preregistration — rung 9 admission and rung 8 differential-mixing-off deck

Date: 2026-10-01

Base: `86a2b4e5c` (`fidelity/nemo-testcases-orca2-decks`).

Scope is NEMO-side only.  Round 2's frozen HD2-P3 through HD2-P5
predictions govern admission of the operator-produced rung-9 record.  This
round then constructs hierarchy rung 8 by removing the three selectors grouped
by Decision 80's differential-mixing rung: double diffusion, river-mouth
diffusivity, and differential internal-wave T/S mixing.  It changes no
legoESM package, card, recipe, physics, configuration, threshold, or carried
state.  Rungs 1 through 7 and main-lane rung 0 remain out of scope.

All record claims are labelled **independent**: NEMO starts from each rung's
own from-rest T/S initialization.

## Frozen oracle reading

The compiled vertical-physics manager reads `ln_zdfddm` in `namzdf` and, when
false, copies `avs=avt` instead of calling double diffusion
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:140-149,255-260,361-368`).
The implicit tracer solve then reuses the temperature matrix for salinity when
double diffusion is off
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/trazdf.f90:171-180`).
The retained `rn_avts` and `rn_hsbfr` values are therefore inert.

The compiled runoff initializer reads `ln_rnf_mouth`, `rn_hrnf`, and
`rn_avt_rnf` together
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcrnf.f90:308-320`).
When the selector is false it zeros `rnfmsk`, `rnfmsk_z`, and `nkrnf`
(`sbcrnf.f90:504-535`), so the river-mouth increment guarded at
`zdfphy.f90:353-357` is absent and both retained numeric values are inert.

The compiled internal-wave module reads `ln_tsdiff` from `namzdf_iwm`
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfiwm.f90:421-427`).
When false, its salt/heat ratio is exactly one and the same wave diffusivity is
added to `avs` and `avt` (`zdfiwm.f90:301-316`).  Internal-wave mixing itself
stays on, so its molecular background reset remains active
(`zdfphy.f90:282-284`; `zdfiwm.f90:438-447`).

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| HD3-P1 rung-9 admission | Round 2's operator record satisfies every frozen no-ice, completeness, and resolved-consequence predicate. | The committed round-2 gate passes normally and all nine record-stage plants fire. | Any plant stays green or any record predicate fails. |
| HD3-P2 one-module deck delta | Rung 8 differs from rung 9 in exactly three parsed assignments and three non-comment lines: `namsbc_rnf.ln_rnf_mouth`, `namzdf.ln_zdfddm`, and `namzdf_iwm.ln_tsdiff`, each true to false. | Complete line and assignment-map gates report only those three deltas; every other byte-level assignment, build pin, input, and protocol value is inherited. | Any other assignment, physical line, CPP key, input, or run-protocol difference. |
| HD3-P3 compiled consequences | Rung 8 retains internal-wave mixing and its molecular background reset, but has no river-mouth mask/increment, no double-diffusive update, and no differential salt/heat wave ratio. | Resolved output reports all three selectors false and `ln_zdfiwm=true`; source-path gates pin zero river-mouth masks, `avs=avt` before waves, unit wave ratio, and the unchanged background reset. | Any removed arm executes, internal-wave mixing turns off, or the background reset changes. |
| HD3-P4 retained parameters inert | `rn_hrnf`, `rn_avt_rnf`, `rn_avts`, and `rn_hsbfr` retain their rung-9 values but cannot affect rung 8 because their guards are false. | Exact deck diff leaves all four values unchanged and compiled-branch checks establish their only consumers are skipped. | A value changes or an unguarded executing consumer is found. |
| HD3-P5 record completeness | The reused two-rank instrumented binary produces 480 finite self-describing operand frames, finite T/U/V/W month products, finite fp64 step-240 ocean restarts, and a complete SHA-256 inventory. | Header-driven parsing reaches physical EOF for every rank-step frame and all product/restart/inventory checks pass. | Missing or malformed frame, bad payload, non-finite value, wrong step/dtype, or incomplete ledger. |
| HD3-P6 acquisition disposition | No rung-8 record exists before the operator run. | Preflight and every synthetic violation pass/fire; round ends `STOPPED_FOR_RECORD` with a committed launcher. | An already-existing admissible rung-8 record is found. |

Failed predictions remain in the receipt as **REFUTED**.

## Controls and landing predicate

The rung-8 gate must refuse at least a fourth deck delta, a missing required
selector delta, a changed retained parameter, a wrong build pin, malformed or
truncated self-describing payloads, a missing/non-finite frame, a non-finite
terminal state, a wrong terminal step, a wrong resolved consequence, and an
incomplete SHA-256 inventory.  Every plant must fire.

This round lands rung-9 admission evidence plus the rung-8 deck, manifest,
gate, launcher, tests, and receipt only if rung 9 admits and rung-8 preflight
and plants pass.  Rung 8 remains **UNMEASURED** until its operator-run
acquisition is admitted.  No GYRE run is required because no model file changes.
