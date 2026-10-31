# SI3 round 29 (`si3_r29`) — O17 design and record audit receipt

Date: 2026-10-31. Base: `920c61d884384e18470962568cda8295f0dfbe89`.
No NEMO build, launch, or integration was run.

## Outcome

**ACQUISITION_NEEDED; O17 composition held.** The admitted O8--O16 master has
the correct unmodified production deck, 100-step window, dynamics disposition,
thermodynamic stage frames, exchange outputs, and restart. It does not have a
complete ice-step-entry state or raw O7 meteorology at every step. The O1--O7
records do not fill those holes over the same 100-step production trajectory.

There is also no existing legoESM production entry point that performs the O17
program. The current dispatcher accepts a typed finalization forcing and calls
O15, O16, and the stress tail, while its earlier O2 forcing path raises
`NotImplementedError` (`packages/ice/legoesm/ice/sea_ice.py:636-708`). O4--O16
therefore do **not** already score through one common full-step entry point.
The exact missing piece is a typed `si3_orca2_o17` production composition entry
that orders the already admitted O4 Prather, O5 H79+aEVP, O6 redistribution and
post-dynamics correction, live O7 surface step, and O8--O16 thermodynamics and
finalization. This is composition wiring and a forcing/response contract, not a
new physical operator.

The round instruction says to stop and name the missing piece if composition
needs one. Consequently this round did not invent that entry point, did not
write an acquisition driver whose read-set would precede the missing production
contract, and did not run the requested one-step gradient program. An O17
acquisition remains required after that entry point fixes the authoritative
step-entry and live-forcing pytrees.

## Measure-first record audit

The target oracle row is the shipped `ORCA2_SAS_ICE` deck: steps 1--100 and
`rn_Dt=10800` (`cfgs/ORCA2_SAS_ICE/EXPREF/namelist_cfg:21-28`),
`nn_fsbc=1` (`cfgs/ORCA2_SAS_ICE/EXPREF/namelist_cfg:59-61`), and CPP keys
`key_si3 key_linssh key_vco_1d3d key_xios key_RK3`
(`cfgs/ORCA2_SAS_ICE/cpp_ORCA2_SAS_ICE.fcm:1`). The O8--O16 acquisition was
preregistered as the unmodified production ice namelist, production dynamics
on but outside stage projections, O7-completed surface products as its forcing
boundary, and 100 steps (`docs/ice/fidelity/si3_r18_o8_o16_master_preregister.md:3-24`).
Those identities agree with O17; the forcing boundary does not.

| Evidence | Deck / window | What it proves | O17 missing data |
|---|---|---|---|
| O4 admitted record | Prather-only diagnostic overlay, 2 steps | Five-category Prather is 0 ULP for both sweep orders; its manifest explicitly disables all other dynamics and thermodynamics (`docs/ice/fidelity/si3_r25_o4_score_receipt.md:3-6,19-29`). | Not the production deck or 100-step carry; no live O7 forcing. |
| O5 admitted record | Dynamics diagnostic overlay, 3 steps | H79+aEVP is 0 ULP on its registered three-step production-JIT comparison; thermo and redistribution are disabled (`docs/ice/fidelity/si3_r12_o5_acquisition_receipt.md:3-14,26-34`). | Not the production deck or 100-step carry. |
| O6 receipt | Local model admission only | The receipt explicitly leaves ORCA2 NEMO-record bit exactness plausible because no ORCA2 record exists (`docs/ice/fidelity/si3_r14_o6_receipt.md:252-255`). | No O6 production record. |
| O7 admitted record | Unmodified production deck, 2 steps | All 91 scored leaves at each of `kt=1,2` are 0 ULP, including composed live surface stages (`docs/ice/fidelity/si3_r13_o7_admission_receipt.md:3-11,64-90`). | Raw meteorology and computed products stop after step 2. |
| O8--O16 master | Unmodified production deck, 100 steps | Every thermodynamic boundary and the O16 exchange/tau boundary is registered; restart pairs are byte-identical (`docs/ice/fidelity/si3_r18_o8_o16_master_preregister.md:15-24,40-58`). | No full pre-dynamics ice-step-entry carry, Prather moments, dynamic stress carry, or raw O7 meteorology. |

The master schema makes the gap mechanical rather than inferential. Kind 21 is
only the thermodynamic `STATE` tuple immediately before category work; kinds
18/15/16/19/20 cover pre-aggregation, flux, and stress boundaries. Its complete
projection registry contains no step-entry or raw-meteorology kind
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o16_master_gate.py:93-100,136-238`).
O8--O12, O13, O14, and O15/O16 nevertheless remain individually admitted at
0 ULP over all 100 steps (`docs/ice/fidelity/si3_r21_o8_o12_score_receipt.md:3-13`,
`docs/ice/fidelity/si3_r22_o13_receipt.md:3-8`,
`docs/ice/fidelity/si3_r23_o14_score_receipt.md:3-15`, and
`docs/ice/fidelity/si3_r24_o15_o16_receipt.md:3-19`). This audit retracts none
of those operator verdicts; it says only that their separate exact-input views
cannot be relabelled as O17 composition evidence.

## Frozen O17 score contract

The design row now freezes the full O17 selector card. In particular,
`core2_precipitation_closure=False` is explicit for the NEMO-comparison path;
the library default is `True` and its source says comparison configurations may
disable it (`packages/ice/legoesm/ice/config.py:524-529`). Dynamics and
thermodynamics are explicitly live in the O17 `SI3ORCA2StateConfig`; the current
state carrier otherwise defaults both flags off for the O1 overlay
(`packages/ice/legoesm/ice/si3_orca2_state.py:92-131`).

One production program must produce two non-interchangeable reports:

1. `INDEPENDENT`: initialize once from the O17 initial state, advance all 100
   steps with live O7 products, and compare every recorded field at every step.
2. `GIVEN-NEMO-ENTRY`: for each step, replace the complete step-entry state and
   live-forcing inputs with that step's NEMO record, execute exactly one same
   production step, and compare every recorded field. No output of this label
   may seed `INDEPENDENT`.

The local bar is exact bits, including signed zero. The trajectory bar is the
already frozen pointwise normalized `1e-15`. The first normal-range non-bit row
is `DEBT` and cites its first admitted owner. The window is all 100 steps; the
scored set is every recorded field at every step; the acquisition subset is
exactly complete step-entry state, live O7 inputs and products, named call-order
boundaries, exchange outputs, and restart. None may be sampled or relaxed.

## O17/C4 differentiability disposition

The design now requires CPU binary64 `jax.value_and_grad` for the exact one-step
production entry, with an unweighted sum of all floating prognostic and exchange
output leaves as the scalar reachability objective. It publishes shape, dtype,
finite, NaN, Inf, and nonzero counts for every floating input leaf; integer and
Boolean scheduling/routing leaves are reported as literal non-differentiable
inputs. Category participation, CFL/sweep selection, limiter and donor routing,
`min`/`max`/`where` switches, fixed iteration counts, convergence masks, category
search, and restart/write schedules are the literal non-differentiable
statements. The primal output must retain every forward bit unchanged. GPU
memory and multi-step tape budgeting are explicitly later work.

No count table is published as a measurement: there is no O17 callable to
compile. `step-entry floating leaves: UNMEASURED`, `live-forcing floating
leaves: UNMEASURED`, and `finite/NaN/Inf/nonzero counts: UNMEASURED`. Running
`value_and_grad` on only O15/O16 would be a different program and is refused.

## Decisions, controls, and review

- Window: fixed at 100 steps; no decision.
- Bars: exact local and fixed `1e-15` trajectory; no decision.
- Scored set and record subset: fixed above; no decision.
- AD scalarization: fixed as a mechanical unweighted output-leaf sum; it is not
  a scientific objective or a new bar.
- Acquisition tool: not authored because the explicit composition-entry stop
  fired first. After the entry exists, the acquisition tool must import the
  O5/O7/O8--O16 machinery and shared extent/read-set gates; it may not copy
  their logic.
- NEMO: not built or run.

`CONFIRMED`: the master/record mismatch and missing production entry are direct
schema/source findings. `PLAUSIBLE`: none promoted to an O17 physics verdict.

## Gates

- The required receipt-citation test file passed 45/45. The round receipt gate
  found 17 citations with zero unmapped citations, failures, or full-map audit
  failures; all 12 built-in controls fired. The cumulative default gate found
  274 citations with the same zero-failure result.
- The documented landing battery passed. The combined process reached the
  known per-process JAX compiler limit in the two-case stage-face-mask file, so
  that file and the following barotropic-state file were completed in fresh,
  serial processes as required by the repository instructions: both mask cases
  passed and the barotropic-state file passed 8/8. No test ran with `pytest -n`.
- The nine CI ratchet files reported 11 failed, 6,936 passed, and 2 skipped.
  Those are the established lane failures in hardcoded constants, inline
  coefficients, and parameter specifications; every named failure is in an
  untouched model, acquisition-fixture, or unit-test file. This round changes
  only the ladder design, this receipt, and the citation registry.
- All commands used CPU fp64 with `CUDA_VISIBLE_DEVICES=""`,
  `JAX_PLATFORMS=cpu`, and `JAX_ENABLE_X64=1`. No NEMO or GPU command ran.

independent review: operator (Claude)
