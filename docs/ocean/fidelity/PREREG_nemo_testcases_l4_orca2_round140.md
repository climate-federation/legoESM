# Preregistration — ORCA2 round 140 upstream-predictor walk

Date: 2026-10-04. Base: `41b927214f679245e7bc74c027d4690da083ea20`.
Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round140`.
All step-36 values are **independent**: rung 0 starts from its own
climatological T/S, zero velocity, and zero sea surface. Given-entry ladder
populations remain separate.

Round 139 admitted finite `pbef=8852.366190269584` and `paft=inf` at the wet
below cell `(j,i,k)=(87,159,4)`. The compiled ORCA2 `fct_up1_2stp` program
builds first-step face fluxes at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:500-526`, midpoint `pt_up1` at
`:528-539`, averaged second-step face fluxes at `:562-596`, `ztra` at
`:598-607`, and the final numerator/divisor update at `:609`.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R140-P1 | The source trace is passive and ordered. | Separately compiled ordinary FCT outputs and every ordinary step-state leaf are bit-identical; registry, passivity, source-order, arithmetic-link, and planted violations all fire. | Reject every source-walk number and stop at the observer defect. |
| R140-P2 | The first non-finite source-ordered intermediate at `(87,159,4)` is an incident averaged horizontal face flux at compiled lines 569-572; first-step fluxes and midpoint remain finite. | All preceding registered intermediates are finite and the first non-finite field is an averaged U/V incident face. | Mark **REFUTED** and retain the actual earliest source-ordered field/statement. |
| R140-P3 | The final after-thickness divisor is finite and positive, while the numerator is non-finite because `ztra` already is. | The recorded face terms reproduce `ztra`; base content plus `dt*ztra` reproduces the numerator; numerator/divisor reproduces `paft=inf`. | Mark **REFUTED** and walk the first broken link or association. |
| R140-P4 | Multiplying rung 0's instantiated `rn_evd`/`K_conv` by `1e4` changes neither the surfaced `avt`/`avm` profiles nor one-step state, proving the stated EVD arm is inert on this card. | Exact array equality for profiles and state, with a plant that changes a live coefficient and is detected. | Mark **REFUTED**; report the active-cell census and magnitude. Decision 94 remains pending, so do not land an EVD fix. |
| R140-P5 | No production statement, deck, selector, carried state, stabilizer, sea-ice field, or `unmeasured_features` entry changes. | Only default-off private diagnostics, gates/tests, citations, and receipts change; standing shared-card gates retain their certified rows. | Hold at the first moved shared-gate row. |

## Round bar

The first source-ordered non-finite arithmetic statement inside
`fct_up1_2stp` owns the walk. A result is admissible only if the observer is
bit-passive and every arithmetic association is mechanically reproduced. The
EVD plant is a read-out only: Decision 94 forbids an independent fix here. No
stabilizer, clipping, threshold relaxation, card choice, or sea-ice change is
authorized.
