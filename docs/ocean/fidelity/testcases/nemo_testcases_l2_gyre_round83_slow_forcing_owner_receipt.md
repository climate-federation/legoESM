# NEMO testcase L2 GYRE phase 3 — round 83 slow-forcing owner receipt

## Outcome

**HELD.** The admitted kt2 records and a production-JIT CPU/fp64/libm trace
were walked through the `stp2d` forcing producer. The frozen prediction that
the complete three-dimensional RHS would be the first non-bit input was
**REFUTED**: reference face thickness is already non-bit on every wet U and V
three-dimensional face. The frozen exact cross-record replay also failed, so
this round makes no exact downstream ownership claim and lands no production
physics.

The magnitude arm nevertheless gives a decisive next walk. Replacing only
live geometry by NEMO's recorded reference geometry moves the depth mean by at
most `1.9852334701272664e-23` U and `2.150669592637872e-23` V, while the
remaining depth-mean gap is `1.0530158009820665e-11` U and
`1.0766011961012138e-11` V. The complete three-dimensional RHS differs at all
17,400 U and 17,100 V wet faces, with maxima
`1.9220276136603893e-09` and `1.966059508480186e-09`. Reference geometry is
therefore a real bit boundary but not the magnitude owner. The next
magnitude-ranked owner is upstream inside the cumulative three-dimensional
RHS, to be walked HPG → LDF → VOR → KEG → ZAD/ADV in Round 84.

Decision 37 remains YES, but its absolute-history candidate and the Round-82
drag/inverse pair remain held. Nothing measured here licenses either pair:
the required kt2 U/V trajectory rows were not moved and no production
statement changed.

## Preregistration, records, and evidence

The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round83.md`, committed as
`4171977bb7dd0ae7bd11d1114f43c2372858f172` before scientific scoring. The
gate was committed before each sealed run and finally stamped
`f979935ddbdffb2528198bc0b307a24e34d93574`. Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round83/`.

The Round-64 inherited-record admission passed with 43 byte-identical
records, 20 classified changed records, and 132 admitted differences. The
consumed kt2 stage projection had zero owned-field differences; its only raw
difference from Round 46 is seven undefined `ww` halo values. The stage record
SHA-256 is
`64c42dbecee9e82639384dfa084899d799ae215a27b44ffaf055c488e0ae2642`.
The independently admitted Round-81 external-step stream retained its 46/70
exact, 24 changed, 264-field census and has SHA-256
`566f3f240d05ce526fc9005c2b4a5112e4e4d90484056d6ec6b8a445e8c6abd6`.

The sealed ordinary result is `round83_slow_forcing_walk_sealed.json`,
SHA-256
`2b5e112541f7a56132a3423ee8683a4d2c16155e73066304dfa758488c55ea23`.
Its status is **REFUTED**. NEMO masks, Kmm bottom velocity, barotropic entry
velocity, and the imported Coriolis row are bit-exact. Besides thickness and
RHS, the already known face drag and inverse-depth rounding boundaries remain
non-bit. The source depth mean differs on all 580 U and 570 V wet faces.

The NEMO-record-only final replay does not cross-calibrate: it differs on all
580 U and 570 V faces, at maxima `2.105283055809266e-09` and
`1.7281484610543052e-09`. There is no direct kt2 wind-stress record, and the
live kt2 stress was used only as the preregistered calibration operand. The
gate therefore prints `WITHHELD_NO_DIRECT_RECORD`; it does not recycle the kt1
stress or claim that RHS substitution recovers final slow forcing.

The three repaired differential controls each printed `FIRED` and exited 1:

- thickness ULP: `round83_e3_ulp_plant_sealed.json`, SHA-256
  `1360415088761547ed46407c73272b28ed9fef11ac34de2bbaa951a68dc288cf`;
- RHS ULP: `round83_rhs_ulp_plant_sealed.json`, SHA-256
  `d9bf746f50c5432a5725f430468ec3aee5d54a22cfd95525845969871d414dec`;
- final-forcing ULP: `round83_final_ulp_plant_sealed.json`, SHA-256
  `db27eb42316223c8b74460dd8bf8fca56ad6f77399545a22b2cff7fbee9af938`.

## Instrument failures and retractions

All failures remain explicit rather than being silently overwritten.

1. The first admission attempt incorrectly required whole-file byte identity
   for the stage record. It stopped before reading a scientific field. The
   corrected gate consumes the already admitted projection and fail-closes on
   its exact seven-halo/zero-owned classification.
2. `round83_instrument_failure_after_ldf.json` selected the intermediate
   `after_ldf` cumulative field instead of the compiled branch's final
   `after_adv` field and joined kt2 live stress to a kt1 stress record. None of
   its values is evidence. The corrected tool selects `after_adv` and
   withholds direct kt2 wind identity.
3. The first thickness plant perturbed an arbitrary already-different wet cell
   and left the aggregate count and maximum unchanged. It stayed green. The
   sealed control targets the maximum-residual cell and moves the oracle one
   ULP away, so the reported maximum changes and the control exits nonzero.
4. The preregistered statement “reference thickness and masks are bit-exact
   before the RHS row” is **REFUTED** in the result, not merely withdrawn in
   prose. The exact-RHS-owner and exact-final-substitution claims are likewise
   absent from the gate's success path.

## Compiled-source findings

The admitted stage/RHS stream was emitted by its own compiled Round-64 branch.
That executable accumulates HPG, LDF, VOR, KEG, and ZAD, and records the final
cumulative vector-form RHS as `after_adv`
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:141-176`). Its write-only
instrument records reference thickness, the complete Krhs arrays, and native
masks without assigning a model operand
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:187-199`). These are the
recorded operands scored in the walk; no generic or dead source branch is
cited.

In the same compiled record producer, the active vector-form branch evaluates
the thickness-times-RHS-times-mask vertical sum and then multiplies by the
reference reciprocal depth
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:202-207`). It next calls
the baroclinic drag correction and records its output
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:225-227`), then adds wind
in a separate written statement
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:229-235`).

The independently acquired Round-81 compiled external-step branch copies that
completed forcing, initializes/removes the separately treated two-dimensional
Coriolis trend, and only then enters the split-explicit evolution
(`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:291-327`). Its
executing drag routine forms the bottom-only face coefficients, selects the
forward Kmm bottom residual, and adds inverse-depth × coefficient × residual
in the written order
(`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:1467-1526`).

These statements establish the measured order. They do not make the failed
cross-record join exact. The first measured bit boundary is the reference-
versus-live thickness input to the vertical sum; the controlled substitution
shows that boundary is not large enough to own the campaign's kt2 forcing
residual. The complete cumulative RHS is the first magnitude-ranked upstream
input and therefore owns the next source walk.

## Rule-12 table

| Lane | Registered disposition |
|---|---|
| GYRE kt2 forcing producer | **MEASURED / NO CANDIDATE.** Thickness is the first non-bit input; its reference-geometry substitution moves the depth mean by only about `2e-23`. The cumulative RHS is the next magnitude-ranked walk. Exact final ownership is withheld because the independent records do not cross-calibrate. |
| GYRE kt=1..10 | **PRESERVED, NOT RERUN.** No production physics changed. The immutable before arm remains `decision36_nemo_face_shear/after_kt1_10.json`; first-over-bar remains kt2 U/V. |
| Every moved GYRE row | **NONE.** This round changes diagnostics/tests/docs only. The held Round-79 and Round-82 candidates are not in a newly promoted production arm. |
| GYRE days 1..30 | **PRESERVED, NOT RERUN.** No eligible short-ladder candidate exists. The immutable before arm remains `decision36_nemo_face_shear/after_day_gap.json`. |
| LOCK_EXCHANGE-zco | **PRESERVED, NOT RERUN.** No shared production statement changed; no new neutrality claim is made. Its next run must score the same split-explicit kt rows if an RHS fix lands. |
| OVERFLOW-zps | **PRESERVED, NOT RERUN.** No shared production statement changed; no new neutrality claim is made. |
| DINO | **SHARED-STATEMENT RISK.** DINO's leapfrog card has its own histories, but it executes the shared momentum RHS and depth-average programs. Its 96–98% regional cancellation forbids a neutrality inference from the GYRE arm. |
| ORCA2 | **UNMEASURED WITH SPEC.** Independently align the complete 3-D RHS and every cumulative operator boundary, reference face thickness, masks, depth mean, drag/wind/Coriolis operands, final forcing, T/S/U/V/SSH, and six histories on native staggered masks. Require elementwise fp64 bit equality and normalized L-infinity through kt=1..10 at bar `1e-15`. Falsifier: any AT-BAR loss, earlier first-over-bar, or wet-point operand/history mismatch. |

No restart or checkpoint representation changed. The Decision-37 loud-failure
contract for incompatible history-bearing restarts remains untouched. No
configuration, selector, coefficient, timestep, stabilizer, carried-state
choice, year harness, reconciliation gate, freshwater pair, #1484 guard,
held manifest, or NEMO source was modified.

## Review and focused verification

The mandatory separate review command was attempted against the complete
Round-83 diff after the Rule-12 table was written. It failed before the review
model started, so there is no `SHIP`/`DO NOT SHIP` verdict to quote. Its
terminal line, quoted verbatim, is:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Per the operator's 2026-09-13 08:30 instruction: **independent review
unavailable in-sandbox**; work continued. The complete captured output is
`round83_codex_review.txt`, SHA-256
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.
No reviewer approval is claimed, and no physics is being shipped.

The clean-tree citation gate passed all 7/7 compiled-source citations with no
unmapped citation or map-audit failure. Shifting the HPG-to-ADV citation by two
lines produced `SYMBOL-NOT-AT-LINE` and exited 1. The ordinary/plant artifacts
have SHA-256
`3b03f4360f89d9bd412f52aff6f685bc5f06aa637bdcac8ac7f291725d5e3109`
and
`e17c04ff30b3639f27751a89177be6806b6c62a0ebad866a20c7f79cd743829f`.

Focused CPU/fp64 verification covered the Round-81 admission/record reader,
Round-82 operand walk, new Round-83 source-chain helpers, and receipt citation
gate: 25 tests passed in 1.69 seconds. Both modified scripts also compile, and
`git diff --check` passes. The test log SHA-256 is
`d9f83af3c2813810b066e09bcce3f94353e387a0f7a3abab1e6ebbf8e0af8ecb`.
The known unrelated `test_rk3_ws_differs_from_rk3_and_is_finite` failure was
not encountered.

## Choices and uncertainty

Choices made: none. This round did not reinterpret Decision 37, choose a
free-surface geometry convention, or authorize a new carried-state format.

The remaining uncertainty is deliberately split. The magnitude result is
measured: reference geometry cannot explain the slow-forcing scale, and the
complete cumulative RHS is the next large upstream input. Exact source
ownership downstream of that input is unmeasured because the independent
kt2 records lack direct wind identity and fail final cross-calibration. No
claim bridges that gap.

## OPEN — exact handoff to round 84

1. Reuse the admitted Round-64 cumulative stage record; no NEMO acquisition is
   needed. Preregister and walk the complete kt2 three-dimensional RHS in the
   compiled order `after_hpg`, `after_ldf`, `after_vor`, `after_keg`,
   `after_zad`, `after_adv`. Score U and V on native wet masks and rank the
   first differing cumulative boundary by magnitude.
2. Read the compiled implementation for whichever operator first creates the
   large residual and compare every input/result in written association. A
   production candidate is eligible only when all earlier inputs are bit-exact
   and the changed statement is cited from the executing compiled branch.
3. Keep the Round-79 histories arm and Round-82 drag/inverse pair held. They
   may be reconsidered only with the upstream RHS owner and only if the
   Decision-36 ladder moves kt2 U/V toward the bar, loses no AT-BAR row, moves
   first-over-bar no earlier, and creates no greater-than-two-ULP worsening.
4. If such a pair passes kt1..10, then run days 1..30 and the recorded
   LOCK_EXCHANGE/OVERFLOW lanes, state DINO shared-statement risk, and retain
   the ORCA2 contract above.

ACQUISITION_NEEDED: NONE

DECISION_NEEDED: NONE
