# NEMO testcase L4 ORCA2 round 60 — controlled OVERFLOW downstream walk

Date: 2026-09-28  
Incoming tip: `58f578850c23f9c11b4d1e5c046cdbabadd2e334`  
Frozen preregistration: `f68a67ea8`  
Final production disposition: **HELD; no model physics lands**  
Evidence: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round60/`

## Outcome

Round 60 controlled the held source-ordered UP3 arm at an identical legoESM
kt=3 entry and walked active U through the available production-boundaries to
kt=4.  These numbers are **independent**: both arms begin from legoESM's own
two-step state, not NEMO's recorded state.  NEMO frames are the oracle at each
boundary; they are not substituted into the candidate.

The UP3 arm first moves at stage-2 `after_adv`: 294 / 16,900 active U values
move, with aggregate direction **TOWARD** NEMO.  The immediately following
stage-2 QCO assignment is the first direction change: 237 values move, 66
toward, 170 away, one neutral, and the aggregate becomes **MIXED**.  Its L2
error improves from `5.148932316449435e-08` to
`5.1485304568328606e-08`, while its L-infinity error worsens from
`1.638423633570918e-08` to `1.6384236342648073e-08`.

This names the first source boundary that destroys the strictly-toward result;
it does **not** name a compensating owner.  No later measured boundary is
strictly AWAY, and the required stage-3 post-`dyn_zdf` / pre-barotropic
`raw_kaa` value has no narrow production observer.  The gate therefore leaves
`compensating_owner: null` and `s3.raw_kaa.u: UNMEASURED` instead of inferring
through the missing frame.

The exact round-59 UP3 candidate was reverted.  `git diff 58f578850 --
packages` is empty.  ORCA2, GYRE, configuration, carried state, and sea ice
are unchanged.

## Frozen predictions

| ID | verdict | evidence |
|---|---|---|
| R60-P1 | **CONFIRMED** | The self-describing round-50 record admits `AT_BAR`; the controlled kt=3 state is hash-bound; the one-ULP plant changes the active-U unequal count 520 -> 521 and exits 2. |
| R60-P2 | **CONFIRMED under the controlled entry** | kt=3 entry moves 0 cells; stage-2 `after_adv` is the first move, 294 / 16,900.  The earlier uncontrolled candidate changed 254 entry cells and is retained as a rejected instrument attempt. |
| R60-P3 | **REFUTED** | Stage-2 `pre_zdf` is not the `after_adv` accumulator.  At stage 2, `Krhs == Kaa`; the QCO assignment overwrites that slot before the writer.  The gate now asserts the measured `pre_zdf == raw_kaa` alias instead of printing the false identity. |
| R60-P4 | **PARTIAL** | QCO is the first boundary where TOWARD becomes MIXED, but the strict TOWARD-to-AWAY compensator predicate does not fire.  No owner is promoted. |
| R60-P5 | **STOPPED at the registered production-observer gap** | Stage-3 ADV/LDF/pre-ZDF are measured; `raw_kaa` is explicitly unmeasured; final/postbar and kt=4 are measured but cannot be used to attribute across that gap. |
| R60-P6 | **CONFIRMED** | Observer runs leave T/S/SSH identical to the ordinary step, sidecars are hash- and field-order-bound, and the active-U one-ULP plant exits 2 with exactly one added mismatch. |
| R60-P7 | **CONFIRMED** | The final `packages/` tree equals the incoming tip; both held candidates remain absent. |

## Compiled source order

The executing instrument records the stage-2 ADV accumulator immediately
after the flux-form call at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:327-363`.
For stages 1 and 2, the selected QCO branch then assigns thickness-weighted
velocity to `Kaa` at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:386-405`.
The admitted stage schedule says stage 2 uses `(Kbb,Kmm,Krhs,Kaa) =
(1,3,2,2)`, so the assignment overwrites the same slot the ADV writer just
recorded.  That is the omitted executed statement that refutes R60-P3.

At stage 3 NEMO next adds LDF, writes `pre_zdf`, calls `dyn_zdf`, and writes
`raw_kaa` at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:412-433`.
It then forms the depth-mean correction and updates active U/V at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:436-448`.
The NEMO record contains both sides.  The missing datum is solely a narrow
legoESM production observer between those two calls; a new NEMO acquisition
would not fill that gap.

## Controlled walk

| boundary | direction | moved | toward / away / neutral | base L-inf -> arm L-inf | maximum arm move |
|---|---|---:|---:|---:|---:|
| kt3 entry | UNCHANGED | 0 | 0 / 0 / 0 | `4.181444884787666e-09` -> same | 0 |
| s2 after ADV | TOWARD | 294 | 96 / 197 / 1 | `2.4407174125198362e-09` -> `2.4407174121742468e-09` | `1.2761002912078194e-10` |
| s2 QCO / pre-ZDF / raw Kaa | MIXED | 237 | 66 / 170 / 1 | `1.638423633570918e-08` -> `1.6384236342648073e-08` | `6.380464014391855e-10` |
| s2 postbar | MIXED | 205 | 87 / 118 / 0 | `1.2991128187089807e-08` -> `1.2991129290373937e-08` | `6.258927239335965e-10` |
| s3 after ADV / after LDF / pre-ZDF | TOWARD | 262 | 124 / 138 / 0 | `2.6472396755999677e-09` -> `2.6472396752679308e-09` | `1.2758038893227523e-10` |
| s3 raw Kaa | UNMEASURED | — | — | — | no production-bound observer |
| s3 postbar / kt4 entry | MIXED | 204 | 89 / 115 / 0 | `2.3323169326405768e-08` -> `2.3323170436628793e-08` | `1.251070035328955e-09` |

Per-cell counts and aggregate norms answer different questions here.  For
example, stage-2 ADV has more away than toward cells while both aggregate
norms improve.  The gate's frozen direction predicate uses both L2 and
L-infinity; it reports the counts too and never converts MIXED into an owner.

The citable comparison is `comparison_reclassified.json` (SHA-256
`1c41a2d051bec7a5d925719aafe4efd47d887271b9532a80a1662a7fbbe0d74c`).
Its base and candidate sidecars have SHA-256
`dd98cc4bd604caea9dbb20299c2cab2408a666ace9e5b348f645e1fd62ed7328`
and `c1826680c38c9d9bb5eb4bfebe4c323ee89ddaa3a0d29041ea1f1bd9f3e1d002`.
The controlled kt=3 entry is
`1d782be6d09446d40b7ef0526464144aa041645279706c73019bc19c6c706719`.

## Failed and superseded measurements

The first candidate run started independently from the candidate's own first
two steps.  Its kt=3 entry already differed from the base in 254 active U
cells, so it could not isolate the downstream effect and is not cited for the
boundary attribution.  It remains in the evidence directory as
`candidate.json`.

Two earlier base attempts were interrupted after the private observer set hit
the process compiler-map limit.  The committed gate materializes each result,
drops the model, clears the JAX cache, and prints observer progress; only the
successful clean-stamped `base.json` is cited.

R60-P3's false “no intervening branch” prediction is retained above.  The
mistake was source-order aliasing, not a dead compiled branch: `Krhs` and
`Kaa` are the same stage-2 slot.

## Review and verification

The separate read-only Codex review did not start a reviewer model.  Its exact
terminal verdict was:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Therefore **independent review unavailable in-sandbox**.  The review artifact
SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

Verification results are recorded in the final round commits and evidence
log: the focused round-60 gate tests pass; the default and round-60 citation
gates plus their shifted-line plant are required to pass; and the single
`tests/ocean/fidelity -n 12` battery is reported with registered reds kept
distinct from new failures.

## Scope ledger

**ASKED.** Walk the held UP3 movement from kt=3 stage-2 after-ADV through kt=4
entry, mechanically name the first direction-changing statement, and retain
failed predictions.

**UNASKED and unchanged.** No configuration, default, forcing, threshold,
stabiliser, carried state, ORCA2 entry, or sea-ice field changed.  The six
ORCA2 ice selectors and `unmeasured_features` remain exactly as carried at
`STOP_SELECTOR_GAP`.

## OPEN

1. Add one narrow, write-only, noninterfering production observer after the
   stage-3 `dyn_zdf` update and before barotropic replacement; rerun the same
   controlled entry and decide whether the strict reversal occurs there or in
   the replacement.  This is legoESM instrumentation, not a NEMO acquisition.
2. If that frame names an owner, preregister the UP3 + owner pair.  Until then,
   the source-ordered UP3 and the separately held QCO/RK candidate stay absent.
3. After the step walk closes, run Decision 52's owed independent-start ORCA2
   ladder, then rank month-scale ORCA2 magnitudes.
4. Sea ice remains out of scope at `STOP_SELECTOR_GAP`.

ASKED: controlled kt3 stage-2-ADV through kt4-entry walk.  
UNASKED: configuration, state, stabiliser, and sea-ice changes.
