# GYRE high-resolution round 21 receipt

Date: 2026-10-20. Status: **HELD FOR THE LDF COEFFICIENT ACQUISITION -- NO
PHYSICS CHANGE**. The admitted source-ordered records make stage-1 lateral
diffusion the first exact-bar mismatch. Its maximum is only
`1.85e-21 m s-2`; the later vertical-advection boundary carries the material
`5.14e-10 m s-2` residual. No single factor has yet removed the first mismatch,
so this round does not name a transcription owner.

## Finding -> disposition -> evidence

The first attempted score refused because legoESM's ordinary diagnostic forms
isolated tendencies in a different arithmetic order from NEMO's cumulative
Krhs. That was an instrument-association defect, not a model result. The scorer
now keeps an independently compiled ordinary step as the passivity control and
uses the pre-existing Round-119 `stage1-source-order` association arm only for
the cumulative comparison. The ordinary traced and untraced final states are
bit-exact (zero mismatched bytes). The association arm is explicitly
diagnostic, is not called passive, and differs from the ordinary final state in
1,023,497 bytes.

All three final-frame admissions close exactly: scalar NEMO's final cumulative
frame equals its inherited completed RHS, vector NEMO's does likewise, and the
legoesm source-order final frame equals its source-order RHS, with zero unequal
U and V words. The scalar/vector NEMO envelope is zero at every cumulative
boundary. Against the certified scalar oracle:

| boundary / contributor | field | legoESM maximum [m s-2] | unequal words | maximum ULP | scalar-vector maximum |
|---|---|---:|---:|---:|---:|
| post-HPG | U | `0` | 0 / 285,600 | 0 | `0` |
| post-HPG | V | `0` | 0 / 284,400 | 0 | `0` |
| post-LDF | U | `1.852884572118782e-21` | 3,789 / 285,600 | 5,120 | `0` |
| post-LDF | V | `3.3881317890172014e-21` | 3,261 / 284,400 | 1,152 | `0` |
| post-VOR | U | `3.3881317890172014e-21` | 1,321 / 285,600 | 768 | `0` |
| post-VOR | V | `6.776263578034403e-21` | 695 / 284,400 | 128 | `0` |
| post-KEG | U | `6.776263578034403e-21` | 2,506 / 285,600 | 382 | `0` |
| post-KEG | V | `6.776263578034403e-21` | 665 / 284,400 | 768 | `0` |
| post-ZAD | U | `5.139395332331717e-10` | 285,600 / 285,600 | `1.41e15` | `0` |
| post-ZAD | V | `4.868790573218769e-10` | 284,400 / 284,400 | `2.21e15` | `0` |

The registered exact bar therefore selects post-LDF U as the **first** term
beyond the twin envelope. Its large ULP count reflects tiny near-zero values;
it is not evidence that LDF owns the material post-ZAD residual. Conversely,
the later post-ZAD magnitude is not eligible to bypass the first boundary.

**Disposition:** record the formed Smagorinsky square-root coefficients at the
same developed state, then substitute scalar NEMO's `ahmt`, `ahmf`, and the
pair one input set at a time. Only a one-field arm that makes both post-LDF
faces bit-exact without moving the exact post-HPG rows can name a single-factor
owner. Pair-only exactness is coupled composition debt; a non-exact pair
exonerates coefficient formation and requires the two operator passes. The
current records contain cumulative RHS fields but not `ahmt`/`ahmf`, so the
factor proof cannot be completed without this acquisition.

The evidence artifact is
`gyrehr_rounds/round21/stage1_source_boundary.json`, SHA-256
`0cd601b8b75b5f131daf3a16030bb29387244ad4db3b1ca5b0e2c554efd7c369`,
produced from clean commit `5f769d6009cf006edbe95140c5321fd34a8f726d`.
The scalar and vector cumulative providers each admit 124 manifested members;
their manifest SHA-256 values are
`3e1c015fc2f1e0e70c2c92d6f7bbb6001136f18ac0638b197f6a2ecf579ca277`
and `dad19d9353c6d978eee2c143c7645540676a83065f281eb4a5c9ae29de16b8dc`.

## Acquisition and controls

The committed writer captures NEMO's post-`ldf_dyn` `ahmt` and `ahmf`
immediately before `stp_2D` at step 4321. It uses the exact common day-180
restart, one rank, the `r4_smag` deck, and separate scalar/vector builds. One
built binary runs with capture disabled and enabled; terminal restarts must be
byte-identical, the control must emit no record, and the enabled record must
have the pinned binary64 header, `126 x 86 x 31` native extents, finite words,
and exact byte size. Both preflights pass, including zero-fuzz source/deck
patches and the modified Fortran syntax compile. The scripts create their
evidence parent directory before checking free space.

The cumulative comparison's one-ULP plant moves exactly one active post-LDF U
word: one of 285,600 words differs, maximum ULP is one, and maximum absolute
movement is `5.293955920339377e-23`. Its artifact is
`gyrehr_rounds/round21/stage1_after_ldf_u_ulp_plant.json`, SHA-256
`48e227086adcff29b6f84a62ba6afc9b1d9e02f9c99ba8c008e8bd7d9d3a13f2`.
The active coefficient plant remains preregistered and is evaluated only after
the acquired arrays define active support.

The source-order regression test is reversion-sensitive: changing the scorer
back to the ordinary trace makes
`test_run_uses_source_order_arm_only_for_cumulative_family` fail. Restoring the
fix returns the focused file to 19 passing tests. Header, shape, finite-value,
paired-provider, exact-only owner, source-order, single-rank, writer passivity,
and evidence-parent refusals are also exercised.

## Inherited pins / OPEN

Inherited unchanged: shipped one-degree digest `56c5171d43504b54`, its
no-worsening ceiling, 31 levels, WS-RK3, split-explicit free surface, TKE/EVD,
Redi tracers, the `r4_smag` deck, 3,600-s step, literal 360-day forcing clock,
scalar-libm certified CPU policy, vector-math climate reference, common
day-180 restart SHA-256
`a8e33a3d22e4483bac777839f6cdc9bdf8c2e348c3a1d638adeafa114fff1c12`,
round-16 four-times twin discriminator, year-5 climate PASS against both NEMO
builds, and all earlier retractions. No card, deck, prognostic arithmetic,
shared physical kernel, long run, or NEMO engine source changed.

OPEN, in order:

1. Run the scalar and vector post-`ldf_dyn` coefficient captures.
2. Score the preregistered `ahmt`, `ahmf`, and pair substitutions plus their
   active one-ULP control.
3. If the pair is not exact, record both rotated bilaplacian passes; do not
   infer the operator owner from the remaining norm.
4. Preserve the round-18 and round-19 HELD labels. Round 21 is likewise
   **UNREVIEWED-BY-GLM**, as the operator records that service unavailable.
5. Post a reviewed result to issue #1455; this acquisition-held result does not
   promote a physics claim.

No five-year run or shared-path change is warranted. The single next
measurement is the common-state coefficient substitution.

Choice audit -- **ASKED**: score the acquired source-order terms, walk the
first one factor by factor, and request only a missing stage-1 factor record.
**UNASKED**: no physics edit, bar change, deck change, long run, or NEMO run was
performed. No user decision is needed.

## Tests and review

The full high-resolution acquisition/ladder/budget/climate/divergence/movie,
round-17/20/21, shipped-ceiling, driver, and citation split reports
`158 passed, 2 deselected`. The time-level registry, one-degree reconciliation,
and phase-3 trajectory compatibility split reports `62 passed, 2 deselected`.
Thus the disjoint full lane battery reports
`FULL_LANE_BATTERY_SPLIT_PASS: 220 passed, 4 deselected`.
Both scalar/vector operator preflights report `PREFLIGHT_OK`; `bash -n`, the
actual modified-Fortran syntax compile, and `git diff --check` pass.

The receipt citation gate passes all four producer-specific compiled-source
citations with zero unmapped citations, failures, or stale map entries. All
nine synthetic controls fire. Shifting the source-order citation by two lines
returns `SYMBOL-NOT-AT-LINE` and a nonzero exit, as required.

Review status is **HELD / UNREVIEWED-BY-GLM**. The read-only Claude reviewer
refused because the CLI is not logged in and the workspace trust record cannot
be written from this sandbox. The read-only Codex reviewer refused while
initialising its app-server client on the read-only host filesystem. GLM is
unavailable as recorded by the operator. These are service failures, not
verdicts or waivers. No physics fix depends on this unreviewed localization;
the result is an exact measured boundary plus the minimal passive acquisition
needed for the proving substitution.

## Compiled-source citations

The cumulative provider computes the mode-32 coefficient before the external
mode at
`GYRE_OMIP_L2_P3_R20RHSS/BLD/ppsrc/nemo/stprk3.f90:184-190`.
Its writer records the shared RHS after HPG, LDF, VOR, KEG, and ZAD in executed
order at
`GYRE_OMIP_L2_P3_R20RHSS/BLD/ppsrc/nemo/stp2d.f90:159-190`.
The selected coefficient branch forms the deformation and lower/upper limits,
converts to the square-root bilaplacian coefficient, and applies the halo at
`GYRE_OMIP_L2_P3_R20RHSS/BLD/ppsrc/nemo/ldfdyn.f90:462-534`.
The selected rotated operator uses that coefficient in its first and second
passes at
`GYRE_OMIP_L2_P3_R20RHSS/BLD/ppsrc/nemo/dynldf_lev.f90:249-330`.
