# GRIDLADDER-VMIX lane — round 19 (vmix_r19): L2 A33 and face operands

Status: **L2 CERTIFIED** (2026-10-19).

The resume closes the held `kt=1` gate, corrects the stage-ratio citations,
and transcribes the stabilising vertical diffusivity and its consuming Redi
flux in NEMO's written arithmetic topology.  Both L2 labels are AT-BAR through
`kt=10`; all non-L2 controls retain their normalized jaxprs and GYRE retains
its ten-step residual archive byte-for-byte.

## Source transcription

NEMO first forms A33 from two bracketed U/V diffusivity pairs
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traldf_iso.f90:782-794`).  Its MSC branch then
divides each vertical face pair by the written metric product, adds the two U
terms and two V terms as bracketed pairs, multiplies the combined sum by
`0.25`, materializes the live squared Kmm thickness, and finishes the
Laplacian branch with `* ze3w_2 * r1_Dt`
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traldf_iso.f90:811-830`).  The selected L2
implementation now preserves those operands, divisions, pair association,
and statement boundaries with `nemo_source_round`; the false branch remains
the historical reciprocal-metric/left-fold/final-division graph
(`packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py:2463-2540`).

The same selected source-order path preserves NEMO's vertical-flux statement
boundaries and uses the existing literal horizontal-divergence and stored-area
evaluators; the unselected branches remain unchanged
(`packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py:2901-3057`,
`packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py:5000-5055`).
The production selector is still ORCA2 L2 only: both configuration fields
default false
(`packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:480-493`) and
the exact L2 predicate requires both true
(`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:4525-4564`).

ORCA2 executes the stored stage-2 `r3u`/`r3v` averages in the `np_HYB` branch
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:222-223`) and consumes that Kmm
in stage 3 (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:229`).  L2 passes
those stored native ratios to the live face builder
(`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:10006-10013`).

## Exact-value gates

The x64 JIT unit test uses traced inputs and an independent NumPy
transcription.  It is array-exact for A33 and `akz`; its fixed former-formula
plant differs in 13 cells
(`tests/ocean/unit/test_gm_redi_latlon_cgrid.py:100-196`).

The jitted production gate checks stored `ahtu`/`ahtv`, explicit `akz`, and
the implicit K33 consumer.  NEMO has 171
nonzero `akz` cells; the new path and an independent replay have zero
mismatches over all 799,200 cells.  The former formula, replayed eagerly on
the same operands, has 50 mismatches
(`tests/ocean/fidelity/test_nemo_testcase_l2_orca2iso.py:414-572`).

The independent review located the original four surface residuals at T
(87,43) and S (86,43), near 11 N 166 E, and T (115,53) and S (116,53), near
57 N 174 W.  Their columns and all 68 three-dimensional S residual cells lay
where the pre-fix stabilising `akz` differed, including 165 of NEMO's 171
nonzero cells.  The copied review report, scripts, logs, and checksums are in
`gridladder_vmix_rounds/round19/review_vmix20/`; no claim here depends on a
temporary path.

The held full-step record test was rerun after all source changes and reports
**1 passed in 466.28 s**.  Its measured selected T/S tuples are both
`(0, 0.0, 0.0)`.
All four omission arms remain non-vacuous and are pinned from that run:

- metric: T `(16433, 0.02054161798762677, 0.0006725116923845033)`, S
  `(16433, 0.020566462820472253, 0.0004266288093426869)`;
- EOS-80: T `(16429, 0.004663857636465485, 0.00013889085230196617)`, S
  `(16429, 0.0014607050156243417, 3.852705067169809e-05)`;
- north fold: T `(138, 0.0013241514577646551, 1.595933265099751e-05)`, S
  `(138, 0.008011780370249966, 9.00824649513366e-05)`;
- Kmm face geometry: T
  `(16428, 5.092867354017372e-05, 4.384825971525139e-07)`, S
  `(16428, 2.572830925373637e-05, 2.698489329967449e-07)`.

## Certification score and controls

The independent label covers `kt=1..10`: **68 / 68** surface rows and **36 /
36** full-depth summaries are AT-BAR.  The given-NEMO-entry scorer covers the
same ten steps through its entry and following-step rows: **68 / 68** surface
rows and **20 / 20** full-depth summaries are AT-BAR.  Evidence and SHA-256
stable inputs are under `gridladder_vmix_rounds/round19/resume_score/`.
Therefore the round-19 outcome is **L2 CERTIFIED**.

Normalized jaxpr hashes are unchanged for the 20 non-DINO cards: G5, V1-V5,
L1, full ORCA2, GYRE, VORTEX, SMT-1..4, rung 0, and OMT-0..4.  DINO is also
unchanged.  All 21 comparisons report `expected_identical: true` under
`gridladder_vmix_rounds/round19/purity_resume/`.

The fresh GYRE ten-step comparison passes all 70 rows with zero ULP worsening.
Its residual archive is byte-identical to the prior artifact at SHA-256
`3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5b`.
Evidence is under `gridladder_vmix_rounds/round19/gyre_resume/`.

No NEMO executable was built or run.  No new record is needed.

## Landing gates

- explicit round-19 receipt citation gate: **PASS**, 12 citations, zero
  failures, zero unmapped citations, zero map-audit failures;
- default ORCA2 landing list: **145 passed in 983.84 s**;
- private-import ratchet: **6 passed in 9.31 s**;
- x64 JIT exact-value unit test: **1 passed in 1.02 s**;
- jitted production exactness gate: **1 passed in 193.87 s**;
- held `kt=1` record/omission gate: **1 passed in 466.28 s**.

independent review: operator (Claude)

codex review: operator

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/vmix_r19/repo
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: LANDED — L2 CERTIFIED; independent 68/68 surface + 36/36 depth and given-entry 68/68 + 20/20 AT-BAR through kt=10; all other cards unchanged
