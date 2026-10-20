# NEMO testcase L2 GYRE higher-resolution round 22 receipt

Date: 2026-10-20. Branch: `fidelity/gyre-high-resolution`.

## Decision

**DEBT / HELD — coefficient/operator composition, no physics edit.** At the
common day-180 `r4_smag` state, legoESM, scalar-math NEMO, and vector-math NEMO
form bit-identical stage-1 Smagorinsky square-root coefficients. Materializing
either coefficient field at the private `dyn_ldf` seam makes the complete
post-LDF U and V accumulators bit-exact to scalar NEMO. Because `ahmt`, `ahmf`,
and the pair all collapse, no single coefficient factor owns the residue. The
measured owner boundary is JAX composition across coefficient formation and
its two-pass consumer. This is a rounding-order debt, not a transcription
defect; no production kernel, physics, deck, or acceptance bar changes.

Artifact:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round22/stage1_ldf_factor_walk.json`,
SHA-256 `df454fd47977235725cd11ee1d00cf2b6da7172a7fa0a5be12e3c48e9eb02f45`,
clean commit `ef4e2079cfdda7718707ff9417fd2542f348a7cb`.

## Finding -> disposition -> evidence

### 1. Merged-source citation drift -> repaired before science

Pierre Gentine's six source-area commits shifted eight mapped
`nemo_testcase_recipe.py` anchors. Each anchor was re-read against the merged
source rather than mechanically renumbered. The citation tests pass, the real
gate passes with `274` citations, zero failures, zero unmapped citations, and
all nine synthetic controls firing. Shifting the planted `stp2d.F90:128`
anchor by two lines exits nonzero with `SYMBOL-NOT-AT-LINE`.

### 2. The acquired coefficient factors -> exactly exonerated

The scorer admits the round-21 scalar and vector roots, verifies every
manifest member, pins producer `52a05e570b607d7fb30ba82d4e9add3812aa7e25`,
and records manifest SHA-256 values
`5c7edb56371d500fc1f2883bb6a1ad4f7f5bbe277e656d1acc02d0fdbbb8ec74`
(scalar) and
`52198602cc548ef9365b4d79e31dd16ec77cc61bdaf14861c9ef0b35d8be6d86`
(vector). The full arrays score:

| formed field | words | legoESM vs scalar | scalar vs vector |
|---|---:|---:|---:|
| `ahmt` | 300,120 | 0 unequal, 0 ULP | 0 unequal, 0 ULP |
| `ahmf` | 306,270 | 0 unequal, 0 ULP | 0 unequal, 0 ULP |

This exact result jointly exonerates the requested coefficient factors:
before-level strain/tension inputs, grid metrics, `rn_csmc`, lower and upper
bounds, `rn_Dt`, masks, and the square-root conversion. NEMO's executed mode-32
branch forms the deformation and bounds, converts the bilaplacian coefficient,
and exchanges its halo at
`GYRE_OMIP_L2_P3_R20RHSS/BLD/ppsrc/nemo/ldfdyn.f90:462-534`.

### 3. The private substitution seam -> repaired and proved live

The first two attempts refused correctly: stage 2/3 received the private
coefficient override, but the separate stage-1 `tendencies()` call did not.
The instrument fix routes the same diagnostic-only hook into stage 1. Its
default remains `None`, which is the production path; no card constructs a
non-default hook. Reverting the routing makes
`test_stage1_routes_smagorinsky_override_into_initial_tendency_call` fail.

The one-ULP control moves exactly one active `ahmf` word at native
`[j=57,i=115,k=0]`, from `171939.92903242097` to
`171939.929032421`, and the coefficient comparator records one unequal word
and one ULP. That single ULP rounds away through the two-pass operator and HPG
addition, so the receipt does not claim it is downstream-observable. A
separate one-word plant at the same acquired support proves the seam: HPG
remains bit-exact, while post-LDF moves eight U words and eight V words
(maxima `1.940357102098917e-17` and `1.9456297167594526e-17`). This separates
the exact-array plant from the non-vacuity test without weakening either.

### 4. Exact factor walk -> composition boundary, no single owner

Before substitution, the first source boundary outside the scalar/vector
NEMO envelope is stage-1 lateral diffusion:

| field | legoESM - scalar NEMO | scalar - vector NEMO |
|---|---:|---:|
| U | 3,789 / 285,600 unequal; max `1.852884572118782e-21`; 5,120 ULP | exact |
| V | 3,261 / 284,400 unequal; max `3.3881317890172014e-21`; 1,152 ULP | exact |

NEMO calls `ldf_dyn` before the external-mode stage at
`GYRE_OMIP_L2_P3_R20RHSS/BLD/ppsrc/nemo/stprk3.f90:184-190`, and its stage-1
shared RHS is accumulated in executed HPG -> LDF -> VOR -> KEG -> ZAD order at
`GYRE_OMIP_L2_P3_R20RHSS/BLD/ppsrc/nemo/stp2d.f90:159-190`.

Each of the `ahmt`-only, `ahmf`-only, and paired materialized arms leaves both
HPG fields bit-exact and makes both post-LDF fields bit-exact (zero unequal
active words). The selected NEMO operator applies the same square-root
coefficient in both rotated bilaplacian passes at
`GYRE_OMIP_L2_P3_R20RHSS/BLD/ppsrc/nemo/dynldf_lev.f90:249-330`.
Since the supplied fields equal the live fields word for word, the only changed
variable is whether coefficient formation and consumption are composed in one
JAX computation or separated by a materialized input. The exact classifier is
therefore `COEFFICIENT_OPERATOR_COMPOSITION`. It explicitly refuses to name
`ahmt`, `ahmf`, any coefficient factor, or either NEMO pass as a transcription
owner.

### 5. Shipped card and shared defaults -> unchanged

The shipped one-degree digest remains `56c5171d43504b54`. Its certified
ten-step trajectory run exits with the expected scientific DEBT, not a harness
failure; all 70 trajectory/barotropic residual rows are byte-for-byte equal to
the admitted round-14 report, and the first-over-bar boundary remains step 3.
The round-9 no-worsening ceilings pass. The ordinary traced state is bit-exact
to the untraced state (zero mismatched bytes); only the private source-order arm
is intentionally non-passive and is never a run candidate.

## Inherited pins / OPEN

Inherited unchanged: the certified `GYRE-zco` one-degree deck and digest,
31 levels, WS-RK3, split-explicit free surface, TKE/EVD, Redi tracers,
scalar-libm CPU policy, `r4_smag` deck and 3,600-s step, literal 360-day
forcing clock, common day-180 restart SHA-256
`a8e33a3d22e4483bac777839f6cdc9bdf8c2e348c3a1d638adeafa114fff1c12`,
round-16 four-times twin discriminator, the year-5 climate PASS against both
NEMO builds, and every prior retraction. No NEMO engine source was modified or
run by the agent.

OPEN: the composition debt remains registered at the exact stage-1 LDF
boundary. It does not warrant forcing JAX materialization into the production
model: its maximum is `3.39e-21`, the five-year statistical card already
passes, and the user's long-run bar is climate statistics rather than
trajectory identity. A future attempt may compare compiler HLO/fusion choices,
but no acquisition or shared-path change is requested from this result.

Round 18 and round 19 remain **HELD**. Round 22 is likewise
**HELD / UNREVIEWED-BY-GLM** as required by the operator; no physics claim or
fix depends on an unavailable review.

## Tests, controls, and review

The focused round-20/22 scorer test file reports `22 passed`. The hook-routing
reversion test fails when the stage-1 keyword is removed. The high-resolution
instrument/acquisition/climate/driver/citation split reports
`173 passed, 2 deselected`; the time-level/one-degree reconciliation/phase-3
compatibility split reports `62 passed, 2 deselected`. Thus the disjoint GYRE
lane battery reports `FULL_LANE_BATTERY_SPLIT_PASS: 235 passed, 4 deselected`.
The shared SWG card/composition supplement reports another `160 passed`.

The round-22 receipt citation gate passes all four compiled-source citations,
with zero failures, zero unmapped citations, zero map-audit failures, and all
nine synthetic controls firing. Shifting the compiled `stprk3` citation by two
lines returns `SYMBOL-NOT-AT-LINE` and a nonzero exit. Keeping the private hook
keyword on the existing physical call line also restores the complete historic
citation-map audit after an intermediate three-line form correctly made it
fail.

Review status is **HELD / UNREVIEWED-BY-GLM**. The read-only Codex reviewer
failed while initializing its app-server client on the read-only host
filesystem. The read-only Claude reviewer reported that the workspace trust
record was unavailable, then produced no review content and was terminated
after the bounded wait. GLM is unavailable as recorded by the operator. These
are service failures, not verdicts or waivers. No production physics fix
depends on this unreviewed composition localization.

Choice audit -- **ASKED**: repair merged citation anchors, score the acquired
scalar/vector coefficients, walk one factor at a time, preserve exact bars,
and gate shared defaults. **UNASKED**: no physics edit, deck change, bar change,
NEMO execution, GPU execution, long run, or external issue update was made.
No user decision is needed.

## Compiled-source citations

The coefficient call precedes the stage-1 external-mode program at
`GYRE_OMIP_L2_P3_R20RHSS/BLD/ppsrc/nemo/stprk3.f90:184-190`.
The acquired cumulative family follows the executed shared-RHS order at
`GYRE_OMIP_L2_P3_R20RHSS/BLD/ppsrc/nemo/stp2d.f90:159-190`.
Mode 32 forms, clamps, converts, masks, and exchanges the coefficients at
`GYRE_OMIP_L2_P3_R20RHSS/BLD/ppsrc/nemo/ldfdyn.f90:462-534`.
The selected level-wise rotated operator consumes them in its two passes at
`GYRE_OMIP_L2_P3_R20RHSS/BLD/ppsrc/nemo/dynldf_lev.f90:249-330`.
