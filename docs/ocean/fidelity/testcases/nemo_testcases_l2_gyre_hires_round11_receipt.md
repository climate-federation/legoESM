# NEMO testcase L2 GYRE high resolution — round 11 receipt

Date: 2026-10-16

Incoming lane tip: `854d2da150da603813f2637e8ae764a7c5459ccf`.

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round11/`.
The exact-rung oracle is the admitted scalar-math step-2 record from round 7.
The climate reference is certified vector-math NEMO; scalar-math NEMO is the
labelled, non-certified twin floor.  The candidate is the registered
`post_round9_219db484b` five-year legoESM run.

## Verdict

**The remaining entry-`kt=2` tracer boundary is a measured coupled
input-composition/fusion debt, not a measured single transcription owner.  No
shared-path change is warranted.**  The exhaustive full-step bisection found two equivalent
three-member temperature minima: formed diffusivity + Kaa thickness + formed
temperature content, or pre-formation heat diffusivity + Kaa thickness +
formed temperature content.  Salinity has the analogous minima, and joint
T/S exactness additionally needs formed salinity content.  Every proper
one-member removal fails the zero-word bar.

The day-30..180 excess over the scalar/vector NEMO twin is spatially
structured, and the preregistered algebraic form has lower log-residual than
the exponential form for every tied field, basin-wide and in its leading
region.  All six stored fields already exceed four times the twin floor at day
30, so the daily, float32 NEMO output cannot distinguish a uniquely first
field.  The tie is reported without a field-order or ratio tie-break.  Bottom
velocity is unmeasured because the registered legoESM daily files store only
surface and external-mode velocities.

## Finding 1 — exact minimal subset

**Finding.**  Production at the scalar step-1 vertical-solve boundary differs
from NEMO in 110,286 temperature words (maximum
`1.0658141036401503e-14 K`, 3 ULP) and 103,327 salinity words (maximum
`2.1316282072803006e-14`, 3 ULP).  The instrument first substituted one
recorded input at a time, then exhausted all 31 non-empty subsets of the five
arrays consumed by the production tracer dispatcher and all 63 non-empty
subsets of those arrays plus upstream heat diffusivity in the unobserved,
compiled full step.  Its disabled seam is bit-exact and all six active plants
fire.

The isolated production tracer dispatcher has the source-ordered minima one
would expect:

| scored result | smallest exact consumed-input set |
|---|---|
| temperature | formed K + temperature content |
| salinity | formed K + salinity content |
| joint T/S | formed K + temperature content + salinity content |

The full fused step requires one additional selected leaf:

| scored result | smallest exact full-step sets |
|---|---|
| temperature | formed K + Kaa thickness + T content; **or** heat K + Kaa thickness + T content |
| salinity | formed K + Kaa thickness + S content; **or** heat K + Kaa thickness + S content |
| joint T/S | formed K + Kaa thickness + T content + S content; **or** heat K + Kaa thickness + T content + S content |

For the first joint minimum, removing each member gives:

| member removed | T maximum / unequal | S maximum / unequal |
|---|---:|---:|
| formed K | `1.0658e-14` / 1,774 | `1.4211e-14` / 1,443 |
| Kaa thickness | `1.0658e-14` / 64,779 | `1.4211e-14` / 61,309 |
| T content | `7.1054e-15` / 67,325 | `0` / 0 |
| S content | `0` / 0 | `1.4211e-14` / 63,449 |

Kaa thickness is bit-identical to NEMO on all 288,000 active words.  Its
12,120 inactive words differ (NEMO retains nonzero geometry where legoESM has
zero), so the new discriminator separately selects the production array, an
active-only oracle array, an inactive-only oracle array, and the full oracle
array.  With formed K plus both recorded contents, **all four selected arms
are jointly bit-exact**, while the all-false-selector baseline retains 64,779
T and 61,309 S unequal words.  Thus selection of the identical active leaf,
not replacement of its values or its inactive words, changes the fused graph.
The isolated dispatcher does not need Kaa thickness.  Formed K and heat K are
interchangeable here because the recorded step-1 isoneutral addition is zero;
neither is a single owner.

This localizes a genuine multi-input composition/fusion coupling at the
production boundary, but does **not** prove that the upstream formations of K
and the two contents are individually faithful expressions.  The independent
review correctly rejected the stronger rounding-only claim: exactness under a
joint intervention cannot exclude multiple upstream transcription
differences.

**Disposition.**  Classified `COUPLED_INPUT_COMPOSITION_DEBT`; stopped without
a physics change.  No single substitution owns the residual, and changing a
shared kernel before tracing the upstream arithmetic would be unjustified.

**Evidence.**  `step1_vertical_minimal_subset_v5.json`, SHA-256
`e97b07b782454c0589811d994133f5661c18219005b7bb1f7094e800987dce6c`,
produced from clean instrument commit `a9d5764ba`.  The expected process exit
is nonzero because this measurement reports DEBT rather than AT-BAR.  The admitted oracle
manifest is `28e73955382c80179e2e244294ccc28dd4a67762fe826d9da19f887580b6d1f0`.

## Finding 2 — day-30..180 excess over the twin envelope

**Finding.**  The extended divergence instrument first reproduces the
independent NEMO-twin SSH checkpoints exactly at their printed precision:
`1.04e-9`, `1.17e-9`, `1.19e-8`, `1.21e-6`, `2.95e-4`, and `3.89e-3` on days
30, 60, 90, 120, 180, and 365.  Its exact and one-ULP controls separate.

At day 30 every measured field already clears the preregistered four-times
twin criterion:

| field | legoESM normalized RMS | twin normalized RMS | ratio |
|---|---:|---:|---:|
| depth-mean U | `2.3434e-7` | `4.2995e-10` | `545` |
| depth-mean V | `1.9657e-7` | `5.6301e-10` | `349` |
| SSH | `1.5270e-7` | `1.0419e-9` | `147` |
| SST | `6.8570e-7` | `0` | unresolved/infinite |
| surface U | `1.6144e-6` | `1.6442e-9` | `982` |
| surface V | `1.2193e-5` | `2.3296e-9` | `5,234` |

Thus “which field diverges first” is a six-way tie at the first registered
day, not an identifiable ordering.  NEMO's daily fields are float32, so the
zero SST twin floor is a storage-resolution bound rather than proof that the
two NEMO trajectories are physically identical.  By day 180 the legoESM/twin
ratios are `7.16`, `5.64`, `8.20`, `6.10`, `3.35`, and `2.53` in the table's
field order.

The signed candidate-minus-twin SSE attribution preserves the six-way tie:

| tied field | leading wet-span region | signed excess share |
|---|---|---:|
| depth-mean U | eastern boundary | `34.97%` |
| depth-mean V | western interior | `34.64%` |
| SSH | western interior | `32.48%` |
| SST | eastern boundary | `47.09%` |
| surface U | eastern boundary | `33.11%` |
| surface V | western boundary | `27.53%` |

All five signed shares are recorded for every field; none is clipped.  For
each of the six fields, both the basin and its own leading-region fit favor
algebraic growth over exponential growth, with positive fitted rates.  For
example, basin SST has log-RMS residual `0.852` versus `1.190`, and its eastern
boundary has `1.971` versus `2.538`.  Within this two-form diagnostic the
algebraic form fits better and the excess remains spatially structured.  That
does not establish causal absence of chaotic growth.

Depth-mean U/V are reported separately using NEMO's static z-coordinate
thickness-weighted daily 3-D velocities and legoESM's stored external-mode
daily means.  Bottom velocity remains `UNMEASURED`: no matching legoESM
bottom field exists in the registered run.

**Disposition.**  Recorded as diagnostic debt, not a climate failure.  The
same run still passes the preregistered year-5 climate bars against both NEMO
builds, and this round supplies no single shared-path owner.

**Evidence.**
`divergence_regional_v3/gyre_hires_divergence_report.json`, SHA-256
`0fcecf14d530aaa121518f32d2808b9c8137b675dc286d7a1c0af2618607a680`,
and `gyre_hires_divergence_arrays.npz`, SHA-256
`d96f9cf31ec96d598d05418ce1835dbedc64b2e49c1f4be171c44913e1eb48b6`,
produced from clean instrument commit `a9d5764ba`.  The report records both
mesh-mask hash sets, the exact argv/worktree stamp, per-field exact/one-ULP
controls, and strict JSON without non-finite values.

## Shared-path disposition and next measurement

No shared-path change is warranted.  The only model change is a private,
non-configurable diagnostic seam exposing salinity content alongside
temperature content; disabled-seam identity is exact.  No deck value,
physical operator, forcing, or production configuration changed.

The single next measurement is an executed-source arithmetic walk of the
recorded step-1 temperature content at its maximum residual: substitute its
recorded elementary RHS terms one at a time, in NEMO source order, to decide
whether its 78,243 unequal words are a transcription or composition before
attempting any shared-path change.  No acquisition is needed for that first
walk because the scalar step-2 record already contains the assembled matrix,
RHS, and surrounding tracer operands.

## Inherited pins / OPEN

Inherited unchanged: the shipped one-degree digest `99b2dfce324b0f77`, the
one-degree no-worsening ceiling, `r4` and `r4_smag` decks, 31 levels, RK3,
split-explicit free surface, TKE/EVD, Redi tracers, 360-day forcing clock,
scalar-libm CPU certification, zero-word exact-rung bar, float32 NEMO daily
storage, thickness-weighted NEMO daily temperature, registered long-run
manifests, and explicit physical-GPU selection.

The explicit clean-tree shipped trajectory check remains `DEBT`, not a false
bit-identity claim: entry `kt=1` is exact, entry `kt=2` is within the existing
`1e-15` statistic bar but T/S/U/V are not word-exact, and all five fields first
exceed that bar at entry `kt=3`.  Its artifact is
`shipped_1deg_trajectory.json`, SHA-256
`2be84fc7c73a9ea1ba91aa5e43fc7536757b9f82d20daf04fd03e5d1273cb476`,
from clean commit `8ab32d086`.  The 70-row no-worsening ceiling passes both
tests, and the explicit GYRE digest ratchet passes at
`99b2dfce324b0f77`.

OPEN, in order:

1. Walk the recorded step-1 temperature-content RHS terms in executed source
   order; the coupled intervention does not prove each formation faithful.
2. Localize the existing entry-`kt=3` SSH debt by scalar step-2 RK stage.
3. If it first appears inside the external mode, acquire scalar step-2
   barotropic substeps; the current scalar step-2 record has substeps for
   step 1 only.
4. Bottom-regional divergence remains unmeasured for this completed run.
5. GitHub issue #1455 could not be updated from this sandbox because network
   access to `api.github.com` was unavailable.

## Tests and controls

The split verification battery's exact aggregate is:

`FULL_LANE_BATTERY_SPLIT_PASS: 241 unique tests passed, 4 deselected`

Two citation cases initially refused exactly because the review disposition
was still an uncommitted tracked edit; after that receipt-only commit, the
complete citation test file passes 17/17.  This is a clean-tree control firing,
not a product failure.  The final committed receipt citation gate reports
`PASS`, two citations, zero failures and every self-test firing.

Controls exercised by the committed instruments are:
disabled full-step seam identity, six active input plants, exhaustive 31- and
63-subset inventories, active/inactive/selected-identity Kaa discrimination,
NEMO-twin benchmark replay, per-field exact/one-ULP divergence detection,
known-answer depth weighting, five wet-span region placement, strict finite
JSON, and synthetic exponential/algebraic growth separation including a
decreasing-series refusal.

## Independent-review disposition

The first exact-rung review returned **HOLD** because the original report
called every multi-input exact subset rounding-only and scored Kaa equality on
active words while replacing a full array.  The claim is downgraded to coupled
input composition, and the v5 artifact adds full-array, active-only,
inactive-only, and selected-identity controls.  The first climate review also
returned **HOLD** because it invented an unregistered SST tie-break, omitted
signed shares, controlled only SSH, had a vacuous depth-weight check, allowed
non-positive fitted growth, omitted mesh/argv provenance, and overstated the
growth interpretation.  The v3 instrument and this receipt disposition every
finding; final re-review is recorded below.

Both re-reviews now **PASS with no HOLD**.  The exact reviewer independently
verified the v5 hash, active/inactive counts, all four Kaa selector arms and
the narrowed classification; its focused suite reports 42 passed.  The
climate reviewer independently verified every mesh hash, all signed-share
sums, twelve positive growth fits, strict parsing/provenance and the v3
evidence hashes; its focused suite reports 10 passed.

## Compiled-source citations

The executed scalar writer zeros the surface coefficient and forms the
source-ordered lower, upper, and diagonal tracer matrix terms at
`GYRE_OMIP_L2_P3_SMW_R7STEP2/BLD/ppsrc/nemo/trazdf.f90:431-457`.
The same compiled routine forms the separate temperature and salinity
contents, performs the forward recurrence, and then backward substitution at
`GYRE_OMIP_L2_P3_SMW_R7STEP2/BLD/ppsrc/nemo/trazdf.f90:527-559`.
