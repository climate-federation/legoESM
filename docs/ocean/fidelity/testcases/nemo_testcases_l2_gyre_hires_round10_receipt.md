# NEMO testcase L2 GYRE high resolution — round 10 receipt

Date: 2026-10-16

Incoming lane tip: `f4b5c6c0d71fd163faba03ead7e07515e893dc28`.

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round10/`.
The five-year legoESM run is the registered native-GPU run
`post_round9_219db484b`.  Vector-math NEMO remains the certified climate
reference; scalar-math NEMO is explicitly labelled non-certified wherever it
is admitted.

## Verdict

**The year-5 climate verdict is PASS against both NEMO builds, but the
round-9 early-divergence expectation is REFUTED.**  The two shared-path fixes
together reduce legoESM's SSH and native-U divergence at every registered
checkpoint through day 365, spectacularly at day 30, but they do not place
days 30–180 on the NEMO-scalar/NEMO-vector envelope.  At day 180 the new
SSH and U normalized RMS differences are respectively `8.20x` and `3.35x`
the twin values.  Under the explicitly post-hoc operational criterion that
both fields are at most four times the twin, chaotic separation first
dominates at day 283 and remains dominant continuously from day 314.

The year-5 eddy-fraction shift from `0.09642` to `0.08907` is `0.00736`,
`2.21x` the two NEMO realizations' same-year spread (`0.00333`).  It is
therefore a measured realization-shift signal, not "within spread".  It is
not evidence for reverting either source-faithful fix: chaotic separation is
already controlling the late first year, the new run is inside every
preregistered year-5 bar against both references, and a single realization
does not identify a shared-path physics defect.

The remaining entry-`kt=2` temperature boundary is last-bit coupled
composition.  No one-variable substitution removes its 3-ULP maximum; all
recorded inputs together make temperature bit-exact.  Salinity remains 2 ULP
because this record and seam expose temperature content only.  No new physics
owner is named and no shared kernel changes in this round.

## Finding 1 — divergence envelope

**Finding.**  The committed instrument uses the round-8 definition: daily
RMS(candidate-reference) divided by the RMS spatial anomaly of vector NEMO,
with SSH on the T mask and native surface U on the U mask.  It first replays
the scalar/vector NEMO twin.  Its SSH values at days 30, 60, 90, 120, 180 and
365 are exactly the independently printed `1.04e-9`, `1.17e-9`, `1.19e-8`,
`1.21e-6`, `2.95e-4` and `3.89e-3`; its exact and one-ULP controls separate.

The baseline is `post_fix_b401ec3f3`, before the round-8 live-QSR and round-9
TKE-shear fixes.  Thus this experiment measures their combined effect; it
does not pretend to attribute a share to either fix.

| day | twin SSH | new SSH | new/old SSH | new/twin SSH | twin U | new U | new/old U | new/twin U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | `1.042e-9` | `1.527e-7` | `0.00911` | `146.6` | `1.644e-9` | `1.614e-6` | `0.00168` | `981.8` |
| 60 | `1.169e-9` | `7.167e-5` | `0.348` | `61301` | `1.309e-9` | `2.243e-3` | `0.874` | `1.71e6` |
| 90 | `1.191e-8` | `6.064e-4` | `0.442` | `50928` | `6.865e-7` | `1.014e-2` | `0.450` | `14771` |
| 120 | `1.207e-6` | `1.124e-3` | `0.500` | `931.4` | `7.178e-4` | `3.267e-2` | `0.732` | `45.5` |
| 180 | `2.953e-4` | `2.422e-3` | `0.312` | `8.20` | `8.232e-3` | `2.761e-2` | `0.432` | `3.35` |
| 365 | `3.891e-3` | `1.307e-2` | `0.681` | `3.36` | `2.329e-2` | `7.042e-2` | `0.631` | `3.02` |

**Disposition.**  The preregistered "factor of a few at days 30–180"
expectation is REFUTED, although the directional expectation is confirmed:
every checkpoint is closer than the old legoESM run.  The first crossings of
`1e-3` normalized RMS are day 39 for U and day 108 for SSH.  Calling those
crossings the onset of chaos would be post-hoc speculation; the conservative
envelope-based statement is the four-times-twin criterion above.

**Evidence.**  `divergence_envelope/gyre_hires_divergence_report.json`
(`sha256 89e7424a3d62ce41f316f01dc9fef693d8703058666034773c889fca2bbe58d7`)
and `gyre_hires_divergence_arrays.npz`
(`sha256 d6089684c1cb891739e08e8c3ea8f926b380051a4c57799814358eaaae6c3dee`),
produced by instrument commit `9453f6a4e`.

## Finding 2 — yearly eddy fractions

**Finding.**  All four series use the same native-U/V 31-day nearest-boundary
high-pass definition and the full five-year daily record.

| year | NEMO vector | NEMO scalar | old legoESM | new legoESM |
|---:|---:|---:|---:|---:|
| 1 | `0.010832` | `0.010828` | `0.011207` | `0.010842` |
| 2 | `0.048782` | `0.051540` | `0.041334` | `0.038766` |
| 3 | `0.095254` | `0.096067` | `0.081062` | `0.071521` |
| 4 | `0.104894` | `0.100056` | `0.114202` | `0.101354` |
| 5 | `0.108264` | `0.104930` | `0.096424` | `0.089067` |

**Disposition.**  The year-5 old/new shift exceeds the observed same-year
NEMO twin spread and is a real realization signal under the requested rule.
It is not called a fidelity regression: year 4 moves into the NEMO envelope,
year 5 remains within the registered 20% climate bar, and separation is
chaotic long before year 5.  A new ensemble member, not another deterministic
shared-path edit, would be needed to estimate a climate-mean effect.

**Evidence.**  The same divergence report records all twenty yearly rows.

## Finding 3 — labelled scalar reference

**Finding.**  The previous scorer refused the scalar build because it lacks
the vector run's resolved receipt and has a different binary hash, even though
its `namelist_cfg` and `namelist_ref` are byte-identical.

**Disposition.**  The default path is unchanged and still requires the exact
certified vector manifest plus resolved-deck receipt.  A separate explicit
pair, `--alternative-nemo-manifest` and `--alternative-nemo-label`, admits
only a manifest at the selected root whose inventory matches and whose two
namelist hashes equal the certified deck.  It stamps the reference
`certified: false`.  Missing labels, external manifests, deck changes and use
of the default vector path against the scalar build all refuse in tests.

The year-5 scalar comparison passes all four bars: KE `0.016005` versus
`0.015668` (`2.15%`), eddy fraction `0.089067` versus `0.104930` (`15.12%`),
jet row `80` versus `80`, and SSH zero-crossing row `52.413` versus `51.658`
(`0.755` row).

**Evidence.**  `scalar_climate_score/gyre_hires_climate_report.json`
(`sha256 abab7123cbe60c9d208edb7ca6de863d55116c9e33872538fe495332f0060e7d`),
whose provenance labels the NEMO reference `scalar_math`, non-certified, and
records verdict `PASS`.

## Finding 4 — remaining entry-kt=2 boundary

**Finding.**  At the scalar step-1 vertical solve, production temperature is
different in 110,286/288,000 wet words, maximum `1.0658141036401503e-14 K`
(3 ULP); salinity is different in 103,327 words, maximum
`2.1316282072803006e-14` (3 ULP).  The one-variable substitutions give:

| substituted recorded input | T max | T unequal | S max | S unequal |
|---|---:|---:|---:|---:|
| `avt` before coefficient formation | `1.0658e-14` | 109,917 | `2.1316e-14` | 103,041 |
| formed tracer coefficient | `1.0658e-14` | 109,917 | `2.1316e-14` | 103,041 |
| Kmm interface thickness | `1.0658e-14` | 110,286 | `2.1316e-14` | 103,327 |
| Kaa cell thickness | `1.0658e-14` | 68,361 | `2.1316e-14` | 64,135 |
| formed temperature content | `1.0658e-14` | 65,432 | `2.1316e-14` | 103,327 |
| all recorded inputs | `0` | 0 | `1.4211e-14` | 63,449 |

Every active one-variable plant reaches temperature, and the disabled seam
is bit-identical to production.  The Kmm/Kaa thickness inputs themselves are
bit-exact; `avt` differs by at most 8 ULP and temperature content by 2 ULP.
Consequently the result rules out every recorded input as a single owner and
rules out the temperature solve operator once all its recorded inputs are
matched.  It does not identify a unique pair or explain salinity, whose
formed content is absent from the seam.

**Disposition.**  Classified `COUPLED_INPUT_COMPOSITION`; stopped without a
physics edit, as required for a composition boundary.  The single next
measurement is a source-ordered minimal-subset bisection beginning with the
formed coefficient plus formed temperature content, with an added recorded
salinity-content seam before making any S claim.

**Evidence.**  `step1_vertical_owner.json`
(`sha256 5a2345a6a84cec8ba896ee6b8764a3f17acbf803bef6d4bcd8ad70396d70248c`),
clean commit `98511f17f`, verdict `CONFIRMED_COMPOSITION`.

## Shared-path disposition

Nothing in the five-year run warrants another shared-path change.  Both
vector and scalar year-5 comparisons pass all climate bars; the divergence
curve shows the source-faithful fixes strongly reduce the early seed but do
not erase chaotic separation; and the only newly examined exact-rung debt is
coupled last-bit composition without a single causal input.  The next action
is the one minimal-subset measurement above, not a five-year rerun or a
configuration change.

## Inherited pins / OPEN

Inherited unchanged: shipped one-degree deck and digest, `r4` and `r4_smag`
namelists, 31 levels, RK3, split-explicit free surface, Redi tracers, TKE/EVD,
360-day forcing clock, scalar-libm CPU certification, zero-word ladder bar,
thickness-weighted daily temperature, registered native-GPU long runs and
explicit physical-GPU selection.  No deck value, physics knob or shared
kernel changed.

OPEN, in order:

1. Run the minimal-subset vertical-input bisection and expose recorded
   salinity content; do not infer a single owner from the all-input arm.
2. The entry-`kt=3` SSH boundary remains the largest early residual after the
   entry-`kt=2` composition boundary.
3. Quantifying climate sensitivity requires an ensemble; the two NEMO builds
   and two legoESM realizations do not estimate an ensemble mean.

## Tests and evidence

The divergence benchmark-shift plant, exact/one-ULP control, all scalar
manifest refusals, disabled vertical seam and all five active input plants
fire.  The focused climate/divergence/ladder set passes 44 tests.  The full
split lane battery's exact aggregate final line is:

`FULL_LANE_BATTERY_SPLIT_PASS: 135 passed, 2 deselected across 11 pytest processes`

The final citation gate on the committed receipt reports `PASS` for both
citations with zero citation and full-map audit failures; its shifted-anchor
self-tests fire.

## Compiled-source citations

The scalar writer captures the assembled tracer coefficient after the
surface coefficient is zeroed and before the source-ordered matrix terms are
formed at
`GYRE_OMIP_L2_P3_SMW_R7STEP2/BLD/ppsrc/nemo/trazdf.f90:431-457`.
The same executed routine forms the surface content RHS, performs the forward
recurrence and then the backward substitution at
`GYRE_OMIP_L2_P3_SMW_R7STEP2/BLD/ppsrc/nemo/trazdf.f90:527-559`.
