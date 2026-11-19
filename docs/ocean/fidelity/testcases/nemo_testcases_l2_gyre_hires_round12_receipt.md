# NEMO testcase L2 GYRE high resolution — round 12 receipt

Date: 2026-10-16

Incoming lane tip: `ea1a57c31dd135937c47e4b01ca1071442f5e72e`.

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round12/`.
The exact-rung oracle is the admitted scalar-math step-2 record from round 7.
The climate reference remains the certified vector-math five-year NEMO run,
and the candidate remains registered run `post_round9_219db484b`.

## Verdict

**The first source-ordered temperature-content term that materially changes
the entry-`kt=2` residual is the accumulated `Krhs` content.  At this recorded
boundary it is a floating-point association/composition debt, not a measured
transcription defect, so no shared-path change is warranted.**  The before
content is word-exact.  Replacing the accumulated term makes the isolated
temperature solve word-exact, but the production and NEMO programs form the
completed content with different associations.

The requested day-30..180 momentum-term attribution is **UNMEASURED**, not an
inference from the existing surface maps.  Its archive preflight passes the
NEMO provider and refuses the completed legoESM run because that run contains
only two-dimensional surface and external-mode fields.  A new observer-free
driver mode and exact operator command are committed; the GPU acquisition is
the only blocker.

## Finding 1 — source-ordered temperature-content walk

**Finding.**  The NEMO literal reconstruction calibrates exactly to the
recorded completed RHS.  The two input boundaries then score as follows:

| source-ordered boundary | unequal active words | maximum | maximum ULP |
|---|---:|---:|---:|
| before content, `e3t(Kbb) * T(Kbb)` | 0 | 0 | 0 |
| completed content after the exact before term | 78,243 | `2.2737367544323206e-13` | 2 |

The ordinary full-step result begins at 110,286 unequal temperature words,
maximum `1.0658141036401503e-14 K` (3 ULP).  The registered interventions are:

| context | substitution | T words before -> after | changed output words | material |
|---|---|---:|---:|---|
| ordinary | before content | 110,286 -> 110,286 | 0 | no |
| ordinary | accumulated `Krhs` content | 110,286 -> 65,432 | 67,418 | yes |
| content isolation | before content | 67,325 -> 67,325 | 0 | no |
| content isolation | accumulated `Krhs` content | 67,325 -> 0 | 67,325 | yes |

The “both addends” arm is identical to the accumulated-only arm because the
before addend is already exact.  The production path evolves the same
mathematical content as a previously associated advection content plus a
separate source increment; the executed NEMO statement materializes
`before + (dt * e3t(Kmm)) * Krhs`.  Thus the registered boundary is
`COMPOSITION_DEBT`.  This classification is deliberately local: it does not
claim that every elementary operator already accumulated into `Krhs` has been
independently proved exact.

**Disposition.**  Stopped without changing the shared tracer kernel.  A
shared-path rewrite merely to select a different fused expression would
violate the campaign rule against forcing bit identity for a measured
composition difference.

**Evidence.**  `step1_temperature_content_walk.json`, SHA-256
`0e44245633915b564feb3706fe88c522a0a16fb15da6f530ec4df07489568d1d`,
from clean instrument commit `b01d8f67afac6964fcf811c849fd1bc7640c4d35`.
The passive trace and all-false selector each move zero T and S words.  The
completed-content one-ULP plants move 251,856 ordinary-context and 251,798
isolation-context temperature words, so both intervention channels fire.

## Finding 2 — the term-budget archive is insufficient

**Finding.**  The committed preflight audited days 1, 30, and 180 and all four
NEMO tile families:

| provider | result | evidence |
|---|---|---|
| NEMO r4_smag | PASS | 8 T, 8 U, 8 V, and 8 W tiles; 1,825 daily records; all registered variables and DOMAIN placement attributes present |
| legoESM `post_round9_219db484b` | REFUSE | no 3-D T, S, U, V, TKE, or layer-thickness snapshots or daily means; manifest has no registered full-state window |

NEMO's T provider contains the thickness-weighted tracers and same-file
thickness; its W provider contains the recorded vertical tracer and momentum
diffusivities.  The legoESM archive has SSH, surface T/U/V, and external-mode
U/V only.  Therefore pressure-gradient, 3-D advection, and vertical-mixing
terms cannot be recomputed over the registered native V faces, and neither a
largest relative term nor the required closure can be reported from this run.

**Disposition.**  `ACQUISITION_NEEDED`.  The GYRE driver now accepts
`--per-step --full-state-window 1:180` and writes 3-D T/S/U/V/TKE/layer
thickness as snapshots and daily means.  T/S use
`mean(tracer*e3t)/mean(e3t)`; other fields use arithmetic per-step means.  The
ordinary compiled `model.step` is called before host-side NumPy accumulation,
so the observer is outside the compiled physics graph.  The manifest records
the deck, commit, native-GPU policy, explicit physical GPU, per-step mode,
window, fields, cadence, and daily hashes.  It refuses a full-state window
without `--per-step`.

**Evidence.**  `momentum_budget_archive_preflight.json`, SHA-256
`0b9ec8ad4503ae5edc5b08f375ee3af01df037d566fdef4311656321888d702c`,
from committed preflight `286a1f77d`.  The exact operator acquisition is:

```text
cd /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-gyrehr-3145520903
scripts/run/run_gyre_hires_gpu.sh --physical-gpu 0 --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/legoesm_runs/r4_smag_gpu_budget_180d_round12 --days 180 --per-step --full-state-window 1:180
```

The operator must run this only after confirming physical GPU 0 is idle; the
campaign rule remains “either idle GPU, never kill a running job.”  No GPU job
was launched from this sandbox.

## Shared-path disposition and single next measurement

No shared-path change is warranted.  The exact-rung finding is a measured
composition debt, and the long-run term owner remains unavailable rather than
guessed.  The single next measurement is the preregistered day-30..180 native
surface-V western-boundary budget from the acquired full-state archive:
advection, pressure/kinetic-energy gradient, total-vorticity/Coriolis,
Smagorinsky lateral diffusion, vertical mixing, surface stress, and closure,
with the carrier selected by `abs(mean legoESM-NEMO) / RMS(NEMO)` only after
the summed difference closes.

## Inherited pins / OPEN

Inherited unchanged: the shipped one-degree digest `99b2dfce324b0f77`, the
one-degree no-worsening ceiling, resolved `r4` and `r4_smag` namelists, 31
levels, RK3, split-explicit free surface, TKE/EVD, Redi tracers, 360-day
forcing clock, scalar-libm CPU certification, the zero-word exact-rung bar,
thickness-weighted NEMO daily T/S, and explicit physical-GPU selection.

The shipped trajectory remains exact at entry `kt=1`, within its existing
statistic bar but non-exact at `kt=2`, and DEBT from `kt=3`; this round makes
no production-model change.  OPEN, in order:

1. Acquire the 180-day matched full-state legoESM archive and perform the
   registered momentum-term closure and attribution.
2. Localize the existing entry-`kt=3` SSH debt by scalar step-2 RK stage.
3. If that debt first enters the external mode, acquire scalar step-2
   barotropic substeps; the admitted scalar record has step-1 substeps only.
4. Post the receipt to GitHub issue #1455 when network access is available.

## Tests and controls

The split lane battery reports:

`FULL_LANE_BATTERY_SPLIT_PASS: 129 passed, 4 deselected across 12 pytest processes`

It covers both new entry points and the inherited acquisition, 30-day ladder,
climate, divergence, exact-rung ladder, movie, shipped ceiling, card
reconciliation, phase-3, and Smagorinsky suites.  Controls include passive
trace identity, disabled-selector identity, completed-content one-ULP plants,
materiality positive and negative cases, a missing required archive field, a
non-finite provider, a refusing-provider joint plant, explicit GPU selection,
full-state-window refusal without per-step mode, thickness-weighted tracer
means, and equality of the observed and ordinary fake-model trajectories.
The receipt citation gate reports PASS: one citation, zero failures, zero
unmapped citations, and all 9/9 self-tests firing.  Its committed-receipt
artifact is `receipt_citation_gate.json`, SHA-256
`b93fd3ae2ffc64822e186817167b79e9467a1aacdb4c7a69932eebb7fe953ba1`;
the citation-gate unit suite passes 17/17.

## Compiled-source citations

The executed scalar routine forms the temperature RHS from the before content
and the accumulated `Krhs` content, then performs the forward and backward
recurrences at
`GYRE_OMIP_L2_P3_SMW_R7STEP2/BLD/ppsrc/nemo/trazdf.f90:527-559`.
