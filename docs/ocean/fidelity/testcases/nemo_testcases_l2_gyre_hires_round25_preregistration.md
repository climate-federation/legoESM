# GYRE higher-resolution round 25 recurring-rounding preregistration

Registered 2026-10-20 before producing either arm of the recurring experiment.

## Question and fixed intervention

Does a stage-1 momentum-composition residue of the measured round-22 size,
when it recurs at every model step, explain the days 30--180
legoESM--NEMO separation?  Start both arms from the same developed day-180
restart used in round 24 (SHA-256
`a8e33a3d22e4483bac777839f6cdc9bdf8c2e348c3a1d638adeafa114fff1c12`).
The control is the ordinary `r4_smag` production step.  At every step in the
intervention arm, add to the stage-1 complete momentum right-hand side the
unchanged round-22 pattern

`materialized-coefficient post-LDF accumulator - live-composed post-LDF accumulator`.

Do not rescale, translate, rotate, or recompute the pattern from the evolving
state.  Its admitted active maximum/count contract remains U
`1.852884572118782e-21` / 3,789 unequal words and V
`3.3881317890172014e-21` / 3,261 unequal words.  Apply it once at each of the
4,320 steps from step 4,321 through step 8,640 inclusive.  An identity
transform must be byte-identical to the ordinary first step, and the recurring
arm must move at least one state word at the first step.  Both arms otherwise
use the same production card, seasonal forcing, compiled-day scan, native fp64
GPU transcendentals, and daily-mean accumulation.  This is a non-certified
sensitivity experiment, not a model configuration or physics change.

## Registered metrics and controls

Use exactly the round-24 scorer.  Write matched daily means of SSH, SST, native
surface U/V, and prognostic depth-mean U/V.  Score recurring-minus-control with
the round-11 normalized-RMS and pattern-correlation definitions on the same
T/U/V masks at days 30, 60, 90, 120, and 180.  Over days 30--180, report the
same five per-row wet-span longitude bins and compare algebraic with
exponential fits in log absolute-RMS space.  Exact-self must score zero; a
one-ULP active plant in each field must score nonzero.

The comparison artifact remains the admitted round-11 array file
`gyrehr_rounds/round11/divergence_regional_v3/gyre_hires_divergence_arrays.npz`,
SHA-256 `d96f9cf31ec96d598d05418ce1835dbedc64b2e49c1f4be171c44913e1eb48b6`.
Its registered day-180 rows are unchanged:

| field | legoESM--NEMO | scalar--vector NEMO | ratio |
|---|---:|---:|---:|
| SSH | 0.0024221113795355876 | 0.00029525242111989495 | 8.203527579379362 |
| SST | 0.00431353791563564 | 0.0007076608127647132 | 6.095487891696821 |
| surface U | 0.027607691307419996 | 0.008231652544169346 | 3.3538455564399534 |
| surface V | 0.029898956932709626 | 0.011828076472453129 | 2.5277953691238366 |
| depth-mean U | 0.00845419214888829 | 0.001180081422270612 | 7.164075282721979 |
| depth-mean V | 0.013121674021533858 | 0.00232612599515822 | 5.64099883189747 |

## Verdicts fixed before acquisition

- **CONFIRM_FULL**: recurring/control normalized RMS at day 180 is at least
  the registered legoESM/NEMO value in all six fields.  A recurring residue of
  the measured size is sufficient to reproduce the entire observed separation.
- **REFUTE_ROUNDING_EXPLANATION**: recurring/control normalized RMS at day 180
  is no larger than the scalar/vector NEMO value in all six fields.  The
  recurring perturbation remains inside the independently measured rounding
  envelope.
- **PARTIAL_OR_INCONCLUSIVE**: every other result.  Report every field; do not
  turn a subset or a post-hoc threshold into either terminal verdict.

Growth class and leading region are descriptive discriminators and cannot
override the six-field terminal rules.  No production-physics, configuration,
or acceptance-bar change follows directly from this experiment.
