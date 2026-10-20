# GYRE higher-resolution round 24 closing-check preregistration

Registered 2026-10-20, before producing either arm of the closing experiment.

## Question and fixed intervention

Does the stage-1 Smagorinsky coefficient/operator composition residue measured
in round 22 explain the days 30--180 legoESM--NEMO separation?  Start both arms
from the shared developed day-180 restart (SHA-256
`a8e33a3d22e4483bac777839f6cdc9bdf8c2e348c3a1d638adeafa114fff1c12`).
The control is the ordinary `r4_smag` production step.  The intervention adds,
once and only once, to the step-4321 stage-1 complete momentum RHS the actual
round-22 pattern

`materialized-coefficient post-LDF accumulator - live-composed post-LDF accumulator`.

It is not a random perturbation and it is not rescaled.  Its admitted active
maximum/count contract is U `1.852884572118782e-21` / 3,789 unequal words and V
`3.3881317890172014e-21` / 3,261 unequal words.  An identity transform must be
byte-identical to the ordinary step, the seeded step must move at least one
state word, and both arms use the same ordinary production model thereafter.
Run both arms for 180 days with native fp64 GPU transcendentals.  This is a
non-certified sensitivity experiment, not a new model configuration.

## Registered metrics and controls

Write matched daily means of SSH, SST, native surface U/V, and prognostic
depth-mean U/V.  Score perturbation minus control with the round-11 normalized
RMS and pattern-correlation definitions on the same T/U/V masks, at days 30,
60, 90, 120, and 180.  Over days 30--180, report the five per-row wet-span
longitude bins and compare algebraic with exponential fits in log absolute-RMS
space.  Exact-self must score zero; a one-ULP active plant in each field must
score nonzero.

The comparison artifact is the admitted round-11 array file
`gyrehr_rounds/round11/divergence_regional_v3/gyre_hires_divergence_arrays.npz`,
SHA-256 `d96f9cf31ec96d598d05418ce1835dbedc64b2e49c1f4be171c44913e1eb48b6`.
Its day-180 registered rows are:

| field | legoESM--NEMO | scalar--vector NEMO | ratio |
|---|---:|---:|---:|
| SSH | 0.0024221113795355876 | 0.00029525242111989495 | 8.203527579379362 |
| SST | 0.00431353791563564 | 0.0007076608127647132 | 6.095487891696821 |
| surface U | 0.027607691307419996 | 0.008231652544169346 | 3.3538455564399534 |
| surface V | 0.029898956932709626 | 0.011828076472453129 | 2.5277953691238366 |
| depth-mean U | 0.00845419214888829 | 0.001180081422270612 | 7.164075282721979 |
| depth-mean V | 0.013121674021533858 | 0.00232612599515822 | 5.64099883189747 |

## Verdicts fixed before acquisition

- **CONFIRM_FULL**: the seeded legoESM/control normalized RMS at day 180 is at
  least the registered legoESM/NEMO value in all six fields.  A residue of the
  measured size is sufficient to reproduce the entire observed separation.
- **REFUTE_ROUNDING_EXPLANATION**: the seeded legoESM/control normalized RMS at
  day 180 is no larger than the scalar/vector NEMO value in all six fields.
  The perturbation remains inside the independently measured rounding envelope.
- **PARTIAL_OR_INCONCLUSIVE**: every other result.  Report every field; do not
  turn a subset or a post-hoc threshold into either terminal verdict.

Growth class and leading region are descriptive discriminators and cannot
override the six-field terminal rules.  No physics, configuration, or
acceptance-bar change follows directly from this experiment.
