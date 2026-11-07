# NEMO testcase L2 GYRE high resolution — round 12 preregistration

Date: 2026-10-16

Incoming lane tip: `ea1a57c31dd135937c47e4b01ca1071442f5e72e`.

No measurement registered here changes a deck, a physical operator, a bar, or
the shared production trajectory.  The exact-rung oracle remains the admitted
scalar-math step-2 record at
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round7/oracle_gyre_r4_scalar_step2/run/`.
The long-run candidate remains registered run `post_round9_219db484b`; the
certified vector-math NEMO run is the climate reference and the admitted
scalar-math NEMO run is the arithmetic twin.

## R12.1 — source-ordered temperature-content walk

The scored boundary is the stage-3 `tra_zdf` temperature output at entry
`kt=2`, under the certified fp64/libm CPU policy.  The exact bar remains zero
unequal active float64 words.  The executed scalar source forms the right-hand
side in this order:

1. before content, `e3t(Kbb) * T(Kbb)`;
2. accumulated tendency content, `(p2dt * e3t(Kmm)) * T(Krhs)`;
3. the sum of those two addends, before the forward and backward recurrences.

The record carries every operand and the completed RHS.  The NEMO literal
rebuild must be bit-exact to the recorded RHS before any intervention is
admitted.  The production trace exposes the already-materialized content
boundaries; it does not feed them back into the trajectory.

For each addend, in the order above, the instrument reports the addend's
active-word exact score and two downstream interventions:

- the ordinary full-step context, with only that content addend replaced;
- a content-isolation context in which the round-11 companion inputs (formed
  diffusivity and selected Kaa thickness) are held at their admitted NEMO
  values while only that addend changes.

The without-substitution row is reported beside every arm.  An effect clears
the registered materiality threshold only if at least one active solved-T word
changes and the lexicographic residual `(unequal words, maximum absolute
error)` decreases.  This is a detection threshold, not a relaxed fidelity
bar.  An exact arm still requires zero unequal words.  Exact and one-ULP
plants must separate, and selecting no term must be bit-identical to its
baseline.

`TRANSCRIPTION_OWNER` requires one recorded addend, by itself, to remove the
residual in the content-isolation context and requires its production
formation to disagree with the executed NEMO expression for a named operand
or association.  If neither addend alone is exact, or if moving the association
without changing an input value changes the result, the result is
`COMPOSITION_DEBT`; quantify it and stop without a shared-path edit.  If both
addends are exact yet the solve is not, the recorded boundary is incomplete
and the result is `UNMEASURED`, never an inferred owner.

## R12.2 — day-30..180 momentum-term attribution

The leading momentum field/region carried forward from round 11 is native
surface V in the western-boundary wet-span bin.  SST had the largest regional
share, but pressure-gradient and Coriolis terms are not temperature-budget
terms; choosing V keeps every term named by the round-12 scope physically
defined.  No claim is made that V was uniquely first: all stored fields were
tied at the first daily sample above the four-times-twin criterion.

For days 30 through 180 inclusive, the requested rows are advection,
pressure/kinetic-energy gradient, total-vorticity/Coriolis, Smagorinsky lateral
diffusion, vertical mixing, surface stress, and their closure.  Each model is
sampled on the same daily convention and the same V-face region.  For each
term the report gives the signed time/region mean, RMS magnitude, signed
legoESM-minus-NEMO difference, and `abs(mean difference) / RMS(NEMO term)`.
The largest finite relative row is the measured carrier only if the summed
term difference closes the diagnosed total difference to the registered
floating-point closure bar.  Otherwise attribution refuses.

Archive admission is part of this measurement.  The existing NEMO run stores
full 3-D daily T/S/U/V and TKE diffusivities, but the registered legoESM run
stores only SSH, SST, surface U/V, and external-mode U/V.  A pressure-gradient,
3-D advection, or vertical-mixing budget cannot be reconstructed from those
2-D products.  Therefore the instrument must refuse the current legoESM
archive rather than fill missing levels, use a surface proxy, or infer a term
from the divergence map.  If the audit confirms those fields are absent, the
round will commit the smallest acquisition path that records matched 3-D
daily means over days 1–180 (scoring days 30–180), with the explicit physical
GPU and exact run commit in its manifest.  The acquisition is evidence work,
not authorization to launch a GPU job from this sandbox.

All new selectors have disabled-path identity controls and active-word plants.
Every new Python entry point receives a focused test.  The final receipt must
state separately what was measured and what remains acquisition-blocked.
