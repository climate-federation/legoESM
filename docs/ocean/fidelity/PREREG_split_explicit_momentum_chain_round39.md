# Split-explicit momentum chain round 39: residual ZAD operand factorial

Status: **FROZEN BEFORE MEASUREMENT**. Round 38's committed day-180 scorer
classified `WZV_REFUTED_AS_ZAD_MAJORITY_OWNER`: NEMO call-1 `ww` removes
70.716% of U and 89.323% of V error, leaving normalized RMS
`4.17258e-6/6.08806e-6`. The Kbb velocity arm is two to four orders worse,
and call 2 is worse than the structurally correct call 1. This amendment peels
the residual using retained inputs only.

## Frozen 2^3 design

The three binary factors are:

* **W**: production diagnosed `w` versus retained NEMO call-1 `ww`;
* **H**: production live face thickness versus a source-literal reconstruction
  of NEMO `e3u/e3v(Kmm)` from the restart SSH and mesh `e3u_0/e3v_0`, following
  `domqco.F90:160-181` and the live metric macros used at
  `dynzad.F90:104-107,115-118`;
* **A**: production vectorized kernel association versus an explicit
  surface-to-bottom transcription of `dynzad.F90:83-119` (`zWf`, neighbour
  sum, shear product, carried interface term, post-factor in source order).

All eight arms use the identical Kmm/current velocity; the scorer must prove
that it is bit-exact to the NEMO restart on wet U/V cells before judging the
factorial. It must also prove that the bridge's T/U/V metric arrays map to the
NEMO mesh arrays within `1e-15`; otherwise A would conflate association with a
metric-value change and the result is invalid. The production W0H0A0 arm must
reproduce round 38 exactly, and the W1H0A0 arm must reproduce its call-1 arm
exactly.

Every arm is scored on the unchanged active 3-D U/V population with the
accumulating `1e-12` normalized-RMS bar. Main effects and all two-/three-way
interactions are reported by the standard signed 2^3 contrast convention.
An operand or interaction owns the round-38 call-1 residual only if its
conditional substitution removes at least 90% of that residual in both U and
V. If W1H1A1 is AT BAR for both components, the direct ZAD operator is closed
and its owner is the earliest factor/interaction meeting that bar. If the full
arm remains DEBT, the unremoved residual remains OPEN and no local owner is
invented.

Round-38 receipt/hash, retained streams, state recipe, selector defaults,
full-halo/interior shapes, source files, and tracked-clean status are fail-
closed. Identity, longitude-roll, sign, wet-NaN, and two-bar controls must fire
for U and V. No held run, NEMO build, GPU, or MPI process is authorized.

### Stopped admission clarification (before any factorial score)

The first scorer attempt stopped before building or printing any arm because
the metric-alignment admission included the structurally dry outer V row.
The south-dropped mapping is exact on every wet V face (0 differing values),
while its sole difference is the final dry boundary row; the alternative
north-dropped mapping differs at 9,818 wet faces. The admission is therefore
clarified to the same active T/U/V populations on which every registered arm
is scored. This changes no factor, bar, input, or ownership rule.
