# DINO channel seasonal attribution result — corrected after adversarial review

**Corrected verdict: all four tested state-derived drivers have NO DETECTED
PHASE RELATION — indistinguishable from random quarter relabelling.** This is
not an affirmative refutation of density, stratification, mixed-layer depth or
thermal-wind involvement. The four-horizon experiment has only three seasonal
degrees of freedom after mean removal, and every driver-advantage interval
straddles zero.

The trivial fixed-fraction explanation remains **REFUTED** by its direct
proportionality residual. Prescribed wind stress remains excluded, but only as
a **ZERO FORCING GAP**, not because wind work is time-independent.

## What changed after review

The `SHIP AFTER CORRECTIONS` review reproduced every recomputable value exactly
but found that the original labels overstated the power of one four-horizon
cycle. Corrections C1--C6 and C8 are applied; the supplied fix list contains no
C7 entry.

- The iid, exact-`r=1`, `N_eff=35` confirmation plant is retracted as a
  deviation from the registered *smooth* plant. It is replaced by a committed
  six-realization AR(1) power sweep.
- All four public driver labels are downgraded from the raw registered
  `REFUTES_PHASE_TRACKING` arm to **NO DETECTED PHASE RELATION —
  indistinguishable from random quarter relabelling**. The raw classifications
  remain in the artifact as `registered_status`.
- The `-1/3` balanced four-phase correlation baseline and three seasonal
  degrees of freedom are printed with the profile matrix.
- The Fisher aggregate is explicitly descriptive: its four horizon entries are
  dependent after mean removal.
- Wind exclusion now rests on identical prescribed stress, hence zero imposed
  cross-model forcing gap. `tau*u` can still be seasonal because `u` evolves.
- Measured controls, including the power sweep, are embedded in the artifact.
- `UNRESOLVED_EFFECTIVE_N` is recorded explicitly in the preregistration as a
  post-result amendment; it cannot enter confirm or non-detection counts.

## Detectable-effect floor

The replacement plant uses six seeds, 35 rows, AR(1) row coefficient
`phi=0.85`, four-horizon anomaly projection, and 5,000 moving-block draws for
each realization. The detectable-effect floor is defined before computation as
the lowest planted tracking coefficient `rho` that confirms in all six
realizations.

| planted rho | confirmations | matched-score range | minimum effective-row range |
|---:|---:|---:|---:|
| 0.60 | 0/6 | 0.286--0.563 | 3.14--9.09 |
| 0.75 | 4/6 | 0.477--0.738 | 3.00--6.91 |
| 0.90 | 6/6 | 0.796--0.897 | 3.00--6.25 |
| 0.95 | 6/6 | 0.911--0.949 | 3.00--6.06 |
| 0.975 | 6/6 | 0.959--0.974 | 3.00--5.97 |
| 0.99 | 6/6 | 0.984--0.990 | 3.00--5.93 |

The resulting design-specific detectable-effect floor is **rho >= 0.90**.
This is an exceptionally strong relation and does not show useful confirm-arm
power at the observed effect scale.

Measured pushback is retained rather than reconciled away: the reviewer's
independent autocorrelated sweep confirmed `0/6` at `rho=0.90` and `2/6` at
`rho=0.95`, while its `rho=0.60` cases scored `0.49--0.68`. The committed plant
uses the subsequently fixed seeds and AR(1)/anomaly construction and therefore
obtains a different floor. Together, the two sweeps show that confirm-arm power
is sensitive to smooth pattern geometry. Both nevertheless place the observed
driver scores (`0.04--0.12`) far below a moderately tracking synthetic driver.

## Target reconstructed at all horizons

The artifact contains all 35-row profiles for all four members, both models,
and days 90, 180, 270 and 360. The SHA-pinned upstream reducer reproduces every
stored gap with maximum absolute residual `0.0 Sv`.

| Horizon | raw gap gross, mean +/- member SD (Sv) | seasonal-anomaly gross, mean +/- member SD (Sv) | ensemble anomaly RMS (Sv/row) |
|---:|---:|---:|---:|
| day 90 | 0.08096 +/- 0.00030 | 0.08782 +/- 0.00281 | 0.004183 |
| day 180 | 0.07205 +/- 0.00475 | 0.03944 +/- 0.00299 | 0.002416 |
| day 270 | 0.12654 +/- 0.00918 | 0.06932 +/- 0.00687 | 0.003884 |
| day 360 | 0.13470 +/- 0.01091 | 0.10696 +/- 0.00840 | 0.005385 |

The ensemble seasonal-anomaly profile correlations are:

```text
          90       180       270       360
90     1.0000    0.0395   -0.6152   -0.2981
180    0.0395    1.0000    0.6362   -0.9165
270   -0.6152    0.6362    1.0000   -0.5449
360   -0.2981   -0.9165   -0.5449    1.0000

balanced four-phase mean-pairwise baseline: -1/3 = -0.3333
observed mean of the six off-diagonal pairs:       -0.2832
```

Mean removal forces the four anomaly fields into a three-dimensional seasonal
subspace; `-1/3` is the balanced equal-energy pairwise baseline. The mean
`-0.283` is therefore near baseline and is not independent evidence for
rotation. The informative descriptive feature is the wide pairwise spread,
from `-0.916` to `+0.636`. Day-360 gross still exactly recovers the upstream
`0.134695 +/- 0.010907 Sv`; target saturation is `0/35` rows.

## Trivial explanation tested first

**REFUTES_FIXED_FRACTION_EXPLANATION.** With model seasonal anomalies
`A_side`, define `C=(A_lego+A_nemo)/2` and `D=A_lego-A_nemo`. The zero-intercept
fit `D=beta*C` has pooled `beta=-0.00509`, NRMSE `0.98356`, and member NRMSEs
`0.98519, 0.98869, 0.98705, 0.97087`. All exceed the registered `0.70` residual
bar, so the gap is not a fixed fraction of a shared rotating field.

The direct `C -> D` phase score is `0.17890` versus best cyclic score `0.33858`;
its advantage interval `[-0.38195, 0.04421]` straddles zero and is interpreted
as a non-detection. Whether legoESM and NEMO rotate identically remains
`UNRESOLVED_EFFECTIVE_N`: all Bartlett counts reach the three-row Fisher floor.
Neither phase result is needed for the proportionality refutation.

## Wind exclusion

**EXCLUDED_A_PRIORI_ZERO_FORCING_GAP.** Both cards prescribe the same zonal
stress from fixed latitude knots, so the imposed stress difference between
models is identically zero. Time-independent stress does **not** imply
time-independent wind work: `tau*u` can vary seasonally through `u`. Wind work
is therefore not claimed absent; prescribed stress is excluded only as an
independent cross-model seasonal forcing-gap driver. Source, namelist and
SHA-256 receipts remain recorded.

## Relabelled driver table

The matched score is the effective-row-weighted Fisher aggregate across the
four dependent horizons. The table retains the original numbers, 5,000/5,000
draws per real leg, and member replication, but reports the corrected
interpretation.

| Leg | corrected verdict | matched score | best cyclic score | matched - null (98.75% interval) | median/min effective rows | saturation |
|---|---|---:|---:|---|---:|---:|
| density inventory | **NO DETECTED PHASE RELATION (u)** | 0.0576 | 0.3333 | -0.2757 `[-0.4803, 0.2091]` | 9.16 / 3.39 | 0/35 |
| vertical stratification | **NO DETECTED PHASE RELATION (u)** | 0.1227 | 0.1984 | -0.0757 `[-0.3553, 0.2516]` | 8.04 / 3.71 | 0/35 |
| mixed-layer depth | **NO DETECTED PHASE RELATION (u)** | 0.0610 | 0.3492 | -0.2882 `[-0.5235, 0.2068]` | 24.86 / 11.21 | 0/35 |
| thermal wind | **NO DETECTED PHASE RELATION (u)** | 0.0403 | 0.1438 | -0.1035 `[-0.2521, 0.2421]` | 24.07 / 4.98 | 0/34 |

Every ensemble mean and all four members receive the corrected non-detection
label. Every advantage interval crosses zero. Density inventory and MLD also
have best cyclic-null scores `0.333` and `0.349`, above the old `0.30` low bar;
the bar lies inside the measured null range and cannot support affirmative
refutation. `(u)` means all tested positions are unsaturated; thermal wind has
34 adjacent-row faces.

## Weighting, provenance, controls and verification

- Density inventory and stratification remain volume/thickness weighted; MLD
  uses the canonical 0.01 kg m-3 relative-to-10 m diagnostic and area weights;
  thermal wind uses the committed depth/thickness integrand and face reducer.
- The audit, atlas, channel reducer and upstream artifact are loaded from
  SHA-256-checked git objects. All launch/member paths, oracle restart clock
  (`15,552,000 s = 180 d`) and the NEMO member-3 exception were reread and
  recorded.
- Seven failure plants fired and are stored under `controls`: pinned-source
  mutation, time-dependent wind signature, layer-average substitution, MLD
  immobility, thermal-wind immobility, independent-row assumption, and an
  independent-row claim for the autocorrelated phase plant.
- The prior and corrected artifacts are numerically identical for every
  transport array, driver array, matched score, best-null score and advantage;
  maximum recomputation residual is `0.0` for each.
- The focused suite passes `11/11`; `ruff` is clean; the prescribed interpreter
  resolves imports from this worktree.

The corrected producer was clean at
`738c2c23d005aa38586b37810ee02d64fe6385a9`. The artifact is
`docs/ocean/fidelity/dino_channel_seasonal_artifact.json`, SHA-256
`5c29782e62764641fae0871d1b3d407c15140609a7f46baf18dd579932c5e043`.
