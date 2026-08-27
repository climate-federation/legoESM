# Per-row rescore of the circumpolar-channel verdict

**Status: POST-REVIEW CORRECTED.** This revision applies both adversarial
reviews' blocking inference and engine changes. The coordinator still owns
promotion to a campaign finding.

**Instrument:** `scripts/validate/ocean_fidelity/dino_1226/channel_rescore.py`
at clean producer `7af72bc33ef560ed0c789c322f28cb572a9d3fc6`.

**Pre-registration:** `PREREG_channel_rescore.md@e2fd09158`, committed before
any verdict360 channel-rescore statistic was computed. Post-review deviations
from that registration are named below rather than silently backdated.

**Artifact:** `docs/ocean/fidelity/dino_channel_rescore_artifact.json`, SHA-256
`fd39b99842e7c20a5f42b069f8ff732e6c515dee0b0ab00bea1ddae2f52ef11b`.
All measurements are offline reductions of the same saved verdict360 states;
no model was stepped and no GPU was used.

## Corrected verdict

> The day-360 channel row-gap profile is smooth (**CONFIRMED**, decider C),
> above the current ensemble spread at profile level (**PLAUSIBLE**, because
> all 35 row floors are unsaturated), and temporally reorganizing
> (**CONFIRMED coherent-pair guard**, while fixed-pattern persistence remains
> **UNRESOLVED**, decider A). The day-360 band net is only 0.254 of its own
> ensemble floor (**UNMEASURABLE for an agreement claim**). Row 13, just
> outside the pinned channel mask, carries +0.032230 Sv (**CONFIRMED measured
> gap**, not a mask-sensitivity experiment).

The emitted verdict is:

`PLAUSIBLE_TEMPORALLY_REORGANIZING_UNRESOLVED[A_fixed_pattern_persistence,D_row_agreement,E_band_sensitivity,B_member_stability]`

The row-level disagreements are therefore not supported as decorrelating
noise. They are coherent, seasonally reorganizing structure, but the stronger
claim that the 17 ppm full-year statistic masks a *confirmed* real pattern
error is withheld: row agreement and band sensitivity are both unresolved.
The observed day-360 gross profile magnitude is **0.135 +/- 0.011 Sv** across
members (descriptive, named denominator: member sample standard deviation),
not the former six-digit member-0 claim.

The former statements “genuine row-level agreement REFUTED,” “member-stable
pattern CONFIRMED” as an independent leg, edge sensitivity `E=3.680`, and the
pairing of full-year 17 ppm with day-360 `X` are all **RETRACTED** in the probe's
printed output and JSON, not only in this prose.

## Registered bars and post-review scores

| leg | named denominator and bars | measurement | label |
|---|---|---|---|
| A: fixed-profile persistence | centered Pearson energy; Bartlett effective rows; persistent if signed median r >=0.70 and block lower >0.30; decorrelating if r <=0.30 and upper <0.70 | signed median **-0.06556**, 95% block CI **[-0.16454, 0.74263]**, median/min effective rows **8.25/6.40** | **UNRESOLVED** |
| A guard: coherent reorganization | post-review guard with denominator `max(pair |r|)`; decorrelation forbidden if any pair clears 0.70 | median |r| **0.25110**, max |r| **0.93487** | **CONFIRMED coherent reorganization present** |
| B: across members | registered Pearson construction, now recognized as the forced amplitude ratio S/(S+N), not independent shape evidence | pooled raw r **0.99672**; day-360 raw r **0.97189** | **DEMOTED construction-forced amplitude** |
| C: spatial structure | roughness denominator `4 sum(g-gbar)^2`; smooth if Z <=0.25 and Z <0.60 | **Z=0.18063**; **1/30** floor-resolved adjacent pairs flips sign | **CONFIRMED physical/smooth structure** |
| D: row agreement | 35 row-specific floors; confirm if >=90% within 2 floors and <=10% materially beyond 5; refute if materially beyond-5 fraction >=25% | control: **4/35** within 2; raw **17/35** beyond 5 is an upper bound; exact repo materiality leaves **3/35** | **UNRESOLVED** |
| E: band sensitivity | `C=|sum gap|/sum|gap|`; confirm if median C <=0.10 and `J=max|row gap|/band floor >=2`; edge term withdrawn | median **C=0.10230**, `J=1.85812` | **UNRESOLVED** |
| day-360 band net | exact same-window band floor **0.0096045 Sv** | net **+0.00243969 Sv = 0.25401 floor** | **UNMEASURABLE below own floor; no agreement claim** |

The moving-block bootstrap retained all **5,000/5,000** draws. Its signed-r
median is **+0.022724**, beside the point estimate -0.065556; the point lies at
the **28.5th percentile** of the bootstrap distribution. This records the skew
called out by review rather than hiding it in an interval.

## Temporal pattern

The four-member median correlations show why signed-r cannot be read as a
noise test:

| horizon pair | median r |
|---|---:|
| day 90 vs 180 | +0.20731 |
| day 90 vs 270 | -0.10989 |
| day 90 vs 360 | **-0.79243** |
| day 180 vs 270 | **+0.89395** |
| day 180 vs 360 | -0.29535 |
| day 270 vs 360 | -0.01124 |

Day 90 versus day 360 is a coherent sign reversal, while day 180 versus 270 is
a strongly aligned profile. A fixed pattern with one sign flip is therefore no
longer allowed to classify as decorrelating noise. One annual cycle cannot
separate seasonal reorganization from elapsed-time evolution.

## Row floors, member spread, and saturation

All four paired members give the same registered D classification:

| member | within 2 floors | raw beyond 5 floors (upper bound) | survives exact repo materiality | D label | X [Sv] |
|---:|---:|---:|---:|---|---:|
| 0 | 4/35 | 17/35 | 3/35 | UNRESOLVED | 0.122876 |
| 1 | 5/35 | 22/35 | 5/35 | UNRESOLVED | 0.130978 |
| 2 | 2/35 | 24/35 | 5/35 | UNRESOLVED | 0.148892 |
| 3 | 2/35 | 23/35 | 5/35 | UNRESOLVED | 0.136035 |

Mean `X` is **0.134695 Sv**, with member sample standard deviation
**0.010907 Sv** (`mean/std=12.35`). This supports the descriptive statement
that the profile amplitude is above the *current* ensemble spread. It is not
promoted because the saturation instrument reports **0/35 saturated rows**.

The band floor is also measured unsaturated, not stamped with a hardcoded
string. Its per-side growth ratios are legoESM **8.930** and NEMO **12.104**
from days 180 to 270, then legoESM **1.397** and NEMO **0.826** from days 270 to
360. The reason emitted by the shared saturation engine is
`lego 180->270 8.93; nemo 180->270 12.10; lego 270->360 1.40`.

## The band and row 13

The full-year 17 ppm statistic and day-360 profile are different windows. At
day 360, the exact control-member group-reduced NEMO transport is
**36.457698 Sv**. The **+0.00243969 Sv** net is therefore **66.918 ppm** with
that named, same-window denominator. More importantly, it is only **0.254** of
the same-window ensemble floor, so it carries no agreement claim either way.

Registered edge `E` is withdrawn. Rows 13 and 49 are not alternative channel
edges: each has **50/52 wet T columns and 49/52 wet U columns**, whereas every
channel row has 52 wet columns. Moving the mask crosses a blocked topology and
changes the functional.

The useful observation exposed by that invalid diagnostic is retained:

| quantity | measured value |
|---|---:|
| row-13 gap | **+0.0322305 Sv** |
| row-13 / day-360 band net | **13.2109x** |
| largest absolute in-channel member-0 row | 0.0178464 Sv |
| row-13 / largest absolute in-channel row | **1.8060x** |
| row-13 / band floor | 3.3558x |

Row 13 is a separate adjacent-basin finding, not evidence that the registered
channel mask is tunable.

## Engine tests and mutation controls

The test suite now exercises `evaluate`, `aggregate_stability`,
`spatial_decider`, and `moving_block_indices` through a synthetic
four-member-by-four-horizon ensemble with a known smooth persistent pattern
and a pattern-removed twin with a different exact verdict string.

| reviewed mutation | post-fix planted result |
|---|---|
| M8: per-row floors replaced by one global floor | **RED**: analytic heteroscedastic floor vector mismatch |
| M11: moving-block bootstrap replaced by identity | **RED**: unique-draw/CI contract fails |
| M12: force `CONFIRMED_PERSISTENT` | **RED**: pattern-removed twin verdict string changes |
| M14: force `CONFIRMED_PHYSICAL_STRUCTURE` | **RED**: pattern-removed twin spatial result and verdict change |
| M15: disable reducer SHA pin | **RED**: planted wrong SHA must raise `SystemExit` |
| M16: member pairs changed to self-comparisons | **RED**: zero-self-pair and day-360 `r<1` contracts fail |

Direct result: **18 passed**; ruff: **clean**. The probe's six arithmetic
plants also all print `PLANT FIRED`; the two former costume plants now invoke
real guards. The lag `n-1` one-pair Bartlett term is omitted and this
post-registration correction is stamped in the artifact.

## Reviewer-number reconciliation

Two requested numerical restatements do not reproduce under their named
denominators, and the committed artifact records both the repo-exact result and
the reviewer variant:

1. `regional_audit.u_is_material` uses the **maximum per-side** last-quarter
   growth, as its source and self-test require. Called verbatim row by row, it
   voids **14 of 17** raw member-0 beyond-5 rows, leaving **3/35**. The review's
   **9 voided / 8 surviving** reproduces exactly only when the two sides are
   first RSS-combined and that combined floor's growth is substituted. The
   artifact records that variant as
   `NOT_SCORED_REPO_RULE_USES_PER_SIDE_GROWTH`. Both give D **UNRESOLVED**, so
   the verdict direction is unchanged. Per the task rule, the repo instrument
   wins.
2. The review's **71 ppm** uses an approximately 34 Sv denominator. The exact
   same-window NEMO group transport is 36.457698 Sv, giving **66.918 ppm**.
   Likewise, row 13 is **1.806x**, not 2.6x, the largest absolute in-channel
   member-0 row. The review's **13x band-net** comparison does reproduce
   (13.2109x).

## Provenance

The artifact records the real scored-day list, launch SHA, all four member
paths and stamps, exact NEMO directories, and the oracle restart clock
(**15,552,000 s = 180 d**). It records that `control_clock` returns `None`, the
actual recovered clock value, the legacy-clock environment before/after the
verified substitute path, `--skip-upstream-self-test=false`, bootstrap retained
and dropped counts, producer cleanliness, and the pinned upstream reducer
SHA-256.

The branch-layout discrepancy remains: this checkout originally lacked the
task's expected `AGENTS.md` and regional-audit files. The rules were read from
committed source `24be4d763`; the exact audit and atlas sources were imported
from `fidelity/dino-regional-audit@102ef501a`, with the audit SHA-256 pinned and
verified before every measurement.
