# Rows 13 and 49 at the DINO channel edges

**Status: corrected after an independent adversarial NO-SHIP review.**  The
reviewer reproduced every original number, then identified circular reuse of
day 360, low-n overclaiming, an incomplete power control, incomparable raw
projection scales, a missing opposite-neighbour veto, an unreachable
asymmetry arm, and an uncommitted artifact.  The corrected result has not yet
received a second adversarial pass.

**Instrument:**
`scripts/validate/ocean_fidelity/dino_1226/row13_edge_attribution.py` at clean
producer `2a1c7ca23a484562e4888061a1ee4595eeb174a0`.

**Pre-registration:** `PREREG_row13_edge_attribution.md` at `ea416520b`,
committed before the first state statistic was computed.  The review-mandated
correction protocol was fixed in the same file at `59646d40d`, before any
corrected statistic was computed; it is explicitly post-registration.

**Artifact:** `docs/ocean/fidelity/dino_row13_edge_attribution_artifact.json`,
SHA-256
`6447be3c028d0a9d98c3c317266eda020f4e1d58677efbd9f50b5e157dc2f8c3`.
It contains every member, horizon, level and longitude.  Everything is offline
from the saved verdict360 states; no model was stepped.

## Retractions

The original day-360-template attribution is **retracted**.  Day 360 defined
the regional template and also dominated the four-point correlation.  The
original **“row 49 CONFIRMED P3 basin-edge”** statement is likewise
**retracted**.  Both retractions are first-class fields in the committed JSON
and are printed before any corrected score.

## Verdict

> **Row 13 is UNRESOLVED_LOW_N.**  With horizon-held-out, RMS-normalized
> templates, the control-member trajectory has `r = +0.981` against P1
> (`p = 0.019`) and `r = -0.868` against the channel (`p = 0.132`).  Absolute
> profile separation is only `0.11`, and channel-side row 14 (`|r| = 0.989`)
> is closer than basin-side row 12 (`0.862`), vetoing a basin route.

> **Row 49 is UNRESOLVED_LOW_N; the P3 ownership claim is withdrawn.**  Its
> control trajectory has `r = +0.914` against held-out P3 (`p = 0.086`) and
> `r = +0.638` against the channel profile (`p = 0.362`).  Although the
> descriptive profile separation is `0.28`, channel-side row 48 (`|r| =
> 0.928`) remains closer than basin-side row 50 (`0.887`) and activates the
> opposite-neighbour veto.  The replacement statement is: **row 49 is a
> low-n, boundary-adjacent saved-state pattern with P3-profile similarity but
> no attributable owner.**

For four horizons, Pearson's null is exactly uniform on `[-1,1]`, so every
unadjusted two-sided receipt is `p = 1 - |r|`.  The `0.70`, `0.50`, and `0.30`
bars therefore correspond to `p = 0.30`, `0.50`, and `0.70`; they are only
descriptive routing bars.  The four 1e-14-nudge members have effective
independent replication approximately one.  Member 0 is primary; members 1--3
are sensitivity traces, and no median-over-members evidentiary gain is claimed.

Both absolute gap verdicts are withheld.  At day 360 row 13 is
`+0.0322305 Sv / 0.0038435 Sv = 8.386` current floors and row 49 is
`+0.00311888 Sv / 0.000324894 Sv = 9.600` current floors, but **both floors are
unsaturated and the audit's exact materiality test says continued floor growth
could overturn both `no` verdicts**.  These are gap **upper bounds**, not
confirmed gaps.

The amplitude-aware symmetry cell now fires independently of locally scaled
floors: row 49's day-360 median absolute gap is `0.1005` of row 13's, an
absolute difference of `0.029543 Sv`, so the saved states show
**DESCRIPTIVE_ROW13_AMPLITUDE_DOMINANCE_UNSATURATED**.  Trajectory symmetry is
still unsupported: control `|r| = 0.247`, exact `p = 0.753`.  This is an
upper-bound amplitude description, not a confirmed physical asymmetry.

## Per-member, per-horizon trajectories

Gap is legoESM minus NEMO, positive eastward.  The floor in each row is the RSS
of that row's own four-member sample standard deviations on the two sides.  The
multiple is for member 0.  Earlier-horizon multiples are descriptive upper
bounds; the registered two-quarter saturation test ends at day 360.

| row | day | m0 [Sv] | m1 [Sv] | m2 [Sv] | m3 [Sv] | row floor [Sv] | m0/floor |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 | 90 | +0.003578944 | +0.003579774 | +0.003579797 | +0.003545431 | 0.000017040 | 210.04 |
| 13 | 180 | −0.006508768 | −0.006594467 | −0.004771926 | −0.006228822 | 0.000755927 | 8.61 |
| 13 | 270 | −0.016842556 | −0.016988870 | −0.018100486 | −0.018004799 | 0.001516530 | 11.11 |
| 13 | 360 | +0.032230452 | +0.033455548 | +0.036496092 | +0.027243104 | 0.003843501 | 8.39 |
| 49 | 90 | +0.000816190 | +0.000816182 | +0.000816206 | +0.000815667 | 0.000000263 | 3104.15 |
| 49 | 180 | +0.002466023 | +0.002617222 | +0.002493821 | +0.002369420 | 0.000102239 | 24.12 |
| 49 | 270 | +0.004615416 | +0.004647341 | +0.004682005 | +0.004422761 | 0.000149104 | 30.95 |
| 49 | 360 | +0.003118879 | +0.002809138 | +0.003720279 | +0.003480727 | 0.000324894 | 9.60 |

Row 13 changes sign twice: small positive at day 90, negative at days 180/270,
then a much larger positive at day 360.  Row 49 stays positive, grows through
day 270, then retreats.  The matched-edge trajectories are weakly anticorrelated
member by member (`r = −0.247, −0.352, −0.072, −0.166`).

Day-360 saturation receipts:

| row | legoESM spread [Sv] | NEMO spread [Sv] | floor [Sv] | saturation reason | exact materiality |
|---:|---:|---:|---:|---|---|
| 13 | 0.00320999 | 0.00211388 | 0.00384350 | lego 180→270 ×1.61; NEMO ×2.98; lego 270→360 ×2.99; NEMO ×1.97 | could overturn; floor needs ×4.19, last-quarter side growth ×2.99 |
| 49 | 0.000275759 | 0.000171794 | 0.000324894 | NEMO 180→270 ×34.93; lego 270→360 ×4.79 | could overturn; floor needs ×4.80, last-quarter side growth ×4.79 |

## Bottom-reference / shear split

The audit reducer is used verbatim.  `bottom ref` is `u_bottom * H`: the
transport of a depth-uniform column at its deepest wet velocity.  It is not a
true depth mean.  `bottom ref + shear = total` exactly for every entry.

### Row 13

| day | member | total [Sv] | bottom ref [Sv] | shear [Sv] |
|---:|---:|---:|---:|---:|
| 90 | 0 | +0.003578944 | −0.019924039 | +0.023502984 |
| 90 | 1 | +0.003579774 | −0.019916943 | +0.023496717 |
| 90 | 2 | +0.003579797 | −0.019916867 | +0.023496663 |
| 90 | 3 | +0.003545431 | −0.001317560 | +0.004862991 |
| 180 | 0 | −0.006508768 | +0.034536785 | −0.041045553 |
| 180 | 1 | −0.006594467 | +0.007930343 | −0.014524810 |
| 180 | 2 | −0.004771926 | +0.008062152 | −0.012834079 |
| 180 | 3 | −0.006228822 | +0.002797278 | −0.009026100 |
| 270 | 0 | −0.016842556 | −0.053019399 | +0.036176842 |
| 270 | 1 | −0.016988870 | +0.006635165 | −0.023624035 |
| 270 | 2 | −0.018100486 | +0.055264535 | −0.073365021 |
| 270 | 3 | −0.018004799 | +0.025671461 | −0.043676260 |
| 360 | 0 | +0.032230452 | −0.003040975 | +0.035271428 |
| 360 | 1 | +0.033455548 | −0.005195185 | +0.038650732 |
| 360 | 2 | +0.036496092 | −0.012770632 | +0.049266725 |
| 360 | 3 | +0.027243104 | −0.095716512 | +0.122959617 |

Row 13's net is **shear-led in the control member at day 360**, but that is
descriptive, not a component verdict.  Its day-360 component floors are
`0.04009 Sv` (bottom reference) and `0.03911 Sv` (shear), making the control
components only `0.076` and `0.902` of their own floors.  The total is 8.386
floors because the two component dispersions strongly cancel in their sum.
Calling this independently resolved shear physics would repeat the campaign's
reduction-cancellation error in reverse.

### Row 49

| day | member | total [Sv] | bottom ref [Sv] | shear [Sv] |
|---:|---:|---:|---:|---:|
| 90 | 0 | +0.000816190 | +0.001703533 | −0.000887342 |
| 90 | 1 | +0.000816182 | +0.001703778 | −0.000887596 |
| 90 | 2 | +0.000816206 | +0.001703688 | −0.000887481 |
| 90 | 3 | +0.000815667 | +0.001717615 | −0.000901948 |
| 180 | 0 | +0.002466023 | +0.002526414 | −0.000060391 |
| 180 | 1 | +0.002617222 | +0.002691101 | −0.000073879 |
| 180 | 2 | +0.002493821 | +0.002516135 | −0.000022314 |
| 180 | 3 | +0.002369420 | +0.002273853 | +0.000095567 |
| 270 | 0 | +0.004615416 | +0.004110270 | +0.000505146 |
| 270 | 1 | +0.004647341 | +0.004270959 | +0.000376382 |
| 270 | 2 | +0.004682005 | +0.004125969 | +0.000556036 |
| 270 | 3 | +0.004422761 | +0.003676346 | +0.000746415 |
| 360 | 0 | +0.003118879 | +0.003064952 | +0.000053927 |
| 360 | 1 | +0.002809138 | +0.002813628 | −0.000004491 |
| 360 | 2 | +0.003720279 | +0.003680173 | +0.000040106 |
| 360 | 3 | +0.003480727 | +0.003280162 | +0.000200565 |

Row 49 is bottom-reference-led across members at day 360.  The control bottom
reference is `9.22` of its own component floor (`0.0003323 Sv`); shear is
`0.52` of its floor (`0.0001032 Sv`).  This descriptive vertical resemblance
to P3 does not assign ownership, and the total gap remains an unsaturated
upper bound.

## Thickness-weighted depth structure

Every one of the 36 level contributions is stored in the artifact.  The table
below gives the member-0 signed contributions in the audit's fixed depth
classes; percentages are shares of `sum_k |level gap|`, not shares of the net.
The centroid uses those absolute thickness-weighted contributions.

| row | day | upper <200 m [Sv] | 200–1400 m [Sv] | >1400 m [Sv] | gross shares upper/interior/abyss | gross [Sv] | centroid [m] |
|---:|---:|---:|---:|---:|---|---:|---:|
| 13 | 90 | +0.001120 | +0.002041 | +0.000418 | 6.1% / 54.0% / 39.9% | 0.018372 | 1558 |
| 13 | 180 | +0.001823 | −0.000249 | −0.008082 | 3.0% / 25.9% / 71.1% | 0.060468 | 2177 |
| 13 | 270 | +0.002394 | +0.001971 | −0.021208 | 5.1% / 35.4% / 59.5% | 0.046683 | 1727 |
| 13 | 360 | +0.004939 | +0.025014 | +0.002277 | 5.0% / 25.1% / 70.0% | 0.099734 | 2129 |
| 49 | 90 | −0.000520 | +0.000580 | +0.000755 | 82.2% / 8.1% / 9.8% | 0.008246 | 295 |
| 49 | 180 | −0.000159 | +0.001394 | +0.001232 | 69.6% / 16.2% / 14.3% | 0.008623 | 468 |
| 49 | 270 | +0.000207 | +0.002052 | +0.002357 | 54.4% / 21.2% / 24.4% | 0.009666 | 775 |
| 49 | 360 | −0.000281 | +0.001618 | +0.001782 | 73.1% / 12.8% / 14.1% | 0.012648 | 494 |

Row 13's day-360 **gross** vertical structure is abyss-dominated (70%), but
the abyssal signed sum is only +0.00228 Sv: large opposite-signed levels cancel
inside that class.  Its net positive gap is carried mainly by 200–1400 m.
Row 49 has the opposite gross ordering: 73% of its day-360 gross structure is
above 200 m, but the upper signed contribution is negative; its positive net
comes from the interior and abyss.  Neither row is accurately described by a
single depth fraction without both signed and gross columns.

## Zonal structure: blocks versus open water

Rows 13 and 49 have identical U topology: blocked columns `[0, 50, 51]`, one
open segment `[1..49]`, and wet-level counts differing slightly inside that
segment.  The verdict reducer scores columns `2..49`.  **All three blocked
columns are outside the scored window**, as is open column 1.  The only scored
open column immediately adjacent to a block is column 49; columns 2–48 are the
open interior.

| row | day | near-block signed [Sv] | open-interior signed [Sv] | gross share near/interior | zonal gross [Sv] |
|---:|---:|---:|---:|---|---:|
| 13 | 90 | −0.000760 | +0.004339 | 7.0% / 93.0% | 0.010918 |
| 13 | 180 | +0.000881 | −0.007390 | 5.1% / 94.9% | 0.017215 |
| 13 | 270 | +0.000281 | −0.017123 | 1.3% / 98.7% | 0.022010 |
| 13 | 360 | +0.001951 | +0.030280 | 3.4% / 96.6% | 0.057465 |
| 49 | 90 | −0.000074 | +0.000890 | 2.5% / 97.5% | 0.002926 |
| 49 | 180 | −0.000063 | +0.002529 | 2.2% / 97.8% | 0.002912 |
| 49 | 270 | −0.000039 | +0.004654 | 0.8% / 99.2% | 0.004693 |
| 49 | 360 | −0.000095 | +0.003214 | 2.8% / 97.2% | 0.003445 |

The gaps are broad over the scored open segment.  At day 360 the single
near-block column carries only 3.4% (row 13) and 2.8% (row 49) of gross zonal
structure.  The blocked columns contribute exactly zero and are not part of
the reducer.  The topological break distinguishes these rows from the channel,
but the measured gap is not localized at that break.

## Corrected leave-one-horizon-out relationship scores

For each scored horizon, the regional template is the mean profile over the
other three horizons and all four members.  The held-out profile is projected
onto that template after normalizing the template to unit spatial RMS.  Thus a
horizon never helps define the template used to score itself, and projections
from regions with different row counts have the common unit `Sv per row`.
Each fold's raw squared norm, RMS and final divisor are in the artifact.

The table uses the control member only.  Exact p-values are unadjusted; there
are four reference comparisons for each of two target rows.

| target | basin profile | channel profile | basin neighbour | channel neighbour | profile separation | corrected verdict |
|---:|---:|---:|---:|---:|---:|---|
| 13 | +0.98 (`p=.02`) | −0.87 (`p=.13`) | +0.86, row 12 (`p=.14`) | +0.99, row 14 (`p=.01`) | +0.11 | **UNRESOLVED_LOW_N** |
| 49 | +0.91 (`p=.09`) | +0.64 (`p=.36`) | +0.89, row 50 (`p=.11`) | +0.93, row 48 (`p=.07`) | +0.28 | **UNRESOLVED_LOW_N** |

Separations are deliberately shown to two decimal places: with four horizons,
additional digits imply resolution the statistic does not have.  Row 13 fails
the `0.20` descriptive separation and the basin route is also vetoed by the
stronger channel-side neighbour.  Row 49 clears the descriptive profile
separation but is vetoed because its closest immediate neighbour remains on
the channel side.  Neither row supports an ownership statement.

The nudge traces do not change that disposition.  Row-13 signed P1 sensitivity
correlations are `[0.962, 0.968, 0.947]`, while channel values are
`[-0.906, -0.875, -0.793]`.  Row-49 P3 sensitivities are
`[0.734, 0.903, 0.953]`, while channel values are `[0.655, 0.465, 0.604]`.
Their close agreement describes perturbation sensitivity only; it is not four
independent tests.

## Review fixes B1--B8

1. **B1:** retracted the day-360-template results and replaced them with
   horizon-held-out templates.
2. **B2:** recorded exact `n=4` null p-values beside every correlation and bar;
   margins are rounded to two decimals and no longer treated as inferential.
3. **B3:** member 0 is primary; the nudge twins are sensitivity traces with
   effective independent replication approximately one.
4. **B4:** the power plant now generates row trajectories and traverses the
   actual held-out template, normalization, projection, Pearson, veto and
   routing path.  A held-out-day poison separately proves non-leakage.
5. **B5:** projections use unit-RMS templates and report raw and normalized
   fold divisors.
6. **B6:** an opposite-neighbour veto prevents either regional arm from
   winning when the closer immediate match lies across the boundary.
7. **B7:** per-horizon amplitude cells compare the two rows directly, without
   requiring row 49 to be inside its locally scaled floor.
8. **B8:** the complete 489 kB JSON artifact is committed beside this report.

Measured pushback on B5: the review correctly measured a roughly `222x`
difference between the original row-13 raw template squared norms.  However,
those were fixed positive scale factors across the original four-horizon
trajectory, and Pearson correlation is exactly invariant to such scaling.
Recomputing all original arms after restoring their denominators changed `r`
by at most `2.22e-16`.  Thus the denominator ratio did not make the original
correlation margins incommensurable; circular reuse and `n=4` were the actual
decisive faults.  Normalization is nevertheless necessary in the corrected
cross-fit because its template scale varies by held-out horizon.

## Provenance and controls

The artifact read and records all available input stamps:

* all four legoESM members: `.launch_sha = a7b940f75c04d824b478de4e1728220e3a71989e`,
  materialized state dtype `float64`, `nemo_ladder_mode = both`, and
  `seasonal_t0_seconds = 15552000` (180 days);
* the older verdict artifacts do **not** carry `twin_start_mode`,
  `vertical_ladder_sha256`, or `producer_git_sha`; those keys are recorded as
  null, not inferred.  Consequently this result makes no new start-mode or
  full-grid-identity claim;
* the audit's clock compatibility path recovered the oracle restart clock as
  exactly 15,552,000 s and ran the twin's own phase guard before allowing the
  legacy-stamp loads;
* NEMO member directories are recorded exactly, including the named relocated
  member 3 at `/tmp/dino_v360_m3`;
* `legoesm.constants`, `legoesm.ocean`, both reducers and the probe resolved
  inside `/tmp/codex-row13-local`; Python was
  `/home/dbalwada/legoESM/.venv/bin/python` under the required explicit
  `PYTHONPATH`;
* regional-audit SHA-256
  `2f2bbc03afed792ea0148002029a6ad6481c93bd2ee37679b4bf615aab392fe7`
  and channel-rescore SHA-256
  `a3b51de6292df22b90fa9bd385dcb22ab47a3049c737f2e2abf1a28eb5a18be9`
  were checked before opening a state.

Mechanical gates: the upstream audit self-test passed; the channel rescore's
6/6 planted violations fired; the row probe's reducer-hash, depth, zonal,
component, thickness, blocked-column, local-floor, end-to-end synthetic
attribution, held-out-day poison, opposite-neighbour veto and amplitude-aware
symmetry plants all fired; direct pytest reports **14 passed**; ruff is clean;
producer SHA and clean state were identical at the start and end of the run.

## Honest campaign disposition

1. **Row 13 is unresolved at low n.**  Its +0.0322 Sv saved-state magnitude is
   broadly distributed across open longitudes and vertically
   cancellation-rich, but neither P1 nor channel ownership survives the
   cross-fit, low-n qualification, and neighbour veto together.
2. **Row 49 is not attributable to P3.**  It has P3-profile and vertical
   resemblance, but row 48 is the closer immediate trajectory match.  Its
   +0.00312 Sv magnitude remains an unsaturated upper bound.
3. **The equal topology has unequal saved-state amplitudes.**  Row 49 is
   one-tenth row 13 at day 360, so the amplitude-aware cell records row-13
   dominance.  Weak four-point trajectory correlation and unsaturated floors
   prevent promotion to a physical symmetry or asymmetry claim.
4. No physics change follows from this state-only measurement.  The row-13
   and row-49 ownership questions both remain open.
