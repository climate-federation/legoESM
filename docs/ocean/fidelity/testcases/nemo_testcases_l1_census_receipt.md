# OVERFLOW-zps `final_water_mass_census`: anatomy, and the k=24 ownership claim REFUTED

Preregistration: `nemo_testcases_l1_census_preregister.md` (`51c33699e`,
frozen before any arm; probe committed first, at `7e2ff4e91` and `fec9c5929`).
Base tree `fd4a798c7` (the isomorphism tip carrying every OVERFLOW fix to date,
including the 3-D umask).  fp64 (`PrecisionPolicy.fp64()` set BEFORE the card
is built, `JAX_ENABLE_X64=1`), CPU.  **NO MODEL CODE CHANGED IN THIS ROUND**
(`git diff fd4a798c7..HEAD -- packages/ src/` is empty), so every trajectory
row, every gate and all six statistics are unchanged from the phantom round.

## Verdict

**The `k=24` family does NOT own the census row: REFUTED, by both frozen
refute conditions.**  The row is not a trajectory-seed effect at all.  It is a
systematic **tracer-transport statistic**: over 17 hours legoESM destroys
**13.16% less** volume-weighted temperature variance than NEMO-FCT2, while two
NEMO runs that differ only in FCT order -- and that are pointwise MORE
decorrelated from each other than legoESM is from NEMO -- differ by 0.58%, and
the fp32 precision arm differs by 0.13%.  The gap is invisible in the first
minute of model time (the per-step variance-destruction rates agree to
**1.05e-9** relative through kt=60, while the destruction itself is 1.2e-2 over
that window) and is already 8.3% by the halfway point.

Two open items are CLOSED at exactly zero by measurement (section 4), and no
arm was run, because none of the enumerated candidates is scale-compatible.

## 1. Census anatomy -- what, where, when (`census_map/census_map.json`)

INSTRUMENT CONTROL, run before any map was written: the probe reproduces the
committed scorer's three numbers for this row to the last digit --
candidate `0.009437984095819751`, floor `0.0005206303447481894`, spread
`0.003362090947063974`.

**WHAT.**  The `cold [10,12)` class is **EXACTLY 0.0 in all four arms** at the
scored time: no water below 12 C survives anywhere in the domain by 61200 s.
So the row is a two-class split, and its max-abs reduction is just the mixed
fraction:

| arm | `cold_10_12` | `mixed_12_18` | `ambient_18_20` | delta vs N2 |
|---|---:|---:|---:|---:|
| `N2` NEMO FCT2 | `0.0` | `0.05986706` | `0.94013294` | -- |
| `N4` NEMO FCT4 | `0.0` | `0.05650497` | `0.94349503` | `-0.00336209` |
| `L64` legoESM  | `0.0` | `0.05042908` | `0.94957092` | `-0.00943798` |
| `L32` legoESM fp32 | `0.0` | `0.05094971` | `0.94905029` | `-0.00891735` |

legoESM holds **15.8% less** intermediate (12-18 C) water on the slope than
NEMO-FCT2; NEMO-FCT4 holds 5.6% less.  **Same sign, legoESM 2.81x further
along the same axis.**

Which of the three possible mechanisms it is, measured:

* NOT a volume / thickness artefact.  The slope volume differs by `1.6e-6`
  relative, and the same-class volume reshuffle contributes `3.4e-5` of
  fraction -- 280x below the row.  The row is 100% temperature classification.
* NOT a heat-content or dense-water bias.  The domain volume-weighted mean
  temperature is identical to `4.1e-13 K`: heat is conserved to roundoff in
  both models, and there is no cold water left to be deficient in.
* NOT bin-edge chatter.  `1.062e10 m3` (3.95% of the slope volume, 532 cells)
  changes class between `L64` and `N2`, with median `|dT| = 0.778 K`, p90
  `2.17 K`, max `6.06 K`, against a median distance to the nearest bin edge of
  `0.212 K`.  The `L32`/`L64` floor pair, for contrast, IS chatter: median
  `|dT| = 0.077 K` at a `0.017 K` edge gap.
* It IS a temperature-distribution difference in the plume: real, several-K
  temperature differences in the descending dense water.

**WHERE.**  The slope band (`500 m < H < 2000 m`) is `x = 23.5 .. 134.5 km`.
The census difference is concentrated at the DEEP end and at the BOTTOM:

| `x` band | `d(mixed fraction)`, `L64 - N2` |
|---|---:|
| 20-40 km | `+0.00025` |
| 40-60 km | `+0.00050` |
| 60-80 km | `+0.00096` |
| 80-100 km | `-0.00275` |
| 100-120 km | `-0.00238` |
| 120-140 km | `-0.00602` |

The peak temperature difference is `6.06 K` at `x = 133.5 km`, depth
`1989.6 m`, column bathymetry `2000.0 m` -- the deepest slope cell, at the
seabed.  `N4 - N2` peaks at `x = 131.5 km`, `1990.0 m`, `5.97 K`: **the same
place, the same size.**  By level, the signal spans k=25..99 and is largest at
k=95..98, the bottom four levels.

**WHEN.**  At the registered midpoint (30590 s) the same map peaks at
`x = 42.5 km`, depth `1230 m`, bathymetry `1400 m` -- mid-slope -- and again
`N4 - N2` peaks at the SAME cell.  The difference travels with the plume:
mid-slope at 8.5 h, deep slope and seabed at 17 h.  The midpoint census
separation is `0.004512` against an `N4-N2` midpoint separation of `0.001193`
(3.78x), so the gap is present early and is not created at the end.  (Only the
final time is scored; the midpoint is reported here, not claimed as a row.)

## 2. The mixing measure -- volume-weighted domain tracer variance

Pure advection preserves tracer variance; every numerical scheme destroys some,
and how much is the scheme's spurious diapycnal mixing.  The domain mean is
identical across arms to `4e-13 K`, so the variances are directly comparable.

| time | `N2` | `N4` | `L64` | `L32` | `L64/N2` | `N4/N2` | `L32/L64` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 s | `2.855649033` | `2.855649033` | `2.855649033` | `2.855648934` | `1.000000` | `1.000000` | `1.000000` |
| 30590 s | `1.140403604` | `1.111873038` | `1.234766852` | `1.234303110` | `1.082745` | `0.974982` | `0.999624` |
| 61200 s | `0.509774191` | `0.506805285` | `0.576870665` | `0.577621471` | **`1.131620`** | `0.994176` | `1.001302` |

And over the first 60 steps, against NEMO's own step entries
(`census_map/variance.json`):

| kt | NEMO | legoESM | ratio |
|---:|---|---|---:|
| 1 | `2.855649032569996` | `2.855649032569996` | `1.000000000000` |
| 10 | `2.854050910702079` | `2.854050910697569` | `0.999999999998` |
| 30 | `2.841780007389422` | `2.841780007282376` | `0.999999999962` |
| 60 | `2.820483075012570` | `2.820483077978081` | `1.000000001051` |

NEMO destroys `3.52e-2 K2` of variance over those 59 steps and legoESM
destroys the same amount to **1.05e-9 relative**.  There is no resolvable
per-step mixing-operator gap in the first ten minutes; the 13% gap opens later.

## 3. Scaling -- the two frozen refute conditions, both MET

**(R1) the calibration pair.**  `N4` differs from `N2` in the tracer advection
order ONLY (the scorer's own namelist diff: `nn_fct_h` 2->4, `nn_fct_v` 2->4,
and the experiment name; no momentum-operator difference of any kind):

| pair | `temperature_linf` | `instantaneous_u_linf` | variance ratio | census |
|---|---:|---:|---:|---:|
| `N4` vs `N2` | `0.356744` | `0.672694` | `0.994176` | `0.003362` |
| `L64` vs `N2` | `0.319981` | `0.657229` | `1.131620` | `0.009438` |
| `L32` vs `L64` | `0.053768` | `0.086588` | `1.001302` | `0.000521` |

legoESM is pointwise CLOSER to NEMO than NEMO-FCT4 is (both fields), yet its
bulk mixing statistic is **22x further away**.  Decorrelation of the observed
magnitude produces a `0.6%` variance difference and a `0.0034` census
difference; the observed gap is `13.2%` and `0.0094`.  R1 MET.

**(R2) the per-step perturbation bound.**  The `L32` arm injects a relative
perturbation of order `eps(f32) = 1.19e-7` into every wet cell at every one of
6120 steps.  The `k=24` injection is `7.06e-12` (kt=2) and `4.18e-9` (kt=3)
normalized, at one or two of 16900 u points -- `1.7e4x` and `28x` SMALLER per
step, and localized.  The fp32 perturbation moves the census by `0.000521`,
5.5% of the gap, and the variance by `0.13%`, 1% of the gap.  Under any
monotone response to seed amplitude the `k=24` family's chaotic channel is
bounded far below the gap.  R2 MET.

CONFIRMED, therefore: the census row is a systematic difference in the tracer
transport's mixing statistics, not a trajectory seed.  The `k=24` family is
REFUTED as its owner.  What remains PLAUSIBLE and unmeasured is *which*
operator: see section 5.

## 4. The k=24 family, enumerated (`census_map/faces.json`)

Section row `j=1`; gate u-face `f` lies between T columns `f` and `f+1`
(model face `f+1`).  CONTROLS, all exact: NEMO's `tmask` equals legoESM's
active T mask cell for cell; NEMO's `umask` equals legoESM's active u mask.

| gate face | x (km) | bottom k, L/R | bathy L/R (m) | `hu_0` (m) | k=24 | k=25 |
|---:|---:|---|---|---:|---|---|
| 20 | 20.5 | 24 / 24 | 500.0 / 500.0 | `500.000` | WET, `e3u_0 = h_u = 20.0000` | dry both |
| 21 | 21.5 | 24 / 24 | 500.0 / 500.0 | `500.000` | WET, `e3u_0 = h_u = 20.0000` | dry both |
| 22 | 22.5 | 24 / 25 | 500.0 / 510.0 | `500.000` | WET, `e3u_0 = h_u = 20.0000` | dry both, RIGHT neighbour wet |
| 23 | 23.5 | 25 / 25 | 510.0 / 513.3 | `510.039` | WET, `20.0000` | WET, `e3u_0 = h_u = 10.0393` (PARTIAL) |
| 24 | 24.5 | 25 / 25 | 513.3 / 517.7 | `513.330` | WET, `20.0000` | WET, `e3u_0 = h_u = 13.3299` (PARTIAL) |

At `k=24` on the injection faces (20, 21, 22) **nothing differs between the two
models and nothing is partial**: `umask = 1` and `lego_u_active = True`; the
face thickness is the full `20.0000 m` in both; both neighbouring T columns are
full cells (`e3t_0 = 20.000 / 20.000`); and the UP3 k-slab stencil's three
neighbours are `[1, 1, 1]` -- every one wet.  That is exactly why the 3-D umask
fix measured `1.00x` on this family: **no masking or thickness rule can act
there.**  The `k=24` injection is an arithmetic difference on identical,
fully-wet, full-thickness operands, and the partial cells do not begin until
`k=25` on gate face 23.

By contrast the already-owned `k=25` family at gate face 23 is wet on a
`10.04 m` partial cell whose SHOREWARD stencil neighbour (gate 22) is dry
(`up3_neighbour_umask = [0, 1, 1]`) -- the phantom's entry point.

**Two open items CLOSED at exactly zero.**  The phantom round's open item 3
asked whether legoESM's stage depth mean, which uses LIVE weights
(`_replace_stage_mean`: `sum(u*h_u_pre)/H_u_pre`), differs from NEMO's, which
uses REFERENCE weights (`stprk3_stg.F90:440`
`zub = uu_b(Kaa) - SUM(e3u_0(:)*uu(:,Kaa)) * r1_hu_0`, with
`hu_0 = SUM(e3u_0*umask)` at `domain.F90:145`).  Measured over EVERY wet u
face:

```
max | e3u_0            - legoESM reference h_u |  = 0.0
max | hu_0             - legoESM H_u           |  = 0.0
max | e3u_0/hu_0       - h_u/H_u               |  = 0.0
```

Under z*/qco every level of a column carries the same `1 + r3u`, so the two
weightings are the SAME weights; the difference is identically zero, not small.
Open item 3 is closed with no arm.  (Open item 2, the UP3 curvature `umask` at
`dynadv_up3.F90:142-143`, is untouched by this round and stays open.)

No new operator replay was built: the stage3-remainder round's candidate table
(X1..X7, replayed on NEMO's own stage operands at kt=2) already ranked this
family's candidates, its winner was landed, and the one candidate it left
UNMEASURED at this site is the weighting above, now measured at exactly zero.
Building a fourth bookkeeping probe against a family whose geometry is
identical in both models would be the anti-pattern the skill names.

## 5. What the oracle leaves as the only remaining suspects

Read from the namelist and from NEMO's own output, not inferred:

* `ln_traldf_OFF = .true.` and `ln_dynldf_OFF = .true.` (`namelist_cfg:78,109`)
  -- NO lateral diffusion of tracers or momentum.
* `ln_drg_OFF = .true.` (`:45`) -- no bottom drag.
* `ln_zdfcst = .true.`, `rn_avm0 = 1.0e-4`, **`rn_avt0 = 0.0`** (`:128-131`);
  `ln_zdfevd = F` (`ocean.output:643`).  NEMO's own 17-hour output confirms it:
  `votkeavt` is EXACTLY `0.0` at all 17000 wet points and `votkeavm` maxes at
  `1.0e-4`.  **There is no vertical tracer-mixing operator in this case at
  all**, so the vertical-mixing suspects named for the slope -- a TKE or EVD
  closure -- do not exist here, and the lateral-diffusion operands do not
  either.
* What remains: the FCT tracer advection (`nn_fct_h=2, nn_fct_v=2,
  nn_fct_imp=1`), the ADVECTIVE BBL (`ln_trabbl = .true.`, `nn_bbl_adv = 2`,
  `rn_gambbl = 20`, `trabbl.F90:415-434`) which is a slope-specific tracer
  transport, and the `ln_zad_Aimp` partition of `w`.
* Aimp status, PLAUSIBLE not confirmed: prior rounds measured it INERT at kt=1
  (NEMO's stage-3 transport dump gives `max Cu_v = 1.66e-3` against
  `Cu_min_v = 0.8`, and legoESM returns `max wi = 0.0` exactly), i.e. every
  verification of that package so far was multiplied by zero.  NEMO's 17-hour
  MEAN `w` gives `max Cu_v = 0.0719` over the wet domain, 11x below the
  threshold -- a WEAK bound, because a time mean underestimates the
  instantaneous peak.  Whether Aimp ever activates during the descent is
  UNMEASURED and is the ranked next step.

## 6. Predictions vs outcomes

| # | prediction | outcome | verdict |
|---|---|---|---|
| R1 | the FCT4/FCT2 calibration pair refutes the seed claim | `N4` is MORE decorrelated from `N2` than `L64` is (`T_linf` `0.3567` vs `0.3200`; `u_linf` `0.6727` vs `0.6572`) and moves the variance `0.58%` and the census `0.00336`, against `13.16%` and `0.00944` | **MET -- claim REFUTED** |
| R2 | the fp32 per-step bound refutes the seed claim | the fp32 arm perturbs every wet cell every step at `1.19e-7` relative, `28x`-`1.7e4x` larger per step than the `k=24` injection, and moves the census `0.000521` (5.5% of the gap) and the variance `0.13%` (1% of the gap).  The two perturbation sizes are both relative to the field scale but are not the same normalization, so this is an order-of-magnitude bound, not an equality | **MET -- claim REFUTED** |
| P1-P6 | the arm's predictions | NOT RUN: both refute conditions fired before any arm | -- |

RETRACTION, first-class (Rule 11).  An earlier reading in this session -- that
the census difference might be a **numerical-mixing gap resolvable per step**
-- was tested and is WRONG in that form: the per-step rates agree to `1.05e-9`
through kt=60.  The gap is real and systematic but does NOT show up as a
constant per-step bias in the early regime; it opens between kt=60 and the
halfway point.  Recorded here rather than dropped, because the natural next
probe (a per-step operator replay at kt<=60) would measure nothing.

## 7. Statistics -- all six rows, before and after

Nothing landed, so before == after, byte for byte
(`phantom_velocity/after/stats/overflow_statistics.json`).  Overall status
`OUTSIDE`; verdict counts `{IND-AT-FLOOR 0, OUTSIDE 1, WITHIN-SPREAD 5}`.

| metric | before | after | fp32 floor | NEMO spread | verdict |
|---|---:|---:|---:|---:|---|
| `final_temperature_histogram_tv` | `0.03300509` | `0.03300509` | `0.00622769` | `0.04423095` | WITHIN-SCHEME-SPREAD |
| `final_water_mass_census` | `0.00943798` | `0.00943798` | `0.00052063` | `0.00336209` | **OUTSIDE** |
| `instantaneous_u_linf` | `0.65722907` | `0.65722907` | `0.08658750` | `0.67269430` | WITHIN-SCHEME-SPREAD |
| `plume_descent_m` | `1.4304351` | `1.4304351` | `0.0036066` | `1499.6233` | WITHIN-SCHEME-SPREAD |
| `plume_front_km` | `2.9624485` | `2.9624485` | `0.1355313` | `121.93071` | WITHIN-SCHEME-SPREAD |
| `temperature_linf` | `0.31998141` | `0.31998141` | `0.05376804` | `0.35674372` | WITHIN-SCHEME-SPREAD |

Trajectory rows kt=1..10 and kt=60 are likewise unchanged from the phantom
round (no model code changed), so no `--compare-to` run was made and no row is
faithful-but-worse in this round.

## 8. Figures

Regenerated with the committed plotter from the newest 6120-step states (the
phantom round's, since nothing landed here) at
`/data/abyssal/dbalwada/nemo-testcases-l1/figures_full_census/`:

| figure | sha256 |
|---|---|
| `overflow_zps_full_temperature_sections.png` | `0e2cb056356d827010cba229f4d3441bcc8451fa52b6376513ac78ac251ec62e` |
| `overflow_zps_full_statistical_metrics.png` | `cfc389f6a0c485f3a6f1e8c71d1bf948de4cec744369bf192033f49359093064` |
| `lock_exchange_zco_full_temperature_sections.png` | `4b480346e9966062efc1bef9ba0ef9ca33110f2b0b28354134b46331108f6926` |
| `lock_exchange_zco_full_statistical_metrics.png` | `b305ea1f2f30e44712c07dbbf3af537468b7ae58a07fc63e6eb7cd1a0bbf6519` |

DISCLOSED: the OVERFLOW panels use the phantom round's fp64/fp32 states and its
scorer JSON; the LOCK panels reuse the older `full_statistical` LOCK states and
report, because no LOCK 61200-step run has been made since -- the phantom round
established LOCK is bit-identical on all 300 trajectory rows through kt=60, but
that is not a statement about 61200 steps.

## 9. Controls run

- the map probe reproduces the committed scorer's candidate, floor and spread
  for this row to the last digit BEFORE writing any map, and refuses otherwise;
- the per-cell contribution decomposition is asserted to sum to the measured
  census delta (`< 1e-12`) for every reported pair, so the by-column and
  by-level maps cannot be a different quantity from the row;
- the temperature classes are computed at each arm's OWN dtype and the volume
  weights in fp64, exactly as the scorer does -- otherwise the fp32 floor row
  is not reproducible (it was not, until this was fixed);
- the faces probe refuses unless NEMO's `tmask` and `umask` equal legoESM's
  active T and u masks cell for cell;
- both slope masks are asserted equal between the two arms of every difference
  map;
- salinity is MEASURED (reported, not asserted) constant to `7e-13` about 35
  in every fp64 arm, so no census class can be moved by a salinity difference;
- `tests/ocean/fidelity/test_nemo_testcase_census_map_probe.py`: 3 tests, each
  shown to FAIL under a synthetic violation (shifting one bin edge fails the
  classification test; swapping the reclassified/same-class split fails both
  channel-separation tests), and passing at HEAD.

## 10. Reviews

Codex CLI and the GLM tool are unavailable on this account, so the DUAL review
runs as two independent reviewer subagents (one on the diff, one on the
mechanism) -- see the session's review record.
