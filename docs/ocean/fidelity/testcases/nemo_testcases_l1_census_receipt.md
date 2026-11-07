# OVERFLOW-zps `final_water_mass_census`: anatomy, and the k=24 ownership claim REFUTED

Preregistration: `nemo_testcases_l1_census_preregister.md` (`51c33699e`,
frozen before any arm; probe committed first, at `7e2ff4e91` and `fec9c5929`).
Base tree `fd4a798c7` (the isomorphism tip carrying every OVERFLOW fix to date,
including the 3-D umask).  fp64 (`PrecisionPolicy.fp64()` set BEFORE the card
is built, `JAX_ENABLE_X64=1`), CPU.  **NO MODEL CODE CHANGED IN THIS ROUND**
(`git diff fd4a798c7..HEAD -- packages/ src/` is empty), so every trajectory
row, every gate and all six statistics are unchanged from the phantom round.

## Verdict

*(Rewritten after dual adversarial review.  The first draft's verdict claimed
the census row IS a mixing statistic and that the `k=24` family was REFUTED as
its owner.  Both reviewers attacked it; five of their points were checked
against this round's own JSON and CONFIRMED, so the verdict below is weaker
than the draft's.  The retractions are in section 6.)*

**What is established.**  The census row is a temperature-classification
difference in the descending plume on the slope -- not thickness (the
same-class volume channel is 280x below the row), not heat content (the domain
mean agrees to `4.1e-13 K` and the cold anomaly's heat deficit to `1.4e-12`
relative), not bin-edge chatter (the reclassified volume carries a median
`0.78 K` difference at a `0.21 K` edge gap).  It sits at the deep slope at the
seabed at 17 h and mid-slope at 8.5 h, peaking in the SAME cells as NEMO's own
FCT4-vs-FCT2 difference.  Separately, legoESM's cold anomaly is diluted into
**10.1% less volume** than NEMO-FCT2's at the scored time, where NEMO's own
tracer-order change moves that by 0.5%.

**What is NOT established, and was claimed in the first draft.**  That the
census row is a readout of mixing.  It is not monotone in dilution: NEMO-FCT4
holds `0.00336` LESS mixed water than FCT2 while being MORE diluted
(effective anomaly volume `+0.50%`), and legoESM holds `0.00944` less while
being LESS diluted (`-10.1%`).  **The row demonstrably moves `0.0034` through a
channel that is not bulk dilution**, so mixing can at most be a term on top of
that.  RETRACTED.

**The `k=24` family is REFUTED as the owner of this row -- by direct
measurement (section 5b), not by the two frozen bounds.**  Review showed both
of those bounds were weaker than the preregistration assumed, so a second
preregistration was frozen and run: a full-duration run seeded at the `k=24`
face itself, at `1e-9 m/s` (the size of the measured kt=3 injection), moves the
census by `2.9e-14` against a `9.4e-3` gap -- eleven orders short.

That measurement also corrects a premise underneath both bounds: **the OVERFLOW
solution is not chaotic on this window.**  A thousandfold larger seed gives only
a `22x` larger response.  So no observed statistic can be manufactured by
trajectory divergence, and the census separation, the FCT4-vs-FCT2 separation
and the fp32 floor are all SYSTEMATIC differences.

Open item 3 is CLOSED at the injection faces and REOPENED elsewhere, where it
is a newly measured, scale-compatible candidate (section 4).  No model code
changed in this round.

## 1. Census anatomy -- what, where, when (`census_map/census_map.json`)

INSTRUMENT CONTROL, run before any map was written: the probe reproduces the
committed scorer's three numbers for this row to the last digit --
candidate `0.009437984095819751`, floor `0.0005206303447481894`, spread
`0.003362090947063974`.

**WHAT.**  The `cold [10,12)` class is **EXACTLY 0.0 in all four arms ON THE
SLOPE** at the scored time, so the row is a two-class split and its max-abs
reduction is just the mixed fraction.  CORRECTED after review: this is a
statement about the slope region only.  In the DOMAIN, `N4` still holds
`2.00e7 m3` below 12 C and its coldest cell is `11.892 C`; `N2`, `L64` and
`L32` hold none, with coldest cells `13.621`, `13.220` and `13.234 C`.  The
cold bin therefore sits at its floor in every arm while the calibration arm is
`0.11 K` from populating it -- the row reads a two-sided distribution
difference one-sidedly, which is a limitation of the statistic.

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
  temperature is identical to `4.1e-13 K` and the cold anomaly's heat deficit
  `int w (20-T)` to `1.4e-12` relative: heat is conserved to roundoff in both
  models, and there is no cold water left to be deficient in.
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

CORRECTED after review, twice.  (a) `1.05e-9` is the residual on the variance
RATIO; the DESTRUCTION agrees to **`8.43e-8`** relative (NEMO
`3.516595756e-02`, legoESM `3.516595459e-02`), and the destruction is
`1.2315e-2` OF THE INITIAL VARIANCE, not `3.52e-2` -- the first draft paired
one normalization with the other.  The operators still agree about five orders
of magnitude better than the effect they produce.  (b) The delta is NOT flat
across the window: it holds near `-1.0e-10` to kt=43, CHANGES SIGN at kt=49,
and then grows `1.62x` per step to `+2.97e-9` at kt=60 -- a factor 119 in the
last ten steps.  So "a per-step replay at kt<=60 would measure nothing" is
FALSE from about kt=50; the signal is emerging exactly where the window ends,
and extending the walk is a real next step (blocked only by NEMO step entries
existing to kt=60 and no further).

## 2b. The DIRECTION, reconciled: legoESM's plume core stays COLDER

POST-HOC, and labelled as such: the anomaly moments were added to the probe
AFTER the census numbers existed, to reconcile a sign the first draft could not
explain.  They are not covered by the preregistration banner at the top.

Higher variance at a common mean is ambiguous on its own, so state it in plume
units.  Write the cold anomaly's first and second moments about the ambient
20 C, `D = int w (20-T)` and `M2 = int w (20-T)^2`.  `D` is the heat deficit and
is conserved by advection; at fixed `D`, `D^2 / M2` is the volume the anomaly
effectively occupies and `M2 / D` its mean amplitude.

| arm | `D` (K m3) | `M2` (K2 m3) | effective volume (1e9 m3) | mean amplitude (K) |
|---|---:|---:|---:|---:|
| `N2` | `1.000000000e+11` | `2.026841794e+11` | `49.3378` | `2.026842` |
| `N4` | `1.000000000e+11` | `2.016751076e+11` | `49.5847` | `2.016751` |
| `L64` | `1.000000000e+11` | `2.254889254e+11` | `44.3481` | `2.254889` |

CONTROL: `D` agrees between `L64` and `N2` to `1.4e-12` relative (and `N4` to
`4.6e-13`) -- the anomaly's heat content is identical, so the moments are
comparable.  **legoESM spreads the SAME cold anomaly over 10.1% LESS volume at
11.3% LARGER amplitude.**  It dilutes the plume LESS, and `N4` sits on the same
side of `N2` (0.5%) as `L64` does, 20x nearer.

Where, by temperature band (from the probe's committed 0.05 K domain histogram;
bin-centre quadrature, so read the STRUCTURE here and the exact moments above):

| band (C) | `N2` volume (1e9 m3) | `L64` volume | `d`volume | `d`M2 (1e9) |
|---|---:|---:|---:|---:|
| 12-14 | `0.140` | `0.640` | `+0.500` | `+20.5` |
| 14-16 | `1.759` | `2.649` | `+0.890` | `+22.0` |
| 16-18 | `14.578` | `10.670` | `-3.908` | `-23.1` |
| 18-19 | `21.868` | `19.836` | `-2.032` | `+0.5` |
| 19-19.5 | `17.209` | `21.860` | `+4.651` | `+2.9` |
| 19.5-19.9 | `34.521` | `32.322` | `-2.199` | `-0.1` |
| 19.9-20 | `249.805` | `251.903` | `+2.098` | `+0.0` |

legoESM keeps **4.6x more** water in the coldest band that survives (12-14 C)
and 1.5x more in 14-16 C, while holding 27% less in the diluted 16-18 C shell
and `2.1e9 m3` more at near-ambient.  The census's `mixed_12_18` bin LUMPS the
enlarged cold core with the shrunken diluted shell, and the shell dominates --
which is why a model that dilutes LESS reads as a DEFICIT on that row.  The
apparent contradiction between "higher variance" and "less 12-18 C water" is
this lumping, and it is resolved: both say legoESM's plume is less mixed.

## 3. Scaling -- the two frozen refute conditions, both MET

**(R1) the calibration pair.**  `N4` differs from `N2` in the tracer advection
order ONLY (the scorer's own namelist diff: `nn_fct_h` 2->4, `nn_fct_v` 2->4,
and the experiment name; no momentum-operator difference of any kind):

| pair | `temperature_linf` | `instantaneous_u_linf` | variance ratio | census |
|---|---:|---:|---:|---:|
| `N4` vs `N2` | `0.356744` | `0.672694` | `0.994176` | `0.003362` |
| `L64` vs `N2` | `0.319981` | `0.657229` | `1.131620` | `0.009438` |
| `L32` vs `L64` | `0.053768` | `0.086588` | `1.001302` | `0.000521` |

On L-infinity legoESM is pointwise CLOSER to NEMO than NEMO-FCT4 is, in both
fields, yet its dilution statistic is 20-23x further away.  THREE corrections
from review, all checked against this round's JSON:

* **The "22x" is time-sampled.**  At the registered MIDPOINT the same
  ratio-of-ratios is `3.31x` on variance and `3.00x` on effective anomaly
  volume (`L64/N2` `1.0827` / `0.9286`; `N4/N2` `0.9750` / `1.0238`); at the
  scored time it is `22.60x` and `20.21x`.  `N4-N2`'s own mixing difference is
  non-monotone and passes near a minimum at the scored time, so the headline
  number depends on WHEN it is read.  Both are reported; neither alone.
* **The L-infinity framing inverts under RMS.**  On the slope, the
  root-mean-square temperature difference is `0.4088 K` for `L64-N2` against
  `0.3922 K` for `N4-N2` -- legoESM is MORE decorrelated by that measure, less
  by L-infinity.  "Pointwise closer" is metric-dependent and is not load
  bearing.
* **R1 is structurally weak for a mixing-sensitive row.**  FCT's numerical
  mixing is limiter-dominated, and `nn_fct_h`/`nn_fct_v` change only the
  antidiffusive flux where the limiter does not clip, so `N4`-vs-`N2` is close
  to a null for mixing BY CONSTRUCTION.  It remains a fair decorrelation-only
  control for the SEED question -- the two share every momentum operator -- but
  it is a poor null for the census row itself.

R1 MET as a bound on the seed channel; NOT sufficient to establish that the row
is a mixing statistic.

**(R2) the per-step perturbation bound.**  The `L32` arm injects a relative
perturbation of order `eps(f32) = 1.19e-7` into every wet cell at every one of
6120 steps.  The `k=24` injection is `7.06e-12` (kt=2) and `4.18e-9` (kt=3)
normalized, at one or two of 16900 u points -- `1.7e4x` and `28x` SMALLER per
step, and localized.  The fp32 perturbation moves the census by `0.000521`,
5.5% of the gap, and the variance by `0.13%`, 1% of the gap.  Under any
monotone response to seed amplitude the `k=24` family's chaotic channel is
bounded far below the gap.  R2 MET **as written, and it is not enough**:
fp32 rounding is UNBIASED noise while the `k=24` injection is a systematic,
always-same-sign operator difference, and a systematic per-step forcing can
bias a statistic where random noise cannot.  The preregistration's R2 does not
bound that channel.  Review also noted the fp32 arm never enters the regime the
row lives in -- it saturates at a normalized temperature L-infinity of `0.054`
while every operator pair in the scorer sits at `0.32`-`0.36`.

A caution recorded rather than buried: over kt=3..60 the variance delta grows
at `1.164x` per step and the `k=24` `u` row at `1.170x` per step -- the same
e-folding to 0.5%, which is the signature of ONE growing mode rather than two
independent channels.  PLAUSIBLE only; the variance delta changes sign near
kt=45, which contaminates the fit.

Because R2 does not bound a systematic seed, a second preregistration was
frozen and the direct measurement run: section 5b.

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

**Open item 3: CLOSED at the injection faces, REOPENED everywhere else.**
The phantom round's open item 3 asks whether legoESM's stage depth mean, which
uses LIVE weights (`_replace_stage_mean`: `sum(u*h_u_pre)/H_u_pre`, with
`h_u_pre = min_cell_to_uface(h_k_pre)` at
`ocean_model_latlon_cgrid.py:4432`), differs from NEMO's, which uses REFERENCE
weights (`stprk3_stg.F90:440`
`zub = uu_b(Kaa) - SUM(e3u_0(:)*uu(:,Kaa)) * r1_hu_0`, with
`hu_0 = SUM(e3u_0*umask)` at `domain.F90:145`).

RETRACTED, from the first draft: it claimed the two weightings are identically
the same weights, because under z*/qco every level of a column carries one
`1+r3u`.  **That is true of NEMO's construction and FALSE of legoESM's**, which
takes a MIN over two columns whose free-surface Jacobians differ, so the
per-level argmin can switch sides and the live weights are not a uniform
rescale of the reference ones.  The first draft measured
reference-against-reference and called the item closed.

Reference against reference, over every wet u face -- still exact, and still
worth having, because it certifies the GEOMETRY:

```
max | e3u_0      - legoESM reference h_u |  = 0.0
max | hu_0       - legoESM H_u           |  = 0.0
max | e3u_0/hu_0 - h_u_ref/H_u_ref       |  = 0.0
```

LIVE against reference, the question that was actually open, per step of a free
run (`faces.json`, `live_vs_reference_stage_mean_weights`):

| kt | max `|live - ref|` weight | wet levels differing `> 1e-12` | at injection faces 20/21/22 |
|---:|---:|---:|---:|
| 1 | `0.0` | 0 | `0.0` |
| 2 | `2.323655e-09` | 26 | `6.94e-18` |
| 4 | `1.252540e-06` | 79 | `6.94e-18` |
| 5 | `3.761720e-06` | 105 | `6.94e-18` |
| 6 | `6.080607e-06` | 134 | `0.0` |
| 9 | `8.609751e-06` | 167 | `6.94e-18` |
| 10 | `7.570658e-06` | 173 | `6.94e-18` |

So: at the `k=24` INJECTION faces the difference is at most **one ULP**
(`6.94e-18`) at every step, because gate faces 20/21/22 have EQUAL bathymetry
on both sides and the min-rule therefore picks a single uniform Jacobian --
the item is closed THERE, which is what the `k=24` question needed.  Everywhere
else it is up to `8.6e-6` on 167 wet levels by kt=9, the SAME ORDER as the
`k=24` injections quoted above (`7.06e-12`, `4.18e-9`), on exactly the
staircase faces where the two neighbouring columns differ in depth.  **That is
a newly measured, scale-compatible candidate for the downslope trajectory
rows** and it is now the ranked next arm.

This table was produced independently twice: the diff reviewer built its own
probe and reported `2.323655e-09`, `1.252540e-06` and `3.761720e-06`, every one
of which appears in this walk to the last digit (the two label steps
differently).  Two converging probes, per Rule 1e, is what establishes it.

(Open item 2, the UP3 curvature `umask` at `dynadv_up3.F90:142-143`, is
untouched by this round and stays open.)

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
  `rn_gambbl = 20`, `trabbl.F90:415-435`) which is a slope-specific tracer
  transport, and the `ln_zad_Aimp` partition of `w`.
* Aimp status, PLAUSIBLE not confirmed: prior rounds measured it INERT at kt=1
  (NEMO's stage-3 transport dump gives `max Cu_v = 1.66e-3` against
  `Cu_min_v = 0.8`, and legoESM returns `max wi = 0.0` exactly), i.e. every
  verification of that package so far was multiplied by zero.  NEMO's 17-hour
  MEAN `w` gives `max Cu_v = 0.0719` over the wet domain, 11x below the
  threshold -- a WEAK bound, because a time mean underestimates the
  instantaneous peak.  Whether Aimp ever activates during the descent is
  UNMEASURED and is the ranked next step.

## 5b. The chaos null at the `k=24` site (second preregistration)

Two full-duration fp64 runs, each differing from the certified run by ONE
number: the initial `u` at the `k=24` injection face itself (section row `j=1`,
model u-face 21, level 24), raised by `1e-12` and by `1e-9 m/s`.  Controls
before each run: the seeded face is wet at that level, EXACTLY one face differs
from the certified initial state, and the amplitude landed.  Scored against the
COMMITTED unperturbed fp64 states with this round's reductions plus the
scorer's own normalized L-infinity, `N2` as the common scale.

| quantity | ARM-A `1e-12 m/s` | ARM-B `1e-9 m/s` | for scale |
|---|---:|---:|---|
| census max-abs distance | `1.311e-15` | `2.934e-14` | gap `9.438e-03`, fp32 floor `5.206e-04` |
| `temperature_linf` normalized | `4.406e-09` | `6.279e-08` | candidate `0.320`, floor `0.0538` |
| `u_linf` normalized | `4.139e-09` | `4.907e-08` | candidate `0.657`, floor `0.0866` |
| tracer-variance ratio | `1.000000000000075` | `1.0000000016` | `L64/N2` `1.1316` |
| anomaly effective-volume ratio | `0.99999999999923` | `0.99999999857` | `L64/N2` `0.8989` |
| anomaly deficit, relative | `-4.07e-13` | `-2.57e-13` | conservation control |
| census by time (0 / mid / final) | `0.0` / `2.2e-16` / `1.3e-15` | `0.0` / `1.6e-15` / `2.9e-14` | |

**This settles it.**  A seed at the `k=24` face at `1e-9 m/s` -- the size of
the measured kt=3 injection -- moves the census row by `2.9e-14` after the full
6120 steps.  That is **eleven orders of magnitude** below the `9.4e-3` gap and
**ten** below the fp32 floor.  The `k=24` family is REFUTED as the owner of
this row by direct measurement, not by a bound.

It also settles something larger, and corrects a premise both refute conditions
rested on: **the OVERFLOW solution is NOT chaotic on this window.**  A
thousandfold larger seed produces only a `22x` larger census response
(`2.93e-14 / 1.31e-15`) and a `14x` larger temperature response -- monotone in
amplitude, nowhere near saturation, and tiny in absolute terms.  Trajectory
divergence therefore cannot manufacture ANY of the observed statistics, which
means the `0.0094` census separation, the `0.00336` FCT4-vs-FCT2 separation and
the `0.00052` fp32 floor are all SYSTEMATIC operator or arithmetic differences.
That strengthens the surviving half of the verdict and removes the chaos
reading entirely.

Predictions: **N1 CONFIRMED** (`1.3e-15 << 3.36e-3`), **N2 CONFIRMED**
(`4.4e-9 << 0.32`), **N3 NEITHER ARM OF THE DISJUNCTION** -- the ratio is
`22x` for a `1000x` seed, so the response is sub-linear but far from saturated;
recorded as written rather than reinterpreted, **N4 CONFIRMED** (both ratios
within `1.5e-9` of 1, against the `0.5%` bar), **N5 CONFIRMED** (deficit
conserved to `4e-13`).

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
- `tests/ocean/fidelity/test_nemo_testcase_census_map_probe.py`: 4 tests, each
  shown to FAIL under a synthetic violation (shifting one bin edge fails the
  classification test; swapping the reclassified/same-class split fails both
  channel-separation tests; the fourth PLANTS an inconsistent census so the
  decomposition-closure guard is shown to fire, which the honest fixtures never
  exercise), and passing at HEAD.  DISCLOSED: the tests cover the probe's pure
  helpers; `arm_state` itself is exercised only by the runs, where its
  instrument control against the committed scorer is the guard.
- every artifact in this round was regenerated at a CLEAN tree after review
  found an earlier `census_map.json` stamped with a commit that did not contain
  four of its fields; the stamp now carries a `-dirty+N` suffix when the tree
  is not clean.

## 10. Reviews

Codex CLI and the GLM tool are unavailable on this account, so the DUAL review
ran as two independent reviewer subagents, one on the diff and one on the
mechanism, both on the `fd4a798c7..241652095` range.  **Both returned
REQUEST-CHANGES, and both were right.**

**Reviewer A (diff).**  (1) BLOCKING: the anomaly-moments commit broke this
probe's own tests and I had not re-run them -- CONFIRMED, fixed, and the
fixture now carries the new keys.  (2) IMPORTANT: open item 3 was closed on a
reference-against-reference measurement and the stated z*/qco basis is false
for legoESM's min-rule -- CONFIRMED at the source
(`ocean_model_latlon_cgrid.py:4432`), measured, and section 4 is rewritten;
their independent values reproduce mine to the last digit.  (3) IMPORTANT: the
`1.05e-9` was a mismatched normalization -- CONFIRMED, corrected to `8.43e-8`.
(4) IMPORTANT: `census_map.json` stamped a commit that lacked four of its
fields -- CONFIRMED, the stamp now reports dirtiness and every artifact was
regenerated clean.  (5) IMPORTANT: section 2b is post-hoc -- ADOPTED and
labelled.  (6) MINOR: `arm_state` untested and the closure guard unexercised --
a fourth test now plants a violation; the `arm_state` gap is disclosed.
(7) `trabbl.F90:415-434` ends one line early -- corrected to `:415-435`.  They
also independently re-derived every receipt table from the JSON and verified
all seven NEMO citations.

**Reviewer B (mechanism).**  (1) BLOCKING: the calibration pair moves the
census the same way while moving the mixing measure the opposite way, so "the
row is a mixing statistic" is not established -- CHECKED against this round's
JSON and CONFIRMED; the verdict is rewritten and the claim RETRACTED.
(2) BLOCKING: the `22x` is time-sampling -- CONFIRMED (`3.31x` at the
midpoint); both are now reported.  (3) BLOCKING: "no water below 12 C in the
domain" is false for `N4` -- CONFIRMED (`11.892 C`), corrected.  (4-5)
IMPORTANT: R2 does not bound a systematic forcing, and the kt<=60 window is
over-read because the delta changes sign at kt=49 then grows `1.62x`/step --
both CONFIRMED from `variance.json`, both corrected.  (6) IMPORTANT: R1 is
question-begging for a mixing-sensitive row, and the L-infinity framing inverts
under slope RMS (`0.4088` vs `0.3922`) -- CONFIRMED, disclosed, and R1 is
demoted to a seed-channel bound.  (7) MINOR: promote the uncensored dilution
measure to a scored row -- NOT taken here; adding a registered metric changes
the scorer and is an ASK item, recorded as such.  Their proposed highest-value
measurement -- a single-cell seed run to full duration -- was ADOPTED, frozen
in a second preregistration, and run: section 5b.

Where the two reviewers overlapped (open item 3, and the strength of R1/R2)
they agreed.  Nothing was averaged; every finding above was re-measured against
this round's own artifacts before it was adopted.
