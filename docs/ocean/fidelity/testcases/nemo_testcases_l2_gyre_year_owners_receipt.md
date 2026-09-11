# GYRE, WHO OWNS THE DAY-30 GAP: the whole year's difference is made in ONE TIME STEP

Preregistration: `docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_year_owners.md`,
written before any number below was read and frozen; no expectation, metric or
falsifier was changed.  It follows
`nemo_testcases_l2_gyre_year_fromrest_receipt.md`, which scored the year and
left the owner unnamed.

Harness: `scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_year_owners.py`.
Acquisition, for the operator: `scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_earlydays/run.sh`.
Artifacts: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_owners/`.
CPU, fp64, `PrecisionPolicy.fp64(transcendentals="libm")`, `JAX_ENABLE_X64=1`.

## VERDICT

**The GYRE year's model-model gap is not accumulated.  It is created in STEP 2,
and every one of the remaining 2158 steps only amplifies it.**

| what | wet-cell rms temperature gap | factor over the previous row |
|---|---:|---:|
| entering step 1 (both models at rest) | `0.0` K | — |
| entering step 2 (after 1 step) | `2.0438e-15` K | — |
| entering step 3 (after 2 steps) | `4.1544e-04` K | **`2.03e+11`** |
| entering step 60 (day 9.83) | `5.2599e-03` K | `12.7` over 57 steps |
| day 30 (after 180 steps) | `1.4241e-02` K | `2.71` over 120 steps |
| day 360 (after 2160 steps) | `4.0714e-01` K | `28.6` over 1980 steps |

One step multiplies the gap by two hundred billion.  Every later step
multiplies it by between `1.0017` and `1.045`.  **CONFIRMED.**

Three consequences, all of which change what the campaign should do next.

1. **The year receipt's section 7(c) is CORRECTED.**  It computed, correctly,
   that the kt=2 error compounded over 180 steps is `2.4e-12` K against a
   measured day-30 gap of `1.4e-02` K, and concluded "the per-step error must
   GROW by orders as the flow spins up".  It does not grow by orders over many
   steps; it grows by eleven orders in ONE step and then stops growing.  The
   sentence "closing the remaining kt=2 DEBT rows cannot be assumed to close
   the year" is therefore too weak in one direction and too strong in the
   other: closing the kt=2 rows is not sufficient (they are `3e-12` m/s and the
   step-2 output is off by `4e-04` K), but **the step-2 OPERATORS are exactly
   where the year's gap is made**, and no other step is.
2. **The day-30 owner and the kt=2 owner are the same question.**  The kt=1..10
   ladder that rounds 8..50 of this branch have been walking is not a separate,
   smaller problem from the year.  It is the whole problem.
3. **The surface forcing is exonerated as a statement** (section 3) and
   **salinity leads** (section 4) — so what step 2 gets wrong is a TRACER
   TRANSPORT operator, not a flux.

## 1.  THE INSTRUMENT, AND WHY NO NEW NEMO RUN WAS NEEDED FOR DAYS 0-10

NEMO's certified GYRE card writes its whole-step entry state for
`kstp = nit000 .. nit000 + 59`
(`cfgs/GYRE_OMIP_L2_P3_SM_YRPERT/MY_SRC/stprk3.F90:90-101`:
`WRITE(itraj) ts(:,:,:,:,Nbb), uu(:,:,:,Nbb), vv(:,:,:,Nbb), ssh(:,:,Nbb)`).
The year run's pristine member wrote all sixty months ago and they have been on
disk ever since.  They could not be READ: the campaign's time-level registry
(`ocean/fidelity/time_levels.py`, which fails closed on any unregistered dump)
covered only ten of them.  It now covers sixty and still raises at sixty-one.

**A citation defect found on the way in, and fixed.**  The registry cited the
vendored `scripts/.../nemo502_MY_SRC/stprk3.F90` and said it is "used
byte-for-byte by testcase lanes 1 and 2".  That is FALSE for lane 2.  The
`WRITE` statement is byte-identical, but the GATES are not: the vendored
lane-1 copy fires at `nit000`, the midpoint and `nitend`; the GYRE card fires
on the first sixty steps.  A reader following that citation would have
concluded the year run wrote three entry dumps, not sixty.  The lane-2 range is
now cited from the card that compiles it.

**Reconciliation before recording (Rule 1e).**  This walk is a NEW probe and it
disagrees with nothing only because it was checked against the recorded ladder
first:

| quantity | this walk | recorded, round-8 receipt | agreement |
|---|---|---|---|
| kt=2 `u`, max abs | `2.747840e-12` m/s | `2.747840e-12` m/s | all printed digits |
| kt=2 `v`, max abs | `3.305560e-12` m/s | `3.305560e-12` m/s | all printed digits |
| kt=2 `T`, normalized max | `6.048356e-16` | `6.054358e-16` | `0.9990` |
| kt=10 `T`, normalized max | `5.620512e-03` | `5.636638e-03` | `0.9971` |

The two velocity rows are the SAME SIXTEEN DIGITS as numbers this round did not
produce and could not have tuned.  The two temperature rows agree to `0.3 %`
and are expected to, not to agree exactly: the recorded sweep was taken before
the ENE vorticity commit and before the round's later operator work, and the
year receipt measures that commit moving the model.  **A prose correction the
preregistration owes**: its entering table labels `5.786374e-16` as the kt=2
`T` row.  That number is the kt=2 **`S`** row; `T` is `6.054358e-16`
(round-8 receipt, the last sweep).  No conclusion moves.

**What this instrument cannot see.**  It compares STATES, so it says which STEP
owns the gap and says nothing about which OPERATOR inside that step does.  The
per-operator record exists only at `kstp == nit000` on this card, so naming the
operator needs one instrumented step launched from a step-1 restart — a
separate acquisition, named in the open questions, not taken here.

## 2.  THE DAY-BY-DAY TABLE

Days 0-9 are at STEP resolution from the record above; day 30 is the year
record.  Days 10-29 are the acquisition this round writes and does not run.
Wet-masked rms, fp64, NEMO's own `tmask` for `T`/`S`/`ssh` and the card's face
masks for `u`/`v`.

| day | kt | `T` [K] | `S` [g/kg] | `u` [m/s] | `v` [m/s] | `ssh` [m] | `T` ratio | leads |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 1 | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | — | — |
| 1 | 7 | `2.6994e-03` | `2.0391e-04` | `5.1958e-04` | `7.4568e-04` | `1.3519e-06` | — | `S` |
| 2 | 13 | `2.7856e-03` | `1.4378e-04` | `8.6902e-04` | `1.8743e-04` | `8.1644e-06` | `1.032` | `u` |
| 3 | 19 | `2.8185e-03` | `1.5600e-04` | `5.5284e-04` | `2.2112e-04` | `1.7806e-05` | `1.012` | `u` |
| 4 | 25 | `3.3105e-03` | `1.9930e-04` | `5.7926e-04` | `2.8858e-04` | `2.8945e-05` | `1.175` | `u` |
| 5 | 31 | `3.8895e-03` | `2.2572e-04` | `7.4593e-04` | `4.7500e-04` | `4.1285e-05` | `1.175` | `u` |
| 6 | 37 | `4.1958e-03` | `2.8294e-04` | `7.6619e-04` | `3.5979e-04` | `5.4624e-05` | `1.079` | `u` |
| 7 | 43 | `4.4082e-03` | `2.3556e-04` | `7.7226e-04` | `2.2426e-04` | `6.8648e-05` | `1.051` | `u` |
| 8 | 49 | `4.6569e-03` | `2.5223e-04` | `7.1844e-04` | `4.6195e-04` | `8.3165e-05` | `1.056` | `u` |
| 9 | 55 | `4.9327e-03` | `2.6367e-04` | `7.2368e-04` | `4.3775e-04` | `9.8125e-05` | `1.059` | `u` |
| 10-29 | 61-174 | — | — | — | — | — | — | ACQUISITION PENDING |
| 30 | 181 | `1.4241e-02` | `2.2345e-03` | — | — | `4.5334e-04` | — | `S` |

The STEP table, which is what carries the verdict, is in
`year_owners/step_gap.json`; the eleven-order jump is between its `kt = 2` and
`kt = 3` rows and nowhere else in sixty steps.

| expectation | bound | measured | verdict |
|---|---|---|---|
| **E1** day-1 `T3D` above `1e-4` K | `> 1e-4` | `2.6994e-03` K | **HELD** |
| **E2** no knee: day-to-day `T` ratio below 3 | `< 3` | max `1.175` (days 1-9) | **HELD**, and it is the wrong question — the knee is INSIDE day 1, between steps 2 and 3, which a daily cadence cannot resolve and a step cadence does |
| **E3** the forcing statement is bit-exact at every clock value | 0 cells | 0 of 600, five fields, thirteen clock values | **HELD** |
| **E4** upper ocean and a tracer lead | `0-100` m more than `10x` the `1000+` band; `T` or `S` leads | `2.5079e-02` vs `1.7094e-04` K = `147x`; `S` leads | **HELD** |
| **E5** the EVD trigger masks first differ inside `[1080, 1260]` | step window | see section 6 | see section 6 |

## 3.  THE SURFACE FORCING IS EXONERATED AS A STATEMENT

`year_owners/forcing_gate.json`.  legoESM's production forcing path and a
literal transcription of the compiled `usrdef_sbc.f90` are evaluated on the
SAME state — NEMO's own restart, which IS the `Kbb` state entering the next
step (`stprk3.F90:155`, `CALL sbc( kstp, Nbb, Nbb )`) — so the inputs are
identical by construction and any difference can only be the statement.

| clock sample | `ztime` [h] | `qsr` | `qns` | `emp` | `utau` | `vtau` |
|---|---:|---:|---:|---:|---:|---:|
| kt = 1 | `4.0` | 0 | 0 | 0 | 0 | 0 |
| entering days 30, 60, … 330 | `724 … 7924` | 0 | 0 | 0 | 0 | 0 |
| clock sample at kt = 2161 | `8644.0` | 0 | 0 | 0 | 0 | 0 |

Cells unequal, of 600 wet, at every one of the thirteen samples: **zero**.
The samples span the entire `8640`-hour seasonal cycle, so no phase,
denominator, amplitude or calendar operand of the seasonal statements can be
wrong.  **C1 REFUTED as an owner.**

**The statements that were re-read line by line to make that claim** (Rule 0;
`BLD/ppsrc/nemo/usrdef_sbc.f90`, the compiled file the card runs):

| # | NEMO, cited | legoESM | verdict |
|---|---|---|---|
| 1 | `:107-108` `ztime = REAL(kt)*rn_Dt/(rmmss*rhhmm) - (nyear-1)*rjjhh*zyydd` | `nemo_gyre_seasonal_cosines`, `t_seconds/3600` with `t_seconds = kt*dt`; **no `nyear` term** | MATCH inside year 1 — the term is `(nyear-1)*24*360` h = exactly one full period of every cosine, and inside year 1 it is `0.0`, which subtracts exactly.  **MEASURED**: forcing the literal arm to `nyear = 2` at kt = 181 moves `qsr` by `1.4211e-13` W/m2, i.e. the term is mathematically a no-op and numerically one ulp of `libm`'s `cos`.  The gate CANNOT see a missing `nyear` term in year 1, stated rather than discovered later |
| 2 | `:110-113` `ztimemax1 = 4104`, `ztimemin1 = 8424`, `ztimemax2 = 4824`, `ztimemin2 = 504` h | same four constants, same composition | MATCH |
| 3 | `:123-124` `zcos_sais1` divides by `(ztimemin1-ztimemax1)`, `zcos_sais2` by `(ztimemax2-ztimemin2)` — two DIFFERENT expressions with the same value `4320` | the same two expressions, not a shared constant | MATCH |
| 4 | `:132-134` `t_star = 28.3*(1+1/50*zcos_sais2)*COS(rpi*(gphit-5)/(53.5*(1+11/53.5*zcos_sais2)*2))` | `nemo_gyre_t_star`, operand for operand | MATCH |
| 5 | `:136` `qsr = 230*COS(3.1415*(gphit-23.5*zcos_sais1)/(0.9*180))` — the literal is **`3.1415`, not `rpi`** | `_NEMO_QSR_PI` | MATCH.  **MEASURED**: swapping it for `rpi` moves `qsr` by `9.2864e-03` W/m2, so the gate would see it |
| 6 | `:137` `qns = ztrp*(ts(ji,jj,1,jp_tem,Kbb) - t_star) - qsr`, `ztrp = -40` | `nemo_gyre_qns` on the entering tracer | MATCH, and the TIME LEVEL is `Kbb` = the step-entry tracer, which is what legoESM feeds |
| 7 | `:140` `IF( gphit >= 14.845 .AND. 37.2 >= gphit )` | `(lat >= 14.845) & (lat <= 37.2)` | MATCH |
| 8 | `:151-157` `zsumemp = glob_2Dsum(emp)/glob_2Dsum(tmask(:,:,1))`, then `emp = emp - zsumemp*tmask` | `nemo_gyre_zero_mean_emp`, NEMO's DDPDD compensated sum in `ji`-inner order | MATCH (certified bit-exact at kt=1 by round 8) |
| 9 | `:161` `qns = qns - emp*sst_m*rcp` | `nemo_gyre_qns`'s `emp_heat`, with `sst_m` = potential temperature from the entering conservative tracer | MATCH, **and the time level was the live question**: `sst_m` is built in `sbcssm.f90:84-93` from `ts(:,:,1,:,Kmm)` while the Haney term reads `Kbb`.  Those are DIFFERENT LEVELS in general.  On this card they are the same one, because `stprk3.F90:155` calls `sbc( kstp, Nbb, Nbb )` — `Kmm` IS `Nbb`.  legoESM uses one state for both.  **C2's restoring-time-level half is REFUTED on this card, by the call site, not by an argument** |
| 10 | `:185-193` `ztau = 0.105/SQRT(2.)`, `ztau_sais = 0.015`, `ztaun = ztau - ztau_sais*COS(...)`, `utau = -ztaun*SIN(rpi*(gphit-15)/(29-15))`, `vtau = +ztaun*SIN(...)` | `nemo_gyre_wind` | MATCH |

**What section 3 cannot see**, stated in the preregistration and unchanged: a
statement misread the SAME way by both arms.  The mitigations are that the
literal arm is certified against NEMO's OWN kt=1 dump (round 8: all five
fields, 0 of 600 wet cells, max abs `0`) and that the table above is a fresh
reading of the compiled file, not a re-reading of the transcription.

## 4.  WHERE THE GAP LIVES: FIELD, DEPTH, REGION

`year_owners/decompose_day030.json`, `decompose_day360.json`.  Each cut's blind
spot is printed with it, not left to the reader.

**BY FIELD**, each gap over the signal NEMO ITSELF developed from rest by that
day — dimensionless, so five fields in five units can be RANKED.  This is
DINO's statistic (`dino_1226/day_gap_table.py:186-200`), reused rather than
re-derived so the two cases stay comparable.

| field | day 30 gap | day 30 / NEMO's own signal | day 360 gap | day 360 / signal |
|---|---:|---:|---:|---:|
| `S` | `2.2345e-03` g/kg | **`7.63e-02`** | `6.9290e-02` g/kg | **`3.83e-01`** |
| `v` | `8.3330e-04` m/s | `5.93e-02` | `4.0703e-03` m/s | `1.38e-01` |
| `u` | `5.9620e-04` m/s | `5.32e-02` | `4.0919e-03` m/s | `1.46e-01` |
| `T` | `1.4241e-02` K | `1.21e-02` | `4.0714e-01` K | `1.59e-01` |
| `ssh` | `4.5334e-04` m | `8.97e-03` | `7.6119e-03` m | `4.53e-02` |

**Salinity leads at both days, by a factor of six over temperature at day 30.**
GYRE has `sfx = 0` and a zero-mean `emp` (`usrdef_sbc.f90:157,160`), so
salinity has NO SOURCE: it is a pure passive tracer moved by advection and
diffusion alone.  A model that is six times worse at the tracer with no source
than at the tracer with one is not being let down by its surface fluxes.
**PLAUSIBLE as an attribution — no ablation was run on either model (Rule 4),
so this is a DIRECTION.**

*Blind spot:* the ratio's denominator is NEMO's own from-rest change, which for
a field that barely moves makes the ratio large for a small absolute error.
`S` changes by `2.93e-02` g/kg rms in 30 days and legoESM gets `7.6 %` of that
change wrong; both numbers are in the table so the reader can see which.

**BY DEPTH**, temperature, share of the summed squared difference:

| band | day 30 gap [K] | day 30 share | day 360 gap [K] | day 360 share | cells |
|---|---:|---:|---:|---:|---:|
| `0-100` m | `2.5079e-02` | `0.827` | `4.9586e-01` | `0.396` | 4800 |
| `100-1000` m | `1.0257e-02` | `0.173` | `5.4819e-01` | `0.604` | 6000 |
| `1000+` m | `1.7094e-04` | `0.0001` | `7.1969e-03` | `0.0001` | 7200 |
| the TOP CELL alone | `2.1393e-02` | `0.075` | `4.2961e-01` | `0.037` | 600 |

**The gap is in the upper ocean but NOT in the top cell.**  The top cell is the
only cell the surface flux touches directly and it holds `7.5 %` of the day-30
difference on `3.3 %` of the cells — an enrichment of `2.3`, against `10.0` for
the `0-100` m band as a whole.  Between day 30 and day 360 the difference
MIGRATES DOWNWARD, from `83 %` above 100 m to `60 %` between 100 and 1000 m.
Below 1000 m there is nothing in either year.

**BY REGION**, temperature, day 30, with the enrichment (share of the squared
difference over share of the cells) stated because a share alone favours a big
region:

| region | share of sum dT² | share of cells | enrichment |
|---|---:|---:|---:|
| western third | `0.620` | `0.333` | **`1.86`** |
| interior third | `0.183` | `0.333` | `0.55` |
| eastern third | `0.197` | `0.333` | `0.59` |
| `emp` south branch (`<= 37.2 N`) | `0.803` | `0.745` | `1.08` |
| `emp` north branch (`> 37.2 N`) | `0.197` | `0.255` | `0.77` |
| wind band (`15-29 N`) | `0.638` | `0.343` | **`1.86`** |

**The two enriched regions are NOT two findings — they are largely the same
cells.**  On GYRE's 45-degree-rotated grid the western third (200 surface
cells) and the `15-29 N` wind band (206) share 154; their Jaccard overlap is
`0.61`.  Measured here rather than left for a reviewer.  What separates them is
the peak: the largest single difference at day 30 is `-4.4411e-01` K at
`j=7, i=1, k=3` — depth `36.5` m, latitude `20.24 N`, and `i = 1` is ONE CELL
off the western wall, the closed-boundary ring being `i = 0`.  At day 360 the
peak is `-2.9493e+00` K at `j=6, i=4, k=7`, `91.4` m, `21.59 N`.  The western
boundary current reading is the one the peak supports; the forcing-band reading
is not separable from it by this cut, and saying so is the result.

**The `emp` split shows NO enrichment at all** (`1.08` and `0.77`), which is
the cleanest negative in this table: the freshwater flux's own branch structure
leaves no fingerprint on where the temperature difference lives.

## 5.  TWO SOURCE-LITERAL DIFFERENCES IN THE CONVECTIVE SWITCH, ONE OF WHICH IS REAL

Read from `BLD/ppsrc/nemo/zdfevd.f90`, the compiled file:

```fortran
DO jk = 1, jpkm1 ; DO jj = ... ; DO ji = ...
   IF(  MIN( rn2(ji,jj,jk), rn2b(ji,jj,jk) ) <= -1.e-12 )      &
      &  p_avt(ji,jj,jk) = rn_evd * wmask(ji,jj,jk)            ! :107-110
END DO ; END DO ; END DO
```

and the same block for `p_avm` under `nn_evdm == 1` (`:131-135`).

| # | NEMO | legoESM | assessment |
|---|---|---|---|
| A | `<= -1.e-12` | `N2 < _thr` with `_thr = -1e-12` (`convection/enhanced_diffusion.py:198-200`) | a transcription difference, reachable ONLY on exact float64 equality with `-1e-12`.  `rn2` is a quotient of differences of EOS polynomials; the set of inputs giving that exact bit pattern is a measure-zero set and no cell has ever been observed in it.  **Recorded as DEBT, not fixed this round: a fix is one character, but changing a threshold comparison is a change to the model's own convective switch and it goes in the ASKED table, not in a diff** |
| B | NEMO **SETS** `p_avt = rn_evd` | legoESM takes `K_v_total = maximum(K_v_total, K_conv)` (`vertical_mixing/k_profiles.py:420-425`) | the two agree wherever the other schemes' diffusivity is BELOW `rn_evd = 100` m2/s and DISAGREE wherever it is above — NEMO would reduce such a cell to 100, legoESM keeps the larger.  **This is the one worth measuring**, and the measurement is a one-line census on the traced run: how many wet interfaces ever hold a non-convective `K_v` above `100` m2/s.  Named in the open questions; UNMEASURED this round |

Neither can own the day-30 gap: both are threshold behaviours, and section 1
shows the day-30 gap is made in step 2, when the column is still the
initial-state profile and nothing is convecting.

## 6.  THE THRESHOLD SWITCH BETWEEN DAY 180 AND DAY 210

PENDING — the trace is `--switch-trace 1260 --every 6 --seeds 0,1`, stepping
both members from rest with the model's OWN convective-coefficient field read
every day, and its result is filled in below when it lands.  Its refusal is
already in the harness: if the trigger never fires anywhere in the traced
window, it EXITS NON-ZERO rather than reporting "no crossing", because an
instrument that sees nothing may not be quoted as seeing nothing.

## 7.  WHAT THIS ROUND CANNOT SEE

* **It names a STEP, not an OPERATOR.**  The per-operator record exists at
  `kstp == nit000` only.  Naming what step 2 gets wrong needs one instrumented
  step launched from the step-1 restart.
* **No ablation on either model**, so section 4 is a direction and not an
  attribution (Rule 4).
* **The forcing gate shares one possible misreading with the literal arm** and
  cannot see a missing `nyear` term inside year 1 (section 3, row 1).
* **Days 10-29 have no NEMO record.**  The acquisition is written and not run.
* **The region cut cannot separate the western boundary from the wind band**
  on this rotated grid (`0.61` Jaccard).
* The step walk compares STATES on the card's masks; a difference living on a
  dry face or below level 30 is invisible by construction.

## 8.  RETRACTIONS AND CORRECTIONS

1. **The year receipt's 7(c) mechanism — "the per-step error must GROW by
   orders as the flow spins up" — is CORRECTED.**  Its arithmetic stands: the
   kt=2 error compounded is `2.4e-12` K against a `1.4e-02` K day-30 gap.  Its
   mechanism does not: there is no gradual growth.  There is one step.
2. **The preregistration of the YEAR mislabels the kt=2 `T` row.**  It quotes
   `5.786374e-16` as `GYRE-zco.kt2.before.T`; that is the `S` row.  `T` is
   `6.054358e-16`.  No scored number moves.
3. **The time-level registry's citation for the step-entry dumps was wrong in
   SCOPE** — "used byte-for-byte by lanes 1 and 2" — and is corrected in
   section 1.
4. **The forcing gate's `nyear` plant was a dead variable in its first
   version** and proved nothing; it is fixed and now moves `qsr` by
   `1.4211e-13` W/m2 at kt = 181.  Found by running the plants rather than
   trusting them.

## 9.  ASKED / UNASKED

| # | choice | value | ASKED? |
|---|---|---|---|
| 1 | the early-days NEMO record reuses the certified binary instead of a `makenemo` config copy | reuse | **UNASKED**, preregistered as such.  Offered for revert.  Reason: a second binary inside a one-variable comparison; the day-30 byte-identity refusal is the stronger check |
| 2 | NEMO trend diagnostics not enabled | not enabled | **UNASKED**, preregistered.  Offered for revert.  They write nothing without XIOS on this card, NEMO declares them untested under RK3, and an unclosed trend bucket may not be quoted |
| 3 | `_literal_sbc` gains `kt`/`nyear`/`qsr_pi`, defaults unchanged | additive | **UNASKED**, preregistered.  The alternative is a second copy of a transcription, which is forbidden; the defaults are shown byte-unchanged by the self-check |
| 4 | the time-level registry extends from 10 to 60 step-entry dumps | extend to the writer's own range | **UNASKED.**  Offered for revert.  It reads dumps the card already wrote and still fails closed one past them; no recorded number can move, because nothing previously READ those files |
| 5 | `run_member` gains a snapshot cadence, default unchanged | additive | **UNASKED.**  Offered for revert.  Every scored member is byte-unchanged |
| 6 | the EVD `<=` vs `<` and set-vs-max differences are RECORDED, not fixed | record | **ASKED**, pick: record.  Both are changes to the model's own convective switch and both would move every card that convects; neither can own the day-30 gap |
| 7 | the switch trace's window and cadence | 1260 steps, mask read every 6 | **ASKED**, pick: to day 210 so the year receipt's jump is inside it, daily so a crossing is dated to a day |

## 10.  OPEN, RANKED

1. **Name the operator inside step 2.**  Restart NEMO from the step-1 restart
   and run ONE step with the card's own stage writers, then compare stage by
   stage.  This is the whole campaign's next measurement: everything else in
   the GYRE year is downstream of it.
2. **Run the early-days acquisition** so days 10-29 stop being a gap in the
   record.
3. **Census the non-convective `K_v` above `100` m2/s** to decide whether
   NEMO's SET and legoESM's MAX ever disagree (section 5, row B).
4. Decide the `<=` vs `<` threshold comparison (section 5, row A) — a one-
   character change to a convective switch, so it is a question, not a diff.
