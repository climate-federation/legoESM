# Receipt — the DINO zdf-divisor GPU arm

Branch `fidelity/dino-zdf-divisor-scaling`, arm run at tip `0d1d49c26`; the
change under test is `d80df3ed7` ("NEMO's `e3w(Kmm)` is the unbranched
implicit-mixing divisor").  Preregistration:
`dino_zdf_divisor_gpu_arm_preregister.md`, committed `f2920ee71` at 21:14:39
on 2026-09-02 — **before** any arm produced a number.

## Verdict, first line

**The implicit-mixing divisor is a real but small contributor to the 20-year
climate divergence, and it is NOT the owner.**  The preregistered refute
condition FIRED: the abyssal water-mass census decisive row had to contract by
at least 50% for the divisor to own it, and it contracted by **7.75%**.  Five of
the six families move toward NEMO, none crosses a verdict boundary, and the
census family stays `REFUTE` at `R_single 96.3` where it was `104.4`.

## The table, with the preregistered prediction beside it

`R_single` is the single-member distance-vs-floor number defined in the
preregistration: `|new_value − frozen NEMO family mean| / frozen floor`.  It is
NOT the preregistered bootstrap `R` and carries no verdict on its own.

| family | preregistered prediction | frozen gap | new gap | contraction | frozen R | R_single | prediction |
|---|---|---:|---:|---:|---:|---:|---|
| `ts_water_mass_census` | **moves INSIDE the floor** (`>= 50%`, ideally 98%) | `-1.092044e-05` | `-1.007422e-05` | **+7.75%** | 104.4 | 96.29 | **MISSED — refute condition fired** |
| `mld_seasonal_cycle` | moves, but less; partial contraction, right sign | `-3.164195e-02` | `-2.324558e-02` | **+26.54%** | 2.993 | 2.199 | **HIT** |
| `acc_series` | must not get worse | `-4.511147e-01` | `-7.401314e-02` | +83.59% | 0.903 | 0.148 | HIT (improved) |
| `density_contrasts` | must not get worse | `7.112532e-04` | `5.387760e-04` | +24.25% | 1.340 | 1.015 | HIT (improved) |
| `basin_row_transports` | **no prediction** | `-4.711734e-03` | `-4.740056e-03` | −0.60% | 86.0 | 86.52 | (none; it did not move) |
| `variability` | **no prediction** | `-7.718259e-07` | `-6.830527e-07` | +11.50% | 9.647 | 8.537 | (none; it improved) |

Decisive statistics, in row order: `north of band.abyss_ge1400m.S_mean.
mean_y16_y20`, `mld.north of band.month05`, `acc.full.month09`,
`density.deep.month04`, `row.190.mean`, `census.north of band.abyss_ge1400m.
S_mean.deseasonalized_std`.

## The count, against its real null — not against a coin flip

The naive null for "how many statistics moved toward NEMO" is 383 of 766.  That
null is WRONG, and the preregistration said so before the run: a member of the
*unchanged* ensemble already scores 213/766, because a single draw is being
compared to a six-member family mean.

| family | statistics | null member (capstone `m0`) | divisor member |
|---|---:|---:|---:|
| `acc_series` | 34 | 4 | **9** |
| `basin_row_transports` | 449 | 133 | **169** |
| `density_contrasts` | 34 | 4 | **17** |
| `mld_seasonal_cycle` | 48 | 17 | **18** |
| `ts_water_mass_census` | 117 | 18 | **70** |
| `variability` | 84 | 37 | **37** |
| **total** | **766** | **213** | **320** |

`320` against a null of `213` is a real, broad shift toward NEMO — the divisor
reaches the climate.  The census family carries most of it (`18 → 70` of 117).
It is a shift in the right direction of the wrong SIZE.

## RETRACTION — the scale-compatible expectation was wrong

Recorded as a first-class retraction, not a footnote.

`dino_zdf_divisor_scaling.md` argued that the removed divisor error was
*scale-compatible* with the abyssal census gap: only `0.2–0.3%` of the removed
divergence rate needed to rectify over 20 years to produce the whole
`1.09e-05 g/kg` offset.  That arithmetic was correct and the conclusion drawn
from it was not.  **The measured contraction is 7.75%, not ~100%.**

What the scaling study got wrong, precisely:

1. **Scale-compatibility is necessary, not sufficient.**  Showing that a bias
   *could* produce a gap if a few tenths of a percent of it rectified is not
   evidence that it *does*.  The study never had a mechanism forcing that
   rectification fraction, and the arm has now measured it to be about 25x
   smaller than the ceiling the argument allowed.
2. **The 5-day rate did not persist.**  The one-step and 5-day signatures were
   real and reproduced exactly (`1.129e-04 K` one step, `1.259e-02 K` at day 4,
   both re-measured here).  A 20-year mean offset is not the time integral of a
   5-day tendency difference: the ocean equilibrates most of it.  Extrapolating
   a short-window tendency to a climate offset is the error, and it is the same
   class as the rate/window confounds this campaign has hit before.
3. **The census owner is elsewhere.**  92% of the abyssal `S_mean` gap survives
   the exact NEMO divisor.  Whatever sets it is still unidentified, and the next
   candidate should not be chosen by scale-compatibility alone.

Standing and unchanged: the divisor fix itself is correct and stays.  It is
NEMO's array bit for bit, it removed a measured `+0.30%` median / `+12.7%`
deepest-level geometry error, and it moves five of six families the right way.
This retraction is about the *attribution*, not the fix.

Also unchanged: even the 7.75% is **not an attribution**.  Attribution needs the
term removed from both models, and the twin no longer has a midpoint arm
reachable from a NEMO card.  The honest statement is "consistent with the
divisor contributing about a twelfth of the census gap".

## Year-1 battery

All three registered bars pass unchanged and nothing moves resolvably; the
divisor is invisible at one year, as preregistered.  Full table in
`dino_zdf_divisor_gpu_arm_preregister.md`.  `epoch_duplicate_identity = true`.

## Provenance

```
git SHA (both arms)      0d1d49c26a7df48819d2c55cbad2f0a7a4388f8b   (worktree /tmp/wt-dino-gpu-arm, clean)
capstone baseline        ddd3a8476afd87da4afa5747eb7e5057490b893c   arms/m0.npz
year-1 baseline          9548be86181a1f362355e0cdc30ce4ac98d14e3a   round-94 receipt
frozen scorer            a2ca7e97c, multi_year_climate_equivalence.py
                         sha256 4f2602ca73bdb1750bfe2554199a0f37611e7327454d171b798bd29cc4ad89ee

sha256 a8d84a2c49141cc605ad72aded77a5a475b473e43c8e25f088ab1855478cb76c  member20y/m0_divisor.npz
sha256 0e9e88a14c4f04b193aad1f0312a9596df3e5436fa4491ffa2d5e73ae5a48bcc  member20y/divisor_member_vs_frozen_floors.json
sha256 9a81a6fbb64f12e434da3cd6e421ffe4634c5092c295d29dd700058dedf58ac6  year1/dino_climate_rebattery_score.json
sha256 cf7d9fada07e517e6b901e50b26e248cd0de8b922c6481b624a2284bfc3ab207  proof/reach_CAPddd3_d5.npz
sha256 c8a31c8fbb6e6d4043707c3557c169e6f39487ac58411158b4606517ca3db08f  proof/reach_PAR9070_d5.npz
sha256 b6de1884ecadaf02499222883b73c57d447d7286e45d3e4fb15196159f18d0b2  proof/reach_BAT9548_d5.npz
sha256 7e7f6b9dcb3af9f50828f0a4c05f23a3f52fdaaeceda26eccccb630f414e1f74  proof/reach_FIX0d1d_d5.npz
```

Run root `/data/abyssal/dbalwada/dino-zdf-divisor-arm/` (runtime outputs are not
tracked).  GPU/wall record:

| arm | GPU | wall | note |
|---|---|---:|---|
| 20-year member m0, control | 1 (Tesla V100S 32 GB) | `13464 s` (3.74 h) | 230,400 leapfrog steps, `STABLE=True`, `blew_up_at_step = -1`; capstone m0 took `15487 s` for the same steps |
| year-1 battery, 4 arms | 0 (Tesla V100S 32 GB) | ~27 min total | `climate_a/b` 360 d, `wall_a/b` 5 d, offline CPU score |
| 5-day reach gate x4 | CPU, fp64 | ~230 s each | the one-variable proof |

The member's `run.json` carries the git SHA, every flag, the full resolved
config, the config diff against capstone `m0`, and the start time.  The producer
pin override is printed in `member20y/score_run.log`; every other
`validate_lego` guard bound normally.

**Instrument validated before its numbers were quoted:** the frozen scorer's
`reduce_legoesm`, re-run on the capstone's own `m0`, reproduces the capstone
reduced archive's member-0 column bit-exactly — 766/766 statistics, worst
absolute difference `0.000000e+00` (`member20y/instrument_validation.log`).

## What this tool now enables

The expensive half of the capstone is the NEMO side: six 20-year MPI members and
their restart archive, which are frozen and paid for.  What this arm added is a
scoring path that takes ONE legoESM member and places it against those frozen
six-member floors, with its own null measured from a member of the unchanged
ensemble.  A candidate fix now costs **one 20-year run, about 3.7 GPU-hours on a
single V100S**, plus two CPU minutes to score — instead of the six-member
ensemble the preregistered bootstrap needs.  It buys a contraction percentage
per family and a toward-NEMO count against a real null, which is enough to rank
candidates and to kill a wrong one; it does NOT buy a family verdict, and any
candidate that looks like it closes a family still has to earn five more members
before that word is used.  For a campaign whose next question is "what actually
owns the abyssal census gap", that is the difference between screening one
hypothesis a day and screening one a week.

## Choices

| choice | status |
|---|---|
| run the two GPU arms, one per GPU | **ASKED** |
| one-variable proof at four commits before launching | **ASKED** |
| write this retraction as a first-class section rather than a caveat | **ASKED** |
| all four year-1 arms on GPU 0 | **UNASKED**, forced; retired by the bit-identical duplicate pair |
| scorer producer-pin override for the new member | **UNASKED**, forced; printed at run time |
| distance-vs-floor table instead of a family verdict | **UNASKED**, forced by the preregistered bootstrap needing an ensemble |
| the null run (capstone `m0` through the same scorer) | **UNASKED**, additive; without it the `320/766` would have been read against the wrong goalpost of `383` |

**REVIEW: UNREVIEWED.** Markdown-only, which the standing rule exempts; the
numbers in it come from committed scripts and hashed artifacts, and the
instrument carries the bit-exact control above. Stated rather than implied.
