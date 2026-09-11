# GYRE from-rest YEAR, round 2: SCORED — the two models are DISTINGUISHABLE at every scored day, and legoESM's own floor is a THRESHOLD SWITCH

Preregistration: `docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_year_fromrest.md`,
written and twice independently reviewed BEFORE any member ran, and frozen: no
metric, threshold, scored row, verdict rule or expectation was changed this
round.  Round 1 (PHASE 0 BLOCKED by the TKE runaway) is kept verbatim at the
end of this file; the blocker's fix and PHASE 0's completion are in
`nemo_testcases_l2_gyre_tke_runaway_receipt.md`.

Artifacts: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_fromrest/`
(the pre-ENE-fix ensemble and the NEMO members) and `.../year_fromrest_head/`
(the branch-tip ensemble, the headline).  SHA-256 in
`nemo_testcases_l2_gyre_year_fromrest_artifacts.sha256`.
CPU, fp64, `PrecisionPolicy.fp64(transcendentals="libm")`, `JAX_ENABLE_X64=1`.

## VERDICT

**GYRE is DISTINGUISHABLE at every one of the twelve scored days, on every
scored row.**  No crossing day.  The preregistration predicted this (P4) and
said in advance what it would mean -- if P4 holds by orders of magnitude "the
year adds little beyond the ten-step ladder except the MAPS" -- so the maps and
the SHAPE of the gap, not the verdict, are the deliverable.

Day 360, control against control, on NEMO's own `tmask`, 18000 wet cells:

| | value |
|---|---:|
| gap, 3-D wet-cell temperature rms | `4.0714e-01` K |
| legoESM's own 4-member spread (max over its 6 pairs) | `2.4943e-02` K |
| NEMO's own 4-member spread (max over its 6 pairs) | `3.3945e-10` K |
| floor (the two in quadrature) | `2.4943e-02` K |
| `gap / (2 * floor)` | `8.161` |
| verdict | **DISTINGUISHABLE** |

Three things matter more than that ratio.

**1. The two models agree on the CIRCULATION and disagree on the upper-ocean
THERMAL field.**  Like for like at day 360 -- each quantity's rms difference
over NEMO's own standard deviation for that quantity -- the barotropic
streamfunction is `0.12 %` while the temperature in the top 100 m is `17.6 %`.
The signed extrema are `+13.0439` vs `+13.0379` Sv and `-29.9392` vs
`-29.9460` Sv.

**2. The gap SATURATES; it does not diverge.**  It rises as `n^1.58` in the
step count over days 30-180, turns over, goes as `n^0.78` over days 180-360,
peaks at `4.5241e-01` K on day 300 and FALLS `10 %` to `4.0714e-01` K by day
360.

**3. legoESM's own ensemble floor is not a noise floor -- it is a THRESHOLD
SWITCH, and the same-binary reproducibility floor is EXACTLY ZERO.**  This is
the round's most consequential finding and it is section 7(d).

## 1. ADMISSION

### The NEMO records

Five NEMO runs, all `STOP 0`; log `year_fromrest/nemo_members_run.log`.  Config
copy `GYRE_OMIP_L2_P3_SM_YRPERT`.  The acquisition script performed its own
checks and printed them; they are quoted, not re-derived.

| check | what it proves | result |
|---|---|---|
| all **twelve** of seed 0's restarts `cmp`-identical to the pristine unpatched binary's | the `nn_pert_seed` source patch is a NO-OP at seed 0, so the control is the unperturbed path | `IDENTICAL` x 12 |
| the four members' final restarts have four distinct SHA-256 | the perturbation reached the state | `all members differ` |
| every scored day has exactly one restart on every member | the verdict is not computed on fewer days than preregistered | `every scored day has a restart on every member` |
| member namelist vs the certified 10-step card, compared as PARSED ASSIGNMENTS | the card is the certified one | `nn_itend 10->2160`, `nn_stock 10->180`, `nn_write 10->2160`, `nn_pert_seed` (new); nothing else |
| all four members' `binary.sha256` equal (added this round) | one binary per ensemble | `578c88f1...` on all four |

**What that does NOT prove**, stated because the admission reads stronger than
it is.  Byte-identity proves the patch is inert at seed 0, not that it is right
at other seeds.  Four distinct file hashes prove the members differ SOMEWHERE
in the file, not that the scored rows have usable spread.  The namelist diff is
about the card, not the run.  None of it excludes a member that reached
`STOP 0` on a degraded trajectory.  The real admission evidence is the MEASURED
NEMO spread -- non-zero on every scored row and day
(`vacuity_gate.both_ensembles_live = true`) -- and the `0.12 %` streamfunction
agreement.  Raised by an independent review and adopted.

### The scored days, checked against a quantity this harness does not use

Days 30, 60, ..., 360; twelve days, `180` steps apart at `rn_Dt = 14400 s`,
`nn_leapy = 30`.  The preregistration argues from the card's own compiled
`restart.F90` and `stprk3.F90` that a restart stamped `kt = n` holds the state
after `n` completed steps.  Checked here against NEMO's own bookkeeping:

| restart | `kt` | `adatrj` | `ndastp` |
|---|---:|---:|---:|
| `..._00000180_restart.nc` | 180 | `30.000000` | `10130` |
| `..._00002160_restart.nc` | 2160 | `360.000000` | `11230` |

**A prose correction the preregistration owes.**  Its section 2 says day 360 is
"26 December".  NEMO's own `ndastp` says `0001-12-30`: under `nn_leapy = 30`
every month has 30 days, so day 360 is 30 December and day 180 is 30 June.  The
sentence used a real calendar.  No scored number moves; the seasonal-phase
statement shifts four days.

### The legoESM members, and a provenance defect this round had to repair

The four members on disk in `year_fromrest/` ran at commit `95cfcc38f18e`,
which **is no longer an ancestor of the branch tip** -- the history was
rewritten while they ran.  The sole model-code difference between that commit
and the tip is one commit, `2fd3560d85f3`, which re-transcribed the ENE
vorticity term; `git diff 95cfcc38f18e <tip> -- packages/` and
`git diff 2fd3560d85f3^ 2fd3560d85f3 -- packages/` are byte-identical (same
SHA-256).  So the on-disk ensemble is a **pre-ENE-fix** model.

All four members were therefore re-run at the branch tip into
`year_fromrest_head/`, against the SAME NEMO members, and the tip ensemble is
the headline.  The model code was byte-identical throughout this round
(`git diff b1b4f422764e HEAD -- packages/ src/` is empty; every commit here
touches only the harness and its tests), so the tip ensemble's members, the
reproducibility member and the scoring all ran one model.

`score` now READS the four member manifests and refuses an ensemble split
across commits, a dirty worktree, a different certified-gate SHA-256, or a
member that is not 2160 steps.  Before this round it stamped only the SCORING
tree, so this exact defect could not have gone red.

## 2. THE ALIGNMENT GATE — and what each leg cannot see

Every number here is an index-by-index difference between two models, so a
frame error does not raise, it returns a plausible number.  `--alignment-gate`
has four legs; each is SHOWN to fail on a synthetic violation
(`--plant alignment-initial | operand-mismatch | mask-shift | frame-flip`, all
exit non-zero) and it runs automatically before `--score`.

| leg | binds | measured |
|---|---|---|
| **A1** legoESM's initial state vs NEMO's BEFORE level at the entry of step 1 | the VALUES both models start from, and the VERTICAL axis | **BIT-EXACT**: `0` cells unequal of 18000 (T), 18000 (S), 17400 (u), 17100 (v), 600 (ssh); max abs difference exactly `0.0` |
| **A2** the card's latitude and depth vs NEMO's `mesh_mask` | the mesh reader | `0.0` deg, `0.0` m; `0` of 704 rounded latitudes and `0` of 21120 rounded depths differ |
| **A2b** the card's wet mask vs NEMO's own `tmask` | the HORIZONTAL SHIFT that latitude cannot see | `0` of 21120 cells differ, 18000 wet on both |
| **A3a** each restart's OWN `nav_lat`/`nav_lon` vs the mesh's `gphit`/`glamt`, identity and every reversal; **legoESM does not enter** | the netCDF restart reader | identity `3.693e-06` deg (float32 storage), nearest reversal `1.415e+01` deg, ratio `3.83e+06` |

**A1 CANNOT SEE THE HORIZONTAL FRAME on this card, and a first version of this
gate claimed it could.  RETRACTED.**  GYRE's day-0 state is horizontally
UNIFORM -- temperature and salinity have a horizontal spread of exactly `0.0` K
at every one of the 30 levels, `u`, `v` and `ssh` are identically zero -- and
NEMO's `tmask` is symmetric under all four horizontal mappings, so A1 passes
BIT-EXACTLY under a reversed `j` or `i`.  It does bind the vertical axis (the
day-0 profile steps by up to `2.275` K between levels) and the values.

**A2 IS EXACTLY BLIND TO A DIAGONAL SHIFT, which is why A2b exists.**  GYRE's
45-degree rotation makes the latitude invariant along the anti-diagonal: in the
interior overlap, `|gphit[j+1,i-1] - gphit[j,i]| = 7.105e-15` deg.  A1 is
horizontally blind and A3a compares two NEMO-side files that would shift
together, so a `(j+1,i-1)` misread passed every leg.  The MASK sees it -- GYRE's
wet rectangle is 30x20 inside a 32x22 array, so shifting it moves the
closed-boundary ring -- and A2b REFUSES to pass vacuously: it measures, per
shift, how many cells the mask discriminates, and fails if any of the eight is
invariant.  Measured, all eight are seen (40 to 98 cells of a 651-682 overlap).

**The frame itself was never wrong, and the DATA is a much weaker witness to
that than the review reported.**  Re-measured here on the interior overlap with
both masks intersected, the day-360 temperature rms under `(j+1,i-1)` is
`0.4866` K against the identity's `0.4071` K -- a factor of **`1.20`**, not the
`3.751` K the reviewer obtained.  The two probes disagreed because the
reviewer's comparison wrapped the array rather than restricting to the overlap,
so land and the opposite edge entered the sum.  Reconciled before either number
was recorded (Rule 1e): the correct statement is that at day 360 the
model-model difference can BARELY discriminate a one-cell diagonal shift, which
is the strongest possible argument for gating on the mask instead.  The full
set, day 360, interior overlap:

| shift of NEMO relative to legoESM | `T3D` rms [K] |
|---|---:|
| identity | `0.4071` |
| `(j+0,i-1)` | `0.4256` |
| `(j+1,i-1)` | `0.4866` |
| `(j-1,i+0)` | `0.5204` |
| `(j+1,i+0)` | `0.5246` |
| `(j+0,i+1)` | `0.5393` |
| `(j-1,i+1)` | `0.5875` |
| `(j-1,i-1)` | `0.5833` |
| `(j+1,i+1)` | `0.6768` |

A one-LEVEL shift is better separated but still only by `2.4x` (`0.9604` and
`1.0465` K), which is what A1's vertical binding is for.

**A3b, a DIAGNOSTIC that gates nothing.**  The identity mapping minimises the
model-model difference at every scored day, but its margin over the nearest
reversal falls from `46.8x` at day 30 to `3.7x` at day 360 -- not because the
frame degrades but because the GAP grows.  A first version of A3 gated on that
margin at `10x` and turned a correct frame red; that is a defect in the
instrument and it is recorded rather than quietly removed.

**What closes the C-grid FACE convention, since A1 cannot.**  `u` and `v` are
identically zero at day 0, so A1's bit-exact `u`/`v` rows say as little about
the face convention as its `T` row says about the horizontal frame.  The
binding evidence is the certified kt=1..10 ladder, which compares `u` and `v`
against NEMO at a state that is no longer at rest and reports
`2.7478406377547115e-12` and `3.305560306813421e-12` m/s absolute at kt=2
(`nemo_testcases_l2_gyre_phase3_round8_receipt.md`).  A one-face shift would
put the difference at the scale of the field's own shear, not nine orders
below it.  That ladder reads NEMO's BINARY record; the year reads the netCDF
restart, and the two are two writers of the same Fortran array, with the
restart's frame pinned to the mesh by A3a.  It matters here because
`PSI_MAX`/`PSI_MIN` are computed from `u`.

**Frames, for the record.**  NEMO's restarts are netCDF `x=32, y=22,
nav_lev=31`, no halo written; NEMO's per-step binary records are the
`nn_hls = 2` allocation `36x26x31`, cropped `[2:-2, 2:-2]`; legoESM's snapshots
are written through the certified 10-step gate's own `lego_fields`.  All three
land on `(22, 32, 30)`.  `18000` cells are wet in three dimensions, `600` at
the surface -- `30x20` of the `32x22` columns, the closed-boundary ring masked
out of every number by NEMO's own `tmask`.

## 3. THE SCORE

`T3D`, the headline row, at the branch tip.  `spread` is the MAX over that
model's 6 within-ensemble pairs (the sample RANGE, about 2 sigma at n=4);
`floor` is the two in quadrature; the bar is `2 * floor`.

| day | spread legoESM [K] | spread NEMO [K] | floor [K] | gap [K] | gap/(2*floor) | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 30 | `6.6516e-10` | `1.4439e-10` | `6.8065e-10` | `1.4241e-02` | `1.046e+07` | DISTINGUISHABLE |
| 60 | `1.3667e-09` | `1.5912e-10` | `1.3759e-09` | `3.0434e-02` | `1.106e+07` | DISTINGUISHABLE |
| 90 | `1.6743e-09` | `2.1442e-10` | `1.6879e-09` | `5.7269e-02` | `1.696e+07` | DISTINGUISHABLE |
| 120 | `2.3427e-09` | `1.2832e-09` | `2.6711e-09` | `9.8439e-02` | `1.843e+07` | DISTINGUISHABLE |
| 150 | `1.6681e-09` | `4.6372e-10` | `1.7313e-09` | `1.5918e-01` | `4.597e+07` | DISTINGUISHABLE |
| 180 | `1.2357e-08` | `2.9863e-10` | `1.2361e-08` | `2.4400e-01` | `9.870e+06` | DISTINGUISHABLE |
| 210 | `1.2202e-03` | `3.5079e-10` | `1.2202e-03` | `3.1845e-01` | `130.5` | DISTINGUISHABLE |
| 240 | `1.9685e-03` | `3.3041e-10` | `1.9685e-03` | `3.7652e-01` | `95.64` | DISTINGUISHABLE |
| 270 | `2.8352e-03` | `2.9437e-10` | `2.8352e-03` | `4.2219e-01` | `74.46` | DISTINGUISHABLE |
| 300 | `6.2537e-03` | `2.5957e-10` | `6.2537e-03` | `4.5241e-01` | `36.17` | DISTINGUISHABLE |
| 330 | `1.3629e-02` | `4.8689e-10` | `1.3629e-02` | `4.4296e-01` | `16.25` | DISTINGUISHABLE |
| 360 | `2.4943e-02` | `3.3945e-10` | `2.4943e-02` | `4.0714e-01` | `8.161` | DISTINGUISHABLE |

Every scored row:

| row | unit | gap day 30 | gap day 360 | floor day 360 | gap/(2*floor) day 360 |
|---|---|---:|---:|---:|---:|
| `T3D` | K | `1.4241e-02` | `4.0714e-01` | `2.4943e-02` | `8.161` |
| `T3D_0_100` | K | `2.5079e-02` | `4.9586e-01` | `3.8370e-02` | `6.462` |
| `T3D_100_1000` | K | `1.0257e-02` | `5.4819e-01` | `2.6241e-02` | `10.45` |
| `T3D_1000P` | K | `1.7094e-04` | `7.1969e-03` | `6.5761e-05` | `54.72` |
| `S3D` | g/kg | `2.2345e-03` | `6.9290e-02` | `3.4445e-03` | `10.06` |
| `SST` | K | `2.1393e-02` | `4.2961e-01` | `1.8298e-02` | `11.74` |
| `SSH` | m | `4.5334e-04` | `7.6119e-03` | `1.8928e-04` | `20.11` |
| `PSI_MAX` | Sv | `3.0349e-04` | `6.0105e-03` | `7.5233e-04` | `3.995` |
| `PSI_MIN` | Sv | `5.6972e-04` | `6.8156e-03` | `1.5672e-04` | `21.74` |
| `QNET` | W | `4.6654e+11` | `8.0114e+13` | `1.7569e+11` | `228.0` |

The preregistered control-vs-control gap and the like-for-like max over the
sixteen cross-model pairs agree to four or more significant figures on every
day, so the choice of statistic is immaterial here; the control pair is what is
tabulated, because that is what section 6 of the preregistration defines.

The 1 mK cell-count diagnostic, of 18000 wet cells:

| day | 30 | 180 | 210 | 360 |
|---|---:|---:|---:|---:|
| between the two models | `5112` | `11850` | `12094` | `12805` |
| within legoESM's own ensemble | `0` | `0` | **`135`** | **`4014`** |
| within NEMO's own ensemble | `0` | `0` | `0` | `0` |

Gates that RAN inside the scoring, so they are measurements and not
assumptions: both ensembles live on every scored day, `max|u|` on a dry u-face
is exactly `0.0` in both models at every scored day (so the unmasked barotropic
streamfunction integrates no phantom transport), one legoESM commit and one
NEMO binary across each ensemble, and the alignment gate above.

### The pre-ENE-fix ensemble, reported beside it

The `year_fromrest/` ensemble (commit `95cfcc38f18e`, before the ENE vorticity
transcription) gives the SAME verdict on every day and a very different floor:

| | day 30 | day 360 |
|---|---:|---:|
| gap [K] | `1.4241e-02` | `4.0797e-01` |
| floor [K] | `5.1984e-10` | `1.6588e-07` |
| `gap/(2*floor)` | `1.370e+07` | `1.230e+06` |

The GAP is the same to four digits.  The FLOOR differs by a factor of 150000,
and section 7(d) is what that is.

## 4. THE §1 BOUNDED-OFFSET TEST, and P1..P5

```
G_gap   = gap_360 / gap_90   = 4.0714e-01 / 5.7269e-02 = 7.109
G_floor = floor_360/floor_90 = 2.4943e-02 / 1.6879e-09 = 1.478e+07
G_floor / G_gap = 2.079e+06  >=  10   ->  BOUNDED_DETERMINISTIC_OFFSET
```

| | bound | tip ensemble | pre-ENE-fix ensemble |
|---|---|---|---|
| **P1** floor_360 below `1e-3` K | `< 1e-3` | `2.494e-02` K — **REFUTED** | `1.659e-07` K — HELD |
| **P2** floor growth below `1e3` | `< 1e3` | `3.665e+07` — **REFUTED** | `319.1` — HELD |
| **P3** the gap inside `[2.8e-3, 3.0]` K every day | band | `[1.424e-02, 4.524e-01]` — **HELD** | `[1.424e-02, 4.527e-01]` — HELD |
| **P4** DISTINGUISHABLE every day, no crossing | REFUTED iff some day has `gap <= 2*floor` | every day, `crossing_day = null` — **HELD** | same — HELD |
| **P5** the shape is GAP_AMPLIFYING or CO_GROWING | REFUTED iff `G_floor/G_gap >= 10` | `2.079e+06` — **REFUTED** | `13.33` — REFUTED |
| floor shape | -- | `FLOOR_GROWING` | `FLOOR_GROWING` |

**P1 and P2 are REFUTED on the tip model and HELD on the pre-fix one, and that
is the finding, not a scoring accident.**  P1's stated reason was that GYRE at
1 degree has no resolved eddy field to grow a perturbation exponentially.  That
reasoning is still right; what it did not bound is the threshold process the
preregistration itself named in the same paragraph -- `ln_zdfevd = .true.`,
`rn_evd = 100.`, `nn_evdm = 1`, a hard branch that is a `1e7` jump on one side.
The preregistration wrote, in advance: "P1 is a prediction about how OFTEN that
switch flips differently between members, which no laminar argument can settle
in advance."  It flips.

**P5's refutation is real; its label's GLOSS does not apply, and that must be
said in the same breath.**  The preregistration wrote
`BOUNDED_DETERMINISTIC_OFFSET` to mean "a growing floor OVERTOOK a flat gap, so
the crossing day is the whole deliverable".  Here the floor is still 1.6 orders
BELOW the gap at day 360 and overtakes nothing; the classification fires on the
two growth ratios alone.  What it correctly detects is that the floor grows far
faster than the gap.  **And P5 is inside its own sampling noise even on the
pre-fix ensemble**: recomputing `G_floor/G_gap` from each of the 6 within-
ensemble pairs rather than from the max gives `5.7, 5.8, 10.0, 13.0, 28.9,
32.7` against a bar of `10` -- two of six say CO_GROWING.  Found by an
independent review.  The `MARGINAL` band the harness applies to the verdict
ratio is not applied to this one, and the preregistration did not ask it to be.

## 5. GYRE NEXT TO DINO

Same protocol, same metric, both from rest.  DINO's pinned artifact is
`/data/abyssal/dbalwada/dino_fromrest_y1/verdict360_fromrest/phase0_floor.json`;
a later re-measurement of DINO's day-30 gap on its fixed card gives
`6.889e-04` K, and the preregistration compares against the pinned table, so
that is what is used.

| | DINO day 30 | DINO day 360 | GYRE day 30 | GYRE day 360 |
|---|---:|---:|---:|---:|
| gap [K] | `2.039e-03` | `3.924e-03` | `1.424e-02` | `4.071e-01` |
| floor [K] | `1.245e-05` | `5.450e-03` | `6.807e-10` | `2.494e-02` |
| `gap/(2*floor)` | `81.8` | `0.360` | `1.046e+07` | `8.161` |
| verdict | DISTINGUISHABLE | **INDISTINGUISHABLE** | DISTINGUISHABLE | DISTINGUISHABLE |

* **DINO's gap barely moves** (1.9x over the year) while its floor grows 438x,
  so chaos overtook a nearly-flat offset and the crossing day is 360.
  **GYRE's gap grows 29x** and its floor grows `3.7e+07`x.
* **GYRE's day-360 gap is 104x DINO's**, and GYRE is the simpler, laminar,
  single-basin case.  That is the uncomfortable number in this receipt.
* GYRE's day-30 floor is 18000x SMALLER than DINO's.  A tiny floor is not a
  better measurement -- it means the instrument hides nothing and every
  operator difference shows.

## 6. THE MAPS

Seven quantities x three columns (legoESM, NEMO, the difference) at day 30 and
day 360: `SST`, `SSS`, `SSH`, and temperature and salinity at the two
preregistered depth-band boundaries.  The figure depths are the band
boundaries, not a new choice.  Each figure carries both models' commits, the
NEMO restart's SHA-256, the mesh's SHA-256 and the masking convention in its
own header, and `maps/figure_provenance.json` repeats them machine-readably.

| figure | SHA-256 |
|---|---|
| `year_fromrest_head/maps/gyre_year_fromrest_maps_day030.png` | `eb806b78aa8d1941590f6f21d6699c3cb8fa4de2ea1b755c17a2a1f5fb43a217` |
| `year_fromrest_head/maps/gyre_year_fromrest_maps_day360.png` | `2ef365c2f4191e35a9fa0c77ecc60dc260f3467c7799f8715ada560faf5a5e31` |
| `year_fromrest_head/maps/figure_provenance.json` | `4461cdb83c70eb659adc3ba8092ee956f0d9afaceab17cd8710daa57dbd5f44a` |
| `year_fromrest/maps/gyre_year_fromrest_maps_day030.png` (pre-ENE-fix) | `a4d419cffbe1ad3353722f66de0b410f9908fe9aa7f13cff413f8e97505f978c` |
| `year_fromrest/maps/gyre_year_fromrest_maps_day360.png` (pre-ENE-fix) | `79cce91a9a997c8afaeb3513ac51c44485ca705a8088273e142204a7e7e6c0fa` |
| `year_fromrest/maps/figure_provenance.json` | `72e0a0d78186bceb065564144e2737832943d2931b76e07b4d1ab9ef0b65b63a` |

**Where the difference lives, read off the same arrays.**

| | day 30 | day 360 |
|---|---|---|
| `max abs dT` | `0.4441` K at `j=7, i=1, k=3` (depth `36.5` m, lat `20.24 N`) | `2.9493` K at `j=6, i=4, k=7` (depth `91.4` m, lat `21.59 N`) |
| share of `sum dT^2` in the western third (`36.7 %` of the cells) | `64.5 %` | `51.8 %` |
| wet cells with `abs dT > 1 mK` | `5112 / 18000` | `12805 / 18000` |

The difference is concentrated in the **western boundary current and the
subtropical thermocline**, at 20-22 N and 36-91 m, and is nearly absent below
1000 m.

## 7. INTERPRETATION

**(a) The gap is BOUNDED, not amplifying.  CONFIRMED.**  `n^1.58` over days
30-180, `n^0.78` over days 180-360, a peak at day 300 and a 10 % fall by day
360.  A trajectory diverging from a neighbour on an attractor does not turn
over.  The two GYRE solutions sit on the same attractor and differ by a bounded
`~0.45` K, about `6 %` of the temperature field's own standard deviation.

**(b) The upper-ocean THERMAL field leads; the CIRCULATION does not.
CONFIRMED.**  Day 360, each row's rms difference over NEMO's own standard
deviation for that same quantity -- like for like, including the
streamfunction, which is given as a FIELD rms rather than as a difference of
extrema:

| row | gap on NEMO's own scale |
|---|---:|
| `T3D` 0-100 m | `17.64 %` |
| `T3D` 100-1000 m | `15.92 %` |
| `SST` | `14.84 %` |
| `S3D` | `9.76 %` |
| `T3D` whole column | `6.02 %` |
| `SSH` | `4.53 %` |
| `T3D` below 1000 m | `0.79 %` |
| `PSI` (field rms) | `0.12 %` |

The barotropic streamfunction is **50x better in relative terms than the whole
column and 150x better than the top 100 m**.  That points the next round at the
TRACER column -- vertical mixing, the surface-flux top cell, lateral tracer
diffusion -- and away from the barotropic solver, which the ten-step ladder's
open DEBT rows are about.  It is a DIRECTION, not an owner: no ablation was run
on either model, so by Rule 4 nothing here is an attribution.  PLAUSIBLE.

**(c) The ten-step ladder does NOT bound the year, and the arithmetic is not
close.  CONFIRMED.**  The kt=2 temperature row is at the bar,
`5.786e-16` normalized, which on a field whose maximum is `23.4954` K is
`1.360e-14` K absolute.  Accumulated COHERENTLY over the 180 steps to day 30
that is `2.447e-12` K.  The measured day-30 gap is `1.424e-02` K --
**`5.82e+09` times larger**; reaching it at that per-step size would take
`1.05e+12` steps, about 480 million years.  So the day-30 gap is not the kt=2
error compounding.  The per-step error must GROW by orders as the flow spins
up, which is what one expects when the mismatch is RELATIVE to a state that
starts at exactly zero: at kt=2 the velocity is still `~0`, so a relative
operator mismatch reads `3e-12 m/s` absolute and says nothing about its size at
`1e-2 m/s`.  **Closing the remaining kt=2 DEBT rows therefore cannot be assumed
to close the year.**  The arithmetic is CONFIRMED; the mechanism is PLAUSIBLE.

**(d) legoESM's ensemble floor is a THRESHOLD SWITCH, its run-to-run floor is
EXACTLY ZERO, and one operator commit moved a rounding-level difference into
the third decimal place.  CONFIRMED.**

PHASE 0b, the preregistration's same-binary reproducibility member: seed 0
re-run at the tip pinned to 4 cores against the ensemble's 80.  Result:
**bit-identical on every saved field at all twelve scored days**, `T3D`
difference exactly `0.0`.  legoESM's GYRE is deterministic across thread
counts, so the run-to-run floor -- the floor the campaign's own question
("each model's own run-to-run spread") literally names -- is **zero**, and the
IC-perturbation floor is the only floor available.  The preregistration said in
advance that if it came out zero "that is itself the finding".  It also
disposes of a reviewer's objection that the tip re-run was confounded by thread
count: it is not, because the model is bit-reproducible, so COMMIT is the only
variable between the two ensembles.

With that established, the two ensembles are a controlled pair, and they
separate cleanly.  Per-member difference between the pre-ENE-fix ensemble and
the tip ensemble, `T3D` rms [K]:

| day | seed 0 (control) | seed 1 | seed 2 | seed 3 |
|---:|---:|---:|---:|---:|
| 150 | `1.68e-09` | `1.27e-09` | `2.00e-09` | `1.73e-09` |
| 180 | `1.27e-08` | `6.42e-09` | `1.21e-08` | `9.72e-09` |
| 210 | **`1.2202e-03`** | `3.18e-08` | `1.65e-07` | `3.61e-08` |
| 360 | **`2.4943e-02`** | `2.25e-08` | `1.02e-07` | `3.15e-08` |

**One member moved five orders of magnitude; the other three did not.**  The
ENE vorticity transcription changes every member's trajectory by `~1e-08` K at
day 180.  Between day 180 and day 210 that `1e-08` became `1.2e-03` for seed 0
alone.  The within-ensemble pairs at the tip say the same thing: at day 210 the
three pairs containing seed 0 are all `1.220e-03` and the three that do not are
`1.9e-08` to `1.7e-07`.

This is the `ln_zdfevd` threshold the preregistration named, doing exactly what
it said: a hard branch (`IF( MIN(rn2,rn2b) <= -1.e-12 ) p_avt = rn_evd`, a
`1e7` jump from `rn_avt0`) converting a rounding-level difference into a finite
one in one cell in one step, during the convective season.  The identification
of EVD as the branch is **PLAUSIBLE** -- it is the only threshold of that size
in the configuration and the timing matches the convective half of the year --
but no cell-level trace was run this round, so it is not CONFIRMED.

Three consequences:

1. **legoESM's ensemble "floor" is not a noise floor; it is a switch
   indicator.**  Its value says how many members have crossed, not how noisy
   the model is.  Averaging or quoting it as a standard deviation would be
   wrong.  NEMO's ensemble, on the same seed, shows `0` cells over 1 mK at
   every day and a spread that grows `2.4x` in a year -- **NEMO does not cross
   the switch within year 1 and legoESM does.**  That asymmetry is itself an
   unscored fidelity difference, and the day-0 spreads agree (`1.270e-10` K
   max-pair in both models, independently rebuilt by a reviewer from
   `mesh_mask`; I reproduced it at `1.2704e-10` K from the harness's own
   transcription), so it is not an artifact of how the perturbation enters.
2. **A last-ulp operator fix has year-scale reach on this card.**  The ENE
   commit was landed on evidence that it is bit-exact at kt=2 stages and moves
   the kt=2 velocity in its last digits; over a year it moved the control
   member by `2.5e-02` K.  The verdict did not move (the gap changed in the
   fourth digit) -- but any FLOOR measured before such a change is stale.
3. **The bar tightened by six orders and the verdict did not flip.**
   `gap/(2*floor)` at day 360 went from `1.23e+06` to `8.16`.  It is still
   DISTINGUISHABLE, and it is no longer distinguishable "by a million".  One
   more such change and this measurement becomes MARGINAL.

## 8. WHAT THIS ROUND CANNOT SEE

* **`QNET` is NOT the independent budget-closing discriminator the
  preregistration hoped for.**  It is computed from legoESM's OWN transcription
  of NEMO's `40 W/m2/K` restoring applied to each model's saved SST, because
  NEMO's own `qns` is not in the members' output.  Its per-day gap correlates
  with the `SST` gap at **`r = 0.9950`**.  It is a re-weighted SST row, blind to
  a forcing-transcription error.  The preregistration is frozen so the row stays
  scored; it is RELABELLED here and in the artifact.
* **A1 is blind to the horizontal frame and to the C-grid face convention**
  (section 2).  A2b, A3a and the kt=2 ladder carry those.
* **The floor is single-model in practice.**  NEMO's `3.39e-10` K in quadrature
  with legoESM's `2.49e-02` K changes the floor in the 9th digit; "combined in
  quadrature" is cosmetic on this card.
* **Four members**, and the `0.41` relative standard error of a spread at
  `n = 4` is not propagated into the verdict.  Immaterial at a ratio of `8.16`
  only because 8.16 is outside the `[0.5, 2]` MARGINAL band -- not by much.
* **No ablation on either model**, so section 7(b) is a direction, not an
  attribution, and 7(d)'s EVD identification is PLAUSIBLE only.
* **The EVD threshold crossing was not traced to a cell.**  That is the single
  cheapest next measurement and it is named in the open questions.
* Western-boundary separation latitude, thermocline depth and the
  subtropical/subpolar transport partition remain unscored; the
  preregistration waived them in writing and this round does not add them.
* Day 360 is one phase of one seasonal cycle and about a third of a
  first-baroclinic basin crossing.

## 9. RETRACTIONS

1. **"A1 binds the frame" — RETRACTED.**  GYRE's day-0 field is horizontally
   uniform and the mask is symmetric under every reversal, so A1 passes
   identically on a flipped frame.  The claim lived one commit.
2. **A3's first gate, "the identity mapping beats every reversal by 10x on the
   model-model difference" — RETRACTED as a gate.**  It turned a correct frame
   red at day 360 because the real gap had grown to `4.07e-01` K against a
   `1.52e+00` K reversal signal.  It survives as a diagnostic.
3. **"A2 closes the horizontal frame" — RETRACTED.**  Latitude is invariant
   along GYRE's anti-diagonal to `7.105e-15` deg.  A2b closes it.
4. **The harness's headline statistic — RETRACTED and replaced by the
   preregistered one.**  It scored the verdict, P3, P4 and the section-1 test on
   the max over the 16 cross-model pairs; section 6 of the preregistration
   defines the gap as control vs control.  The two agree here, but the code was
   quoting a statistic the document does not name.
5. **The preregistration's "day 360 is 26 December" — RETRACTED.**  NEMO's own
   `ndastp` says `0001-12-30` under `nn_leapy = 30`.  No scored number moves.

## 10. REVIEWS

Two independent fresh reviews: one BEFORE any gap was quoted, on the alignment
and admission plan; one AFTER, briefed to try to make the verdict pass on a
misaligned frame, on mismatched days or on a wrong floor, and to check the
figures' provenance.  **Codex was occupied on this branch's kt=2 momentum
ladder and GLM-5.2 was unavailable, so both reviewers were Claude instances
started with no shared context.**  Stated because the repository's default is
codex + GLM and this round did not meet it.

Every finding was RE-MEASURED before it was acted on.  All of them reproduced.

### Review 1 — before scoring

| finding | measured | what changed |
|---|---|---|
| A1 has ZERO horizontal frame power | horizontal spread `0.0` K at all 30 levels; `u`,`v`,`ssh` identically `0.0`; mask symmetric under all four mappings | claim RETRACTED in code and here; the gate now measures and reports its own blind spot |
| the headline verdict is the max over 16 cross-model pairs; the preregistration says control vs control | the two agree to 4+ digits | the control pair is now the statistic of record |
| `QNET` is a rescaled `SST` row | `r = 0.9950` | measured into the artifact, relabelled |
| P4 was refutable by the `MARGINAL` label, which the preregistration does not contain | vacuous at the observed ratios | P4 now applies its written rule |
| the vacuity gate never ran on the verdict path | NEMO's spread is non-zero everywhere | both ensembles now required live; test fails on a dead one |
| the streamfunction integrates `u` with no mask | `max|u|` on a dry u-face is exactly `0.0`, both models, every day | now a gate |
| the restart's axis order was assumed | holds | now read |
| the tip re-run is confounded by thread count unless PHASE 0b runs | **bit-identical at all 12 days** | PHASE 0b run; the objection is refuted by measurement |

### Review 2 — after scoring

| finding | measured | what changed |
|---|---|---|
| the on-disk ensemble's commit is unreachable and its one delta is the momentum operator | the two diffs are byte-identical | the tip ensemble is the headline; both are reported |
| `score` never reads a member manifest, so a split-commit ensemble has no red | -- | `ensemble_provenance` added, four synthetic violations each shown to raise |
| PHASE 0b was preregistered ASKED and not run | -- | run; see 7(d) |
| P5 is inside its own sampling noise | per-pair `G_floor/G_gap` = `5.7 … 32.7` against a bar of `10` | reported in section 4 |
| A2 is exactly blind to a `(j+1,i-1)` shift | `7.105e-15` deg — REPRODUCED; but the reviewer's `3.751` K for the gap under that shift did NOT reproduce (interior overlap gives `0.4866` K, a factor `1.20`) — the reviewer's probe wrapped instead of restricting to the overlap | A2b added, with a measured non-vacuity check; the corrected, much weaker discrimination is what makes A2b necessary rather than optional |
| the u-face slice is bound only by the kt=2 ladder, in another artifact | the reviewer's `1.4-1.6x` margin is on `u`; the temperature row's own i-shift margin re-measured here is `1.05-1.32x`, i.e. weaker still | stated in section 2 |
| the restart's vertical axis is checked by dimension NAME only | REPRODUCED: `max|nav_lev - gdept_1d| = 1.719e-04` m; level 31 is dry | recorded |
| the floor is single-model in practice | NEMO's contribution changes the 9th digit | section 8 |
| the spread asymmetry is real and conservative | REPRODUCED: day-0 max-pair spread `1.2704e-10` K from legoESM's own transcription, and the reviewer independently rebuilt NEMO's expression from `mesh_mask` and got the same | section 7(d) |
| PSI's "%" was gap/extremum while the T rows were rms/std | PSI field rms is `0.12 %` | section 7(b) made like-for-like |

The day-mapping attack (B) closed with no finding: `adatrj = 30.0` at
`kt = 180`, and the restart writes `tn/un/vn/sshn` from `Kbb` after the
`Nbb <-> Naa` swap in the card's own compiled step routine.

## 11. ASKED / UNASKED

Every scientific choice in the preregistration's own table -- run length,
cadence, member count, seeds, scored rows, the perturbation, the verdict rule
-- is unchanged and was ASKED there.  This round added the following.

| # | choice | value | ASKED? |
|---|---|---|---|
| 1 | re-run the four legoESM members at the branch tip, because the ones on disk ran at a commit that is no longer an ancestor of it | run them; the tip is the headline, the pre-fix ensemble is reported beside it | **UNASKED.**  Offered for revert: every number can be read off the pre-fix ensemble, and both are in this receipt.  Taken because reporting a verdict for an orphaned commit would be a stale figure, and because the pair turned out to be the round's main finding |
| 2 | run the preregistration's PHASE 0b reproducibility member | run it, pinned to 4 cores against the ensemble's 80 | ASKED in the preregistration (item 12, "run it"); the thread count is this round's pick and is the thing PHASE 0b exists to vary |
| 3 | the alignment gate's thresholds (`1e-4` deg and `1e4` on the coordinate identity; bit-exact on A1; zero cells on A2b) | as written | **UNASKED.**  Instrument thresholds on geometry identities, not on physics; no scored number depends on them -- the gate raises or it does not |
| 4 | the figures' two subsurface depths | the preregistered depth-band boundaries, `100 m` and `1000 m` | not a choice -- taken from the preregistration's section 4 |
| 5 | where the maps are written | `<root>/maps/` | measurement-neutral |

No preregistered metric, threshold, scored row, verdict rule or expectation was
changed.  Every harness change this round either ADDS a gate, fixes a crash in a
path that had never been executed, or moves a statistic TOWARD the frozen
document.

## 12. OPEN, ranked

1. **Trace the EVD threshold crossing to a cell**, between day 180 and day 210
   on the tip's seed 0.  It is the cheapest measurement in the campaign, it
   turns 7(d)'s PLAUSIBLE into CONFIRMED or kills it, and it decides whether
   legoESM's convective switch is transcribed the way NEMO's is.
2. **Re-measure the floor after any operator change.**  A floor measured before
   a last-ulp fix is stale, demonstrated here at a factor of 150000.
3. The tracer column -- vertical mixing, the top cell, lateral tracer diffusion
   -- is where the year says the remaining GYRE gap lives.  It needs an
   ablation on BOTH models before it is an attribution.
4. NEMO does not cross the switch in year 1 and legoESM does.  That is an
   unscored fidelity difference with no owner yet.

---

# ROUND 1, kept verbatim (superseded by round 2 above)

# GYRE from-rest YEAR, round 1: PHASE 0 IS BLOCKED — the TKE closure diverges at day 6

Preregistration: `docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_year_fromrest.md`
(written and independently reviewed twice BEFORE any member ran).
Artifacts: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_fromrest/`,
SHA-256 in `phase0_blocker_artifacts.sha256`.

## VERDICT

**The GYRE year cannot be run.**  legoESM's certified GYRE card
(`GYRE-zco`, `build_gyre_zco_card`) aborts at **step 48 of 2160**, and the
reason is not the abort's message.

The owner is identified, not merely localised: in ONE column -- the south-east
corner of the wet domain, at the deepest interior interface -- the model's
turbulent kinetic energy obeys

```
e(n+1) = e(n) + (0.5 * rn_ediss * dt / L) * e(n)^1.5
```

with the coefficient matching `0.5 * 0.7 * 14400 / 306.5 = 16.4437` to five
significant figures over five consecutive steps.  That is the EXPLICIT half of
NEMO's semi-implicit dissipation split running as a SOURCE with nothing on the
implicit diagonal to balance it.  `e` reaches `3.5e40 m2/s2` by step 47, the
eddy viscosity reaches `5.6e21 m2/s`, temperature and salinity go infinite and
the velocities go NaN.  The sea surface inherits the NaN inside step 48 and the
vertical-geometry guard in `eos.py:736-740` is what finally reports it.

This is a **NEW, previously unmeasurable finding**: the kt=1..10 ladder that
certified this card stops 38 steps before the instability starts.  "Never
measured beyond ten steps" turns out to mean "cannot currently be measured
beyond 47".

CONFIRMED.  Three readings, all stopping at the same step: the year harness
itself (all four members), a bare stepping walk, and the geometry trace.  They
are three READINGS OF ONE CODE PATH, not three independent instruments -- all
call the certified gate's `_surface_forcings` and `LatLonCGridOceanModel.step`
-- so they establish reproducibility, not independence.  (Corrected after
review; the first version of this line claimed independence.)

## The measurement

The probe is COMMITTED, not a heredoc: `--census` on the year harness
(`nemo_testcase_l2_gyre_year_fromrest.py`).  It is fp64 / CPU / scalar-libm and
steps the card through the SAME functions the ten-step gate uses
(`nemo_testcase_l2_gyre_phase3_gate._surface_forcings` +
`LatLonCGridOceanModel.step`), and it EXITS NON-ZERO on the first non-finite
value rather than printing a table someone has to read:

```
STATUS NONFINITE at step 47: u (17250 cells), v (16950 cells),
                             T (16350 cells), S (16350 cells)
```

It reports one step EARLIER than the model's own abort, and it names the
fields.

| step | max `tke` [m2/s2] | max `tke_avm` [m2/s] | eta, u, v, T |
|---:|---:|---:|---|
| 1 | `5.538e-03` | `8.188e-02` | healthy |
| 20 | `2.208e-03` | `1.136e-01` | healthy |
| 30 | `2.205e-03` | `6.237e-01` | healthy |
| 36 | `5.779e-03` | `2.286e+00` | healthy |
| 37 | `1.298e-02` | `3.426e+00` | healthy |
| 38 | `3.726e-02` | `5.805e+00` | healthy |
| 39 | `1.554e-01` | `1.185e+01` | healthy |
| 40 | `1.161e+00` | `3.241e+01` | healthy |
| 41 | `2.173e+01` | `1.402e+02` | healthy |
| 42 | `1.687e+03` | `1.235e+03` | healthy |
| 43 | `1.141e+06` | `3.212e+04` | healthy |
| 44 | `2.004e+10` | `4.257e+06` | healthy |
| 45 | `4.665e+16` | `6.495e+09` | healthy |
| 46 | `1.657e+26` | `3.871e+14` | healthy |
| 47 | `3.507e+40` | `5.631e+21` | **T, S = inf; u, v = NaN (17250 faces)** |
| 48 | — | — | abort |

**The dynamics are healthy the entire time.**  Right up to step 46:
`eta` in `[-4.90e-2, 4.77e-2] m`, `u` in `[-8.11e-2, 5.44e-2] m/s`,
`v` in `[-2.97e-2, 1.52e-1] m/s`, `T` in `[0, 23.83] C`,
`uu_b` in `[-1.11e-2, 1.98e-2] m/s`.  The advective CFL is about `0.011`.
There is no barotropic mode, no CFL violation and no front.  Only `tke`,
`tke_avm`, `tke_avt` and `tke_dissl` move, and they move together.

`tke` FALLS from `5.5e-3` at step 1 to `2.2e-3` by step 20-30, then TURNS
AROUND between steps 30 and 36 and runs away, ACCELERATING.  An earlier draft
of this receipt read that acceleration as a THRESHOLD crossing and pointed at
the convective switch.  **That reading is RETRACTED**: the acceleration is
fully explained by the recursion identified below, which needs no threshold,
and two independent reviews said so before any further compute was spent.  An
adjacent paragraph claiming `tke_avt` tracked `tke_avm/10` and then became
equal is **also RETRACTED** -- the two are maxima taken independently and need
not sit in the same cell, and the log contradicts the claim anyway (they are
EQUAL at steps 10 and 20 and are NOT equal at step 46).

## THE OWNER, and it is not a mystery

The census already contains NEMO's mixing length.  `dissl` is a RATE,
`sqrt(en)/L` (`zdftke.F90:717`), not a length -- so `L = sqrt(tke)/tke_dissl`
is readable straight off the table:

| step | 36 | 38 | 40 | 41 | 42 | 43 | 44 | 45 | 46 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `L` [m] | 306.93 | 306.86 | 306.71 | 306.60 | 306.52 | 306.50 | 306.50 | 306.50 | 306.51 |
| `e(n+1)/e(n)^1.5` | 29.5 | 21.6 | 17.4 | 16.65 | **16.4669** | **16.4426** | **16.4439** | **16.4454** | **16.4419** |

and `0.5 * rn_ediss * dt / L = 0.5 * 0.7 * 14400 / 306.5 = ` **`16.4437`**.

**The recursion is `e(n+1) = e(n) + (0.5 * rn_ediss * dt / L) * e(n)^1.5`**,
matched to five significant figures over five consecutive steps.  That is the
EXPLICIT half of NEMO's semi-implicit dissipation split (`zdftke.F90:419`)
running as a SOURCE with no implicit counterpart on the diagonal (`:414`).
Shear, buoyancy and TKE diffusion are numerically absent from the balance at
that cell: nothing else in the closure can produce an `e^1.5` map with that
coefficient.

This identification came from an independent review of this receipt, from the
numbers already in it.  **The arithmetic was re-derived here before it was
adopted** -- a reviewer's finding is a hypothesis, not an instruction -- and it
reproduces: the measured ratios are `16.4419` to `16.4669`, the prediction is
`16.4437`.

### WHERE, measured rather than inferred

The reviewer predicted the peak would sit on the deepest active interface, and
named the one-line measurement that would settle it: print `argmax(tke)`.  Run
(`phase0_blocker_census_argmax.log`):

| steps | argmax(tke) | |
|---|---|---|
| 30-34 | wanders, `k=0`, `tke ~ 2e-3` | surface, decaying |
| **35-47** | **`j=1, i=30, k=28`**, pinned | the runaway |

`(j=1, i=30)` is the **south-east CORNER column of the wet domain** -- wet `j`
runs 1..20 and wet `i` runs 1..30, and that column has 2 wet neighbours of 4.
`k=28` is the **second-deepest of 30 levels** (`bottom_level = 29`), at
`gdept = 3849.90 m`.  So the runaway is ONE COLUMN, at the deepest interior
interface, i.e. the last interior row of the vertical tridiagonal solve.

`L` pins at `306.50 m` there from step 42, which is LARGER than any thickness
in GYRE's ladder (`max e3t_1d = 301.10 m`, `max e3w_1d = 300.98 m`), so `L` is
at a geometric ceiling built from more than one spacing.  Which ceiling, and
whether the bottom TKE boundary condition is the reason the last row takes a
dissipation add-back without an implicit counterpart, is **PLAUSIBLE, not
measured**: the reviewer's reading is that NEMO applies its bottom TKE
Dirichlet unconditionally while legoESM gates it on a `bottom_tke_bc` flag that
defaults to False and that this card never sets.  That is a code reading, and
it is the next round's first measurement.

## Why the reported error names the wrong thing

The abort message is `raw-mesh e3w_int must contain only finite values > 0`
(`eos.py:736-740`).  That guard is correct and it is not the owner:

* at step 47's END the live `e3w` computed from the prognostic state is
  perfectly healthy — stretch in `[0.9999877, 1.0000119]`, minimum `e3w`
  `10.1210 m`, zero non-positive cells;
* inside step 48, the trace of every live-`e3w` producer
  (`phase0_blocker_e3w_trace.log`) shows call sites 11-19 healthy with the
  same stretch, and only the LAST TWO calls of the step returning `NaN`;
* the `NaN` is in the **stretch**, i.e. in `eta`
  (`nemo_r3t_stretch`, `eos.py:863-923`, floors at `1e-6` and guards dry
  columns, so it cannot manufacture a non-finite value from a finite `eta`).

So the chain is: TKE runs away -> the eddy coefficients run away -> `T`, `S`
go infinite and `u`, `v` go NaN at step 47 -> `eta` inherits the NaN inside
step 48 -> the vertical-geometry guard fires.  **Attributing this to the
geometry, to the free surface, or to the barotropic solver would be wrong**;
each was checked and each is healthy at the last finite step.

## What this does and does not say

* **CONFIRMED**: the card aborts at step 48; TKE is the diverging field; the
  dynamics are healthy through step 46; the guard that reports it is
  downstream.
* **CONFIRMED**: the owning term is the explicit half of the TKE dissipation,
  acting as an `e^1.5` source with no implicit counterpart; the coefficient
  matches `0.5 * rn_ediss * dt / L` to five significant figures over five
  steps; the runaway is one corner column at the deepest interior interface.
* **RETRACTED**: the enhanced-vertical-diffusion convective switch as the
  trigger.  Two independent reviews refuted it -- the implied stratification at
  that cell is a strong pycnocline, not a convecting one, and the diffusivity
  ratio that was offered as evidence does not say what the earlier draft said
  it said.
* **PLAUSIBLE, NOT MEASURED**: that the missing implicit counterpart is the
  bottom TKE boundary condition, which legoESM gates behind a flag defaulting
  to off and which NEMO applies unconditionally.  Code reading only.
* **NOT ATTEMPTED**: no fix.  A change to a certified card's closure is a
  numerics change requiring both adversarial reviews and, almost certainly, a
  configuration decision that is not the agent's to make.  It is reported, not
  patched.
* **NEMO's own side is UNMEASURED, and probably need not be spent here.**  No
  NEMO GYRE year has been run in this campaign.  GYRE is NEMO's standard
  demonstration configuration and is routinely integrated for years, and
  NEMO's semi-implicit split gives `e_new/e_old -> 1/3` at large `e`, which
  analytically cannot produce this map.  So the one-sided conclusion stands --
  legoESM cannot reach a year on this card -- and the NEMO members are better
  spent after the closure is fixed than before.

## The operator script was dry-run, and it was broken

`run.sh` cannot be executed by the agent, so the part of it that can silently
refuse a CORRECT input was extracted and run by hand against the certified
namelist.  It refused, with `REFUSE: 197 changed rows`.

The bug: the namelist check compared the two files LINE BY LINE with a `zip`,
so the moment `nn_pert_seed` was inserted every subsequent line shifted by one
and the whole file read as changed.  The operator would have hit that on the
first member.  It also inserted the new row immediately after the
`&namusr_def` header, ahead of the group's own comment line.

Fixed: the check now compares PARSED ASSIGNMENTS (`key = value`, skipping
comment and group lines), so it is insensitive to position, and the row is
inserted just before the group's closing `/`.  Re-run against the certified
namelist it reports exactly

```
    nn_itend       10 -> 2160
    nn_stock       10 -> 180
    nn_write       10 -> 2160
    nn_pert_seed   (new) -> 1
```

and a plant that additionally flips `rn_Uv` from `2.0` to `4.0` exits
non-zero naming `rn_Uv`.  It also refuses if FEWER than all three rows change,
so a writer that silently failed to reach `nn_itend` cannot produce a ten-step
run wearing a year's name.

## Consequence for the round

| phase | status |
|---|---|
| preregistration | **DONE**, and revised after two independent reviews |
| harness + tests | **DONE**; `--self-check` green, 18 direct tests pass, three of five plants exercised |
| NEMO `run.sh` + MY_SRC patches | **WRITTEN**, not run.  Its PHASE-0 gate correctly refuses, because `phase0_floor.json` does not and cannot exist |
| PHASE 0 (legoESM ensemble) | **BLOCKED at step 48 of 2160** |
| PHASE 1 (NEMO ensemble) | not started |
| verdict, floor, maps | **UNMEASURED** |

Every preregistered expectation P1-P5 is therefore **UNMEASURED**, not held and
not refuted.  The 40-hour entering condition from the preregistration's section
0 (`T3D` rms `2.768e-3 K` at ten steps) stands, and remains the only
legoESM-vs-NEMO GYRE number beyond kt=2.

## Reproduce

```
PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:\
packages/ice:packages/land:packages/ml:packages/tools:src \
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_year_fromrest.py \
  --census 50 --census-start 36
```

about 6 minutes on CPU, exit status 1.  The card requires
`transcendentals='libm'`, which the precision policy rejects on GPU, so CPU is
not a preference here.  Output preserved as
`phase0_blocker_census_committed.log`.

## Reviews of this round, and what they changed

Two independent fresh reviews ran on this diff and this receipt, on top of the
two that ran on the preregistration before any simulation.  Codex was occupied
on this branch's kt=2 ladder and GLM-5.2 was unavailable, so all four reviewers
were Claude instances with no shared context.

| finding | what changed |
|---|---|
| the owning term is determined by the receipt's own table: `dissl` is a rate, so `L = sqrt(tke)/dissl`, and `e(n+1)/e(n)^1.5` matches `0.5*rn_ediss*dt/L` | the receipt now IDENTIFIES the term instead of calling it UNMEASURED.  The arithmetic was re-derived here before adoption |
| the convective-switch trigger is refuted; the acceleration needs no threshold | that attribution RETRACTED |
| the `tke_avt` / `tke_avm` ratio paragraph is contradicted by the log it cites, and compares two independently-taken maxima | paragraph DELETED |
| "three independent probes" is one code path read three ways | corrected to "three readings of one path" |
| the committed census reduced min/max over the FINITE subset, so a half-infinite field printed a healthy range | fixed to raw min/max; a test pins it by source |
| the floor is a max over six within-model pairs while the gap was a single control pair -- an extreme of six against one draw, inflating the floor ~2x toward INDISTINGUISHABLE | the headline ratio now uses the max over the sixteen cross-model pairs, the same order statistic; the control-pair ratio is reported beside it |
| NEMO's restart fields were never checked for finiteness while legoESM's were | finiteness gate added to the NEMO reader |
| a non-finite ratio was reported as `UNMEASURED_ZERO_FLOOR` | distinct `UNMEASURED_NONFINITE` verdict |
| only `T3D` reached the exit status; salinity, SSH, PSI and QNET had verdicts nothing could see | every row's day-360 verdict is now printed |
| the "no sqrt(2)" test grepped one exact spelling, and the constant existed only to be absent | the constant is DELETED; the check is now a property (no root factor multiplies a floor) with its own non-vacuity probe |
| the time-level citation is off by 43 lines | it named the shipped `stprk3.F90`; the card compiles its MY_SRC copy.  Both are now cited |
| "15 direct tests, every plant fires" over-claims | corrected below |

**Plant coverage, stated honestly.**  Eighteen direct tests pass.  Three of the
five plants are exercised (`perturbation-zero`, `perturbation-relative`,
`operand-mismatch`), each shown to raise.  `floor-inflate` and `gap-zero` live
inside `score()` and are exercised only when member data exists, which it does
not while PHASE 0 is blocked.  They are UNEXERCISED, and that is a debt, not a
detail.

## Artifacts, SHA-256

All under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_fromrest/`.

| file | sha256 |
|---|---|
| `phase0_blocker_tke_census.log` | `6c62febdec708ff092e39052b3f9b497c653edb54ae9940568f0f52588181349` |
| `phase0_blocker_e3w_trace.log` | `c7d35b0fc893997a8665dee8aaf75e06749b3ad72816e68db773c6fa0aa91f02` |
| `phase0_blocker_census.py` | `1bd66258160d21f7c1b064e4f8c975d9032bf01d67b210253e4b85aacda8b626` |
| `phase0_blocker_census_committed.log` | `d29bc20d4cae9a451ecba38ac89741f4e4fc1f594f76542637fe52765242a0a9` |
| `phase0_blocker_census_argmax.log` | `8488369f94b3ce3ee8cf7e8f541b88b90ed97f7a06ff88976aad431395e7da37` |
| `lego_seed0.log` | `ff005d91ef3659cb359c3404ce560afcb4d673ea48c6d14c79ef9916ad4adad3` |

`phase0_blocker_census.py` is the throwaway that FOUND this; it is superseded
by the harness's `--census` mode and kept only so the first reading is
reproducible.  `lego_seed0.log` is the year harness's own first failure, i.e.
the blocker as it appeared in production rather than in a probe.
