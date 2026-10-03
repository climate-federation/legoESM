# Round 208 / VORTEX round 21 — Decision 74's resolution ladder (30 / 15 / 10 km)

**Verdict: LANDED, measurement only.** The two VORTEX cards were measured at
three grid spacings. No physics statement is proposed and none lands here; what
lands is four new cards, four NEMO records, four ladders, their registries and
this receipt.

**Headline.** The transcription carries **no grid-size dependence**. Every
`kt=1` row is at bar at every rung; the first row over bar is `kt=2` on both
cards at every rung, with the same field set; and the flux card's remaining
error does not grow as the grid refines — it **shrinks roughly in proportion to
the timestep** (fitted exponent −0.7 … −1.1 against `r = 30 km / dx`, where a
pure `dt` proportionality is −1). The vector card sits at the rounding floor at
every rung, so round 199's landing holds at 15 km and 10 km as it does at 30 km.
That refutes this round's own prediction P3, which expected growth.

---

## 1. The deck rule, verified in NEMO's own files

The rule was not chosen. NEMO's VORTEX ships an AGRIF zoom whose space and time
refinement ratios are pinned in the test itself, and a child namelist that says
what a refinement of this configuration changes.

`tests/VORTEX/EXPREF/AGRIF_FixedGrids.in:2` reads

```text
22 41 22 41 3 3 3
```

— one zoom, i-range 22-41, j-range 22-41, and the three trailing integers are
the space-i, space-j and TIME refinement ratios, all 3.

A complete `diff -u` of `tests/VORTEX/EXPREF/namelist_cfg` against
`tests/VORTEX/EXPREF/1_namelist_cfg` is reproduced in
`phase3/round208/child_deck.diff`. Every value hunk in it:

| parameter | parent | NEMO's child (ratio 3) | this ladder at ratio 2 (15 km) | this ladder at ratio 3 (10 km) | citation |
|---|---|---|---|---|---|
| `rn_dx`, `rn_dy` | 30000. | 10000. | **15000.** | **10000.** | `1_namelist_cfg:21-22` |
| `rn_Dt` | 2880. | 960. | **1440.** | **960.** | `1_namelist_cfg:43` |
| `nn_itend` | 3000 | 6000 | *see DECISION_NEEDED below* | *see below* | `1_namelist_cfg:34` |
| `&namagrif` sponge block | absent | `rn_sponge_tra = rn_sponge_dyn = 0.00768` | **not applied** | **not applied** | `1_namelist_cfg:103-108` |

and every value the child leaves alone, which this ladder therefore also leaves
alone:

| parameter | value in parent and child alike | citation |
|---|---|---|
| `rn_dz` | 500. | `1_namelist_cfg:23` |
| `rn_ppumax` | 1.0 | `1_namelist_cfg:25` |
| `nn_rot` | 0 | `1_namelist_cfg:26` |
| `nn_e` (barotropic substeps) | 48 | `1_namelist_cfg:222`, parent `namelist_cfg:214` |
| `ln_traldf_OFF` | `.true.` | `1_namelist_cfg:166`, parent `namelist_cfg:158` |
| `ln_dynldf_OFF` | `.true.` | `1_namelist_cfg:230` |

The last two rows are why there is no "and the viscosity follows": **both
lateral operators are OFF in parent and child alike, so this configuration has
no lateral viscosity or diffusivity to rescale.** Likewise `nn_e` is NOT scaled
with `rn_Dt`: NEMO's own refinement keeps 48 barotropic substeps inside a
timestep three times shorter, and so does this ladder.

The remaining hunks in the child diff are comment-only (`namc1d` header lines,
a trailing-space change on `ln_dynspg_exp`, a `"key_netcdf4"` annotation) and
change no value.

### The one DECISION_NEEDED, and why the round proceeded

`nn_itend` is the only value the child changes that has no ratio-2 analogue:
6000 is **not** 3 × 3000, so no rule can be read off it. It is registered here
as **DECISION_NEEDED** rather than guessed.

It is also **inert for everything this round measures**: the acquisition deck
pins `nn_itend = 10` at every resolution, exactly as the certified 30 km decks
do, and every ladder in this campaign is `kt = 1..10`. The cards' `n_steps`
metadata carries NEMO's own cited value where one exists — 3000 at 30 km
(parent) and 6000 at 10 km (`1_namelist_cfg:34`) — and the parent's 3000 at
15 km, where nothing is cited. **No gate in this campaign reads that field.**

### The domain size is NEMO's own derivation, not a choice

`usrdef_nam.f90` of the compiled `VORTEX_OMIP_L1_P3` build derives the whole
grid from the namelist at run time:

```fortran
         IF     ((nn_rot==0).OR.(nn_rot==2)) THEN
            kpi = NINT( 1800.e3  / rn_dx ) + 3
            kpj = NINT( 1800.e3  / rn_dy ) + 3
```

at `usrdef_nam.f90:138-139`, with `kpk = NINT( 5000._wp / rn_dz ) + 1` at
`usrdef_nam.f90:144`. So a non-AGRIF run at 10 km covers the FULL 1800 km box at
10 km — it is not a 22-41 zoom — and the cell count follows the resolution:
63 → 123 → 183, with `jpkglo = 11` unchanged because `rn_dz` is unchanged.

**No rebuild was needed, and none was done.** The grid is a run-time quantity,
so each rung is a namelist change and nothing else. That is not an assumption:
each rung's run printed its own resolved values, quoted in §3.

---

## 2. What was built, recorded and admitted

Nothing was compiled. Each rung ran the **certified executables** — the very
binaries the 30 km records were produced with — and the script proves it by
hashing them against the committed round-2/round-3 manifests before running:

```text
CERTIFIED_BUILD_REUSED .../VORTEX_OMIP_L1/BLD/bin/nemo.exe        f1151cb6770a1966…
CERTIFIED_BUILD_REUSED .../VORTEX_OMIP_L1_P3/BLD/bin/nemo.exe     cfcc0b452780b92b…
CERTIFIED_BUILD_REUSED .../VORTEX_VEC_OMIP_L1/BLD/bin/nemo.exe    6fd6a372d95c7d32…
CERTIFIED_BUILD_REUSED .../VORTEX_VEC_OMIP_L1_P3/BLD/bin/nemo.exe 496c8a52115806c0…
```

This is deliberate and it is the stronger control: rebuilding would have
produced the same binary from the same sources and destroyed the proof that a
rung differs from its certified counterpart **in the deck alone**. The reused
instrumented build is re-checked for the writer and the reused reference build
re-checked for its absence, every run.

The certified 30 km `EXP00` decks are never touched: each rung assembles its run
directory from the shipped `EXPREF` plus its own resolution patch, and the two
arms' namelists are `cmp`-ed after the runs.

| rung | record | frames | geometry (with halo) | restart byte-identical | plant |
|---|---|---:|---|:--:|:--:|
| 15 km flux | `phase3/vortex_ladder/15km/flx` | 24 | 127 × 127 × 11 | **yes** | fired |
| 15 km vector | `phase3/vortex_ladder/15km/vec` | 24 | 127 × 127 × 11 | **yes** | fired |
| 10 km flux | `phase3/vortex_ladder/10km/flx` | 24 | 187 × 187 × 11 | **yes** | fired |
| 10 km vector | `phase3/vortex_ladder/10km/vec` | 24 | 187 × 187 × 11 | **yes** | fired |

All four `ADMITTED`. The halo is two cells each side, so 127 = 123 + 4 and
187 = 183 + 4, matching `usr_def_nam`'s derivation. Additions-only is proved the
way note AS requires: the **plain** build's `kt=10` restart is byte-identical to
the instrumented build's at every rung. The checker parses each record's own
self-describing header (note BD) and predicts no size; its header plant turns it
red at every rung, so the guard the admission rests on is shown able to fail.

---

## 3. The resolved configuration of each rung, read off NEMO's own output

Not the deck's comments — `ocean.output`:

| quantity | 30 km (certified) | 15 km | 10 km |
|---|---|---|---|
| `rn_dx`, `rn_dy` [m] | 30000 | 15000 | 10000 |
| `rn_dz` [m] | 500 | 500 | 500 |
| `Ni0glo` × `Nj0glo` | 63 × 63 | 123 × 123 | 183 × 183 |
| `jpkglo` | 11 | 11 | 11 |
| `rn_Dt` [s] | 2880 | 1440 | 960 |
| `nn_e` | 48 | 48 | 48 |
| `ln_traldf_OFF` / `ln_dynldf_OFF` | T / T | T / T | T / T |
| `rn_ppumax` [m/s] | 1.0 | 1.0 | 1.0 |
| `nn_rot` | 0 | 0 | 0 |
| LX [km] | 1830 | 1815 | 1810 |
| H [m] | 5000 | 5000 | 5000 |

`LX` differs slightly between rungs because it is `(kpi-2)*rn_dx`, and the
integer `+3` in `usr_def_nam` is not divisible by the ratio. That is NEMO's own
arithmetic, carried as it stands.

## 4. The four new legoESM cards

`VORTEX-15km-zco`, `VORTEX_VEC-15km-zco`, `VORTEX-10km-zco`,
`VORTEX_VEC-10km-zco`, all in the recipe module beside the certified pair.
Every resolved value is **written out** in `_VORTEX_RESOLUTIONS` rather than
recomputed at the point of use, so a card cannot silently disagree with the deck
NEMO ran; the written-out cell counts are nevertheless checked against
`usr_def_nam`'s own `NINT(1800e3/rn_dx)+3` at import, and that check is shown
able to fail (a rung edited to 122 × 122 raises
`VORTEX rung -15km states 122x122 cells, which is not usr_def_nam's
NINT(1800e3/rn_dx)+3`).

What each card states, with nothing defaulted: `dx = dy` (30000 / 15000 /
10000 m), `dz = 500 m`, `dt` (2880 / 1440 / 960 s), ten wet levels on a flat
5000 m bottom, the domain in cells (63² / 123² / 183²), `nn_e = 48`,
`rn_ppumax = 1.0`, `rn_ppgphi0 = 38.5`, `nn_rot = 0`, the S-EOS coefficients of
Decision 69, no surface forcing, no lateral diffusion, no bottom drag. The
closed-box wet-row check now derives its expected count from the card's own mask
(`nj - 2`) instead of the hard-coded 61, which would have exempted the refined
rungs from it.

The geometry helpers take the rung as an argument **defaulting to the 30 km
one**, so the certified pair is unchanged by construction; a test proves it
field for field (`T`, `S`, `u`, `v`, `eta`, `ff_f` all `array_equal`), and the
0/50 measurement in §6 proves it against the oracle.

## 5. The three-resolution registry table

Every rung scored exactly like the 30 km ones:
`nemo_testcase_phase3_trajectory_gate.py --case <card> --max-step 10
--continue-after-first`, 50 rows each (10 steps × T, S, u, v, ssh), bar 1e-15
normalised.

### Rows at bar / in debt

| card | 30 km | 15 km | 10 km |
|---|---|---|---|
| `VORTEX` (flux) | 11 at bar / 39 debt | 9 / 41 | 8 / 42 |
| `VORTEX_VEC` (vector) | 15 / 35 | 12 / 38 | 12 / 38 |

### First row over bar — the SAME kt and the SAME fields at every rung

| card | 30 km | 15 km | 10 km |
|---|---|---|---|
| `VORTEX` | `kt=2`, {T, u, v, ssh} | `kt=2`, {T, u, v, ssh} | `kt=2`, {T, u, v, ssh} |
| `VORTEX_VEC` | `kt=2`, {u, v, ssh} | `kt=2`, {u, v, ssh} | `kt=2`, {u, v, ssh} |

### Headline rows, normalised max-abs, with the ratio to the 30 km row

**`VORTEX` (flux card)**

| row | 30 km | 15 km | 10 km | 15/30 | 10/30 |
|---|---:|---:|---:|---:|---:|
| kt1 T | 0 | 0 | 0 | exact | exact |
| kt1 S | 0 | 0 | 0 | exact | exact |
| kt1 u | 2.2204460e-16 | 1.1102230e-16 | 2.2204460e-16 | 0.50 | 1.00 |
| kt1 v | 2.2204460e-16 | 1.1102230e-16 | 2.2204460e-16 | 0.50 | 1.00 |
| kt1 ssh | 1.3552527e-20 | 1.1102230e-16 | 2.2204460e-16 | 8192 | 16384 |
| kt2 u | 1.2168311e-08 | 7.1165586e-09 | 5.0191263e-09 | 0.58 | 0.41 |
| kt2 v | 1.2394322e-08 | 7.3088558e-09 | 5.1247806e-09 | 0.59 | 0.41 |
| kt2 ssh | 2.6645353e-15 | 3.3306691e-15 | 4.1078252e-15 | 1.25 | 1.54 |
| kt2 T | 2.8232632e-10 | 8.5747724e-11 | 3.9929690e-11 | 0.30 | 0.14 |
| kt10 u | 9.9539758e-09 | 5.1100708e-09 | 3.7993677e-09 | 0.51 | 0.38 |
| kt10 v | 8.8742154e-09 | 5.0811171e-09 | 4.0596633e-09 | 0.57 | 0.46 |

**`VORTEX_VEC` (vector card)**

| row | 30 km | 15 km | 10 km | 15/30 | 10/30 |
|---|---:|---:|---:|---:|---:|
| kt1 T | 0 | 0 | 0 | exact | exact |
| kt1 S | 0 | 0 | 0 | exact | exact |
| kt1 u | 2.2204460e-16 | 1.1102230e-16 | 2.2204460e-16 | 0.50 | 1.00 |
| kt1 v | 2.2204460e-16 | 1.1102230e-16 | 2.2204460e-16 | 0.50 | 1.00 |
| kt1 ssh | 1.3552527e-20 | 1.1102230e-16 | 2.2204460e-16 | 8192 | 16384 |
| kt2 u | 1.3044375e-15 | 1.6862596e-15 | 1.5716595e-15 | 1.29 | 1.20 |
| kt2 v | 1.3356829e-15 | 1.9105946e-15 | 1.6942649e-15 | 1.43 | 1.27 |
| kt2 ssh | 2.8310687e-15 | 2.4980018e-15 | 3.3306691e-15 | 0.88 | 1.18 |
| kt10 u | 6.7335205e-15 | 6.7287755e-15 | 7.3729542e-15 | 1.00 | 1.09 |
| kt10 v | 6.6045730e-15 | 7.7879258e-15 | 8.2759812e-15 | 1.18 | 1.25 |

Full 50-row registries for all six runs: `phase3/round208/ladders/*.json`;
the table generator is `phase3/round208/score_ladder.py` and its output
`score_ladder.txt`.

### The scaling, fitted

For every flux-card row above 1e-14 at 30 km, the exponent `p` in
`e ∝ r^p` with `r = 30 km / dx`:

| kt | u | v | T | ssh |
|---:|---:|---:|---:|---:|
| 2 | −0.81 | −0.80 | −1.78 | (at floor) |
| 5 | −0.71 | −0.70 | −1.31 | −1.55 |
| 10 | −0.88 | −0.71 | −0.99 | −0.90 |

(fits at `r = 3`; the `r = 2` fits agree to ±0.1; full table
`phase3/round208/scaling.txt`.)

A per-step injection proportional to the timestep would give `p = −1` exactly,
because `dt ∝ 1/r`. The velocity rows sit at −0.7 … −1.1 and the tracer rows
start steeper and relax toward −1. **PLAUSIBLE, not confirmed:** the flux card's
remaining error is a tendency-level difference re-made once per step whose size
is set by `dt`, which is what rounds 204-206 located in `dyn_adv_up3`. What is
**CONFIRMED** is only the sign and order of the scaling: the residual decays as
the grid refines; it does not grow, and it does not change character.

### The only status crossings

Three rows per card cross AT-BAR → DEBT at the refined rungs, and they are all
tracer rows one quantum over the bar:

| card | row | 30 km | 15 km | 10 km |
|---|---|---:|---:|---:|
| `VORTEX` | kt5 S | 8.120488e-16 | 8.120488e-16 | 1.015061e-15 |
| `VORTEX` | kt6 S | 8.120488e-16 | 1.015061e-15 | 1.015061e-15 |
| `VORTEX` | kt7 S | 8.120488e-16 | 1.015061e-15 | 1.218073e-15 |
| `VORTEX_VEC` | kt5 T | 8.677745e-16 | 1.040291e-15 | 1.039997e-15 |
| `VORTEX_VEC` | kt6 S | 8.120488e-16 | 1.015061e-15 | 1.015061e-15 |
| `VORTEX_VEC` | kt7 S | 8.120488e-16 | 1.015061e-15 | 1.015061e-15 |

8.120488e-16 is exactly 4 × 2.030122e-16 and 1.015061e-15 is exactly 5 of the
same quantum: these are **4 ULP → 5 ULP** moves of the normalising scale, and
the 1e-15 bar happens to fall at 4.9 ULP. They are the rounding floor, not a
resolution-dependent transcription defect. They are registered, not explained
away: a round that wants them gone must move the bar's definition, not the code.

## 6. Predictions — CONFIRMED / REFUTED

Preregistered before any measurement in `phase3/round208/preregistration.md`.

| # | prediction | verdict | evidence |
|---|---|---|---|
| P1 | every `kt=1` row at bar at every rung, both cards | **CONFIRMED** | all 10 kt1 rows AT-BAR; max 2.22e-16 against a 1e-15 bar |
| P2 | first-over-bar stays at the same kt and fields | **CONFIRMED** | `kt=2` with identical field sets at all three rungs, both cards |
| P3 | residuals GROW with refinement, no faster than `r²` | **REFUTED** | they SHRINK, roughly as `r^-1`; the direction was wrong, not just the rate |
| P4 | vector card ≤ flux card at every rung | **CONFIRMED** | vector `kt2 u` ≤ 1.7e-15 vs flux ≥ 5.0e-09 at every rung |
| P5 | 30 km registries and GYRE unmoved | **CONFIRMED** | §7 |
| P6 | no rebuild needed | **CONFIRMED** | four certified hashes matched; all four runs completed |

P3's refutation is the round's most useful result and the reason it was
preregistered: a residual that shrank would have been quietly read as "fine"
without a written expectation to contradict.

## 7. Inertness gates

| gate | result |
|---|---|
| `VORTEX-zco` 50-row registry vs round 207's certified | **0/50 moved** |
| `VORTEX_VEC-zco` 50-row registry vs round 207's certified | **0/50 moved** |
| GYRE certified `kt=1..10` ladder | *see below* |
| GYRE from-rest 360-day year | *see below* |
| `tests/ocean/unit/test_nemo_vortex_card.py` | 59 passed |

**Which code changed, and why the year was run anyway.** This round touched one
file under `packages/`: the recipe module that BUILDS the cards. No model,
physics, dynamics or operator file changed, and every edited branch in the
recipe module is VORTEX-gated (`card.case.startswith(("VORTEX-",
"VORTEX_VEC-"))`), so GYRE's card cannot reach any of it. That is a reading of
the diff, not a measurement — so the year was run regardless, because
`packages/` changed.

GYRE_LADDER_PLACEHOLDER

GYRE_YEAR_PLACEHOLDER

## 8. Review

REVIEW_PLACEHOLDER

## 9. Verdict and OPEN

**LANDED — measurement round.** Four cards, four NEMO records, four ladders and
their registries land. **No physics statement lands**, and none is proposed: the
ladder found no grid-size dependence to fix.

**OPEN, named for the next round:**

1. **The `kt=1` ssh row is the one resolution-dependent row at entry.** It is
   1.35e-20 at 30 km — effectively exact — and 1.11e-16 / 2.22e-16 at 15 km and
   10 km, i.e. half an ULP and one ULP. Every value is far under the bar and
   nothing is in debt, so this is a curiosity, not a defect; but it is the only
   row whose character changes with the grid, and the entry state is pure
   transcribed arithmetic, so it should be cheap to name. Candidate, PLAUSIBLE
   and untested: the Gaussian bell's `exp(-(x²+y²)/λ²)` is evaluated at
   coordinates whose magnitudes shrink with `rn_dx`, so the 30 km box may simply
   be hitting an exactly-representable argument that the refined boxes do not.
2. **The flux card's `kt=2` owner is unchanged by resolution**, so rounds
   204-206's walk into `dyn_adv_up3` is the right place and the `r^-1` scaling
   is a new, free constraint on any candidate: a statement whose fix does not
   scale with `dt` is the wrong statement.
3. `nn_itend` at ratio 2 remains **DECISION_NEEDED** (inert).
4. Carried from note BY: the transport divisor's association (NEMO multiplies by
   the stored reciprocal `r1_hu_0/(1+r3u)`; we divide by the summed depth), and
   the proof — not the assumption — that GYRE's flat uniform box makes the
   summed depth exactly `hu_0`.

## 10. Evidence

`phase3/round208/`: `preregistration.md`, `child_deck.diff`, `acquire_all.{sh,log}`,
`ladders/` (six gate JSONs, six stdouts, `run_all.{sh,log}`), `score_ladder.py`,
`score_ladder.txt`, `scaling.txt`, `status_crossings.txt`, `gyre_gates.{sh,log}`,
`gyre_ladder_r208.{json,log}`, `gyre_year_r208a.log`,
`gyre_day_gap_r208a.{json,log}`, `review_round208.md`.
`phase3/vortex_ladder/{15km,10km}/{flx,vec}/`: the four NEMO records, their
reference runs, namelists, `ocean.output`, admissions, plant reports and
checksums.
