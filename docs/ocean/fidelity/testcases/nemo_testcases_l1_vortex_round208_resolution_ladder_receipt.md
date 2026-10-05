# Round 208 / VORTEX round 21 — Decision 74's resolution ladder (30 / 15 / 10 km)

**Verdict: LANDED, measurement only.** The two VORTEX cards were measured at
three grid spacings. No physics statement is proposed and none lands here; what
lands is four new cards, four NEMO records, four ladders, their registries and
this receipt.

**Headline.** The transcription carries **no grid-size dependence**. Every
`kt=1` row is at bar at every rung; the first row over bar is `kt=2` on both
cards at every rung, with the same field set; and the flux card's remaining
error does not grow as the grid refines — it **shrinks**, with a fitted exponent
against `r = 30 km / dx` of −0.7 … −1.1 on the velocity rows (a pure `dt`
proportionality is −1) but **as steep as −1.8 on the early tracer and `ssh`
rows**, which is nearer `r^-2` and is NOT dt-proportional. "Shrinks roughly like
the timestep" is therefore true of the velocities only, and §5 reports the two
families separately. The vector card sits at the rounding floor at every rung,
so round 199's landing holds at 15 km and 10 km as it does at 30 km. All of this
refutes this round's own prediction P3, which expected growth.

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
| `&namagrif` sponge block | absent | `rn_sponge_tra = rn_sponge_dyn = 0.00768` | **unreachable** | **unreachable** | `1_namelist_cfg:103-108` |

The sponge is not a choice this round declined: none of the four certified
builds compiles `key_agrif` (their `cpp_*.fcm` resolves to
`key_qco key_vco_1d key_RK3`), so the `&namagrif` block is **structurally
unreachable** in every run here, and the acquisition also deletes the child deck
files from the run directory so nothing can read them by accident.

And every value the child leaves alone, which this ladder therefore also leaves
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
change no value. One of them **strengthens** the rule and was missed in the
first draft: the commented-out modified-Leapfrog timestep `!MLF rn_Dt` also goes
1440 → 480, i.e. divided by the same 3, so NEMO scales the time step by the
refinement ratio on both of its integrators.

### The one DECISION_NEEDED, and why the round proceeded

`nn_itend` is the only value the child changes that has no ratio-2 analogue:
6000 is **not** 3 × 3000, so no rule can be read off it. It is registered here
as **DECISION_NEEDED** rather than guessed.

It is also **inert for everything this round measures**: the acquisition deck
pins `nn_itend = 10` at every resolution, exactly as the certified 30 km decks
do, and every ladder in this campaign is `kt = 1..10`. The cards' `n_steps`
metadata carries NEMO's own cited value where one exists — 3000 at 30 km
(parent) and 6000 at 10 km (`1_namelist_cfg:34`) — and the parent's 3000 at
15 km, where nothing is cited. **No gate in this campaign reads that field** —
verified by grep rather than asserted: `NEMOTestcaseCard.n_steps` has writers
and no readers. The reviewer's preferred shape (carry `None` at 15 km so a
future reader fails loudly) is right and is deliberately NOT done in a
measurement round, because `n_steps` is typed `int` and changing that is a
production-code change this round has no mandate for. It is carried as an OPEN
item.

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
`usr_def_nam`'s own `NINT(1800e3/rn_dx)+3` at import — transcribed as
half-away-from-zero, which is Fortran's `NINT` and not Python's `round`
(half-to-even); no shipped rung reaches a `.5` case, but a future one would
resolve to a different box under the wrong rule. The check is shown able to
fail by a test that calls it with a poisoned rung (a 20 km rung declaring
63 × 63 raises
`... which is not usr_def_nam's NINT(1800e3/rn_dx)+3 = 93x93`), and with the
`raise` deleted that test fails `DID NOT RAISE`.

What each card states, with nothing defaulted: `dx = dy` (30000 / 15000 /
10000 m), `dz = 500 m`, `dt` (2880 / 1440 / 960 s), ten wet levels on a flat
5000 m bottom, the domain in cells (63² / 123² / 183²), `nn_e = 48`,
`rn_ppumax = 1.0`, `rn_ppgphi0 = 38.5`, `nn_rot = 0`, the S-EOS coefficients of
Decision 69, no surface forcing, no lateral diffusion, no bottom drag. The
closed-box wet-row check now derives its expected count from the card's own mask
(`nj - 2`) instead of the hard-coded 61, which would have exempted the refined
rungs from it.

The geometry helpers take the rung as an argument **defaulting to the 30 km
one**, so the certified pair is unchanged by construction. A test checks the
default argument against the explicit `"30km"` rung field for field (`T`, `S`,
`u`, `v`, `eta`, `ff_f` all `array_equal`) — but **that test compares new code
with new code and cannot see a change to the 30 km arithmetic**. The real proof
is the 0/50 measurement in §7, against a registry produced before this round,
backed by reading every substitution in the diff: each is `_VORTEX_NI → res.ni`
(the same `int`) or `_VORTEX_DX_M → res.dx_m` (the same `float` object, passed
unconverted), with operand order untouched.

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

Full 50-row registries for all six runs: `phase3/round208/ladders/*.json`. The
table, the exponent fit, the status crossings and the inertness control are all
produced by one COMMITTED scorer,
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex_round208_resolution_ladder.py`
(direct test `tests/ocean/fidelity/test_nemo_testcase_l1_vortex_round208_ladder.py`),
whose full output is `phase3/round208/resolution_ladder.txt`.

### The scaling, fitted

For every flux-card row above 1e-14 at 30 km, the exponent `p` in
`e ∝ r^p` with `r = 30 km / dx`:

| kt | u | v | T | ssh |
|---:|---:|---:|---:|---:|
| 2 | −0.81 | −0.80 | −1.78 | (at floor) |
| 5 | −0.71 | −0.70 | −1.31 | −1.55 |
| 10 | −0.88 | −0.71 | −0.99 | −0.90 |

(fits at `r = 3`; full table, both fits per row, in
`phase3/round208/resolution_ladder.txt`.)

**The two fits per row do NOT agree to a tenth**, and an earlier draft of this
receipt said they did: across the 35 fitted rows the largest
`|p(r=2) − p(r=3)|` is **0.26** (`kt9 v`), with six more rows at 0.16-0.17. The
scorer now prints that spread so it cannot be eyeballed again. Three points is
thin for a power law and the spread says so.

A per-step injection proportional to the timestep would give `p = −1` exactly,
because `dt ∝ 1/r`. **Two families, not one:** the velocity rows sit at
−0.7 … −1.1 at every `kt`, consistent with `dt`; the early tracer and `ssh` rows
are much steeper (`kt2 T` −1.78, `kt3 ssh` −1.74, `kt4 ssh` −1.69), nearer
`r^-2`, and only relax toward −1 by `kt≈8`. **PLAUSIBLE, not confirmed:** the
velocity family is a tendency-level difference re-made once per step whose size
is set by `dt`, which is what rounds 204-206 located in `dyn_adv_up3`; the
steeper tracer family is **unexplained** and is carried as an OPEN item rather
than folded into the same sentence. What is **CONFIRMED** is only the sign and
order: every fitted residual decays as the grid refines, none grows, and no row
changes character.

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
| that comparison's own non-vacuity | the identical code reports **45/50** and **46/50** against the 15 km rung, and the scorer raises if the control finds no move |
| GYRE certified `kt=1..10` ladder | *see below* |
| GYRE from-rest 360-day year | *see below* |
| `tests/ocean/unit/test_nemo_vortex_card.py` + `tests/ocean/fidelity/test_fidelity_card_constructibility.py` | **94 passed** |
| receipt citation gate | **PASS**, 274 citations, 0 unmapped, 0 failing, 0 map entries failing audit; its own can-fail test green |
| `tests/ocean/fidelity/test_nemo_testcase_l1_vortex_round208_ladder.py` (the scorer's direct test) | **5 passed** |

**One re-anchor.** The rungs inserted about 110 lines into the recipe module,
which moved three `CITATION_MAP` entries and the one prose span that cites
them off their symbols. They were re-anchored BY SYMBOL from the gate's own
audit, once, in the same commit — `:2303 → :2413`, `:2306 → :2416`, and the
last endpoint of `370,615,2287 → 2397` — as the citation re-anchor rule
requires. No NEMO source was touched by this round, so no NEMO citation moved.

**Which code changed, and why the year was run anyway.** This round touched one
file under `packages/`: the recipe module that BUILDS the cards. No model,
physics, dynamics or operator file changed, and every edited branch in the
recipe module is VORTEX-gated (`card.case.startswith(("VORTEX-",
"VORTEX_VEC-"))`), so GYRE's card cannot reach any of it. That is a reading of
the diff, not a measurement — so the year was run regardless, because
`packages/` changed.

### GYRE certified `kt=1..10` ladder — UNCHANGED

The 50-row registry is identical to round 207's, row for row, and the stricter
cellwise comparison against the same NEMO oracle is clean. Both comparisons ran
offline from the two runs' residual sidecars; the plants are the proof the
comparison can fail.

```text
plant=None             exit=0  ORACLE_RELATIVE_COMPARE PASS: rows=954 max_worsening_ulps=0 first_over_bar={'T','S','u','v','ssh'} kt=3 -> unchanged
plant=worsen-3ulp      exit=1  ORACLE_RELATIVE_COMPARE FAIL: rows=954 max_worsening_ulps=3
plant=at-bar-to-debt   exit=1  ORACLE_RELATIVE_COMPARE FAIL: rows=954 max_worsening_ulps=0
plant=improve          exit=0  ORACLE_RELATIVE_COMPARE PASS: rows=954 max_worsening_ulps=0
```

**954 rows, 0 moved, 0 ULP**, `first_over_bar` `kt=3` on all five fields,
unchanged; the 50-row registry comparison independently reports **0/50 moved**.
The run's own worktree stamp records no dirty tracked path.


### GYRE from-rest 360-day year, scored against the note-BZ pin

| day | certified (note BZ) | this round | delta (K) | floor units |
|---:|---|---|---:|---:|
| 30 | `2.3432465132112266e-06` | `2.3432465132112266e-06` | 0 | 0.000 |
| 60 | `1.4793247973304582e-05` | `1.4793247973304582e-05` | 0 | 0.000 |
| 90 | `1.633271203963844e-05` | `1.633271203963844e-05` | 0 | 0.000 |
| 120 | `0.00010965907352116351` | `0.00010965907352116351` | 0 | 0.000 |
| 180 | `6.115335539300055e-05` | `6.115335539300055e-05` | 0 | 0.000 |
| 240 | `6.58170609494473e-05` | `6.58170609494473e-05` | 0 | 0.000 |
| 300 | `5.466049845187051e-05` | `5.466049845187051e-05` | 0 | 0.000 |
| 360 | `5.4077419367442036e-05` | `5.4077419367442036e-05` | 0 | 0.000 |

**BYTE-IDENTICAL on all eight certified days**, and the three certified
snapshot digests reproduce note BZ's to every character:

| day | note BZ's pin | this round |
|---|---|---|
| 030 | `4e36c106403b495e…` | `4e36c106403b495e…` |
| 240 | `8b9cd60475626373…` | `8b9cd60475626373…` |
| 360 | `e3e0a068346c7866…` | `e3e0a068346c7866…` |

So note BZ's certified GYRE year stands unchanged and is not re-pinned.


## 8. Review

One fresh `code-reviewer` subagent, no prior context on this round, told to try
to break seven named things (citations, the deck rule, hidden choices, the
scorer, the CONFIRMED/PLAUSIBLE split, test vacuity, and whether the 30 km
arithmetic survived the parameterisation). It was given a read-only brief and —
after it started a parallel `pytest -n 12` battery that this lane's
one-battery-at-a-time rule forbids and that was contending with the GYRE
certification job — was told to continue statically. That battery was killed;
the omission was in the brief, not in the review.

**Verdict: BLOCK**, on findings 2, 5 and 6. All three accepted and fixed in
commit `674f3b037`, before landing; the GYRE gates were then re-run from the
start on the fixed tree rather than carried over.

| # | finding | severity | disposition |
|---:|---|---|---|
| 1 | every citation verified line by line; the reviewer's own parent↔child diff reproduces `child_deck.diff`; one hunk MISSED that strengthens the rule (`!MLF rn_Dt` 1440→480, also /3) | OK | receipt now states it |
| 2 | "the `r = 2` fits agree to ±0.1" is **false by the round's own evidence** — 13 of 36 rows exceed 0.1, worst 0.26 | MAJOR | **fixed**: claim deleted, real spread (0.26) quoted, and the committed scorer now PRINTS it so it cannot be eyeballed again |
| 3 | the headline exponent range hides the tracer rows (`kt2 T` −1.78, `kt3 ssh` −1.74 — nearer `r^-2`, not dt-proportional) | MAJOR | **fixed**: headline and §5 now report two families and carry the steep one as an OPEN item |
| 4 | `scaling.txt` / `status_crossings.txt` had no committed generator | MAJOR | **fixed**: one committed scorer with its own direct test replaces all the inline probes |
| 5 | **the non-vacuity test was itself vacuous** — it built a bad rung and asserted only that two integers differ; deleting the import-time `raise` left it green | **BLOCKER** | **fixed**: the loop body is now `validate_vortex_resolution` and the test calls it with a poisoned rung; with the `raise` removed the test fails `DID NOT RAISE` (shown) |
| 6 | **the four new cards escaped every per-card gate** — neither `_ALL_NEMO_TESTCASE_CARDS` (the Decision 75 no-library-default check) nor the constructibility parametrize listed them | **BLOCKER** | **fixed**: both lists extended; the constructibility test's momentum branch now keys off the DECK, not an equality against `"VORTEX-zco"`, which would have sent a flux rung down the vector arm |
| 7 | `nn_itend` is labelled DECISION_NEEDED but a value ships (classic fix-behind-a-default shape); reviewer verified by grep that `n_steps` has writers and **no readers** | MAJOR, mitigated | accepted as correct; NOT done here because `n_steps` is typed `int` and retyping it is a production change a measurement round has no mandate for. Carried as OPEN |
| 8 | no-rebuild is sound and **stronger than claimed**: no certified build compiles `key_agrif`, so the sponge is structurally unreachable rather than a declined choice; each rung's own `ocean.output` confirms every resolved value; `cn_exp` matches the restart pattern | OK | receipt strengthened |
| 9 | "0/50 moved" is genuinely non-vacuous: distinct files, differing `legoesm_git_sha` (so the 30 km rung really was re-run on this tip), key-set equality asserted before diffing, exact `!=` on value AND status; all §5 numbers reproduce | OK | — |
| 10 | the 30 km-equality test compares new code with new code and cannot see a 30 km arithmetic change; §4 overstated it | MINOR | **fixed**: §4 now says the real proof is the 0/50 measurement plus reading every substitution |
| 11 | Python `round()` is half-to-even, Fortran `NINT` is half-away-from-zero | MINOR | **fixed**: `_vortex_nint` transcribes Fortran's rule, with its own test |
| 12 | 30 km arithmetic bit-identical by inspection — every substitution is the same `int`/`float` object, operand order untouched | OK | — |
| 13 | "59 passed" unverified (no pytest allowed) | UNVERIFIED | re-run serially after the fixes: **94 passed** across the two card test files, plus **5 passed** for the new scorer test |

Two findings were the same shape as defects this campaign has shipped before —
a guard that cannot fail, and a new artefact that slips past the gate written to
catch exactly its failure mode — and neither would have been caught by the
author.


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
4. **The steep tracer/`ssh` scaling at early `kt`** (`p ≈ −1.7 … −1.8`, nearer
   `r^-2` than `r^-1`) is unexplained and is a second, separate family from the
   velocity rows. It may be the same statement seen through the tracer equation,
   or a second one.
5. `NEMOTestcaseCard.n_steps` should be `None` where no value is cited, so a
   future reader fails loudly instead of consuming the parent's run length. It
   has no reader today.
6. Carried from note BY: the transport divisor's association (NEMO multiplies by
   the stored reciprocal `r1_hu_0/(1+r3u)`; we divide by the summed depth), and
   the proof — not the assumption — that GYRE's flat uniform box makes the
   summed depth exactly `hu_0`.

## 10. Evidence

`phase3/round208/`: `preregistration.md`, `child_deck.diff`, `acquire_all.{sh,log}`,
`ladders/` (six gate JSONs, six stdouts, `run_all.{sh,log}`),
`resolution_ladder.txt` (the committed scorer's full output), `gyre_gates.{sh,log}`,
`gyre_ladder_r208.{json,log}`, `gyre_year_r208a.log`,
`gyre_day_gap_r208a.{json,log}`, `review_round208.md`.
`phase3/vortex_ladder/{15km,10km}/{flx,vec}/`: the four NEMO records, their
reference runs, namelists, `ocean.output`, admissions, plant reports and
checksums.
