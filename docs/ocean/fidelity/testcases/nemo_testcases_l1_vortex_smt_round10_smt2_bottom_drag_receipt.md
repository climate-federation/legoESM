# VORTEX_SMT round 10 (lane round 222) — rung SMT-2: linear bottom drag

Decision 93's seamount mini-ladder, second rung.  One module of the deck moves
to ORCA2 rung 0's values: `&namdrg`.  **ROUND STATUS: the RUNG LANDS, the
STATEMENT the rung exposed is HELD** — its owner is named, cited and its
magnitude predicts the measured row to 1.5 %, but the exact transcription needs
an operand this round could not thread and measure under its gates.

Preregistration: `docs/ocean/fidelity/PREREG_nemo_testcases_l1_vortex_smt_round10_smt2_bottom_drag.md`
(committed before any SMT-2 measurement).  Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/round10/`.

## 1. The deck diff — one hunk, two lines, each cited

Against the SMT-1 deck (`namelist_cfg_smt1_vec_een.patch`), the SMT-2 deck
(`namelist_cfg_smt2_vec_een.patch`) differs in `&namdrg` and nowhere else:

```
@@ -111,7 +111,8 @@
 &namdrg        !   top/bottom drag coefficient                          (default: NO selection)
 !-----------------------------------------------------------------------
-   ln_drg_OFF  = .true.   !  free-slip       : Cd = 0                  (F => fill namdrg_bot
+   ln_drg_OFF  = .false.  !  SMT-2: rung 0 resolves .false. (namelist_ref:812; its cfg leaves it unset)
+   ln_lin      = .true.   !  SMT-2: ORCA2 rung-0 namelist_cfg:270 -- linear drag Cd = Cd0*Uc0
 /
```

ORCA2 rung 0 writes exactly ONE line in this module —
`orca2_rounds/round83/acquisition/orca2_rung0_restart_list_10step_a_np2/namelist_cfg:270`
`ln_lin = .true.` — and writes no `&namdrg_bot` block at all, so every
companion resolves from the reference namelist.  This deck does the same and
states the two values the cfg must carry (the shipped VORTEX deck's explicit
`ln_drg_OFF = .true.` has to be withdrawn or `zdfdrg` would see two selections).

**CORRECTION to the round-222 brief.**  The brief cited `rn_Cd0 = 1.e-3 :823`,
`rn_Uc0 = 0.4 :824` and `ln_boost = .false. :828`.  Those three lines are
`&namdrg_top`.  `drg_init` reads `namdrg_bot` when `cd_topbot == 'BOTTOM'`
(`zdfdrg.f90` drg_init, the `CASE( 'BOTTOM' )` arm sets `cl_namdrg =
'namdrg_bot'`), so the block this rung resolves is `namelist_ref:832-840`:
`rn_Cd0 = 1.e-3` **:834**, `rn_Uc0 = 0.4` **:835**, `ln_boost = .false.`
**:839**.  Same values, different block; the citations below use the corrected
lines.  `&namdrg ln_drgimp = .true.` is `namelist_ref:817`, as the brief said.

| knob | SMT-1 | SMT-2 | where it comes from |
|---|---|---|---|
| `ln_drg_OFF` | `.true.` (deck) | `.false.` | `namelist_ref:812`; rung 0 leaves it unset |
| `ln_lin` | `.false.` | `.true.` | rung-0 `namelist_cfg:270` |
| `ln_non_lin` / `ln_loglayer` | `.false.` | `.false.` | `namelist_ref:814`, `:815` |
| `ln_drgimp` | `.true.` | `.true.` | `namelist_ref:817` |
| `rn_Cd0` | — | `1.e-3` | `namelist_ref:834` (`&namdrg_bot`) |
| `rn_Uc0` | — | `0.4` | `namelist_ref:835` |
| `ln_boost` | — | `.false.` | `namelist_ref:839` |

**The compiled run reads every one of them back** (`ocean.output:651-668` of the
record):

```
   Namelist namdrg : top/bottom friction choices
      free-slip       : Cd = 0                  ln_drg_OFF  =  F
      linear  drag    : Cd = Cd0                ln_lin      =  T
      implicit friction                         ln_drgimp   =  T
   Namelist namdrg_bot : set BOTTOM friction parameters
      drag coefficient                        rn_Cd0   =    1.0000000000000000E-003
      characteristic velocity (linear case)   rn_Uc0   =   0.40000000000000002       m/s
      set a regional boost of Cd0             ln_boost =  F
   ==>>>   linear BOTTOM friction (constant coef = Cd0*Uc0 =    4.0000000000000002E-004 )
```

**R10-P1 CONFIRMED** (one hunk, two lines).  **R10-P2 CONFIRMED** (every echoed
value, including the resolved `Cd0*Uc0 = 4.0e-4`).

## 2. NEMO: build, record, sanity

New build directories `VORTEX_SMT2_VEC_R8_OMIP_L1` (plain) and
`..._P3` (instrumented, round 9's recorder patches unchanged); cpp keys
unchanged (`key_vco_1d3d`, `key_qco`, `key_RK3`).  Round 9's builds untouched.
Build 279 s + 261 s plus two links of 10 s and 31 s; both runs `STOP 0`;
whole acquisition ≈ 12 min wall.  Record **ADMITTED**, with
`restart_byte_identical: true` — the instrumented and plain kt=10 restarts are
byte-identical, so the recorder is additions-only (**R10-P4 CONFIRMED**), and
the header plant fired.

Sanity, from the admitted record (`smt2_probe.json`), against SMT-1's own:

| kt | NaN | max abs ssh [m] | SMT-1 | max abs u [m/s] | SMT-1 |
|---|---|---|---|---|---|
| 1 | 0 | 0.916157370471783 | identical | 0.8722609616581942 | identical |
| 2 | 0 | 0.8867942411954015 | 0.8867942200253586 | 0.8841705024234918 | 0.8841850132337148 |
| 10 | 0 | 0.8002135129879558 | 0.799982099472118 | 1.01127897443355 | 1.0112669258982572 |

**R10-P3 PARTIALLY REFUTED, and the refutation is reported as such.**  No
non-finite field at any step and both maxima well inside 2x, but the drag does
NOT reduce the field maxima at this horizon: at kt=10 `max|u|` is 1.2e-5 m/s
LARGER than SMT-1's, not smaller.  The prediction was wrong because the maxima
live in the surface vortex core while the linear drag acts on the deepest wet
level alone; a bottom sink has no reason to lower a surface extremum over ten
steps.  Over 100 days it may; that comparison is preregistered below, not run.

**Where the drag acts** (`smt2_probe.json`, from the card's own resolved
geometry, which the geometry gate pins to NEMO's at 0 ULP): 3721 wet columns,
3660 wet U faces and 3660 wet V faces.  NEMO puts the coefficient on
`iku = mbku(ji,jj) = MIN(mbkt(ji,jj), mbkt(ji+1,jj))` and on no other level.
The bottom level is NOT uniform over the seamount — **R10-P5 CONFIRMED**:

| bottom level (0-based) | T columns | U faces | V faces |
|---|---|---|---|
| 7 | 5 | 8 | 8 |
| 8 | 56 | 62 | 62 |
| 9 | 3660 | 3590 | 3590 |

The level index understates the spread: because these are z PARTIAL steps, the
bottom-cell THICKNESS varies continuously even within level 9.  That is what
section 5 turns out to be about.

**NEMO's own signature of the module** (SMT-2 minus SMT-1, same record format,
max abs per field): kt=2 `u` 7.163e-04, `v` 5.184e-04, `ssh` 5.862e-07, `T`
8.378e-10; kt=10 `u` 3.903e-03, `v` 3.719e-03, `ssh` 2.314e-04, `T` 2.262e-03.

## 3. The legoESM card

`VORTEX_SMT2_VEC-zps`, vector deck only (ORCA2 is vector-invariant, so the
flux-form seamount card is not carried up the ladder).  It is the SMT-1 card
plus the bottom-drag block and nothing else — a field-by-field comparison of
the two resolved configurations is a committed test, and the moved set is
exactly `{bottom_drag, zdf_drag_in_matrix, barotropic_drag_substep}`.

Every value is stated, none inherited:

* `bottom_drag_scheme = "nemo_linear"` — NEMO's `np_lin`, the new branch
  transcribed this round (section 4).
* `bottom_drag_cd0 = 1.0e-3`, `bottom_drag_uc0 = 0.4` — `namelist_ref:834`, `:835`.
* `bottom_drag_bbl_thickness = 0`, `bottom_drag_bg_velocity = 0`,
  `bottom_drag_r = 0` — `ln_boost = .false.` (`namelist_ref:839`) and NEMO
  drags the bottom cell alone, never a K&E99 band.
* `bottom_drag_cdmax`, `bottom_drag_z0`, `bottom_drag_ke0` — stated at the
  reference values even though `np_lin` reads none of them, so the card records
  what the namelist resolved instead of letting a library default speak.
* `zdf_drag_in_matrix`, `zdf_baroclinic_only`, `barotropic_drag_substep` all
  true — `ln_drgimp = .true.` (`namelist_ref:817`) with `ln_dynspg_ts = .true.`
  is ONE composition in NEMO and is selected here as one, exactly as the GYRE
  card selects it.
* The EVD composition field every EVD card must state is inherited from the
  SMT-1 block verbatim: `evd_composition = "nemo_replace"` with `nu_conv = 0`
  (`nn_evdm = 0`, rung-0 `namelist_cfg:410`) and `rn_evd = 100`
  (`:411`) — the field round 221 gave no default, so a card that dropped it
  would raise.

The card validator refuses a SMT-2 card that drops the scheme, either
coefficient, the no-boost/bottom-cell-only trio, or any member of the implicit
composition; and — after the reviewer's R-3, which found the guard nested one level too
deep — it refuses ANY OTHER VORTEX card that acquires a drag law, scheme or
implicit composition, because every one of those decks resolves `ln_drg_OFF`
(`namelist_cfg:114`).  Thirteen card tests
plus six new ones: **19 passed**.

**Geometry identity: 18/18 rows EXACT, 0 ULP**, the round-220 row set including
`e3u_0`, `e3v_0`, `e3f_0`, `e3w_0` and the three masks; `GEOMETRY IDENTICAL`.
**Initial state: bit-identical to SMT-1's** (committed test over `T`, `S`, `u`,
`v`, `eta`, `uu_b`, `vv_b` and `h_partial`/`bottom_level`/`is_active`).

## 4. What was transcribed

`zdfdrg.f90`'s `np_lin` branch is a rate that never reads the velocity:
`drg_init` stores `pCd0 = rn_Cd0 * zmsk_boost` once (with `zmsk_boost = ssmask`
because `ln_boost` is false), `zdf_drg_lin` then writes
`pCdU(ji,jj) = - pCd0(ji,jj) * rn_Uc0`, and `l_zdfdrg = .FALSE.` so neither is
updated again for the whole run.  The shared bottom-drag law gains it as
`bottom_drag_scheme = "nemo_linear"`, returning the constant `cd0 * uc0` in the
same multiplication order NEMO uses (measured exactly equal to `1.0e-3 * 0.4`).

The reference velocity is a new configuration field with **no default inside
the law**: a caller that selects the linear scheme without stating `rn_Uc0` is
refused by name, rather than silently given one.  The guard that keeps the
legacy MOM6 rate out of NEMO's `ln_drgimp` diagonal listed the two
velocity-dependent laws by name and is widened to the three NEMO laws.

## 5. The ladder, the registry, and the owner walk

### 5a. SMT-2 against SMT-1, row by row (50 rows each)

**First over bar: kt=2, fields T/u/v/ssh — the SAME ROW as SMT-1, never
earlier.**  kt=1 is at the bar on all five rows (T/S exactly 0, u/v one quantum
2.220446e-16, ssh 1.355253e-20) and is bit-for-bit SMT-1's.

| kt | field | SMT-2 | SMT-1 | ratio |
|---|---|---|---|---|
| 2 | T | 3.657428e-12 | 3.618737e-12 | 1.011 |
| 2 | S | 6.090366e-16 | 4.060244e-16 | 1.500 (AT-BAR both) |
| 2 | **u** | **2.193091e-04** | 1.267539e-09 | 1.73e+05 |
| 2 | v | 2.693376e-05 | 3.058414e-10 | 8.81e+04 |
| 2 | ssh | 3.938355e-10 | 2.664535e-15 | 1.48e+05 |
| 3 | T | 8.387269e-07 | 2.394062e-11 | 3.50e+04 |
| 3 | u | 4.343767e-04 | 5.443096e-09 | 7.98e+04 |
| 10 | T | 2.334802e-05 | 1.484277e-08 | 1.57e+03 |
| 10 | u | 2.003889e-03 | 1.237110e-06 | 1.62e+03 |
| 10 | v | 8.059764e-04 | 3.128077e-07 | 2.58e+03 |
| 10 | ssh | 9.666538e-06 | 1.673112e-08 | 5.78e+02 |

43 of 50 rows move against SMT-1 and exactly one row LEAVES the bar (kt=4 S,
a uniform-salinity row carrying no signal: `rn_b0 = 0`, so S is a constant
field and its row is quantisation).  Every moved row is registered here; this
is a NEW CARD on a new deck, not a change to SMT-1's registry, whose rows are
re-measured unmoved in section 6.

### 5b. The owner, walked in NEMO's stage order

The rung's own module is where the row comes from, so the walk is the drag.
`rCdU_bot` is a time-constant scalar, so **R10-P7 was right that the
coefficient is not the owner**; the walk goes to its consumers.  The
discriminating measurement is WHICH LEVEL the kt=2 `u` residual lives on,
taken from the gate's own residual arrays with the card's own mask:

| | max abs residual, kt=2 u |
|---|---|
| at the bottom level (`mbku`) | 2.193091e-04 |
| at every level above it | 6.438978e-06 |

A factor 34 — the row is a BOTTOM-CELL statement, not the barotropic one
(the near-uniform 6.44e-6 over all ten levels IS the barotropic component, and
it is 34x smaller).  So the owner is `dynzdf.f90:306` (and its V twin `:473`,
and the barotropic re-add `:166`/`:168`), which read

```
zwd(ji,iku) = zwd(ji,iku) - zDt_2 *( rCdU_bot(ji+1,jj)+rCdU_bot(ji,jj) )
            / (e3u_3d(ji,jj,iku) *(1._wp+r3u(ji,jj,Kaa)*umask(ji,jj,iku)))
```

Next operand: the prefactor.  `zDt_2 = rDt*0.5` (`dynzdf.f90:116`) multiplies
the SUM of the two T-point rates, i.e. `rDt` times their AVERAGE, which is
exactly what legoESM applies — that operand is NEMO's and is not the owner.

Next operand: the DIVISOR.  **THE FIRST NON-BIT STATEMENT IS the bottom-cell
scale factor.**  NEMO divides by `e3u_3d(ji,jj,iku)`, its own U-point scale
factor, which over z partial steps is the SHALLOWER neighbour's thickness —
the min rule this card's geometry gate already pins to NEMO's at 0 ULP.
legoESM divides by `dz_u_open`, and `dz_u_open = interp_cell_to_uface(dz_cell)`
is, in that helper's own words, the "simple average of the two cells sharing
each lon face".

Measured by the committed probe
`nemo_testcase_l1_vortex_smt_round10_drag_divisor_walk.py`, from the card's
resolved geometry alone:

| | VORTEX_SMT2_VEC-zps | GYRE-zco |
|---|---|---|
| bottom U faces | 3660 | 580 |
| faces where the two divisors differ | **2484** | **0** |
| max abs divisor difference | 85.19 m | 0.0 |
| max relative divisor error | **1.340** | 0.0 |
| thinnest bottom cell, NEMO vs legoESM | 50.67 m vs 91.40 m | identical |
| worst predicted damping ratio lego/NEMO | **0.43278** | 1.0 |

**NON-VACUITY, by the measurement the campaign requires.**  The statement's own
arithmetic predicts a worst-case damping ratio of **0.43278**.  Independently,
the two ladders give the drag's realised effect on kt=2 `u` as
`lego(SMT2) - lego(SMT1)` against `NEMO(SMT2) - NEMO(SMT1)`, cell by cell over
the 272 bottom cells where NEMO's effect exceeds 1e-7: median ratio 0.99471,
p05 0.70130, **minimum 0.42618 at face (33, 16, 9)**.  The prediction evaluated
AT THAT SAME FACE is **0.43278**, which is also its minimum over all 3660
bottom faces — so the two extrema are PAIRED, not two unrelated minima:
predicted 0.43278 against measured 0.42618 at one cell, **1.5 %**.  Both sides
are emitted by the one committed probe (`--smt2-residuals/--smt1-residuals`),
which finds the row by name through the gate's own registry rather than by
index arithmetic, so the number in this paragraph is re-runnable.  The divisor is the owner, and the probe is not measuring a
coincidence: on GYRE-zco, whose bottom is flat, it predicts and finds EXACTLY
zero difference on all 580 bottom faces.

### 5c. Why the statement is HELD and not landed

NEMO's divisor is `e3u_0(iku) * (1 + r3u(Kaa)*umask(iku))` — the REFERENCE
min-rule face thickness times the live stretch factor at the AFTER time level.
legoESM carries both operands (the card's `e3u_0` is a 0-ULP geometry row, and
`nemo_qco_live_face_geometry_from_operands` already builds `e3u` from them),
but they are not threaded into the drag block, which sees only the T-point
`dz_cell`.  Substituting a min-rule average of the LIVE T thickness would move
the row by about three decades and still leave the O(`r3`) ~ 2e-4 relative
remainder, i.e. it is a partial transcription whose remainder would be
unmeasured.  Landing a partial transcription blind is what this lane's rule
12 exists to prevent, and this round's remaining budget could not thread the
exact operand AND re-run the eleven registries, the GYRE ladder and year, and
the DINO month gate behind it.  So the statement is held, with the exact fix
and its falsifier preregistered for the next round (section 9).

**R10-P8 holds for what DID land**: the rung's first-over-bar row is kt=2, the
same row as SMT-1's and never earlier, and no row of any OTHER card's registry
left the bar (section 6).

## 6. Inertness, and the gates

The round's model change is additive by construction: a new drag law reachable
only by naming it, a new configuration field, and the reference velocity
threaded into five existing call sites without changing any existing scheme's
result.  The registries are the measurement of that, not the argument for it.

**ELEVEN certified card registries: 0/50 ROWS MOVED EACH, TOTAL_MOVED_ROWS 0**,
every first-over-bar unchanged, scored against the lane tip's own baseline
(`phase3/round221/baseline`): VORTEX_SMT1_VEC-zps, VORTEX_SMT_VEC-zps,
VORTEX_SMT-zps, VORTEX-zco, VORTEX_VEC-zco, VORTEX-15km-zco,
VORTEX_VEC-15km-zco, VORTEX-10km-zco, VORTEX_VEC-10km-zco, LOCK_EXCHANGE-zco,
OVERFLOW-zps.

**GYRE certified ten-step ladder: 50 rows, 0 MOVED**, first over bar kt=3
(T/S/u/v/ssh) unchanged against the round-221 clean run.  GYRE selects
`ln_non_lin`, not `ln_lin`, so none of the new code executes on it — and the
divisor statement held in section 5c is additionally inert on it by arithmetic
(0 of 580 bottom faces differ).

**GYRE from-rest certified year: NOT RE-RUN.**  This round's model change is
additive only — a new drag law reachable only by naming it, a new
configuration field, and a widened allow-list — and the ten-step ladder at 0
moved rows is the measurement of that.  Per note BZ the year is required when
PRODUCTION changes; nothing a GYRE path executes changed here.  Stated rather
than silently skipped.  (A year run was launched and aborted on its own
git-SHA stamp when this round's reviewer fixes were committed mid-run; it is
not quoted as a result.)

**DINO from-rest month gate**: run by `land.sh` as part of the landing, since
the round touches `packages/`; its line is quoted with the push below.

**Focused tests: 28 passed** (`tests/ocean/unit/test_nemo_vortex_smt_card.py`
19, including the six new SMT-2 rows and the reviewer's R-3/R-4 additions;
`tests/ocean/fidelity/test_nemo_round221_evd.py` 9, the EVD composition this
card inherits).

**Citation gate: PASS**, 0 unmapped citations, 0 failures, 0 map entries
failing audit, after the committed diff-based re-anchor tool remapped the
eleven files this round moved lines in.

**ONE FRESH ADVERSARIAL REVIEWER: SHIP WITH FIXES.**  Codex is paused on this
account, so DUAL review is a STATED GAP, not a silent one.  Nine findings,
five REAL, all five taken, and three of them changed the artifact:

* **R-3 (taken, real defect).**  The guard forbidding a drag law on a deck that
  resolves `ln_drg_OFF` sat INSIDE the mini-ladder block, so it only ever
  reached rungs SMT-1 and SMT-2; every other VORTEX card could have taken
  `nemo_linear` with `bottom_drag_r = 0` and passed.  Hoisted one level, and a
  test now proves it fires on VORTEX_SMT_VEC-zps and VORTEX_VEC-zco.
* **R-5 (taken, and it was the right call).**  The measured ratios in section
  5b had NO COMMITTED PROBE behind them — a throwaway number carrying the
  section's whole non-vacuity argument.  The committed walk now reads the two
  ladders' own residual files and emits them.
* **R-6 (taken).**  Predicted-min against measured-min was an UNPAIRED
  comparison of two extrema.  The probe now reports the measured argmin cell
  and evaluates the prediction AT THAT CELL: (33, 16, 9), predicted 0.43278,
  measured 0.42618 — and that face is also where the prediction is minimal, so
  the pairing is real and the claim stands rather than being downgraded.
* **R-8 (taken).**  Two citation line numbers were wrong in the round whose
  headline is a citation correction: `zdfdrg.f90:259` is `SELECT CASE(ndrg)`,
  the assignment is `:262`; `dynzdf.f90:117` is `zDt_4`, `zDt_2` is `:116`.
  Both fixed and re-verified against the compiled source.
* **R-4 (taken).**  Three more mutation rows in the validator test.
* Reviewer verified sound and found NOT-REAL: the `np_lin` transcription
  including the masking and the multiplication order (it reproduced
  `1.0e-3*0.4 = 4.00000000000000019E-04` byte-equal to the run's echo); that
  `bottom_drag_uc0`'s default is unreachable outside the linear branch; the
  two-line deck diff and every namelist citation including this round's
  correction of the brief; and that the probe's face-index convention is a
  uniform relabelling that leaves every extremum correct.
* One reviewer note NOT acted on and recorded instead: the round-10 NEMO
  probe's `_bottom_level_map` docstring states the (i, i+1) convention over a
  u-line that computes (i-1, i).  It is a comment defect in a counting
  instrument whose histogram is convention-independent (both conventions give
  the same multiset of face bottom levels); it is named here rather than
  fixed in the same breath as the landing.

**THREE GATES WERE ALREADY RED ON THE LANE TIP** and are untouched (none reads
a file this round edits): `test_param_specs[shortwave_penetration.py]`,
`test_no_inline_physics_coeffs[land/restart.py]`,
`test_dispatch_hardening::test_no_dispatch_guard_removed`
(`land/slab_land.py::step_land`).

## 7. ORCA2 pointer

ORCA2 rung 0 is exactly this switch set: `&namdrg ln_lin = .true.`
(`namelist_cfg:270`) with no `&namdrg_bot` block, hence `rn_Cd0 = 1.e-3`
(`namelist_ref:834`), `rn_Uc0 = 0.4` (`:835`), `ln_boost = .false.` (`:839`),
`ln_drgimp = .true.` (`:817`) — the resolved rate is a constant
`rCdU_bot = -4.0e-4 m/s` on every wet column, never updated.  Three things its
lane should take from this round:

1. **`linear_implicit_bottom_drag` can come off the ORCA2 card's
   `unmeasured_features` tuple** once the card states the five values above:
   the law now exists (`bottom_drag_scheme="nemo_linear"`) and the implicit
   composition it needs (`zdf_drag_in_matrix` + `zdf_baroclinic_only` +
   `barotropic_drag_substep`) is the same one the GYRE card already runs.  The
   card must also state `bottom_drag_uc0`, which has no default in the law.
2. **ORCA2's bottom is partial-cell everywhere, so the held divisor statement
   is LIVE on it, and large.**  On this seamount it is a 134 % divisor error on
   2484 of 3660 bottom faces and a 2.3x under-damping at the worst one.  Before
   its lane reads anything into a rung-0 bottom-momentum residual, it should
   run this round's committed divisor probe against its own card and quote the
   face count — the number is free, it needs no run.
3. The owner walk's compiled lines, for the rung-0 walk to take directly:
   `zdfdrg.f90` `drg_init` / `zdf_drg_lin` (the constant), `dynzdf.f90:306`
   and `:473` (the implicit diagonal over partial cells), `dynzdf.f90:160-169`
   (the barotropic removal and bottom-stress re-add), and
   `dynspg_ts.f90:1245-1246` / `:1271-1272` (the external mode's own drag).
   On this rung the barotropic part is 34x smaller than the bottom-cell part;
   ORCA2 should not assume that ratio, because its `H` and its substep count
   differ.

## 8. Choices

**UNASKED list: EMPTY.**  Every value this rung resolves comes from ORCA2 rung
0's own namelist or from the reference namelist it leaves unwritten, and each
is cited above.  Nothing that rung 0 fails to pin was chosen here: the one
place where this deck must differ textually from rung 0's — withdrawing the
shipped VORTEX deck's explicit `ln_drg_OFF = .true.`, which `zdfdrg` would
reject beside `ln_lin` — resolves to the same value rung 0 resolves to
(`namelist_ref:812`), and it is written out rather than left implicit.

One ASKED-and-stated scope note: the three `np_lin` never reads
(`rn_Cdmax`, `rn_z0`, `rn_ke0`) are stated on the card at their reference
values.  That changes no number; it records the namelist.

## 9. OPEN — for the next round

1. **LAND THE DIVISOR STATEMENT.**  Thread NEMO's own
   `e3u_0(iku)*(1 + r3u(Kaa)*umask(iku))` (and the V twin) into the
   `zdf_drag_in_matrix` diagonal and the `zdf_baroclinic_only` RHS correction
   in place of `interp_cell_to_uface(dz_cell)`.  PREDICTION: SMT-2's kt=2 `u`
   row falls from 2.193091e-04 to below 1e-6, leaving the 6.44e-6-class
   barotropic component as the new leader; GYRE's registry stays 0/50 by
   arithmetic (0 of 580 bottom faces differ); OVERFLOW-zps is inert because it
   does not select `zdf_drag_in_matrix`; DINO is partial-cell and is the one
   card that must be MEASURED, not predicted.  FALSIFIER: the kt=2 `u` row does
   not fall by at least two decades, or any GYRE row moves, or the DINO month
   gate goes red — any of those and the statement goes back to held.
2. **The 100-day SMT-2 comparison** is preregistered and NOT run (the deck
   patch `namelist_cfg_smt2_vec_een_100d.patch` and the `smt2vec100d` variant
   are committed, reusing this round's binaries by hash).  PREDICTION: the
   day-100 T rms lands within 2x of SMT-1's; and, unlike the ten-step horizon
   where R10-P3 was refuted, `max|u|` at day 100 IS lower than SMT-1's, because
   a hundred days is long enough for a bottom sink to reach the core.
   FALSIFIER: either half wrong.
3. **The barotropic component** (6.44e-6, near-uniform over depth) is the
   second term and has not been walked.  Its candidates are
   `dynspg_ts.f90:1245-1246` and `:1271-1272`; the per-substep record the
   round-196 instrument writes would resolve it without new NEMO time.
4. **Rung SMT-3** (lateral tracer diffusion: lap/iso/msc, `nn_aht_ijk_t=20`)
   and **rung SMT-4** (lateral momentum diffusion) remain, per Decision 93.
5. The kt=4 S row that left the bar carries no signal (`rn_b0 = 0`, uniform
   salinity) but is registered rather than excused.
