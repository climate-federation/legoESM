# Grid ladder — G5 round 4 (gridladder_r34)

Status: **LANDED. G5 CERTIFIED.** Every field is bit-identical to NEMO under
both labels at kt 1..10: surface rows 68/68 and all-level rows 0 unequal
cells, INDEPENDENT and GIVEN-NEMO-ENTRY.

Preregistration `PREREG_nemo_testcases_gridladder_g5_round4.md` (`1d7dd78bd`,
before any measurement). Record `gridladder_rounds/round31/oracle_g5_r31`
(admitted round 2). Landing commit `6c1cebfd7`. Evidence
`gridladder_rounds/round34/` (`manifest.sha256` 1c2a6e0488f68d00):
`g5_stages_kt1.json` 871c934003a717de, `g5_independent_kt1_6.json`
dfaf60a62b1e64cb, `g5_carried_kt7.json` 193179af8c81b10b,
`g5_stages_kt5_given.json` 703fa3783883ede3, `g5_stages_kt9_given.json`
03e0767ed43e7500, `g5_stages_kt5_s2fix.json` 5ba419ff93456f4d,
`g5_stages_kt1_both.json` 2df435266712c892, `g5_stages_kt5_both.json`
00d092d334eb39f7, `g5_stages_kt9_both.json` 6d42d38f229b714e,
`g5_independent_kt1_10.json` 56fc2b4123c7ada3, `g5_given_kt1_10.json`
f32083764655eae8, `g5_independent_plant_score.json` e0e00d57d8dc3672,
`consumer_plants.log` 31946f0138f284e9. The two kt 1..10 ladders and the
plant are stamped clean at `6c1cebfd7` (packages tree `b357597fd558`); the
stage and carried arms ran on the working tree (`-dirty` stamps) and are
identified by their arm and option state below.

## 1. Where the INDEPENDENT chain first leaves NEMO (P1 CONFIRMED)

Not kt 7 ssh. The first leave at all is **kt 1, stage 2**: T 1 cell
(`[26,131]`, 1.110e-16) and S 2 cells (first `[6,44]`, 7.105e-15), level 1,
full cells; entry and stage 1 are bit-exact, stage 3 returns to exact. The
first step-final leave is kt 2 S (2 cells, 7.105e-15); independent S then
carries 2, 2, 3, 4, 5 cells (kt 2..6). The reviewer's kt 7 ssh (428 cells,
1.887e-15) reproduces exactly; it is the first leave of a DYNAMICS field.

## 2. Carried state at kt 7 (P2, P3 CONFIRMED)

The new `carried` arm diffs every leaf the card carries out of kt 6 (18
fields) against the state built from NEMO's kt-6 restart, then steps kt 7 from
swapped states.

| leaf | unequal | max abs |
|---|---:|---:|
| S | 5 (`[11,51]`, `[126,55]`, `[141,43]`, `[142,139]`, `[146,41]`, level 1) | 1.421e-14 |
| w (card diagnostic; restart has none) | 430552 | 3.235e-02 |
| T, u, v, eta, uu_b, vv_b, six barotropic histories, all others | 0 | 0 |

| kt 7 started from | ssh | uu_b / vv_b | u / v (all-level cells) | S |
|---|---|---|---|---|
| card's carried state | 428, 1.887e-15 | 490 / 500 | 9873 / 11084 | 5 |
| NEMO restart | 0 | 0 | 0 | 1 |
| carried, NEMO's S swapped in | 0 | 0 | 0 | 1 |
| NEMO restart, card's S swapped in | 428, 1.887e-15 | 490 / 500 | 9873 / 11084 | 5 |
| carried, all restart fields swapped in (card's w kept) | 0 | 0 | 0 | 1 |

The carried salinity alone owns the kt 7 leave; `w` is not read. So the drift
is a last-bit S leave produced inside single steps, which the restart erases
and the independent chain keeps. From NEMO's restart it appears at kt 5 in
stage 2 (T 1, S 1) and stage 3 (S 1), and at kt 9 in stage 3 only (S 3).

## 3. Walk in compiled order — two transcription defects

`stp_RK3_stg` stages 1-2 call `tra_sbc_RK3`, whose non-linear-ssh branch
divides by the stored Kmm thickness:
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/trasbc.f90:284`
(`z1_rho0_e3t = r1_rho0/(e3t_3d*(1+r3t(Kmm)*tmask))`, SUBROUTINE
`tra_sbc_RK3` at :220), applied at
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/trasbc.f90:285-286`;
the stage's runoff divisor is the same stored thickness
(`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/sbcrnf.f90:281`,
`sbc_rnf_div`). At stage 2, `r3t(Kmm)` is stage 1's
`r2_3*r3t(Kbb) + r1_3*r3ta`
(`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stprk3_stg.f90:177`, np_HYB).

1. **Stage 2 (first non-bit statement).** The card divided by the thickness
   interpolant `h(Kbb) + (h(Kaa) − h(Kbb))/3`, not NEMO's
   `e3t_0*(1+r3t(Kmm))`. G4 round 9 fixed the same thing at stage 3. Option
   `rk3_stage2_tracer_nemo_stored_thickness` uses the card's existing stage-1
   ratio (`nemo_r3t_rk3_stage1_stretch`, already the stage-2 time-step
   weight).
2. **Stage 3 salt.** NEMO adds `sfx*z1_rho0_e3t` to a cleared Krhs
   (`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/trasbc.f90:308`), then
   runoff. The card rescaled its Kbb salt-flux tendency,
   `(1e3*(1e-3*sfx))/(rho0*h(Kbb))*h(Kbb)/h(Kmm)`. G4 never tested this: its
   `sfx` was zero (round 9's registered gap). Option
   `rk3_stage3_salt_nemo_written` transcribes the statement. It reads NEMO's
   recorded `sfx` from a new optional forcing field `salt_flux_sfx`, because
   `1e3*(1e-3*sfx)` is not bitwise `sfx` (round 2 measured 474 cells).

| arm (stage-local, all levels) | kt 1 stage 2 | kt 5 stage 2 | kt 5 stage 3 | kt 9 stage 3 |
|---|---|---|---|---|
| both options off (control) | T 1, S 2 | T 1, S 1 | S 1 | S 3 |
| + stage-2 stored thickness | exact | exact | S 1 | S 3 |
| + stage-3 `sfx` statement (landed) | exact | exact | exact | exact |

With both options on, all 21 stage rows (7 fields × 3 stages) are bit-exact at
kt 1, 5 and 9. Turning either option off brings its cells back (control rows
above).

Selection: both options default False and only G5 selects them
(`build_orca2g1_flat_card`'s `ORCA2G5-ice` block; identity checks
"stage-2 Kmm e3t" and "stage-3 salt Krhs" assert G5-only). G1, G1b, G2, G3,
G4 and every other card are unchanged. Caller grep: both fields are set only
in `nemo_testcase_recipe.py`. Selecting the salt option without a recorded
`sfx` or without the stage-3 stored thickness raises.

## 4. Both labels, kt 1..10 (scored at `6c1cebfd7`)

| label | surface rows | not bit-identical | all-level rows | unequal cells |
|---|---:|---:|---:|---:|
| INDEPENDENT (kt 1..10) | 68 | 0 | 36 | 0 |
| GIVEN-NEMO-ENTRY (entries 1/3/5/7/9 + two-step) | 68 | 0 | 20 | 0 |

Round 3's debt (independent kt 9 U partial 1.25e-15 relative; kt 10 U/V) and
every last-bit S row (both labels) are gone. Prediction 5 CONFIRMED. The
instrument is not vacuous: `--plant score` exits 1 (kt 1 ssh DEBT, 1 cell).

**G5 is CERTIFIED**: every field is bit-identical under both labels at kt 1..10.

## 5. Decision 122 (ORCA2_OMIP_L4 build, each opened at its SUBROUTINE)

| statement | ORCA2_OMIP_L4 line | SUBROUTINE |
|---|---|---|
| stage-1/2 Kmm divisor | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:284` | `tra_sbc_RK3` (:220) |
| stage-1/2 EMP dilution | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:285-286` | `tra_sbc_RK3` |
| stage-3 salt flux | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:308` | `tra_sbc_RK3` |
| runoff `h_rnf = e3t(Kmm)` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:281` | `sbc_rnf_div` (:238) |
| stage-1 `r3t(Kaa)` (stage-2 Kmm) | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:177` | `stp_RK3_stg` (:78, np_HYB) |

The text and line numbers match the G5 build: `stp_RK3_stg` lines 78-250 are
identical. The full ORCA2 card does not select either option, so the effect
there is UNMEASURED.

## 6. Top-drag consumers — fast controls (review caveat)

New `tests/ocean/fidelity/test_nemo_testcase_g5_top_drag_consumers.py` runs
on the synthetic VORTEX card (no deck inputs) with the top-drag path selected.
The coefficient is a traced `jax.jit` argument. It holds 6 tests, 124 s
together, none marked slow:

- the coefficient map equals `r1_2*((0+0)+(r(i+1)+r(i)))` exactly on every
  interior U and V face;
- the coefficient reaches the step (nonzero vs zero `rCdU`);
- omitting the residual, the substeps or `dyn_zdf` each moves u/v/ssh, by
  3.7e-4, 1.4e-4 and 4.8e-3 m/s respectively;
- the salt option refuses a missing `sfx`.

Non-vacuity (`consumer_plants.log`): I planted each removal in production
code and its test failed. The three omission defaults were flipped to True one
at a time, and the coefficient's `0.5` became `0.25`. Each plant turned its
test red (4/4). Tree restored, checked with `git status --porcelain`.

## 7. Unmeasured labels

`ORCA2G5_UNMEASURED` is now `("nemo_post_si3_surface_records",)`.

- **Restored: `nemo_post_si3_surface_records`.** The card is fed NEMO's
  post-SI3 ocean input (D123 route), so SI3 itself never runs here and stays
  unmeasured.
- **Not restored: `g5_record_not_admitted`.** The record was admitted in
  round 2 (passivity PASS), so the label describes no unmeasured row.
  G1/G1b/G3/G4 still carry their `*_record_not_admitted` labels after
  admission; that is inconsistent, but out of scope.
- **Removed: `kt9_independent_all_level_u_continuation`** (closed by §4).

## Tests and review

- Focused G5 + G2-ladder (CPU fp64, `-m "not slow"`): 22 passed, 8 deselected.
- G1 tools 31 passed; G3 9 passed; G4 29 passed. The G1 ladder fast set
  (85 tests) did not finish inside the 580 s budget; G1's card is unchanged.
- New consumer file: 6 passed. Plants: 4/4 red.
- Private-import + dispatch: 29 passed, 1 failed. That failure is the known,
  pre-existing slab-land `step_land` guard.
- Citation gate: 17 passed. I reanchored 107 citations mechanically with the
  hunk-shift script, saved as evidence `reanchor.py`. One span grew from 174
  to 199 lines; its two pinned endpoint statements are unchanged.
- Single review (codex): see below.

ASKED: first-leave measurement; carried-state comparison; compiled-order walk;
the cited G5-selected landings; both labels kt 1..10; Decision-122 lines; the
UNMEASURED labels; fast per-consumer tests; the citation gate; one review.

UNASKED, offered for revert:
- the new optional forcing field `salt_flux_sfx`. It is needed to transcribe
  `sfx` bitwise and is inert unless the G5 option is on.
- the new `carried` arm in the ladder instrument. It has no direct test; it is
  exercised only by this round's record run.

UNVERIFIED: the full ORCA2 card's response (pointer only).
