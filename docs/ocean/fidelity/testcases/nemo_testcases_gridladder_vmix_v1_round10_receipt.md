# Vertical-mixing lane — V1 round 10 (vmix_r10)

## Correction (resume after the independent review's HOLD, 2026-10-17)

**Withdrawn:** "the (116,45) residue is a compiler last-bit effect, irreducible" (section 3, and the status
lines below). It was walked downstream of the real site.

**Owner (reviewer MEASURED, reproduced here):** the step-entry shear's sea-level ratio. The card forms
`r3v = num*r1_hv0/e1e2v` and feeds `e3w*(1 + r3v)`. NEMO stores `pr3u`/`pr3v` as arrays first
(`domqco.f90:266-269`) and only then adds 1 (`zdfsh2.f90:104-108`). Under compilation XLA rewrites the
division by the constant face area into a multiply by its reciprocal, then fuses that multiply with the
"1 +" into one FMA. That is the miss.

Reviewer's measurements:
- kt 2 with FMA disabled globally: every field bit-for-bit.
- Guarding only the `r3v` quotient: every field bit-for-bit, in both columns ((116,45) and (26,133)).
- Guarding every other product or quotient outside the vertical-coordinate module, or the `r3u` quotient
  alone: no change. Not libm, not subnormal flushing, not the Prandtl tail.
- On a scratch copy, `r3u` and `r3v` in the existing rounding guard: INDEPENDENT kt 1-3 bit-for-bit on every
  field (kt-3 sea level 189 cells -> 0); GIVEN-NEMO-ENTRY kt 1-3 exact.

Section 3's own finding stands as measured (the Prandtl tail equals NEMO when evaluated outside the step);
its conclusion does not. The (26,133) column was the same owner, not a separate one (section 5).

### Landed

- `066f5c712`: `TKEConfig.nemo_sh2_r3_source_rounded` (V1-selected, default False) rounds the step-entry
  `r3u`/`r3v` quotients before the "1 +". It refuses the in-solve shear and any other metric source.
- `1c5c3fe66`: map entries the guard shifted, re-anchored (+9 below it). `a28038597`: rounds 8-9 receipts
  mapped. This receipt's own entries land with this correction.
- The `066f5c712` message names `dom_qco_r3c`. The step-entry ratio at kt >= 2 is the previous step's
  stage-3 `dom_qco_r3c_RK3`; `dom_qco_r3c` builds only the kt-1 entry. Both have the same statement (table).

Evidence: `gridladder_vmix_rounds/round10b/` (`manifest.sha256`, 24 files).

| measurement | result |
|---|---|
| INDEPENDENT kt 1-3, guard on, before commit (`indep1_3_guard.log`) | 39 rows, 0 unequal cells |
| flag-off step jaxprs, ORCA2 and DINO, base `991cad2ba` vs `066f5c712` (`jaxpr_all.log`) | identical (138,747-line ORCA2) |
| new record test, V1 kt 1-2 every row and level | passes; flag off in the V1 recipe: red (kt-2 v, T, S; levels u, v, T, S, en, avm, avt, dissl) |
| the shear builder compiled alone (toy grid, and the real card at its kt-1 state) | guard on = guard off; the FMA forms only inside the full step |

So the unit test covers wiring only; the record test carries the claim.

PLAUSIBLE, not measured separately: NEMO multiplies by a stored reciprocal (`r1_e1e2u = 1/e1e2u`,
`domhgr.f90:170`), so the compiled card's reciprocal multiply, once stored, is NEMO's own form.

### Re-score at `066f5c712` (clean worktree), both labels kt 1..10

- **INDEPENDENT:** kt 1-3 bit-for-bit on every field and level. kt 4: surface exact; levels S/en/avt/dissl
  differ in 1-4 cells, at the bar. **First over the bar: kt-6 T, 3.64e-14 in 2 cells (1.21e-15
  normalised)**, with level rows T/avm/avt/dissl (6-14 cells). Worst row: kt-10 u, 1.65e-14 normalised.
  18 rows over the bar, all kt 6-10.
- **GIVEN-NEMO-ENTRY (kt 1/3/5/7/9):** every one-step row at the bar; kt-2 and kt-4 two-step now exact
  (round 10 had v/T/S at kt 2, u at kt 4). **Only row over the bar: two-step kt-6 T, 3.64e-14 in 2 cells
  (1.21e-15 normalised)** — the round-10 debt, unchanged.

**V1 is NOT certified.** Next over-bar row, both labels: kt-6 T (2 cells), with avm/avt/dissl level rows.
Its first trace is the kt-4 level residue (S/en/avt/dissl, 1-4 cells), which both labels show.

| statement | ORCA2_OMIP_L4 line | SUBROUTINE |
|---|---|---|
| `pr3u`/`pr3v` stored (stage 3, the next step's entry) | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/domqco.f90:266-269` | `dom_qco_r3c_RK3` (:237), called at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:167` in `stp_RK3_stg` (:78) |
| `pr3u`/`pr3v` stored (kt-1 entry) | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/domqco.f90:214-217` | `dom_qco_r3c` (:188), called at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/domqco.f90:177` in `dom_qco_zgr` (:160) |
| `e3w_1d*(1+r3u)` and `e3w_1d*(1+r3v)` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfsh2.f90:98-108` | `zdf_sh2` (:56) |
| `r1_e1e2u = 1/e1e2u` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/domhgr.f90:170` | `dom_hgr` (:58) |

V1 build (`ORCA2V1_OMIP_L4_R1FULLENTRY`) and `ORCA2_OMIP_L4`: `domqco.f90` (`9a3b302cd3f8868f`), `zdfsh2.f90`
(`e8bb2398700d0d3d`) and `stprk3_stg.f90` byte-identical. The full ORCA2 and DINO cards run this statement
too; it stays off there (D130), and their flag-off programs are unchanged.

Choices made this resume — ASKED (brief): land the guard on both quotients, V1-selected; correct this
receipt; re-score; map rounds 8-10. UNASKED, offered for revert:
- the guard test is a record test (2.5 min, skips without the V1 record), because no isolated compile shows
  the FMA; the unit file keeps only the default and the two refusals;
- the flag refuses non-`nemo_qco_live_face` metric sources and the in-solve shear (round-9 pattern);
- evidence for this resume is under `round10b/`, not `round10/`;
- in the round-8 and round-10 receipts, the bare `stprk3_stg.f90`/`trazdf.f90` citations are now written with
  the V1 build path (the bare names resolve to a GYRE build), and round 8's in-solve shear citation in the TKE module moved from line 3368 to 3385.

UNVERIFIED: why INDEPENDENT kt 5 has more unequal surface-velocity cells than GIVEN kt 5 (3/4 vs 2/2) when
kt 4 is bit-for-bit (a carried field outside the restart, PLAUSIBLE); the kt-4 level residue's owner.

---

Status: **HELD. V1 is not certified. Nothing landed in model code.**
- The kt-3 attribution is now MEASURED by a committed probe. kt-3 sea level comes from the kt-2 T/S residue.
- The kt-2 T/S residue comes entirely from the kt-2 diffusivity (`avt`), and only that.
- The diffusivity residue at (116,45) is classified as **last-bit association from compilation (PLAUSIBLE)**, not a
  transcription defect. The same source statements give NEMO's bits whenever they are evaluated outside the production
  fusion. Three NEMO-form candidates moved nothing, so none landed.

Decisions in force: D129 = 1, D130 = 1. Codex review: paused (quota).

Preregistration: `PREREG_nemo_testcases_gridladder_vmix_v1_round10.md`, frozen at `4af3fc41e` before any measurement.

Commits:
- `4af3fc41e` preregistration;
- `592c9b60e` committed probe `v1_substitution_probe.py` and its direct test;
- this receipt.

Record: `gridladder_vmix_rounds/round1/oracle_v1_r1` (admitted round 3). Evidence: `gridladder_vmix_rounds/round10/`
(`manifest.sha256`, 40 files).

| file | sha256 prefix |
|---|---|
| `v1_substitution_probe.py` (= committed `592c9b60e`) | dd2dcf0f4f84aa02 |
| `entry_kt3.json` / `closure_kt2.json` | 2e9f586c9d475c12 / 7c30c6f374228f17 |
| `replay_avt_116_45.log` / `avt_tail_jit.log` / `n2_bundle_kt2.log` | dc2adf27249eb290 / 5d69927afca31229 / be5555a149e0b98d |
| `scratch_tail_kt2.npz` / `operand_ulp_search.log` / `candidates_not_landed.diff` | 9410c6a0a0fb610a / b3323deeceb29ccd / 2ad23badc82d7652 |
| `indep1_10.log` / `given_all.log` | 323c41f5fdd79439 / 6cafb7ea16d0cd55 |

The probe ran from the working tree at `4af3fc41e`, before it was committed. The committed file is identical to the
evidence copy (same SHA-256).

## 1. Committed substitution probe (P1 CONFIRMED)

The card's own chain states after kt 1 and kt 2 were regenerated at HEAD. The kt-2 rows reproduce round 9 exactly
(T 5, S 4, en 2, avm 1, avt 3, dissl 2 cells). Each arm is a fresh model.

Layout check: NEMO's kt-2 entry fields differ from the card's in exactly the ladder's residue cells:
T 5, S 4, en 2, avm 1, avt 3, dissl 2, u 1, v 2; sea level, barotropic transports and the six histories 0.

ENTRY substitution at kt 3 (unequal cells vs NEMO's kt-3 restart; sea level normalised by max|ssh|):

| arm | ssh | T | S | u | avt_k |
|---|---|---|---|---|---|
| plain | 189 / 1.29e-15 | 13 | 6 | 5,557 | 12 |
| self (identity control) | bit-identical to plain | | | | |
| NEMO's kt-2 T/S | **0** | 3 | 0 | 34 | 3 |
| NEMO's kt-2 TKE fields | 189 / 1.29e-15 | 11 | 4 | 5,556 | 12 |
| NEMO's full kt-2 state | **0** | **0** | **0** | **0** | **0** |

This reproduces the round-9 reviewer's scratch result. The probe is now committed with a direct test. The test fails when
the unequal-cell count is planted wrong (`1 failed, 2 passed`).

## 2. The kt-2 T/S residue is the diffusivity's (P2 CONFIRMED)

CLOSURE substitution at kt 2 (NEMO's kt-2 `avm_k`/`avt_k` swapped in where the solves read the closure's output):

| arm | u | v | T | S |
|---|---|---|---|---|
| plain | 1 | 2 | 5 | 4 |
| self (the card's own viscosity: identity control) | bit-identical to plain | | | |
| NEMO's `avm_k` | 0 | 0 | 5 | 4 |
| NEMO's `avt_k` | 1 | 2 | **0** | **0** |
| both | 0 | 0 | 0 | 0 |

So `tra_zdf` (`ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/ppsrc/nemo/trazdf.f90:67`, `tra_zdf_imp` :134, `avt` read at :185-194) is bit-for-bit on V1 at kt 2. The T/S
residue is inherited from the diffusivity alone.

The diffusivity residue sits in two separate columns:
- **(116,45), NEMO w-levels 3-4.** Energy and viscosity are exact here, so P3 is **FALSIFIED** for this column. It owns
  all 4 S cells, 4 of the 5 T cells, and the kt-3 sea-level maximum.
- **(26,133), w-level 2.** Energy (2 cells), viscosity and dissipation also differ here. Not walked this round.

## 3. Walk at (116,45): the source is NEMO's; the compiled copy is not

| measurement | result |
|---|---|
| numpy replay of NEMO's `zri`/`pdlr`/`avt` (`zdftke.f90:390-405`, `:709-717`) from the card's passive kt-2 trace | equals NEMO at both cells; differs from the card's published value |
| the card's own Prandtl tail (`_prandtl_number` + `K_H`) under `jax.jit`, fed the traced operands | equals NEMO at all 4 levels |
| the card's in-model `rn2` vs `rn2b` (step-entry bundle, jit) | identical; equal to the traced N2 in every cell |
| in-model capture of the tail's `K_H` as an extra output (scratch hook, step passive) | `K_H * wmask` equals NEMO's `avt_k` in **every** cell (0 unequal); the published `avt` in the same build differs in exactly the 3 residue cells |
| plant `K_H * 2` on that statement | published `avt` moves in 398,565 + 15,554 cells, so this statement does produce it |

Three NEMO-form candidates were tried in the model. All three left every kt-1 and kt-2 row bit-identical to before.
None landed (`candidates_not_landed.diff`; tree verified clean with `git status --porcelain`):
1. `avt = pdlr*avt` (`:717`) instead of `K_M/(1/pdlr)` (a `+raise` plant proved the branch runs on V1);
2. 1 plus each statement materialized (`zri` :401, `p_pdlr` :405, `avt` :717);
3. 2 plus `p_sh2` materialized before `zdiv = p_sh2 + rn_bshear` (`:397`);
4. (scratch) the Prandtl step fed the same shear array the RHS reads: no change.

Offline ulp search: with NEMO's chain, a +1-ulp shear or a -1-ulp N2 operand reproduces the card's published value at
both cells. A viscosity change is excluded, because the published viscosity is exact there.

**Classification (PLAUSIBLE; WITHDRAWN, see Correction):** last-bit association introduced when XLA compiles the production step. One Prandtl
operand reaches that statement one ulp off, in a fused copy that cannot be observed without changing the compile. It is
not a transcription difference: the source, evaluated outside that fusion, equals NEMO in every cell. Under the brief,
nothing lands.

**What would make this false:** an in-model operand dump that does not perturb the compile, showing `p_sh2` and `rn2b`
equal to the traced ones at (116,45). No such instrument exists. Every observer tried here (trace, extra output)
re-fuses the program.

## 4. Both labels, kt 1..10 (re-scored at HEAD)

The ladders ran on HEAD's package code (no package change this round). Every row is identical to round 9's
`indep1_5.log` + `indep6_10.log` and `given.log` (line-by-line diff empty).
- INDEPENDENT: kt 1 exact. kt 2 at the bar on every field (worst S 5.25e-16, 4 cells).
- INDEPENDENT, first over the bar: kt-3 sea level, 189 cells, 2.39e-15 m absolute = 1.29e-15 normalised; kt-3 T
  6.46e-15 (13 cells) and `avt_k` 8.36e-14 (12 cells). kt 4..10: as round 9's section-4 table.
- GIVEN-NEMO-ENTRY (kt 1/3/5/7/9): unchanged from round 9. The only DEBT is two-step kt 6: T 3.64e-14 in 2 cells, plus
  the level rows T/avm/avt/dissl (6-14 cells).

**P5 FALSIFIED: V1 is NOT certified.**

## 5. Next owner (registered, not landed)

- (116,45): compile association (section 3). Untested remedy, in the pattern rounds 6 and 9 landed for N2 and the TKE
  matrix: materialize every operation of the step-entry shear builder (`zdf_sh2`, `zdfsh2.f90:98-113`). That is a
  choice, so it is put to the user (DECISION_NEEDED).
- (26,133): en/avm/avt/dissl at w-level 2, the 2 TKE energy cells round 9 named. Next: a cell-level replay through the
  Langmuir source, the penetration increment and the mixing length.

## 6. Decision 122 (ORCA2_OMIP_L4)

`zdftke.f90` (`0f290190a33e4713`) and `trazdf.f90` (`029056166047354c`) are byte-identical between the V1 build
(`ORCA2V1_OMIP_L4_R1FULLENTRY`) and `ORCA2_OMIP_L4`.

| statement | ORCA2_OMIP_L4 line | SUBROUTINE |
|---|---|---|
| `zdiv = p_sh2 + rn_bshear`; `zri = rn2b*p_avm/zdiv` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:397-401` | `tke_tke` (:199) |
| `p_pdlr = MAX(0.1, ri_cri/MAX(ri_cri, zri))` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:405` | `tke_tke` (:199) |
| `p_avt = MAX(zav, avtb_2d*avtb)*wmask` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:710` | `tke_avn` (:531) |
| `p_avt = MAX(p_pdlr*p_avt, avtb_2d*avtb)*wmask` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:717` | `tke_avn` (:531) |
| implicit tracer solve reads `avt` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:185-194` | `tra_zdf_imp` (:134), called from `tra_zdf` (:67) |

Pointer only: the full ORCA2 card was not measured this round.

## 7. Selection, purity, tests

- No package, recipe or config change landed. Every card's program is unchanged; the flag-off jaxpr question does not
  arise.
- New test `test_nemo_testcase_v1_substitution_probe.py` (3 tests): layout cut, unequal counter, arm guard. Plant:
  counting `d > 0.1` instead of `d != 0` gives 1 failed.
- Runs, all synchronous, CPU, fp64: listed in section 8.

## 8. Test runs

Clean tree at `592c9b60e`, JAX_ENABLE_X64=1, `land.sh` PYTHONPATH: citation gate, private cross-imports, the new probe
test and the V1 card test: **40 passed**. No package code changed, so the TKE/recipe suites were not re-run.

## Choices made

ASKED (this round's brief): commit the attribution probe; walk the kt-2 T/S residue starting from the avt hypothesis;
re-score both labels.

UNASKED, offered for revert:
- the probe has no NEMO kt-1 restart to control against (the record has none), so its layout control is the
  entry-minus-card count instead;
- the (116,45) residue is classified as compile association and nothing lands; the remedy is put to the user;
- the round-9 trace (`tke_trace_kt2_fix4.npz`, same package code apart from round 9's raise guard) was reused as the
  replay input instead of being regenerated.

UNVERIFIED:
- which in-model operand (the shear or N2) is the one-ulp copy;
- which `tra_zdf_imp` branch (:185/:189/:194) V1 takes;
- the (26,133) owner.
