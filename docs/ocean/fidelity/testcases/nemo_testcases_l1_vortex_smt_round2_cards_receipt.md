# Round 212 / VORTEX_SMT round 2 — the hook reviewed, the gdept statement, and two seamount cards

**Decision 88 (user, 2026-10-03), operator note CC addendum 2.** Round 1 built
NEMO's side: a Gaussian seamount through NEMO's own `usr_def_zgr` hook, partial
steps under `key_vco_1d3d`, four admitted records. This round reviews that
hook, closes round 1's open finding as a cited statement, and declares the two
legoESM cards.

Predictions were written before any measurement:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/round2/predictions.md`.

---

## 0. THE FORTRAN HOOK, ADVERSARIALLY REVIEWED — FAITHFUL-WITH-DEFECTS

A fresh reviewer was given the ~310-line hook, `tests/VORTEX/MY_SRC/usrdef_zgr.F90`,
`tests/OVERFLOW/MY_SRC/usrdef_zgr.F90`, `tests/IWAVE/MY_SRC/usrdef_zgr.F90`,
`tools/DOMAINcfg/src/domzgr.F90::zgr_zps`, `src/OCE/DOM/domzgr.F90` and
`dom_oce.F90`, and one question: is every statement a faithful transcription,
and what is missing or wrong.

**Verdict: FAITHFUL-WITH-DEFECTS — no statement changes the resolved ocean
geometry.** The 1-D part is byte-verbatim against VORTEX's own `zgr_z`.

| # | finding | class | action |
|---|---|---|---|
| 1 | the control print's `MINVAL`/`MAXVAL` of bathymetry, `k_bot` and bottom `e3t` all sat inside `IF(lwp)`, so on two ranks they were **rank-0-local** while the text read "as resolved on this rank" | REAL DEFECT, instrument only | **FIXED**: every reduction is now taken by all ranks over the interior window through `mpp_min`/`mpp_max` before the `IF(lwp)`; the surviving i-row print is relabelled RANK-0 SUBDOMAIN ONLY, which its local `MINLOC` makes it |
| 2 | the land ring leaves the hook with `k_top = 1` and a partial `e3t`, because `dom_zgr` masks it only afterwards (`domzgr.F90:303-315`) | LATENT, inert at 1.1e-4 m (the west wall is 4L from the seamount) | **not a code change** — but it IS the behaviour the legoESM cards had to reproduce, and they do; see §3 |
| 3 | four wrong citations (`domzgr.F90:242-243`; `zgr_zps` lbc_lnk pair, `WHERE(e3==0)` repairs, row duplication) | COSMETIC | **FIXED** |
| 4 | the `jj=1 -> jj=2` duplication was justified as "ORCA-specific"; it is unconditional in `zgr_zps` | prose wrong, omission still exact (`kfillmode=jpfillcopy` performs the same copy) | **FIXED** |
| 5 | the other three exclusions (OVERFLOW's `pe3u=pe3t` shortcut, ice-shelf branches, the `WHERE(e3==0)` repair) | NOT-A-DEFECT | kept, reasons now in the module header |
| 6 | clean: interface/optional-argument match under `key_vco_1d3d`, `DO_2D(L,R,B,T)` argument order vs the `ji+1`/`jj+1` reads, 2-rank halo correctness, which array is MIN'd with which, `e3f` from `e3v` not `e3u`, loop direction, `jpkm1` vs `jpk`, the `ik+1` write at `ik=jpkm1`, bathymetry units/sign/centre, literal kinds and expression association | NOT-A-DEFECT | — |

**The defect, MEASURED, not merely reviewed.** `ocean.output` of the admitted
flux record prints `k_bot min = 0`; the global interior minimum read from the
same run's `mesh_mask.nc` is **8**. The bathymetry and bottom-`e3t` extrema
happen to agree because rank 0 contains the summit. Round 1's receipt quoted
those two from the sanity probe's global `mesh_mask` read and is therefore
still right; the print was wrong on `k_bot min` and is now global.

**RE-ACQUISITION: NOT DONE, and why.** The fix is print-and-comment only — the
diff touches comments, `WRITE` statements, three new local declarations and
`mpp` reductions consumed by nothing but a `WRITE` — so NEMO's ocean output
cannot move. Re-running the committed tool anyway (so the committed hook is the
one that produced the evidence, and byte-identity is shown rather than argued)
required moving round 1's four build directories aside, and **that action was
refused by this session's permission boundary.** The four round-1 records
therefore stand as admitted, produced by a hook identical to the committed one
except in its printing. OPEN item 1 below; it is cheap (two 2-second runs plus
their builds) and is the first thing round 3 should do.

---

## 1. THE gdept STATEMENT — round 1's guess REFUTED, the real one PROVED TO THE BIT

Round 1, PLAUSIBLE and carried open: *"the depths handed to `istate` are rebuilt
from the 3-D scale factors on the `vco_1d3d` path rather than taken from the
1-D reference."*

**REFUTED, from NEMO's own macros.** `src/OCE/DOM/domzgr_substitute.h90:71`
opens one block for `key_vco_1d .OR. key_vco_1d3d`, and inside it

```
:75      #define  DEPt_0(i,j,k)   gdept_1d(k)
:76      #define  DEPw_0(i,j,k)   gdepw_1d(k)
:78      #define  gdept_0(i,j,k)  gdept_1d(k)
:80      #define  e3w_0(i,j,k)    e3w_1d(k)
```

Only the `E3t_0/E3u_0/E3v_0/E3f_0` macros split on the key (`:82-100`). So
`gdept`, `gdepw` and `e3w` are the SAME 1-D ladder under both keys; nothing is
rebuilt. (The hook reviewer reached the same reading independently.)

**The real statement.** Under `key_qco`,

```
domzgr_substitute.h90:139   gdept(i,j,k,t) = ( gdept_1d(k) Tisf(r3t,risfdep,i,j,t)
                       :51   Tisf(r3,isf,i,j,t)  ->  ) Time(r3,i,j,t)        [no key_isf]
                       :50   Time(r3,i,j,t)      ->  *(1._wp + r3(i,j,t))
                     i.e.   gdept = gdept_1d(k) * ( 1 + r3t(i,j,t) )
domqco.F90  dom_qco_r3c      pr3t(ji,jj) = pssh(ji,jj) * r1_ht_0(ji,jj)
domain.F90:139-144           ht_0 = SUM_k e3t_0(:,:,jk) * tmask(:,:,jk)
domain.F90:158               r1_ht_0 = ssmask / ( ht_0 + 1 - ssmask )
istate.F90:127-130           zgdept(:,:,jk) = gdept(:,:,jk,Kbb) ; CALL usr_def_istate( zgdept, ... )
```

Under `key_vco_1d` `e3t_0` is `e3t_1d`, so `ht_0` is 5000 m in every wet column.
Under `key_vco_1d3d` `e3t_0` is `e3t_3d`, so **`ht_0` IS the seamount
bathymetry**. The vortex's initial `ssh` is non-zero
(`usrdef_istate.F90:177-183` runs before `dom_qco_zgr`, which runs before
`istate`), so the two runs hand `usr_def_istate` different depths. The initial
state never reads the bottom; it reads a stretching factor whose denominator is
the bottom.

**The bit-level test.** A committed probe
(`nemo_testcase_l1_vortex_smt_round2_gdept_statement.py`) reconstructs NEMO's
own initial-T statement (`tests/VORTEX/MY_SRC/usrdef_istate.F90:69-88`) in fp64
from each run's own `mesh_mask.nc` and its own recorded kt=1 `ssh`:

| | seamount run | flat run |
|---|---:|---:|
| `ht_0` min / max [m] | 4000.0 / 5000.0 | 5000.0 / 5000.0 |
| wet cells | 37 144 | 37 210 |
| reconstruction max abs error [K] | **0.0** | **0.0** |
| cells reconstructed EXACTLY | 37 144 / 37 144 | 37 210 / 37 210 |

and on the 37 144 cells wet in both:

| | value |
|---|---|
| recorded max \|ΔT\| | `2.7873925644072983e-05` K |
| **predicted** max \|ΔT\| | `2.7873925644072983e-05` K |
| max \|predicted − recorded\| | **0.0** |
| same argmax cell | yes, `(j,i,k) = (31,29,9)` |
| max depth difference | `8.2887106e-03` m |
| cells with ΔT ≠ 0 | 3 273 |

**CONTROL.** Re-running the same reconstruction for the seamount run with the
FLAT run's `r1_ht_0` substituted and nothing else changed gives
`max |ΔT| = 0.0` against the flat reconstruction — the mechanism accounts for
the difference entirely, and the control is not a perturbation of a zero
(the uncontrolled difference is 2.79e-05 K).

P1 and P3 CONFIRMED. **This is a statement, not a tweak**, and the legoESM
cards build their initial state from depths built the same way (§3).

---

## 2. THE RULE legoESM NEEDED, AND WHY IT IS NOT THE ONE IT HAD

Pre-impl search (RULE 4), pasted rather than claimed:

```
$ grep -rn "def create_partial\|def create_full_step\|bottom_index_rule" packages/ocean/legoesm/ocean/vertical.py
1438:def create_partial_cell_coordinate(   ...   bottom_index_rule: str = "interface"
1597:def create_full_step_coordinate(
$ grep -n "def build_overflow_zps_card" packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py
983:def build_overflow_zps_card() -> NEMOTestcaseCard:
```

So partial-cell machinery EXISTS (`create_partial_cell_coordinate`, used by the
OVERFLOW zps card, and by ORCA2 through its `domain_cfg`), and it was EXTENDED,
not duplicated. What it did not have is the rule this deck runs.

NEMO ships **two different zps bottom rules in its user domains** and they are
not equivalent:

| | legoESM `"nemo_tpoint"` | legoESM `"nemo_zps_e3min"` (new) |
|---|---|---|
| source | `usrdef_zgr` T-point form | `tests/OVERFLOW/MY_SRC/usrdef_zgr.F90:204,209-212`, which the SMT hook transcribes |
| index depths | `gdept_1d(k)` | `gdepw_1d(k) + ze3min`, `ze3min = 0.1*rn_dz` |
| implied floor on a uniform ladder | **half** a cell (250 m here) | **a tenth** of a cell (50 m here) |
| bottom thickness | `MIN(H - gdepw(k), dz)` | `MIN(H, gdepw(k+1)) - gdepw(k)` — NEMO's own association, which is the same number in exact arithmetic and not in fp64 |

They disagree for every column whose bathymetry lands in
`[gdepw+50, gdepw+250)`. The unit test pins one: a 4560 m column reaches level
10 under the new rule and stops at 9 under the old one. That test is the
non-vacuity proof — reverting the card to `"nemo_tpoint"` makes it fail.

`min_partial_thickness` is required by, and only by, the new rule, so neither
rule can silently run with the other's floor.

---

## 3. THE TWO CARDS, AND THE GEOMETRY IDENTITY PROOF

`VORTEX_SMT-zps` and `VORTEX_SMT_VEC-zps` are the shipped 30 km VORTEX decks in
every namelist value — `rn_dx = rn_dy = 30000`, `rn_dz = 500`, ten levels,
`rn_Dt = 2880`, `rn_ppgphi0 = 38.5`, `rn_ppumax = 1.0`, `nn_e = 48`,
`nn_bt_flt = 3`, `rn_bt_alpha = 0.07`, both lateral operators OFF,
`rn_avm0 = 1.0e-4`, `rn_avt0 = 0`, `ln_zad_Aimp = F`, `rn_shlat = 0`, the
shipped S-EOS with `rn_a0 = 0.28` — and they differ from the flat pair in the
bottom and in nothing else. The two momentum programs are the flat cards'
(decision 73), validated by the same branch of `validate_nemo_testcase_card`
rather than by a second copy of it.

**The bathymetry is an explicit field recipe, constants printed:**

```
h(x,y) = 5000 m  -  1000 m * exp( -((x - x0)^2 + (y - y0)^2) / (150 km)^2 )
(x0, y0) = (-300 km, 0 km)
```

`glamt`/`gphit` are in KILOMETRES (`tests/VORTEX/MY_SRC/usrdef_hgr.F90:78`) and
`glamt` grows with `i` at `nn_rot = 0` (`:108`), so west is `-glamt` and `x0` is
negative. The association is the hook's: `-(zx*zx + zy*zy) / (L*L)`.

**Every geometry statement, and where it comes from:**

| card statement | NEMO source |
|---|---|
| `k_bot` = count of `gdepw_1d(jk) + ze3min <= zht`, `jk = 1..jpkm1` | the hook's `DO jk = jpkm1,1,-1 ; WHERE( zht < pdepw_1d(jk)+ze3min ) k_bot = jk-1` (OVERFLOW:209-212); the downward loop's last write is the complement of the strict `<`, hence `<=` |
| `ze3min = 0.1 * rn_dz` = 50 m | OVERFLOW:204 |
| `e3t(ik) = MIN(zht, gdepw_1d(ik+1)) - gdepw_1d(ik)`, `e3t(ik+1) = e3t(ik)` | OVERFLOW:221-225 |
| `e3u = MIN(e3t(i), e3t(i+1))`, `e3v = MIN(e3t(j), e3t(j+1))` | `zgr_zps:1166-1167` |
| `e3f = MIN(e3v(i), e3v(i+1))` — from `e3v`, not `e3u` | `zgr_zps:1194` |
| the land ring is NOT masked out of `e3t` | `dom_zgr` masks it only after `usr_def_zgr` returns, `domzgr.F90:303-315`; NEMO's own `mesh_mask` carries `mbathy = 10` and `e3t = 499.99997817 m` there |
| `gdept_0`, `gdepw_0`, `e3w_0` stay on the 1-D ladder | `domzgr_substitute.h90:71-80` — this is NEMO's zps shape under this key, not a simplification |
| `ht_0/hu_0/hv_0 = SUM_k e3*_0*mask`, `hf_0 = SUM_k e3f_0*vmask(j)*vmask(j,i+1)` | `domain.F90:139-150` |
| initial `T`,`u`,`v` on `gdept_1d(k)*(1 + ssh*r1_ht_0)` with the card's own `ht_0` | §1 |

**GEOMETRY IDENTITY — 12 rows per card, every one EXACT, zero ULP**
(`nemo_testcase_l1_vortex_smt_round2_geometry_gate.py`, against each run's own
`mesh_mask.nc`; the gate pins the fp64 policy itself, because under the default
float32 policy the operands lose the partial cells and the gate goes red — that
is its own non-vacuity demonstration):

| field | `VORTEX_SMT-zps` | `VORTEX_SMT_VEC-zps` |
|---|---|---|
| `k_bot` (`mbathy`) | EXACT, 0 differing | EXACT, 0 differing |
| `e3t_0` (all 11 records, land ring included) | EXACT | EXACT |
| `e3u_0` | EXACT | EXACT |
| `e3v_0` | EXACT | EXACT |
| `e3f_0` | EXACT | EXACT |
| card `e3t_0` operand | EXACT | EXACT |
| card `e3u_0` / `e3v_0` / `e3f_0` operands | EXACT | EXACT |
| card `tmask` / `umask` / `vmask` | EXACT | EXACT |

P4 CONFIRMED. The seamount as the model resolved it: `k_bot` 8 / 9 / 10 over
5 / 56 / 3908 wet columns, summit cell FULL (500.0000 m, the 300 km offset lands
on a T-point and the summit on a w-level), bottom `e3t` 50.6710 … 500.0000 m,
66 T-cells dried at levels 9 and 10.

---

## 4. THE kt=1..10 LADDERS

Scored by the shared trajectory gate against the round-1 admitted records
(`--case VORTEX_SMT-zps|VORTEX_SMT_VEC-zps --max-step 10
--continue-after-first`), bar `1e-15`.

### Headline rows

| row | `VORTEX_SMT-zps` | `VORTEX_SMT_VEC-zps` | the FLAT card, same row |
|---|---|---|---|
| kt1 T | `0.0` AT-BAR | `0.0` AT-BAR | `0.0` AT-BAR |
| kt1 u / v | `2.220446e-16` / `2.220446e-16` AT-BAR | same | same |
| kt1 ssh | `1.355253e-20` AT-BAR | `1.355253e-20` AT-BAR | same |
| kt2 u | `6.308422e-08` | `1.748385e-07` | flux `1.216831e-08`, vector `1.304438e-15` |
| kt2 v | `4.742908e-08` | `1.607001e-07` | flux `1.239432e-08`, vector `1.335683e-15` |
| kt2 ssh | `3.726197e-07` | `3.664757e-07` | flux `2.664535e-15`, vector `2.831069e-15` |
| kt2 T | `4.259549e-10` | `1.733725e-09` | flux `2.823263e-10`, vector `3.466140e-16` |
| kt10 u | `3.976053e-06` | `8.666645e-06` | flux `9.953976e-09`, vector `6.733520e-15` |
| kt10 v | `4.380958e-06` | `5.743456e-06` | flux `8.874215e-09`, vector `6.604573e-15` |
| first over bar | **kt=2**, T/u/v/ssh | **kt=2**, T/u/v/ssh | flux kt=2 T/u/v/ssh; vector kt=2 u/v/ssh |
| registry | 10/50 AT-BAR, 40/50 DEBT | 9/50 AT-BAR, 41/50 DEBT | — |

**kt=1 IS AT THE BAR ON BOTH CARDS, ALL FIVE ROWS.** T is EXACTLY zero on
37 144 wet cells. That is the §1 statement landing: before it, the cards would
have carried NEMO's 2.79e-05 K as an initial-state error. P5 CONFIRMED.

The full 50-row registries are in
`phase3/vortex_smt/round2/registries.md` and in the two ladder JSONs.

### The first non-bit producer — NAMED AS A CANDIDATE, NOT LANDED

This round does not land a physics statement; it names the owner for round 3,
with the controlled comparison that identifies it.

**The control is the flat pair, same harness, same ten steps, one variable
(the bottom).** The flat VECTOR card is AT THE BAR at kt=2 on every field
(`1.3e-15`); the seamount vector card is at `1.75e-07`. So **the whole of the
seamount cards' kt=2 error is owned by partial-cell statements** — there is no
inherited flat-card debt underneath it on the vector arm. (On the flux arm the
flat card already carries `1.2e-08`, so its own debt is a floor there; the
vector card is the clean instrument and is the one to walk.)

**PLAUSIBLE, with the discriminator named.** The kt=2 `ssh` error is
`3.726197e-07` on the flux card and `3.664757e-07` on the vector card — within
2% of each other across two DIFFERENT momentum programs, while the flat pair's
kt=2 `ssh` is at the bar on both. A quantity that is the same size under
flux-form UP3 and under vector-invariant momentum is produced by a statement
the two programs SHARE, which on this identity is the free-surface /
split-explicit barotropic path and its partial-cell operands (`e3u(Kbb) =
e3u_0*(1+r3u)` over the substeps, `hu_0`/`r1_hu_0`, the transport divergence),
not the momentum advection or the vorticity triad. `ssh` is also the first
thing that moved in NEMO's own seamount-vs-flat comparison at kt=2 (round 1,
2.3e-03 m), which is consistent but is NOT independent evidence.

**The falsifier, and round 3's first measurement:** score the stage/substep
terms of the barotropic solve on the SMT vector card against the admitted
record the way rounds 196-200 did for the flat cards. If the per-substep
barotropic terms are at the bar and the error enters elsewhere (the HPG over
partial steps, or a bottom-level loop), the shared-statement reading is
refuted and the walk moves there. Both candidates are reachable from the
records already admitted; neither needs a new NEMO run.

