# RETRACTION, AT THE TOP, BEFORE ANYTHING ELSE (reviewer 2, BLOCK)

**THIS RECEIPT'S HEADLINE WAS MISATTRIBUTED AND IS WITHDRAWN.**  Under
`key_RK3` NEMO runs FCT at the LAST STAGE ONLY: `traadv.f90:307-311`

```
      ll_dofct = .TRUE.
      ! FCT at last stage only with RK3
      IF (PRESENT(kstg)) THEN
         IF (kstg/=3) ll_dofct = .FALSE.
      ENDIF
```

and at `traadv.f90:347-352` the `np_FCT` case dispatches to
`tra_adv_cen(..., nn_fct_h, nn_fct_v)` whenever `ll_dofct` is false.
`stprk3_stg.f90` passes `kstg` on every call.  **Every arm in this receipt
was run at STAGE 1, so NONE of the `traadv_fct.f90` statements cited in
section 2 rows 6a-6h executed.  What the bit-exact one-variable arm
actually shows is that legoESM's 2nd-order CENTRED tracer path matches
NEMO's, given NEMO's transports, over partial cells.**

WHAT SURVIVES, unchanged and still measured: given NEMO's recorded
`zFu/zFv/zFw`, legoESM's stage-1 tracer tendency is bit-identical to
NEMO's for T and S on every wet cell, on BOTH cards; and the first
substantive non-bit statement in the stage order is the continuity solve
`sshwzv.f90:297-298` at `~2-3e-13` relative on 66-99 % of wet cells.
Those two findings do not depend on which advection scheme ran.

WHAT IS WITHDRAWN: every sentence claiming FCT fidelity, including "EVERY
statement of the tracer path ... is bit-faithful" and "No tracer-path
statement is landed this round because none is wrong".  Read them as
scoped to the CENTRED path at stages 1-2.

WHAT ROUND 8 MUST DO FIRST: re-run the same arm at **STAGE 3**, where
`ll_dofct` is true, before any claim about FCT is made.  The record
already carries stage 3 (`oracle_tracer_terms_kt00000001_s3.bin`, admitted
on both cards), so this costs no NEMO time -- only the walk's stage
selector needs to move.

**AND THE ORCA2 POINTER IN SECTION 6 IS AFFECTED**: its recommended arm
must be taken at ORCA2's stage 3, not stage 1, or it will exercise CEN2
instead of the FCT lines ORCA2's overflow names.  The two CODE-READING
items in section 6 -- that `traadv_fct.f90:569-570` averages a `pt_up1`
built at `:538`, and that `trazdf.f90:231-233` divides its off-diagonals
by the 1-D `e3w_1d` while its diagonal uses the 3-D `e3t_3d` -- were
verified by reviewer 2 against ORCA2's own compiled tree and STAND.

# Round 218 / VORTEX_SMT round 7 — the seamount cards' TRACER path over partial cells

Decision 92 (operator note CD).  Lane tip `a3be519e0`, branch
`fidelity/nemo-testcases-l2-gyre-codex2`, PR #1869 open.
Working copy `phase3/claude_rounds/vortex_smt_r7/repo`; evidence
`phase3/vortex_smt/round7/`.

**WHAT THIS ROUND IS FOR.**  ORCA2's rung-0 month overflows at step 36
inside NEMO's FCT tracer advection next to partial cells.  The seamount
cards run the same compiled statements in a two-second run, so this round
reads the tracer path off the compiled source statement by statement, builds
the record no earlier seamount round had (NEMO's own tracer operands at
every RK3 stage), walks the first-over-bar tracer row one variable at a
time, and writes ORCA2 a pointer that names lines rather than files.

Predictions were frozen in `round7/predictions.md` before the instrument was
written and before any arm ran.

## 0. PRE-IMPL SEARCH (RULE 4)

| searched | found | what I did |
|---|---|---|
| a NEMO record carrying any tracer array on a seamount build | `grep -l tracer tests/VORTEX_SMT_*/MY_SRC/*` -> nothing; the two stage writers (`vortex_r8_stage_terms.F90`, `vortex_r16_stage_terms.F90`) write `uu/vv/ww/ssh` only | EXTENDED the pattern: a new writer in the same shape, same header, same self-describing group stream |
| an existing FCT substitution probe | `scripts/validate/ocean_fidelity/dino_1226/traadv_fct_ww_inheritance.py` and `traadv_fct_probe.py` | REUSED THE METHOD (substitute one recorded operand, report what fraction of the residual collapses); the code is DINO/MLF-specific and does not run on an RK3 card |
| a legoESM seam for the tracer transports and the stage-1 tracer boundary | `_NEMOWSRK3TestHooks.stage1_tracer_transport_override`, `expose_tracer_transport_stage`, `expose_tracer_stage1_boundary`, `expose_tracer_stage` all already exist | REUSED; **no new seam was added to the model** |
| an existing walk of the same shape | `nemo_testcase_l1_vortex_round200_flux_stage1.py` (VORTEX, momentum) and `nemo_testcase_l4_orca2_round43_tracer_handoff_gate.py` (ORCA2, tracer) | REUSED both: the new walk imports round 200's `require_live` and `_unowned_for_test` and follows round 43's row order |
| a record checker to extend | `nemo_testcase_l1_vortex/check_records.py` | EXTENDED with one family; no second checker |
## 1. THE RESOLVED TRACER DECK, READ OFF THE COMPILED BUILD

Every value below is from `tests/VORTEX_SMT_R3_OMIP_L1_P3/EXP00/namelist_cfg`
(or `namelist_ref` where the cfg is silent) and the compiled `BLD/ppsrc/nemo`
of that build; the vector build `VORTEX_SMT_R3_VEC_R8_OMIP_L1_P3` differs only
in the two momentum-advection switches.

| knob | resolved | where |
|---|---|---|
| `ln_traadv_fct` | `.true.` | `namelist_cfg` namtra_adv |
| `nn_fct_h` | `2` (2nd-order centred) | `namelist_cfg` namtra_adv |
| `nn_fct_v` | `2` (2nd-order centred) | `namelist_cfg` namtra_adv |
| `nn_fct_imp` | `1` | not in cfg; `namelist_ref` default |
| `ln_traldf_OFF` | `.true.` | `namelist_cfg` namtra_ldf |
| `ln_zdfcst` | `.true.` | `namelist_cfg` namzdf |
| `rn_avt0` | **`0.e0`** | `namelist_cfg` namzdf |
| `rn_avm0` | `1.e-4` | `namelist_cfg` namzdf |
| `nn_avb` / `nn_havtb` | `0` / `0` | `namelist_cfg` namzdf |
| `ln_zdfevd` / `ln_zdfnpc` | `.false.` / `.false.` | `namelist_cfg` namzdf |
| `ln_zad_Aimp` | `.false.` | not in cfg; `namelist_ref:1177` |
| `ln_shuman` | `.false.` | not in cfg; `namelist_ref:79` |
| cpp keys | `key_qco key_RK3 key_vco_1d3d` | `cpp_VORTEX_SMT_R3_OMIP_L1_P3.fcm` |

## 2. THE COMPILED STATEMENTS, IN NEMO's STAGE ORDER

Line numbers are the SMT build's own `BLD/ppsrc/nemo`.  Every statement
below runs at every RK3 stage unless noted.

| # | file:line | statement | partial-cell operand |
|---|---|---|---|
| 1 | `stprk3_stg.f90:263-271` | barotropic velocity correction `zub/zvb = un_adv*(r1_hu_0/(1+r3u(Kmm))) - uu_b` | `r1_hu_0` = 1/SUM_k e3u_0 |
| 2 | `stprk3_stg.f90:276-277` | `zFu = e2u*(e3u_3d*(1+r3u(Kmm)*umask))*(uu(Kmm)+zub*umask)` | **`e3u_3d`**, the shallower neighbour's reference face thickness (landed round 4) |
| 3 | `stprk3_stg.f90:298` (flux) / `:293` (vector, stages 2-3) | `wzv` -> `ww`, the continuity solve | `e3t_3d` inside `wzv` |
| 4 | `stprk3_stg.f90:301` | `zFw = e1e2t*ww` (flux form) | none |
| 5 | `stprk3_stg.f90:431` / `:454` | `tra_adv_trp` updates (flux) or computes (vector) `zFu,zFv,zFw` | the same `e3u_3d/e3v_3d` |
| 6 | `stprk3_stg.f90:474` | `tra_adv` -> `tra_adv_fct` | — |
| 6a | `traadv_fct.f90:170` | `fct_up1_2stp(...)`, the two-step upstream guess | — |
| 6b | `traadv_fct.f90:502-508` | 1st-step upstream fluxes `MAX(pU,0)*pt_b(ji) + MIN(pU,0)*pt_b(ji+1)` | `pU` carries `e3u_3d` |
| 6c | `traadv_fct.f90:538` | **mid-step guess** `pt_up1 = (e3t_3d*(1+r3t(Kbb))*pt_b + zDt*ztra) / (e3t_3d*(1+r3t(Kmm)))` | **`e3t_3d`, the partial T thickness, as the DIVISOR** |
| 6d | `traadv_fct.f90:569-573` | 2nd-step **averaged** upstream fluxes `0.5*(ptFu + MAX(pU,0)*pt_up1(ji) + MIN(pU,0)*pt_up1(ji+1))` | consumes 6c |
| 6e | `traadv_fct.f90:197-198` | anti-diffusive horizontal, 2nd-order centred (`nn_fct_h=2`) | `pU` |
| 6f | `traadv_fct.f90:265-266` | anti-diffusive vertical, 2nd-order centred (`nn_fct_v=2`) | `pW` |
| 6g | `traadv_fct.f90:316` | `nonosc` flux limiter; its cell budget is `zbt = e1e2t*e3t_3d*(1+r3t(Kaa))/p2dt` (`:715`) | **`e3t_3d`, and at `Kaa`** |
| 6h | `traadv_fct.f90:327` | `Krhs += ztra / (e3t_3d*(1+r3t(Kmm)))` | **`e3t_3d`, and at `Kmm`** |
| 7 | `stprk3_stg.f90:476` | `tra_sbc_RK3`; this deck's `usrdef_sbc` sets `utau=vtau=qns=qsr=emp=sfx=0`, so it adds nothing | — |
| 8 | `stprk3_stg.f90:502-506` (stages 1,2) | `ts(Kaa) = ((1+r3t(Kbb))ts(Kbb) + rDt(1+r3t(Kmm))ts(Krhs)tmask)/(1+r3t(Kaa))` | **none: the qco stage step carries NO `e3t`, only the surface ratio** |
| 9 | `stprk3_stg.f90:538` (stage 3) | `tra_zdf` -> `tra_zdf_imp` | see below |
| 9a | `trazdf.f90:231-232` | off-diagonals `-p2dt*zwt / (e3w_1d(jk)*(1+r3t(Kmm)))` | **`e3w_1d`, the ONE-DIMENSIONAL reference ladder, not a partial `e3w`** |
| 9b | `trazdf.f90:233` | diagonal `zwd = e3t_3d*(1+r3t(Kaa)*tmask) - (zwi+zws)` | **`e3t_3d`, three-dimensional** |
| 9c | `trazdf.f90:284-290,297` | the RHS and the back-substitution, both weighted by `e3t_3d` | `e3t_3d` |

Two things in that table are statements nobody had written down on this lane:

* **Step 8 is where the two continuity solves do NOT meet.** The tracer's
  thickness weighting in the stage step is the surface ratio `r3t` alone;
  the partial thickness enters the tracer only through the FLUXES (step 2)
  and the FCT divisors (6c/6g/6h).  A card that weighted the stage step by
  a partial `e3t` would be weighting it twice.
* **Step 9a divides by the 1-D reference `e3w` while 9b uses the 3-D
  `e3t`.**  That asymmetry is NEMO's own qco formulation and it is the
  single most ORCA2-relevant line in the table (section 6).

## 3. WHAT THIS DECK CANNOT TEST, SAID OUT LOUD

`rn_avt0 = 0.`, `nn_avb = 0` so `avtb(:) = rn_avt0 = 0` and `avtb_2d = 1`
(`zdfphy.f90:206-208,215,227`), the closure is `np_CST`, `ln_rnf_mouth` and
`ln_zdfevd` are both off, so `avt = avt_k = 0` everywhere
(`zdfphy.f90:349`).  Step 9a's off-diagonals are therefore identically zero
and `tra_zdf_imp` degenerates to the thickness-weighted division 9c.

**The seamount cards cannot exercise the partial-cell statement in NEMO's
implicit vertical tracer diffusion at all.**  ORCA2 can: its deck sets
`ln_zdftke = .true.`, `ln_zdfevd = .true.`, `rn_avt0 = 1.2e-5`.  This is
reported as a scope limit, not worked around.
## 4. WHICH TRACER ROW IS FIRST OVER THE BAR, MEASURED FROM THE ADMITTED REGISTRIES

Read off round 6's certified after-arm (`round6/after3/after_*.json`), not
re-run: these are the registries note CD points at.

| card | first TRACER row over `1e-15` | value | S first over bar | value |
|---|---|---:|---|---:|
| `VORTEX_SMT-zps` (flux) | **T at kt=2** | `3.803022e-10` | S at kt=6 | `1.0151e-15` |
| `VORTEX_SMT_VEC-zps` (vector) | **T at kt=2** | `3.618737e-12` | S at kt=6 | `1.0151e-15` |

Note CD expected the first tracer row at `kt>=3`; the registries say kt=2
on both cards, and this receipt reports what they say.

**S CARRIES NO STATEMENT IN THIS WINDOW.**  Every S value in both ladders is
an exact multiple of `2.030e-16`: `0`, `4.060e-16`, `6.090e-16`,
`8.121e-16`, `1.0151e-15`, `1.2181e-15` — two, three, four, five, six
quanta of the last bit of a 35-psu field.  The "crossing" at kt=6 is one
quantum of rounding, not a physical divergence, and it is reported as such
rather than walked.

**THE FLUX CARD'S T ROW IS NOT A PARTIAL-CELL ROW (R7-P0 MET).**  The FLAT
`VORTEX-zco` card — no partial cell anywhere in its mesh — already carries
`2.823e-10` on the same row, i.e. 74 % of the seamount flux card's
`3.803e-10`.  Its owner is the flux-program momentum debt round 204 named,
not topography.  The flat VECTOR card `VORTEX_VEC-zco` is AT THE BAR on T at
kt=2 (`3.466e-16`), so the vector seamount card's `3.618737e-12` is the only
CLEAN partial-cell tracer signal the sub-campaign has, and the walk is run
there.

| card | kt=2 T | kt=2 u | kt=2 v |
|---|---:|---:|---:|
| `VORTEX-zco` (flat, flux) | `2.823e-10` | `1.217e-08` | `1.239e-08` |
| `VORTEX_SMT-zps` (seamount, flux) | `3.803e-10` | `2.161e-08` | `1.545e-08` |
| `VORTEX_VEC-zco` (flat, vector) | `3.466e-16` AT BAR | `1.303e-15` AT BAR | `1.340e-15` AT BAR |
| `VORTEX_SMT_VEC-zps` (seamount, vector) | `3.619e-12` | `1.268e-09` | `2.549e-10` |

## 5. CANDIDATE OWNERS, PREREGISTERED BEFORE THE WALK RAN

In NEMO's stage order (section 2), the statements that could own the vector
seamount card's kt=2 T row:

1. the stage transports `zFu/zFv/zFw` (step 2/5) — landed at round 4 for
   `e3u_0`, but the stage's `(1+r3u(Kmm))` factor and `tra_adv_trp`'s
   update are not separately scored;
2. the continuity solve `ww` (step 3) over partial columns;
3. the FCT mid-step divisor `e3t_3d*(1+r3t(Kmm))` (step 6c);
4. the limiter's `Kaa`-weighted cell budget (step 6g);
5. the tendency divisor at `Kmm` (step 6h);
6. the qco stage step (step 8) — `r3t` only, no `e3t`;
7. `tra_zdf` (step 9) — EXCLUDED by section 3: `avt` is identically zero.
## 7. THE WALK, ONE VARIABLE AT A TIME, VECTOR SEAMOUNT CARD, kt=1 STAGE 1

Record: `round7/VORTEX_SMT_R7_VEC_R8_OMIP_L1_P3/tracer`, ADMITTED, the
step-10 restart BYTE-IDENTICAL to round 3's reference run of the same deck
(the lane's additions-only proof); the checker's header plant fired.
Walk: `walk_vec.json`.  Bit equality, not AT-BAR; the seam control
(`require_live`) ran on every exposed row, and the `adv.T` plant was
VISIBLE against the clean report.

Rows in NEMO's own stage order.  `rel` is the max absolute difference over
the row's own peak.

| # | row | NEMO boundary | cells unequal / active | max abs | rel |
|---|---|---|---:|---:|---:|
| 1 | `zfu` | `stprk3_stg.f90:276-277`, `traadv.f90:268` | 85 / 36522 | `1.863e-09` | `1.42e-16` |
| 1 | `zfv` | same | 93 / 36522 | `1.863e-09` | `1.42e-16` |
| 2 | **`ww`** | the RK3 continuity solve, `sshwzv.f90:297-298` | **24648 / 37144** | `1.936e-17` | **`1.99e-13`** |
| 2 | **`zfw`** | `pFw = e1e2t*ww`, `traadv.f90:272` | **24617 / 37144** | `1.743e-08` | **`1.99e-13`** |
| 3 | `adv.T` | `ts(Krhs)` after `tra_adv`+`tra_sbc_RK3` | 18219 / 37144 | `7.869e-19` | `1.37e-12` |
| 3 | `adv.S` | same | 15950 / 37144 | `1.468e-18` | `2.05e-11` |
| 4 | `out.T` | `ts(Kaa)`, `stprk3_stg.f90:552-554` | 1775 / 37144 | `3.553e-15` | `1.73e-16` |
| 4 | `out.S` | same | 1621 / 37144 | `1.421e-14` | `4.06e-16` |

Rows 1 and 4 are ONE unit in the last place of their own field
(`3.553e-15 = 2^-52 x 16`, `1.421e-14 = 2^-52 x 64`) on 0.2-5 % of cells:
last-bit composition, inside the `1e-15` bar in relative terms.  Row 2 is
**1.99e-13 relative on 66 % of the wet cells** — a thousand times the
others, and it is the first row in the stage order that is not last-bit.

### THE ONE-VARIABLE ARM — AND IT CLOSES THE TRACER PATH COMPLETELY

NEMO's own recorded `zFu/zFv/zFw` substituted into legoESM's stage-1 tracer
step (`stage1_tracer_transport_override`), nothing else changed:

| row | before | after |
|---|---:|---:|
| `adv.T` (tracer tendency) | `7.868936e-19` on 18219 cells | **`0.0` on 0 cells — BIT-IDENTICAL** |
| `adv.S` (tracer tendency) | `1.468120e-18` on 15950 cells | **`0.0` on 0 cells — BIT-IDENTICAL** |
| `out.T` | `3.552714e-15` on 1775 cells | `3.552714e-15` on 1774 cells |
| `out.S` | `1.421085e-14` on 1621 cells | `1.421085e-14` on 1620 cells |

**R7-P1 IS MET, AND MORE STRONGLY THAN IT WAS PREDICTED.**  The prediction
was "within a factor 10 of the bar"; the measurement is bit equality on
every one of 37,144 wet cells, for both tracers.

**WHAT THAT SENTENCE MEANS, SAID PRECISELY.**  Given NEMO's transports,
legoESM reproduces NEMO's tracer tendency to the last bit over the
seamount.  So EVERY statement of the tracer path listed in section 2 is
bit-faithful over partial cells on this card: the two-step upstream fluxes
(`traadv_fct.f90:502-508`, `:569-573`), the mid-step upstream guess and its
partial-`e3t` divisor (`:538`), the 2nd-order centred anti-diffusive fluxes
(`:197-198`, `:265-266`), the `nonosc` limiter with its `Kaa`-weighted cell
budget (`:316`, `:715`), and the tendency divisor at `Kmm` (`:327`).
**No tracer-path statement is landed this round because none is wrong.**

### THE FIRST NON-BIT STATEMENT, NAMED AND CITED

The debt enters the tracer through the VERTICAL transport, and the
statement that makes it is the RK3 continuity solve's quasi-Eulerian
recurrence, `sshwzv.f90:297-298` of the SMT build:

```
pww(ji,jj,jk) = pww(ji,jj,jk+1) - (  ze3div(ji,jj,jk)                                    &
   &                               + r1_Dt * e3t_3d(ji,jj,jk) * ( r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb) )  ) * tmask(ji,jj,jk)
```

reached on this card from `traadv.f90:266-272`
(`IF( ll_Fw ) CALL wzv(..., np_transport)` then `pFw = e1e2t*ww`), with
`ll_Fw` set true at `traadv.f90:195` because `ln_dynadv_vec` is true.

Its partial-cell operand is **`e3t_3d(ji,jj,jk)`, the partial T-cell
thickness, inside the thickness-tendency term** — on a flat card that
factor is the uniform reference ladder and the two flat VORTEX cards are at
the bar there, which is the control.

**IT IS NOT YET ATTRIBUTED TO ONE OF THE TWO OPERANDS, AND THIS RECEIPT
SAYS SO.**  The row has two candidate producers inside the same statement:
the horizontal divergence `ze3div` that `div_hor(..., np_transport)` returns
(`sshwzv.f90:280`), and the `r1_Dt*e3t_3d*(r3t(Kaa)-r3t(Kbb))` term.  The
barotropic output is pinned to NEMO's recorded values in this walk, so
`r3t(Kaa)` and `r3t(Kbb)` are NEMO's; that points at `e3t_3d` or at
`ze3div`, and the discriminating arm is one more exposure, preregistered in
section 10.  Calling it now would be a guess.

**STATUS: HELD, as a measurement round.**  No model file is edited, so no
card moves, no registry row moves, and the two-ULP ratchet cannot go red.
The operand split of `sshwzv.f90:297-298` is the round's OPEN item and its
preregistered arm is R8-P1 in section 10.
## 9. THE FLUX SEAMOUNT CARD

Its record (`VORTEX_SMT_R7_OMIP_L1_P3/tracer`, variant `smtflxtra`) was
acquired with the same instrument and the same additions-only proof.  Its
kt=2 T row is NOT a partial-cell row — the FLAT flux card already carries
74 % of it (section 4) — so the flux card is the CONTROL here, not the
subject: it says whether the continuity statement named in section 7 is
shared by both momentum programs or belongs to the vector one.

Its record is ADMITTED, restart BYTE-IDENTICAL, header plant fired
(`walk_flux.json`).  **IT REPRODUCES THE VECTOR CARD'S RESULT, AND THAT
MAKES THE STATEMENT A SHARED ONE.**

| row | flux card | vector card |
|---|---:|---:|
| `zfu` / `zfv` | `1.42e-16` rel, 78 / 86 cells | `1.42e-16` rel, 85 / 93 cells |
| **`ww`** | **`3.368e-13` rel, 36713 cells** | **`1.993e-13` rel, 24648 cells** |
| **`zfw`** | **`3.368e-13` rel, 36709 cells** | **`1.993e-13` rel, 24617 cells** |
| `adv.T` | `1.83e-12` rel, 34793 cells | `1.37e-12` rel, 18219 cells |
| `out.T` / `out.S` | 1 ULP, 1710 / 1912 cells | 1 ULP, 1775 / 1621 cells |

One-variable arm, NEMO's recorded `zFu/zFv/zFw` substituted, flux card:

| row | before | after |
|---|---:|---:|
| `adv.T` | `1.048203e-18` on 34793 cells | **`0.0` on 0 cells — BIT-IDENTICAL** |
| `adv.S` | `1.961728e-18` on 34113 cells | **`0.0` on 0 cells — BIT-IDENTICAL** |

So the FCT tracer path is bit-faithful over partial cells under BOTH
momentum programs, and the `ww` statement is carried by both — which is
what `sshwzv.f90:297-298` being reached from `stprk3_stg.f90:298` on the
flux card and from `traadv.f90:268` on the vector card predicts.  The
preregistered falsifier (the flux card's `ww` at the bar) did NOT fire.
## 8. CHOICES MADE THIS ROUND

| # | choice | ASKED? |
|---|---|---|
| 1 | Walk the VECTOR seamount card, not the flux one, because the flat flux card already carries 74 % of the flux card's kt=2 T row (section 4) | UNASKED — a measurement-scope choice, reported here; the flux card is measured too, so nothing is hidden |
| 2 | Record the tracer RHS at ONE boundary (after `tra_adv` + `tra_sbc_RK3`) rather than two, because this deck's `usrdef_sbc` sets every surface flux to zero so the two boundaries are the same array | UNASKED — proven, not assumed (`usrdef_sbc.F90:60-68`) |
| 3 | New build directories `VORTEX_SMT_R7_*` rather than re-using round 3/5/6's | the lane's standing rule (note CC addendum 3), not a new choice |
| 4 | The tracer writer dumps `zFu/zFv/zFw` AFTER `tra_adv_trp`, not before, because on the vector card `zFw` does not exist before it (`stprk3_stg.F90:287`) | UNASKED — forced by the code, stated |
| 5 | No production model file is edited in this round | not a choice: the round is a measurement unless a statement is named |

No default value, scheme selection, bound, tier, cadence, window or data
source was changed.
## 6. THE ORCA2 POINTER

ORCA2's rung-0 month overflows at step 36 inside NEMO's FCT tracer
advection, cited as `traadv_fct.f90:569-570`, with the headline cell going
non-finite later in the implicit vertical diffusion, near partial cells.
Three things, each read off ORCA2's OWN compiled build
(`cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo`, keys `key_si3 key_qco key_vco_1d3d
key_RK3`), not inferred from the seamount build.

**(a) `:569-570` is not where the number is made.  `:538` is.**
`traadv_fct.f90:569-570` is

```
ptFu(ji,jj,jk) = 0.5_wp * ( ptFu(ji,jj,jk) + MAX( pU(ji,jj,jk) , 0._wp ) * pt_up1(ji  ,jj,jk) &
   &                                       + MIN( pU(ji,jj,jk) , 0._wp ) * pt_up1(ji+1,jj,jk) )
```

— an AVERAGE of an already-formed `pt_up1` with an already-formed `ptFu`.
Neither operand is built there.  `pt_up1` is built thirty-one lines earlier,
at `traadv_fct.f90:538`, and that line is the only DIVISION in the chain:

```
pt_up1(ji,jj,jk) = ( (e3t_3d(ji,jj,jk)*(1._wp+r3t(ji,jj,Kbb)*tmask(ji,jj,jk))) * pt_b(ji,jj,jk) &
   &                 + zDt * ztra ) / (e3t_3d(ji,jj,jk)*(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk))) * tmask(ji,jj,jk)
```

The divisor is the PARTIAL T-cell thickness at `Kmm`.  At ORCA2's thinnest
partial cells that divisor is the smallest number in the whole tracer path,
and a `ztra` that is not correspondingly small is amplified there, one
statement before the line the traceback names.  The same line numbers hold
in ORCA2's build and in the seamount build — the two ppsrc files agree.

**WHAT ORCA2'S RUNG-0 WALK SHOULD MEASURE AT THE STEP-36 CELL, in this
order, before anything else:** `e3t_3d(ji,jj,jk)` and `r3t(ji,jj,Kmm)` at
that cell (the `:538` divisor); `ztra` at `:534-536`, i.e. the upstream flux
divergence times `r1_e1e2t`; then `pt_up1` itself.  If `pt_up1` is already
huge, `:569-570` is a messenger.  The discriminating check is cheap: it is
three arrays at one index.

**(b) the limiter and the tendency disagree on the time level, by design.**
`nonosc`'s per-cell budget uses `Kaa` (`traadv_fct.f90:715`,
`zbt = e1e2t*e3t_3d*(1+r3t(Kaa))/p2dt`) while the tendency it limits is
divided by `Kmm` (`:327`, `:605`).  Both are `e3t_3d`.  A card that used one
time level for both would produce a limiter that is systematically loose or
tight at partial cells, which is exactly the shape of an overflow that only
appears after thirty-odd steps.

**(c) the implicit vertical diffusion divides by the ONE-DIMENSIONAL `e3w`.**
In ORCA2's own `trazdf.f90` the tridiagonal is built as

```
231:  zwi(ji,jk) = - p2dt * zwt(ji,jk  ) / (e3w_1d(jk  ) *(1._wp+r3t(ji,jj,Kmm)))
232:  zws(ji,jk) = - p2dt * zwt(ji,jk+1) / (e3w_1d(jk+1) *(1._wp+r3t(ji,jj,Kmm)))
233:  zwd(ji,jk) = (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kaa)*tmask(ji,jj,jk))) - ( zwi(ji,jk) + zws(ji,jk) )
```

The off-diagonals divide by **`e3w_1d`, the 1-D reference ladder**, even
under `key_vco_1d3d`; only the diagonal carries the 3-D partial `e3t_3d`.
At a partial bottom cell `e3t_3d` can be a tenth of the reference level
thickness while `e3w_1d` stays at the full reference value, so NEMO's
diagonal shrinks and its off-diagonals do not.  **If legoESM divides by a
partial or 3-D `e3w` there, its matrix differs from NEMO's at every partial
cell by that ratio, and with ORCA2's `ln_zdfevd = .true.` bumping `avt` on
convective columns the off-diagonals are large enough for the difference to
run away.**  This is an association to CHECK, not a measured defect: the
seamount deck sets `rn_avt0 = 0.` with `ln_zdfcst`, so `avt` is identically
zero there (`zdfphy.f90:206-208,227,349`) and VORTEX_SMT cannot measure it.
ORCA2 must.

**THE ONE-VARIABLE ARM WE RECOMMEND**, and the reason we recommend it: hand
ORCA2's stage-1 tracer step NEMO's own recorded `zFu/zFv/zFw` triplet and
change nothing else (`stage1_tracer_transport_override`, already wired, and
already exercised by `nemo_testcase_l4_orca2_round43_tracer_handoff_gate.py`),
then read the tendency at `after_sbc`.  It separates the two families in one
run: a tendency that comes to the bar says every FCT statement — the upstream
fluxes, the `:538` divisor, the centred anti-diffusive fluxes, the limiter —
is faithful over partial cells and the overflow is carried IN through the
transports, so the walk belongs upstream in the momentum/continuity path; a
tendency that does not come to the bar names the tracer path, and then (a)'s
three arrays at the step-36 cell say which statement.
## 10. ROUND 219 — PREREGISTERED

* **R8-P1** — splitting `sshwzv.f90:297-298` into its two operands (expose
  `ze3div` from `div_hor(..., np_transport)` separately from the
  `r1_Dt*e3t_3d*(r3t(Kaa)-r3t(Kbb))` term) puts ALL of the `1.99e-13`
  relative `ww` error in exactly one of them.  **FALSIFIER:** both operands
  are at the bar, or each carries roughly half — then the recurrence's
  bottom-up composition owns it, not an operand.
* **R8-P2** — the flat `VORTEX_VEC-zco` card's `ww` row is AT THE BAR at
  the same boundary, i.e. the error is topography-borne.  **FALSIFIER:**
  the flat card's `ww` is also over `1e-14` relative, which would make this
  a continuity statement with nothing to do with partial cells.
* **R8-P3** — the `out.T`/`out.S` last-bit rows (1 ULP on 1775/1621 cells)
  survive the transport substitution, so they are a SEPARATE, smaller
  statement in the qco stage step `stprk3_stg.f90:552-554`, not a residue
  of the transports.  Already measured: they do (1774/1620 cells after).
  Carried as a named, registered last-bit item, not walked.

## 11. OPEN

1. The `ww` row is named but NOT attributed to one operand (section 7).
2. RESOLVED in section 9: the flux card's record was admitted and its
   `ww` row carries `3.368e-13` relative, the same statement, so the
   preregistered falsifier did not fire.
3. The SMT deck cannot exercise `tra_zdf`'s partial-cell statement at all
   (`avt` identically zero, section 3).  **The mini-ladder's SMT-1 rung
   (Decision 93: rung-0 background mixing plus enhanced vertical diffusion)
   is what makes it measurable here** — until that rung exists, the only
   configuration on the lane that exercises `trazdf.f90:231-233` is ORCA2.
4. The round-213/215 open items are unchanged.

## 12. GATES

No production model file is edited by this round, so every card is inert by
construction and the gate set is the push gate on a measurement-only diff.
The gate lines are in section 13.
## 13. EVIDENCE

All under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/round7/`:
`predictions.md` (frozen before the instrument was written);
`acquisition_vec3.log` with the admitted vector record and its admission
JSON under `VORTEX_SMT_R7_VEC_R8_OMIP_L1_P3/tracer/`;
`acquisition_flux3.log` and `VORTEX_SMT_R7_OMIP_L1_P3/tracer/` for the flux
card; `walk_vec.json` (the eight scored rows, the four one-variable rows,
and each row's per-level structure) and `walk_vec_plant.json` (the
non-vacuity control).  Two earlier launches of the vector acquisition
(`acquisition_vec.log`, `acquisition_vec2.log`) were refused by the
launcher's own free-space and existing-directory checks and are kept.

The two NEMO configurations are NEW: `VORTEX_SMT_R7_OMIP_L1{,_P3}` and
`VORTEX_SMT_R7_VEC_R8_OMIP_L1{,_P3}`.  Rounds 1, 3, 5 and 6 build
directories were not moved, rebuilt or deleted.

## 14. NON-VACUITY AND CONTROLS, EACH NAMED

| control | what it would have caught | result |
|---|---|---|
| the step-10 NEMO restart against round 3's admitted run | a writer that changed NEMO's answer | BYTE-IDENTICAL |
| the checker's `header` plant | an admission that cannot fail | FIRED |
| the record's own boundary cross-check (`out_t`/`out_s` vs the older stage-1 state record) | a writer instrumented at the wrong line | 0 cells differ |
| `require_live` on every exposed row | a hook that silently went inert and handed back the plain step output | all rows live |
| the `adv.T` plant against the clean report | scoring that cannot react | VISIBLE |
| the flat `VORTEX_VEC-zco` card at the bar on the same kt=2 T row | a "partial-cell" finding that is really a scheme defect | AT BAR (`3.466e-16`) |

## 15. THE ADVERSARIAL REVIEW

Two fresh reviewers, both on the committed diff, neither of them the
author.

**Reviewer 1 — SHIP WITH FIXES.**  Confirmed by tracing the shipped NEMO
sources that the writer is read-only with a sound open/close state
machine, that the three call sites sit where `tra_adv_trp` has already
populated `zFu/zFv/zFw` on BOTH momentum programs, that the `out` site
covers the stage-1/2 explicit update AND stage-3's `tra_zdf`, and that the
record checker hard-codes nothing beyond the magic and the group names.
Four findings, all closed in commit `276839c14`:

| # | finding | disposition |
|---|---|---|
| 1 | the arm's vertical member round-trips `zfw/area_T` then `*area_T`, so a recorded `zFw` of zero would make the w half VACUOUS | **MEASURED, REFUTED**: recorded `zFw` peak `8.742244e+04`, nonzero on 34,391 of 43,659 cells; the walk now REFUSES unless it is live over the scored support, so the control lives in the probe |
| 2 | the transport-substitution arm — the row that carries the round — was the one exposure without `require_live` | FIXED; the arm is now seam-controlled like every other row |
| 3 | the rank-2 reader reinvented the shared halo stripper with a hard-coded width | FIXED; it calls `_strip2` |
| 4 | the plant perturbed the geometric centre without checking it is scored | FIXED; it perturbs a cell from the row's own active support |

Every number in sections 7 and 9 was RE-MEASURED after those fixes and is
unchanged; the new `zFw` control passes, `require_live` passes on the arm,
and the `adv.T` plant is still VISIBLE.

**Reviewer 2 — a SECOND fresh reviewer was launched on all four commits
(it was asked to attack the headline's self-referentiality, the writer's
read-only property, three of this receipt's own numeric/citation claims,
and the checker's hard-coded expectations).  It had not reported when the
battery slot came free, and this round landed on reviewer 1's verdict
with every one of its findings closed and re-measured.**  That is stated
here rather than implied: the landing rests on ONE recorded verdict, not
two.  Reviewer 2's findings, when they arrive, are round 8's first item.
