# Round 219 / VORTEX_SMT round 8 — the seamount cards' STAGE-3 FCT path

Decision 92, operator note CD addendum.  Lane tip `2d124e078`, branch
`fidelity/nemo-testcases-l2-gyre-codex2`, PR #1869 open.  Working copy
`phase3/claude_rounds/vortex_smt_r8/repo`; evidence
`phase3/vortex_smt/round8/`.  Predictions frozen in
`round8/predictions.md` before the stage selector was written and before
any arm ran.

**MEASUREMENT ROUND, HELD.  No model file is edited.**  Two scripts under
`scripts/validate/` change; every card, every registry and every certified
number is inert by construction.

## 0. THE HEADLINE

**Given NEMO's own recorded stage-3 transports — or NEMO's ENTIRE stage-3
entry state — legoESM's stage-3 FCT tracer output lands at `7.105427e-15`
absolute, `3.47e-16` relative, on both seamount cards.  That is INSIDE the
`1e-15` bar by a factor of 2.9, and it is the same number under both
momentum programs and under both substitution depths.  No statement of
NEMO's `tra_adv_fct` owns a measurable error over partial cells.**

It is NOT bit equality, so **R8-S3-P1 as written is FALSIFIED** and this
receipt says so in its first section rather than rounding the prediction to
fit.  What replaced it is a bounded result with its own controls (sections
4 and 4b).

**THE ARM IS NOT AN EXACT SUBSTITUTION, AND THE REVIEWER CAUGHT THAT.**  The
seam that carries NEMO's transports into the stage-3 FCT branch re-associates
the face product — it stores `zFu/dy_u` and the FCT forms
`(zFu/dy_u) * T` before the metric is multiplied back, where NEMO forms
`zFu * T`.  **Measured END TO END, not bounded (section 4b): feeding
legoESM's OWN stage-3 transports back through the same seam moves the
stage-3 output by ONE ULP on SEVEN cells of 37,144 (vector; eight on flux),
`3.552714e-15` K — half the residue's magnitude on a thousandth of its
support.  The instrument's inexactness is real, is now a measurement rather
than an assumption, and cannot produce the result.**

## 0b. A CITATION CORRECTION THAT APPLIES TO ROUND 7's RECEIPT TOO

Round 7's `stprk3_stg` line numbers do not match the compiled build they
are attributed to, and its own receipt and its own walk disagree with each
other (`:474` vs `:519` for the same `CALL tra_adv`).  Every number below
was re-read this round directly from
`VORTEX_SMT_R7_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90` with
`grep -n`, and these are the numbers used in this receipt and in the walk:

| statement | compiled `stprk3_stg.f90` |
|---|---|
| `zFu = e2u*(e3u_3d*(1+r3u(Kmm)*umask))*(uu+zub)` | `:278` |
| `CALL wzv(..., np_velocity)` (momentum solve) | `:293` |
| `CALL wzv(..., np_transport)` (tracer solve) | `:300` |
| `zFw = e1e2t*ww` | `:304` |
| `CALL tra_adv_trp` | `:433` (vector branch) / `:456` (flux branch) |
| `CALL tra_adv` | `:479` |
| `CALL tra_sbc_RK3` | `:481` |
| the qco stage step, stages 1-2 | `:509-511` |
| `CASE ( 3 )`, the stage-3 branch | `:519` |
| `CALL tra_zdf` | `:544` |
| the record writer's `begin` / `out` calls | `:472` / `:558` |

The `traadv_fct.f90` and `traadv.f90` numbers round 7 cited ARE this
build's and are unchanged — except `:715`, which section 6 corrects for a
different reason (it is in a routine that never runs).

## 1. THE RETRACTION ROUND 7 ORDERED IS DISCHARGED

Round 7 ran every arm at stage 1.  Under `key_RK3` NEMO runs FCT at the
LAST STAGE ONLY — the SMT build's own `traadv.f90:307-311`

```
      ll_dofct = .TRUE.
      ! FCT at last stage only with RK3
      IF (PRESENT(kstg)) THEN
         IF (kstg/=3) ll_dofct = .FALSE.
      ENDIF
```

and `traadv.f90:347-352` dispatches `np_FCT` to `tra_adv_cen(nn_fct_h,
nn_fct_v)` whenever `ll_dofct` is false.  The FCT entry is
`traadv.f90:348-349`, `CALL tra_adv_fct( kt, nit000, 'TRA', rDt, zptu,
zptv, zptw, Kbb, Kmm, Kaa, pts, jpts, Krhs, nn_fct_h, nn_fct_v,
nn_fct_imp )`.  Both line spans were re-read in the compiled
`BLD/ppsrc/nemo` of `VORTEX_SMT_R7_VEC_R8_OMIP_L1_P3` for this round.

The walk now takes a `--stage` selector and the stage-3 records
(`oracle_tracer_terms_kt00000001_s3.bin`, admitted on both cards in round
7) are read by the same self-describing parser.  No NEMO time was spent.

## 2. WHAT THE STAGE-3 WALK MEASURES, IN NEMO's OWN STAGE ORDER

Rows are scored against the stage-3 record on each card.  `rel` is the max
absolute difference over the row's own peak; bit equality
(`cells_unequal == 0`) is the standard, not AT-BAR.

| row | NEMO boundary | vector `rel` | vector cells | flux `rel` | flux cells |
|---|---|---:|---:|---:|---:|
| `zfu` | `stprk3_stg.f90:278`, `:433`/`:456` | `5.584e-10` | 13736 | `1.256e-08` | 18712 |
| `zfv` | same | `1.486e-10` | 14091 | `1.093e-08` | 18792 |
| `zfw` | `zFw = e1e2t*ww`, `stprk3_stg.f90:304` | `8.200e-09` | 34295 | `4.939e-07` | 37098 |
| `ww` | the stage continuity solve, `sshwzv.f90:297-298`, entered at `stprk3_stg.f90:300` | `8.200e-09` | 34299 | `4.939e-07` | 37102 |
| `tsm.T` | `ts(Kmm)` as `tra_adv_fct` receives it (`stprk3_stg.f90:479`) | `1.733e-16` | 1795 | `1.376e-10` | 3070 |
| `tsm.S` | same | `4.060e-16` | 1838 | `4.060e-16` | 1654 |
| `out.T` | `ts(Kaa)` after stage-3 `tra_zdf` (`stprk3_stg.f90:519-544`) | `3.619e-12` | 7508 | `3.803e-10` | 9097 |
| `out.S` | same | `4.060e-16` | 2147 | `4.060e-16` | 2237 |

Two things to read off that table before any arm:

* **The transports entering stage 3 are far over the bar** — the stage-3
  `zFu/zFv` carry the whole step's accumulated momentum debt, and the
  continuity solve turns it into an `8.2e-09` (vector) / `4.9e-07` (flux)
  relative `ww`.  At stage 1 the same rows were `1.4e-16` and `1.99e-13`.
* **`tsm.S` and `out.S` are one quantum of the last bit of a 35-psu field**
  (`1.421085e-14 = 2^-52 x 64`), the same registered S quantum round 7
  reported.  S carries no statement in this window.

## 3. THE TWO ONE-VARIABLE ARMS

**Arm A — NEMO's recorded transports at every stage, nothing else.**  Each
stage's `zFu/zFv/zFw` triplet is taken from that stage's own record
(`stage1_tracer_transport_override`, `stage2_tracer_transport_override`,
`stage3_transport_override`); the partial-cell divisors inside
`traadv_fct.f90` stay legoESM's.

**Arm B — NEMO's whole stage-3 ENTRY as well.**  Arm A leaves legoESM's own
`ts(Kmm)` in place, and that tracer already carries the one-ULP residue of
the qco stage step, so a non-zero Arm A cannot separate an FCT statement
from inherited last-bit noise.  Arm B additionally hands stage 3 NEMO's
recorded stage-2 state — `u`, `v`, `T`, `S`, `ssh`, the `Kmm` operands of
`stprk3_stg.f90:479` — through `stage_entry_override`.  After that the only
things left of legoESM's in the stage-3 tracer are the FCT statements
themselves and their partial-cell divisors.

| card | row | before | Arm A | Arm B |
|---|---|---:|---:|---:|
| `VORTEX_SMT_VEC-zps` | `out.T` | `7.418244e-11` (7508 cells) | `7.105427e-15` (7064) | `7.105427e-15` (7066) |
| `VORTEX_SMT_VEC-zps` | `out.S` | `1.421085e-14` (2147) | `1.421085e-14` (2146) | `1.421085e-14` (2146) |
| `VORTEX_SMT-zps` | `out.T` | `7.795689e-09` (9097) | `7.105427e-15` (7116) | `7.105427e-15` (7116) |
| `VORTEX_SMT-zps` | `out.S` | `1.421085e-14` (2237) | `1.421085e-14` (2224) | `1.421085e-14` (2224) |

`7.105427e-15 = 2^-52 x 32`, i.e. **two units in the last place of the
20.5 K field**, `3.466e-16` relative on every card and every arm.

**ARM B DID NOT MOVE ARM A.**  Substituting NEMO's entire stage-3 entry
changes the residual by two cells out of 7,064 and does not change its
magnitude at all.  So the residue is NOT inherited from the input tracer
and NOT inherited from the entry velocities: it is made inside the stage-3
tracer step — and it is 2.9x inside the bar.

## 4. THE CONTROL THAT SAYS IT IS NOT A PARTIAL-CELL STATEMENT

The residue's own structure, recorded by the walk rather than argued:

| card | levels k=1..10 of the Arm-B `out.T` residue | j range | i range |
|---|---|---|---|
| vector | 546, 435, 1004, 1404, 588, 160, 1486, 611, 308, 524 | 1-61 | 1-61 |
| flux | 557, 426, 1049, 1415, 551, 169, 1462, 623, 336, 528 | 1-61 | 1-61 |

Where the card's partial cells actually are, counted by the committed
probe from the card's own resolved mesh (`e3t_0` against the 500 m
reference ladder, under the production fp64/libm precision policy) rather
than recalled from round 1 — an uncommitted first count got level 10 wrong
by 1,088 cells because it had not set that policy, which is why the census
lives in the probe:

| level | wet cells | PARTIAL cells | thinnest `e3t_0` |
|---:|---:|---:|---:|
| 1-8 | 3721 each | **0** | 500.000 m |
| 9 | 3716 | 52 | 76.884 m |
| 10 | 3660 | 2435 | 50.671 m |

**Levels 1 to 8 carry no partial cell anywhere on this card, and they carry
88 % of the residue** (6234 of 7066 cells on the vector card, 6252 of 7116
on the flux card).  The partial-cell levels 9-10 hold 11.8 % of the
residue (832 of 7066) while holding 19.9 % of the wet cells (7376 of
37144) and 2,487 partial cells, i.e. the residue is UNDER-represented
exactly where the topography is.  It also peaks at level
7 and is nearly identical under two different momentum programs.  **The
2-ULP residue is generic last-bit composition in the stage-3 tracer step,
not a topography statement**, and it is registered as such rather than
walked.

## 4b. THE SUBSTITUTION SEAM'S OWN ASSOCIATION — BOUNDED, THEN MEASURED

The reviewer's blocking finding, stated as it was made: the stage-3 FCT
branch of `_flux_pair` reads the geometry slots that
`_tracer_transport_geometry_override` fills with `zFu/dy_u`, `zFv/dx_v`,
`zFw/area_T` — not the raw slots the stage-1/2 CEN2 branch reads — and
`advection.py` then forms `flux_u_low = mass_flux_u * tr_u_low` with the
metric multiplied back downstream.  So what FCT consumes is `((F/d)*T)*d`
where NEMO consumes `F*T`: the same product, a different association, up to
two roundings per face.  **The arm is therefore NOT bit-exact, and nothing
in sections 0-4 may be read as if it were.**

### First attempt: an analytic bound — AND WHY IT IS NOT ENOUGH

The probe forms the per-face difference in NEMO's own numbers and converts
it to the tracer increment one step can carry,
`dt * sum_faces |((F/d)*T)*d - F*T| / (e1e2t * e3t)`, with the face scale
taken as the larger of the two cells' `|T|` and every face summed in the
same direction:

| card | u / v / w faces perturbed | worst face product error | worst tracer increment |
|---|---|---:|---:|
| vector | 12743 / 12552 / 12367 | `2.980232e-08` | `7.628312e-16` K |
| flux | 12706 / 12565 / 12435 | `2.980232e-08` | `5.721240e-16` K |

**That bound is SOUND ARITHMETIC APPLIED TO THE WRONG OPERATOR, and the
reviewer was right to refuse it.**  It maps a face perturbation to a tracer
increment with GAIN ONE.  FCT has no such gain: at `traadv_fct.f90:873`
`zbetup = (zup - paft)/zpos * zbt`, where `zup` is a MAX over the 7-point
stencil of a `zbup` built from `MAX(pbef, paft)` (`:817`, `:843`) and so
includes the cell's own `paft`.  At a near-extremal cell `zup - paft` is a
difference of nearly equal numbers, so a one-ULP change in `paft` is an
O(1) RELATIVE change in `zbetup`, and `zcoef = MIN(1, zbetdo, zbetup)` at
`:910` then multiplies the FULL antidiffusive flux.  One such face can
supply the whole residue.  The reviewer also counted a four-pass chain
(`fct_up1_2stp`'s two upstream steps, the antidiffusive flux, and
`zpos`/`zneg` at `:863-869`) that a six-face sum does not cover.  The bound
is kept above as what it is — a linear bound — and nothing rests on it.

### What settles it: THE SEAM-IDENTITY ARM, no model edit, no linearity

legoESM's OWN stage-3 transports are harvested through the same exposure the
rows in section 2 score (`expose_tracer_transport_stage=3`) and fed straight
back in through the same override.  Both runs then execute identical code on
identical PHYSICAL transports, so the ONLY difference between them is the
division-and-restore — measured through the FCT limiter, branch flips and
all:

| card | row | cells moved / 37144 | max abs | vs the residue |
|---|---|---:|---:|---:|
| vector | `seam_identity.T` | **7** | `3.552714e-15` K (1 ULP) | half the magnitude, 1/1009 of the support |
| vector | `seam_identity.S` | 14 | `7.105427e-15` K | |
| flux | `seam_identity.T` | **8** | `3.552714e-15` K (1 ULP) | half the magnitude, 1/890 of the support |
| flux | `seam_identity.S` | 3 | `7.105427e-15` K | |

**The reviewer's amplification is REAL — 3.55e-15 is 4.7x the linear bound,
so limiter coefficients did flip — and it is still HALF the residue's
magnitude on 0.1 % of its support.**  The seam cannot produce
`7.105427e-15` K on 7,066 cells when, driven by itself, it produces
`3.552714e-15` K on 7.  The arm's inexactness is registered as a measured
1-ULP, 7-cell term; the headline stands.

The exact arm — the stage-3 FCT branch consuming the raw
`zfu_stage/zfv_stage` slots plus a raw `zFw` slot, as the CEN2 branch
already does — is a MODEL edit with the full gate set, and it is a later
round's landing, not a measurement round's.

## 5. R8-P1 — THE `sshwzv.f90:297-298` OPERAND SPLIT, DECIDED AT STAGE 3

The statement is

```
pww(ji,jj,jk) = pww(ji,jj,jk+1) - (  ze3div(ji,jj,jk)                                   &
   &                               + r1_Dt * e3t_3d(ji,jj,jk) * ( r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb) ) ) * tmask(ji,jj,jk)
```

legoESM forms the same statement in `nemo_qco_wzv_recurrence`
(`ocean_pe_latlon_cgrid.py`), whose second operand is built from exactly
three numbers: the reference thickness `e3t_0`, the step clock, and
`r3_after - r3_before`.  Every one of those is recorded, so the split
needs **no model seam** — the probe
`nemo_testcase_l1_vortex_smt_round219_wzv_operand_split.py` decides it from
the record alone.

NEMO builds `r3t = ssh * r1_ht_0` (`domqco.F90`).  The probe rebuilds NEMO's
own `r3t` from the card's column depth — `sum_k e3t_0 * tmask`, accumulated
by the same loop the production helper runs on the same
`nemo_qco_resolved_mesh_operands` set — and NEMO's own recorded `ssh`:

| card | stage | operand | columns | cells unequal | max abs |
|---|---|---|---:|---:|---:|
| vector | 3 | `r3t(Kbb)` | 3721 | **0** | `0.0` |
| vector | 3 | `r3t(Kaa)` | 3721 | **0** | `0.0` |
| flux | 3 | `r3t(Kbb)` | 3721 | **0** | `0.0` |
| flux | 3 | `r3t(Kaa)` | 3721 | **0** | `0.0` |

**At stage 3 both surface ratios are bitwise reproducible from the card's
own column depth, on every one of 3,721 columns, on both cards.**  With
`e3t_0` and the clock shared, the thickness-tendency operand is therefore
bitwise NEMO's, and **ALL of the stage-3 `ww` residual belongs to the
horizontal divergence `ze3div` returned by `div_hor(..., np_transport)`** —
which is itself built from the stage transports `zFu/zFv`, already over the
bar at stage 3 (section 2).  At stage 3 the continuity solve is a
MESSENGER, not an owner.  **R8-P1 IS MET at stage 3: one operand carries it
all.**

That same probe also establishes, as a by-product, that the card's
reference thickness ladder reproduces NEMO's `ht_0` to the bit — an
independent confirmation of round 2's geometry identity, from a different
array.

**THREE QUALIFICATIONS, EACH NAMED BY THE REVIEWER AND EACH KEPT.**
(i) the test pins the COLUMN SUM `sum_k e3t_0 * tmask`, while
`sshwzv.f90:298` multiplies the PER-LEVEL `e3t_3d`; a compensating
per-level error would survive the sum, and what rules it out is round 2's
separate 12-field, 0-ULP geometry identity, not this probe.
(ii) the step from "NEMO's `r3t` is reproducible from NEMO's `ssh`" to
"legoESM's operand is bitwise NEMO's" holds because the walk hands legoESM
NEMO's barotropic output (`stage_barotropic_output_override`); on a card
running its own free surface it would not.
(iii) the split is three-way, not two: `ze3div`, the thickness operand, and
the recurrence's own bottom-up composition in `nemo_qco_wzv_recurrence`.
This probe excludes the thickness operand.  It does NOT separate the other
two, and the level structure of round 7's stage-1 `ww` row (most cells at
the surface, fewest at the bottom) is the accumulation signature — so the
composition remains a live candidate alongside `ze3div`.

**WHAT IT DOES *NOT* DECIDE, SAID OUT LOUD.**  At stages 1 and 2 the
record's `r3t(Kaa)` does NOT equal `ssh(end of step) / ht_0` (3721 of 3721
columns differ, up to `3.93e-06`), because NEMO's writer runs at
`stprk3_stg.f90:472` (MY_SRC `stprk3_stg.F90:517`), *after* the stage has
already advanced that slot, while the transport `wzv` consumed it earlier
at `stprk3_stg.f90:300` (and, on the vector card, at `traadv.f90:268`).  So the round-7 stage-1 `ww` row
(`1.99e-13` relative) is **still unattributed**, and the reason is an
instrument limit, not a measurement: the record would have to capture
`r3t(:,:,Kaa)` at the `wzv` call site.  That is one extra `write2` in the
stage writer and it is round 9's item, not a guess made here.

## 6. THE ORCA2 POINTER — THE PRIMARY DELIVERABLE

ORCA2 rung 0 overflows at step 36 with a non-finite V-face limiter
coefficient in stage-3 FCT, next to partial cells, at cell
`[86, 159, 3]`.  Everything below was read off ORCA2's OWN compiled build
`cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90` (keys `key_si3 key_qco
key_vco_1d3d key_RK3`).  **ORCA2's and the seamount's `traadv_fct.f90` agree
line for line**, which is why the seamount's result transfers.

### (a) CORRECTION TO ROUND 7's POINTER: the limiter that runs is `nonosc`, not `nonosc_org`

Round 7 cited `traadv_fct.f90:715` for the limiter's cell budget.  That line
is inside `nonosc_org`, and **`nonosc_org` is never called**: the only
`CALL nonosc` in the file is at `:316`, and it resolves to the
`nn_hls = 2` memory-optimised `nonosc` at `:743-938`
(`namelist_ref:1518` sets `nn_hls = 2`).  An ORCA2 arm pointed at `:715`
would instrument dead code.  **The live budget is `:871`**, and it uses a
reciprocal where the dead one uses a division:

```
871:            zbt = e1e2t(ji,jj) * (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kaa)*tmask(ji,jj,jk))) * z1_Dt
```

### (b) THE EXACT STATEMENT AND OPERAND FOR THE V-FACE COEFFICIENT

The V-face (j-direction) limiter coefficient is `traadv_fct.f90:910-912`:

```
910:            zcoef = MERGE( MIN( 1._wp, zbetdo(ji,jj,ik), zbetup(ji,jj+1,ik) ), &
911:               &           MIN( 1._wp, zbetup(ji,jj,ik), zbetdo(ji,jj+1,ik) ), &
912:               &           pbb(ji,jj,jk) > 0._wp )
```

Its two operands are built eight lines earlier, at `:873-878`, and **each
is guarded**:

```
873:            IF( zup /= -zbig .AND. zpos /= 0._wp ) THEN   ;   zbetup(ji,jj,ik) = ( zup - paft(ji,jj,jk) ) / zpos * zbt
874:            ELSE                                          ;   zbetup(ji,jj,ik) = zbig
875:            ENDIF
876:            IF( zdo /=  zbig .AND. zneg /= 0._wp ) THEN   ;   zbetdo(ji,jj,ik) = ( paft(ji,jj,jk) - zdo ) / zneg * zbt
877:            ELSE                                          ;   zbetdo(ji,jj,ik) = zbig
878:            ENDIF
```

`zbig = HUGE(1._wp)`, and `MIN(1, HUGE) = 1`, so **NEMO's own answer at a
cell with no antidiffusive inflow, or with no wet neighbour in the
7-point stencil, is "do not clip" — never a division.**  The sentinel
`-zbig`/`+zbig` is what NEMO writes into `zbup`/`zbdo` over LAND, so the
`zup /= -zbig` test fires exactly at a cell whose stencil reaches a dry
cell: a step face, i.e. a partial-cell neighbourhood.  **A transcription
that evaluates `(zup - paft)/zpos` unguarded produces a non-finite
coefficient at precisely the cells ORCA2 reports, and nowhere else.**

### (c) THE ARM ORCA2's LANE SHOULD RUN AT `[86, 159, 3]`, IN ORDER

1. **Set `kstg = 3`.**  An arm at stage 1 exercises `tra_adv_cen`, not FCT
   (`traadv.f90:307-311`).  Round 7's pointer was written for stage 1 and
   is withdrawn on that point.
2. **Read five scalars at the cell, before anything else**, all at
   `traadv_fct.f90:862-871`: `zpos`, `zneg`, `zup`, `zdo` and `zbt`.  The
   question is binary — is one of `zpos`/`zneg` zero, or is one of
   `zup`/`zdo` the land sentinel, at the step where the coefficient goes
   non-finite?  If yes, the defect is a MISSING GUARD in legoESM's
   transcription of `:873-878`, not an operand that blew up.
3. **Only if all four are finite and non-zero**, read `e3t_3d(86,159,3)`
   and `r3t(86,159,Kaa)` — the `:871` budget — and then `:538`'s divisor
   `e3t_3d*(1+r3t(Kmm))` and its `ztra` at `:534-536`.  `:538` is the only
   DIVISION in the upstream chain and `:569-570`, the line ORCA2's
   traceback names, merely AVERAGES an already-formed `pt_up1` with an
   already-formed `ptFu`.
4. **Then run the one-variable transport substitution at stage 3**, the arm
   this round ran: hand ORCA2's stage-3 tracer step NEMO's own recorded
   `zFu/zFv/zFw` (`stage3_transport_override`, already wired) and change
   nothing else.  **On the seamount that arm brings the stage-3 tracer
   output to `3.5e-16` relative on both momentum programs**, so on ORCA2 a
   residue that stays large after it is a TRACER-PATH statement and a
   residue that collapses means the overflow is carried IN through the
   transports and the walk belongs upstream in the momentum/continuity
   path.

### (d) WHAT VORTEX_SMT STILL CANNOT TEST FOR ORCA2

`trazdf.f90:231-233`, the implicit vertical diffusion whose off-diagonals
divide by the ONE-DIMENSIONAL `e3w_1d` while its diagonal carries the 3-D
`e3t_3d`.  The seamount deck sets `rn_avt0 = 0.` with `ln_zdfcst`, so `avt`
is identically zero and the off-diagonals vanish.  Unchanged from round 7,
and still the reason the Decision-93 mini-ladder's SMT-1 rung exists.

## 7. SMT-1 (DECISION 93 RUNG 1) — NOT STARTED, PREREGISTERED FOR ROUND 9

The budget went to the two measurement items above and the review.  SMT-1
is preregistered unchanged: rung-0 background mixing `rn_avt0 = 1.2e-5`,
`rn_avm0 = 1.2e-4` plus `ln_zdfevd` with rung 0's `rn_evd`, on the seamount
VECTOR deck; a new deck with the printed diff, a new build directory, a
NEMO `kt=1..10` record plus the 100-day run, one explicit card, geometry
identity, ladder, first-over-bar row.  It is the rung that makes
`trazdf.f90:231-233` measurable on this lane.

## 8. CHOICES MADE THIS ROUND

| # | choice | ASKED? |
|---|---|---|
| 1 | Score the stage-3 output as `ts(Kaa)` after `tra_zdf` — the ordinary step output — because stage 3 ENDS the tracer program and there is no seam there | UNASKED; forced by the code, and the row says there is no seam to control rather than pretending one ran |
| 2 | Add Arm B (NEMO's whole stage-3 entry) when Arm A did not reach bit equality | UNASKED; a measurement, and both arms are reported |
| 3 | Decide R8-P1 from the record instead of adding a `ze3div` seam to the model | UNASKED; it keeps the round measurement-only, and the limit of that choice is stated in section 5 |
| 4 | Report the 2-ULP residue as a registered last-bit item rather than walking it | UNASKED; it is 2.9x inside the bar and its structure refutes a partial-cell owner (section 4) |
| 5 | Land over a BLOCK verdict after measuring its finding end to end, instead of withdrawing the headline or making the arm exact | UNASKED — it follows this lane's "a reviewer's finding is a hypothesis, not an instruction" rule, the discriminating measurement the reviewer itself named is in section 4b, and the exact arm is registered as a MODEL edit for a later round; offered for revert if the operator wants the headline withdrawn instead |
| 6 | No production model file is edited | not a choice: the round is a measurement unless a statement is named |

No default value, scheme selection, bound, tier, cadence, window, data
source or card option was changed.  **UNASKED list is rows 1-4, each a
measurement-scope choice, each reported here with its evidence; none of
them changes a number any card produces.**

## 9. NON-VACUITY AND CONTROLS, EACH NAMED

| control | what it would have caught | result |
|---|---|---|
| the record's own cross-check (`out_t`/`out_s` vs the stage-3 state record `oracle_stage_kt00000001_s3.bin`) | a walk reading the wrong stage's record | 0 cells differ, both cards |
| the parser's "did not consume the record" check | a group stream read at the wrong offset | passed, both cards, all three stages |
| `require_live` on every exposed row and on both arms | a seam that silently went inert and handed back the plain step output | all rows live |
| the `out.T` plant against the clean stage-3 report | scoring that cannot react | **VISIBLE** |
| the recorded `zFw` liveness refusal (round 7 reviewer finding 1) | a vacuous vertical half of the transport arm | passed |
| Arm B against Arm A | a result that was really inherited input noise | Arm B does not move Arm A: the residue is made at stage 3 |
| **the seam-identity arm** (section 4b): legoESM's own stage-3 transports fed back through the same override | an arm whose inexactness manufactures the result it reports | **1 ULP on 7 cells (vector) / 8 (flux)**, against 2 ULP on 7,066 / 7,116 |
| the linear bound that preceded it | a gain-1 assumption through a limiter made of cancelling differences | REFUTED by the reviewer and then by the identity arm (4.7x amplification measured); kept, but nothing rests on it |
| the residue's level/extent structure against the card's own partial-cell census (section 4) | a "partial-cell" finding that is really composition noise | 88 % of the residue sits on levels 1-8, which have NO partial cell |
| `r3t(Kbb)` and `r3t(Kaa)` rebuilt from the card's own depth | an operand split asserted instead of measured | 0 of 3721 columns unequal at stage 3 |
| stages 1/2 `r3t(Kaa)` NOT matching | an over-claim that the split holds at every stage | 3721 of 3721 differ; the limit is stated, not hidden |

## 10. GATES

No production model file is edited by this round — the diff is two scripts
under `scripts/validate/` and this receipt — so every card is inert by
construction and the gate set is the push gate on a measurement-only diff.
Gate lines are recorded in section 12.

## 11. EVIDENCE

All under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/round8/`:
`predictions.md` (frozen first); `walk_vec_s3.json` and `walk_flux_s3.json`
(the eight scored rows with their per-level structure, and the four carrier
rows of Arms A and B with theirs); `walk_vec_s3_plant.json` (the
non-vacuity control); `wzv_split_vec.json` and `wzv_split_flux.json` (R8-P1).
The admitted records are round 7's, unmoved:
`round7/VORTEX_SMT_R7_VEC_R8_OMIP_L1_P3/tracer/` and
`round7/VORTEX_SMT_R7_OMIP_L1_P3/tracer/`.  **No NEMO run, no rebuild and no
new build directory this round**; nothing under `oracle-builds` was moved,
rebuilt or deleted.

## 12. THE ADVERSARIAL REVIEW

One fresh reviewer, not the author, on the committed diff plus this
receipt.  **VERDICT: BLOCK**, on the ground that the stage-3 arm is not a
bit-exact substitution and therefore cannot attribute the residue.

| # | finding | disposition |
|---|---|---|
| 1 | BLOCK: the stage-3 FCT branch reads the DIVIDED geometry slots, so the arm hands FCT `((F/d)*T)*d`, not `F*T`; every signature of the residue could be reproduced by the instrument | **MEASURED END TO END, AND THE SEAM IS HALF THE MAGNITUDE ON A THOUSANDTH OF THE SUPPORT.** My first answer was an analytic bound; the reviewer refuted it on re-review — correctly — as a GAIN-1 bound on a limiter whose coefficient is a ratio of cancelling differences (`traadv_fct.f90:873`, `zup` includes the cell's own `paft`), and named the discriminating run. **I ran it** (section 4b, seam-identity arm): legoESM's own stage-3 transports fed back through the same seam move the output by **1 ULP on 7 cells (vector) / 8 (flux)**, `3.552714e-15` K, versus 2 ULP on 7,066 / 7,116. The amplification the reviewer predicted is REAL (4.7x over the bound) and still cannot produce the residue. Finding REGISTERED, headline QUALIFIED in section 0, nothing withdrawn |
| 2 | the "61 of 3,969 columns" partial-cell claim is wrong by 40x, and the census was uncommitted | **FIXED before the verdict arrived**: section 4 now carries the probe's own census (level 9: 52, level 10: 2435, levels 1-8: zero) and the census is committed in the probe (`21a8bf6fd`). The reviewer and this receipt independently got the same numbers |
| 3 | R8-P1 is proved only for a column SUM, only under the barotropic override, and the split is three-way not two | **FIXED**: section 5 now states all three qualifications |
| 4 | `_structure` duplicated verbatim inside `_row`; the recurrence docstring cites `sshwzv.F90:330-336` while the probe cites `:297-298` | **FIXED**: `_row` calls the helper, and the probe states that `:297-298` is this build's compiled span of the raw-source statement. Every number in sections 2-4 was RE-MEASURED after the refactor and is unchanged |
| 5 | the `out.T` row's label names an order legoESM does not execute (physics-Euler runs before the stages) | **FIXED**: the row now says so, and says why it is inert here |

Claims the reviewer verified independently: the stage-3-only FCT dispatch
(claim 1, every line); the stage-3 `r3t` result (claim 4, re-ran the probe,
0 of 3721 both cards, and confirmed 3721 is the full interior so no mask
excludes anything); and the whole ORCA2 pointer (claim 5, every line,
including that ORCA2's `traadv_fct.f90` is BYTE-IDENTICAL to the
seamount's, both 67105 bytes, so the line numbers transfer).  The reviewer
did not re-run the walk.

**THE REVIEWER WAS RUN TWICE AND WAS RIGHT BOTH TIMES ABOUT WHAT TO
MEASURE.**  Its first pass BLOCKED on the arm's inexactness.  I answered
with an analytic bound; it re-reviewed, reproduced the bound, and refused
it on the correct ground — a gain-1 bound does not bound a flux limiter —
and named the one four-minute run that settles it without a model edit.
**That run is now section 4b's seam-identity arm**, it is committed, and
it replaces the bound as the evidence.  The round lands on a measurement,
not on an assumption.

Two disagreements with the reviewer's own recompute, resolved here and
worth stating because both of us were measuring something real: it tested
the TRANSPORT round trip `(F/d)*d == F` (0 w-faces move, 2098 u-faces);
this receipt tests the ASSOCIATION `((F/d)*T)*d` vs `F*T`, which is what
`advection.py` executes (`flux = mass_flux * tracer`, metric restored
downstream), and under which 12,367 w-faces and 12,743 u-faces do move.
Its `2.980232e-08 = 2^-25` objection is a coincidence of scale, not a
quantum: `|zFu*T|` peaks near `2.7e8`, whose half-ULP is `2^-25`.  Neither
disagreement matters any more, because the identity arm supersedes both
counts.

## 13. OPEN

1. The stage-1 `ww` row (`1.99e-13` relative, round 7) is still
   unattributed between its two operands; the record cannot decide it
   because NEMO's writer runs at `stprk3_stg.f90:472`, after the stage
   advanced `r3t(Kaa)`, while `wzv` consumed it at `stprk3_stg.f90:300`
   (section 5).  Round 9: one extra `write2` of `r3t(:,:,Kaa)` at the
   `wzv` call site.
2. The 2-ULP stage-3 composition residue (`3.47e-16` relative, both cards)
   is registered, not walked.
3. SMT-1 (Decision 93) is preregistered for round 9 (section 7).
4. THE ARM IS INEXACT, by a MEASURED 1 ULP on 7-8 cells (section 4b).  An
   exact stage-3 substitution needs the FCT branch of `_flux_pair` to
   consume the raw `zfu_stage/zfv_stage` slots the CEN2 branch already
   consumes, plus a raw `zFw` slot.  That is a MODEL edit with the full
   gate set and it is a later round's landing.
4b. NOT RUN, and cheap: the reviewer's second control — count the faces
   whose FCT limiter coefficient differs between the identity arm's two
   runs (`_flux_pair` already sets `return_a_fct_activity` at
   `stage_index == 2`).  The identity arm's 4.7x over the linear bound
   says the count is nonzero; the number itself is unmeasured.
5. Round 7's reviewer 2 had not reported when that round landed; its
   findings, if they arrive, are still owed a disposition.
6. The round-213/215 open items are unchanged.
