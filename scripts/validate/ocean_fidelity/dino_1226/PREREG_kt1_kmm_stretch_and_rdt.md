# PREREG — the stretch's TIME LEVEL in `tra_ldf`, and `rDt` on the leap-frog card

Written BEFORE the code change, as required. Two claims, two tables, one
falsifier each. Every NEMO row cites the COMPILED source the DINO binary runs
(`cfgs/DINO/BLD/ppsrc/nemo/*.f90`) with the `src/OCE` original where the
statement is unchanged by the preprocessor.

---

## CLAIM 1 — every geometric operand of NEMO's `tra_ldf` is at `Kmm`, and legoESM builds three of them at `Kaa`

### 1a. Which time level is `Kmm` during the tracer stage

DINO does NOT compile `src/OCE/stpmlf.F90` — it compiles
`cfgs/DINO/MY_SRC/stpmlf.F90`, and the line numbers below are therefore taken
from the COMPILED file `cfgs/DINO/BLD/ppsrc/nemo/stpmlf.f90`. (An earlier draft
of this table cited `src/OCE`; the physics was identical but the provenance
rule was not followed, and a claim reviewer caught it.)

`stpmlf.f90:504` calls `tra_ldf( kstp, Nbb, Nnn, ts, Nrhs )` — so inside the
routine `Kbb = Nbb` and `Kmm = Nnn`.

`r3t(:,:,Nnn)` is NOT written anywhere between step entry and `tra_ldf`:

| where (ppsrc `stpmlf.f90`) | statement | writes |
|---|---|---|
| `:233` | `dom_qco_r3c( ssh(:,:,Naa), r3t(:,:,Naa), r3u(:,:,Naa), r3v(:,:,Naa) )` | **Naa** only |
| `:235` | `dom_qco_r3c( ssh(:,:,Nnn), r3t(:,:,Nnn), … , r3f )` | Nnn — but guarded `ln_dynspg_exp`, and DINO's `namelist_cfg:351-352` sets `ln_dynspg_exp=.false.`/`ln_dynspg_ts=.true.` ⇒ **dead** |
| `:350` | `dom_qco_r3c( ssh(:,:,Naa), r3t(:,:,Naa), r3u, r3v, r3f )` after `dyn_spg` | **Naa** only |
| **`:425`** | **`ssh_atf( kstp, Nbb, Nnn, Naa, ssh )` → `sshwzv.f90:445` `pssh(:,:,Kmm) = pssh(:,:,Kmm) + rn_atfp*( … )`** | **`ssh(:,:,Nnn)` IS overwritten here, BEFORE `tra_ldf` at `:504`** |
| `:426` | `dom_qco_r3c( ssh(:,:,Nnn), r3t_f, r3u_f, r3v_f )` | a SEPARATE array `r3t_f` |
| `:571` | `r3t(:,:,Nnn) = r3t_f(:,:)` | Nnn — but AFTER `tra_atf_qco` at `:568`, i.e. after the whole tracer stage |

So at `tra_ldf`, `r3t(Kmm)` is the **step-entry, UNFILTERED** ratio.

**The `:425` row was missing from the first draft of this table and a claim
reviewer supplied it. It matters, and it names the invariant this change
actually rests on.** From `:425` to `:571` NEMO deliberately holds two
different "now" heights: `ssh(:,:,Nnn)` is Asselin-filtered, `r3t(:,:,Nnn)` is
not, and `tra_ldf` at `:504` sits between them. legoESM passes an *eta*, not an
*r3t*, so the two are only equivalent because **legoESM's Asselin filter on eta
runs at the END of the step, after the whole tendency pass**
(`ocean_model_latlon_cgrid.py:11097-11103`, `eta_f = _asselin(...)`), and the
filtered value it stores becomes the NEXT step's `state.eta` — which is exactly
what NEMO's `:426`→`:571` `r3t_f` handoff does. That invariant is now cited
here instead of being assumed.

**At kt=1 this cannot be tested.** `sshwzv.f90:443` guards the filter with
`IF( .NOT.l_1st_euler )`, so `ssh_atf` is a no-op on the from-rest first step
and the two heights coincide. The kt=2 record is the discriminator for this as
well as for `rDt`.

### 1b. The operand table — NEMO `traldf_iso.f90` (ppsrc) vs legoESM

Operator entered with `msc_stabilize=T` (`namelist_cfg:267 ln_traldf_msc=.true.`).

| # | quantity | NEMO statement (ppsrc `traldf_iso.f90`) | NEMO level | legoESM builds it from | legoESM level | verdict |
|---|---|---|---|---|---|---|
| N1 | `zA11` (u-face `e3u`) | `:231` `e2_e1u * (e3u_3d*(1+r3u(ji,jj,Kmm)*umask))` | **Kmm** | `face_thickness_u` ← `redi_flux_eta = state.eta` | Nnn | MATCH |
| N2 | `zA22` (v-face `e3v`) | `:232` `e1_e2v * (e3v_3d*(1+r3v(ji,jj,Kmm)*vmask))` | **Kmm** | `face_thickness_v` ← same | Nnn | MATCH |
| N3 | A33 vertical-flux divisor `e3w` | `:284` `e1e2t / (e3w_3d(jk+1)*(1+r3t(ji,jj,Kmm))) * wmask` | **Kmm** | `nemo_iso_a33_e3w(z_coord, e3t, jacobian, …)`, `jacobian` ← `state_new.eta` | **Kaa** | **DIFF** |
| N4 | tendency divisor `e3t`, interior `jk` branch | `:292` `r1_e1e2t / (e3t_3d*(1+r3t(ji,jj,Kmm)*tmask))` | **Kmm** | `e3t = dz_ref * jacobian` (gm_redi_latlon_cgrid.py:2430) | **Kaa** | **DIFF** |
| N5 | tendency divisor `e3t`, `jk == jpkm1` branch | `:305` the same statement in the bottom-level branch (T and S share ONE `DO jn` loop — an earlier draft called these the T and S halves, which is wrong) | **Kmm** | same object | **Kaa** | **DIFF** |
| N6 | `ze3w_2` inside `akz` | `:823`,`:828` `(e3w_3d*(1+r3t(ji,jj,Kmm)))**2` | **Kmm** | same `nemo_iso_a33_e3w` → same `jacobian` | **Kaa** | **DIFF** |
| N7 | the slopes `uslp/vslp/wslpi/wslpj` | carried from `ldf_slp( … Nbb, Nnn )` (`stpmlf.F90:199`) | Kmm geometry, Kbb density | `native_slope_eta = state.eta`, `native_prd_jacobian` ← `eta_before` | Nnn / Nbb | MATCH |
| N8 | the tracer `pt(:,:,:,jn,Kbb)` | `:281` etc. | Kbb | `_T_gm_in`/`_S_gm_in` = Nbb under `nemo_mlf` | Nbb | MATCH |

**So the change is exactly N3–N6: the operator's own volume geometry moves from
`Kaa` to `Kmm`.** Nothing else. The density and the slopes already carry their
own explicitly-threaded time levels and are untouched — which is what keeps this
a ONE-VARIABLE change against the recorded A0 row.

### 1b'. SCOPE — corrected by the claim review, and it is WIDER than the first draft said

The first draft said "the change is exactly N3-N6, nothing else, and it is
inert unless the caller passes the new height". **Both halves of that are
false, and the reviewer proved it from the code:**

1. Both model call sites pass `redi_kmm_eta` **unconditionally**, so nothing
   about this is opt-in. That is deliberate (the user's instruction for this
   round is "land it in the shared step, no knob"), but it must be stated as
   what it is.
2. DINO selects `redi_flux_face_thickness_evaluation='nemo_qco_live'`
   (`experiments/dino.py:1386`), where the u/v flux faces were ALREADY on the
   Kmm height, so on DINO exactly N3-N6 move. The library DEFAULT is
   `'tpoint_jacobian'` (`config.py:414`), where
   `e3u_flux = e3v_flux = e3t` — so on any OTHER `nemo_iso_lap` card **N1 and
   N2 move too**: six operands, not four.
3. `compute_ocean_jacobian` is `(1+r3t)` only on an
   `OceanPartialCellCoordinate`, which is what the DINO card builds
   (`experiments/dino.py` `create_partial_cell_coordinate`). On an
   `OceanZStarCoordinate` it is `(eta+H_bathy)/H_max` — still the live
   thickness for that ladder (`sum(dz_ref) == H_max`), but a different formula
   from NEMO's `r3t`. Moving its TIME LEVEL is the same statement; claiming it
   is "NEMO's `(1+r3t)`" would not be.

Rule 12 therefore requires the card sweep to enumerate every `nemo_iso_lap`
card, not only the two that turn the stabilising correction on.

### 1c. Pre-registered prediction, and the falsifier

The kt=1 ladder already measured the N3 operand alone (arm `T1`, commit
`23dc6b26`):

```
A0 as the model runs it   T 1.000005  rms 7.716e-13   S 1.000005  rms 1.348e-13
T1 e3w at Kmm=Nnn         T 1.000002  rms 3.769e-13   S 1.000002  rms 6.679e-14
```

A claim reviewer rejected the first draft's P1 ("at most the T1 value … may go
further") as unfalsifiable — it admitted every value in `[1.0, 1.000002]`.
Replaced with numbers that can be wrong:

* **PREDICTION P1 (falsifiable)**: the landed T ratio lands in
  `[1.0000000, 1.0000030]` and the landed T residual rms lands in
  `[0, 4.0e-13]`. Anything outside either interval refutes it. The T1 arm
  already moved N3 alone from `1.000004833 / 7.716e-13` to
  `1.000002 / 3.769e-13`; N4/N5 are the same `(1+r3t(Kmm))` on the same cells,
  so the landed value must not be WORSE than T1's and must not overshoot below
  1.0 by more than the 4.1e-5 relative size of the operand change.
* **PREDICTION P2**: it cannot be worse than A0. The two stretches differ by at
  most 4.1e-5 relative (the gate prints both spans) and the operator is
  first-order in the operand.
* **FALSIFIER**: the landed ratio moves AWAY from 1.0, or the residual rms
  exceeds A0's `7.716e-13`. Then these operands are not one object and the
  change is reverted.
* **CONTROL (rebuilt; the first draft's was unrunnable).** The reviewer pointed
  out that after the fix there is no "N3 alone" arm left to reproduce, because
  the diff deliberately gives all four operands ONE argument. The control is
  therefore a REGRESSION WITNESS in the other direction: the gate's new `V0`
  arm feeds the operator the post-barotropic `Naa` jacobian the step itself
  built (captured from the step, not rebuilt), and **must reproduce round 3's
  recorded `1.000004833 / 1.000004723` to 2e-8**. `W1` restores `Naa` AND the
  pre-fix midpoint `e3w` and must reproduce the original `0.996571 / 0.996696`.
  Two witnesses, two recorded bars, one statement each.
  `--plant-kmm` feeds `V0` the Kmm jacobian while labelling it Naa; the gate
  must then refuse.

### 1d. UNMEASURED by this record, stated up front

`akz` is identically zero on all 342134 wet cells of the kt=1 from-rest record
(measured, commit `23dc6b26`), so **N6 is multiplied by nothing this record can
see**, and the implicit half of the explicit/implicit split is likewise inert.
N6 is landed as a transcription fix on the cited statement, not as a residual
this record scores.

---

## CLAIM 2 — `rDt`, and the RETRACTION it forces

### 2a. What NEMO does

| where | statement | value |
|---|---|---|
| `domain.F90:288` | `rDt = 2._wp * rn_Dt` | MLF: **2·rn_Dt** |
| `stpmlf.F90:114-116` (ppsrc `:131-133`) | `IF( l_1st_euler ) THEN ; rDt = rn_Dt` | kt=nit000 from rest: **rn_Dt** |
| `stpmlf.F90:467-470` (ppsrc `:617-620`) | `rDt = 2._wp*rn_Dt ; l_1st_euler = .FALSE.` | restored at the END of that first step |

DINO `namelist_cfg:116` `rn_Dt = 2700.`, so rDt = 2700 s at kt=1 and 5400 s
from kt=2 on.

### 2b. Every `rDt` consumer the DINO card executes, and what legoESM passes

| # | NEMO consumer | statement | legoESM operand | verdict |
|---|---|---|---|---|
| R1 | `tra_ldf` → `akz` threshold | `traldf_iso.f90:829-830` `zcoef0 = rDt*(pakz + pah_wslp2/ze3w_2)` | `dt=dt` into `gm_redi_tracer_tendency_latlon`, where the step's `dt` is `_step_impl(state, rdt, …)` and `rdt = (1.0 if euler_start else 2.0)*dt` (`ocean_model_latlon_cgrid.py:10880`, `:10913`) | **MATCH — see retraction** |
| R2 | `tra_zdf` implicit solve | `trazdf.f90:99` `tra_zdf_imp('TRA', rDt, …)`; `:115`,`:119` `…*rDt` | `_apply_implicit_vmix(naa_expl, rdt, …)` (`:11009`) | MATCH |
| R3 | `tra_adv` FCT | `traadv.f90:267` `tra_adv_fct(…, rDt, …)` | same `_step_impl(state, rdt, …)` pipeline | MATCH |
| R4 | `dyn_zdf` | `dynzdf.f90:159-168`, `:303-310` | `dt_mom = rdt / cfg.dt_mom_ratio` (`:11002`) | MATCH **iff `dt_mom_ratio == 1`** — printed by the gate, not assumed |
| R5 | `ssh_nxt` | `sshwzv.f90:140` `pssh(Kaa) = pssh(Kbb) - rDt*(…)` | the barotropic/ssh stage of the same `_step_impl` | MATCH |
| R6 | `dyn_spg_ts` barotropic substep | `dynspg_ts.f90:1203` `rDt_e = rn_Dt / nn_e` — **`rn_Dt`, not `rDt`** | `_barotropic_substep_scale = 1 if euler_start else 2` (`:10909`) | MATCH (already certified by the substep ladder) |
| R7 | `wzv` restart term | `sshwzv.f90:527` `zdt = 2._wp*rn_Dt` with the comment *"MLF: 2\*rn_Dt and not rDt (for restartability)"* | — | **UNMEASURED**: legoESM's equivalent is not enumerated here |

### 2c. RETRACTION

Commit `4731822a` registered, and PR #1728's last comment repeated, that
*"NEMO's a33 uses `rDt` … while the card passes the base `dt`"*, listed as an
UNMEASURED row the kt=1 record could not score. **That is wrong.** Both
leap-frog entries already scale the timestep before the tendency pipeline sees
it — `rdt = (1.0 if _euler_start else 2.0) * dt`, and the tendency pipeline is
entered as `self._step_impl(state, rdt, …)`, so the `dt` the isoneutral
operator receives IS `rDt`. The row is retracted here rather than carried.

The gate now PRINTS the `dt` the captured `tra_ldf` call received next to
`rn_Dt` and `2·rn_Dt`, so this is a measurement and not a second reading of the
source.

### 2d. What the kt=1 record still cannot settle

At kt=1 `rDt == rn_Dt`, so R1–R5 are all **degenerate** on this record: a card
that wrongly used `rn_Dt` at every step would score identically here. The
discriminating measurement is a **kt=2** record, where `rDt = 2·rn_Dt` and the
`akz` threshold `MAX(zcoef0 - 0.5, 0)` can actually fire. `DINO_KT2_TRENDS`
(`nemo_dino_kt2_trends/run.sh`) acquires it. Until that record exists every R
row above is CONFIRMED-BY-CITATION and UNMEASURED-BY-RECORD, and this file says
so rather than letting the table read as a pass.

---

## Rule 12 — the per-card disposition this change must satisfy

The change is scoped to the `nemo_iso_lap` branch of
`gm_redi_tracer_tendency_latlon` and to `compute_isoneutral_K33_latlon`, and is
inert unless the caller passes the new Kmm sea-surface height. Required before
landing:

1. `ldf_e3w_card_sweep.py` (extended) enumerates every recipe/YAML/test-case
   card and prints which ones reach the changed block.
2. The kt=1 from-rest step-1 snapshot re-measured on the DINO card.
3. Every OTHER card fingerprinted byte-identical, or its moved rows registered.
4. The certified 90-day twin measured byte-unchanged, or its rows registered.

## What the two claim reviews killed, and what survived

Two fresh independent Claude agents (codex is on the GYRE lane, GLM
unavailable), both run on this file BEFORE the code existed.

| finding | disposition |
|---|---|
| `ssh_atf` (`stpmlf.f90:425`) overwrites `ssh(:,:,Nnn)` before `tra_ldf` and was missing from the table | **ACCEPTED** — table row added, and the legoESM invariant it rests on is now cited (`:11097-11103`), together with the fact that kt=1 cannot test it |
| every `stpmlf` citation pointed at `src/OCE`, but DINO compiles `cfgs/DINO/MY_SRC` | **ACCEPTED** — all line numbers retargeted at `BLD/ppsrc/nemo/stpmlf.f90` and re-verified |
| N4/N5 are the interior and bottom `jk` branches, not "T half / S half" | **ACCEPTED** — relabelled; the statement and the fix are unchanged |
| "exactly N3-N6, nothing else" and "inert unless the caller passes it" are both false off the DINO card | **ACCEPTED** — §1b' added; the sweep is widened to every `nemo_iso_lap` card |
| P1 could not fail | **ACCEPTED** — replaced with two numeric intervals |
| the CONTROL was unrunnable after the fix | **ACCEPTED** — replaced by the `V0`/`W1` regression witnesses with their recorded bars, plus `--plant-kmm` |
| `msc_stabilize=T` was asserted from NEMO's namelist, never from legoESM's resolved config | **ACCEPTED** — the gate prints the resolved value |
| CLAIM 1 (`r3t(Kmm)` is the step-entry ratio; every `traldf_iso` operand is Kmm) | **SURVIVES** — reviewer independently grepped the whole operator body: every `r3` is `Kmm`, every `pt` is `Kbb`, nothing else is time-indexed |
| DINO runs the `traldf_iso_lap` copy at `:231`; the `:457`/`:633` twins are inside `traldf_iso_blp` and dead | **SURVIVES, and it is new** — `ln_traldf_lap=T`/`_blp=F`/`_iso=T` ⇒ `nldf_tra = np_lap_i` ⇒ `traldf.f90:110` |
| CLAIM 2's retraction (the card already passes `rDt`) | **SURVIVES** — `_step_impl`'s second positional is `dt`, both leap-frog entries pass `rdt`, nothing rescales in between |
| `_euler_start = state.u_before is None` is NOT equivalent to NEMO's `l_1st_euler` on a RESTART (`domain.f90:383-415` also forces Euler on a stored-`rdt` mismatch, on a missing `sshb`, and on `ln_1st_euler`) | **ACCEPTED as a NEW finding, not fixed here** — it is a restart-path defect, this round runs from rest, and fixing it needs a restart record. Registered. |
| the from-rest spread floor may be dead on arrival: the certified verdict-360 record shows the NEMO/legoESM spread growing 2200x between day 90 and day 360 while the from-rest gap grows only 1.8x | **ACCEPTED** — the harness now runs a legoESM-only 4-member pre-check FIRST and refuses to spend the NEMO members unless the floor is in a preregistered window |
| NEMO members need no source patch: `nn_pert_seed` already exists in DINO's `usrdef_istate` and in the compiled `ppsrc` | **ACCEPTED and independently verified** (`BLD/ppsrc/nemo/usrdef_istate.f90:188-194`, `MY_SRC/usrdef_nam.F90:81,126`) — the members are a namelist line |
| 4-vs-4 permutation gives at best p = 2/70 = 0.0286, and a 3-D rms is a category error for a mean-difference floor | **ACCEPTED** — the from-rest harness uses pairwise distances, and reports the floor-crossing day rather than a day-360 binary |

## ASKED / UNASKED

**UNASKED, and named because a reviewer named it: the flux-face height and the
cell-volume height are now ONE argument, where before the flux face had its own
and the volume had none.** That removes a separation that existed. It is
deliberate — `traldf_iso` indexes both at `Kmm` with one `r3t`, and the
previous round shipped a bug precisely because the two halves of the `akz`
split could be given different `e3w` — but it is a design choice made inside
the diff, so it is recorded here rather than left implicit. Offered for revert
in the same breath: splitting them again is a two-line change.

**Everything else: none.** No default value moved, no config field was added,
no YAML key changed. `redi_flux_eta` → `redi_kmm_eta` is an internal keyword
with two call sites, both already passing the `Kmm` height.
