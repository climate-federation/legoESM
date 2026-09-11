# PREREG — the kt=2 THICKNESS TIME LEVEL in the leap-frog's dissipative pass

Written BEFORE the code change.  Companion to `PREREG_kt1_kmm_stretch_and_rdt.md`,
which established WHY kt=2 is the only record that can see this (at kt=1 from
rest `max|sshb| = 0`, so every `Kbb`/`Kmm` height coincides and the statement is
degenerate).

## §1 — THE ALIGNMENT TABLE (Rule 0: every row is a compiled statement)

`stpmlf.f90`'s tracer chain, in call order, with the time level of the
SEA-SURFACE-HEIGHT STRETCH (`r3t`/`r3u`/`r3v`, i.e. `e3 = e3_0*(1+r3)`) that
each routine reads.  Lines are `cfgs/DINO/BLD/ppsrc/nemo/*.f90` (the COMPILED
source), which is what the certified binary ran.

| stpmlf | routine | what the height scales | NEMO's level | citation |
|---|---|---|---|---|
| `:201` | `eos( ts, Nbb, rhd )` | the DEPTH the before-density is evaluated at (`zh = gdept_3d*(1+r3t(Knn))`, `Knn=Nbb`) | **Kbb** | `eosbn2.f90:272` (`eos_insitu_New_t(pts,ktts,Knn,…)`), `:362`. The 3-argument call with an INTEGER second argument can only bind `eos_insitu_New` (`:256`); `eos_insitu(pts,prd,pdep,…)` is excluded on type. DINO sets `ln_seos = .true.`, so `:362` is the live branch and `:325` (TEOS/EOS80) is dead |
| `:216` | `ldf_slp( kstp, rhd, rn2b, Nbb, Nnn )` | every `gdept`/`gdepw`/`e3u`/`e3v`/`e3w` in the slope assembly and both limiters | **Kmm** | `ldfslp.f90:177, 216, 267, 268, 281-286, 336, 337, 345` |
| `:459` | `tra_sbc( kstp, Nnn, ts, Nrhs )` | the first-layer thickness the surface flux is divided by | **Kmm** | `trasbc.f90` (already landed; §2 below) |
| `:484` | `tra_adv( kstp, Nbb, Nnn, Naa, ts, Nrhs )` → `ldf_eiv_trp( …, Kmm, Krhs, 'TRA' )` | NOTHING — the eiv streamfunction carries `e2u`,`wslpi`,`aeiu`,`wumask` and no `e3` at all | n/a (Kmm transport) | `ldftra.f90:944-947` (psi), `:951-952` (the increment), `traadv.f90:251` |
| `:491`–`:504` | `tra_ldf` → `traldf_iso_lap( kt, Kbb, Kmm, … )` | `zA11`/`zA22` face areas; the A33 `e3w` divisor; the tendency `1/e3t`; the `ze3w_2` inside `akz` | **Kmm** (every one) | `traldf_iso.f90:231, 232, 284, 292, 305, 823, 828` |
| `:507` | `tra_zdf` → `tra_zdf_imp( 'TRA', rDt, Kbb, Kmm, Krhs, pts, Kaa, … )` | off-diagonals `e3w`; the diagonal `e3t`; the RHS's two terms | **Kmm** / **Kaa** / **Kbb**+**Kmm** | `trazdf.f90:222, 223, 232, 233` (Kmm); `:224, :234` (Kaa); `:285, :289` (Kbb) and `:286, :290` (Kmm) |
| `:568` | `tra_atf_qco` → `tra_atf_qco_lf` | `ze3t_b`/`ze3t_n`/`ze3t_a`; the filtered divisor `ze3t_f` | **Kbb**/**Kmm**/**Kaa**; `r3t_f` | `traatf_qco.f90:299, 300, 301, 312` |
| — | `dyn_ldf` → `dynldf_lev_lap( kt, Kbb, Kmm, … )` | the DIVERGENCE term and its transports; the CURL term; the trend divisor | **THREE-WAY**: `Kbb` for the divergence, an UNTIMED `r3f` for the curl, `Kmm` for the divisor | `dynldf_lev.f90:128, 129, 130` (Kbb); `:124` (`r3f`, no time index at all — a 2-D array last written by the PREVIOUS step's `dom_qco_r3c` at `stpmlf.f90:350` from `ssh(Naa)`); `:136, :140` (Kmm) |

**The one subtlety that is easy to get backwards, and is therefore written
down:** `ssh_atf` overwrites `ssh(:,:,Nnn)` at `stpmlf.f90:425`, but
`r3t(:,:,Nnn)` is NOT refreshed from it until `:571`, after `tra_atf_qco`.
So every `r3t(…,Kmm)` above is the **UNFILTERED step-entry** now-height, not the
Asselin-filtered one.  legoESM's step-entry `state.eta` is exactly that object.

## §2 — WHAT legoESM DOES TODAY, and the three rows that differ

The leap-frog runs the dissipative operators in a second `_step_impl` pass over
a state whose whole geometry is substituted to the before level
(`ocean_model_latlon_cgrid.py:10492-10494`, `nbb = state._replace(…,
eta=state.eta_before)`).  Three geometry consumers read `state.eta` inside that
pass, so all three silently become `Kbb`:

| # | legoESM operand | feeds | NEMO's level | today in the Nbb pass | verdict |
|---|---|---|---|---|---|
| 1 | `native_slope_eta` (`:6108`, `:6167` at `3f889bb3b8dc..HEAD`) | `compute_nemo_native_slopes` — the `ldf_slp` geometry | **Kmm** (`ldfslp.f90:177…345`) | `Kbb` | **DIFF** |
| 2 | `redi_kmm_eta` (`:6122`, `:6170` at HEAD) | `_kmm_J` (the `1/e3t`, A33 `e3w`, `ze3w_2`) AND `_flux_eta` (the `zA11`/`zA22` face thicknesses, since the card resolves `redi_flux_face_thickness_evaluation='nemo_qco_live'`) | **Kmm** (`traldf_iso.f90:231…828`) | `Kbb` | **DIFF** |
| 3 | `_eta_slope` → `_gm_native_prd_J` (`:6020` at HEAD) | the DEPTH of the before-density `rhd` | **Kbb** (`eosbn2.f90:325, :362` with `Knn=Nbb`) | `Kbb` | **MATCH — do not touch** |

Row 3 is the reason the substitution is not "pass the now-state into the Nbb
pass".  One of the three operands is genuinely `Kbb` and moving it would be a
new defect.  Row 4 (`dynldf_lev_lap`'s OUTER divisor) is momentum, is
**MIXED** in NEMO, and is NOT landed this round — it is named with its
measurement in §5.

## §3 — THE ARMS, PREREGISTERED, one operator at a time

Each arm is one operand moved from `Kbb` to the true `Kmm`, measured on the
kt=2 record through `kt2_leapfrog_gate.py`'s existing `tra_ldf` row (the spy
identifies NEMO's call by an EXACT match against the `Kbb` tracer, so a
mis-identified call cannot be scored).

| arm | operand moved | prediction |
|---|---|---|
| A0 | none (baseline) | T ratio `0.999995181`, S `0.999995294` — the recorded value, re-measured, not retyped |
| A1 | row 2 only (`redi_kmm_eta`) | ratio moves to `1.000000015 ± 5e-9`; **`cells!=` stays > 0** — this is DEBT that shrinks, not a pass |
| A2 | A1 + row 1 (`native_slope_eta`) | direction UNPREDICTED. The slopes enter the tendency multiplicatively, so this can move the ratio either way; the preregistered decision rule is below |
| A3 | rows 1+2 with row 3 ALSO moved (the "pass the now-state" version) | must be WORSE than A2, or the `eos(ts,Nbb,·)` reading in §1 is wrong |

**GIVEN-INPUTS ROW for every arm** (Rule 7, and the thing that makes these
comparable): each arm is one run of the SAME gate, on the SAME kt=1 restart
from `RUN_FROMREST_KT1`, the SAME kt=2 trend record, the SAME mesh, fp64, the
same `dt = rn_Dt = 2700` with `rdt = 2*rn_Dt` printed by the gate before it
scores.  Only the named operand differs.

**DECISION RULE, written before the numbers exist.**  An arm LANDS only if its
`tra_ldf` residual rms falls AND no other scored row in the gate rises.  A2
lands on that rule alone — the alignment table says `Kmm` regardless of which
way the ratio moves, and a faithful change that worsens a metric is Rule 8, not
a reason to revert.  A3 never lands; it exists only as the falsifier.

**FALSIFIER.**  If A1 does not move the `tra_ldf` ratio at all, then
`redi_kmm_eta` is not reaching the operator in the Nbb pass and every number in
this document is about a dead operand.  The gate prints the operator's captured
kwargs, so this is checkable rather than assumed.  **It is ONE-SIDED, and that
limit is written down rather than discovered later:** `redi_kmm_eta` feeds TWO
consumers (`_kmm_J` and, because this card resolves
`redi_flux_face_thickness_evaluation='nemo_qco_live'`, `_flux_eta`), so a moved
ratio proves only that AT LEAST ONE of them is live.  A half-wired operand
passes this falsifier.

## §4 — RULE 12: what else executes this statement

The edit is a new private keyword on `_step_impl` that is passed by exactly ONE
caller — the leap-frog's Nbb pass.  Two OTHER sites (`_unsplit_ab2_step`,
`:11232-11250`) pass the same two operands from their own `state.eta`; they do
not go through `_step_impl` at all and are untouched, and the sweep covers them
rather than arguing about them.  `_nemo_mlf_step` (the `nemo_mlf` recipe) never
substitutes the geometry — it runs ONE pass at Nnn and hands only the Kbb
TRACERS to the operator through `_ldf_state` — so it is already on NEMO's split
and does not move.  Every other caller leaves it `None` and is
bit-identical BY CONSTRUCTION, which is a claim the sweep MEASURES rather than
asserts: `kt2_kmm_card_sweep.py` fingerprints every recipe that resolves
`outer_integrator in ('leapfrog','nemo_mlf')` together with a GM/Redi block,
runs one step of each, and hashes the after-state before and after the edit.

* Cards whose reference IS NEMO move toward it and that is the point.
* Cards that are NOT on the leap-frog (forward-Euler / AB2 / RK3 recipes, the
  GYRE lane, MPAS) must be BYTE-IDENTICAL; a changed hash there is a failure.
* The certified 90-day twin is re-measured, not assumed.

## §5 — NAMED, NOT LANDED (each with its measurement)

1. **`dynldf_lev_lap`'s outer divisor.**  NEMO divides the u/v trend by
   `e3u/e3v(Kmm)` (`dynldf_lev.f90:136, :140`) while building the transports
   from `e3u/e3v/e3t(Kbb)` (`:128-130`).  legoESM's Nbb pass uses `Kbb` for
   both.  Measurement: the gate's `u`/`v` rows against NEMO's kt=2 `un`/`vn`,
   with a `utrd_ldf` trend row if the record carries one.
2. **The eiv's STAGE.**  NEMO adds the bolus transport inside `tra_adv`
   (`traadv.f90:251`), i.e. in the `Kmm` advective pass; legoESM evaluates it in
   the `Kbb` dissipative pass.  That is a stage difference, not a height
   difference, and it needs its own record.
3. **The two surface statements** (the restoring flux's time level; NEMO's
   `0.5*(sbc_tsc_b + sbc_tsc)` two-step average) still stand — both need new
   carried state, which the user has not decided.  No kt>=2 surface row may be
   called AT BAR while they stand.

## §6 — WHAT THIS CANNOT SEE

* `akz` fires on 0 wet cells at kt=2 (the census is in the gate), so ONLY the
  `ze3w_2` height at `traldf_iso.f90:823, :828` is multiplied by nothing.  The
  A33 `e3w(Kmm)` divisor at `:284` is NOT in that hole — `ln_traldf_msc =
  .true.` (`namelist_cfg:267`) makes the explicit A33 flux live through
  `ah_wslp2` regardless of `akz`, so arm A1 does score it.  A spun-up record is
  still required for the `ze3w_2` pair.
* The gate's STATE rows are not attributable to any leap-frog statement while
  the surface carries (`qns_b`, `sbc_hc_b`, …) are unbridged; only the
  per-operator `tra_ldf` row is.

## §7 — CLAIM REVIEW (fresh independent Claude agent, run on this document BEFORE the code)

It opened every cited line itself.  Claims 1-4 CONFIRMED with exact citations;
claim 2 was found UNDERSTATED (`Kbb` appears in `ldf_slp` only in the signature
`:114` and its declaration `:140` — it is a dead dummy argument).  Claim 5
CONFIRMED but INCOMPLETE, and that omission is now the `r3f` cell in §1: a
change that moved the whole interior of `dynldf_lev_lap` to `Kbb` would have
introduced a defect in the curl term.  Six smaller corrections (the legoESM
line numbers, the `eosbn2` dead branch, the `ldftra` span, the one-sided
falsifier, the `akz` scope, the `_unsplit_ab2_step` sites) are folded in above.

One of its findings does NOT stand, and the reason is a timestamp: it reported
that `_kmm_geometry_eta` was already wired and therefore that "A0 is really
A1".  It read the WORKING TREE after the edit had been applied.  Arm A0 was
measured at 05:35:43 on the clean tree, before any edit existed, and it
reproduces the recorded baseline — T ratio `0.999995181`, S `0.999995294`,
`cells!= 342090/342134` — to every printed digit.  The reviewer could not see
that, and the correction is recorded here rather than argued about.
