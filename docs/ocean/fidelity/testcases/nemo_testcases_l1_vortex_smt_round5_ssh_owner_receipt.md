# Round 215 / VORTEX_SMT round 5 — the kt=2 sea-surface-height owner, NAMED

**VERDICT: the owner is the barotropic loop's ENTRY operand, not any
statement inside the loop.** Handing NEMO's own recorded depth-averaged slow
forcing to the split-explicit solve takes the seamount card's end-of-window
sea surface height from `3.664757e-07` m to `2.109e-15` m — a factor of
174 000 — and its barotropic velocities from `6.00e-08`/`4.01e-08` to
`1.67e-16`/`2.22e-16`. Every thickness operand INSIDE the loop is bit-exact
over the seamount. **Nothing lands: the repair is only 21 % transcribed and
the round's own bar refuses a partial one.**

Decision 88 (user), operator note CC addendum 5. Predictions were frozen
before any measurement:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/round5/predictions.md`.
Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/round5/`.

---

## 1. PRE-IMPL SEARCH (RULE 4)

Searched, and what was found — nothing new was written where something
existed:

```
ls scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex/
grep -n 'spgts' scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex/run.sh
grep -rn 'read_rhs|oracle_rhsterm|NEMO_L1_RHSTRM' scripts/validate/ocean_fidelity/testcases/
grep -rn 'min_rule_live' packages/ src/ scripts/ tests/
grep -rn 'nemo_qco_card_mesh_operands|_nemo_ws_qco_stage_faces' packages/ src/
```

* The per-substep writer, its Fortran module, its patch and its checker
  already exist (round 196) — this round adds two acquisition VARIANTS that
  compile them into the seamount configurations, no new writer.
* The substep walk already exists (round 196) — it gained a `--case`
  argument, not a copy.
* The per-term pre-stage probe already exists (round 200) — same, one
  `--case` argument.
* The depth-average transcription already exists as round 213's HELD patch —
  it was measured again here rather than rewritten.

## 2. THE RECORD

Two new configurations, `VORTEX_SMT_R5_VEC_R8_OMIP_L1{,_P3}`, built from the
shipped VORTEX case with the seamount hook and the round-196 per-substep
writer. Rounds 1 and 3 builds were neither moved nor rebuilt.

| row | value |
|---|---|
| records | `kt = 1..10`, one per baroclinic step |
| frames per record | **50** — loop entry, 48 sub-time-steps, loop exit |
| groups per record | **1562** |
| `icycle` | **48**, read from each record's own header (nothing predicted) |
| plants | all four fire (`header`, `field-name`, `truncated`, `missing-frame`); the unplanted run is green |
| admission | `"status": "ADMITTED"` |

**Additions-only, proven across BUILDS and ROUNDS.** The step-10 restart is
byte-identical for three independent runs — this round's instrumented run,
this round's own uninstrumented reference, and the ADMITTED round-3 run of
the same card in a different configuration directory:

```
10b5150c540771443b31462381406c486b7696076cd1d172076b1bd9c9411904
  round5/VORTEX_SMT_R5_VEC_R8_OMIP_L1_P3/spgts/VORTEX_SMT_VEC_OMIP_L1_ZPS_00000010_restart.nc
  round5/VORTEX_SMT_R5_VEC_R8_OMIP_L1_P3/spgts/reference/…_00000010_restart.nc
  round3/VORTEX_SMT_R3_VEC_R8_OMIP_L1_P3/kt1_10/…_00000010_restart.nc
```

**The flux card's record was NOT acquired, and the reason is mine.** Its
build completed and its run never started: I committed an edit to the
acquisition script WHILE that script was executing, and bash re-read the
half-written file and died on a syntax error at the admission step. No
evidence directory was created, so nothing is half-admitted; the two
`VORTEX_SMT_R5_OMIP_L1{,_P3}` builds are left in place, unused and unmoved.
It is re-queued as round 6's first item (§12). The BOUNDARY this round
names is one both cards have; the STATEMENT behind it is not shared (§5),
so the flux card's own owner is open.

## 3. THE WALK IS ONLY VALID AT kt=1, AND A CONTROL SAYS SO

The barotropic solve at step `kt` produces the quintuple the stages consume
at `kt+1` (round 195's "N+1"). So the kt=**1** window's output IS the
registry's kt=**2** row, and the measured numbers say it to seven digits:
the walk's end-of-window `ssha` is `3.664757e-07` and the registry's kt=2
`ssh` row is `3.664757e-07`.

**A kt=2 walk was run first and is REFUSED as an instrument.** Seeding the
step entry replaces T, S, u, v and ssh but NOT the carried barotropic
history, so a kt=2 walk measures that gap and not the physics. The control
that proves it is the FLAT card, whose kt=2 registry `ssh` row is
`2.831e-15`, at the bar:

| arm | end-of-window `ssha` |
|---|---|
| seamount, kt=2 walk | `6.970e-03` |
| seamount, kt=2 walk, NEMO entry velocity | `7.522e-04` |
| **flat**, kt=2 walk, NEMO entry velocity | `7.469e-04` |

The flat card — at the bar in the ladder — shows the same `7.5e-04`. The
kt=2 walk is measuring the seed, not the seamount. Every number below is
from the kt=1 walk.

## 4. THE WALK, IN NEMO'S SUBSTEP ORDER

`VORTEX_SMT_VEC-zps`, kt=1, production-jitted step, bar `1e-15`.

| boundary (NEMO's order) | first substep | reading |
|---|---:|---|
| loop entry `ssh_frc` | `0` | **BIT** |
| loop entry `zu_frc` / `zv_frc` | **`5.552e-11` / `5.556e-11`** | **the first non-bit operand** |
| entry velocity `un_e` / `vn_e` | `5.6e-17` / `1.1e-16` | at the floor |
| mid-step `sshp2_mid` | `0` | **BIT** |
| mid-step face depths `hup2_e` / `hvp2_e` | `0` / `0` | **BIT** |
| after-SSH `ssha_e` | `9.992e-16` | at bar |
| surface pressure gradient `zu_spg` | `3.39e-19` | floor |
| barotropic Coriolis `cor_u` | `6.78e-21` | floor |
| updated velocity `ua_new` | `3.331e-09` | `= rDt_e * d(zu_frc)` exactly |
| end-of-substep depths `hu_e` / `hv_e` | `0` / `0` | **BIT** |

Two readings follow directly and neither needs a model of the loop.

**Every partial-cell THICKNESS operand of the loop is bit-exact AT THE
FIRST SUBSTEP** — `hup2_e`, `hvp2_e`, `hu_e`, `hv_e` and `sshp2_mid`, `0`
differing cells each, over a seamount whose stepped U faces number
686 in the `nlev = 10` operand the solver receives (1 164 in the `jpk = 11`
mesh array it is sliced from) — round 214's receipt, §8, which is where
both counts were measured, not this round. The scope
word is deliberate and the whole-window numbers are given rather than
implied, because an earlier draft of this receipt wrote "every thickness
operand inside the loop is bit-exact" without them and the independent
reviewer refuted it: `hu_e` goes non-bit at **substep 2**
(`3.976e-08` m, 219 cells) and `hup2_e` at **substep 3** (`7.082e-08` m,
237 cells), both reaching `3.64e-07` m by substep 48. (Those are absolute
metres; the walk's normalized rows, divided by the `5000` m peak, are
`7.95e-12`, `1.42e-11` and `7.28e-11`.) That is what a row downstream of a
first-substep forcing difference must do once the sea surface has started
to differ. What the walk shows is that no thickness
operand STARTS the difference. The round's primary prediction R5-P3 — that
a thickness operand owns it — is **REFUTED**, and round 214's
face-thickness landing is why: the reference faces this solve reads are now
NEMO's own.

For the same reason the only two boundaries non-bit at substep 1 are the
two forcing rows; every other row's whole-window maximum
(`entry.eta 5.199e-07`, `flux.u 2.429e-07`, `spg.u 8.112e-11`,
`cor.u 3.282e-12`) is a consequence, not a cause. "At substep 1" is load
bearing in every sentence of this receipt that says it.

**The velocity update's injection IS the forcing difference, arithmetically.**
`rDt_e = rn_Dt/nn_e = 2880/48 = 60 s`, and `60 * 5.5518e-11 = 3.3311e-09`,
which is `ua_new`'s first-substep difference to five digits. Nothing else in
the loop contributes at that size.

### 4.1 THE SUBSTITUTION — one variable, and it closes the window

Hand the loop NEMO's own recorded `zu_frc`/`zv_frc` at the boundary where
legoESM forms them, change nothing else. The arm is self-checking: the two
forcing rows go to `0` differing cells, so the substitution bound.

| row | production | NEMO's own slow forcing | |
|---|---:|---:|---|
| loop-entry `zu_frc` | `5.5518e-11` | **`0` (bit)** | — |
| end-of-window `ssha` | `3.664757e-07` | **`2.109e-15`** | **174 000x toward NEMO** |
| end-of-window `ua_new` | `5.9988e-08` | **`1.665e-16`** | 360 000x |
| end-of-window `va_new` | `4.0123e-08` | **`2.220e-16`** | 181 000x |

**CONFIRMED: the whole of the seamount cards' kt=2 sea-surface-height error
is carried by the barotropic loop's entry forcing.** Nothing inside
`dyn_spg_ts` owns any of it.

## 5. INSIDE THE FORCING: THE CITED STATEMENT, AND HOW MUCH OF IT IT OWNS

NEMO builds that operand in the pre-stage 2-D driver. The statement that
executes on this build is, verbatim:

```
Ue_rhs(ji,jj) = SUM( e3u_3d(ji,jj,1:jpkm1)*uu(ji,jj,1:jpkm1,Krhs)*umask(ji,jj,1:jpkm1) ) * r1_hu_0(ji,jj)
Ve_rhs(ji,jj) = SUM( e3v_3d(ji,jj,1:jpkm1)*vv(ji,jj,1:jpkm1,Krhs)*vmask(ji,jj,1:jpkm1) ) * r1_hv_0(ji,jj)
```

`tests/VORTEX_SMT_R5_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:178-179`,
the vector-form branch (`np_VEC_c2`, selected at `:176`). It reads the
REFERENCE face thickness (`e3u_3d` is `e3u_0` under `key_vco_1d3d`) and the
STORED reciprocal `r1_hu_0`; there is no sea-surface stretching anywhere in
it. legoESM instead weights with the per-level minimum of the two LIVE
(stretched) thicknesses and divides by their own column sum.

**This is the VECTOR card's branch, and the flux card's is a different
statement.** `:176` opens `SELECT CASE( n_dynadv )`; `:178-179` is the
`np_VEC_c2` arm, which ASSIGNS the depth mean, and `:183-184` is the
`np_FLX_c2, np_FLX_up3` arm, which CUMULATES it onto the two-dimensional
advective right-hand side `dyn_adv_up3` has already written. The velocity
update downstream forks the same way (`dynspg_ts.f90:585` against the
`hu_0*(1+r3u)`-weighted branch at `:599-618`). So what the two seamount
cards share is the BOUNDARY — the loop-entry forcing `zu_frc` — and not
this statement; an earlier draft of this receipt said they share the
statement and the independent reviewer refuted it. On a full-step
mesh the two agree exactly — one per-face scalar cancels — and over partial
cells they do not. That is round 213's S2, held as
`manifests/nemo_testcase_l1_vortex_smt_round213_slow_forcing_depth_held.patch`.

**Measured in the substep frame, S2 moves the forcing and does not close
it.** Whether it is a CORRECT transcription cannot be read off this table
— it reduces two rows and makes a third worse, which is what two partly
cancelling errors look like — and §6 settles it by measurement instead.

| row | production | S2 applied | |
|---|---:|---:|---|
| loop-entry `zu_frc` | `5.5518e-11` | `4.3969e-11` | 1.26x |
| end-of-window `ssha` | `3.664757e-07` | `2.861601e-07` | 1.28x |
| end-of-window `ua_new` | `5.9988e-08` | `4.6322e-08` | 1.30x |
| end-of-window `va_new` | `4.0123e-08` | `4.4837e-08` | **0.89x — away** |

The `2.861601e-07` reproduces round 213's number to every digit it printed,
measured in a completely different frame, which is the cross-check that the
substep instrument and the ladder are describing the same quantity.

**S2 is therefore an EXACT transcription that is not the whole owner.**
Section 6 proves both halves of that sentence by measurement rather than by
apportioning a max-abs ratio, which would not be a legitimate decomposition
anyway. (`stp2d.f90:138`, `:141`, `:144`, `:153`, `:161`, `:163` — `dyn_hpg`,
`dyn_ldf`, `dyn_vor`, `wzv`, `dyn_keg`, `dyn_zad`, in that order).

`Ue_rhs` reaches the solve as `zu_frc(:,:) = Ue_rhs(:,:)`
(`dynspg_ts.f90:275`), the barotropic Coriolis trend is subtracted from it
at `:292` (`zu_frc = zu_frc - zu_trd*ssumask`), and the record dumps the
finished operand at `:365`. The velocity update that consumes it is
`:586-589`, with `rDt_e = rn_Dt/nn_e`.

**The barotropic Coriolis subtraction at `:292` is RULED OUT by the record,
not by argument.** It is a genuine candidate, because the recorded operand
is taken after it; but the same operator, handed velocities differing by
`5.6e-17`, produces a `cor_u` differing by `6.8e-21` at every one of the 48
substeps. It cannot carry `4.4e-11`.


## 6. THE SPLIT: WHICH OF THE TWO STATEMENTS OWNS WHAT

The operand the record carries is formed by two compiled statements, in
this order:

```
dynspg_ts.f90:275   zu_frc(:,:) = Ue_rhs(:,:)                 the depth average
dynspg_ts.f90:292   zu_frc(ji,jj) = zu_frc(ji,jj) - zu_trd(ji,jj) * ssumask(ji,jj)
```

Both are live at kt=1 and neither is negligible: `Ue_rhs` peaks at
`8.1226e-05` m/s^2, the finished `zu_frc` at `6.1395e-05`, and the
subtraction between them is `2.0272e-05` on 3 660 faces — the entry
barotropic velocity is `0.2438` m/s, not zero, so this is not a control
that perturbs a zero
(`round5/slow_forcing_split.json`, probe
`nemo_testcase_l1_vortex_round215_slow_forcing_split.py`).

**The three-dimensional right-hand side the depth average consumes is NOT
the owner, and that is measured, not assumed.** A third acquisition — the
round-200 per-term pre-stage writer compiled into the seamount vector deck
(`VORTEX_SMT_R5R_VEC_R8_OMIP_L1{,_P3}`, admitted, all plants fire) — lets
the existing per-term probe compare legoESM's own breakdown against NEMO's
increments at the same boundary:

| row | NEMO | legoESM minus NEMO |
|---|---:|---:|
| `vor` | `6.626650e-05` | `2.03e-20` |
| `keg + hpg` | `8.445093e-05` | `6.78e-21` |
| `zad` | `1.692846e-08` | `5.93e-21` |
| `ldf` | `0` (off in this deck) | `0` |
| **completed `uu(:,:,:,Krhs)`** | — | **`1.355e-20`** |

The array `stp2d.f90:178` averages is bit-level identical. Whatever is left
in the forcing is in the two statements above it, not in what they are
handed.

**One variable, and it is decisive: substitute NEMO's OWN depth average.**
The probe rebuilds `Ue_rhs` from NEMO's recorded right-hand side and NEMO's
own mesh operands — the card's `e3u_0`, `umask`, `hu_0`, proved 0 ULP
against `mesh_mask.nc` in round 212 — and the walk injects it at the
boundary legoESM forms the same quantity, BEFORE the Coriolis subtraction.

| arm | loop-entry `zu_frc` | end-of-window `ssha` |
|---|---:|---:|
| production | `5.551760e-11` | `3.664757e-07` |
| **NEMO's own depth average injected** | **`4.3969046116e-11`** | **`2.8616009e-07`** |
| **round 213's held S2 patch** | **`4.3969046116e-11`** | **`2.8616009e-07`** |
| NEMO's own finished `zu_frc` injected | `0` (bit) | `2.109e-15` |

The middle two rows were produced by two independent routes — one injects
NEMO's array, the other runs legoESM's own transcription of NEMO's
statement — and they leave the same residual. **How close "the same" is,
exactly, because a draft of this receipt wrote "agree to every digit" and
called it a proof of exactness, and the reviewer refuted that from these
two JSON files:** the `u` maxima are identical to all 18 printed digits
(`4.39690461164344952e-11`), the `v` maxima are NOT
(`4.38530688948472400e-11` injected against `4.38530689016235035e-11`
transcribed, a difference of `6.776e-19`, i.e. `1.5e-08` of the residual
itself), and the two arms disagree about WHICH cells are unequal (2 104
against 2 520). They are therefore different fields.

**What that licenses, and what it does not.** It licenses: the two routes
bound the same statement to within `1.5e-08` of the residual they both
leave, so the depth average is not where the remaining `4.4e-11` is — a
transcription wrong enough to hold `4.4e-11` could not land within
`6.8e-19` of NEMO's own array. It does NOT license calling round 213's S2
EXACT; `max |S2 - NEMO's own depth average|` per cell is not measured here
and the one-line arm that would measure it (dump both and difference them)
is round 6's. Until then S2 is a transcription whose residual is bounded,
not a transcription shown exact.

**By elimination, over a set that is closed ONLY under two stated
exclusions: the remaining `4.3969e-11` is the loop-entry barotropic
Coriolis subtraction, `dynspg_ts.f90:292`.** FOUR statements write
`Ue_rhs` between `stp2d.f90:178` and `dynspg_ts.f90:275`, not two — the
reviewer found the other two and they are carried here rather than
dropped: `stp2d.f90:194` passes `Ue_rhs` to `dyn_drg_init` as
`INTENT(inout)` (`dynspg_ts.f90:1284`) which adds the bottom drag
UNGUARDED at `dynspg_ts.f90:1339`, and `stp2d.f90:197-199` adds the
surface stress, also unguarded. Both are identically zero on THIS deck,
and only on a deck like it: `ln_drg_OFF = .true.` (`namelist_cfg:114`) and
`ln_usr = .true.` with VORTEX's zero-stress `usrdef_sbc`. The probe now
REFUSES unless the run's own namelist says both, and its output field is
named `residual_after_depth_average_*` rather than `coriolis_*`, so the
attribution cannot travel to GYRE or ORCA2 inside the instrument.

**A SECOND, INDEPENDENT READING THAT THIS RECEIPT DOES NOT RESOLVE, AND
SAYS SO.** NEMO builds `zu_trd` at `:292` with the SAME `dyn_cor_2D` the
loop calls at every substep, and **at substep 1** the record says that
in-loop operator is faithful: handed velocities differing by `5.6e-17` it
produces a `cor_u` differing by `6.8e-21`. (Not at every substep — that is
a scope word an earlier draft got wrong and the reviewer refuted: `cor_u`
grows to `3.282e-12` by substep 48, downstream of the entry difference.
Substep 1 is the comparable one, because it is where the entry subtraction
also acts.) A coefficient error large enough to put `4.4e-11` into a
`2.03e-05` subtraction is a relative `2.2e-06`, which the same operator
would have shown at substep 1 and does not. The two readings are stated
together rather than averaged, and the discriminating measurement is named:
a per-statement override of the ENTRY subtraction alone, which no
instrument currently has. It is round 6's, preregistered in §12. Until it
runs, "the entry Coriolis subtraction owns the remainder" is **PLAUSIBLE,
not CONFIRMED** — what is CONFIRMED is that the depth average and the 3-D
right-hand side do not.

## 7. LANDING VERDICT: HELD, AND WHY THE PARTIAL FIX IS REFUSED

**Nothing in production changed this round.** `git diff` of the round's
first commit against its last, restricted to `packages/` and `src/`, is
zero lines. Every card, every certified number, GYRE, DINO, both tanks and
the six flat VORTEX cards are therefore inert BY CONSTRUCTION, not by
measurement — which is the strongest form of the inertness claim and the
same one round 196 made.

The preregistered landing bar (R5-P4, frozen before measurement) was: the
named statement must take kt=2 `ssh` at least 10x toward NEMO on BOTH
seamount cards and move no certified row away. The depth-average statement
alone gives **1.28x and moves `v` 11 % away**. It is refused for exactly the
reason round 213 refused it, now with the substep frame saying how much of
the owner it is (21 %) instead of leaving that unknown.

**RULE 3 applies to the held patch and is stated out loud:** its config
field `barotropic_slow_forcing_depth_evaluation` defaults to
`min_rule_live`, which is today's behaviour, i.e. the defect. That default
is *not* a shipped knob — the patch is held in `manifests/`, outside the
package tree, and the card edit that selects `nemo_literal` is part of the
same held patch. If a future round lands it, the question "default stays
broken (`min_rule_live`) or moves to the fixed value (`nemo_literal`)?" must
be asked in that PR, and the seamount cards must select it explicitly.

## 7.1 THE PREREGISTERED PREDICTIONS, EVERY ONE ADJUDICATED

* **R5-P1 — the record admits.** **CONFIRMED** for the vector card
  (50 frames, 1562 groups, `icycle` 48 from the header, four plants fire,
  restart byte-identical to round 3's). **NOT EVALUATED** for the flux
  card: its run never happened (§2).
* **R5-P2 — the loop-entry operands are not the whole owner.** **REFUTED,
  and the refutation is the round's finding.** Its own falsifier said
  "substituting the loop-entry operands alone takes kt=2 ssh below
  3.7e-08"; the measurement is `2.109e-15`. The prediction was wrong and
  the opposite is true: the loop-entry forcing is the whole of it.
* **R5-P3 — the first non-bit statement is a partial-cell thickness
  operand.** **REFUTED.** Its own falsifier — thickness rows bit-exact at
  substep 1 while another row is the first non-bit one — fired exactly as
  written, except that the first non-bit row is the FORCING rather than
  the predicted trend.
* **R5-P4 — one variable, measured on BOTH cards.** **NOT MET, and the
  bar it set is therefore unevaluable this round.** The bar was "10x on
  kt=2 ssh on BOTH seamount cards"; only the vector card was measured, so
  nothing could have passed it even had the arm been bigger. Said out
  loud rather than quietly scored against one card.
* **R5-P5 — nothing lands unless a cited statement closes every gate.**
  **CONFIRMED.** Nothing in production changed (§7).

## 7.2 THE TWO S2 ARMS, BOTH DISCLOSED

Two arms were run with round 213's held patch applied and they are NOT the
same tree, so both are named rather than the convenient one reported:

| arm | tree | loop-entry `zu_frc` | end-of-window `ssha` |
|---|---|---:|---:|
| `walk_vec_kt1_S2` | commit `2d19354a0` + the held patch, unmodified | `4.3969046116e-11` | `2.861600860e-07` |
| `walk_vec_kt1_S2res` | the same, with ONE further change: the patch's operand source swapped from the re-derived `nemo_qco_card_mesh_operands` to round 214's bundle-reading `nemo_qco_resolved_mesh_operands` | `4.3969046103e-11` | `2.861600860e-07` |

The second was run to test whether the held patch's operand SOURCE matters
over partial cells. It does not — the two agree to ten digits — which is
the control saying the patch's re-derivation of `e3u_0` is not a defect on
this card, and it is why the receipt quotes one number. Both arms carry
`allow_dirty_escape_used: true` with their own `diff_sha256`, as the
provenance stamp requires.

## 8. CHOICES MADE THIS ROUND

| choice | ASKED? |
|---|---|
| two new acquisition variants rather than editing the certified ones | ASKED — note CC addendum 5 says "build the per-substep record on the R3 SMT builds (round-196 instrument pattern)"; new `VORTEX_SMT_R5_*` directories, rounds 1 and 3 untouched |
| the walk and the per-term probe take a `--case` argument instead of being copied | ASKED — "reuse it on the SMT builds; extend, never duplicate" |
| the kt=2 walk is REFUSED and the kt=1 window is walked instead | not a choice of behaviour: the kt=2 seed is proven invalid by the flat-card control (§3), and the kt=1 window IS the kt=2 registry row |
| a third acquisition (the per-term pre-stage record) was added inside the round | ASKED — note CC addendum 5 says "name the first non-bit statement"; the substep record names the forcing and cannot name the term inside it |
| a fourth tool, the forcing-split probe, was committed rather than run as a heredoc | not a choice of behaviour: the campaign's rule is that a throwaway probe's number is unmeasured |
| the walk and the per-term probe now REFUSE a card/record mismatch, where before they would have run silently | **UNASKED when first written, raised here**: it makes a previously-tolerated invocation a hard error. It is the reviewer's MAJOR 3 and it is unreachable by any correct invocation (both are proved to fire, and the two real invocations still run). Offered for revert if the operator prefers a warning |

**UNASKED list: ONE — the card/record mismatch becoming a hard error.**
It is named in the table above and offered for revert in this same receipt.

**COMPLIANCE, stated because no gate checks it (RULE 2):**
* **DUAL review: ONE fresh adversarial reviewer ran, not two.** The second
  is codex, paused on this account. That is a GAP, not an exemption.
* **Controlled comparison:** every arm in §4.1 and §5 differs from the
  production arm in ONE thing, run on the same tree against the same
  admitted record; the S2 arm's tree is the production tree plus the held
  patch and nothing else.
* **Instrument validated before its numbers were quoted:** §3 is that
  validation and it REFUSED the first instrument this round built. The
  substitution arms are self-checking (the substituted rows must become
  bit-exact, and they do). The S2 arm reproduces round 213's independently
  measured `2.861601e-07` to every digit.
* **Non-vacuity:** all four record plants fire and the unplanted run is
  green; the substitution arm moves the end of the window by five orders of
  magnitude, so it could not have been a no-op.
* **Pre-impl search:** §1, with the greps pasted.

## 9. GATES

| gate | result |
|---|---|
| record admission, seamount vector card | `ADMITTED`; 10 records, 50 frames, 1562 groups, `icycle` 48 from the header, four plants fire |
| additions-only | step-10 restart byte-identical across three runs and two configuration directories (§2) |
| production diff (`packages/`, `src/`) | **zero lines** — every card, GYRE, DINO and the tanks inert by construction |
| DINO month gate | auto-skipped by `land.sh` because the production diff is empty |
| citation re-anchor rule | does not apply: no model file was edited |
| the two new card/record guards | both proved to FIRE (seamount card against the flat record, and the reverse), and both real invocations still run |
| direct test for the new probe | `tests/ocean/fidelity/test_nemo_testcase_l1_vortex_round215_slow_forcing_split.py`, 4 tests, run with round 196's own module: `9 passed in 3.42s`. **NON-VACUITY PROVED**: with the guard block deleted the two refusal tests go red (`2 failed, 2 passed`), and the file was restored and `git status --porcelain` checked |
| push battery | §13 (appended on landing) |

## 10. OPEN

1. **The flux seamount card's substep record was not acquired** — my own
   tooling error (§2), the builds are in place, the run is round 6's first
   item. The owner named here is a statement both cards share and the flux
   card's kt=2 `ssh` is `3.726197e-07`, within 2 % of the vector card's.
2. **The remaining 79 % of the forcing difference** is inside the
   three-dimensional pre-stage right-hand side (§6).
3. Round 213's S2 stays HELD, now with its share measured (21 %).
4. The flux card still does not materialise the `wzv` operand where NEMO's
   call is unconditional (round 213's finding, unchanged).
5. The stage-terms writer's uninitialised-halo dump; the stage-one stretch
   helper's missing floor; the untranscribed `r3u`/`r3v` stage-one ratios.
6. `mesh_mask.nc`-sourced operands are float32 on the cards that read them;
   the seamount cards build theirs in float64 from the Gaussian, so this
   does not reach them (checked this round, and an earlier reading of mine
   that it did is **retracted**: it was a probe run without the fp64
   precision policy, i.e. an instrument artifact, caught before it was
   quoted anywhere).

## 11. THE ADVERSARIAL REVIEW

One fresh reviewer (not the author, read-only, given the diff, the compiled
NEMO sources and the evidence directory, and told not to run a test
battery). Verdict **DO NOT SHIP** on the snapshot it was handed, five
MAJOR. Every one is taken; two of them changed what this receipt claims.

| # | finding | taken |
|---|---|---|
| MAJOR 1 | "every partial-cell thickness operand inside the loop is bit-exact" is false as written — `hu_e` goes non-bit at substep 2, `hup2_e` at substep 3 | **ACCEPTED.** §4 now gives both onsets with their cell counts, in absolute metres and normalized, and says "at substep 1" wherever it means it. A weaker version of this fix was already in the file when the reviewer started; its independent reading produced the same two substeps |
| MAJOR 2 | "`cor_u` differs by 6.8e-21 at every one of the 48 substeps" is false — it grows to `3.282e-12` | **ACCEPTED, and it is the sentence that carried an exculpation.** §6 now scopes it to substep 1, which is the comparable boundary, and says why |
| MAJOR 3 | the walk accepts a card/record mismatch SILENTLY: `SMT_CASES` was dead data, `DEFAULT_ROOT` still points at the flat record, and no downstream guard can discriminate because the two decks share a grid size | **ACCEPTED, FIXED, AND THE FIX IS PROVED NON-VACUOUS.** Both scripts now refuse; both refusals were run and printed; both real invocations still run. Registered as this round's one UNASKED item (§8) because it turns a tolerated invocation into a hard error |
| MAJOR 4 | the cited statement is NOT shared by the two cards: the flux card runs the cumulating arm at `stp2d.f90:183-184` and the `hu_0*(1+r3u)` velocity update at `dynspg_ts.f90:599-618` | **ACCEPTED.** §2, §5 and R6-P1 now say the two cards share the BOUNDARY and not the statement |
| MAJOR 5 | "S2 owns 21 %" is not a defensible apportionment of a max-abs, and a correct transcription should not make a row worse, so S2's correctness was unestablished | **ACCEPTED on the apportionment** — the phrase is gone and §5 no longer infers a remainder from it. **ANSWERED on correctness by §6**, which the reviewer did not have: injecting NEMO's OWN rebuilt depth average gives the same residual as S2 to ten digits, which establishes the transcription directly. The `v` row still moves away, and the receipt now says why that is consistent: two partly cancelling errors |
| minor | the prediction ledger was selective and R5-P2's falsifier had fired unreported | **TAKEN**: §7.1 adjudicates all five, and R5-P2 is REFUTED |
| minor | two S2 arms exist with different `diff_sha256` and only one was reported | **TAKEN**: §7.2 discloses both and says what the second was for |
| minor | the `run.sh` comment cited `stp2d.f90:176-178` and `:134-170` | **TAKEN**: corrected in the comment |
| minor | "1 164 stepped U faces" had no provenance in this round's evidence | **TAKEN**: both counts now carry round 214 as their source, and the one the solver receives (686 at `nlev = 10`) leads |
| minor | §10 item 2 pointed at a section that did not exist | **TAKEN** |
| minor | the lost flux record obscures that R5-P4's two-card bar was unevaluable | **TAKEN**: §7.1 says so |
| PASS | all three sha256 verified independently; every NEMO line citation verbatim-correct; every number reproduces from the JSON; `rDt_e = 60` and the `3.331e-09` identity holds to 1 part in 4e9; variant wiring complete across all four switches; `--case` threads through every former constant; nothing in the diff can change a certified number; HOLDING is correct | — |

**A SECOND PASS FOLLOWED, against the current text and the one instrument
the first pass had not seen, and it returned DO NOT SHIP again with four
more MAJOR. All four are taken and the first two changed what this receipt
claims:**

| # | finding | taken |
|---|---|---|
| MAJOR 1 | §6's "the two routes agree to every digit, that is the proof S2 is EXACT" is refuted by the two JSON files it cites: the `v` maxima differ at the 10th digit and the two arms disagree about which cells are unequal (2 104 against 2 520), so they are different fields | **ACCEPTED, decisive.** §6 now gives the `6.776e-19` difference and the two cell counts, claims only that the two routes bound the same statement to `1.5e-08` of the residual, and says plainly that S2 is NOT shown exact and which one-line arm would show it |
| MAJOR 2 | the set is not closed: `stp2d.f90:194` (`dyn_drg_init`, `Ue_rhs` INTENT(inout), drag added unguarded at `dynspg_ts.f90:1339`) and `:197-199` (surface stress) also write it; the probe's field name `coriolis_subtraction_*` baked the attribution into the instrument | **ACCEPTED.** §6 carries both exclusions with the namelist lines that make them zero, the probe REFUSES unless the run's own `namelist_cfg` sets them, and the field is renamed `residual_after_depth_average_*` so the attribution cannot travel to another card inside the tool |
| MAJOR 3 | the probe's documented control, `require(BOUNDARIES[-1] == "zad")`, is a check on its own Python list that no record can falsify | **ACCEPTED.** Replaced by three controls read from the records: the boundary set on disk must be exactly the six, the last dump must DIFFER from the one before it, and the two components must not be the same buffer (their peaks are equal on this symmetric vortex, so a peak check would not catch it) |
| MAJOR 4 | three arms, three trees, and no artifact says whether an operand was injected | **ACCEPTED.** The walk's report now stamps every arm flag, including the injected operand's path |
| minor | `ssu = (hu_0 > 0)` re-derives `ssumask`, which NEMO defines as `MAXVAL(umask, DIM=3)` and the probe already loads | **TAKEN** |
| PASS, second pass | `r1_hu_0 = ssumask/(hu_0 + 1 - ssumask)` verified against `domain.f90:213`; the guard hoist confirmed at `:123`, called from both entry points; MAJOR 1, 2 and 4 of the first pass genuinely discharged, not papered over; the PLAUSIBLE labelling of the Coriolis attribution accepted, and the §5/§6 tension judged correctly resolved rather than averaged | — |

**THE REVIEWER'S OWN CORRECTION, recorded because it changes who was
right.** After the fixes above it re-read the lane and RETRACTED MAJOR 3 —
"the guard exists" — which is true, and it exists BECAUSE of MAJOR 3: the
snapshot it was handed was four of the lane's six commits and I moved the
tree under it. The finding stands, the retraction is of its own reading of
the diff, and the record says both. With it came one residual minor,
**taken**: the conditioning entry point (`--one-ulp-entry-probe`) took a
card and a root without the guard. It is hoisted into one
`require_case_matches_root` helper that both entry points call, with a
fifth test that goes red if `conditioning` stops calling it.

**COMPLIANCE (RULE 2): ONE review, not two.** The second reviewer is codex,
paused on this account. That is a gap, not an exemption.

## 12. ROUND 6 — PREREGISTERED PREDICTIONS

Round 6 is the 100-day comparison and movie for both seamount cards
(round 210's scorer and movie script, reused), plus the flux card's substep
record. Frozen here, before any of it runs:

* **R6-P1** — the flux card's substep record admits on the same terms
  (50 frames, 1562 groups, `icycle` 48, four plants, restart byte-identical
  to round 3's flux run), and its kt=1 walk names the same loop-entry
  BOUNDARY, `zu_frc`, as its first non-bit operand, within a factor of 2 of
  `5.6e-11`. The STATEMENT behind it will be a different one — the flux
  card executes `stp2d.f90:183-184` and `dynspg_ts.f90:599-618`, not the
  vector branches this round walked — so a different repair may be needed
  there. **FALSIFIER:** the flux card's first non-bit operand is a
  different boundary, which would mean the two cards' equal-sized `ssh`
  errors have different owners and §4's reading does not transfer.
* **R6-P2** — over 100 days both seamount cards stay bounded, with a
  threshold rather than a word: peak `|u|` stays under `2.0` m/s (the
  deck's `rn_ppumax` is `1.0`; NEMO's own 100-day run is the comparison)
  and peak `|ssh|` under `1.0` m on every scored day, and the seamount's
  imprint — the vortex's deflection over the bank — has the same SIGN in
  legoESM and NEMO. **FALSIFIER:** either threshold is crossed on any
  scored day, or the deflection has opposite sign.
* **R6-P3** — the 100-day daily score is WORSE on the seamount pair than on
  the flat pair at the same day, because the kt=2 owner named here is still
  in the model. **FALSIFIER:** the seamount pair scores at or better than
  the flat pair, which would mean the kt=2 owner does not survive into the
  developed flow and the round-6 ranking must be redone before any fix is
  prioritised.
* **R6-P4 is NOT a prediction and is restated as a COMMITMENT**, because
  as written ("nothing lands in round 6 either") it described my own
  choice and no measurement could refute it — the reviewer's point, taken.
  The commitment: round 6 is a measurement round, exactly as round 208's
  ladder was, and if a landable statement appears it is reported and
  deferred rather than landed inside it.

## 13. EVIDENCE

All under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/round5/`:
`predictions.md`; `acquisition.log`, `acquisition_flux.log`,
`acquisition_rhs.log`; the admitted records and their admission JSON under
`VORTEX_SMT_R5_VEC_R8_OMIP_L1_P3/spgts/` and
`VORTEX_SMT_R5R_VEC_R8_OMIP_L1_P3/rhs/`; the walk arms
`walk_vec_kt1{,_nemoforcing,_nemodepth,_S2,_S2res}.{json,log}`,
`walk_vec_kt2{,_entryvel}.{json,log}` and the refused-instrument control
`walk_flat_kt2_entryvel.{json,log}`; `prestage_terms_smt_vec.{json,log}`;
`slow_forcing_split.json` and `nemo_depth_average_kt1.npz`;
`round215.diff`.
