# NEMO-testcases L2 GYRE round 103 receipt: the TKE matrix and RHS statement walk

Date: 2026-09-17. Branch `fidelity/nemo-testcases-l2-gyre-codex2`, incoming tip
`e92f8aff20521b23c6d062a5baf8c46882361efc`.

## Verdict

**HELD; no production physics and no configuration landed.** The compiled block
`zdftke.f90:399-473`, which the Round-101 walk reported as one row, is split
into one row per compiled write and driven from NEMO's recorded stage entry
through the real production step.

**WHAT THE DIFF TOUCHES, stated in the verdict so a mechanical scope audit is
not blindsided by the diffstat.** This round is measurement-only in substance
but NOT by line count. It changes 45 lines of the SHARED turbulence solver,
`packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py`, which DINO and
every other card also run. Every one of those lines is reachable only when a
caller passes `return_statement_trace=True` - a static Python flag that no
production path, no card and no default sets. The change is four added trace
fields, a return widened on that diagnostic branch alone, and a loud refusal
when a trace is requested on a column too short for the literal assembly to
run. No production numeric line moved, so no production number can move. The
same disclosure appears in the DINO row of the per-testcase table below; it is
repeated here because a scope audit reads the verdict, not the lane table.

**SCOPE OF THE HEADLINE, stated before it rather than after it.** The block
writes FIVE outputs. The first in compiled execution order is `p_pdlr`
(`zdftke.f90:421`), and it is **NOT MEASURED** this round: it never enters
`en` (it is read once, at `zdftke.f90:712`, where it multiplies `p_avt`), so
it is a separate consumer chain and it gets its own walk. Every "first
non-bit" claim below is therefore scoped to **the `en` path**, which is the
chain that produces the block's exit value and the 979-cell post-sweep gap
(979 of 20,416 owned, 979 of 17,400 wet) this round was asked about. It is
NOT a claim about the block's first non-bit output in general; `p_pdlr` could
be non-bit and nothing here would know.

On the `en` path, the first statement whose output is not bit-identical is the
**right-hand-side assignment at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:439-442`**:

> `en(ji,jj,jk) = en(ji,jj,jk) + rn_Dt * ( p_sh2 - p_avt*rn2 + zfact3*dissl*en ) * wmask`

11,993 of 20,416 owned cells differ - equivalently 11,993 of the 17,400 WET
cells, since the other 3,016 owned cells are dry and agree by construction -
maximum absolute `5.488912518947231e-10`. Both denominators are given for
every headline count in this receipt; see "Denominators" below. The three
matrix writes that precede it inside the same loop body are each **BIT with
zero unequal cells**: `zd_up` (`:434`), `zd_lw` (`:435`) and `zdiag`
(`:436`).

The magnitude of that statement's error is carried entirely by ONE of its five
operands, the shear production `p_sh2`. The production `p_sh2` differs from
NEMO's recorded `sh2` in 17,400 of 20,416 owned cells - which is exactly the
wet-cell total, so 17,400 of 17,400 wet - at maximum
`3.811744924985501e-14`, and

```
rn_Dt * max|delta p_sh2| = 14400.0 * 3.811744924985501e-14
                         = 5.488912691979121e-10
observed RHS max         = 5.488912518947231e-10
ratio                    = 0.9999999684761081
```

so the budget closes to 3.2e-8 relative. **No candidate fix exists inside this
block.** `p_sh2` is computed by a different statement in a different routine
(`zdfsh2`), consumed here as an operand, and under decision 41 it must be
walked at its own stage. The round therefore names the statement and stops;
the trajectory ladder and the days 1-30 arm were not run because there is no
candidate to judge, and every headline number is unchanged by construction.

No NEMO source was modified and neither `makenemo` nor `mpirun` was run. No
card, default, scheme selection, threshold, coefficient, carried state,
restart schema, year harness, reconciliation gate, freshwater pair or #1484
guard changed.

## Registration

The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round103.md`, committed as
`30006264843d` before any new measurement, with `1579a71ef2de` correcting an
invented tip hash in it. Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round103/`.

## Premises carried in from after the round-102 receipt

1. The GYRE card selects `tke_langmuir_evaluation="nemo_literal"` (user
   decision 42, card only) with the safe-sqrt gradient guard, so the
   production arm IS the literal arm. **This buys attribution, not accuracy**:
   the post-sweep row is 979 cells on both arms and the literal arm is
   marginally worse in the last bits (`6.809688229969524e-12` against
   `6.809688013129089e-12`).
2. Round 102's attribution of its 3,223-cell post-Langmuir difference to the
   ordered `zpelc` recurrence is **REFUTED** by the operator's one-variable
   measurement; the owners are the `imlc` index and the missing surface mask,
   mutually redundant. That recurrence is inert and was not walked.
3. An ORCA2 inventory and a corrected passivity gate landed separately; not
   used here.

## Instrument validation before any new science

The consolidated Round-46/51 gate was re-run at the incoming tip in
`--mode stage-tke-walk` with nothing changed, purely to reproduce known
values. Artifact `instrument_repro.json`, SHA-256
`e674f45302126d5642aaea289fa853ea21cb06762ac8b18577f82aa27811764c`. Its
`NEMO_TKE_RECORDED` arm reproduces round 102's literal-arm rows exactly:

| boundary | round 102 literal arm | this tip |
|---|---|---|
| `en_entry` | 0 / 0 | 0 / 0 |
| `en_after_boundaries` | 0 / 0 | 0 / 0 |
| `en_after_langmuir` | 0 / 0 | 0 / 0 |
| `rhs_pre_sweep` | 11,993 / `5.488912518947231e-10` | 11,993 / `5.488912518947231e-10` |
| `en_post_sweep` | 979 / `6.809688229969524e-12` | 979 / `6.809688229969524e-12` |

Cells are `unequal / max abs`.

A second, independent instrument check validates the REFERENCE side. The
committed probe
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round103_tke_block_replay.py`
rebuilds NEMO's four block outputs from NEMO's OWN recorded operands in the
compiled source association and compares them with NEMO's OWN recorded
outputs. All four rows are BIT over all 20,416 owned cells
(`record_replay.json`, SHA-256 below, stamped clean at `c571dab4439f`). That
simultaneously validates the transcription and the index mapping: had either
been wrong, NEMO could not reproduce itself through it.

Its plant is instructive and is recorded as a failure that was caught. The
first version advanced one bit of `avm` at cell `[0,0,2]`, where `tmask` is
zero, so `zcof` annihilates it and every row stayed green; the probe refused
its own plant with exit 2 rather than reporting a green run. The repaired
plant selects a cell that is wet at both `jk` and `jk-1`, reports the baseline
operand it multiplies (`5.361255775252422e-4`, not a zero), and makes `zd_up`
and `zd_lw` each differ in exactly one cell, exit 1. `zdiag` does NOT move
under that plant: the induced change is ~1e-17 against a diagonal near 1,
below its own spacing. That is honest and is stated rather than hidden.

## Compiled-source basis

Two NEMO builds are involved and their compiled `zdftke.f90` files are
**identical outside the recorder calls** - the two files differ in ZERO lines
once the `r54_`/`r101_` call lines are removed. That was measured, not
assumed, and it settles for this file the operator's Round-101 note-N concern
(b) that the two builds might compute different things. The same statements
therefore carry different line numbers in each build, and each citation names
the producer of the array it describes.

Round-101 build, `GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90`:
the three `zfact` scalars at `:259-261`; the post-Langmuir record callback at
`:395`; `zcof` at `:426`; `zzd_up` at `:429-430`; `zzd_lw` at `:431-432`; the
three matrix writes at `:434-436`; the right-hand side at `:439-442`; the
record callbacks at `:470` and `:473`; the sweep and floor at `:475-495`.

Round-59 build, `GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90`: the
same statements at `:417`, `:420-421`, `:422-423`, `:425-427` and `:430-433`,
with the write call that produced the matrix arrays at `:463` and the writer
body at `GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/l2_r54_tke.f90:172-190`.

## The statement inventory, and what is excluded

Between the post-Langmuir callback and the right-hand-side callback, exactly
these output-bearing assignments execute on this card:

| order | output | statement | disposition |
|---:|---|---|---|
| 1 | `p_pdlr` | `zdftke.f90:421` | EXCLUDED, UNMEASURED-WITH-SPEC |
| 2 | `zd_up` | `zdftke.f90:434` | measured |
| 3 | `zd_lw` | `zdftke.f90:435` | measured |
| 4 | `zdiag` | `zdftke.f90:436` | measured |
| 5 | `en` RHS | `zdftke.f90:439-442` | measured |

`p_pdlr` is first in compiled order and is deliberately excluded from the
`en` walk rather than silently skipped. The reason is cited, not assumed: it
is written once at `:421` and read at exactly one place, `:712` inside
`tke_avn`, where it multiplies `p_avt`. It never enters `en`, so it belongs to
a separate consumer chain and gets its own walk (see OPEN).

The wave-coupled surface block at `:451-468` does NOT execute, on two
independent conditions: `cpl_phioc` is set `.TRUE.` only in
`sbccpl.f90:629`, which GYRE (forced, no coupler) never reaches, and
`ln_phioc = .false.` in `EXP00/namelist_ref:593`. `nn_pdl = 1` in the same
namelist, which is why row 1 exists at all.

## The walk, given NEMO's recorded stage entry, through the production step

Binding artifact `tke_matrix_walk.json`, SHA-256 recorded below, stamped clean
at commit `c571dab4439fb353d97066453725d12a3610f831`. Every row covers all
20,416 owned cells (`22 x 32` horizontal by NEMO levels `2:jpkm1`) with no
wet-mask exception and no numeric tolerance.

**Denominators, both of them.** The scored domain is every owned cell, 20,416
of them. Of those, 17,400 are WET and 3,016 are DRY; the dry cells agree by
construction, so a fraction taken against 20,416 always reads better than the
wet-cell fraction it implies. Both are therefore given side by side below and
at every headline count above. The wet count was MEASURED from the Round-59
record's own masks over the scored slice - `tmask` and `wmask` select the same
17,400 cells there, cell for cell, which is the same agreement the reviewer
confirmed independently - not assumed from the domain. Scoring keeps the full
20,416-cell denominator with no mask exception; this is a presentation of the
same rows, not a re-score.

| row | unequal | of 20,416 owned | of 17,400 wet |
|---|---:|---:|---:|
| `en` RHS, `zdftke.f90:439-442` | 11,993 | 58.7% | 68.9% |
| `p_sh2` operand, `zdftke.f90:439` | 17,400 | 85.2% | 100.0% |
| `en` post-sweep at the block exit | 979 | 4.8% | 5.6% |
| swap arm vs the production RHS | 11,029 | 54.0% | 63.4% |
| swap arm vs NEMO's recorded RHS | 2,663 | 13.0% | 15.3% |
| `zd_up`, `zd_lw`, `zdiag` | 0 | 0% | 0% |

The `p_sh2` row's 17,400 EQUALS the wet-cell total. That the unequal set IS
the wet set was not separately measured, so it is reported as the equality of
two counts and nothing more.

| order | output | statement | unequal | max abs | class |
|---:|---|---|---:|---|---|
| 2 | `zd_up` | `zdftke.f90:434` | 0 | 0 | **BIT** |
| 3 | `zd_lw` | `zdftke.f90:435` | 0 | 0 | **BIT** |
| 4 | `zdiag` | `zdftke.f90:436` | 0 | 0 | **BIT** |
| 5 | `en` RHS | `zdftke.f90:439-442` | 11,993 | `5.488912518947231e-10` | DEBT |

**The first non-bit statement ON THE `en` PATH is `zdftke.f90:439-442`.**
`p_pdlr` (`zdftke.f90:421`) precedes all of these in compiled order and is
unmeasured; see the scope note in the verdict.

The consumed operand rows, over the same domain:

| operand | source | unequal | max abs | class |
|---|---|---:|---|---|
| `p_sh2` | model's own, `tke_shear_evaluation_stage="step_entry"` | 17,400 | `3.811744924985501e-14` | DEBT |
| `p_avt` | recorded, installed by the entry bridge | 0 | 0 | BIT |
| `dissl` | recorded, installed by the entry bridge | 0 | 0 | BIT |
| `en` (post-Langmuir) | production, measured earlier in this walk | 0 | 0 | BIT |

## Ownership: the one-variable swap, and what it refuted

The first version of this round attributed the miss to `p_sh2` with a
SET-INCLUSION test - every cell where the right-hand side differs is a cell
where `p_sh2` differs (11,993 of 11,993; zero RHS-unequal cells have a
bit-equal `p_sh2`). The independent review refuted that as non-causal: the
same statement also consumes `p_avt`, `rn2`, `dissl`, the post-Langmuir `en`
and `wmask`, and coincident errors in those would pass the same test. The test
was replaced by a real one-variable swap through the committed transcription.

Counts are over the same 20,416 owned cells, of which 17,400 are wet; the
"Denominators" table above gives every row against both.

| arm | reference | unequal | max abs | class |
|---|---|---:|---|---|
| every operand recorded | NEMO's recorded RHS | 0 | 0 | BIT |
| only `p_sh2` swapped to the production value | the production RHS | 11,029 | `1.1102230246251565e-16` | AT-BAR |
| only `p_sh2` swapped to the production value | NEMO's recorded RHS | 2,663 | `5.488912657725109e-10` | DEBT |

**WHAT WAS PREREGISTERED AND WHAT WAS NOT.** P3 as frozen is the SET-INCLUSION
test, and nothing else: "every cell where the production RHS is unequal is a
cell where `p_sh2` is unequal". The one-variable swap is **POST HOC**, added
after the review refuted the inclusion test as non-causal, and registered as
post hoc in addendum 1 to the preregistration. This receipt does not claim the
swap was predicted, and it does not claim P3 predicted an exact swap.

Reading the numbers honestly:

- **The magnitude is `p_sh2`, and only `p_sh2`.** Substituting that single
  operand and nothing else moves the row from bit-exact to
  `5.488912657725109e-10`, against the production row's
  `5.488912518947231e-10` - the same number to eight significant figures - and
  the independent budget `rn_Dt * max|delta p_sh2| = 5.488912691979121e-10`
  closes to 3.2e-8 relative. Two independent routes to the same number.
- **The swap is NOT exact, and that is a limit on what can be claimed.** It
  leaves 11,029 cells differing at `1.1102230246251565e-16` from the production
  output, and it names 2,663 DEBT cells where the production step names 11,993.
  So "only `p_sh2` differs" holds for the MAGNITUDE and does not hold at the
  last bit. The receipt claims the former and not the latter.
- **What the residual is has NOT been determined.** `1.11e-16` is one unit in
  the last place for values in `[0.5, 1)` and sits six orders of magnitude
  below the `p_sh2` term, so it cannot own the block's DEBT or the 979-cell
  post-sweep gap. Two candidate causes remain open and this round does not
  choose between them: XLA fusion / fused-multiply-add reordering the same
  expression inside the full step (the effect operator note L confirmed on the
  held Round-89 member), or a genuine association difference between the
  model's right-hand-side expression and NEMO's. A discriminating measurement
  is named in OPEN. Calling it "fusion" now would be a mechanism asserted, not
  measured.
- The cell-count inflation from 2,663 to 11,993 is consistent with that
  last-bit residual pushing already-marginal cells across bit equality, but
  that too is PLAUSIBLE, not measured.

The gate's verdict field therefore reads `REFUTED`, not a confirmation, and
the receipt says so.

## Plants

`stage-tke-matrix-ulp` advances exactly one bitwise cell of the recorded
`zdiag` reference on the binding `NEMO_TKE_RECORDED` arm.

**WHAT PROVES THE PLANT FIRED, corrected.** An earlier version of this receipt
cited the gate's exit code. That citation was WRONG and is withdrawn: the gate
returns 1 whenever a plant is requested, whether or not the plant landed, so
its exit status discriminates NOTHING. The real proof is internal and is a
hard check, not a report: the targeted row must move from its clean count to
clean + 1, and the gate aborts if it does not. For this plant the `zdiag` row
goes from 0 unequal cells to exactly 1, at index `[0, 0, 0]`, and the plant
target is named `GYRE-zco.kt2.tke_matrix.production_step.zdiag` rather than
null, so the round-102 null-target failure is not repeated. Both the clean
count and the planted count are recorded per row in `tke_matrix_plant.json`,
which is what a reader should check.

**AND THE PRINTED LABEL IS FIXED.** The gate used to print `STATUS PASS`
while a plant was firing, because the plant paths prove themselves with that
internal check and never touch the report's status field. In a campaign whose
logs get scraped, a gate that prints success while deliberately failing is a
trap. A plant run now prints `STATUS PLANT-FIRED` and writes the same string
into its JSON, so neither the log nor the artifact can read as a pass. This
follows the two companion probes rather than inventing a scheme: the round-103
block replay derives its status from its rows and prints `STATUS FAIL` under
its plant, and the receipt citation gate exits 2 when its plant fails to fire.

Before and after, same command, `--plant stage-tke-matrix-ulp`:

| | printed label | exit |
|---|---|---:|
| before | `STATUS PASS` | 1 |
| after | `STATUS PLANT-FIRED` | 1 |

The exit code is unchanged and still proves nothing; the label no longer lies.
The clean run is unaffected and still exits 0 with `STATUS PASS`.

Both runs were re-done under the fix, at commit `96d098d549c2`, into new
artifacts so nothing frozen was overwritten. The plant's own rows, from
`tke_matrix_plant_relabelled.json`, are the evidence the label is not:

| row | clean | planted | plant index |
|---|---:|---:|---|
| `zd_up` | 0 | 0 | none |
| `zd_lw` | 0 | 0 | none |
| `zdiag` | 0 | **1** | `[0, 0, 0]` |
| `en` RHS | 11,993 | 11,993 | none |

Only the targeted row moved, by exactly one cell, and `plant_target` is
`GYRE-zco.kt2.tke_matrix.production_step.zdiag`.

The record replay's operand plant is described above under instrument
validation: its first version was a control that perturbed a zero, the probe
refused it, and the repaired version fires on a live cell.

Both plants fire through the same code path the science uses. The production
rows are measured through `LatLonCGridOceanModel.step` / `_step_jitted`, not
through an isolated closure (operator note L-amend); the replay arms are
labelled "recorded-operand replay of the compiled statement" and are never
called production.

## Prediction ledger

| # | prediction | outcome |
|---|---|---|
| P1 | the five output-bearing assignments between `:395` and `:473`, in that order; the wave-coupled block does not execute | **CONFIRMED** |
| P2 | first non-bit output is the `en` RHS at `:439-442`; `zd_up`, `zd_lw`, `zdiag` all BIT with 0 unequal cells | **CONFIRMED** |
| P3 | the RHS miss is inherited from `p_sh2`: the production `p_sh2` is non-bit, and every RHS-unequal cell is a `p_sh2`-unequal cell | **CONFIRMED AS FROZEN** - `p_sh2` is non-bit (17,400 cells, max `3.811744924985501e-14`) and zero of the 11,993 RHS-unequal cells has a bit-equal `p_sh2`. **But the test it specified is weak** and an independent reviewer refuted it as non-causal; the stronger one-variable swap that replaced it is POST HOC and is reported as such above. The swap supports the MAGNITUDE claim and does NOT reproduce the production output exactly. |
| P4 | no candidate lands; status HELD | **CONFIRMED** |

## Rule 12 and the testcase dispositions

No numerical or configuration candidate was created, so the 954-row ladder and
the days 1-30 arm would have had nothing to score. The immutable round-96/97
before arm remains the before arm and is unchanged by construction.

| headline | before arm | after this round |
|---|---|---|
| kt2 T RMS | `1.4210854715202004e-14` | unchanged; AT-BAR |
| kt2 S RMS | `2.1316282072803006e-14` | unchanged; AT-BAR |
| kt2 U RMS | `2.7377110452773967e-12` | unchanged; first-over-bar |
| kt2 V RMS | `3.284922138989399e-12` | unchanged; first-over-bar |
| kt3 T RMS | `1.627497246303733e-4` | unchanged; magnitude target |
| kt3 S RMS | `6.327735185607253e-6` | unchanged |
| day-30 T RMS | `1.2397011295506804e-2 K` | unchanged; magnitude target |

No AT-BAR row can have left the bar and first-over-bar cannot have moved,
because no candidate arm exists. This is not an improvement claim.

| lane | disposition |
|---|---|
| GYRE stage twin | instrument extended and PASS; the first owned stage is still kt1 stage 1 with W unresolved/held. This kt2 TKE walk is a magnitude diagnostic and does not advance stage order. |
| GYRE kt=1--10 | no candidate; no 954-row score; first-over-bar remains kt2 U/V |
| GYRE days 1--30 | no candidate; immutable day-30 T RMS retained |
| LOCK_EXCHANGE-zco | no shared numerical or card change landed; no new tank-fidelity claim |
| OVERFLOW-zps | no shared numerical or card change landed; partial-cell behaviour remains in spec |
| DINO | **SHARED-STATEMENT RISK:** DINO runs the shared TKE program and the shared solver, whose trace signature changed. The change is WRITE-only and reachable solely via `return_statement_trace`, which no production path requests; no DINO number can move. If a future round lands anything in this block, DINO needs its own production-stage and trajectory assessment. |
| ORCA2 | **UNMEASURED-WITH-SPEC:** resolve its integrator and score these same statement rows before any shared landing here. |

## Independent adversarial review

`codex exec --sandbox read-only` was available this session and returned a
full review of the diff, the claims and the compiled source. It was run TWICE. Its first-pass
verdict was **DO NOT SHIP**, verbatim:

> DO NOT SHIP: the gate can falsely attribute the RHS error to `p_sh2`, and
> its claimed replay/control is not committed.

It raised three findings. **All three were upheld and fixed; none was argued
away.**

1. *High.* "the new `CONFIRMED_INHERITED_P_SH2` verdict is not causal. It only
   checks that RHS-different cells are a subset of `p_sh2`-different cells...
   NEMO's RHS also consumes `p_avt`, `rn2`, `dissl`, prior `en`, and `wmask`.
   Coincident errors in those operands would produce the same 'confirmed'
   result. A swap/replay holding every operand fixed except `p_sh2` is
   required." **Upheld.** The subset test was demoted to a reported number
   explicitly labelled as not being the evidence, and the one-variable swap
   above replaced it, registered as POST HOC in addendum 1 to the
   preregistration. Running that swap then showed the inclusion test was not
   the whole story: the swap is not exact.
2. *High.* "C6's replay is not shipped... an uncommitted replay cannot support
   the shipped claim or its plant." **Upheld.** The replay is now
   `scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round103_tke_block_replay.py`,
   carries a worktree stamp, has a direct test with a non-vacuity arm, and the
   stage gate imports its transcription so there is exactly one copy of the
   arithmetic.
3. *Medium.* "the recorded provenance is false. The report says the Round-59
   record writer was called from the Round-101 source at line 472; the actual
   Round-59 compiled call is `zdftke.f90:463`. Round-101 line 472 is a
   different binary." **Upheld.** Corrected, and the two builds were then
   diffed: they are identical outside the recorder calls, which is now stated
   with both builds' line numbers.

It also reported that "C2, C3, and C5 survived attack: shapes are exact,
`tmask == wmask` over all 20,416 scored cells, and the deepest recorded upper
diagonal is live." The `tmask == wmask` result is worth keeping: the model's
literal assembly uses `w_active` where NEMO's `zcof` uses `tmask`
(`zdftke.f90:426`), and on this card those two masks agree on every scored
cell, so that difference is measurably inert here. It would NOT be inert on a
grid with an overhang, and any future card must re-check it.

A SECOND pass was then run against the fixed diff, this receipt and the
preregistration. It confirmed that finding 2, both plants, the trace-only
production safety and the HELD status check out, and returned **DO NOT SHIP**
again with three further findings. **All three were upheld and fixed; none was
argued away.**

4. *High.* "'first non-bit statement' is false globally. `p_pdlr` executes
   first but remains unmeasured; exclusion is valid only for the `en` chain.
   The receipt admits this... while making the unconditional headline."
   **Upheld.** The headline is now scoped to the `en` path in the verdict
   itself, before the claim rather than after it, and the scope note says
   plainly that `p_pdlr` could be non-bit and nothing here would know.
5. *High.* "the receipt rewrites P3. The preregistration specifies only the
   non-causal subset test, not an exact swap. The receipt falsely says exact
   reproduction was preregistered and 'REFUTED as written'." **Upheld, and it
   is the more serious of the two.** P3 as frozen is the inclusion test and it
   is CONFIRMED; the swap is post hoc. Calling a confirmed prediction refuted
   misstates the frozen record exactly as badly as the reverse would. The
   ledger and the swap section are corrected, and the swap is registered as
   POST HOC in addendum 1 to the preregistration.
6. *Medium.* "F3 persists in the frozen preregistration: it still says the
   Round-59 record came from Round-101 `zdftke.f90:472`." **Upheld.** The
   preregistration is frozen and is not rewritten; addendum 1 corrects it in
   place, dated, with the measured reason the conclusion still holds.

The review logs are `codex_review.log`, SHA-256
`97be6f91a469f2ee6fb202948283e678d9c03d46fd9a18b4fb5fa23481c6ae02`, and
`codex_review2.log`, SHA-256
`fc9c9ca3083cff2e41a70e1b42ec1a9dd800106542c39ebb7860413d0884736f`. Neither pass returned SHIP; the second pass's blockers
are prose and record-keeping defects with no numerical consequence, and every
one of them is fixed above. A third pass was not run, so **no review has
returned SHIP on the final text** - that is stated rather than implied, and the
next round should treat it as an open item if anything here is built on.

## Tests

Focused CPU/fp64 suite over the new walk, the consolidated stage gate, the
Round-54/59 operand reader, the citation gate and the TKE carried-coefficient
physics: **120 passed in 32.50 s**, pytest's own summary line verbatim
`120 passed in 32.50s`. JUnit artifact `focused_pytest.xml`, SHA-256
`f086a2d9679c3b682c9e63fb4e727f07cc9f7b1b04bdeeeec45c04f6734471e0`.

The count moved from 119 to 120 because the review fixes above add ONE test,
covering the plant label: a plant run must not be able to print a success
label, in either direction. No test was removed, weakened or renamed.

**FINGERPRINT AUDIT.** An earlier version of this receipt recorded a
`focused_pytest.xml` fingerprint that did NOT match the file on disk: the
suite was re-run after the receipt was committed and the hash was never
refreshed. The substance was never in doubt - that run was 119 tests, zero
failures - but a fingerprint that does not verify defeats the entire point of
a receipt, so it is treated as a defect and not as a typo. EVERY SHA-256 in
this receipt was then re-computed against the file it names. The receipt
carried EIGHT distinct fingerprints at the time of the review
(`instrument_repro.json`, `record_replay.json`, `record_replay_plant.json`,
`tke_matrix_walk.json`, `tke_matrix_plant.json`, `focused_pytest.xml`,
`codex_review.log`, `codex_review2.log`): SEVEN verified unchanged and only
`focused_pytest.xml` was stale. The preregistration carries none. Every file
in the evidence directory was hashed as well, and the two citation-gate
artifacts that previously carried no fingerprint now carry one. The receipt
now carries TWELVE distinct fingerprints and ALL TWELVE verify against the
file each one names, re-checked after the last edit to this text.

The citation suite passes 16/16 on a clean tree. Three anchor shifts were
caused by this round's insertions into the stage gate and were re-anchored by
RIGID shifts with both endpoints moved by the same delta and the pinned extent
unchanged: `+187`, then `+1` twice as the diff grew. No citation was weakened
or deleted.

`tests/ocean/fidelity/test_nemo_testcase_worktree_stamp.py::test_every_report_emitter_stamps_the_worktree`
remains red with the SAME NINE offender files it had at the incoming tip. The
new replay probe was briefly a TENTH offender and was fixed before this
receipt by stamping its report; the counts were compared against the incoming
tip mechanically (11 report dicts / 10 unstamped in the stage gate, both
before and after).

## Evidence

All under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round103/`.

| artifact | SHA-256 |
|---|---|
| `instrument_repro.json` | `e674f45302126d5642aaea289fa853ea21cb06762ac8b18577f82aa27811764c` |
| `record_replay.json` | `b339e8ebf809a046b140efd303e5791175f410f2e5b39a63aa499adbcdc7a466` |
| `record_replay_plant.json` | `573bb97f3e426ee796d5855a432efa5b62367458a5b927d6bc0843ba7ee7a9ab` |
| `tke_matrix_walk.json` | `241bb02b065fb730d99f22824f3f1f94004b41ceb97a257e76c994a6c7a5e542` |
| `tke_matrix_plant.json` | `4cc4e7836b49987a69be14faab0aff3f96cf7e9f42e27915e41c0ad6a93274ef` |
| `focused_pytest.xml` | `f086a2d9679c3b682c9e63fb4e727f07cc9f7b1b04bdeeeec45c04f6734471e0` |
| `codex_review.log` | `97be6f91a469f2ee6fb202948283e678d9c03d46fd9a18b4fb5fa23481c6ae02` |
| `codex_review2.log` | `fc9c9ca3083cff2e41a70e1b42ec1a9dd800106542c39ebb7860413d0884736f` |
| `citation_gate.json` | `dc26774979dd38f7c5b10c633165d50f6c9894bfebb8553b4b6b71928b1814c8` |
| `citation_gate_plant.json` | `8074f0fe81df8aff521f4fbca1e77ba823deb815297db7914add90df52ff052a` |
| `tke_matrix_walk_relabel_check.json` | `b4020efa8d13b6be70c7745efdba7c7b9b03fe23505b7f80834b210a1b6c6b8f` |
| `tke_matrix_plant_relabelled.json` | `1921438fea4ee182bd303eff3cef0d237cec0b790355ea0f22a741964551801e` |

The citation gate run is 274 citations, zero failures, `STATUS PASS`, exit 0;
its plant on `domqco.F90:189-209` fails with `SYMBOL-NOT-AT-LINE` and exits 1,
and that probe exits 2 if its plant does NOT fire, so its exit code is
evidence where this gate's is not.

The last two rows are the re-run under the corrected plant label. THE CLEAN
RE-RUN REPRODUCES EVERY NUMBER IN THIS RECEIPT BIT-FOR-BIT - all four block
rows, the `p_sh2` operand row, all three one-variable-swap rows and the
`REFUTED` verdict, compared field by field against the original
`tke_matrix_walk.json` - which is the measurement that the label change is
numerically inert, rather than an assertion that it must be. The original
artifacts are untouched and keep their own fingerprints above; nothing was
overwritten.

A fingerprint pins ONE run of the tool that produced it. Re-running the suite
or a gate after this receipt is committed invalidates the hash even when every
number is identical, which is exactly how the stale `focused_pytest.xml`
fingerprint arose. Refresh the hash in the same commit, or do not re-run.

## OPEN - round 104

1. **Walk `zdfsh2`, not this block.** The named statement's arithmetic is not
   the owner; its `p_sh2` operand is, and `p_sh2` carries the whole
   `5.4889e-10` right-hand-side magnitude and therefore the 979-cell,
   `6.8096882e-12` post-sweep gap at the block exit. The model computes it
   under `tke_shear_production="nemo_face_native_now2"`,
   `tke_shear_avm_weighting="nemo_face"`,
   `tke_shear_metric_source="nemo_qco_live_face"`, evaluated at
   `tke_shear_evaluation_stage="step_entry"`. Round 104 should subdivide that
   transcription against `zdfsh2.f90:83-114` in the Round-59 build the same
   way this round subdivided the matrix block. The Round-59 record already
   stores `sh2`, so a new acquisition is needed ONLY if per-statement
   boundaries inside `zdfsh2` are wanted; the final `sh2` comparison needs
   nothing new. NOTE the campaign's known open item #1455 on this exact chain.
2. **Decide what the `1.1102230246251565e-16` residual is.** The one-variable
   swap leaves 11,029 cells differing at one unit in the last place between a
   NumPy transcription of `zdftke.f90:439-442` and the model's evaluation of
   the same statement inside the full step. Two candidates, not chosen between
   here: XLA fusion / FMA reordering (operator note L's confirmed effect), or a
   genuine association difference. DISCRIMINATING MEASUREMENT: evaluate the
   model's right-hand-side expression on the same operands (a) eagerly, (b)
   under `jax.jit` of an isolated closure, and (c) inside the production step,
   and compare all three against the NumPy transcription. If (a) matches NumPy
   and (c) does not, it is fusion; if (a) already differs, the model's
   association differs from NEMO's and that is a second, in-block owner at the
   AT-BAR level. It cannot own the DEBT either way - it is six orders of
   magnitude too small - so it does not block item 1.
3. **`p_pdlr` (`zdftke.f90:421`) is still UNMEASURED-WITH-SPEC.** It is first
   in compiled order and feeds `p_avt` at `zdftke.f90:712`, so it owns part of
   the diffusivity the NEXT step consumes even though it never touches `en`.
   Exposing it needs a return threaded out of the model's Prandtl helper. The
   Round-59 record already stores `pdlr`, so again no acquisition is needed.
4. **Do not re-walk what is closed.** `zd_up`, `zd_lw` and `zdiag` are BIT
   given NEMO's entry and should not be re-measured except as regression. The
   ordered `zpelc` recurrence is measurably inert (operator note Q). Held
   patches under `manifests/` were not re-evaluated: none names a statement in
   `zdftke.f90:424-443`.
5. **No review has returned SHIP on the final text.** Two adversarial passes
   ran; the first returned DO NOT SHIP on three findings that were numerical
   and procedural, the second on three that were prose and record-keeping. All
   six were fixed, but a third pass was not run, so the fixes are unreviewed.
   Round 104 should re-review this receipt before building on it.
6. **No acquisition is pending.** The admitted Round-59 and Round-101 records
   contain every array this round and the next one need.
