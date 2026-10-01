# NEMO testcase fidelity: round 195 / VORTEX round 11 — the post-ZAD owner and the held candidate's second statement

**Status: HELD.** Nothing lands in production. Two measured results:

1. The first non-bit producer of the kt=2 U/V residual that survives round
   194's held candidate (`1.2462615e-08` / `1.0562746e-08`) is the
   **external-mode (barotropic) solve**, whose output the per-stage
   correction adds at
   `VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:421-429`.
   The stage momentum right-hand side, the velocity update and the
   correction arithmetic are all already at the compiled-rounding floor.
2. The held candidate's single card edit selects **two** NEMO statements,
   not one, and they fail the certified gate for **two different reasons**.
   The literal stage continuity recurrence alone changes no row's STATUS and
   buys none of the velocity gain (kt=2 u identical to production's last
   bit), yet still breaks the two-ULP cellwise ratchet on 33 of the 50
   certified rows; the second per-stage solve alone owns both the kt=2 U/V
   improvement and the kt=5 salinity AT-BAR loss — and that loss is a
   **single floating-point quantum**.

Evidence root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round195/`.
Frozen preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l1_vortex_round195.md`,
committed as `71808510d` before any measurement.

## Compiled program

The RK3 step solves the external mode once, before the stages, and saves
its result at
`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:133-135`
("save ssh, uu_b, vv_b at N+1 (computed in dynspg_ts)"). Every stage then
replaces the depth mean of its own updated velocity with that saved
barotropic velocity at
`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:421-429`:
`zub = uu_b(Kaa) - SUM(e3t_1d*uu(:,Kaa))*r1_hu_0`, then
`uu(jk,Kaa) = uu(jk,Kaa) + zub*umask`. The stage-term instrument writes the
velocity on both sides of that block — `update_u`/`update_v` before it and
`out_u`/`out_v` after it (`vortex_r8_stage_finish`, patched in at
`stprk3_stg_record` line 463 of the shipped `.F90`) — so the correction and
its operand can be scored separately.

`uu_b(Kaa)`/`vv_b(Kaa)` and `un_adv`/`vn_adv` are therefore **operands of the
stage, produced outside it**. That is what this round substitutes.

## One-variable ownership of the remaining kt=2 U/V residual

Both arms are the round-194 held candidate card, the same production-jitted
`model.step`, driven from NEMO's recorded stage entry; the **only** variable
is whether NEMO's recorded barotropic quintuple
(`ssh`, `uu_b`, `vv_b`, `un_adv`, `vn_adv`, from
`phase3/round192/oracle_stage23_terms/oracle_bt_frames_kt00000001.bin`)
replaces legoESM's own solved one.

| stage | barotropic output | normalized max abs vs NEMO | status |
|---:|---|---:|---|
| 2 | NEMO recorded | `1.11022302462515654e-16` (u and v) | AT-BAR |
| 3 | NEMO recorded | `2.22044604925031308e-16` (u and v) | AT-BAR |
| 2 | legoESM's own solve | u `1.24589062771391176e-08`, v `1.05595161158665909e-08` | DEBT |
| 3 | legoESM's own solve | u `1.24589062771400705e-08`, v `1.05595161158674909e-08` | DEBT |

Removing the substitution degrades the stage output by `1.1e+08`x. The
numbers it degrades *to* are the trajectory's own kt=2 rows under the
candidate (`1.2462615e-08` / `1.0562746e-08`) to three significant figures,
so this is the same error, not a second one.

Everything upstream is already exact under the candidate, from the same two
runs: the cumulative momentum accumulator at the KEG and ZAD boundaries is
`1.355e-20 .. 2.033e-20` (round 194's result, reproduced here). The HPG and
VOR rows in the same file read `6.72e-06` — that is NOT a contradiction and
NOT an error in the accumulator: round 193 established that legoESM
associates the accumulator's terms in a different ORDER than NEMO, so the
intermediate boundaries disagree while the completed sum agrees, and the
round-193 `--source-order` arm makes the intermediate boundaries bit-exact
too without changing any certified number. The relevant quantity here is the
completed accumulator, which is the ZAD boundary. With the
barotropic operand supplied the whole stage reproduces NEMO's stage velocity
at `1.1e-16`. Since the correction replaces exactly the depth mean, the
baroclinic part of the velocity update is proven exact by the AT-BAR row and
only the depth-mean (barotropic) part can carry the residual.

Corroborating, unchanged by either statement: the certified kt=2 sea-surface
row is `3.709009770e-08` in production, in arm B and in the held candidate
— the same external-mode solve, the same error, untouched by anything the
two momentum statements do.

**The first non-bit producer is named, but its own first non-bit statement is
not**: the round-192 record carries the barotropic solve's *output* frames
only, not its substep operands. Naming the statement inside `dyn_spg_ts`
needs a new NEMO record — see OPEN.

## The held candidate is two statements — one-variable decomposition

`nemo_stage_momentum_wzv_executes` returns False unless
`wzv_call2_evaluation == "nemo_literal"`, and that same field independently
selects the literal `sshwzv` continuity recurrence for **every** stage
transport (momentum and tracers alike). So round 194's one-line card edit
turns on two things. Three arms, each a clean commit, each scored against
the same certified round-191 reference with the **unchanged** two-ULP gate:

| arm | card fields | kt=2 u | kt=5 S | rows with a cell past 2 ULP | worst cell worsening (ULP) | gate |
|---|---|---:|---|---:|---:|---|
| A control | production (`generic`, unset) | `3.369329786e-06` | `8.120488409e-16` AT-BAR | 0 | `0` | **PASS** |
| B | `nemo_literal`, split `False` | `3.369329786e-06` | `8.120488409e-16` AT-BAR | 33 | `18.688` | FAIL |
| C (round 194's held candidate) | `nemo_literal`, split `True` | `1.246261511e-08` | `1.015061051e-15` DEBT | 41 | `2063747067` | FAIL |

Arm A is the instrument control and it is **exact**: production in this fresh
worktree reproduces the certified round-191 residual fields with
`max_worsening_ulps = 0`, so every move below is the statement's, not the
worktree's.

Arm B keeps **22 of the 50** row maxima bit-identical to production,
including both headline rows (kt=2 u and kt=5 S); the other 28 move by at
most `5.9e-10` relative, 21 of them toward NEMO and 7 away, and **no row
changes status**. It buys none of the candidate's kt=2 velocity gain — kt=2 u
is identical to production's last bit — and still breaks the ratchet on 33 of
the 50 certified rows. The one cell each violating row reports (the gate
names a row's FIRST offending cell, not its worst) is **disjoint** from arm
C's set: 0 of 33 shared. So the literal recurrence is a last-bit reshuffle
with no measurable benefit and a real ratchet cost.

An earlier draft of this receipt said arm B reproduced *every* row maximum to
16 digits. That is wrong and is corrected above: 28 of 50 move. The
independent review below caught it; the substantive verdict is unchanged.

Arm C minus arm B is the second per-stage continuity solve alone. It owns
the whole kt=2 U/V improvement and both salinity status moves — the gate
records two, not one: kt=5 S `AT-BAR -> DEBT` and kt=7 S `DEBT -> AT-BAR`
(round 194's receipt reported only the loss).

**The kt=5 salinity loss is one quantum.** On that row the residual scale's
quantum is `2.030122102e-16`; production is `8.120488409e-16` = exactly 4.0
quanta, the candidate is `1.015061051e-15` = exactly 5.0 quanta, and the
AT-BAR bar of `1e-15` sits at 4.926 quanta — between them. The row is at the
compiled-rounding floor on both sides and the bar happens to fall inside the
last bit.

## Predictions (frozen before measurement), kept with their verdicts

* **P1 — the second difference is the tracer-side literal call-2, not the
  momentum split.** **PARTLY REFUTED, and the refutation is the finding.**
  The prediction that arm B reproduces the kt=5 S `AT-BAR -> DEBT` change is
  REFUTED: arm B's row statuses are identical to production. The prediction
  that arm B reproduces the two-ULP cellwise FAIL while leaving kt=2 U/V
  unmoved is CONFIRMED (33 rows, worst 18.688 ULP, kt=2 u identical to 16
  digits). So the veto has two owners, not one: the ratchet half is the
  literal recurrence, the status half is the split.
* **P2 — the two statements are separable and additive.** CONFIRMED for
  separability (B shows none of C's kt=2 improvement), REFUTED for
  additivity: B's and C's flagged cells share nothing.
* **P3 — the first non-bit producer of the remaining kt=2 U/V residual is
  the barotropic correction's operand.** **CONFIRMED.** The falsifier was
  "within 10x"; the measured factor is `1.1e+08`.
* **P4 — nothing lands.** CONFIRMED. No card selection changed; the
  production diff for this round is empty (below).

## Plants (non-vacuity), all run this round

* Stage walk, `--plant s2.zad.u` on the production-barotropic arm:
  `STATUS PLANT-FIRED`, exit 1, the planted row reads
  `max=1.00000000000159361e+00` against its unplanted `1.8905e-09` sibling
  (`phase3/round195/plant_stage_walk.log`).
* Two-ULP gate on the control's own residual fields:
  unplanted `PASS 0 ULP exit 0`; `--compare-plant at-bar-to-debt` fires
  `status crossed AT-BAR -> DEBT`, exit 1; `--compare-plant worsen-3ulp`
  fires `3.000 row-scale oracle ulp; bar is 2 ulp`, exit 1
  (`armA_control_plant_*.json`).
* Citation gate self-test: see below.

## Gates and tests

* **Instrument control (the most important line):** production card, fresh
  worktree, against the certified round-191 reference —
  `ORACLE_RELATIVE_COMPARE PASS: rows=50 max_worsening_ulps=0`.
* **Citation gate** on this receipt: `"status": "PASS"`,
  `"unmapped_citations": []`, `"citations_found": 2`, worktree clean at
  commit `5155d1318da844f2ab586d4183f2e4411202fcbb`
  (`phase3/round195/citations.json`). Its planted control, shifting
  `stprk3_stg.f90:421-429` by two lines, fires
  `SYMBOL-NOT-AT-LINE ... that symbol identifies line 421` and exits 1
  (`citations_plant.json`).
* **Focused battery**, run serialized after confirming no other pytest was on
  the host: the citation-gate tests, the three VORTEX walk-script tests, the
  VORTEX card test and the NEMO recipe test —
  `128 passed in 357.99s (0:05:57)` (`phase3/round195/focused_pytest.log`).
  The lane's full required battery is the one the operator's `land.sh` runs
  at the push gate; its decisive line is recorded with the push.
* **Production diff for this round is empty.** `git diff` of the round's
  first commit against its last, restricted to `packages/` and `src/`,
  is zero lines: the two scratch arms were committed and reverted, so no
  card, no model file and no certified number can move. GYRE, DINO, the
  tanks and the generic NEMO-GYRE recipe are untouched by construction.
  The DINO month gate is run by `land.sh` regardless.

## Landing verdict: HELD

Round 194's candidate stays held, now with its veto attributed. Nothing in
production changed; this round lands the preregistration, the extended stage
walk (one new measurement arm on an existing script), two citation-map
entries and this receipt.

## OPEN — round 196 / VORTEX round 12

Three items, in priority order.

1. **DECISION_NEEDED (operator).** The held candidate's veto now has two
   separable owners and one of them is a single last-bit quantum. Choose:
   (a) keep both statements held until the barotropic owner below is fixed,
   so the whole vector kt=2 program lands at once; (b) land the pair with
   the kt=5 salinity row registered as a one-quantum move at the floor (an
   exception of the Decision-76 kind, since the row is 4 vs 5 quanta of
   `2.030122102e-16` with the bar at 4.926 quanta) — the 33-row ratchet
   break from the literal recurrence would still have to be waived, and it
   buys nothing on its own, so this is not recommended; (c) rule that a
   NEMO-literal statement which changes no row's status may not be held
   back by the cellwise ratchet alone, i.e. amend Decision 71. My pick is
   (a).
2. **ACQUISITION_NEEDED.** Naming the first non-bit statement inside the
   external-mode solve needs NEMO's barotropic substep operands, which no
   existing record carries (round 192 records the solve's *output* frames
   only). Build the acquisition on the VORTEX_VEC_R8_OMIP_L1_P3 build with a
   self-describing per-substep record of `dyn_spg_ts`'s accumulators, in the
   round-35 format and with a header-parsing checker (operator note BD), and
   stop for the operator to run NEMO. Until then the barotropic owner is
   named but not resolved.
3. The flux card's (`VORTEX-zco`) own kt=2 owner is still untouched since
   round 4. Decision 74's 30/15/10-km ladder stays blocked (note BO).

## Independent adversarial review (fresh reviewer, this round)

Codex is out of budget on this account for this window and the GLM tool is
not reachable in this session, so the mandatory second opinion was a fresh
reviewer agent given the diff, the evidence root and the claims, with no
knowledge of how they were produced. Its verdict was **DO-NOT-SHIP until two
things are fixed**, and both are now fixed in this file:

* **DEFECT (accepted, corrected above).** The claim that arm B reproduced
  *every* row maximum to 16 digits is false: 22 of 50 are bit-identical, 28
  move (largest `5.9e-10` relative, plus one quantum-level salinity row at
  kt=4 that improves 4 quanta -> 3), 21 toward NEMO and 7 away, no status
  change. Re-measured here before correcting.
* **BLOCKER (accepted, corrected above).** The gates section pointed at a
  "Battery" section that did not exist; the citation-gate and focused-battery
  lines are now quoted in place.
* **NIT (accepted, corrected above).** "Everything upstream is already
  exact" did not explain the `6.72e-06` HPG/VOR rows sitting in the same
  file; round 193's accumulator-order finding is now cited there.
* The reviewer independently CONFIRMED: the instrument control at 0 ULP; the
  empty production diff and both scratch commits round-tripping to zero
  (`git diff 9d384eaf7^ 4ac38a903` and `git diff 2b37155a0^ f8ff46de2` are
  each 0 lines); the one-quantum salinity arithmetic (it recomputed the
  quantum as `spacing(35)/35 = 2.0301221021717148e-16` and found the kt=5 and
  kt=7 salinity rows swap values exactly, which is stronger than this receipt
  claimed); that the script extension's default path is byte-identical to
  before the edit and the new keys overwrite nothing; and that scoping the
  owner to the whole external-mode routine rather than a statement inside it
  is honest, because the finer attribution is declared ACQUISITION_NEEDED.
* The reviewer could not re-derive the "flagged cells are disjoint" claim
  from the persisted JSON alone and left it UNVERIFIED. It is measured here
  from the two comparison reports' own violation lists; the sentence has been
  tightened to say exactly what a violation line reports (a row's first
  offending cell, not its worst).

No finding changes the HELD verdict or the owner attribution.
