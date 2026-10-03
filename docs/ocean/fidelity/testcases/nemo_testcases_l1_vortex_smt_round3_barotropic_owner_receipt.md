# Round 213 / VORTEX_SMT round 3 — the records re-acquired; the partial-cell face-thickness statement MEASURED and HELD

**VERDICT: the re-acquisition LANDS; both candidate statements are HELD.** A
fresh adversarial reviewer returned **DO NOT SHIP** on the first candidate and
was right (§2.4): the repair as written replaced one wrong array with a
differently wrong one. No model code changes in this commit.

**Decision 88 (user, 2026-10-03), operator note CC addendum 3.** Two items:
re-acquire the four seamount records with the committed (print-fixed) hook and
prove them bit-identical, then walk the kt=2 owner over partial cells in NEMO's
own order and land the first cited statement.

Predictions were written before any build, run or ladder:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/round3/predictions.md`
(its ADDENDUM, naming the two candidate statements S1 and S2, was written
before either was applied and before any ladder was scored).

---

## 1. THE RE-ACQUISITION — EVERY INITIALISED BYTE IDENTICAL, AND ONE REAL FINDING

Four new NEMO configurations, `VORTEX_SMT_R3_{,VEC_R8_}OMIP_L1{,_P3}`, built
from the committed hook by four new `run.sh` variants
(`smtflxr3 smtvecr3 smtflx100dr3 smtvec100dr3`).  Round 1's four build
directories and four evidence directories were not touched, so the comparison
is a measurement.  All four records `ADMITTED`, header plant fired on all four,
every instrumented restart byte-identical to the un-instrumented build's.

| record | files compared | bit-identical to round 1 |
|---|---:|---|
| `VORTEX_SMT_R3_OMIP_L1_P3/kt1_10` | 27 | **27 / 27** |
| `VORTEX_SMT_R3_VEC_R8_OMIP_L1_P3/kt1_10` | 29 | 27 / 29 |
| `VORTEX_SMT_R3_OMIP_L1_P3/day100` | 3265 | **3265 / 3265** |
| `VORTEX_SMT_R3_VEC_R8_OMIP_L1_P3/day100` | 3267 | 3265 / 3267 |

("files" = every `oracle_*.bin`, both arms' step-`N` restarts and
`mesh_mask.nc`.)

**The two exceptions, diagnosed rather than waved past.** Both are the vector
card's stage-terms record at kt=1 (stages 2 and 3), the round-192 writer.  In
each 6.7 / 7.5 MB file exactly **28 bytes** differ; the headers and every
group's metadata are equal, and every differing byte lies in the `ww` group's
payload at elements 0,1,2,3,4,8,9 of 49 379 — i.e. `(i = 1..10, j = 1, k = 1)`
of the `(67,67,11)` Fortran-ordered array, the SOUTHERN HALO ROW that every
consumer strips (`plane[2:-2, 2:-2]`).  The values are denormals of order
`1e-310` on both sides: memory NEMO never writes.

**CONTROL, and it is decisive.** The R3 executable was re-run a second time,
unchanged, into a fresh directory (`round3/repeat_vec`).  Its two stage-terms
files differ from the R3 run's **as well**, while
`oracle_step_entry_kt00000001.bin` is byte-identical across R1, R3 and the
repeat.  So the 28 bytes are run-to-run nondeterminism of unwritten memory on
ONE unchanged binary — not the hook, not the build, not physics.

**P1 CONFIRMED on every initialised value; the hook fix is proven print-only.**
The R3 records are now the admitted ones (the trajectory gate's
`DEFAULT_ORACLE_ROOTS` points at them).

**P2 CONFIRMED.** `ocean.output` now prints the GLOBAL `k_bot min = 8` where
round 1's rank-local print said `0`; bathymetry `4000.000 / 5000.000` m and
bottom `e3t` `50.6710 / 500.0000` m are unchanged.  That is the whole of the
round-2 fix doing its job.

**OPEN, instrument defect (new):** the stage-terms writer dumps `ww`'s
uninitialised halo.  Harmless today (every consumer strips the halo) but it
makes two otherwise identical runs produce different files, which is exactly
what a byte-identity check is for.  Zero the array before the dump, or dump
the interior only.

---

## 2. THE WALK, IN NEMO'S ORDER, OVER PARTIAL CELLS

NEMO's `stp_2D` order on this deck, read off the compiled SMT ppsrc:
`eos / dyn_hpg / dyn_ldf / dyn_vor` → `CALL wzv` (`stp2d.f90:153`) →
`dyn_keg / dyn_zad` (vector) or `dyn_adv_up3` (flux) → the vertical averaging
into `Ue_rhs/Ve_rhs` (`stp2d.f90:177-186`) → `dyn_drg_init`, wind → `dyn_spg_ts`.

### 2.0 The static depths are EXACT — the walk does not start there

`hu_0 = SUM_k e3u_0*umask` and `r1_hu_0 = ssumask/(hu_0 + 1 - ssumask)`
(`domain.f90:193-215` of `VORTEX_SMT_OMIP_L1_P3`) are built by the card from
operands round 2 already proved exact at zero ULP, by the same sum.  **P3
holds**; the first non-bit statement is a CONSUMER of the depths, not a depth.

### 2.1 S1 — THE FIRST NON-BIT STATEMENT: the reference U/V face thickness

**legoESM aliased `e3u_0 = e3v_0 = e3t_0` whenever a card carries NEMO's raw
mesh operands.** The comment said so: *"NEMO's own mesh: e3u_0/e3v_0 are e3t_0
on the full-step meshes this branch serves"*.  Over partial cells that is
false.  NEMO's reference face thickness is the **shallower neighbour's**
reference T thickness —
`e3u_0(i,j,k) = MIN( e3t_0(i,j,k), e3t_0(i+1,j,k) )`,
`e3f_0` from `e3v_0` — built by `tools/DOMAINcfg/src/domzgr.F90::zgr_zps` and
reached through the `E3u_0` macro of `domzgr_substitute.h90` under
`key_vco_1d3d`.  Round 1 printed the very number: at `glamt = -480` km NEMO's
`e3u` is **132.1 m** where `e3t` is **263.1 m**.

MEASURED, geometry only, before any time step: on `VORTEX_SMT_VEC-zps` the
aliased and the correct face thickness differ on **1 084 wet U cells** by up to
**170.38 m**.

**Where it executes.** The alias lives in the one shared qco operand builder
(`nemo_qco_resolved_mesh_operands`), whose consumer on this card is
`nemo_qco_wzv_operands` — the `zad_qco_evaluation="nemo_literal"` arm that
every NEMO testcase card selects, i.e. NEMO's `CALL wzv` at `stp2d.f90:153`.

**RETRACTED, by the reviewer and verified:** an earlier draft of this receipt
and the first commit message called that "the FIRST statement in the step that
reads a U-face thickness".  It is not.  `CALL dyn_vor` runs at `stp2d.f90:142`
and `vor_een` reads `e3u_3d`/`e3v_3d` at `dynvor.f90:766-767`, before `wzv`.
What is true is narrower and is what the measurement supports: the EEN
vorticity path reads the card's OWN `e3u_0`/`e3v_0` — the bundle round 2
proved exact against `mesh_mask.nc` at zero ULP — so it is not affected by the
alias, and `wzv` is the first statement that reaches the ALIASED operand.

**THE DEFECT IS REAL; THE REPAIR TRIED THIS ROUND IS NOT CORRECT.** The arm
measured below deleted the alias and applied the shared min rule inside the
one kernel.  The reviewer refuted that repair (§2.4) and it is HELD, not
landed (`manifests/nemo_testcase_l1_vortex_smt_round213_e3u0_min_rule_held.patch`).
The numbers below are therefore a MEASUREMENT of what the statement is worth,
not a landing.

#### S1 — ONE VARIABLE, BEFORE vs AFTER, both cards, bar 1e-15

| kt=2 row | `VORTEX_SMT-zps` before | after | `VORTEX_SMT_VEC-zps` before | after |
|---|---|---|---|---|
| T | 4.259549e-10 | 4.259549e-10 | 1.733725e-09 | **4.539846e-11** (38x) |
| u | 6.308422e-08 | 6.308422e-08 | 1.748385e-07 | **6.004125e-08** (2.9x) |
| v | 4.742908e-08 | 4.742908e-08 | 1.607001e-07 | **4.037530e-08** (4.0x) |
| ssh | 3.726197e-07 | 3.726197e-07 | 3.664757e-07 | 3.664757e-07 |

The flux card is **exactly unmoved**.  An earlier draft explained that by
"NEMO calls `dyn_zad` only in the vector branch of `stp2d`'s `SELECT CASE`".
**RETRACTED:** `CALL wzv` at `stp2d.f90:153` sits OUTSIDE that `SELECT CASE`
(which opens at `:157`), so NEMO computes `wzv` unconditionally.  The correct
reading of the unmoved flux card is a FINDING, not a control: legoESM
materialises this operand only for the `zad` arm, which the flux card does not
run, where NEMO's `wzv` is unconditional.  Whether that conditionality is
itself a fidelity gap on the flux card is an OPEN item.

### 2.4 THE ADVERSARIAL REVIEW, AND WHY THE REPAIR IS HELD

A fresh reviewer (not the author, read-only, given the diff, the receipt and
the compiled NEMO sources) returned **DO NOT SHIP**, with one finding that is
decisive and three that correct this receipt.  Its findings, and what was done:

| # | finding | taken |
|---|---|---|
| MAJOR 1 | the rebuilt face thickness is NOT NEMO's: `min_cell_to_vface` ZEROES the northern row, so on `VORTEX_SMT_VEC-zps` the rebuilt `e3v_0` differs from the card's own NEMO-verified bundle on **630 cells, by up to 500.0 m** (the whole `j=62` row is 0.0 where NEMO carries 500.0), and `e3u_0` differs on 5 cells of `i=62` by 3.05e-05 m (periodic WRAP vs the card's COPY).  It also SPLITS provenance: the vorticity path reads the card's correct `e3u_0` and the qco path would read a different one. | **ACCEPTED, decisive.** The repair is withdrawn and HELD.  The correct repair is to READ the card's own `nemo_een_barotropic.e3u_0/.e3v_0` (round 2 proved them exact against `mesh_mask.nc` at 0 ULP), or to attach them as raw operands, not to re-derive them. |
| MAJOR 2 | the "bit-identical on a full-step mesh" argument holds for `e3u_0` and NOT for `e3v_0`, whose northern row comes back exactly 0.0 where NEMO never has 0 (`usrdef_zgr.F90:196` init, `lbc_lnk jpfillcopy :219`, `zgr_zps:1182` zero repair).  The four zero-ULP ladders pass only because that row is the stripped land ring — a card whose north row is NOT land (ORCA2's fold, DINO) would move. | **ACCEPTED.** The claim as written is false and is withdrawn with the repair. |
| MAJOR 3 | "the FIRST statement in the step that reads a U-face thickness" is refuted: `CALL dyn_vor` at `stp2d.f90:142` precedes `CALL wzv` at `:153`, and `vor_een` reads `e3u_3d`/`e3v_3d` at `dynvor.f90:766-767`. | **ACCEPTED and RETRACTED in §2.1.** |
| MAJOR 4 | the flux card's unmoved rows are not the control the receipt claimed, because `CALL wzv` is OUTSIDE the `SELECT CASE`. | **ACCEPTED and RETRACTED in §2.1**, and re-stated as a finding. |
| MAJOR 5 | untested blast radius: `GYRE-zco` pending, DINO absent, and two of the four "inertness" rows cannot move by construction. | **ACCEPTED**, §3. |
| minor 6 | `tools/DOMAINcfg/src/domzgr.F90::zgr_zps` is an OFFLINE TOOL, not compiled into these builds; the statements that actually execute are `tests/VORTEX_SMT_VEC_R8_OMIP_L1_P3/MY_SRC/usrdef_zgr.F90:213,216`.  The rule and the macro resolution (`domzgr_substitute.h90:92-95`, `E3u_0 -> e3u_3d` under `key_vco_1d3d`) are confirmed correct. | **ACCEPTED**: the executing citation is the hook's own lines; `zgr_zps` is where the rule was transcribed FROM, which is a different claim and is now said that way. |
| minor 7 | the rewritten unit test's prose claims it measures a vertical velocity and it does not; two lines are dead.  The test is non-vacuous. | moot — the test is withdrawn with the repair. |
| minor 8 | the oracle repoint checks out; `0eb82c675` is additive and refuses unknown variants. | noted. |

The reviewer also recorded, unprompted, that the step-order walk, the
preregistered S2 falsifier and the decision to HOLD S2 on a 1.3x result
against a 10x bar were done well.  That does not change the verdict.

### 2.2 THE ssh OWNER IS STILL OPEN — round 2's falsifier, answered

Round 2's candidate was "the partial-cell operands of the shared free-surface /
split-explicit barotropic path", because kt=2 `ssh` is the same size under both
momentum programs.  S1 does not move `ssh` at all, so the shared-statement
reading **survives** and S1 is not that statement.

### 2.3 S2 — the depth average of the slow forcing: TRANSCRIBED, MEASURED, HELD

NEMO writes, at `stp2d.f90:177-186` of `VORTEX_SMT_VEC_R8_OMIP_L1_P3`:

```
Ue_rhs(ji,jj) = SUM( e3u_3d(ji,jj,1:jpkm1)*uu(ji,jj,1:jpkm1,Krhs)
                     *umask(ji,jj,1:jpkm1) ) * r1_hu_0(ji,jj)
```

— the REFERENCE face thickness and the STORED reciprocal, with **no ssh
stretching anywhere in the statement**.  legoESM uses the per-level minimum of
the two LIVE (stretched) thicknesses divided by their own column sum.  On a
full-step mesh those weights are the reference weights times one per-face
scalar that cancels; over partial cells the per-level minimum can follow a
different column than the reference minimum does, and it does not cancel.

MEASURED, weights only, no time step — max |w_legoESM − w_NEMO|:

| card | difference | relative |
|---|---|---|
| `VORTEX_SMT_VEC-zps` | 4.629562e-06 | 3.704e-05 |
| `VORTEX_VEC-zco` (flat) | 2.775558e-17 | 2.776e-16 |
| `OVERFLOW-zps` (partial cells, zero initial ssh) | 3.469447e-18 | 8.674e-17 |

Applied on top of S1, it moves the cards like this (kt=2):

| row | flux S1 | flux S1+S2 | vector S1 | vector S1+S2 |
|---|---|---|---|---|
| T | 4.259549e-10 | 4.248031e-10 | 4.539846e-11 | 4.846989e-11 |
| u | 6.308422e-08 | 4.783816e-08 | 6.004125e-08 | 4.638456e-08 |
| v | 4.742908e-08 | 5.078597e-08 | 4.037530e-08 | 4.604105e-08 |
| ssh | 3.726197e-07 | **2.861386e-07** | 3.664757e-07 | **2.861601e-07** |

`ssh` improves by 1.30x on both cards — **short of the preregistered
one-order-of-magnitude falsifier**, and `v` moves AWAY from NEMO on both.  So
S2 is a correct transcription that is NOT the owner, and landing it would move
a certified-shape row the wrong way on no evidence that it is the next
statement in the chain.  **HELD**, as a committed patch
(`manifests/nemo_testcase_l1_vortex_smt_round213_slow_forcing_depth_held.patch`),
with its measurement above.  Round 4 decides it after the barotropic substep
record names the owner.

---

## 3. INERTNESS — MEASURED, NOT ASSUMED

S1 is a change to a SHARED kernel, so inertness is a claim about every card
that reaches it.

| card | rows | rows moved | cellwise residual arrays differing | `first_over_bar` |
|---|---:|---:|---:|---|
| `VORTEX-zco` | 50 | **0** | 0 / 150 | unchanged |
| `VORTEX_VEC-zco` | 50 | **0** | 0 / 150 | unchanged |
| `LOCK_EXCHANGE-zco` | 50 | **0** | 0 / 150 | unchanged |
| `OVERFLOW-zps` (the other partial-cell card) | 50 | **0** | 0 / 150 | unchanged |

**TWO OF THOSE FOUR ROWS ARE WEAK, and the reviewer said so.**
`LOCK_EXCHANGE-zco` and `OVERFLOW-zps` do not carry NEMO's raw mesh operands,
so they cannot reach the changed branch at all; their zeros are a
construction, not a test.  The two VORTEX rows are real (they carry the raw
set and are full-step).  `GYRE-zco` and DINO — the two other raw-mesh cards,
i.e. exactly the ones the branch changes — were NOT measured under the arm,
because the arm was withdrawn before the gates finished.  That gap is one more
reason this statement is HELD.

Because the landed commit changes NO file under `packages/` or `src/`, the
GYRE year and the DINO month gate are inert BY CONSTRUCTION this round; the
GYRE kt=1..10 ladder is run anyway and reported in §3.1.

### 3.1 WHAT THE LANDED COMMIT ITSELF MOVES: NOTHING

With both statements withdrawn, the only behavioural question left is the
oracle repoint.  Both seamount ladders were re-run on the FINAL tree against
the R3 records and reproduce round 2's registry exactly:

| card | kt=2 T / u / v / ssh | AT-BAR | `first_over_bar` |
|---|---|---:|---|
| `VORTEX_SMT-zps` | 4.259549e-10 / 6.308422e-08 / 4.742908e-08 / 3.726197e-07 | 10 / 50 | kt=2 T,u,v,ssh |
| `VORTEX_SMT_VEC-zps` | 1.733725e-09 / 1.748385e-07 / 1.607001e-07 / 3.664757e-07 | 9 / 50 | kt=2 T,u,v,ssh |

Every value is identical to round 2's, to every digit it printed — which is
the repoint's inertness, measured rather than inferred from the byte
comparison.

`GYRE-zco` kt=1..10 ladder on the final tree: see the landing line.

---

## 4. CHOICES MADE THIS ROUND

| choice | ASKED? |
|---|---|
| four new `run.sh` variants into NEW build directories | ASKED — note CC addendum 3 item 1 |
| the R3 records become the admitted ones (`DEFAULT_ORACLE_ROOTS` repointed) | ASKED — same note; and they are bit-identical, so the repoint is inert |
| BOTH candidate statements HELD rather than landed | FORCED by the evidence, not chosen: S1's repair was refuted by the reviewer (§2.4 MAJOR 1/2), S2 missed its own preregistered falsifier (§2.3). No model file changes in this commit. |
| the GYRE year and the DINO month gate not run | FORCED: the landed commit touches no file under `packages/` or `src/`, so both are inert by construction and `land.sh` skips the DINO gate on the same test. Said out loud rather than left implicit. |

**UNASKED list: EMPTY.**

**COMPLIANCE, stated because no gate checks it (RULE 2):** ONE fresh
adversarial review was run (§2.4), not two. Per the lane's standing note the
second reviewer is codex, which is paused on this account; the second opinion
is therefore MISSING, and that is a gap rather than an exemption. It matters
less than usual here only because the one review that did run returned DO NOT
SHIP and the statement is held.

---

## 5. ORCA2 POINTER

ORCA2 is the other partial-cell card on this identity, and **its whole domain
is partial cells**, so S1 is the most ORCA2-relevant statement VORTEX has
produced.  ORCA2's card carries exactly the operand set that triggered the
alias: `nemo_e3t_0` is the real 3-D partial `e3t_0` read from its
`domain_cfg`, and `nemo_hu_0/nemo_hv_0` are the real summed depths, while the
true `e3u_0/e3v_0/e3f_0` are attached only to the EEN barotropic operand
bundle — which the vorticity path reads and the qco path does not.  So before
this round ORCA2's `wzv`/`div_hor` ran with `e3u_0 = e3t_0` on every stepped
face in the global ocean.

**What to measure at ORCA2's rung 0, and where:** re-run the ORCA2 rung-0
kt=1..N trajectory with and without this commit and read the `wzv` / vertical
velocity rows first; the discriminating field is the horizontal divergence's
U-face transport, and the cells to look at are the ones where
`e3u_0 != e3t_0` in ORCA2's own `domain_cfg` (every shelf break and every
ridge).  A move there is the statement landing, not a regression; a move
anywhere the mesh is full-step would be a defect in this fix.

---

## 6. ONE DECISION, ONE LINE (RULE 3)

**S2's default: stays `min_rule_live` (today's live min-rule weights) or moves
to `nemo_literal` (NEMO's reference `e3u_0 * r1_hu_0`)?**  My pick: **stay for
now**, because the measurement above shows it is not the kt=2 owner and it
moves one row away from NEMO; revisit in round 4 with the per-substep record.
The patch is committed and the measurement is in §2.3, so nothing has to be
re-derived to take the other branch.

**S1 is NOT a decision, it is unfinished work:** the defect is real and
measured, the repair tried here was wrong, and the correct repair is named in
§2.4 (read the card's own NEMO-verified `e3u_0`/`e3v_0` instead of
re-deriving them). It is the first item of round 4, and until it lands every
raw-mesh card — GYRE, DINO and ORCA2 included — still runs `e3u_0 = e3t_0`
in the qco path.

---

## 7. ROUND 4 — PREREGISTERED PREDICTIONS (the 100-day comparison)

Round 4 is decision 88's deliverable (4): both SMT cards against NEMO's own
shipped 100-day run (`nn_itend = 3000` steps of `rn_Dt = 2880` s, daily
restarts), scored with the GYRE year scorer's definitions, exactly as round 210
did for the flat cards.  Written now, before it runs:

* R4-P1. The ten-step floor does NOT survive: both cards' wet 3-D T rms grows
  from `~1e-10` K at kt=10 to `>1e-4` K by day 100, because the kt=2 residual
  is a `ssh`/transport error that the vortex's own advection amplifies.
* R4-P2. The two cards' error CURVES differ in shape: the vector card starts
  lower (its flat twin is at the bar) and the flux card starts at its own
  inherited `1.2e-08` floor, so the flux curve is above the vector curve for
  the first days and the two converge once the seamount-driven error
  dominates — the crossing day is the number to report.
* R4-P3. The error is NOT uniform: its maximum sits over the seamount and
  along the vortex's path west of it, not where the vortex is at day 100.
* R4-P4. NEMO's own day-100 states (round 1, admitted) are unchanged, so the
  comparison needs no new NEMO run.
* FALSIFIER for R4-P1: a flat (sub-1e-9) T-rms curve over 100 days would mean
  the kt=2 residual is a bounded bookkeeping offset rather than a growing
  trajectory error, and the whole SMT walk would be reclassified as cosmetic.

---

## 8. EVIDENCE

All under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/round3/`:
`predictions.md`, `acquire.log`, `record_identity.txt`, `repeat_vec/`,
`ladder_fix1_*.json`, `ladder_fix12_*.json`, `inert/`, and the four admitted
record directories `VORTEX_SMT_R3_*`.
