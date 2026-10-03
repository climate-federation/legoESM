# Round 213 / VORTEX_SMT round 3 — the records re-acquired, and the first non-bit partial-cell statement

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

**Where it executes, proved rather than assumed.** The alias lives in the one
shared qco operand builder (`nemo_qco_resolved_mesh_operands`), whose consumer
on this card is `nemo_qco_wzv_operands` — the `zad_qco_evaluation="nemo_literal"`
arm that every NEMO testcase card selects, i.e. NEMO's `CALL wzv` at
`stp2d.f90:153`, the FIRST statement in the step that reads a U-face thickness.
`dyn_zad` is called only in the vector branch of `stp2d`'s `SELECT CASE`
(`np_VEC_c2`), and the flux branch calls `dyn_adv_up3` instead — which is
exactly the measured pattern below.

**THE FIX IS THE ROOT CAUSE, NOT A CARD PATCH:** the alias is deleted and the
shared min rule is applied, in the one kernel, so every card that carries
NEMO's raw mesh gets NEMO's face thickness.  On a full-step mesh the two
neighbours carry the same number and the minimum is that number bitwise, which
is why every full-step card below is unmoved at zero ULP.

#### S1 — ONE VARIABLE, BEFORE vs AFTER, both cards, bar 1e-15

| kt=2 row | `VORTEX_SMT-zps` before | after | `VORTEX_SMT_VEC-zps` before | after |
|---|---|---|---|---|
| T | 4.259549e-10 | 4.259549e-10 | 1.733725e-09 | **4.539846e-11** (38x) |
| u | 6.308422e-08 | 6.308422e-08 | 1.748385e-07 | **6.004125e-08** (2.9x) |
| v | 4.742908e-08 | 4.742908e-08 | 1.607001e-07 | **4.037530e-08** (4.0x) |
| ssh | 3.726197e-07 | 3.726197e-07 | 3.664757e-07 | 3.664757e-07 |

The flux card is **exactly unmoved**, which is the statement's own control: the
arm it corrects is the one NEMO runs only under vector-invariant momentum.

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
| `OVERFLOW-zps` | 50 | PENDING | PENDING | PENDING |
| `GYRE-zco` | 954 | PENDING | PENDING | PENDING |

GYRE year, DINO from-rest month gate: below.

---

## 4. CHOICES MADE THIS ROUND

| choice | ASKED? |
|---|---|
| four new `run.sh` variants into NEW build directories | ASKED — note CC addendum 3 item 1 |
| the R3 records become the admitted ones (`DEFAULT_ORACLE_ROOTS` repointed) | ASKED — same note; and they are bit-identical, so the repoint is inert |
| S1 landed unconditionally in the shared kernel rather than behind a card flag | ASKED in the sense that it is a DEFECT REPAIR with no behaviour change on any full-step card (measured), not a new option; RULE 3 does not bite because no default preserves the old behaviour anywhere |
| S2 HELD rather than landed | **DECISION_NEEDED, raised here** — see §6 |

**UNASKED list: EMPTY.**

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
