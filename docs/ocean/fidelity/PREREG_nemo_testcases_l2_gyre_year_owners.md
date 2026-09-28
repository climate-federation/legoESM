# PREREG — WHO OWNS THE GYRE DAY-30 GAP, and WHAT FLIPS BETWEEN DAY 180 AND DAY 210

Written BEFORE any number in this round was measured.  It is the follow-up to
`PREREG_nemo_testcases_l2_gyre_year_fromrest.md`, whose round-2 receipt
(`testcases/nemo_testcases_l2_gyre_year_fromrest_receipt.md`) scored the year
and left two ranked open items; this document turns items 1 and 3 of that
receipt's section 12 into falsifiable measurements.

Harness: `scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_year_owners.py`
(new, committed; a throwaway probe's number is unmeasured).
NEMO acquisition: `scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_earlydays/run.sh`
(written by the agent, run by the operator).
Artifacts: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_owners/`.
CPU, fp64, `PrecisionPolicy.fp64(transcendentals="libm")`, `JAX_ENABLE_X64=1` —
the card forbids any other platform.

## 0.  The entering condition, quoted not re-derived

| what | value | source |
|---|---:|---|
| day-30 3-D wet-cell `T3D` gap, control vs control | `1.4241e-02` K | year receipt §3 |
| the floor it is judged against at day 30 | `6.8065e-10` K | year receipt §3 |
| the same gap after TEN steps (40 h) | `2.7682e-03` K | year preregistration §0 |
| the kt=2 temperature row | `5.786e-16` normalized = `1.360e-14` K | round-8 receipt |
| the kt=1 surface forcing, all five fields, against NEMO's own dump | **BIT-EXACT**, 0 of 600 wet cells | round-8 receipt §"post-fix eligibility" |
| day-360 gap / day-360 floor | `4.0714e-01` K / `2.4943e-02` K | year receipt §3 |

**The arithmetic that sets this round's target.**  `2.768e-03` K at kt=10 is
already `19 %` of the day-30 gap, and the remaining 170 steps multiply it by
only `5.1`.  So the day-30 gap is not made at day 30: **most of it is made in
the first ten steps, in a window the ten-step ladder already instruments
per step**, and the rest is a slow amplification.  A day-by-day table is the
measurement that says which of those two it is, and no such table exists on
GYRE.

## 1.  What is acquired, and why the acquisition is not a rebuild

NEMO's GYRE record has restarts at `kt = 10` and then nothing until `kt = 180`
(day 30).  The acquisition fills days 1..30 at DAY resolution, and gets days
0..10 at STEP resolution for free, because the certified card's own MY_SRC
writer dumps the step-entry state for `kstp = nit000.. nit000+59`
(`cfgs/GYRE_OMIP_L2_P3_SM_YRPERT/MY_SRC/stprk3.F90:90-92`) and the existing
10-step run simply stopped before most of them were written.

`run.sh` stages a NEW RUN DIRECTORY against the ALREADY-CERTIFIED BINARY and
changes three namelist rows (`nn_itend 10 -> 180`, `nn_stock 10 -> 6`,
`nn_write 10 -> 180`).  It does not build.  Rationale, and it is a choice on
the ASKED/UNASKED table: a `makenemo` copy would put a SECOND BINARY inside a
comparison whose whole purpose is that only the time axis varies, and the
run's own instrument check would then be unable to separate a build difference
from a physics one.  The check that makes this safe is preregistered here:
**the new run's `kt = 180` restart must be BYTE-IDENTICAL to the year
control's `kt = 180` restart**, which proves the restart cadence changed no
arithmetic and that this record and the scored year are one trajectory.

**NEMO's trend diagnostics are NOT acquired, and that is a decision.**
`&namtrd`'s 3-D outputs go through `iom_put`, and this card compiles without
`key_xios` (`cpp_GYRE_OMIP_L2_P3_SM_YRPERT.fcm` = `key_qco key_vco_1d3d
key_RK3`), so `ln_tra_trd`/`ln_dyn_trd` would write nothing.  `ln_glo_trd`
does write without XIOS, but NEMO 5 itself prints
`'The trends diagnostics are a work in progress: they are not yet fully tested
or functional'` for every trend flag under RK3 (`src/OCE/TRD/trdini.F90`,
`trd_init`), and Rule 5 forbids quoting an unclosed trend bucket at all.  The
per-operator record this campaign actually uses is the card's own MY_SRC stage
writer, which is exact and closes by construction; it fires at `kstp == nit000`
only, so a per-operator record at a DAY boundary is acquired by restarting
from that day's restart and running ONE instrumented step.  That is a separate
run and a separate ASKED choice; it is not taken this round.

## 2.  The three candidates, RANKED IN ADVANCE, each with its falsifier

**C1 — THE SURFACE FORCING'S SEASONAL TIME FUNCTIONS.**  The kt=1 dump
certifies `qsr`, `qns`, `emp`, `utau`, `vtau` BIT-EXACT — at ONE value of the
clock, `ztime = 4` hours of a 8640-hour cycle.  Every seasonal statement in
`usrdef_sbc` (`BLD/ppsrc/nemo/usrdef_sbc.f90:107-124,178-188`) is a function of
that scalar, so a wrong phase, a wrong denominator, a wrong calendar operand or
a missing `nyear` term can be invisible at kt=1 and finite on day 30.

*The measurement, and it needs NO new NEMO run.*  The forcing is analytic in
`gphit` and `ztime` except for `qns`, which reads `ts(:,:,1,jp_tem,Kbb)` — the
state ENTERING the step.  NEMO's restart at `kt = n` IS that entering state for
step `n+1`.  So score, on **NEMO'S OWN STATE**, at every day boundary the year
record holds:

* the LITERAL `usrdef_sbc` transcription (the one the kt=1 dump certifies), and
* legoESM's CURRENT forcing path,

field by field, BIT-EXACT bar.  **REFUTED if every field is bit-exact at every
one of the twelve day boundaries** — C1 is then exonerated as a statement, and
the realized forcing difference between the models is a CONSEQUENCE of the
state gap, not a cause.  **CONFIRMED if any field differs on any day at a
magnitude that can move `1.4e-02` K in 180 steps.**

*What this cannot see, stated in advance:* a statement that is wrong in BOTH
the literal transcription and legoESM, i.e. a misreading of the Fortran shared
by both.  The mitigation is Rule 0 — the compiled `usrdef_sbc.f90` is re-read
statement by statement in the receipt, with line citations, and every operand
whose value cannot be read off the source (`nyear`, `nyear_len(1)`, `ndate0`)
is read from the run's OWN record (`ndastp` in each restart, `ocean.output`).

**C2 — THE SOLAR PENETRATION AND THE RESTORING TIME LEVEL** (DINO's day-30
owners).  `traqsr` two-band and the level at which the restoring reads the
tracer.  Entering evidence AGAINST C2 being a live defect on GYRE: the kt=1
`qsr_2BD` accumulated increment is BIT-EXACT (round-8), and `sbc` is called
`CALL sbc( kstp, Nbb, Nbb )` (`MY_SRC/stprk3.F90:155`), so NEMO's `sst_m`
(built from `ts(:,:,1,:,Kmm)`, `sbcssm.f90:84-93`) and the Haney term's
`ts(...,Kbb)` are THE SAME LEVEL on this card — which is what legoESM does.
**REFUTED if that call site reads `Nbb, Nbb` and the kt=1 QSR row is
bit-exact** (both are re-read and re-quoted, not assumed).  Kept on the list
because DINO's owner was here and because "exonerated at kt=1" is not
"exonerated at day 30" — the depth-band decomposition below is what would
raise it again: C2 predicts the gap is concentrated in the TOP CELL and the
two-band e-folding depths (0.35 m, 23 m), i.e. almost entirely in `T3D_0_100`
with `SST` leading.

**C3 — THE MIXED-LAYER / CONVECTION SWITCH.**  `ln_zdfevd = .true.`,
`rn_evd = 100.`, `nn_evdm = 1` (certified `namelist_cfg &namzdf`).  This is a
threshold, not a smooth term, so it cannot make a gap grow smoothly — it makes
a gap JUMP.  **It is therefore predicted NOT to own the day-30 gap** (which
the year receipt shows growing as a power law from day 30 to day 180) **and
predicted to own the day-180 -> day-210 crossing**, which jumped five orders in
one 30-day block on ONE member.  Falsifier for the second half: if the first
step at which the two members' EVD trigger masks differ is NOT inside days
180-210, or if the trigger masks never differ while the gap jumps, the EVD
identification is REFUTED and the jump has another owner.

## 3.  The day-30 decomposition — what is computed, and what each cut can and cannot see

On NEMO's own `tmask`, fp64, index-by-index, after the year harness's own
alignment gate has passed (it is the gate that makes an index-by-index
difference meaningful at all).

| cut | rows | blind to |
|---|---|---|
| FIELD | `T`, `S`, `u`, `v`, `ssh` — each as a wet rms AND as a fraction of `rms(NEMO(day) - NEMO(rest))` | which cells; a field whose own from-rest signal is ~0 has an undefined fraction and is reported as such |
| DEPTH | the three preregistered bands `0-100`, `100-1000`, `1000+`, plus the TOP CELL alone | horizontal structure |
| REGION | western third / interior / eastern third of the wet rectangle, and the forcing bands (the `emp` split at `37.2 N`, the wind's `15-29 N` half period), each as a share of `sum dT^2` | it is a share, so a region that is large is favoured; the cell COUNT and the per-cell mean are reported beside it |

The FRACTION-OF-NEMO'S-OWN-SIGNAL table is the only one of the three that can
name a LEADING FIELD, because the five fields are in five different units.  It
is DINO's statistic (`dino_1226/day_gap_table.py:186-200`), reused rather than
re-derived so the two cases remain comparable.

**A share is not an attribution (Rule 4).**  No ablation is run on either model
this round, so every statement produced by section 3 is a DIRECTION.  It is
labelled PLAUSIBLE wherever it appears.

## 4.  Expectations — falsifiable, committed before any number was read

* **E1 — the day-by-day gap is already large on DAY 1.**  `T3D` at day 1
  (`kt = 6`) is **above `1e-4` K**.  *Reason:* it is `2.768e-03` K at kt=10 and
  the ladder's rows grow monotonically over kt=1..10.
  **REFUTED if day 1 is below `1e-4` K.**
* **E2 — the day-by-day gap has NO knee.**  Between day 1 and day 30 the `T3D`
  gap is monotone and its day-to-day ratio never exceeds `3`.  *Reason:* a
  threshold process would show a jump; a smooth operator mismatch would not.
  **REFUTED if any day-to-day ratio is `>= 3`, which would name a DAY and turn
  this round into a step hunt inside it.**
* **E3 — the forcing statement is exonerated.**  Every one of `qsr`, `qns`,
  `emp`, `utau`, `vtau` is BIT-EXACT between legoESM's current path and the
  literal transcription, on NEMO's own state, at all twelve day boundaries.
  **REFUTED if any field differs anywhere.**  This is a CALIBRATED expectation,
  not a blind one: the kt=1 row is already known bit-exact, and this predicts
  that the time dependence carries it.
* **E4 — the day-30 gap leads in the UPPER OCEAN and in TEMPERATURE.**  `T3D`'s
  `0-100` m band exceeds its `1000+` m band by more than `10x` in K, and the
  fraction-of-signal table names `T` or `S` rather than `u`, `v` or `ssh`.
  *Reason:* the year receipt's day-360 table has `T3D_0_100` at `17.6 %` of
  NEMO's own scale and `PSI` at `0.12 %`.  **REFUTED otherwise.**
* **E5 — the day-180 -> day-210 jump has an EVD trigger-mask crossing inside
  it.**  Stepping two members of the TIP model side by side from rest, the
  first step at which their enhanced-vertical-diffusion trigger masks differ
  lies in `[1080, 1260]`.  **REFUTED if it lies outside, or if no step in the
  year has a differing mask.**  If REFUTED and the masks differ EARLIER while
  the gap stays at `1e-8` K, that is also informative and is reported as such:
  it would mean the switch flips often and harmlessly, and the day-210 jump is
  a different object.

## 5.  Gates, and the plant that proves each can fail

Every gate in the new harness exits NON-ZERO on its plant, and each plant is
exercised by the committed unit test.

| gate | what it binds | plant |
|---|---|---|
| forcing statement gate | legoESM's forcing == the literal `usrdef_sbc`, bit-exact, on NEMO's own state | `--plant forcing-phase` shifts `ztime` by one step; `--plant forcing-qsr-pi` swaps the source's literal `3.1415` for `rpi`; `--plant forcing-nyear` restores the `(nyear-1)` term as if the run were in year 2 |
| day-by-day table | the two sides are the same day, the same mask, the same metric | `--plant day-offset` reads NEMO one day late |
| alignment | reuses the year harness's own `--alignment-gate` rather than a second copy | that gate's own four plants |
| switch trace | the trigger mask is the one the model uses, not a re-derivation | `--plant switch-blind` freezes the mask so no crossing can ever be found, and the trace must then REFUSE rather than report "no crossing" |

## 6.  ASKED / UNASKED

| # | choice | value | ASKED? |
|---|---|---|---|
| 1 | the early-days NEMO record reuses the CERTIFIED BINARY instead of a `makenemo` config copy | reuse | **UNASKED.**  Offered for revert.  Taken because a second binary inside a one-variable comparison cannot be separated from a physics difference, and because the byte-identity check at `kt = 180` against the scored year control is a stronger instrument than a build-provenance hash.  The operator's own instruction named a config copy; this is the one place this round departs from it, and it is named here rather than buried |
| 2 | NEMO trend diagnostics not enabled | not enabled | **UNASKED.**  Offered for revert.  Reason in section 1: they write nothing without XIOS on this card, NEMO declares them untested under RK3, and Rule 5 forbids quoting them unclosed |
| 3 | the early-days record's cadence | `nn_stock = 6` (one day), `nn_itend = 180` (30 days) | **ASKED**, pick: one day, so the record's days are the day-by-day table's rows |
| 4 | the legoESM daily member's cadence | `--snap-steps 6`, 30 days, seed 0 | **ASKED**, pick: the same days as the NEMO record; seed 0 is the control by the year preregistration's definition |
| 5 | the region cut's boundaries | the wet rectangle's thirds, and the forcing's OWN latitudes (`37.2 N` for `emp`, `15-29 N` for the wind) | **ASKED**, pick: these.  The thirds match the year receipt's existing "western third" statistic so the two are comparable; the forcing latitudes are read off `usrdef_sbc`, not chosen |
| 6 | the switch trace's two members | tip seed 0 and tip seed 1 | **ASKED**, pick: these, because the year receipt measures the jump as seed 0 moving away from seeds 1/2/3 |
| 7 | `_literal_sbc` gains a `kt` argument (default 1) in the round-16 discriminator | additive, default unchanged | **UNASKED.**  Offered for revert.  The alternative is a SECOND copy of the literal transcription, which the repository forbids outright; the default keeps round 16's own number byte-unchanged |

No expectation, threshold, metric or scored row in this document was written
after a number was read.  Any that later must change is recorded as a
RETRACTION in the receipt, never edited here.
