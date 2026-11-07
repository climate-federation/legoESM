# Round 52 preregistration — GYRE WS-RK3 live stage operands

Date: 2026-09-11.  Frozen code base:
`471bf6b399de116997631a2697911b7e87c56b09`.  Oracle:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round46/oracle_kt2_stage`.
All new results are CPU, fp64, production-JIT measurements.  The committed
NEMO executable and records are read-only; no NEMO build or integration is
authorized.  The score is the ordinary testcase bar
`max(abs(candidate-reference))/max(max(abs(reference)),1) <= 1e-15`.

## Source-locked program

The compiled GYRE branch first calls `stp_2D(kstp,Nbb,Nbb,Naa,Nrhs)` and
then invokes stages 1, 2 and 3 with fixed `Kbb=Nbb` and successively swapped
`Kmm`/`Kaa` (`stprk3.f90:190,200-215`).  The pre-stage program calls HPG,
LDF, VOR, WZV and ZAD on `Kbb`, in that order (`stp2d.f90:141-175`); its
initial `rDt=rn_Dt` is set at `domain.f90:308-310`.  In the shared stage
routine, stage 1 sets `rn_Dt/3`, stage 2 sets `rn_Dt/2`, and stage 3 sets
`rn_Dt` (`stprk3_stg.f90:138-146,195-200,239-244`) before the vector arm
calls WZV with `(Kbb,Kmm,Kaa)` at `:329-335`.  WZV uses the then-live
`r1_Dt` in the written stretching recurrence (`sshwzv.f90:277-298`).

At stage 3, the caller hands `dyn_ldf` `(Kbb,Kmm)`
(`stprk3_stg.f90:690-712`).  The compiled level operator reads Kbb velocity
and Kbb `e3t/e3u/e3v` in its curl/divergence input (`dynldf_lev.f90:121-130`),
then divides the output curl by Kmm `e3u/e3v` (`:132-140`).  Thus the six
thickness operands are `(e3t_Kbb,e3u_Kbb,e3v_Kbb,e3f_live,e3u_Kmm,e3v_Kmm)`.

This is new evidence against Round 21's isolated clock discriminator, which
paired legoESM's full-step SSH endpoints with only a changed denominator.
Round 52 scores the clock at the complete live stage-2 accumulator and
trajectory boundaries; if those registered rows do not improve, the new
causal claim is REFUTED and the contradiction remains open rather than being
papered over.

## Card A — stage-2 WZV/ZAD clock

Change only the shared WS-RK3 stage program: stage 2 passes `dt/2` both to
the WZV transport construction and to the literal ZAD continuity recurrence.
The pre-stage/stage-1 RHS retains the full `stp_2D` clock; stage 3 retains the
full clock.  There is no public or private selection knob.

Prediction: against the kt=1 stage record, stage-2 post-ZAD/post-ADV U and V
accumulators strictly decrease in max absolute error, and neither stage-3
post-ADV accumulator worsens.  The independent kt=1 exit, observed as kt=2
stage-1 `u_Kmm/v_Kmm`, strictly improves from
`2.7478404751243857e-12 / 3.305560306813421e-12`; AT-BAR means normalized
error `<=1e-15`.  The card is REFUTED if either exit field fails to decrease,
any registered accumulator worsens, or any GYRE/LOCK_EXCHANGE/OVERFLOW
kt=1..10 first-over-bar row moves earlier.  A one-cell comparison plant and a
commit-stamp plant must exit nonzero.

## Card B — stage-3 LDF Kbb/Kmm thickness family

After Card A is measured and retained, pass the source-locked six-tuple above
only to the existing literal LDF kernel in stage 3.  Velocity remains Kbb;
the stage's HPG/VOR/ADV operands remain Kmm.  No selector or fallback changes.

Prediction: the live stage-3 LDF handed-operand `e3t_Kbb` error
`1.1029155336927943e-4` becomes bit-exact, and the stage-3 post-LDF U/V
accumulator max errors strictly decrease without worsening any earlier
accumulator.  The kt=1-exit U/V errors strictly improve relative to Card A;
AT-BAR is `<=1e-15`.  REFUTED if either required operand is non-bit, either
post-LDF field fails to improve, or any Rule-12 first-over-bar moves earlier.
The same plants must exit nonzero.

## Rule-12 table contract

For each card, run the trajectory gate for GYRE-zco, LOCK_EXCHANGE and
OVERFLOW through kt=10 before and after, with `--continue-after-first` and an
ULP residual sidecar comparison.  Every moved `(kt,field)` row is reported;
the first-over-bar may stay or move later, never earlier.  LOCK_EXCHANGE and
OVERFLOW execute the same `rk3_ws` program; the result records their resolved
literal/generic WZV and LDF-on/off arms rather than assuming transfer.
ORCA2 is `UNMEASURED_WITH_SPEC`: run the same kt=1..10 gate once a pinned
independent ORCA2 record with the identical schema exists.  DINO is a
separate leap-frog (`nemo_mlf`) program, so neither code path executes there;
the shared PE literal WZV/LDF statement remains the only regression risk and
is covered by focused tests, not claimed dynamically.

## Conditional ZAD card

Only after Cards A and B, apply the already committed
`manifests/nemo_testcase_l2_gyre_round47_rejected_stage_zad.patch` without
editing its physics.  CONFIRM and retain only if all 57 formerly worsened
Rule-12 rows vanish (no positive ULP delta), no first-over-bar moves earlier,
and the kt=1 stage ZAD/ADV accumulators improve.  Otherwise mark REFUTED and
revert that candidate only.  If kt=2 becomes AT-BAR, inspect the committed
kt=3 stage inputs and name its first unequal operand plus a one-variable
discriminating measurement; otherwise that conditional owner walk is not
entered.

## Self-review / ASKED boundary

Pre-code review found one implementation site, existing literal WZV and LDF
operand seams, and no need for a new configuration choice.  Forbidden and
unentered: NEMO edits/runs, TKE, forcing modules, year harness, tolerances,
carried-state redesign, and any GYRE-only branch.  The user-requested
independent Claude review follows this work; this round makes no claim of
having supplied that external review.

## Card A2 addendum — paired stage-2 Kaa/clock handoff

Card A's isolated denominator change is REFUTED: kt=1 exit U/V became
`2.019902651313472e-6 / 3.9550723470321045e-6`, and the Rule-12 comparison
failed. This is the Round-21 half-state: legoESM still handed WZV the full-step
Kaa SSH while changing only `r1_Dt`. NEMO stage 2 constructs Kaa=N+1/2 at
`stprk3_stg.f90:215-235` and consumes that Kaa with `rn_Dt/2` at `:199-200,
333`; stage 3 restores Kaa=N+1 and full dt at `:240-248`.

Before measuring the replacement, freeze the paired candidate: stage 2 alone
hands WZV both `_eta_live_one_half` and `dt/2`; literal ZAD retains `dt/2`.
CONFIRM only if stage-2 post-ADV remains AT-BAR, stage-3 accumulators and kt=1
exit U/V strictly improve over the original base (`2.7478404751243857e-12 /
3.305560306813421e-12`), and all three Rule-12 cards have no worsened row or
earlier first-over-bar. Otherwise Card A remains REFUTED and is reverted.

## Card B rebase after Card A refutation

Both A arms met their falsifiers, so Card B is measured on the restored
`471bf6b399de` physics baseline. Its source tuple and thresholds above are
unchanged. The additional fail condition is any positive Rule-12 worsening
relative to the already captured baseline artifacts; Card A artifacts are not
used as B's reference.
