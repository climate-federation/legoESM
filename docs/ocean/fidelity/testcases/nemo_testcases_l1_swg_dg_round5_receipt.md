# SWG double-gyre round 5 receipt

Date: 2026-10-12

Status: **STOPPED FOR DECISION — THE DYNAMIC RECORDS ARE HONEST, BUT THE
FORCING ZERO BAR NEEDS AN EXPLICIT RULING**.  Localisation finds no record
discontinuity.  The stage and month nonzero residuals arise inside the
certified CPU production calculation at last-bit scale, while the only
numerically equal IEEE mismatch is NEMO's signed-zero meridional wind stress.
The exact IEEE bar has not been changed.  No physics, configuration, or
numerical method was changed, and no NEMO build, NEMO run, or GPU run was
launched in this round.

## Provenance and unchanged controls

The incoming tip was `6030fa3f1`.  The implementation commits are
`09671b614c0d68808c57553caf8fa5eba43756da` and `eb6bf946e`.  The Round-5
preregistration was committed before measurement.  All new comparisons use
the already admitted Round-1 stage and month records.

The no-argument shipped card remains the default.  Its digest remains
`f9d46bfd08f23702`; its certified given-entry 338/338, chained 338/338,
independent 50/50, carried 29/29, and month 690/690 rows are unchanged.  The
parameterised card still obtains `nn_SWG`, `rn_dx`, `rn_domsiz`, `rn_tau`,
`rn_modified_grav`, `rn_Uv`, `rn_Lv`, `rn_rfr`, `rn_beta`, `rn_f0`, and
`rn_Dt` from the resolved namelist and rejects unsupported values.

## Stage-debt localisation

The canonical report is `round5/debt_localization.json`.  The committed
summary log prints the decisive counts.

1. A production step from rest reproduces every dynamic field in the admitted
   kt-2 stage-1 entry exactly: zero unequal IEEE words across all nine fields
   and all regions.  This proves that the entry record has not corrupted the
   first production state.
2. Replaying kt-2 stage 1 from those admitted operands produces the original
   debt.  SSH has 1,278 unequal wet wall-adjacent cells, with maximum absolute
   residual `3.3881317890172014e-21` m.  The replay also has last-bit residuals
   in V and the four thickness-ratio fields; the maximum is
   `6.617444900424222e-24` m/s in V and `1.3234889800848443e-23` in `r3v`.
   `r3f` includes 1,076 dry/storage and 1,072 wall-adjacent unequal words.
3. A second production step differs from the admitted kt-3 stage-1 entry in
   45,973 dynamic words.  SSH has 189 unequal words with maximum absolute
   residual `1.3552527156068805e-20` m; V and V-before each have 22,421 unequal
   words with maximum `2.6800651846718097e-22` m/s.

The operands and transition record are therefore honest.  The nonzero debt is
a CPU/backend arithmetic boundary inside the production stage, not a
record/representation fault.  This does not contradict the operator's
independent native-GPU end-to-end result: NEMO and legoESM instantaneous
restarts were bit-identical at years 1 and 2, over 1,261,440 steps.  Those
operator comparisons remain supporting, non-certified evidence; the CPU
ladder remains the formal gate.

## Signed-zero representation debt and requested ruling

At 6.25 km, `vtau` has 644 unequal IEEE words but zero numerical residual:
638 are wet interior, two wall-adjacent, and four dry/storage.  At 25 km the
same effect accounts for 164 unequal words.  NEMO forms `ztauv` as `-0.0` when
`rn_theta=0` and then multiplies it by the latitude cosine, so its zero sign
changes with the cosine.  The card stores numerically identical `+0.0`.

The Round-4 exact IEEE bar is retained in every report.  The requested ruling
is whether **forcing-only** comparisons should treat `+0.0` and `-0.0` as
equal while every prognostic/state comparison remains IEEE-word exact.  The
recommendation is **yes**: the sign is created solely by multiplying an exact
zero by the steady cosine and is dynamically inert.  This is a bar decision,
not an implementation permission; no gate was relaxed in this round.

## Thirty-day localisation

The canonical `round5/month.json` contains all 690 rows after a fresh 12,960-
step certified-CPU integration from rest: 249 AT-BAR and 441 DEBT, exactly the
Round-4 classification.  Day 1 is the first signed-zero forcing debt.  The
first nonzero difference is the day-1 SSH restart: 22,487 unequal wet cells,
maximum `9.367506770274758e-17` m.  At day 30 the peak is
`1.6431300764452317e-14` m/s in U, normalized
`9.20175294902847e-15`.

Crucially, NEMO's day-1 instantaneous restart at kt 432 is bit-identical to
its next recorded stage entry at kt 433 in SSH, U, V, `utau`, and `vtau`.
There is no record or sampling discontinuity.  The month debt is accumulated
certified-CPU last-bit arithmetic, not daily-mean contamination or a damaged
record.

## Matched daily-mean output

The non-certified GPU driver now has `--daily-means`.  One compiled-day scan
advances the unchanged production three-stage program and accumulates SSH, U,
and V in the same left-associated online order as IOIPSL.  Each v2 NPZ retains
the end-of-day snapshot and adds explicitly named daily-mean arrays, sampling
labels, and the 1,728-sample count.  Certified CPU mode refuses this option so
the formal per-step ladder path remains unchanged.

The direct production control uses a non-steady tiny grid.  Its compiled mean
equals a separately associated mean of every post-step state and differs from
the final snapshot.  Dropping the sampling metadata or using daily snapshots
causes the climate reader to refuse.  The real NEMO year-1 archive also passes
the committed daily-mean admission: 32 files per field, fp32 storage,
`online_operation=ave(x)`, 50-second sampling, and 86,400-second writes.

The climate result is intentionally unmeasured.  The existing legoESM years
contain end-of-day snapshots and must not be relabelled.  The operator must
run the committed GPU script with `--daily-means`, after which the committed
climate instrument can compare NEMO and legoESM days 1-365 as **year 1,
spin-up**.  Years 2-3 remain the registered climate window.

## Tests, preflights, and evidence

The final numerical implementation battery is archived in
`round5/direct_tests.log`: **191/191 passed** on CPU with x64 enabled.  It
includes the parameterised card, production ladder, daily-mean driver, and
climate reader tests.  Focused Ruff, bytecode compilation, and
`git diff --check` pass.  The citation-gate mapping extension is checked again
by the explicit receipt gate and its planted failure.

All four no-run acquisition preflights pass in
`round5/acquisition_preflights.log`: shipped stage, shipped month, Maddison
stage, and Maddison month.  Every target is still created from shipped `SWG`
before the selected deck overlay; the shipped and Maddison month cadences are
24 and 432 steps/day.  No acquisition is requested for the already admitted
ladders.

The evidence root is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/round5/`.
`debt_localization.log`, `month.log`, `nemo_daily_mean_preflight.log`, and the
structured JSON reports are the canonical measurement records.

Two attempts to obtain fresh independent review were blocked by the execution
environment: Claude did not pass its workspace-trust prompt in noninteractive
mode, and Codex could not initialize its app-server state in a read-only
sandbox.  Their failure logs are archived as `claude_review.log` and
`codex_review.log`; neither is represented as a scientific review.  The
substantive diff therefore remains unreviewed and must not be shipped without
the campaign's required two independent reviews.

## Open and inherited items

- The forcing signed-zero bar needs the explicit ruling above.
- The daily-mean GPU year and year-1 spin-up climate comparison need operator
  acquisition.  The finer-grid years 2-3 comparison remains open.
- The unrelated inherited GYRE/LOCK_EXCHANGE/OVERFLOW digit pins remain stale.
  The pinned values are `5850bbcfcd351c29` / `4ec82201f0fae7bf` /
  `7f1ec89b3e6578f5`; current values are `f02d2f42a570f413` /
  `d5dfdf83b3a1e0ab` / `83de0010e1170032`.  This lane does not re-pin them.

## Oracle citations

The executed SWG_DG step swaps the just-computed level into `Nbb`, then sends
that level to both diagnostics and restart output at
`SWG_DG/BLD/ppsrc/nemo/stprk3.f90:308-321`.  Its daily writer takes SSH, U,
and V from that `Kmm` level at
`SWG_DG/BLD/ppsrc/nemo/diawri.f90:457,536-538`.

NEMO constructs the meridional stress amplitude as the negative sine and then
multiplies it by the fixed 2,000 km cosine at
`SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_sbc.f90:99-106`.  For `rn_theta=0`, that
sequence explains the dynamically inert signed zeros without changing the
wind's numerical value.

IOIPSL's accumulated-mean routine and its left-associated update live at
`ext/IOIPSL/src/mathelp.f90:3086-3119`; that executed association is what the
new legoESM daily-mean accumulator reproduces.
