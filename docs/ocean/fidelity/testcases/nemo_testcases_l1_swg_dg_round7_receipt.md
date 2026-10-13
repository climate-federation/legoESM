# NEMO testcase L1 SWG_DG round 7 receipt

**Date:** 2026-10-12  
**Status:** STOPPED_FOR_DECISION — the additive record writer and the ladder's
chained composition are exonerated.  The certified per-step program has a
last-bit prognostic debt from step 2; no physics, configuration, production
arithmetic, acquisition artifact, or fidelity bar changed.

## Frozen instrument and evidence

The preregistered discriminator is
`nemo_testcases_l1_swg_dg_round7.py`; its five direct tests cover restart
assembly, exact/signed-zero classification, missing-restart refusal,
plant binding, and the end-to-end synthetic path.  It replays the admitted
`kt=1..10` entries through the production three-stage program under certified
`PrecisionPolicy.fp64(transcendentals="libm")` on CPU.  Every floating-point
comparison reports IEEE-word and numerical equality separately on the
field-specific wet mask.  A one-ULP change to a nonzero wet U input changes
two stage-1 SSH words, so the control is binding.

The canonical result is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/round7/per_step_restart_discriminator.json`,
SHA-256
`b4d68b93a31e83e15c45837c767514b61b2414e81cd8de6f7ae034ae8de55ce6`.
It stores every compared row, admitted-input hashes, resolved namelist, masks,
policy, backend, JAX versions, and worktree stamp.  The successful run log has
SHA-256
`548bc789bf80282a9ed95a44c53c1b7fcf3b8256b1af88a9b310ea6d1ae59782`.
An earlier execution refused before producing a JSON artifact because a NumPy
integer was not serializable; its output is not cited as evidence.

## Comparison (i) — writer boundary versus restart: EXACT

For every available `N=1..9`, the record's `kt=N+1`, stage-1 entry is
word-identical to NEMO restart N: all 63 mapped SSH/U/V, carried U/V, and
before-stress rows pass.  This independently confirms that the additive
writer captures NEMO's actual whole-step boundary.  `N=10` is UNMEASURED,
not inferred: the sanctioned writer stops at `kt=10`, so there is no `kt=11`
stage-1 entry to compare with restart 10.

## Comparisons (ii) and (iii) — certified program versus restart: DEBT

At step 1, legoESM's from-rest production SSH, U, and V are word-exact to the
restart; only 640 wet forcing V-stress words differ as `+0.0` versus `-0.0`.
The first numerical prognostic difference appears at step 2:

- SSH: 189 unequal wet words, maximum absolute residual
  `1.3552527156068805e-20` m;
- V: 22,421 unequal wet words, maximum absolute residual
  `2.6800651846718097e-22` m/s;
- U: word-exact.

By step 10, SSH has 278,318 unequal wet words with maximum residual
`2.168404344971009e-19` m, U has 3,143 at
`2.710505431213761e-20` m/s, and V has 133,084 at
`2.541098841762901e-20` m/s.  Across all ten steps, production-state versus
restart has 64/100 exact rows, 26 numerical-debt rows, and 10 signed-zero-only
rows.

The chained ladder reaches exactly the same candidate SSH/U/V words as the
certified production loop after every step `N=1..10`.  It is exact at step 1
and first differs from restart N in the same SSH/V rows at step 2.  Its 27/70
exact and 43/70 numerical-debt restart rows therefore do not identify a
different ladder composition: they reproduce the certified production
composition.

## Comparison (iv) — stage-1 output versus stage-2 entry: DEBT

The first dynamic intermediate debt is the accepted `kt=2` finding.  On wet
cells, stage-1 replay versus NEMO's stage-2 entry differs in 1,278 SSH words
by at most `3.3881317890172014e-21` m, and in 408,318 V words by at most
`6.617444900424222e-24` m/s; the associated ratio fields also differ in their
last bits.  Across `kt=1..10`, comparison (iv) has 33/130 exact rows, 77
numerical-debt rows, and 20 forcing-signed-zero rows.

The executed source applies the stage-1 U/V and ratio halo exchanges and swaps
the new level into `Nnn` before stage 2.  The additive writer then writes
`ssh`, `uu/vv`, and all ratio arrays at `Nnn`.  Between that close and the
first stage-2 use of `Nnn`, NEMO zeros only the separate `Nrhs` arrays; it does
not touch a recorded array.  The final stage applies its halo operations,
assigns `Nbb=Naa`, and writes the restart from that same level.  Thus the
existing dump is already at the array, time level, and moment consumed by
stage 2.  A second writer dump would duplicate the first and is not proposed.

Together with comparison (i), this makes the writer innocent.  Together with
the candidate equality in comparison (iii), it also makes the ladder's stage
composition innocent.  The remaining measured owner is the certified
materialized per-stage/per-step arithmetic composition, although the evidence
does not isolate one source statement within that composition.

## Reconciliation with the long GPU trajectory

The operator's year-1, year-2, and year-3 endpoint comparisons remain valid:
the non-certified native GPU run is word-identical to NEMO at all three yearly
restarts.  Its manifest identifies a distinct execution composition, one
jitted `lax.scan` per 1,728-step day.  Round 3 already measured last-bit
differences between that fast scan and the certified Python/materialized-stage
loop.  Exact yearly scan endpoints therefore do not establish exactness of
the formal certified ladder composition, and the step-2 restart result does
not contradict the operator's long-run result.  Certifying the scan would be
a separate campaign decision; this round does not silently substitute it.

## Preserved gates, tests, and preflights

The exact bar is unchanged.  The shipped no-argument card digest remains
`f9d46bfd08f23702`; its mounted rows remain given-entry 338/338, chained
338/338, independent 50/50, carried 29/29, and month 690/690.  The final
SWG-focused battery records **243/243 passed** in `round7/direct_tests.log`;
it includes the card, production stage, mounted shipped ladders, acquisition
checks, climate instrument, GPU-driver tests, round 6, and all five direct
round-7 tests.  Its log SHA-256 is
`8d96afffc4067ac3226d4dfdddb0e1cdd131df8825662dc75309d2d395df15a0`.

All four no-run acquisition preflights pass in
`round7/acquisition_preflights.log` (SHA-256
`b48a33b4c0b3fd03e488238860070d3c2ff369328cd8d0ca95f612c8802eb379`):
shipped and Maddison stage/month, with every target created from shipped
`SWG` and the selected case overlaid afterward.  The daily cadences remain 24
and 432 steps/day.  No `makenemo`, `mpirun`, or NEMO engine command ran.
Focused citation-gate tests record 17/17 passed.  Focused Ruff and
`git diff --check` pass.

## Decision needed

My pick is to retain exact IEEE comparison for prognostic fields, compare
forcing-only `+0.0` and `-0.0` as numerically equal, and carry this certified
per-step arithmetic debt openly.  The alternative is to authorize a separate
campaign that certifies the one-JIT-per-day scan composition; neither changing
the exact prognostic bar nor altering arithmetic is justified by this round.

## Open and inherited items

- The forcing signed-zero ruling remains open until the user accepts or
  rejects the recommendation above.  The current emitted bar is unchanged and
  therefore continues to report those rows as representation DEBT.
- The unavailable `kt=11` entry leaves comparison (i) at `N=10` UNMEASURED.
  Because `N=1..9` all pass and the capture point is source-identified, no new
  acquisition is requested solely for that row.
- The unrelated inherited GYRE/LOCK_EXCHANGE/OVERFLOW digit pins remain stale.
  Expected values are `5850bbcfcd351c29` / `4ec82201f0fae7bf` /
  `7f1ec89b3e6578f5`; current values are `f02d2f42a570f413` /
  `d5dfdf83b3a1e0ab` / `83de0010e1170032`.  This lane does not re-pin them.

## Oracle citations

The executed stage-1 path exchanges U/V and ratio halos, then swaps `Naa` into
`Nnn`, at
`SWG_DG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:204-210`.  The additive writer
records stage 2's `ssh(:,:,Nnn)`, `uu/vv(:,:,Nnn)`, ratio fields, carried
fields, and stresses at
`SWG_DG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:220-235`.  The next statements
zero `Nrhs`, then stage 2 consumes `Nnn` in advection, vorticity, pressure,
and SSH at
`SWG_DG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:238-255`.  The final stage's
halo exchange, level assignment, and restart write are at
`SWG_DG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:360,366,376`.
