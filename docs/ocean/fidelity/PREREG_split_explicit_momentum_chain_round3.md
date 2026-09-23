# PREREGISTRATION — split-explicit momentum chain round 3

Frozen before the first round-3 numerical execution.  This round imports the
already-built T-point restart-stress correction from unmerged PR #1695 and
continues row 1.1 from the faithful `zu_frc` residual measured in round 2.

## Imported correction and merge order

The implementation is reused by cherry-pick, not reimplemented:

- upstream `73ad090d411027ae3815f6ae8ea7b7ff0f85f0dc` builds the analytic
  DINO T-point prior-stress reconstruction, content/time/stagger receipt, CLI
  selector, fail-closed gate, and tests;
- upstream `f862e1554911f0901ab15b7ede80de7a74d2f1ac` makes that faithful carry
  the harness default and retains an explicit legacy U-as-T reproduction arm;
- local cherry-picks are `8327ffb10bc` and `7061c59af51`, respectively.

This branch therefore has a merge-order dependency on PR #1695.  If #1695
lands first, these two patch-equivalent cherry-picks must be dropped while the
round-3 receipts remain.  If this branch lands first, #1695 must rebase or omit
the same two changes.  Neither branch may land a second implementation of the
carry correction.

Round 2 measured the corrected carry as a same-input counterfactual before the
default changed.  Its admitted receipt is reused without rescoring: U residual
reduction `0.9928695526681799`, prediction correlation
`0.999974655141223`, and prediction normalized error
`0.007130447331820113`, yielding `CONFIRMS_BRIDGE_WIND_SOURCE`.  Round 3 will
label the centred-wind row `FIXED-BY-#1695` only if the imported commits are
ancestors of the measurement commit, the default/legacy selector tests pass,
and the regenerated round-2 receipt reproduces those values under the frozen
round-2 bars.

## Source-ordered continuation

The faithful U assembled residual remains DEBT at
`E=1.9539648558722173e-6`.  The existing term order and NEMO sources remain:

| Order | Constituent | Active NEMO source | Round-2 faithful status |
|---:|---|---|---|
| 1a | kinetic-energy gradient | `dynadv.F90:89-95`; `stpmlf.F90:309-314` | NEAR-CLASS; REFUTES_CARRY |
| 1b | vertical advection | `dynadv.F90:97-103`; `dynzad.F90:81-119`; `stpmlf.F90:309-314` | DEBT; first ordered peel |
| 2 | total EEN vorticity/Coriolis | `dynvor.F90:147-193`; `stpmlf.F90:315-318` | DEBT |
| 3 | lateral friction | `dynldf.F90:69-119`; `stpmlf.F90:319-322` | DEBT; UNRESOLVED |
| 4 | hydrostatic pressure gradient | `dynhpg.F90:348-413`; `stpmlf.F90:324-328` | NEAR-CLASS |
| 5 | REST depth mean | `dynspg_ts.F90:316-339` | ledger prerequisite |
| 6 | pre-loop 2-D Coriolis removal | `dynspg_ts.F90:358-370` | DEBT |
| 7 | baroclinic-residual drag | `dynspg_ts.F90:372-400` | DEBT |
| 8 | atmospheric pressure | `dynspg_ts.F90:404-421` | structural zero, `ln_apr_dyn=F` |
| 9 | centred wind | `dynspg_ts.F90:423-459` | `FIXED-BY-#1695` candidate |
| 10 | assembled forcing | `dynspg_ts.F90:507-525` | faithful DEBT |

Vertical advection reuses the committed ZAD term decomposition rather than
walking it again.  The flux-vs-advective defect is already fixed by
`84c169eb770`; the bottom/straddling-face mask defect is already fixed by
`5bdcf219edc`.  The surviving day-180 depth-mean term error will be scored
against the faithful assembled residual using the exact round-2 attribution
classifier.  If it REFUTES_CARRY, order advances to vorticity; otherwise row
1.1 stops there.  The same rule then applies to vorticity.  No term may be
skipped because a later term has a larger correlation.

## New measurements and frozen bars

The probe reruns the committed round-2 measurement on CPU/fp64 from the same
18 existing `RUN_SEQDUMP_D180_1R` dumps, intercepting its already-validated
raw term/error operands without altering the round-2 algorithm.  It reports:

1. the faithful U attribution for every term, in source order;
2. every two-term sum involving the first unresolved source-ordered term;
3. the vorticity plus pre-loop-Coriolis sum, because both individual errors
   are larger than the faithful residual and have opposite signed response;
4. a lateral-friction-only time-level counterfactual
   `F_cf = F_faithful - LDF_NOW + LDF_BEFORE`.

`LDF_BEFORE` is evaluated through the production diagnostics path with only
`ldf_state=(T_before,S_before,u_before,v_before)` changed.  This mirrors NEMO
`dyn_ldf(Kbb)` at `dynldf.F90:69-119`; all other tendency operands, the
faithful stress carry, state, geometry, clock, and reduction are held fixed.
It is an offline forcing-operand counterfactual, not an admitted production
fix and not a replacement for the actual-F-slow closure receipt.

For residual `R` and candidate error `T`, retain the round-2 axes:

- `gain = RMS(T)/RMS(R)`;
- `removal = 1 - RMS(R-T)/RMS(R)`;
- `corr = corr(T,R)`;
- CONFIRMS_CARRY iff `corr >= 0.99`, `0.90 <= gain <= 1.10`, and
  `removal >= 0.90`;
- REFUTES_CARRY iff `abs(corr) <= 0.20` and `removal <= 0.10`; otherwise
  UNRESOLVED.

For the time-level substitution, `P=F_cf-F_faithful` confirms the source only
if `RMS(P+R)/RMS(R) <= 0.10`, `corr(P,-R) >= 0.99`, and
`1-RMS(F_cf-NEMO)/RMS(R) >= 0.90`.  It refutes only if those axes are
respectively `>=0.90`, `<=0.20`, and `<=0.10`; otherwise it is unresolved.
A confirming candidate is downgraded if any measured partner reverses at
least half its removal.  Identity and orthogonal classifier plants must still
confirm and refute, and a synthetic perfect correction/zero correction must
traverse the time-level classifier as CONFIRM/REFUTE.

## Continuation rule

After each source-ordered disposition, the next term may be scored.  Row 1.1
closes only if the corrected assembled U and V forcing both reach the campaign
bar.  Otherwise row 1.1 remains open at the first unresolved term and rows
1.2, 1.3, and execution-chain rows 2--6 remain ordered-blocked.  Descriptive V
scores cannot open independent ownership while U row 1.1 is open.

No new NEMO instrumentation is required; all oracle operands are existing
dumps and the BEFORE arm uses an existing production diagnostics hook.
Therefore this round allocates no SLOT block.  Any future NEMO writer must be
separately preregistered under the held-SLOT protocol before it is built.

## Reviewer-required integration amendment before rerun

The first accepted execution exposed two packaging defects under independent
review, neither of which changes a science operand or bar.  Before the rerun:

1. import the exact latest committed `PREREG_endwall_wind_placement.md` that
   `endwall_tpoint_bridge_gate.py` names and hashes, so the reused #1695 gate is
   executable from this branch rather than depending on a file outside it;
2. resolve the T-point carry CLI default after parsing the start mode: the
   faithful carry remains default for a bridged start, while
   `--legacy-euler-start` selects no prior-stress carry.  An explicit request
   for faithful prior stress with an Euler start must still fail closed;
3. extend the machine selector receipt to require both legacy-Euler booleans
   false, in addition to the four already-frozen default/legacy-carry fields.

These are import-completeness and control-path repairs only.  The round-3 term
ladder, source order, raw operands, attribution classifiers, cancellation
bars, continuation rules, and all reported science thresholds above remain
frozen.  The previous artifact is superseded; only a clean CPU/fp64 rerun from
a commit containing this amendment and the selector-receipt extension may be
packaged as the accepted result.
