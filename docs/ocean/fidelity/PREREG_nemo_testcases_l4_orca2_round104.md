# Preregistration — ORCA2 round 104 EEN coefficient zero signs

Date: 2026-10-02. Base: `d99367ba27b1f1e4da6311e0ea2c2bf778618c3a`.
Scope is ocean only. Every ORCA2 number is **independent**: hierarchy rung 0
starts from NEMO's own from-rest state. Sea ice, the shipped ORCA2 card, and
its `unmeasured_features` tuple are unchanged.

## Frozen source order and candidate

The executing NEMO `np_EEN` arm constructs live U/V thickness factors as
`e3u_3d * (1 + r3u * umask)` and `e3v_3d * (1 + r3v * vmask)`, then applies
exactly one neighbor-face mask inside each coefficient recurrence
(`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:1213-1265`). The current
literal builder additionally multiplies both live thickness arrays by their
own masks before the recurrence
(`barotropic_latlon_cgrid.py:986-987`), even though the card-carried reference
thickness is nonzero at 386,170 dry U cells and 384,025 dry V cells. The first
one-variable arm removes only those two extra terminal masks. It retains the
compiled neighbor masks, operand order, accumulation, and final scale.

This is a shared statement. Before any landing, the round must print which
cards execute it and apply the registered ORCA2, GYRE, DINO, tank, citation,
and push gates. No configuration value, selector, threshold, stabilizer, or
carried state may change.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R104-P1 | The extra early face masks are the first owner of round 99's non-fold signed-zero coefficient debt. | With only those masks removed, all eight coefficients have zero non-fold bit differences against the admitted two-rank NEMO record. | Any non-fold bit difference remains, or any previously equal non-fold magnitude moves: **REFUTED**; retain the failed arm and continue through accumulation/final scale in source order without landing it. |
| R104-P2 | The northern-fold magnitude debt is independent of the zero-sign owner. | The candidate retains exactly the previously registered 66 `ffv_nw` and 67 `ffv_ne` fold-row magnitude differences and no other magnitude differences. | Any magnitude support changes outside those two fold rows: stop and reconcile the instrument before a claim. |
| R104-P3 | Exact coefficients close the first two `dyn_cor_2D` applications while the later 68-cell substep-2 U residual remains. | Substep-1 U/V are bit-exact; substep-2 U retains 68 unequal cells with the registered maximum near `2.9617669311254642e-8`; substep-2 V remains bit-exact. | Any other application movement: stop and name the changed row; do not combine owners. |
| R104-P4 | A one-bit change in one candidate coefficient is detected through both coefficient and consumer scoring. | Both plants exit nonzero at the named cell/row. | Either plant stays green: the instrument is invalid and no result is citable. |
| R104-P5 | GYRE either does not move or passes its standing Decision 43/45/55/59 gate; no ORCA2 AT-BAR row leaves the bar and the first-over-bar checkpoint is not earlier. | Mechanical gates pass with every moved row registered. | Any gate violation: hold the model statement and name the exact failing row. |

## Landing bar

The statement lands only if R104-P1 through P5 pass, the production builder is
bit-exact on every non-fold coefficient given NEMO operands, the rung-0 and
rung-7 ten-step gates pass, shared-card gates pass, focused tests and the
prescribed ocean-fidelity battery are run one at a time, the citation gate and
its plant fire, and the separate read-only Codex review is recorded. Otherwise
the round is **HELD** at the first falsified prediction.

## Post-refutation preregistration — loop-bound arm

Committed after R104-P1 was measured and **REFUTED**, before measuring this
second arm. Removing the early masks caused 2,115–2,693 new non-fold magnitude
differences per coefficient and maxima up to `2.5812540150707704e-3`; that
failed prediction and its artifact remain part of the record.

The source-order discriminator is the loop domain. NEMO evaluates the U and V
recurrences only for `jk=1:mbku(ji,jj)` and `jk=1:mbkv(ji,jj)`
(`dynspg_ts.f90:1215-1234,1242-1261`). The vectorized builder instead evaluates
all levels and turns out-of-column terms into signed zeros with an extra local
mask multiplication. Arm 2 removes the two early local masks, preserves the
compiled neighbor-mask products, and conditionally updates the accumulator
only where the local face mask says that level belongs to the NEMO loop.

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R104-P6 | Explicit `mbku/mbkv`-equivalent loop gating owns the non-fold zero signs without changing magnitudes. | All eight coefficients have zero non-fold bit differences; only the registered 66/67 fold magnitudes remain. | Any non-fold bit difference or any new magnitude difference: **REFUTED** and hold without a model landing. |
| R104-P7 | The source loop arm closes substep-1 U/V and leaves the independent 68-cell substep-2 U fold residual. | Same consumer rows as R104-P3. | Any other row moves: hold and name it. |
| R104-P8 | Replacing the conditional update with unconditional masked arithmetic makes the synthetic dry-column control fail. | Direct test observes positive-zero accumulator for the skipped loop and a negative zero for the planted unconditional term. | Control stays green: no landing. |
