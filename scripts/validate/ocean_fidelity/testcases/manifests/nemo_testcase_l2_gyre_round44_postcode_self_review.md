# Round 44 post-code self-review

Reviewed against the compiled GYRE sources and the frozen round-44
preregistration.

## Findings dispositioned

1. **Corrected before certification:** the first handoff draft tested the
   optional `_step_impl(config=None)` argument rather than the resolved
   `_cfg_b`, so it was a no-op.  The provisional ZAD exposure caught this;
   the rejected arm used `_cfg_b.vertical_momentum_scheme`.
2. **Corrected before certification:** NEMO's recorded `after_zad-after_keg`
   maximum includes accumulator rounding and is not the raw physical ZAD
   maximum.  The gate now prints it beside the model's raw already-computed
   `diag_vertadv` maximum, as requested, but does not score those unlike
   quantities against each other.  Given-input accumulator exactness remains
   owned by the round-41 gate.
3. **Rejected by Rule 12:** the first tracer-transport handoff improved
   GYRE's kt=2 maxima, but worsened individual cells and 57 trajectory rows.
   The corrected velocity-form handoff then moved kt=2 U/V to AT-BAR and the
   first-over-bar to kt=3, but still produced 55 cellwise violations (maximum
   worsening 160015661435.125 row-scale ulps).  Both were removed from the
   operator-facing tree.  Their clean-clone commits are evidence, not proposed
   commits; no production physics remains in this delta.
4. No new model hook remains.  The gate uses the pre-existing WRITE-only
   stage-2 velocity exposure, then runs the shared production WZV and vertical
   momentum kernels outside the model step.  That keeps the model file and
   all mechanically checked receipt line numbers unchanged.
5. The gate refuses tracked dirt, untracked files, and a mismatched full SHA.
   Its ww/wsd/ZAD/card/stamp plants all have nonzero-exit paths.
6. No tolerance, stabilizer, NEMO source, record, scheme default, or pending
   slope-scope choice changed.  The temporary clean-clone commit is evidence
   only; the operator must commit the listed primary-checkout paths and rerun
   gates at that resulting SHA.  The retained changes are preregistration and
   one fail-closed measurement gate; there is no production physics delta.

Independent Claude review remains pending as directed by the user.
