# Round 67 pre-code self-review: stage-3 LDF order

Date: 2026-09-12. Completed after the admitted-record substitution and before
editing the production stage program.

## Claim and source walk

NEMO's compiled stage clears `Krhs`, then calls advection/SBC
(`stprk3_stg.f90:827-868`); stage 3 calls QSR, LDF, then ZDF at `:917-965`.
`trazdf.f90:416-479` adds isoneutral K33 to the implicit matrix, while
`:547-565` constructs content from Kbb tracer plus Kmm-weighted Krhs before the
solve. legoESM constructs SBC/QSR rates at
`ocean_model_latlon_cgrid.py:6223-6314`, adds GM/Redi to `T_mid/S_mid` at
`:7474-7480`, but its later WS call restarts at `state.T/state.S` and the saved
stage-2 result (`:7639-7677`). Only `stage_source_rates[2]` reaches the content
statement (`:1905-1917`) passed to ZDF (`:8038-8150,10417-10438`). Therefore
the existing GM/Redi result is discarded on this program, not applied after
the solve.

The source-literal edit is one statement: for `rk3_ws`, add the signed
concentration tendency returned by the already-executed GM/Redi operator to
`_stage_source_rates[2]`; retain every other program's existing concentration
update. This introduces no field/default/card choice. K33 remains exclusively
in the implicit matrix.

## Measurement audit

The clean stamped report at `07f8399ad8e5b06b7f398f5ceac32b48c6c1e326`
passed its all-oracle exact rebuild and one-ULP control. Routed-live T content
is `5.954039670542e-5`, versus the round-66 baseline `1.679392691671e-3`
(28.2x improvement); the pre-edit content override predicts kt3 T/S
`8.916073070964e-7`/`7.235655630211e-8`.

The component rows identify complete FCT advection (`6.196948294061e-11 K/s`)
as the primary remaining Krhs owner. SBC is `1.052042188174e-12`, QSR
`1.168556066120e-16`, and the already-known model LDF mismatch is
`3.901917581501e-12`. Thus the preregistered owner is confirmed, but its phrase
“post-SBC substitution does not remove the floor” is **REFUTED**: post-SBC is
cumulative and necessarily replaces advection too. The component split, not
that cumulative-arm phrase, discriminates the owner. Post-LDF reproduces the
frozen round-66 Kmm-association row (`1.359694579151e-10` content), so neither
K33 nor a new thickness association owns the routed pre-solve floor.

## Rule-12 risk review

- GYRE executes the changed statement; its exact override prediction, kt=1--10
  ladder, first-over-bar, and day 1--30 register are mandatory.
- LOCK_EXCHANGE/OVERFLOW use the same `rk3_ws` function but their resolved NEMO
  namelists set `ln_traldf_OFF=.true.` and their legoESM cards set
  `gm_redi=None`; before/after trajectories must nevertheless be bit-identical.
- DINO uses the separate modified-leapfrog branch; the static branch must leave
  it byte-identical. ORCA2 remains UNMEASURED-with-spec.

Two attempts to invoke the requested read-only nested Codex review produced no
verdict: the first could not initialize its read-only state; the isolated-state
retry could not reach the Codex service. This is recorded, not treated as an
approval. The separately required post-diff invocation remains mandatory.

Self-review verdict: **EDIT ELIGIBLE**, subject to every preregistered gate.

## Follow-up self-review before the association edit

The first implementation was correctly REFUTED by the exact kt3 criterion.
The discrepancy is traced to the helper's two-sum content association at
`ocean_model_latlon_cgrid.py:1991-1997`, not to LDF sign, level, mask, or K33.
The follow-up rewrites that one shared statement into the literal compiled
`trazdf` Krhs association; it does not alter fluxes, stages 1/2, the matrix, or
the solver. Because it executes on tank cards even when LDF is absent, their
bit-inert waiver is withdrawn and their complete before/after rows are now a
landing gate. Follow-up self-review verdict: **EDIT ELIGIBLE**.
