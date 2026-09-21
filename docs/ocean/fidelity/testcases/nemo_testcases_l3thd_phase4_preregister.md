# Lane 3b SI3 thermodynamics Phase 4 preregistration

Tracker: `climate-federation/legoESM#1699`

Status: **PREREGISTERED before adding the snow-temperature operand dump,
evaluating kt4239/kt5285, or changing the legoESM implementation.**

## Fixed protocol and identity

The oracle, forcing, one-hour step, CPU backend, explicit fp64 policy, and
`1e-15` boundary bar remain fixed.  The only supported identity remains the
ORCA1-resolved one-category HFN column: BL99 3+3/P07, `nn_icesal=2` with
`rn_sinew=.75`, ponds and lateral melt off.  No public selector or alternate
ice model is introduced.

NEMO calls `ice_var_glo2eqv(2)` before atmospheric coupling and `ice_thd`
(`icestp.F90:182-206`).  That conversion reconstructs each snow-layer
temperature only for `v_s > epsi20`, clamps it to `[rt0-100,rt0]`, and otherwise
sets it to `rt0` (`icevar.F90:404-416`; `epsi20=1e-20` at
`par_ice.F90:125-126`).  `ice_thd_1d2d` then copies `t_s` independently into
the selected-category array before zeroing volumetric snow enthalpy when
`h_s*a_i <= epsi20` (`icethd.F90:343-355,418-435`).  BL99 treats every
`h_s>0` column as snow-present and stores that copied temperature as its old
snow operand (`icethd_zdf_bl99.F90:159-198`).

The current legoESM execution reconstructs snow temperature from category
enthalpy in `bitz_lipscomb.py::_si3_zdf_bl99_step`, resets only exactly absent
snow, and does not apply NEMO's upper/lower temperature bounds.  The comparison
therefore currently lacks a registered oracle `t_s` operand at the
post-`ice_thd_1d2d`, pre-ZDF time level.

## Kt4239 discriminator and scaling

Add one write-only field, `t_s_1d(1:npti,1:nlay_s)`, to a versioned copy-only
ZDF-input stream.  Its Rule-1d registry entry is: **current ice step, after
`ice_var_glo2eqv(2)` and `ice_thd_1d2d`, immediately before
`ice_thd_zdf`**, sourced to `icestp.F90:182-206`,
`icethd.F90:343-355,418-435`, and `icethd_zdf_bl99.F90:189-199`.

Primary hypothesis **H4239-SNOW-TEMPERATURE-BOUNDS**: kt4239 is owned by
legoesm's missing NEMO snow-temperature reconstruction bounds, not by an
unrecoverable carried state.  It is confirmed only if:

1. the new dump at kt4239 equals the source replay from global `v_s/e_s`
   using NEMO's `v_s > epsi20` predicate and `[rt0-100,rt0]` clamp;
2. the first material exact-entry disagreement remains POST_ZDF and the
   dumped `t_s` differs from legoESM's unbounded reconstructed operand;
3. a private `_nemo_snow_temperature_bounds=False/True` arm changes no other
   input and improves kt4239 POST_ZDF `e_i` by at least 100-fold;
4. the effect scales from the dumped operand difference through the ZDF
   matrix/solution, with the arm delta explaining the observed row.

It is refuted if the dump contains a value that cannot be reproduced from the
registered global state and active reset rule, if the first mismatch precedes
ZDF, or if the one-variable arm improves by less than 100-fold.  If confirmed,
the NEMO bound/reset is unconditional within this SI3 identity and lands
unbranched; the private disabled hook remains gate-only.

## Kt5285 discriminator

After the kt4239 repair, enumerate the first remaining branch split.  The
registered candidate is the BL99 surface-energy branch: NEMO chooses cold
surface when `t_su_1d < rt0` and fixed melting otherwise, separately for
snow-present (`h_s_1d>0`) and snow-free rows
(`icethd_zdf_bl99.F90:433-513`).  Confirm only if the same exact entry plus the
new `t_s` operand yields different Boolean conditions before repair and equal
conditions after repair.  Otherwise name the first differing condition from
the executed branch trace rather than retaining this hypothesis.

## Full-year and machine-readable acceptance

The post-fix gate must run all 8,760 exact-entry operators and the continuous
trajectory.  It will retain an `oracle_entry_operator_sweep.per_step` array of
8,760 records, each containing that step's maximum normalised error, its
field/sub-call owner, and its over-bar row count.  It must also retain the
largest exact-entry outlier with step, sub-call, field, absolute numerator,
normalisation denominator, and normalised quotient.  A plant that removes one
per-step record or corrupts the largest-outlier attribution must exit nonzero.

Report without changing the bar: first over-bar frame; total over-bar rows;
continuous normalised errors for `t_su/e_i/h_i/h_s` at steps
1, 10, 100, 1000, 3000, 5000, and 8760; and the six established phenomenology
rows before/after.  Labels remain numeric/DEBT unless the registered bar or
precision-floor classification is actually attained.

## Choice register

- ASKED — register and first-divergence the kt4239 snow-temperature operand.
- ASKED — quote and test the active NEMO tiny/zero-snow reset and carry rules.
- ASKED — preregister a one-variable arm, scaling test, and identity-internal
  repair where NEMO has no switch.
- ASKED — identify the first remaining branch split at/after kt5285 with both
  sides' conditions.
- ASKED — rerun the year gate with the requested growth and phenomenology rows.
- ASKED — commit all 8,760 oracle-entry per-step records and largest-outlier
  numerator/denominator attribution with fail-closed controls.
- ASKED — CPU only, copy-only NEMO instrumentation, explicit-path local-git
  commits/bundle, no shipped-tree edits, and no push.
- UNASKED — alternate SI3 identities, public physics switches, threshold/bar
  changes, forcing changes, coupled bulk-flux certification, GPU/MPI work, or
  deletion of any earlier run root.
