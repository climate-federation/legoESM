# PREREGISTRATION — end-wall wind-stress placement

Frozen before the first execution of `endwall_wind_placement.py` on this
branch.  The source alignment was read first; no array in the four KT lanes
was numerically reduced before this file was committed.

## Version 2 control correction — frozen before rerun

The first execution produced **no valid measurement**: its 2x planted control
was applied to the whole B1 state and failed (`1.108443e-10 m/s` against an
absolute `1e-12` gate).  That control premise was wrong.  B1 is downstream of
the state-dependent barotropic recurrence, so it is not the linear operator
the control claimed to test.  No threshold is relaxed and no failed score is
reused.

The rerun moves the same 0x/1x/2x control to `surface_stress_faces`, the
single-owner linear stress sign/interpolation/rotation helper that supplies
both placements.  It requires exact equality of `tau(0x)=0` and
`tau(2x)=2*tau(1x)`; a planted `nextafter` mutation of one returned wet-face
value must make that equality fail.  Whole-step 2x departures are printed as
diagnostics only and carry no gate.  The primary operands and all four science
statistics below are unchanged from version 1.

## Version 3 arm correction — frozen before second rerun

The version-2 rerun also produced **no valid measurement**.  Its helper
control reported `zero_exact=False` and `double_exact=False` because the arm
scaled only the current stress.  This card applies
`0.5*(state.tau_*_prev + surface_forcing.tau_*)`; leaving the previous carry at
1x means the alleged 0x arm is physically a 0.5x arm and the alleged 2x arm is
1.5x.  The control correctly exposed that confound.

Version 3 scales `state.tau_x_prev/state.tau_y_prev` and the current forcing
together.  Those two fields are one centred wind input, not two variables.
The exact helper control, primary operands, statistics and bars are otherwise
unchanged.  Neither invalid run yielded a classified term score.

## Version 4 mask-loader correction — frozen before third rerun

The version-3 execution passed the helper and structural setup controls, then
refused before its first statistic: the raw `mesh_mask.nc` in this lane is
already `199x52`, and the probe stripped a second two-cell halo, producing a
`195x48` mask against `199x52` operands.  No score was computed.  The rerun
uses `g.umask/g.vmask` from `multistep_replay.build_replay_ic`, the same shared
loader and already-aligned masks that built the bridged state.  Criteria and
science operands are unchanged.

## Claim and inputs

Candidate: NEMO places the centred surface-stress increment inside `dyn_zdf`,
whereas the shipped `nemo_dino_kamm_mlf` card places the same increment in the
explicit momentum RHS before the implicit solve.  NEMO also adds the same
centred stress independently to `zu_frc`; this registration does not confuse
that barotropic add with the vertical placement.

The primary offline state is kt=5761.  NEMO inputs are the committed
`RUN_SEQDUMP_KT5761_1R/zdf_dump_{u,v}1_{pre,post}stress.bin` brackets and
`wnd_dump_z{u,v}_frc_inc.bin`.  The legoESM side reuses
`multistep_replay.build_replay_ic`, `post_tendency_stage_birth`'s stage hook,
and the production `surface_stress_faces`/barotropic call.  It runs the same
bridged state with 0x, 1x and 2x stress; every other input is identical.

The dead oracle slot being repaired offline is
`utrd_store(:,:,:,jpdyn_tau)` / `vtrd_store(:,:,:,jpdyn_tau)`: the probe fills
its top level with `(poststress-prestress)/rDt` instead of reading the emitted
all-zero slot.  This is an analysis repair only; it does not alter the oracle.

## Exact offline numbers

For the whole wet u-face domain and separately for the southern end-wall row
`j=1`, the probe will print:

1. `zdf_increment_err_norm`: RMS of
   `(lego B1[wind]-lego B1[zero]) - (NEMO poststress-NEMO prestress)`, divided
   by the RMS of the NEMO bracket, both in m/s.
2. `zdf_increment_corr` and `zdf_increment_rms_ratio` for the same operands.
3. `fslow_err_norm`: RMS of
   `(lego F_slow[wind]-lego F_slow[zero]) - NEMO wnd_dump_z*_frc_inc`, divided
   by the NEMO RMS, both in m/s2.  This is also the dynamic check of premise 4:
   the supplied slow forcing reaches the loop in tendency units, without a
   second face-depth scaling.
4. `placement_delta_rms`: the numerator RMS in item 1, in m/s.  This, not the
   dead `utrd_tau`, is the measured placement term.

The 2x planted control is applied at `surface_stress_faces` as specified in
version 2 above.  The meridional stress is a structural-zero control on this
DINO forcing and must be exactly zero on the oracle side and no larger than
`1e-12` on the legoESM side.  A hook count other than one per arm, a non-finite
value, a failed entry-state identity gate, or a failed control invalidates
every score.

## Interpretation frozen in advance

This offline term test **CONFIRMS algebraic placement equivalence** when, on
both the domain and `j=1`, `zdf_increment_err_norm <= 0.05`, correlation is at
least 0.999, and `fslow_err_norm <= 0.05`.  It **CONFIRMS a material placement
DIFF** when either end-wall normalized error is at least 0.25.  Values between
are `PLAUSIBLE/UNRESOLVED`.  These bars classify the measured term, not
ownership of sea-surface flicker; the quantities have different units and no
post-hoc transfer coefficient will be invented.

Ownership requires the one-variable free-run arm below.  Its exact primary
number is the last-half, per-cell-first zonal-wall 2dt amplitude ratio
`A_lego/A_nemo`, with wall variance share reported alongside it.  Starting
from the bridged day-180 state:

- **CONFIRMS ownership:** implicit-placement arm ratio `<= 1.25` and wall share
  `<= 0.08`, while the control reproduces `2.89` within its registered
  interval and NEMO's share is about `0.04`.
- **REFUTES ownership:** implicit-placement arm ratio `>= 2.30` and wall share
  `>= 0.68` (at least 80% of the present 2.89 ratio and 85% share remain).
- Anything else is **UNRESOLVED** and is reported without a causal label.

The required implementation must make `surface_stress_implicit=True`
compatible with the active leapfrog reconciliation before running; the current
constructor deliberately refuses that combination.  Exact invocation after
that faithful path exists:

```sh
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf results/dino_1455/wind_place_implicit.npz \
  --days 5 --bridge-before --save-3d
```

The arm must stamp `surface_stress_implicit=True`; an otherwise identical
control stamps `False`.  The existing per-step eta extractor and
`eta_flicker_decay.py` score the first 160 samples.  No GPU arm is run here.
