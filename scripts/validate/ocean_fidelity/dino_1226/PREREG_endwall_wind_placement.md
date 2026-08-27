# PREREGISTRATION — end-wall wind-stress placement

Frozen before the first execution of `endwall_wind_placement.py` on this
branch.  The source alignment was read first; no array in the four KT lanes
was numerically reduced before this file was committed.

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

The 2x arm is a planted violation/control: both the pre-solve deposit and the
slow-forcing deposit must satisfy
`max|delta_2x - 2*delta_1x| <= 1e-12 * max(1, max|delta_2x|)`.  The meridional
stress is a structural-zero control on this DINO forcing and must be exactly
zero on the oracle side and no larger than the same numerical bound on the
legoESM side.  A hook count other than one per arm, a non-finite value, a
failed entry-state identity gate, or a failed control invalidates every score.

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

