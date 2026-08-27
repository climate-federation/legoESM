# PREREGISTRATION — end-wall wind source and placement response

Frozen before the first execution of `endwall_wind_placement.py` on this
branch. The source alignment was read first; no array in the KT lane was
numerically reduced before version 1 was committed.

## Audit trail: four refused runs

No result from these runs is reused.

1. Version 1 put the exact 0x/1x/2x linearity control on the downstream B1
   state. It fired because the intervening barotropic recurrence is
   state-dependent; the premise was invalid.
2. Version 2 moved the control to `surface_stress_faces`, but scaled current
   stress without the previous centred-stress carry. The exact-zero and
   exact-doubling controls correctly failed.
3. Version 3 scaled both stress time levels, then refused on a shape mismatch:
   it stripped a halo from an already aligned `199x52` mask.
4. Version 4 passed setup controls, then refused because it compared unequal
   control-flow spans: NEMO's direct `dyn_zdf` deposit against legoESM's B1
   state after the rotating barotropic loop. The valid lego operand is the
   direct production helper result `rDt*tau/(rho0*dz0)`; B1 is diagnostic.

Version 5 made that operand correction without changing the science bars.
Independent review then found that its prose still called a reconstructed
source operand a measured *placement response*. Version 6 restricts the
offline claim to source-operand agreement. Neither the alternative implicit
arm nor its tridiagonal response runs in the offline probe. The dynamic
placement response and eta ownership require the free-run A/B below.

Version 6 also proves exact 1x entry identity and that 0x/2x change stress
fields only; counts every stress, barotropic and momentum-vmix hook; stamps all
lane-selector environment values; and SHA-256 hashes the probe, current
preregistration, six operands, and every cited NEMO source. This is required
because the DINO `MY_SRC` files are not pinned by the oracle Git revision
alone. No science threshold changes.

## Candidate and inputs

Candidate: NEMO places the centred surface-stress increment inside `dyn_zdf`,
whereas the shipped `nemo_dino_kamm_mlf` card places the same increment in the
explicit momentum RHS before the implicit solve. NEMO independently adds the
same centred stress to barotropic `zu_frc`; this registration treats that as a
separate entry point.

The primary offline state is kt=5761. NEMO inputs are
`RUN_SEQDUMP_D180_1R/zdf_dump_{u,v}1_{pre,post}stress.bin` and
`wnd_dump_z{u,v}_frc_inc.bin`. The legoESM side reuses
`multistep_replay.build_replay_ic`, `post_tendency_stage_birth._load`, and the
production `surface_stress_faces` and barotropic call. It runs the same bridged
state with 0x, 1x and 2x centred stress; every non-stress input must be exact.

The oracle `utrd_store(:,:,:,jpdyn_tau)` /
`vtrd_store(:,:,:,jpdyn_tau)` slot is emitted to output but never populated by
the DINO dump path. The probe does not fill or mutate the slot; it reconstructs
the equivalent source operand from `(poststress-prestress)/rDt`.

## Exact offline numbers

For the whole wet u-face domain and separately for southern row `j=1`, print:

1. `zdf_increment_err_norm`: RMS of
   `lego rDt*tau/(rho0*dz0) - (NEMO poststress-NEMO prestress)`, divided by
   NEMO RMS, in m/s.
2. Direct-increment correlation and lego/NEMO RMS ratio.
3. `fslow_err_norm`: RMS of
   `(lego F_slow[wind]-lego F_slow[zero]) - NEMO wnd_dump_zu_frc_inc`, divided
   by NEMO RMS, in m/s2. This directly tests premise 4: `F_slow` reaches the
   loop in tendency units without another face-depth scaling.
4. The numerator RMS for items 1 and 3 in their native units.

The exact 0x/1x/2x control is on `surface_stress_faces`: `tau(0x)=0` and
`tau(2x)=2*tau(1x)` must be bit-exact, and a one-ULP planted mutation must make
the equality fail. Meridional forcing must be exactly zero on the oracle side
and no larger than `1e-12` on the legoESM side. A hook count other than one per
arm, non-finite data, or failed entry identity invalidates every score.

## Frozen interpretation

The offline source test **CONFIRMS algebraic source-operand equivalence** when,
on both the domain and `j=1`, direct-increment normalized error is `<=0.05`,
direct-increment correlation is `>=0.999`, and `F_slow` normalized error is
`<=0.05`. Correlation was not registered as an `F_slow` gate. It **CONFIRMS a
material source DIFF** when either end-wall normalized error is `>=0.25`.
Values between are `PLAUSIBLE/UNRESOLVED`.

These bars classify source operands only. They do not classify the response
through the implicit solve or sea-surface ownership, and no post-hoc transfer
coefficient will be invented.

## Free-run placement/ownership discriminator — GPU handoff

The exact primary number is the last-half, per-cell-first zonal-wall 2dt
amplitude ratio `A_lego/A_nemo`; wall variance share is reported alongside it.

- **Control validity:** explicit-placement ratio in `[2.60, 3.18]`
  (registered 2.89 ±10%) and wall share in `[0.75, 0.95]`
  (registered 0.85 ±0.10).
- **CONFIRMS ownership:** implicit-placement ratio `<=1.25` and wall share
  `<=0.08` while the control is valid.
- **REFUTES ownership:** implicit-placement ratio `>=2.30` and wall share
  `>=0.68` while the control is valid.
- Otherwise: **UNRESOLVED**.

The active shipped outer integrator is `leapfrog` with the split-explicit
solver, which accepts either placement. The twin harness exposes the existing
config field as a one-variable selector and stamps the resolved model value.
Exact implicit arm:

```sh
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf results/dino_1455/wind_place_implicit.npz \
  --days 5 --bridge-before --save-3d --surface-stress-implicit
```

Exact explicit control:

```sh
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf results/dino_1455/wind_place_explicit.npz \
  --days 5 --bridge-before --save-3d --no-surface-stress-implicit
```

The artifacts must stamp `surface_stress_implicit=True` and `False`,
respectively. Use the existing per-step eta extractor and
`eta_flicker_decay.py` to score the first 160 samples. No GPU arm runs here.
