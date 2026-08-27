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

Version 6's first rerun produced **no valid measurement**: the strengthened
counter found two production helper calls, not the preregistered one. Source
tracing shows that active leapfrog evaluates `_step_impl` first on Nnn for the
advective/barotropic pass and again on Nbb for the dissipative pass whose
barotropic result is discarded (`ocean_model_latlon_cgrid.py:8396-8455` and
the following Nbb pass). Version 7 registers the live order as
`stress(Nnn), barotropic(Nnn), stress(Nbb), final momentum-vmix`. The source
operand is captured from the first call, whose explicit tendency feeds the
live barotropic pass. Swapping the first two event labels is the planted
violation and must fail the order equality. No score from the refused run is
reused and no science threshold changes.

Evidence re-review found that version 7 still *bypassed* rather than wired the
allocated diagnostic slot. Version 8 registers the actual offline wiring:
load `utrd_tau/vtrd_tau` from `DINO_00005764_restart.nc`, require their emitted
3-D arrays to be exact zero, copy them, populate only top level `k=1` with
`(poststress-prestress)/rDt`, require every lower level to remain exact zero,
and feed all source scoring through `rDt*wired_slot[k=1]`. Controls plant a
nonzero in the emitted slot, a nonzero at a lower level, and a material top-
level reconstruction error; each must fire. The restart file joins the hashed
inputs. This changes the data route, not the registered quantity or bars.

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
`vtrd_store(:,:,:,jpdyn_tau)` slot is emitted to the step-5764 restart but
never populated by the DINO dump path. The offline probe preserves that donor,
copies its 3-D arrays, and wires the copies as registered in version 8.

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
and no larger than `1e-12` on the legoESM side. Each arm must show the exact
event sequence `stress, barotropic, stress, momentum-vmix`; a different count
or order, non-finite data, or failed entry identity invalidates every score.

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
  --days 5 --bridge-before --save-step-eta --surface-stress-implicit
```

Exact explicit control:

```sh
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf results/dino_1455/wind_place_explicit.npz \
  --days 5 --bridge-before --save-step-eta --no-surface-stress-implicit
```

The artifacts must stamp `surface_stress_implicit=True` and `False`,
respectively. `--save-step-eta` makes the primary `eta` payload exactly 160
per-step fp64 samples, writes relative `t_seconds=2700..432000`, stamps
`capture_every_steps=1`, and refuses a non-fp64 materialized state. Daily eta
is retained separately as `eta_daily`.

Rebuild the registered NEMO comparator from its existing certified per-step
run (the extractor verifies kt, `rDt=2700`, fp64, and NEMO's Asselin identity):

```sh
.venv/bin/python scripts/validate/ocean_fidelity/dino_1226/eta_wave_twin.py \
  extract-nemo --run-dir /tmp/dino_eta_waves/nemo_5d \
  --kt0 5760 --nsteps 160 \
  --out results/dino_1455/nemo_day180_5d_eta.npz
```

Score each arm against that same comparator with the exact registered tool:

```sh
.venv/bin/python scripts/validate/ocean_fidelity/dino_1226/eta_flicker_decay.py \
  --nemo results/dino_1455/nemo_day180_5d_eta.npz \
  --lego results/dino_1455/wind_place_implicit.npz \
  --out results/dino_1455/wind_place_implicit_flicker.json

.venv/bin/python scripts/validate/ocean_fidelity/dino_1226/eta_flicker_decay.py \
  --nemo results/dino_1455/nemo_day180_5d_eta.npz \
  --lego results/dino_1455/wind_place_explicit.npz \
  --out results/dino_1455/wind_place_explicit_flicker.json
```

The NEMO extractor's binary md5 must remain the registered
`d3cf9242289b633d671013d0299803a3`; any missing run or changed receipt is a
STOP, not permission to substitute daily output. No GPU arm runs here.
