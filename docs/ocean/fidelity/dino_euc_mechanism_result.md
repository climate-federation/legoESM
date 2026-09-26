# DINO equatorial-undercurrent mechanism probe

Date: 2026-08-27  
Branch: `fidelity/dino-euc-mechanism-codex`  
State: shared NEMO day-180 restart; matched first step `kt=5761`

## Reported outcome

**FIX-FIRST verdict: SHIP `d86d97d49`; live PREFIX/FIX response
`CONFIRMED_DAY10_DAY90_DO_NO_HARM`.** Both round-6 reviewers found no blocker;
the day-10 causal response, day-90 persistence, and global do-no-harm bars all
CONFIRM. The prior headline that vertical mixing was unsupported is retracted.
The audited shear spans
5--26 m, so the 10.14 m interface is shear-setting, not peripheral. At that
interface legoESM's momentum viscosity is 2.0514 times NEMO's while the model's
shear is about 0.68 times NEMO's; `A_v * shear` is consequently similar
(`2.16e-5` versus `1.55e-5`). Excess near-surface viscosity predicts both
observed signs: weaker 5--26 m shear and a deeper zero crossing.

The closure root is now isolated: **equatorial near-surface TKE energy is
2.255635 times NEMO in geometric mean across the 47 excess columns at the
shear-setting interface.** The resulting mixing-length excess is algebraic,
not independent.

The TKE-equation decomposition named the implementation owner:
**pre-fix legoESM incorrectly applied the `1e-4` surface-TKE minimum to its first
interior interface under `tke_surface_bc_level="nemo_z0"`.** NEMO holds
`rn_emin0` only at the separate surface row (`jk=1`) and solves/floors `jk=2`
with `rn_emin=1e-6`. A preregistered replay removing only that misplaced
interior clamp closes 94.35% of the median log-energy gap and moves all 47
columns toward NEMO; the residual energy ratio is 1.0543.

The faithful production fix is now built in commit
`d86d97d496463b9317764426573af958fe5f20b9`. A clean, committed real-path
replay—not the earlier emulation—exactly reproduced the registered residual:
the 47-column geometric-mean energy ratio is `1.054275004`, and all 47 columns
move toward NEMO. Mixing length falls to `1.026778945` and momentum viscosity
to `1.054275003` in geometric mean.

The review-corrected offline bar confirms because 10--40 m NRMS is 0.68865 and
the single shear-setting 10.14 m ratio exceeds 1.25. The original
`UNRESOLVED_CLOSURE_DIFFERENCE` label is retained in the artifact only as a
retracted result from a physically invalid two-consecutive-level clause.

The coordinator's live 10-day PREFIX/FIX run now confirms the registered
causal chain. This lane did not run a GPU; it reads and records the executor's
completed artifacts below. The runner that produced those arms is now
committed, and its historical frozen-`A_v` verdict is explicitly superseded.

## ROUND-6 RETRACTIONS — DO NOT CITE THE SUPERSEDED CLAIMS

**RETRACTED: the frozen-`A_v` arm did not close 89% of a physical gap and is
UNINFORMATIVE about the clamp fix.** At 5.03/15.32/25.96/37.01 m its BASE
velocity errors versus NEMO were `+0.0553/-0.0173/-0.0953/-0.0124 m/s`
(RMS `0.0561`), while NEMO_AVM errors were
`+0.0520/+0.0697/+0.0689/+0.0333 m/s` (RMS `0.0579`). The frozen arm shifted
the upper 37 m roughly `+0.06 m/s` eastward and degraded upper-ocean RMS by
about 3%. Its apparent scalar-shear improvement was sign cancellation. Both
`top_depth` and `shear_5_26` are dominated by `u(26 m)`, so they were one
effective metric, not two. The arm's `REFUTE` was a bar artifact.

**RETRACTED: the four one-term replays' `0.0` closure values do not exonerate
production, buoyancy, dissipation, or the surface term.** The misplaced
`1e-4` post-solve clamp saturated every replay, giving
`arm_max_abs_change_at_10m=0.0`; those arms could not respond. Only the
direct factor ratios `1.0468/0.9663/1.0002/1.0000` remain legitimate evidence.

## Live PREFIX/FIX A/B — CONFIRMED at day 10

Executor artifacts: `/tmp/dino_euc_mechanism/ab_prefix.{log,json}` and
`ab_fix.{log,json}`. Runner SHA-256
`b2de13ca7039a6954a7c3e115af249e834dc4b9be83343cc0188e56893d12845`
was byte-identical for both arms; day-0 identity and all bridges were exactly
zero, and arm A reproduced the prior baseline bit-for-bit.

`PREFIX (2cb678259)`: top_depth `34.836 m`, shear `0.017060`; `FIX
(26d706b14)`: top_depth `33.103 m`, shear `0.023512`; NEMO control
`32.744 m / 0.024218`.

The pre-registered physics-review bars, registered before scoring, read:

| bar | result | verdict |
|---|---|---|
| P1 top-four-level velocity-error RMS | `FIX/PREFIX = 0.00614/0.05612 = 0.109` | **CONFIRM** (`<=0.70`) |
| P2 shear-error ratio | `0.099` | **CONFIRM** (`<=0.50`) |
| P3 per-level errors | all improved; same-sign `+0.0074/-0.0053/-0.0074/-0.0036` | **CONFIRM** |
| P4 crossing | `33.103 m`; no overshoot (frozen arm overshot to `28.27 m`) | **CONFIRM** |
| persistence at day 90 | shear ratio PREFIX/NEMO `0.7106` (instrument check reproduces recorded ~0.70); FIX/NEMO `0.9731`, inside registered `[0.85,1.20]` | **CONFIRM** |
| global do-no-harm | both arms `CERTIFIED`, exit 0, `PASS 5 | FAIL 0` at level 5x; every PREFIX-to-FIX move is 2--5 orders inside tolerance | **CONFIRM** |

Overall live label: **`CONFIRMED_DAY10_DAY90_DO_NO_HARM`**.

The persistence extraction is the regional-audit probe imported at
`91153f2aa`, applied to equator-row wet-zonal-mean `u` at 5--26 m against NEMO
`RUN_VERDICT360_M0`, `kt=8640`. Its preregistered bar is CONFIRM in
`[0.85,1.20]` and approximately `0.70` REFUTE; PREFIX=`0.7106` validates the
instrument and FIX=`0.9731` confirms persistence.

For do-no-harm, FIX versus PREFIX changes are: ACC `3.82e-3 Sv` against a
`4.55e-1` threshold; upper contrast `2e-6` against `5.5e-4`; deep contrast
approximately `1e-9` against `2.25e-4`; S-band sigma maximum approximately
`1e-6` against `4.75e-4`; and S-band sigma mean below `1e-6` against
`4.75e-4`. FIX ACC=`64.9876 Sv`, PREFIX=`64.9838 Sv`, and NEMO=`65.3692 Sv`,
so the fix moves ACC toward NEMO. The gate self-test also passes: a synthetic
violation fails all five metrics.

Executor provenance: clean worktrees with `dirty_tracked_files=0`; identical
`kamm_twin_90d.py` harness; 2,880 stable, finite steps per arm; day-0 gate
maxima all zero; ladder=`both`, ladder SHA-256
`9536f62732ab8823b2899d26ae0c4582cca09e4b2f49d9e17ff1234f88584ff9`;
start=`bridged`; control dtype=`float64`. Standard invocation:
`kamm_twin_90d.py nemo_dino_kamm_mlf <out>.npz --days 90 --save-3d --bridge-before`.
Artifacts are `/tmp/dino_euc_mechanism/twin90_{prefix,fix}/` and
`gate_{prefix,fix}.log`.

## Why this was the discriminator

The regional audit found the following stable EUC signal at days 90, 180,
270, and 360:

| day | legoESM shear | NEMO shear | ratio | legoESM core/top (m) | NEMO core/top (m) |
|---:|---:|---:|---:|---:|---:|
| 90 | 0.01548 | 0.02178 | 0.711 | 34.9 | 32.0 |
| 180 | 0.01460 | 0.02147 | 0.680 | 34.5 | 31.1 |
| 270 | 0.01448 | 0.02029 | 0.714 | 33.1 | 30.1 |
| 360 | 0.01427 | 0.01983 | 0.719 | 31.6 | 28.4 |

Its prescribed matched-state test was to feed the shared state into the
shipped closure, compare the resulting coefficient profile with NEMO's dumped
coefficient, then replace only momentum viscosity in a short arm.  The audit's
wet-zonal-mean zero crossing and two-level shear definitions are retained for
the planned response readout.  Thickness weighting here uses the W-interface
control thickness `0.5*(e3t_0[k]+e3t_0[k+1])`.

## Pre-registration and controls

The original bars were committed before the first coefficient score in
`f3a9f7ae26bee72dbe3656c890cb2db10bfe9377`:

- offline confirm: NRMS at least 0.25 plus two consecutive level ratios outside
  [0.75, 1.25] in the same direction;
- offline refute: NRMS at most 0.10 and every level ratio in [0.90, 1.10];
- otherwise unresolved;
- response confirm: at least half of the BASE-to-NEMO core-depth gap closed,
  in the correct direction, without worsening absolute shear error by more
  than 10%; response refute: at most 10% closed or motion in the wrong
  direction.

Review A3/A4 showed that the consecutive-level clause makes a surface-anchored
mechanism impossible to confirm even though the top interface sets the metric.
The reissue and decomposition attribution bar were committed in
`7d73e69ea41f2955371706f070527a706412f9e7` before the decomposition. It permits
a single 5--26 m shear-setting interface to confirm when NRMS clears 0.25 and
the viscosity sign predicts weak shear plus a deeper core.

The controls were able to fail.  A one-interface mapping error raised closure-
`avm` NRMS from 0.68865 to 1.26861; a dry-cell coefficient plant was detected
in 86 cells; an empty wet mask raised; and disabling EVD changed the realized
tracer coefficient by 99.99897 m2/s.  The bridge independently printed exact
day-0 identity for T, S, eta, u, v, all before levels, and TKE.

## Mixing profiles

All values below are equatorial wet-zonal means in m2/s.  NEMO's closure dumps
are from `tke_dump_{avm,avt}_final.bin`; realized coefficients are from the
post-EVD `dump_{avm,avt}.bin`.

| depth (m) | lego `avm_k` | NEMO `avm_k` | ratio | lego `avt_k` | NEMO `avt_k` | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 10.139 | 1.47767e-3 | 7.20307e-4 | 2.0514 | 1.32626e-3 | 5.83845e-4 | 2.2716 |
| 20.593 | 3.00487e-4 | 3.04698e-4 | 0.9862 | 2.67725e-4 | 2.75115e-4 | 0.9731 |
| 31.429 | 4.47538e-4 | 4.41291e-4 | 1.0142 | 3.78498e-4 | 3.70527e-4 | 1.0215 |

At 31.4 m, 19.23% of equatorial columns receive 100 m2/s EVD.  Including those
cells produces tiny 26--40 m realized-profile NRMS values, but those statistics
are **not evidence of closure fidelity**: the 100 m2/s EVD values dominate the
normalizer. They are retained only as a caveated diagnostic in the artifact.
At the shear-setting 10.14 m interface EVD does not erase the signal: the
realized momentum-viscosity ratio is still exactly 2.0514.

The earlier 0.999632 correlation / 1.000812 ratio offset scan used hardcoded
`RUN_GDB` year-5 data, not the day-180 lane. It is dropped from the day-180
physics argument; only the direct day-180 mapping controls are dispositive.

## TKE-closure decomposition at 10.14 m

The attribution uses 49 of 50 wet equatorial columns (98% eligible); 47 have
direct legoESM/NEMO `avm > 1.25`. Per column it closes

```
log(avm_L/avm_N) = log(Ck_L/rn_ediff_N)
                 + log(mxl_L/zmxlm_N)
                 + 0.5 log(en_L/en_N)
```

to `2.22e-16` maximum log residual and reconstructs direct avm with zero
reported relative error. The registered `DISTRIBUTED_OR_UNRESOLVED` label and
50/50 log split are retained only as historical arithmetic. They are **not an
independent decomposition result**: on the active buoyancy limb,
`zmxlm=sqrt(2 en/N²)`, so `avm ∝ zmxlm*sqrt(en) ∝ en`. Equal energy/length
log shares are therefore an algebraic identity and a consistency check.

| piece | median absolute log score | score share | geometric-mean factor | sign agreement |
|---|---:|---:|---:|---:|
| TKE energy, `sqrt(en)` | 0.384323 | 50.0000% | 1.501877 | 100% |
| mixing length, `zmxlm` | 0.384323 | 50.0000% | 1.501877 | 100% |
| coefficient/stability, `Ck/rn_ediff` | 0 | 0% | 1.000000 | 0% |

Thus the 2.05x zonal-mean excess is not carried by `rn_ediff` or a hidden
momentum stability function. The equal factors establish one TKE-energy root,
not two distributed roots.

### Single-root test

The follow-up bar was committed before reading these ratios in
`ecf236af95672bea1a18253e7824803a21fedeff`. Its verdict is
`CONFIRM_TKE_ENERGY_SINGLE_ROOT`:

| diagnostic over 47 excess columns | result |
|---|---:|
| geometric-mean `en_L/en_N` | 2.255635114 |
| median `en_L/en_N` | 2.156842650 |
| `en`-ratio IQR | [2.091538442, 2.239767452] |
| geometric-mean `(mxl_L/sqrt(en_L))/(mxl_N/sqrt(en_N))` | 0.9999999987 |
| normalized-ratio IQR | [0.9999999956, 1.0000000020] |
| columns within normalized [0.90, 1.10] | 100% |
| legoESM buoyancy-limited limb active | 100% |
| NEMO buoyancy-limited limb active | 100% |
| distance-bounded limb active, either model | 0% |

The executed NEMO branch constructs the raw length as
`zmxlm=MAX(rmxl_min,SQRT(2*en/MAX(rn2,rsmall)))` at
`cfgs/DINO/MY_SRC/zdftke.F90:757-760`, before the `nn_mxl=3` distance bounds at
`:799-812`. Because the raw buoyancy limb is active in every excess column and
`mxl/sqrt(en)` is invariant between models, the apparent 1.501877x length
factor is exactly propagation of the TKE-energy factor:
`sqrt(2.255635114)=1.501877`. The length scale carries no independent
infidelity.

**Root finding:** equatorial near-surface TKE energy is approximately 2.26x
NEMO at the 10.14 m shear-setting interface.

## Model paths verified

The DINO run is TKE, not the historical generic-card KPP path.  Its executed
namelist has `ln_zdftke=T`, `ln_zdfevd=T`, `nn_evdm=1`, `rn_evd=100`,
`rn_avm0=1.2e-4`, `rn_avt0=1.2e-5`, and `nn_etau=1`
(`RUN_VERDICT360_M0/namelist_cfg:384-401`); `ocean.output:731-782` confirms the
effective branch and defaults.

NEMO computes shear before dispatching TKE (`src/OCE/ZDF/zdfphy.F90:264-286`),
copies `avm_k/avt_k` to realized coefficients and then applies EVD
(`zdfphy.F90:311-323`).  The TKE path constructs the stress surface anchor,
buoyancy mixing length, top/down bounds, `avm`, and Prandtl-corrected `avt` at
`src/OCE/ZDF/zdftke.F90:575-724`.  EVD uses
`MIN(rn2,rn2b)<=-1e-12` and, for `nn_evdm=1`, sets momentum viscosity too
(`src/OCE/ZDF/zdfevd.F90:92-120`).

The executed instrumented DINO branch states the coefficient composition
directly at `cfgs/DINO/MY_SRC/zdftke.F90:832-837`:
`zsqen=SQRT(en)`, `zav=rn_ediff*zmxlm*zsqen`, and
`p_avm=MAX(zav,avmb)*wmask`. Its `nn_pdl` block at `:841-844` modifies only
`p_avt`; there is no independent stability-function multiplier on `p_avm`.
The realized `dump_avm`/`dump_avt` time levels are now registered from the
current `zdf_phy` call (`MY_SRC/stpmlf.F90:210`) through the exact write sites
(`MY_SRC/ldftra.F90:955-956`).

legoESM selects the NEMO TKE configuration in
`packages/ocean/legoesm/ocean/experiments/dino.py:2890-2941` and layers the
same enhanced-diffusion momentum option at `:3234-3259`.  The production call
and coefficient handoff to implicit mixing are in
`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:6347-6519`;
the closure entry is `physics/vertical_mixing/k_profiles.py:108`.  The import
receipt resolves that module inside this worktree.

## Frozen-`A_v` short-run design — superseded

This registered design is retained for audit history, but its physics verdict
is retracted as described above. Its committed runner is
`scripts/validate/ocean_fidelity/dino_1226/euc_substitution_run.py`; the runner
still calculates the historical Cz bar but prints it as superseded and emits
`SUPERSEDED_FROZEN_AVM_UNINFORMATIVE`.

Run two 10-day, 320-step arms from the same exact bridge on GPU 0:

1. `BASE`: shipped `nemo_dino_kamm_mlf`.
2. `NEMO_AVM`: at every step, replace only the non-EVD momentum viscosity
   passed to the implicit vertical momentum solve with frozen NEMO day-180
   `tke_dump_avm_final` at jk=2..36.  Keep legoESM K_v, TKE evolution, forcing,
   all other physics, and cells where its native trigger sets K_m=100 m2/s.

Required controls before scoring are bit-identical initial arms; unchanged K_v
and TKE returned by the hook; dry/wet and vertical-shift failures; a doubled-
K_m first-step tendency plant; BASE reproduction of archived day-10 control;
and read-back/re-recording of both NPZ provenance stamps.

The archived one-GPU control reports 93 s through day 10 (including JIT) and
637 s through day 360.  Budget 93 s per separately compiled arm, about 186 s
for the pair, plus approximately 50 MiB total when retaining only day-0/day-10
3-D states and daily reducers.

## TKE-equation term decomposition — executed

The offline pass used the same 47 excess columns and captured legoESM's actual
single TKE solve. BASE replay reproduced the captured matrix result within
`8.48e-16` relative and reconstructed final post-`etau` energy within
`7.76e-16`. The one-level-shift control worsened mean absolute log error from
0.8134 to 1.6971, and a 2x production plant changed energy by
`4.56e-4 m2/s2`, proving the replay path can fail.

The four direct input-factor ratios are legitimate measurements, but their
one-term replay responses are saturated and cannot attribute or exonerate:

| candidate (causal direction) | geometric-mean factor | replay label |
|---|---:|---|
| production `sh2_L/sh2_N` | 1.046834 | `SATURATED_NOT_EVIDENCE` |
| buoyancy sink `(avt_N rn2_N)/(K_H,L N2_L)` | 0.966319 | `SATURATED_NOT_EVIDENCE` |
| dissipation `dissl_N/dissl_L` | 1.000199 | `SATURATED_NOT_EVIDENCE` |
| held surface energy `en_sfc,L/en_sfc,N` | 1.000000 | `SATURATED_NOT_EVIDENCE` |

For every row, the misplaced `1e-4` clamp held the result fixed and
`arm_max_abs_change_at_10m=0.0`. The formerly printed `0% closure` and
`0% toward NEMO` values are retracted as evidence.

The surface receipts are exact, not merely close: analytical DINO `taum`, held
surface `en`, and surface `zmxlm` ratios are 1.000000 in all 47 columns; both
models reconstruct `en_sfc=max(1e-4,67.83*taum/1026)` and NEMO surface `avm`
with zero reported relative error. Thus the surface coefficient is not a
one-line coefficient mismatch.

The registered four-input result was `UNRESOLVED_TKE_EQUATION_OWNER` because
the response arms were saturated. The preserved response ledger instead
exposes the structural boundary-placement error:
legoESM's matrix energy is exactly `1e-4` in every excess column, whereas
NEMO's solved `jk=2` energy ranges from `2.2275e-5` to `4.9311e-5`. The
post-solve `etau` increment is only `1.0324e-13`, so it cannot explain the
gap.

The preregistered residual arm is
`CONFIRM_MISPLACED_SURFACE_MIN_CLAMP`: BASE is pinned in 100% of columns,
NEMO is below `1e-4` in 100%, lowering only the replay clamp to `1e-6` closes
94.3549% of the median log gap, and all 47 columns move toward NEMO. The arm's
geometric-mean energy ratio is 1.054275.

NEMO makes the separation explicit. The executed surface BC is
`en(ji,jj,1) = MAX( rn_emin0, zbbrau * taum(ji,jj) )` at
`cfgs/DINO/MY_SRC/zdftke.F90:361`. After solving `jk=2..jpkm1`, it applies
`MAX(en,rn_emin)` only over those interior rows at `:561-565`. Pre-fix
legoESM first applied `tke_background` and then unconditionally reset
`e_new[...,0]` to at least `tke_surface_min` at
`packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:1377-1386`, even
though the `nemo_z0` branch has already created a distinct virtual surface
row.

### Faithful fix built and verified

Commit `d86d97d496463b9317764426573af958fe5f20b9` guards the
`tke_surface_min` clamp with `surface_bc_level == "interior_pinned"`. Under
`nemo_z0`, solved interface 0 now retains only `tke_background` (`rn_emin`);
the separate virtual z=0 Dirichlet row consumes `cfg.tke_surface_min` as its
`rn_emin0` floor. Follow-up commit
`b127bc29ae88b74ba523eb5f2031f2117e713c47` routes that field into the row;
previously the row read module constant `_NEMO_TKE_EMIN0`, making the knob a
silent no-op under `nemo_z0`. A coupled-solve regression proves changing the
field changes the interior answer. This matches the executed DINO source at
`cfgs/DINO/MY_SRC/zdftke.F90:361` for the surface and `:564-565` for solved
interior rows.

The shipped-card reachability audit resolves every DINO recipe as follows:

| recipe | mixing scheme | surface-BC layout | behavior changed? |
|---|---|---|---|
| `legoesm_default` | KPP | `interior_pinned` | no |
| `mitgcm` | KPP | `interior_pinned` | no |
| `nemo_dino_kamm` | TKE | `nemo_z0` | **yes** |
| `nemo_dino_kamm_mlf` | TKE | `nemo_z0` | **yes** |
| `nemo_paper` | TKE | `interior_pinned` | no |
| `oceananigans` | CATKE | `interior_pinned` | no |
| `veros` | TKE | `interior_pinned` | no |
| `audit_orca1_tke_card.py` | auditor expectation, not a shipped card | expects `nemo_z0` | expectation follows corrected behavior |

Thus `nemo_z0` is oracle-card-only among shipped cards: the direct oracle card
and its MLF inheritance are the only affected recipes. Generic callers and
audit scripts that explicitly select `nemo_z0` also receive the correction.
All non-`nemo_z0` paths retain the old operation byte-for-byte and are pinned
by exact-array tests for both `None` and explicit surface-Dirichlet inputs.

The hand-computed two-interface regression was red before the fix: with zero
production, buoyancy, diffusion, and old energy `1e-6`, the legacy branch
returned `[1e-4, 1e-6]` instead of the NEMO-faithful `[1e-6, 1e-6]`. Round-6
review reproduced **2 failures** at `2cb678259`; the prior report's “1 failed”
receipt is retracted.

Exact clean-process test ledger:

- **52 passed in 37.95s** — `env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH="$PWD/packages/ocean:$PWD/packages/core:$PWD/src:${PYTHONPATH:-}" /home/dbalwada/legoESM/.venv/bin/python -m pytest -q tests/ocean/unit/test_tke_nemo_terms.py`
- **76 passed in 129.34s** — `env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH="$PWD/packages/ocean:$PWD/packages/core:$PWD/src:${PYTHONPATH:-}" /home/dbalwada/legoESM/.venv/bin/python -m pytest -q tests/ocean/unit/test_tke_nemo_identity.py tests/ocean/unit/test_tke_prognostic.py`
- **57 passed in 110.32s** — `env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH="$PWD/packages/ocean:$PWD/packages/core:$PWD/src:${PYTHONPATH:-}" /home/dbalwada/legoESM/.venv/bin/python -m pytest -q tests/ocean/unit/test_tke_integration.py tests/ocean/unit/test_combined_pipeline_tke_evd_gate.py tests/ocean/unit/test_implicit_vertical_mixing.py tests/ocean/unit/test_tke_post_mixing.py tests/ocean/unit/test_tke_dry_wmask.py tests/ocean/unit/test_tke_n2_before_advection.py`
- **36 passed, 9 warnings in 43.05s** — `env -u JAX_ENABLE_X64 CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONPATH="$PWD/packages/ocean:$PWD/packages/core:$PWD/src:${PYTHONPATH:-}" /home/dbalwada/legoESM/.venv/bin/python -m pytest -q tests/ocean/unit/test_mpas_tke.py`

For audit completeness, **34 passed, 2 failed, 9 warnings in 39.41s** —
`env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH="$PWD/packages/ocean:$PWD/packages/core:$PWD/src:${PYTHONPATH:-}" /home/dbalwada/legoESM/.venv/bin/python -m pytest -q tests/ocean/unit/test_mpas_tke.py`. Those failures are the known float64-input/float32-policy scan mismatch, not a passing receipt.

The committed verifier calls the fixed production closure on the exact shared
day-180 state. It first reconstructs the frozen 47-column cohort and recovers
the prior energy ratio `2.255635114`, then scores the real fixed result:

| real fixed quantity at 10.138751 m | geometric mean | IQR | min--max |
|---|---:|---:|---:|
| `en_L/en_N` | 1.054275004 | [1.034854588, 1.056921536] | 1.024404728--1.172218662 |
| `mxl_L/mxl_N` | 1.026778945 | [1.017278032, 1.028066522] | 1.012128809--1.082690471 |
| `avm_L/avm_N` | 1.054275003 | [1.034854591, 1.056921531] | 1.024404727--1.172218659 |

Verdict: `CONFIRM_REAL_FIX_MATCHES_OFFLINE_REPLAY`. Energy moves toward NEMO
in 47/47 columns. The complete per-column receipt is stored under
`faithful_surface_floor_fix.per_column` in the JSON artifact; its `x=3..49`
energy ratios are all reported there (range 1.024405--1.172219).

The remaining `1.054275` energy ratio has a plausible owner rather than being
left unattributed. Direct shear production is `sh2_L/sh2_N=1.046834`; the
closure self-limitation `en ∝ P^(2/3)` predicts approximately `1.031`, close
to the measured residual. **Label: `PLAUSIBLE_MOMENTUM_SHEAR_RESIDUAL`.** The
remaining mismatch is the momentum/shear field difference, itself plausibly
downstream of the same prior over-mixing.

The day-10 executor A/B confirms the registered chain: fixing the clamp drives
the near-surface closure toward NEMO, strengthens 5--26 m shear from 0.017060
to 0.023512 s⁻¹, and shoals the crossing from 34.836 to 33.103 m without
overshooting the 32.744 m control. The day-90 shear ratio remains near NEMO
(`0.9731`) and both 5x acceptance gates certify `PASS 5 | FAIL 0`, completing
the persistence and global do-no-harm confirmations.

## Provenance

The machine-readable receipt is `dino_euc_mechanism_artifact.json`.  Key input
SHA-256 values are:

- shared restart: `0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e`;
- NEMO seqdump executable: `545b383996e644591492e0fc7c82663a2d47272e2c7172ad8bbc58e5cd7d5239`;
- closure `avm`: `1240ccb86e674edd1309831e2292a244aaabc3bca2f245b530fd3b88b59c275f`;
- realized `avm`: `ad8104d352838ff36c1c09cb3b21231eacabeb00ce7bd2b5477cc40ead4faed0`;
- mesh mask: `3285fc4af36854a38b4e6f7985ab0372b95424398750a23b628935da02f72622`.
- NEMO final TKE energy: `b7a28c78a10dc3ed15b424197f7139d53c42cdfea6fe4191d96796b635722565`;
- NEMO final mixing length: `60ed331574d813885fb29399b2dbf041a6913ea965d45c09e2890eef8b3ef830`.
- NEMO shear production: `2c18e0926b2b81c07ba3c5c371067d051ffea2d2fad0cc75c444af1654746f53`;
- NEMO carried dissipation: `facba1529b41577cf12b5d4cf5f43dc0ec9fed63ec4b073de17748204e8e15df`;
- NEMO surface `utau`: `9c6943a02c5221feeb5b56d023418e69860d31d5ea0ded51437e7886811641f8`.

The scorer hash-pins and reuses the committed bridge/dump reducers.  It writes
the receipt, reads it back, and rejects a provenance mismatch.
The faithful-fix receipt was regenerated from clean committed HEAD
`b9946ccf93bd2f51f05f2c14dbb8cad0db3b7939` on the named campaign branch and
records `dirty:false`. It SHA-256-pins the production fix, regression test,
verifier, shared restart, NEMO energy/mixing fields, mesh, and every imported
reducer.

Repository note: no `AGENTS.md` exists in this checkout, its tracked tree,
`.agents`, `.codex`, `/home/dbalwada/legoESM`, `/home/dbalwada`, or `/tmp`.
That missing instruction source is recorded rather than silently substituted;
the explicit campaign instructions in the task were followed as binding.
