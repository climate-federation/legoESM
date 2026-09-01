# Lane 1 OVERFLOW-zps non-finite investigation receipt

Status: **CONFIRMED OUTSIDE; initiating owner remains UNMEASURED.**  NEMO 5.0.2
completes the pinned 6,120-step `key_qco + key_RK3` case; the certified legoESM
card first becomes non-finite after completed step 2,877 in fp64 (2,879 in
fp32).  No damping, clipping, diffusion, limiter, or card selector was added.

## Executive finding

The new every-step comparison separates initiation from terminal amplification.
The common state is exact at kt=1.  Instantaneous U and SSH first leave NEMO's
roundoff-padded wet range at kt=2 and become gross at kt=3, at the developing
front near x=21 km and z=490 m (U).  T remains inside NEMO's global wet range
through kt=60.  By kt=2,601 the largest instantaneous U error is 17.9033 m/s at
x=41 km, z=1,250 m and the SSH error is 3.82650 m near x=54.5 km; the dynamics
are already far outside NEMO before the terminal tracer event.

At completed steps 2,865--2,876, maximum one-step T increments grow from
1.88257 K to 314,761 K with raw successive ratios
`1.04, 1.11, 1.13, 1.43, 1.89, 1.12, 2.32, 1.72, 3.10, 2.37, 2.63, 574`.
The maxima migrate along the slope around x=41.5--43 km and z=850--1,250 m.
This is a late amplifying mode, not the earlier polynomial trajectory-debt
signature.  T/S/volume content remain closed to `5.84e-16`, `3.29e-16`, and
`1.80e-16` relative respectively, classifying the event as redistribution,
not an unbalanced source.

The same-input scale arm at completed step 2,875 is decisive but limited:
zeroing only tracer vertical transport removes 99.961% of the next T increment
at x=41.5 km, z=970 m, while changing U and SSH by exactly zero in that step.
Thus the current explicit vertical tracer flux is **PLAUSIBLE terminal T
amplifier**, but cannot be the initiating U/SSH owner.  NEMO's complete
adaptive-implicit RK3 package remains the highest source-attested mismatch; a
faithful unbranched implementation is required before it can receive a
`CONFIRMED` root-owner label.

## Rule 0: executed source and configuration

The matched NEMO rerun changes only `cn_exp`, `nn_itend/nn_stock`, and the
write-only dump condition.  Its `mesh_mask.nc` byte-matches the certified mesh
(`4692b893...`).  The resolved run selects adaptive implicit vertical
advection, constant mixing, `rn_avm0=1e-4`, and `rn_avt0=0`; Richardson, TKE,
GLS, enhanced-diffusion convection, and all other specific convection closures
are off (`overflow_zps/namelist_cfg:123-140`; matched `ocean.output:630-660`).
`zdfphy.F90:193-213` consequently dispatches no specific convection scheme.
Missing stabilizing mixing/convection is **SOURCE-EXONERATED**.

NEMO evaluates its adaptive Wicker--Skamarock partition twice in RK3 stage 3,
for velocity and transport (`src/OCE/stprk3_stg.F90:281-303`).  The live
criterion combines horizontal and vertical Courant numbers, a bottom-up
running maximum, and the 0.8/1.1 thresholds
(`src/OCE/DYN/sshwzv.F90:696-873`).  Its explicit and implicit tracer pieces
enter FCT and the vertical tridiagonal solve
(`src/OCE/TRA/traadv_fct.F90:141-167,286-330`;
`src/OCE/TRA/trazdf.F90:207-225`).

legoESM's only implementation is the older local vertical-only momentum form
in `packages/ocean/legoesm/ocean/vertical.py:1716-1909`; it is applied after
the completed RK3 program in
`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:4996-5054`.
The tracer WS-RK3 stages receive unpartitioned vertical transport and have no
matching implicit tracer solve.  This is a **SOURCE-CONFIRMED MISMATCH**, not a
confirmed failure owner.

## Comparator frame, inventory, and dtypes

The committed probe compares NEMO's instantaneous Nbb step-entry fields at
`kt` against legoESM's instantaneous prognostic fields after `kt-1` completed
steps.  T/S/SSH use T centres, U uses C-grid U faces, and V has no active
meridional face in this three-row closed tank.  Every reduction is elementwise
L-infinity on the phase-3 gate's certified common wet mask; no depth or
substep-time averaging is applied.  The exact kt=1 control checks 17,000 T/S
cells, 16,900 U faces, and 200 SSH cells bit-for-bit.

The legoESM baseline ran on CPU with storage, compute, accumulation, control,
all five state arrays, and all five coordinate arrays recorded as fp64.  The
NEMO rerun also used CPU and was invoked directly, without `mpirun`.  Its 278
new dumps cover kt=2,601--2,878.  The scorer never calls a `nan*` reduction:
non-finite rows are inventoried explicitly and make the JSON verdict red.

## First-divergence and approach ledger

| matched state | measurement | disposition |
|---|---|---|
| kt=1 | T/S/U/SSH exact in the certified frame | VERIFIED |
| kt=2 | U range excess `8.45e-7 m/s`; SSH low-range excess `1.24e-7 m` | FIRST OUTSIDE |
| kt=3 | U and SSH relative range excursions exceed `1e-6` | FIRST GROSS |
| kt=60 | T still within NEMO's global wet range | VERIFIED BOUND |
| kt=2,601 | T/U/SSH normalized L-inf `0.3933/3.1491/2.2768` | OUTSIDE |
| completed 2,870 | legoESM T range first exceeds 20 C in the approach window (`20.8693 C`) | OUTSIDE |
| completed 2,871 | T range expands to `4.3978..25.8267 C` | AMPLIFYING |
| completed 2,876 | T increment `314,761 K`; U/SSH increments `3,392.87 m/s` / `7,992.10 m` | AMPLIFYING |
| completed 2,877 | T/S/U/V non-finite; SSH finite but `1.66e86 m` | CONFIRMED OUTSIDE |

The exact first T-only global-range transition lies between kt=61 and kt=2,601
and is **UNMEASURED**.  This does not weaken the initiating-order result: the
state as a whole first leaves NEMO at kt=2 in U/SSH, while the every-step
approach window establishes the slope-front terminal locus.

## Frozen causal arms and reconciliation

All arms are private harness hooks with reference `experimental harness
ablation; no reference model`; none is a public selector or constructible
Frankenstein configuration.

| arm | prestated discriminator | movement | label |
|---|---|---|---|
| zero tracer vertical transport from initialization | support only if failure is >=100 steps later or >3,200 and T increment halves | failure moves 2,877 -> 240 (2,637 earlier) | PLAUSIBLE; current vertical transport is necessary, full adaptive-package ownership unresolved |
| same-input vertical-transport scale at completed 2,875 | effect must be >=0.1 of observed T increment before any owner label | effect/increment `0.999609`; U and SSH effect exactly zero | PLAUSIBLE terminal T amplifier; initiating owner REFUTED |
| omit `un_adv` primary-transport time average | support only if failure moves >=100 later and kt=2,601 U/SSH errors both halve | failure unchanged at 2,877; U/SSH move `8.62e-5` / `9.54e-6` relatively | REFUTED_PRIMARY |

The primary-transport arm cites NEMO's always-live time mean at
`src/OCE/DYN/dynspg_ts.F90:509,641,843`; disabling it is localization only and
cannot be shipped.  The tracer ablation's earlier failure falls outside both
frozen terminal corridors, so it is not post-hoc promoted to confirmation.

Reconciled suspect order after the arms:

1. complete adaptive-implicit RK3 tracer **and momentum** composition --
   source-confirmed mismatch, root owner PLAUSIBLE;
2. remaining barotropic stage/substep arithmetic on partial-cell slope --
   UNMEASURED after the primary-average refutation;
3. BBL stage transport -- legoESM-side CONFIRMED live, NEMO-side PLAUSIBLE;
4. prior ~75% unowned kt=2 T term -- UNMEASURED but not scale-compatible with
   the initiating late U/SSH growth on current evidence;
5. vertical mixing/convection stabilizer -- SOURCE-EXONERATED.

The near-identical fp64/fp32 failure steps (2,877/2,879) remain **PLAUSIBLE
DETERMINISTIC STRUCTURAL DIVERGENCE**, strongly inconsistent with slow roundoff
accumulation but not promoted to `CONFIRMED`: no matched fp32 approach-window
signature was recorded in this round.

## Provenance and controls

Run root: `/data/abyssal/dbalwada/nemo-testcases-l1/stability/`.

| artifact | SHA-256 |
|---|---|
| NEMO matched `namelist_cfg` | `a98d58159ca792b49da51606eafe79f0fa32e8a284f696b11eaebeb959157e50` |
| NEMO dump binary | `4337874f60ba9c3a7c2bdcf6b7121bfff04cda2cc88ddc89e5089137ecdbccb9` |
| NEMO dump-only `stprk3.F90` | `48ae371ee0bb8730cf1c84cad501f6dc1c57715e790ecc8c1f3ae507182f6700` |
| NEMO `ocean.output` | `9edf2b400c74dd95b499071ce0f199f972f0728a278dbfa3f5e905b906516381` |
| baseline `run.json` | `4098b4b1725d361d0a66d8c7dd72802235576be95729550272133c141c14a098` |
| baseline `matched_score.json` | `0dff011511dc2c1370678168f39dd75c3b34d412d19834ce85bb640ea61ce4e0` |
| baseline `run_summary.json` | `7b1ead3d21c0107eff420415405e2815830c544bbbbd709b5204b0e533eb419d` |
| paired-step scale | `f4de0f9f13987bb9c2dfdbbfed083a82679fe7a874d78ad5d25a8f9fd64f3e54` |
| tracer-arm comparison | `0fe2732288b7c10180ec2e18b1e718af6564e643ca8378c7fa91fc3314aeea21` |
| primary-average comparison | `e623d0a0c1665590922cd36abc26cae3ed15ae96ada6e1f7900a06353ec7a52f` |

All machine artifacts stamp probe commit `c658e8d3139d`.  The baseline's model
semantics are the certified f7e044e card; the intervening model change adds only
an inert-by-default private test hook, proven by direct hook-isolation tests.
The committed controls plant a 50 C wet cell, a non-finite wet value, and a
shifted step; the gross/non-finite and time-alignment paths all go red.  The
focused CPU/fp64 run passed **13 tests**: 5 stability-probe controls plus 8
real WS-RK3 tracer-path tests:

```text
PYTHONPATH=src:packages/core:packages/grids:packages/ocean \
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest -q \
  tests/ocean/unit/test_nemo_overflow_stability_probe.py \
  tests/ocean/unit/test_nemo_ws_tracer_rk3.py
```

`py_compile` and `jq empty` also passed for the probe and all six pinned JSON
artifacts.

Independent adversarial review status is **UNMEASURED** for this round; no
shipping claim is made beyond the measured OUTSIDE finding and labeled owner
ledger.

## Resolved-physics coverage continuation (2026-09-01)

**Outcome: the tested register is exhausted without a confirmed initiating
owner.**  The file-driven coverage pass eliminates a missing NEMO convection,
momentum-advection-form, momentum-LDF, or barotropic-selector mechanism.  The
current legoESM adaptive-momentum approximation is a **PLAUSIBLE late
amplifier**: removing only that rewrite delays failure by 1,051 steps and
suppresses the baseline terminal growth, but the arm still becomes non-finite
at step 3,928 rather than completing the oracle's 6,120 steps.  BBL is
**REFUTED_PRIMARY** at the terminal event's scale.  The kt=2 U/SSH initiating
arithmetic owner therefore remains **UNMEASURED**.

### Pre-implementation search and fail-closed coverage

Searched before adding any arm: the existing resolved-namelist parser and
file-side planted-control gate in
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_oracle_gate.py`, the
certified card builder and validator in
`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py`, the canonical
barotropic filter helpers in `barotropic_common.py`, the adaptive vertical
operator in `vertical.py`, the WS-RK3 call path in
`ocean_model_latlon_cgrid.py`, and the existing private BBL hook.  Found and
reused all of them; no duplicate parser, solver, public selector, or
Frankenstein card was added.

The committed gate inventories all **108** file-side keys in the executed
`namdyn*`, `namzdf`, and `namtra*` blocks.  Exact ledger equality is required;
the planted extra file key fails as missing, and a flux-to-vector card mutation
fails before integration.  The disposition is:

| status | rows | meaning |
|---|---:|---|
| VERIFIED | 67 | live card binding or source-bound executed-off selector |
| WAIVED | 40 | dead-family operand; inventory stability only, not review |
| UNMEASURED | 1 | active `namzdf.ln_zad_aimp` arithmetic and time levels |

The matched resolved namelist SHA-256 is
`2ed353704f87a1a9fa0cb09ec33452f238cd442ccf2afa7db591acd4898d84d0`.
It has `ln_zdfevd=F`, `ln_zdfnpc=F`, and every other specific convective
closure off.  NEMO's executed dispatcher consequently selects “no specific
scheme used” (`src/OCE/ZDF/zdfphy.F90:181-197`) and the constant closure with
`rn_avm0=1e-4`, `rn_avt0=0` (`:151-179,207-215`).  Enabling EVD, convection,
or tracer diffusion would be an oracle-absent Rule-9 stabilizer and was not
armed.  Momentum is flux-form UP3 (`src/OCE/DYN/dynadv.F90:78-90`), while
`ln_dynldf_off=T` returns before viscosity arrays are allocated
(`src/OCE/LDF/ldfdyn.F90:220-239`).

The per-case external-mode composition is **VERIFIED** against the resolved
values, not a lane default: `ln_bt_fw=T`, `ln_bt_auto=T`, `nn_bt_flt=1`,
`rn_bt_alpha=0`, runtime `nn_e=3`.  NEMO's executed weight builder
(`src/OCE/DYN/dynspg_ts.F90:1041-1108,1223-1240`) and legoESM both produce
primary `[0,1,1,1]/3`, raw secondary `[3,3,2,1]`, divisor 9, and four loop
iterations.  The live zero-alpha back interpolation is the same
`0.614/0.285/0.088/0.013` arm (`:1676-1711`).  This verifies the registered
substep count/filter composition; it does not convert the still-unowned kt=2
U/SSH arithmetic residual into a match.

### Scaling-first arms and reconciliation

Both arms are private experimental harness ablations with **no reference
model**.  Public cards still run NEMO's named configuration.

| arm | frozen scale result | trajectory movement | label |
|---|---|---|---|
| suppress legoESM's current post-program adaptive momentum rewrite | same-input step-2,875 U effect `2,766.778 m/s`, ratio `0.815468` of the baseline increment | non-finite `2,877 -> 3,928`; kt=2,877 U error `621.703 -> 3.90185` (`159.336x` reduction); step-2,876 T increment `314,761 -> 0.0967489 K` (`3.25338e6x`) | **PLAUSIBLE amplifier**, not CONFIRMED |
| suppress Campin--Goosse BBL transport | effect at the registered T and U increment maxima exactly zero; global T effect only `5.733e-8` of the increment | full arm forbidden below the frozen 0.1 scale floor | **REFUTED_PRIMARY** |

Arm D's step-3,928 failure is again a slope-front redistribution event: its
last finite large increments localize near x=46--51 km and z about 530--630 m,
and volume remains closed to `1.80e-16` relative.  It cannot receive a
CONFIRMED label because it does not complete 6,120 steps.  The movement instead
shows that legoESM's current post-stage adaptive rewrite is a powerful
late-time amplifier.  NEMO does not expose a no-adaptive switch, so the arm is
not a correction and cannot ship.

The sole coverage UNMEASURED row is source-specific.  NEMO constructs its
Wicker horizontal/vertical Courant partition bottom-up at RK3 stage 3
(`src/OCE/DYN/sshwzv.F90:710-847`), then uses the explicit and implicit
transports in the resolved optimized `nn_fct_imp=1` two-step predictor
(`src/OCE/TRA/traadv.F90:220-227`;
`src/OCE/TRA/traadv_fct.F90:140-145,526-536`) and the momentum/tracer implicit
applications at their NEMO time levels.  legoESM's current local vertical-only,
post-program approximation is not that package.  The next implementation debt
is therefore the **source-exact, unbranched NEMO RK3 `ln_zad_Aimp` package**;
it must be implemented as one scheme identity, not as public micro-selectors.
Until that package runs the oracle duration and the kt=2 U/SSH term is owned,
the initiating owner remains **UNMEASURED**.

The fp64/fp32 failure steps 2,877/2,879 still support a deterministic structural
interpretation, not roundoff accumulation.  This remains labeled PLAUSIBLE,
not CONFIRMED, because no matched fp32 approach-window diagnostic was added.

### Continuation artifacts

Run root remains
`/data/abyssal/dbalwada/nemo-testcases-l1/stability/`; no new NEMO run was
needed after the coverage table.

| artifact | SHA-256 |
|---|---|
| resolved coverage JSON | `b22281a4f2389edcd29b354e663369c2d13398df7716fe5e576057b81d68bed5` |
| adaptive-momentum same-input scale | `d7ef743ae95d5f7553f5763673d53f7ec04fa0662040587872e9c9b73762e2b1` |
| adaptive-momentum arm `run.json` | `3d3a51b3a15ea96150bfa187cdcc807ee4a758e11c049b431fd0c11e113d580d` |
| adaptive-momentum matched score | `8d642cbd0385bcde27ae9baad880757e61cc92318cb03958500f30a516f60679` |
| adaptive-momentum run summary | `3162b99bf49bc155e01b1c0a110f8846bd0533b8892d49870d8415a2a6e4279f` |
| BBL same-input scale | `a87ec2fdfb5aa8b8c8805a76d6edc614bf216eabd3a56d3fbcf6804f55cfc6a0` |
| machine verdict | `cfb67081d01c5ca59bca581634600fc6414f290d1dcb9d8ff1cc82af5181e8ce` |

The coverage JSON and machine verdict stamp scorer commit `4865140afcbf`;
the full adaptive arm stamps runner commit `5e68669f5e8f`.  The focused current
breakdown is **31 tests**: 11 stability-probe/coverage/label controls plus 20
adaptive-implicit vertical-advection unit tests.  All ran CPU/fp64 with
`JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`; every state and geometry dtype in the
arm receipt is `float64`.
