# Lane 3b rung 3.5b: SI3 ice-side bulk-flux receipt

Date: 2026-09-04

Tracker: `climate-federation/legoESM#1699`

Implementation commit: `96edcff2ee5`
Verdict: **CONFIRMED AT-BAR for the registered ORCA1 ice-side arm**

## Scope and pre-implementation search

This rung covers the accepted C1D_OMIP_L3 oracle's `blk_ice_1`, `ice_alb`,
`blk_ice_2`, and `ice_flx_other` boundaries for all 8,760 hourly steps.  The
mandatory search found the shared scheme validator and numerical bulk machinery
in `packages/core/legoesm/core/bulk_flux.py`, the existing ice dispatcher in
`packages/ice/legoesm/ice/sea_ice.py`, bulk configuration in
`packages/ice/legoesm/ice/config.py`, and existing saturation helpers in
`packages/core/legoesm/thermo.py`.  No
`packages/coupler/legoesm/coupler/bulk_flux.py` exists in this checkout; coupler
callers import the core implementation.  No existing SI3 `ice_alb`, Goff-ice
derivative, or `ice_flx_other` implementation was found.  The implementation
therefore extends those existing files.  It adds no second bulk module.

The public selector is `nemo_si3_constant`.  Validation makes only the
ORCA1-resolved `jpl=1`, no-pond, NEMO-constant identity constructible.  Alternate
formula arms remain private test hooks.

## Resolved oracle and source

ORCA1 selects NCAR for the ocean side and constant air--ice coefficients at
`/data/abyssal/dbalwada/ORCA1-omip/EXPREF/namelist_cfg:129-142`:

| resolved item | value printed by accepted `ocean.output` | executing source |
|---|---:|---|
| ocean bulk | NCAR | `sbcblk.F90:300-329` |
| ice bulk | constant | `sbcblk.F90:333-350,1097-1107` |
| `Cd_ice=Ce_ice=Ch_ice` | `1.0e-3` | `sbcblk.F90:1100-1102` |
| snow cover / blowing | `nn_snwfra=2`, `rn_snwblow=.66` | `icevar.F90:1580-1585,1619-1628` |
| flux distribution | `nn_flxdist=-1` | `icesbc.F90:176-180` |
| transmission | `nn_qtrice=0` | `sbcblk.F90:1322-1346` |
| conduction emulator / ponds | false / false | `icesbc.F90:181-190`; `icealb.F90:141-146` |

ORCA1 does not set `ln_ECMWF`; false resolves from the shared reference and the
exclusive NCAR choice.  The unmodified shipped C1D deck instead selects ECMWF
and `1.4e-3` (`cfgs/C1D/EXP_SASICE/namelist_cfg:96-117`).  That shipped arm is
**UNCERTIFIED** by this rung.  The ocean-side NCAR producer is also
**UNCERTIFIED**; only its registered outputs when consumed by the ice-side path
are checked here.

The source read before implementation was the untouched NEMO 5.0.2 tree:

- stress, coefficient selection, and Exner inputs: `sbcblk.F90:1048-1177`;
- radiative/turbulent/mass/precipitation/transmission fluxes:
  `sbcblk.F90:1180-1395`;
- Goff ice saturation and derivative: `sbc_phy.F90:665-711,727-790`;
- albedo and snow fraction: `icealb.F90:124-185`,
  `icevar.F90:1564-1601`;
- ice/ocean and lead fluxes: `icesbc.F90:310-437`.

### Corrected active `ice_flx_other` branch

The initial preregistration said "no-dynamics branch."  That statement is
retracted.  C1D suppresses the dynamics timestep with `ln_c1d` at
`icestp.F90:139-140`, but the accepted namelist retains `ln_icedyn=.true.`.
Thus `ice_flx_other` executes the ice-relative-velocity branch at
`icesbc.F90:322-341`; it does not execute the atmospheric-stress fallback at
`:342-346`.  The operand stream registers the ice and ocean velocity stencil,
`drag_io`, and all selecting flags.  The legoESM transcription covers the
executing branch only.

## Oracle provenance and immutable inputs

Official archive:
`https://gws-access.jasmin.ac.uk/public/nemo-vol1/sette_inputs/r5.0.0/C1D_v5.0.0.tar.gz`

| artifact | digest |
|---|---|
| `C1D_v5.0.0.tar.gz` | MD5 `9456e6a0a84d40630ad1804fd4061caf` |
| `C1D_v5.0.0.tar.gz` | SHA-256 `54a2ceefd9126e180676e68eaa28ded85cc3b93ea3f0dda0fb964a035a4fc382` |
| `ERA5_NorthGreenland_surface_84N_-36E_1h_y2018.nc` | SHA-256 `e5ec49445d2569019c45dec24255b9c7daf050079444b0e6e6d86a5b82317afe` |
| resolved `namelist_cfg` | SHA-256 `6151c0fdd2431d07c7897d5852846a620edd58c55255f11b7fb3569803b5342c` |
| resolved `namelist_ice_cfg` | SHA-256 `da7b4fc5865edf6a6a912d6316a51f6b845aaf7e8b87e9da121c6f0a46278033` |

Both archive hashes were recomputed in this round.  No synthetic forcing was
used.

## WRITE-only instrument and time registry

Run root:
`/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_sasice_bulk_phase1`

Build source copy:
`/data/abyssal/dbalwada/nemo-testcases-l3/nemo502_si3bulk_phase1_src`

The config-local writer records a 16-byte magic, version, step, stage, value
count, fp bits, and fp64 payload.  It never assigns a model field.  Exposure of
private `sbcblk` operands is instrumentation-only.

| frame | values | registered time level | source boundary |
|---|---:|---|---|
| `POST_BLK_ICE_1` | 17 | current step, after stress and before halo link | `icesbc.F90:83-108` |
| `POST_BLK_ICE_2` | 50 | current step, after albedo/air--ice flux, before `ice_flx_other` | `icesbc.F90:149-194` |
| `POST_ICE_FLX_OTHER` | 39 | current step, immediately before `ice_thd` | `icesbc.F90:194-199`; `icestp.F90:182-206` |

The second frame includes five first-owner operands internal to `blk_ice_2`:
`q_sat`, surface Exner temperature, longwave, sensible heat, and longwave
derivative.  This distinguished a real predicate difference from cancellation.

Instrument validation:

- 26,280 frames, exactly three per step; 8,374,560 bytes;
- bulk stream SHA-256
  `57868f3212646bdf6b0c4add0f48701c0082718331bc76a153050c6d1d44dfe9`;
- stabilized exchange stream unchanged at
  `091395cf604e83d88fbf458c4ef76ac3df2d5a65dac9cdc502e224cc1d1af2e4`;
- thermodynamics stream unchanged at
  `7fc9df2707a85581075e3c69b26784155151a55640a5693eb32c34fb710ea49b`;
- ZDF input stream unchanged at
  `cd1b15c821f19442a840e99c067c640e5146b754fc137a2c81e88856d6ea7efd`;
- `ocean.output` unchanged at
  `3e47d39f061ca1f9fa111a46b01bb3d1dbcfa10be73422fced0abc9e4af25430`;
- ocean restart unchanged at
  `84ed40c5e5d46f9830f4c203b3e3a79f6dfffaf142dc4c44347648e2cd9f5265`;
- ice restart unchanged at
  `b61cb8443e14b3f0868ef125f621c0b1748aff7bc827dfab0d4044fc3f773b4e`.

The direct CPU run completed step 8760 and wrote both final restarts.  It used
`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=1`, and no MPI launcher.

## Gate result

The committed gate JSON is
`nemo_testcases_l3thd_bulk_phase1_gate.json`, SHA-256
`12463d58a47dbf6579c889179e298085c2ad0c834e029a07738480f8eef7a5ff`.
It sets the precision policy to fp64, prints NumPy `float64`, JAX x64 true, and
CPU backend.

The bar is pointwise
`abs(legoESM-NEMO)/max(abs(NEMO),1) <= 1e-15`.  Across 26 registered outputs
and 8,760 steps, all 227,760 rows are AT-BAR and zero are over bar.  The largest
normalised row is step 861 `POST_BLK_ICE_1.utau_ice`,
`2.220446049250313e-16`.  Eighteen fields whose bulk value remains unchanged to
the end writer have an additional 157,680 exact input-legitimacy comparisons;
their maximum difference is zero.

This is a measured boundary verdict.  It is not a certification of the
uncertified ECMWF or ocean-NCAR arms and not a coupled ocean claim.

### First owner and private arm

Before the fix, three melt-season steps carried positive minimum-subnormal
snow.  NEMO's IEEE scalar `h_snow == 0` predicate treated them as snow-present;
XLA's floating comparison flushed them to zero and selected melting bare-ice
albedo.  The private ordinary-comparison arm produces 35 over-bar rows beginning
at step 5842.  The unbranched identity uses a bit-preserving zero test; it
returns the gate to zero over-bar rows.  This is **CONFIRMED** by the registered
operand and one-variable arm.  Other post-hoc causal stories are not asserted.

## Coverage register

The machine-readable JSON dispositions all 36 exchange fields.  Twenty-three are
`VERIFIED` either as a computed output or a registered bulk input.  Thirteen are
`WAIVED` with an owner: thermodynamic state/fluxes, ocean-side NCAR diagnostics,
inactive implicit-drag storage, or fields overwritten after this boundary.
Nothing is absent.  In particular, end-of-step `alb_ice` is recomputed after
thermodynamics and end `utau/vtau` pass through `ice_update_tau`; the gate does
not compare those different time levels.

## Controls and tests

All seven CLI plants exited 1: `blk_ice_1`, `ice_alb`, `blk_ice_2`,
`ice_flx_other`, `stream_hash`, `coverage`, and `selector`.  The first four
perturb nonzero registered outputs; the remaining three prove fail-closed
provenance, inventory, and identity selection.

Targeted test command ended with:

```text
tests/ocean/fidelity/test_nemo_si3_bulk_flux_gate.py ....... [100%]
7 passed
```

The pre-existing SI3 phase-2 suite ended with `13 passed`.  The touched-file
constants/inline-coefficient ratchets ended with `9 passed, 1 skipped`.
The broad ratchet command ended with `12 failed, 3627 passed, 2 skipped`: one
failure was the new test's literal freezing point and was corrected before the
touched-file rerun; the remaining eleven are pre-existing missing ML/tools
discovery roots and unrelated FV3/DINO/ocean-TKE sites.  No broad-suite green
claim is made.

## Committed source hashes at `96edcff2ee5`

| file | SHA-256 |
|---|---|
| config-local `sbcblk.F90` | `087e551a49e5bb0aefe93e80f1bca817365b1f68d84dc59b4178eff87c010cea` |
| config-local `icesbc.F90` | `75cdaaeec423f27c79983730828cb6a9978f1d0dbbfac9a671a7e083a95ea74e` |
| `nemo_si3_bulk_flux_gate.py` | `1adb464da63d0264fb24221e4a358982260a4e114afa3f8319f621bc75da0985` |
| gate tests | `f11e5f099369f5373fedbc67fee6ada40fc4fbf3b5977c7730e193b25b9a8326` |
| `constants.py` | `ac6c447e28fbcff5b0d37386e69328315088809a0b13cb7b35a2c9173793a1ce` |
| `thermo.py` | `41eafe070c883cce1504117082b3545338308cb412e090a24782829ba8e46407` |
| core `bulk_flux.py` | `981f294faca78f6fa78ffea50712258fe546df22ae7bb480e3533cd126f5daff` |
| ice `config.py` | `0342b881062ae111c881b0d3362b72887445cade8d740ed0bbb9d5185ac5578a` |
| `c1d_omip_l3.py` | `38c73e14d94392776529c7e580176da80f5b1d1ca7b3818954362a83032144da` |
| `sea_ice.py` | `fc13c49775ec70dc184ef17b72390fef34a132deaa15e39a955260494fff3d06` |

## Debt and boundaries

| item | status |
|---|---|
| shipped C1D ECMWF/1.4e-3 ice arm | **UNCERTIFIED** |
| ocean-side NCAR bulk producer | **DEBT — not certified by this ice-side rung** |
| coupled `sbc_ssm`, EOS freezing, flux/tau update, ocean response | **DEFERRED to reviewed rung 3.6 design** |
| closed C1D thermodynamic-column residuals | **MIXED DEBT**, retained in phase-6 receipt |

## Choices

| choice | state | disposition |
|---|---|---|
| Diagnose/stabilize exchange stream before bulk work | ASKED | Completed first. |
| ORCA1 constant ice arm, fp64, CPU, hourly full year, 1e-15 | ASKED | Executed as directed. |
| Config-local WRITE-only operand stream | ASKED | Three registered frames per step. |
| Preserve NEMO's positive-subnormal snow predicate | ASKED | Required by the selected SI3 identity; private ablation measured. |
| Correct the falsely preregistered `ice_flx_other` branch | ASKED | Oracle-executing source controls scope. |
| Certify shipped ECMWF/1.4e-3 | UNASKED | Not done. |
| Certify ocean-side NCAR | UNASKED | Not done. |
| Add a second bulk module or public mixture hooks | UNASKED | Not done. |
| Modify the shipped NEMO tree | UNASKED | Not done. |

## Flagged for future deletion

Nothing was deleted.  Intermediate `bulk_phase1.stdout`,
`bulk_phase1_v2.stdout`, their stderr files, and the pre-final bulk streams in
the copied run root are stale development artifacts **FLAGGED FOR FUTURE
DELETION**; they remain retained.  Historical drift streams and all earlier run
roots also remain retained.
