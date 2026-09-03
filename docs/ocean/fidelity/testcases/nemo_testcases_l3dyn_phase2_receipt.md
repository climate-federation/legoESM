# SI3 lane 3 — phase-2 receipt (rung 3.1 only)

Issue: climate-federation/legoESM #1699

Branch: `fidelity/nemo-testcases-l3-si3dyn-codex`

Oracle root: `/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv1d/final`
Gate artifact:
`/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv1d/phase2/nemo_si3_phase2_gate.json`

## Verdict

**Overall: DEBT.**  The source-built geometry, cold-start entry, and every
registered ordinary prognostic at all 40 available step boundaries are
**AT-BAR** under the immutable normalized pointwise `1e-15` class.  The final
Prather restart-moment endpoint is **DEBT**: five of 50 in-scope moment rows
are outside the bar.  No broader equivalence claim is made.

| registered group | rows | maximum normalized absolute error | result |
|---|---:|---:|---|
| geometry (18 coordinates/metrics/Coriolis + 4 masks) | 22 | `2.7755575615628914e-17` | AT-BAR |
| `kt=1` entry (11 packed tracers + surface temperature + U/V) | 14 | `1.5959278840984339e-16` | AT-BAR |
| post-step ordinary trajectory (13 fields, steps 1--40) | 520 | `6.315272866215738e-16` | AT-BAR |
| final restart moments (5 moments x 10 ORCA1-scope tracers/layers) | 50 | `2.32620646580274e-15` | **DEBT** |

The first outside-bar row is `restart_moment.sxa` at
`1.0062235299375275e-15`.  The other four are `sxxa`
(`2.32620646580274e-15`) and `sxxe_l01/l02/l03`
(`1.770047618885865e-15` each).  These are measured endpoint discrepancies,
not a relaxed tolerance.

Every scored floating oracle and legoESM array is `float64`; exact mask rows
are boolean.  JAX reported the CPU backend and the card verified
`PrecisionPolicy.fp64()`.

## Phase-1 review closure carried into this phase

The rung-3.2 maximum-concentration endpoint-equality predicate was retracted
before this implementation.  `ICE_ADV2D/EXPREF/README:64-65` says only that
Prather conserves maximum values relative to UM and creates side lobes; it
provides no numerical tolerance.  `tests/README.rst:181-186` documents the
case's thickness overshoot, while `icedyn_adv_pra.F90:418-420` warns that the
scheme is not perfectly bounded.  Equality of `kt=1` and the final endpoint
cannot classify an intermediate trajectory.  The phase-1 gate now reports the
full step-boundary maximum series as MEASURED-UNCLASSIFIED and documentation
conformance as **UNMEASURED**.  Its non-vacuous control constructs equal
endpoints with an intermediate excursion and stays non-green.  The focused
phase-1 suite is included in the combined result below.

## Implementation and exact source contract

No second ice model or transport module was added.  The opt-in
`si3_prather` kernel lives in the existing `legoesm.ice.transport` module;
the existing PPM and donor-cell paths and their defaults are unchanged.  It
ports the five-moment 1-D program from the shipped oracle source:

* CFL cycle selection and halo progression:
  `src/ICE/icedyn_adv_pra.F90:116-132`;
* construction of extensive transported contents: `:218-240`;
* split dispatch and post-transport recovery: `:253-375`;
* x limiter, slab extraction, donor remnant, and receiver merge: `:499-719`;
* low-concentration `Hbig` correction: `:946-1002`;
* zero-snow `Hsnow_pra` arm: `:1082-1142`;
* nonnegative cleanup call: `:418-421` and `icevar.F90`;
* zero cold-start moments: `:1361-1370`;
* five-moment restart enumeration: `:1383-1497`.

The kernel accepts a packed `(x-with-halo, y-with-halo, tracer)` extensive
state and five equally shaped prognostic moment arrays.  Its x sweep now
preserves NEMO's general donor ordering: positive slabs are computed and
removed first (`icedyn_adv_pra.F90:570-616`), then negative slabs are computed
from those updated donor boxes (`:618-664`), before the two receiver merges
(`:666-707`).  The synthetic dual-outflow control at
`test_nemo_si3_phase2_gate.py:143-188` exercises a cell exporting through both
faces and checks its content, first moment, and second moment against that
ordered scalar program.  The card rejects nonzero V, so it cannot silently
claim the unported 2-D alternating sweep.
The rung resolves two CFL subcycles.  `Hbig` uses the dynamics-entry thickness
for both subcycles because SI3 does not recover the intensive `ph_i` work field
until after the dynamics call (`icedyn_adv_pra.F90:142-160,355-367`).  Both
volume and concentration indices are now mandatory (`transport.py:916-923`),
so the public dispatch cannot silently omit `Hbig`; the omission control at
`test_nemo_si3_phase2_gate.py:124-140` raises.  A deliberate threshold ablation
is non-vacuously outside the bar by more than `1e-11` at entry `kt=16`.

## Rung-3.1 independent-review HOLD and closure

The first phase-2 receipt failed to disclose that
`tests/test_no_inline_physics_coeffs.py` rejected both new source files.  The
independent review therefore placed this rung on HOLD.  The failure was real:
Prather coefficients, CFL thresholds, `Hbig` thresholds, and fixed testcase
profile values appeared as anonymous arithmetic literals.

They now live in sourced module-level provenance blocks:

* `transport.py:77-85` names the limiter/merge coefficients, the two CFL
  thresholds, the `0.15` `Hbig` threshold, and `rn_himax=99 m`, citing
  `icedyn_adv_pra.F90:124-126,553,681,702,1000` and
  `cfgs/SHARED/namelist_ice_ref:46` (not overridden by ORCA1);
* `nemo_testcase_recipe.py:43-69` names the shipped grid, clock, ORCA1 deck,
  prescribed-velocity, initial-profile, temperature, and salinity values,
  citing `namelist_cfg`, `namelist_ice_cfg`, `usrdef_*`, `icedyn.F90`, and
  `make_initice.py` line by line.

Targeted ratchet results after the correction are **2 passed** for the two
source files under `test_no_inline_physics_coeffs.py`, and **25 passed** for
all selected touched transport/card/test paths under
`test_no_hardcoded_constants.py`.  The complete ratchets were also run.  They
retain two unrelated inline-coefficient failures (`gm_redi_latlon_cgrid.py`,
`tke.py`: 352 passed / 2 failed) and five unrelated hardcoded-constant
failures (FV3 coupling plus three existing DINO tests: 3386 passed / 5 failed /
2 skipped).  None of those seven files is changed by this dispatch.

The card holds immutable selectors separately from an explicit
`ICEAdv1DState`; all five moments are leaves of that prognostic state.  The
`icevar.F90:611-708` cleanup is included, notably its `t_su=sst_m+rt0` update
at `:695`.  `sst_m` is loaded from the phase-1 oracle's
`output.init_ice.nc:sst`; no analytic substitute is used.  A card restart
persists all nine array leaves (contents, U/V, surface temperature, five
moments) plus the clock and a SHA256 identity over every continuation selector,
mask, prescribed velocity, and empty-cell temperature.  Loading only that NPZ after
seven steps gives a bitwise-identical next step.  Dropping, retyping, or
perturbing a moment makes the continuation control red.
The general production `DynamicSeaIceState` remains 12 fields; general
production run-restart integration for this opt-in card is **UNMEASURED**.

The complete card step was compiled with `jax.jit`, and reverse-mode
differentiation returned finite, nonzero fp64 sensitivity.

## Card selector ledger — no implicit arms

| selector/value | named source |
|---|---|
| `jpl=1` | ORCA1 `EXPREF/namelist_ice_cfg:24` |
| `nlay_i=3`, `nlay_s=3` | ORCA1 cfg `:25-26` |
| `ln_dynADV1D=T` | shipped ICE_ADV1D cfg `:38` |
| `ln_adv_Pra=T`, `ln_adv_UMx=F` | ORCA1 cfg `:75-76` |
| `ln_icethd=F` | shipped ICE_ADV1D cfg `:27`; this is an advection-only rung |
| `nn_icesal=2` scope, bulk salinity carry | ORCA1 cfg `:108` |
| ponds off | ORCA1 cfg `:151-152` |
| landfast off | user §8 pick; ORCA1 cfg `:46` is deliberately not selected |
| 59 x 59, `dx=dy=4 m`, `dt=2 s`, 40 steps | shipped ICE_ADV1D case and dossier §4 |
| convergent U, V exactly zero | shipped `src/ICE/icedyn.F90:144-157` |
| concentration ramp/thickness notch, float32 input quantization | shipped `tests/ICE_ADV1D/EXPREF/make_initice.py:85-113` |
| empty-cell surface temperature | phase-1 oracle `output.init_ice.nc:sst` plus named `T_freeze`; executed assignment `icevar.F90:695` |

The phase-1 oracle retained the testcase reference's `nn_icesal=4` and ponds
on (`namelist_ice_ref:199,241`) rather than ORCA1's option 2/ponds-off rows.
Therefore its bulk `sv_i` trajectory, option-4 layer salinity moments, and pond
fields/moments cannot certify this ORCA1-limited card and remain loudly
**UNMEASURED**.  No Frankenstein comparison is made across those arms.

## Geometry, coverage, and controls

The card reuses `create_beta_plane_cgrid_geometry`: 59 x 59 T cells,
`dx=dy=4 m`, zero Coriolis, a closed one-cell rim, and the shipped two-cell
halo.  The phase-2 gate imports the phase-1 frame reader and coverage checker,
then enforces exact manifest set equality.  Inventory counts are 35 mesh
arrays, 109 restart arrays, 281 ice-namelist keys, and 325 dynamics-namelist
keys; every discovered entry retains one VERIFIED/WAIVED/UNMEASURED reason.
The candidate gate separately requires exact set equality against all 19
arrays in the phase-1 `FRAME_REGISTRY`.  Nine are scored at every boundary;
ponds and stresses have sourced inactive waivers.  The nonzero derived
`snwice_mass`/before-level carries and bulk-versus-layer salinity fields are
UNMEASURED; they are not mislabeled as inactive.

Two executed source calls are explicitly inactive on this rung.  `Hsnow_pra`
is non-binding because the measured maximum absolute snow content over all 40
steps is `0.0` (`icedyn_adv_pra.F90:1117-1129`).  Candidate contents at both
Prather subcycle boundaries have measured minimum `0.0`; applying every
volume/concentration/salt/enthalpy/age zeroing condition from
`icevar.F90:759-837` gives a measured maximum potential state change of `0.0`.
The cleanup called at `icedyn_adv_pra.F90:418-421` is therefore
WAIVED-INACTIVE here.  These are rung-local dispositions, not claims about
other cases.

All planted violations went red in real CLI runs:

| control | observed failure |
|---|---|
| unaccounted mesh array | `missing=['PLANTED_UNACCOUNTED_FILE_ARRAY']` |
| unaccounted frame array | exact-set mismatch names `PLANTED_UNACCOUNTED_FRAME_ARRAY` |
| geometry perturbation | `geometry.e1t`, normalized error `0.25` |
| entry-state perturbation | `entry.kt00000001.v_i`, normalized error `1.0` |
| active moment perturbation | `prather moment carry changed split continuation` |
| dropped moment | `ICE_ADV1D restart missing ['moment_4']` |
| retyped moment | `ICE_ADV1D restart moment_0 shape/dtype mismatch` |

The restart-loader unit control also injects nonzero `v_ice`; reconstruction
then re-runs the complete card validator and rejects the out-of-scope 2-D
state.  This selector/state control is distinct from the CLI comparison
plants above.

## Verification record

* `pytest` over the phase-1/phase-2 fidelity gates and ice transport/state
  units: **68 passed, 1 skipped**.
* Focused phase-2 suite: **14 passed**; this includes complete-card JIT, reverse-mode,
  source-essential `Hbig`, restart carry, geometry/state plants, coverage
  plant, general dual-outflow ordering, mandatory Hbig indices, full
  trajectory, and loud endpoint DEBT.
* The earlier two adversarial approvals predated this independent-review HOLD;
  this section supersedes the earlier review status rather than concealing it.
* Ruff (`F,E501,I`): clean for every changed Python file.
* `mypy --follow-imports=skip --ignore-missing-imports` reports no issues in
  the new card and phase-2 gate.  The natural `mypy --ignore-missing-imports`
  traversal including the whole existing transport module retains eight
  pre-existing `no-any-return` findings and two pre-existing optional-fold
  narrowing findings; the five new fixed-tuple findings in the Prather code
  are resolved.  No existing production branch was changed to silence them.
* The unrelated `tests/ocean/unit/test_ocean_run_restart.py` baseline is red:
  65 failures because pre-existing TKE slots `tke_avm`, `tke_avm_surface`,
  `tke_avt`, and `tke_dissl` are unclassified.  This dispatch changes neither
  ocean state nor ocean restart code.

## Artifact hashes

| artifact | SHA256 |
|---|---|
| phase-2 gate JSON | `c658fbc39272ff6848310dd707067e1518ff85856b39f444eb03cebf66b84a78` |
| oracle `mesh_mask.nc` | `ba0e884eab64dd4ef659b21e6369d5999bda20e1e243c5c541ae5b93008148ad` |
| oracle `output.init_ice.nc` | `7b9affc6f958cec9be7d193fbd021da11dd95fa2b115dd76c48ef8788492cb3f` |
| oracle final ice restart | `bc49d8dd9633210a1c8f759b60c27919d48a789637dffae7ef7ee66682488dca` |
| changed `transport.py` | `48111ec1c0058c22b966ffb127173f18414e8f50ef848a486726d19613679fd5` |
| card recipe | `df689ceeb0747a334bebce2683d92d4161b7ba3e12e4c678d214a80f02d86323` |
| phase-2 gate source | `9fa55f6876378842b60cd3d975850355f64934f92995f97bbe5f4d4425a95927` |

## Loudly UNMEASURED / deferred

Rungs 3.2 and 3.3; within-step x/y split states; ORCA1 option-2 salinity
against the phase-1 option-4 oracle; option-4 layer-salinity moments; pond
fields and moments; candidate alignment of nonzero `snwice_mass` and its
before level; separation of `zapsmall` from `zapneg`; thermodynamics;
rheology; ridging/rafting; general production run-restart integration of the opt-in
card state; coupled ice--ocean comparison; and landfast L16 (OFF here,
**UNVERIFIED-deferred to lane 4**).  No legoESM claim is made for any item in
this paragraph.
