# SI3 lane 3 — phase-2 receipt (rungs 3.1 and 3.2)

Issue: climate-federation/legoESM #1699

Branch: `fidelity/nemo-testcases-l3-si3dyn-codex`

Phase-2 commit ledger:

* rung-3.1 independent-review HOLD closure:
  `fd2dd6c24e6cc70ada3f1c156b631c3da64d5091`;
* rung-3.2 preregistration: `690fbdd8ee0281398fba9bf54073d5db2502478f`;
* rung-3.2 implementation and fixed gate:
  `9f9c4a4a1681f7cb88bb381e369dbbf378dec2a7`.

Oracle roots: `/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv1d/final` and
`/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv2d/final`.
Gate artifacts are the corresponding `phase2/nemo_si3_phase2_gate.json` files.

## Rung 3.1 verdict

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

* `transport.py:75-83` names the limiter/merge coefficients, the two CFL
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

* `pytest` over the phase-1 oracle gate and both phase-2 rung gates:
  **48 passed**.  The independent ice state/transport regression set is
  **33 passed, 1 skipped**.
* Focused phase-2 suite: **14 passed**; this includes complete-card JIT, reverse-mode,
  source-essential `Hbig`, restart carry, geometry/state plants, coverage
  plant, general dual-outflow ordering, mandatory Hbig indices, full
  trajectory, and loud endpoint DEBT.
* Focused rung-3.2 suite: **13 passed**; this includes complete-card JIT/grad,
  all-tracer `kt=1` and first-step exactness, y-limiter orientation,
  alternating-order plant, source-correction plant, all-80-moment restart
  carry, geometry/inventory plants, ordered frame/header provenance, exact row
  registration, selector/velocity/clock/extra-restart-key rejection, and the
  complete 485-step gate.
* The coefficient and hardcoded-constant ratchets were run together:
  **3,734 passed, 2 skipped, 7 failed**.  All seven selected checks covering
  every ratchet-rostered touched file pass.  The seven full-suite failures are
  pre-existing, unrelated paths: `gm_redi_latlon_cgrid.py`, `tke.py`, FV3
  coupling and its test, and three DINO ocean tests.  No such failure is
  hidden or attributed to this lane.
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

## Rung 3.2 — ICE_ADV2D

### Verdict and fixed gate

**Overall: UNMEASURED; numeric fidelity: DEBT.**  The unresolved candidate
`snwice_mass` and before-level contract prevents a green exit independently of
the numeric score.  The 99 x 99 oracle geometry and all 20 cold-start boundary
fields are **AT-BAR**.  The trajectory remains at the immutable pointwise
`1e-15` class through 15 completed steps; the first measured divergence is
`trajectory.post_step_00000016.szv_i_l01`.  Later accumulated errors and the
final moment endpoint remain DEBT.  These labels report the gate; no tolerance
was relaxed and no equivalence claim is made.

| registered group | rows | AT-BAR / DEBT | maximum normalized error |
|---|---:|---:|---:|
| geometry | 22 | 22 / 0 | `1.9334496211159188e-16` |
| `kt=1` entry | 20 | 20 / 0 | `2.7671398542734975e-16` |
| post-step trajectory | 9,700 | 4,322 / 5,378 | `1.130489336137675e-14` (`post_step_474.v_i`) |
| all active restart moments | 80 | 10 / 70 | `7.703351410561056e-14` (`sxyap`) |
| **numeric total** | **9,822** | **4,374 / 5,448** | **DEBT** |

The corrected gate has 9,822 unique rows: 22 geometry rows; 20 fields at each of 486
boundaries (entry `kt=1`, 484 later entry frames, and the step-485 restart);
and all 80 active oracle moment arrays discovered in the restart.  It requires
exact set equality, not only the count, and fails closed unless the phase-1
mesh/frame inventories are exact.  The 20 boundary fields are the
16 transported tracers, the option-4 carried-but-not-transported `sv_i`
diagnostic, surface temperature, and prescribed U/V.

The committed preregistration said 6,391 rows, `nn_icesal=2`, ponds off, and 50
moments.  That was a wrong pre-implementation assumption.  The already pinned
oracle did not take those ORCA1 overlays: the phase-1 receipt and resolved
namelist show shared-reference `nn_icesal=4` and level ponds on.  Silently
keeping the smaller roster would violate the user's all-moments requirement.
The preregistration is preserved as evidence of the correction; this receipt
and the gate carry the corrected fixed count.  No result obtained under the
incorrect roster is cited.

### Source program and state

The existing `legoesm.ice.transport` module now includes the 2-D extension of
the same selectable Prather program.  It ports:

* alternating odd x-then-y / even y-then-x dispatch from
  `icedyn_adv_pra.F90:253-351`;
* the x sweep from `:499-719` and y sweep from `:722-943`, including the
  y-oriented `sy/syy` limiter at `:757-791` and all `sxy` flux/merge terms;
* Hbig, Hsnow, and nonnegative cleanup from `:946-1142` and
  `icevar.F90:726-839`; and
* the bi-periodic boundary refresh after correction from
  `icedyn_adv_pra.F90:432-479` and shipped `usrdef_nam.F90:99`.

The first implementation accidentally reused the x-oriented limiter in the y
sweep.  The full trajectory gate exposed a step-4 concentration jump.  After
using the actual `sy/syy` limiter, steps 1--15 are at bar.  A second full-gate
run exposed a pond error when the patch crossed the periodic boundary: source
corrections had been applied independently to halo cells.  Refreshing halos
from corrected physical cells, as the source orders it, removed that error.
Neither failed run is represented as a result.

The card is fp64, CPU-only, `jpl=1`, `nlay_i=nlay_s=3`, 485 steps of 1,200 s on
the pinned 3 km geometry, CFL 0.2 / one Prather cycle, with prescribed
`u_ice=v_ice=0.5 m/s`.  Thermodynamics and landfast are off.  To reproduce the
actual oracle restart, its packed prognostic state has 16 independently
advected tracers: ice/snow volume, ice/pond area, age, six energy layers, three
option-4 salt layers, pond volume, and pond-lid volume.  The five Prather
moments are restart-carried for every tracer (80 arrays).  The option-4
`sv_i` array is a separate carried diagnostic because the executed source
transports `pszv_i`, not `psv_i` (`icedyn_adv_pra.F90:232-240,368-375`).
Deriving `sv_i` from the layers would disagree with the oracle after step 1.
All fixed testcase/namelist coefficients are named in the module provenance
block at `nemo_adv2d_testcase_recipe.py:43-75`; the Prather arithmetic
coefficients remain at `transport.py:75-83`.  These exact paths pass both
ratchet scans where rostered; no `# coeff-ok` escape was added.

The restart contains ten state arrays (packed contents, carried bulk salt,
U/V, surface temperature, and five moment packs), its completed-step clock,
and the complete card-contract hash, including the ordered tracer and moment
registries.  Round-trip and split continuation are bitwise exact.  Perturbing,
dropping, retyping, or adding a key goes red; selector, prescribed-velocity,
and clock changes are also rejected.

The independent physics review found two shipped-case-inactive defects before
closure.  The neighborhood maximum now includes SI3's `epsi20` floor
(`icedyn_adv_pra.F90:1519-1534,1561-1571`); pond entry thickness and Hbig use
the source's `a_ip>epsi20`, outer surviving-ice guard, and
`v_ip/MAX(epsi20,a_ip)` (`icevar.F90:306-316` and
`icedyn_adv_pra.F90:985-995`).  An isolated-pond control has finite reverse-mode
gradient, a zero-area transported pond is rescued, and isolated snow retains
the nonzero `a_i*epsi20` floor.  The `zapneg` port also updates concentration
before testing snow volume, preserving the statement order in
`icevar.F90:759-760,807-816`.  The reviewer reran the discriminating controls
and approved these corrections.

### Non-vacuous controls and scope

The shipped Gaussian and equal x/y velocities are symmetric enough that the
two split orders nearly cancel.  The order control therefore plants an
asymmetric cross moment; flipping the parity then exceeds the same bar.  A
separate synthetic y-sweep test proves `sy/syy`, not `sx/sxx`, are limited.
The Hsnow snow-load and pond-area caps are exercised on a non-binding-to-binding
synthetic transition.  Geometry, state, unaccounted mesh-array, unaccounted
frame-array, active-moment perturbation, dropped-moment, and retyped-moment
controls all go red.  The gate validates every one of the 485 ordered frame
names and headers and stamps their ordered SHA256 aggregate; a wrong `kt`
header and a duplicate comparison row each go red.  Both independent reviews
approve the final physics and gate code after their HOLD items were closed.

The complete card step passes `jax.jit` and reverse-mode differentiation with
finite, nonzero fp64 gradients.  All oracle and candidate numerical arrays
recorded by the gate are `float64`; masks are boolean.  The source-effect ocean
residual ledgers are not modeled by this uncoupled card and are not compared.

## Artifact hashes

| artifact | SHA256 |
|---|---|
| phase-2 gate JSON | `9224796a51edd63dcdbabe8365ce270d28d0ff78b25c03854e78eb3dfba7e853` |
| oracle `mesh_mask.nc` | `ba0e884eab64dd4ef659b21e6369d5999bda20e1e243c5c541ae5b93008148ad` |
| oracle `output.init_ice.nc` | `7b9affc6f958cec9be7d193fbd021da11dd95fa2b115dd76c48ef8788492cb3f` |
| oracle final ice restart | `bc49d8dd9633210a1c8f759b60c27919d48a789637dffae7ef7ee66682488dca` |
| changed `transport.py` | `8a90dede63faa5937077c7e15e756b5a49943db2fa21d6bad5e444bc63b9ea05` |
| card recipe | `df689ceeb0747a334bebce2683d92d4161b7ba3e12e4c678d214a80f02d86323` |
| phase-2 gate source | `e70306dcc8302a16877cb93894f9309e612240d915c54b061c1da78b35ee5c65` |

Rung-3.2 artifacts (gate provenance parent
`690fbdd8ee0281398fba9bf54073d5db2502478f`):

| artifact | SHA256 |
|---|---|
| phase-2 gate JSON | `ffdf6474e3c5c39b273eb571039662be850614d9b7eeb8ca0328b33ff458fbc5` |
| oracle `mesh_mask.nc` | `a74a6cd55b11084715b0e4e964f820ea9c2e222a73bd02f960edf73b58970503` |
| oracle `output.init_ice.nc` | `477b4a1c7c4907dca49323824c5af300448cdab1e3800c5b8f4b5c4b9865b389` |
| oracle final ice restart | `bbbecb7eae63ebc014f4b868cf11bde319b6b3716e5e38a92d5ff88da71f33cf` |
| ordered 485-frame SHA256 aggregate | `8f6fa12be7ca329a3520b31317b97dea60d7cdf4fed728e47f574b20f9480122` |
| `transport.py` | `8a90dede63faa5937077c7e15e756b5a49943db2fa21d6bad5e444bc63b9ea05` |
| rung-3.2 card | `4c044afac5128009db1015063b82c795970b24385b4330859d752e05c8d959f8` |
| rung-3.2 gate source | `2e6176fb8e81e44e65d4a69787967608c99718b261698fbfb54a8d25df09eaca` |
| rung-3.2 unit controls | `a0308bff7200a6e739c1c764777db8d3c62dd64f98c87472d2b7942c8a9774b8` |

## Loudly UNMEASURED / deferred

Rung 3.3; within-step x/y split states; ORCA1 option-2 salinity (the rung-3.2
oracle resolves option 4); candidate alignment of nonzero `snwice_mass` and its
before level; separation of `zapsmall` from `zapneg`; thermodynamics;
rheology; ridging/rafting; general production run-restart integration of the opt-in
card state; coupled ice--ocean comparison; and landfast L16 (OFF here,
**UNVERIFIED-deferred to lane 4**).  No legoESM claim is made for any item in
this paragraph.
