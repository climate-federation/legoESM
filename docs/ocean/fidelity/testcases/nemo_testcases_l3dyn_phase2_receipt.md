# SI3 lane 3 — phase-2 receipt (rungs 3.1, 3.2, and 3.3)

Issue: climate-federation/legoESM #1699

Branch: `fidelity/nemo-testcases-l3-si3dyn-codex`

Phase-2 commit ledger:

* rung-3.1 independent-review HOLD closure:
  `fd2dd6c24e6cc70ada3f1c156b631c3da64d5091`;
* rung-3.2 preregistration: `690fbdd8ee0281398fba9bf54073d5db2502478f`;
* rung-3.2 implementation and fixed gate:
  `9f9c4a4a1681f7cb88bb381e369dbbf378dec2a7`;
* round-2b disclosure correction: `43658998cc9`;
* rung-3.2 written-order replay preregistration and final classification:
  `146199c9b3e` and `aa6dc122b43`;
* rung-3.3 initial design preregistration: `af53e78bd33`;
* rung-3.3 independent-review design revision: `9b369d9d082`;
* rung-3.3 C-grid aEVP implementation boundary: `54d0d5dad96c38b194ec56addd6dc051439a52e5`;
* rung-3.3 trajectory/restart preregistration: `535e1b83c5e`;
* rung-3.3 replay, full trajectory, and restart gate: `ba07bc7b202`.

Oracle roots: `/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv1d/final`,
`/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv2d/final`, and
`/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv2d_rhg/final`.
Rung-3.1/3.2 gate artifacts are the corresponding
`phase2/nemo_si3_phase2_gate.json` files.  The rung-3.3 full artifact is
`ice_adv2d_rhg/phase2/nemo_si3_phase2_rung33_trajectory.round5.json`; its earlier
partial artifact is at
`ice_adv2d_rhg/phase2/nemo_si3_phase2_rung33_partial.json`
and its A-grid preservation control is
`ice_adv2d_rhg/phase2/nemo_si3_phase2_rung33_agrid_guard.json`.

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

The pre-implementation search failed to name the existing ocean-tracer SOM at
`packages/ocean/legoesm/ocean/advection_som.py`; this is a disclosure defect,
now recorded in both phase-2 documents.  It does not supply an identical
kernel to reuse.  Ocean SOM is 3-D with nine moments beyond the mean
(`advection_som.py:35-40,92-159`); SI3 is 2-D with five moments beyond the
content.  Its limiter uniformly rescales all moments from
`|s_d|+|s_dd|<=content` (`advection_som.py:47-85`), unlike SI3's directional
clip/curvature/cross-moment program (`icedyn_adv_pra.F90:534-568,757-791`).
Its donor update removes two face fluxes simultaneously from one donor state
(`advection_som.py:200-245`), unlike SI3's positive-loss then updated-donor
negative-loss order (`icedyn_adv_pra.F90:570-664`).  Its receiver keeps the
pre-merge `sx_cell` in the second-moment expression
(`advection_som.py:286-314`), whereas SI3's statements use the just-updated
first moment (`icedyn_adv_pra.F90:678-704`).  The five common outgoing-slab
polynomials are mathematically related but not written-order identical
(`**`/nine-field packing versus explicit products/five arrays), so extracting
them would perturb the exact arithmetic under test.  No limiter or complete
moment-update helper should be shared; ocean SOM remains unchanged.

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

### Rung-3.2 DEBT classification — not re-association

The round-2 replay was preregistered at commit `146199c9b3e`.  It advanced the
pinned candidate through 15 steps, then executed even step 16 (y then x) with
the production JAX arm and with the same source-ordered operation graph under
NumPy scalar ufuncs.  The latter preserves the NEMO statement sequence for the
y limiter/sweep (`icedyn_adv_pra.F90:757-943`), x limiter/sweep (`:534-717`),
and post-split corrections (`:405-479,946-1142`).

The production and written-order results are byte-identical: the complete
contents-plus-80-moment SHA256 is
`9f8586e8c84df4c004aeea015b06643de7f02bfc8732fd59bcbac561df709ac5`
for both, every production/replay moment distance is zero ULP, and there is no
within-step differing assignment to name.  Against oracle entry `kt=17`, both
give the same first-gate-DEBT `szv_i_l01`: normalized maximum
`1.1095589247903137e-15`; at the maximum-absolute-error cell `(50,51)` the
oracle is `2.604251475454491`, both candidates are `2.604251475454495`, and
the distance is 9 ULP.  The two-ULP discriminator therefore **REFUTES
re-association** for this step.

The first available differing operand is inherited at step entry.  At
`kt=16`, `szv_i_l01` is already 13 ULP at its worst cell and 7 ULP at the
maximum-absolute-error cell, while remaining numerically AT-BAR
(`8.541871855375954e-16`).  The two-ULP history first differs at cold entry
`kt=1` (maximum 3 ULP).  `iceistate.F90:357-360` computes salt as
`(sz_i*v_i)*r1_nlay_i`, with a stored reciprocal.  The card instead formed
`sv_i/nlay_i`.  Replaying the source multiplication at the `kt=1`
maximum-error cell reduces the discrepancy from 2 ULP to 1 ULP and halves the
normalized maximum from `2.349678350102868e-16` to
`1.174839175051434e-16`; its domain-wide maximum remains 3 ULP.  Thus the
measured classification is **INHERITED STEP-ENTRY DEBT**, with the first owned
operand being initial layer-salt arithmetic order.  The unseen oracle Prather
moments remain a stated blind spot, so this one-step measurement does not
attribute every later base-field or final-moment residual.

For roster clarity, the fixed gate has 7 of its 20 boundary fields AT-BAR over
all 486 boundaries: the transported `oa_i`, `v_ip`, `v_il`, plus carried
`sv_i`, surface temperature, U, and V.  Counting only the 16 transported
channels gives 3/16, not 7/16; the layer-resolved energy and salt channels are
separate registered fields.  The other transported fields and 70/80 final
moment rows remain DEBT, with smooth growth into the `1e-14` class.  No broader
owner is inferred from that scaling.

The planted cross-moment control changes `v_i` by 5,913,692 ULP, proving the
replay comparison is live.  The focused replay suite passes 2/2.

## Rung 3.3 — C-grid aEVP partial implementation boundary

### Review revision before code

The C-grid adaptive-EVP design was first preregistered at `af53e78bd33`.
Independent line review required six changes; all were folded into the design
alone at `9b369d9d082`, before candidate code.  The revised design:

* moves the selectable C-grid solver beside the existing A-grid `evp_solver`
  in `dynamics.py`, not `rheology.py`;
* records exact one-based NEMO and zero-based Python extents for every shear,
  deformation, stress, divergence, component-update, and final-diagnostic loop;
* preserves `zmsk` on `stress1/2` but not `stress12`, and leaves the H79
  `at_i>epsi10` guard outside shared `ice_strength`;
* enumerates the static-friction arms and `rn_lf_relax=1e-5`, then waives them
  dead because landfast OFF makes basal stress zero and reduces the branch to
  the impossible `zRHS<0 AND zRHS>=0`;
* identifies the fast mask as a file field read by `icedyn.F90:113-118`, not a
  consequence of the landfast selector; and
* retracts parity reversal and simultaneous U/V as controls on this zero-
  Coriolis, one-rank periodic card.  With `zmf=0`, precomputed cross velocities,
  and stresses fixed before the pair, they share no live updated operand; the
  full halo exchange removes the remaining loop-extent distinction.  The
  replacement stress-divergence-weight plant binds on the actual card.

The revision also distinguishes the ORCA1 reference deck's `jpl=5`,
`nlay_i/nlay_s=10/5` from its checked EXPREF overlay's explicit `1/3/3`
override.  ORCA1 selects `ln_dynALL` and landfast; this rung explicitly selects
the shipped case's `ln_dynRHGADV`, landfast OFF, `jpl=1`, and `3/3`.  It imports
only the cited ORCA1 H79/aEVP/drag choices.  The revised design SHA256 is
`f70d7b4d7cebcf22645e69df89d6294172319e3b6ff59cd3e5b1d991dd8bae7c`.

### Implemented boundary and measured result

The existing A-grid arm and defaults are untouched.  The same dynamics module
now contains a separately selected same-index C-grid state/config/forcing
contract (`dynamics.py:103-176`) and the SI3 solver at `:920`.  Its helpers
transcribe F shear (`:747`), T deformation (`:776`), and transformed-stress
divergence (`:829`); the H79 presence guard remains in the caller at `:965`.
All T/U/V/F leaves retain NEMO's common `(103,103)` two-halo allocation.  The
card in `nemo_adv2d_rhg_testcase_recipe.py:131-319` rejects every selector
cross-product outside the resolved one-category, landfast-off card, performs
dynamics before reusing the sole Prather implementation, then reuses the same
Hbig/Hsnow/zapsmall corrections.

SI3 uses unfloored square roots in both the deformation and drag paths.  A
primal floor would change the oracle.  The implementation therefore defines an
exact `sqrt(x)` primal and an explicit finite zero subgradient only at the
mathematically undefined `x=0` cusp (`dynamics.py:179-204`).  The complete 100-
subcycle kernel compiles under `jax.jit`; reverse mode on an active perturbed
ice state returns finite, nonzero fp64 gradients.

The partial fail-closed gate records exactly 68 rows: 22 geometry rows plus 23
registered fields at cold entry and after completed step 1.  It red-flags a
missing or duplicate row.  All 22 geometry rows, all cold-entry rows, both
velocities, and all 16 transported tracer rows after dynamics plus Prather are
AT-BAR at the immutable normalized pointwise `1e-15` class.  The three stress
carries are **DEBT**:

| first completed step row | normalized maximum absolute error | result |
|---|---:|---|
| `stress1_i` (T) | `2.677729266548652e-14` | DEBT |
| `stress2_i` (T) | `1.8877857221034826e-14` | DEBT |
| `stress12_i` (F) | `1.7983397985341945e-14` | DEBT |
| `u_ice` (U) | `4.996003610813204e-16` | AT-BAR |
| `v_ice` (V) | `2.632442874794805e-16` | AT-BAR |

These are measured gate labels, not trajectory claims.  A source-ordered
stress replay has not yet been completed, so the stress owner is
**UNMEASURED**.  Per the preregistration, no re-association label is assigned
without that replay and the two-ULP discriminator.  The 485-frame sweep stops
at this clean boundary and remains **UNMEASURED**; no later frame was scored.

The production stress-divergence weight plant changes its registered force by
`5.270939248713802e-05` normalized and is red as required.  Selector/Coriolis,
geometry, row-omission, T/F stress-mask, and stress-DEBT controls are also
covered.  The existing A-grid card was executed from the preregistration commit
and from the implementation commit in one committed guard: both full-state
SHA256s are
`e4ecb3561c33b7d246ef8122807dcb7c9f690dc96c20dd6d9e5a024d0b9351d1`
and every array is byte-identical.

Focused rung-3.3 tests: **8 passed in 13.50 s**.  They cover fp64 selectors,
Frankenstein rejection, the binding force plant, the deliberate missing F-
stress `zmsk`, full-100 JIT/gradient, the loud first-step stress debt, exact
tripwire rows, geometry plant, and the A-grid before/after guard.  Ruff
`F,E501,I` is clean on all seven changed Python files.

Both repository ratchets were run over 3,748 rostered checks: **3,739 passed,
2 skipped, 7 failed**.  Every touched-file row passed.  The failures are the
same unrelated branch debt, disclosed with exact current locations:
`gm_redi_latlon_cgrid.py:1443`; `tke.py:1454,2235,2237` (plus its other
reported bit-mask/table sites); `fv3_native_physics_coupling.py:113`;
`test_fv3_physics_coupling.py:145,176`;
`test_dino_vertex_area_nemo.py:89`;
`test_dino_vface_zonal_width_nemo.py:210`; and
`test_dino_wall_balance.py:115`.  None is modified here.

### Round-5 harness review closure

The path-dependent A-grid guard was replaced by `git -C` against the checkout
containing the guard.  It now executes both pre-existing arms before and after
the rung-3.3 boundary.  EVP is byte-identical at SHA256
`e4ecb3561c33b7d246ef8122807dcb7c9f690dc96c20dd6d9e5a024d0b9351d1`;
mEVP is byte-identical at
`83b4727c5d1d35698368a1697f21b87a537dd424678bb4aa431c96190b840282`.
No repository or worktree path is embedded in the command.

The earlier stress-divergence plant was defective as a trajectory control: it
compared two calls to the same helper and never reached a registered model
row.  It is retracted.  The replacement changes the weight inside the real
100-subcycle solver, advances the complete dynamics-plus-Prather card, and
scores `trajectory.post_step_00000001.u_ice` against the oracle.  The planted
row is DEBT at `3.896295744876266e-06` under the immutable gate metric
(`7.752461843608436e-06` relative to the oracle U maximum), so it binds through
the scored trajectory.

Gate `status` and process exit are now derived from registered row statuses
(and, for the partial gate, its nonempty unmeasured registry); neither result
is a hand-written outcome.  Every pointwise row now reports both the campaign
metric `max_abs / max(max|oracle|,1)` and the companion field-relative value
`max_abs / max|oracle|`.  Zero-reference rows use `0` for an exact comparison
and `null` for a nonzero residual, rather than hiding an undefined ratio.

Review requested rejecting every `rn_ishlat` other than zero, but the pinned
oracle resolves `rn_ishlat=2` (`ocean.output:658` and
`output.namelist.ice:225`).  Relabeling it zero would violate the selector
contract even though this all-wet periodic geometry makes the distinction
numerically inert.  The implemented closure instead reproduces the resolved
nonzero F-mask repair exactly (`icedyn_rhg_evp.F90:215-224`) and rejects every
value other than `2`; a reverted/zero selector test fails closed.  Likewise,
`drag_io` is now a same-shape T-point forcing field and the solver forms
`0.5*(drag_io(i,j)+drag_io(i+1,j))` at U and the north-neighbor form at V,
matching `icedyn_rhg_evp.F90:307-313`.  A nonuniform-field test prevents a
future scalar-only regression.

All six SI3 fidelity test modules moved from `tests/ocean/fidelity/` to
`tests/ice/fidelity/`.  The combined relocated phase-1 and phase-2 set is
**63 passed in 136.78 s** on CPU with x64 enabled; the focused revised
rung-3.3 set is **13 passed in 19.34 s**.

### Round-4 stress replay and full trajectory closure

**Overall rung-3.3 result: DEBT.**  The exact bar remains normalized pointwise
`1e-15`.  The completed gate advances all 485 outer ice steps on CPU in fp64,
consumes entry frames `kt=2..485`, switches to the final ice restart for step
485, and registers 11,235 trajectory/restart rows: 2,811 AT-BAR and 8,424
DEBT.  The fixed registry order is U, V, stress1, stress2, stress12, then all
16 transported tracers, `sv_i`, and surface temperature.  Its first over-bar
row is therefore `post_step_00000001.stress1_i` at
`2.677729266548652e-14`; this is the same reviewed step-1 measurement, not a
new execution-mode result.

The stress replay starts from the exact bytes in oracle entry `kt=1`, including
the three restart-carried zero stresses.  Subcycle 1 is explicitly recorded as
vacuous for stress: zero velocity and zero stress make all three updates zero.
The gate therefore also replays subcycle 2, the first active stress update,
from byte-identical subcycle-1 carries.  An independent NumPy transcription of
`icedyn_rhg_evp.F90:189-741` and the same production statements executed as a
Python-written-order loop are byte-identical for U, V, and all three stresses:
**0 ULP for every carry**, satisfying the preregistered two-ULP discriminator.
Lowering the same subcycle through `jax.lax.fori_loop` changes the
maximum-error stress cells by 2, 3, and 3 ULP for stress1, stress2, and
stress12.  After 100 NumPy-written-order subcycles, the oracle residuals are
`1.2668826637434482e-14`, `2.960067032769315e-14`, and
`1.477542204264423e-14`.  Thus the measured classification is
**RE-ASSOCIATION in the compiled loop**, not a changed strain, delta/viscosity,
mask, or alpha/beta formula.  The production gate remains DEBT; this
classification does not relax its bar.

The different step-1 labels for stresses and velocities are also measured,
not inferred from relative magnitudes.  The stress absolute residuals are
`6.343725544866174e-11`, `2.2566837287740782e-11`, and
`1.0913936421275139e-11`.  Applying the exact transformed-stress divergence
weights from `icedyn_rhg_evp.F90:495-510` gives maximum force residuals
`1.0061285138363018e-14` (U) and `1.199396137963049e-14` (V).  Dividing cell by
cell by only the mandatory `m/dt*(beta+1)` term, with `beta>=50`, bounds the
velocity response by `1.6216806403927668e-16` and
`1.6743699418381982e-16`; omitting the nonnegative drag makes these conservative
bounds.  The observed U/V errors are `4.996003610813204e-16` and
`2.632442874794805e-16`, both AT-BAR.  Spatial differencing plus the large
active momentum denominator therefore explains how a normalized stress DEBT
can coexist with AT-BAR velocity at that boundary.

Requested growth table.  Each cell is `immutable-gate / field-relative`; the
second value is diagnostic and does not replace or relax the first value's
`1e-15` classification bar:

| step | U | V | stress1 | stress2 | stress12 | `a_i` | `v_i` | `v_s` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `4.996e-16 / 9.941e-16` | `2.632e-16 / 1.100e-14` | `2.678e-14 / 2.678e-14` | `1.888e-14 / 1.888e-14` | `1.798e-14 / 1.798e-14` | `3.331e-16 / 3.681e-16` | `2.484e-16 / 2.484e-16` | `8.327e-17 / 4.657e-16` |
| 10 | `1.110e-15 / 2.205e-15` | `6.251e-16 / 7.691e-12` | `4.761e-13 / 4.761e-13` | `3.649e-11 / 3.649e-11` | `4.489e-11 / 4.489e-11` | `1.332e-15 / 1.444e-15` | `8.755e-16 / 8.755e-16` | `2.359e-16 / 1.329e-15` |
| 50 | `1.110e-15 / 2.205e-15` | `9.845e-16 / 1.207e-9` | `5.461e-12 / 1.744e-11` | `8.875e-10 / 1.053e-8` | `4.525e-10 / 1.166e-8` | `4.885e-15 / 5.351e-15` | `2.433e-15 / 2.433e-15` | `4.441e-16 / 2.561e-15` |
| 100 | `1.110e-15 / 2.205e-15` | `9.823e-16 / 3.871e-7` | `4.711e-12 / 1.531e-8` | `7.228e-10 / 7.966e-6` | `3.529e-10 / 7.027e-6` | `5.329e-15 / 5.807e-15` | `4.495e-15 / 4.495e-15` | `6.106e-16 / 3.636e-15` |
| 200 | `3.442e-15 / 6.837e-15` | `2.558e-15 / 5.947e-7` | `7.252e-9 / 4.428e-4` | `5.835e-9 / 7.039e-1` | `1.194e-9 / 4.055e-4` | `1.499e-14 / 1.659e-14` | `1.126e-14 / 1.126e-14` | `2.054e-15 / 1.286e-14` |
| 485 | `7.105e-15 / 1.411e-14` | `5.513e-15 / 1.528e-6` | `9.387e-9 / 7.771e-4` | `8.959e-9 / 7.735e-1` | `1.562e-9 / 7.609e-4` | `1.066e-13 / 1.181e-13` | `6.753e-14 / 6.753e-14` | `1.008e-14 / 7.160e-14` |

The per-boundary rows show smooth accumulated floating-point separation; no
single threshold/mask/parity branch event like the rung-3.1 column event is
observed.  The full artifact retains every boundary and maximizing cell rather
than only this requested sample.

The round-5 reviewer-requested compounding rate is now named explicitly.  The
clearest approximately three-percent window in the registered table is the
`a_i` absolute residual from step 10 to step 50:
`(4.884981308350689e-15 / 1.3322676295501878e-15)^(1/40) - 1 =
3.3015375767554644%` per outer step.  This is a windowed rate, not a claimed
constant exponent over all 485 steps.  Its source is the **nonlinear
sensitivity of the rheology trajectory**: the compiled-loop stress
re-association changes the carried C-grid velocity, and that velocity feeds the
otherwise shared Prather arm on the following outer step.  The absence of a
discrete predicate change in the per-boundary registry is why this is recorded
as compounding nonlinear sensitivity rather than a branch event.

At the final restart, all 23 ordinary registered fields are loaded and scored:
four are AT-BAR (`oa_i`, `v_il`, carried `sv_i`, and surface temperature) and
19 are DEBT.  All 80 discovered Prather moments are loaded and scored: 10 are
AT-BAR (the five age and five pond-lid moments) and 70 are DEBT; the maximum is
`syye_l03` at `4.3579848610592794e-11`.  The card restart now persists packed
contents, carried salt, surface temperature, U/V, all three stresses, all five
16-family moment packs, the exact clock, and a selector/parameter contract
hash.  Save/load is byte-identical and a step-7 split continuation is
byte-identical.  Missing, retyped, and perturbed stress and moment controls all
go red.  Derived `snwice_mass` and `snwice_mass_b` remain loudly UNMEASURED as
candidate prognostics; they have not been silently dropped from coverage.

The shipped README phenomenology is checked separately on both models.  It
describes the square concentration/Gaussian volume, Prather maximum/side-lobe
behavior, rheology-generated velocity, and the below-1-kg-m-2 zero-ocean-
velocity switch (`tests/ICE_ADV2D/EXPREF/README:48-55,64-66`).  Both models
produce nonzero rheology velocities over the run (oracle U range
`[0,0.5043842101574245]`, V range
`[-0.023923408982296886,0.023923408982296886]`; legoESM differs only beyond
the displayed digits).  Across all 485 boundaries, every face satisfying the
source low-mass/low-concentration predicate is exactly at the zero ocean
velocity in both models; at least 6,495 U and 6,495 V faces bind per boundary.
These two predicates are VERIFIED.  Maximum series and negative/local-extrema
censuses are recorded for both models; maximum conservation and “side lobes”
remain **MEASURED-UNCLASSIFIED** because the README supplies neither a numeric
tolerance nor a side-lobe predicate.  No endpoint-equality substitute is used.

Round-4 verification: the focused rung-3.3 kernel/gate/replay/restart suite is
**10 passed**.  Targeted ratchets are **1 passed** for the touched card under
`test_no_inline_physics_coeffs.py` and **3 passed** for the touched card/gate
paths under `test_no_hardcoded_constants.py`.  The complete two-ratchet run is
**3,749 passed, 2 skipped, 7 failed**; the seven failures are the same unrelated
paths and exact sites disclosed above.  Ruff `F,E501,I` is clean on all four
round-4 Python files.  Mypy with skipped imports reports no issues in the card,
replay, or full gate.

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
| written-order replay JSON | `670fd440fcd678130cc0ab5492502d996e35c36a93f49bd041eac9df73dba0ac` |
| written-order replay source | `dd396635f9e3f98842d74704a9888ee96926d87856164d31b8a95b526f022db1` |
| written-order replay controls | `7573ed5250a113c5f4389ca3c6a5f49e2a87497511d5ebd6ce59d8797db7cb28` |

Rung-3.3 partial-boundary artifacts (implementation commit
`54d0d5dad96c38b194ec56addd6dc051439a52e5`):

| artifact | SHA256 |
|---|---|
| revised preregistration | `f70d7b4d7cebcf22645e69df89d6294172319e3b6ff59cd3e5b1d991dd8bae7c` |
| partial gate JSON | `ac95f2deffb3dfd4d4401fb25d3b73a65379b165b34aaac0ba13b0836b066fcf` |
| A-grid before/after guard JSON | `21652495f9028e7a2243bc7dcff2c3928adbe3014f316198c27d812e3396fa58` |
| `dynamics.py` | `67631aafbfbb2b839c3d53cc2fb431048b04be01e3f20f80fe2625945584e250` |
| rung-3.3 card | `9d86f060c11a43e65e6b27fdbdada5b2233dab70ca90e9d0a2eed2768dba8b26` |
| partial gate source | `0941aa928a6d9c174a83bde25230e850a5b7e5483326b32b3436a4217fb6d74f` |
| A-grid guard source | `d571f4ac213a88349b39afa7e4cd9ad5a4b4ed2d8eb6f090836005ceff96ec08` |
| kernel controls | `e17f85ae4de91eae0789615f1be4f04b15df3facdc8c0693b094af9d12971fa1` |
| gate controls | `2ace7a40e4983ef62aa1d39dc6bae6cc8da7972d623e11bc4e59cfd3ee69f188` |

Rung-3.3 round-4 artifacts (implementation commit `ba07bc7b202`):

| artifact | SHA256 |
|---|---|
| trajectory/restart preregistration | `7da368cbb30fe6ae46cd1632223874ef2c9a0514d1b1aebd1b62dfcb29710f12` |
| independent written-order replay JSON | `04267dbc77d229c2a752e3210574eb3331547a1f84860b4d8f3d17e4086116e6` |
| full 485-step trajectory/restart JSON | `5895312d316819b68723f0ee6cef43a09e77978779c745d38e689ed8e0ed6684` |
| rung-3.3 card/restart source | `b92ca8cf64667ac137cbf48d1e9b033fe12b81e2d1c57b86b78671522f6d2056` |
| written-order replay source | `ce078169135bc2003af63c3f4752db4e3c55899fd88faffb9443282435ee97a5` |
| trajectory/restart gate source | `a605af06726e1be62f009ba7acdc0bcc3773c7ebd28c6ac4b66d82583a1d93cd` |
| rung-3.3 controls | `e681a258acf0fcc9fc3c020ff09a282ba7c54a3009e433227282cdc8adfc09c7` |

Rung-3.3 round-5 harness-review artifacts:

| artifact | SHA256 |
|---|---|
| full trajectory with companion relative rows | `2135a1f39677e1eada47752b4b5542dd454d4fa96a38b207510a707604a920a6` |
| scored-row planted-control partial gate | `4f5141e5bc228f198a9bb82cc8a3eafc711a993b83732d8beca0629baae4a8fe` |
| path-independent EVP/mEVP guard | `6f5fd71d159d7e0edc2219d7d76ef06f323662f4e5e557f147a8e193a8ecc317` |
| unchanged written-order replay | `04267dbc77d229c2a752e3210574eb3331547a1f84860b4d8f3d17e4086116e6` |
| `dynamics.py` | `059a99ad24d6f95a4e8034ec296cdc9778219d2a56f2bf87f47c89851a422c65` |
| rung-3.3 card | `3cac7bc999a7f5cbc41f8b546f0b6a3c6ca744b7aa7fe948c48df92ce75c8245` |
| common phase-2 scorer | `d61709508604d645119c13cf6c7c265512498249bd29c37bf98cfbf8f484e4f3` |
| partial gate | `f93ef19f8ffb598c671ace4035fe1fd5bf53af8845379c2d2bd16bfeca3e69b7` |
| full gate | `3eb4293ef7c3ed444aa9a36902bdf290e45fb2be6cf46bd7c1f0384283cd6713` |

Round-5 follow-up adds direct pytest execution of the 610-line full trajectory
gate's scorer, fail-closed shape/dtype/finiteness paths, phenomenology shape
census, and CLI artifact/derived-exit path.  This closes the earlier coverage
gap; it does not rerun or relabel the scientific trajectory.  The direct file
is **6 passed in 0.11 s**.  The two requested ratchets both include this test
file and the unchanged trajectory gate: inline-physics reports 17 passing
tests plus one skip after its two checkout-discovery failures, while the
hardcoded-constant run reports 2,436 passed and one skip after five unrelated
checkout/legacy-test failures.  The new test itself is exercised and green in
both runs; its literal test fixtures are at lines 25--38 and 41--63, and its
CLI/derived-exit control is at lines 80--92.  Ruff is clean.

| round-5 follow-up artifact | SHA256 |
|---|---|
| direct full-trajectory-gate pytest | `1eea43ce06a884c1237ce7c47396bca539ad0a5b91693c642122a38e9de2d664` |

## Rung 3.4 implementation boundary: ICE_RHEO dynALL

Status: **KT1-VERIFIED; DEBT-FIRST-DIVERGENCE at completed step 2**.  This is
an honestly stopped first-divergence boundary, not a completed 720-step
legoESM trajectory or restart claim.

The pre-implementation search found the existing Lipscomb-2007 path in
`packages/ice/legoesm/ice/ridging.py` and no rafting implementation.  Its
per-category weight is proportional to `a*exp(-h/e*)`.  SI3 instead constructs
the normalized cumulative-area distribution `G`, applies the exponential
difference at `icedyn_rdgrft.F90:433-470`, partitions that participation into
ridge and raft fractions at `:474-503`, and iterates its ordered redistribution
at `:287-341,598-624,667-903`.  The two participation/update algebras are not
identical, so factoring either into a nominally shared helper would change an
existing scheme.  The new `apply_si3_jpl1_ridging` is consequently a selectable
sibling **inside the existing ridging module**; `apply_ridging` and its default
remain unchanged.  H79 strength is not reimplemented there: the existing
C-grid rheology arm supplies `rn_pstar=2e4` and `rn_crhg=20` before
redistribution.

The new arm is deliberately closed over the measured selector composition:
`jpl=1`, exponential participation and ridge distribution, ridge plus raft,
`rn_astar=.03`, `rn_murdg=3`, `rn_csrdg=.5`, `rn_hstar=25`,
`rn_hraft=.75`, `rn_craft=5`, and zero ridge porosity.  Its participation,
partition, normalization, donor/receiver carry, and 19-iteration ceiling cite
`icedyn_rdgrft.F90:209-341,398-624,667-903`.  Because the case resolves
`ln_icethd=F`, SI3 itself overwrites the deck's four `0.5` snow/pond retention
values to `1` at `:1244-1247`; honoring the unevaluated deck value produced a
`v_s` debt of `9.37e-14`, while reproducing the executed value closed kt=1.

The oracle-pinned card consumes the entry frame and every geometry array rather
than synthesizing the 1000 by 1000 grid.  It validates fp64 throughout,
retains 32 transported fields and five Prather moments per field (160 moment
leaves), uses the existing C-grid aEVP and Prather arms, and rejects every
selector cross-product outside this case.  Its outer step is exactly
`rhg -> adv -> rdgrft -> cor`, the shipped dynALL dispatcher at
`icedyn.F90:130-135`.  Existing A-grid EVP/mEVP and existing Lipscomb cards are
untouched.

The full kt=1 gate scores eleven geometry/mask arrays plus all 32 transported
fields, surface temperature, bulk salt diagnostic, U/V, and three stresses:
50 rows are AT-BAR.  The worst row is `stress1_i`, normalized maximum
`6.869504964868156e-16`; every numeric candidate row is `float64` and the three
mask rows are `bool`.
The binding internal trajectory plant changes `v_s` by `1e-10` and is caught
as DEBT (`9.999999439624929e-11` normalized), proving that the scored trajectory
path—not a helper self-comparison—goes red.

The committed first-divergence sweep advances the full dynALL card and stops at
the first over-bar entry boundary.  Completed step 1 remains at bar.  At
completed step 2 / oracle entry kt=3, `stress1_i` first exceeds the bar:
normalized and relative error `4.383271722679783e-15`, absolute error
`5.856426454897701e-15`, oracle scale `1.3360856514086425`, at `(20,59)`.
At that boundary `stress2_i=1.8943180357666733e-15` normalized and
`stress12_i=1.27675647831893e-15`; velocities and every advected/redistributed
field are still AT-BAR.  This first owner is the carried nonlinear C-grid aEVP
stress trajectory.  Per the preregistered stopping rule, boundaries after kt=3,
the candidate final step/restart, and all 160 candidate restart moments remain
**UNMEASURED**; the existence of a completed NEMO restart does not measure a
legoESM restart.

Focused verification is **9 passed** for card/geometry/wind/selector/kt=1 and
ridging controls, **13 passed** for the rung-3.2 transport regression, and
**8 passed** for the unchanged rung-3.3 C-grid kernel.  The inline-physics
ratchet is **3 passed** and the hardcoded-constant ratchet is **5 passed** on
all touched production/gate/test paths; Ruff is clean.

| rung-3.4 artifact | SHA256 |
|---|---|
| kt=1 gate JSON | `5da50c8dca4c3fba55ff3baccdf4bf95495fdb2e0fe567e576425330ac891e6a` |
| binding planted gate JSON | `e3597b7338d5d302a57e6a3f0932051768289ab4526e5f7ccfd0d97181cd6b2c` |
| first-divergence JSON | `8ff7ff42bde1d4e18481a2f1d084b775313275f2b17ae9f288122eee12be9972` |
| existing-module SI3 ridging source | `192f49c35f0222323f1f5517c0dd9de306d871fb9fb879cc939ccb632a2eb384` |
| oracle-pinned card source | `a841e486981ee98efaf5ce40d1c298178454d24e39f28c16ca96f2a7d9ac5c0c` |
| kt=1 gate source | `b3bc5657686a7c7cc595f103090fcd5cb7ac7b182237a146ed73cd87f79740a5` |
| first-divergence source | `ab3166934623a0573abfa173b1e6168be245761a2816bf3248bd34fde6b45572` |
| direct card/control pytest | `dcf4fc024dc8a678a0f81d1c98ed3a1d7a2029e897b5ac97bbf96f9113053f67` |

## Loudly UNMEASURED / deferred

Within-step x/y Prather split states; ORCA1
option-2 salinity (the rung-3.2 oracle resolves option 4); candidate alignment
of nonzero `snwice_mass` and its
before level; separation of `zapsmall` from `zapneg`; thermodynamics;
general production run-restart integration outside the
opt-in card's now-verified restart path; coupled ice--ocean comparison; and
landfast L16 (OFF here,
**UNVERIFIED-deferred to lane 4**).  No legoESM claim is made for any item in
this paragraph.

## End-of-task ASKED / UNASKED choice list

**ASKED choices:** resolve the rung-3.1 review HOLD in its own commit; implement
rungs 3.1 and 3.2 against the pinned shipped cases; fp64/CPU only and `jpl=1`;
Prather as a selectable arm of the existing transport module; prognostic,
restart-carried moments; geometry, `kt=1`, first-divergence and restart gates;
planted controls; preserve production defaults; explicit-pathspec commits;
classify rung-3.2 DEBT with a written-order replay; preregister and independently
revise the rung-3.3 design before code; place the selectable C-grid arm beside
the existing A-grid solver; preserve the A-grid result byte-for-byte; implement
the fp64 card, geometry/cold-entry/first-step gate, binding controls, tripwire
rows, and JIT/gradient test; stop at an honest committed boundary if the sweep
is not reached; then classify the stress debt with a source-written-order
subcycle replay, show the stress-divergence/velocity arithmetic, sweep all 485
boundaries, validate all prognostics/three stresses/80 moments at restart, and
measure the shipped README phenomenology on both models; branch bundle; no
push and no shipped-NEMO modification; make the A-grid guard checkout-relative
and cover EVP plus mEVP; route the stress-divergence plant through a scored
trajectory row; derive gate status/exit from rows; close the `rn_ishlat` and
2-D `drag_io` generalization gaps; publish field-relative errors beside the
immutable gate metric; and place SI3 tests under `tests/ice/fidelity/`.
Then preregister rung 3.4, create production and shipped-override copies of
ICE_RHEO, apply only the cited ORCA1 aEVP/ridging deck with landfast off,
measure the stale override, and proceed to the existing-module jpl=1
ridging/rafting arm only if the shipped oracle can run; then accept the
21-error shipped-control failure as an upstream NEMO defect, use the sole
buildable override-excluded copy as the oracle, run its documented 720 steps,
extend Appendix-A coverage to ridging/rafting state, implement the selectable
jpl=1 SI3 redistribution arm, wire the existing arms in dynALL order, and gate
geometry, kt=1, and first divergence.  Also add direct pytest coverage for the
rung-3.3 trajectory gate and name its measured 3.3015375767554644%/step window
as nonlinear rheology sensitivity.

**UNASKED choices:** no default change to existing ice transport; no ocean-SOM
replacement or arithmetic-changing common refactor; no rung-3.3 claim beyond
the measured gate rows; no thermodynamics, ridging/rafting, landfast,
coupled-ocean, multi-category, or general production-restart claim outside the
new card contract; no analytic oracle; no tolerance relaxation; no numerical
classification of the README's qualitative maximum/side-lobe sentence; no
claim that a DEBT or UNMEASURED row is matched or faithful; no relabeling of
the pinned oracle's `rn_ishlat=2` as zero merely because the two branches are
inert on this all-wet periodic card.
No repair or modernization of either shipped ICE_RHEO override; no deletion or
modification in the shipped NEMO tree; no numerical stale-source trajectory
comparison fabricated from an unbuildable control; no multi-category,
thermodynamic, landfast, alternate-rheology, or alternate-ridging selector arm;
no claim past the measured rung-3.4 first-divergence boundary; no tolerance
relaxation, GPU, MPI, or push.
