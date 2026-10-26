# SI3 lane 3 — phase-2 receipt (rungs 3.1 through 3.4)

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
* round-8 active-regime preregistration and gates: `bafb805f09a`,
  `d9bc45316c6`, and `eb8e25bb94e`;
* executed-source `ato_i` correction: `98e26f63578`;
* rung-3.3 mechanism relabel and dual error columns: `cc72d174fd4` and
  `e50934521ba`;
* exact SI3 clamp exclusions and category-axis guard: `af79634728e` and
  `3489594617c`;
* 720-frame SI3 shear replay and active-window hardening: `c5722694454`,
  `d53bcb2ace1`, `72b17eba2d1`, and `5739462e852`.
* round-10 aEVP source-rounding preregistrations: `968524ca6b9f` and
  `c61371946e79`;
* shared scalar-libm policy import: `f8b46e31e065`;
* source-exact C-grid aEVP implementation and gates: `0458d8d4a8ee`.

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

**CONFIRMED arithmetic:** the round-5 reviewer-requested compounding rate is
named explicitly.  The
clearest approximately three-percent window in the registered table is the
`a_i` absolute residual from step 10 to step 50:
`(4.884981308350689e-15 / 1.3322676295501878e-15)^(1/40) - 1 =
3.3015375767554644%` per outer step.  This is a windowed rate, not a claimed
constant exponent over all 485 steps.  Those two endpoints are only about six
and twenty-two ULPs of a max-one field, respectively.  **PLAUSIBLE, not
measured:** nonlinear rheology sensitivity could compound the carried
stress/velocity re-association before Prather advection on the following outer
step.  No one-ULP perturbation experiment was run, so neither the 3.3015% rate
nor the absence of a registered discrete predicate change identifies that
mechanism.  The source of the windowed growth remains UNMEASURED at the
precision floor.

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

Status: **KT1-VERIFIED; DEBT-FIRST-DIVERGENCE at completed step 2; ACTIVE
RIDGING PROGNOSTICS AT-BAR at completed step 8, with stresses and 60/160
Prather moments DEBT**.  This is not a completed 720-step legoESM trajectory
or restart claim.

### Round-8 `ato_i` review disposition

**CONFIRMED -- reviewer premise retracted.**  The review requested changing
`ato_i` into the first Prather tracer, deleting its nonnegative clamp, and
adding five restart moments.  That would depart from the executed NEMO 5.0.2
oracle.  `icedyn_adv.F90:88-91` passes `ato_i` into the Prather routine, but
the routine's transported-field list and calls at
`icedyn_adv_pra.F90:218-300,306-350` contain no open-water content or moments.
Instead, after all category tracers are advected and `ice_var_zapneg` runs,
SI3 reconstructs `pato_i` with the explicit `MAX(0, ...)` balance at
`icedyn_adv_pra.F90:418-430`.  The module's complete saved-moment declarations
at `:34-46` likewise contain no `ato_i` family, and the measured ICE_RHEO
restart roster has 160 moment arrays for the existing 32 tracers and no open-
water moment.  The card's current balance and clamp are therefore retained;
`ato_i` is not added to `ICE_RHEO_TRACERS`, and no synthetic oracle moment is
created.  The direct card registry test now locks that exclusion.

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
46 informative rows are AT-BAR and four oracle-zero age/pond rows are
UNINFORMATIVE.  The worst informative row is `stress1_i`: normalized maximum
`6.869504964868156e-16`, field-relative maximum
`2.3534714128145477e-15`.  **CONFIRMED:** AT-BAR is classified only by
`max_abs/max(max|oracle|,1) <= 1e-15`; the field-relative column is reported
beside it as a diagnostic and is not the bar.  This preserves the campaign's
immutable max-one normalization and avoids dividing the classification by
zero-valued fields; it is an absolute `1e-15` bound for sub-unit fields, not a
relative `1e-15` claim.  Every numeric candidate row is `float64` and the three
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
field are still AT-BAR.  **CONFIRMED:** this first owner is a carried stress row
produced by the C-grid aEVP arm.  Its causal growth mechanism remains
**UNMEASURED**; the word "nonlinear" names the executed equations, not a
measured attribution.  At the round-7 stopping boundary, boundaries after
kt=3, the candidate final step/restart, and all 160 candidate restart moments
were **UNMEASURED**.  The round-9 production-JIT walk below supersedes that
historical status with measured DEBT rows; the completed NEMO restart alone
was never treated as a legoESM measurement.

### Round-8 active-redistribution measurement

The dominant active-window debt is **`active_moment.sxxe_l01`**, normalized
`3.492459543785742e-9` and field-relative `2.095149123314519e-8`.  It is about
six orders of magnitude above the `stress1_i` field-ordering row
(`3.69646870789917e-15`); the moment row, not the ordering convention, is the
headline debt.

**CONFIRMED:** kt=1 exercised only a near-rest redistribution.  The committed
scan uses SI3's own closing equations at `icedyn_rdgrft.F90:243-252` and its
`epsi10=1e-10` redistribution cutoff at `:594-595,623-624`.  It selects the
first completed step for which `max(closing_net * rDt_ice) > epsi10`.  The
preregistered step-9 prediction is **REFUTED**: completed step 8 is first.

| completed step | max `|delta_i|` (s^-1) | max `opning` (s^-1) | max `closing_net` (s^-1) | max redistributed-area demand |
|---:|---:|---:|---:|---:|
| 1 | `3.0061855342332605e-13` | `2.7113227985048475e-13` | `7.198999503613095e-14` | `2.1596998510839286e-12` |
| 2 | `1.2352371163148635e-12` | `1.1023968777106495e-12` | `4.4283150994362443e-13` | `1.3284945298308732e-11` |
| 3 | `2.768515840614345e-12` | `2.4787076188339375e-12` | `1.005965253149349e-12` | `3.017895759448047e-11` |
| 4 | `4.95129308062187e-12` | `4.434634687243251e-12` | `1.2503461933351708e-12` | `3.751038580005512e-11` |
| 5 | `7.648365214050838e-12` | `6.868841495203859e-12` | `1.4074162153614428e-12` | `4.222248646084329e-11` |
| 6 | `1.1046486778854914e-11` | `9.919305591468476e-12` | `1.6447005069310737e-12` | `4.934101520793221e-11` |
| 7 | `1.4799919508798402e-11` | `1.3320082382328334e-11` | `2.806337228778459e-12` | `8.419011686335376e-11` |
| **8** | `1.9342643933785387e-11` | `1.7409121169648706e-11` | `5.026618518018429e-12` | **`1.5079855554055287e-10`** |
| 9 | `2.4304903870948512e-11` | `2.1871114286978893e-11` | `8.29273242782463e-12` | `2.487819728347389e-10` |
| 10 | `2.9719096219813545e-11` | `2.676083258881891e-11` | `1.4387189385894725e-11` | `4.3161568157684173e-10` |
| 11 | `3.577153008668886e-11` | `3.326794358626695e-11` | `2.265510692352959e-11` | `6.796532077058877e-10` |
| 12 | `5.563712280414257e-11` | `5.37152630871105e-11` | `2.652089574912816e-11` | `7.956268724738448e-10` |
| 13 | `7.357397885514287e-11` | `7.066946512312491e-11` | `3.024124582300152e-11` | `9.072373746900456e-10` |

The aligned CPU/fp64 replay starts from oracle entry frame 8 and NEMO's
step-7 restart-carried moments, then scores entry frame 9 and the NEMO step-8
restart.  The only NEMO changes are `nn_itend=8` and `nn_stock=1`; the physics
namelist is byte-identical to the 720-step oracle.  The isolated ridge/raft arm
changes `a_i` and `ato_i` by `3.43068906616395e-10`; 944,617 wet cells take one
redistribution shift and 51,387 take two (3,996 take none).  **CONFIRMED:** the
finite-amplitude completed-step rows for `a_i`, `v_i`, and `v_s` remain AT-BAR
with maximum absolute errors `2.220446049250313e-16`,
`4.440892098500626e-16`, and `2.7755575615628914e-17`.  Thus this window binds
the active area/volume redistribution, including repeat shifting, but it does
not exercise the 19-shift ceiling or certify the zero age/pond channels.

The four oracle-zero state rows `oa_i`, `a_ip`, `v_ip`, and `v_il` are
**UNINFORMATIVE**, never AT-BAR.  Of the other 206 rows, 143 are AT-BAR and 63
are DEBT.  The preregistered first-over-bar prediction is **CONFIRMED**:
`stress1_i` is first, at normalized and relative
`3.69646870789917e-15` (`1.4210854715202004e-13` absolute).  All three stresses
are DEBT.  Restart-carried moment scoring is complete: 100/160 rows are AT-BAR
and 60/160 are DEBT; the largest is `sxxe_l01`, normalized
`3.492459543785742e-9`, field-relative `2.095149123314519e-8`.  Ridging does
not update Prather moments, so this moment result is a transport/state-carry
debt, not evidence that the active redistribution row failed.  No causal
mechanism for that debt is claimed.

The removed final open-water and broad pond-lid clamps are numerically inactive
on this measured window: open water stays positive (`0.009999990410122006`
before, `0.00999999040071575` after) and pond-lid volume is exactly zero before
and after.  This is **CONFIRMED zero trajectory movement**, not permission to
retain non-oracle stabilizers.  The replacement follows SI3 literally:
`ato_i` is absent from `ice_var_roundoff`, while only
`-epsi10 < v_il < 0` is clipped (`icevar.F90:871`).  The selected arm now also
raises on any state with a category axis; its five-category ORCA1 control is a
passing test of the loud `jpl>1` boundary.

The active scored-row plant binds independently of the already-red gate:
`active.step8.v_s` moves from **AT-BAR** at
`2.7755575615628914e-17` normalized to **DEBT** at
`9.999999439624929e-11` normalized (`1.0101009284553422e-9`
field-relative).  The gate now asserts this exact row transition and fails if
the scoring path stops seeing the plant; its overall exit code is not used as
the plant proof.

| round-8 artifact | SHA-256 |
|---|---|
| source-significant regime scan | `c5501e3e3c7bdd3b55f5b647994dfabcdb5cb00b88a7d3a46ddb1c1f9bdcf279` |
| aligned active-window gate | `700325d0b149245b9ce81629e628b3dd7b7993dedc5df23a062ba5f205496551` |
| aligned active-window scored-row plant | `6887bbb76be599aaa37af0f232caf0fd5d59fb2da93c0464c91d4ae65a5f3c1a` |
| refreshed kt=1 dual-column gate | `c5baa6177de981add5c371e21d013a72bbe038d47f681bf318113a4601bf6872` |
| 720-frame NEMO `sishea` evaluation | `3c741bf041896d2c6be354ce54338a0b050e7f36a9949bae3a390a766d1acf87` |
| full oracle gate with `sishea` (expected nonzero MEASURED-UNCLASSIFIED exit) | `34ac7cabe31d7ea742409b3552c14ec101437880cea4090bf2f59c2953e358da` |
| NEMO step-7 / step-8 moment restarts | `34ab003e6564008bc7ecacac5500689cbbbaa23d9259ba146a3e2e6bbcaaf9d1` / `6b826fd00e13133abd293e447e2a16e4d781f5f993709946d515a5aa0a4e59d8` |
| direct eight-step NEMO stdout / stderr | `dc666b95a6cdfd9cc674b3935555d72642597d13ef87617a935836745ad26a58` / `ee867a804e8f26013c5190b38ea5e4be17ed1a5c5fcfcdf8932a9f8cfbb3f36c` |
| SI3 ridging implementation | `4bc4aad794490ef696b18405691df4f84029b937078428edc4f238a30a066145` |
| final oracle / kt=1 / trajectory gate sources | `01f226c15d2a17efaa91d698112e022d860a534f188546d4a0fefde6fc073226` / `07dcdeeed71390fa84720de3cf2dff93d74cd74284449743145ee51e7ea69baf` / `e58ea2d7c2f4a2ff0a30647913e644578da9aa1d33ab1bf5c9cd94685eec488e` |

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

The oracle dependency subsequently completed all 720 steps at
`/data/abyssal/dbalwada/nemo-testcases-l3/ice_rheo/final`; its exhaustive
coverage status is VERIFIED and its ordered-frame aggregate is
`3e347b438776c077a75c8c02ab3e43ac9158037d164ef48a2e5017dd152ddd27`.
The copied non-XIOS run did not emit `sishea`, but round 8 closes that chosen
measurement gap by evaluating NEMO's own formula on every one of the 720
entry-frame velocity fields and the oracle mesh.  The replay follows F-point
shear `icedyn_rhg_evp.F90:793-796`, T-point tension `:802-806`, four-F-point
weighting `:810-813`, and `sishea=sqrt(zdt**2+zds**2)*zmsk` at `:815-816`;
`zmsk` comes from `at_i>=epsi10` at `:190-191`.  **CONFIRMED measurement:**
the final frame has maximum `1.778565218457925e-4 s^-1`, p95
`6.658604946931389e-7 s^-1`, and p99 `1.6653117537818702e-5 s^-1`.
The README gives no numeric sharpness band and this run has no EAP comparator,
so the phenomenology remains **MEASURED-UNCLASSIFIED**, not passed.  The native
SI3 conservation check still resolves off and is WAIVED-INACTIVE.

Two post-run harness gaps failed closed before that artifact was accepted and
are disclosed here: the two committed rung-3.4 decks intentionally have
different filename stems, and the endpoint velocity-response loader initially
omitted `v_ice`.  The gate now carries explicit contracts for both.  The
documented-shear path now has a selective frame reader plus a binding velocity
perturbation/mask test and fails loudly if neither output nor the registered
reconstruction inputs exist.  It is a replay of the executed oracle source,
not an analytic substitute for the README.  Final focused-test and ratchet
results are recorded in the round-8 verification paragraph below.

### Round-8 verification and ratchet disclosure

The combined CPU/fp64 oracle/card/trajectory/ridging selection is **45 passed
in 123.79 s**; the post-refactor oracle-gate rerun is **28 passed in 11.76 s**,
and Ruff is clean.  The relevant controls are visible at
`test_nemo_si3_oracle_gate.py:97-122` (selective-reader failure plus binding
shear/mask plant), `test_nemo_si3_phase2_rung34_trajectory_gate.py:27-75`
(closing, zero-row, 160-moment, and dual-normalization controls), and
`test_si3_jpl1_ridging.py:121-168` (literal SI3 roundoff exclusions and the
five-category rejection).  Production guard/roundoff code is at
`ridging.py:574-593,920-939`; the classifier and both reported columns are at
`nemo_si3_phase2_rung34_gate.py:83-124`.  ORCA1/diagnostic numeric values are
module-level, source-annotated declarations at
`nemo_si3_oracle_gate.py:57-90`, not inline coefficients.

The exact touched-file ratchet selection is **6 passed in 0.94 s**.  Applying
the same AST detectors directly to all three touched gate scripts and
`ridging.py` reports `inline=[]` and `hardcoded=[]` for each.  The mandatory
repo-wide invocation was also run and is honestly red: **3,752 passed, 2
skipped, 7 failed**.  All seven failures are outside this lane's diff:
inline-coefficient debt in `gm_redi_latlon_cgrid.py` and `tke.py`, plus
hardcoded-constant debt in `fv3_native_physics_coupling.py` and four existing
grid/ocean tests.  No receipt claims a green repository-wide ratchet, and no
unrelated file was edited to conceal it.

### Round-9 Prather-moment owner and 9--720 walk

The source census is **CONFIRMED** and refutes the proposed legoESM-only
moment-consistency adjustment.  Across `src/ICE`, the saved Prather families
occur only in `icedyn_adv_pra.F90`: the outer routine dispatches the ordered
`adv_x`/`adv_y` writes at `:253-350`, exchanges their halos at `:432-479`, and
calls restart output at `:487`; `adv_pra_init` allocates them at `:1145-1200`;
and `adv_pra_rst` reads, zero-initializes, or writes them at `:1203-1503`
(including zero initialization at `:1365-1379` and output at `:1397-1497`).
No rheology, redistribution, correction, or thermodynamics routine writes a
moment.  legoESM likewise carries the moment tuple unchanged between calls to
its existing Prather transport arm (`nemo_rheo_testcase_recipe.py:646-704`).

The first replay artifact reproduces the production-input endpoint: 60/160
moment rows are **DEBT**, led by `sxxe_l01` at
`3.492459543785742e-9` normalized and `2.095149123314519e-8`
field-relative.  Replacing both transport velocities with the oracle values
still leaves 60 DEBT rows before the written-order correction, so the broad
"velocity-only" prediction is **REFUTED**.  Its operand trace first differs
after the y limiter in `v_i.sx`, at interior `(16,2)`, by 136 ULP; the limiter
itself has no velocity operand (`icedyn_adv_pra.F90:757-791`).

The narrower preregistered arm changes only the four face transported-area
expressions from `abs(velocity)*dt` to NEMO's constructed
`alpha*donor_area` ordering (`icedyn_adv_pra.F90:582,628,805,851`).  It does
not move the production-input owner or its 60-row count, so the predicted
order-of-magnitude production move is **REFUTED**.  It does, however, change
the discriminating oracle-input arms: oracle V removes the 30 y-owned rows,
oracle U removes the 30 x-owned rows, and the eager written-order oracle-U+V
arm changes 60 DEBT rows to **160/160 byte-exact** endpoint moments (zero rows
over two ULP).  The literal transported-area correction is therefore retained
and **CONFIRMED necessary in that eager replay**.  Round 11 repeats the
oracle-U/V discriminator through production JIT and **REFUTES** the former
exclusive velocity-owner conclusion: 60 moment rows remain DEBT and 140/160
are over two ULP under JIT.  The claim that no endpoint transport algebra
remained is withdrawn; compiled Prather statement association is the next
owner.  The production-JIT walk separately records seven JIT/eager step-9
DEBT leaves: four moment leaves at
`1.7461744827862447e-9`--`5.819814907770393e-9` normalized and three stress
leaves at `3.2306858985948695e-15`--`9.864812791606474e-15`; no compiler
mechanism beyond this measured execution association is claimed.

The full production-JIT walk completed every candidate step from 9 through
720 on CPU under `PrecisionPolicy.fp64()` and scored the final NEMO restart.
The classificatory bar remains `max_abs / max(max_abs(oracle),1) <= 1e-15`;
the field-relative column is diagnostic and is not the bar.  The complete
ordinary-field first-crossing register is:

| first completed step | fields | first normalized / field-relative range |
|---:|---|---|
| 9 | `e_s_l01..l05` | `1.4674225963469599e-15` / same |
| 9 | `stress1_i`, `stress2_i`, `stress12_i` | `4.7141934749322356e-15`--`1.2793429089114667e-14` / same |
| 9 | `szv_i_l01..l10` | `1.2460415037437435e-15` / same |
| 10 | `a_i` | `1.1102230246251565e-15` / `1.1214373875723981e-15` |
| 10 | `e_i_l01..l10` | `1.3471512235101867e-15` / same |
| 11 | `v_i` | `1.233581125301565e-15` / same |
| 19 | `u_ice`, `v_ice` | `1.3189768620446002e-15`--`1.8401947480194917e-15` / `4.0854902331162955e-10`--`5.6533027936468151e-10` |
| 25 | `v_s` | `2.8588242884097781e-15` / `2.8877012059595306e-14` |
| never through 720 | `t_su`, `sv_i` | exactly zero error at the restart |
| UNINFORMATIVE | `oa_i`, `a_ip`, `v_ip`, `v_il` | oracle and candidate identically zero |

The requested growth rows below are measured from that same uninterrupted
JIT walk; each entry is `normalized / field-relative`:

| step | `u_ice` | `v_ice` | `stress1_i` | `stress2_i` | `stress12_i` | `a_i` | `v_i` | `v_s` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | `8.656e-21 / 1.106e-14` | `6.965e-21 / 1.475e-14` | `5.141e-15 / 5.141e-15` | `4.714e-15 / 4.714e-15` | `1.279e-14 / 1.279e-14` | `6.661e-16 / 6.729e-16` | `4.486e-16 / 4.486e-16` | `5.551e-17 / 5.607e-16` |
| 10 | `2.377e-20 / 2.356e-14` | `1.186e-20 / 1.989e-14` | `1.704e-14 / 1.704e-14` | `1.229e-14 / 1.229e-14` | `1.613e-14 / 1.613e-14` | `1.110e-15 / 1.121e-15` | `7.850e-16 / 7.850e-16` | `9.714e-17 / 9.813e-16` |
| 50 | `7.901e-7 / 2.199e-2` | `4.286e-7 / 2.611e-2` | `4.431e-2 / 4.431e-2` | `3.568e-2 / 3.568e-2` | `4.306e-2 / 4.306e-2` | `5.040e-8 / 5.091e-8` | `6.358e-8 / 6.358e-8` | `6.295e-9 / 6.358e-8` |
| 100 | `1.628e-8 / 9.383e-5` | `8.699e-9 / 1.177e-4` | `2.508e-4 / 2.508e-4` | `3.097e-4 / 3.097e-4` | `3.555e-4 / 3.555e-4` | `9.390e-8 / 9.485e-8` | `1.330e-7 / 1.330e-7` | `1.317e-8 / 1.330e-7` |
| 200 | `7.290e-6 / 5.292e-3` | `1.387e-5 / 2.401e-2` | `2.197e-2 / 2.197e-2` | `1.116e-1 / 1.116e-1` | `6.813e-2 / 6.813e-2` | `1.548e-6 / 1.563e-6` | `1.597e-6 / 1.597e-6` | `1.581e-7 / 1.597e-6` |
| 485 | `3.423e-3 / 2.276e-2` | `4.772e-3 / 4.543e-2` | `6.985e-2 / 6.985e-2` | `2.793e-1 / 2.793e-1` | `3.224e-1 / 3.224e-1` | `4.548e-3 / 4.562e-3` | `4.319e-3 / 4.319e-3` | `4.573e-4 / 4.319e-3` |
| 720 | `8.720e-2 / 2.438e-1` | `7.649e-2 / 2.358e-1` | `1.444e-1 / 1.444e-1` | `2.390e-1 / 2.390e-1` | `7.659e-1 / 7.659e-1` | `5.123e-2 / 5.138e-2` | `3.638e-2 / 3.638e-2` | `4.843e-3 / 3.638e-2` |

**CONFIRMED endpoint:** of 39 ordinary restart rows, 33 are DEBT, `t_su` and
`sv_i` are AT-BAR, and the four zero age/pond receivers are UNINFORMATIVE.
Of 160 restart-moment rows, all 140 informative rows are DEBT; the remaining
20 are the exact-zero moment packs belonging to the four UNINFORMATIVE
age/pond receivers and are not promoted as evidence.  The largest endpoint
moment is `sxya`, normalized and relative `1.6259034728302497`; the largest
ordinary endpoint row is `stress12_i`, `0.76592339279538857`.  This is a
completed restart validation and a loud trajectory **DEBT**, not a fidelity
claim.  The trajectory growth is measured; its nonlinear owner is
**UNMEASURED**.

The literal excessive-removal scan evaluates
`apartf*closing_gross*rDt_ice > a_i` from
`icedyn_rdgrft.F90:600-612`.  It finds zero firing cells and zero open-water
corrections in every one of the 719 transitions supplied by entry frames
1--720.  At completed step 719 the maxima are `|delta|=2.2216755820843106e-4
s^-1`, `opning=1.8418422419436972e-4 s^-1`, `closing_net=5.447690481734209e-5
s^-1`, and redistributed-area demand `1.6343071445202627e-3`.  Thus the
clamp is **CONFIRMED unexercised through the available frame-to-frame scan**;
completed step 720 has no kt=721 entry frame and remains UNMEASURED by this
instrument.  The age/pond receivers likewise remain UNINFORMATIVE:
this rung executes `ln_icethd=F`, while the production ORCA1 deck itself sets
`ln_pnd=.false.` (`/data/abyssal/dbalwada/ORCA1-omip/EXPREF/namelist_ice_cfg:151`).
The copied ICE_RHEO oracle resolves `LN_PND=T` in `output.namelist.ice:78`, but
without thermodynamics no nonzero pond receiver is produced; no pond or age
production claim is made.

The active-window planted control remains bound at row level after the
transport correction: `active.step8.v_s` moves from AT-BAR
`2.7755575615628914e-17` to DEBT `9.999999439624929e-11` normalized.  The
moment replay's velocity plant changes 140 scored moment rows, with
`sxice` moving from exact to `1.4445620896407168e-26`; both controls are
independent of an already-red overall exit code.

| round-9 artifact | SHA-256 |
|---|---|
| initial eager walk target (interrupted; zero bytes; REJECTED) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| production-JIT full walk + final restart score | `d5436efe847704c1fa77ef85e8c94db1cc652fbba063ccfc7ebca15c117153e8` |
| excessive-removal scan | `829f122cdac6484d379f79067b9c66d52eb5c03dbd1ff7b5f418231d717be8af` |
| first four-arm moment replay | `2fa4a60fcbca5d1e91f3eebbc3501b0b1418ba1f26c8195fd0cf3427ea039b52` |
| written-order transported-area replay | `761317721545db3afe98c28ba94c551c8a0db1bb5b1f680a23cb2dd7c9d0a415` |
| corrected active-window gate / row plant | `ebf355b1c8c1410c70f1ebbdc1bb09f5c682f03c89c17741310f8073e0bf48fa` / `933b8aa90b04c374f20d85dffa6127b4a459f770f3010c09d9ce107acaeb0420` |
| final NEMO restart scored by the walk | `e5161e64a5c8e4d9180420dca58f99a43088952d0e9826c802d94ac1693b4bf6` |

The measurement implementation spans commits `125032b53d9` through
`cb0b1a3688c`; the source-order transport change is `8af0852eeb4`.  The
accepted full-walk artifact records `execution_path=JIT (CPU, fp64)` and
`walk_completed_steps=[9,720]`.  The empty `full_walk.json` and the initially
empty `full_walk_jit.json` were inspected before the latter was rerun in one
attached foreground process; no background Python process was left behind.
Focused CPU/fp64 verification is **39 passed in 198.58 s**, followed after the
import-only/style cleanup by **9 passed in 22.34 s** for the affected replay
and ridging tests.  The non-vacuous source-order test is
`test_nemo_si3_phase2_adv2d_gate.py:152-177`; replay and arm derivation tests
are `test_nemo_si3_phase2_rung34_moment_replay.py:28-57`; per-field register,
JIT/eager, and row-plant tests are
`test_nemo_si3_phase2_rung34_trajectory_gate.py:76-134`; and the real
clamp/category guards are `test_si3_jpl1_ridging.py:137-185`.  Direct
application of the inline-coefficient and hardcoded-constant detectors reports
`inline=[]` and `hardcoded=[]` for each of the nine touched Python paths.
The implemented ordering is visible at `transport.py:813-865`, branch
instrumentation at `ridging.py:637-714,899`, the row-level plant at
`nemo_si3_phase2_rung34_trajectory_gate.py:323-342`, and the full walk at
`:414-545`.  Ruff is clean on the same set with only the repository's
Fortran-name `N803/N806` exceptions.

### Round-10 aEVP source materialization: exact operator, full-card DEBT

**Headline result: the C-grid aEVP operator is BIT-EXACT from exact oracle
intensive inputs, but rung 3.3 remains DEBT end to end.**  The completed-step
condition for continuing through steps 2--720 is therefore REFUTED, and no new
long walk was run.  This section supersedes the Round-4 statement that the
compiled association itself was unavoidable debt; it does not rewrite the
historical measurements that exposed it.

The one shared rounding implementation was brought from SI3-thermodynamics
commit `86a8eb21d189`; the direct identity/JIT/gradient test is byte-for-byte
the GYRE test from `2a7b1f7ae258`.  The implementation of
`nemo_source_round` was not changed.  Its docstring alone now states the
measured contract correctly: `optimization_barrier` by itself does not force
the required materialization; the finite-classification `select` plus
`copysign` identity is the operative compiled guard.  The shared scalar-libm
transcendental module and its test are byte-for-byte from GYRE commit
`61180a6776c4`; the compatible `PrecisionPolicy.transcendentals` field was
ported onto this branch.  Native transcendental evaluation remains the global
default.  Only the two NEMO C-grid testcase cards select `fp64/libm`.

The production C-grid arm uses the shared helper after each written operation
registered in the Round-10 preregistration: F shear
(`icedyn_rhg_evp.F90:392-399`), T shear/divergence/tension/delta (`:401-424`),
the delta floor and `P/delta` halo (`:423-427`), the separately recomputed wide
T operands (`:430-440`), adaptive alpha/beta (`:442-477`), T/F stress updates
(`:457-489`), stress divergence (`:493-510`), cross velocities (`:512-514`),
drag/Coriolis/RHS/mass velocity updates (`:530-737`), and the full periodic
halo exchange (`:634,741`).  In particular, `stress12` now literally computes
`1/(alpha_f+1)` and then multiplies the numerator, matching `:476-489`; direct
division was the final one-ULP subcycle-2 discrepancy.  This is one shared
C-grid implementation with no card switch.  The old association exists only
as a private one-variable replay ablation.

The final CPU/fp64 production-JIT replay starts with the exact bytes in oracle
entry `kt=1`.  After subcycle 1, the first active stress update at subcycle 2,
and all 100 subcycles, every registered carry is byte-exact:

| registered aEVP row after 100 subcycles | nonzero / cells | max ULP | status |
|---|---:|---:|---|
| `u_ice` | `0 / 9801` | 0 | BIT-EXACT |
| `v_ice` | `0 / 9801` | 0 | BIT-EXACT |
| `stress1_i` | `0 / 9801` | 0 | BIT-EXACT |
| `stress2_i` | `0 / 9801` | 0 | BIT-EXACT |
| `stress12_i` | `0 / 9801` | 0 | BIT-EXACT |

There is consequently no first non-bit-exact internal aEVP operand for exact
oracle inputs.  The independent NumPy written-order replay, production JIT,
and NEMO endpoint agree on all five carries.  The private ablation binds:
removing only `nemo_source_round` makes all five rows nonzero; the three stress
maxima are `3.1377567211166024e-11`, `2.4726887204451486e-11`, and
`7.474909580196254e-12` for stress1, stress2, and stress12.  The existing A-grid
preservation guard remains byte-identical for both pre-existing paths: EVP
SHA256 `e4ecb3561c33b7d246ef8122807dcb7c9f690dc96c20dd6d9e5a024d0b9351d1`
and mEVP SHA256
`83b4727c5d1d35698368a1697f21b87a537dd424678bb4aa431c96190b840282`.

The full card exposes an earlier representation boundary.  Its prognostic
transport state stores extensive `field*cell_area`, then the dynamics forcing
recovers intensives by division.  Against the original oracle intensives that
round trip is exact for `a_i`, `v_ip`, and `v_il`, but changes `v_i` in
`164 / 10609` cells by at most one ULP (`1.1102230246251565e-16`) and `v_s` in
`172 / 10609` cells by at most one ULP (`1.3877787807814457e-17`).  Feeding
only those bridged inputs through the now-exact aEVP operator is sufficient to
produce stress maxima `3.2741809263825417e-11`,
`4.5361048250924796e-11`, and `6.30961949354969e-12` after 100 subcycles.
The actual completed-step gate remains DEBT in exactly the stress family:

| full-card completed-step row | max absolute | normalized / relative | status |
|---|---:|---:|---|
| `stress1_i` | `6.02540239924565e-11` | `2.543362923424347e-14 / 2.543362923424347e-14` | DEBT |
| `stress2_i` | `3.1036506698001176e-11` | `2.5962997588627242e-14 / 2.5962997588627242e-14` | DEBT |
| `stress12_i` | `7.275957614183426e-12` | `1.1988931990227963e-14 / 1.1988931990227963e-14` | DEBT |

Thus the old 2/3/3-ULP compiled-stress association is **CONFIRMED fixed**.
The first full-model discrepancy is instead a **CONFIRMED remaining
representation defect** before rheology.  The Round-9 720-step separation
therefore includes amplification downstream of a real nonexact input, not an
unavoidable source-association floor.  Whether this bridge exclusively owns
the entire old step-720 state is UNMEASURED; no exclusive nonlinear mechanism
claim is made.

The preregistered transport/moment consequence is not promoted.  Round 9
already measured 160/160 endpoint moments byte-exact when the oracle U/V inputs
are supplied, but the ordinary card does not supply bit-exact dynamics inputs.
Because completed step 1 is not byte-exact, the explicit stop condition barred
the step-2--8 active-window rescore and a new step-9--720/restart walk.  The
external `round10_active_window.json`, produced during operand descent before
the stop disposition was finalized, is retained but REJECTED as acceptance
evidence.  The accepted Round-9 full walk/restart remains the only 9--720
trajectory evidence and remains loudly DEBT.

The secondary H79 arm is also measured from a WRITE-only copied-oracle dump.
NEMO forms H79 strength at `icedyn_rdgrft.F90:1048-1056`, and consumes it as
`P/(delta+rn_creepl)*zmsk` at `icedyn_rhg_evp.F90:420-427`.  On ICE_RHEO entry
frame 8, vector `np.exp` leaves `94,986 / 1,008,016` `P/delta` cells nonzero,
with maximum `0.00390625`; scalar glibc `libm.exp` gives
`0 / 1,008,016`, max zero.  This confirms the scalar-libm operand and refutes
vector math as an oracle substitute.  The broader secondary prediction of a
byte-exact ordinary active-window step is UNMEASURED under the primary stop
condition.

The copied NEMO instrumentation was confined to
`/tmp/si3-round10-nemo/BLD_clean` and
`/tmp/si3-r34-round10-BLD`; run products are under the external Round-10 roots.
No file in the shipped NEMO source or shipped testcase was modified.  The
ICE_RHEO copied-probe `kt=8` entry hash
`3ace5cd197d80c22e93a3ac84c5beebc5c2e43489dfdb57993adde2196a25dc9`
is byte-identical to the accepted `final/` frame, so the local dump did not
change the model trajectory.

Requested Rule-8/Rule-12 and control rows are explicit rather than folded into
an overall exit.  Source materialization moves five exact-input aEVP rows from
nonzero in the private arm to `0 / 9801`; it also exposes, rather than masks or
reverts for, the two nonexact extensive-bridge input rows and three full-card
stress DEBT rows.  The scored solver plant moves
`trajectory.post_step_00000001.u_ice` from clean
`3.3306690738754696e-16` normalized (AT-BAR) to
`3.896295744987288e-6` (DEBT); its self-check keys on that row transition, not
the already-red gate exit.  The Round-9 active plant remains independently
bound at row level: `active.step8.v_s` moves from AT-BAR
`2.7755575615628914e-17` to DEBT `9.999999439624929e-11`.  Exact rows print
their numerator and denominator as `0 / n`; there is no zero-denominator
division or hidden field count.

| round-10 artifact | SHA-256 |
|---|---|
| exact-input 100-subcycle replay + private ablation + bridge arm | `63923aa77f09cb00a68509168bfa773d208415b3f7956263b073fdc9e8f70bea` |
| full rung-3.3 step-1 card gate (expected nonzero exit) | `876b89129aee21b9412b6ea82a7dddb378b18e701881c13db2f906ad3a49d0dc` |
| ICE_RHEO H79 scalar-libm discriminator | `24b8279e7c4603689095e2577a576f12b475eb6af2343cca27b17e32414fc2b0` |
| rung-3.3 copied-oracle subcycle 1 / subcycle 2 dumps | `c0a5f06865cacd6940d5c53c7f0fce0f7b8e5179bda139761b5a810830b9ad15` / `d218b65e18e03a30a9d2b610e0a0ae0bf76591b781ad66078b49ccddda547641` |
| ICE_RHEO copied-oracle subcycle-1 dump | `367a8fb7fab87825d425eec32250427517681c713fbc5e33164e8e8433edecd4` |
| shared source-rounding module / exact imported test | `2c064668a804e572970229e8360929106d7b75722fea13c7c866024691fc8d42` / `e797dfaeb2762d4297f5c9f4759d03fdbb3547c5e1dc703934d43f25d9db6302` |
| exact imported transcendental module / test | `bb324d5245160c4115094d198fd8d3ae2f58694c6cd29c2f67c94049dc572724` / `524bd716f96feb65a05eb2a345aa65334d49d2d8c523b55df53a3316d3604f9f` |

Focused CPU/fp64 verification is **103 passed in 144.05 s**.  The explicit
touched-file coefficient/constant selection is **18 passed in 1.18 s**.
The required repository-wide ratchet was also run and is honestly red:
**3,751 passed, 2 skipped, 7 failed**.  All seven failures are outside this
lane's diff: the same two inline-coefficient and five hardcoded-constant rows
already listed above.  Ruff passes every Round-10 path with the repository's
legacy Fortran-name and exact-import ordering exceptions (`N803`, `N806`,
`I001`).

### Round-11 shared-core convergence

The shared precision files and their direct tests are now the canonical blobs
from GYRE commit `c83f73c23ff82cd7148a68bee020ec3568e5d1d8`, fetched from
`origin/fidelity/nemo-testcases-l2-gyre-codex2`.  Before checkout, a
whitespace-insensitive diff against the Round-10 tree classified
`precision.py` as formatting-only and `source_rounding.py` as docstring-only.
The canonical `transcendentals.py` is a functional superset: it adds the
reviewed scalar-libm `sin`/`cos` paths and their custom JVPs to this lane's
existing `exp`/`tanh` paths.  No SI3-local functional behavior was absent from
the canonical files, so there is **no request back to the GYRE lane** and no
local reapplication.

The canonical file SHA-256 values are `c2eab4ae531621dce8f15ab5e8c80ff7989831feb1471ae69dbdea8a7964272a`
(`precision.py`), `c5364ed24c14a2217e25426a485a6e498c593cabfa8c89b4a7e43cbf81a8cc12`
(`transcendentals.py`), and
`74c5cc38e212438bad89bef2b23856550a0f105291f430e0e058c05a53b98b17`
(`source_rounding.py`).  Their direct CPU/fp64 suite is **80 passed**.  The
post-swap production-JIT aEVP exact-input replay is unchanged: each of
`u_ice`, `v_ice`, `stress1_i`, `stress2_i`, and `stress12_i` remains
`0 / 9801` non-bit-exact cells after 100 subcycles.  The replay artifact is
`round11_core_swap_replay.json`, SHA-256
`63923aa77f09cb00a68509168bfa773d208415b3f7956263b073fdc9e8f70bea`.
This **CONFIRMS** that converging the core blobs changes no aEVP output bit for
exact oracle inputs.

### Round-11 intensive bridge and conditional stop

The bridge prediction is **CONFIRMED for dynamics and REFUTED end to end**.
NEMO aggregates the category intensives directly at `icevar.F90:123-132`, and
dynALL passes those values to rheology before transport at
`icedyn.F90:130-135`.  Prather alone forms `z0* = field * e1e2t` at
`icedyn_adv_pra.F90:218-245` and recovers fields in the written order
`z0 * r1_e1e2t * tmask` at `:355-381`.  The two legoESM oracle cards now keep
intensives at their outer-step boundary, use one shared source-rounded
pack/unpack pair around the existing Prather kernel, and feed rheology the
unmodified intensive carry.  The rung-3.3 gate now constructs its acceptance
state from the pinned oracle entry bytes and the card constructor now requires
that entry frame; there is no analytic acceptance fallback.  Rung-3.1/3.2
transport APIs and all existing ice defaults are unchanged.

The one-variable legacy arm remains the Round-10/11 replay's
`field * 9,000,000 / 9,000,000` bridge: it moves `v_i` in `164 / 10609` cells
and `v_s` in `172 / 10609`, then produces nonzero stresses.  Removing only
that pre-rheology trip makes the production-JIT completed-step-1 dynamics
rows byte-exact:

| rung-3.3 completed-step-1 dynamics row | nonzero / cells | max absolute |
|---|---:|---:|
| `u_ice` | `0 / 9801` | 0 |
| `v_ice` | `0 / 9801` | 0 |
| `stress1_i` | `0 / 9801` | 0 |
| `stress2_i` | `0 / 9801` | 0 |
| `stress12_i` | `0 / 9801` | 0 |

The result is not end-to-end bit-exact.  Fourteen transported-field rows are
nonzero at step 1, all below the immutable normalized bar; representative
counts are `v_i 357 / 9801`, `v_s 356 / 9801`, and `a_i 428 / 9801` under the
whole-step JIT.  At completed step 2 those inputs reach the exact aEVP arm and
`stress1_i` is the first DEBT row: max absolute
`1.0129497240995988e-10`, normalized/relative
`1.61972013779979e-13`.  The rung-3.3 production-JIT continuation completed
through its 485-step restart; the requested early window has 0, 3, 4, 4, 3,
7, 7, and 12 DEBT rows at steps 1--8 respectively.  At step 8 the dominant
row is `stress12_i`, normalized/relative `1.8964858428467975e-11`.
This **CONFIRMS** compiled Prather transport/field recovery as the next owner
boundary after the now-exact dynamics operator; it does not assign a narrower
source statement without a discriminator.

The full-size active step-8 window remains **DEBT** under production JIT.  Its
dominant row is `active_moment.sxxe_l01`, `4.072356563078472e-9` normalized
and `2.443033104890113e-8` relative.  Sixty of 160 moment rows are DEBT.
`stress1_i`, `stress2_i`, and `stress12_i` are also DEBT at
`3.0033808251680753e-15`, `2.8422262044713913e-15`, and
`8.864931383506739e-15` normalized, while both velocity rows remain AT-BAR.

The queued JIT discriminator changes the ownership result.  With oracle U/V,
the production JIT still reports **60/160 DEBT** moments and **140/160 over
two ULP**, led by `syye_l01` at `2.402933046985467e-9` normalized and
`3.739057008472341e-8` relative.  Thus the Round-9 eager `160/160` result does
not transfer to the production execution path: aEVP U/V is not the exclusive
moment owner, and XLA association inside Prather is the explicit next owner.
The velocity plant's clean-vs-planted delta reports
`over_two_ulp_count=140`; its owner `plant_delta.syye_l01` moves by
`6.65614941427987e-7`, and `--plant` asserts the count then exits 1.  The
active trajectory plant independently moves `active.step8.v_s` from AT-BAR
`2.7755575615628914e-17` to DEBT `9.999999439624929e-11` and exits 1.  The
rung-3.3 internal solver plant moves its scored U row to
`3.896295745042799e-6` normalized.  These are row-level Rule-8/Rule-12 controls,
not inferences from already-red clean exits.

Because completed step 1 is not bit-exact end to end, the preregistered
condition for a new ICE_RHEO steps 9--720/restart walk is **REFUTED**.  No new
720-step trajectory was run and no claim is made that the old step-720
`u_ice` 24% / `stress12_i` 77% relative separation is exclusively amplified
roundoff.  The next required owner is the first compiled Prather operation
that separates from the written-order replay; the historical Round-9 walk
remains DEBT evidence only.

| round-11 artifact | SHA-256 |
|---|---|
| canonical-core exact-input replay | `63923aa77f09cb00a68509168bfa773d208415b3f7956263b073fdc9e8f70bea` |
| rung-3.3 step-1 whole-JIT gate | `13840218d73645130451b7a0d76033ccd4f2b1de9248ba4582b7a96d0b214ac7` |
| rung-3.3 485-step whole-JIT trajectory/restart | `0569ef972f46301ff6b20ddd1e0dafc237dfab3815060c6a8501cbeb42388ff6` |
| ICE_RHEO active step-8 whole-JIT gate | `9b3f412b99eb77a1e42e4be6f4e34679379878ebd61838057725f034372ba89b` |
| ICE_RHEO active row plant | `1651f0d24172bfde7ef2ec1c37a2fc385090b88d677429c1aee1ffe338f02e38` |
| JIT oracle-U/V moment discriminator / planted invocation | `9406a2e63b4d7f7a400ad4cbb1df18045cb21c117f688f29ffece72aa7a6635d` / same bytes; both clean debt and planted mode exit 1, while planted mode selects and asserts its independent clean-vs-planted delta rows |
| A-grid EVP/mEVP preservation guard | `6f5fd71d159d7e0edc2219d7d76ef06f323662f4e5e557f147a8e193a8ecc317` |

Focused CPU/fp64 verification is **119 passed in 142.93 s**.  The checkout-
relative A-grid preservation guard passes for both EVP and mEVP; their state
hashes remain
`e4ecb3561c33b7d246ef8122807dcb7c9f690dc96c20dd6d9e5a024d0b9351d1`
and `83b4727c5d1d35698368a1697f21b87a537dd424678bb4aa431c96190b840282`,
respectively.  Ruff passes all Round-11 Python paths with the repository's
established `N803`, `N806`, and
`I001` exceptions.  The required repository-wide coefficient/constant ratchet
remains honestly red at **3,751 passed, 2 skipped, 7 failed**: the same two
inline-coefficient and five hardcoded-constant failures outside this round's
diff that are enumerated above.  No ratchet failure names a Round-11 changed
file.

The interrupted zero-byte `round11_moment_replay_jit.json` is retained and
**REJECTED**; it is not cited as evidence.  No background process remains.
No shipped NEMO source or testcase was modified, and no artifact was deleted.

### Round-12 Prather source exactness and pre-Round-13 measurements

The primary production-JIT discriminator **CONFIRMS** the registered Prather
prediction: with the oracle step-8 U/V inputs, all **160 / 160** restart-moment
rows are byte-exact and zero rows exceed two ULP.  Every row reports
`0 / 1000000` non-bit-exact cells.  Replacing only
`transport.nemo_source_round` by identity in the private ablation leaves just
20/160 rows exact, makes 140/160 non-bit-exact and over two ULP, and puts
60/160 rows in DEBT.  Its owner is `sxxe_l01`, `3.4020464778627968e-9`
normalized / `2.0409097388836545e-8` relative.  This is a one-variable
**CONFIRMED** compiled-association owner, not an inference from endpoint
growth.

The one shared Prather program now materializes every executed NEMO-written
operation in `icedyn_adv_pra.F90`: entry extensive contents at `:218-245`;
the x/y limiters and their ordered `MIN`/`MAX`/`ABS` operands at
`:534-568,757-791`; positive flux and donor recurrences at
`:570-597,793-820`; negative flux and donor recurrences at
`:618-664,841-888`; receiver merges at `:666-707,890-931`; parity/sweep
recursion at `:253-350`; halos at `:432-480`; and intensive recovery at
`:355-381`.  Products, quotients, sums, and differences are consumed in that
source order through the canonical, unchanged
`core.source_rounding.nemo_source_round`.  The implementation remains the
existing `packages/ice/legoesm/ice/transport.py` arm; there is no second model,
card selector, or default change.  The public H79 keyword is now the documented
`source_exact` (default false); only the already-selected C-grid caller enables
it.

The completed rung-3.3 step 1 is also byte-exact end to end under CPU/fp64/
scalar-libm production JIT: all 23 scored trajectory fields, including every
transported field, U/V, and the three stresses, report `0 / 9801`.  Preserving
the already-packed bit pattern when NEMO's Hbig, Hsnow, or `ice_var_zapneg`
predicate is false removes the former artificial content/area/content no-op
round trips (`icedyn_adv_pra.F90:946-1133`; `icevar.F90:759-837`).  This is
source branch semantics, not a stabilizer.

The independent-oracle-entry sweep measures the next boundary as follows; all
rows remain below the immutable normalized bar through step 8:

| completed step | first/nonzero row | nonzero / cells | normalized max | relative max |
|---:|---|---:|---:|---:|
| 1 | none | `0 / 9801` for all 23 | 0 | 0 |
| 2 | `a_ip` | `2 / 9801` | `2.7755575615628914e-17` | `1.5322831957764684e-16` |
| 3 | `a_ip` | `2 / 9801` | `2.7755575615628914e-17` | `1.5239174778496237e-16` |
| 4 | `a_ip` | `2 / 9801` | `2.7755575615628914e-17` | `1.5228419408782595e-16` |
| 5 | `a_ip` | `4 / 9801` | `2.7755575615628914e-17` | `1.5224112534471245e-16` |
| 6 | `a_ip` | `6 / 9801` | `2.7755575615628914e-17` | `1.516088615795205e-16` |
| 7 | `a_ip` | `10 / 9801` | `2.7755575615628914e-17` | `1.5098624118666638e-16` |
| 8 | `a_ip` | `6 / 9801` | `5.551115123125783e-17` | `3.0191683285628125e-16` |

These are the historical pre-correction measurements.  Round 13 located the
first `a_ip` loss at NEMO's intensive-recovery/correction ordering
(`icedyn_adv_pra.F90:355-381,405-421`), corrected that ordering, and reran the
whole gate.  The current result is 11,155/11,155 ordinary boundary rows
bit-exact through 485 steps; the final restart retains 33/80 DEBT moment rows.

The full-size active-ridging window does not satisfy the registered exactness
condition.  Its dominant debt is again
`active_moment.sxxe_l01`, `3.492459543785742e-9` normalized /
`2.095149123314519e-8` relative; 63/210 rows are DEBT.  The first DEBT row in
field registry order is `stress1_i` at `3.0033808251680753e-15`; stress2 and
stress12 are `2.8422262044713913e-15` and
`8.864931383506739e-15`.  U/V remain AT-BAR but are non-bit-exact in
`901299 / 1000000` and `938096 / 1000000` cells, with relative diagnostics
`7.751535113774344e-15` and `7.687361842728817e-15`.  Because the oracle-U/V
Prather arm is 160/160 exact, Round 12 named the full-size ICE_RHEO C-grid
aEVP velocity/stress chain as the next discriminator, not as a measured
defect.  The Round-13 exact-input probe below **REFUTES** an aEVP operator
defect at the strongly deforming step-9 state.  The moment amplification
remains **CONFIRMED**.

The earlier self-imposed condition that the active window also be byte-exact
was stricter than the dispatch and is **WITHDRAWN before the long run** in the
committed preregistration correction.  The user's actual condition was exact
completed step 1, which is satisfied.  The production-JIT ICE_RHEO walk
therefore ran continuously from the aligned step-8 state through the final
step-720 restart.

The corrected preregistered forecast is **CONFIRMED**: within the long-walk
registry, completed step 9 is already over bar and `stress12_i` is the
normalized owner at `1.2387066062187695e-14`; stress1 and stress2 are
`5.0023523591146394e-15` and `3.92220897114362e-15`.  The first non-bit-exact
ordinary rows at step 9 include `v_i` (`249816 / 1000000`) and `v_s`
(`243161 / 1000000`); U/V are non-bit-exact in `928059 / 1000000` and
`953428 / 1000000` cells.  This means the carried trajectory does not begin
from a merely later amplified bit-exact operator sequence.

Per-field first-over-bar coverage is complete:

| fields | first completed step over bar |
|---|---:|
| `stress1_i`, `stress2_i`, `stress12_i` | 9 |
| `e_s_l01`--`e_s_l05` | 11 |
| `szv_i_l01`--`szv_i_l10` | 12 |
| `e_i_l01`--`e_i_l10` | 13 |
| `v_i` | 14 |
| `u_ice` | 19 |
| `v_ice` | 20 |
| `a_i` | 24 |
| `v_s` | 25 |
| `t_su`, `sv_i` | never; byte-exact at restart |
| `oa_i`, `a_ip`, `v_ip`, `v_il` | UNINFORMATIVE; oracle identically zero |

The requested growth table reports `normalized / relative`; the immutable bar
uses the first number, `max_abs/max(max|oracle|,1)`, while the second is a
diagnostic:

| step | `u_ice` | `v_ice` | `stress1_i` | `stress2_i` | `stress12_i` | `a_i` | `v_i` | `v_s` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | `8.841e-21 / 1.130e-14` | `6.194e-21 / 1.312e-14` | `5.002e-15 / 5.002e-15` | `3.922e-15 / 3.922e-15` | `1.239e-14 / 1.239e-14` | `0 / 0` | `1.121e-16 / 1.121e-16` | `1.388e-17 / 1.402e-16` |
| 10 | `2.614e-20 / 2.591e-14` | `1.458e-20 / 2.445e-14` | `1.618e-14 / 1.618e-14` | `1.155e-14 / 1.155e-14` | `1.569e-14 / 1.569e-14` | `0 / 0` | `3.364e-16 / 3.364e-16` | `4.163e-17 / 4.205e-16` |
| 50 | `8.056e-7 / 2.242e-2` | `4.328e-7 / 2.636e-2` | `3.838e-2 / 3.838e-2` | `3.501e-2 / 3.501e-2` | `4.789e-2 / 4.789e-2` | `5.064e-8 / 5.115e-8` | `6.259e-8 / 6.259e-8` | `6.196e-9 / 6.259e-8` |
| 100 | `1.786e-8 / 1.030e-4` | `2.536e-8 / 3.430e-4` | `2.830e-4 / 2.830e-4` | `3.794e-4 / 3.794e-4` | `4.306e-4 / 4.306e-4` | `1.137e-7 / 1.149e-7` | `1.806e-7 / 1.806e-7` | `1.788e-8 / 1.806e-7` |
| 200 | `8.457e-6 / 6.139e-3` | `1.450e-5 / 2.509e-2` | `2.404e-2 / 2.404e-2` | `1.155e-1 / 1.155e-1` | `5.882e-2 / 5.882e-2` | `1.518e-6 / 1.534e-6` | `1.549e-6 / 1.549e-6` | `1.534e-7 / 1.549e-6` |
| 485 | `3.940e-3 / 2.620e-2` | `4.292e-3 / 4.086e-2` | `7.041e-2 / 7.041e-2` | `3.411e-1 / 3.411e-1` | `3.658e-1 / 3.658e-1` | `6.322e-3 / 6.342e-3` | `6.016e-3 / 6.016e-3` | `6.370e-4 / 6.016e-3` |
| 720 | `8.380e-2 / 2.343e-1` | `8.239e-2 / 2.540e-1` | `1.375e-1 / 1.375e-1` | `2.266e-1 / 2.266e-1` | `7.460e-1 / 7.460e-1` | `4.072e-2 / 4.084e-2` | `2.807e-2 / 2.807e-2` | `3.736e-3 / 2.807e-2` |

The final restart validation scores all 39 ordinary rows and all 160 Prather
moment rows: 33/39 ordinary rows and 140/160 moments are DEBT; `t_su` and
`sv_i` are byte-exact, and the four age/pond state rows remain UNINFORMATIVE.
The final moment owner is `restart_moment.sxya`, normalized and relative
`1.1916055671330519`, with `994065 / 1000000` non-bit-exact cells.  Thus the
old endpoint scale is reproduced after removing the compiled Prather defect:
`u_ice` is 23.427% relative and `stress12_i` 74.598% relative at step 720.

This Round-12 mechanism sentence is **WITHDRAWN** by the Round-13
discriminator.  The isolated Prather operator is byte-exact with oracle U/V,
and the full C-grid aEVP operator is also byte-exact at every registered source
boundary when supplied the exact strongly deforming oracle entry.  The
endpoint is therefore measured amplification of an already-present fp64
trajectory seed, not evidence of a remaining full-size aEVP algorithm defect.
Whether such a seed is mathematically unavoidable is **UNMEASURED**; its
measured starting magnitude is at the fp64 precision floor.

Both plants bind at row level.  The Prather velocity plant makes
`plant_delta.syye_l01` DEBT at `6.656046011101788e-7` normalized,
`3 / 1000000` cells, and `49112950923852` maximum ULP; `--plant` asserts that
exact row and returns 1.  Independently, the active field plant moves
`active.step8.v_s` from `1.3877787807814457e-17` normalized (AT-BAR) to
`9.999999439624929e-11` (DEBT) and returns 1.

| round-12 artifact/file | SHA-256 |
|---|---|
| production-JIT exact-input discriminator + private ablation + binding plant | `bb8a26cda75cb782c1cff99bd636f452e4062ae5f6095b182db2302c16f9c746` |
| rung-3.3 completed-step-1 gate | `883db2d0aaaf3ecec39da1787de0ec2ccabec9c35fc8222c23e8fb9e668162fe` |
| rung-3.3 independent-entry 485-step/restart sweep | `43aa822f971728421e4ac2605c0cee87e1985e0737f3107cdabdb6679f26f693` |
| ICE_RHEO active step-8 gate / row plant | `8d06bc76b7ed3b6350b0180a51ddf1ac2a9d9507b50b692750b67eeb2c158edc` / `d6bd5840502763e996959a4082d970571981d8b898719ca8619decf074f5465f` |
| ICE_RHEO production-JIT steps 9--720 + final restart | `0069a67d229b6f4e498cf47b161135af9e1f8a0f28ca5cd90f6e78870581a28b` |
| shared ice transport / source-correction recipe | `ac26cca22c0ba3c4d8aaa0b60553d914af7e0a875f1f3aba5d64be0eb387f710` / `ce50898330c07db5794abd14a785e515af46c4add64e1ca57ed29f1cdec96678` |
| H79 API / C-grid caller | `dacc1528d5660d8bc1c18b1fad7ade48ff2c5fbcd85e27d7697bb4ab04984174` / `aedfc926c3017c469d42402f4d2ff2406985784526feeab8f19a9dcec7f253c2` |
| exact discriminator gate / association test | `f3a943dd03f44304035a48cef00ea475bd810ed2d6371da8ee51829111cd3982` / `bf430a12141411ca85b82546cff4158dfe470af15a4ad34d54c4247930d013e4` |

The preregistration, implementation, and pre-run condition-correction commits
are `a1859b95d3b3`, `02dc6f312671`, and `9c265726375c`.  Focused CPU/fp64 verification is
**63 passed in 654.33 s**, plus the new compiled-association tripwire
**1 passed in 4.06 s**.  Every touched-file coefficient/constant ratchet node
passes (**10 selected nodes**: nine in the implementation commit and the new
test node).  The repository-wide ratchet remains honestly red at
**3,751 passed, 2 skipped, 7 failed**; the same seven unrelated paths are
listed in the Round-10/11 receipt and no failure names a Round-12 file.  Ruff
passes every Round-12 Python path with the established `N803`, `N806`, `I001`
exceptions.  No background process remains, no shipped NEMO source/testcase
was modified, no artifact was deleted, and the canonical core files are
unchanged.

### Round-13 active-regime discriminator and completed walk

#### Crash recovery and artifact audit

The 2026-09-05 `/tmp` exhaustion interrupted this round without a clean exit.
The checkout's native worktree metadata still named `5401f02a7548`, while the
lane's required `/tmp/codex-si3dyn-git` metadata named the three completed
Round-13 commits through `dec3700da55a`; the latter is authoritative. Every
Round-13 Python source was reread and AST-parsed, both edited campaign
documents were reread, all surviving JSON was parsed end to end, and the copied
`icedyn_rhg_evp.F90` instrument source reaches its normal module end. The setup
dump, all 100 numbered subcycle dumps, and the step-9 entry frame are present;
the subcycle dumps are uniformly 387,078,552 bytes. **No surviving file was
truncated.** The interrupted
`ice_rheo/round13/full_walk_intensive_order.json` was absent, so it was the
only lost output and was rerun in the foreground through step 720 and the
final restart. No `run.sh` was handed off, and no background process remains.

#### Exact-input active aEVP discriminator

The preregistered defect prediction is **REFUTED**. A WRITE-only instrument in
a copy of the NEMO source (the shipped tree is unchanged) records the 48
registered source-boundary families for every one of the 100 subcycles at the
strongly deforming completed-step-9 state. The registry covers deformation,
delta/floor, pressure and viscosities, and all three alpha/beta stress updates
from `icedyn_rhg_evp.F90:392-489`; stress divergence and cross-grid velocity
from `:493-514`; and the ordered force, mass/drag/Coriolis, raw velocity,
pre-halo, and post-halo paths from the even and odd updates at `:532-634` and
`:638-741`. With exact oracle U/V, stresses, extensive state, and forcing,
CPU/fp64/scalar-libm production JIT is byte-exact for **4,800 / 4,800**
operand-family/subcycle rows. Every family has zero nonexact subcycles and
maximum ULP zero. The five production endpoints (`u`, `v`, `stress1`,
`stress2`, `stress12`) are each `0 / 1,000,000`, and the instrumented and
uninstrumented endpoints are likewise five times `0 / 1,000,000`.

The control binds to a scored row independently of the overall exit status:
`subcycle001.force_u` moves from `0 / 1,000,000`, BIT-EXACT, to
`1 / 1,000,000`, DEBT, with one ULP (`5e-324` absolute). The plant asserts
that exact row transition and exits 1.

Therefore the former phrase "remaining full-size aEVP defect" is
**WITHDRAWN**. The active C-grid aEVP implementation has no measured
source-boundary defect when it receives exact inputs. The 720-step endpoint is
**CONFIRMED amplification of a measured fp64-scale carried-state seed**, not
evidence for a distinct aEVP algorithm error. Calling that seed
mathematically unavoidable, or calling the trajectory chaotic, would exceed
the measurement and remains **UNMEASURED**. The endpoint-effective,
non-monotone stress12 growth from step 9 to 720 is 0.04462597985810006 per
step, or a 4.56366975759781% factor per step (0.06438167983609172 bits/step);
this is an endpoint exponent, not a Lyapunov exponent or a mechanism claim.

#### Secondary transport-order result

The historical rung-3.3 `a_ip` one-ULP row was not pond physics. The
one-variable arm shows its extensive Prather result is exact and the bit is
lost only when the card performs correction arithmetic before recovering the
intensive field. NEMO first divides extensive contents back to intensive
fields at `icedyn_adv_pra.F90:355-381`, then applies Hbig, Hsnow, and zapneg
corrections at `:405-421`. The shared card path now preserves that literal
order and avoids an extra extensive/intensive round trip. After the change,
the rung-3.3 gate has **11,155 / 11,155** ordinary trajectory rows byte-exact
through all 485 boundaries. The final restart remains honestly DEBT in 33/80
moment rows; this change does not relabel those rows.

#### ICE_RHEO steps 9--720 and restart

The crash-replacement walk completed all 712 requested transitions from the
aligned completed-step-8 state through step 720 under production JIT,
CPU/fp64, and scalar libm. Within that long-walk register, the first
non-bit-exact row at completed step 9 is `v_i`, `249816 / 1000000`, normalized
and relative `1.1214373873213058e-16` (AT-BAR). All three stresses are already
DEBT at step 9: stress1 `5.0023523591146394e-15`, stress2
`3.92220897114362e-15`, and stress12 `1.2387066062187695e-14`. The exact-input
discriminator above proves those full-card errors arrive through a carried
input difference rather than being created by the isolated aEVP operator. The
first full-card operation that creates that carried seed is the explicitly
named next owner and is **UNMEASURED** at this stopping boundary.

The per-field first-over-bar table and normalized/relative growth table above
remain numerically unchanged in the crash-replacement artifact. At step 720,
`u_ice` is 23.42708457% relative, `v_ice` 25.40449316%, and `stress12_i`
74.59834626%. Final-restart coverage is 39/39 ordinary prognostic rows plus
160/160 Prather moment rows: 33/39 ordinary and 140/160 moment rows are DEBT;
`t_su` and `sv_i` are byte-exact; the four oracle-zero age/pond receiver rows
remain UNINFORMATIVE. The restart owner is `restart_moment.sxya`, normalized
and relative `1.1916055671330519`, with `994065 / 1000000` non-bit-exact cells.
The scientific gate exits 1, as required for these DEBT rows.

| Round-13 artifact/file | SHA-256 |
|---|---|
| exact-input active aEVP probe | `912087782bfa304aba21945938af90bb1192516cbcc073a5fd92d2c93e2e10a5` |
| binding active aEVP row plant | `490aea91dedd53a6810ba6eed1fada2f88457ff5532a658002dbbd26692b1915` |
| active completed-step-8 window | `8d06bc76b7ed3b6350b0180a51ddf1ac2a9d9507b50b692750b67eeb2c158edc` |
| production-JIT steps 9--720 plus restart | `be085a8714b987116403df9a01b47631853059f5879e0b06665ef8d3d9c28fc9` |
| corrected rung-3.3 485-boundary trajectory/restart | `e6d937b408d7000802db4fb0952ca0cd93c5b793d452c5f2293f2528cb411074` |
| copied instrument source `icedyn_rhg_evp.F90` | `f1c92a8e815f28e9178acdd8e9e307f598b8057b329e8c2e2abece627271b3fa` |
| copied NEMO executable | `6509d09e3997f9d0a3b936a1bdd96ec67c712277b70cd53e7726acdbd43efe3f` |
| setup / subcycle 1 / subcycle 100 dumps | `308fc3cc490e2e7769f15e8e1b2c4ad6b06369d6b9d35ad846bba9f5909622a7` / `75374a29372213787d1c26e318348424be45606d82d81a77ca6813ef4a3a108a` / `2265774f3ad86808dea8a91426502b534d377f4d0208335d0dc4b679d6102380` |
| copied step-9 oracle entry frame | `cdda1ab4322e869e5075a9b90a5005217dd2ed472c101b44170833b27fe6732a` |

The preregistration, exact-input probe, and source-order commits are
`59915bcf8949`, `918dcdfe74f3`, and `dec3700da55a`. The combined CPU/fp64
science-and-ratchet run reports **2,697 passed, 2 skipped, 8 failed**; all
science tests pass. The eight failures are the two discovery-sanity nodes
caused by the intentionally restricted test invocation, the inline-ratchet
budget roster, and pre-existing hardcoded-constant findings in unrelated
FV3/core/grid/ocean files. Re-running the ratchets on every Round-13 touched
Python file with the full package path gives **3/3 selected inline nodes** and
**6/6 selected hardcoded-constant nodes** passing. The new active-probe unit
tests pass 2/2 and the trajectory-gate tests pass 10/10. A crash-recovery
focused rerun exposed and corrected one cross-rung test assertion: the
rung-3.1 gate's `first_divergence` correctly names restart-moment DEBT rather
than being null. The correction is commit `3d57b234c6a3`; the complete
post-correction focused set is **26 passed in 119.78 s**. No canonical core
file changed.

### Round-14 carried-seed producer and ORCA1 coverage boundary

#### The `v_i` seed has a measured producer

The first probe **REFUTED** its preregistered surface-stress prediction. A
source-written `usrdef_sbc.F90:124-140` arm did not move `v_i`, `v_s`, energy,
or salinity, made both velocity maxima larger, and reduced only two already-red
stress maxima. The probe's initial permissive "any row improves" decision is
**WITHDRAWN**; artifact `ice_rheo/round14/seed_probe.json` is retained as a
rejected decision record, not cited as an ownership result.

The corrected one-variable arm **CONFIRMS** the producer of the Round-13
step-9 `v_i` seed. Starting from the oracle `kt=8` entry and step-7 restart
moments, replacing only the completed step-8 U/V and three stresses by their
oracle values makes source-exact Prather produce **160 / 160** exact moments.
With the shared source-rounded redistribution ledger, all **40 / 40** next-
entry fields and **160 / 160** moments are byte-exact. Disabling only the
ledger rounding restores **27 / 40** non-bit-exact fields: `v_i`, `v_s`, five
snow-energy layers, ten ice-energy layers, and ten option-4 salt layers. The
`v_i` row is `248995 / 1000000` non-bit cells with normalized/relative maximum
`1.1214373875196422e-16`; `a_i` remains `0 / 1000000` in both arms.

The producer is therefore the jpl=1 donor/survivor/receiver inventory
association at `icedyn_rdgrft.F90:715-720,734-741,779-792,874-890`: SI3
separately forms ridge and raft inventories, writes the surviving inventory as
`field * (1-afrdg-afrft)`, and adds the two receiver contributions in that
order. The shared `packages/ice/legoesm/ice/ridging.py` arm now uses the
canonical `nemo_source_round` at those source boundaries; there is no card
switch or second implementation. Its private identity-rounding ablation is the
one-variable control. This is a **CONFIRMED compiled-association defect**, not
a stored-versus-rederived-state design gap.

The source-exact ledger removes all 27 mean-inventory differences from the
full completed-step-8 result. Five carried dynamics rows remain non-bit-exact
(`u_ice`, `v_ice`, and three stresses), and the resulting Prather moments
remain non-bit-exact when those velocities are used. The tested surface-stress
arm does not own them. Their first internal step-8 forcing/aEVP source statement
remains **UNMEASURED**; no claim that the entire 720-step seed has been removed
is made, and no new 9--720 walk is claimed in this round.

The complete carry census also finds a real state-coverage gap:
`snwice_mass` and `snwice_mass_b` are nonzero in `996004 / 1000000` oracle
cells, with maxima `1848.3300187678121` and `1848.3300186039578` kg/m2, but
the card has no corresponding state leaves. NEMO explicitly carries and
updates them at `iceupdate.F90:187-193`; legoESM currently reconstructs the
instantaneous ice/snow mass inputs from volume fields and cannot represent the
before level. That is **UNMEASURED design debt**, although it is dynamically
inert for this oracle because resolved `ln_ice_embd=F`
(`final/namelist_ref:199`, `final/ocean.output:410`).

The row-level plant binds independently of the ordinary scientific debt:
`entry.step9.a_i` moves from `0 / 1000000`, BIT-EXACT, to
`1 / 1000000`, DEBT, normalized `1.000000082740371e-10`; planted mode asserts
that transition and exits 1. The first plant attempt is **REJECTED**: a
singleton category axis caused a fail-closed shape error before scoring and
wrote no artifact. Commit `e3ccffa59f3e` routes the plant through the same
category-collapsed oracle accessor as the clean row; the complete rerun supplies
the evidence above.

The queued immutable-array review finding was also **CONFIRMED and fixed**.
The failing assignment was in `nemo_si3_phase2_adv2d_replay.py:158` (the named
test file itself has only 41 lines): `np.asarray(...).copy()` now precedes
NumPy in-place Hbig replay. The two-test replay suite passes. Its historical
above-bar assertion had also become stale after source-exact Prather; the
measured row is below the normalized bar but remains a nonzero 24-ULP owner,
which is what the control now asserts. The secondary rung-3.3 `a_ip` item was
already closed by the Round-13 source-order fix: all 485 `a_ip` boundary rows
are exact and the whole ordinary register is **11155 / 11155** exact in
`ice_adv2d_rhg/round13_trajectory_intensive_order.json`.

#### ORCA1 dynamics selector coverage

The comparison below distinguishes the actual ORCA1 cfg-over-ref result from
the oracle namelist that ran. The latter is
`ice_rheo/final/namelist_ice_cfg`; its resolved values are cross-checked in
`ice_rheo/final/ocean.output:458-565,652-742`. "Covered" refers only to the
measured jpl=1 CPU/fp64 card and its recorded oracle, never to an unrun selector
cross-product. The shipped ICE_RHEO cfg itself selects `ln_rhg_EVP=F`,
`ln_aEVP=F`, and EAP on (`tests/ICE_RHEO/EXPREF/namelist_ice_cfg:48-58`);
the certified copy's aEVP values are the explicitly cited ORCA1 overlay, not a
claim about that stale shipped selector.

| selector | ORCA1 resolved value | certified oracle value | coverage |
|---|---|---|---|
| category count | `jpl=1` (`ORCA1 cfg:24`, overriding ref `:24=5`) | `1` (`final cfg:24`) | **COVERED**, jpl>1 rejected |
| vertical ice/snow layers | `3 / 3` (`ORCA1 cfg:25-26`) | `10 / 5` (`final cfg:25-26`) | **UNCOVERED cross-product** |
| thermodynamics | `ln_icethd=T` (`ORCA1 cfg:32`) | `F` (`final cfg:28`) | **UNCOVERED** |
| outer dynamics | `ln_dynALL=T` (`ORCA1 cfg:44`) | `T` (`final cfg:37`) | **COVERED** for `rhg -> adv -> rdgrft -> cor` |
| lateral boundary coefficient | `rn_ishlat=2` (ORCA1 ref `:57`) | `2` (`final cfg:41`) | **COVERED** on this all-wet periodic geometry |
| landfast L16 | `T` (`ORCA1 cfg:46`), `rn_lf_*` from ref `:59-63` | `F` (`final cfg:42`; output `:658`) | **UNMEASURED / UNCOVERED** |
| H79 strength | `T`, `rn_pstar=2.0e4`, `rn_crhg=20` (`ORCA1 cfg:52-54`) | same (`final cfg:47-49`; output `:687-689`) | **COVERED** in the landfast-off arm |
| strength smoothing | `F` (`ORCA1 cfg:55`) | `F` (`final cfg:50`) | **COVERED** |
| ridge distribution / participation | exponential / exponential (`ORCA1 cfg:57-60`) | same (`final cfg:51-58`) | **COVERED** for jpl=1 |
| ridging / rafting | `T / T`, `rn_porordg=0` (`ORCA1 cfg:61-65`) | same (`final cfg:59-68`) | **COVERED** only with `ln_icethd=F`; thermodynamic retention coupling is uncovered |
| EVP / aEVP | `EVP=T` (`ORCA1 cfg:70`), `aEVP=T`, `nn_nevp=100`, check off (ref `:108-116`) | same (`final cfg:73-81`; output `:723-727`) | **COVERED**, 100 subcycles |
| EAP | `F` (ORCA1 ref `:109`) | `F` (`final cfg:80`) | selector exclusion covered; no EAP trajectory claim |
| Prather advection | `T` (`ORCA1 cfg:75`) | `T` (`final cfg:90`; output `:742`) | **COVERED**, 32 tracers / 160 moments for this oracle |
| ice-ocean drag | `rn_Cd_io=5.0e-3` (ORCA1 ref `:136`) | same (`final cfg:97`; output `:565`) | **COVERED** for the case forcing |
| Coriolis field | normal ORCA1 spherical field; no ICE_RHEO `ln_corio=F` override | zero (`final/namelist_cfg:21`; `usrdef_hgr.F90:133-140`) | **UNCOVERED ORCA1 forcing arm** |
| coupled ocean/atmosphere forcing fields | production ORCA1 inputs | ICE_RHEO analytic wind and zero prescribed ocean current | **UNCOVERED cross-product**; only the case forcing is scored |
| melt ponds | `ln_pnd=F` (`ORCA1 cfg:151-152`) | resolved level ponds/lids on (`final/output:547-558`) | **UNINFORMATIVE / UNCOVERED**: pond state is zero here |
| salinity option | `nn_icesal=2` (`ORCA1 cfg:108`) | option 4; ten carried layers | **UNCOVERED cross-product** |

The shipped-test search is conclusive for the requested choice point:
**0 / 10** shipped `tests/*/EXPREF/namelist_ice_cfg` files set
`ln_landfast_L16`; no shipped NEMO test case exercises that selector. The L16
terms themselves are the `zvU/zvV`, `zvCr`, `ztaux_base`, `ztauy_base`, and
`tau_icebfr` branch at `icedyn_rhg_evp.F90:333-363`; the certified run takes
the explicit zero branch at `:364-373`. A future user choice is therefore
needed between a copied ICE_RHEO variant with an L16 namelist/input contract or
an ORCA2-based variant using the ORCA1 ice deck. Neither new oracle arm was
started here.

| Round-14 artifact/file | SHA-256 |
|---|---|
| rejected first seed decision record | `07129b91d063303735763ec20e287ee7bd6b07e205ad98fd497f37deb66b1099` |
| corrected seed-producer / private-ledger-ablation gate | `897af8e527dcbf33714a93736fb2473799d6b9b5b8aad1f44388bb86b70f345b` |
| binding row plant | `99ef272ff61d642b11bf9d0af9bce0609d8b190f04cfc2fba8b48a327c743255` |
| corrected rung-3.3 trajectory (`a_ip` closure) | `e6d937b408d7000802db4fb0952ca0cd93c5b793d452c5f2293f2528cb411074` |
| shared ridging implementation | `22e1e6e98bd8fce6c493b463bb48082f431a6cddf001fd0122897479a7a0635d` |
| committed Round-14 producer gate | `789fcfde9b9062dd3f3035bf7771346bdd0f6c414ab747d16f648908d3b081c2` |

The Round-14 preregistration, revised prediction, implementation, immutable-
array review fix, and row-plant repair commits are `26b62bcc966a`,
`4dd73f9eb0f0`, `5909255ba06f`, `ce330724d9aa`, and `e3ccffa59f3e`; the replay
constant-ratchet cleanup is `edd776582a0a`. Focused CPU/fp64 verification is
**30 passed in 227.57 s** for both ridging suites plus the producer-probe
controls, and **2 passed in 12.06 s** for the repaired ADV2D replay suite. The
five selected inline/hardcoded ratchet nodes pass in 0.80 s; applying the same
detectors directly to the two touched scripts reports zero hardcoded hits and
zero inline hits. Ruff passes all six touched Python paths with only the
campaign's established `N803`, `N806`, and `I001` exceptions. No NEMO
executable was run in the sandbox, no shipped source/test was changed, no file
was deleted, no canonical core file changed, and no handoff `run.sh` was needed.

## Loudly UNMEASURED / deferred

Within-step x/y Prather split states; ORCA1
option-2 salinity (the rung-3.2 oracle resolves option 4); candidate alignment
of nonzero `snwice_mass` and its
before level; separation of `zapsmall` from `zapneg`; thermodynamics;
general production run-restart integration outside the
opt-in card's now-verified restart path; coupled ice--ocean comparison; and
landfast L16 (OFF here,
**UNVERIFIED-deferred to lane 4**).  No legoESM claim is made for any item in
this paragraph. The jpl=1 redistribution statement producing the `v_i`
component of the seed is now CONFIRMED above; the first source statement
producing the remaining step-8 U/V/stress differences, and whether any residual
seed is mathematically unavoidable, remain **UNMEASURED**.

## End-of-task ASKED / UNASKED choice list

The mandatory round-7--13 choice record is tabular so scope decisions cannot be
lost inside narrative:

| round | choice | disposition | evidence/disposition |
|---|---|---|---|
| 7 | Treat the shipped-control 21-error build failure as an upstream NEMO defect; use the override-excluded copy as the oracle | ASKED | Preserved UNBUILDABLE control; no source repair |
| 7 | Run 720 oracle steps, add the in-module `jpl=1` SI3 ridge/raft arm, dynALL card, geometry/kt=1/first-divergence gates, and direct rung-3.3 gate tests | ASKED | Completed through the first-divergence stop |
| 7 | Modify/delete shipped files; fabricate a stale-control trajectory; enable landfast, thermodynamics, alternate schemes, MPI/GPU, or push | UNASKED | None performed |
| 8 | Find the first source-significant redistribution frame and score its completed step with aligned NEMO moments | ASKED | First significant completed step 8; 210 scored rows |
| 8 | Mark the four oracle-zero age/pond state rows UNINFORMATIVE | ASKED | Four state rows, and only those four, carry that label |
| 8 | Make `ato_i` a Prather tracer if supported by executed NEMO | ASKED | Premise REFUTED by `icedyn_adv_pra.F90:218-350,418-430`; no synthetic moments added |
| 8 | Measure the rung-3.3 one-ULP sensitivity or relabel its mechanism | ASKED | Chose the authorized PLAUSIBLE label with precision-floor caveat |
| 8 | Add relative diagnostics without replacing the normalized bar | ASKED | Both columns emitted; normalized max-one metric remains classificatory |
| 8 | Remove non-oracle final clamps, reject category-axis input, and calculate `sishea` from NEMO's formula | ASKED | Implemented, tested, and measured |
| 8 | Add thresholds, physical arms, moment families, default changes, or any shipped-NEMO edit beyond these choices | UNASKED | None performed |
| 9 | Bind the active plant by a scored-row transition and headline the dominant moment debt | ASKED | `active.step8.v_s` AT-BAR to DEBT; `sxxe_l01` headlines the active window |
| 9 | Census moment writers and run U-only, V-only, U/V, and written-order transported-area arms | ASKED | No between-call moment mutation; eager exact U/V plus written order is 160/160 byte-exact; the former exclusive ownership statement is superseded by the Round-11 JIT discriminator |
| 9 | Walk completed steps 9--720 with production JIT/fp64/CPU, score the restart, and scan the excessive-removal branch | ASKED | Completed; 173 informative final DEBT rows; no clamp firing in 719 available frame transitions |
| 9 | Treat zero age/pond receivers as trajectory evidence | UNASKED | Four state rows and their 20 zero moment rows remain UNINFORMATIVE; ORCA1 `ln_pnd=F` disclosed |
| 9 | Change schemes/defaults, relax the bar, infer a nonlinear owner, edit shipped NEMO, use GPU/MPI, push, or delete artifacts | UNASKED | None performed; the interrupted zero-byte artifact is retained and flagged REJECTED |
| 10 | Import the shared rounding helper and direct test at their source commits; source-round every executed C-grid aEVP statement without a card switch | ASKED | Imported from `86a8eb21d189`/`2a7b1f7ae258`; exact-input aEVP is `0 / 9801` for all five carries |
| 10 | Import the shared scalar-libm path and select it only for the two NEMO C-grid cards | ASKED | Module/test from `61180a6776c4`; global and A-grid defaults remain native |
| 10 | Bind row-level controls, enumerate exact numerators as `0 / n`, and classify the first remaining operand | ASKED | Private ablation and scored U plant bind; `v_i`/`v_s` extensive bridge is the first full-card nonexact input |
| 10 | Continue the active window and steps 9--720 if and only if completed step 1 is byte-exact | ASKED | Condition REFUTED by three stress DEBT rows; no new conditional walk or restart claim |
| 10 | Change the shared helper implementation, relax the bar, change A-grid/default schemes, edit shipped NEMO, use GPU/MPI, commit large dumps, delete artifacts, push | UNASKED | None performed; copied-build WRITE-only dumps stay external |
| 11 | Converge three shared core files and direct tests to canonical GYRE commit `c83f73c23ff8`, without locally reapplying functional differences | ASKED | Completed; no SI3-only functional behavior required a request back; exact-input replay remains five times `0 / 9801` |
| 11 | Replace the pre-rheology extensive round trip with NEMO's intensive outer-step carry and source-ordered Prather pack/unpack | ASKED | Step-1 U/V/stresses are byte-exact; transported fields remain non-bit-exact and first make stress1 DEBT at step 2 |
| 11 | Rerun the oracle-U/V discriminator under production JIT and bind the moment plant on exit | ASKED | Prior exclusive velocity ownership REFUTED; clean JIT has 60 DEBT/140 over-two-ULP rows; planted mode asserts the count and exits 1 |
| 11 | Run steps 9--720 only if completed step 1 is bit-exact end to end | ASKED | Condition REFUTED by 14 nonzero transported rows; no new 720-step/restart run |
| 11 | Modify the canonical core helper, relax bars, change defaults/schemes, edit shipped NEMO, delete artifacts, use GPU/MPI, commit large dumps, or push | UNASKED | None performed; external zero-byte interrupted artifact retained and rejected |
| 12 | Source-round the single shared SI3 Prather program in NEMO operand order and retain the old association only as a private arm | ASKED | Clean production JIT is 160/160 byte-exact; private identity arm is 20/160 exact, 140/160 nonexact, 60 DEBT |
| 12 | Require byte-exact transported fields at completed step 1, then measure steps 2--8 and the active-ridging window | ASKED | Step 1 is 23/23 exact; first nonexact is step-2 `a_ip`; active window is 63/210 DEBT |
| 12 | Bind plants to named scored rows and report `0 / n` denominators | ASKED | `plant_delta.syye_l01` and `active.step8.v_s` are asserted row transitions; exact discriminator rows are `0 / 1000000` |
| 12 | Continue ICE_RHEO steps 9--720 after the ordered step-1/steps-2--8/active measurements | ASKED | Over-strict local condition withdrawn before run; completed through the step-720 restart, 33/39 ordinary and 140/160 moment rows DEBT |
| 12 | Rename/document the public H79 source-rounding keyword without changing its default | ASKED | `source_exact=False`; only the existing C-grid caller selects true |
| 12 | Modify canonical core files, relax the bar, add a card switch/model, change defaults, edit shipped NEMO, use GPU/MPI, delete artifacts, commit large dumps, or push | UNASKED | None performed |
| 13 | Extend the copied NEMO instrument to active step 9 and discriminate all aEVP source operands for all 100 subcycles with exact oracle inputs | ASKED | 48/48 families and 4,800/4,800 family/subcycle rows byte-exact; preregistered defect arm REFUTED |
| 13 | If the exact-input aEVP path is exact, classify the endpoint as amplification, report its measured exponent, and stop chasing a solver defect | ASKED, conditional | Condition CONFIRMED; endpoint-effective stress12 factor is 4.56366975759781%/step; mathematical inevitability and chaos remain UNMEASURED |
| 13 | Locate and literally repair the secondary rung-3.3 `a_ip` one-ULP operand if cheap | ASKED | NEMO intensive recovery precedes corrections (`icedyn_adv_pra.F90:355-381,405-421`); corrected shared card order gives 11,155/11,155 ordinary boundary rows exact |
| 13 | Rerun the ICE_RHEO steps 9--720 production-JIT walk and final restart after the discriminator | ASKED | Completed after crash recovery; first long-walk nonexact row is step-9 `v_i`, final debt is 33/39 ordinary plus 140/160 moments |
| 13 | Correct the cross-rung test assertion exposed by the post-crash verification rerun | UNASKED, in-scope harness correction | Restored the rung-3.1 restart-moment first-divergence contract in `3d57b234c6a3`; post-fix focused set is 26/26 |
| 13 | Modify canonical core files, change defaults/schemes, relax the bar, edit shipped NEMO, use GPU/MPI, commit multi-MB dumps, delete artifacts, or push | UNASKED | None performed; WRITE-only dumps and JSON stay external, shipped tree untouched |
| 14 | Census the step-9 carry and name the step-8 producer of the `v_i` seed with preregistered one-variable arms | ASKED | Surface-stress prediction REFUTED; jpl=1 ridging inventory ledger CONFIRMED by 40/40 fields plus 160/160 exact moments and a 27-field private ablation |
| 14 | Fix the immutable ADV2D replay mutation and close the secondary `a_ip` item | ASKED | NumPy copy landed; replay suite 2/2; `a_ip` is exact at all 485 corrected boundaries |
| 14 | Compare ORCA1's resolved dynamics deck with the certified oracle and find a shipped landfast test | ASKED | Coverage table above; none of 10 shipped ice-test namelists enables L16 |
| 14 | Add a new landfast, multi-category, thermodynamic, pond, or salinity arm; run NEMO in the sandbox; alter canonical core/defaults/bar; delete files; use GPU/MPI; push | UNASKED | None performed; the next landfast oracle requires a user scope choice |

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
rung-3.3 trajectory gate and name its measured 3.3015375767554644%/step window;
round 8 explicitly authorized relabeling the unperturbed mechanism attribution
as PLAUSIBLE with its precision-floor caveat instead of running the optional
one-ULP sensitivity experiment.

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
no AT-BAR or causal-owner claim past the rung-3.4 first-divergence boundary
(round 9 measures the continuation as DEBT); no tolerance relaxation, GPU,
MPI, or push.
