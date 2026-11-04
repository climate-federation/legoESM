# SI3 round 29 (`si3_r29`) — O17 design and record audit receipt

Date: 2026-10-31. Base: `44ca9044917b13ac14e074223f436385a78eb058`.
No NEMO build, launch, or integration was run.

## Outcome

**ACQUISITION_NEEDED; the unified O17 producer is now scorable.** The admitted O8--O16 master has
the correct unmodified production deck, 100-step window, dynamics disposition,
thermodynamic stage frames, exchange outputs, and restart. It does not have a
complete ice-step-entry state or raw O7 meteorology at every step. The O1--O7
records do not fill those holes over the same 100-step production trajectory.

The audit in this opening section predates the Resume 1--3 implementation. The
typed composition entry and one-trajectory acquisition/scoring tooling now
exist; the operator-produced unified O17 record and its through-entry numerical
verdict still do not.

There is also no existing legoESM production entry point that performs the O17
program. The current dispatcher accepts a typed finalization forcing and calls
O15, O16, and the stress tail, while its earlier O2 forcing path raises
`NotImplementedError` (`packages/ice/legoesm/ice/sea_ice.py:642-725`). O4--O16
therefore do **not** already score through one common full-step entry point.
The exact missing piece is a typed `si3_orca2_o17` production composition entry
that orders the already admitted O4 Prather, O5 H79+aEVP, O6 redistribution and
post-dynamics correction, live O7 surface step, and O8--O16 thermodynamics and
finalization. This is composition wiring and a forcing/response contract, not a
new physical operator.

The round instruction says to stop and name the missing piece if composition
needs one. Consequently this round did not invent that entry point, did not
write an acquisition driver whose read-set would precede the missing production
contract, and did not run the requested one-step gradient program. An O17
acquisition remains required after that entry point fixes the authoritative
step-entry and live-forcing pytrees.

## Measure-first record audit

The target oracle row is the shipped `ORCA2_SAS_ICE` deck: steps 1--100 and
`rn_Dt=10800` (`cfgs/ORCA2_SAS_ICE/EXPREF/namelist_cfg:21-28`),
`nn_fsbc=1` (`cfgs/ORCA2_SAS_ICE/EXPREF/namelist_cfg:59-61`), and CPP keys
`key_si3 key_linssh key_vco_1d3d key_xios key_RK3`
(`cfgs/ORCA2_SAS_ICE/cpp_ORCA2_SAS_ICE.fcm:1`). The O8--O16 acquisition was
preregistered as the unmodified production ice namelist, production dynamics
on but outside stage projections, O7-completed surface products as its forcing
boundary, and 100 steps (`docs/ice/fidelity/si3_r18_o8_o16_master_preregister.md:3-24`).
Those identities agree with O17; the forcing boundary does not.

| Evidence | Deck / window | What it proves | O17 missing data |
|---|---|---|---|
| O4 admitted record | Prather-only diagnostic overlay, 2 steps | Five-category Prather is 0 ULP for both sweep orders; its manifest explicitly disables all other dynamics and thermodynamics (`docs/ice/fidelity/si3_r25_o4_score_receipt.md:3-6,19-29`). | Not the production deck or 100-step carry; no live O7 forcing. |
| O5 admitted record | Dynamics diagnostic overlay, 3 steps | H79+aEVP is 0 ULP on its registered three-step production-JIT comparison; thermo and redistribution are disabled (`docs/ice/fidelity/si3_r12_o5_acquisition_receipt.md:3-14,26-34`). | Not the production deck or 100-step carry. |
| O6 receipt | Local model admission only | The receipt explicitly leaves ORCA2 NEMO-record bit exactness plausible because no ORCA2 record exists (`docs/ice/fidelity/si3_r14_o6_receipt.md:252-255`). | No O6 production record. |
| O7 admitted record | Unmodified production deck, 2 steps | All 91 scored leaves at each of `kt=1,2` are 0 ULP, including composed live surface stages (`docs/ice/fidelity/si3_r13_o7_admission_receipt.md:3-11,64-90`). | Raw meteorology and computed products stop after step 2. |
| O8--O16 master | Unmodified production deck, 100 steps | Every thermodynamic boundary and the O16 exchange/tau boundary is registered; restart pairs are byte-identical (`docs/ice/fidelity/si3_r18_o8_o16_master_preregister.md:15-24,40-58`). | No full pre-dynamics ice-step-entry carry, Prather moments, dynamic stress carry, or raw O7 meteorology. |

The master schema makes the gap mechanical rather than inferential. Kind 21 is
only the thermodynamic `STATE` tuple immediately before category work; kinds
18/15/16/19/20 cover pre-aggregation, flux, and stress boundaries. Its complete
projection registry contains no step-entry or raw-meteorology kind
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o16_master_gate.py:93-100,136-238`).
O8--O12, O13, O14, and O15/O16 nevertheless remain individually admitted at
0 ULP over all 100 steps (`docs/ice/fidelity/si3_r21_o8_o12_score_receipt.md:3-13`,
`docs/ice/fidelity/si3_r22_o13_receipt.md:3-8`,
`docs/ice/fidelity/si3_r23_o14_score_receipt.md:3-15`, and
`docs/ice/fidelity/si3_r24_o15_o16_receipt.md:3-19`). This audit retracts none
of those operator verdicts; it says only that their separate exact-input views
cannot be relabelled as O17 composition evidence.

## Frozen O17 score contract

The design row now freezes the full O17 selector card. In particular,
`core2_precipitation_closure=False` is explicit for the NEMO-comparison path;
the library default is `True` and its source says comparison configurations may
disable it (`packages/ice/legoesm/ice/config.py:524-529`). Dynamics and
thermodynamics are explicitly live in the O17 `SI3ORCA2StateConfig`; the current
state carrier otherwise defaults both flags off for the O1 overlay
(`packages/ice/legoesm/ice/si3_orca2_state.py:92-131`).

One production program must produce two non-interchangeable reports:

1. `INDEPENDENT`: initialize once from the O17 initial state, advance all 100
   steps with live O7 products, and compare every recorded field at every step.
2. `GIVEN-NEMO-ENTRY`: for each step, replace the complete step-entry state and
   live-forcing inputs with that step's NEMO record, execute exactly one same
   production step, and compare every recorded field. No output of this label
   may seed `INDEPENDENT`.

The local bar is exact bits, including signed zero. The trajectory bar is the
already frozen pointwise normalized `1e-15`. The first normal-range non-bit row
is `DEBT` and cites its first admitted owner. The window is all 100 steps; the
scored set is every recorded field at every step; the acquisition subset is
exactly complete step-entry state, live O7 inputs and products, named call-order
boundaries, exchange outputs, and restart. None may be sampled or relaxed.

## O17/C4 differentiability disposition

The design now requires CPU binary64 `jax.value_and_grad` for the exact one-step
production entry, with an unweighted sum of all floating prognostic and exchange
output leaves as the scalar reachability objective. It publishes shape, dtype,
finite, NaN, Inf, and nonzero counts for every floating input leaf; integer and
Boolean scheduling/routing leaves are reported as literal non-differentiable
inputs. Category participation, CFL/sweep selection, limiter and donor routing,
`min`/`max`/`where` switches, fixed iteration counts, convergence masks, category
search, and restart/write schedules are the literal non-differentiable
statements. The primal output must retain every forward bit unchanged. GPU
memory and multi-step tape budgeting are explicitly later work.

No count table is published as a measurement: there is no O17 callable to
compile. `step-entry floating leaves: UNMEASURED`, `live-forcing floating
leaves: UNMEASURED`, and `finite/NaN/Inf/nonzero counts: UNMEASURED`. Running
`value_and_grad` on only O15/O16 would be a different program and is refused.

## Decisions, controls, and review

- Window: fixed at 100 steps; no decision.
- Bars: exact local and fixed `1e-15` trajectory; no decision.
- Scored set and record subset: fixed above; no decision.
- AD scalarization: fixed as a mechanical unweighted output-leaf sum; it is not
  a scientific objective or a new bar.
- Acquisition tool: not authored because the explicit composition-entry stop
  fired first. After the entry exists, the acquisition tool must import the
  O5/O7/O8--O16 machinery and shared extent/read-set gates; it may not copy
  their logic.
- NEMO: not built or run.

`CONFIRMED`: the master/record mismatch and missing production entry are direct
schema/source findings. `PLAUSIBLE`: none promoted to an O17 physics verdict.

## Gates

- The required receipt-citation test file passed 45/45. The round receipt gate
  found 17 citations with zero unmapped citations, failures, or full-map audit
  failures; all 12 built-in controls fired. The cumulative default gate found
  274 citations with the same zero-failure result.
- The documented landing battery passed. The combined process reached the
  known per-process JAX compiler limit in the two-case stage-face-mask file, so
  that file and the following barotropic-state file were completed in fresh,
  serial processes as required by the repository instructions: both mask cases
  passed and the barotropic-state file passed 8/8. No test ran with `pytest -n`.
- The nine CI ratchet files reported 11 failed, 6,936 passed, and 2 skipped.
  Those are the established lane failures in hardcoded constants, inline
  coefficients, and parameter specifications; every named failure is in an
  untouched model, acquisition-fixture, or unit-test file. This round changes
  only the ladder design, this receipt, and the citation registry.
- All commands used CPU fp64 with `CUDA_VISIBLE_DEVICES=""`,
  `JAX_PLATFORMS=cpu`, and `JAX_ENABLE_X64=1`. No NEMO or GPU command ran.

independent review: operator (Claude)

## Resume 2 measurement and disposition

The probe now keeps every scheduler value outside the differentiated pytree.
The literal static inputs are `kt=1`, `jpl=5`, `nlay_i=10`, `nlay_s=5`,
`nn_hls=2`, `n_subcycles=100`, `category_count=5`, `ln_icedyn=True`, and
`ln_icethd=True`.  `jax.eval_shape` completes without converting a traced
integer to Boolean.  The floating pytree is an 8 by 7 full tile with a 4 by 3
owned region: eight owned columns contain ice in five categories, snow, and
ponds; four owned columns are open water; ice/ocean velocities, wind stress,
ocean stress, heat, freshwater, and precipitation forcings are nonzero.

The exact compiled CPU-fp64 `value_and_grad` program was run twice under the
required `timeout 3600`.  Both runs returned exit 124 without a compiler
diagnostic.  The resource-accounted current-source run measured 3607.36 s wall
including timeout reap and 18,061,120 KiB peak RSS (about 17.2 GiB).  The same
exact transform under `jax.disable_jit()` also returned exit 124 after 3604.83 s
and measured 17,710,772 KiB peak RSS.  Thus neither execution produced a
numerical gradient or a primal-from-gradient result; forward-bit equality for
that transformed program is unmeasured, not failed.

Trace-only accounting completed in 43 seconds. Argon's corrected recursive
count for the reviewed graph is 576,364 equations. `_thermo_category_loop`
owns 342,178 (the five O9--O11 category paths contribute about 191k and the O12
pond path about 151k), `si3_ice_thd_do` owns 53,490, and `si3_dyn_orca2` owns
48,325. The remaining named and inlined program is 132,371 equations. The
earlier attribution of 179,770 equations to a catch-all bucket is retracted:
those equations were nested beneath the thermodynamic owner. The reproducible
recursive/source-owner counter is
`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_jaxpr_count.py`.
The same counter on the Resume 3 source reports 577,982 total: 342,170 in the
thermodynamic loop, 53,489 in open-water thermodynamics, 48,325 in Prather
dynamics, and 133,998 remaining. It retains the reviewed `ea3e570` snapshot
above separately, so the Decision 194 measurement is not silently rewritten by
the Resume 3 carry/tail changes and their surrounding composition wiring.

The complete 206-leaf publication is
`docs/ice/fidelity/si3_r29_o17_ad_leaf_table.md`.  Shape and dtype are measured
for every floating entry/forcing leaf.  Finite, NaN, Inf, and nonzero gradient
counts are `UNMEASURED_TIMEOUT` on every row.  Consequently no leaf is called
zero and no zero reason is asserted.  The probe contains reason classification
for a completed run (overwritten live-state fields, masks/routing, disabled
landfast drag, inactive bulk-salinity Prather moments, and exact local zeros),
but publishing any of those as observed zero leaves would fabricate evidence.

The production dispatcher and frozen O17 card are wired, and the structural
preflight proves the admitted call order.  Its swapped-rheology/advection and
dropped-redistribution plants both fire.  The O4 forward/JIT exact cases passed
before only its final AD test reached its 1800-second timeout; O5 passed 17/17,
O7 passed 19/19, the O8--O16 master scorer passed 30/30, O6 passed 28 selected
tests (one deselected), O12 passed 80/80, finalization passed 8/8, and the new
O17 unit file passed 6/6.  These retain every certified component's forward
bits.

The nine repository ratchets report the established baseline: 11 failed,
6,936 passed, and 2 skipped.  The failures are the three pre-existing
hardcoded-constant rows (`test_si3_o13_itd_remap.py`, `test_si3_thd_do.py`, and
the O4 stub writer), five pre-existing inline-coefficient rows
(`bitz_lipscomb.py`, `si3_orca2_state.py`, `geothermal.py`, `mle.py`, and
`mle_latlon_cgrid.py`), and three pre-existing parameter-spec rows
(`si3_orca2_state.py`, `mle.py`, and `shortwave_penetration.py`).  O17 adds no
ratchet failure; its frozen deck literals carry explicit provenance comments.

They are not, however, a through-entry composition proof.  The retained O4,
O5, O7, and O8--O16 records are different projections/runs and do not contain
one complete O17 step-entry plus live forcing.  The composition scorer now
reports both `INDEPENDENT` and `GIVEN-NEMO-ENTRY` as `ACQUISITION_NEEDED`
instead of relabelling separate component scores as an O17 result.  A true
eager/JIT through-entry proof, and meaningful swap/drop numerical red scores,
require the unified O17 record named by the acquisition tool.

### Decision needed

1. Convert the O9--O12 category/layer/Picard Python loops to bit-preserving JAX
   control flow and place checkpoint boundaries between already admitted rung
   calls.  This changes traced operator structure but not intended forward
   arithmetic; every O4--O16 exactness test must remain 0 ULP. **Pick.**
2. Add stagewise custom VJPs around the admitted calls, preserving the current
   forward graph but taking on hand-maintained reverse rules.
3. Replace the full-step objective with separate per-stage gradients.  This is
   a different program and is not recommended.

No option was applied in Resume 2 because the operator explicitly prohibited
operator-structure changes.  Until option 1 or 2 is approved, the C4 numerical
reachability table remains measured as timed out rather than passing.

## Resume 3 review closure

This section is authoritative for the review of `ea3e570732`. No NEMO command
was run. The O2 production boundary again raises the original
`NotImplementedError("ice_update_flx" ...)` class/message contract; both
certified O2 gate cases pass unchanged. O17 still refuses the incomplete O2
forcing as a production full-step input, so restoring the exception contract
does not invent any forcing or weaken either gate.

The production step now includes every cited tail operation rather than taking
it as forcing: `t_bo` is formed at `src/ICE/icestp.F90:136-138`;
`u_ice_b/v_ice_b` at the first `store_fields` boundary
(`src/ICE/icestp.F90:151`, `src/ICE/icestp.F90:421-423`); `a_i_b/at_i_b` at the
second `store_fields` boundary (`src/ICE/icestp.F90:184`,
`src/ICE/icestp.F90:399-418`); `diag_set0` resets owned `t_si` at
`src/ICE/icestp.F90:488`; category unpack retains `t_si` at
`src/ICE/icethd.F90:163,518`; and the post-`ice_cor(kt,2)` ageing, T/U/V halo
links, category sum, and melted-edge velocity zeroing are transcribed from
`src/ICE/icethd.F90:183-201`. C-grid deformation now occurs between aEVP and
Prather advection, matching `src/ICE/icedyn.F90:132-134` and
`src/ICE/icedyn_rhg_evp.F90:791-827`.

The admitted live O9--O12 category loop now returns its carried module flux
state. Salinity's `sfx_res/sfx_bri` are retained from
`src/ICE/icethd_sal.F90:231,239,489,596,624`; diffusion's
`hfx_err_dif/hfx_dif` are carried from
`src/ICE/icethd_zdf_bl99.F90:828-831`; the already admitted category
aggregation is used directly. The five-snapshot replay arm is unchanged. The
live-arm unit test numerically proves all four carries across all five
categories.

The O17 unit gate executes one mixed open/ice, five-category, snow-and-pond
entry through the real `si3_o17_step` to the O7 surface boundary in eager and
compiled modes. It uses the explicit redistribution plant because O6 already
has its admitted eager/JIT numerical suite, and it stops before the 342k
thermodynamic category equations whose compilation Decision 194 reserves for
the next round. The separate live O9--O12 test executes all five category
carries, and the post-thermodynamic ageing/halo/velocity tail runs in eager and
compiled modes. Red controls numerically change the velocity pair presented to
Prather when rheology/advection are swapped, ORCA2 deformation when the fold is
dropped, and O7 heat flux when `frq_m` is halved; `ln_ice_embd=True` refuses.
The production dispatcher test calls the live composed entry, and the
record-reader test proves self-described extents and truncation refusal.
Removing any of those live arms therefore fails a numerical assertion rather
than an AST-list or `callable()` check.

The former four-component O17 driver is removed. The operator tool now invokes
the extended O8--O16 master exactly once. That master retains its passive
reference twin, then runs one instrumented, unmodified 100-step
`ORCA2_SAS_ICE` production trajectory. Per-step/per-rank O17 kinds 22--34 cover
entry, rheology, advection, redistribution, dynamic correction, thermodynamics,
exit, Prather entry/exit, flux-other, both raw bulk boundaries, and the private
stress exit. Records are self-describing and include full prognostic state,
all 55 Prather arrays, dynamic stress, category bounds, delayed CFL, raw O7
meteorology, precipitation, ocean/grid/mask forcing, all carried fluxes, O16
exchange fields, and all stress outputs. The manifest names exactly those
kinds. Production-flag FP-census and same-program gates cover every observer.

The scorer reads that one trajectory and calls `si3_o17_step` for every step
and rank. `GIVEN-NEMO-ENTRY` enforces bitwise equality (including signed zero)
at the named boundaries and exit; `INDEPENDENT` seeds only step 1 from NEMO and
enforces the fixed `1e-15` trajectory bar thereafter. A non-PASS score prevents
the top-level `RUN_COMPLETE` marker and exit zero. The read-only preflight uses
stub NEMO only for driver retention/control flow and separately executes live
numeric swap/fold plants. RAM, disk, and one global wall-clock gates remain.
The operator entry is:

`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_acquisition/run.sh`

Decision 194 selected option 1: bit-preserving JAX loop conversion with
checkpoints. It is deliberately not implemented in this round; it is the next
round's work. The corrected Argon ownership numbers above are the frozen
measurement motivating that decision.

### Resume 3 final verification

All verification used CPU binary64 with `CUDA_VISIBLE_DEVICES=""`,
`JAX_PLATFORMS=cpu`, and `JAX_ENABLE_X64=1`; every pytest command named its
test file, ran serially, and used no xdist worker. No NEMO command ran.

The complete `tests/ocean/fidelity/test_nemo_si3_*.py` sweep returned 206
passing assertions. The O8--O12 master scorer passed 30/30 in 703.01 s and the
extended O8--O16/O17 acquisition suite passed 28/28 in 493.97 s. The O4
Prather file completed its eight forward cases before the deliberately deferred
full-step AD compile reached the 1800-second cap, which is the Decision 194
boundary rather than a numerical red. The scalarmath-v2 file returned four
passes and the pre-existing external-state failure `A MY_SRC is not verbatim`;
neither that test nor its gate changed from `ea3e570732`.

The complete `tests/ice/unit/test_si3_*.py` sweep returned 404 passing
assertions, two deselections, and one expected xfail. O12 passes 81/81 after the
exact-input five-snapshot replay arm was kept compatible with its historical
mock entries while the live arm continues to require and carry
`hfx_err_dif/hfx_dif`; the old composer test and the new five-category live
carry test both pass. O17 passes 7/7, including the eager/JIT numerical
composition test. The only remaining red is the pre-existing
`test_production_path_plants_fire[sum_first_category]` in `test_si3_surface.py`
(23 passed, 1 failed, 1 xfailed); neither the test nor `si3_surface.py` changed
from `ea3e570732`, and the earlier O12 receipt already records the same inert
plant after `e96c1875f8`.

The documented landing files pass after the mechanical O12 citation reanchor:
receipt citations 45/45, private imports 6/6, TKE terms 71/71, NEMO recipe
25/25, freshwater closure 16/16, stage-face masks 2/2, and prognostic
barotropic state 8/8. The nine repository ratchets report the unchanged failure
set: 11 failed, 6,944 passed, and 2 skipped. The three hardcoded-constant rows
are `test_si3_o13_itd_remap.py`, `test_si3_thd_do.py`, and the O4 stub writer;
the five inline-coefficient rows are `bitz_lipscomb.py`, `si3_orca2_state.py`,
`geothermal.py`, `mle.py`, and `mle_latlon_cgrid.py`; the three parameter-spec
rows are `si3_orca2_state.py`, `mle.py`, and `shortwave_penetration.py`.

The unified acquisition's final read-only preflight exited zero after proving
the real preprocessed-source placement, patched bounds fixture, writer layout,
observer-free same program, 100-step stub-driver retention, unchanged cfg
digests, and independent admission. Its final line is
`PREFLIGHT_OK: no real NEMO integration was run; worst-case master 140965556460
bytes (131.28 GiB)`. A top-level `RUN_COMPLETE` remains impossible until the
real record has been scored successfully.

## Resume 1 implementation (operator ruling applied)

The earlier stop above is historical.  The operator confirmed that O17's one
new piece is the typed full-deck composition entry, not a new physical
operator.  Resume 1 therefore implemented that composition while retaining the
stop on any previously unadmitted physical kernel.

The dispatcher now accepts `SI3O17StepState` plus
`SI3O17LiveForcing`.  The explicit `SI3O17Card` writes the live O4--O16 deck,
including `ln_icedyn=True`, `ln_icethd=True`, 100 aEVP subcycles, five
categories, 10+5 layers, 10800 s, `nn_hls=2`, all surface selectors, and
`core2_precipitation_closure=False`.  The production call graph is:

1. O7 `blk_ice_1` and stress exchange;
2. O5 H79 strength and aEVP;
3. O5 C-grid deformation at the rheology exit;
4. O4 Prather advection, then O6 redistribution/rafting and `ice_cor(kt,1)`;
5. the remainder of O7 (`ice_alb`, `blk_ice_2`, `ice_flx_other`);
6. O8 frazil, O9--O12 category/pond work, O13 remap, and O14 open water;
7. O15 correction, O16 aggregation/flux update, and tau update.

No new physical formula was added.  The only supporting changes are production
wiring: the admitted deformation routine can select the admitted ORCA2 fold
exchange, and the admitted O9--O12 loop can carry its flux tuple live between
categories while preserving its old five-snapshot exact-record replay arm.

The read-only O17 preflight imports the O5, O7, O8--O16, read-set, and shared
extent tools.  It mechanically verifies the call graph above, the 100 x 10800 s
contract, both score labels, and the required complete entry/raw
meteorology/products/boundaries/exchanges/restart record set.  Both a swapped
O5/O4-order plant and a dropped-O6 plant fail.  The acquisition driver defaults
to this read-only preflight, writes its source/resource/retention/admission
manifest before an explicit operator run, and routes output only below
`si3_rounds/round<k>/`.

Resume 1 initially left the differentiability probe with an ice-free synthetic
state.  Resume 2 above replaces it with the active mixed-regime state and
records the bounded compiled/eager measurements and per-leaf disposition.

independent review: operator (Claude)

## DECISION 194 — traced O9–O12 loops

DECISION 194 (option 1 of the Resume 2 decision) converted the O9–O12
category, layer and iteration Python loops to JAX control flow with static trip
counts.  Every converted loop keeps its statements, their order and their
source-rounding barriers; only the iteration mechanism changes, and only when
the call is traced (`jax.jit`, `jax.grad`): an eager call still runs the Python
loop.  No default, configuration field, scheme selection or physics changed,
no knob was added, and the O17 composition module was not edited.

What was converted (read off the code):

- the O12 composer's `DO jl = 1, jpl` category loop (O9 + O10 + O11 +
  `ice_thd_1d2d(jl, 2)`) and the `pnd_LEV` category loop: one shared helper
  runs them as a `lax.scan` over the stacked category inputs with the body
  under `jax.checkpoint`, so reverse mode keeps only the category carry;
- the `ice_thd_dh` surface melt/sublimation and basal melt layer passes:
  static-trip `fori_loop`s over the stacked layer carriers;
- the shared `ice_var_vremap`/`snw_ent` old-segment loop: a `fori_loop`
  updating every new interface elementwise in the same old-segment order
  (O14 open-water growth uses the same helper);
- the salinity-4 harmonic-mean permeability: one `fori_loop` over the inner
  layer with all start layers carried together.

One barrier was added.  With the category body scanned, XLA contracted the
BL99 bottom conductive flux into fused multiply-adds in its two consumers (the
`ice_thd_dh` available bottom heat and the `hfx_err_dif` budget): 1–12 ULP in
the basal-growth fluxes and `hfx_err_dif` on the five-category fixture
(measured).  NEMO's object code does not fuse there (DECISION 158), so the
product is now materialised like its neighbours; the fixture is then 0 ULP.

Left as Python loops (small; inside an already-traced body or a few
statements each): the `ice_thd_dh` snow loops, the salinity-4 per-layer and
RJW velocity loops, the `ice_perm_eff` layer loop, the `ice_thd_pnd`
vanished-pond loop, the vremap cumulative sums, and the BL99 matrix rows
inside the existing Picard `fori_loop`.  No checkpoint was placed between the
O17 rung calls (that module is being edited by another round, and a measured
checkpoint variant compiled slower; see below).

### Measurements (CPU fp64, libm arm, probe tile unless stated)

Recursive JAXPR equations of the O17 objective, by owner (the committed
counter, parent 9998276020 → this round):

| owner | parent | DECISION 194 |
|---|---:|---:|
| O9–O12 thermodynamic owner | 342,170 | 35,858 |
| O14 open water (shares vremap) | 53,489 | 9,659 |
| O4 Prather dynamics | 48,325 | 48,325 |
| everything else | 133,998 | 133,998 |
| total | 577,982 | 227,840 |

Forward bits:

- five-category O12 composition fixture (6 × 6, live flux carry, 499 output
  leaves): eager and `jax.jit` both bit-identical to the parent; the committed
  test pins that digest (`e74fcfa0…73bf`);
- full O17 step, eager: all 679 published leaves bit-identical to the parent;
- full O17 step, `jax.jit`: no finite non-zero value changed; 588 elements in
  51 leaves differ only in the sign of zero (572) or of a NaN (16).  581 of the
  588 now equal the parent's eager value (the parent's compiled program had
  folded `0 + (-0)` against compile-time-zero flux accumulators, which the
  scanned carry no longer exposes — mechanism plausible, not instrumented); the
  other 7 are NaN signs in the known open-water NaN columns.  Parent eager vs
  parent jit differed in 607 elements; final jit vs parent eager differs in 33.

Compile cost (wall, peak RSS):

| program | parent | DECISION 194 |
|---|---:|---:|
| O12 composition fixture, forward jit (compile + run) | 254 s | 16 s |
| full O17 step, forward jit | 568 s, 22.4 GB | 230 s, 7.2 GB |
| O12 composition fixture, reverse mode | not measured | 141 s, 7.3 GB |
| one O9 category, reverse mode | not measured | 27.5 s |
| full O17 step, the AD probe's `value_and_grad`, `timeout 3600` | exit 124 | exit 124 |
| same program, `timeout 10800` | — | exit 124 (≥ 62 GB observed) |
| finite-masked full-step `value_and_grad`, `timeout 10800` | — | exit 124 (≥ 173 GB observed) |

Before the `ice_thd_dh` layer passes were traced (category scan, vremap and
permeability already traced) the O12 reverse mode took 729 s and 43.9 GB and
one O9 category 216 s: the straight-line layer passes dominated the reverse
compile.  The parent's own O9 reverse compile aborted before finishing, so no
parent number is quoted for those rows.

The full step still does not compile.  Isolated per-rung reverse-mode
compiles on recorded probe inputs: `ice_update_tau` 0.8 s, O12 composition
119 s, O13 remap 11 s, O14 open water 12 s, O16 46 s, O15 96 s, aEVP 19 s,
O6 `ice_cor` 396 s, O4 Prather 878 s (50 GB), and O6 redistribution did
not finish within about 2.4 h (at least 148 GB observed).  The pre-thermodynamics half of the step alone (O7 → O5 →
O4 → O6 → O7, no O8–O16) needs 82 s to lower and 1131 s to compile (55 GB);
with `jax.checkpoint` around the aEVP, Prather, redistribution and
`ice_cor` calls it needed 1292 s (60 GB).  The remaining compile cost is
therefore outside the O9–O12 owner, and the whole program compiles far
slower than the sum of its rungs.

Gradient checks: the AD probe produced no per-leaf counts (both runs timed
out), so the per-leaf table stays unmeasured.  At the O12 composition level
the reverse pass is finite on all 171 floating input leaves (157 carry a
non-zero gradient).  Central differences of the same compiled forward
(relative step 1e-4, objective = output sum weighted by 1/(1+max|leaf|)):

| leaf | step | AD | FD | rel. error |
|---|---:|---:|---:|---:|
| category 3 `h_i`, cell (3,2) | 2.34e-4 | 4.184320186 | 4.184320175 | 2.6e-9 |
| category 1 `a_i`, cell (3,2) | 2.48e-5 | 16.23678458 | 16.23678420 | 2.4e-8 |
| category 3 `e_i`, cell (3,2), layer 4 | 3.31e+4 | −3.317754415e-7 | −3.317754416e-7 | 2.6e-10 |
| category 2 `h_s`, cell (2,3) | 8.65e-6 | 104.0114698 | 104.0114690 | 7.1e-9 |
| pond `a_ip`, cell (2,3), category 1 | 6.53e-6 | 2.648828360 | 2.648828386 | 9.9e-9 |
| category 1 `qns_ice` forcing, cell (3,2) | 6.94e-3 | 0.06519869983 | 0.06519869923 | 9.1e-9 |

A full-step finite-difference check was not possible: it needs the full-step
reverse pass, which did not compile within 10800 s.  (A trial with every rung
under its own `jax.jit` and no outer `jit` aborted after 19 min with LLVM
"Cannot allocate memory"; not pursued.)

Plants (the committed test's forward pin): reversing the category scan, reversing
the `ice_thd_dh` layer order, skipping the first vremap segment, joining a
permeability start layer one step late, and removing the new bottom-flux
barrier each change the digest; stopping the gradient on the scanned category
inputs zeroes the checked thickness gradient.

Exactness battery, run serially file by file (default `-m 'not slow'`, CPU
fp64), parent 9998276020 against 17caeb46bf: every one of the 33 SI3 unit and
fidelity files, the receipt-citation gate, the four lane-3 `si3thd` gates and
the two shared-kernel unit files reports the same summary line before and
after (the new DECISION 194 file: 2 passed).  The three failures are present in
both runs and unrelated: one `test_si3_surface` production-path plant
(`sum_first_category`), the `scalarmath_v2` gate's "MY_SRC is not verbatim"
host check, and an atmosphere-config tail assertion in `test_pr_f_fixes`.  The
O4 file needed 2 h 58 min before and 3 h 58 min after on the shared host (its
reverse-mode test; the per-file limit was raised from 3 h to 4 h for the after
run only).  The O8–O16 master scorer over all 100 steps: before and after,
`PASS_ALL_O8_O13_0_ULP`, `PASS_O14_0_ULP` and `PASS_ALL_O15_O16_0_ULP`.

### Decision needed

1. The full-step reverse pass still does not compile, and the remaining cost
   is outside O9–O12: O6 redistribution alone did not compile in about 2.4 h,
   O4 Prather needs 878 s and O6 `ice_cor` 396 s, and the whole step compiles
   far slower than the sum of its rungs.
   1. Apply the same bit-pinned loop tracing to the O6 redistribution, O6
      `ice_cor`, O4 Prather, O15 and O16 rungs, one rung per round, largest
      first.  **Pick.**
   2. Keep each admitted rung in its own XLA compile unit and let JAX chain
      the true full-step gradient through them (no hand-written VJP); this
      needs O17 glue changes, and an eager-glue trial aborted with an LLVM
      memory error.
   3. Stop here and use rung-level gradients.
2. The compiled full-step forward changes only zero and NaN signs (581 of 588
   now match the eager program).  Accept as is (**pick**), or require the
   parent's compiled signed zeros.

independent review a: Gemini Argon HOLD, Gemini flash HOLD (both confirmed
the loop orders, 0 ULP against the parent eager and jit on three fixtures,
the plants and the master scorer; the five items below were raised).

### Resume 1 (review a)

1. **Eager path pinned.**  The forward test now runs the composition under
   `jax.disable_jit()` (concrete inputs, so the Python-loop path) and under
   `jax.jit` (the scan/`fori_loop` path) and requires both digests to equal the
   parent digest.  Reversing only the eager category loop changes the eager
   digest (`77c261c4…6770`) while the compiled one still matches.
2. **FMA audit.**  The reviewers' `acc + e_old[old - 1] * weight` is the
   vremap interpolation (icevar.F90 lines 1020–1022), whose two products are
   already materialised.  Counting x86 FMA instructions in the dumped XLA CPU
   kernels and mapping each back to its multiply: a jitted vremap (20,000
   random columns) had one FMA, the new-layer thickness `zhnew`
   (icevar.F90 line 1007) fused into the first interface addition (line 1012),
   value-identical there (adding +0) and 0 bit differences against eager on
   200,000 outputs; the parent's O12 program had 12 such vremap FMAs.  The
   same audit of the O12 composition found a value-changing one inherited from
   the parent: the SSL snow transmission (icethd_zdf_bl99.F90 line 209) fused
   into the absorption difference (line 211) — 7,754 of 20,000 `ab_s`
   elements differed from eager on a random input, although the composition
   fixtures and the master records happened not to expose it.  Both products
   are now materialised.  Afterwards the vremap and SSL kernels have no FMA and
   match eager bit for bit; the O12 composition keeps only FMAs on the exact
   halving `0.5 * (t_i(jk) + t_i(jk+1))`, which cannot change a value; in the
   full O17 forward program the only O9–O12 FMAs left are those exact halvings;
   the others are outside O9–O12 (O17 composition and tail, O4, O5).  A new unit check pins the SSL statement
   (removing its barrier gives 7,754 mismatches); no input was found on which
   the `zhnew` barrier changes bits, so it rests on the audit, not on a test.
   The full O17 forward (`jit`) stays bit-identical to 2428dd0472 on all 679
   published leaves, and eager to the parent on all 1,209 trace leaves.
3. **Citations.**  The 101 parent-blob pins are gone.  87 citations into the
   two edited files (81 pure moves from DECISION 194 plus 6 from the new SSL
   barrier) are re-lined on the live HEAD files with the existing re-line
   helper and keep their anchor text; 21 cite statements DECISION 194 replaced
   and now cite the replacement statements, anchored on HEAD's text.  The seven
   receipts that cite replaced code carry a DECISION 194 note.  The gate's
   checking code is unchanged.  No citation of si3_thd_pnd.py lines 862-970 exists in
   this receipt at 2428dd0472 (searched), so none was changed.
4. **Finite differences.**  Each of the six cases is judged by its own
   relative error (no absolute floor); its rounding-noise bound is at most
   1.2e-7 of its gradient.  Stopping the gradient on the scanned inputs fails
   all six (relative error 1.0 each); scaling the scanned inputs' tangent by
   1 + 1e-5 also fails all six (1.0e-5, 9.9e-6, 1.0e-5 for the 3.3e-7
   enthalpy gradient, 1.0e-5, 1.0e-5, 1.0e-5).
5. **Scored O17 fields, jit against eager, at the final source** (report
   only): 21 elements — `state.oa_i` and the after-thermodynamics `oa_i`,
   9 elements each, differ in value by up to 1.5e-16 relative (one ULP), and
   one NaN sign each in `wfx_snw_sub`, `wfx_spr` and `wfx_sub`.  The parent
   has the same 18 `oa_i` differences (from an FMA in the O17 thermodynamic
   tail, outside O9–O12) plus 12 sign-only differences in other fluxes and
   `emp`; it matched eager on the `wfx_spr` NaN sign, which is new here.
   Final jit against parent jit in scored fields: 11 elements, all zero or NaN
   signs.  User decision 2 (accept the sign changes) is still pending.

Final checks from a clean checkout of 594d6720f1, serial, CPU fp64, largest
resident set 7.9 GB (sampled every 20 s): the DECISION 194 unit file
`3 passed`; the O12 file `81 passed`; the citation gate `45 passed`; the O8–O16
master-scorer tests `30 passed`; and the O9 `15 passed`, O10 `44 passed`, O14
`10 passed`, lane-3 `si3thd` gates `22`, `8`, `13`, `17 passed` and shared BL99
unit file `12 passed`, each the same count as at the parent.  Master scorer,
100 steps: `PASS_ALL_O8_O13_0_ULP`, `PASS_O14_0_ULP`,
`PASS_ALL_O15_O16_0_ULP`.  The full-step `value_and_grad`, the AD probe, the O6
reverse compiles and the O4 file were not run in this resume.

independent review: pending (author: Claude)

## DECISION 198 round 1 — traced O6 redistribution

Review b of b17f14b73c: Gemini Argon LAND, Gemini flash LAND.  DECISION 200
accepts the compiled zero/NaN sign changes; the one-ULP `oa_i` difference from
the O17 thermodynamic tail stays out of scope.  This round extends the
DECISION 194 method to the O6 redistribution only; O6 `ice_cor`, O4 Prather,
O15 and O16 are untouched.

What changed (read off the code): the `rdgrft_shift` donor-category loop
(`DO jl1 = 1, jpl`, icedyn_rdgrft.f90 lines 722–939) became one body that an
eager call runs as the Python loop and a traced call runs as a static-trip
`fori_loop` with the category state as carry; the receiver loop
(`DO jl2`, lines 846–937) stays unrolled inside the body, and the ridging-pass
loop was already a `fori_loop`.  No statement, statement order, barrier or
MIN/MAX operand order changed: the module's barrier sweep and MIN/MAX audit
tests read the same source text.  An object-code audit of the compiled
forward redistribution (10 ice and 5 snow layers) finds no FMA instruction,
before or after.

Measurements (CPU fp64, libm arm; every command that could pass 20 GB ran
under a watchdog that kills only its own child above 120 GiB, after checking
at least 250 GiB available and no process above 100 GiB):

- **Forward bits.**  The O6 unit fixture at 3/2 and at 10/5 ice/snow layers:
  eager and `jax.jit` both bit-identical to b17f14b73c (22 result leaves each,
  19 ridging cells, 5 passes).  The full O17 step: eager and `jax.jit` both
  bit-identical to b17f14b73c on all 1,209 trace leaves, signed zeros and NaN
  bits included, so no compiled field moved.
- **Equations** (the committed counter): 228,050 at b17f14b73c → 198,323;
  the redistribution owner 43,529 → 13,802; the O9–O12 owner (35,918), open
  water (9,809) and Prather (48,325) unchanged.  The counter had not been
  re-pinned after the resume-1 barriers (it reported 228,050 against a pinned
  227,840); both snapshots are now recorded.
- **Compile.**  Isolated O6 reverse pass (`value_and_grad` of a weighted sum of
  all redistribution outputs, 10/5 layers): 12.5 s to lower, 107.6 s to
  compile, 5.0 GiB peak, all 21 input gradients finite and non-zero.  The
  parent's was not re-run here; on the probe's recorded inputs it had not
  finished after about 2.4 h at 148 GB (DECISION 194 measurement).  Forward
  jit of the full step: 154 s, 7.3 GiB → 114 s, 5.7 GiB.  The O6 unit file:
  `28 passed, 1 deselected in 219.82s`, 3.4 GiB peak (1,769 s at the parent).
- **Full-step `value_and_grad`** (the AD probe's program; allowed because the
  isolated O6 reverse compile stayed under 60 GB; run once): stopped by its
  10,800 s limit while still compiling, peak 79.8 GiB, so the per-leaf table
  stays unmeasured.  The remaining cost is outside O6 redistribution (O4
  Prather 878 s, O6 `ice_cor` 396 s, O15 96 s, O16 46 s isolated).
- **FD** (relative step 1e-4, objective = outputs weighted by
  1/(1+max|output|), each case by its own relative error; rounding noise at
  most 9.1e-8 of the gradient):

| leaf (O6 fixture) | AD | FD | rel. error |
|---|---:|---:|---:|
| `a_i` (6,4) cat 5 | 0.3667349272 | 0.3667349267 | 1.4e-9 |
| `v_i` (6,4) cat 5 | 0.07407449431 | 0.07407449425 | 7.5e-10 |
| `v_s` (6,4) cat 5 | 0.3946576485 | 0.3946576486 | 1.3e-10 |
| `e_i` (5,3) layer 1, cat 3 | 2.201112618e-9 | 2.201112617e-9 | 2.1e-10 |
| `v_ip` (6,4) cat 5 | 0.9765167278 | 0.9765167447 | 1.7e-8 |
| `oa_i` (5,3) cat 3 | 3.610880287e-7 | 3.610880287e-7 | 1.3e-10 |

- **Plants** (new test, each restored afterwards): reversing the traced donor
  order changes the compiled digest to `6c77559b…` (eager unchanged); removing
  the barrier between the receiver product and its sum changes the compiled
  digest to `943b21f4…` (removing the one on the ridged-volume sum did not
  change any bit, so it is not used as the plant); stopping the gradient on
  the donor-loop carry gives a zero gradient in all six FD cases; scaling the
  carry tangent by 1 + 1e-5 per donor step gives relative errors of
  2.5e-4 in all six (compounded over 5 donors × 5 passes).
- **Citations.**  The SI3 receipts' four citations into the redistribution
  module are pinned to historical content (DECISION 177 blobs and commits) and
  still resolve; the full map audit and every SI3 receipt pass, so nothing
  needed re-anchoring.  The gate is unchanged.

Final checks from a clean checkout of b7c4412990, serial, CPU fp64: the
DECISION 194 file `3 passed`; the DECISION 198 file `2 passed`; the O12 file
`81 passed`; the citation gate `45 passed`; the O8–O16 master-scorer tests
`30 passed`; master scorer over 100 steps `PASS_ALL_O8_O13_0_ULP`,
`PASS_O14_0_ULP`, `PASS_ALL_O15_O16_0_ULP`.  The O4 Prather file was not run.

independent review: pending (author: Claude)

## DECISION 198 round 2 — traced O4 Prather subcycles

Round 1 (743146b590) is under review; this round works on top of it and
touches O4 Prather advection only (O6 `ice_cor`, O15 and O16 untouched).
DECISION 200 accepts compiled sign-of-zero and NaN-sign changes; none occurred
here.

What changed (read off the code): under a JAX transform the subcycle loop
(`DO jt = 1, icycle`, icedyn_adv_pra.f90 lines 156–498) runs as one
three-step `lax.scan` whose body is checkpointed; each step uses the same
halo width, sweep parity and `icycle >= jt` selection as the three unrolled
calls it replaces.  The alternating sweep order (lines 267–367) becomes a
`lax.cond` on the traced parity; both branches hold the same two sweep calls
in the same order as before.  Eager calls keep the three unrolled calls and a
Python branch.  No statement, barrier or MIN/MAX operand order changed.  An
object-code audit of the compiled forward finds no FMA instruction, before or
after (probe tile through the O4 step, and the two-dimensional fixture below).

Measurements (CPU fp64, libm arm; every command that could pass 20 GB ran
under a watchdog that kills only its own child above 120 GiB, after checking
at least 250 GiB available and no process above 100 GiB):

- **Forward bits.**  The O4 step on the probe tile at one, two and three
  subcycles (both sweep parities): eager and `jax.jit` bit-identical to
  743146b590 (134 leaves each).  A 12 × 10 all-ocean fixture with ice in every
  cell, two-dimensional velocities and the O17 card's configuration (5
  categories, 10/5 layers, LEV ponds, salinity option 4) at two and three
  subcycles: eager and jit digests equal the parent's.  The full O17 step:
  eager and jit bit-identical to 743146b590 on all 1,209 trace leaves, signed
  zeros and NaN bits included.
- **Equations** (the committed counter): 198,323 at 743146b590 → 178,146;
  the Prather owner 48,325 → 28,148 (the six traced sweep substeps become
  four: two per `lax.cond` branch); the other owners unchanged.  The round-1
  snapshot is kept.
- **Compile.**  Isolated O4 reverse pass on the probe's recorded inputs (the
  per-rung instrument that gave 878 s): 743146b590 13.7 s to lower and 846.9 s
  to compile, 39.0 GiB peak; now 19.4 s and 223.1 s, 16.2 GiB.  (The earlier
  50 GB came from a process that had already compiled eight other rungs.)
  Forward jit of the O4 step on the probe tile, compile and run: about 55 s →
  12 s.
- **Full-step `value_and_grad`** (the AD probe's program; allowed because the
  isolated O4 reverse compile stayed under 30 GB; run once): stopped by its
  10,800 s limit while still compiling, peak 50.3 GiB (79.8 GiB in round 1),
  so the per-leaf table stays unmeasured.  O6 `ice_cor` (396 s), O15 (96 s)
  and O16 (46 s) are the rungs not yet traced.
- **FD** on the two-dimensional fixture at three subcycles (relative step
  1e-4, objective = outputs weighted by 1/(1+max|output|), each case by its
  own relative error; rounding noise at most 5.1e-9 of the gradient):

| input | AD | FD | rel. error |
|---|---:|---:|---:|
| `v_i` (6,5) cat 3 | 205.3812379 | 205.3812381 | 9.7e-10 |
| `a_i` (6,5) cat 3 | 8.589927548 | 8.589927551 | 3.0e-10 |
| `v_s` (4,6) cat 5 | 153.8986185 | 153.8986184 | 6.3e-10 |
| `e_i` (6,5) layer 3, cat 3 | 6.268041510e-9 | 6.268041516e-9 | 9.8e-10 |
| `v_ip` (4,6) cat 5 | 787.3811739 | 787.3811754 | 1.9e-9 |
| `u_ice` (4,6) | −19.09890745 | −19.09890748 | 1.5e-9 |
| `v_ice` (4,6) | −17.90208879 | −17.90208892 | 7.5e-9 |

  Reverse-mode values equal the parent's to 1.6e-15 relative on all 33
  scanned cases.  Two scanned cases disagree with central differences in
  both programs: `v_i` and `a_i` at cell (5,4), category 1 (AD 5.864 vs FD
  4.775; 3.667 vs 6.731).  Measured one-sided slopes differ there at steps
  from 1e-3 to 1e-7 (8.13e-8 and 0 for one area output against `v_i`), so the
  input sits on a branch boundary; reverse and forward mode both return one
  of the two one-sided slopes (the right one for that output against `v_i`,
  the left one for an area output against `a_i`).  PLAUSIBLE, read off the
  code: the
  thickness-capping test in `Hbig_pra` compares h after the sweep with the
  pre-advection maximum, which is equal at a local-maximum cell that only
  exports.  These cases are left out of the test.
- **Plants** (new test, each restored afterwards): reversing the traced
  subcycle order changes the compiled digest to `79ddec22…` (on the probe
  tile, whose meridional velocity is zero, it changed nothing, which is why
  the test uses the two-dimensional fixture); removing the barrier around one
  product of the pre-advection mass diagnostic changes the compiled digest to
  `c702ec2b…`; stopping the gradient on the scan carry fails all seven FD
  cases (zero gradient for the five state inputs); scaling the carry tangent
  by 1 + 1e-5 per subcycle gives relative errors of 3.0e-5 for the five
  state inputs and 1.1e-5 and 1.7e-5 for the two velocities, which enter
  outside the carry (all seven fail).
- **Citations.**  The traced subcycles moved every later line of the state
  module; 24 citations into it are re-lined live through the DECISION 194
  mechanism (two re-pin a closing-parenthesis occurrence number) and the
  grown dispatcher anchors on HEAD's text, with a note in the r8 O4 receipt.
  The gate logic is unchanged and passes (274 citations).
- **O4 file** (run once, alone, under the watchdog): `9 passed in 1897.02s`,
  13.5 GiB peak (2 h 58 min before DECISION 194, 60–148 GB at the parent per
  the round brief).

Final checks from a clean checkout of 67137462f8 (the round's later commits
change only this receipt, the leaf table and comments in the new test),
serial, CPU fp64: the DECISION 194 file
`3 passed`; the DECISION 198 round-1 file `2 passed`; the new O4 file
`2 passed`; the O12 file `81 passed`; the O6 file `28 passed, 1 deselected`;
the citation gate `45 passed` and its default run `PASS`; the O8–O16
master-scorer tests `30 passed`; master scorer over 100 steps
`PASS_ALL_O8_O13_0_ULP`, `PASS_O14_0_ULP`, `PASS_ALL_O15_O16_0_ULP`.

independent review: pending (author: Claude)

## DECISION 198 round 3 — traced O6 ice_cor loops

Round 2 (56b768300a) is under review; this round works on top of it and
touches O6 `ice_cor` only (O15, O16 and the round-2 `Hbig_pra` tie untouched).

Today's order (isolated reverse compile on the probe's recorded inputs, each
rung in its own process at 56b768300a, lower + compile, peak):

| rung | lower s | compile s | peak GiB |
|---|---:|---:|---:|
| O6 `ice_cor` | 7.1 | 449.8 | 20.3 |
| O4 Prather (traced in round 2) | 19.3 | 227.3 | 16.1 |
| O12 composition | 29.1 | 117.2 | 9.0 |
| O6 redistribution (traced in round 1) | 14.7 | 116.3 | 7.1 |
| O15 | 7.7 | 100.5 | 8.2 |
| O16 | 4.0 | 48.1 | 4.6 |
| aEVP | 6.6 | 17.1 | 3.7 |
| O14 open water | 4.3 | 11.1 | 3.8 |
| O13 remap | 3.8 | 9.3 | 3.7 |
| `ice_update_tau` | 0.3 | 0.5 | 3.2 |

`ice_cor` was still the largest.

What changed (read off the code): under a JAX transform the `ice_itd_reb`
passes (iceitd.f90 `DO jl = 1, jpl-1` at line 628 and `DO jl = jpl-1, 1, -1`
at line 664, each calling `itd_shiftice`), the salinity bounds
(icecor.f90 lines 114–122, `DO jl` / `DO jk`) and the `ice_var_zapsmall` flux
sums (icevar.f90 lines 633–645 salt, 664–690 heat, 695–722 water) run the same
bodies as static-trip `fori_loop`s through one helper; eager calls keep the
Python loops.  No statement, barrier or MIN/MAX operand order changed: the O6
file's barrier sweep and MIN/MAX audit read the edited source and pass.  An
object-code audit of the compiled forward `ice_cor` (3/2 and 10/5 layers)
finds no FMA instruction, before or after.

Measurements (CPU fp64, libm arm; every command that could pass 20 GB ran
under a watchdog that kills only its own child above 120 GiB, after checking
at least 250 GiB available and no process above 100 GiB):

- **Forward bits.**  The O6 `ice_cor` fixture at 3/2 and 10/5 ice/snow layers
  and the probe's recorded `ice_cor` inputs: eager and `jax.jit` bit-identical
  to 56b768300a (24 result leaves each; the 3/2 fixture rebins cells in all
  eight passes).  The full O17 step: eager and jit bit-identical to 56b768300a on
  all 1,209 trace leaves, signed zeros and NaN bits included.
- **Equations** (the committed counter): 178,146 at 56b768300a → 162,751;
  inside `ice_cor` 20,327 → 4,932; the named owners unchanged.  The round-2
  snapshot is kept.
- **Compile.**  Isolated `ice_cor` reverse pass, same instrument: 7.1 s to
  lower and 449.8 s to compile, 20.3 GiB → 3.3 s and 7.1 s, 3.4 GiB.
- **Full-step `value_and_grad`** (the AD probe's program; allowed because the
  isolated `ice_cor` reverse compile stayed under 30 GB; run once): stopped by
  its 10,800 s limit while still compiling, peak 29.2 GiB (50.3 GiB in round 2,
  79.8 GiB in round 1), so the per-leaf table stays unmeasured.
- **FD** on the O6 fixture (3/2 layers; relative step 1e-4; objective =
  outputs weighted by 1/(1+max|output|); the quotient is taken element by
  element so the objective's large unchanged part does not swamp it; each
  case by its own relative error):

| input | AD | FD | rel. error |
|---|---:|---:|---:|
| `v_i` (5,2) cat 2 | 0.7245951532 | 0.7245951537 | 7.9e-10 |
| `a_i` (5,2) cat 2 | 0.6663113636 | 0.6663113637 | 6.5e-11 |
| `v_i` (5,5) cat 1 | 11.95596085 | 11.95596085 | 3.2e-12 |
| `a_i` (5,5) cat 1 | −0.01023501231 | −0.01023501241 | 1.0e-8 |
| `szv_i` (5,2) layer 2, cat 2 | 0.02216291618 | 0.02216291618 | 9.7e-13 |
| `v_s` (5,5) cat 1 | 0.3796893385 | 0.3796893385 | 4.6e-13 |

  A scan of all 1,057 non-zero interior entries of seven inputs: reverse-mode
  values and quotients identical to the parent's (largest relative difference
  0.0).  1,017 agree to 1e-6.  The other 40 are excluded, with the same
  reverse-mode values in the parent.  39 have inputs of 1e-11 or less, all in
  categories whose input ice volume is below 1e-10; there the step falls
  below the outputs' rounding: 37 quotients
  move by more than 1e-3 between steps 1e-4 and 1e-6, one (`v_i` (3,3)
  category 1) gives a zero quotient at both steps, and one (`szv_i` (2,4)
  layer 2, category 1) splits one-sided at step 1e-6.  The 40th, `a_i` (6,4)
  category 4, is curvature (1.7e-6 at step 1e-4, 6.3e-9 at 1e-6).  No entry
  shows a stable one-sided split, the mark of a branch boundary.
- **Plants** (new test, each restored afterwards): reversing every traced
  loop changes the compiled digest to `59ebad1b…`; removing one barrier in
  the salinity-bound flux update (icecor.f90 lines 114–122) changes it to
  `ce6d1e8a…`; stopping the gradient on the loop carry gives a zero gradient
  in all six FD cases; scaling the carry tangent by 1 + 1e-5 per iteration
  gives relative errors from 4.2e-5 to 3.7e-4 in all six.
- **Citations.**  The full-map audit finds every citation at its anchor
  without re-lining, so the gate script is unchanged; it passes.
- **O6 file** (alone, under the watchdog): `28 passed, 1 deselected in
  210.22s`, 3.4 GiB peak.

Final checks from a clean checkout of a3e6e1f5e3 (the round's later commit
changes only this receipt and the leaf table), serial, CPU fp64: the DECISION
194 file `3 passed`; the DECISION 198 round-1 and round-2 files `2 passed`
each; the new ice_cor file `2 passed`; the O12 file `81 passed`; the O6 file
`28 passed, 1 deselected`; the citation gate `45 passed` and its default run
`PASS`; the O8–O16 master-scorer tests `30 passed`; master scorer over 100
steps `PASS_ALL_O8_O13_0_ULP`, `PASS_O14_0_ULP`, `PASS_ALL_O15_O16_0_ULP`.

independent review: pending (author: Claude)

## DECISION 198 round 4 — full-step reverse-compile and runtime diagnosis

DECISION 209 (2026-11-03) scopes DECISION 198 round 4 as a pure diagnosis at
`22af29b743a8` (rounds D194 and D198 r1–r3 landed) with **no model code changes**
(`git diff --stat 22af29b743a8..HEAD -- packages` is empty; forward bits are
unchanged by construction).

### 1. Question and candidate hypotheses

After DECISION 194 and DECISION 198 rounds 1–3, every isolated rung's reverse
compile (`lower + compile`) finishes in minutes (`o4_prather` 238.4 s compile,
`o12_thermo` 111.2 s, `o6_rdgrft` 110.2 s, `o15_cor` 75.9 s, `o16_update_flx`
43.6 s, `o5_aevp` 20.1 s, `o14_open_water` 13.0 s, `o13_remap` 10.4 s,
`o6_ice_cor_dyn` 7.0 s, rest ≤ 5.3 s; sum across all 14 rungs = 643.5 s
`.compile()` and 88.7 s `make_jaxpr + lower`), yet running the full-step AD
probe (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_ad_probe.py`)
still did not finish within `10,800 s` (peak `29.2 GiB` in Round 3).

Four candidate explanations were pre-registered for discrimination:

- **(a) JAX tracing/lowering vs XLA compile (and runtime execution inside the
  probe)** — separated by timing `jax.make_jaxpr(fwd)`,
  `jax.make_jaxpr(jax.value_and_grad(fwd))`, `.lower()`, `.compile()`, and
  `compiled(inputs)` independently on the full 14-rung step (`--prefixes 14
  --compile-only`) and on growing prefixes (`k = 1..14`).
- **(b) One or a few XLA passes scaling super-linearly with module size** —
  separated by capturing `TF_CPP_VMODULE=hlo_pass_pipeline=1,cpu_compiler=2,buffer_assignment=1,ir_compiler=2`
  into scratch logs (`--xla-pass-log`) and comparing per-pass wall times, buffer
  counts, and pre/post-optimization HLO instruction counts between isolated
  rungs (`k-k`), growing prefixes (`1..k`), and the full step (`1..14`).
- **(c) Composition / specific rung jumps** — separated by measuring both
  `.compile()` and `compiled(inputs)` across the 14 isolated rungs (`1-1..14-14`),
  growing prefixes (`k = 1..8, 12, 14`), and suffixes (`8-14`, `11-14`), then
  isolating any rung whose addition causes a jump in compile or execution time
  down to the individual function and field.
- **(d) Reverse pass of uncounted glue (state packing, 69-field post-dynamics
  halo exchange, `glo2eqv`, `agg(kn=1)`, surface/frazil/tail glue, outer step)**
  — separated by `--glue-only` (which mocks the 10 major D198 rungs with identity
  projections on their recorded outputs while differentiating through all
  inter-rung glue and the 4 previously uncounted stages `k = 1, 6, 7, 12` in the
  real `si3_o17_step`) and by measuring `k = 1, 6, 7, 12` in isolation.

### 2. Instrument and direct unit test

`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_ad_probe.py` was
extended (without duplicating scripts) to provide:

1. Canonical 14-rung execution-order registry `RUNG_SPECS` (`k = 1..14`)
   covering every top-level stage of `si3_o17_step`:
   `1: o7_blk1_strength`, `2: o5_aevp`, `3: o4_prather`, `4: o6_rdgrft`,
   `5: o6_ice_cor_dyn`, `6: postdyn_halos_agg`, `7: o7_surface_o8`,
   `8: o12_thermo`, `9: o13_remap`, `10: o14_open_water`, `11: o15_cor`,
   `12: thermo_tail`, `13: o16_update_flx`, `14: o16_update_tau`.
2. Separate stage timers (`fwd_make_jaxpr_seconds`, `vg_make_jaxpr_seconds`,
   `lower_seconds`, `lower_total_seconds`, `compile_seconds`,
   `execution_seconds`), peak RSS (`ru_maxrss` in KiB/GiB), and pre/post-opt HLO
   instruction and computation counts (`_count_hlo_text`).
3. CLI flags `--prefixes`, `--slices`, `--isolated-funcs`, `--glue-only`,
   `--checkpoint-rungs`, `--compile-only`, `--xla-pass-log`, and `--vmodule`,
   with provenance (`git_sha`, dirty status, `argv`, env vars) stamped in every
   JSON artifact and `/tmp` paths rejected.
4. Direct unit test `tests/ice/unit/test_si3_d198_r4_compile_diagnosis.py`
   (`2 passed in 42.01s`), which runs `diagnose_slices` on prefixes `k = 1, 2`
   and verifies planted violations on `/tmp` log paths, invalid slice bounds,
   planted HLO instructions, and planted XLA `fusion` pass durations.

### 3. Measurements (CPU fp64, `22af29b743a8`)

All runs used `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1` on
`/home/dbalwada/legoESM/.venv/bin/python` with `TMPDIR` and all logs/outputs
under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r4_scratch`
(after verifying `free -g` ≥ 930 GiB available and no process > 16 GiB RSS).

#### Table 1 — All 14 isolated rungs (`k-k`) and `--glue-only` (`MEASURED`)

*(Sources: `diag_p1_3_glue.json`, `diag_iso2_7.json`, `diag_iso8_14.json`,
`diag_exec_8_14.json`, `diag_exec_9_10_12_13_14.json`)*

| k | Rung label | Fwd eqns (top / rec) | VG eqns (top / rec) | `make_jaxpr(vg)` s | `lower()` s | `compile()` s | `execute()` s | Pre-opt HLO | Post-opt HLO | HLO passes s | Buffer assign s (buffers) | Backend JIT s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `o7_blk1_strength` | 670 / 1,345 | 1,753 / 3,646 | 0.28 | 0.17 | 0.87 | 0.0041 | 2,276 | 8,267 | 0.29 | 0.01 (8,049) | 0.47 |
| 2 | `o5_aevp` | 1,616 / 10,194 | 5,145 / 29,805 | 3.91 | 2.50 | 20.10 | 0.0927 | 34,883 | 130,813 | 11.54 | 1.54 (115,119) | 6.73 |
| 3 | `o4_prather` | 2,344 / 28,487 | 6,527 / 107,555 | 18.35 | 4.07 | 238.39 | 0.1145 | 66,532 | 905,032 | 151.78 | 4.77 (882,128) | 80.56 |
| 4 | `o6_rdgrft` | 1,329 / 13,934 | 3,289 / 41,298 | 14.59 | 7.30 | 110.19 | 0.0604 | 89,416 | 406,889 | 53.76 | 14.08 (341,390) | 50.55 |
| 5 | `o6_ice_cor_dyn` | 564 / 5,216 | 2,546 / 12,091 | 1.39 | 1.76 | 7.03 | 0.0094 | 20,189 | 57,657 | 3.71 | 0.42 (47,623) | 2.44 |
| 6 | `postdyn_halos_agg` | 5,044 / 7,405 | 11,760 / 17,737 | 2.50 | 0.99 | 5.28 | 0.0053 | 12,809 | 33,531 | 2.61 | 0.09 (30,739) | 1.78 |
| 7 | `o7_surface_o8` | 2,588 / 5,518 | 6,499 / 14,980 | 1.27 | 0.54 | 5.01 | 0.0143 | 8,808 | 49,201 | 2.35 | 0.11 (48,388) | 2.14 |
| 8 | `o12_thermo` | 3,900 / 38,088 | 11,241 / 148,711 | 25.11 | 7.03 | 111.16 | 1.0457 | 105,679 | 555,014 | 70.24 | 3.21 (521,098) | 35.54 |
| 9 | `o13_remap` | 5,383 / 11,687 | 14,401 / 32,121 | 2.72 | 1.18 | 10.42 | 0.0130 | 17,360 | 101,938 | 5.62 | 0.31 (97,527) | 3.58 |
| 10 | `o14_open_water` | 5,356 / 10,634 | 14,517 / 29,038 | 3.65 | 1.50 | 13.03 | 0.0116 | 23,215 | 109,512 | 6.86 | 0.45 (100,361) | 4.67 |
| 11 | `o15_cor` | 10,589 / 22,265 | 27,890 / 62,327 | 5.72 | 2.25 | 75.91 | >150 (40.8 in `1..12`) | 35,637 | 552,597 | 38.70 | 1.81 (538,549) | 33.76 |
| 12 | `thermo_tail` | 1,594 / 1,798 | 4,025 / 4,536 | 0.72 | 0.38 | 1.32 | 0.0119 | 3,353 | 7,300 | 0.37 | 0.07 (6,759) | 0.60 |
| 13 | `o16_update_flx` | 5,232 / 12,637 | 14,649 / 35,276 | 2.70 | 1.25 | 43.62 | **>10,800 (hang)** | 18,190 | 212,055 | 11.66 | 0.56 (208,217) | 30.56 |
| 14 | `o16_update_tau` | 1,175 / 1,751 | 3,467 / 5,061 | 1.08 | 0.26 | 1.02 | 0.0044 | 2,492 | 8,130 | 0.25 | 0.08 (8,125) | 0.49 |
| **Sum** | **14 isolated rungs** | **47,384 / 170,959** | **127,709 / 544,183** | **84.01** | **31.18** | **643.52** | — | **440,839** | **3,137,936** | **359.74** | **27.51 (2,954,072)** | **253.87** |
| — | `--glue-only` (10 D198 rungs mocked) | 11,146 / 17,772 | 25,356 / 43,745 | 4.96 | 1.95 | 19.03 | 0.0233 | 30,550 | 120,208 | 10.39 | 0.62 (115,447) | 6.41 |

#### Table 2 — Growing prefixes `1..k` and multi-rung suffixes (`MEASURED`)

*(Sources: `diag_p1_3_glue.json`, `diag_p4_5.json`, `diag_p6_7.json`,
`diag_p8.json`, `diag_p12.json`, `diag_p14.json`, `xla_s8_14.log`,
`xla_s8_8_11_14_slice_11_14_o15_cor_to_o16_update_tau.log`. Remaining
intermediate prefixes `k = 9, 10, 11, 13` are scripted in
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r4_scratch/operator_run.sh`.)*

| Slice / Prefix | Fwd eqns (top / rec) | VG eqns (top / rec) | `make_jaxpr(vg)` s | `lower()` s | `compile()` s | `execute()` s | Peak RSS GiB | Pre-opt HLO | Post-opt HLO | `fusion` s | `copy-insertion` s | `buffer_assignment` s (buffers) | `llvm_ir` + `codegen` wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1..1` (`o7_blk1_strength`) | 670 / 1,345 | 1,753 / 3,646 | 0.28 | 0.17 | 0.87 | 0.0041 | 3.16 | 2,276 | 8,267 | 0.09 | 0.04 | 0.01 (8,049) | 0.21 + 0.22 |
| `1..2` (`+ o5_aevp`) | 2,178 / 11,431 | 6,569 / 33,122 | 4.34 | 2.14 | 19.11 | 0.0919 | 3.98 | 36,780 | 141,185 | 2.09 | 3.22 | 1.28 (125,282) | 1.98 + 1.34 |
| `1..3` (`+ o4_prather`) | 4,201 / 39,597 | 12,538 / 140,119 | 23.01 | 6.05 | 257.69 | 0.1810 | 17.18 | 102,351 | 1,044,467 | 104.23 | 20.66 | 8.45 (1,005,657) | 31.09 + 33.05 |
| `1..4` (`+ o6_rdgrft`) | 5,420 / 53,421 | 15,385 / 180,975 | 36.79 | 13.48 | 392.35 | 0.2894 | 18.50 | 191,314 | 1,450,657 | 124.08 | 38.46 | 33.04 (1,346,333) | 39.78 + 50.30 |
| `1..5` (`+ o6_ice_cor_dyn`) | 5,739 / 58,392 | 17,359 / 192,494 | 40.51 | 15.75 | 435.12 | 0.2683 | 20.50 | 210,728 | 1,507,314 | 130.48 | 42.34 | 41.04 (1,392,968) | 42.31 + 53.90 |
| `1..6` (`+ postdyn_halos_agg`) | 10,645 / 65,659 | 28,727 / 209,839 | 40.80 | 17.07 | 432.77 | 0.2698 | 19.29 | 222,670 | 1,539,143 | 126.50 | 42.11 | 41.73 (1,422,028) | 43.92 + 56.30 |
| `1..7` (`+ o7_surface_o8`) | 12,851 / 70,699 | 34,347 / 223,834 | 43.36 | 17.83 | 455.24 | 0.2966 | 21.75 | 230,477 | 1,591,552 | 130.91 | 43.57 | 43.49 (1,473,466) | 43.14 + 57.55 |
| `1..8` (`+ o12_thermo`) | 16,288 / 108,228 | 44,452 / 371,303 | 71.48 | 23.18 | 639.69 | 1.3189 | 24.34 | 334,853 | 2,135,526 | 163.95 | 67.85 | 65.99 (1,983,458) | 55.01 + 67.13 |
| `1..12` (`+ o13,o14,o15,tail`) | 36,984 / 152,002 | 98,015 / 491,497 | 81.95 | 29.86 | 791.90 | 42.0981 | 31.08 | 408,734 | 2,901,500 | 193.92 | 81.63 | 81.93 (2,721,187) | 72.38 + 96.75 |
| **`1..14` (full O17 step)** | **39,930 / 162,737** | **103,347 / 503,371** | **79.79** | **27.67** | **803.95** | **>10,800 (hang)** | **32.12** | **402,843** | **3,052,370** | **200.29** | **78.48** | **76.72 (2,869,611)** | **74.10 + 109.75** |
| `8..14` (`o12` to `update_tau`) | 29,638 / 94,552 | 76,650 / 302,163 | 47.00 | 13.79 | 274.20 | — | — | 194,207 | 1,622,550 | 73.17 | 31.02 | 22.84 (1,563,689) | 50.85 + 31.83 |
| `11..14` (`o15` to `update_tau`) | 16,134 / 35,895 | 43,727 / 98,948 | 10.69 | 3.43 | 122.40 | — | — | 52,053 | 786,876 | 30.89 | 12.23 | 4.86 (771,846) | 27.66 + 26.86 |

#### Table 3 — Full-step (`k = 14`, `prefix_1_14_o16_update_tau`) compile time breakdown (`MEASURED`)

*(Source: `diag_p14.json` and `xla_p14.log`)*

| Phase / XLA Pass | Calls | Wall s | Max single call s | Share of `.compile()` (`803.952 s`) | Share of total (`911.405 s`) |
|---|---:|---:|---:|---:|---:|
| **`jax.make_jaxpr(fwd)`** | 1 | 6.455 | 6.455 | — | — |
| **`jax.make_jaxpr(value_and_grad(fwd))`** | 1 | 79.785 | 79.785 | — | 8.75% |
| **JAX `.lower()` (MLIR / StableHLO emission)** | 1 | 27.668 | 27.668 | — | 3.04% |
| **Subtotal: `lower_total_seconds` (`make_jaxpr(vg) + lower`)** | — | **107.453** | — | — | **11.79%** |
| **XLA HLO pass pipeline (`hlo_passes_wall_s`)** | — | **465.004** | — | **57.84%** | **51.02%** |
| — `fusion` (CPU `InstructionFusion`) | 1 | 200.293 | 200.293 | 24.91% | 21.98% |
| — `copy-insertion` | 1 | 78.476 | 78.476 | 9.76% | 8.61% |
| — `simplify-while-loops` | 50 | 27.615 | 1.636 | 3.43% | 3.03% |
| — `algsimp` | 53 | 19.578 | 1.612 | 2.44% | 2.15% |
| — `tree_reduction_rewriter` | 100 | 14.426 | 0.166 | 1.79% | 1.58% |
| — `layout-assignment` | 1 | 12.112 | 12.112 | 1.51% | 1.33% |
| — `while-loop-invariant-code-motion` | 50 | 10.429 | 1.245 | 1.30% | 1.14% |
| — `tuple-simplifier` | 52 | 9.322 | 2.445 | 1.16% | 1.02% |
| — `constant_folding` | 50 | 7.419 | 1.175 | 0.92% | 0.81% |
| — `cpu-parallel-task-assigner` | 1 | 7.395 | 7.395 | 0.92% | 0.81% |
| — all other HLO passes | — | 77.939 | — | 9.69% | 8.55% |
| **XLA CPU backend JIT (`backend_jit_total_s`)** | 1 | **306.660** | 306.660 | **38.14%** | **33.65%** |
| — `buffer_assignment` (`2,869,611` buffers, `51.65 MiB` alloc) | 1 | 76.724 | 76.724 | 9.54% | 8.42% |
| — `llvm_ir_emit` (HLO → LLVM IR across `10,009` kernels) | 1 | 74.099 | 74.099 | 9.22% | 8.13% |
| — `llvm_codegen` (`10,009` modules; `965.43 s` thread-sum) | 10,009 | 109.752 | 43.567 | 13.65% | 12.04% |
| — backend scheduling / thunk emission / linking | — | 46.085 | — | 5.73% | 5.06% |
| **HLO module setup / pre-pass & post-JIT wrapper overhead** | — | **32.288** | — | **4.02%** | **3.54%** |
| **Subtotal: `.compile()` (`compile_seconds`)** | 1 | **803.952** | 803.952 | **100.00%** | **88.21%** |
| **Total compile (`lower_total_seconds + compile_seconds`)** | — | **911.405** | — | — | **100.00%** |

#### Table 4 — Isolation of the `> 10,800 s` probe timeout to `si3_var_agg(..., kn=2).tm_i` inside `si3_o16_step` (`MEASURED`)

*(Sources: sub-function and per-field isolation runs on the recorded `si3_o16_step`
inputs at `22af29b743a8`)*

| Unit under `jax.value_and_grad` | `lower()` s | `compile()` s | `execute()` s | Post-opt HLO | Notes |
|---|---:|---:|---:|---:|---|
| `si3_o16_step` (`k = 13`, baseline `22af29b743a8`) | 1.25 | 43.62 | **>10,800 (hang)** | 212,055 | Calls `si3_var_glo2eqv(kn=2)`, `si3_var_agg(kn=2)`, `si3_ice_update_flx` |
| — `si3_var_glo2eqv(..., kn=2)` alone | 0.26 | 2.69 | **0.0046** | — | Executes in 4.6 ms |
| — `si3_ice_update_flx(...)` alone | 0.35 | 6.92 | **0.0033** | — | Executes in 3.3 ms |
| — `si3_var_agg(..., kn=1)` (`k = 6`, post-dynamics) | 0.99 | 5.28 | **0.0053** | 33,531 | `kn <= 1` returns before `tm_i` / `tm_s` loops (`si3_orca2_state.py` line 1141) |
| — `si3_var_agg(..., kn=2)` alone (all 21 fields) | 0.67 | 28.12 | **>10,800 (hang)** | 131,806 | Enters `kn > 1` block (`si3_orca2_state.py` lines 1143–1210) |
| — `si3_var_agg(kn=2)` fields `hm_i, hm_s` | 0.08 | 0.86 | **0.0007** | — | Elementwise `jnp.where` |
| — `si3_var_agg(kn=2)` field `tm_si` | 0.14 | 1.66 | **0.0014** | — | 5-step `jpl` loop |
| — `si3_var_agg(kn=2)` fields `om_i, sm_i` | 0.18 | 2.09 | **0.0014** | — | 5-step `jpl` loop |
| — `si3_var_agg(kn=2)` field `tm_s` | 0.30 | 3.51 | **0.0023** | — | 25-step `jpl * nlay_s` loop (`5 * 5 = 25`) |
| — `si3_var_agg(kn=2)` fields `hm_ip, hm_il` | 0.21 | 2.36 | **0.0015** | — | 5-step `jpl` loop |
| — **`si3_var_agg(kn=2)` field `tm_i` alone** | **0.28** | **20.23** | **>10,800 (hang)** | — | **50-step unrolled `jpl * nlay_i` loop (`5 * 10 = 50`, `si3_orca2_state.py` lines 1171–1178)** |
| **In-memory test: `si3_var_agg(..., kn=2)` with `lax.scan` on `tm_i, tm_s`** | **0.35** | **2.60** | **0.0021** | — | **0.000e+00 max abs/ULP forward diff on all 21 fields** |
| **In-memory test: `si3_o16_step` with `lax.scan` on `tm_i, tm_s`** | **0.88** | **6.64** | **0.0098** | **60,095** | **Compile `43.62 s -> 6.64 s` (6.6x); execute `>10,800 s -> 0.0098 s`** |

### 4. Candidate verdicts (labeled `MEASURED` or `PLAUSIBLE`)

1. **Candidate (a) — JAX tracing/lowering vs XLA compile vs runtime execution (`MEASURED`)**:
   - **MEASURED**: On the full 14-rung O17 step (`prefix_1_14_o16_update_tau`),
     `jax.make_jaxpr(fwd)` takes `6.455 s`, `jax.make_jaxpr(value_and_grad(fwd))`
     takes `79.785 s`, and `.lower()` takes `27.668 s` (`lower_total_seconds =
     107.453 s`, 3.04 GiB RSS, `402,843` pre-opt HLO instructions). XLA
     `.compile()` takes **`803.952 s` (`13.40 min`)** at **`32.120 GiB` peak
     RSS**, producing `3,052,370` post-opt HLO instructions. Thus **full-step
     compilation (`lower + compile`) completes in `911.405 s` (`15.19 min`)**.
   - **MEASURED**: In Round 3's log (`work/si3_d194_scratch/probe_d198r3.log`),
     XLA's `SlowOperationAlarm` (`[Compiling module jit_objective for CPU]`,
     which wraps `CpuCompiler`'s backend JIT phase) printed completion
     (`5m12.90s`) 15 minutes after job start — matching today's `5m06.66s`
     backend JIT and `15.19 min` total compile. Immediately after `.compile()`
     returned, `nemo_si3_o17_ad_probe.py` executed `compiled(inputs)` before
     printing any output, whereas Round 3's isolated-rung script (`ad_rungs.py`)
     only called `.lower(diff).compile()` and never called `compiled(diff)`.
     Therefore **`911.4 s` (`8.4%`) of the `10,800 s` timeout was compile time
     and `> 9,880 s` (`> 91.6%`) was a runtime execution hang in
     `compiled(inputs)`**.

2. **Candidate (b) — XLA passes scaling with module size (`MEASURED`)**:
   - **MEASURED**: Full-step `.compile()` (`803.952 s`) is **1.25x** the sum of
     the 14 isolated rungs' `.compile()` (`643.521 s`), and post-opt HLO
     (`3,052,370` instructions) is **7.58x** pre-opt HLO (`402,843`
     instructions).
   - **MEASURED**: Five XLA phases account for **`71.1%` (`571.64 s` /
     `803.95 s`)** of full-step `.compile()`:
     1. HLO `fusion` (`200.293 s`, 1 call, `24.9%`);
     2. Backend `llvm_codegen` (`109.752 s` wall / `965.433 s` thread-sum across
        `10,009` kernel modules, `13.7%`);
     3. HLO `copy-insertion` (`78.476 s`, 1 call, `9.8%`);
     4. Backend `buffer_assignment` (`76.724 s` over `2,869,611` buffers,
        `9.5%`, which scales **2.79x** relative to the sum of isolated rungs'
        `27.51 s` because whole-step live ranges coexist in one module);
     5. Backend `llvm_ir_emit` (`74.099 s`, `9.2%`).

3. **Candidate (c) — Composition / specific rung jumps (`MEASURED` + `PLAUSIBLE`)**:
   - **MEASURED (compile jumps)**: Four rungs account for **`83.2%` (`535.64 s`
     / `643.52 s`)** of isolated `.compile()` and drive the prefix `.compile()`
     jumps: `k = 3` (`o4_prather`, `+238.58 s` in prefix `1..3`), `k = 4`
     (`o6_rdgrft`, `+134.67 s` in prefix `1..4`), `k = 8` (`o12_thermo`,
     `+184.45 s` in prefix `1..8`), and `k = 11` (`o15_cor`, `75.91 s` isolated)
     + `k = 13` (`o16_update_flx`, `43.62 s` isolated; `+152.20 s` across
     `1..8 -> 1..12`). In particular, `si3_o15_step` (`si3_ice_cor` in
     `packages/ice/legoesm/ice/si3_finalize.py` lines 553–730) still unrolls the
     `itd_shiftice`, salinity-bound, and `zapsmall` loops (`27,890` top-level VG
     eqns, `75.91 s` `.compile()`), whereas `si3_ice_cor_dyn` (traced in Round 3,
     `k = 5`) has `2,546` top-level VG eqns and compiles in `7.03 s`.
   - **MEASURED (runtime execution jumps)**: Prefixes `1..1` through `1..8`
     execute `compiled(inputs)` in `0.0041 s – 1.3189 s`. Prefix `1..12`
     executes in `42.0981 s` (`+40.78 s` from `k = 11` `si3_o15_step`'s unrolled
     `ice_cor` loops). Adding `k = 13` (`si3_o16_step`) causes `compiled(inputs)`
     to hang for **`> 10,800 s`**, isolated specifically to the 50-step unrolled
     `tm_i_acc` loop in `si3_var_agg(..., kn=2)`
     (`packages/ice/legoesm/ice/si3_orca2_state.py` lines 1171–1178).
   - **MEASURED (HLO inspection) + PLAUSIBLE (XLA CPU emitter mechanism)**:
     In `si3_var_agg(..., kn=2)`, `tm_i` unrolls `50` iterations (`jpl=5 *
     nlay_i=10`) of
     `tm_i_acc = jnp.where(vt_i > _EPSI20, sr(tm_i_acc + sr(sr(t_i[:, :, jk, jl] * v_i[:, :, jl]) * r1_nlay_i)), rt0)`,
     where `sr` (`nemo_source_round`,
     `packages/core/legoesm/core/source_rounding.py` lines 14–27) is
     `jnp.where(jnp.isfinite(value), value, jnp.copysign(jnp.abs(value), value))`.
     MEASURED in the post-opt HLO dump of `si3_var_agg(..., kn=2)`: XLA fuses all
     50 backward steps and their `slice`/`pad`/`dynamic-update-slice` scatters
     into two 2,859-line `kLoop` fusions (`%fused_computation.1` and
     `%fused_computation.2`, `backend_config={"outer_dimension_partitions":["3"]}`,
     matching the 3 `tf_XLAEigen` worker threads observed spinning at 100% CPU).
     PLAUSIBLE from XLA CPU's `FusedIrEmitter`: because the per-step cotangent
     feeds multiple branch/select operands (`isfinite`, `copysign`, `abs`,
     `vt_i > _EPSI20`) and non-elementwise `slice`/`pad` index expressions inside
     a single `kLoop` fusion, `FusedIrEmitter` re-emits upstream value
     evaluations at each use instead of caching per-step temporaries, turning a
     50-step diamond recurrence into `O(2^50)` runtime work.
   - **MEASURED (in-memory verification without touching `packages/`)**:
     Replacing the unrolled `tm_i_acc` and `tm_s_acc` loops in
     `si3_var_agg(..., kn=2)` with `jax.lax.scan` preserves `0.000e+00` ULP
     forward agreement across all 21 fields of `SI3AggregatedVars`, cuts
     `si3_var_agg(..., kn=2)` reverse compile from `28.12 s` to `2.60 s` and
     execution from `> 10,800 s` to `0.0021 s`, and cuts `si3_o16_step` reverse
     compile from `43.62 s` to `6.64 s` and execution from `> 10,800 s` to
     `0.0098 s`.

4. **Candidate (d) — Uncounted glue, state packing, 69-field halo exchange,
   outer step (`MEASURED` — REFUTED)**:
   - **MEASURED**: `--glue-only` (all 10 major D198 rungs mocked; all inter-rung
     glue, 69-field `si3_apply_state_halos_orca2`, `si3_var_glo2eqv`,
     `si3_var_agg(kn=1)`, `o7_blk1_strength`, `o7_surface_o8`, `_thermo_entries`,
     `_thermodynamic_tail`, `SI3UpdateFluxForcing`, and `SI3O17Trace` packing
     live) lowers in `1.952 s`, compiles in `19.034 s` (`120,208` post-opt HLO),
     and executes in `0.0233 s`. The four previously uncounted rungs in isolation
     (`k = 1, 6, 7, 12`) sum to `2.08 s` `lower()`, `12.48 s` `.compile()`, and
     `0.036 s` `execute()`.

### 5. Next changes pointed to by this diagnosis (no model changes made this round)

1. **Primary change (eliminates the `> 10,800 s` runtime hang in `compiled(inputs)`
   and cuts `si3_o16_step` reverse compile from `43.62 s` to `6.64 s`)**:
   In `si3_var_agg` (`packages/ice/legoesm/ice/si3_orca2_state.py` lines 1171–1187),
   when `kn > 1` and tracing under a JAX transform, replace the unrolled 50-step
   `tm_i_acc` (`jpl * nlay_i`) and 25-step `tm_s_acc` (`jpl * nlay_s`) Python
   loops with static-trip `jax.lax.fori_loop` / `jax.lax.scan` carries in exact
   `(jl, jk)` order (already verified in memory to be `0.000e+00` ULP
   bit-identical on all 21 `SI3AggregatedVars` fields).
2. **Secondary change (cuts `si3_o15_step` isolated reverse compile from
   `75.91 s` to `~7 s` and removes its `~40.8 s` execution cost in `1..12`)**:
   Route `si3_o15_step` (`si3_ice_cor` in
   `packages/ice/legoesm/ice/si3_finalize.py` lines 553–730) through the traced
   `fori_loop` helpers (`itd_shiftice`, salinity bounds, `ice_var_zapsmall`)
   already landed for `si3_ice_cor_dyn` in Round 3.

independent review: operator

## DECISION 198 round 5 — traced `si3_var_agg(kn=2)` and `si3_o15_step` loops

### Summary of changes (`3dd560f7f05f` -> Round 5)

1. **Primary (`si3_var_agg(kn=2)` layer-sum loops)**:
   In `packages/ice/legoesm/ice/si3_orca2_state.py` lines 1176–1211, when `kn > 1`
   and `_si3_traced(a_i, v_i, v_s, t_i, t_s)` is True (traced under a JAX
   transform), replaced the unrolled 50-step `tm_i_acc` (`jpl * nlay_i = 5 * 10`)
   and 25-step `tm_s_acc` (`jpl * nlay_s = 5 * 5`) Python loops with static-trip
   `jax.lax.scan` carries (`_scan_layer_temp`) over the flattened `(jl, jk)` index
   sequence `(jl * nlay + jk for jl in range(jpl) for jk in range(nlay))`, citing
   NEMO 5.0.2 `src/ICE/icevar.F90` lines 181–191 (`tm_i`) and lines 192–200
   (`tm_s`). The eager path (`not _si3_traced(...)`) is preserved untouched.

2. **Secondary (`si3_o15_step` redistribution and `ice_cor` loops)**:
   - Extracted `si3_ice_cor_salinity_bounds` (`src/ICE/icecor.f90` lines 114–122,
     50-step `sz_i` clipping loop with static-trip `jax.lax.fori_loop` when traced)
     in `packages/ice/legoesm/ice/itd.py` lines 1344–1404, and threaded optional
     `traced: bool | None = None` through `si3_ice_itd_reb`,
     `si3_ice_cor_salinity_bounds`, and `si3_var_zapsmall`.
   - Routed `si3_o15_step`'s `_ice_itd_reb` (`packages/ice/legoesm/ice/si3_finalize.py`
     lines 440–594) and `si3_ice_cor` (`packages/ice/legoesm/ice/si3_finalize.py`
     lines 597–819) through `si3_ice_itd_reb`, `si3_ice_cor_salinity_bounds`, and
     `si3_var_zapsmall` when `_si3_traced(...)` is True and
     `not _plant_strip_transfer_barriers`. The eager path and explicit plant path
     remain unchanged.

3. **Codex candidate audit (isolated reverse tracing, lowering, compilation, and execution under `timeout 120`)**:
   All five candidate regions from `review_d198r4_codex.log` were measured via
   `scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_ad_probe.py --reproduce-d198r5`
   with tracing (`jax.jit(jax.value_and_grad(fn)).trace(...)`), MLIR lowering
   (`.lower()`), XLA compilation (`.compile()`), and execution timed separately:

   | Candidate region | Mode | Tracing (s) | Lowering (s) | Compilation (s) | Lower+Compile (s) | Execute (s) | First-call total (s) | Disposition |
   | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
   | `si3_var_agg(kn=2)` (`si3_orca2_state.py` lines 1176–1211, 50+25 steps) | unrolled | `0.1599` | `0.2194` | `20.9963` | `21.2158` | `0.0040` | `21.3797` | **Converted (Primary)** |
   | `si3_var_agg(kn=2)` (`si3_orca2_state.py` lines 1176–1211, 50+25 steps) | scan | `0.0954` | `0.0562` | `1.3900` | `1.4461` | `0.0028` | `1.5443` | **Converted (`14.7x` lower+compile speedup)** |
   | `si3_o16_step` (`si3_finalize.py` lines 1018–1182, calls `si3_var_agg(kn=2)`) | unrolled | `0.6929` | `0.2640` | `42.2690` | `42.5330` | `0.0252` | `43.2511` | **Converted via Primary** |
   | `si3_o16_step` (`si3_finalize.py` lines 1018–1182, calls `si3_var_agg(kn=2)`) | scan | `0.4830` | `0.1112` | `6.1436` | `6.2547` | `0.0086` | `6.7463` | **Converted (`6.8x` lower+compile speedup)** |
   | `si3_o15_step` (`si3_finalize.py` lines 440–819: `_ice_itd_reb` + `si3_ice_cor` 50-step `sz_i` + `zapsmall`) | unrolled | `0.9037` | `0.7611` | `73.3627` | `74.1238` | `345.9344` (`>120 s`!) | `420.9619` | **Converted (Secondary)** |
   | `si3_o15_step` (`si3_finalize.py` lines 440–819: `_ice_itd_reb` + `si3_ice_cor` 50-step `sz_i` + `zapsmall`) | traced `fori_loop` | `0.5881` | `0.2074` | `5.9396` | `6.1470` | `0.0097` | `6.7448` | **Converted (`12.1x` lower+compile, `35,663x` execute speedup)** |
   | `si3_o15_step` slice `11..11` in `si3_o17_step` (`4` inputs) | traced `fori_loop` | `0.4606` | `0.1811` | `4.4694` | `4.6505` | `0.0063` | `5.1173` | **Verified via `--slices 11-11`** |
   | `si3_o16_step` slice `13..13` in `si3_o17_step` (`4` inputs) | scan | `0.3886` | `0.0956` | `5.1079` | `5.2035` | `0.0060` | `5.5982` | **Verified via `--slices 13-13`** |
   | `si3_o15..o17` tail slice `11..14` in `si3_o17_step` (`4` inputs) | traced `fori_loop` + scan | `0.8588` | `0.2896` | `10.1725` | `10.4622` | `0.0097` | `11.3307` | **Verified via `--slices 11-14` (was `118.43 s` in r4)** |
   | O12 10-step pond permeability loop (`si3_thd_pnd.py` lines 247–260) | unrolled | `0.0363` | `0.0117` | `0.2233` | `0.2351` | `0.0005` | `0.2719` | **Not converted (`0.235 s` compile, `0.0005 s` exec)** |
   | O7/O9 10-step layer enthalpy sum (`si3_orca2_state.py` lines 768–777, 997–998) | unrolled | `0.0211` | `0.0076` | `0.0884` | `0.0960` | `0.0004` | `0.1175` | **Not converted (`0.096 s` compile, `0.0004 s` exec)** |

### Proofs

1. **Proof 1 — Forward bits of O4–O16 and full O17 step (1,209 leaves) unchanged vs `3dd560f7f05f` (eager and `jax.jit`, signed zeros and NaN bits included) + negative-zero fixture**:
   - **Full O17 step (`1,209` leaves)**:
     - Eager (`si3_orca2_o17_step`): `0` raw `uint64` bit diffs across all `1,209`
       leaves vs `3dd560f7f05f` eager (`sha256 = 2a1e9da3ab8fb24f44186c944132c94e864939e6a78c30a6b8429d0d4504532f`,
       preserving all `17,734` `-0.0` entries and `44` NaNs bit-for-bit).
     - Compiled (`jax.jit(si3_orca2_o17_step)`): `0` raw `uint64` bit diffs across
       all `1,209` leaves vs `3dd560f7f05f` `jax.jit` (`sha256 = e9ac6d417425adbc424ce8470041e3fe980e57d5e9891449d9f8586ac6a9052c`,
       preserving all `17,734` `-0.0` entries and `44` NaNs bit-for-bit;
       uncached first-call `jax.jit` forward compile+run improved from `80.02 s`
       on `3dd560f7f05f` to `67.39 s` on Round 5).
   - **Subphase fixtures (`si3_var_agg(kn=2)`, `si3_o15_step`, `si3_o16_step`)**:
     Pinned in `tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py`:
     - `si3_var_agg(kn=2)` on `si3_o16_step` inputs (`21` float64 output leaves):
       `eager == jax.jit == 3dd560f7f05f` with `0` raw bit diffs
       (`sha256 = 7e9e4c30fb8800c9f5f87d0fa7703097d7f4616c218d9699bb725517777e7bc7`).
     - `si3_o15_step` (`103` output leaves):
       `eager == jax.jit == 3dd560f7f05f` with `0` raw bit diffs
       (`sha256 = d16fb687141990f4593ab6bf83fbda1b03c173c1aa3640ac3b7ecba51065785d`,
       preserving all `69` `-0.0` entries across `13` leaves).
     - `si3_o16_step` (`66` output leaves):
       `eager == jax.jit == 3dd560f7f05f` with `0` raw bit diffs
       (`sha256 = 95a889ca4c0940b30bb5084e191dcdeff20cc66727cb59337d31ef4c36d4a58a`).
   - **Negative-zero fixture (`build_negative_zero_var_agg_fixture`)**:
     Constructed in `scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_ad_probe.py`
     and tested in `test_si3_var_agg_negative_zero_fixture_survives_eager_and_jit`:
     injects IEEE-754 `-0.0` (`0x8000000000000000`) into every input leaf of
     `si3_var_agg(kn=2)` at cell `(5, 7)` (`a_i=-0.0`, `v_i=-0.0`, `v_s=-0.0`,
     `t_i=-0.0`, `t_s=-0.0`, `e_i=-0.0`, `e_s=-0.0`, `sv_i=-0.0`, `a_ip=-0.0`,
     `v_ip=-0.0`, `v_il=-0.0`, `t_su=-0.0`, `oa_i=-0.0`, plus `-0.0` at `(6, 8)`
     where `at_i == +0.0` and at `(7, 9)` where `a_i > 0` and `v_i == -0.0`).
     All `17` negative-zero output entries across `15` leaves (`vt_i`, `vt_s`,
     `st_i`, `at_i`, `ato_i`, `et_i`, `et_s`, `tm_su`, `tm_si`, `om_i`, `sm_i`,
     `at_ip`, `hm_ip`, `vt_ip`, `vt_il`) survive identically in eager and
     `jax.jit` (`0` raw `uint64` bit diffs,
     `sha256 = 7fb0616b803f7b0d620776479e65a2a89df712d166d5e03923a9b3dddb466c41`).

2. **Proof 2 — Planted loop-order swap and planted barrier removal fail `test_si3_d198_r5_var_agg_o15_traced.py`**:
   - **Plant 1 (loop-order swap in `si3_var_agg(kn=2)`)**:
     In `packages/ice/legoesm/ice/si3_orca2_state.py` lines 1185–1188, swapping
     the scan index table order from `(jl, jk)`:
     `tuple(jl * nlay + jk for jl in range(jpl) for jk in range(nlay))`
     to `(jk, jl)`:
     `tuple(jl * nlay + jk for jk in range(nlay) for jl in range(jpl))`
     causes `test_si3_var_agg_kn2_eager_equals_jit_and_parent_digest` to FAIL
     (`AssertionError: si3_var_agg(kn=2) eager vs jit raw-bit mismatch`,
     compiled digest shifts from
     `7e9e4c30fb8800c9f5f87d0fa7703097d7f4616c218d9699bb725517777e7bc7` to
     `143944c1febff4c8c0b2db7e654fcc54e29f7376832ea8283611cc507ce8b10e`).
     Restored and verified clean.
   - **Plant 2a (barrier removal in `si3_var_agg`)**:
     In `packages/ice/legoesm/ice/si3_orca2_state.py` line 1119, removing `sr(...)`
     around `t_su * a_i`:
     `jnp.sum(t_su * a_i, axis=0)` instead of `jnp.sum(sr(t_su * a_i), axis=0)`
     causes `test_si3_var_agg_kn2_eager_equals_jit_and_parent_digest` to FAIL
     (`AssertionError: si3_var_agg(kn=2) eager vs jit raw-bit mismatch`,
     compiled digest shifts to
     `5d3f3d5a36a39bdfdd01777fa30901e337f2e051391cf8d2958ac8810f1502ee`).
     Restored and verified clean.
   - **Plant 2b (barrier removal in `si3_ice_cor_salinity_bounds` used by `si3_o15_step`)**:
     In `packages/ice/legoesm/ice/itd.py` line 1375, removing `sr(...)` around
     `s_i_new[:, :, jl]`:
     `s_i_max = jnp.minimum(s_i_new[:, :, jl], rn_simax)` instead of
     `s_i_max = jnp.minimum(sr(s_i_new[:, :, jl]), rn_simax)`
     causes `test_si3_o15_ and_o16_eager_equals_jit_and_parent_digest` to FAIL
     (`AssertionError: si3_o15_step eager vs jit raw-bit mismatch`,
     compiled digest shifts from
     `d16fb687141990f4593ab6bf83fbda1b03c173c1aa3640ac3b7ecba51065785d` to
     `48a1fc2b4689154faccf57e8872689a22eac72be43544f3c5e82f8f71350cb69`).
     Restored and verified clean.

3. **Proof 3 — Per-entry-relative central FD checks on `si3_var_agg(kn=2)` outputs (`rtol=1e-6`)**:
   - `test_si3_var_agg_kn2_per_entry_fd_gradients` checks reverse-mode AD cotangents
     against central finite differences (`eps = 1e-6`) on `tm_i` and `tm_s` with
     respect to `t_i`, `v_i`, `t_s`, and `v_s` across all active cells (`8`
     input-output pairs): baseline passes with max per-entry relative errors
     `1.954e-10` (`t_i -> tm_i`), `3.679e-11` (`v_i -> tm_i`),
     `1.519e-10` (`t_s -> tm_s`), `3.211e-11` (`v_s -> tm_s`).
   - **Plant 3 (`jax.lax.stop_gradient` on `_scan_layer_temp` carry)**:
     In `packages/ice/legoesm/ice/si3_orca2_state.py` line 1193, wrapping
     `jax.lax.stop_gradient(acc_next)` fails `4` FD cases (`t_i -> tm_i` rel err
     `1.0`, `v_i -> tm_i` rel err `0.08629`, `t_s -> tm_s` rel err `1.0`,
     `v_s -> tm_s` rel err `0.10824`). Restored and verified clean.
   - **Plant 4 (`1 + 1e-5` tangent scaling on `_scan_layer_temp` carry)**:
     In `packages/ice/legoesm/ice/si3_orca2_state.py` line 1193, scaling tangent
     by `1.0 + 1e-5` via `acc_s + (1.0 + 1e-5) * (acc_next - acc_s)` fails `4`
     FD cases (`t_i -> tm_i` rel err `8.000e-5`, `v_i -> tm_i` rel err `5.323e-5`,
     `t_s -> tm_s` rel err `8.000e-5`, `v_s -> tm_s` rel err `1.505e-4`, all
     exceeding `rtol=1e-6`). Restored and verified clean.

4. **Proof 4 — Equation counter updated with historical snapshots kept**:
   `scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_jaxpr_count.py`
   preserves `PRE_D194_479088F27B`, `D194_920AA4AF24`, `D198_R1_4DD73128E5`,
   `D198_R2_56B9AC5819`, and `D198_R3_22AF29B743` (`162,751` total recursive
   equations, `88,876` `remaining_named_and_inlined`), and updates `EXPECTED` to
   Round 5's measured counts:
   - `total_equations_recursive`: `162,751` -> `141,604` (`-21,147` equations,
     `-13.0%` vs Round 3, `-44.3%` vs pre-D194 `254,322`).
   - `top_level_equations`: `73,906` -> `73,893` (`-13`).
   - `scan` equations: `5` -> `7` (`+2` from `tm_i` and `tm_s` in `si3_var_agg(kn=2)`).
   - `while` (`fori_loop`) equations: `26` -> `44` (`+18` from routing `si3_o15_step`
     through `si3_ice_itd_reb`, `si3_ice_cor_salinity_bounds`, and `si3_var_zapsmall`).
   - `remaining_named_and_inlined`: `88,876` -> `67,729` (`-21,147`, well below the
     `110,000` ceiling).

5. **Proof 5 — Citation gate re-anchored and passing**:
   Re-anchored the `29` shifted citations in `packages/ice/legoesm/ice/si3_orca2_state.py`
   and `2` shifted citations in `packages/ice/legoesm/ice/si3_finalize.py` in
   `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py`;
   `pytest tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py` passes
   (`45 passed`).

6. **Operator acquisition script for full-step `value_and_grad` execution and >20 min verification battery**:
   Written to `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r5_scratch/operator_run.sh`
   (runs full-step `si3_o17_step` reverse compile + execute with separated
   tracing/lowering/compilation/execution timers and peak RSS under a 120 GiB
   child-only memory watchdog and `timeout 10800`, followed by the clean-checkout
   test and scorer battery).

independent review: operator

