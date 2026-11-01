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
