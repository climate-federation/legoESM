# NEMO testcases L2 GYRE — lane-1/lane-2 merge reconciliation receipt

Merge: `c9526e585` ("Merge fidelity/nemo-branch-isomorphism-audit into lane-2
GYRE"), parents `b2f7c298411f` (lane-2 GYRE tip,
`fidelity/nemo-testcases-l2-gyre-codex`) and `e4263111f1e7` (lane-1
isomorphism-audit tip, `fidelity/nemo-branch-isomorphism-audit`), authored
2026-09-03. Full hunk-by-hunk resolution narrative is in the merge commit
message; this receipt is the per-hunk table an independent review asked for,
extended with three rows for undisclosed bundled changes it found in the M1
hunk of `ocean_model_latlon_cgrid.py`.

## Per-hunk table

| hunk | file | resolution | NEMO citation | verdict |
|---|---|---|---|---|
| B1 kwarg list | barotropic_latlon_cgrid.py | iso (`nemo_flux_form_update_test_override` added) | — | as merged |
| B2 substep trace frame | barotropic_latlon_cgrid.py | UNION — one name-keyed frame: iso's 19 names + l2's `trd_u`/`trd_v` | dynspg_ts.F90:686-701 (Coriolis+drag as one trend) | as merged |
| B3 scan dispatch | barotropic_latlon_cgrid.py | iso (trace is a harness artifact; l2 duplicate + trailing `if return_trace` removed as unreachable) | — | as merged |
| B4/B5/B6 plumbing | barotropic_latlon_cgrid.py | iso (`_return_substep_trace` and `_nemo_substep_trace_test_hook` were the same mechanism; one kept) | — | as merged |
| M1 rk3_ws momentum guard (flux/vector-invariant union) | ocean_model_latlon_cgrid.py | UNION — two complete momentum programs; iso UP3 selector split applied to the flux arm | dynadv_up3.F90:166,169-172 | as merged |
| M2 barotropic seed | ocean_model_latlon_cgrid.py | UNION of both test-hook blocks | — | as merged |
| M3/M4 implicit vmix | ocean_model_latlon_cgrid.py | UNION (l2 `effective_K_test_override` + iso `nemo_aimp_*`) | trazdf.F90:219-221, dynzdf.F90:200-203 | as merged |
| pgf_quadrature allow-list | ocean_model_latlon_cgrid.py | WIDENED to `{"nemo_sco","adcroft"}` (not weakened — both pair the same e3w(Kmm) trapezoid recurrence) | dynhpg.F90:270-296 (hpg_zco), :343-374 (hpg_sco) | as merged |
| pgf_scheme identity | testcase card validation | RELOCATED from model- to card-validation (`validate_nemo_testcase_card` pins `pgf_scheme="nemo_sco"`) | — | as merged |
| **M5 barotropic live-split + `nemo_ab3am4` filter guard** | ocean_model_latlon_cgrid.py | **DELETED** (verified in this receipt) | dynspg_ts.F90:199-200 (`ll_bt_av=.FALSE.` iff `nn_bt_flt==3`), :217-223 (cold-start `ll_init=.TRUE.` when `LN_RSTART=F`), :270-302 (RK3 Phase-1 branch, the one this build compiles per `provenance/cpp_GYRE_OMIP_L2_P3.fcm: key_RK3`; NOT the MLF branch at :303-452), :296/:299 (pre-step `dyn_cor_2D`/subtraction), :535-543 (mid-step za1/za2/za3, unconditional on `nn_bt_flt`), :487 (`un_e` seeded from `puu_b(:,:,Kmm)` under `LN_BT_FW=T`), :689 (live substep `dyn_cor_2D`), :1270 ("Demange time filter" label for `nn_bt_flt=3`) | **CONFIRMED fix.** GYRE's own namelist selects `NN_BT_FLT=3` with `LN_BT_FW=T` (`output.namelist.dyn:488,492`), and NEMO's mid-step AB3 extrapolation (:535-543) is not conditioned on `nn_bt_flt` at all — no NEMO code path avoids it, so the deleted guard's "use the boxcar filter instead" escape does not exist in NEMO. The guard's own claim ("substep-0 would not cancel bit-exactly") is also false at a cold start: `ll_init=.TRUE.` at `kt==nit000` (confirmed `ln_rstart=F` at `ocean.output:226`), so jn=1 gets za1=1 and `un_e==puu_b(:,:,Kmm)` exactly — the pre-step and jn=1 live Coriolis calls are bit-identical inputs and cancel exactly; jn>=2 and every later kt do not, which is NEMO's intended AB3 evolution, not an incompatibility. Test: `test_live_coriolis_split_runs_with_the_ab3am4_filter` (constructs the GYRE card, asserts no raise). Non-vacuity: temporarily restored the deleted `raise` and confirmed the test fails (`ValueError: ... incompatible with barotropic_time_filter="nemo_ab3am4"`), then reverted. |
| **M6 barotropic Coriolis stencil requirement (ene/een)** | ocean_model_latlon_cgrid.py | **WIDENED** from hardcoded `("een","een_metric")` to `("ene","ene_metric")` when `vorticity_scheme=="ene_total"`, else `("een","een_metric")` (verified in this receipt) | dynspg_ts.F90:1326 (`dyn_cor_2D_init` `SELECT CASE(nvor_scheme)`), :1327-1381 (`np_EEN` 3-point triads), :1383-1410 (`np_ENE`/`np_MIX` 2-point Sadourny coefficients, `ff_f(ji,jj)` north pair / `ff_f(ji,jj-1)` south pair), :1483-1506 (`dyn_cor_2D` applies whichever set `dyn_cor_2D_init` built); dynvor.F90:63-75 (`np_ENE=1`, `np_EEN=3` parameters), :829-875 (`nvor_scheme` is a single module variable set once, from the SAME `ln_dynvor_ene`/`ln_dynvor_een` namelist flags the 3-D vorticity operator reads) | **CONFIRMED fix.** GYRE's namelist selects `LN_DYNVOR_ENE=T`, `LN_DYNVOR_EEN=F` (`output.namelist.dyn:469,471`), so `nvor_scheme=np_ENE` globally — the SAME variable `dyn_cor_2D_init` switches on. The old hardcoded `een`-only requirement would have refused the ENE stencil NEMO's own barotropic solver builds for this run. Test: `test_ene_vorticity_requires_the_ene_barotropic_coriolis_not_een`. Non-vacuity: temporarily hardcoded `_required_bt=("een","een_metric")` and confirmed the test fails (`ValueError: vorticity_scheme="ene_total" with ... requires ... ('een', 'een_metric') ... Got barotropic_coriolis='ene_metric'`), then reverted. |
| **M7 rk3_ws `gm_redi` guard** | ocean_model_latlon_cgrid.py | **NARROWED** from "refuse any `gm_redi`" to "refuse only `kappa_GM != 0`" (verified in this receipt) | traadv.F90:208 (`IF( ln_ldfeiv .AND. .NOT. ln_traldf_triad ) THEN ! Add the eiv transport`); ldftra.F90:536 (`IF( .NOT.ln_ldfeiv ) THEN !== Parametrization not used ==!`); `packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi.py:11-13` (skew-flux bolus `psi = kappa_GM * S`, both flux terms linear in `kappa_GM`) | **CONFIRMED fix.** NEMO gates the GM bolus transport by `ln_ldfeiv` independently of `ln_traldf_iso` (Redi diffusion); GYRE's namelist runs `LN_TRALDF_ISO=T` with `LN_LDFEIV=F` (`output.namelist.dyn:352,363`) — pure Redi, no bolus, which is exactly what the narrowed guard now admits. The old "refuse any `gm_redi`" guard would have refused NEMO's own GYRE configuration. Test: `test_rk3_ws_admits_pure_redi_but_still_refuses_a_staged_gm_bolus` (asserts the GYRE card's `kappa_GM==0.0`, `kappa_Redi>0.0`, and that setting `kappa_GM=1.0` still raises). Non-vacuity: temporarily restored "refuse any non-None `gm_redi`" and confirmed the test fails (`ValueError: NEMO rk3_ws does not yet support staged GM bolus transports`), then reverted. |

Two accuracy corrections made to the in-code comments while verifying M5 (both
comment-only, no logic change): the citation `dynspg_ts.F90:359` was the MLF
(non-RK3) branch's copy of the pre-step Coriolis subtraction, which this GYRE
build never compiles (`provenance/cpp_GYRE_OMIP_L2_P3.fcm` selects `key_RK3`
only) — corrected to `:296`, the RK3 branch actually executed; and the M6
comment's blanket "the subtraction must ALSO be EEN" was stale after the
widening (true only for the `een_total` arm) — corrected to state both arms.

## Item 2 — gate script provenance and before/after

### The bug and the fix

`nemo_testcase_l2_gyre_phase3_gate.py` read `trace.substeps` (the barotropic
solver's substep record) by fixed position (`BT_SUBSTEP_NAMES.index(name)`).
The merge unified the two branches' rival `return_trace` mechanisms
(receipt Item 1 / hunks B2-B6) into ONE name-keyed dict carrying iso's 19
names plus l2's `trd_u`/`trd_v`. A positional-only reader raises `KeyError: 0`
against that dict (an int key on a dict of string keys) on the merged tree.

Fix (already drafted, completed and verified here): `bt_frame(substeps, name)`
dispatches on `isinstance(substeps, dict)` — dict path keys by name (via
`BT_TRACE_KEY` for the two renamed fields), tuple/list path indexes by
`BT_PRE_MERGE_ORDER.index(name)` (b2f7c298's own `BT_SUBSTEP_NAMES` order,
verbatim). Both the pre-merge tree (`b2f7c298411f`, whose own
`ocean_model_latlon_cgrid.py` returns a positional tuple from
`_return_barotropic_substeps=True`) and the merged tree (name-keyed dict) are
read by the SAME committed script — no separate normalisation step needed,
because the public trace-request kwarg name (`_return_barotropic_substeps`)
is identical on both trees (only its internal plumbing differs: b2f7c298
seeds `_return_substep_trace`, the merge seeds iso's
`_nemo_substep_trace_test_hook` — receipt Item 1's B4/B5/B6 row). Also fixed:
`read_bt_substeps` now accepts the oracle dump's actual on-disk format
(`NEMO_L2_BTSUB_1`, 18 fields, no `cor_u`/`cor_v`) in addition to the planned
but never-generated format 2; a `--without-oracle-ene-coefficients` flag
WAIVES the ENE-coefficient arm (recorded `UNMEASURED`, not silently skipped)
since `oracle_bt_ene_coeff_kt00000001.bin` does not exist on disk (confirmed:
absent from `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyre_kt1_10/`).

### Runs

| | commit | worktree | command | report sha256 |
|---|---|---|---|---|
| BEFORE | `b2f7c298411f` (`fidelity/nemo-testcases-l2-gyre-codex`) | `/tmp/wt-gyre-before` (temporary, removed after) | `cd /tmp/wt-gyre-before && PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_phase3_gate.py --without-oracle-ene-coefficients --output /data/abyssal/dbalwada/nemo-testcases-l2/reconcile/before/legoesm_phase3_gate.json` | `a4f8683ab8c2fad15d54cb082054ad38bafd10ff51c6ca28ace51ec461dd49b8` |
| AFTER | `196a65f5df` (this branch, `c9526e585` + Item 1 commit) | `/tmp/wt-gyre` | same command, `cd /tmp/wt-gyre`, `--output .../reconcile/after/legoesm_phase3_gate.json` | `f467f08979ee0215d3ca5ed18ff8a6bb92bebbb8923f38be0e36b8fe274ddb99` |

Both runs exit 1 (`status: DEBT`) — expected per the skill's own framing: DEBT
is a scientific finding, not a harness failure. `--without-oracle-ene-coefficients`
is required on both because the instrumented run never emitted the ENE dump
(true on both trees, not a merge artifact).

### Reproduced before/after table (kt=1..10, eval protocol identical on both)

| check | BEFORE | AFTER | verdict |
|---|---|---|---|
| kt=1 T/S/u/v/ssh | all `exact=True`, absolute_max=0.0 | all `exact=True`, absolute_max=0.0 | EXACT, both sides, UNCHANGED |
| first_over_bar | kt=2, fields `[T,S,u,v,ssh]` | kt=2, fields `[T,S,u,v,ssh]` | UNCHANGED |
| exact_prefix_entering (kt=1..10) | `[T,T,F,F,F,F,F,F,F,F]` | `[T,T,F,F,F,F,F,F,F,F]` | UNCHANGED |
| kt=2..10 rows (45 = 9 kt x 5 fields) | — | 23 better / 22 worse / 0 identical | median\|ratio-1\|=8.584e-4, max=1.212e-1 (`ssh` kt=8, `0.5246e-3`->`0.5881e-3`) |
| barotropic substep rows (800 = 50 substeps x 16 names) | 126 AT-BAR / 674 DEBT | 126 AT-BAR / 674 DEBT | tally UNCHANGED; 382 identical / 214 better / 204 worse; median\|ratio-1\|=7.289e-14 (both sides fp64-close); the single max-ratio pair (`u_entry` jn=1, ~1e-24 vs 0.0) is float noise between two AT-BAR values, not a regression |
| overall `status` | DEBT | DEBT | UNCHANGED |

This reproduces the merge commit message's own before/after table (which used
a third common-revision script, `efdef59b2`, because neither endpoint's own
script ran standalone at the time) to the same precision: 23/22,
8.6e-4/1.2e-1, 382/214/204, 7.4e-14, and 126/674 all match. The movement in
the kt=2..10 and barotropic rows is EXPECTED (Rule 8, oracle-fidelity skill):
the RK3 ladder-collapse re-association the iso side landed was proven for
`f=0` only, and GYRE rotates, so its `O(dt^2 f)` Matsuno channel changed.
Nothing here is claimed MATCHED; the case remains DEBT, unchanged from
before the merge.

Raw reports: `/data/abyssal/dbalwada/nemo-testcases-l2/reconcile/{before,after}/legoesm_phase3_gate.json`.
