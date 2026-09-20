# GYRE NEMO-fidelity ORCA2 merge receipt — 2026-09-20

Status: **MERGED; GYRE BIT-IDENTICAL; SIX OF SEVEN ORCA2 GATES REPRODUCED.**
The model-hunk inventory below was committed on clean GYRE lane tip
`4cac617cd` before merging ORCA2 tip `4092639c3d32`.  The only unreproduced
historical gate imports the separate L3 SI3 implementation, which is absent
from both parents and is therefore a decision rather than merge repair.

## Scope and provenance

The merge target is branch `fidelity/orca2-on-lane`.  ORCA2 tip
`4092639c3d32` was 170 commits ahead of and 1,241 commits behind the target;
their merge base is `03c6e8d96ff7`.  Preregistration commit `44c2cfefb`, merge
commit `5498e8031`, integration repair `faad1b9a0`, rigid citation re-anchor
`fc4b7e573`, and selector/mask union-test reconciliation `2a6318c42` form the
finished unit.  Evidence is written under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_merge/`.

The lane already admits the native post-boundary/pre-Langmuir TKE frame, but
does not contain the certified ORCA2 assembly.  The ORCA2 branch's Phase-2y
handoff admits `VARIANT_ORACLE_ORCA1ICE`; its Phase-2w handoff pins the
icebergs-off `VARIANT_ORACLE_V2` and stops the ordered TKE walk at the first
non-bit statement, `zpelc` at kt=2.  Phase 2z staged two later SI3 frames but
did not execute or admit them.

## Pre-merge model-hunk inventory

This table records every non-merge ORCA2-side commit that changes `packages/`
before conflict resolution.  The synchronization merge `5bbb9ee56` imported
the common GYRE parent `03c6e8d96ff7`; it has no unique package delta relative
to that second parent and is not an ORCA2-authored model hunk.

| ORCA2 commit | Package hunk and purpose |
|---|---|
| `2c60066a9` | Adds the fail-closed, external-deck `ORCA2-zps` card; reads native 3-D partial cells, masks, metrics, initial T/S and analytic ice-load SSH; selects EOS-80/EEN/RGB/RK3; admits EEN in the shared WS-RK3 composition; reads literal `ff_t/ff_f`. |
| `af54d11ef` | Separates native NEMO F-point `ff_f` from generic V-face Coriolis, threads it through geometry padding/slicing, and routes only EEN/ENE vorticity consumers through the native F-point map. |
| `3a3f12e03` | Makes iceberg state explicit card metadata and pins the comparison card to `icebergs_enabled=False` with no iceberg inputs. |
| `ee1bbfcc2` | Adds the shared pure-JAX NEMO `fld_read` bilinear/bicubic map, source-index decoder, scalar-libm grid rotation, and east/north-to-i/j rotation for ORCA cards. |
| `5260fa8ae` | Selects `nemo_fld_read` explicitly on the ORCA2 card and removes that arm from its unresolved-feature list. |
| `c650babc6` | Adds policy-controlled scalar-libm `log/log10` and the source-statement NEMO RGB shortwave/chlorophyll/extinction path, wired with live/reference depth operands. |
| `ec576f51a` | Registers the ORCA2 stage-1 WZV diagnostic record's NEMO time levels; diagnostic metadata only. |
| `020a5043b` | Threads surface runoff into the shared WZV transport divergence and factors the literal horizontal-divergence and bottom-up W recurrences into shared helpers. |
| `d061ce395` | Adds private WRITE-only ORCA2 stage-1 W exposure/external-endpoint test hooks and strengthens vector/C2 card validation; no public card can select the hooks. |
| `f31af67d4` | Corrects the ORCA2 card validator's kinetic-energy-gradient selector spelling from `centered` to the actual `c2`. |
| `e30e7a93b` | Temporarily allows the private W diagnostic to construct around the then-uncertified EOS-80 geometric-depth guard. |
| `7aa560bc5` | Narrows that EOS bypass to construction of the exact private diagnostic combination, leaving ordinary execution fail-closed. |
| `1b7ea8be3` | Extends the same private construction-only bypass to the stage-1 tracer-boundary diagnostic. |
| `4b7eb88ee` | Maps native NEMO U metrics and rotation to redundant legoESM U faces with native U(i) at index `i+1` and the periodic last face at index 0. |
| `311654c63` | Applies NEMO's F-origin north-fold overwrite to live QCO vorticity thickness and its live stretch before EEN consumption. |
| `b51b3d519` | Adds shared NEMO diffusive-BBL geometry, EOS-80 alpha/beta coefficients, coefficient gate and bottom-tracer tendency; wires independent advective/diffusive BBL selectors and selects `nn_bbl_ldf=1` on ORCA2. |
| `d3e3aafc4` | Carries the frozen free-slip `fe3mask` separately from later slip/strait `fmask` edits and uses it in partial-cell live EEN thickness. |
| `01dcc1149` | Admits the measured EOS-80 geometric-depth combination in the model validation allow-list. |
| `9f11b5784` | Replaces the provisional adjacent-ULP BBL quotient search with the source-ordered, materialized ordinary IEEE division. |
| `fde040e91` | Seeds ORCA2's carried pre-closure TKE, viscosity/diffusivity, surface viscosity and dissipation fields from the resolved NEMO initialization, including the latitude-band tracer-diffusivity factor. |
| `076be0932` | Selects and validates ORCA2's RK3 `zdf_sh2` identity: face-native Nbb×Nbb velocity, face AVM weighting, and live-QCO face metric. |
| `a20407e3b` | Selects the executed ORCA2 bathymetry-relative bottom TKE Dirichlet boundary and corrects its config documentation. |
| `50c2b643c` | Renames the RK3 shear selector from ambiguous `now2` to the NEMO time-level name `nbb2` across validation and vertical-mixing dispatch. |
| `2efc0ce15` | Restores NEMO `nn_eice=1` as `tanh(10*fr_i)` through one shared dispatcher and selects it on ORCA2; default remains mode 0. |
| `ab223d8cd` | Removes remaining ambiguous “now” wording for Nbb; comments/docstrings only. |
| `bb1c60beb` | Adds NEMO `nn_eice=2` as raw ice fraction to the same shared dispatcher and validators; no default or existing card selection changes. |

## Textual conflicts

There were exactly four textual conflicts.  Every resolution was derived from
the executed compiled source, not by choosing a parent wholesale.
All ORCA2 source citations in this table are relative to
`cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/`; the MLF citations are relative to
`cfgs/DINO/BLD/ppsrc/nemo/`.  The two selector citations are the named
configuration's `EXP00/namelist_cfg`.

| File | Lane intent | ORCA2 intent | Resolution | Executed NEMO statement |
|---|---|---|---|---|
| `ocean_model_latlon_cgrid.py` | Preserve the GYRE stage/test hooks, process trace and first-step MLF ordering. | Add the narrowly scoped private EOS-80/geometric diagnostic construction bypass and carry runoff/bathymetry into WZV. | Union.  The bypass remains private and construction-only; runoff and `H_bathy` are shared operands.  Both RK3 shear aliases reach one Nbb-by-Nbb implementation, and stage calls use their literal stage-local `dt` rather than an erroneous common outer value. | ORCA2 `eosbn2.f90:1279-1332` forms EOS-80 coefficients from geometric depth; `sshwzv.f90:126-137,345-355` forms horizontal divergence and the runoff correction; `sbcrnf.f90:253-260` supplies runoff.  DINO `stpmlf.f90:131-134,451-466,534-535,617-620` and `istate.f90:139-141` fix the first-step MLF ordering. |
| `ocean_pe_latlon_cgrid.py` | Retain GYRE's shortwave selector/water type and diagnostic operand. | Add RGB's explicit step length plus native `ff_f`, north-fold and live-`e3f` EEN operands. | Union.  GYRE keeps 2BD and ORCA2 keeps RGB; selecting both configuration seams raises a named error instead of silently choosing one.  The native F-point/fold/thickness path is selected only where configured. | ORCA2 `traqsr.f90:210-217,1080,1147-1157` executes RGB, selected by `EXP00/namelist_cfg:136`; GYRE selects 2BD at its `EXP00/namelist_cfg:78`.  ORCA2 `dynvor.f90:738-770,907-937` consumes `ff_f`, folded live thickness and `fe3mask`. |
| `nemo_testcase_recipe.py` | Keep every GYRE card value, the existing-state `uu_b`/`vv_b` guard and its defaults. | Add a distinct fail-closed ORCA2 card and the six-field BBL expectation. | Union.  ORCA2 is a separate card, explicitly requests prognostic `uu_b`/`vv_b`, and the GYRE card and all defaults are unchanged. | ORCA2 `dynspg_ts.f90:136-164,797-860` establishes the external-mode state pair; `trabbl.f90:182-195` establishes the BBL operands. |
| `test_tke_nemo_terms.py` | Retain the lane's `ln_mxl0` tests. | Add `nn_eice=1/2` dispatch tests. | Union of both independent test families. | ORCA2 `zdftke.f90:252-264` dispatches the three ice-attenuation modes and `zdftke.f90:835-856` initializes the mixing length. |

The compiled stage aliases were also checked directly: GYRE
`stprk3.f90:159-168` and ORCA2 `stprk3.f90:170-179` both feed the Nbb velocity
pair to `zdfsh2.f90:83-113`.  This is why the lane's existing
`nemo_face_native_now2` spelling and ORCA2's source-named
`nemo_face_native_nbb2` spelling are compatibility aliases for one arithmetic
arm, not two implementations.

## Model-hunk execution audit on the GYRE card

`yes` means some code introduced by the hunk executes; `inert` means only a
construction/default/alias path is encountered and cannot change the selected
arithmetic; `no` means the GYRE card cannot reach it.  For every `yes` and
`inert` row, the clean committed trajectory comparison in the next section is
the binding proof of bit identity.

| ORCA2 commit | GYRE execution | Reason |
|---|---|---|
| `2c60066a9` | inert | The ORCA2 card is not constructed; only the widened shared identity registration is visible. |
| `af54d11ef` | yes | GYRE EEN consumes the separately carried native `ff_f`; it is the same source operand the lane used before the refactor. |
| `3a3f12e03` | inert | New iceberg metadata takes its off/default value. |
| `ee1bbfcc2` | no | The GYRE card does not select ORCA2 `fld_read`. |
| `5260fa8ae` | no | ORCA2-card selector only. |
| `c650babc6` | no | GYRE selects NEMO 2BD, not RGB. |
| `ec576f51a` | no | ORCA2 diagnostic metadata only. |
| `020a5043b` | yes | The factored WZV recurrence executes; GYRE runoff is absent, so its arithmetic is the former recurrence. |
| `d061ce395` | no | Private ORCA2 test hooks are unselectable by the public GYRE card. |
| `f31af67d4` | no | ORCA2-card validator only. |
| `e30e7a93b` | no | Private ORCA2 diagnostic bypass only. |
| `7aa560bc5` | no | Private ORCA2 diagnostic bypass only. |
| `1b7ea8be3` | no | Private ORCA2 diagnostic bypass only. |
| `4b7eb88ee` | no | ORCA2 native-U mapping only. |
| `311654c63` | yes | The fold-aware live-thickness helper is called; non-tripolar GYRE applies no fold overwrite. |
| `b51b3d519` | inert | New BBL fields remain zero/off on GYRE. |
| `d3e3aafc4` | yes | EEN consumes `fe3mask`; on GYRE it is exactly the former `fmask`. |
| `01dcc1149` | no | ORCA2 EOS-80/geometric validation combination only. |
| `9f11b5784` | no | ORCA2 diffusive-BBL division only. |
| `fde040e91` | no | ORCA2 carried-TKE initialization only. |
| `076be0932` | inert | ORCA2's Nbb-named RK3 alias is added; GYRE retains its existing alias to the same arm. |
| `a20407e3b` | no | ORCA2 bottom-TKE boundary selection only. |
| `50c2b643c` | inert | Selector spelling compatibility only; the GYRE selection is preserved. |
| `2efc0ce15` | inert | The dispatcher is shared, but GYRE keeps `nn_eice=0`. |
| `ab223d8cd` | inert | Comments/docstrings only. |
| `bb1c60beb` | inert | Mode 2 becomes valid but is not selected; no default moves. |

## GYRE trajectory proof

Both arms were clean, committed trees: `before/` is lane tip `4cac617cd` in
`/tmp/gyre-before-orca2-920`; `after_final/` is final package commit
`fc4b7e573` in `/tmp/orca2merge-2647811388`.  Commit `2a6318c42` changes tests
only (`git diff fc4b7e573..2a6318c42 -- packages` is empty), so this evidence
covers the delivered implementation.  Both arms used CPU, fp64, the prescribed
`PYTHONPATH`, and the same commands: the ladder with `--trajectory-only
--max-step 10`; `nemo_testcase_offline_compare.py`; and the 30-day member with
`--member 0 --days 30 --snap-steps 6 --tag daily` followed by day-gap scoring.

| Comparison | Result |
|---|---|
| Offline oracle-relative comparison | `PASS`: 70 certified rows, maximum worsening 0 ULP, no violations. |
| `ladder.json` trajectory | 10 step blocks / 50 rows exactly equal; barotropic-state rows exactly equal. |
| `ladder.residuals.npz` | 210 arrays in the same order; every array passes `np.array_equal`; both files have SHA-256 `de4eea43cab0236e98b85021a13a4c34b6b121cf178ffa59a3303864504b0e23`. |
| Daily member | `day001.npz` through `day030.npz`, 30/30 byte-identical. |
| Day-gap scores | Days 1 through 30, every row exactly equal. |

The merge verdict is therefore **GYRE trajectory bit-identical: YES**.  The
earlier `after/` arm at `faad1b9a0` also passed; `after_final/` is the binding
post-re-anchor arm.

## ORCA2 gate reproduction

No NEMO binary was built or launched.  The merged-tree gates consumed only the
certified records under the Phase-2v/2y and Round-20 evidence roots.  Six of
seven recorded verdicts reproduce:

| Gate / evidence | Merged-tree result | Reproduction |
|---|---|---|
| Phase-1 full record/schema gate with planted controls (`orca2/phase1_gate.json`) | `PASS`; 4 record groups; all 10/10 plants `PASS_NONZERO`; SHA-256 `cd34c630d67aa6d19bc9fdb163d581be0d347f674b474ddde8fb2414f7f5d93f`. | yes |
| Phase-2v admission and ordered TKE walk (`orca2/tke_walk.json`) | Admission `PASS`: 101 twin and 100 inherited records.  First non-bit statement remains `zpelc` at kt=2: 39,290/242,135 cells, max absolute residual `1.7763568394002505e-15`, max row ULP 4, first cell `[1,49,2]`. | yes; the only recursive JSON difference from the archived artifact is non-verdict metadata `resolved_card.langmuir_evaluation`, archived `vectorized`, merged `nemo_literal`. |
| Phase-2y ORCA1-ice admission (`orca2/orca1ice_admission.json`) | `PASS`, 116/116 records and 26/26 plants; exact recorded SHA-256 `d3c60bf16a84f8739f8106541bea2bfcf4941e75b17e7a4f1bd0d5f7cd371297`. | yes, byte-identical |
| Phase-2y resolved ice namelist (`orca2/ice_namelist_resolved.json`) | 185 fields; exactly the recorded one-field variant difference, `namini.nn_iceini_file` 0 versus 1; self-test passes; exact recorded SHA-256 `278a891ad15cf7aecc6f8a60533634af6a79076b9650e573a840261236554013`. | yes, byte-identical |
| Round-20 Phase-1 schema over the Phase-2q root (`orca2/round20_phase1_schema.json`) | Expected exit 1 / `FAIL` for five extra records; exact archived SHA-256 `6574215cedafa818a3ef9900ced494d643fac4c95ea2aec8391d91df4b1f403f`. | yes, expected fail reproduced |
| Round-20 SI3 stream-header gate (`orca2/round20_si3_stream_headers.json`) | `VALID`; stream counts 5/140/10/25/15 for reassociation/thermodynamics/exchange/ZDF/bulk; exact archived SHA-256 `741bb4b0f7bf41ead4875726c1c01ae35689bb4c95897b2960213b8576b5494d`. | yes, byte-identical, using the unchanged historical gate at `9e4fe6f537f4` |
| Round-20 SI3 exact-input admission (`orca2/round20_exact_input_import.log`) | Cannot start: its unchanged line 18 imports `SI3ThermoConfig`, which is absent from the merged tree, lane parent, and ORCA2 parent. | **no**; the archived `STOP_SELECTOR_GAP` requires the separate L3 SI3 implementation. |

The unreproduced row is not a changed physics verdict and cannot be repaired by
choosing another conflict side: neither merge parent contains the class needed
to import its gate.  Importing that separate implementation is left explicit
under OPEN.

## Citation gate and tests

The first merged-tree citation run failed 17 receipt references and reported
45 stale current-tree map keys.  Each correction in `fc4b7e573` was a rigid
line-number shift only: old and new source blocks were compared at equal
extent and all 45 were text-identical.  The same shifts were applied to every
occurrence in the Round-8 receipt.  Two unchanged BBL keyword lines were moved
later within the same recipe constructor solely to preserve the old cited
block extent; their values and defaults did not change.  The final citation
test is `16 passed`; the CLI artifact `tests/citation_gate_final.json` reports
`PASS`, 274 citations, 0 unmapped, 0 failures, 0 map-audit failures, and all
planted violations firing.

The required four-file push gate
(`test_nemo_testcase_receipt_citation_gate.py`, `test_tke_nemo_terms.py`,
`test_nemo_recipe.py`, `test_real_freshwater_closure.py`) is **121 passed** in
381.63 s (`tests/four_file_push_gate_final.log`).

The one requested broad invocation, `pytest tests/ocean/fidelity
tests/ocean/unit -n 12`, collected 8,241 tests but eight xdist workers aborted
inside the JAX compiler with `Not properly terminated`; after five minutes
without new output at 96%, the deadlocked coordinator was interrupted.  All
624 IDs named by that run were re-run in smaller worker groups.  One remaining
compiler abort/stall left 29 IDs, which were then run one at a time in fresh
processes.  That isolated pass produced 21 passed and 8 failed.  The same eight
IDs on clean lane tip `4cac617cd` produced 2 passed and 6 failed.  The two
merge-only failures were stale assertions exposed by the union: one still
treated now-valid NEMO `nn_eice=2` as an invalid MPAS value, and one expected
legacy `fmask` where compiled NEMO freezes `fe3mask` before slip edits.  Commit
`2a6318c42` reconciles those tests; both focused tests pass.

The final eight-ID comparison is 6 failed / 2 passed on both trees, with
identical failing-ID sets: four pre-existing f32 advection-gradient cases
whose `custom_vjp` cannot JVP, and two pre-existing carried-TKE helper-fixture
cases lacking `_nemo_ws_test_hooks`.  Thus the broad-run attribution is **0
merged-only failures**, despite the resource-aborted aggregate run.  Raw and
isolated logs, exact ID lists, and both-tree comparisons are retained under
`tests/` in the evidence directory.

## Independent review

In-sandbox `codex exec` review is unavailable.  The operator will run the
required independent Claude review before shipping; no self-review is offered
as a substitute.

## Choices

- No configuration default, GYRE card value, scheme selection, threshold,
  cadence, state, or data source moved.
- ORCA2 is a separate card.  The only shared selector expansion is explicit:
  both RK3 shear names map to the same compiled Nbb-by-Nbb arm, and selecting
  both shortwave configuration seams fails loudly.
- No recorded ORCA2 verdict was hidden or rewritten.  The one unavailable gate
  is reported as unavailable and retained as an operator decision.

## OPEN

The next ORCA2 fidelity round is the shared-TKE `zpelc` statement at **kt=2**,
the first measured non-bit row in the admitted Phase-2v stream.  There is no
registered kt=1 non-bit statement: Phase-2w records the actual NEMO kt=1 entry
gate as exact.  Changing this to kt=1 would contradict the handoff and the
reproduced record.

One assembly decision remains: whether to merge the separate L3 SI3
implementation so the historical Round-20 exact-input admission gate can
import `SI3ThermoConfig` and replay its archived `STOP_SELECTOR_GAP` verdict.
That implementation is not present on either parent and was not silently
introduced by this merge.
