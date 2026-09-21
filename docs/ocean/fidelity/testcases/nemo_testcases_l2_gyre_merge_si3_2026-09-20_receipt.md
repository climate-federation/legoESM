# GYRE NEMO-fidelity SI3 merge receipt — 2026-09-20

Status: **MERGED; GYRE AND ORCA2 CONFIRMED; ONE EXTERNAL SI3 PROVENANCE
GATE BLOCKED.**

This receipt covers Decision 47: merge the certified L3 SI3 thermodynamics
tip `4e2904637273` and SI3 dynamics tip `fd8b9c130806` onto ORCA2 lane tip
`598ef772d00f`, in that order, before any ORCA2 integration round with ice.
The delivery branch is `fidelity/si3-on-lane`; evidence is written under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_merge/`.  No NEMO binary
will be compiled or launched.

## Branch inventory and order

The thermodynamics handoffs build the BL99 column from the shared constants,
bulk-flux and thermodynamic utilities through the C1D-OMIP scalar-math-v2
admission.  Its package delta is:

- `packages/core/legoesm/{constants.py,thermo.py}`;
- `packages/core/legoesm/core/bulk_flux.py`;
- `packages/core/legoesm/timestepping/tridiagonal.py`;
- `packages/ice/legoesm/ice/{__init__.py,config.py,scm.py,sea_ice.py,state.py}`;
- new `packages/ice/legoesm/ice/{bitz_lipscomb.py,c1d_omip_l3.py,constants_config.py}`;
- the retained future/reference `packages/ice/legoesm/ice/_future/bitz_lipscomb.py`.

The dynamics handoffs then add source-ordered scalar math, Prather transport,
ridging, rheology and landfast/aEVP behavior.  Its package delta is:

- `packages/core/legoesm/core/precision.py` and new
  `packages/core/legoesm/core/{source_rounding.py,transcendentals.py}`;
- `packages/ice/legoesm/ice/{__init__.py,dynamics.py,rheology.py,ridging.py,transport.py}`;
- new SI3 dynamics fidelity recipe modules under
  `packages/ice/legoesm/ice/fidelity/`.

The thermodynamics receipts culminate in a shared Thomas solver and define
the thermodynamic state/configuration consumed by the dynamics work.  Nothing
in the dynamics receipts reverses that dependency.  The registered merge
order is therefore thermodynamics first, dynamics second.

The handoff inventory was read rather than inferred from the package diff.
For thermodynamics it is the Phase-1, Phase-2, Phase-2b, bulk Phase-1,
exchange-drift, and scalar-math-V2 preregistrations/receipts under
`docs/ocean/fidelity/testcases/nemo_testcases_l3thd_*`; for dynamics it is
`nemo_testcases_l3dyn_phase1_receipt.md`,
`nemo_testcases_l3dyn_phase2_receipt.md`, and the rung-32/33/34
preregistrations.  The separate scoping branch contributes only
`si3_lane3_scoping_dossier.md`.  The resulting merge commits are
`54f3c842a05a` (thermodynamics) and `95aa1feaa2f4` (dynamics); measurement
preregistration was committed as `dfa2026b40db` before any gate was run.

The ORCA2 card receives SI3 only through its existing
`unmeasured_features` field: `si3_jpl5_layered_prather_state`.  The merged L3
implementation does not silently select itself on ORCA2, and no existing
card default is allowed to move.  In particular, the historical Round-20
exact-input gate is expected to import `SI3ThermoConfig` successfully and then
retain its recorded fail-closed `STOP_SELECTOR_GAP` verdict for ORCA2's
unsupported five-category configuration.

## Predicted and resolved textual conflicts

`git merge-tree` predicted one conflict for the thermodynamics merge and
three more for dynamics after thermodynamics.  The actual merges produced
exactly those four files and no others.

| File | Lane intent | SI3 intent | Registered resolution | Executed NEMO anchor |
|---|---|---|---|---|
| `packages/core/legoesm/timestepping/tridiagonal.py` | Preserve the lane's undecorated primal inside the custom-VJP backward solve, required for forward-over-reverse AD. | Add the source-order selector used by SI3 BL99. | Union: the custom VJP calls the undecorated implementation and forwards `operation_order`; neither side's semantic fix is dropped. | C1D `icethd_zdf_bl99.f90:536-560` materializes the unnormalised forward elimination and reverse substitution. |
| `packages/core/legoesm/core/transcendentals.py` | Provide policy-controlled source-ordered `log` and `log10` used by ORCA2 RGB. | Provide policy-controlled `exp`, `tanh`, `sin`, and `cos` used by SI3. | Union all six through the single existing precision-policy selector; unsupported names raise loudly. | ORCA2 `icealb.f90:184-189,218-234` executes `LOG` and `EXP`; `icedyn_rdgrft.f90:1077-1084` executes H79 `EXP`; `icedyn_rhg_evp.f90:350-374` executes landfast `EXP`. |
| `tests/unit/test_transcendentals.py` | Pin `log`/`log10` policy and gradients. | Pin the four dynamics functions, their domains, JIT identity and custom JVPs. | Union the independent test families; log tests retain a positive domain while odd/even SI3 functions retain the dynamics domain. | Same source anchors as the implementation conflict. |
| `packages/ice/legoesm/ice/__init__.py` | Export thermodynamic config/state/C1D surfaces. | Export dynamics/ridging/transport surfaces. | Explicit export union with duplicate auto-union lines removed. | ORCA2 `icestp.f90:186-217,250-262` executes dynamics and thermodynamics in distinct ordered calls, so both public surfaces are required. |

Focused post-resolution controls are already green: the Thomas custom-VJP and
SI3 thermodynamics Phase-2 tests are 22 passed; the combined transcendental,
Thomas, thermodynamics, ridging and landfast set is 56 passed.  These are
conflict controls, not substitutes for the preregistered reproduction gates.

## Measurement preregistration

All production measurements below use the committed merged tree, CPU-only
JAX, fp64, the repository-wide package `PYTHONPATH`, and
`/home/dbalwada/legoESM/.venv/bin/python`.

### GYRE invariance

The fixed reference is
`phase3/orca2_merge/before/`, produced at GYRE lane tip `4cac617cd928`.
The candidate will be produced under `phase3/si3_merge/gyre/`.

| Measurement | CONFIRM | REFUTE |
|---|---|---|
| Phase-3 ladder, `--trajectory-only --max-step 10` | Offline comparison `PASS` with zero worsening; all 10 step blocks / 50 trajectory rows equal. | Any failed comparison, moved row, changed verdict, or changed first-over-bar location. |
| `ladder.residuals.npz` | Same keys/order and `np.array_equal` for every array. | Any missing, extra, reordered, dtype/shape-changed, or unequal array. |
| Thirty-day member, `--member 0 --days 30 --snap-steps 6 --tag daily` | `day001.npz` through `day030.npz` byte-identical to the fixed reference. | Any missing file or byte difference. |
| Day-gap score for days 1–30 | Complete report exactly equal to the fixed reference report. | Any row/value/status difference. |

Any moved GYRE row is a wrong merge hunk and will be repaired, never accepted
or registered as a new baseline.

### SI3 L3 gates

Every thermodynamics and dynamics gate named by the two branches' receipts
will be replayed against its named immutable L3 record.  A reproduction counts
only when the merged-tree verdict equals the recorded verdict, including
recorded fail-closed stops.  Every planted violation exercised by the branch
tests must still turn red.  Missing records, import-only success, a different
verdict, or a plant that stays green refutes reproduction.

### ORCA2 seven-gate set

The first six expected results are frozen by the preceding ORCA2 merge
receipt.  The seventh expected result is the archived Round-20 verdict, not a
new claim.

| Gate | Registered result |
|---|---|
| Phase-1 full schema | `PASS`, four record groups, 10/10 planted controls. |
| Phase-2v ordered TKE walk | Admission `PASS`; first non-bit statement `zpelc` at kt=2 with the archived row values. |
| Phase-2y ORCA1-ice admission | `PASS`, 116/116 records and 26/26 plants. |
| Phase-2y resolved ice namelist | 185 fields, only `namini.nn_iceini_file` differs between variants, self-test passes. |
| Round-20 Phase-1 schema | Expected exit 1 / `FAIL` for five extra records. |
| Round-20 SI3 stream headers | `VALID`, stream counts 5/140/10/25/15. |
| Round-20 exact-input admission | Gate imports; verdict remains `STOP_SELECTOR_GAP`. |

### Configuration and model-hunk audit

No existing selection/default may move on GYRE, ORCA2, LOCK_EXCHANGE,
OVERFLOW, DINO, or any OMIP deck.  The source audit will classify each merged
surface per card as `yes`, `inert`, or `no`.  `yes` and `inert` require an
execution or configuration proof; newly selected SI3 behavior on an existing
card other than the already recorded ORCA2 unresolved-feature field refutes
the merge.

### Citation, push, and broad test gates

After results are written, citations may move only by rigid re-anchor: old and
new cited blocks must have equal extent and identical text.  The citation
gate and its planted controls must pass.  Then the four-file push gate runs
`test_nemo_testcase_receipt_citation_gate.py`, `test_tke_nemo_terms.py`,
`test_nemo_recipe.py`, and `test_real_freshwater_closure.py` together.

Finally there will be exactly one combined invocation of
`tests/ocean/fidelity` and `tests/ocean/unit` with `-n 12`.  Compiler-symbol
materialization failures or `MemoryError` are infrastructure failures; only
their failing IDs are rerun in isolation.  The reported regression count is
the set difference against the frozen base-tip failing-ID set.  Only new
failing IDs count.

## Results

### Merge and configuration audit

Both `--no-ff` merges completed in the registered order.  The four predicted
conflicts were the four actual conflicts.  No unpredicted file conflicted, and
the focused conflict controls passed first as 22/22 and then as 56/56.

The merged dynamics gates initially exposed one repository-wide provenance
ratchet failure: eight new report dictionaries did not stamp their producing
worktree.  Commit `ed601563ef37` adds `worktree_stamp()` to those eight
emitters.  The ratchet is now 10/10, all eight scripts compile, and a clean-tree
rerun of the rung-33 A-grid guard is still byte-identical while stamping branch
`fidelity/si3-on-lane` and commit `ed601563ef37`.

No existing card recipe or selector changed.  `SeaIceConfig` retains
`thermo_scheme="zero_layer"`, `bulk_scheme="constant"`,
`dynamics="none"`, `transport="none"`, and one category.  The two textual
default refactors are value-preserving: emissivity remains `0.97`, and Cd/Ch
remain `1.5e-3`; the appended Ce default is also `1.5e-3`.  ORCA2's sole new
relationship to this merge is still the field already present at the base
tip, `unmeasured_features=("si3_jpl5_layered_prather_state",)`.

| Card/deck | Merged model hunk executes? | Selection/default proof | Verdict |
|---|---|---|---|
| GYRE | Existing shared Thomas calls can execute; the new SI3 source-order arm is inert under its unchanged default. | No GYRE recipe changed; the ten-step ladder and thirty-day member below are bit-identical. | `inert` |
| ORCA2 | No SI3 thermodynamics/dynamics step is selected.  The historical admission imports `SI3ThermoConfig` only to test the boundary. | No ORCA2 recipe changed; the pre-existing unresolved-feature tuple is byte-identical at base and merge tips; seven gates reproduce. | `no` |
| LOCK_EXCHANGE | No SI3 selector is selected. | No card/config file changed; zero-layer/default dispatch remains selected. | `no` |
| OVERFLOW | No SI3 selector is selected. | No card/config file changed; zero-layer/default dispatch remains selected. | `no` |
| DINO | No SI3 selector is selected. | No card/config file changed; zero-layer/default dispatch remains selected. | `no` |
| Existing OMIP decks | No existing deck was edited or newly selects SI3. | Only new retained L3 testcase namelists were added. | `no` |
| New C1D/ADV/RHG L3 evidence decks | Yes, explicitly and only inside their new testcase recipes. | Their resolved namelists and manifests are the inputs to the 20-program table below. | `yes` |

The executable-source conflict resolutions remain anchored in compiled NEMO,
not in either Git side: C1D
`BLD/ppsrc/nemo/icethd_zdf_bl99.f90:536-560` is the Thomas elimination and
back-substitution; ORCA2 `BLD/ppsrc/nemo/icestp.f90:186-217,250-262` calls
dynamics before thermodynamics; ORCA2 `BLD/ppsrc/nemo/icealb.f90:184-189,218-234`
uses `LOG`/`EXP`; ORCA2
`BLD/ppsrc/nemo/icedyn_rdgrft.f90:1077-1084` and
`BLD/ppsrc/nemo/icedyn_rhg_evp.f90:350-374` execute the H79 and landfast
exponentials.  These are compiled `ppsrc` arms for the named configurations,
not dead source arms.

### GYRE invariance — confirmed

The candidate is `si3_merge/gyre/`; the fixed reference is
`orca2_merge/before/` at `4cac617cd928`.

| Measurement | Result |
|---|---|
| Ten-step trajectory-only ladder | The scientific ladder remains the recorded `DEBT`, as expected.  Offline comparison is `PASS`: 70 certified rows, zero moved/worsened cells, zero status change, and zero largest residual worsening ULP.  All ten trajectory blocks (50 rows) and ten barotropic-state blocks are equal. |
| Residual sidecar | Same 210 keys in the same order; every array passes `np.array_equal`.  Both files have SHA-256 `de4eea43cab0236e98b85021a13a4c34b6b121cf178ffa59a3303864504b0e23`. |
| Thirty-day member | All `day001.npz` through `day030.npz` are byte-identical. |
| Day-gap score | All 30 rows and `max_day_to_day_ratio=10.268914926740052` are equal.  The only JSON difference is the required producer-worktree stamp. |

Therefore **GYRE is bit-identical**.  No moved row was registered.

### SI3 L3 reproduction — 19/20 programs

The denominator is the six thermodynamics and fourteen dynamics gate programs
shipped by the two branches.  A recorded scientific `DEBT`, `UNMEASURED`, or
first-divergence stop counts as reproduced only when the same fail-closed
verdict returns; it is not relabelled as a pass.

| Thermodynamics program | Recorded verdict | Merged-tree reproduction |
|---|---|---|
| `nemo_si3thd_oracle_gate.py` | `VERIFIED` | `VERIFIED`; exact JSON SHA-256 `7e8fcf463fecd83adacbbc94fc70ebe36719ada39edd7b5b42ee46823115b8b0`. |
| `nemo_si3thd_phase2_gate.py` | `DEBT`, first owner at the pre-BL99 `qns_ice` entry time level | Same `DEBT` and owner. |
| `nemo_si3thd_phase2b_year_gate.py` | full-year `DEBT` / MIXED DEBT | Same full-year scientific payload and `DEBT`; only the precision-policy representation now explicitly includes `transcendentals='native'`. |
| `nemo_si3_bulk_flux_gate.py` | `AT-BAR` | `AT-BAR`, 227,760 comparisons and zero over-bar rows; exact SHA-256 `12784cbf13f77a993aa7f11c5dfab08cfbfe97d7db3236c7a544512a6e1178ed`. |
| `nemo_si3_exchange_drift_gate.py` | two 8,760-step streams byte-identical | Same, zero differing bytes; exact gate SHA-256 `2bde178decf1c6d49bc29d83b207102f56e76a15ff933137773785ca9761b0e0`. |
| `nemo_si3_scalarmath_v2_gate.py` | `REPRODUCIBLE` | **Not reproduced:** fail-closed `A MY_SRC is not verbatim` before any scientific comparison. |

The scalar-math failure is external evidence drift, not a candidate numeric
change.  In the retained A source tree, `icethd.F90`, `icethd_dh.F90`, and
`icethd_zdf_bl99.F90` now hash differently from both V1 and V2-B.  Their A-side
mtimes are 2026-09-04 17:35, after the A executable's 08:24 build time.  The
gate correctly refuses to attribute that executable to the later source.
Neither the external record nor the bar was edited to make it pass.

The whole-file hashes in the dynamics table are the exact clean-tree artifacts
from preregistration commit `dfa2026b40db`.  The later integration repair
`ed601563ef37` adds the required producer-worktree object to those JSON reports,
so a post-repair whole-file hash necessarily changes even when every scientific
row is identical.  That metadata-only change is not presented as a new exact
hash claim; the post-repair stamped A-grid rerun is the execution control.

| Dynamics program | Recorded verdict | Merged-tree reproduction |
|---|---|---|
| `nemo_si3_oracle_gate.py` | rungs 3.1/3.2 `DEBT`, rung 3.3 `VERIFIED` | Same three verdicts. |
| `nemo_si3_phase2_gate.py` | ADV1D `DEBT` | Same `DEBT`. |
| `nemo_si3_phase2_adv2d_gate.py` | `UNMEASURED`, numeric `DEBT` | Same. |
| `nemo_si3_phase2_adv2d_replay.py` | `IMPLEMENTATION_OR_UNMEASURED_MOMENT_INPUT_DEBT` | Same classification. |
| `nemo_si3_phase2_round10_h79_probe.py` | `BIT-EXACT` | `BIT-EXACT`; exact SHA-256 `24b8279e7c4603689095e2577a576f12b475eb6af2343cca27b17e32414fc2b0`. |
| `nemo_si3_phase2_round13_active_aevp_probe.py` | `BIT-EXACT` | `BIT-EXACT`; exact SHA-256 `912087782bfa304aba21945938af90bb1192516cbcc073a5fd92d2c93e2e10a5`. |
| `nemo_si3_phase2_round14_seed_probe.py` | `PRODUCER-NAMED` | `PRODUCER-NAMED`; exact SHA-256 `897af8e527dcbf33714a93736fb2473799d6b9b5b8aad1f44388bb86b70f345b`. |
| `nemo_si3_phase2_rung33_agrid_guard.py` | A-grid EVP/mEVP byte-identical | Same; repeated after the stamp repair from clean commit `ed601563ef37`. |
| `nemo_si3_phase2_rung33_gate.py` | `UNMEASURED`, numeric `AT-BAR` | Same. |
| `nemo_si3_phase2_rung33_replay.py` | `BIT-EXACT` | Same. |
| `nemo_si3_phase2_rung33_trajectory_gate.py` | `DEBT` | Same. |
| `nemo_si3_phase2_rung34_gate.py` | `KT1-VERIFIED` | Same. |
| `nemo_si3_phase2_rung34_moment_replay.py` | step-8 eager/JIT `AT-BAR` | Same, exit 0. |
| `nemo_si3_phase2_rung34_trajectory_gate.py` | `DEBT-FIRST-DIVERGENCE` at completed step 2 | Same stop and first-divergence boundary. |

The optional rung-34 full-walk mode is not a twenty-first shipped program and
was not substituted for the default first-divergence gate.  It was stopped
after completing only steps 9 and 10 when its observed rate projected to
hours; it makes no claim.  The merged SI3-focused test battery collected 182
tests: 181 passed and the sole failure is the same external scalar-math
provenance refusal above.

### ORCA2 reproduction — 7/7

| Gate | Recorded verdict | Reproduced verdict |
|---|---|---|
| Phase-1 full schema | `PASS`, four record groups, 10/10 plants | Same; exact SHA-256 `cd34c630d67aa6d19bc9fdb163d581be0d347f674b474ddde8fb2414f7f5d93f`. |
| Ordered TKE walk | Admission `PASS`; `DEBT` first at kt=2 `zpelc`, 39,290/242,135 unequal, max absolute `1.7763568394002505e-15`, max row ULP 4 | Same; exact SHA-256 `c7ef6ed73efe06c1f8a894c9e76f9e6b8f4051a579df420ed54a8cd62ef1f947`. |
| ORCA1-ice admission | `PASS`, 116/116 records, 26/26 plants | Same; exact SHA-256 `d3c60bf16a84f8739f8106541bea2bfcf4941e75b17e7a4f1bd0d5f7cd371297`. |
| Resolved ice namelist | 185 fields; only `namini.nn_iceini_file` differs | Same; exact SHA-256 `278a891ad15cf7aecc6f8a60533634af6a79076b9650e573a840261236554013`. |
| Round-20 Phase-1 schema | expected exit 1 / `FAIL`, five extra records | Same; exact SHA-256 `6574215cedafa818a3ef9900ced494d643fac4c95ea2aec8391d91df4b1f403f`. |
| Round-20 SI3 stream headers | `VALID`, counts 5/140/10/25/15 | Same; exact SHA-256 `741bb4b0f7bf41ead4875726c1c01ae35689bb4c95897b2960213b8576b5494d`. |
| Round-20 exact-input admission | `STOP_SELECTOR_GAP` | The historical gate imports the merged `SI3ThermoConfig`, scores zero physics rows, and returns the same six unsupported selectors: categories, ice layers, snow layers, salinity scheme, ponds, lateral melt.  Exact SHA-256 `c4bdf3141b35b565fb1b6ea2d9031158a5caa50674b8651445a23c9f906e3833`. |

### Citation, push, and broad tests

The rigid citation gate passed 16/16 with no re-anchor required.  The four-file
push gate passed 124/124 in 372.29 seconds.  The required single invocation of
`tests/ocean/fidelity tests/ocean/unit -n 12` collected 8,304 tests and reached
96% before eight JAX compiler workers aborted (`Fatal Python error: Aborted` /
`Not properly terminated`); the coordinator then stopped making progress.
No second broad battery was run.

All 455 IDs left in flight or flagged by the aborted run, spanning 56 files,
were rerun in fresh isolation.  The two extraction-only groups passed 29/29
and 67/67.  The remaining candidate failures classify as follows:

- the six IDs in the preceding ORCA2 receipt's frozen base set reproduce
  unchanged: four f32 advection-gradient tests and two carried-TKE tests;
- `test_f90_recurrence_oracle_nonuniform_rho` has the same
  `2.34866651e-15` maximum absolute mismatch when run alone at both candidate
  and exact base tip `598ef772d00f`, so it is base debt, not a merge failure;
- the worktree-stamp ratchet was a merge integration defect and was fixed;
  its final rerun is 10/10; and
- the only new failing ID is
  `test_nemo_si3_scalarmath_v2_gate.py::test_full_v2_gate_and_plants`, caused
  by the external overwritten A-side source ledger described above.

The base-tip set difference is therefore **one new test failure**.  Evidence
for every result above is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_merge/`.

An independent in-sandbox Codex review was unavailable.  The operator's
Claude review remains required before integration.  This sandbox also had no
GitHub connector, so the operator must post the receipt/verdict to issue
`#1455`.

## OPEN

**ORCA2 ice Round 1 is selector closure at the exact-input boundary, before
any integration score.**  It must add the ORCA2 identity actually recorded by
NEMO: five categories, 10 ice layers, 5 snow layers, salinity scheme 4, ponds,
and lateral melt.  Only after that gate scores physics may the first ORCA2
integration with ice begin.  The present merge deliberately leaves historical
Round 20 at `STOP_SELECTOR_GAP`; the L3 one-category implementation does not
close `si3_jpl5_layered_prather_state` by assertion.

The remaining evidence decision is external: restore or supply an immutable
copy of the V2-A `MY_SRC` that was actually built into its retained executable,
then rerun the scalar-math provenance/byte gate.  Rebuilding or editing the
oracle was outside this CPU-only merge task and would destroy the historical
provenance question rather than answer it.
