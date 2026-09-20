# GYRE NEMO-fidelity SI3 merge receipt — 2026-09-20

Status: **MEASUREMENTS PREREGISTERED; RESULTS PENDING.**

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

Pending.

## OPEN

Pending measurements.  If all gates confirm, the first ORCA2 round with ice
is the existing Round-20 exact-input admission replay followed by a new ORCA2
integration round that explicitly closes `si3_jpl5_layered_prather_state`;
this merge does not pretend the L3 one-category implementation closes that
five-category debt.
