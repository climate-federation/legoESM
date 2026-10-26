# NEMO testcase L2 GYRE phase 3 — round 124 day-240 process ranking receipt

Date: 2026-09-20

Incoming tip: `af3f7215060fc17c71adc6794817c710df8ee471`

Status: **HELD — the independently closed day-180-to-240 budget ranks the
stage-3 vertical-diffusion bucket first at `+2.4168271578053416e-2 K` signed
day-240 carry, but this diagnostic round lands no physics and does not yet
attribute that bucket to a coefficient, closure, or scalar statement.**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round124/`

## Outcome first

The Round-123 oracle record is admitted, and an independent CPU/fp64 legoESM
run from rest produced the corresponding 360-step process trace through the
production `self._step_jitted` path.  The diagnostic state was never carried:
all 360 full-state comparisons against a separate ordinary compiled step were
byte-identical.  Its generated day-180 and day-240 T, S, u, v and SSH
snapshots are also byte-identical to the immutable admitted year member.

The measured day-240 temperature RMS is
`1.6446741930292448e-2 K`.  The largest process projection is **vertical
diffusion**, `+2.4168271578053416e-2 K` (`146.9486886 %` of the signed endpoint
projection), born most strongly during days 190--200 in the upper 100 m,
western third and southern latitude band.  It is strongly cancelled by
lateral diffusion, `-1.6338204693572344e-2 K`, while advection reinforces it
by `+1.0718353088678008e-2 K`.  The absolute signed carries total
`5.511204485073970e-2 K`, a `3.3509399664` cancellation ratio.

This result names a **process bucket**, not a sub-owner.  The vertical row is
the difference between each independently evolving model's solved `Taa` and
its own post-lateral-diffusion normalized content.  It can contain differences
in the live entry, diffusivity profiles, implicit matrix, free-surface
weights, and solve association.  It is not evidence that TKE, one matrix
coefficient, or one last-bit expression owns the month-scale error.

No shared physics, configuration, restart schema, carried state, card default
or stabilizer changes in this round.  Decision-43/45 landing gates therefore
do not run, and the immutable before arms do not move.

## Frozen preregistration ledger

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round124.md`, committed
before the first process measurement as full commit
`6d4b0008a267bd1e32129d32077c7e2a751fcf79`.

- **P0 CONFIRMED.** The oracle has exactly 360 frames, each 1,415,300 bytes,
  for steps 1081--1440 at `rDt=14400`.  Its chain and decoded step closure are
  exact.  Both restart files are byte-identical to the admitted year run.
  All five oracle plants printed `STATUS PLANT-FIRED` and exited 1.
- **P1 CONFIRMED after one rejected attribution and one provenance-superseded
  run.** The final v3 trace was produced at clean commit
  `98327e7a2d5841da0b90bd63f19c8a6ebf07a2f8`, contains 360 frames, has zero
  unequal carried-state bytes, and reproduces every immutable endpoint field
  bit-for-bit.  The v1 process labels were rejected as described below and
  are not used in the result.
- **P2 CONFIRMED.** Wrong-stamp, one-ULP and full-production effect plants
  printed `STATUS PLANT-FIRED` and exited 1.  The effect plant moved the
  post-SBC and all downstream boundaries while moving no upstream boundary
  and zero carried-state bytes.
- **P3 CONFIRMED.** Components reconstruct the full day-240 error array to
  `3.552713678800501e-15 K`; signed carries sum to the endpoint RMS within
  `-3.469446951953614e-18 K`.  The mechanically largest absolute signed carry
  is vertical diffusion.  Day-30 process carry remains
  **UNAVAILABLE-BY-RECORD** rather than being inferred.
- **P4 CONFIRMED.** Only a private diagnostic hook, its existing owner
  instrument, controls, tests, citation mappings, preregistration and this
  receipt land.  The hook is absent from public recipe/config fields and is
  statically off for every ordinary model.

## Compiled record and process order

The cited program is the exact preprocessed branch that produced the record,
not the uninstrumented source card.  It enables the writer only for stage 3
and steps 1081--1440, then writes the header, timestep, `Tbb` and the three
free-surface ratios at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:818-830`.

The compiled stage program then calls tracer advection, writes its accumulator,
calls the RK3 surface boundary condition, and writes that accumulator at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:861-869`.
At stage 3 it calls penetrative shortwave and lateral diffusion and records
both post-call accumulators at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:930-952`.
Finally it calls `tra_zdf`, writes the solved `Taa`, and closes the record at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:964-970`.

Inside the compiled implicit routine, the before tracer content and the
middle-level accumulated RHS form the content RHS at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/trazdf.f90:545-560`; the forward
and backward vertical recurrences produce the after tracer at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/trazdf.f90:563-578`.  These
statements define the measured vertical-diffusion bucket.  They do not by
themselves identify which operand owns its independent-trajectory difference.

## Oracle-record admission

Normal admission is
`round124/oracle_process_record_validation.json`:

| control | measured result |
|---|---:|
| records / steps | 360 / 1081--1440 |
| bytes per frame / total | 1,415,300 / 509,508,000 |
| `Taa(step) -> Tbb(step+1)` unequal cells | 0 |
| max decoded per-step closure | `0.0 K` |
| geometry / advection visits | 6,480,000 / 6,480,000 |
| surface / shortwave visits | 216,000 / 3,672,000 |
| lateral / vertical visits | 6,479,999 / 6,480,000 |

Independent `sha256sum` and `cmp` checks reproduced:

| restart | SHA-256 | admitted-year comparison |
|---|---|---|
| step 1080 | `6c0c7a950b30b9d59dbf2673833ddf462a5f8ea5650f496f2f772e1e17092976` | byte-identical |
| step 1440 | `96529a98da0e0d89b328632a826a9d41593f81f0d1a917350f28f184d49b163a` | byte-identical |

The five persisted plant reports are
`plant_process_{stamp,truncation,sbc_ulp,sbc_effect,trajectory_ulp}.json`.
Each exited 1.  The raw RHS ULP moved its recorded operand but, as already
retracted in Round 123, was swallowed by the normalized temperature content;
the separate calibrated effect plant moved the decoded surface row.  No plant
prints PASS while firing.

## legoESM production trace, rejection and correction

The committed instrument extends only
`nemo_testcase_l2_gyre_year_owners.py`.  It runs seed zero from rest, uses the
same forcing helpers and resolved GYRE-zco recipe, calls `model.step` on CPU in
fp64, and records steps 1081--1440.  On every recorded step it evaluates a
trace model and a separate ordinary model from the same entry, compares the
entire returned state pytree byte-for-byte, and carries only the ordinary
state.  Its manifest binds every file to the clean producer commit.

The first completed trace under `round124/lego_process_trace/` is preserved
but **rejected for process attribution**.  Its provisional surface bucket had
3,348,195 nonzero visits although NEMO's compiled SBC row and the physical
source are top-only.  The new fail-closed validator applied to that immutable
record exits 1 with:

```text
GATE FAILED: legoESM surface-boundary bucket moved 3132195 subsurface wet cells; a non-surface process was misclassified
```

Cause: v1 subtracted the NEMO-live stage-3 shortwave reconstruction from the
combined production tendency.  The production physics pipeline's actually
executed Kbb shortwave component is not bit-identical to that replacement, so
their subsurface residual was incorrectly labelled surface forcing.  The
correction re-evaluates the same pure physics callable with production's same
cell-centred velocity proxy, subtracts that executed component to obtain the
top-only surface source, and leaves every remaining stage-3 heat contribution
in the shortwave bucket.  It changes diagnostic attribution only.

V2 corrected the attribution and passed scientifically, but citation audit
found that its trace-only block sat inside an already pinned production-source
range.  The block was moved immediately after that range so both pinned
endpoints shift rigidly and the original extent stays unchanged.  Fail-closed
provenance then required a new run.  All 11 v3 trace arrays and both endpoint
NPZ files are byte-identical to v2; v2 is preserved but superseded.

The authoritative trace is `round124/lego_process_trace_v3/`; its report and
final validation are `lego_process_trace_v3_report.json` and
`lego_process_trace_v3_validation_stamped.json`:

| control | v3 result |
|---|---:|
| traced frames / wall time | 360 / `1926.0 s` |
| carried-state unequal bytes | 0 |
| chained temperature unequal cells | 0 |
| generated-vs-immutable cells unequal, day 180 (T/S/u/v/SSH) | 0 / 0 / 0 / 0 / 0 |
| generated-vs-immutable cells unequal, day 240 (T/S/u/v/SSH) | 0 / 0 / 0 / 0 / 0 |
| surface top-cell visits / expected | 216,000 / 216,000 |
| surface subsurface visits | 0 |
| shortwave visits | 3,672,000 |
| max decoded per-step closure | `7.105427357601002e-15 K` |

The v3 wrong-stamp, trace-ULP and production-effect reports are
`plant_lego_process_{stamp,ulp,effect}_v3.json`; all exited 1.  The effect
plant moved `Bsbc/Bqsr/Bldf/Bpre/Taa` by 1/1/1/1/3 cells, moved
`Tbb/q_Kbb/q_Kmm/q_Kaa/B0/Badv` by zero cells, and left the separately
compiled carried state byte-identical.

The whole-tree worktree-stamp ratchet subsequently caught one Round-124
provenance defect: the trace **validation** report retained the immutable
producer stamp but did not stamp the tree that performed validation.  Commit
`bc4607ca01a4b793c0c2a5878214df8a1c48f14c` adds that independent stamp.  The
ratchet failed before the fix and passes after it; its synthetic missing-key
arm remains the non-vacuity control.  Both the standalone validation and the
final budget now name that clean validating commit.  Removing every
`worktree` object leaves the pre-fix and post-fix scientific payloads exactly
equal.

For audit only, the v1 surface and shortwave signed carries were
`-2.9933008766379514e-3 K` and `-8.393958572063607e-9 K`.  V2 assigns
`-2.9943959019919585e-3 K` and `+1.0866313955171657e-6 K`; their combined
carry is unchanged within `8.2e-17 K`.  Only v3 appears in the verdict table.

## One ranked day-240 table

The final machine-readable result is
`round124/day240_process_budget_v4_stamped.json`.  Its scientific payload is
identical to `day240_process_budget_v3.json`; v4 adds the validation-worktree
stamp described above.  A carry is the signed projection of
that process-difference array onto the independently measured day-240 error;
ranking is by its absolute value.  Local rows use their own local projection,
so they locate the owner but are not additive whole-domain carries.

| rank | owner / compiled stage | day-240 signed carry (K) | projection | component RMS (K) | day-30 carry | strongest birth | largest depth / longitude / latitude partition |
|---:|---|---:|---:|---:|---|---|---|
| 1 | vertical diffusion / stage-3 `tra_zdf` solve | `+2.4168271578053416e-2` | `+1.469486886` | `5.571209911849351e-2` | UNAVAILABLE-BY-RECORD | days 190--200, `+5.6288525578725495e-3 K` | 0--100 m / west third / south <=37.2 N |
| 2 | lateral diffusion / stage-3 `tra_ldf` | `-1.6338204693572344e-2` | `-0.993400685` | `3.177093926579452e-2` | UNAVAILABLE-BY-RECORD | days 230--240, `-3.8933619888259063e-3 K` | 0--100 m / west third / south <=37.2 N |
| 3 | advection / stage-3 `tra_adv` | `+1.0718353088678008e-2` | `+0.651700691` | `3.3465631122984923e-2` | UNAVAILABLE-BY-RECORD | days 190--200, `+2.362568743894072e-3 K` | 0--100 m / west third / south <=37.2 N |
| 4 | surface boundary / stage-3 `tra_sbc_RK3` | `-2.9943959019919585e-3` | `-0.182066206` | `1.4870528651722227e-2` | UNAVAILABLE-BY-RECORD | days 230--240, `-7.084020913155213e-4 K` | 0--100 m / west third / south <=37.2 N |
| 5 | incoming independent-state gap | `+8.916820923885074e-4` | `+0.054216336` | `3.580551011866709e-3` | UNAVAILABLE-BY-RECORD | before day 180 | 100--1000 m / east third / north >37.2 N |
| 6 | shortwave / stage-3 `tra_qsr` | `+1.0866313955171657e-6` | `+6.606970548e-5` | `9.056728608436352e-5` | UNAVAILABLE-BY-RECORD | days 210--220, `+1.8379413209574634e-7 K` | 0--100 m / east third / north >37.2 N |
| 7 | free-surface content geometry | `-5.086465932617549e-8` | `-3.092689090e-6` | `5.150099776399684e-7` | UNAVAILABLE-BY-RECORD | days 190--200, `-2.7752310234786937e-8 K` | 0--100 m / west third / south <=37.2 N |
| 8 | explicit floating-point closure | `+6.264033818400789e-16` | `+3.808677637e-14` | `2.9650269776502056e-14` | UNAVAILABLE-BY-RECORD | days 210--220, `+2.905951397344723e-16 K` | 0--100 m / west third / south <=37.2 N |

Budget controls:

| endpoint/control | value |
|---|---:|
| day-30 T3D RMS, context only | `6.890484901489568e-5 K` |
| day-240 T3D RMS | `1.6446741930292448e-2 K` |
| signed carry sum | `1.6446741930292445e-2 K` |
| signed sum minus endpoint | `-3.469446951953614e-18 K` |
| max array reconstruction residual | `3.552713678800501e-15 K` |
| lego interval closure RMS | `2.9638147004890224e-14 K` |
| NEMO interval closure RMS | `5.478300364444701e-16 K` |
| cancellation ratio | `3.350939966366927` |

The day-30 endpoint is reproduced for context.  A day-30 **process carry** is
not available because the admitted process record begins at day 180; no other
probe is silently substituted.

## Landing and cross-card scope

Nothing physical lands.  The current admitted headlines therefore remain:

- kt2 T/S: `1.4210854715202004e-14 K` /
  `2.1316282072803006e-14 psu`;
- kt2 U/V: `2.7377110452773967e-12` /
  `3.2849219221489645e-12 m s-1`;
- kt3 T/S: `8.600419718618468e-7 K` /
  `6.979441735666114e-8 psu`;
- day-30 T3D RMS: `6.890484901489568e-5 K`;
- day-240 T3D RMS: `1.6446741930292448e-2 K`; and
- day-360 T3D RMS: `1.1223573910167267e-2 K`.

The private trace hook is statically off on ordinary construction and has no
public card field.  GYRE production, generic NEMO-GYRE, DINO, LOCK_EXCHANGE
and OVERFLOW therefore execute no changed production statement.  Their
certified values do not move and are not re-baselined.  ORCA2 remains
**UNMEASURED-WITH-SPEC**: repeat this independent-trajectory process budget
with its native compiled order, grid/masks, endpoint snapshots, full-state
passivity comparison and the same projection/closure controls before making
an ORCA2 magnitude claim.

No configuration or carried-state choice is exposed by this process ranking,
so `DECISION_NEEDED` is `NONE`.

## Independent adversarial review

The required command was run with `codex exec --sandbox read-only` against the
complete incoming-tip-to-receipt diff.  Its verbatim terminal result was:

```text
Error while loading conda entry point: conda-anaconda-tos (cannot import name 'validate_prefix_exists' from 'conda.cli.install' (/home/dbalwada/miniconda3/lib/python3.13/site-packages/conda/cli/install.py))
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

The command exited 1.  Independent review unavailable in-sandbox.  It emitted neither
`SHIP` nor `DO NOT SHIP`; no verdict is fabricated.

## Verification

The machine-readable inventory is
`round124/whole_ocean_test_audit.json`.  Collection found 8,219 node IDs.  The
first combined `-n 12` run reached 98% before repeated JAX compiler worker
aborts stopped progress; it was interrupted with exit 130 and emitted neither
a terminal summary nor JUnit.  A lower-concurrency whole-fidelity retry had
the same long-tail problem and was also interrupted with exit 130 and no
JUnit.  The required trees were therefore split by file, as the repository
instructions require when a process approaches the compiler limit.  Those 25
non-overlapping partitions recorded 8,199 unique node IDs; the 20 interrupted
cases were then run from fresh processes.

Every partition's exact terminal summary follows.  `INTERRUPTED` means the
outer 15-minute bound wrote a partial JUnit file; those omitted cases are all
dispositioned below.

| suite | exact pytest summary |
|---|---|
| fidelity 01 | `368 passed, 2 skipped in 13.74s` |
| fidelity 02 | `236 passed, 3 skipped in 176.13s (0:02:56)` |
| fidelity 03 | `234 passed in 58.16s` |
| fidelity 04 | `197 passed in 912.02s (0:15:12)` — INTERRUPTED, one omitted |
| fidelity 05 | `1 failed, 215 passed, 2 skipped in 95.64s (0:01:35)` |
| fidelity 06 | `137 passed in 5.18s` |
| unit 01 | `6 failed, 438 passed, 1 xfailed, 1 warning in 284.64s (0:04:44)` |
| unit 02 | `334 passed, 1 skipped, 1 xfailed, 4 warnings in 359.26s (0:05:59)` |
| unit 03 | `262 passed, 2 warnings in 904.13s (0:15:04)` — INTERRUPTED, fifteen omitted |
| unit 04 | `4 failed, 544 passed, 9 skipped, 10 warnings in 86.48s (0:01:26)` |
| unit 05 | `3 failed, 401 passed in 556.91s (0:09:16)` |
| unit 06 | `1 failed, 260 passed, 14 skipped, 2 warnings in 74.61s (0:01:14)` |
| unit 07 | `398 passed, 4 warnings in 170.62s (0:02:50)` |
| unit 08 | `265 passed in 120.34s (0:02:00)` |
| unit 09 | `2 failed, 441 passed, 5 skipped, 6 warnings in 215.16s (0:03:35)` |
| unit 10 | `1 failed, 408 passed, 42 warnings in 196.54s (0:03:16)` |
| unit 11 | `1 failed, 250 passed, 3 skipped, 2 warnings in 915.30s (0:15:15)` — INTERRUPTED, four omitted |
| unit 12 | `584 passed, 80 skipped, 7 warnings in 455.45s (0:07:35)` |
| unit 13 | `2 failed, 253 passed, 2 skipped, 10 warnings in 242.45s (0:04:02)` |
| unit 14 | `299 passed in 61.43s (0:01:01)` |
| unit 15 | `292 passed, 3 skipped, 2 warnings in 151.02s (0:02:31)` |
| unit 16 | `302 passed, 31 skipped, 1 warning in 167.58s (0:02:47)` |
| unit 17 | `2 failed, 472 passed in 186.75s (0:03:06)` |
| unit 18 | `340 passed in 128.93s (0:02:08)` |
| unit 19 | `90 passed in 42.29s` |

The fresh-process disposition is mechanical:

- the omitted fail-closed LOCK plant passed: `1 passed in 1094.58s
  (0:18:14)`;
- the four omitted production-JIT NEMO-WS cases passed: `4 passed in 598.85s
  (0:09:58)`;
- the omitted Coriolis groups passed: `28 passed in 11.30s`;
- the Round-124 worktree-stamp regression failed in fidelity partition 05,
  was fixed, and its exact rerun is `1 passed in 1.76s`;
- the 16 raw failures absent from the old 87-ID reference were rerun together:
  `6 failed, 10 passed, 1 warning in 101.24s (0:01:41)`.  The ten passes were
  compiler-pressure transients.  The same six failures then reproduced on the
  untouched incoming commit `af3f7215060fc17c71adc6794817c710df8ee471`:
  `6 failed, 1 warning in 11.41s`; and
- the sole remaining omitted case, unrelated CORE-II cache creation, blocks
  inside `xarray.Dataset.to_zarr -> zarr.sync.wait`.  With an explicit
  60-second diagnostic timeout its current-tree summary is `1 failed in
  60.88s (0:01:00)` and its incoming-tip summary is `1 failed in 60.96s
  (0:01:00)`, with the same stack.  This is an environment/event-loop timeout,
  not a Round-124 source difference.

The six failures present in both the incoming and Round-124 trees, but absent
from the older 87-ID reference file, are:

1. `test_freesurface_helmholtz_adjoint_mpas.py::test_production_step_grad_finite_f32_and_scan`;
2. `test_nemo_match_recipe.py::test_mpas_factory_builds_valid_model_and_one_step_is_finite`;
3. `test_partial_cells_phase1.py::TestFlatBottomBitExact::test_flat_bottom_nonzero_eta`;
4. `test_partial_cells_phase2.py::TestFlatBottomBitExact::test_nonzero_eta_flat_bottom`;
5. `test_tke_carried_coefficients.py::test_step_entry_n2_bundle_fails_closed_without_raw_w_mesh`; and
6. `test_tke_carried_coefficients.py::test_step_entry_n2_bundle_matches_live_geometry_construction`.

Six failures from the supplied 87-ID reference were observed unchanged: the
four `test_advection_grad_underflow.py::test_model_rollout_grads_finite_f32`
parameters, `test_baroclinic_decomposition_bit_identical`, and
`test_nemo_sco_step_pgf.py::test_f90_recurrence_oracle_nonuniform_rho`.
After fresh-process adjudication the audit contains **zero Round-124-introduced
failures**.

The final owner self-check prints `self-check: all checks passed`, including
the process-layout, decoded-effect, trace-boundary and ULP controls.  The
earlier owner-only test run reported `15 passed, 1 skipped`; the focused owner
plus citation suite reported `32 passed in 31.51s`.  Against clean receipt
commit `078fc8a362fa0d9ecfe14351730913c711829c50`, the final focused owner,
citation and report-stamp suite has 33/33 passing tests (`33 passed in
33.51s`; `round124/focused_tests_post_audit.xml`).  The citation gate records
six citations, zero failures, zero unmapped citations and zero map-audit
failures in `round124/citation_gate_post_audit.json`.  Shifting the compiled
writer citation from line 818 to line 820 exits 1 with
`SYMBOL-NOT-AT-LINE`, recorded in
`round124/citation_gate_post_audit_shift_plant.json`.

## OPEN — round 125

Remain on the magnitude program.  The sole candidate is the largest admitted
day-240 owner: the stage-3 vertical-diffusion bucket defined by the compiled
`tra_zdf` content and solve at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/trazdf.f90:545-560` and
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/trazdf.f90:563-578`.

1. Reproduce the admitted vertical carry
   `+2.4168271578053416e-2 K`, its days-190--200 birth, and the endpoint
   closure before any new decomposition.
2. Decompose that bucket by **magnitude on the independent trajectories**:
   live entry/content, free-surface weighting, vertical diffusivity profile
   family, implicit matrix construction, and solve/association.  Rank the
   resulting contributions at day 240.  Do not resume a last-bit walk and do
   not assume the earlier TKE/bn2 rounding rows carry this magnitude.
3. The largest admitted vertical sub-owner becomes the sole implementation
   candidate.  If it requires a configuration or carried-state choice, stop
   with one explicit `DECISION_NEEDED`; otherwise preregister its exact
   compiled statement and run the Decision-43 month/ladder and Decision-45
   full-year gates before landing.
4. Any new NEMO acquisition must be requested only if the existing admitted
   records cannot distinguish those magnitude terms; no acquisition is
   requested by Round 124 itself.

The lateral and advection rows are cancellation context, not simultaneous
candidates.  No downstream or smaller process may land before the vertical
bucket is resolved or mechanically refuted as the largest actionable owner.
