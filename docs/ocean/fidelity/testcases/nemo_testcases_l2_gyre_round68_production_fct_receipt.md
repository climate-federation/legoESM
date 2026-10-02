# NEMO-testcases L2 GYRE round-68 production-FCT receipt

Date: 2026-09-12. Final disposition: **HELD / REFUTED; no production physics
change was attempted or remains**.

## Verdict and first non-bit statement

The preregistered production-FCT prediction is **REFUTED** by its exact
baseline-reconstruction control. The compiled production temperature content
and the host reconstruction of the same source association differ in 3 of
18,000 wet cells, maximum `2.842170943040401e-14 K m` (one ULP at the largest
cell); salt is exactly equal. Therefore the host-constructed candidate is not
an exact predictor and the source-derived LDF routing statement is not eligible
to land in this round.

The first non-bit statement in this walk is the association at
`ocean_model_latlon_cgrid.py:1862-1912`: production forms its stage-3 content
as the already-materialized FCT advection content plus the Kmm-weighted source,
while round 68's NumPy reconstruction of that statement moves three temperature
cells by one ULP. This is an instrument boundary, not a new NEMO-physics
attribution. The next discriminator must execute the candidate source tuple
through the same compiled JAX helper rather than reconstruct its result on the
host.

## Source record and admission

Compiled GYRE clears and accumulates tracer `Krhs` through advection, SBC, QSR,
and LDF at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:827-950`, then calls
ZDF at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:917-965`. The active
isoneutral operator reads Kbb tracer gradients at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traldf_iso.f90:183-192`, forms its
Kmm-metric face fluxes at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traldf_iso.f90:230-246`, and adds
their divergence to `Krhs` at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traldf_iso.f90:287-305`. ZDF forms
the Kbb tracer plus Kmm-weighted `Krhs` content before the solve at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90:547-565`.

The admitted round-64 record again reports 43/63 exact inherited records, 20
changed records, and all 132 consumed values admitted. The all-oracle content
rebuild has zero unequal cells. The content one-ULP plant changes exactly one
wet cell (`2.842170943040401e-14 K m`). Wrong worktree and wrong record-producer
plants both exit 1 before creating an output report. The successful report is
stamped to `1908f25c208594e620b7ac015480127036553da0`, fp64 model content,
fp64 geometry, and fp64 record.

## Frozen measurement

Report `round68_production_fct_before.json` has SHA-256
`4291acf842ea2f446740c925ab0612dd47323bc2e5f42465f4f86ff8eae2c0dd` and
status **REFUTED**.

| Row | Temperature | Salinity |
|---|---:|---:|
| production baseline rebuild unequal / maximum | `3 / 2.842170943040401e-14` | `0 / 0` |
| predicted routed content maximum vs NEMO | `5.954039670541533e-5 K m` | `7.651457053725608e-6` |
| predicted routed kt=3 maximum vs NEMO | `8.916073106490785e-7 K` | `7.235656340753849e-8` |
| production prediction vs round-67 recomputation unequal / maximum | `5908 / 2.2737367544323206e-13` | `5628 / 1.8189894035458565e-12` |

The magnitude prediction would improve temperature content by 28.2x and is
inside the frozen round-67 floor plus measured association. Those non-exact
benefits do not override the failed exact control. The earlier LDF source-order
claim remains source-confirmed, but no new claim of trajectory improvement is
made.

## Rule 12 and card disposition

| Card | Changed statement | Disposition |
|---|---|---|
| GYRE | none; candidate stopped before edit | kt=1..10 and days 1..30 **UNREACHED**, not passed |
| LOCK_EXCHANGE | none | exact before/after trajectory **UNREACHED**; no source move to score |
| OVERFLOW | none | exact before/after trajectory **UNREACHED**; no source move to score |
| DINO | none; separate modified-leapfrog program | execution comparison **UNREACHED**; final production diff is empty |
| ORCA2 | none; native record absent | **UNMEASURED WITH SPEC**: resolve native card; record post-SBC/QSR/LDF Krhs and pre/post-ZDF T/S at kt=1..10; run cumulative/content and trajectory gates; require exact replay, every moved row registered, no AT-BAR loss, and no earlier first-over-bar boundary |

Decision 36's after arm remains the unchanged trajectory baseline. No AT-BAR
row moved because no production row moved. No configuration/default, carried
state, stabilizer, NEMO source/build/run, year harness, reconciliation gate,
freshwater pair, #1484 guard, or held manifest was changed.

## Review and focused checks

The required separate `codex exec --sandbox read-only` review was attempted on
the complete round-68 diff. The direct invocation produced, verbatim,
**“Error: failed to initialize in-process app-server client: Read-only file
system (os error 30)”**. A writable isolated-state retry initialized the client
but could not reach the service; its terminal result, verbatim, was
**“ERROR: Reconnecting... waiting for network”**. Neither invocation emitted a
`VERDICT:` line. The review requirement is therefore **UNMET/BLOCKED**, not an
approval; no production diff is being shipped.

Round-63 admission/plant tests, round-66 operand tests, and round-67/68 content
tests pass 14/14. Python compilation and diff checks pass. The shifted-citation
plant exits 1 and the unplanted citation gate passes all seven mapped citations.

## ASKED / UNASKED and OPEN

| State | Item | Disposition |
|---|---|---|
| ASKED | configuration choice | none encountered |
| UNASKED | configuration, carried state, stabilizer, NEMO, or harness change | none performed |

OPEN for round 69: preregister a JIT-native causal arm that passes the captured
GM/Redi rate into the same production WS helper's stage-3 source tuple, then
freezes that helper's returned content and post-solve kt=3 metric row. The arm
must reproduce the untouched baseline exactly before its LDF addition, retain
the failed three-cell host reconstruction as a retraction, and use the same
round-64 oracle/admission record. Only an exact JIT-native prediction may
reconsider the source-derived LDF routing; complete-FCT advection remains the
largest residual owner after it.
