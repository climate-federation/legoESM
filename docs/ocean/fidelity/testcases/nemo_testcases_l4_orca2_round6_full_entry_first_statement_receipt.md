# NEMO testcase Lane 4 — ORCA2 card round 6 full-entry receipt

Date: 2026-09-22

Starting tip: `0dff1f286a32b8dd460b6965cfc965f2ed16b1d2`

Preregistration: `625002a6a23175968a4f3eb5cb784dea3f815818`

Status: **HELD.**  The returned round-5 record is present and admitted: both
twins have all 20 rank-local surface frames and all 20 rank-local step-entry
frames.  The first full-domain non-bit statement is earlier than the predicted
RK stage: at kt=1 the independent card omits NEMO's active ORCA2 hand edits to
initial T/S, so T is first.  The production step then refuses EOS-80 at the BN2
boundary before stage 1.  Consequently the kt=10 carried magnitude is
**UNMEASURED**, not zero and not extrapolated.

No model package, card selector, scientific configuration, stabilizer, NEMO
source, or sea-ice registry entry changed.  Sea ice remains out of scope and
the six-entry `unmeasured_features` tuple is unchanged.

## 1. Gate and record inventory

This is the consolidated ORCA2 inventory requested by the card round.  A PASS
admits only the stated record or boundary, not downstream model arithmetic.

| gate or record | mechanically admitted | UNMEASURED-with-spec or current boundary |
|---|---|---|
| Phase-1 shipped-deck record | 90 frozen streams, schema/EOF, ordinary-output passivity, 10/10 plants. | No legoESM stage arithmetic; iceberg-enabled root is not the active card root. |
| Phase-2v V2 record | 101/101 twin admission and ordered TKE walk through `zWlc2`. | Historical five-category root only; first TKE debt remains kt=2 `zpelc`. |
| Phase-2y `VARIANT_ORACLE_ORCA1ICE` | 116/116 raw-exact twins, schemas/passivity, 26/26 plants; A is pinned. | The old T/S/u/v exact claim covered rank 0 only; round 6 retracts full-domain T/S identity. |
| Phase-2y resolved ice namelist | 185 fields; only registered initial-file selector differs. | Selector equality is not SI3 physics identity. |
| Round-20 Phase-1 schema | Expected FAIL for five additional streams. | Frozen 90-stream parser intentionally cannot admit the larger inventory. |
| Round-20 SI3 headers | VALID counts 5/140/10/25/15. | Schema only; no ice integration claim. |
| Round-20 exact-input admission | `STOP_SELECTOR_GAP`. | Five categories, 10 ice layers, five snow layers, salinity scheme 4, ponds, and lateral melt remain unsupported and untouched. |
| Round-4 full surface record | 20/20 surface frames and raw twin equality. | It originally lacked rank-1 entry frames. |
| Round-5 full-entry acquisition | PASS, 136 oracle streams per twin, 107 inherited streams passive. | Record is complete; candidate arithmetic is now blocked by initial T/S transcription and EOS-80 BN2 support, not acquisition. |
| Round-6 ladder gate | Full-domain entry/surface decode, Decision-52-only SSH bridge, exact kt=1 chlorophyll reconstruction, production JIT call. | `STOP_PRODUCTION_EOS80_BN2_GAP`; stage 1 through kt=10 are unmeasured. |

The live card registry, including order, remains:
`staged_gm_eiv`, `linear_implicit_bottom_drag`,
`internal_wave_mixing`, `spatial_lateral_viscosity`,
`freshwater_budget_carry`, and `si3_jpl5_layered_prather_state`.

## 2. B3 target census and admission

The acquisition script names target configuration
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY` and roots
`orca1ice_surface_entry_every_step_{a,b}_np2` under the round-5 acquisition
directory.  `ls -la` was captured for that exact target and both roots before
any inference about filenames.

| twin | all `oracle_*` | surface rank 0 / rank 1 | entry rank 0 / rank 1 |
|---|---:|---:|---:|
| A | 136 | 10 / 10 | 10 / 10 |
| B | 136 | 10 / 10 | 10 / 10 |

Every surface file is 4,101,508 bytes and every entry file is 14,288,048
bytes.  The compiled writer has a conventional filename on the writing rank
and a rank-qualified filename on every other rank; neither stream is behind a
root-only return.  Thus the B3 missing-record diagnosis is closed: the files
were under the expected names and no re-acquisition is needed.

The acquisition's own JSON is PASS with 20 surface frames, 20 entry frames,
107 inherited streams passive, raw-exact surface twins, and raw-exact rank-1
entry twins.  The round-6 gate independently verifies every one of its 30
registered new-file digests.  The census JSON SHA-256 is
`e914ec977c24b96aef3efbb742cc6177c5f80013dafad74deaa05c73d9578b36`.

## 3. Decision-52 labels and first statement

The labels below are binding; independent and twin results are not mixed.

| quantity | label | result |
|---|---|---|
| Card before the SSH bridge | independent | T first: 1,283/799,200 cells unequal, max 20.515075852794034 °C; S 720/799,200, max 0.3500000000000014 PSU; u/v exact; SSH differs as previously recorded. |
| Card after replacing only SSH | independent with Decision-52 SSH | SSH, u, and v are bit-exact; T and S retain the same independent mismatch. |
| Given-NEMO-entry twin | given NEMO's entry | **STOP_INITIAL_T_S_TRANSCRIPTION**; it is not constructed by silently replacing T/S. |
| First non-bit statement | independent | kt=1 entry T, before RK stage 1. |
| kt=10 carried T magnitude | independent | **UNMEASURED_STOP_PRODUCTION_EOS80_BN2_GAP**. |

The source owner is the executed ORCA2 initialization branch.  With resolved
`cn_cfg=ORCA`, `nn_cfg=2`, and `ln_tsd_dmp=true`, it subtracts the Alboran Sea
T/S increments and assigns the Red Sea deep-temperature profile at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dtatsd.f90:218-253`.
The card currently stops after the input-file interpolation and mask, before
these edits.  The old rank-0-only identity never sampled their global-column
region; the newly acquired rank-1 slab does.

The frozen prediction R6-P6 is therefore **REFUTED**: T is first at entry, not
stage-1 T.  This is also why the card's independent entry cannot yet be called
the Decision-52 twin.

## 4. Production execution stop

The gate reconstructs kt=1 chlorophyll from the input file and NEMO clock at
0/13,320 unequal cells, builds the production `LatLonCGridOceanModel`, retains
JIT, fp64, scalar-libm and CPU, and calls the ordinary step through the private
already-materialized stage observer.  The call refuses before RK stage 1:
legoesm's BN2 helper does not accept the card's `eos80` form.

NEMO's compiled BN2 routine receives already computed alpha/beta, interpolates
them to W points, and forms signed N2 at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/eosbn2.f90:1587-1647`.
This is an implementation gap, not permission to switch ORCA2 to `seos` or
`teos10`; no selector was changed.  The gate exits 4 with
`STOP_PRODUCTION_EOS80_BN2_GAP` and writes the partial report before refusing.

The binding ladder JSON SHA-256 is
`f4d8c3c944fa19accf44d019e9306e8bccc46176e79a4570b8f90f8bc6c0bb16`.
It is stamped to clean commit `aee86fd07fa6bf417af4bd4e94a8a7c6da54586e`.

## 5. Prediction ledger and controls

| ID | verdict | evidence |
|---|---|---|
| R6-P1 | **CONFIRMED** | Exact target and both roots exist; their listings are retained. |
| R6-P2 | **CONFIRMED** | Compiled surface and entry writers are rank-aware; all kt=1..10 rank files exist. |
| R6-P3 | **CONFIRMED** | Both twins have surface 10+10 and entry 10+10 with invariant sizes. |
| R6-P4 | **CONFIRMED** | Existing admission is PASS, 107 inherited streams are passive, and the ladder clears both record-gap stops. |
| R6-P5 | **REFUTED** | Replacing only SSH leaves 1,283 T and 720 S unequal cells on the full domain. |
| R6-P6 | **REFUTED / kt=10 UNMEASURED** | First is kt=1 entry T; EOS-80 BN2 refuses before stage 1, so no kt=10 number exists. |

The one-representable-step kt=1 T plant exits 1 at the named entry-identity
refusal.  The acquisition-digest plant exits 1 on the first surface frame.
Their stderr SHA-256 values are respectively
`3995104ee2be58f124d3b6422bc1c94b1d79c014f9d4f224295d0f484c33bb79`
and
`9bc8826aa60ab126cb4d5006dbca9ff4173d14d7685562927feb3a57199443e6`.
Unit coverage separately proves absent rank-local files restore the named
surface and entry record stops, and exact comparison distinguishes signed
zero bits.

## 6. Implementation scope, review, and tests

The gate extends the existing round-1 ladder; no parallel executor, parser, or
launcher was created.  No file under `packages/` changed, so the shared-model
GYRE base/tip trajectory requirement was not triggered.  No NEMO run was
attempted and no acquisition script is needed.

The required separate `codex exec --sandbox read-only` review was attempted.
It stopped before reading the diff because its in-process app-server could not
initialize on the read-only filesystem (`Read-only file system (os error
30)`).  Verdict: **independent review unavailable in-sandbox**.

Focused round-6 tests pass 10/10.  The single required
`tests/ocean/fidelity -n 12` battery completed in 2,440.07 seconds: 1,511
passed, seven skipped, and exactly one failure.  That failure is the listed
pre-existing SI3 scalar-math provenance red, `A MY_SRC is not verbatim`; no
new failing ID appeared.  The captured stdout SHA-256 is
`9ca26f80f807428df046db4dd72ffba3422ff01e988a12c996e97e8631156d08`.

The receipt citation gate passes with both compiled citations mapped and no
failure, unmapped citation, or map-audit failure.  Its rigid two-line shift of
the `dtatsd` citation exits 1 at `SYMBOL-NOT-AT-LINE`, proving the control
fires.  The citation-gate unit suite passes 16/16 after the round-6 maps are
committed.

## 7. OPEN

1. In the next round, perform Decision 52's already scheduled independent
   initial-state transcription, including the active ORCA2 `dtatsd` hand
   edits, and gate all full-domain kt=1 T/S/SSH cells against NEMO.  This is a
   model-package change and therefore requires the complete GYRE base/tip
   trajectory and 30-day identity proof.
2. After initial-entry identity, implement the compiled EOS-80 BN2 statement
   without changing the ORCA2 selector, then resume this same ladder at stage
   1.  Do not register a kt=10 magnitude until the production path reaches it.
3. The recorded `rnf_tsc`/`rnf_tsc_b` surface operands remain a later explicit
   tracer-source boundary; this round did not reach them.
4. GitHub issue 1455 remains an operator-post action because no GitHub
   connector is installed in this environment.

## Choices

ASKED: Decision 52's NEMO-entry SSH bridge remains the only explicit entry
replacement.  UNASKED: none.  No new decision is needed for the already
mandated independent initial-state transcription.
