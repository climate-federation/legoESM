# Lane 3b exchange-stream drift ticket

Date: 2026-09-04  
Tracker: `climate-federation/legoESM#1699`  
Preregistration commit: `1539915d36f`  
Verdict: **CONFIRMED inactive/uninitialized storage; FIXED and reproducible**

## Result

Every historical byte difference is outside a valid C1D exchange value.  The
2,120-byte records have identical headers and 260 fp64 payload values.  Across
all eight retained full-year streams, no reduced value, full-array center
(Fortran-linear index 12), or active `A2D(1)` center (index 4) changes.  The
union of 52,702 changed bytes is:

| field/member | frame(s) | record byte offsets | cause |
|---|---:|---:|---|
| `rCdU_ice[0]` | 1--8760 | 616--621 | Entire field inactive/uninitialized in SAS; only this allocator residue varied. |
| `utau[0]` | 1 | 1288--1293 | Halo invalid at the writer's call site. |
| `vtau[0,2:11,13:23]` | 1 | within 1488--1687 | Halos invalid at the writer's call site. |
| `emp[0]` | 1 | within 1712--1911 | Halo invalid at the writer's call site. |

For the receipt-pinned transitions, differing-byte totals are 52,702 for
`998f...` to `7f22...`, 52,702 for `7f22...` to `6eef...`, and 52,681 for
`6eef...` to `f02d...`.  The complete machine-readable map, including the
additional retained `gate`, phase-2b, and phase-4 streams, is
`nemo_testcases_l3thd_exchange_drift_historical.json`.

## Source ownership

The record layout comes from the config-local writer
`nemo502_si3thd_MY_SRC/icestp.F90:246-269`, the allocations at shipped
`sbc_ice.F90:124-146` and `sbc_oce.F90:184-216`, and the `A2D` bounds at
`do_loop_substitute.h90:53-57,72-85`.

`sbcmod.F90:474-482` calls `ice_stp` and explicitly says `utau`, `vtau`, and
`emp` are not valid on halos at that point.  Their halo link occurs only after
the writer returns (`sbcmod.F90:529-537`).  `rCdU_ice` is more strongly
inactive: `zdfdrg.F90:52` defaults `ln_drgice_imp=.false.` when
`zdf_phy_init` is absent; only the OCE driver calls that initializer
(`OCE/nemogcm.F90:463`), while SAS does not.  The sole assignment is guarded by
that false flag (`iceupdate.F90:388-396`).  Its nominal center's retained bits
spell allocator-like ASCII and are not a physical value.

## Config-local writer correction and two-run proof

No shipped NEMO file or shipped configuration was changed.  The config-local
writer serializes zero for all of inactive `rCdU_ice`.  At `nit000` it also
serializes zeroed `utau/vtau/emp` payloads with only the registered `A2D(0)`
column copied.  It never writes to a model array.  Later halo payloads are
preserved, although consumers must still waive them because their time level is
not the active column's time level at this call site.

Two independent source copies were rebuilt with `makenemo -n C1D_OMIP_L3 -m
conda -j 8 -y`; each was run directly for 8,760 hourly steps on CPU, without
an MPI launcher.  Both exit 0 and produce byte-identical exchange streams:

| artifact | rebuild A SHA-256 | rebuild B SHA-256 |
|---|---|---|
| config-local `MY_SRC/icestp.F90` | `c2384d4d4ff96c4aa9c880ca7de07b7180355e361e2c8584f59fba84e53ec982` | same |
| `nemo.exe` | `92d89cb1ce3e4bf8e495967b8f9549e3633577a5bff2c3d45d9392f9d4cb7856` | `25f24cd2c0927b19bd5e1db18bcf42fca2f037fa61d2d5df59a54a77219b370b` |
| deterministic exchange stream | `091395cf604e83d88fbf458c4ef76ac3df2d5a65dac9cdc502e224cc1d1af2e4` | same |
| thermodynamics stream | `7fc9df2707a85581075e3c69b26784155151a55640a5693eb32c34fb710ea49b` | same |
| ZDF-input stream | `cd1b15c821f19442a840e99c067c640e5146b754fc137a2c81e88856d6ea7efd` | same |
| `ocean.output` | `3e47d39f061ca1f9fa111a46b01bb3d1dbcfa10be73422fced0abc9e4af25430` | same |
| ocean restart | `84ed40c5e5d46f9830f4c203b3e3a79f6dfffaf142dc4c44347648e2cd9f5265` | same |
| ice restart | `b61cb8443e14b3f0868ef125f621c0b1748aff7bc827dfab0d4044fc3f773b4e` | same |

Different executable hashes reflect the independent absolute build roots; all
scientific products above are identical.  The exact stability result is in
`nemo_testcases_l3thd_exchange_drift_stability.json`.

## Downstream disposition

- Phase 1's frame count and `fr_i=0.9` range remain valid.  Its exchange gate
  checked finite payloads and `fr_i`, not pointwise values.  The earlier phrase
  “writes every array” remains a writer inventory, not semantic verification;
  a correction is added to the phase-1 receipt.
- Phase 2's production result uses the dedicated ZDF-input stream.  Its retained
  legacy owner arm alone reads exchange `qns_ice/dqns_ice`; both are bitwise
  identical in every retained stream.  Phase 3 onward year gates do not read
  the exchange stream.
- ZDF-input files are not claimed byte-identical across schema revisions:
  phase 2 is 1,261,440 bytes (`5522eadc...`), while phase 6 is 1,471,680 bytes
  (`cd1b15c...`).  Only like-schema phase-6 rejected/write-only files are
  identical.
- The column card's old `7f22...` pin is superseded by deterministic
  `091395cf...`; rung 3.5b uses the latter.

## Scalar-math oracle V2 update (2026-09-04)

User Decision 4 required a fresh oracle built with
`arch-conda-scalarmath.fcm`, SHA-256
`132f7a0500c4f0e86d8d3bf7864974a82e1dea5d83166dcfdfaf409e2ca04561`,
whose production flags append `-fno-tree-vectorize`.  Independent copied source
trees/configurations and run roots `c1d_omip_l3_sasice_scalarmath_v2_a` and
`c1d_omip_l3_sasice_scalarmath_v2_b` each completed all 8,760 steps directly
on CPU without an MPI launcher.  `nm -D` reports zero `_ZGV*` symbols in both
executables.

The V2 exchange stream is byte-identical between rebuilds and to the retained
stable V1 stream: SHA-256
`091395cf604e83d88fbf458c4ef76ac3df2d5a65dac9cdc502e224cc1d1af2e4`,
18,571,200 bytes, 8,760 records.  Thus the exchange drift ticket remains
**CONFIRMED / REPRODUCIBLE** under scalar math.  The bulk, thermodynamics, ZDF,
DH, remap, and reassociation streams and both restarts are likewise A/B and
V1/V2 byte-identical.  The three annual model-output NetCDF payloads are
bit-identical but their global `TimeStamp` attributes differ; that metadata is
not part of the exchange stream.  Full hashes and the byte inventory are in
`nemo_testcases_l3thd_scalarmath_v2_gate.json`.

## Controls and tests

The exact identity gate compares all bytes, not just active fields.  Plants in
a reduced active field, an invalid full-array halo, and active index 4 of
`rCdU_ice` all exit red.  The last plant also proves that historical
classification cannot silently excuse a future active-center change.

## Decisions

| Choice | State | Disposition |
|---|---|---|
| Diagnose and stabilize before rung 3.5b | ASKED | Complete. |
| Preserve all retained roots | ASKED | Complete; none deleted. |
| Normalize only serialized data | ASKED | Complete; model state untouched. |
| Use two independent rebuilds and reruns | ASKED | Exact stream identity confirmed. |
| Recheck exchange reproducibility under scalar-math oracle V2 | ASKED | Two fresh builds/runs retain the same `091395cf...` hash. |
| Reuse an old drifting hash for rung 3.5b | UNASKED | Rejected. |
| Treat ZDF schema revisions as content drift | UNASKED | Rejected and disclosed. |
| Modify shipped NEMO | UNASKED | Forbidden and not done. |

## Flagged for future deletion

Nothing was deleted.  All historical roots remain retained evidence.  Their
nondeterministic exchange streams are superseded for new certification but
flagged, not removed.
