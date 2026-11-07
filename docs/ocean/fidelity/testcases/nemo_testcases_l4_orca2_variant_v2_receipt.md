# NEMO testcase Lane 4 — ORCA2 icebergs-off VARIANT oracle V2

Date: 2026-09-06

Parent: `110f7521ca2cbaf74fe18ba85d74dbaa4a583008`

## Accepted root and reproducibility rule

The accepted `VARIANT_ORACLE_V2` root is:

`/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2i_rhsrank0_a_10step_np2`

The acceptance rule is mechanical: **two independent executions of the same
hash-pinned scalar-math binary, resolved icebergs-off deck, two-rank layout,
and launcher must produce the complete frozen 92-record inventory byte for
byte.**  Arm A and arm B both completed ten steps and produced 92 records;
the committed streaming gate reports 92 / 92 raw identical and complete
inventories.  Its JSON SHA-256 is
`f4d38050a5c5faeddb2ca7998ed52861be20531bc39ee9d3fd380de156446461`.
A planted byte in the RHS record passes through that comparator, exits 1, and
reports 91 / 92; JSON SHA-256
`e9755d23f9099e5407c77c7fef71679981f52ad311572729e3d0045ccfb80ac3`.

The full record manifest has 92 entries and SHA-256
`2d1a69e430d8cf3fca0c6f642578c7f32ee6dd0c6f6823fb1d9abeced3d11f8e`:

`/data/abyssal/dbalwada/nemo-testcases-l4/phase2j/variant_v2_all_92_records.sha256`

The phase-1-compatible 90-record subset manifest has SHA-256
`dcccb742d29f2c5c261c37b2ed31164ca18456ad212b3af3466672b577cc9352`.

## Schema, identity, and controls

Both twins pass the complete legacy 90-record magic/header/frame/monotonicity/
payload-size/EOF/finiteness walk and the surface-input plus two-frame O1
schemas.  A and B acquisition JSON digests are respectively
`ed0d93d75e7c7316582e8b273c1ceb6af92b3842fe9efbcbe2dd8821e2bf062c`
and `5651eb96dd5641e0fcfe02d69b86ce9cb0a03df2065adc20209b27d78890a5a1`.

Against the uninstrumented icebergs-off ten-step control, arm A is exact for
all ordinary byte-compared files, all four ocean/SI3 restart shards, and all
eight history data payloads under the raw-data-variable comparison with only
the declared global timestamp exception.  The Phase-1 gate reports PASS and
all ten controls `PASS_NONZERO`: bad magic, wrong level, truncation, trailing
byte, missing file, extra file, NaN, SI3 one-ULP, restart byte, and asymmetric
restart inventory.  Its JSON SHA-256 is
`785c52f41bb840cf4bd0040de3d603d9f929e1f000af8ab856797b87f7386c9a`.
The O1 header-count, one-ULP payload, and undefined-slot controls also exit
nonzero.

The restart/ocean-output manifest has SHA-256
`2470c3de56a2c0439962582485a87548e78a958d5b2e1190462825cb8fab92ae`:

| artifact | SHA-256 |
|---|---|
| `ORCA2_00000010_restart_0000.nc` | `28fbf31286d90f31e6f72083b942a322a4d2cbfb18db6bf4983d76bae82c7280` |
| `ORCA2_00000010_restart_0001.nc` | `4f7873be26a96f9694470e5e7b67b93e7956180045effe948e2d3080eca4a326` |
| `ORCA2_00000010_restart_ice_0000.nc` | `4015292b4663e27a8e44109c1c001603d1d82519339d8cc1285c6d4f268d8be2` |
| `ORCA2_00000010_restart_ice_0001.nc` | `f6bea2738a8e10286653d09ee8bf1bdd18a91111c9b22c3260200975f847a229` |
| `ocean.output` | `7590d8ff2afc830e73baa3238e98a87b7be44235179ce3f0c008ebab0aa7dea6` |

`ocean.output` records the step-10 restart writes, contains no `E R R O R`,
and the launcher records `MPIRUN_RC=0` and `RUN DONE` for each twin.

## Superseded roots

All earlier roots remain on disk and are retained evidence, but none is the
comparison oracle.  The following are explicitly `SUPERSEDED_PRE_V2`:

- `variant_icebergs_off_instrumented_10step_np2`
- `variant_icebergs_off_o1_instrumented_10step_np2`
- `variant_icebergs_off_o1_schemafix_instrumented_10step_np2`
- `variant_icebergs_off_phase2f_canonical_a_10step_np2`
- `variant_icebergs_off_phase2f_canonical_b_10step_np2`
- `variant_icebergs_off_phase2g_o1canon_a_10step_np2`
- `variant_icebergs_off_phase2g_o1canon_b_10step_np2`
- `variant_icebergs_off_phase2h_systematic_a_10step_np2`
- `variant_icebergs_off_phase2h_systematic_b_10step_np2`

Arm B is `V2_REPRODUCIBILITY_TWIN`, not a second oracle root.  The
uninstrumented ten-step run remains the identity control, and the
uninstrumented 30-day run remains the statistics reference.  Nothing was
deleted or renamed.

## ASKED / UNASKED

| item | status | disposition |
|---|---|---|
| pin twin A after 92 / 92 | ASKED | accepted as `VARIANT_ORACLE_V2` |
| require two independent byte-identical executions | ASKED | satisfied, complete 92-record inventory |
| hash every record | ASKED | 92-entry external manifest, hash-pinned above |
| hash restarts and `ocean.output` | ASKED | five-entry manifest and per-file digests above |
| re-run identity and plants | ASKED | Phase-1 ten plants plus O1 three plants green |
| preserve prior variants | ASKED | retained and marked `SUPERSEDED_PRE_V2` |
| alter the resolved icebergs-off deck | UNASKED and forbidden | not done |
| execute legoESM numerics in this acceptance step | UNASKED | deferred until after the V2 pin |
