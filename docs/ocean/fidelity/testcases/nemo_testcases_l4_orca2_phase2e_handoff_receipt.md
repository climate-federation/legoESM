# NEMO testcase Lane 4 — ORCA2 Phase-2e schema-fix handoff

Date: 2026-09-06

Parent: `b35e96792cbb415f9d4f3b3c16e17e7330e14ea8`

Corrective preregistration: `6dba63974`

Result: **STOP before O1 scoring.  The executed O1 record is malformed and
strict inherited-record identity is 84 / 91; one schema-fixed run is
prepared.**

No legoESM numerical operator was added or changed.  The NCAR bulk, RGB QSR,
EOS/HPG, external mode, transports, FCT, BBL, and TKE/EVD/IWM boundaries were
not entered.  SI3 remains `UNMEASURED_PENDING_ICE_MERGE` and its exchange
fields remain oracle-supplied.

## 1. Executed run provenance

Run:
`/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_o1_instrumented_10step_np2`

The unchanged launcher has SHA-256
`88590198b3222fc8c76eb5c68404f24c19cc3c98abf0978ecdc0f1bd3c6d768c`.
The run reached `time.step=10`, `MPIRUN_RC=0`, `RUN DONE`, in 12.898 seconds
and produced 92 `oracle_*.bin` files.  `STOP 0` in standard output is NEMO's
normal successful termination.  The complete 188-item executed-run manifest
is 17,579 bytes, SHA-256
`84f154edccfc9cb8c3e0303003f85d87000e87bd70fd2abae644bca949471196`.

## 2. O1 schema finding

**CONFIRMED — malformed frame 2.**  The new record is 3,113,568 bytes,
SHA-256
`982347ad6f617e388104c108a782b21e79d2ee95583872849cd865b8e24c4f31`.
Frame 1 has valid magic and header `(1,1,0,90,148,9,0,64)` and exactly
`9*90*148` f64 values.  Frame 2 starts at the derived offset with valid magic
and header `(1,1,1,90,148,20,0,64)`, but has 23,232 trailing bytes.  The gate
exits 1 with `trailing payload`; its 54-byte JSON and stdout each have SHA-256
`fd2e1c1cf015fe62be68b43e391b9b5fef37f3a29de518a2a7d2a968d40f8ee1`.

The source cause is exact.  `src/OCE/SBC/sbc_oce.F90:184-216` allocates
`emp`, `utau`, and `vtau` on full `jpi*jpj=94*152`, while the other frame-2
fields and declared header use `A2D(0)=90*148`.  The copied writer at
`cfgs/ORCA2_OMIP_L4/MY_SRC/sbcblk.F90:710-713` wrote those three unsliced.
The excess is exactly
`3 * (94*152 - 90*148) * 8 = 23,232` bytes.  This is a writer defect, not a
model-physics result.

Because the base record is invalid, its mutations are not admissible planted
controls: any mutation would fail vacuously after an already-failing schema.
The gate correctly stops before running or crediting its count and one-ULP
plants.

## 3. Instrument-inertness and inherited records

**CONFIRMED — ordinary outputs are inert.**  The dynamic identity gate passes
against the accepted icebergs-off uninstrumented control:

- 4 / 4 restart shards are exact bytes;
- 8 / 8 history payloads are exact apart from the registered global creation
  timestamp;
- layout, mesh masks, initialization output, resolved namelists, `time.step`,
  and normalized `ocean.output` pass their frozen exact rules.

**CONFIRMED — inherited raw record identity is 84 / 91, not 91 / 91.**
The seven differing files are:

- `oracle_bt_advmean_operands_kt00000001.bin`
- `oracle_bt_drag_operands_kt00000001.bin`
- `oracle_bt_ordered_operands_kt00000001.bin`
- `oracle_bt_substeps_kt00000001.bin`
- `oracle_ocean_surface_input_kt00000001.bin`
- `oracle_rkstage3_wzv_kt00000001.bin`
- `oracle_slow_forcing_kt00000001.bin`

The full 40,027-byte identity JSON has SHA-256
`f3da75ce9657bd201734ec06eff397fc9e86acecdb2419f5b7005605a2044963`.
All 90 inherited schemas and the post-`sbc` schema independently pass; the
30,364-byte summary has SHA-256
`869857b953aed24c1f289eecba969068db01822960803737f9ff4137dc1c5e65`.

For the post-`sbc` record, inspection is exact: its 24 differing f64 values
are the first four elements of each of `utauU`, `vtauV`, `utau_b`, `vtau_b`,
`rnf_b`, and `rnf_tsc_b`; values are pointer-like finite subnormals in the
`4.6e-310..6.9e-310` range.  The other six streams have repeated byte-offset
patterns at carried-array positions.  It is **PLAUSIBLE**, not yet an accepted
waiver, that all seven raw differences are the same uninitialized-slot class.
The ordinary-output identity proves the added writer did not alter carried
physics, but it does not make raw record bytes equal.

The strict comparison is retained in the acquisition gate.  It reports every
record and exits nonzero unless the exact count equals the actual total; no
literal `84` or `91` is used as a pass condition.

## 4. First boundary and required decision

The ordered ladder remains stopped at **O1** before arithmetic:

| boundary | owner | disposition |
|---|---|---|
| mapped nine-field `fld_read` result | `ORCA2_OWNER` | `UNMEASURED_INVALID_ORACLE_RECORD` |
| NCAR bulk leaf given mapped inputs | `LANE3B_OWNER` | NOT ENTERED |

This is not an over-bar result; there is no valid O1 score.  Review must choose
how strict identity treats source-undefined record slots:

1. register those slots outside the oracle payload and compare every
   source-defined cell bitwise; or
2. regenerate the accepted VARIANT records with all affected writers using
   canonical writer-local values for undefined slots.

The lane does not choose between these without user authority.  Under either
choice, the O1 record itself must first be regenerated with a valid header.

## 5. WRITE-only repair and prepared run

After preregistration, the copied configuration changed only the frame-2 WRITE
list to `emp(A2D(0)), utau(A2D(0)), vtau(A2D(0))`.  Array sections occur only
in the I/O list; no model field is assigned.  The copied `MY_SRC/sbcblk.F90`
now has SHA-256
`fcb1d0add456f709b49c78e1e6a1c12fd79c59b80da81fe2f55ef7fe03d6c3ab`.

The scalar-math rebuild contains `-fdefault-real-8 -O3 -funroll-all-loops
-fno-tree-vectorize`; its `_ZGV*` census is empty.  The local Conda entry-point
warning is preserved in the successful build log.

| artifact | bytes | SHA-256 |
|---|---:|---|
| `build/nemo_ORCA2_OMIP_L4_phase2e_o1_schemafix.exe` | 54,920,400 | `a39000462c7da6278faa197757ffa6fb78593e3001862054cc197d617b1aa20a` |
| build log | 10,994 | `2b1b3f170c26165bde40035137cf67504db13509f670e41c277e6c21f5e54d64` |
| 14-file `MY_SRC` manifest | 2,100 | `91bac34d01d7eb0517637e66ff7ebcaa74a9570a9785eeb413809d2828457cf3` |
| `_ZGV*` census | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| prepared 63-item run census | 5,343 | `121b3b7827ba64653178f93b150d217a10ced598342eb82f915d9611e08d2078` |
| launcher | 2,282 | `44b826fdea3b063e31242088755337093b5997e5232942d1280fa6a76662752c` |

Run exactly:

```text
/data/abyssal/dbalwada/nemo-testcases-l4/runs/
  variant_icebergs_off_o1_schemafix_instrumented_10step_np2/run.sh
```

The directory contains 19 copied deck files, 40 absolute immutable input
symlinks, an absolute binary symlink, two manifests, and the launcher.  It has
no run output.  The launcher verifies every hash, refuses stale output, uses
Bash `time`, sets one thread per library, and runs
`mpirun -np 2 --oversubscribe ./nemo`.

## 6. Coverage and time-level register at stop

| family | level/boundary | disposition |
|---|---|---|
| VARIANT ocean T/S/u/v/ssh entry | kt=1, `Kbb=1` | VERIFIED `0 / n` from Phase-2d |
| SI3 jpl=5/Prather producer | pre-ocean exchange | `UNMEASURED_PENDING_ICE_MERGE` |
| SI3-to-ocean exchange | current/before levels recorded by writer | `ORACLE_SUPPLIED` |
| mapped nine CORE fields | post-`fld_read`, kt=1 `fnow` | INVALID O1 record; replacement prepared |
| open-ocean NCAR result | post-`blk_oce_2`, pre-SI3 | INVALID O1 record; replacement prepared |
| RGB QSR through TKE/EVD/IWM entry | registered Phase-2d levels | NOT ENTERED |

## 7. ASKED / UNASKED

| item | status | disposition |
|---|---|---|
| validate O1, both frames, and plants | ASKED | frame 1 valid; frame 2 invalid; plants withheld as non-binding |
| compare inherited records and restarts | ASKED | 84 / 91 raw records; 4 / 4 restarts; ordinary identity PASS |
| continue ordered ladder | ASKED | stopped at invalid O1 before any score |
| report first over-bar boundary | ASKED | none measured; O1 is `UNMEASURED_INVALID_ORACLE_RECORD` |
| prepare required rerun and stop | ASKED | one schema-fixed directory prepared; not executed by agent |
| add strict inherited-record comparison to gate | UNASKED necessary validation | fail-closed; actual loop counts |
| choose an undefined-slot waiver/canonicalization policy | UNASKED and authority-expanding | not chosen |
| change any legoESM numerical operator | UNASKED and forbidden at this stop | not done |
