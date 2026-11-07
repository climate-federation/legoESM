# NEMO testcase Lane 4 — ORCA2 Phase-2b exchange handoff receipt

> **SUPERSEDED — DO NOT RUN.** User Decision 7 (2026-09-06) selects the
> icebergs-off comparison variant.  This shipped-deck handoff remains the
> `SHIPPED_DECK_RECORD`, but its launcher now exits 69 before MPI.  The active
> handoff is `nemo_testcases_l4_orca2_phase2c_variant_handoff_receipt.md`.

Date: 2026-09-04

Base: `f4ac8a2a3a1f4deaf857ba1fa6ba1ee9b645a50f`

Preregistration: `109152161`

Schema correction before execution: `383dcefdc`

Instrument, gate, and launcher preparation: `9a816aaaa`

Result: **STOP at B2b-X — one NEMO run prepared**

No legoESM numerical operator was added or executed in this continuation.
The first required ocean input is absent from the accepted record set, so the
ordered ladder stops before B3a.  The only NEMO change is a configuration-local,
rank-0, WRITE-only record in the copied `ORCA2_OMIP_L4`; no shipped NEMO file
or deck file was edited.  The selected `ln_icebergs=.true.` deck remains exact.

## 1. Confirmed boundary gap

The existing `oracle_si3_exchange_frames.bin` records the state inside
`ice_stp`, immediately after `ice_update_tau`
(`cfgs/ORCA2_OMIP_L4/MY_SRC/icestp.F90:217-238,250-291`).  It is not the
surface state delivered to ocean physics.

The source-ordered crossing is:

| order | executed operation | source | consequence for ocean operands |
|---:|---|---|---|
| 1 | bulk surface condition | `src/OCE/SBC/sbcmod.F90:415-424` | creates open-ocean stress/heat/freshwater fields |
| 2 | SI3 step and flux/stress update | `sbcmod.F90:474-480`; `src/ICE/iceupdate.F90:145-183,187-203,388-406` | replaces/composes `qsr,qns,emp,sfx,fr_i`, ice mass and ice-ocean drag/stress |
| 3 | inherited Lane-3 exchange record | `cfgs/ORCA2_OMIP_L4/MY_SRC/icestp.F90:237-238` | records the intermediate state only |
| 4 | iceberg step | `sbcmod.F90:484`; `src/OCE/ICB/icbstp.F90:71-142` | executes at odd `kt` for resolved `nn_fsbc=2` |
| 5 | iceberg-to-ocean melt/heat | `src/OCE/ICB/icbthm.F90:286-295` | changes `emp` and `qns` |
| 6 | runoff and prescribed runoff-iceberg climatology | `sbcmod.F90:494`; `src/OCE/SBC/sbcrnf.F90:128-168` | changes `rnf`, `fwficb`, heat and salt runoff content |
| 7 | annual freshwater carry | `sbcmod.F90:498`; `src/OCE/SBC/sbcfwb.F90:292-295` | changes `emp` and `qns` |
| 8 | final full halo exchange | `sbcmod.F90:529-537` | supplies tripolar/cyclic halo values |
| 9 | U/V stress interpolation | `sbcmod.F90:539-547` | creates `utauU,vtauV` consumed by ocean momentum |
| 10 | final record | copied `MY_SRC/stprk3.F90:158-159,416-441` | exact post-`sbc`, pre-EOS/ZDF/external-mode boundary |

Promoting the intermediate record to item 10 would silently omit executed
operators and violate the no-Frankenstein rule.  This is
**CONFIRMED MISSING_ORACLE_OPERAND**, not a measured legoESM mismatch.

SI3 dynamics/thermodynamics are labelled
`UNMEASURED_PENDING_ICE_MERGE`; the iceberg producer is labelled
`UNMEASURED_PENDING_USER_DECISION`.  Their recorded outputs will be labelled
`ORACLE_SUPPLIED` when the ocean walk resumes, never certified legoESM output.

## 2. WRITE-only instrument and schema

The new writer is in the copied configuration only:

```text
/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/
  cfgs/ORCA2_OMIP_L4/MY_SRC/stprk3.F90
SHA-256 1d8d6ef62550d757859c282865e14d7f24fb28a51069799cf27271cad1b59bf7
```

It adds two visibility imports, one guarded call immediately after `sbc`, and
one local writer.  The only assignment is to the writer-local `magic` string;
all model arrays occur only in `WRITE` statements.  `IF(.NOT.lwp) RETURN`
gives the stream a single rank-0 owner.  The writer opens, writes, closes, and
does not alter any time level or arithmetic expression.

The schema is magic `NEMO_L4_SBCIN_1`, 13 native four-byte integers, and
binary64 payload.  Header allocation-class counts are derived from the write
list and checked against the NEMO declarations:

| class | arrays/families | shape at this layout |
|---|---:|---:|
| explicit full | 20 | `94 x 152` |
| `A2D(0)` 2-D | 12 | `90 x 148` |
| `A2D(1)` 2-D | 1 (`rCdU_ice`) | `92 x 150` |
| `A2D(0),jpts` | 2 (`rnf_tsc,rnf_tsc_b`) | `90 x 148 x 2` each |

The derived payload is `512,680` f64 values and the expected record size is
`4,101,508` bytes.  The gate walks all 35 named fields/families and fails on
bad magic, any header/allocation count, derived size, non-finite payload, or
post-run integrity digest.

The initial preregistration draft incorrectly flattened the allocation classes
to 33 full arrays.  It was caught before execution.  That build and prepared
directory are retained but retracted; their launcher now exits 69 before any
MPI command.  The correction is first-class in the preregistration and tool.

## 3. Scalar-math build

Command:

```text
PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:... \
  ./makenemo -n ORCA2_OMIP_L4 -m conda-scalarmath -j 8
```

The incremental build preprocessed one file and compiled `stprk3`, its
dependent `nemogcm`, and the executable entry point.  The compile line retains
`-fdefault-real-8 -O3 -funroll-all-loops -fno-tree-vectorize`.

| artifact | bytes | SHA-256 |
|---|---:|---|
| `build/nemo_ORCA2_OMIP_L4_phase2b_exchange_schema.exe` | 54,904,016 | `27f02c52b9509201319136deecdf36321ce31324c260796fe646f3bad73a91ec` |
| `build/build_ORCA2_OMIP_L4_phase2b_exchange_schema.log` | 4,711 | `2262044d9422a8c5553a09a3fd3f147918ef74b5dfeeef0ee734aa614f3d7dc3` |
| `build/phase2b_exchange_schema_MY_SRC.sha256` | 2,100 | `5dd1f5079839e1bccb91efbd7ff694ed384bffce2bff941c27aab941ec4df337` |
| `build/phase2b_exchange_schema_ZGV.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

`nm -D` finds zero `_ZGV*` symbols.  The linked toolchain and scalar-math arch
are unchanged from accepted Phase 1.

## 4. Prepared user-shell run

Execute exactly this directory's `run.sh`:

```text
/data/abyssal/dbalwada/nemo-testcases-l4/runs/
  instrumented_phase2b_exchange_schema_10step_np2
```

Packaging is the established Lane-4 policy: 19 regular deck-file copies, 40
absolute read-only input symlinks into the immutable unpacked archive, and one
absolute `nemo` symlink to the exact binary above.  `namelist_cfg` remains the
accepted ten-step deck (`nn_itend=10`, `nn_stock=10`, `nn_istate=1`).  The
smallest supported layout remains `-np 2`, automatically resolved as
`jpni=2,jpnj=1`; one processor in X is invalid in the shipped iceberg path.

| item | SHA-256 |
|---|---|
| binary through run symlink | `27f02c52b9509201319136deecdf36321ce31324c260796fe646f3bad73a91ec` |
| `deck_files.sha256` | `e1974d9db5974f54d451fe1112b8f22a404516adcc7cbcf0b035fbe1cccba70b` |
| `input_files.sha256` | `3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5` |
| accepted launcher | `56dc3104e2f7c56d35d16c87a5687d4d3f7fc0fc3b55e4a492abda43fba72351` |
| 63-item prepared census | `b3d61cd6833e22286d522e7b68daa9ea66022a006a2f07b4b8e2c72362b266f9` |

The launcher checks directory identity, refuses stale output, checks binary,
deck and every input, fixes one thread per library, uses Bash `time`, and runs:

```text
mpirun -np 2 --oversubscribe ./nemo
```

The earlier never-executed flat-schema directory is preserved at
`runs/instrumented_phase2b_exchange_10step_np2`.  Its current 63-item census
has SHA-256
`6bb61199452c58ca4661b6552e1fa3f2cee71128a5f47a91291d2ec45f284a9a`;
its retained launcher SHA-256 is
`7edbff8a2f9930148407d6eecc257c1aeb89487cbf9ac85491d0c4a28898e5c6`
and exits 69.  It must not be run.

## 5. Validation frozen before execution

`nemo_testcase_l4_orca2_phase2b_exchange_gate.py` reuses the accepted Phase-1
90-stream schema walk and `validate_identity` directly.  It presents the old
parser with a temporary 90-file symlink view so the one newly registered file
cannot weaken the frozen inventory.  After execution it will:

1. validate all 90 old streams;
2. validate the new post-`sbc` stream and every named allocation class;
3. run byte identity against
   `runs/uninstrumented_10step_np2`, with dynamic restart/history counts and
   the established timestamp/timing exclusions; and
4. run five binding new-stream plants.

A synthetic correctly sized stream passed the schema walk at
`512680 / 512680` values and `4,101,508` bytes.  The same validator returned
nonzero for all five plants: bad magic, wrong derived class count, truncated
payload, NaN, and a one-ulp active-field mutation bound by the record digest.
This synthetic check validates the tool, not the absent oracle result.

## 6. Coverage and time-level registry at the stop

| family | registered level/boundary | disposition |
|---|---|---|
| final `utau,vtau,utauU,vtauV,taum` | after `sbc`, before EOS/ZDF; current forcing | AWAITING_ORACLE_RUN |
| `utau_b,vtau_b,qns_b,emp_b,sfx_b` | kt=1 before carries initialized at `sbcmod.F90:549-575` | AWAITING_ORACLE_RUN |
| final `qsr,qns,emp,sfx` | post-SI3/post-iceberg/post-FWB current forcing | AWAITING_ORACLE_RUN |
| `qsr_tot,qns_tot,emp_tot,wndm` | bulk/SI3 current surface diagnostics/operands | AWAITING_ORACLE_RUN |
| `rnf,rnf_b,rnf_tsc,rnf_tsc_b,fwficb` | current/before runoff levels after `sbc_rnf` | AWAITING_ORACLE_RUN |
| `fwfice,fr_i,snwice_mass,snwice_mass_b,snwice_fmass,rCdU_ice` | SI3 current/previous ice cadence state delivered to ocean | ORACLE_SUPPLIED_PENDING_RUN; producing SI3 operator unmeasured |
| `utau_icb,vtau_icb` | pure atmospheric stress supplied to current iceberg step | AWAITING_ORACLE_RUN |
| iceberg calving, heat, floating melt, stored heat | after current odd-kt `icb_stp` | ORACLE_SUPPLIED_PENDING_RUN; producing iceberg operator unmeasured |
| ocean `T,S,u,v,ssh` | `Kbb=1`, already B2-O bit-exact | VERIFIED from Phase 2 entry gate |
| EOS/HPG/external mode/transports/tracers/ZDF | first consumers after this stop | NOT ENTERED |

Every field in the new write list has a boundary and ownership disposition.
No missing field is represented by a zero or inferred from another time level.

## 7. ASKED / UNASKED

| item | status | action |
|---|---|---|
| continue ocean ladder using exact exchange operands | ASKED | stopped to acquire the actual final operands |
| SI3/iceberg producers not certified | ASKED | explicit pending labels; no producer code entered |
| retain shipped `ln_icebergs=.true.` while decision is pending | ASKED | exact deck retained |
| any needed NEMO rerun is prepared and executed by user shell | ASKED | one accepted directory above; no MPI run by agent |
| rank-0 record plus documented halo mapping | ASKED inherited | `lwp` only; full/reduced allocation shapes explicit |
| add a post-`sbc` writer | UNASKED necessary instrumentation | first source-valid point containing all final ocean operands |
| detect and retract flat-size schema before execution | UNASKED control outcome | old build/directory preserved, launcher refuses |
| extend shared legoESM numerics before operand validation | UNASKED and forbidden by stop rule | not done |

## 8. Resume condition

After the user-shell run, re-run the committed exchange gate with the retained
record SHA-256, the uninstrumented ten-step control, and planted controls.  If
all schemas, ordinary-output identity, and plants pass, relabel the selected
fields `ORACLE_SUPPLIED` and resume B3a at NCAR/CORE `sbcblk` in NEMO order.
The shared implementation to reuse is
`packages/ocean/legoesm/ocean/bulk_flux_omip.py::air_sea_fluxes`, together
with the certified Lane-3b bulk changes when their scheduled merge lands;
forcing placement remains the shared Lane-2 WS-RK3 path.  No ORCA2-local
numerical branch is authorized.
