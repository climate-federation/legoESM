# NEMO testcase Lane 4 — ORCA2 Phase-2d O1 acquisition handoff

Date: 2026-09-06

Parent: `6e935823577b3a60056e96bfca429f43e84ecac3`

Phase-2d preregistration: `9ebd3c91e`; O1 acquisition addendum:
`d433d2f48`

Result: **V0–V3 and C0–C1 PASS; STOP at O1 missing-oracle boundary; one
replacement NEMO run prepared**

The accepted comparison oracle is now the icebergs-off `VARIANT`.  The three
user-shell runs are valid, the ten-step instrument is observationally inert,
the card carries the identical option and remains bit-exact at ocean entry.
No legoESM ocean-stage arithmetic was added or scored.  O1 cannot assign a
mismatch to `fld_read` versus the NCAR bulk leaf from the existing final
post-`sbc` stream, so the ordered stopping rule requires one additional
WRITE-only record.  No SI3 operator or iceberg model was entered.

## 1. Accepted VARIANT oracle

The copied deck differs from the shipped-deck record only by
`ln_icebergs=.false.`.  `ln_rnf_icb=.false.`, `nn_ice=2`, `nn_fsbc=2`, all
other resolved namelist values, inputs, scalar-math flags, and the two-rank
`jpni=2,jpnj=1` layout are unchanged.

| arm | terminal step | wall seconds | result | complete-file manifest |
|---|---:|---:|---|---|
| instrumented ten-step | 10 | 12.264 | `MPIRUN_RC=0`, `RUN DONE` | `a7c1a30836c4651a99f71ad748ce292c2f66a2f7d4d16b0d2e046150fd4fe081` (187 files) |
| uninstrumented ten-step control | 10 | 13.822 | `MPIRUN_RC=0`, `RUN DONE` | `f9edd367755fe8d67f1f7b123a38aafbb15b39bd4a99001ebdb34b6315155fb3` (96 files) |
| uninstrumented 30-day | 240 | 209.101 | `MPIRUN_RC=0`, `RUN DONE` | `6f82643f2d15bf9084cfbc39b782ea141270465b2d211ae2d4ee19cea64bfdbf` (93 files) |

Each `ocean.output` contains zero `E R R O R`, `ERROR`, `STOP`, or `FAILED`
matches.  The launchers were executed unchanged and retain SHA-256 values
`e9f16a444616e14faefdb633adbe4f68981e4c04599829585e696935f9d1e63e`,
`b647d4a3c1f302f26d44bd078e1787e879828c8ea794807dbd96920e69b519a5`,
and `af46ce7583c4d077a4db4f45ba6dd83bd229819bfb17d7aac7833bdbb471022a`.
The complete per-file hashes live in the three manifests under
`/data/abyssal/dbalwada/nemo-testcases-l4/phase2d/`.

### Record schema and integrity

The accepted JSON is `phase2d/variant_oracle_gate_final.json`, 116,573 bytes,
SHA-256
`43c6d0d26503548ecec54f9c16d62ac60de1e3477cc82ee621ff6f7d2d9d71e3`.
It validates exactly 91 streams: all 90 frozen Phase-1 records plus
`NEMO_L4_SBCIN_1`.  The frozen-record manifest is
`790f7e0b46d9beaa11e62a4d0e8bef9091e40fd8a635b4c3db990288ed6c908a`;
the 91-record manifest is
`d528c89cc82c9c5cc2ae8b203e40cd296ce1067f8742fc5f06daeda8c1588684`.

Every fixed stream passes magic, complete header, derived payload count,
exact size, finite payload, and EOF checks.  Append streams pass monotone
frame/header walks with observed counts: bulk 15, exchange 10,
thermodynamics 140, ZDF inputs 25, and reassociation 5.  The post-`sbc`
stream walks all 35 fields and derives 512,680 f64 values and 4,101,508
bytes from 20 full, 12 reduced, one halo-1, and two reduced-3D families.  Its
SHA-256 is
`42d1f9735a17652e6d00d4e1641cb513b67bcc221d11605d27a7971483ed16eb`.
All six inactive iceberg slots are exact zero.

### Instrument identity, per ordinary file

The validator reports its actual loop counts: 4 restart shards and 8 history
payloads.  NetCDF history equality compares raw data-variable bytes and
fill/mask metadata; only the registered global creation timestamp differs.
Launcher/timing artifacts are excluded by the frozen rule.

| file | comparison | instrumented SHA-256 |
|---|---|---|
| `layout.dat` | exact bytes | `1e85d5ae647a3fbfdc6a843440d7be3e409bf2a52648cab67549cc8ca3962832` |
| `layout.nc` | exact bytes | `61392c48ee1435e4beb0b84297995b13add16749dd4078588df7bb6a7637494b` |
| `mesh_mask_0000.nc` | exact bytes | `0f373c6609d287bc818b1f6ef96bafdcc0b620c730219018de9728369d01ec5f` |
| `mesh_mask_0001.nc` | exact bytes | `0b6903ce4e508bbd3f3c05dfbd8290548549ed52b13f081810e717071e977933` |
| `output.init_0000.nc` | exact bytes | `850a292e77d7471d0b08bfe2103f0b4c39713847702d3fbaa33a2e4884fd1d2b` |
| `output.init_0001.nc` | exact bytes | `ed2823852ab626d65f53c8c07063ff065d18851fabd740a9624f55249ace9148` |
| `output.init_ice_0000.nc` | exact bytes | `22e52f569e3ac9a3e9136d0172f29780d774f3b2569714d422858e085ed62bcd` |
| `output.init_ice_0001.nc` | exact bytes | `01555e31709cd398ab48fea133a5f6072cfa454b63c79309476fe04cf2f5ee25` |
| `output.namelist.dyn` | exact bytes | `5bf037a14eb8a1ecf39bbb20a2adfdf523b2d4134ca2860a1138de4ab0dc87fd` |
| `output.namelist.ice` | exact bytes | `2ccc2d012159bcc1d182dc48bf34ff2abdfb09aecb45ff227cd30ebadd9a8603` |
| `time.step` | exact bytes | `83e4e460507d78e2bd843e9a6961b3b204c2b5229499b5e2f10f11b3a7d5bb78` |
| ocean restart rank 0 | exact bytes | `28fbf31286d90f31e6f72083b942a322a4d2cbfb18db6bf4983d76bae82c7280` |
| ocean restart rank 1 | exact bytes | `4f7873be26a96f9694470e5e7b67b93e7956180045effe948e2d3080eca4a326` |
| SI3 restart rank 0 | exact bytes | `4015292b4663e27a8e44109c1c001603d1d82519339d8cc1285c6d4f268d8be2` |
| SI3 restart rank 1 | exact bytes | `f6bea2738a8e10286653d09ee8bf1bdd18a91111c9b22c3260200975f847a229` |
| grid-T history rank 0 | exact variables except timestamp | `1b965ca269d3ee4d26634ce92c9bb938238ac1a346a449d13fd9ee28e7eb20c5` |
| grid-T history rank 1 | exact variables except timestamp | `f2c42a5b75d06b6635d96911f5e4ed0ccf2e9b5bb2aa10ed8dda258f586f2560` |
| grid-U history rank 0 | exact variables except timestamp | `3bd2c172a9c9cf88a5cc905ed2726fbe0ed92c28f4a447be7a8d04188f3d811a` |
| grid-U history rank 1 | exact variables except timestamp | `d8e877ca87818b08c9c31111295b22428e77c7d08ab7c8aaa3f7dca8c607480f` |
| grid-V history rank 0 | exact variables except timestamp | `33e30053bae24d4639021fbcb0a185110b3808d439a5e2508cc6c9678c95544a` |
| grid-V history rank 1 | exact variables except timestamp | `44abe366b23e971118114f2be145dc4cd47b2c2861d77f65fe73ca5e4b428a53` |
| grid-W history rank 0 | exact variables except timestamp | `9d70337d41c954c89ffcf96f36c6fdde2db5ed44896a6a0251ca0989b2463094` |
| grid-W history rank 1 | exact variables except timestamp | `40438539e85d32c3c1c26f03ce51a2f0522a07e77fa432e63ee20ba43321b5e3` |
| `ocean.output` | exact after dump-notice removal | `7590d8ff2afc830e73baa3238e98a87b7be44235179ce3f0c008ebab0aa7dea6` |

The four-restart inventory is itself a resolved-option fact: no iceberg
restart or trajectory is expected when `ln_icebergs=F`; an asymmetric
optional-file plant fails through `validate_identity`.

## 2. One-variable variant effect

The deck comparison and resolved output confirm that only `ln_icebergs`
changed.  Source inspection confirms the inactive call and write cone:
`src/OCE/SBC/sbcmod.F90:457-484` guards iceberg stress and `icb_stp`, and
`src/OCE/ICB/icbthm.F90:286-295` writes only iceberg melt to `emp` and latent
heat to `qns`.

Measurement confirms every recorded pre-effect operand exactly:

| record | exact scope |
|---|---|
| ocean step entry kt=1 | complete 14,288,048 bytes |
| SI3 Prather kt=1 stages 0 and 1 | complete 94,300,860 bytes each |
| SI3 bulk | all 3 kt=1 frames, 956-byte prefix |
| SI3 exchange | kt=1 frame, 9,450,856-byte prefix |
| SI3 thermodynamics | all 28 kt=1 frames, 68,614,920-byte prefix |
| SI3 ZDF | all 5 kt=1 frames, 1,281,080-byte prefix |

Of 90 common streams, 8 remain complete-file exact and 82 differ only after
the first possible iceberg effect has entered the coupled state.  Those later
files are diagnostic, not a pre-effect identity claim.  The new variant-only
post-`sbc` stream has exact-zero `utau_icb`, `vtau_icb`, calving, calving heat,
floating melt, and stored iceberg heat.  Together with the one-variable deck
and the source write cone, this confirms the preregistered removal: iceberg
stress slots are inactive and the only removed physical source channels are
iceberg `emp/qns`.  A same-boundary shipped post-`sbc` file does not exist, so
no stronger direct field-by-field shipped comparison is claimed.

## 3. Controls

The accepted oracle gate exits 0.  Ten inherited schema/identity plants, six
surface-stream plants, and the cross-run pre-effect plant all return
`PASS_NONZERO`.  This includes the binding restart-byte plant through
`validate_identity`, optional restart-inventory asymmetry, malformed magic,
wrong level, truncation, trailing data, NaN, one ULP, wrong derived count, and
nonzero inactive iceberg data.

During development, the first inventory-asymmetry plant mistakenly wrote
through a temporary symlink and truncated the preserved shipped
`ORCA2_00000010_restart_icb_0000.nc`.  It was detected immediately.  That one
file was restored from the accepted byte-identical uninstrumented control to
its documented 1,392,672-byte SHA-256
`1dc69243e465964a99217c5729df5f21ee23a583507e7cbdb274502700eb3926`.
The plant now unlinks before materializing its mutation.  A complete shipped
Phase-1 regression passes; its retained JSON SHA-256 is
`cd34c630d67aa6d19bc9fdb163d581be0d347f674b474ddde8fb2414f7f5d93f`.
The failed development outputs remain preserved and are explicitly retracted.

## 4. Icebergs-off card and entry gate

`NEMOTestcaseCard` now has explicit coupled-deck metadata.  The ORCA2 card
sets `icebergs_enabled=False`, `iceberg_inputs=()`, and removes
`iceberg_state` from unresolved features because the operator is inactive,
not waived.  The structural validator rejects an enabled or nonempty iceberg
arm.  Other cards retain tri-state `None` metadata; no default physics choice
was imposed on them.

The production-JIT CPU gate explicitly installs
`PrecisionPolicy.fp64(transcendentals="libm")` and rejects disabled JIT.  It
passes against the VARIANT entry record:

| field | unequal / n | level |
|---|---:|---|
| T | 0 / 399600 | Kbb=1, 30 active records |
| S | 0 / 399600 | Kbb=1, 30 active records |
| u | 0 / 399600 | Kbb=1, 30 active records |
| v | 0 / 399600 | Kbb=1, 30 active records |
| ssh | 0 / 13320 | Kbb=1 |

The retained JSON is 1,555 bytes, SHA-256
`aad0f9a637484f5f05debf54e5e58e1eeed966d8a71eda557770bf03d610a202`.
All eleven entry plants, including the new iceberg-option plant, exit 1.
Focused unit tests pass `2 / 2`.

## 5. Coverage and time-level registry at this stop

| family | level/boundary | disposition |
|---|---|---|
| tripolar geometry, masks, e3 fields, `ff_t`, separate `ff_f` | static | VERIFIED |
| ocean T/S/u/v/ssh | kt=1 `Kbb=1` | VERIFIED `0/n` on rank-0-owned `90x148x30` |
| full SI3 jpl=5 layered/Prather producer | ice level before odd kt=1 step | `UNMEASURED_PENDING_ICE_MERGE` |
| SI3-to-ocean stress, heat, freshwater, salt and drag fields | after SI3, before ocean consumers | `ORACLE_SUPPLIED`; producer not certified |
| iceberg fields | inactive at `ln_icebergs=F` | VERIFIED zero/no inputs/no restart |
| final `utau/vtau/qsr/qns/emp/sfx`, runoff and FWB carry | post-`sbc`, pre-EOS | VERIFIED oracle records; card consumer pending O1 |
| mapped CORE/NCAR atmospheric fields | post-`fld_read`, kt=1 current `fnow` | MISSING in accepted set; acquisition prepared |
| open-ocean NCAR outputs | post-`blk_oce_2`, pre-SI3 | MISSING in accepted set; acquisition prepared |
| EOS/HPG and TKE/EVD/IWM entry | `Kbb=1`, before stage 1 | NOT ENTERED; shared debt owner `GYRE_OWNER`, IWM file owner `ORCA2_OWNER` |
| 65-substep external mode and north fold | stage-1 slow forcing | NOT ENTERED; shared owner `GYRE_OWNER`, fold owner `ORCA2_OWNER` |
| stage-1 transports | `(Kbb,Kmm,Krhs,Kaa)=(1,1,3,3)` | NOT ENTERED; `GYRE_OWNER` |
| FCT tracer advection/surface forcing | stage 1 | NOT ENTERED; shared FCT `GYRE_OWNER`, ORCA2 forcing `ORCA2_OWNER` |
| RGB QSR, BBL, geothermal, lateral/ZDF closure | stage 3 | NOT ENTERED; owner split retained from preregistration |

All oracle-supplied exchange arrays are read at their recorded current/before
levels; none is relabelled as a legoESM SI3 output.  The card still fails
closed on seven selected unmeasured features: `nemo_fld_read_forcing`, staged
GM/EIV, linear implicit bottom drag, IWM, spatial viscosity, freshwater carry,
and SI3 layered/Prather state.

## 6. O1 stop and prepared replacement run

The first unresolved ordered boundary is **O1 / MISSING_ORACLE_OPERAND**.
The 91-stream set records final post-`sbc` fields and three sampled SI3 bulk
columns, but not the full mapped atmospheric input or the pre-SI3 open-ocean
bulk result.  Scoring the final field alone would make ownership ambiguous
between ORCA2 `fld_read`/rotation and the Lane-3b NCAR arithmetic.  No
implementation was attempted across that ambiguity.

The copied configuration's `MY_SRC/sbcblk.F90:559-560,631-635,672-715` now
adds one rank-0, kt=1-only WRITE-only stream.  Frame 0 writes the nine selected
`sf%fnow` fields immediately after `fld_read`; frame 1 writes 20 processed
operands/intermediates/open-ocean outputs immediately after `blk_oce_2` and
before SI3.  All model fields appear only in `WRITE` lists; only writer-local
filenames, magic, units, and status are assigned.

Schema `NEMO_L4_BLKIO_1` carries `(version,kt,kind,nx,ny,nfields,halo,bits)`.
The two exact headers are `(1,1,0,90,148,9,0,64)` and
`(1,1,1,90,148,20,0,64)`.  The derived record is 3,090,336 bytes.  The frozen
acquisition gate requires the full 92-record inventory, revalidates the 90
legacy records and surface record, calls dynamic ordinary-output identity,
and schema-walks both new frames.  Synthetic count and one-ULP plants both
return `PASS_NONZERO`.  The retained 1,261-byte synthetic-schema log has
SHA-256
`3256b2a69675db6c1708aade22d7896ed6914709a2c9064b1df4286de541a3e9`.
The first local evidence command named a nonexistent `.venv`; its empty
failed log is preserved as `o1_synthetic_schema.failed_missing_venv.log` and
is not evidence.

The scalar-math rebuild retained `-fdefault-real-8 -O3
-funroll-all-loops -fno-tree-vectorize`; zero `_ZGV*` symbols were found.
The nonfatal local Conda entry-point warning is retained in the successful
build log.

| artifact | bytes | SHA-256 |
|---|---:|---|
| `build/nemo_ORCA2_OMIP_L4_phase2d_o1.exe` | 54,920,400 | `c9c25e1aeb17d88f8b41e6a3f263d0684678067d3aba2931913effe3da55503b` |
| build log | 10,867 | `c074be7470f1435c26f3475e17eeab75691ab8dd97429da63560ce347bb27cb3` |
| 14-file `MY_SRC` manifest | 1,092 | `7009f39fe854f89173bceba26fcf4bb8a8034cb228650e63a8586905a52d49a1` |
| `_ZGV*` census | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| copied `MY_SRC/sbcblk.F90` | source | `75177950f32faa52bb3e59111b6adec6b4b07cf8f6044720a8873810671a0eed` |

Run exactly:

```text
/data/abyssal/dbalwada/nemo-testcases-l4/runs/
  variant_icebergs_off_o1_instrumented_10step_np2/run.sh
```

The directory has 19 regular deck copies, 40 immutable absolute input
symlinks, one absolute binary symlink, two manifests, and the launcher.  Its
63-item prepared census SHA-256 is
`512ba04697f897991005542a5ae9115419f48b902639269cb065152219a3be16`;
the launcher is
`88590198b3222fc8c76eb5c68404f24c19cc3c98abf0978ecdc0f1bd3c6d768c`.
It hash-checks everything, refuses stale output, sets one thread per library,
uses Bash `time`, and invokes `mpirun -np 2 --oversubscribe ./nemo`.

## 7. ASKED / UNASKED

| item | status | disposition |
|---|---|---|
| accept and pin the three icebergs-off runs | ASKED | PASS; `VARIANT` is the comparison oracle |
| compare instruments to uninstrumented control per file | ASKED | PASS; dynamic 4 restart / 8 history counts |
| bind card to icebergs off with no inputs | ASKED | PASS; metadata, guard, plant, entry `0/n` |
| ocean ladder in source order and stop at first decision/gap | ASKED | stopped at O1 missing operand before arithmetic |
| oracle-supplied SI3 exchange | ASKED | registered input substitution; producer remains unmeasured |
| prepare any required NEMO rerun and stop | ASKED | one directory above; agent did not invoke MPI |
| add full mapped-input/pre-SI3 bulk writer | UNASKED necessary instrumentation | source-required to assign O1 ownership; WRITE-only |
| repair dynamic inactive-component identity inventory | UNASKED gate correction | same equality rule, now fail-closed for resolved option |
| restore accidentally modified preserved symlink target | UNASKED corrective action | disclosed above; exact documented bytes restored and regression passed |
| implement `fld_read`, bulk, EOS/HPG, external mode, transport or SI3 now | UNASKED and forbidden by stop | not done |

## 8. Resume condition

After the user-shell replacement run, the acquisition gate must pass: 92
records, derived O1 headers/sizes, the existing surface digest, frozen
90-record manifest, dynamic ordinary-output identity against the accepted
variant control, and both O1 plants.  Only then may O1 implement the shared
source-pinned `fld_read` map/rotation and compare it separately from the
already shared NCAR bulk leaf.  The first over-bar result must retain its
registered `ORCA2_OWNER` or `LANE3B_OWNER` label.  Shared ocean-operator debt
later in the ladder is handed to `GYRE_OWNER`; it is not fixed in Lane 4.
