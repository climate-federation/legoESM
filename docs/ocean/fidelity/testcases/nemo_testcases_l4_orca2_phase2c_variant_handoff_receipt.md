# NEMO testcase Lane 4 — ORCA2 Phase-2c variant handoff receipt

Date: 2026-09-06

Parent: `1b83a3bcbeee08c299303067f5ca2f5c8bda69a1`

Preregistration commit: `82974b40d`

Coriolis repair commit: `af54d11ef`

Run preparation commit: `ade8a9ffd`

Result: **STOP — three icebergs-off VARIANT oracle runs prepared; no MPI run**

This handoff implements User Decision 7, closes the tripolar Coriolis review
blocker, rebuilds the configuration-local WRITE-only instrument, and prepares
all three requested user-shell runs.  It does not execute any run and does not
enter a legoESM ocean or SI3 numerical operator.  Shipped NEMO source and deck
files are untouched.

## 1. Frozen one-variable comparison variant

The comparison oracle is the shipped `ORCA2_ICE_PISCES` deck copied as
`ORCA2_OMIP_L4`, with exactly one scientific configuration change:

| resolved option | `SHIPPED_DECK_RECORD` | `VARIANT` oracle |
|---|---:|---:|
| `ln_icebergs` | `.true.` | `.false.` |

`ln_rnf_icb=.false.`, `nn_ice=2`, `nn_fsbc=2`, all other resolved options,
and every input remain unchanged.  The ten-step pair has the already accepted
run controls `nn_itend=10`, `nn_stock=10`, `nn_istate=1`; the 30-day arm has
`nn_itend=240`, `nn_stock=240`, `nn_istate=0` at `rn_Dt=10800 s`.  Direct
deck-manifest comparison against the corresponding prior arm changes only
`namelist_cfg`; its content diff changes only the one `ln_icebergs` value.

All existing Phase-1 and Phase-2b artifacts are retained and labelled
**SHIPPED_DECK_RECORD**.  They are superseded as comparison operands, not
deleted.  The former icebergs-on Phase-2b launcher is retained but now exits
69 before MPI; its SHA-256 is
`895684724ca23e424f6c2991186f522f201f3b94697980b044b8087cc0761d3d`.
Its preserved pre-supersession census remains in the Phase-2b record; a new
63-item current census records the guarded launcher and has SHA-256
`72a4891487ade34553f0e3631c1fb8dab46d1e577251e50982ae611af38db649`.

### Source-ordered expected change

- `src/OCE/ICB/icbini.F90:73-80` reads the option and returns before allocating
  iceberg stress and grid storage when it is false.
- `src/OCE/SBC/sbcmod.F90:457-470` computes `utau_icb,vtau_icb` only inside
  the active-iceberg guard.
- `src/OCE/SBC/sbcmod.F90:474-480` still calls SI3, while line 484 is the
  guarded `icb_stp` call that becomes inactive.
- `src/OCE/ICB/icbthm.F90:286-295` shows the sole iceberg thermodynamic
  write-back to ocean: floating melt is subtracted from `emp`, and calving
  heat is added to `qns`.

Therefore every operand before the first possible `icb_stp` effect is frozen
to byte identity.  At the post-`sbc` boundary the only allowed physical
differences are the removed iceberg contributions to `emp` and `qns`;
`utau_icb`, `vtau_icb`, and the four inactive iceberg-grid diagnostic slots
must be zero.  This is a preregistered expectation, not a measurement yet.

## 2. Tripolar Coriolis review blocker closed

The earlier loader wrongly replaced generic v-face `f_v` by NEMO's native
F-point `ff_f` whenever a domain file supplied the latter.  The repair:

1. restores the pre-change `f_v` construction byte-for-byte from analytic
   T-point Coriolis plus the historical adjacent-row average;
2. stores native `ff_f` as a separate optional geometry field;
3. carries that field through the shared halo, MPI-slice, and fidelity bridge
   plumbing; and
4. selects it only in the NEMO EEN/ENE vorticity path.  Generic vertex and
   face-Coriolis consumers continue to read `f_v`.

This matches NEMO's staggering: `src/OCE/DYN/dynvor.F90:350-378` uses `ff_t`
in the T-point energy arm, while the ENE path uses `ff_f` at lines 449-490 and
the EEN path uses it at lines 748-783.  The GYRE fixture pins its old `f_v`
bytes while separately pinning `ff_f`; the ORCA2 entry gate independently
pins the legacy `f_v` expression and literal domain-file `ff_f`.

The repaired CPU/fp64 entry gate is byte-identical to its pre-repair artifact:
both are 1,538 bytes with SHA-256
`d58e1f336eaa7910aa3a83a8d0a0bb7c5b032d7a3039f8cf0b60e8917b8d7703`.
Its `T`, `S`, `u`, `v`, and `ssh` entry comparisons remain `0 / n`.  The
binding `coriolis_swap` plant exits nonzero (`FAIL: geometry mismatch:
ff_f_grid`); retained log SHA-256 is
`84fccf7d19dec25931c39584d667cf8836f8952fb95c0fb2e2a0397f9330269a`.
The focused recipe suite passes `28 / 28`; its 521-byte log has SHA-256
`55a3c3b4c29e698629b29cee539e384afe0f2fb16f71b85ff9079f2bba96b80d`.

This is geometry/source-field separation, not a new numerical scheme.  No
measured shared-operator debt was altered.

## 3. Icebergs-off-safe WRITE-only instrument

The accepted post-`sbc` writer is in the copied configuration only:

```text
/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/
  cfgs/ORCA2_OMIP_L4/MY_SRC/stprk3.F90
SHA-256 045ca27f0679bdbf4c11cd150e3ccd56ab0a84ac8b0c6beac733de09b3eb757f
```

Because `icbini` returns before allocation when icebergs are disabled, the
writer must not dereference those arrays.  It imports the resolved flag and,
only in the inactive arm, fills a writer-local `zicb_zero(jpi,jpj)` scratch
array and writes it in the six frozen diagnostic slots.  The active arm writes
the original arrays.  `IF(.NOT.lwp) RETURN` retains one rank-0 writer; no model
array is assigned and no arithmetic used by NEMO is changed.

The record remains `oracle_ocean_surface_input_kt00000001.bin`, magic
`NEMO_L4_SBCIN_1`, 13 native four-byte header integers, and 35 binary64
fields/families.  Counts are derived from the write inventory: 20 full
`94x152` arrays, 12 reduced `90x148` arrays, one halo-1 `92x150` array, and
two reduced `90x148x2` families.  The derived payload is 512,680 values and
the exact record size is 4,101,508 bytes.

The committed gate now verifies the one-variable variant namelist, labels the
result `VARIANT`, walks every field, and requires all six inactive slots to be
exact zero.  A synthetic schema-valid record passed all 35 fields.  Six
binding mutations each exited nonzero: bad magic, bad derived count,
truncation, NaN, one-ulp active-field change, and nonzero inactive iceberg
data.  The 274-byte retained gate log has SHA-256
`a9b602b34b9a3893cc6d1feb0516cd58e57c91e214632d3bef7c8e782320813c`.
The synthetic record itself is test evidence only, not an oracle record.

## 4. Scalar-math executables

The instrumented executable was rebuilt with:

```text
PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:... \
  ./makenemo -n ORCA2_OMIP_L4 -m conda-scalarmath -j 8
```

The compile command retains `-fdefault-real-8 -O3 -funroll-all-loops
-fno-tree-vectorize`.  `nm -D` reports zero `_ZGV*` symbols.

| artifact | bytes | SHA-256 |
|---|---:|---|
| instrumented variant executable | 54,904,016 | `b31fc33edd3109a41640f9fb59f915d90a52f1f0509fde7c33254cf46b896f28` |
| instrumented build log | 4,642 | `9777ab99e03c848eb3390ecf8cc362c17c82a18edc37d5b2d5b7e88c79dcf73d` |
| instrumented `MY_SRC` manifest | 2,100 | `54b3bff73af525906e1117cf26af23c75609de4307a44c169c9156ff001edfba` |
| `_ZGV*` census (empty) | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| accepted uninstrumented scalar-math executable | 54,746,696 | `c4907e476cf3969052b44c5c7fa966f3dac493e8cfb563f6554c8f3a27186343` |

## 5. Prepared runs

Each directory contains 19 regular copies of deck files, 40 absolute
read-only symlinks to the immutable unpacked input archive, one absolute
binary symlink, its manifests, and a self-contained executable `run.sh`.
There are 63 prepared entries in each directory.  Every launcher refuses a
wrong directory, a stale output/record/restart, or a binary, deck, or input
hash mismatch before invoking MPI.  It sets one thread per numerical library,
uses Bash's `time` keyword, and runs
`mpirun -np 2 --oversubscribe ./nemo`.  The two-rank `jpni=2,jpnj=1` protocol
is retained so the variant pair and the rank-0 record map are directly
comparable with the accepted control.

| arm | run directory | binary | launcher SHA-256 | 63-item census SHA-256 |
|---|---|---|---|---|
| instrumented 10-step | `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_instrumented_10step_np2` | instrumented variant | `e9f16a444616e14faefdb633adbe4f68981e4c04599829585e696935f9d1e63e` | `0229bc1e9b92427e6f6b107962453fa3eac6f33dd5b11cf008b15acf7c21eb5a` |
| uninstrumented 10-step identity control | `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_uninstrumented_10step_np2` | accepted uninstrumented | `b647d4a3c1f302f26d44bd078e1787e879828c8ea794807dbd96920e69b519a5` | `28bb86dfd516ea826b2e308a5889eda61e0fb27c20af9f9736e1c080d7dd253e` |
| uninstrumented 30-day | `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_uninstrumented_30day_np2` | accepted uninstrumented | `af46ce7583c4d077a4db4f45ba6dd83bd229819bfb17d7aac7833bdbb471022a` | `60493aa81a7c8188dba9821b9b877238a9f82285e92d49d77f5e4404e7ccc05e` |

Both ten-step deck manifests have SHA-256
`e2cb4c552360491fa9dcea0649661e5f44a9d972769d40a4ecfc2aa70a097059`;
the 30-day deck manifest is
`2ce5f93ae9d6355e1d4d94e2f5e4576f98eec3ec0e61f3abb69d8c149028683a`.
All three input manifests are
`3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5`.
All manifest checks pass, launcher copies equal the committed recipes, and no
run output exists at handoff.

The user shell should run `./run.sh` from each directory above, one at a time.
No icebergs-on directory should be run.

## 6. Frozen validation after execution

On resume, the lane will first validate the runs, before any legoESM stage:

1. require zero launcher status, `RUN DONE`, step 10/10/240, and clean
   `ocean.output` for the three arms;
2. SHA-256 every output, restart shard, and all 91 instrument streams;
3. walk the frozen 90-stream schema plus the post-`sbc` derived schema,
   including magic, monotone `kt`, exact sizes, finite payload, and exact-zero
   inactive iceberg slots;
4. run byte identity for every ordinary output and all dynamically discovered
   restart shards in the ten-step pair, excluding only registered
   timing/timestamp metadata;
5. require every pre-`icb_stp` operand to equal the `SHIPPED_DECK_RECORD` and
   restrict post-`sbc` differences to the preregistered iceberg field set; and
6. run the inherited nine Phase-1 plants and all six surface-record plants,
   each with a required nonzero exit.

Only after those gates pass does the ocean ladder resume with variant records.

## 7. Ownership and next boundary

SI3 dynamics and thermodynamics remain
`UNMEASURED_PENDING_ICE_MERGE`; until the certified ice-lane code lands, their
post-SI3 exchange fields are `ORACLE_SUPPLIED` inputs.  Iceberg operators are
inactive by Decision 7.  No ice operator is claimed by this lane.

The subsequent ocean walk follows NEMO execution order using the shared
canonical WS-RK3 identity.  Lane 4 may implement only ORCA2-specific north
fold, `fld_read` forcing, RGB chlorophyll, runoff, geothermal, IWM, BBL, and
icebergs-off setup work.  Any first over-bar result in shared EOS/HPG,
transports, ZDF/TKE, or external mode is registered with its exact boundary
and a `GYRE_OWNER` label, then handed to the GYRE lane without a Lane-4 fix.

## 8. ASKED / UNASKED

| item | status | disposition |
|---|---|---|
| icebergs-off comparison oracle, no other deck change | ASKED, Decision 7 | frozen one-variable VARIANT |
| preserve shipped-deck records | ASKED | labelled `SHIPPED_DECK_RECORD`; superseded for comparison only |
| do not run prior icebergs-on directory | ASKED | launcher exits 69; directory retained |
| prepare all three MPI runs, then stop | ASKED | three hash-guarded directories above; no MPI executed |
| keep full instruments plus post-`sbc` dump | ASKED | 90 legacy streams plus one registered surface stream |
| repair `f_v`/`ff_f` separation and prove entry unchanged | ASKED review blocker | repaired, tested, planted, byte-identical entry artifact |
| enforce Lane-4 operator ownership | ASKED standing | no shared numerical debt fixed |
| rank-0 record | ASKED inherited | `lwp` writer; `jpni=2,jpnj=1` layout retained |
| local-zero inactive diagnostic slots | UNASKED enabling instrumentation | avoids unallocated reads; WRITE-only; schema and plant bind it |
| rebuild instrumented executable | UNASKED operational consequence | scalar-math build; uninstrumented executable unchanged |
| deck copies plus immutable absolute input/binary symlinks | UNASKED inherited packaging | disclosed and fully hash-guarded |
| any ocean-stage or SI3 numerical work | UNASKED and outside this stop | not done |

No unasked scientific configuration choice was made.  External reviewer CLIs
could not complete under the restricted network, so this implementation is
explicitly **UNREVIEWED** and this receipt is the independent-review handoff.
