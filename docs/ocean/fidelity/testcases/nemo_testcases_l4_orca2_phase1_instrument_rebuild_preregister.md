# NEMO testcase Lane 4 ORCA2 — full-instrument rebuild preregistration

Date frozen: 2026-09-04 (America/New_York)
Scope: corrective Phase-1 instrument build and one replacement ten-step oracle
record run only. No legoESM code or numerics are authorized.

## Why the first record run is retracted

The user-shell executions completed successfully:

| run | terminal step | MPI result | wall/user/sys (s) | complete-file manifest SHA-256 |
|---|---:|---|---:|---|
| `instrumented_10step_np2` | 10 | `MPIRUN_RC=0`, `RUN DONE` | 11.724/16.918/4.115 | `b633486295afca2c8f6930258cdcb4a9dc86d0cf0bc0eb0bcda372814505a4b6` |
| `uninstrumented_30day_np2` | 240 | `MPIRUN_RC=0`, `RUN DONE` | 188.768/363.542/10.378 | `efe02aa764feb07ba045ee25c0bd304240b7780e39c01376dec741b4520413f9` |

The first instrumented run emitted only 63 `oracle_*.bin` files:

- ten `oracle_step_entry`, ten `oracle_bt_frames`, and 30 `oracle_stage`;
- one each of `oracle_rhs`, `oracle_rgb_chl`, and `oracle_zdf_entry`;
- ten `oracle_si3_prather` entry/exit frames for the five odd ice steps.

It emitted none of the registered Lane-2 detailed EOS, HPG, barotropic
substep, transport, tracer, QSR, or WZV records and none of the registered
Lane-3 SI3 bulk, exchange, ZDF-input, reassociation, or thermodynamic records.
This fails the frozen inventory in
`nemo_testcases_l4_orca2_phase1_preregister.md`; it cannot be waived after
measurement.

The build audit found that `cp -p` had preserved the old mtimes on copied
Lane-2/Lane-3 sources.  At the first build, the generated files under
`BLD/ppsrc/nemo` were newer than 11 corresponding config-local sources, and
FCM did not preprocess the replacements.  Source writer markers versus
preprocessed writer markers were nonzero/zero for:

```text
dynhpg 1/0       dynspg_ts 9/0   dynvor 1/0      eosbn2 2/0
icesbc 1/0       icestp 2/0       icethd 4/0      sbcblk 1/0
stp2d 2/0        stprk3_stg 11/0 traadv 1/0
```

Only the locally edited `stprk3`, `traqsr`, and `icedyn_adv_pra` replacements
had entered the first binary.  A `strings` census of that retained binary
independently contains only the seven record-name families actually observed.
This is a build-dependency failure, not a runtime inactive-arm result.

The incomplete run nevertheless provides a useful non-perturbation witness:
all six terminal ocean, iceberg, and SI3 restart shards are byte-identical to
the uninstrumented ten-step control.  It does not prove the missing writers
are WRITE-only.  The eight native-IOM history shards are not byte-identical:
their embedded `TimeStamp` attributes differ between executions.  The three
timing NetCDF files also differ by construction.  These provenance/timing
differences are recorded now; the final receipt will report the corrective
run's ordinary-output comparison per file rather than claiming blanket byte
identity.

## Correction frozen before measurement

All 14 config-local `MY_SRC/*.F90` files were touched to invalidate FCM's
incremental dependency cache.  No source bytes changed: the sorted SHA-256
manifest before and after the timestamp update has the same SHA-256,
`59cba6adb1a40607e09b4c03ac57af1ea50e27d4c3ecb67fa7ac7d543752d08a`.
The copied configuration was rebuilt with the same conda `nemo-build`
toolchain and `arch-conda-scalarmath.fcm`, including compiler-wide
`-fno-tree-vectorize`.

```text
binary  /data/abyssal/dbalwada/nemo-testcases-l4/build/nemo_ORCA2_OMIP_L4_instrumented_full.exe
bytes   54904000
SHA256  40e9ac050297b02443e0b897403eb5cf767efc197b2ab046de5b17ec4d6571ee
build   /data/abyssal/dbalwada/nemo-testcases-l4/build/build_ORCA2_OMIP_L4_instrumented_full.log
SHA256  f9006bb6afc44350e6ee4f8f97d1a0e5d9070e9d6d108eed1f42eadaa903e094
_ZGV*   zero dynamic symbols; empty census SHA256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

The rebuilt preprocessed sources now contain the registered writer markers.
A binary `strings` census contains the slow-forcing, barotropic substep and
operand, EOS/HPG, RK transport/tracer/QSR/WZV, SI3 bulk/exchange/ZDF/
reassociation/thermodynamic, Prather, RGB, stage, RHS, and TKE/ZDF record-name
families.  This is a static build witness only; emitted files and schemas must
still pass the post-run gate.

## Corrective run deck and launcher

The only authorized corrective execution is:

```text
/data/abyssal/dbalwada/nemo-testcases-l4/runs/instrumented_full_10step_np2
```

It is the same registered CPU-only `jpni=2`, `jpnj=1` arm, because the shipped
iceberg code rejects one processor in x.  The 19 XML/namelist deck members are
regular-file copies.  The 40 immutable input payloads and exact binary are
absolute symlinks.  `namelist_cfg` is byte-identical to the first instrumented
ten-step deck: only `nn_itend=10`, `nn_stock=10`, and `nn_istate=1` differ from
the shipped scientific deck.

```text
namelist_cfg       62f4746cf3846254c18af73bcde53e42f7d4cb9ebcdaf5f63cc9ed5316a9f8f4
deck manifest      e1974d9db5974f54d451fe1112b8f22a404516adcc7cbcf0b035fbe1cccba70b
input manifest     3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5
run.sh             c88d5bc7fad7974960bcdef838b6ab4250bb35abf98adef88304d155ef39c512
prepared census    1aeb1f8d761a561c94b7447d5e1f84e6ef503b6e652a075bae0caca10802563e
```

`run.sh` verifies its directory, clean output state, binary, namelist, deck,
and inputs; fixes the thread counts; uses Bash's `time` keyword; and runs
exactly `mpirun -np 2 --oversubscribe ./nemo`.  Detailed writers retain NEMO's
`lwp` convention: rank 0 writes its local `94 x 152 x 31` subdomain including
two-cell halos.  The two ordinary restart shards provide the global state;
the receipt will register the halo stripping and rank mapping.

## Frozen outcome

The correction is **CONFIRMED** only if the user-shell run exits zero, prints
`RUN DONE`, reaches step 10, writes all six restart shards, emits every
registered record family exactly as its writer schema requires, and the
ordinary-output comparison is reported per file.  The six state restart
shards must be byte-identical to the uninstrumented control.  The record gate
and all registered planted controls must pass before the Phase-1 receipt is
written.

Missing or extra record files, malformed headers, payload-size errors,
non-finite defined-domain values, a restart byte mismatch, or any changed
resolved selector **REFUTES** the correction.  No second post-hoc inventory or
mixed-binary (Frankenstein) record set is permitted.

## ASKED / UNASKED disposition

| Item | Disposition |
|---|---|
| user-shell MPI execution | ASKED; the agent only prepares and verifies |
| Bash `time` substitution | ASKED after `/usr/bin/time` was found absent; exact scripts re-hashed |
| removal of empty exit-127 launcher outputs | ASKED user action, limited to the failed launcher's own files |
| retain the incomplete 63-record run | standing no-delete rule; evidence is hash-pinned |
| touch 14 `MY_SRC` mtimes | UNASKED operational cache invalidation; before/after content hashes are identical |
| rebuild and one corrective run | UNASKED enabling correction required by the frozen inventory |
| physics, CPP keys, namelists, MPI layout, input payloads | unchanged; no new choice |
| any legoESM work | out of scope and forbidden |

Phase 1 remains open at this stop.  The final executed-arm audit, schema and
planted-control results, coverage/time-level registers, final manifest, and
independent-review receipt wait for this one corrective user-shell run.
