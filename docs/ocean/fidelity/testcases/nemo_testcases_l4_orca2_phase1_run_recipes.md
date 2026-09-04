# NEMO testcase Lane 4 ORCA2 — Phase-1 run recipes

Status at the corrective stop: all three originally prepared executions have
run.  The uninstrumented ten-step control and 30-day reference are valid, but
the first instrumented build is **retracted as the record oracle**.  Preserved
source timestamps caused FCM to reuse stale preprocessed sources for 11 of 14
config-local overrides, so that run emitted only 63 of the preregistered
records.  The run is retained as evidence; it is not silently replaced.  A
corrected full instrumented build and a fresh ten-step directory are registered
in `nemo_testcases_l4_orca2_phase1_instrument_rebuild_preregister.md`.

This remains an intermediate Phase-1 hand-off, not the oracle receipt and not
a Phase-2 fidelity claim.

## Verified two-rank constraint and control

The preregistered one-rank attempt is retained at
`/data/abyssal/dbalwada/nemo-testcases-l4/runs/uninstrumented_10step`.
It exited nonzero because the shipped iceberg path stops at
`src/OCE/ICB/icbini.F90:112` when `lk_mpp .AND. jpni == 1`:

```text
icbinit: having ONE processor in x currently does not work
```

This is an executed constraint of the oracle, not an analyst choice. NEMO's
automatic smallest working decomposition for two ranks is `jpni=2`, `jpnj=1`,
`jpnij=2`. The global domain is `184 x 152 x 31`; rank 0 reports local
`94 x 152 x 31`, `nn_hls=2`, `nimpp=1`, and `njmpp=1`. The completed control
is:

```text
/data/abyssal/dbalwada/nemo-testcases-l4/runs/uninstrumented_10step_np2
```

The user's shell ran the prepared `mpirun -np 2 --oversubscribe ./nemo`
recipe. `run.user.time.log` records wall/user/sys `14.138/21.205/4.557 s`,
`MPIRUN_RC=0`, and `RUN DONE`; `time.step` is the formatted value `10`.
No error/abort/non-finite diagnostic is present. `ocean.output` records the
terminal ocean restart at step 10 and the staggered SI3 restart at ice step 9.
All six restart shards exist:

| restart shard | bytes | SHA-256 |
|---|---:|---|
| `ORCA2_00000010_restart_0000.nc` | 28,454,360 | `9ba6054040f11ea38a6637c4e6e7e26261359540fbf2d5be2cf63b78857dd508` |
| `ORCA2_00000010_restart_0001.nc` | 28,454,360 | `d06e099765da5e6eba9f2f6ecb8aa66b20fb034fa506456daeb8da0cb8fea1ba` |
| `ORCA2_00000010_restart_icb_0000.nc` | 1,392,672 | `1dc69243e465964a99217c5729df5f21ee23a583507e7cbdb274502700eb3926` |
| `ORCA2_00000010_restart_icb_0001.nc` | 1,386,328 | `5b183bb7e79003a2d36325247246d9265b8030e21ebc4ffd3507ab6a9450e5b1` |
| `ORCA2_00000010_restart_ice_0000.nc` | 104,228,128 | `5266345352fecb609820d3e62f495019d2bd05d4e0bdb40d0a327bd6c9dafd20` |
| `ORCA2_00000010_restart_ice_0001.nc` | 104,228,128 | `2634bd96f623fe476c3c389ea232b03583793686dde8781120427320676b68e2` |

The complete 97-file post-run census, including ordinary NetCDF output,
resolved namelists, deck, linked input payloads, logs, and executable, is
`manifests/uninstrumented_10step_np2_all_files.sha256` under the Lane-4 data
root. Its SHA-256 is
`924867d6e066d450715a0997ecfd1264005b02abe8f4d05228cf29870fff03a7`.

## Executed selector witness

The completed control confirms the registered physical path:

- `ocean.output:222` selects third-order Runge--Kutta with `rDt=10800 s`,
  and `:1772`, `:1787`, and `:1819` execute stages 1, 2, and 3. This is the
  `key_RK3` WS-RK3 path, not MLF.
- `ocean.output:355` reports QCO variable volume active.
- `ocean.output:675` reports SI3 and `:930` selects Prather advection;
  `:972` selects TKE closure.
- `ocean.output:1192-1197` selects RGB light with file chlorophyll and a
  vertical chlorophyll profile, while `ln_qsr_bio=F`.
- `ocean.output:1367` selects the split-explicit free surface.
- No `trc_init`, `trc_stp`, PISCES, or other TOP execution call appears.

`output.namelist.dyn:98` prints the inherited reference value `LN_TOP=T`.
This file is emitted before `src/OCE/DOM/domain.F90:353-356` applies the
absence of `key_top` and sets `ln_top=.false.`. The runtime call graph is
compile-guarded at `src/OCE/nemogcm.F90:496-499` and
`src/OCE/stprk3_stg.F90:502-506`; the absence of their messages is the direct
execution witness. Thus the superficially true value in the early namelist
echo is not an executed TOP selector and does not refute the oracle definition.

## Retracted first instrumented build

The first instrumented binary used the same copied configuration, key set, conda
`nemo-build` compiler/link environment, and `conda-scalarmath` arch as the
control. Only config-local `MY_SRC` writers differ. The final executable is:

```text
/data/abyssal/dbalwada/nemo-testcases-l4/build/nemo_ORCA2_OMIP_L4_instrumented.exe
bytes   54763256
SHA256  fc4b48a425f2064209e5a683f0e14f06bc561795cacae243941b5da8d90baa31
_ZGV*   0 dynamic symbols
keys    key_si3 key_qco key_vco_1d3d key_RK3
```

`instrumented_MY_SRC.sha256` pins all 14 override sources. The build log
retains both the first rejected RGB-writer compile and the successful rebuild:
under QCO, `gdepw(i,j,k,Kmm)` is a CPP scalar-index expression, not an
addressable rank-3 object. The corrected WRITE-only frame stores its exact
operands `gdepw_1d` and `r3t(:,:,Kmm)`; it performs no model assignment or
replacement arithmetic. The final build log SHA-256 is
`55232c9d145aaeacc3c636cb29a688ed32691c4a4042a49130ceb2e6d934d826`.

The intended Lane-2 overrides are `dynhpg`, `dynspg_ts`, `dynvor`, `eosbn2`, `stp2d`,
`stprk3`, `stprk3_stg`, and `traadv`. The Lane-3 overrides are `sbcblk`,
`icesbc`, `icestp`, and `icethd`. Lane 4 adds frames in `traqsr` for the RGB
chlorophyll read and in `icedyn_adv_pra` for every selected Prather moment,
and extends the post-`zdf_phy` frame with `avs` and TKE `en`. `usrdef_sbc` is
not used because this case executes `sbcblk`.

Detailed binary streams follow NEMO's own `lwp` convention: rank 0 alone
writes its full local `94 x 152 x 31` domain, including the two-cell halos.
They are intentionally not gathered and are never presented as a global
field. Reassembly/coverage uses the two ordinary NEMO restart shards as the
global state record; detailed frames are rank-0 operator records whose mapping
is fixed by `jpni=2`, `jpnj=1`, `nimpp=1`, `njmpp=1`, and `nn_hls=2` above.
Phase 2 must strip the registered halos before comparing interior cells. This
layout-note option was explicitly requested; changing decomposition makes the
detailed record set incommensurate.

The later record census proved that this binary contains writer strings only
for `stprk3`, `traqsr`, and `icedyn_adv_pra`.  Its 63 files comprise ten step
entries, ten barotropic frames, 30 stage frames, one stage-1 RHS, one RGB
chlorophyll frame, one ZDF entry, and ten Prather entry/exit frames.  The
registered Lane-2 detailed operands and Lane-3 SI3 exchange/thermodynamic/bulk
frames are absent.  Therefore the following preparation record is historical,
not an executable Phase-1 hand-off.

## Executed but coverage-refuted instrumented ten steps

Directory:

```text
/data/abyssal/dbalwada/nemo-testcases-l4/runs/instrumented_10step_np2
```

It contains regular-file copies of all 19 XML/namelist deck files, absolute
read-only symlinks to all 40 unpacked archive members, and an absolute symlink
named `nemo` to the exact instrumented binary above. Only the preregistered
run controls differ from the shipped namelist: `nn_itend=10`, `nn_stock=10`,
and `nn_istate=1`. The hashes are:

```text
namelist_cfg       62f4746cf3846254c18af73bcde53e42f7d4cb9ebcdaf5f63cc9ed5316a9f8f4
deck manifest      e1974d9db5974f54d451fe1112b8f22a404516adcc7cbcf0b035fbe1cccba70b
input manifest     3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5
original run.sh    f74b55fb0adf991e24bf0339caf761f216a19478e992210fac63d406fe1b487d
63-file census     4b320ee3847d1553d0f0098d721a44014c9c9f7d7c2ea639a9ec5d0873d8a958
```

`run.sh` refused an unexpected directory, stale output, binary/deck/input
hash mismatch, or missing archive member. It fixes one thread per math/runtime
library and executes exactly:

```text
mpirun -np 2 --oversubscribe ./nemo
```

The user's first launch found that `/usr/bin/time` was not installed and exited
127 before MPI started.  The user removed only that failed launcher's empty
`run.user.stdout.log`, `run.user.time.log`, and `run.launcher.log`, then changed
the launcher to Bash `time` and reran it.  The retained launcher SHA-256 is
`9ef46f677c98cae57f8686151ba79a9aafd7f135a7d3a358dbf250a239fc7027`.
It records `MPIRUN_RC=0`, `RUN DONE`, step 10, and wall/user/sys
`11.724/16.918/4.115 s`.  The complete 163-file census is
`manifests/instrumented_10step_np2_all_files.sha256`, whose SHA-256 is
`b633486295afca2c8f6930258cdcb4a9dc86d0cf0bc0eb0bcda372814505a4b6`.

## Executed uninstrumented 30 days

Directory:

```text
/data/abyssal/dbalwada/nemo-testcases-l4/runs/uninstrumented_30day_np2
```

The copy/symlink policy and guarded launcher were identical. At the shipped
`rn_Dt=10800 s`, 30 days is exactly 240 steps. Only `nn_itend=240` and
`nn_stock=240` differ from the shipped scientific deck; `nn_istate` remains
the shipped zero. The hashes are:

```text
binary             c4907e476cf3969052b44c5c7fa966f3dac493e8cfb563f6554c8f3a27186343
namelist_cfg       483ee5196fc880a09929fa59d9fcdd2f32871c4dd4a01db94839875f6d7efd65
deck manifest      4f8c480d03061ddd44218b0913dabc901fcaa2741b59bcc7d169730ded54d4db
input manifest     3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5
original run.sh    788c7b209965cb5541045e945a531eaedab5b21a49be44e220a0ec9c8f0c5a65
63-file census     76133cd32cb980eecbe2ca0d243e8d1e3c2e2da0c8d09d00a334103e83d24ad1
```

The same failed-first-launch and Bash-`time` substitution occurred here before
MPI started.  The retained launcher SHA-256 is
`24137b7e8471667361320458a910b940cad799d6f5f689e863640eb0cbae78b1`.
It records `MPIRUN_RC=0`, `RUN DONE`, step 240, and wall/user/sys
`188.768/363.542/10.378 s`; `ocean.output` has no `E R R O R`.  The complete
97-file census is `manifests/uninstrumented_30day_np2_all_files.sha256`, whose
SHA-256 is
`efe02aa764feb07ba045ee25c0bd304240b7780e39c01376dec741b4520413f9`.

## Choice disposition at this stop

| Item | Disposition |
|---|---|
| two MPI ranks | ASKED smallest supported layout; forced by executed iceberg constraint |
| `jpni=2`, `jpnj=1` | FACT resolved automatically by NEMO, not a choice |
| rank-0 `lwp` detailed writers plus layout note | ASKED choice among the two permitted record layouts |
| regular deck copies and absolute input/binary symlinks | UNASKED operational packaging, disclosed and hash-guarded |
| 240-step endpoint | ASKED 30 days, mechanically derived from shipped `rn_Dt` |
| user-shell execution | ASKED; all MPI launches remain outside the agent sandbox |
| Bash `time` substitution | ASKED after `/usr/bin/time` exit 127; exact retained launchers are re-hashed above |
| deletion of three empty failed-launch files in each directory | ASKED user action; only launcher-owned empty files, before successful relaunch |
| corrective rebuild/rerun | UNASKED enabling correction; required by the frozen record inventory, with no physics or deck change |

After the corrective user-shell run returns, Phase 1 still requires record
parsing and planted controls, the final byte-identity comparison against the
completed control, the active-input/coverage/time-level registers, final
manifest, executed-arm audit, and independent-review receipt. No Phase-2
legoESM work is allowed before that receipt.
