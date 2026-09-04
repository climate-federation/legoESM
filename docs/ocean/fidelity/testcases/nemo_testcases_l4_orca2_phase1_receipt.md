# NEMO testcase Lane 4 — ORCA2 oracle Phase-1 receipt

Re-closed after independent review: 2026-09-04 (America/New_York; supplied
archive provenance is dated 2026-09-05)

Session: `01a06d99-f562-7b11-bc63-e9b112877f54`

Independent-review status: **FIXED AND VERIFIED**

Verdict: **CONFIRMED — PHASE 1 CLOSED**

Scope: NEMO 5.0.2 oracle only. No legoESM card or legoESM numerical result was
created.

The accepted oracle is the single scalar-math binary
`nemo_ORCA2_OMIP_L4_instrumented_reviewfix.exe` and its ten-step run at
`/data/abyssal/dbalwada/nemo-testcases-l4/runs/instrumented_reviewfix_10step_np2`.
All 90 registered record files pass the frozen schema and manifest gate; the
six restart shards are byte-identical to the uninstrumented control. This is
one binary, one compiler-wide arithmetic policy, and one run: no field was
substituted from another build or precision (the no-Frankenstein rule).

Independent review found no scientific-output or provenance defect, but did
require two WRITE-only writer-source corrections: derived SI3 payload-count
metadata and armed-only EOS diagnostic scratch allocation. The replacement
run executed those corrections and passed the complete gate. The former
`instrumented_rank0_schema_10step_np2` arm remains preserved but is
superseded; no record from it is mixed into this oracle.

The companion artifact index is
`nemo_testcases_l4_orca2_phase1_manifest.md`. The executable data remain below
`/data/abyssal/dbalwada/nemo-testcases-l4`; no multi-megabyte artifact is in
git.

## 1. Oracle definition and source provenance

The shipped reference is `cfgs/ORCA2_ICE_PISCES` in NEMO git
`dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796`. `makenemo -r
ORCA2_ICE_PISCES -n ORCA2_OMIP_L4 -m conda-scalarmath` made a configuration
copy; no shipped file was modified. Its exact keys are:

```text
key_si3 key_qco key_vco_1d3d key_RK3
```

Only shipped `key_xios` and `key_top` are absent. `key_xios` removal selects
the native-IOM `dia_wri` arm at `src/OCE/stprk3.F90:237-245`, the same
no-XIOS convention used by Lanes 1 and 2; restart I/O remains NEMO-native.
`key_RK3` selects `stp_RK3`, whose three calls are at
`src/OCE/stprk3.F90:192-207`; `key_qco` selects the variable-volume WS-RK3
updates at `src/OCE/stprk3_stg.F90:151-167` and `:550-554`. This is not MLF.

`nn_ice=2` is retained. `src/OCE/SBC/sbcmod.F90:474-480` dispatches to
`ice_stp`, so this is the ORCA2 + SI3 five-category oracle, not ocean-only.

### Why `key_top` is excluded

The resolved source deck defaults `ln_top=.true.` at
`cfgs/SHARED/namelist_ref:67`, but absence of `key_top` forces the runtime
logical false at `src/OCE/DOM/domain.F90:353-356`. Both passive-tracer entry
points are compile/runtime guarded: `src/OCE/nemogcm.F90:496-499` at
initialization and `src/OCE/stprk3_stg.F90:502-506` at every RK stage. No
`trc_init`, `trc_stp_rk3`, PISCES, or TOP runtime message occurs.

The possible physics feedbacks are inactive:

- `ln_qsr_bio=.false.` is inherited from
  `cfgs/SHARED/namelist_ref:427`. The output resolves RGB chlorophyll
  (`ocean.output:1192-1210`), and `tra_qsr` dispatches to `qsr_RGBc` at
  `src/OCE/TRA/traqsr.F90:172-185`, not the `np_BIO` arm. The latter is the
  only light arm that requires TOP (`:1414-1419`).
- `ln_dm2dc=.false.` at `cfgs/SHARED/namelist_ref:203` and
  `ln_trcdc2dm=.false.` at `cfgs/SHARED/namelist_top_ref:112` exclude the
  TOP daily-shortwave route through `sbcblk`.
- ORCA2's `ln_trcldf_tra=.true.` and TOP's inherited
  `ln_trcldf_OFF=.false.` make the guarded TOP condition in `trazdf` reduce
  to the same active-tracer slope condition.
- NPC, OSMOSIS, and MFC selectors all resolve false; their guarded calls at
  `src/OCE/stprk3_stg.F90:591-599` do not execute.

Therefore PISCES has no TOP-to-ocean or TOP-to-SI3 feedback in this shipped
physical state. The early resolved echo still prints inherited `LN_TOP=T` in
`output.namelist.dyn:98`; this precedes the compile-key correction in
`domain.F90` and is not an executed selector. The direct call-graph and absent
runtime messages make the exclusion **CONFIRMED**, not an inferred omission.

The final directory retains verbatim `namelist_cfg`, `namelist_ice_cfg`,
`namelist_ref`, `namelist_ice_ref`, `output.namelist.dyn`, and
`output.namelist.ice`. Their individual hashes are in the artifact manifest;
the 19-file deck manifest is
`e1974d9db5974f54d451fe1112b8f22a404516adcc7cbcf0b035fbe1cccba70b`.

## 2. Inputs

The immutable `ORCA2_ICE_v5.0.0.tar.gz` is 1,365,673,011 bytes, MD5
`872eea51f22c5dabaffd268c0220eba6` (the published value), and SHA-256
`5d47eab85c591fe0fd7e63a80892f3264b6387edf93cbb975a71b2f2f5fdf1a4`.
The user's shell fetched it on 2026-09-05 from
`https://gws-access.jasmin.ac.uk/public/nemo-vol1/sette_inputs/r5.0.0/ORCA2_ICE_v5.0.0.tar.gz`
because the agent sandbox had no network. It was unpacked below `inputs/`
without changing the archive.

Exactly 40 files were read by the resolved active descriptors. Every filename
and digest is reproduced in the companion manifest from the retained
`input_files.sha256` (digest
`3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5`).
The run deck is copied; those inputs and the executable are absolute symlinks,
all checked by `run.sh` before launch. Stale/inactive archive members were
neither linked nor deleted.

## 3. Scalar-math builds and executions

Conda environment `nemo-build` supplied `mpif90`. The arch is the established
`arch-conda.fcm` plus compiler-wide `-fno-tree-vectorize`; its SHA-256 is
`132f7a0500c4f0e86d8d3bf7864974a82e1dea5d83166dcfdfaf409e2ca04561`.
All retained scalar-math binaries have zero `_ZGV*` dynamic symbols: their
census files are empty with SHA-256 `e3b0c442...b855`.

| binary | SHA-256 | build-log SHA-256 |
|---|---|---|
| uninstrumented | `c4907e476cf3969052b44c5c7fa966f3dac493e8cfb563f6554c8f3a27186343` | `b4622a7fb64cabd0fd7f8133b4a61d08c7d67947ba9d50aa87d622bb875360c8` |
| superseded WRITE-only | `07cec34c5683e6a3e37fa432ac5796ab43c747ab6f468fd131a8be796996dbb8` | `42614a4b1c2b62129acdfa93569f2d172c51bbd09a0c8616422ca9aae854d028` |
| accepted review-fixed WRITE-only | `ca2355aa777adc47825c4b777c5fa4e89cc60dc2c509a0e2f379e439acefd025` | `96a38f0415970a5d3987d67374d1583ad30a19725d1b199465a60b46857526a7` |

The accepted binary is 54,899,920 bytes. Its 14 config-local `MY_SRC`
overrides are pinned by manifest SHA-256
`78e1465fe7d9cea4d3adf24fb369b6a458c7bb7a912cc75f85d7ca6177e59152`.
They only open/write diagnostic streams or copy values into writer-local
temporaries. The final seven `IF (.NOT.lwp) RETURN` guards select rank-zero
file ownership and do not assign a model field.

### MPI layout is an oracle constraint

The retained one-rank attempt at `runs/uninstrumented_10step` stopped at
`src/OCE/ICB/icbini.F90:112` with `icbinit: having ONE processor in x
currently does not work`. The smallest supported layout is therefore two MPI
ranks, automatically resolved as `jpni=2`, `jpnj=1`, `jpnij=2`. It is an
executed fact, not an analyst choice.

The global tripolar domain is `184 x 152 x 31`. Each rank has `94 x 152 x 31`
including `nn_hls=2`; rank 0 has `nimpp=1,njmpp=1`, rank 1
`nimpp=91,njmpp=1`. Detailed records use NEMO's `lwp` convention and contain
rank 0 only. Global reconstruction uses both native restart shards: remove
the two-cell halos, place rank 0 at its `nimpp/njmpp` offset and rank 1 at its
offset, resolve the cyclic overlap and north-fold exactly as `layout.nc`
records. Detailed records must never be described as already global.

All launches were CPU-only, one at a time, from the user's shell with
`mpirun -np 2 --oversubscribe ./nemo`, one thread for each math/runtime
library. Each self-contained launcher checked its directory, binary, deck,
and input hashes and teed stdout.

| run | run controls | result | bash time wall/user/sys (s) |
|---|---|---|---|
| uninstrumented identity | `nn_itend=10, nn_stock=10, nn_istate=1` | step 10, RC 0, `RUN DONE` | 14.138 / 21.205 / 4.557 |
| superseded instrumented | same | step 10, RC 0, `RUN DONE` | 13.050 / 19.126 / 4.561 |
| accepted review-fixed instrumented | same | step 10, RC 0, `RUN DONE` | 12.280 / 17.597 / 4.499 |
| uninstrumented statistics reference | `nn_itend=240, nn_stock=240`; shipped `nn_istate=0` | step 240, RC 0, `RUN DONE` | 188.768 / 363.542 / 10.378 |

At shipped `rn_Dt=10800 s`, 240 steps are exactly 30 days. The only scientific
deck change in either ten-step run is the requested run length/control stock;
the 30-day run changes only endpoint/stock. No error, abort, NEMO control
failure, or non-finite diagnostic appears; all requested restarts were
written. Run/output/restart hashes are in the companion manifest.

The prepared launchers initially used `/usr/bin/time`. That executable is not
installed, so the first attempt in each of the first two prepared directories
exited 127 before MPI. The user removed only the three empty launcher-owned
files and substituted Bash `time` with
`TIMEFORMAT='wall_seconds %R\nuser_seconds %U\nsys_seconds %S'`. The
superseded and accepted final instrumented launchers already contained that
substitution and ran unchanged. The accepted launcher's hash is
`756d6ac26e851ecba33b0ef8a6061238e4b8ca289d9e2011b4b0e8747c3d7415`.
The complete retained launcher census is:

| run directory | `run.sh` SHA-256 | status |
|---|---|---|
| `instrumented_10step_np2` | `9ef46f677c98cae57f8686151ba79a9aafd7f135a7d3a358dbf250a239fc7027` | Bash-time substitution; retracted record set |
| `instrumented_full_10step_np2` | `c88d5bc7fad7974960bcdef838b6ab4250bb35abf98adef88304d155ef39c512` | executed unchanged; retracted record set |
| `instrumented_rank0_10step_np2` | `91ad44b8d12703ecc8f91eb032a3dcce5067a9190b3fe6f54bd2300a9842f7d4` | executed unchanged; retracted record set |
| `instrumented_rank0_schema_10step_np2` | `59e794abc299cc34897eddd905e492edf8fe41dde456977283abf0161fc767ea` | executed unchanged; preserved superseded run |
| `instrumented_reviewfix_10step_np2` | `756d6ac26e851ecba33b0ef8a6061238e4b8ca289d9e2011b4b0e8747c3d7415` | executed unchanged; accepted oracle |
| `uninstrumented_30day_np2` | `24137b7e8471667361320458a910b940cad799d6f5f689e863640eb0cbae78b1` | Bash-time substitution; accepted statistics reference |

## 4. Full record-schema gate

The committed fail-closed gate is
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l4_orca2_phase1_gate.py`
(SHA-256 `3d0b61762a4dbe0acbdaa3b52791696346a73afc84a78bb7388a167999aa00ce`).
The final invocation used the frozen 90-file record manifest and
`--plant-controls`. Result: `PASS`; JSON SHA-256
`dc42802b3f925c4197b75003431e3ba1762bfa18a39a038f8a864f10cc553577`.

For every frame the gate consumes exactly 16 ASCII bytes of magic, exact-width
32-bit header integers, then the declared native binary64 values. It checks
magic, version, kt/stage/category/time-level sequence, dimensions, bit width,
declared payload count, exact file size/EOF, manifest hash, and finiteness in
blocks. Thus a size match alone cannot pass.

### Frozen file inventory

| stream family | files/frames | magic | exact schema result |
|---|---:|---|---|
| step entry | 10/10 | `NEMO_L1_ENTRY_1` | kt 1..10; alternating Kbb 1,3; 14,288,048 B/file |
| barotropic frame | 10/10 | `NEMO_L1_BTFRM_1` | kt 1..10; Kaa 3,1 alternating; 457,256 B/file |
| RK stage state | 30/30 | `NEMO_L1_STAGE_1` | stages 1,2,3 for every kt; Kaa sequence checked; 14,288,052 B/file |
| stage transport | 3/3 | `NEMO_L1_TRANSP_1` | kt1 stages 1..3; Kmm 1,3,2; 10,630,320 B/file |
| stage-1 momentum RHS | 1/1 | `NEMO_L1_RHS___1` | Krhs=3; 7,086,892 B |
| ZDF/TKE entry | 1/1 | `NEMO_L4_ZDF___2` | Kbb=1; 13,453,548 B |
| slow forcing | 1/1 | `NEMO_L2_SLOW_2` | Kbb=1,Krhs=3 plus seven truthful section counts; 22,814,420 B |
| barotropic substeps | 1/65 | `NEMO_L2_BTSUB_2` | substep ids 1..65 monotonic; 147,588,780 B |
| barotropic drag | 1/65 | `NEMO_L2_BTDRG_1` | substep ids 1..65 monotonic; 89,386,028 B |
| barotropic advective mean | 1/65 | `NEMO_L2_BTADV_2` | substep ids 1..65 monotonic; 74,984,772 B |
| ordered barotropic operands | 1/2 | `NEMO_L2_BTORD_2` | substep ids 1,2 monotonic; 11,504,296 B |
| EEN coefficients | 1/1 | `NEMO_L2_ENECO_1` | Kmm=1,Krhs=3; 852,524 B |
| stage-1 transport operands | 1/1 | `NEMO_L2_TRPOP_2` | Kmm=1; 29,490,480 B |
| stage-2 terms/HPG operands/HPG literal | 3/3 | `NEMO_L2_RKTRM_1`, `NEMO_L2_HPGOP_1`, `NEMO_L2_HPGLT_1` | Kmm=3,Krhs=2; 28,347,444 / 10,630,316 / 21,489,200 B |
| stage-2 preupdate/operands/EOS | 3/3 | `NEMO_L2_RKPRE_1`, `NEMO_L2_RKSTG_1`, `NEMO_L2_EOSOP_1` | Kbb/Kmm/Krhs/Kaa = 1/3/2/2; exact 7,086,908 / 35,876,028 / 46,065,036 B |
| RK tracer operands | 2/2 | `NEMO_L2_RKTRA_1` | stage1 `1/1/3/3`, stage2 `1/3/2/2`; 53,494,332 B/file |
| tracer transport/stage3/QSR/WZV | 4/4 | `NEMO_L2_TRTRP_1`, `NEMO_L2_RKTR3_1`, `NEMO_L2_QSR___1`, `NEMO_L2_WZVOP_1` | stage3 exact levels; 10,630,332 / 57,037,756 / 3,650,036 / 10,630,320 B |
| RGB chlorophyll | 1/1 | `NEMO_L4_CHL_001` | Kmm=2, 3-D/profile flag=1; 335,464 B |
| Prather entry/exit | 10/10 | `NEMO_L4_PRA_001` | odd kt 1,3,5,7,9, stages 0,1; jpl=5,nlay_i=10,nlay_s=5; 94,300,860 B/file |
| SI3 bulk append | 1/15 | `NEMO_L3BULK_001` | odd kt monotonic; stages 0/1/2; payloads 17/50/39; 4,780 B |
| SI3 exchange append | 1/10 | `NEMO_L3XCHG_001` | kt 1..10 monotonic; 1,181,352 f64/frame; 94,508,560 B |
| SI3 thermo append | 1/140 | `NEMO_L3THD_001` | five odd kt, 28 frames/ice step; category/stage order and compressed npti checked; 345,301,800 B |
| SI3 ZDF inputs append | 1/25 | `NEMO_L3ZIN_002` | odd kt x categories 1..5; count=`(13+nlay_s)*npti`; 6,681,880 B |
| SI3 reassociation append | 1/5 | `NEMO_L3REA_001` | kt=3, categories 1..5; count=`(3+3*nlay_i+2*nlay_s)*npti`; 3,156,440 B |

The record census SHA-256 is
`70c3779bc11e4b3df41ebf756abf25362dab82d9622c87829012618412cf273a`.
All five append streams reached exact EOF after the listed frames: there is no
concurrent-write interleaving, truncation, or trailing payload.

All nine independent planted violations exit nonzero and are recorded
`PASS_NONZERO`: extra file, missing file, malformed magic, wrong time level,
truncated payload, trailing byte, binary64 NaN, one-ulp active SI3 mutation,
and one-byte restart identity mismatch.

The binding identity plant materialized the instrumented ocean restart,
changed one byte at offset 128, and called `validate_identity` itself. The
standalone child reported
`identity mismatch: ORCA2_00000010_restart_0000.nc` and exited `1`; the
aggregate planted-control gate records `restart_identity_byte=PASS_NONZERO`.

## 5. WRITE-only byte-identity control, per ordinary file

The comparison rule was frozen before inspection. Deterministic ordinary files
must be exact bytes. Native-IOM history files may differ only in their global
wall-clock `TimeStamp`; the gate compares every dimension, variable, variable
attribute, global attribute other than `TimeStamp`, and the raw stored-value
bytes of every data variable with automatic masking/scaling disabled.
`ocean.output` may differ only by registered `LANE*`/`L2_*` writer notices.
Launcher and timing products are excluded because instrumentation necessarily
changes runtime and stdout. No scientific field is excluded.

Exact-byte files:

| file | shared SHA-256 | result |
|---|---|---|
| `ORCA2_00000010_restart_0000.nc` | `9ba6054040f11ea38a6637c4e6e7e26261359540fbf2d5be2cf63b78857dd508` | EXACT |
| `ORCA2_00000010_restart_0001.nc` | `d06e099765da5e6eba9f2f6ecb8aa66b20fb034fa506456daeb8da0cb8fea1ba` | EXACT |
| `ORCA2_00000010_restart_icb_0000.nc` | `1dc69243e465964a99217c5729df5f21ee23a583507e7cbdb274502700eb3926` | EXACT |
| `ORCA2_00000010_restart_icb_0001.nc` | `5b183bb7e79003a2d36325247246d9265b8030e21ebc4ffd3507ab6a9450e5b1` | EXACT |
| `ORCA2_00000010_restart_ice_0000.nc` | `5266345352fecb609820d3e62f495019d2bd05d4e0bdb40d0a327bd6c9dafd20` | EXACT |
| `ORCA2_00000010_restart_ice_0001.nc` | `2634bd96f623fe476c3c389ea232b03583793686dde8781120427320676b68e2` | EXACT |
| `layout.dat` | `1e85d5ae647a3fbfdc6a843440d7be3e409bf2a52648cab67549cc8ca3962832` | EXACT |
| `layout.nc` | `61392c48ee1435e4beb0b84297995b13add16749dd4078588df7bb6a7637494b` | EXACT |
| `mesh_mask_0000.nc` | `0f373c6609d287bc818b1f6ef96bafdcc0b620c730219018de9728369d01ec5f` | EXACT |
| `mesh_mask_0001.nc` | `0b6903ce4e508bbd3f3c05dfbd8290548549ed52b13f081810e717071e977933` | EXACT |
| `output.init_0000.nc` | `376b6f3170b7231056215d62e38ebf1619704e76289e5a87c4e0c0c678033a32` | EXACT |
| `output.init_0001.nc` | `7dec10870109c6c2bc9c6ae01fec905c4ce4f4b406a268f0cb154d1a902fd04b` | EXACT |
| `output.init_ice_0000.nc` | `dbe61bd953c2acacf6410af5addb79bef186d40be06fb9afe7a8caad2f2a1dfb` | EXACT |
| `output.init_ice_0001.nc` | `01555e31709cd398ab48fea133a5f6072cfa454b63c79309476fe04cf2f5ee25` | EXACT |
| `output.namelist.dyn` | `31ceebedb15a0e427d9fc530029cc98e18db6749545ddc354ca749d2192e4148` | EXACT |
| `output.namelist.ice` | `2ccc2d012159bcc1d182dc48bf34ff2abdfb09aecb45ff227cd30ebadd9a8603` | EXACT |
| `time.step` | `83e4e460507d78e2bd843e9a6961b3b204c2b5229499b5e2f10f11b3a7d5bb78` | EXACT |
| `trajectory_icebergs_00010101-00010102_0000.nc` | `3f503096312c530ca92ce0b37279493a63874b31c664b659f3d8ac75657f3e1c` | EXACT |
| `trajectory_icebergs_00010101-00010102_0001.nc` | `cb2e3c86298eb1fa31cadd3ee21b6d29abb3b6c2d465df6e66670eeb43e001c6` | EXACT |

Native-IOM files, each exact except global `TimeStamp`:

| file | control SHA-256 | instrumented SHA-256 | result |
|---|---|---|---|
| `...grid_T_0000.nc` | `f042130de9608c38b5e9f6691d70dea490a01b7e625024938a0e3f096946a3f7` | `131362bfd6e19473f7c9dc5df895e95c69e5aad1bf475e56822892d456cdbe3c` | EXACT DATA/METADATA |
| `...grid_T_0001.nc` | `f15fb7f02132d7eb037a368f464d8eeee1f497593baca5a1a80d6b1faf7f1a72` | `2bb7b10e61923e72419348b9b473f81df16574167a6f05f304b4a576c1ffe51d` | EXACT DATA/METADATA |
| `...grid_U_0000.nc` | `bd8bdded908746eaf10e699307fb3f58a51179b5a6169de162673118f4dba3be` | `1b82db7139536c594529899b123834adf6fa7478879144629b498ba95d0a13fe` | EXACT DATA/METADATA |
| `...grid_U_0001.nc` | `c496f4af2e35a2951cb20341b9f4d9599416d2c0c10fecac342986cec371d6f1` | `242abf429be62f46616dbc6cb78d765cab023307d9d547e4babb22d48cf197d4` | EXACT DATA/METADATA |
| `...grid_V_0000.nc` | `266b32c0a5509d7c50907e99690422d35808cf97920f9ca66376f149c36c6e50` | `eb00d78d103db9cde78fab0744a018b4f7b07eb251a8c1ad4f282e30d0e99eba` | EXACT DATA/METADATA |
| `...grid_V_0001.nc` | `3538dc2a1b75b46c245c4f376824dc9a7609bc3078a0e7b04ddcafef10906ba9` | `09adfd661275a7a16b3b898373c02759183a722ee98c2e4fcfc83df2c9e5d416` | EXACT DATA/METADATA |
| `...grid_W_0000.nc` | `34f31b75eddf5a87ad0c76fa29faf29d81b66f8813d23e0d28a4462547f6e692` | `2635bbf387e1f3f1abc2c63db73a06b1d2c71300c4534d65df63cbb471b45473` | EXACT DATA/METADATA |
| `...grid_W_0001.nc` | `18ba9179726757bead40cb24286f295bbdcc8ee0f693e18c41a2ca9c5bf96e71` | `c6d16063d56fa5a17cee4aa5e44ad136faa1d35b66a5b9d47421d347a5262558` | EXACT DATA/METADATA |

`ocean.output` is exact after dropping only writer notices (control
`644c5223...10fea`, instrumented `06c41ebf...a534`). Per the frozen rule, the
excluded files are `communication_report.txt`, `timing.output`,
`timing_gnuplot.sh`, `timing_step.nc`, `timing_ts_allmpi_step.nc`,
`timing_tsum_allmpi_t1_t10.nc`, `run.user.stdout.log`, `run.user.time.log`, and
`run.launcher.log`. This proves the instruments are WRITE-only for ordinary
model results.

`validate_identity` increments its counters inside the successful comparison
loops rather than reporting constants. For this run it reports
`restart_shards_exact=6` and `history_payloads_exact=8`.

## 6. Executed-arm audit in NEMO order

`ocean.output` confirms EOS-80 (`:163-164`), RK3 and 10800 s (`:222`), QCO
(`:355`), NCAR bulk (`:594-607`), SI3 (`:675`), Prather (`:930`), TKE
(`:972`), FCT (`:1283`), vector-form momentum (`:1320`), EEN (`:1336-1338`),
SCO HPG (`:1356`), split-explicit surface (`:1367`), and runtime stages 1/2/3
(`:1783,:1802,:1843`). The ordered descent through the executed step is:

| order | executed arm | selector/call evidence | Phase-1 record disposition |
|---:|---|---|---|
| 1 | surface means, CORE/NCAR bulk forcing | `sbcmod.F90:403-424`; NCAR dispatch `sbcblk.F90:453-460` | bulk operands and exchange fields VERIFIED |
| 2 | SI3 every `nn_fsbc=2` ocean steps | `sbcmod.F90:474-478`; `icestp.F90:126-159` | five ice steps, bulk stress VERIFIED |
| 3 | SI3 dynamics then Prather advection, five categories | `icestp.F90:164-184`; `icedyn_adv_pra.F90:109-139,488` | all first/second moments at entry/exit VERIFIED |
| 4 | per-category flux, thermodynamics, ocean-flux update | `icestp.F90:189-224`; `icethd.F90:116-231`; `nn_flxdist=-1` at output `:812-819` | 140 thermo + 25 ZDF + 5 reassociation + exchange frames VERIFIED |
| 5 | active iceberg model, then runoff | `sbcmod.F90:484-494`; output `:1380-1446` | ICB state/restarts and runoff exchange VERIFIED |
| 6 | EOS-80 `eos_rab` + `bn2` on `Nbb` | `stprk3.F90:153-160`; EOS selector `eosbn2.F90:2008-2011,2216-2217` | EOS operands, T/S entry and derived closure inputs VERIFIED |
| 7 | TKE vertical physics, enhanced vertical diffusion, double diffusion, differential internal-wave mixing | `stprk3.F90:163-165`; selectors at output `:972-1036` (`ln_zdfevd` `:977`, `ln_zdfddm` `:983`, `ln_zdfiwm` `:988`) | `en,avm,avt,avs` VERIFIED |
| 8 | standard Madec isoneutral slopes; spatial/time-varying tracer diffusivity and EIV; 3-D file viscosity | `stprk3.F90:169-180`; output `:1116-1182`, including `ln_ldfeiv` `:1144` | inputs/state VERIFIED; scratch slope/coefficient temporaries WAIVED below |
| 9 | stage-one slow momentum RHS: vector invariant, implicit linear bottom drag, wind | `stp2d.F90:175-230`; drag selectors output `:1094-1097` | complete ordered slow-forcing operands VERIFIED |
| 10 | QCO split-explicit barotropic solver, 65 substeps, EEN | `dynspg.F90:179-181,230-250`; output `:1367`; `dynspg_ts.F90` substep loop | recurrence, drag, EEN, ordered first two steps and `un_adv/vn_adv` VERIFIED |
| 11 | WS-RK3 stages 1,2,3 | `stprk3.F90:192-207`; stage selector `stprk3_stg.F90:116-218` | every stage Kaa state VERIFIED |
| 12 | QCO geometry and WZV transports | `stprk3_stg.F90:151-167,281-301`; `ln_dynadv_vec=T` at `ocean.output:1315` selects the WZV calls | QCO states, transports, stage-3 WZV operands VERIFIED |
| 13 | stages 2/3 EOS-80, SCO HPG, total-vorticity EEN, vector invariant momentum advection | `stprk3_stg.F90:309-334`; HPG dispatch `dynhpg.F90:170-204`; runtime `:1854-1856` | literal EOS/HPG/EEN/RHS operands VERIFIED |
| 14 | stage-3 iso-level Laplacian momentum mixing then implicit momentum ZDF | `stprk3_stg.F90:395-430`; output `:1162-1182,1864` | boundary states/RHS VERIFIED; operator scratch WAIVED |
| 15 | transport construction including EIV and Fox-Kemper mixed-layer eddy transport, FCT tracer advection, surface tracer flux each stage | `stprk3_stg.F90:456-521`; EIV/MLE insertion `traadv.F90:343-346`; FCT dispatch `traadv.F90:355-370`; `ln_mle=T` output `:1288` | `zFu,zFv,zFw` and tracer operands VERIFIED |
| 16 | stage-3 RGB chlorophyll QSR, standard isoneutral Laplacian, variable geothermal bottom heat, diffusive/advective BBL, and tracer restoring | `stprk3_stg.F90:568-598`; RGB dispatch `traqsr.F90:172-185`; BBL and damping selectors output `:1247-1256`; runtime `:1882-1915` | chlorophyll/QSR/tracer boundary states VERIFIED |
| 17 | QCO tracer update and implicit tracer ZDF; native-IOM output/restarts/control | `stprk3_stg.F90:540-610`; `stprk3.F90:222-270` | after states, six global restart shards and outputs VERIFIED |

Inactive high-leverage calls are explicitly disposed: TOP is compile-excluded;
MLF, TEOS-10, flux-form momentum, explicit free surface, CICE, user-defined
SBC, OSMOSIS, NPC, MFC, ice shelves, open boundaries, tides, AGRIF, and
atmospheric coupling do not execute. Adaptive implicit vertical advection is
also inactive: `ln_zad_Aimp=F` at `ocean.output:968`, and all `wAimp` calls
are guarded by it at `src/OCE/stprk3_stg.F90:283-299`. The iceberg model is active with test
icebergs (`nn_test_icebergs=10`); `calving.nc` is present and hash-pinned, but
the executed `ln_use_calving=F` selector (`ocean.output:1444-1446`) means its
calving field does not seed this short run. Phase 2 must preserve that exact
distinction.

### Independent-review selector re-verification

After the row-12 correction, every other audit row was re-checked against the
accepted run's resolved selector and its source dispatch. Re-verified without
further correction: rows 1 (`ln_blk=T`, `ln_NCAR=T`, output `:517,:594`), 2
(`nn_ice=2`, `nn_fsbc=2`, `:513,:526`), 3 (`ln_icedyn=T`, Prather, `:706,:930`),
4 (`ln_icethd=T`, `nn_flxdist=-1`, `:707,:812`), 5 (`ln_rnf=T` plus the
executed iceberg banner, `:534,:1383`), 6 (EOS-80, `:163-164`), 7 (TKE/EVD/DDM/IWM,
`:972,:977,:983,:988`), 8 (Laplacian/isoneutral/EIV and file viscosity,
`:1116,:1121,:1144,:1182`), 9 (linear implicit drag, `:1094,:1097`), 10
(split-explicit, `:1363-1367`), 11 (three runtime stage banners,
`:1783,:1802,:1843`), 13 (vector/EEN/SCO, `:1315,:1329,:1351`), 14
(Laplacian/level momentum diffusion and TKE ZDF, `:1162,:1165,:972`), 15
(FCT/MLE, `:1273,:1288`), 16 (RGB/geothermal/BBL/restoring,
`:1192,:1227,:1247,:1256`), and 17 (`key_qco`, split-explicit, native-IOM and
restart call sites cited above). Row 12 was the sole executed/inactive
misclassification found by this full pass.

## 7. Coverage register

Here **VERIFIED** means present in the accepted oracle, manifest-pinned,
schema/finite checked, and covered by the ordinary-output identity test. It
does not claim a legoESM match; that is Phase 2.

| oracle state/exchange family | disposition | evidence |
|---|---|---|
| active tracers `tn,sn`, stage `ts`, surface `sst_m,sss_m` | VERIFIED | entry/all stages/tracer operands plus ocean restart |
| 3-D momentum `un,vn`, stage `uu,vv`; barotropic `ubb_e,vbb_e,ub_e,vb_e,uu_b,vv_b`; `un_adv,vn_adv` | VERIFIED | entry/stages/RHS, complete external-mode records, ocean restart |
| free surface `sshn,ssha,sshb_e,sshbb_e,ssh_m`; QCO live thickness/geometry and transports | VERIFIED | entry/stages, slow/RK/EOS/HPG records, ocean restart |
| TKE and diffusivities `en,avt_k,avm_k` plus dumped `avm,avt,avs` | VERIFIED | ZDF frame and ocean restart |
| freshwater/filter carry `a_fwb,a_fwb_b,emp_corr,frq_m,e3t_m`, delayed coupling buffers, `fraqsr_1lev` | VERIFIED | exact ocean restarts; QSR/slow/exchange frames where consumed |
| SI3 core `v_i,v_s,a_i,t_su,u_ice,v_ice,oa_i,a_ip,v_ip,v_il,e_s_l01..05,e_i_l01..10,sv_i,szv_i_l01..10` | VERIFIED | global thermo frames plus both exact SI3 restart shards |
| five-category thermodynamic compressed states `a_i_1d,h_i_1d,h_s_1d,t_su_1d,e_i_1d,e_s_1d,sz_i_1d` | VERIFIED | 140-frame thermo stream, jpl=5 |
| all Prather moments: ice/snow/area/age, five `c0` levels, ten ice-energy levels, salinity, `a_ip/v_ip/v_il` moment families | VERIFIED | 10 entry/exit files; all 210 SI3 restart variables enumerated by the gate audit |
| ice stresses/deformation moments and `snwice_mass,snwice_mass_b` | VERIFIED | SI3 restart, bulk and exchange streams |
| full ice/ocean exchange: category `qns_ice,qsr_ice,qla_ice,dqla_ice,dqns_ice,tn_ice,alb_ice,qevap_ice`; `qml_ice,qcn_ice,qtr_ice_top`; stresses, evaporation, precipitation, freshwater, heat, salt, `fr_i`, freeze temperature and drag | VERIFIED | `NEMO_L3XCHG_001` every kt and bulk/ZDF frames |
| iceberg `calving,calving_hflx,stored_ice,stored_heat` and particle `lon,lat,xi,yj,uvel,vvel,mass,thickness,width,length,number,year,day,mass_scaling,mass_of_bits,heat_density` | VERIFIED | both exact ICB restart shards and trajectory files |
| restart coordinates/control metadata (`nav_lon,nav_lat,nav_lev,numcat,time_counter,kt,kt_ice,ndastp,adatrj,ntime,nn_fsbc,kount`) | VERIFIED | variable census and byte-exact restart identity |
| rank-1 detailed operator internals | WAIVED | `lwp` rank-zero layout was requested; global carried state is covered by both restart shards and layout-based reassembly |
| transient local work arrays not crossing a call boundary (loop scalars, local stencil increments, diagnostic-only reductions) | WAIVED | not prognostic/exchange state; causal boundary operands and outputs are recorded instead |
| TOP/PISCES tracers | WAIVED / NOT CARRIED | compile-excluded by oracle definition after the no-feedback audit |
| inactive scheme state (CICE, OSMOSIS, NPC, MFC, BDY/ISF/AGRIF/coupler) | WAIVED / NOT EXECUTED | selectors and compile keys above |

There are no unaccounted prognostic or ocean/SI3/iceberg exchange families.
The restart inventory is 42 ocean variables, 210 SI3 variables, and 21 ICB
variables; coordinate/control variables are explicitly included rather than
silently ignored.

## 8. Time-level registry

The integer IDs are NEMO array slots, not physical labels inferred from names.
For odd kt, step entry has `Kbb=1`; for even kt `Kbb=3`. At kt=1 the WS-RK3
calls are stage 1 `(Kbb,Kmm,Krhs,Kaa)=(1,1,3,3)`, stage 2
`(1,3,2,2)`, stage 3 `(1,2,3,3)`. The mapping is proved by
`stprk3.F90:195-207` and recorded in every relevant header.

| record / every dumped array | registered level |
|---|---|
| `step_entry`: `ts,uu,vv,ssh` | `Kbb` |
| `bt_frames`: `uu_b,vv_b` | `Kaa`; `un_adv,vn_adv` are interval-integrated transports for `Kbb -> Kaa`, not slot arrays |
| `stage`: `ts,uu,vv,ssh` | stage `Kaa` |
| `rhs`: `uu,vv` | `Krhs` |
| `zdf_entry`: `avm,avt,avs,en` | sampled immediately after `zdf_phy(kt,Kbb,Kbb,Krhs)`; closure carry at step `Kbb`, no separate RK slot for coefficients |
| `slow_forcing`: `e3u_3d,e3v_3d,r1_hu,r1_hv` | `Kbb`; `uu,vv` at `Krhs`; masks/constants/forcing and successive `Ue_rhs,Ve_rhs` snapshots are non-slot operands |
| `transport`: `zFu,zFv,zFw` | constructed from stage `Kmm` (1,3,2 for stages 1,2,3) |
| stage-1 transport operands | `uu` and live geometry at `Kmm=1`; masks/metric/source arrays are non-slot |
| stage-2 term/HPG literal/HPG operands | density/live geometry at `Kmm=3`; accumulating momentum at `Krhs=2` |
| stage-2 preupdate/operands | all slot arrays explicitly `Kbb/Kmm/Krhs/Kaa=1/3/2/2` |
| stage-2 EOS operands | input `ts` and geometry at `Knn=Kmm=3`; destination context `Krhs=2`; constants are non-slot |
| RK tracer operands stage 1 | `Kbb/Kmm/Krhs/Kaa=1/1/3/3` |
| RK tracer operands stage 2 | `Kbb/Kmm/Krhs/Kaa=1/3/2/2` |
| tracer transport and stage-3 tracer frame | `Kbb/Kmm/Krhs/Kaa=1/2/3/3` |
| stage-3 QSR: `qsr`, temperature increment | forcing at stage `Kmm=2`; increment accumulated into `Krhs=3` |
| stage-3 WZV operands | transport built from `Kmm=2`; result feeds stage 3, no stored prognostic time slot |
| RGB: `sf_chl%fnow`, profile, depths/attenuation inputs | forcing sampled for kt=1 and stage `Kmm=2`; no prognostic slot |
| barotropic substep/drag/advmean/ordered records | external-mode subcycle `m=1..65` within `Kbb -> Kaa`; each frame is tagged by substep, never mapped to Kmm by guess |
| SI3 bulk, exchange, thermo, ZDF, reassociation, Prather arrays | SI3 current state at odd ice steps; Prather stage 0/1 means advection entry/exit, thermo 0..7 means routine checkpoints, and category IDs are not ocean K slots |
| ocean restart | terminal `Nbb` state as written at `stprk3.F90:256`; its own `kt/ndastp` metadata is authoritative |
| SI3 restart | last executed ice state (`kt_ice=9`) because `nn_fsbc=2`; its own metadata is authoritative |

An unregistered filename, kt/stage/category sequence, or time-level ID raises;
there is no default-to-now behavior.

## 9. Retractions and cross-lane defect ticket

Rule 11 history is preserved; none of these directories was deleted:

1. `instrumented_10step_np2` (63 records) is retracted because preserved
   timestamps let FCM reuse 11 stale preprocessed overrides. It does not meet
   the frozen inventory.
2. `instrumented_full_10step_np2` (90 names) is retracted because both MPI
   ranks concurrently wrote the SI3 append files. Sequential parsing exposed
   interleaving/corruption.
3. `instrumented_rank0_10step_np2` (90 names) is retracted because
   `l3rea_dump` declared `18*npti` but wrote `43*npti`. The old gate now rejects
   it as `oracle_si3_reassoc_operands.bin: bad payload size header`.
4. `instrumented_rank0_schema_10step_np2` passed its contemporary gate and has
   no scientific-output defect, but independent review required the writer to
   derive that count from `nlay_i/nlay_s` and required EOS scratch allocation
   only while armed. It is preserved as a superseded arm, not used as a source
   of replacement records.

The accepted correction changes only config-local writer ownership guards and
derived truthful header metadata. No model array or arithmetic expression is
changed; the ordinary-output identity result above proves that claim
observationally.

### Independent-review correction: armed-only EOS scratch

The config-local EOS operand writer no longer creates thirteen unconditional
`(jpi,jpj,jpk)` automatic arrays on every EOS call.  In both instrumented EOS
entry points, `MY_SRC/eosbn2.F90:268-280,492-508`, those arrays are allocatable
and are allocated only while `l2_dump_armed` is true; after the record is
closed they are deallocated at `:353-354,646-647`.  Their initialization,
population, order, type, dimensions, and record schema are unchanged.  This is
a WRITE-only instrumentation-lifetime change and is included in the corrective
instrumented rebuild and the accepted replacement ten-step run.

### Independent-review correction: history payload identity

The identity gate disables NetCDF automatic masking and scaling and compares
the contiguous raw stored-value bytes of every data variable.  Consequently a
difference underneath a mask, including a differing fill value in the payload,
is now a failure; `np.ma.allequal` is no longer used.  Dimension, dtype,
variable-attribute, and non-`TimeStamp` global-attribute checks remain in force.

### Cross-lane ticket: Lane-3 reassociation header

The defect originated in the Lane-3/C1D writer
`MY_SRC/icethd.F90:l3rea_dump`, carried by
`docs/ocean/fidelity/testcases/nemo_testcases_l3thd_exchange_drift_ticket.md`.
The retained Lane-3 source witness is
`nemo502_si3thd_phase6_src/cfgs/C1D_OMIP_L3/MY_SRC/icethd.F90:298-315`
under `/data/abyssal/dbalwada/nemo-testcases-l3`; its header expression at
`:309` is the defective `18*npti`.
The Lane-3 source writes, per active category, `t_su` (1), `t_i` (10), `t_s`
(5), `sz_i` (10), `e_i` (10), `e_s` (5), `qns_ice` (1), and `dqns_ice` (1):
`43*npti` binary64 values, while its inherited header says `18*npti`.

Lane 3 must derive the payload count from the write-list dimensions, not
replace `18` with another literal: `(3 + 3*nlay_i + 2*nlay_s)*npti`, which
resolves to `43*npti` for `nlay_i=10,nlay_s=5`. It must add a sequential
header-validity check for every
`NEMO_L3REA_001` frame that verifies: exact 16-byte magic; version; monotonic
kt/category; `nlay_i=10`, `nlay_s=5`, `wp=64`; positive `npti`; declared
`count == (3 + 3*nlay_i + 2*nlay_s)*npti`; exactly that many finite binary64 payload values; and exact
EOF after the expected category frames. A mere whole-file byte count is
insufficient. The C1D Lane-3 oracle streams may otherwise carry the same
malformed metadata even when their payload bytes are intact.

## 10. ASKED / UNASKED disposition

| action/choice | disposition |
|---|---|
| shipped copy `ORCA2_OMIP_L4`; exclude only `key_top,key_xios`; retain RK3/QCO/SI3 | ASKED, exact |
| scalar-math build and zero `_ZGV*` | ASKED, exact |
| supplied archive provenance; unpack without archive modification | ASKED, exact |
| CPU, smallest supported layout | ASKED; two ranks forced by executed iceberg constraint |
| `jpni=2,jpnj=1` | FACT resolved by NEMO, not a choice |
| rank-zero detailed records plus documented halo reassembly | ASKED layout choice |
| user-shell MPI execution | ASKED |
| Bash-time substitution after `/usr/bin/time` exit 127 | ASKED; launcher re-hashed |
| removal of only empty failed-launch outputs before successful relaunch | ASKED user action, disclosed |
| regular deck copies and immutable absolute input/binary symlinks | UNASKED operational packaging; hash-guarded |
| uninstrumented ten-step run | UNASKED enabling control; required to prove WRITE-only identity |
| terminal `nn_stock=240` | UNASKED operational edit requested for terminal restart capture |
| port Lane-2/3 writers; add RGB, TKE, and Prather frames | ASKED coverage, exact |
| rebuild after stale FCM objects | UNASKED enabling correction; first run preserved/retracted |
| add seven `lwp` ownership guards | UNASKED enabling WRITE-only correction; second run preserved/retracted |
| derive `l3rea_dump` count as `(3+3*nlay_i+2*nlay_s)*npti` | ASKED independent-review correction; third run preserved/retracted |
| move adaptive implicit vertical advection to inactive and re-check every audit row | ASKED independent-review correction |
| route the restart-byte plant through `validate_identity` | ASKED independent-review correction; plant exits nonzero |
| report identity counts from the actual loops | ASKED independent-review correction |
| commit accepted/control launchers | ASKED independent-review correction |
| allocate EOS dump scratch only while armed | ASKED independent-review WRITE-only correction |
| compare native-IOM stored-value bytes with masks disabled | ASKED independent-review correction |
| replacement user-shell execution | ASKED; executed unchanged, RC 0 |
| final gate and planted mutations | ASKED mechanical acceptance |
| legoESM code, card, score, or physics change | UNASKED and forbidden in Phase 1; none performed |

## 11. Exact Phase-2 needs — no work performed here

Phase 2 must build the legoESM ORCA2 card on the already shared canonical
NEMO blocks. It may select existing faithful arms but may not add card-local
numerics or fork an operator. In particular it needs:

1. The NEMO WS-RK3 + QCO identity: exact `Kbb/Kmm/Krhs/Kaa` stage ordering,
   QCO thickness weighting, stage transports/WZV, 65-step split-explicit
   barotropic recurrence and correction, with this receipt's time-level
   registry used fail-closed.
2. The ORCA2 tripolar `184 x 152 x 31` geometry from
   `ORCA_R2_zps_domcfg.nc`, including north-fold/cyclic halos, masks, partial
   steps, row/metric coefficients, and explicit two-shard `domain_cfg.nc` /
   restart handling. Rank-zero detailed records require the documented halo
   strip/reassembly before comparison.
3. Lane-1 gates unchanged for WS-RK3 stage state/RHS/transport and
   split-explicit state identity; Lane-2 gates unchanged for EOS/BN2, SCO HPG,
   EEN/vector momentum, barotropic substeps/drag/mean, FCT transport, QSR and
   time-level registry; Lane-3 gates unchanged for SI3 bulk/exchange,
   thermodynamics and conservation, after correcting/validating the
   reassociation header ticket above. Every gate retains a planted failure.
4. Genuinely new ORCA2 forcing and boundary work: `sbcblk` NCAR/CORE fields
   (wind, air temperature/humidity, pressure, radiation, rain/snow and their
   bicubic/bilinear weights), surface-current feedback settings, runoff and
   river-mouth treatment, freshwater budget carry, geothermal bottom heat,
   RGB chlorophyll file plus analytical vertical profile, initial/restoring
   fields, internal-wave mixing, spatial viscosity, and the ORCA2 calendar.
5. Genuinely new coupled state: SI3 `jpl=5`, ten ice and five snow layers,
   every Prather first/second moment and reassociation operand, TKE/EVD/IWM
   carry, iceberg dynamics/trajectories and ocean exchange. The calving input
   must be present, while this executed short-run selector remains
   `nn_test_icebergs=10, ln_use_calving=F`; do not silently turn file-driven
   calving on.
6. A Phase-2 coverage gate driven from all 42 ocean, 210 SI3, and 21 iceberg
   restart variables plus all record schemas above. Every item must be
   VERIFIED or explicitly WAIVED; comparisons run fp64 under one shared
   compiler/arithmetic provenance policy and cannot mix oracle binaries.

The independent-review replacement run has passed its complete record,
identity, coverage, time-level, and planted-control gates. Phase 1 closes here.
No legoESM configuration, numerical score, or implementation follows this
receipt.
