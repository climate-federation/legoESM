# NEMO testcase Lane 4 ORCA2 oracle — Phase 1 preregistration

Date frozen: 2026-09-05 (America/New_York)  
Lane: `ORCA2_ICE_PISCES` -> `ORCA2_OMIP_L4`  
Scope: NEMO 5.0.2 oracle definition, build, run, records, and provenance only.
No legoESM card or legoESM numerical comparison is in scope.

This document is frozen before the first `ORCA2_OMIP_L4` model execution. It
follows the Lane-1 oracle convention and Rule 12 of the oracle-fidelity skill:
the reference is one compiler-wide scalar-math NEMO binary, all records are
native binary64 values written by that binary, and no field may be silently
substituted from another executable, compiler, precision, or source tree.

## Registered oracle definition

The shipped reference is `cfgs/ORCA2_ICE_PISCES` at NEMO git
`dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796`. The copied configuration is
`ORCA2_OMIP_L4`. Its CPP keys will be exactly:

```text
key_si3 key_qco key_vco_1d3d key_RK3
```

Relative to the shipped key list, only `key_xios` and `key_top` are excluded.
As in Lanes 1 and 2, removing `key_xios` selects NEMO's compiled native-IOM
`dia_wri` path (`src/OCE/stprk3.F90`, the `#if ! defined key_xios` arm) rather
than requiring an external XIOS service. Native restart writing remains
enabled.

`key_top` is excluded only if the completed static audit and resolved namelist
prove that PISCES is passive with respect to physical ocean and SI3 evolution.
The preregistered decisive facts are:

- `cfgs/SHARED/namelist_ref:67` defaults `ln_top=.true.`, but
  `cfgs/SHARED/namelist_ref:427` defaults `ln_qsr_bio=.false.` and ORCA2 does
  not override it. Thus `src/OCE/TRA/traqsr.F90` must select the configured RGB
  chlorophyll light path, not TOP-provided irradiance.
- `cfgs/SHARED/namelist_ref:203` has `ln_dm2dc=.false.` and
  `cfgs/SHARED/namelist_top_ref:112` has `ln_trcdc2dm=.false.`, excluding the
  TOP daily-shortwave feedback guarded in `src/OCE/SBC/sbcblk.F90`.
- ORCA2 sets `ln_trcldf_tra=.true.` while TOP's reference has
  `ln_trcldf_OFF=.false.`; the `key_top` guard in `src/OCE/TRA/trazdf.F90`
  therefore reduces to the same active-tracer slope condition.
- the compatibility-only `key_top` checks for NPC/OSM/MFC vertical physics are
  inactive because all three selectors resolve false.

If any executed TOP-to-OCE or TOP-to-SI3 feedback is found, or if disabling
TOP changes any physical selector beyond the already registered `ln_top`, the
oracle definition is **REFUTED** and work stops. No alternative key or namelist
will be improvised.

`nn_ice=2` and `key_si3` are retained. The oracle is therefore ORCA2 ocean +
SI3, with the shipped five-category SI3 defaults, not an ocean-only oracle.
`key_RK3` plus `key_qco` registers the WS-RK3 path; the MLF path is excluded.

## Inputs and immutable deck

The user supplied the published archive at
`/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0.tar.gz`:

```text
bytes   1365673011
MD5     872eea51f22c5dabaffd268c0220eba6
SHA256  5d47eab85c591fe0fd7e63a80892f3264b6387edf93cbb975a71b2f2f5fdf1a4
URL     https://gws-access.jasmin.ac.uk/public/nemo-vol1/sette_inputs/r5.0.0/ORCA2_ICE_v5.0.0.tar.gz
fetch   user's shell, 2026-09-05; agent sandbox had no network
```

It is unpacked, without modifying the archive, below the same `inputs`
directory. Every file actually opened through a resolved `sn_*` descriptor,
domain configuration, initial condition, restart, or weights descriptor will
be listed with byte count and SHA-256. Archive members named only in inactive
namelist arms will be listed separately as excluded, never deleted.

The receipt will preserve verbatim copies of `namelist_cfg`,
`namelist_ice_cfg`, `namelist_ref`, and `namelist_ice_ref`, plus NEMO's resolved
`output.namelist.dyn` and `output.namelist.ice`. Any target deck change other
than the registered CPP exclusions and run-control edits below is a hard fail.

## Build registration

Build command family:

```text
./makenemo -r ORCA2_ICE_PISCES -n ORCA2_OMIP_L4 -m conda-scalarmath ...
```

The environment is conda `nemo-build`, using its `mpif90`. The arch file must
be `arch/arch-conda-scalarmath.fcm`, byte-recorded in the manifest, and must be
the established `arch-conda.fcm` plus compiler-wide
`-fno-tree-vectorize`. The binary is accepted only if `nm -D` finds exactly
zero `_ZGV*` symbols. The uninstrumented executable is built and retained
first; the same copied configuration is then rebuilt after config-local
`MY_SRC` WRITE-only instruments are installed. Both executables and both build
logs are retained and hash-pinned. Shipped source and shipped configuration
files are never edited.

## Registered executions

All runs are CPU-only, one MPI process, invoked directly unless NEMO requires
`mpirun -np 1`. A larger layout is not authorized automatically: if one rank
cannot run, the smallest working layout and the one-rank failure must be
recorded before proceeding.

1. **Instrumented ten-step oracle:** copied shipped deck with only
   `nn_itend=10`, `nn_stock=10`, and `nn_istate=1`; all other scientific
   namelist values unchanged.
2. **Uninstrumented ten-step identity control:** identical ten-step deck but
   `nn_istate=1`, executed with the retained uninstrumented binary. Its ocean
   and ice restart payloads must be byte-identical to the instrumented run.
   This extra control is the direct non-vacuity check that the writers do not
   perturb model arithmetic.
3. **Uninstrumented 30-day reference:** 240 steps at the shipped
   `rn_Dt=10800 s`, with `nn_itend=240` and `nn_stock=240`; no other scientific
   namelist change. The stock edit is the requested terminal-restart capture,
   not a physics choice.

For every run, the receipt will record command, environment, one-rank layout,
start/end timestamps, `/usr/bin/time -v` wall time, exit status, terminal
iteration, `ocean.output` stability/control numerics, and all restart hashes.

## Registered WRITE-only record set

All streams use Fortran unformatted stream records: a 16-byte ASCII magic,
32-bit integer version/time-level/dimension metadata, then native Fortran
column-major IEEE-754 binary64 payloads. Every stream hard-fails unless
`STORAGE_SIZE(1._wp)==64`. Time levels are recorded numerically in each header;
the receipt maps them to `Kbb`, `Kmm`, `Kaa`, and `Krhs` at the exact call
site. No model prognostic or exchange array is assigned by a writer.

Reused, source-identical Lane-2 instruments where the same routine executes:

- `stprk3.F90`: per-step entry state; `stp2d` barotropic frame; three Kaa
  stage states; stage-1 RHS; post-`zdf_phy` coefficients.
- `stp2d.F90`: stage-1 slow/barotropic forcing operands and the substep
  integration frame.
- `dynspg_ts.F90`: every split-explicit substep, including `un_adv/vn_adv`.
- `stprk3_stg.F90`: WS-RK3 stage operands, pre-update RHS, corrected Kaa,
  transports, tracer boundaries, and QSR before/after frames.
- `eosbn2.F90`, `dynhpg.F90`, and `traadv.F90`: executed EOS/BN2, SCO HPG,
  tracer-advection, WZV, and stage-3 transport operands.

`usrdef_sbc.F90` is explicitly not ported because ORCA2 executes `sbcblk`.
ORCA2/SI3-specific frames are:

- Lane-3 `sbcblk.F90` + `icesbc.F90` NCAR-bulk operands;
- Lane-3 corrected `icestp.F90` exchange frame, including deterministic
  inactive/invalid halos and the guarded `rCdU_ice` payload;
- Lane-3 `icethd.F90` global category/layer thermodynamic entry, internal, and
  exit frames;
- a Lane-4 RGB frame at the chlorophyll read and `tra_qsr` application;
- a Lane-4 Prather frame containing every first and second moment at entry and
  exit of the selected `icedyn_adv_pra` arm;
- a Lane-4 closure frame adding `en` (TKE) to `avm/avt/avs` after
  `zdf_phy`.

The inventory is accepted only if all ten step-entry files, 30 stage files,
ten barotropic-frame files, all executed substep frames, all five-category SI3
state/exchange frames, the Prather moments, TKE, RGB chlorophyll, and every
registered kt=1 detailed operand stream exist exactly once and parse fully.
Internal scheme-local work arrays that are not persistent state may be WAIVED
only by name, source lifetime, and producer/consumer boundary in the final
coverage register.

## Outcomes fixed before measurement

**CONFIRMED** means all of the following hold:

- exact key/deck definition above; completed source audit finds no active TOP
  physics feedback;
- both builds exit zero, both binaries are binary64 scalar-math builds, and
  each dynamic-symbol census has zero `_ZGV*` entries;
- every active input exists and is hash-pinned; no unlisted input is opened;
- the three registered runs exit zero at steps 10, 10, and 240 with no NEMO
  control failure or non-finite physical state;
- instrumented and uninstrumented ten-step ocean/ice restarts are byte-exact;
- every required record is present, uniquely registered, header-valid,
  correctly sized, finite on its defined domain, and assigned an explicit
  time-level disposition.

Any failed clause is **REFUTED** or **UNMEASURED**, never downgraded to
PLAUSIBLE. Static source conclusions without a direct execution witness are
labeled **PLAUSIBLE**. No Phase-2 fidelity claim is made in this phase.

## Planted controls

The committed record gate must exit nonzero for each independently planted
violation: malformed magic; wrong time-level id; truncated payload; appended
trailing byte; missing registered file; extra unregistered file; one binary64
NaN in a defined payload; and a one-ulp mutation of one active SI3 category
state. The restart-identity check also receives a one-byte planted mismatch and
must exit nonzero. A control is not counted unless its mutation lands in a
field/domain the passing gate reads.

## ASKED / UNASKED choice register

| Choice | Status | Registered disposition |
|---|---|---|
| shipped `ORCA2_ICE_PISCES` copy named `ORCA2_OMIP_L4` | ASKED | exact |
| remove `key_top` and `key_xios`; retain other shipped keys | ASKED | exact; fail if TOP feedback exists |
| retain `nn_ice=2`, SI3 five-category physics | ASKED | exact |
| scalar-math arch and zero `_ZGV*` | ASKED | exact |
| supplied archive provenance and checksums | ASKED | exact |
| CPU, smallest supported MPI layout | ASKED | try one direct rank first |
| ten-step instrumented run controls | ASKED | exact |
| uninstrumented 30-day run | ASKED | 240 steps derived from shipped dt |
| uninstrumented ten-step identity run | UNASKED enabling control | retained; proves WRITE-only identity |
| terminal `nn_stock=240` for 30-day run | UNASKED operational edit | retained solely for requested restart hash |
| reuse Lane-2 and Lane-3 writers | ASKED | exact where executed routines match |
| RGB, Prather, TKE frames | UNASKED enabling coverage | retained because ASKED coverage names them |
| output without XIOS | ASKED by key exclusion | native IOM, matching Lanes 1/2 |
| any legoESM card or numerical score | UNASKED/out of scope | forbidden in Phase 1 |

## Stop boundary

Phase 1 ends with the oracle receipt, preregistration, coverage/time-level
registers, artifact manifest, local-git commit, and review bundle. Phase 2 is
listed only as future requirements. No legoESM configuration or model code is
created or executed.
