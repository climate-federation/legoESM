# NEMO testcase Lane 4 — ORCA2 Phase-2m EEN acquisition handoff

Date: 2026-09-06

Parent: `ce0f353e25a282b80ce4d134d1f937af24d1ed5e`

Preregistration commit: `efb268b54`

Status: **STOP FOR USER-SHELL MPI ACQUISITION.**  The two icebergs-off twin
directories are prepared and hash-guarded.  No MPI/NEMO run was launched from
the sandbox.  Tracer ownership, BBL, and ZDF/TKE/IWM were not entered across
this acquisition boundary.

## 1. Crash recovery and repository state

At recovery, the ordinary worktree resolved to `ce0f353e25a` and the localgit
branch also resolved to that parent before the preregistration commit.  The
worktree was clean apart from the preregistration file represented through the
separate localgit object store.  `/tmp` had 136 GiB free (39% used) before
large writes.  The explicitly flagged `/tmp/codex-orca2-r21` probe worktree was
not read, modified, or deleted.

Every artifact written around the interrupted window was re-read.  No receipt,
preregistration, gate, launcher, or MY_SRC source was truncated.  The first
build-environment attempt selected the system Perl and stopped before NEMO
compilation because `Text::Balanced` was unavailable; it produced no retained
measurement or build log.  The in-flight scalar-math build subsequently
reached `Compilation successful`, but was not admitted alone: the identical
command was rerun incrementally and then a second, fresh 17-MiB source/config
copy was compiled completely in an independent BLD tree.  The fresh build also
ended successfully.  No partial build or gate result is admitted as evidence.

One recovery-process deviation is explicit.  The first incremental rerun used
the original full-build log pathname, replacing that log before it was copied.
The initial full-build text is therefore not retained (its then-measured hash
was `481404ed...`).  The incremental rerun log and the independent 134-second
full rebuild log are both retained and hash-pinned below.  No source, binary,
run directory, record, or user result was removed or overwritten.

Two fail-before-compile rebuild-staging attempts are also flagged.  An
alternate `-t` BLD request was rejected because `makenemo` treated it as a new
configuration; its log pathname was then inadvertently reused.  The retained
failed `build_ORCA2_OMIP_L4_phase2m_een_rebuild.log` is the next attempt, which
correctly refused a fresh mini-tree missing `cfgs/ref_cfgs.txt` and
`cfgs/work_cfgs.txt`.  After those index files were copied, `rebuild2.log`
records the complete successful build.  Neither failed attempt compiled or
ran NEMO, and neither is measurement evidence.

## 2. Why these are the requested external-EEN operands

The source walk makes one citation correction explicit:

- `dynvor.F90:918-950` allocates and constructs the masked, folded, nonzero
  reference divisor `e3f_0vor` for `np_EEN`.
- `domqco.F90:233-246` constructs live F-point `r3f`; the executing QCO macro
  at `domzgr_substitute.h90:125-130` defines
  `e3f_vor=e3f_0vor*(1+r3f*fe3mask)`.
- `dynspg_ts.F90:1326-1379`, not `dynvor.F90:448-535`, is the executing frozen
  external-mode EEN builder.  It evaluates each `q=ff_f/e3f_vor`, forms four U
  and four V `zpvo` triads, and accumulates the eight coefficients handed to
  GYRE.  `dynvor.F90:448-535` is the distinct 3-D momentum `vor_ene` program;
  dumping that arm would not discriminate the frozen external coefficients.

This agrees with the committed GYRE Round-21 request at `e2378f057ca`, which
identifies `dynspg_ts.F90` as the live consumer and holds after the live-divisor
arm improves all eight rows without reaching the bar.  The acquisition
therefore records the operands of the actual over-bar boundary.

## 3. WRITE-only instrument

The shipped checkout remains untouched.  The build-only tree is
`/data/abyssal/dbalwada/nemo-testcases-l4/build/nemo_5.0.2_phase2m_een`.
Its configuration-local override is:

`cfgs/ORCA2_OMIP_L4/MY_SRC/dynspg_ts.F90`

The source changes pass `kt` into the already executing
`dyn_cor_2D_init` call (`:306`, with the nonexecuting preprocessor alternative
at `:387`), then:

- arm only `lwp .AND. kt==nit000 .AND. nvor_scheme==np_EEN` at `:1526`;
- allocate eleven writer-local arrays only while armed at `:1529-1533`;
- zero every slot before copying at `:1535-1541`;
- copy `e3f_0vor`, materialize live `e3f_vor`, and evaluate the exact scalar
  `q` division at `:1542-1547`;
- copy the ordinary U and V `zpvo` scalars immediately after they are formed,
  at `:1571-1575` and `:1604-1608`; and
- write the four frozen streams and deallocate every temporary at
  `:1720-1762`.

The added assignments target only `l4_*` arrays, I/O metadata, and the local
arming flag.  No NEMO model array, coefficient, scalar before its ordinary use,
or prognostic field is assigned.  This is a WRITE-only instrument.

Source hashes:

| artifact | SHA-256 |
|---|---|
| inherited ORCA2 config override | `c1542a517a40627aeea099c0955728cda85eb49b3ec5aa9d98068f371dbcb74c` |
| Phase-2m config override | `3430c945df0d8e2dbdb7fe6f23599bcf24af19f81d4c2a16555c10c7453456c1` |
| `arch-conda-scalarmath.fcm` | `132f7a0500c4f0e86d8d3bf7864974a82e1dea5d83166dcfdfaf409e2ca04561` |

## 4. Frozen schema and canonical slots

All four files have a 16-byte blank-padded magic followed by ten native
int32 values `(version,kt,Kmm,nvor,jpi,jpj,jpk,nfields,bits,payload_count)`.
For this two-rank decomposition the common header resolves to version 1,
`kt=1`, `Kmm=1`, `nvor=3`, `(jpi,jpj,jpk)=(94,152,31)`, and binary64.
Each full field has `94*152*31 = 442,928` values.

| stream | magic | ordered payload | derived bytes | level |
|---|---|---|---:|---|
| `oracle_een_e3f0vor_kt00000001.bin` | `NEMO_L4_E3F0_1` | `e3f_0vor` | 3,543,480 | static/reference |
| `oracle_een_e3fvor_kt00000001.bin` | `NEMO_L4_E3FV_1` | live `e3f_vor` | 3,543,480 | derived from `Kmm=1` |
| `oracle_een_q_kt00000001.bin` | `NEMO_L4_QEEN_1` | `ff_f/e3f_vor` | 3,543,480 | derived from `Kmm=1` |
| `oracle_een_zpvo_kt00000001.bin` | `NEMO_L4_ZPVO_1` | U `nw,ne,sw,se`; V `nw,ne,sw,se` | 28,347,448 | derived from `Kmm=1` |

The arrays retain Fortran `(i,j,k)` order.  The two-cell halo is zero in every
field.  The unused `jpk` slot is zero in live `e3f_vor` and `q`.  Each `zpvo`
field is nonzero only on the corresponding source-defined U or V face through
`mbku`/`mbkv`; dry, below-bottom, halo, and inactive slots are zero.  The
committed gate derives those face-defined slots from `mesh_mask_0000.nc` and
checks header, exact EOF size, finiteness, and canonical zeros before scoring.

Its binding plants alter (1) a header integer, (2) a defined payload bit
through the twin raw-identity checker, and (3) a canonical halo zero.  Each is
preregistered to exit nonzero through the production checker after acquisition.

## 5. Scalar-math build

Command, in the isolated build-only NEMO copy:

```text
PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/bin:/bin \
./makenemo -n ORCA2_OMIP_L4 -m conda-scalarmath -j 8
```

The initial build ended with `Compilation successful` and recorded a
130-second total.  Its mandatory post-crash incremental rerun also ended
successfully.  A clean-room copy at
`/data/abyssal/dbalwada/nemo-testcases-l4/build/nemo_5.0.2_phase2m_een_rebuild`
then compiled all 474 dependency-scanned files from the same source hash in
134 seconds; its log is
`build_ORCA2_OMIP_L4_phase2m_een_rebuild2.log`, SHA-256
`aae8b7cc599d9ce1f6278fe6cb3037b0fecef6150f56e1fefe62dac9dec15a01`.
The compiler lines contain `-fno-tree-vectorize`, and both completed binaries
have zero `_ZGV*` dynamic symbols.  Their overall ELF hashes differ because
the independent build embeds its distinct absolute build root; no
cross-directory binary-reproducibility claim is made or required.  The staged
binary, retained unchanged throughout and used by both twin launchers, is
`/data/abyssal/dbalwada/nemo-testcases-l4/build/nemo_ORCA2_OMIP_L4_phase2m_een.exe`,
SHA-256 `309fdc81d667789e3bc94206ea14e0f2e07ec92234566af7d90953b7dbc3c4b0`.
`nm -D` reports **0 `_ZGV*` dynamic symbols**.  The incremental verification
log hashes `b64dec1516055527b83b25550b7bda4ac0076a4f83ea3d733d0167fed3dd1abb`.

## 6. Prepared twins

Run A:

`/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2m_een_a_10step_np2`

Run B:

`/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2m_een_b_10step_np2`

Both retain the exact accepted Phase-2j V2 deck: `ln_icebergs=.false.`,
`nn_itend=10`, `nn_stock=10`, `nn_istate=1`, `rn_Dt=10800 s`, and
`nn_fsbc=2`.  Deck files are byte copies; the 40 deck inputs are absolute
read-only symlinks into the unpacked `ORCA2_ICE_v5.0.0` input root; `nemo` is
an absolute symlink to the staged binary.  The deck and input manifests hash
`e2cb4c552360491fa9dcea0649661e5f44a9d972769d40a4ecfc2aa70a097059`
and `3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5`.
Both were checked in place.

Each `run.sh` refuses the wrong directory or pre-existing outputs, verifies
the binary and both manifests, fixes all thread counts to one, and launches:

```text
mpirun -np 2 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
```

under Bash's `time` keyword, appending `MPIRUN_RC`, completion UTC, and
`RUN DONE`/`RUN FAILED` to `run.user.time.log`.  NEMO's iceberg initialization
constraint makes the previously resolved `(jpni,jpnj)=(2,1)` the smallest
supported layout; it is an oracle fact rather than a discretionary choice.

## 7. Admission after the user runs both directories

The acquisition is not pinned until all conditions pass:

1. both launchers finish at `time.step=10`, `MPIRUN_RC=0`, with four active
   restart shards (two ocean and two SI3; TOP is compiled out);
2. all four new schemas pass and the two new record sets are byte-identical;
3. all 93 inherited V2 records are raw-byte identical between twins and to
   the Phase-2j V2 root;
4. the full twin inventory is **97 / 97** raw-byte identical;
5. Phase-1 `validate_identity` reports ordinary outputs and restarts identical
   to the Phase-2j identity control, dynamically counting actual files; and
6. all three new plants and the inherited identity plants exit nonzero.

There are no consumed-field exceptions in the ORCA2 inventory: systematic
canonicalization already removed every undefined slot from the 93 inherited
records.  The GYRE Round-21 exception for a pre-consumer `zFw` slot does not
apply here.  Raw identity is therefore the admission rule for every ORCA2
record.

## 8. ASKED / UNASKED

| item | status | disposition |
|---|---|---|
| EEN primitive discriminator dumps | ASKED | four canonical WRITE-only streams staged |
| user-shell twin MPI runs | ASKED | two directories prepared; not executed here |
| Round-21 consumed-field admission rule | ASKED | no ORCA2 exceptions; all 97 must be raw exact |
| source citation check | ASKED | actual external `q`/`zpvo` consumer is `dynspg_ts`, documented above |
| repair shared external-mode code | UNASKED and forbidden | no legoESM physics changed |
| tracer/FCT ownership walk | ASKED after acquisition | held behind this rerun boundary |
| BBL and TKE/EVD/IWM continuation | ASKED after tracer | not entered |
| read or clean `/tmp/codex-orca2-r21` | explicitly forbidden | flagged only; untouched |
| delete prior artifacts | forbidden | nothing deleted |
| shipped NEMO edits | forbidden | none; isolated build-only copy used |
| overwrite initial build log during recovery rerun | UNASKED deviation | disclosed above; recovered with a retained independent full rebuild |

## Stop receipt

This handoff ends before any MPI execution and before any legoESM operator
change.  Run A and B one at a time with their unchanged `run.sh` files, then
resume Lane 4 for the 97-stream admission and the source-ordered discriminator
walk.
