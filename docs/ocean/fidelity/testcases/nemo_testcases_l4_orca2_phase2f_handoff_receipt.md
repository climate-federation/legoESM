# NEMO testcase Lane 4 — ORCA2 Phase-2f instrument-hygiene handoff

Date: 2026-09-06

Parent: `eac79aa7d006836cf2b58b0b6f09b480caa735ea`

Preregistration: `0f4f92a50`

Defined-cell gate: `d1a3506a2`

Twin launchers: `9ebf771d2`

Result: **STOP for two user-shell NEMO runs.**  The schema-fixed O1 record and
all source-defined inherited slots are valid.  The seven nondeterministic
streams now use canonical WRITE-only views, and two fresh 10-step directories
have been prepared from one rebuilt executable.  No legoESM numerical operator
was entered or changed.

## 1. Schema-fixed run validation

Run:
`/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_o1_schemafix_instrumented_10step_np2`

The retained user-shell run has `time.step=10`, `MPIRUN_RC=0`, `RUN DONE`, a
14.4-second wall time, and 92 records.  The O1 record has two complete frames:

- input header `(1,1,0,90,148,9,0,64)` and `9*90*148` f64 values;
- output header `(1,1,1,90,148,20,0,64)` and `20*90*148` f64 values;
- no trailing bytes, and every payload value finite;
- header-field-count and digest-bound one-ULP controls each return
  `PASS_NONZERO`.

Ordinary-output identity against the accepted icebergs-off uninstrumented
control remains exact under the frozen Phase-1 rules: the loop reports 4 / 4
restart shards exact and 8 / 8 history payloads exact apart from the registered
global creation timestamp.  Other ordinary files and normalized
`ocean.output` pass unchanged.

The interim inherited-record result is **CONFIRMED 84 / 91 raw, 91 / 91 on
source-defined bytes**.  The seven raw differences are exactly the registered
instrument-hygiene streams; no defined-cell difference remains.  The gate
reports rather than hardcodes these per-stream f64 counts:

| stream | compared source-defined | excluded canonical-zero class |
|---|---:|---:|
| barotropic advective mean | 5,627,627 | 3,745,432 |
| barotropic drag | 6,708,387 | 4,464,829 |
| ordered barotropic operands | 879,763 | 558,268 |
| barotropic substeps | 11,208,080 | 7,240,480 |
| post-`sbc` surface input | 272,183 | 240,497 |
| stage-3 `wzv` | 700,023 | 628,761 |
| slow forcing | 1,478,719 | 1,373,074 |

The exclusions are enumerated as rank-0 halo bands, masked land, and inactive
or unallocated schema components.  `wi` has zero elements under resolved
`ln_zad_Aimp=.false.` and therefore no frozen payload slot; the three stored
stage-3 arrays are both `ww` frames and `pFw`.  A one-ULP mutation in a compared
slot of each of the seven streams traverses the same comparator and returns
`PASS_NONZERO`.

The complete 994,665-byte gate JSON and stdout each have SHA-256
`c778b59ac130fbcaf21e184246e2b99fb03d025bf3a13feaacc928f4bbc85d89`.

## 2. Canonical WRITE-only implementation

The config-local helper at
`cfgs/ORCA2_OMIP_L4/MY_SRC/l4_oracle_canon_subroutines.h90:1-79` creates an
automatic result initialized entirely to `0._wp`, maps full, `A2D(1)`, and
`A2D(0)` shapes by their halo offset, and copies an operand only where the
rank-0 cell is both owned and wet on its T/U/V/F staggering.  Its arguments are
`INTENT(in)`.  It never assigns a model field.  The copied writers use the view
only in their stream `WRITE` lists:

- `stp2d.F90`: slow forcing;
- `stprk3.F90`: final surface inputs, retaining explicit zero-only iceberg
  placeholders for `ln_icebergs=.false.`;
- `traadv.F90`: `ww` and `pFw`, with the inactive unallocated `wi` never
  dereferenced (`traadv.F90:228-248`);
- `dynspg_ts.F90`: substeps, drag, advective mean, and ordered operands.

Frozen magic, headers, field order, frame count, and payload counts do not
change.  The source digests are:

| copied source | SHA-256 |
|---|---|
| canonical helper include | `202bd34777188124fe15260d9834a96e86a6fa03c2b20d5311105d5cd652418c` |
| `stp2d.F90` | `552bda13b973699cf7c383d659dfc476271f4167b7a9df2eada6961b23581763` |
| `stprk3.F90` | `0c547e53b477e69070a9f5f59039faacb9e4f9fbd84290dbea2cdcf0c23e4675` |
| `traadv.F90` | `4b47090d5b638be9fa6938fb82d35b0feb0d58ba72e332201f5049a6444d1fd8` |
| `dynspg_ts.F90` | `596bb5bcad9d1795810fa9a8b1aa84cce7f0ca9fdd0cb99ed1a8cbce4203f6d8` |

## 3. Scalar-math build

The copied configuration was rebuilt with `makenemo -n ORCA2_OMIP_L4 -m
conda-scalarmath`.  The log records `-fdefault-real-8 -O3
-funroll-all-loops -fno-tree-vectorize`; `nm -D` reports zero `_ZGV*` symbols.
The recurring local Conda entry-point warning is retained in the successful
build log.

| artifact | bytes | SHA-256 |
|---|---:|---|
| `build/nemo_ORCA2_OMIP_L4_phase2f_canonical.exe` | 55,129,776 | `5befe21268dc7e2487525aadf4dec4fb1660451eb70bc33c2ffa8e6ceda196c5` |
| `build/build_ORCA2_OMIP_L4_phase2f_canonical.log` | 7,388 | `eb4e03322be645cab982501843ae51bd632617f28ad876908520d07248fc2cc5` |
| `build/phase2f_canonical_MY_SRC.sha256` | 2,270 | `5a9bbfdaed3e7d4da05b6fb389fdd7cb058255b9bbcb33a29809220fe1d563a9` |
| `build/phase2f_canonical_ZGV.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

## 4. Prepared twin runs

Both directories contain 19 copied deck files, 40 absolute immutable input
symlinks, one absolute binary symlink, two manifests, and the launcher: 63
prepared entries and no run output.  The resolved namelists are the same
icebergs-off VARIANT (`ln_icebergs=.false.`), still with `nn_itend=10`,
`nn_stock=10`, two ranks, `jpni=2`, `jpnj=1`, one thread per library, CPU, and
Bash timing.  The launchers check the binary, deck manifest, every deck file,
the input manifest, and every input before `mpirun -np 2 --oversubscribe`.

| arm | launcher SHA-256 | prepared-manifest SHA-256 |
|---|---|---|
| A | `c1bf68b6a71e31813bb00ac46355c8bd39d1576bdfd6362d99ea1a91b2ab84da` | `cc6b165c962fc638df48490f4375d8ddbc6a9a287de1fca52e2b2f6fb7ff2589` |
| B | `2abd81f23c9c4fbbeee81cb9df5c3be659788c169a2d2764a1ff5b2fb8ecc62d` | `9ca72ba32c2a539b93ac1b7ad09c9f2a56e14b7e0bf1c863f6f99b56eae1b160` |

The different launcher digests encode only their deliberately different
absolute `EXPECTED_DIR` values.  The binary and all deck/input digests are
identical.

Run A and B one at a time, unchanged:

```text
/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2f_canonical_a_10step_np2/run.sh
/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2f_canonical_b_10step_np2/run.sh
```

## 5. Next mechanical boundary

On resume, the lane must schema-walk all 92 records in both runs, require 92 / 92
raw-byte record identity between A and B, run all O1/inherited/ordinary plants,
and independently bind each run to the uninstrumented VARIANT control.  Only
then may these canonical records supersede the schema-fixed instrument run and
the O1 `fld_read` mapping ladder resume.  The ordered numerical needs remain:
mapped nine CORE fields, certified Lane-3b NCAR bulk, RGB QSR, shared EOS/HPG,
external mode with tripolar north fold, transports, FCT, BBL, then TKE/EVD/IWM
entry.  Shared-operator debt is registered for GYRE; only ORCA2-owned fold,
`fld_read`, RGB chlorophyll, runoff, geothermal, IWM, BBL, and icebergs-off
setup may be changed here.  SI3 remains `UNMEASURED_PENDING_ICE_MERGE` with
oracle-supplied exchange fields.

## 6. ASKED / UNASKED

| item | status | disposition |
|---|---|---|
| canonicalize all undefined slots | ASKED | zero-first WRITE-only views on owned wet regions; inactive allocations never dereferenced |
| interim compare defined cells and enumerate exclusions | ASKED | 91 / 91; actual per-stream counts above |
| two reproducibility runs if cheap | ASKED | two 10-step directories prepared from one executable |
| O1 two-frame schema and plants | ASKED | PASS; two binding plants |
| ordinary-output identity | ASKED | 4 / 4 restart and 8 / 8 history counts derived by loops |
| continue ocean ladder | ASKED after hygiene | deferred at required NEMO rerun boundary |
| execute MPI from agent shell | UNASKED and prohibited | not done |
| change legoESM arithmetic or shared operators | UNASKED and forbidden | not done |
| delete/supersede prior run artifacts | UNASKED and forbidden | none deleted; prior runs remain retained |
