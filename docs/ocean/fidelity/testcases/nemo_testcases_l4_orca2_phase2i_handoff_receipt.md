# NEMO testcase Lane 4 — ORCA2 Phase-2i handoff receipt

Date: 2026-09-06

Parent: `c8be7befc8c730f80ab72a952a7c8f9104e6203a`

Status: **STOP — replacement oracle twins require user-shell execution; the
ocean ladder's first new debt is a shared external-mode boundary.**  No
legoESM SI3 or iceberg operator was entered.  VARIANT V2 remains deliberately
unpinned until the replacement twins prove 92 / 92 raw-record identity.

## 1. Systematic-twin disposition and RHS cause

The completed Phase-2h twins each contain the frozen 92-record inventory.  A
fresh run of the committed streaming reproducibility gate reports 91 / 92 raw
exact.  Its JSON digest is
`d220672a86b7a68d16da49f938a490ee22c86da671a8080077f750709de76dcb`.
The sole difference is:

| stream | A SHA-256 | B SHA-256 | bytes |
|---|---|---|---:|
| `oracle_rhs_kt00000001.bin` | `18a961621997f0f52d1293fec6aaeeb4424b24fb79cb469c6d75c73bfdfd6e5b` | `006fca37d1a3e337f7be3ce47a528e423266d03c2debf9baa915fb0ac47c4979` | 7,086,892 |

Both headers are valid: `NEMO_L1_RHS___1`, version 1, `kt=1`, `Krhs=3`,
`jpi=94`, `jpj=152`, `jpk=31`, binary64.  The two `uu/vv` payloads already
pass through zero-first `l4_canon_3d` views, but `l1_dump_rhs` alone lacked the
rank-zero visibility return.  Thus both ranks could concurrently
`STATUS='REPLACE'` the same pathname.  The repair adds only
`IF (.NOT.lwp) RETURN` before `OPEN` in the configuration-copy
`MY_SRC/stprk3.F90`; it assigns no model array and changes no model expression,
call site, header, or payload.  This is a WRITE-only instrument repair.

The other 91 streams are reproducible evidence, but are not separately named
V2: campaign acceptance is atomic over the frozen inventory.  The Phase-2h
runs remain preserved as `CANONICAL_CANDIDATE_91`, superseded for final
acquisition rather than deleted.

## 2. Scalar-math rebuild and replacement twins

The repaired `ORCA2_OMIP_L4` configuration copy was rebuilt with
`conda-scalarmath` (`-fno-tree-vectorize`).  Dynamic-symbol inspection finds
zero `_ZGV*` symbols.

| artifact | bytes | SHA-256 |
|---|---:|---|
| `nemo_ORCA2_OMIP_L4_phase2i_rhsrank0.exe` | 55,519,664 | `ea6953d6872f97d11f8acade9f2a71c73fb04b2298a2e7f30e1656e4301ccbc9` |
| successful build log | 4,643 | `d38af430cb5637bbba5ca9f6e8df5fc9b5057180a03ae8d88e56d4c80a8b11e6` |
| 15-file `MY_SRC` manifest | 2,270 | `e62c231467cd653c83faf49a8f3e78a4efdba3f6a083290d6dde4ebafd1234ed` |
| zero-line `_ZGV` result | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

An initial build invocation used the base environment, failed before compile
because FCM could not load Perl `Text::Balanced`, and was immediately retried
with the documented `nemo-build` environment.  The same redirected log path
was reused, so the successful redirection overwrote that failed diagnostic
log.  This accidental artifact-preservation miss is declared here; no run,
oracle record, input, shipped file, or prior campaign artifact was removed.

Two independent directories are prepared.  Namelists and XML are copied;
the 40 deck inputs and exact binary are symlinked and hash-guarded.  Each
launcher requires CPU, one OpenMP thread, `jpni=2`, `jpnj=1`, two MPI ranks,
bash `time`, an empty output namespace, and the expected working directory.

| arm | prepared-files SHA-256 | launcher SHA-256 |
|---|---|---|
| `variant_icebergs_off_phase2i_rhsrank0_a_10step_np2` | `c240d73b1e581f194ecfb52d6a0922a0c560ca1c23aefa8599a037876b6a36a7` | `a776c4005814a6d80ead6a20e28426869108f8c9ba4b2b71d9e99fe42ccc0544` |
| `variant_icebergs_off_phase2i_rhsrank0_b_10step_np2` | `6c680ad31edec5038e65a3e43783653dcb77f359fc63febcb8029a067cecd61c` | `a836815b2e641d23cdf7862c577d6921e8696471c76a1f1165c5267ec21cba43` |

After execution, each run must complete ten steps, pass the full Phase-1
schema/header walk, ordinary-output identity against the uninstrumented
icebergs-off control, and all plants.  Only 92 / 92 raw-identical records
between A and B permits arm A to be pinned as `VARIANT_ORACLE_V2`.

## 3. ORCA2 RGB source identity

The shared `nemo_qsr_rgb` arm now transcribes the executing
`qsr_RGBc` program in `cfgs/ORCA2_OMIP_L4/MY_SRC/traqsr.F90:255-496` rather
than reusing the generic chlorophyll parameterization.  It retains that
generic arm unchanged.  The shared implementation carries NEMO's statement
order through `nemo_source_round`, including the Morel-Berthon profile,
`NINT(41+20*LOG10(chl)+1.e-15)`, the 61-row B/G/R table, scalar-libm
`log`/`log10`/`exp`, resolved extinction levels 2/8/19/22, band recurrence,
next-W-level mask, and final source insertion.  The card supplies the live
`gdepw/e3t` and reference ladders; there is no ORCA2 conditional in the
operator.

The record schema correction is material: `oracle_rgb_chl` contains one full
`jpi*jpj` chlorophyll field, then one A2D(0) `qsr`, a 31-value `gdepw_1d`, and
one full `jpi*jpj` `r3t`.  The old parser's total byte count happened to match
despite swapping full and A2D shape classes.  The QSR target is the actual
writer expression `(RHS_before + source_increment) - RHS_before`, so the gate
uses the stage-3 tracer record's pre-QSR temperature RHS and applies the final
source-rounding subtraction.

Result under production JIT, CPU, fp64 and scalar-libm: **0 / 233,341**
rank-zero owned wet T cells, maximum absolute error 0.  Measurement JSON
SHA-256 is
`9f19c5c51d96a1ea9a66eb95263c9a6f4e540b15cf68a0d1fc008a3fa6c11d40`.
The oracle/oracle one-ULP plant exits 1 at 1 / 233,341 (stdout SHA-256
`e5cffa2a0c66073742c9bbc0024ad0b106103a70d1e1b513b410ef3309df59bd`).
This boundary is **CONFIRMED AT_BAR, ORCA2_OWNER**.

## 4. Continued ocean ladder

O1-M remains the already-certified nine-field `fld_read` mapping (nine times
0 / 13,320).  O1-B remains `LANE3B_OWNER` debt and is bypassed only with its
oracle frame-2 outputs.  SI3 aggregate exchange fields remain
`ORACLE_SUPPLIED`; SI3 stays `UNMEASURED_PENDING_ICE_MERGE`.

All new rows use production JIT, CPU, fp64 plus scalar-libm, exact f64
cellwise comparisons, and the rank-zero owned/stencil-valid domain:

| order | boundary | result | owner/disposition |
|---:|---|---|---|
| 1 | O1-M nine CORE `fld_read` inputs | nine × 0 / 13,320 | `ORCA2_OWNER`, certified in Phase-2h |
| 2 | O1-B NCAR bulk | DEBT | `LANE3B_OWNER`; oracle bulk outputs supplied |
| 3 | O2 `tra_qsr` RGB | 0 / 233,341 | `ORCA2_OWNER`, CONFIRMED AT_BAR |
| 4 | O3 stage-2 EOS-80 source program | thirteen × 0 / 233,341 | `GYRE_OWNER_SHARED_EOS`, CONFIRMED AT_BAR |
| 5 | O4 stage-2 SCO HPG | three U fields × 0 / 221,640; three V fields × 0 / 222,048 | `GYRE_OWNER_SHARED_HPG`, CONFIRMED AT_BAR |
| 6 | O4-EXT-A frozen EEN coefficients | all eight fields DEBT; first `ffu_nw` 8,308 / 8,568, max abs `8.056177274927652e-05` | `GYRE_OWNER_SHARED_EXTERNAL_MODE`, first new over-bar boundary; STOP |
| 7 | external recurrence/north fold, stage transports, FCT, BBL, TKE/EVD/IWM | NOT ENTERED | stopped at earlier external-mode debt |

EOS-80 is the full thirteen-field source walk (`T`, `S`, pressure and all
intermediate polynomial fields through `prd`); its plant exits 1 at
1 / 233,341.  SCO HPG walks the six source-order cumulative/intermediate/final
fields and excludes one canonical-halo stencil band; its plant exits 1 at
1 / 221,640.  Their JSON digests are respectively
`e09cfc472f5ce6b6c4a671c9755ab92042a4e956599e329473b0d3697f5de442`
and `014f08f9c903ea99d9284303f1632fc42c1e9453e70de10e387dbf2029460e5c`.

The external gate walks the first executing external-mode operation:
`dyn_cor_2D_init(Kmm)` after the resolved `ln_dynvor_een` selector
(`dynvor.F90:889-892`) and before `dyn_cor_2D` consumes its eight frozen
fields (`dynspg_ts.F90:1495-1697`).  The committed gate calls the shared
production literal builder under JIT and maps the full card to the rank-zero
A2D(0) domain.  Results are:

| field | unequal / n | max abs |
|---|---:|---:|
| `ffu_nw` | 8,308 / 8,568 | 8.056177274927652e-05 |
| `ffu_ne` | 8,304 / 8,568 | 7.727286259909532e-05 |
| `ffu_sw` | 8,331 / 8,568 | 6.983696920301593e-05 |
| `ffu_se` | 8,323 / 8,568 | 7.253667168386182e-05 |
| `ffv_nw` | 8,285 / 8,589 | 8.320420480597231e-05 |
| `ffv_ne` | 8,299 / 8,589 | 7.900097225433845e-05 |
| `ffv_sw` | 8,310 / 8,589 | 7.257335949618652e-05 |
| `ffv_se` | 8,320 / 8,589 | 7.788661883968118e-05 |

This is a **CONFIRMED shared-external-mode debt**, not a north-fold result:
the frozen coefficients precede the substep recurrence and its boundary
exchange.  Under the standing ownership rule it is handed to the GYRE lane
with record
`variant_icebergs_off_phase2h_systematic_a_10step_np2/oracle_bt_ene_coeff_kt00000001.bin`,
SHA-256 `e7282ddc105300f8ee101d00ba9859eeed487a9c480947ae55fd5aac9f2c9363`.
Lane 4 made no external-mode change.  The binding control starts from an exact
oracle/oracle comparison, changes one live `ffu_nw` cell by one ULP, and exits
1 at exactly 1 / 8,568; its stdout digest is
`e84dbaf2745f5ffb8a75661eb6ffd646731d0e67d9ad11e30f0ef8e6c8168ebe`.

The card state coverage and time-level registry are unchanged: input T/S/u/v/
ssh remain exact at whole-step entry; O1 bulk and SI3 are explicitly supplied
operands; RGB is applied to stage-3 `Krhs` using live `Kmm` thickness/depth;
EOS/HPG consume stage-2 `Kaa=3` and write `Krhs`; frozen EEN consumes `Kmm=1`.
No prognostic or exchange field was waived by a silent stub.

## 5. Verification and provenance

Targeted unit coverage is 88 passed: scalar-libm transcendental dispatch/JVP,
generic and NEMO RGB behavior, source-literal shortwave, and ORCA2 card
selectors.  Each of the RGB, EOS, HPG, and EEN gates has a nonzero binding
plant.  The artifact manifest is
`nemo_testcases_l4_orca2_phase2i_artifacts.sha256`.  Data remain under
`/data/abyssal/dbalwada/nemo-testcases-l4`; no multi-megabyte artifact was
added to git.  The shipped NEMO tree is untouched; the sole NEMO edit is the
configuration-copy WRITE-only guard.

## 6. ASKED / UNASKED

| item | status | disposition |
|---|---|---|
| systematic RHS writer diagnosis and zero-first/rank-zero repair | ASKED | guard added; no physics assignment |
| rebuild and prepare independent twins | ASKED | complete; MPI execution left to user shell |
| pin V2 before complete twin identity | UNASKED and forbidden | not done; requires 92 / 92 |
| preserve Phase-2h twins | ASKED | retained as superseded `CANONICAL_CANDIDATE_91` |
| source-literal shared RGB identity | ASKED | 0 / 233,341; generic arm retained |
| walk EOS/HPG/external mode | ASKED | first new debt found at frozen EEN coefficients |
| repair shared external mode in Lane 4 | UNASKED and forbidden | registered and handed to `GYRE_OWNER` |
| infer north-fold status from pre-fold debt | UNASKED | not claimed; north fold remains unmeasured downstream |
| use bulk/SI3 outputs as certified operators | UNASKED and forbidden | explicitly `ORACLE_SUPPLIED` only |
| execute MPI | UNASKED and prohibited | user-shell handoff below |
| failed-build log preservation | UNASKED diagnostic | failed log accidentally overwritten by successful retry; disclosed |
| delete prior run/data evidence | UNASKED and forbidden | no prior campaign evidence deleted |

## 7. Exact handoff

Run these directories **one at a time**, using the already-present `run.sh`:

1. `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2i_rhsrank0_a_10step_np2`
2. `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2i_rhsrank0_b_10step_np2`

On resume: validate both complete schemas, identity and plants; require 92 / 92
raw exact; then pin A as `VARIANT_ORACLE_V2`.  The ocean numerical ladder
remains stopped at O4-EXT-A pending disposition by the shared GYRE owner.
