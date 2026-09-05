# NEMO testcase Lane 4 — ORCA2 Phase-2h systematic-writer handoff

Date: 2026-09-06

Parent: `c19944cea7ed4d2ca5da97a70237509e3892a9b6`

This is a run handoff, not a VARIANT V2 acceptance receipt.  The systematic
writer audit and scalar-math rebuild are complete, two fresh run directories
are prepared, O1-M remains certified, O1-B is handed to Lane 3b, and the
post-bulk ladder reaches an ORCA2-owned RGB debt.  VARIANT V2 is deliberately
un-pinned until the replacement twins prove 92 / 92 raw-byte identity.

## 1. Phase-2g twin disposition

Both completed Phase-2g twins independently pass the acquisition gate,
including every frozen record schema, 4 / 4 restart shards, 8 / 8 history
payloads under the timestamp rule, the surface record, O1's two frames, and
the binding controls.  Their gate JSON digests are
`e60ad49d013bf4834e3623d382cbf1c2eaa19f2857bb9063fb493ab83a522167`
(A) and
`9baf506f59eac4a23cec50891fbf76bf93dc272c308d20fe9dfca013ae7166e1`
(B).

The new fail-closed reproducibility gate sees complete 92-record inventories
on both sides but exits 1 at 91 / 92 raw exact.  Its source-layout decoder
locates all 430 unequal f64 values in one record:

| stream | A digest | B digest | differing operands |
|---|---|---|---|
| `oracle_rkstage1_transport_operands_kt00000001.bin` | `156a60f889c90f2d0bee5af27b2970875a7cd2413c8638a895e4e1a6bfc61aa7` | `1711aa88aa1e78b44cdd719b6af5963c24fe42a30aaf49dfb44b36f48c363952` | `zub`: 232; `zvb`: 198 |

Those values are unowned east-halo storage.  All 91 other records, including
O1, are raw exact.  An A-versus-A one-byte plant passed through the same
reproducibility comparator exits 1 at 91 / 92, with only the planted O1 record
reported unequal.  The earlier Phase-2g “91-record VARIANT V2” designation is
therefore superseded by `CANONICAL_CANDIDATE_91`; it is not the campaign's
canonical complete record set.

As a polarity control, A versus itself exits 0 at 92 / 92 through the same
streaming comparator (JSON SHA-256
`02c1e0fcbe03ec73b9370f2f0cbc08563ab44af3175d1ad17714974f47e5e84e`).

## 2. Systematic WRITE audit and rebuild

The exhaustive operand/shape/risk table is
`nemo_testcases_l4_orca2_phase2h_writer_audit.md`.  All fourteen config-local
F90 overrides were walked.  Every full horizontal-domain payload now passes
through a zero-first rank-owned wet T/U/V/F view; already-owned `A2D(0)`,
packed `1:npti`, and completely assigned fixed vectors remain direct.  The
rank-4 helper covers ice categories and layers.  Optional arrays remain behind
their allocation-owning selectors.  All helpers take the model fields as
`INTENT(in)` and assign only function-result storage.

The first rebuild attempt is preserved and flagged: it failed at compile time
because three ice overrides did not import the masks needed by the new helper.
Adding explicit read-only `dom_oce` imports fixed the configuration copy; it did
not change a model expression.  The successful `conda-scalarmath` build has
`-fno-tree-vectorize`; `nm -D` finds zero `_ZGV*` symbols.

| artifact | bytes | SHA-256 |
|---|---:|---|
| `build/nemo_ORCA2_OMIP_L4_phase2h_systematic.exe` | 55,519,664 | `c47a1a6bd9b1873f28cf3eaa34a6264cd3ed65e24d16f8d145c1137e2671a308` |
| successful build log | 77,616 | `7146ac801ee88981a4dac1115970588550eec3062200d436bf60e0ad50e198b3` |
| preserved failed build log | 37,452 | `efd317c60b1abb9d66b6bd8cf59fadbeddf853aef1e7750b3cc682dbabe780d0` |
| 15-file MY_SRC manifest | 1,190 | `4ea891a7810f5818b6f3c6d15bb5dbcd49a44a4b8d808f6756fda6e9460106ff` |
| zero-line `_ZGV*` result | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

No shipped NEMO file was modified.

## 3. O1-M certification and Lane-3b bulk handoff

The O1 record used by this handoff is:

`/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2g_o1canon_a_10step_np2/oracle_sbcblk_o1_kt00000001.bin`

It is 3,090,336 bytes with SHA-256
`751b2d9181778e81f01bc5872d47100afc0fc3ad02c4ffcccc9613a0af2ad045`.
The independently repeated numerical gate uses production JIT on CPU with
fp64 and scalar-libm.  O1-M is `AT_BAR`: 0 / 13,320 for each of `wndi`,
`wndj`, `tair`, `humi`, `qsr_down`, `qlw_down`, `precip_raw`, `snow_raw`, and
`slp`.

Frame 0 is recorded immediately after `CALL fld_read` and before humidity,
temperature, precipitation, or bulk transformations
(`MY_SRC/sbcblk.F90:559-560,672-693`).  Its header is
`(version,kt,frame,nx,ny,nfield,reserved,bits)=(1,1,0,90,148,9,0,64)`.
NEMO performs field updates and vector rotation before forming `fnow`
(`src/OCE/SBC/fldread.F90:200-227`); the generic time-interpolating arm, when
selected, uses the source-ordered before/after expression at `:216-227`.

Frame 1 is appended after `blk_oce_1` then `blk_oce_2`
(`MY_SRC/sbcblk.F90:624-635`) and before SI3, iceberg, runoff,
freshwater-budget, or final halo assembly (`:696-717`).  Its header is
`(1,1,1,90,148,20,0,64)`.  Field order is:

`theta_air, q_air, precip, sst, ssu, ssv, tsk, ssq, cd_du, sensible, latent,
evap, qlwn, qsr, qns, emp, utau, vtau, taum, wndm`.

For the resolved NCAR arm, `cd_du` and `qlwn` are source-unowned and are
canonical zero fields.  The other eighteen fields are defined.  `qsr`, `qns`,
`emp`, `utau`, `vtau`, `taum`, and `wndm` are the pre-SI3 open-ocean bulk
outputs; a consumer must not mistake them for the post-SI3 aggregate surface
fields.

This is the requested `LANE3B_OWNER` handoff.  On oracle-supplied mapped
inputs, the current shared NCAR leaf is over bar in all seven scored outputs:
`theta_air` 117 / 8,794, `ssq` 3,964 / 8,794, and `utau`, `vtau`, `sensible`,
`latent`, `evap` each 8,794 / 8,794.  The ice-thermo lane must certify the
source-ordered `blk_oce_1`/`blk_oce_2` Large & Yeager path against this exact
record and preserve the two-frame semantics above.  Lane 4 made no shared
bulk arithmetic change.

## 4. Post-bulk ocean ladder

The bulk boundary was bypassed only by explicit operand substitution.  Frame-1
bulk outputs and the oracle SI3 aggregate exchange remain
`ORACLE_SUPPLIED`; this does not certify either bulk or SI3.  The actual
`tra_qsr` input is the post-SI3 aggregate `qsr` carried in the RGB input frame.
SI3 remains `UNMEASURED_PENDING_ICE_MERGE`.

The committed RGB gate parses `oracle_rgb_chl_kt00000001.bin` and
`oracle_qsr_stage3_kt00000001.bin`, uses the deck's resolved
`ln_qsr_rgb=T`, `nn_chldta=1`, `nn_chlprfl=1`, `rn_abs=0.58`, and
`rn_si0=0.35`, and calls the production shared `nemo_qsr_rgb` implementation
under JIT, CPU, fp64, scalar-libm.  NEMO selects `qsr_RGBc` and its
source-ordered Morel-Berthon/profile and band recurrences in
`MY_SRC/traqsr.F90:255-496`; the dump is taken after `fld_read` at `:298-302`.

| boundary | result | owner | disposition |
|---|---|---|---|
| O1-M `fld_read` | AT_BAR, nine times 0 / 13,320 | `ORCA2_OWNER` | CERTIFIED |
| O1-B NCAR bulk | DEBT | `LANE3B_OWNER` | handed to ice-thermo lane; oracle outputs substituted |
| O2-RGB `tra_qsr` | DEBT, 149,647 / 233,341 wet T cells; max abs `6.341827702190138e-06` | `ORCA2_OWNER` | first over-bar boundary after substitution |
| EOS/HPG through TKE/EVD/IWM | not entered | registered owners | stopped after first post-bulk debt |

The RGB one-variable plant compares an oracle copy through the identical
cellwise scorer, changes one wet f64 by one ULP, and exits 1 at 1 / 233,341.
The measurement JSON has SHA-256
`236e507cb354cef9aaf45e805c29f087ab803d37805d2b800d812104a11a8e76`;
the plant stdout digest is
`e5cffa2a0c66073742c9bbc0024ad0b106103a70d1e1b513b410ef3309df59bd`.
The RGB debt is CONFIRMED; its detailed arithmetic correction is deferred
until the canonical V2 acquisition clears, so no result is mixed across
oracle versions.

## 5. Prepared replacement twins

Both directories contain copied, hash-pinned namelists/XML and symlinks to the
unchanged input deck and exact systematic binary.  They use CPU, one OpenMP
thread, and the oracle-required `jpni=2`, `jpnj=1` two-rank layout.  The
launchers use bash `time`, tee stdout, record `MPIRUN_RC`, and refuse an
unexpected directory, output overwrite, binary mismatch, deck mismatch, or
input mismatch.

| arm | prepared-manifest digest | launcher digest |
|---|---|---|
| A | `3c3c437452df02a2dbbbd4ed3f6bd87c185857319acf6215136043f3dd3dc20e` | `b0d479d2c9db7b66a25489d608e6dbe272cd9c4ca6c68733890414fa58f0a381` |
| B | `f339aff29876373dddaac09d6149509964f9e2d5bc4c00a230c56073e3347570` | `137772b2c9c41691cc91c7a91fd73092e195e57aeeea688b5d5d206f821eaf45` |

Acceptance after execution is fixed: each arm must finish 10 steps, pass the
complete schemas, ordinary-output identity, and all plants; then the
reproducibility gate must report exactly 92 / 92 raw identical.  Anything less
leaves VARIANT V2 unpinned.

## 6. ASKED / UNASKED

| item | status | disposition |
|---|---|---|
| systematic audit of every MY_SRC writer | ASKED | complete array-payload census and zero-first canonicalization |
| rebuild and prepare twin runs | ASKED | complete; MPI execution handed to user shell |
| retain Phase-2g evidence | ASKED | both runs preserved and labelled candidate/superseded, not deleted |
| pin VARIANT V2 before 92 / 92 | UNASKED and forbidden | not done |
| certify nine CORE mappings | ASKED | 0 / 13,320 each, retained |
| hand NCAR bulk to Lane 3b | ASKED | exact record/schema/semantics above |
| continue with oracle-supplied bulk/SI3 operands | ASKED | reached first post-bulk debt at RGB |
| fix shared NCAR bulk in Lane 4 | UNASKED and forbidden | no shared arithmetic change |
| fix ORCA2 RGB before canonical acquisition | UNASKED sequencing choice | deferred to keep one oracle version per score |
| failed ice-mask-import build | UNASKED diagnostic | preserved and flagged; successful correction is import-only |
| execute MPI | UNASKED and prohibited | user shell action required |
| delete any prior run/artifact | UNASKED and forbidden | nothing deleted |

Stop: execute the two directories in section 5, one at a time.  After 92 / 92
is proven, pin arm A as VARIANT V2 and resume the ORCA2-owned RGB identity.
