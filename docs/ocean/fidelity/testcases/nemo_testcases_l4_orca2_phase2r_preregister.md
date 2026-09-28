# NEMO testcase Lane 4 — ORCA2 Phase-2r preregistration

Date: 2026-09-06

Parent: `3e93cef88cc4`

Status: **PREREGISTERED BEFORE REPLACEMENT ADMISSION, SH2 SCORING, OR BBL
RE-MEASUREMENT.**  All numerical rows use CPU production JIT, binary64,
explicit scalar-libm policy, oracle-relative cellwise scoring, and the
rank-zero reconstruction.  No sandbox MPI/NEMO execution is authorized.

## P2R-A — schema-fix admission

The two replacement roots must each contain exactly 95 records.  All 95 must
be raw-identical between twins; 94 must be raw-identical to the Phase-2p root.
Only `oracle_zdf_sh2_operands_kt00000001.bin` may differ, solely because the
header count changed from the rejected 9,358,640 to the `SIZE(...)`-derived
9,267,648.  Ordinary output admission remains four exact restart shards and
eight exact history payloads under only the frozen timestamp/timing exclusions.

The real admission validator must walk these schemas to exact EOF:

| stream | required derived schema |
|---|---|
| `oracle_zdf_sh2_operands_kt00000001.bin` | 13-int header; 3 A2D and 18 full 3-D fields plus 1 A2D and 3 full 2-D fields; 9,267,648 binary64 values |
| `oracle_zdf_entry_kt00000001.bin` | one full `avm` plus three rank-zero-interior `avt/avs/en` arrays |
| `oracle_si3_zdf_inputs.bin` | every `NEMO_L3ZIN_002` frame; `(13+nlay_s)*npti`, monotone `kt`, exact EOF |

Header-count, canonical-zero, owned-payload, inherited-identity, restart and
history plants must traverse the real validators and exit nonzero.  Twin A is
then pinned as `VARIANT_ORACLE_V2_ZDF_EXTENSION`; twin B is its independent
reproducibility witness.  The Phase-2p twins remain retained and are flagged
`SUPERSEDED_REJECTED_MALFORMED_ZDF_HEADER`.

## P2R-B — `zdf_sh2` statement ladder

The resolved no-Stokes source arm is `zdfsh2.F90:78-100`.  The gate invokes
the production shared `avm_weighted_shear_production` through JIT with the
admitted NEMO operands.  The source ladder is:

1. U/V now and before vertical differences (`:80-89`);
2. live Kmm/Kbb face-thickness products (`:83,88`);
3. previous `avm_k` adjacent-T sums (`:80,85`);
4. division then `wumask/wvmask` multiplication (`:83-89`);
5. adjacent-face sums, coast factors, and literal 0.25 combine (`:92-95`);
6. surface and bottom zeroing (`:97-100`).

Every statement boundary is rounded with `nemo_source_round`.  The bar is
zero unequal bits on every owned wet W cell with complete face stencil; rows
also report row-scale ULP maxima and the first cell's fold/partial-bottom/
coast/interior class.  One-variable association arms isolate only the first
departing statement.  A one-bit target plant must exit nonzero.

Pre-closure `avm_k`, `avt_k`, and `en` are scored against the production card's
carried entry state where represented; absent state is explicitly
`UNMEASURED`, not synthesized.  These are input-pipeline/time-level rows, not
an SH2 arithmetic claim.  Any ORCA2-only input departure (fold, partial cell,
`taum/fr_i/rCdU_bot`) belongs to Lane 4.  Any departure in the same shared
SH2 arithmetic belongs to GYRE and is handed off with gate, record hash, and
exact invocation; Lane 4 does not repair it.

If SH2 is exact, the same gate enters `tke_tke` at its first executed source
statement in `zdftke.F90`, using only recorded operands.  It stops at the first
non-bit or missing operand and does not guess through an unrecorded boundary.

## P2R-C — diffusive-BBL compiler discriminator

The existing nextafter/argmin `_nemo_bbl_scalar_divide` is not accepted as
production physics glue without proof.  Arm A replaces it with source-rounded
plain division whose numerator and denominator are each materialized by
`nemo_source_round`, matching `trabbl.F90:194-200`.  Acceptance requires the
three frozen rows to remain 0 / 8,489, 0 / 8,554, and 0 / 231,519, with
GYRE/LOCK/OVERFLOW unchanged and every coefficient/pre-tracer/post-RHS plant
nonzero.

If Arm A fails, the gate emits the first counterexample's numerator, divisor,
plain result, oracle result and signed ULP delta, plus lowered CPU HLO.  Only
that evidence permits retaining the emulation, which must become an explicit
shared numeric selector and pass `jax.test_util.check_grads(order=2)` for its
JVP.  There is no ORCA2-specific divide fork.

## P2R-D — carried EEN review

The prior GYRE 0 / 21,120 row is audited for exact step/stage scope.  A
single-entry-state result must be labelled as such; it is not silently called
kt=1..10.  If the existing gate can exercise ten steps without a NEMO rerun,
the ten-step scope is measured; otherwise the row remains explicit one-state
Rule-12 selector coverage.

NEMO ENS also divides by/reciprocates live `e3f_vor` under QCO at
`dynvor.F90:621-624,717-720`.  Therefore LOCK/OVERFLOW “EEN-unreachable” means
only that legoESM's current helper is dispatched by
`een_e3f_scheme == "nemo_avg4"`.  It is not proof that future ENS cards may
ignore `fe3mask`; any ENS/QCO upgrade must rerun that operand gate.

## ASKED / UNASKED

| action | classification | disposition |
|---|---|---|
| admit and pin schema-fix twins | ASKED | fail closed on any schema/identity/plant failure |
| score SH2 and carried entry fields | ASKED | source ladder; first non-bit boundary only |
| change shared SH2/TKE arithmetic | forbidden in Lane 4 | hand to GYRE |
| replace BBL mimicry with materialized division | ASKED | accept only at all frozen bars |
| retain emulation without cell/HLO/gradient evidence | forbidden | none |
| state exact GYRE EEN row scope | ASKED | measured, not inferred |
| register ENS/e3f caveat | ASKED | code-path scope, not physics waiver |
| alter BBL defaults | ASKED decision pending | no change |
| sandbox MPI/NEMO | forbidden | none |
| shipped edit, deletion, push | forbidden | none |
