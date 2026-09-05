# NEMO testcase Lane 4 — ORCA2 Phase-2q preregistration

Date: 2026-09-06

Parent: `124d9f5cea63`

Status: **PREREGISTERED BEFORE ZDF ADMISSION, SH2 SCORING, OR BBL
RE-MEASUREMENT.**  Measurements use production JIT on CPU, binary64, the
explicit scalar-libm policy, oracle-relative cellwise scoring, and the
rank-zero reconstruction.  No sandbox MPI/NEMO run is authorized.

## P2Q-A — ZDF twin admission

The user reports two successful 10-step twins with 95 records apiece and
95/95 raw twin identity.  Admission independently measures those claims.  It
expects exactly 94 records shared with the Phase-2n BBL root and one added
`oracle_zdf_sh2_operands_kt00000001.bin`; all 94 inherited records must be raw
byte-identical.  The ZDF twin must also retain exact ordinary restart/history
payload identity under the campaign's timestamp-only exclusion.

The Phase-2p preregistration predicted the four Phase-2m EEN streams would be
present and therefore predicted 99 records.  That prediction is **REFUTED**:
the Phase-2p source assembly applied the SH2 patch series without the distinct
EEN writer patch.  The omission was not intended, but it changes no model
array and does not invalidate either acquisition.  The four EEN streams remain
pinned only at the Phase-2m twin-A root; the ZDF extension is the 95-file root.
The receipt and tool must stop claiming a 99-file combined inventory.

Three ZDF-related streams are schema-walked even though two are inherited:

| stream | schema requirement |
|---|---|
| `oracle_zdf_sh2_operands_kt00000001.bin` | `NEMO_L4_ZSH2_1`; 13-int header; `21*jpi*jpj*jpk + 4*jpi*jpj` binary64 payload; exact EOF; canonical zero outside each field's grid |
| `oracle_zdf_entry_kt00000001.bin` | `NEMO_L4_ZDF___2`; seven-int header; `jpi*jpj*jpk + 3*(jpi-4)*(jpj-4)*jpk` binary64 payload; exact EOF |
| `oracle_si3_zdf_inputs.bin` | inherited SI3-ZDF exchange schema and exact EOF from the Phase-2b gate |

Header/payload/canonical-slot plants must pass through the real validators and
exit nonzero.  Twin-A becomes the `VARIANT_ORACLE_V2_ZDF_EXTENSION`; twin-B is
the reproducibility witness.  Every retained artifact is SHA-256 pinned.

## P2Q-B — shared `zdf_sh2` boundary

The resolved no-Stokes arm is frozen from `zdfsh2.F90:78-100`:

1. face-native now-minus-below and before-minus-below velocity differences;
2. live `e3uw/e3vw(Kmm)*e3uw/e3vw(Kbb)` divisor;
3. previous `avm_k` summed across the two T cells bracketing each U/V face;
4. `wumask/wvmask` multiplication;
5. the `0.25` T-point combine and coast factors; and
6. explicit surface/bottom zeroing.

The production shared operator is
`vertical_mixing._shared.avm_weighted_shear_production`, called through a
production-JIT arm with the exact NEMO operands.  Each source statement is
separated with `nemo_source_round`.  The bar is 0 / n bitwise unequal owned
wet W points.  The gate also scores `avm_k`, `avt_k`, and `en` at the
pre-closure boundary against the card's carried entry state where such state
exists; unavailable carried state is reported `UNMEASURED`, never fabricated.
The first non-bit statement and cell class (fold, partial-cell bottom, coast,
or interior) are recorded.  A one-bit target plant must exit nonzero.

`zdf_sh2` is geometry-independent shared closure arithmetic owned by GYRE.
Lane 4 will not land a shared SH2 change.  A departure first proven in an
ORCA2-only fold or partial-cell operand is Lane-4-owned and must be separated
before routing the remainder.

## P2Q-C — BBL division review

The current `_nemo_bbl_scalar_divide` nextafter/argmin routine is treated as a
reviewed hypothesis, not an accepted production numeric.  The first arm
replaces it with plain division after `nemo_source_round` materialization of
the numerator and divisor, preserving the source association in
`trabbl.F90:187-200`.  Acceptance requires the three frozen diffusive-BBL rows
to remain 0 / 8,489, 0 / 8,554, and 0 / 231,519, all binding plants to exit
nonzero, and inactive GYRE/LOCK/OVERFLOW Rule-12 rows to remain 0 ULP.

If that arm fails, the gate records a concrete cell's numerator, divisor,
plain quotient, NEMO target and ULP relation, plus lowered CPU HLO showing the
division lowering.  Only then may the emulation remain, and it must become an
explicit selectable numeric with a second-order `jax.test_util.check_grads`
test on its JVP.  No card-specific BBL fork is permitted.

The review also requires this ASKED decision row to be present before any
future default is changed:

| field | current default | proposed default | per-card values | disposition |
|---|---:|---:|---|---|
| `bbl_diffusive_option` / `bbl_aht_m2_s` | `0` / `0.0` | `0` / `0.0` | ORCA2 `1` / `1000`; OVERFLOW `0` / `1000` (`nn_bbl_adv=2`); GYRE and LOCK `0` / `0` | ASKED; decision pending with user |

Every new configuration field must receive an ASKED/UNASKED row in the same
commit that introduces it.

## ASKED / UNASKED

| action | classification | disposition |
|---|---|---|
| admit and pin the reported ZDF twins | ASKED | fail closed on schema, twin, inherited, or history/restart mismatch |
| preserve Phase-2m EEN pins separately | ASKED | no synthetic combined root |
| score SH2 and pre-closure state | ASKED | source-statement gate; shared debt routes to GYRE |
| change shared SH2 arithmetic | UNASKED | no change in Lane 4 |
| replace BBL divide mimicry with barrier-materialized division | ASKED | retain only if all exact rows and controls hold |
| retain divide emulation without counterexample/HLO/gradient test | UNASKED | forbidden |
| alter BBL config defaults | ASKED decision, pending | no default change this phase |
| sandbox MPI/NEMO | forbidden | none |
| shipped-tree edit, deletion, push | forbidden | none |
