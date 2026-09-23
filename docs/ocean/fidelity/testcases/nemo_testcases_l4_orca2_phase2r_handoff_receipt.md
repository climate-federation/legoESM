# NEMO testcase Lane 4 — ORCA2 Phase-2r stop receipt

Date: 2026-09-06

Starting parent: `3e93cef88cc4`

Status: **STOP AT THE ORCA2 `zdf_sh2` CARD-SELECTOR BOUNDARY.**  The
schema-corrected ZDF twins are admitted as the latest
`VARIANT_ORACLE_V2` extension.  The cold-start `avm_k`, `avt_k`, and `en`
inputs are now bit-exact.  The accepted frame has identically zero velocity,
so both the current production card and the source-literal face-native SH2
routine score 0 / 224,547; that result is explicitly vacuous and does not
certify shared SH2 arithmetic.  The first remaining boundary is the ORCA2
card's wrong SH2 selector tuple.  No shared SH2/TKE arithmetic was changed and
no `tke_tke` row is claimed beyond that boundary.

The carried BBL review is closed: nextafter/argmin mimicry was removed from
production, ordinary division on source-rounded operands remains bit-exact,
and all three binding plants exit nonzero.  The EEN cross-card scope is also
closed on all ten GYRE oracle entry states.

## 1. Schema-fix twin admission and V2 pin

**CONFIRMED:** the user-shell twins

- `variant_icebergs_off_phase2q_zdf_schemafix_a_10step_np2`, and
- `variant_icebergs_off_phase2q_zdf_schemafix_b_10step_np2`

each completed ten steps with `MPIRUN_RC=0`, `RUN DONE`, and 95 records.
Their retained launcher logs are `/tmp/orca2_p2q_run_a.log` and
`/tmp/orca2_p2q_run_b.log`, SHA-256
`998ed80dce556f042a705dd2f1ddaf5ba379d39c8c53d968139897f3e7f13676`
and
`dade742233631c4f333a5076628b6a6236688ef84eab1b47fbf6e57b445f5cda`.
The launchers remain unchanged at their run roots, SHA-256
`f1f030b108c250f0eee1fe87499ebdfbda0cbc5512460d4a6748419e82515213`
and
`419c925746bf4c1806916157eca10bb009bb78e66a3a78208781b8e86959275d`.
No MPI/NEMO process ran in the sandbox.

**CONFIRMED:** the committed admission gate walks all three schemas to exact
EOF and reports:

| stream | derived schema result | SHA-256 |
|---|---:|---|
| `oracle_zdf_sh2_operands_kt00000001.bin` | 9,267,648 binary64 values; 74,141,252 bytes | `1e23fa06140cb137c3f75b8354de231a76f2098becda35d0b874c229efa951f7` |
| `oracle_zdf_entry_kt00000001.bin` | 1,681,688 binary64 values; 13,453,548 bytes | `da27af4d951fa8b5ad9416d8956be1ca9b4f26afa60036201d484e09a210eb8c` |
| `oracle_si3_zdf_inputs.bin` | 25 exact frames: kt 1,3,5,7,9 × five categories; 6,679,576 bytes | `a41c354a18c7ebccc310f6e39477d3d6270b591e3899d9f7f4fcb80f7203b07b` |

The mixed SH2 allocation is derived from the writer expressions: three A2D
and 18 full 3-D fields, then one A2D and three full 2-D fields.  The SI3 frame
count is `(13+nlay_s)*npti` with `nlay_s=5`; no payload constant substitutes
for either derivation.

**CONFIRMED:** twins A/B are 95 / 95 raw-identical.  The replacement has 94 /
94 raw identity to the Phase-2n inherited set and 94 / 94 identity to Phase-2p
when the malformed SH2 stream is excluded.  The ordinary-output identity gate
reports four restart shards exact and eight history payloads exact; its
timestamp/timing exclusions are unchanged.  Header-count, canonical-zero,
owned-payload, SI3-header, restart-identity, and history-payload plants all
return `PASS_NONZERO` through their real validators.

Twin A is pinned as `VARIANT_ORACLE_V2_ZDF_SCHEMAFIX_EXTENSION`; twin B is its
independent witness.  The 95 record hashes, ordinary-output hashes, schemas,
and plant results are frozen in
`/data/abyssal/dbalwada/nemo-testcases-l4/phase2r/zdf_schemafix_admission.json`,
SHA-256
`bb0b10ed685c6e46a5f71eb095ca70e331b87a2740921d7e7b027d369e2a5214`.
The Phase-2p twins are retained and flagged
`SUPERSEDED_REJECTED_MALFORMED_ZDF_HEADER`; nothing was deleted.  The four EEN
streams remain pinned on Phase-2m twin A; no synthetic combined root is
claimed.

## 2. Card construction and pre-SH2 entry

**CONFIRMED:** constructing the production ORCA2 model initially stopped in a
guard, not physics: `eos_depth="geometric"` admitted TEOS-10 and S-EOS but
omitted the already-certified EOS-80 arm.  NEMO `eosbn2.F90:260` uses live
`gdept` for the joint `np_teos10/np_eos80` branch.  The allow-list now includes
`nemo_eos80` at
`ocean_model_latlon_cgrid.py:3766-3776`; the guard remains active for an
uncertified EOS.  The focused constructibility tests pass for both admitted
EOS-80 and rejected `wright` cases.  This is a selector guard correction; it
does not change EOS arithmetic.

**CONFIRMED:** the ORCA2 card's cold-start ZDF input pipeline now transcribes
the resolved initialization at `nemo_testcase_recipe.py:1185-1224`:

- `zdfphy.F90:151-173`: `nn_avb=0`, `rn_avm0=1.2e-4`,
  `rn_avt0=1.2e-5`, and the three `nn_havtb=1` latitude `WHERE` statements;
- `zdftke.F90:841-844,918-923`: `ln_zdfiwm=T` forces `rn_emin=1e-10`
  before the cold-start `tke_rst` assignment; and
- the card carries `avm_k`, `avt_k`, `en`, the surface `avm_k`, and `dissl`
  as state, rather than re-deriving them inside an ORCA2 operator fork.

The unchanged entry gate remains exact after this addition: T, S, u, and v
are each 0 / 399,600 and SSH is 0 / 13,320.  The new ZDF gate scores every wet
rank-zero W cell:

| entry field | unequal / n | status |
|---|---:|---|
| `avm_k_pre` | 0 / 224,547 | **CONFIRMED AT-BAR** |
| `avt_k_pre` | 0 / 224,547 | **CONFIRMED AT-BAR** |
| `en_pre` | 0 / 224,547 | **CONFIRMED AT-BAR** |

The entry rerun is
`/data/abyssal/dbalwada/nemo-testcases-l4/phase2r/orca2_entry_rerun.json`
(SHA-256
`893f38dfa12c0ee7054e4ebfb2a23f23b16fbfe1f96c42f17137112210228d12`).

## 3. SH2 score, retraction, and handoff

NEMO executes `zdf_sh2` before `zdf_tke` at `zdfphy.F90:264-286`.  The live
no-Stokes arm is `zdfsh2.F90:78-100`: Kmm/Kbb face differences, face-native
QCO thickness products, adjacent-T `avm_k` sum, W-face masks, coast doubling,
and the final 0.25 combination.

**CONFIRMED / AT-BAR BUT VACUOUS:** all four admitted velocity operands
`u_Kbb/u_Kmm/v_Kbb/v_Kmm` contain zero nonzero values.  Consequently:

| arm | unequal / n | disposition |
|---|---:|---|
| current production card (`squared_centered/tpoint/tpoint_jacobian`) | 0 / 224,547 | `AT_BAR_VACUOUS_ZERO_GRADIENT` |
| shared source-literal face-native routine on the recorded NEMO operands | 0 / 224,547 | `AT_BAR_VACUOUS_ZERO_GRADIENT` |

**RULE 11 RETRACTION:** a zero cold-start SH2 result cannot discriminate the
source statements and is not called shared-operator certification.  The plant
changes one scored target bit and exits nonzero, proving the comparator, but it
cannot create missing physical shear.

**CONFIRMED / FIRST BOUNDARY:** after the entry-state repair, the first
departure is the ORCA2 card selector tuple.  It remains
`squared_centered/tpoint/tpoint_jacobian`, while the resolved RK3 oracle calls
the source arm as face-native Nbb×Nbb at step entry, face-summed `avm_k`, and live QCO face
metrics.  Owner: `LANE4_ORCA2_CARD_SELECTOR`.  Selecting the faithful tuple
also exposes a shared dispatch requirement: the `nemo_face_native_nbb2`
metric path must use Nbb as both Kmm and Kbb instead of requiring the MLF Nnn
slot.  Because this phase is forbidden to alter shared SH2/TKE code,
the selector is registered and the walk stops before `tke_tke`.

The reproducer handed to `GYRE_OWNER_SHARED_TKE` is:

```text
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=packages/core:packages/ocean \
  /home/dbalwada/legoESM/.venv/bin/python \
  scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_phase2r_zdf_sh2_gate.py \
  --deck-root /data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0 \
  --oracle-root /data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2q_zdf_schemafix_a_10step_np2 \
  --plants
```

Its JSON/log SHA-256 is
`58f1b632a95551406bae737eb039eb7624c7202f3017a8c9d013ba22b771f101`.
The shared handoff status is `UNMEASURED_ZERO_GRADIENT`, not `DEBT`; GYRE needs
a nonzero-shear source frame to certify statement arithmetic.  Lane 4 retains
the card-selector owner.

## 4. Diffusive BBL carried review closure

**CONFIRMED:** `_nemo_bbl_scalar_divide` and its nextafter/argmin search were
removed.  The single shared diffusive BBL implementation now materializes the
completed numerator and the `e3t(Kmm)` denominator separately, then performs
plain division (`bbl_adv.py:317-328`), matching `trabbl.F90:194-200`.

The discriminator explains what the helper worked around.  Without the
materialized divisor, the diagnostic flat-division arm differs at 48
temperature and one salinity cell.  With both operands passed through
`nemo_source_round`, ordinary division is exact; no alternate quotient,
numeric selector, custom JVP, HLO exception, or gradient exception remains.

Two independent executions of the final gate produced byte-identical JSON:
SHA-256
`d76901e82ae14a8f6d7d66ab647d81a9819335f0d67354ba2b6810c9efaf9626`.
The decisive rows are:

```text
ahu_bbl                 AT_BAR  0 /   8,489
ahv_bbl                 AT_BAR  0 /   8,554
temperature_Krhs_post   AT_BAR  0 / 231,519
salinity_Krhs_post      AT_BAR  0 / 231,519
```

**CONFIRMED:** the three controls are binding and exit 1 for their intended
scorers:

```text
FAIL: planted BBL coefficient target rejected through coefficient scorer
FAIL: planted pre-BBL tracer target rejected through entry scorer
FAIL: planted post-BBL Krhs bit rejected through production scorer
```

**CONFIRMED / RULE 12:** GYRE, LOCK_EXCHANGE, and OVERFLOW all retain
`bbl_diffusive_option=0`; their no-BBL temperature/salinity rows remain 0 /
21,120, 0 / 7,800, and 0 / 60,600 respectively.  OVERFLOW's separate
advective-BBL selector remains 2 and is not changed by this diffusive arm.

## 5. EEN cross-card scope and ENS caveat

**CONFIRMED / CORRECTED SCOPE:** the earlier GYRE 0 / 21,120 EEN Rule-12 row
was one cold-start entry-eta operand evaluation: zero ocean steps and zero RK
stages.  The gate now reads the hash-pinned GYRE NEMO `BEFORE` entry records at
kt=1 through kt=10 and evaluates the old/new live-e3f builders before every RK
stage.  The result is **0 / 211,200**.  It is ten observed input states, not a
claim that Lane 4 executed a GYRE model trajectory.  All ten input hashes are
in `e3f_scope_rerun.json`, SHA-256
`9f5bc44cf6c076cfc78c65d30c35c0e7d1005c1e15caa381aa2cf51f9dca732d`;
the planted operand bit exits 1 through the production scorer.

**CONFIRMED CAVEAT:** NEMO ENS also consumes `e3f_vor` at
`dynvor.F90:623,719`.  Therefore LOCK_EXCHANGE/OVERFLOW
`AT_BAR_UNREACHABLE_EEN_SELECTOR` means only that legoESM currently dispatches
this helper through `een_e3f_scheme == "nemo_avg4"`.  A future ENS/QCO card
must re-check `fe3mask`; the row is not a physics waiver.

## 6. Verification, rules, and ownership

All numerical gates ran with CPU production JIT, binary64, and explicit
scalar-libm policy.  The focused test run is four passed, 25 deselected
(`test_fidelity_card_constructibility` and diffusive `test_bbl_adv`), log
SHA-256
`5ce412729bc79aeffedb2ce3f9bdfc579a51ae50ecfd8f3d233100143fd80d27`.

| change / claim | Rule 8 same-operator check | Rule 12 cross-card check | disposition |
|---|---|---|---|
| EOS-80 geometric allow-list | no EOS arithmetic changed; uncertified `wright` still raises | existing TEOS-10/S-EOS cards construct unchanged | **CONFIRMED** |
| ORCA2 cold-start ZDF carry | source-literal card input fields; no operator fork | other cards are not modified | **CONFIRMED 0 / 224,547 each** |
| SH2 arithmetic | current and source-literal outputs both zero because inputs are zero | no shared change landed | **PLAUSIBLE / UNMEASURED_ZERO_GRADIENT** |
| diffusive BBL divide | same `apply_bbl_diffusive_tendency` implementation | GYRE/LOCK/OVERFLOW diffusive selector 0, 0 movement | **CONFIRMED** |
| EEN GYRE row | same old/new operand builders at ten actual entry states | LOCK/OVERFLOW remain current-code-path unreachable only | **CONFIRMED with ENS caveat** |

The BBL defaults remain `bbl_diffusive_option=0` and `bbl_aht_m2_s=0.0`.
The proposed defaults are also 0 / 0.0, matching NEMO reference
`ln_trabbl=.false.`; per-card values remain ORCA2 1 / 1000, OVERFLOW 0 / 1000
with `nn_bbl_adv=2`, and GYRE/LOCK 0 / 0.  The user's default decision remains
pending; Phase 2r changes no default and adds no new configuration field.

The Phase-2p ZDF runs, the malformed Phase-2p record, and every earlier oracle
root remain retained and flagged.  The shipped NEMO checkout was not edited.
No multi-megabyte artifact is added to git.

## 7. ASKED / UNASKED

| action | classification | disposition |
|---|---|---|
| admit and pin schema-fix twins | ASKED | completed; A is V2 extension, B witness |
| retain and flag Phase-2p twins | ASKED | retained; superseded, not deleted |
| repair EOS-80 construction guard | UNASKED but necessary to execute the requested production score | narrow allow-list correction with non-vacuity test |
| source-literal ORCA2 `avt_k/en` cold-start state | ASKED by ORCA2-only ownership rule | completed; three entry rows exact |
| call zero SH2 a shared certification | forbidden by Rule 11 | retracted mechanically in the gate |
| change shared SH2/TKE arithmetic | forbidden in Lane 4 | none; handoff remains UNMEASURED |
| change ORCA2 SH2 selector before its shared NOW² metric dispatch exists | UNASKED / would make the card fail construction at the next consumer | registered at first boundary; no silent selection |
| replace BBL emulation with materialized plain division | ASKED | completed; all frozen rows exact twice |
| retain custom quotient/JVP | disallowed absent counterexample/HLO | removed; no exception needed |
| scope GYRE EEN row and extend kt=1..10 | ASKED | completed on ten hash-pinned oracle entry states |
| register ENS caveat | ASKED | in gate and receipt |
| change BBL defaults | ASKED decision pending | unchanged |
| sandbox MPI/NEMO, shipped edit, deletion, push | forbidden | none |

## 8. Stop boundary / next work

Phase 2r stops at `LANE4_ORCA2_CARD_SELECTOR` before `tke_tke`.  The next
round must:

1. make the shared `nemo_face_native_nbb2` SH2 entry dispatch consume Nbb for
   both QCO metric factors under RK3, owned and reviewed by the GYRE/shared-TKE
   lane;
2. set the ORCA2 card to the exact face-native Nbb² / `nemo_face` /
   `nemo_qco_live_face` tuple, with a nonzero-shear discriminator rather than
   relying on the vacuous cold start; and
3. only after that selector boundary closes, walk the first executed
   `tke_tke` statement and continue through EVD/DDM/IWM.  SI3 remains
   `ORACLE_SUPPLIED` / `UNMEASURED_PENDING_ICE_MERGE`.

No NEMO rerun is requested by this receipt.

Session ID: `01a06d99-f562-7b11-bc63-e9b112877f54`.
