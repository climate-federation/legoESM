# NEMO testcase Lane 4 — ORCA2 Phase-2y handoff receipt

Date: 2026-09-05

Parent: `eeea6a2382a8` (phase-2y preregistration)

Status: **P2Y-1 ADMITTED.** `VARIANT_ORACLE_ORCA1ICE` is pinned. This receipt
inventories the two SI3 lanes that now coexist under Lane 4 and closes the
resolved-namelist and source-hygiene review items carried from phase-2x. No
legoESM numerics changed. No MPI/NEMO run was launched from the sandbox
during this receipt (the phase-2x twins had already been executed by the
user shell before this session started).

## 1. P2Y-1 admission

**CONFIRMED PASS.** `nemo_testcase_l4_orca2_phase2y_orca1ice_admission_gate.py`
run against
`variant_orca1ice_phase2x_{a,b}_10step_np2` with baseline
`variant_icebergs_off_phase2v_tke_a_10step_np2`: 116/116 streams present in
both twins, 116/116 twin-raw-identical, the 90-stream Phase-1 inventory
(76 allocation-invariant via the frozen composite parser + 14 variable-legacy
SI3 streams via the header-driven parser) and the 11 orca2_l4 extensions all
decode to exact EOF, ordinary-output identity (15 exact-byte files, 4 restart
shards, 8 history payloads, normalized `ocean.output`) is PASS with
`icebergs_enabled=false`, and all 26 plants (9 tailored schema plants + 5
legacy-format plants × 3 mutations + the twin-byte plant + the inherited
100-stream V2 regression plant) bind. The gate's JSON is byte-identical
(SHA-256 `d3c60bf16a84f8739f8106541bea2bfcf4941e75b17e7a4f1bd0d5f7cd371297`) to
the artifact already sitting at
`/data/abyssal/dbalwada/nemo-testcases-l4/phase2y/orca1ice_admission.json`
from the prior session's own run — independent reproduction, no
reconciliation needed (oracle-fidelity Rule 1e).

**Pinned:** twin **a** (`variant_orca1ice_phase2x_a_10step_np2`) is
`VARIANT_ORACLE_ORCA1ICE`. Twin **b** is retained only as its
reproducibility witness, per the preregistration's explicit UNASKED
disposition (not a second root).

**First differing OCEAN stream vs the icebergs-off `VARIANT_ORACLE_V2` root**
(`variant_icebergs_off_phase2v_tke_a_10step_np2`, 101 streams, all of which
are common with `VARIANT_ORACLE_ORCA1ICE`'s 90+11): the committed probe
`nemo_testcase_l4_orca2_phase2y_v2_divergence_probe.py` (JSON
`nemo_testcases_l4_orca2_phase2y_v2_divergence.json`) finds only 2 of 101
common streams byte-identical (`oracle_een_e3f0vor_kt00000001.bin`, a static
grid metric, and `oracle_sbcblk_o1_kt00000001.bin`, an ice-independent
atmosphere-forcing operand). The chronologically first record,
`oracle_step_entry_kt00000001.bin` (the write-only `ts/uu/vv/ssh(...,Nbb)`
dump at the top of `stp_RK3`, `stprk3.F90:88-101` in the shipped-numbering
citation, before `sbc()` is even called for kt=1), has **T, S, u, v exactly
bit-identical** between roots; only `ssh` differs, at 8794 of 14288 rank-0
cells, max/mean `|delta|` = 1.548e-2 m. This is **CONFIRMED expected**, not a
WRITE-only failure: `iceistate.F90:400-406` (SI3's initial-state routine,
called once during `nemo_init`, before the step loop starts) sets
`snwice_mass = tmask * SUM(rhos*v_s + rhoi*v_i + rhow*(v_ip+v_il), dim=3)`
(summed over ice categories) and then subtracts it from `ssh` at **both**
`Kmm` and `Kbb` — i.e. the sea-level correction for the *analytically
initialized* snow+ice load. `VARIANT_ORACLE_V2` resolves `jpl=5` ice
categories; `VARIANT_ORACLE_ORCA1ICE` resolves `jpl=1` (confirmed from each
run's own `namelist_ice_ref`/`namelist_ice_cfg`). The same analytic
cold-start discretized into a different category count gives a different
total ice+snow volume even under `nn_iceini_file=0` on both sides, so `ssh`
already differs before the first per-step `sbc`/SI3 call. Every later
stream inherits this (plus the subsequent per-step SI3 feedback into surface
forcing), which is why 99 of 101 common streams differ. This is evidence the
two roots must stay separate oracle families, exactly as the preregistration
anticipated.

## 2. The two SI3 lanes

### Lane V2 (icebergs-off, `jpl=5`)

Root: `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2v_tke_a_10step_np2`

SI3-touching streams (part of the 90-stream Phase-1 inventory; every other
stream in that inventory is pure-ocean and carries no ice state):

| stream | SHA-256 |
|---|---|
| `oracle_si3_exchange_frames.bin` | `3d96ff548a788529676072f0f6e352b380c711a7acab6362dd1519bb3886a26c` |
| `oracle_si3_thd_frames.bin` | `f10e6c6a8fbee5dfbefc50afbf5ee05e6ece70821181a847f7ebb6e02acd5a6e` |
| `oracle_si3_zdf_inputs.bin` | `a41c354a18c7ebccc310f6e39477d3d6270b591e3899d9f7f4fcb80f7203b07b` |
| `oracle_si3_reassoc_operands.bin` | `7ac8a004aca4adb9b5ef6dfb50bce94fc6223863fee58f98faf56fcffddcdd26` |
| `oracle_si3_bulk_operands.bin` | `3895867017affa1c9827ae0374d50f8e69bfed78b603c70647a6516f60cc1dbc` |
| `oracle_si3_prather_kt00000001_s0.bin` | `500e3b0b08e89c180bd1de0a72a06930349fbbe2da7d1c1b206cb30c79eb4f0f` |
| `oracle_si3_prather_kt00000001_s1.bin` | `5c746ddfff8af9cf77fd603d1df2f04183f71ef3358b95293ab867f16eabfffc` |
| `oracle_si3_prather_kt00000003_s0.bin` | `5ff2b413e61ff19ad6dee6320ab912523cb8c58f17ff4c5aaead6974ec5c6b07` |
| `oracle_si3_prather_kt00000003_s1.bin` | `cd57529d09845bbd9286ba51c9b8f162852c1bd1005e22a9d136827b846a795f` |
| `oracle_si3_prather_kt00000005_s0.bin` | `59b02ce228c308b78382038799b6acd0f244340d0953548ef5fd52378c9d4911` |
| `oracle_si3_prather_kt00000005_s1.bin` | `99160af29eaa7bb8b5997914f86cb3f2726923761deef427303e14b9657f97b8` |
| `oracle_si3_prather_kt00000007_s0.bin` | `3be263712002f76ad471a9ad22e2e74c50d905085b154de46910095756d9e3f1` |
| `oracle_si3_prather_kt00000007_s1.bin` | `7e368e44589a078482011c65516fb693c01a441c74fe102aadf81d3a6502ac38` |
| `oracle_si3_prather_kt00000009_s0.bin` | `9a29c90995b586931e7b2910d4957185ecddbfd0b3f2aa459b78d1edad2ebd2f` |
| `oracle_si3_prather_kt00000009_s1.bin` | `32b2284bb6de5c3a8d9bebfe3b9ec14321b59a915fe3c9cbc96df94df5c64307` |

Ice namelist: SI3 standard ORCA2 defaults (`cfgs/SHARED/namelist_ice_ref`,
SHA-256 `055ab5e6839c12dbae61bf1c3bcdda46a8f95cf275b7f294d68d7f8e73179307`)
with no override touching `jpl`/`nlay_i`/`nlay_s`/`nn_icesal` — resolves
`jpl=5`, `nlay_i=1` (single-layer ice class since neither cfg sets
`nn_icesal=2`... see the resolved value in §3 below), `nn_icesal` per ref
default. Full 90-key resolution not repeated here; the V2 root's ref/cfg
pair is the "reference" side of the §3 comparison.

### Lane ORCA1ice (`jpl=1`, `nn_icesal=2`)

Root: `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_orca1ice_phase2x_a_10step_np2`

Every SI3 stream (the 14-item variable-legacy family plus the 15-item
TAILORED family; the full 116-entry manifest with every stream's SHA-256 is
the admission gate's own JSON, cited in §1 above):

| stream | SHA-256 |
|---|---|
| `oracle_si3_exchange_frames.bin` | `fd538f4a93987fc614a2acb9f2a7645b74726b9bd9ca42928db6164cae7db2aa`\* |
| `oracle_orca1ice_dyn_kt00000001.bin` | `3d64b7f129284f994e6e6d9203270b585e4d6430c6d17ef3182e9ae1c11e6d8d` |
| `oracle_orca1ice_dyn_kt00000003.bin` | `dc090637f64a28a4ea8be9cadc8b4c2b3dfa958800b948779f6f1f9c1284da66` |
| `oracle_orca1ice_dyn_kt00000005.bin` | `3bda617c3582c55ab716816c7c877c62c229fea62934fd6ee55c673c045429c3` |
| `oracle_orca1ice_prather_kt00000001_f0.bin` | `1a86c612980e4cb0e5d4d580edbafd24d8f1c256cc967ccf814d4de0e89a49dc` |
| `oracle_orca1ice_prather_kt00000001_f1.bin` | `1cffc86ff6cdd606efa5c4b27fcff3daa61b249be13fc9f203d987f88277c0b4` |
| `oracle_orca1ice_prather_kt00000003_f0.bin` | `4af504ee2e409d35d0896582def14c18ef8332ba8be99f4de6b254bab2976cab` |
| `oracle_orca1ice_prather_kt00000003_f1.bin` | `a81252d02675e5121052a7a9421a3afaf6dc30de209cf4d6dd99965db263a374` |
| `oracle_orca1ice_prather_kt00000005_f0.bin` | `5df709ac1cac18f14a85a7dec6464aac7ae400d55d327ff0df07295896b6eb41` |
| `oracle_orca1ice_prather_kt00000005_f1.bin` | `e16d6f06ca4045cfc3838d46b9050f09527b3b0963edf81625a540281e071f7d` |
| `oracle_orca1ice_thd_kt00000001_f0.bin` | `0624d37f62f7c3c7de6f5e093bc2d76e8b4bae8d6ab0d47f5c5c4bbcb5daf29d` |
| `oracle_orca1ice_thd_kt00000001_f1.bin` | `2eb52ee643e3d7ee334c3e10f37bda4e1afc56c7eb584ed403c24b85405b196e` |
| `oracle_orca1ice_thd_kt00000003_f0.bin` | `639f317077cea460daa73bf8a27189f11a0c84a1d8013543bddd8f4af13feb23` |
| `oracle_orca1ice_thd_kt00000003_f1.bin` | `72312a04e48c1d77ca3c775f519904eee14df6cd85b502e858c45d1615cc70ed` |
| `oracle_orca1ice_thd_kt00000005_f0.bin` | `804a03cfa71c370770f882d7e5772b571f0b03ff5b18bd630c2e05986cba6bb4` |
| `oracle_orca1ice_thd_kt00000005_f1.bin` | `c28486e4a2aa976c93925ea6d2abf67d742ce0bbec528c9ab0d72568cd1025b2` |

\* the remaining 13 variable-legacy streams (`oracle_si3_thd_frames.bin`,
`oracle_si3_zdf_inputs.bin`, `oracle_si3_reassoc_operands.bin`,
`oracle_si3_bulk_operands.bin`, the 10 `oracle_si3_prather_kt*_s*.bin`) are
present and hashed in the admission JSON's `records` array; omitted here for
length, not omitted from the gate.

Ice namelist: `namelist_ice_ref` = `cfgs/SHARED/namelist_ice_ref` (same
SHA-256 as the V2 lane above — the run directory's copy is byte-identical to
the shared file), overridden by the Phase-2x `namelist_ice_cfg` (SHA-256
`6b647863137b518b95ff97f83975d9afcb3944b7f6e8d63e45a05f494f5edc89`, per the
phase-2x receipt), which reproduces ORCA1's own 91 active assignments
field-for-field except the registered `nn_iceini_file` input-grid
substitution.

## 3. Resolved-namelist diff (P2Y-4a)

**CONFIRMED.** The Phase-2x ice-namelist gate compares only *cfg override
text* against *cfg override text* — it never reads either side's
`namelist_ice_ref`, so it cannot see whether a ref-level difference survives
resolution. `nemo_testcase_l4_orca2_phase2y_ice_namelist_resolved_gate.py`
resolves both sides properly (`{**parse(ref), **parse(cfg)}`, the same
two-pass semantics NEMO's own namelist read uses) and diffs the full
185-field resolved namelist. Result (JSON
`nemo_testcases_l4_orca2_phase2y_ice_namelist_resolved.json`):

**Exactly one resolved field differs: `namini.nn_iceini_file`** (variant
resolves `0`, ORCA1 resolves `1` — the already-registered, deliberate
input-grid substitution).

The five named ref-level differences are all **masked** — both cfgs override
every one of them to the identical value, so `resolved_equal=true` for all
five despite differing ref defaults:

| field | variant ref default | ORCA1 ref default | both cfgs resolve to | reachable under `nn_icesal=2`? |
|---|---:|---:|---:|---|
| `rn_time_gd` | 1.728e+5 | 1.73e+6 | 1.73e+6 | **yes** — `icethd_sal.F90` `CASE(2)` (lines 207-249) reads it, gated live by `ln_drainage=.true.` (resolved identically on both sides) |
| `rn_Rc_RJW` | 8.0 | 9.0 | 9.0 | no — read only inside `CASE(4)` |
| `rn_alpha_GN` | 4.6e-4 | 5.1e-4 | 5.4e-4 | no — read only inside `CASE(4)` |
| `rn_Rc_GN` | 8.1 | 10.0 | 6.4 | no — read only inside `CASE(4)` |
| `rn_alpha_CW` | 5.0e-7 | 4.8e-7 | 9.e-7 | no — read only inside `CASE(4)` |

Reachability is read directly from `icethd_sal.F90`'s
`SELECT CASE(nn_icesal)`: `CASE(2)` ("time varying salinity with linear
profile", Vancoppenolle et al. 2005) is the branch both sides resolve
(`nn_icesal=2` on both), and its body (lines 207-249) references only
`ln_drainage`, `rn_time_gd`, `ln_flushing`, `rn_time_fl`, `rn_sal_himin`,
`rn_sal_fl`, `rn_sal_gd`, `rn_simin`, `rn_sinew`. `rn_Rc_RJW`, `rn_alpha_GN`,
`rn_Rc_GN`, `rn_alpha_CW`, `rn_alpha_RJW`, `nn_sal_scheme` are declared in
the same `NAMELIST/namthd_sal/` (so both sides still resolve a value for
them) but are referenced only inside `CASE(4)` (Gravity Drainage and
Flushing, lines 256+) — structurally unreachable code under `nn_icesal=2`
regardless of their resolved value. The self-test plants a resolved-only
difference (a field only one side's cfg touches) and a zero-diff control to
prove the diff mechanism actually fires and can pass.

## 4. `stprk3.F90` licence hygiene (carried review item)

**CONFIRMED, separate commit.** The last full NEMO source tracked in git,
`scripts/validate/ocean_fidelity/testcases/nemo502_MY_SRC/stprk3.F90`, was
also discovered **stale**: it matched neither the file actually compiled
into the current ORCA2 Lane-4 binaries nor the GYRE lane's own copy (a
three-way hash mismatch confirmed against the shipped base, the GYRE `cfgs/`
copy, and the `ORCA2_OMIP_L4` `cfgs/` copy). It has been replaced with a
unified diff against the hash-pinned shipped NEMO 5.0.2 `OCE` source
(`d12b246db6b77b122ef1c53a485a3d742ce4acf113a58684a639c9031f0a9e1a`) plus the
tracked include it needs
(`l4_oracle_canon_subroutines.h90` — original write-only masking glue, not
NEMO-derived). Patch replay is exact: applying
`stprk3.F90.patch` to the shipped base reproduces the file actually used to
build every current ORCA2 Lane-4 binary byte for byte
(SHA-256 `9d0318fda246ef1ed3df50172b9d72a5e0b5df879b9a661b38622c9078078989`).
Untouched copies of shipped base / applied file / patch / header are hashed
under
`/data/abyssal/dbalwada/nemo-testcases-l4/phase2y/stprk3_licence_hygiene/MANIFEST.sha256`.

**Flagged, not fixed (out of Lane-4 scope):**
`packages/ocean/legoesm/ocean/fidelity/time_levels.py` cites this same git
path at specific line numbers for the GYRE lane (testcase lanes 1/2), and
that citation was *already* stale before this commit — the GYRE `cfgs/`
copy's step-entry trigger condition had independently diverged from the
committed text at the exact cited lines. This is unrelated to Lane 4;
flagging for whoever owns Lane 1/2 provenance.

## 5. Anomalous mtimes in the shared NEMO build tree (flag only)

Per the coordinator's instruction: while investigating where the ORCA1-ice
binary is actually built (`/tmp/nemo-orca2-phase2p/cfgs/
ORCA2_ORCA1ICE_OMIP_L4/MY_SRC/`, **not**
`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_OMIP_L4/` — the two
trees are unrelated and happen to share a similar config-naming convention;
the coordinator's own message confirms both are shared, retained provenance
and must not be reset), the following files carry mtimes I did not create
and did not touch, listed here as a flag only — **nothing was reverted**:

| file (in `/tmp/nemo-orca2-phase2p/cfgs/ORCA2_ORCA1ICE_OMIP_L4/MY_SRC/`) | mtime (2026-09-05) | apparent owner (by stream family) |
|---|---|---|
| `dynhpg.F90`, `dynvor.F90` | 00:08 | dynamics/vorticity — pre-existing Lane-4 rkstage2 instrumentation |
| `eosbn2.F90`, `traqsr.F90` | 00:08-00:09 | EOS/shortwave — pre-existing Lane-4 instrumentation |
| `icestp.F90`, `icedyn_adv_pra.F90`, `icethd.F90` | 00:14 (pre-edit) | phase-2w/2x SI3 |
| `stprk3.F90` | 00:43 | phase-2x-era step/frame writer (the file now patch-tracked, §4) |
| `stprk3_stg.F90` | 06:54-06:58 | RK3-stage writer, unclear phase |
| `l4_oracle_canon_subroutines.h90` | 06:58 | shared canon helper |
| `dynspg_ts.F90` | 08:25 | barotropic solver — unclear phase |
| `zdfphy.F90` | 08:24 | ZDF entry — phase-2o/2p family |
| `zdftke.F90` | 10:38 | TKE — phase-2u/2v family (P2Y-2/3 reads this file without editing it) |
| `icedyn_rhg_evp.F90`, `icedyn_adv_pra.F90`, `icethd.F90` | 11:13-11:35 | phase-2x's own edits, immediately before its 11:37:58 build |

All mtimes in this specific directory are `<=` phase-2x's own build-start
timestamp (11:37:58); nothing here has been touched since that build
completed, and the full directory (18 files) matches the phase-2x receipt's
three cited applied hashes exactly (icethd.F90, icedyn_rhg_evp.F90,
icedyn_adv_pra.F90 — verified in this session before any edit was made).
This is reported as a finding per the coordinator's instruction, not acted
on.

## ASKED / UNASKED

| item | status | disposition |
|---|---|---|
| admit the ORCA1-ice twins per P2Y-1 | ASKED | CONFIRMED PASS, pinned as `VARIANT_ORACLE_ORCA1ICE` |
| retain twin b as witness only | ASKED (preregistered UNASKED) | CONFIRMED, not pinned as a second root |
| report first differing ocean stream vs V2 | ASKED | CONFIRMED `oracle_step_entry_kt00000001.bin`, ssh only, traced to `iceistate.F90:400-406` |
| compare RESOLVED ice namelists | ASKED | CONFIRMED, one field differs, five named ref-differences all masked |
| assess reachability under `nn_icesal=2` | ASKED | CONFIRMED via `icethd_sal.F90` SELECT CASE, four of five unreachable |
| replace stale `stprk3.F90` full source with a patch | ASKED | CONFIRMED, patch replay exact |
| use the phase-2x-frozen copy, not the live one, for the patch | ASKED-enabling, confirmed by independent peer review | CONFIRMED |
| reset/overwrite any shared MY_SRC tree | explicitly forbidden by coordinator | not done |
| create an isolated new config for phase-2z instead | ASKED by coordinator | see phase-2z receipt |
| edit shipped NEMO, delete artifacts, commit multi-MB data, push, run MPI | forbidden | not done |
