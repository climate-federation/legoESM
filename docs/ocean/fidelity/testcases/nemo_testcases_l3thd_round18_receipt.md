# SI3 lane 3b round 18 receipt — cross cards and NCAR ocean bulk

Date: 2026-09-05

Tracker: `climate-federation/legoESM#1699`

Parent: `c7ca3866725b8e1856d18663d3abe8dbb8406870`

Preregistration: `9aa377a1319`

Result: **MIXED.**  The handed-off ORCA2 O1 Large & Yeager NCAR boundary is
**CONFIRMED AT_BAR and bit-identical: 0 / 158,292 source-owned wet-cell
rows**.  The cross-card audit did not close.  It found an older branch
composition problem after restoring NEMO's complete UP3 selector, and the
canonical GYRE integration arm moved slightly more cells away from NEMO than
toward it.  Those rows remain DEBT owned by the GYRE lane.  AT_BAR and
bit-identical are reported separately throughout.

No shipped NEMO file, shipped configuration, ORCA input, or retained run was
modified or deleted.  All executions were CPU/fp64.  Numerical outputs and
multi-megabyte sidecars are retained under `/data`, not git.

Round-18 commit ledger: preregistration `9aa377a1319`; complete UP3 card
selection `c9967f54d9d`; masked-stage zero materialization `a200b4bd4a6`;
shared NCAR implementation and gate `5378383d822`; initial receipt
`7f3d7bdd40c`.

## 1. Cross-card selector resolution

NEMO does not define a legitimate split program consisting of horizontal
flux-UP3 plus a separately disabled vertical advection arm.  In
`dynadv.F90:35-46`, `ln_dynadv_OFF` is the complete no-momentum-advection
program (`np_LIN_dyn`) and `ln_dynadv_up3` is the complete flux-UP3 program
(`np_FLX_up3`).  The executable dispatch has only vector, flux-C2, and
flux-UP3 cases (`dynadv.F90:78-90`); the vector case explicitly calls vertical
`dyn_zad`, while UP3 enters the single `dyn_adv_up3` implementation.  Init
maps each selector to one enum and requires exactly one (`:128-134`).

Therefore the validator was correct.  The old legoESM LOCK/OVERFLOW cards
were wrong: they selected horizontal flux-UP3 while inheriting vertical OFF.
Commit `c9967f54d9d` selects the already shared `nemo_up3` vertical identity on
those two cards.  The one-column coupled slab continues to select OFF/OFF.
No validation was weakened and no numerical arm was added.

Restoring the complete program exposed non-finite dry-face workspace.  The
ranked arm in preregistration addendum A1 confirmed that IEEE `NaN * 0`
entered a live UP3 neighbour stencil.  Commit `a200b4bd4a6` materializes zero
at the existing masked WS-RK3 boundaries.  This makes the LOCK stage sweep
finite but is not sufficient for OVERFLOW; it is a prerequisite correction,
not a claim that the three slab changes caused the remaining debt.

## 2. Direct current-branch cross-card results

These are actual executions of the checked-in stage and trajectory gates
after commits `c9967f54d9d` and `a200b4bd4a6`.  Exit 1 denotes measured DEBT;
exit 2 denotes a fail-closed non-finite/schema error.

| gate | exit | measured result |
|---|---:|---|
| LOCK stage | 1 | stage 1 AT_BAR, `2.5587171270657905e-17`, `120 / 2,540` non-bit; stage 2 first DEBT, `9.76564494417978e-11`, `140 / 2,540`; stage 3 `1.7120836275389892e-10`, `140 / 2,540` |
| OVERFLOW stage | 2 | `kt1.stage1.faithful.instantaneous_u` candidate non-finite; no numerical verdict |
| LOCK kt1..10 trajectory | 1 | stops fail-closed at `kt3.before.T` candidate non-finite |
| OVERFLOW kt1..10 trajectory | 1 | stops fail-closed at `kt2.before.T` candidate non-finite |

The LOCK stage row-scale ULP maxima are `0.115234375` at stage 1,
`439805.54931640625` at stage 2, and `771053.9187011719` at stage 3.  Thus
stage 1 is AT_BAR but not bit-identical; stage 2 is the first bar failure.
The historical lane-1 exact-card artifact remains untouched at
`nemo-testcases-l1/stage3_remainder/after/gates/lock_stage_sweep_kt2.json`
(SHA-256 `fe6a61fa...2faab`): all three stages were AT_BAR there.  A detached
`dad9dd30487` bisection reproduced the restored-UP3 dry/non-finite issue before
the round-15/16/17 shared changes.  Consequently it is not attributed to
those changes.

The existing LOCK stage plant and OVERFLOW trajectory plant each exited 1
(SHA-256 `ad871663...e2a61` and `4779cb19...7111`).  They prove the production
gate exits red, but because the unplanted cards are already DEBT they do not
turn these rows into a cross-card certification.

### Exact current-branch GYRE block

The GYRE production gate is absent from this branch.  Running the canonical
`c83f73c23ff8` gate against this branch's packages fails before model
construction with:

```text
ImportError: cannot import name '_nemo_literal_barotropic_coriolis'
```

The retained stdout has SHA-256 `ed177965...f667`.  This is the exact reason
the current checkout cannot execute the current GYRE card: the branch predates
the GYRE round-19 canonical selector/helper surface.  Validation was not
weakened and the missing GYRE implementation was not copied piecemeal.

## 3. Canonical GYRE integration discriminator (Rule 8/12)

To measure the shared Kmm change without a Frankenstein import, a detached
integration worktree started from canonical GYRE `c83f73c23ff8`.  Its newer
shared implementation already subsumes the lane's ZDF-input ordering and QCO
layer-thickness identities.  The sole non-subsumed production arm was the
source-literal Kmm seed from `dynspg_ts.F90:484-493`, composed in temporary
commit `980e06a41222e6d3b2cb690c05a9f183b79b1515`.  A WRITE-only gate extension
stored per-cell candidates and oracle values; it did not alter model results.
Both baseline and arm used production JIT, CPU, fp64, scalar libm, the same
Round-19 Oracle V2, and kt=1..10.

The preregistered prediction that live GYRE kt=2 velocity would move toward
NEMO is **REFUTED**: no field moves at kt=1 or kt=2.  The first movement is at
kt=3.  The first-over-bar register remains `kt=2: T,S,u,v` in both arms.

| entry step | moved fields | changed cells | toward NEMO | away from NEMO | largest worsening, row-scale ulp |
|---:|---|---:|---:|---:|---:|
| 1 | none | 0 | 0 | 0 | 0 |
| 2 | none | 0 | 0 | 0 | 0 |
| 3 | T,S,u,v | 24,156 | 12,265 | 11,887 | 2.96875 |
| 4 | T,S,u,v,ssh | 33,780 | 17,558 | 16,221 | 69 |
| 5 | T,S,u,v,ssh | 36,261 | 17,363 | 18,898 | 6,506 |
| 6 | T,S,u,v,ssh | 41,640 | 19,635 | 22,005 | 12,110 |
| 7 | T,S,u,v,ssh | 51,476 | 25,783 | 25,693 | 16,928 |
| 8 | T,S,u,v,ssh | 58,652 | 29,632 | 29,020 | 50,921.890625 |
| 9 | T,S,u,v,ssh | 62,891 | 31,669 | 31,222 | 41,531.4375 |
| 10 | T,S,u,v,ssh | 65,168 | 32,991 | 32,177 | 37,397 |

Across the 39 moved step/field rows, 374,024 candidate cells change;
186,896 residuals improve, 187,123 worsen, and five change bits without
changing absolute residual.  The committed receipt does not compress this to
a favorable direction.  The external Rule-12 JSON enumerates every row with
candidate movement, before/after residual, improved/worsened counts, and the
row-scale ULP.  Owner-of-record is the GYRE lane for all 39 rows.

The later canonical LOCK/OVERFLOW integration sweep was also launched after
the temporary composition, but the CPU process was terminated while compiling
the first stage gate and produced no score.  It is **UNMEASURED**, not silently
promoted from GYRE's round-19 baseline.  The direct current-branch results in
section 2 are the only round-18 LOCK/OVERFLOW numerical claims.

### Shared-change register

| shared change | canonical composition | measured Rule 8/12 disposition |
|---|---|---|
| round-15 ZDF input before stage-3 mean imposition | already subsumed by canonical GYRE | no separable arm in this composition; GYRE owner-of-record |
| round-16 QCO layer thickness under `nemo_literal` | already subsumed by canonical GYRE | no separable arm in this composition; GYRE owner-of-record |
| round-17 Kmm barotropic seed | temporary source-literal arm | kt1-2 unchanged; mixed movements start kt3; prediction of kt2 improvement refuted |
| round-17 quadratic drag | GYRE already selects its source-defined quadratic drag | no selector movement; not a separable GYRE arm |

## 4. NCAR pre-implementation search and resolved oracle

The search found one existing shared home in
`packages/core/legoesm/core/bulk_flux.py`: `validate_bulk_scheme`, the
Large--Yeager neutral helper, q-saturation and MOST utilities, and the
previously certified `nemo_si3_constant` arm.  The coupler already dispatches
through this core machinery.  At this round the older
`packages/ocean/legoesm/ocean/bulk_flux_omip.py` was reduced to a compatibility
facade over that shared implementation.  Round 19 removes that facade and
routes every caller directly to the one production implementation, selected
as `nemo_ncar`.

The oracle is the ORCA2 lane's canonical O1 twins:

- `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2g_o1canon_a_10step_np2`;
- the corresponding `_b_` twin.

Both O1 records are 3,090,336 bytes and SHA-256
`751b2d9181778e81f01bc5872d47100afc0fc3ad02c4ffcccc9613a0af2ad045`.
Frame 0 is `(1,1,0,90,148,9,0,64)` and contains the nine mapped CORE fields;
frame 1 is `(1,1,1,90,148,20,0,64)`.  The comparison is over the 8,794 wet
rank-0 T cells.  Mapping is oracle-supplied and is not certified here.

The resolved ORCA2 deck selects `ln_NCAR=.true.` at ORCA2
`namelist_cfg:101`; `rn_zqt=rn_zu=10` and `rn_pfac=rn_efac=1`.  NEMO reads and
preprocesses the forcing at `sbcblk.F90:559-629`, dispatches NCAR at
`:833-838`, calls the bulk formulas at `:895-922`, and assembles stress at
`:929-943`.  `blk_oce_2` forms heat, longwave, freshwater, and the continued
left-associated non-solar sum at `:1024-1050`.

## 5. Source-literal NCAR identity

The implementation follows the executed NEMO statements rather than a
textbook reconstruction:

- `sbcblk_algo_ncar.F90:110-115`: fixed iteration count, equal-height test,
  and 0.5 m/s wind floor;
- `:117-135`: virtual-temperature stability guess and neutral Cd/Ch/Ce;
- `:147-217`: five fixed iterations, turbulent scales, Obukhov inverse,
  `|zeta|<=10`, height correction, and coefficient updates;
- `:244-255`: neutral drag, including its separately materialized wind-cubed
  then wind-sixth power and cyclone branch;
- `:278,290`: heat and moisture neutral coefficients;
- `:312-326,352-363`: stable/unstable momentum and scalar profiles;
- `sbc_phy.F90:235-282,321-392`: pressure/temperature and density;
- `sbc_phy.F90:428-489,533-566,630-751`: latent heat, moist heat capacity,
  Obukhov and q-saturation;
- `sbc_phy.F90:883-925,1004-1027`: turbulent and longwave assembly.

Every source statement is rounded once through the one shared
`nemo_source_round`.  NEMO's EXP calls use the shared scalar-libm precision
policy.  The shared policy does not provide LOG/LOG10; the machine-specific
O1 measurement confirms native JAX LOG/LOG10/POW bits at every executed wet
cell, so no private transcendental was invented.  Constants and algorithm
coefficients live in `legoesm.constants`, not inline in the formulas.

The ranked walk found three concrete expression owners: removing an
unfaithful q/qsat clipping branch cleared the 117 large `theta_air` rows;
materializing a same-shaped NEMO constant prevented XLA's divide-by-constant
reciprocal rewrite; and reusing NEMO's round-tripped `ptsk` plus preserving the
continued qns accumulator order cleared the remaining heat/freshwater rows.
These are unbranched parts of the `nemo_ncar` identity.

## 6. NCAR gate, coverage, and controls

The gate is
`scripts/validate/ocean_fidelity/testcases/nemo_ncar_o1_bulk_gate.py`.  It
checks both record hashes, both headers, EOF, the 8,794-cell mask, CPU, fp64,
production JIT, explicit scalar-libm, runtime provenance, and the complete
output disposition.  Bit claims fail closed outside Python 3.13.0, JAX/JAXLIB
0.10.0, and NumPy 2.4.4.

| outputs | status | bit unequal / n | over bar / n |
|---|---|---:|---:|
| theta_air, q_air, precip, sst, ssu, ssv, tsk, ssq | VERIFIED | each `0 / 8,794` | each `0 / 8,794` |
| sensible, latent, evap, qsr, qns, emp | VERIFIED | each `0 / 8,794` | each `0 / 8,794` |
| utau, vtau, taum, wndm | VERIFIED | each `0 / 8,794` | each `0 / 8,794` |
| `cd_du` | WAIVED | not scored | inactive `ln_abl`; canonical writer zero only |
| `qlwn` | WAIVED | not scored | inactive MFS optional result; canonical writer zero only |
| **total source-owned** | **VERIFIED** | **`0 / 158,292`** | **`0 / 158,292`** |

There is no first non-bit operand and no first over-bar row.  The result is
both **CONFIRMED AT_BAR** and **CONFIRMED bit-identical** on the registered
stack.  The gate also reports row-scale ULP and nonzero-oracle relative error;
all maxima are zero.

The coverage register contains **20 fields**: 18 VERIFIED/scored fields and
two source-backed WAIVED fields (`cd_du`, `qlwn`).  All 18 independent
in-process one-ULP plants for the VERIFIED rows are required to produce a
non-bit row and are stamped `PASS_NONZERO`, exit code 1.  The separate qns CLI
plant also exits 1 and changes exactly `1 / 8,794` qns rows.  A missing stream
raises named `GateError`, and a malformed header does likewise.

## 7. Tests

```text
47 passed in 38.08s
```

This is the combined selector, stage/trajectory control, NCAR gate, and legacy
LY09 compatibility set.  The touched-file constants ratchet invocation is:

```text
2 passed, 1 skipped in 0.54s
```

The skip is `constants.py`, which is the canonical constants definition file;
both executable files touched by the NCAR implementation pass.  The initial
combined invocation (before correcting two new gate-test expectations)
produced 3,413 passes and 2 skips; its other five failures were ratchet rows in
pre-existing unrelated FV3/DINO files.  None is touched by this lane.  Unlike
the retracted round-2 claim, no new-file failure is described as pre-existing.

## 8. Evidence hashes

| artifact | bytes | SHA-256 |
|---|---:|---|
| NCAR gate JSON | 14,881 | `30fe378e3e15f07f7eb0e79e67b8f8e717e2fc38d21384859bafc73c1c1c6dca` |
| NCAR qns plant JSON | 15,392 | `4141d3cbbd9f444a8ec1fc17c3570bee5106379d3e9a949158fa87f8e94fb05b` |
| direct LOCK stage JSON | 13,914 | `23f9e97cd92dc3ff2d6b44945d344bf60a26242ab2de4a15e548180c7d514f6e` |
| direct OVERFLOW stage stdout | 67 | `cea425831c5e1400afbf5ee582eb6ac4d8b585375fa5932b633cae0a4c07d17a` |
| direct LOCK trajectory stdout | 58 | `1630581dca65cb6e972d1f54ea8ec8bebda19675aaad4678b1963a76ffe8e3a7` |
| direct OVERFLOW trajectory stdout | 53 | `75ab761c3847038736ae90c777b71f4a26f5e274f12e9c13f41bf9df1665b52b` |
| current-branch GYRE import failure | 929 | `ed177965f413efdb4b89f488197092f686fda54eff648c94c87958f9d61df667` |
| canonical GYRE baseline | 28,163 | `865463e5bd6c18c6a242b9fd8880a2d85d8b58ad04d21c36ab3a83792e6f856a` |
| canonical GYRE plus Kmm arm | 28,174 | `a5160222d3be711b3e93d6b79915676d1bca121400317664678c5c6b76d753f5` |
| complete cellwise Rule-12 register | 23,398 | `aa60cf45a5101853efa47312b8fe74c8068705837d87a2115da0d23efcbe0e24` |
| GYRE baseline residual sidecar | 8,311,571 | `76260e15db3d02e34e8ef1cf1abfec645c0f2888c21e2d8f58685215d598568a` |
| GYRE arm residual sidecar | 8,311,903 | `17499e11c7b5eed876d70d48b6b3d634cb962a4166b2e9ce38240cc8b375c99f` |

Roots are `/data/abyssal/dbalwada/nemo-testcases-l3/round18_{cross_cards,ncar_bulk}`.

## 9. FLAGGED FOR FUTURE DELETION (nothing deleted)

- `c1d_omip_l3_coupled10m_r13_oracle{,_b,_c,_d,_e,_f,_g,_h}` are the eight
  unpinned round-13 iteration roots.  `_i` remains the accepted pinned root.
- `c1d_omip_l3_coupled10m_r17_oracle_a` is the failed run whose first build
  used the wrong assumed-shape lower bound.
- `/tmp/codex-si3thd-r18-bisect`,
  `/tmp/codex-si3thd-r18-gyre-gate`,
  `/tmp/codex-si3thd-r18-gyre-baseline2`, and
  `/tmp/codex-si3thd-r18-gyre-integration` are retained diagnostic trees.

## 10. ASKED / UNASKED

| choice or action | state | disposition |
|---|---|---|
| resolve OFF versus flux-UP3 from NEMO source | ASKED | validator retained; inconsistent L1 cards repaired |
| run LOCK/OVERFLOW stage and kt1..10 gates | ASKED | run; direct results are DEBT/error as listed, never promoted |
| run GYRE production-JIT kt1..10 and Rule 8/12 | ASKED | current-branch API block recorded; canonical integration discriminator completed |
| expected kt2 GYRE movement toward NEMO | ASKED prediction | REFUTED; kt1-2 unchanged and later movement mixed |
| certify ORCA2 O1 NCAR bulk in the shared machinery | ASKED | 20 registered fields (18 VERIFIED plants bind; two WAIVED); 0 / 158,292 source-owned rows |
| import the newer GYRE numerical stack into lane 3b | UNASKED | not done; would be a large cross-lane merge |
| certify O1 mapping, `cd_du`, `qlwn`, RGB, SI3, or downstream ocean fields | UNASKED | outside this boundary / explicitly waived |
| delete retained roots or diagnostics | UNASKED | none deleted; flagged only |
| modify shipped NEMO, use GPU, add large runtime output to git, push | UNASKED | none done |

## 11. CONFIRMED / PLAUSIBLE

**CONFIRMED:** NEMO's complete momentum-advection selector semantics; the
old LOCK/OVERFLOW card contradiction; the direct cross-card outcomes; the
canonical GYRE per-cell before/after movement; both canonical O1 hashes and
schemas; complete NCAR coverage disposition; `0 / 158,292` source-owned bit
identity; and all NCAR plants.  **UNMEASURED:** a successful post-composition
canonical LOCK/OVERFLOW sweep, due CPU compilation termination.  There is no
PLAUSIBLE numerical fidelity claim in this receipt.

Round-18 measurements and this receipt are Codex-internal.  No independent
review verdict is claimed without a branch artifact naming the reviewer,
commit, and verdict.
