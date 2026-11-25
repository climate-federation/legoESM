# ORCA2 round 293 — fold-image V-depth operand split

Date: 2026-10-16

Status: **HELD**.  The independent OMT-4 kt=1 stage-1 substep-1 live
V-face-depth debt is the reference face depth itself: the production
reconstruction differs from NEMO's recorded `hv_0` on 13/13,320 slab slots,
maximum 719.0 m at `[j=147,i=58]`.  Replacing only `hv_0` by the card's raw
NEMO operand makes the active depth bit-exact, 0/8,589 unequal.  The
preregistered surface-mask/T-neighbour cancelling partner is **REFUTED at
this boundary**: both later arms are null after `hv_0` closes it.  No package,
card, configuration, or carried-state change lands.

Every scientific number below is **independent OMT-4, kt=1 stage 1,
external substep 1**.  No given-entry result is mixed into the table.

## Provenance and instrument

- Lane base: `89831cfbe`.
- Frozen preregistration: `PREREG_nemo_testcases_l4_orca2_round293.md`,
  commit `f67985f51`; its two dated instrument clarifications were committed
  before the accepted measurement.
- Accepted gate and result classifier: `0f2b416ed`; zero-V-null test:
  `3e2d5728d`.
- Evidence root:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round293/`.
- Parent boundary: round 292's immutable `measurement.json` at commit
  `477688578`.
- Oracle operands: admitted round-222 `oracle_bt_substeps` and
  `oracle_bt_advmean_operands` twins.  Card geometry: instantiated OMT-4
  card, production D114 route, fp64/libm on CPU.
- Instrument: two independent implementations evaluated on the admitted,
  bit-exact substep-1 midpoint SSH — the production `_nemo_ssh_avg_apply`
  and round 245's source-ordered `_v_depth_replay`.  They are array-identical
  on the complete slab and independently reproduce round 292's 13-cell /
  719 m boundary before any operand substitution is read.
- Search-before-build reused the production face-depth function, round 245's
  replay/native mappings, round 235's admitted readers, and round 292's
  exact midpoint-state registry.  No parser, formula, model runner, or NEMO
  observer was duplicated.
- The original three-arm full-step process and its isolated baseline were
  killed during JAX compilation before emitting any report.  Their empty and
  warning-only logs are retained; no number from them is used.  The accepted
  offline replay completed in 21 seconds.

## Source-ordered operand result

The source statement's direct operands compare as follows before
substitution:

| operand | unequal / slab slots | maximum | disposition |
|---|---:|---:|---|
| reconstructed `hv_0` | 13 / 13,320 | 719.0 m | **first non-bit operand** |
| `r1_e1e2v` | 0 / 13,320 | 0 | exact |
| south/north `e1e2t` | 0 / 13,320 | 0 | exact |
| compact surface `ssvmask` | 35 / 13,320 | 1 | non-bit, but null at substep 1 after raw `hv_0` |

The active-support causal table is:

| arm | V depth unequal / 8,589 | max (m) | `zhV` unequal / 8,589 |
|---|---:|---:|---:|
| production reconstruction | 13 | 719.0 | 0 |
| raw NEMO `hv_0` only | 0 | 0 | 0 |
| raw `hv_0` + raw V surface mask | 0 | 0 | 0 |
| raw `hv_0` + mask + T-pivot north neighbour | 0 | 0 | 0 |

Thus one source-exact operand closes the named first boundary.  The mask and
T-neighbour do not form a cancelling pair **at substep 1**; their round-245
pair acts only after sea level makes that surface term live.  The immediate
transport cannot validate causality here because midpoint V is exactly zero,
so `zhV` is already bit-exact with the wrong depth.  This null is recorded as
a refutation, not as consumer closure.

## Frozen prediction ledger

| ID | Verdict | Mechanical result |
|---|---|---|
| R293-P1 | **CONFIRMED** | Twins admit; card selectors print; CPU/fp64/libm and the offline/passive protocol are pinned. |
| R293-P2 | **CONFIRMED** | Independent replay is array-identical to production and reproduces 13 cells / 719 m. |
| R293-P3 | **REFUTED** | The production route's reconstructed `hv_0`, not an area operand, is first non-bit. |
| R293-P4 | **REFUTED** | Raw `hv_0` alone closes; raw surface mask and T-neighbour are null at this boundary. |
| R293-P5 | **REFUTED — NULL CONSUMER** | `zhV` was already exact because midpoint V is exactly zero. |
| R293-P6 | **CONFIRMED** | All nine plants print `STATUS PLANT-FIRED` and exit 2. |

## Compiled source citation

The executing ORCA2 record build computes both mid-step face depths in
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:509-519`; its V statement reads
the domain's stored `hv_0` first, then adds the two-area sea-level average and
surface mask.  The round-293 split follows that written association and shows
that the first operand alone owns substep 1.

## Validation

- Measurement/classification JSON and all plant logs are retained under the
  evidence root.  Measurement/classification SHA256 values are
  `7598210c766848e8bb8fa085ec7433ec33cb7f4f775f4a5c820eec00bb2901a9`
  and `2ccc466ae9387df2aae0b9c1686af0dcc3fb9b409f6a69962f9606497348f0a9`.
- Focused round-291/292/293 and citation tests pass 25/25.  Log SHA256:
  `34e1648c85c361d205c8ef6a3ffeeedb85cd109972359bba4cf7a52b263e7de8`.
- Citation gate, planted citation shift, and cumulative default gate:
  PASS / plant exits 1 / PASS, with zero unmapped citations.  Artifact
  SHA256 values are
  `3c2cee63ac560b90f989cb204f4bd3065bdb032c105ac365cd0df02f39210ed1`,
  `76b887f7c3aa3b23b7d88bf3c3bf9160152f0b6161556795a18aa1ae31dfd0f8`,
  and `71c7af60f3aec05cd50a6c8d56958eb42eca93f8076a34c2c7b20db26ee5cc0e`.
- The one permitted `tests/ocean/fidelity -n 12` battery reached 94%, then
  all real pytest processes disappeared without a terminal summary,
  reproducing the registered late worker-loss pattern; it was not repeated.
  Nine registered branch-tip reds printed first: GYRE round-129 record hash,
  two round-33 mutation anchors, OMT-1 module-edge ratchet, round-228 citation
  map, GYRE year-owner layout plant, dirty-escape scope, worktree stamping,
  and SI3 scalar-math provenance.  All three round-293 tests passed.  Log
  SHA256: `ad9ba3fda24266ac1ff96028395e26213143afbf22d1dc5e3ac5ff0750cd0d5d`.
- Independent review: operator.  In-sandbox read-only review attempt:
  unavailable — `codex exec --sandbox read-only` exited before reading the
  diff because its app-server client attempted a write.  Log SHA256:
  `eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.
- No model or card file changed, so ORCA2/GYRE/DINO/tank trajectory gates are
  off-claim.

## Configuration choices

- **ASKED:** none.
- **UNASKED:** none.
- Decision 125's completed-stage top-V exchange, the resolved stage mask,
  and the vector-form fold unit remain off.
- Decision 126 remains pending; shipped runoff is unchanged and the decision
  is not re-raised.
- Sea ice and the shipped card's `unmeasured_features` tuple are unchanged.

## OPEN

1. Keep raw `hv_0` inside the private vector-form fold unit: although it
   closes the first substep-1 boundary, the complete unit is known to expose
   later compensating debt and partial landing is forbidden.  Resume the
   source-ordered substep table with **only raw `hv_0` substituted** and name
   the next valued statement; the round-245 mask/T-neighbour pair is the
   registered candidate once the surface term becomes live.
2. Keep the resolved stage mask and completed-stage top-V exchange held until
   that next boundary is attributed atomically.
3. Leave shipped runoff unchanged while Decision 126 remains pending.
