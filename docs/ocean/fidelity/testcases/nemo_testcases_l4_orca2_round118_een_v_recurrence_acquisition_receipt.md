# ORCA2 round 118 — V EEN recurrence acquisition

Date: 2026-10-03. Base `bbf91aa1429f5a7796e5bf4d4d73ca5a929a5093`.
Every eventual ocean number from this rung-0 record is **independent** because
rung 0 starts from NEMO's own from-rest state. No model physics, card field,
configuration value, carried state, stabilizer, sea-ice selector, or
`unmeasured_features` entry changed.

## Verdict

**STOPPED_FOR_RECORD.** Rounds 115--117 close all four U-grid EEN recurrence
paths only after NEMO's southern V-mask association and ordinary IEEE-zero
addition are applied. A shared production landing remains forbidden because
the four V-grid paths are unmeasured.

This round commits a self-describing, rank-complete additions-only recorder
for northwest, northeast, southwest, and southeast V. Its exact-source syntax
proof and all five record-independent launcher controls pass. The new record
directory is absent, so R118-P1 through R118-P4 remain unmeasured and no ocean
number is inferred from preflight.

The operator action is:

```text
/tmp/autopilot-orca2-944922297/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round118_een_v_recurrence_acquisition/run.sh --run
```

It creates target `ORCA2_OMIP_L4_R118EENVREC` and run directory
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round118/acquisition/orca2_rung0_een_v_recurrence_ranked_10step_np2`.

## Compiled source boundary

The compiled rung-0 program evaluates northwest, northeast, southwest, then
southeast V. Each statement performs one carried addition of live V
thickness, the corresponding neighboring U thickness, that neighboring U
mask, and the source-ordered `zpvo` coefficient
(`ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/dynspg_ts.f90:1324-1327`).

The patch places before/after calls around these four untouched statements and
one dump after the complete V loop. It removes or rewrites zero NEMO source
lines. For every path it records the four inputs, separately materialized
product, accumulator before, and accumulator after, plus `mbkv`. Admission
requires both the product and recurrence to replay bitwise, so plausible but
nonexecuted arithmetic cannot admit the record.

## Admission contract and controls

The checker reads every field name, rank, dimension, and payload length from
the record header. It requires exactly-once coverage of the `148 x 180` global
domain, rejects the dummy level and all writes outside `mbkv`, and compares
each path's last executed value with the admitted round-105 V accumulator using
the explicit native `(i,j)` to global `(j,i)` association.

Observational passivity requires all twenty kt=1..10 restart shards and all
eight inherited round-105/107/110/116 record shards to be byte-identical to
the admitted round-116 run. The launcher pins the source, CPP keys, binary,
namelist, deck manifest, input manifest, patch, writer, checker, launcher, and
preregistration by SHA-256 content. Its target, run directory, and absolute
per-rank output path are new.

The exact Fortran writer and patched source compile under the pinned rung-0
includes. Layout, absolute-path, five-recorder environment, producer-content,
and rank-log controls each exit 69 with `STATUS PLANT-FIRED`. Runtime header,
field-name, dimensions, truncation, missing-field, duplicate-rank,
bottom-index, product, recurrence, inherited-terminal, association, and
restart-byte controls are committed but cannot execute until the record
exists. Preflight and control logs are under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round118/`.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R118-P1: recorder is observationally passive | **UNMEASURED**, operator record absent |
| R118-P2: all four products replay from recorded operands | **UNMEASURED**, operator record absent |
| R118-P3: all four recurrences replay under host IEEE addition | **UNMEASURED**, operator record absent |
| R118-P4: northwest V is the next source walk and any zero-sign-only recurrence difference closes under the registered arm | **UNMEASURED**, no recurrence value is inferred from preflight |
| R118-P5: no production landing before the streams admit | **CONFIRMED**, no `packages/` file changes |

## Validation and review

The focused parser/schema test passes 5/5 and the round-107/109/110/114--118
chain passes 26/26. The single required `tests/ocean/fidelity -n 12` invocation
collected 2,355 tests and reached 94% before reproducing the registered
xdist-controller stall; it was interrupted after six failure markers without
a summary and is **incomplete, not PASS**. Re-running the six registered IDs
in isolation reproduces the same six pre-existing reds: SI3 scalar-math source
provenance, round-51 private-arm scope, round-35 escape scoping, worktree
stamping, the round-129 certified-year fixture, and the `hires_lane_surface`
case-board row. No new round-118 failure was exposed.

The separate read-only Codex review returned **independent review unavailable
in-sandbox**: `failed to initialize in-process app-server client: Read-only
file system`.

No model or card file changed, so ORCA2 rung-0/rung-7, GYRE, DINO, tank, and
generic-card trajectories cannot move and are not represented as rerun gates.

## OPEN

1. Operator runs the committed launcher. Admission must prove restart and
   inherited-stream identity, product and recurrence replay, terminal
   accumulator identity, two-rank coverage, and every runtime plant.
2. Walk northwest, northeast, southwest, and southeast V in compiled order.
   Preserve every failed frozen prediction as `REFUTED`; do not extrapolate
   the U result.
3. If all eight U/V paths close, land the southern mask association and
   IEEE-zero recurrence semantics together under the full ORCA2, GYRE, DINO,
   tank, generic-card, citation, and push gates. Otherwise retain the U result
   and follow the earlier V owner.
4. Resume the northern-V cancelling pair and later 68-cell substep-2 U
   residual only after this coefficient path is complete. The package-exposed
   rung-0 card and independent 240-step month remain open hierarchy work.

## UNVERIFIED

- Runtime observational passivity and arithmetic sufficiency until the
  operator completes acquisition.
- The first non-bit statement and arithmetic sufficiency of all four V paths.
- The combined shared implementation under the full landing gates; no
  production implementation was attempted.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.
