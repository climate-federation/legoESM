# ORCA2 round 116 — remaining U EEN recurrence acquisition

Date: 2026-10-03. Base `f0975a020b1b8b010c9a8aab7d4e91c27dd39f5c`.
All eventual ocean numbers from this rung-0 record are **independent** because
rung 0 starts from NEMO's own from-rest state.

## Verdict

**STOPPED_FOR_RECORD.** Round 115's first non-bit statement remains the
carried EEN vertical recurrence addition: northwest U differs from NEMO only
in exact-zero sign bits, and the source-faithful IEEE addition arm closes its
1,618 accumulator-before and 5,197 accumulator-after differences. The three
later U recurrences are not present in the admitted stream, so this round does
not extrapolate the northwest result or land a partial production change.

A committed additions-only acquisition now records northeast, southwest, and
southeast U on every executed level and both ranks. Its exact-source syntax
proof and five record-independent controls pass. The record directory is
absent, so R116-P1 through R116-P4 remain unmeasured and no ocean number is
quoted from the instrument.

The operator action is:

```text
/tmp/autopilot-orca2-d5JviC/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round116_een_u_recurrence_acquisition/run.sh --run
```

It creates target `ORCA2_OMIP_L4_R116EENUREC` and run directory
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round116/acquisition/orca2_rung0_een_u_recurrence_ranked_10step_np2`.

## Compiled source boundary

The compiled rung-0 program evaluates northwest U, then northeast, southwest,
and southeast U in one vertical loop. Each recurrence is one carried addition
of a four-factor product: live U thickness, the corresponding neighboring V
thickness, that neighboring V mask, and the source-ordered `zpvo` coefficient
(`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1263-1273`).

The round-107 admitted stream surrounds only northwest U. The round-116 patch
adds before/after calls around the three untouched source statements and one
dump after the loop. It deletes or rewrites zero NEMO source lines. The writer
stores, for each missing recurrence, the four inputs, separately materialized
product, accumulator before, and accumulator after, plus `mbku`. The checker
requires both `term = e3u * e3v * mask * zpvo` and
`after = before + term` to replay bitwise; therefore a separately evaluated
term that does not describe the executed binary recurrence refuses admission.

## Admission contract and controls

The stream is self-describing: the checker reads every field name, rank,
dimension, and payload length from its header. It requires exactly-once
coverage of the `148 x 180` global domain, rejects the dummy level and all
writes outside `mbku`, and compares each recurrence's last executed value to
the admitted round-105 terminal accumulator with the native `(i,j)` to global
`(j,i)` association explicit.

Observational passivity requires all twenty kt=1..10 restart shards and all
six inherited round-105/107/110 streams to be byte-identical to the admitted
round-110 run. The launcher pins the source `dynspg_ts`, CPP keys, binary,
namelist, deck manifest, input manifest, patch, writer, checker, launcher, and
preregistration by SHA-256 content. Its target, run directory, and absolute
per-rank output path are new.

The exact Fortran writer and patched source compile under the pinned rung-0
includes. The layout, absolute-path, four-recorder environment,
producer-content, and rank-log controls each exit nonzero with
`STATUS PLANT-FIRED`. Runtime header, field-name, dimensions, truncation,
missing-field, duplicate-rank, bottom-index, product, recurrence,
inherited-terminal, association, and restart-byte controls are committed but
cannot execute until the record exists.

Preflight and control logs are under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round116/`.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R116-P1: recorder is observationally passive | **UNMEASURED**, operator record absent |
| R116-P2: NE/SW/SE products replay from recorded operands | **UNMEASURED**, operator record absent |
| R116-P3: NE/SW/SE recurrences replay under host IEEE addition | **UNMEASURED**, operator record absent |
| R116-P4: northeast U is the next source walk and its zero signs close under the round-115 arm | **UNMEASURED**, no recurrence value is inferred from preflight |
| R116-P5: no partial production landing before the streams admit | **CONFIRMED**, no `packages/` file changes |

## Validation and review

The focused parser/schema test passes 5/5. The separate read-only Codex review
was attempted and returned **independent review unavailable in-sandbox**
(`failed to initialize in-process app-server client: Read-only file system`).
No model file changes, so the rung-0/rung-7 ORCA2, GYRE, DINO, tank, and
generic-card trajectories do not move and are not relabelled as rerun results.

The focused round-107/109/110/114/115/116 chain passes 20/20. The single
required `tests/ocean/fidelity -n 12` invocation collected 2,347 tests and
reached 95% before reproducing the registered xdist-controller stall. It was
terminated without a summary after seven failure markers, so it is
**incomplete, not PASS**. The new round-116 tests pass both alone and in the
focused chain; the stalled quiet-mode run did not expose test IDs, so the
seventh marker beyond round 115's six registered reds remains unlocalized and
is not represented as a new round-116 failure.

## OPEN

1. Operator runs the committed launcher. Admission must prove restart and
   inherited-stream identity, product and recurrence replay, terminal
   accumulator identity, two-rank coverage, and every runtime plant.
2. Walk northeast, southwest, and southeast U in compiled order. Preserve any
   failed frozen prediction as `REFUTED`; do not assume northwest behavior.
3. Land the shared arithmetic only if all four U and four V coefficient paths
   are bit-exact and the full ORCA2/GYRE/DINO/tank/generic-card gates pass.
4. Then resume the northern V cancelling pair and later 68-cell substep-2 U
   residual. The package-exposed rung-0 card and independent 240-step month
   remain open hierarchy deliverables.

## UNVERIFIED

- Runtime observational passivity and arithmetic sufficiency of the new
  record until the operator completes acquisition.
- Whether northeast, southwest, and southeast U share northwest U's
  signed-zero recurrence boundary.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.
