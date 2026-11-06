# SI3 round 30 O17 carried-state free-run receipt (DECISION 210)

Evidence date: 2026-11-06.  Scored revision: `01009fda6`.  This receipt records
the completed DECISION 210 free run without adding a scientific interpretation.

## Evidence and integrity

Evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round30_o17_free_run`.
Its `SHA256SUMS` verifies both listed files with `sha256sum -c SHA256SUMS`:

- `free_run_report.json`: `00c5c2db6bc3202f2d3d2ab7fcea077432f1e4ad5fd833c7da98708568c21798`
- `launch_free_run.sh`: `62e0b7485ae2bb55cba70319009ba57ce1d835fdb161f9b9573b42f74c344004`

`status_lines.txt` records launch after the independent GIVEN-NEMO-ENTRY
rescore and completion with exit `0`.  `launch_free_run.sh` records CPU-only
execution (`CUDA_VISIBLE_DEVICES=""`, `JAX_PLATFORMS=cpu`,
`JAX_ENABLE_X64=1`), the O17 record under
`round29/oracle_o17_full_step`, `--execution eager`, `--expect-steps 100`,
`--free-run`, and `--n-steps 100`.  The launcher did not run NEMO.

## Carried-state implementation at `01009fda6`

The scored function is `free_run()` in
`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:758-877`
at `01009fda6`.  Step 1 reads the recorded entry
(`:809-811`).  For every later step, `prev_out = carried[rank]`,
`entry = carry_step_state(prev_out, records, card)`, and
`_carried_mismatches(entry, prev_out, full, h)` are executed at `:812-815`;
`carry_step_state` returns `prev_out` at `:489-496`.  The per-step report writes
`carry_chain_intact = len(carry_mismatches) == 0` at `:839-846`, and the current
output becomes the next carried state at `:848`.

## Recorded measurements

All keys below are from `free_run_report.json`.

| report key or counted report path | recorded value |
|---|---:|
| `mode` | `FREE_RUN` |
| `execution` | `eager` |
| `production_entry` | `legoesm.ice.si3_o17.si3_o17_step` |
| `steps` | integers `1..100` (`len(steps) = 100`) |
| `ranks` | `[0]` (`len(ranks) = 1`) |
| `ulp_bar` | `0` |
| `status` | `PASS` |
| `different` | `0` |
| `nonfinite` | `0` |
| `first_difference` | `null` |
| `per_step[*].carry_chain_intact` | `true` for `100 / 100` step-rank rows |
| `per_step[*].fields` | `198` fields in every step-rank row |
| counted field rows | `19,800` (`100` `per_step` rows × `198` `fields` keys) |
| counted scalar values | `4,906,947,200` |
| `scorer_wall_seconds` | `56973.55184951797` |
| `scorer_peak_rss_kib` | `10,342,400` |

The scalar count uses the `198` keys in `per_step["1:0"].fields`, resolves each
key to its recorded field shape in the same O17 evidence, and sums
`49,069,472` scalar values for one step.  The report contains `100` `per_step`
rows with the same `198` keys, so the recorded comparison count is
`49,069,472 × 100 = 4,906,947,200` scalar values.

## Scope not covered

This receipt does not cover more than the single recorded rank (`ranks = [0]`),
the stand-alone SAS ice-only deck, or JIT execution (`execution = eager`).
