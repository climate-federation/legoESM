# ORCA2 round 345 one-ULP envelope arm

`run.sh` defaults to preflight. Operator-only `--run` copies admitted round-342
arm A's day-1 (`kt = 8`) double-precision restart pair, advances only finite wet
surface `tn` values by one float64 ULP toward positive infinity, then restarts
the same binary at step 9 and continues through step 2920. The perturbed
restart remains day 1 of the envelope.

The cold-start, forcing, and damping inputs are unchanged and re-hashed against
arm A. `restart_sst_ulp_perturbation.json` records both rank-file digests,
variable, dtype, direction, and selected/changed counts; admission re-derives
the pattern and proves every other restart value and all metadata are identical.

Failed arms created by this invocation move to `refused/` with their builds
retained. Existing arms never move. Re-admit a completed retained arm directly:

```bash
run.sh --admit-existing /absolute/path/to/refused/<retained-arm>
```
