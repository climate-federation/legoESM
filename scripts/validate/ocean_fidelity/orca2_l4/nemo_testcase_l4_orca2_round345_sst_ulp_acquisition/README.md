# ORCA2 round 345 restart control and one-ULP envelope arms

`run.sh` defaults to preflight. Operator-only `--run` creates two continuations
from admitted round-342 arm A's day-1 (`kt = 8`) double-precision restart pair.
The control changes no bit. The envelope arm advances only finite wet surface
`tn` values by one float64 ULP toward positive infinity. Both restart the same
binary at step 9 and continue through step 2920.

Admission requires the control to be byte-identical to arm A at every daily
restart before the perturbed arm can set an envelope.

The cold-start, forcing, and damping inputs are unchanged and re-hashed against
arm A. `restart_sst_ulp_perturbation.json` records both rank-file digests,
variable, dtype, direction, and selected/changed counts; admission re-derives
the pattern and proves every other restart value and all metadata are identical.

Failed arms created by this invocation move to `refused/` with builds retained.
`--admit-existing` verifies producer-commit ancestry plus manifest and binary
digests, logs the attempt in `admit_existing.log`, keeps failures in
`refused/`, and promotes a passed pair to its canonical run-root paths:

```bash
run.sh --admit-existing /absolute/path/to/refused/<retained-arm>
```
