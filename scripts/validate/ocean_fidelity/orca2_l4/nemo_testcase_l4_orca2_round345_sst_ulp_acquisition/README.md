# ORCA2 round 345 restart control and one-ULP envelope arms

`run.sh` defaults to preflight. Operator-only `--run` creates two continuations
from admitted round-342 arm A's day-1 (`kt = 8`) double-precision restart pair.
The control changes no bit. The envelope arm advances only finite wet surface
`tn` values by one float64 ULP toward positive infinity. Both restart the same
binary at step 9 and continue through step 2920.

Admission compares every common variable's dtype, shape, and raw data bytes in
both ranks at every continuation output (`kt = 16..2920`). It requires exactly
`DELAY_re8_fwb2` and `DELAY_im8_fwb2` to exist only in arm A, and refuses every
other missing/extra variable or unequal common-variable byte.

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

`run.sh --coupled-c4 --run` applies the same acquisition to the admitted
coupled C-ladder `REF_C4` deck.  It stages REF_C4's exact 59 deck/input files,
then changes only the ocean and SI3 restart names and run start to continue the
ocean+ice restart quartet at `kt = 8`.  The control must reproduce all common
ocean and SI3 continuation-variable bytes at every daily output through
`kt = 2920`.  For both ocean and SI3, only `DELAY_im8_fwb2` and
`DELAY_re8_fwb2` may be absent.  NEMO writes the SI3 delayed state at
`src/ICE/icerst.F90:140`; continuation-read buffers are skipped by
`src/OCE/IOM/iom.F90:1953-1981`, so those two fields are not re-emitted.  The
perturbed arm changes the same finite wet surface `tn` values by one float64
ULP and leaves both SI3 restart shards byte-identical.

Coupled admission has the same quarantine and promotion contract as the
standalone profile.  After both NEMO arms finish, admit the canonical pair with:

```bash
run.sh --coupled-c4 --admit-existing /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round31/coupled_c4_perturbed_twin
```

If the original admission already moved the pair into `refused/`, pass either
retained coupled arm path instead; the gate validates both retained arms before
promoting them together and records the attempt in `admit_existing.log`.

The coupled admission writes `round345_coupled_c4_sst_ulp_metrics.json`.  Its
12 complete months contain all Decision-188 ocean metrics plus
`ICE_AREA_NH`, `ICE_AREA_SH`, `ICE_VOLUME_NH`, and `ICE_VOLUME_SH`.  The
companion admission binds the report to REF_C4's binary, re-hashed namelists
and inputs, unchanged day-1 SI3 state, and the exact control continuation.

To regenerate only that report after admission, without running NEMO, use:

```bash
python scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round345_sst_ulp_gate.py \
  --coupled-c4 \
  --build-coupled-c4-envelope-report /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round31/coupled_c4_perturbed_twin
```

This also refreshes the report SHA-256 stored in the existing passed admission.
