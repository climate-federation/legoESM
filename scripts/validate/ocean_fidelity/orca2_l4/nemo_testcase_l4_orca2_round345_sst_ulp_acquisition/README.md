# ORCA2 round 345 one-ULP envelope arm

`run.sh` defaults to read-only preflight. The operator-only `--run` mode
reuses the admitted round-342 observer-free K0 binary and runs one additional
2,920-step year. It stages the round-342 inputs, replaces only the staged
potential-temperature file by an audited regular copy, and advances every
finite wet surface float32 value in all 12 records one ULP toward positive
infinity.

The generated `initial_sst_ulp_perturbation.json` records the resolved source,
source/perturbed/domain SHA-256 digests, selected count, exact ULP distance,
and the fact that NEMO reads this file for both cold-start T/S and the configured
T/S damping target. No NEMO source or model option is changed.
