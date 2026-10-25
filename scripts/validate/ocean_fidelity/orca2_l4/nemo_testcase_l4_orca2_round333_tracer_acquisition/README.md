# ORCA2 round-333 kt=2 tracer acquisition

This is an operator-only, additions-only extension of the reviewed round-328
acquisition driver and its round-332 source-order arm. The default invocation
is read-only:

```bash
scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round333_tracer_acquisition/run.sh --preflight-only
```

The wrapper imports the complete parent driver, audits exactly six pinned
substitutions (fresh configuration, evidence path, run name, source directory,
stage observer patch, and extended writer), and therefore retains every
inherited plant, syntax proof, passivity check, same-program proof, memory/disk
guard, and admission gate.

Only the operator may request the NEMO run:

```bash
scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round333_tracer_acquisition/run.sh --run
```

The added passive stream is
`oracle_rktracer_stage3_kt00000002.bin`. It uses the already reviewed RKTR3
schema and captures rank 0 at zeroed Krhs, after advection, after surface
forcing, after QSR, after LDF, and immediately after the implicit tracer solve.
The existing kt=2 stage-2 and stage-3 frames provide the Kmm input and
post-boundary-condition output cross-checks. No configuration value changes.
