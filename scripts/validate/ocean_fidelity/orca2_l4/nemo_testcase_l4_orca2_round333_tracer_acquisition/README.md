# ORCA2 round-333 kt=2 tracer acquisition

This is an operator-only, additions-only extension of the reviewed round-328
acquisition driver and its round-332 source-order arm. The default invocation
is read-only:

```bash
scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round333_tracer_acquisition/run.sh --preflight-only
```

The wrapper imports the complete parent driver and audits six pinned
substitutions (fresh configuration, evidence path, run name, source directory,
stage observer patch, and extended writer). It retains every inherited plant,
syntax proof, same-program proof, memory/disk guard, and admission gate. Its
arithmetic baseline additionally removes every r332/r333 observer call before
the normalized floating-point instruction census.

Only the operator may request the NEMO run:

```bash
scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round333_tracer_acquisition/run.sh --run
```

The added passive streams are
`oracle_rktracer_stage3_kt00000002_r0000.bin` and `_r0001.bin`. They record raw
declared `jpi,jpj,jpk` arrays (no canonicalizer), with a field checksum before
every payload. Both ranks are captured at zeroed Krhs, after advection, surface
forcing, QSR, LDF, and the enabled BBC/BBL/damping group, then immediately
after the implicit solve. The finish call is deliberately after the inherited
kt=1 tracer-dump block. The existing kt=2 stage-2 and stage-3 frames provide
the rank-0 Kmm input and post-boundary-condition cross-checks. No configuration
value changes.

Retained output can be re-admitted without rebuilding or running NEMO by
setting `R328_EXISTING_RUN` to its path under this round's `refused/` directory
and invoking `--admit-existing`. Admission trusts the run manifest's producer
commit and `nemo.sha256`, never the current HEAD; admission logs remain beside
the retained evidence.
