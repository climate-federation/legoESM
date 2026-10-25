# ORCA2 round-333 kt=2 tracer and `traldf_iso_lap` acquisition

This is an operator-only, additions-only extension of the reviewed round-328
acquisition driver and its round-332 source-order arm. The default invocation
is read-only:

```bash
scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round333_tracer_acquisition/run.sh --preflight-only
```

The wrapper imports the complete parent driver and audits its pinned
substitutions (fresh configuration, evidence path, run name, source directory,
stage observer patch, and extended writers). It retains every inherited plant,
syntax proof, same-program proof, memory/disk guard, and admission gate. Its
arithmetic baselines are independently preprocessed from the unpatched NEMO
source. The separate `traldf_iso.o` census uses the record build's production
`-O3 -funroll-all-loops -fno-tree-vectorize` flags and must be identical with
and without the internal observer; a changed-arithmetic plant proves that
comparison is live. NEMO's build flags are never changed to make this pass.

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

The additional per-rank streams are
`oracle_r333_traldf_kt00000002_r0000.bin` and `_r0001.bin`. At the stage-3
`tra_ldf` boundary, the arm is active only for `kt=2`; `traldf_iso_lap` then
records both tracer passes (T and S) in NEMO execution order. The 71 fields
include the full declared extents of the input time level, before/after RHS and
exact per-pass increment, horizontal/vertical gradients, tensor coefficients,
individual flux components and totals, reciprocal/divisor operands, source
geometry and masks, pre-add divergence and divisor, and coverage flags. They
also include the executed `traldf_iso_a33` inputs (`e1u`, `e2v`, `rDt`, and
`r1_Dt`), its pre-limit coefficient and limiter intermediates. The routine has
no `kpass` loop in the
record build; its `jn=1,itra` loop is the two-pass T/S iteration. Records retain
IEEE signed zeros and carry explicit rank, origin, extent, field-order, and XOR
checksum metadata.

All capture calls are outside the NEMO loop nests; exact initialized gradient
ranges are copied by array section. The internal writer is compiled and run on
a two-rank synthetic fixture under `-fcheck=bounds`. A second production-flag
fixture executes the observer-bearing loop bodies and compares them bitwise
with an independent unpatched implementation. Preflight includes
source-placement plants for both source and preprocessed `ppsrc/nemo`, the outer
boundary and each internal region, plus header, field-order, truncation,
checksum, derived time-level, tracer-pass, extent, finite-payload, signed-zero,
wet-coverage, exact-increment, RHS replay, and A33 replay plants. The replay
gate reconstructs NEMO's recorded RHS and A33 statements bitwise before a
record can be admitted. No preflight path builds or executes NEMO.

Retained output can be re-admitted without rebuilding or running NEMO by
setting `R328_EXISTING_RUN` to its path under this round's `refused/` directory
and invoking `--admit-existing`. Admission trusts the run manifest's producer
commit and `nemo.sha256`, never the current HEAD; admission logs remain beside
the retained evidence.
