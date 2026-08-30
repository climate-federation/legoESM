# Split-explicit momentum chain: rounds 28--32 result

Date: 2026-08-30. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

## Production result through row 1.4

Round 28 built the complete literal EEN coefficient path at commit
`27a3682209688bd872f448965ecdb5b65f10b1a1`. The bridge carries the raw NEMO
F-point Coriolis, U/V/F thicknesses, masks, reference depths, and all eight
horizontal metrics. `_nemo_literal_een_coefficients` at
`barotropic_latlon_cgrid.py:692-820` materializes the four U and four V corner
coefficients in the executed NEMO association: triads
`dynspg_ts.F90:1517-1528,1544-1555`, surface-to-bottom reduction
`:1530-1534,1557-1561`, and post factors `:1535-1538,1562-1565`.
`een_barotropic_coriolis` consumes that object at
`barotropic_latlon_cgrid.py:905-980`; both the pre-loop and live substep paths
share it. The selector is faithful by default only on `nemo_dino_kamm` and
`nemo_dino_kamm_mlf` (`dino.py:1470`); all other cards retain the generic byte
pin (`state.py:1132`). Missing operands and unknown selectors fail closed.

Artifact
`/tmp/dino_split_explicit_momentum_chain_round28_literal_builder.json`,
SHA-256
`0e8757a2c10cd99cbb8394f639e15beb84c05b244f4eeb349718785763e9106e`,
classifies `PRODUCTION_LITERAL_EEN_COEFFICIENTS_AT_BAR`. All eight coefficient
RMS errors are zero or `2.82e-18`; applied U is exact and applied V is
`1.29e-19` RMS with `1.28e-17` maximum/NEMO-RMS.

The unchanged production replay,
`/tmp/dino_split_explicit_momentum_chain_round28_production_replay.json`,
SHA-256
`1b4e83ec06416711cfc2b866c878cccb2a23988e8953c9b440a9e94c22bc2be9`,
puts row 1.3 SSH/U/V at `1.90e-20`/`2.08e-16`/`2.01e-16` and all five row-1.4
outputs between `3.26e-16` and `4.78e-16`: every registered field is AT BAR.
The target reduction from the old roughly `2e-7` row-1.3 production error is
therefore confirmed.

Focused bridge, config, JIT/gradient, literal-builder, AL81
energy/enstrophy, and conservation tests pass (157 passed, one expected
failure in the broad physics set; the later focused routing set adds 37
passes). The sole strict-suite failure is pre-existing and unrelated: the
stale `grid_type` allowlist in `test_validate_strict_coverage.py`.

## Ordered rows 2 and 3

Round 29 artifact
`/tmp/dino_split_explicit_momentum_chain_round29_rows2_3.json`, SHA-256
`3cd09d6bf3f7f5bb9ee51705a1e5e8cbf3671ff19cec6ebf407efd9e883d60d5`,
scores the second `div_hor` call at `stpmlf.F90:349-376` /
`divhor.F90:172-181` as its registered `CEILING`: production normalized RMS
`8.06495e-5`, zero shift uniquely best. The source-ordered offline divergence
is `3.44234e-5`; this is a transport/metric association ceiling, not a new
physical-term owner, and it legally releases row 3.

The second `dom_qco_r3c` call (`stpmlf.F90:378-394`,
`domqco.F90:153-185`) is the next formal stop. T/U/V/F normalized RMS values
are `3.3185e-16`, `3.0750e-16`, `2.9791e-16`, and `2.6900e-16`, but their
strict maximum/NEMO-RMS values are `2.07e-15`--`2.50e-15`; all four classify
NEAR-CLASS under the unchanged pointwise bar.

Round 30's preregistered 2^3 factorial binds production versus NEMO
`pssh_final`, vector versus source-ordered reference-depth reductions
(`domain.F90:139-160`), and live division versus materialized NEMO metric/post
factors (`domhgr.F90:144-160`, `domqco.F90:165-181`). Artifact
`/tmp/dino_split_explicit_momentum_chain_round30_qco_factorial.json`, SHA-256
`fe2e8023c176d455ba98253840d68eef76fb50ec2ad8c8d5fa03938fd1d61ccc`,
classifies `INHERITED_FROM_ROW_1_4`:

* depth association is inert;
* literal metric/post-factor association removes only an independent
  face-operation roundoff (U/V/F remain AT BAR under oracle SSH either way);
* substituting the unchanged NEMO `pssh_final` makes T exactly zero, and the
  fully literal P1D1M1 arm makes all four outputs exactly zero.

Thus QCO depths and local composition are exonerated. The strict row-3 residue
is deterministic amplification of row-1.4 `pssh_final`, which is itself AT
BAR under its registered accumulation gate (`3.26280e-16` normalized RMS,
`1.94207e-15` maximum/NEMO-RMS). Bars are not relaxed: row 3 remains the
ordered stop.

## Refuted update candidate and held next measurement

Round 31 tested the already-held round-26 identity for the explicit momentum
commit (`dynspg_ts.F90:700-705,719-732`) in production. The literal selector
changed none of the row-1.3 or row-1.4 metrics, failing the preregistered 90%
improvement bar. The candidate and tests were removed at
`4baa531ef641ce4a4cb965c53678c2626f828ec2`; no inert physics knob survives.
The binding replay is
`/tmp/dino_split_explicit_momentum_chain_round31_production_replay.json`,
SHA-256
`7ce94a68528e0e1b0fe950eeaa1ca194bcd9bbdfa0c8545459bbd7141888643d`.

The first missing discriminant is now the 68-substep SSH/U/V trajectory: does
the at-bar first-substep velocity residue enter SSH during the recurrence, or
only during final boxcar accumulation? Round 32 preregisters and supplies the
three-stream full-halo SLOT instrument, a 223/223+3 exact bracket, a
checkout-local CPU/fp64 scorer, and planted controls in
`dino_split_explicit_momentum_chain_round32_handoff.md`. It is DESIGNED, NOT
RUN. No NEMO, GPU, or MPI process was launched here.

Current ordered disposition: rows 1.1--1.4 are owned and at their registered
bars; row 2 is at its registered ceiling; row 3 is locally exonerated but
formally blocked by inherited row-1.4 last bits. Rows 4 (`dyn_zdf`), 5 (`wzv`),
6 (`mlf_baro_corr`), then the free-surface filter, momentum RHS, tracer tail,
and remaining registry chains are ORDERED-BLOCKED pending the round-32 held
trajectory.

Round-32 execution addendum: the host completed the build, both sequential
arms, and the exact 223/223+3 bracket. The first scorer attempt is invalid and
makes no science claim: it intercepted `scan` while the faithful card executes
`fori_loop`, and it treated the pre-update trace as post-update. The corrected
scorer tees the 68-row `fori_loop`, compares carry-in to NEMO substep-start,
and binds trace rows 1/2 to the existing entry/post-substep-1 dumps. Only the
scorer block in `dino_split_explicit_momentum_chain_round32_resumed_handoff.md`
must rerun; the NEMO artifacts remain admitted.

## Official round-32 closure and ordered promotion

The official corrected scorer artifact is
`/tmp/dino_split_explicit_momentum_chain_round32_substep_trace.json`, SHA-256
`0b284d7c8880646850daee86b2b85fb13fb65540402e2e9a02760b0ed60ba86f`.
It binds commit `588cbae24e8605008cf3ccac5dff55fb9c5460a9`, the admitted NEMO source,
binary, patch, bracket, and three trace streams, and classifies
`NO_STRICT_SSH_FAILURE`. There is no first strict SSH, U, or V failure in any
of the 68 split-explicit substeps. The final U/V point errors are
`4.83e-16`/`4.44e-16`, below the registered `1.4e-14` linear-accumulation
bound by a scalar factor of about 30: the remaining last-bit errors cancel
rather than accumulate. The actual substep-1, full 68-row capture, entry and
post-substep-1 alignment, `fori_loop` restoration, identity, and five-
`nextafter` planted controls all fire. The retained four-`nextafter` plant is
documented as structurally insufficient and is superseded by the firing
five-`nextafter` plant.

This receipt certifies the split-explicit recurrence end to end. Rows
1.1--1.4 and row 2 are promoted from the ordered gate. Row 3 is also promoted
as `INHERITED-BOUNDED`: its local QCO depth and composition operands remain
exonerated by the round-30 factorial, and the certified recurrence proves that
the inherited at-bar SSH last bits never create a strict downstream failure.
This is an ordered-gate release, not a relabeling of the row-3 NEAR-CLASS
pointwise statistic or a relaxation of its bar. The next executable registry
row is row 4 (`dyn_zdf`), followed by row 5 (`wzv`), row 6
(`mlf_baro_corr`), the free-surface filter, momentum RHS, and tracer tail.
