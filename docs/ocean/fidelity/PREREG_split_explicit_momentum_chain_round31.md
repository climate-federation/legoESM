# Preregistration: source-ordered barotropic momentum commit, round 31

Date: 2026-08-30. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. Frozen before changing production or
replaying it. Round 30 is bound at SHA-256
`fe2e8023c176d455ba98253840d68eef76fb50ec2ad8c8d5fa03938fd1d61ccc`.
Its eight-arm offline factorial classifies row 3
`INHERITED_FROM_ROW_1_4`: P0D1M1 remains `NEAR-CLASS`, while substituting the
NEMO row-1.4 `pssh_final` makes P1D1M1 exactly zero for all four QCO outputs.
Depth association is inert and literal metric/post-factor association removes
only the independent face-operation roundoff.

## Candidate and source

The earliest nonzero production residue feeding the row-1.4 recurrence is the
row-1.3 first-substep velocity commit (U/V normalized RMS
`2.0785e-16`/`2.0103e-16`; SSH `1.9042e-20`). NEMO materializes bottom stress
as `zu_trd = zu_cor + ((zCdU_u*un_e)*hur_e)` at
`dynspg_ts.F90:700-705`, then commits
`un_e + rDt_e*((zu_spg+zu_trd)+zu_frc)` at `:719-732`. Round 26's held operand
receipt proved both literal identities bit-exact (`0/9758`, `0/9868`). Current
production instead presents Coriolis, drag, PGF, and forcing in a different
addition order to XLA.

Add `barotropic_momentum_update_evaluation = generic | nemo_literal`.
`generic` must retain the current expression byte-for-byte. `nemo_literal`
must materialize Coriolis-plus-drag, PGF-plus-trend, plus forcing, the `dt`
product, the carry addition, and mask application in the NEMO source order,
using optimization barriers so XLA cannot reassociate them. The two DINO
fidelity cards select `nemo_literal`; every other card remains explicitly
`generic`. Unknown selectors must fail. Tests must include a red operand-order
plant, JIT and gradient execution, exact two-card routing, and generic pins.

## Frozen gates

Replay the unchanged round-6 production scorer on CPU/fp64 and then the
unchanged round-29 row-3 scorer. Controls and populations do not change.

* **CONFIRMED:** row 1.3 SSH/U/V, all five row-1.4 fields, and all four row-3
  QCO fields classify `AT BAR`. Row 3 is released and the ordered walk resumes
  at row 4.
* **PARTIAL:** row 1.3 U/V improve by at least 90% but any row-3 field remains
  `NEAR-CLASS` or `DEBT`. Keep the implementation only if its literal helper
  separately matches the round-26 held outputs at the strict pointwise bar;
  stop and localize the next recurrence operand.
* **REFUTED:** either row-1.3 velocity worsens, improvement is below 90%, or a
  control/default/red test fails. Revert the candidate and stop.

Rows 4--6 and later chains remain blocked until the production row-3 gate is
released. No new NEMO writer, GPU, or MPI execution is authorized.
