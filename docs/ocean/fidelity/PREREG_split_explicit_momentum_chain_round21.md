# Preregistration amendment: row-1.3 surface PGF, round 21

Date: 2026-08-29. Frozen after the committed round-20 recurrence and before
the PGF production replay. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 20 puts row 1.2 at exact zero for all 9,920/9,758/9,868 SSH/U/V points
and advances the ordered stop to row 1.3. The same artifact reports exact
midstep U/V and exact substep-1 SSH under held slow forcing, while final U/V
remain DEBT (`4.182607878554959e-7`, `1.0511418030286768e-6`).

The next executed operand is the back-interpolated SSH and surface pressure
gradient at `dynspg_ts.F90:766-780`. The already bound QCO writer makes
`zsshp2_e` bit-exact. The registered PGF comparison applies NEMO's literal
`-zldg*(ssh_neighbor-ssh)*r1_e1u/e2v` expression to that exact operand and
compares it with the production gradient. Both components use the POINTWISE
`1e-15` normalized-RMS and maximum-error/NEMO-RMS bars.

If the U and V PGF are exact after a correction, row 1.3 advances to the pure
in-loop EEN Coriolis dump at `dynspg_ts.F90:783-805`. If either fails, the
first failing component remains the stop. The correction is a new
`barotropic_pgf_evaluation` selector: `generic` byte-pins every non-fidelity
card; `nemo_literal` is default only on the two DINO fidelity cards and uses
the carried `dx_u/dy_v` face metrics with NEMO multiplication order. A planted
nonuniform `dy_v` test must separate it from the generic reconstructed metric.
