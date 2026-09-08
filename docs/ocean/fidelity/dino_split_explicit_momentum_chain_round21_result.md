# Split-explicit momentum chain: literal seed and PGF, round 21

Date: 2026-08-29. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Row 1.2 is **FIXED and exact in production**. The first implementation replay
stopped because it reconstructed the NEMO reference ladder through the generic
min-face z-star path. The corrected implementation consumes the bridge-carried
`e3t_0`, `hu_0/hv_0`, and native T/U/V areas directly and preserves
the live BEFORE thickness at `dynspg_ts.F90:571-579`, the source-ordered
vertical reduction at `istate.F90:149-155`, and the face-thickness association
at `domqco.F90:166-169`. The authoritative round-20 and round-21 recurrences
both report zero error on SSH and all 9,758 U plus 9,868 V seed points.

The ordered walk then found and fixed the next operand. NEMO evaluates the
surface PGF at `dynspg_ts.F90:766-780` with native U/V face metrics. The former
regular/Mercator production operator rebuilt its V spacing by averaging T-cell
heights; the literal selector instead consumes carried `dx_u/dy_v` and keeps
NEMO multiplication order. The committed acceptance reports U and V both
bit-exact: `E=0`, maximum/NEMO-RMS `0`, mismatches `0/9758` and `0/9868`.
Its artifact is `/tmp/dino_split_explicit_momentum_chain_round21_pgf.json`,
SHA-256 `c2b890f8ec70847bfc69f565cb254e022d86621b726a1d37aef85059a62e8709`.

The full held-forcing recurrence improves immediately: substep-1 V error falls
from `1.0511418030e-6` to `1.8479e-7`; final U/V errors fall from
`6.2217e-6/1.1093e-5` to `2.7807e-6/4.6426e-6`. These are propagation checks,
not ownership bars. Row 1.3 remains open at the next executed operand, pure EEN
Coriolis (`dynspg_ts.F90:783-805`); bottom stress, final update, row 1.4, rows
2--6, and the remaining chains stay ordered-blocked.

| ordered row | post-fix receipt | disposition |
|---|---:|---|
| 1.2 SSH/U/V barotropic seed | `E=0`; `0/9758`, `0/9868` | FIXED, AT BAR |
| 1.3 QCO continuity composition | prior literal metric result retained; substep-1 SSH `E=1.904e-20` | AT BAR |
| 1.3 surface PGF U/V | `E=0`; zero bit mismatches | FIXED, AT BAR |
| 1.3 live EEN Coriolis | next executed operand | HELD round-22 coefficient peel |
| 1.3 bottom stress/final update | no legal ordered measurement yet | ORDERED BLOCKED |
| 1.4 full held-forcing U/V | `2.7807e-6` / `4.6426e-6` propagation only | ORDERED BLOCKED |
| 2--6 | no legal ordered measurement yet | ORDERED BLOCKED |
| free-surface filter, momentum RHS, tracer tail | no legal ordered measurement yet | ORDERED BLOCKED |

Round 22 preregisters the eight frozen `ffu/ffv` arrays at
`dynspg_ts.F90:1491-1667` and the four-term live application at
`dynspg_ts.F90:1670-1693`. Its held cascade is the first point at which a new
NEMO execution is required, so the walk stops there under the ordered-row rule.

Tests: 23/23 CPU/fp64. No NEMO process, GPU, `mpirun`, or push was used.
