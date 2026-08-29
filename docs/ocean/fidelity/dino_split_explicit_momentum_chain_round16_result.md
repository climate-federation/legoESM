# Split-explicit / momentum-commit chain: round 16

Date: 2026-08-29. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. Measured package commit
`3e76b2166e0`.

## Association rows 9.4--9.6

The existing day-180 dumps close all three multiplication-association rows:

| Row | NEMO expression | Result |
|---:|---|---|
| 9.4 | `(e2u * ua_e) * zhup2_e` | unique bit-exact U transport |
| 9.5 | `(e1v * va_e) * zhvp2_e` | unique bit-exact V transport |
| 9.6 | `(dU + dV) * r1_e1e2t` | zero mismatches on 9,920 wet T points |

For row 9.6, divide-after-sum differs at 2,655/9,920 points. These are the
executed source expressions at `dynspg_ts.F90:698-704,722-724`. The production
path now preserves these associations under
`barotropic_continuity_evaluation="nemo_literal"`.

Adversarial review caught and closed a selector-routing defect before any
climate arm: the first implementation let the selector alter face-depth/drag
physics while both values still took literal divergence. Production now keys
H_u/H_v only on `barotropic_face_depth` and keys generic versus literal
divergence only on `barotropic_continuity_evaluation`. A one-substep test with
nonzero drag and `g=0` requires both arms to return bit-identical transports
and velocities while SSH is bit-distinct; the combined continuity/drag/
partial-cell set passes 40/40 on CPU/fp64.

At the measured production commit, the hardened production acceptance reports
rows 9.1--9.7 all `AT BAR` with normalized error and maximum error both exactly
zero. It binds checkout-local production modules, the exact mesh/restart and
dump hashes, CPU/fp64, session, and controls. The production acceptance calls
the face-depth, metric-transport, and divergence helpers; its SSH value is the
registered formula applied to those production operands. The independent full
recurrence closes the first-substep SSH output.

## Ordered chain state

The hardened recurrence remains stopped at row 1.2 under the frozen
pointwise `1e-15` bar:

| Row | Operand | Status | E | max error / NEMO RMS |
|---:|---|---|---:|---:|
| 1.2 | `sshn_e` | AT BAR | 0 | 0 |
| 1.2 | `un_e` | NEAR-CLASS | 2.29625968916112e-16 | 5.57431633372995e-15 |
| 1.2 | `vn_e` | NEAR-CLASS | 2.223767115037694e-16 | 4.26422750960742e-15 |

The 3-D BEFORE velocities, wet-level counts, and total face thicknesses are
exact. That makes NEMO's left-accumulated per-level thickness x velocity
reduction the prime candidate (`dynatf_qco.F90:254-267`, copied into the
split-explicit seed at `dynspg_ts.F90:571-579`), but it is **not owned or
bounded yet**: total thickness equality does not prove that every per-level
face thickness and multiplication operand is bit-identical. The next legal
peel is a bit comparison of the per-level U/V face weights followed by the
left-accumulation association.

Rows after the stop retain targeting evidence only:

| Row | Operand/output | Targeting E | Registry disposition |
|---:|---|---:|---|
| 1.3 | `ssh_substep1` with exact seed | 1.90418589978210e-20 | ordered-blocked |
| 1.3 | `ub_substep1` with exact seed | 4.18260787855496e-7 | ordered-blocked |
| 1.3 | `vb_substep1` with exact seed | 1.05114180302868e-6 | ordered-blocked |
| 1.4 | `puu_b_final` | 6.22167444653010e-6 | ordered-blocked |
| 1.4 | `pvv_b_final` | 1.10928061109024e-5 | ordered-blocked |
| 1.4 | `pssh_final` | 4.20792959287899e-6 | ordered-blocked |
| 1.4 | `un_adv_final` | 3.36866897283752e-6 | ordered-blocked |
| 1.4 | `vn_adv_final` | 9.17972945992041e-6 | ordered-blocked |
| 2--6 | all registered rows | not crossed | ordered-blocked |

If row 1.2 clears, row 1.3 resumes at the first velocity-update operands in
source order: back-interpolated SSH and pressure gradient
(`dynspg_ts.F90:766-780`), in-loop Coriolis (`:783-805`), explicit bottom
stress (`:818-825`), then final vector association (`:838-850`). Existing
substep-1 Coriolis dumps and exact SSH dumps cover the first comparisons;
bottom-stress dumps must be re-inventoried before instrumentation.

## Structural owner and climate release

NEMO's V-point latitude and zonal metric are explicit:
`zvj=j-nn_jeq_s+0.5`, `gphiv=asin(tanh(...zvj...))`, and
`e1v=R*cos(gphiv)*dlon` (`usrdef_hgr.F90:98,108,113`). The former legoESM
path averaged adjacent tracer latitudes before taking the cosine. The corrected
V-face metric plus literal continuity path is now production-selected and
passes the hardened day-180 operand acceptance exactly.

The climate intervention is therefore released independently of the ordered
registry stop. Its frozen question is:

> Does the NEMO-faithful V-face Mercator metric plus literal QCO continuity
> materially reduce the frozen day-360 southern-basin deficit (-0.951912 Sv)
> and the five-day wall flicker?

The committed preregistration uses a 2x2 metric x association design so the
headline contrast is not confounded and the main effects plus interaction are
reported. No GPU arm was run in this round.

## Bound artifacts

- acceptance: SHA-256
  `a8da092619b04760071ddba66bd7a3f7a522824e9caa314025cc5f3a21cffb08`
- recurrence: SHA-256
  `315e7f93362b2d9694bfea4ab6636e6763c3b4e1dfa362fa0d01ba1d02d31121`
- combined adjudication: SHA-256
  `21925ad16b32f6e7240243d6b08eb882bf835c772daccb7839382c558b6def2a`

The earlier round-13/14 artifacts are unbound diagnostics superseded by these
hardened receipts. No NEMO process, GPU, `mpirun`, push, or new instrumentation
was used.
