# Split-explicit / momentum-commit chain: rounds 11--14

> **Superseded admission notice.** The original round-13/14 receipts described
> below did not mechanically bind checkout-local production imports plus the
> exact mesh/restart. They are `UNBOUND_DIAGNOSTIC`, not citable verdicts. The
> hardened round-16 result reproduces the measurements and supersedes this
> document.

Date: 2026-08-29.  Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

## Result

The multiplication-association peel is closed.  On the existing day-180
operand dumps, NEMO's literal U transport `(e2u * ua_e) * zhup2_e` and V
transport `(e1v * va_e) * zhvp2_e` are bit-exact; the production alternatives
are bit-distinct and outside the frozen per-element bar.  For divergence,
`(dU + dV) * r1_e1e2t` is the unique bit-exact expression: it has zero
mismatches on 9,920 wet T points, whereas `(dU + dV) / area` differs at 2,655
points.  These are the executed NEMO expressions at
`cfgs/DINO/MY_SRC/dynspg_ts.F90:698-704,722-724`.

Production now uses those literal associations in the `nemo_ssh_avg` QCO
continuity path. Round 13 called the production face-depth, metric-transport,
and divergence helpers on the bound day-180 inputs, then applied the registered
SSH formula to those production operands. Every row
9.1--9.7 was bit-exact (`E=0`, maximum absolute difference `0`), including
both walls and the prognostic `ssha_e` update.  The relevant production tests
pass: 3/3 literal-continuity tests and 38/38 tests in the combined continuity,
drag, and partial-cell set, all CPU/fp64.

## Ordered continuation

Round 14 then reran the unchanged committed full-recurrence wrapper under the
round-6 bars, with the already-owned NEMO slow forcing held at the production
entry.  The authoritative `forcing_only` arm stops at row 1.2:

| Row | Operand/output | Gate result | Normalized RMS error | Maximum error / NEMO RMS |
|---:|---|---|---:|---:|
| 1.2 | `sshn_e` | AT BAR | 0 | 0 |
| 1.2 | `un_e` | NEAR-CLASS | 2.2962596892e-16 | 5.5743163337e-15 |
| 1.2 | `vn_e` | NEAR-CLASS | 2.2237671150e-16 | 4.2642275096e-15 |

The frozen POINTWISE per-element bar is `1e-15`; therefore the U/V seed
cannot be promoted as AT BAR.  The 3-D BEFORE velocity is bit-exact and prior
receipts show identical wet-level counts and total face thickness. The
remaining row-1.2 difference makes the vertical reduction/multiplication
association used to construct NEMO's
`puu_b/pvv_b(Kbb)`: NEMO initializes and left-accumulates the vertical
transport at `dynatf_qco.F90:254-267`, then copies it into `un_e/vn_e` at
`dynspg_ts.F90:571-579`; legoESM uses the fused vertical reduction in
`ocean_tendency_common.py:540-590` through
`barotropic_latlon_cgrid.py:232-235` the prime candidate. It is not owned or
bounded until the per-level face weights are compared bitwise; equality of the
total face thickness is insufficient. This is a roundoff-class operand stop,
not evidence of a physical seed defect, but the preregistered bar forbids
skipping it.

The exact-seed targeting arm supplies a useful bound without changing that
verdict:

| Row | Operand/output | Gate result | Normalized RMS error |
|---:|---|---|---:|
| 1.3 | `ssh_substep1` | AT BAR | 1.9041858998e-20 |
| 1.3 | `ub_substep1` | DEBT | 4.1826078786e-7 |
| 1.3 | `vb_substep1` | DEBT | 1.0511418030e-6 |

Thus the metric fix is the sole owner of the continuity residual, but it is
not the sole owner of the complete barotropic velocity recurrence.  With the
seed and slow forcing held, the first physical DEBT is the combined velocity
update at `dynspg_ts.F90:838-850`.  Its next ordered operand ladder is the
back-interpolated SSH and pressure gradient (`:766-780`), in-loop Coriolis
(`:783-805`), explicit bottom stress (`:818-825`), and final update association
(`:838-850`).  Existing `cor2d_dump_{ua,va,zu_trd,zv_trd}_substep1` files cover
the Coriolis arm; the pressure-gradient operand is reconstructible from the
now-exact SSH dumps.  A bottom-stress operand dump is not claimed present and
must be re-inventoried before any writer is designed.

Row 1.4 remains targeting-only (`E=6.2216744465e-6` U,
`1.1092806111e-5` V, `4.2079295929e-6` SSH, `3.3686689728e-6` U transport,
and `9.1797294599e-6` V transport).  Registry rows 2--6 remain
**ORDERED-BLOCKED** and were not promoted or rescored across the stop.

## Climate question and release state

The headline climate question is:

> Does the NEMO-faithful V-face Mercator metric plus literal QCO continuity
> association materially reduce the frozen southern-basin transport deficit
> (`Gbasin90`, approximately -0.95 Sv at day 360) and the independently frozen
> wall-flicker statistic, relative to the corrected-carry baseline?

This non-release was superseded after hardened production acceptance showed
the structural fix is in the twin path. Round 16 releases an independent 2x2
climate intervention without promoting any ordered registry row. No GPU,
NEMO MPI run, or climate arm was launched in rounds 11--14.

## Artifacts

- `dino_split_explicit_momentum_chain_round11_artifact.json`: association
  four-corner measurement.
- `dino_split_explicit_momentum_chain_round12_artifact.json`: exact divergence
  tie-break.
- `dino_split_explicit_momentum_chain_round13_artifact.json`: production-path
  continuity acceptance.
- `dino_split_explicit_momentum_chain_round14_artifact.json`: ordered full
  recurrence replay and stop.

All measurements used existing dumps, CPU/fp64, the day-180 lane, and the
session ID above.  No new NEMO instrumentation or SLOT block was required.
