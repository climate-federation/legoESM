# DINO split-explicit / momentum-commit chain: round-5 result

Date: 2026-08-29. Session:
`01a04e34-d1fb-73e0-b25a-177641f0a246`. Clean measurement commit:
`9e108f164e8`.

## Verdict

Row 1.1 is **FULLY DISPOSED as a component-wise composition**.

- U remains the accepted round-4 `L+V+C` composite: lateral friction and
  pre-loop Coriolis are positive contributors, vorticity is a canceller, and
  every other term is bounded.
- V is a confirmed `L+V+C+P` composite.  The full oracle substitution removes
  `0.997227273491` of the faithful residual, with correlation
  `0.999998160346`, gain `1.002010662876`, and squared residual energy
  `Q=7.688012291872e-6`.  Every remaining named term and the explicit
  assembly remainder is below the frozen `0.10` bound.

The V Shapley allocation is:

| Term | Allocation | Disposition |
|---|---:|---|
| lateral friction | `+0.133701275958` | OWNED_POSITIVE_CONTRIBUTOR |
| total EEN vorticity | `+0.064497260189` | OWNED_POSITIVE_CONTRIBUTOR |
| pre-loop Coriolis removal | `-0.043244995596` | OWNED_CANCELLER |
| hydrostatic pressure gradient | `+0.845038771437` | OWNED_POSITIVE_CONTRIBUTOR |

The allocations sum to the full squared-energy benefit
`0.999992311988` exactly at the registered `1e-12` resolution.  Material V
interactions are vorticity × Coriolis `+0.082515069391`, vorticity × pressure
`-0.124451136797`, and Coriolis × pressure `+0.083128126635`; every other
interaction is bounded.  Field-level interaction contrasts remain below
`1.39e-17*RMS(r0)`, confirming these are cancellation terms in the energy
reducer rather than nonlinear forcing physics.

## Source-ordered ownership

| Term | Active NEMO source/time level | U | V |
|---|---|---|---|
| kinetic-energy gradient | `dynadv.F90:89-95`; `stpmlf.F90:309-314` | bounded | bounded |
| vertical advection | `dynadv.F90:97-103`; `dynzad.F90:81-119`; `stpmlf.F90:309-314` | bounded | bounded |
| total EEN vorticity | `dynvor.F90:143-194`; `stpmlf.F90:315-318`, Kmm/NOW | canceller | positive contributor |
| lateral friction | `dynldf.F90:73-115`; `stpmlf.F90:319-322`, Kbb/BEFORE | positive contributor | positive contributor |
| hydrostatic pressure gradient | `dynhpg.F90:348-413`; `stpmlf.F90:324-328`, Kmm/NOW | bounded | positive contributor |
| pre-loop 2-D Coriolis removal | `dynspg_ts.F90:358-370`, Kmm/NOW | positive contributor | canceller |
| baroclinic-residual drag | `dynspg_ts.F90:372-400` | bounded | bounded |
| centred wind | `dynspg_ts.F90:423-459` | bounded after #1695 | bounded |
| faithful assembly remainder | signed ledger remainder | bounded | bounded (`9.97e-12`) |

This is ownership, not a multi-term production-fix authorization.  The
individual substitutions are not independently safe because their material
interactions change sign and magnitude.

## Continuation

The exact all-eight-term endpoint is the registered held row-1.1 control for
the downstream chain.  It may replace `F_slow_u/F_slow_v` with the existing
NEMO `spg_dump_zu_frc.bin`/`spg_dump_zv_frc.bin` fields in an offline
production replay.  Rows 1.2--6 remain blocked only until that replay is
separately preregistered and run in execution order.

All eight `P=0` corners and the pressure singleton reproduced their committed
receipts within `1e-12`; all Möbius/Shapley identities and planted controls
passed.  The run was clean, CPU/fp64, existing-dump-only, strict JSON, and
used no NEMO execution, writer, SLOT, GPU, or MPI.

Machine receipt:
`docs/ocean/fidelity/dino_split_explicit_momentum_chain_round5_artifact.json`.
