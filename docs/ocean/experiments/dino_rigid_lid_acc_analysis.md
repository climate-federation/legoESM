# DINO rigid-lid ACC: barotropic-instability analysis

**Question (user):** the DINO ACC transport is ~50 Sv (vs the Kamm et al. 2025 R1
equilibrium ~206 Sv) — "too low". Is the barotropic solver the problem, and does
the rigid-lid (streamfunction) solver fix it?

**Short answer:** the rigid lid *diagnoses* the problem (it shows the
free-surface 44 Sv plateau is artificially low — its η-solve dissipation caps the
ACC) but does **not** fix it: the rigid-lid ACC is **barotropically unstable** at
1° and runs away. The runaway is **not** a bottom-form-stress bug; it is the
intrinsic non-eddy-resolving ACC instability. A stable physical ACC at 1° needs
an eddy closure that matches NEMO's 206 Sv, not a barotropic-solver swap.

## What the rigid lid showed (correct part)

With the full Veros-faithful stack (`barotropic_solver="rigid_lid"` auto-applies
`outer_integrator="ab2"` + `coriolis_scheme="explicit_ab2"` + `ab2_scope="advective"`
+ `dt_mom_ratio=9`), DINO rigid_lid spins up cleanly for ~250 days: ACC 0→52 Sv,
**climbing past the free-surface solvers' ~44 Sv plateau**. So the free-surface
44 Sv is artificially low (the η-solve dissipation caps the depth-integrated
transport — a null mode of the free-surface solve). The user's instinct that
"50 Sv is too low" is correct.

## The runaway (the open problem)

Beyond ~day 270 the ACC runs away **super-exponentially** (growth rate itself
accelerating): 52→106→168→343→1166→NaN by day 360.

### Ruled out (≈12 GPU experiments)

| hypothesis | test | result |
| --- | --- | --- |
| cache / setup | bare flip vs faithful stack | bare flip NaN by day 60; advective drag-fold delays to day 270 |
| cold-start | hot-start from a 120 d implicit_cn-spun state | still runs away (41→456→NaN) |
| Coriolis time-stepping | ab2 + explicit_ab2 without advective | runs away faster |
| bottom-drag routing | `ab2_scope="advective"` (folds drag into F_slow) | delays 4× but does not fix |
| GM / baroclinic-eddy saturation | Visbeck κ_GM ×10 (2000→20000) + α ×4 | **no effect** — near-identical runaway |
| grid-scale (2Δx) mode | biharmonic `B_h_barotropic` 3e11–1e13 | **no effect** — biharmonic is weak at large scale |

### What controls it

Only the **Laplacian lateral viscosity A_h** (strong at large scale) damps the
runaway — and only by **over-damping the ACC**:

| A_h [m²/s] | day-400 ACC | state |
| --- | --- | --- |
| 15 012 (DINO default ×1) | NaN | runaway |
| 21 017 (×1.4) | 308 (climbing) | runaway |
| 24 019 (×1.6) | 91 (creeping) | marginal |
| 27 021 (×1.8) | 50 | ~stable |
| 30 024 (×2.0) | 27 | stable, over-damped |

There is **no A_h that is both stable and high**: the stable branch caps the ACC
at ≤ ~50 Sv (≈ the free-surface value). Veros's rigid-lid ACC is stable because
it uses A_h ~ 2.2e5 m²/s (dx³ scaling at 2°) — ~15× DINO's Munk-scaled,
paper-matched A_h — i.e. it stabilises by heavy viscosity at coarse resolution.

## Diagnosis

The instability is a **large-scale barotropic shear instability of the ACC jet**
that the 1° (non-eddy-resolving) model cannot saturate:
- it is large-scale (biharmonic-insensitive, Laplacian-dampable),
- it is barotropic (GM/baroclinic-insensitive),
- the form stress is present (the depth-averaged PGF, hence bottom-pressure
  torque, enters the streamfunction RHS via `curl(F_slow)` —
  `rigid_lid_latlon_cgrid.rigid_lid_step`); it is *not* missing.

The **free surface masks** the instability through its η-solve numerical
dissipation, at the cost of capping the ACC at 44 Sv. The **rigid lid exposes**
it (no η restoring/dissipation), so without resolved eddies (or a barotropic
eddy closure) to saturate the jet the transport is either over-damped (low ACC)
or unstable (runaway).

## Recommendation / next direction

- **implicit_cn** stays the practical stable DINO solver (bounded, but caps the
  ACC low).
- **rigid_lid** is correct + selectable for the **≤~250 d spin-up demonstration**
  (it reveals the true, higher ACC the free surface masks).
- A **stable physical ~200 Sv ACC at 1°** is an eddy-closure problem (match the
  paper's NEMO 206 Sv), not a barotropic-solver bug. Note GM is *not* the lever
  to raise the ACC (stronger GM flattens isopycnals → lowers the thermal-wind
  shear → lowers the ACC). The free-surface 44-vs-206 gap is a model-vs-NEMO
  discrepancy (barotropic-solver dissipation / forcing / density), i.e.
  OMIP-faithful territory.

Code: `ocean/experiments/dino.py` (rigid_lid auto-applies the faithful stack),
`ocean/dynamics/rigid_lid_latlon_cgrid.py` (solver),
`ocean/dynamics/ocean_model_latlon_cgrid.py` (`_step_impl` F_slow + `B_h_barotropic`;
`_ab2_step` calls `_step_impl` so the biharmonic is in the faithful path).
Config: `scripts/experiment/dino/rigid_lid_latlon_365d.yaml`.
