# MITgcm barotropic-gyre oracle: the residual is a spatial energy-conservation defect

**Status (2026-06-17):** root cause *identified and quantified*; the fix is a scoped dycore
project (an energy-conserving C-grid momentum scheme), not a config or wiring change.

## Summary

legoESM reproduces MITgcm's `tutorial_barotropic_gyre` to **0.9997 eta pattern correlation at
10 steps**, but its 3-year **equilibrium is 6-9× too energetic** (`|u|max ≈ 0.18-0.31` m/s vs
MITgcm's `0.031`). A full per-term + per-wiring oracle investigation localized this to a single,
precisely-measured cause:

> **legoESM's nonlinear barotropic C-grid discretization is not energy-conserving, and the
> defect is GRID-SCALE / gradient-dependent (an enstrophy-conservation defect).** In the
> **inviscid, unforced** limit (where the continuous advection + Coriolis + free-surface
> equations conserve kinetic energy exactly):
> - a *smooth* gyre initial condition is roughly conserved (ratio ≈ 0.98 over 1000 steps);
> - the *equilibrated MITgcm WBC* state spuriously **gains ≈46% KE per 28 days**;
> - a sharp Munk-like western-boundary jet **loses ≈52%**.
>
> The injection/loss is **dt-independent** (1.4604 / 1.4600 / 1.4598 at dt = 1200 / 600 / 300 s) —
> a *spatial* discretization defect, not a time-stepping error — and its **sign is
> state-dependent**, the hallmark of a scheme that fails to conserve energy/enstrophy at sharp
> (grid-scale) features such as the western boundary current.

MITgcm's equilibrium momentum balance is **inertial** (advection ≈ Coriolis ≈ PGF ≈ `1.5e-6`;
friction — lateral viscosity, side-drag, bottom drag — all negligible, `Um_ImplD = 0`). Because
that equilibrium is essentially frictionless, legoESM's spurious energy source has nothing to
balance it, so the gyre runs up to a much higher amplitude before the discrete nonlinearity
finally saturates it.

## What was checked and ELIMINATED (all source-grounded against the MITgcm build)

Per-term tendency diffs on a bridged MITgcm state confirmed every spatial operator matches:

| Term | legoESM vs MITgcm | verdict |
|---|---|---|
| Lateral viscosity (`Um_Diss`) | corr 0.99996 | match |
| No-slip side-drag (`USidDrag`) | ratio 1.000, corr 1.0000 | **bit-exact** |
| Coriolis (`Um_Cori`) | ratio 1.000, corr 0.97 | match |
| Wind (`Um_Ext`) | ratio 1.000, corr 0.99 | match |
| Pressure gradient (`Um_dPhiX`) | ratio 0.998, corr 1.0000 | match |
| Bottom drag (`Um_ImplD`) | 0 in both (`viscAz=0`, `no_slip_bottom` inert) | match |

Eliminated as the cause (with evidence):
- **Lateral viscosity operator** — the "2.7× too weak" scare was an artifact of MITgcm's
  `Um_Diss` diagnostic *including* the side-drag; `Um_Diss − USidDrag` matches legoESM to 1e-15.
- **Momentum-advection magnitude** — the probe's near-zero `vortcor_u` is the (correctly-zero)
  baroclinic *perturbation* advection; the full advection reaches the barotropic mode via the
  depth-mean `F_slow` (`ocean_model_latlon_cgrid.py:1609`). Advection IS active: the gyre
  equilibrium depends on the scheme (flux_form 0.19 vs vector_invariant 0.36).
- **Time integrator / Coriolis placement** — implementing MITgcm's wiring (one AB2 predictor
  carrying advection + EXPLICIT Coriolis, then the implicit free-surface solve; `explicit_ab2` +
  `implicit_cn` with the barotropic solver's own Coriolis gated off) did **not** close the gap;
  energy injection is dt-independent, ruling out the AB2/forward-Euler distinction.
- **Bottom drag / friction** — negligible in MITgcm at equilibrium; not the missing sink.

## The wiring difference (necessary but not sufficient)

MITgcm is **unsplit**: one AB2 momentum predictor `{advection + explicit Coriolis(v^n) +
viscosity + sidedrag + wind, no surface PGF}` → one implicit `cg2d` free-surface solve → velocity
correction. legoESM **operator-splits** barotropic/baroclinic: advection (forward-Euler) →
depth-mean `F_slow` forcing to the barotropic solver, with Coriolis a separate stage. The wiring
was made matchable (see "Change made"), but the energy leak persists because it is in the
*spatial energy budget* of the C-grid operators in combination — orthogonal to the time wiring.

## Wiring: verified matchable, but NOT the fix (not committed)

The wiring CAN be made to match MITgcm: a single AB2 momentum predictor carrying advection +
EXPLICIT Coriolis, then the implicit free-surface solve. This was prototyped by allowing
`coriolis_scheme="explicit_ab2"` with `barotropic_solver="implicit_cn"` (gating off the
implicit-CN solver's own forward-backward Coriolis so Coriolis is applied once, in the AB2
predictor). It is valid, stable, holds the 10-step match (0.9996), and passes
`tests/ocean/unit/test_ab2_scope.py` (no regression to the `explicit_ab2`+`rigid_lid` path the
Veros recipes use). **But it does NOT close the equilibrium gap** — the energy injection is
dt-independent and gradient-dependent, i.e. spatial, so it is orthogonal to the time wiring.
Because matching the wiring without an energy-conserving spatial scheme makes the symptom *worse*
(it removes upwind advection's accidental dissipation that was masking the leak), the wiring
change was reverted; the gyre recipe stays at its prior best-matching config until the
energy-conserving momentum scheme below lands. The exact wiring edits are recorded here so the
follow-up project can re-apply them on top of the conserving scheme.

## The actual fix (scoped, not done here)

An **energy-conserving (and ideally enstrophy-conserving) C-grid momentum scheme** for the
nonlinear barotropic dynamics — the Arakawa/Sadourny family: an energy-conserving Coriolis
discretization plus an enstrophy-conserving advective (vorticity-flux) form, with the
PGF/divergence pair kept discretely adjoint. The regression test
`tests/ocean/unit/test_barotropic_energy_conservation.py` pins the current defect (inviscid
unforced KE growth) so that work has a target to drive to ~0.

## Reproduce

```
# MITgcm reference (needs gfortran; conda env `mitgcm-build`):
python scripts/data/generate_mitgcm_barotropic_gyre_reference.py \
    --ref-root <ref> --optfile <linux_amd64_gfortran + gcc15 relax flags>
# Energy-conservation probe (inviscid, unforced, from MITgcm equilibrium): see
# the test, or /tmp/econs_dt.py in the investigation session.
```
