# ORCA12 Marmara instability: diagnosis, not a verified fix

## Status

The active pressure-gradient implementation has a known truncation error.
The available evidence does **not** establish that error as the feedback
mechanism behind the reported exponential velocity growth. No production
numerics, scheme defaults, or user-staged files were changed in this diagnosis.
The requested instability fix and a regression that fails without that fix
remain outstanding.

## Recovered provenance and executed path

The existing run `n16_dt150_j27391836` reproduces the supplied speed sequence
in its `run.log`. Its artifacts are under
`/work/bd1083/b309178/diffESM/legoesm_pg/ocean_hires_runs/smoke_orca12/`.
Its `tripole/eorca1/run_config.json` records:

- `pgf_scheme="adcroft"`, `pgf_quadrature="cell_integral"`, Wright EOS;
- vector-invariant momentum, centered KE gradient, AL81 vorticity;
- forward-Euler outer integration, Euler tracer integration;
- `vertical_momentum_scheme="upwind_perturbation"`;
- `adaptive_implicit_vertadv=false`, `coriolis_scheme="matsuno_split"`.

The CLI's unset PGF override reaches `NEMOMatchTripoleRecipeConfig` in
`packages/ocean/legoesm/ocean/fidelity/nemo_match_recipe.py:192`, which selects
`adcroft`. The resolved artifact confirms this is the actual selection,
not merely a current default.

`ocean_pe_latlon_cgrid.py:1240` constructs density and pressure on actual
partial-cell thicknesses. `ocean_tendency_common.py:315` computes
`p_k = g sum(l<k, rho'_l h_l) + g rho'_k h_k/2`.
The call at `ocean_pe_latlon_cgrid.py:4552` reaches the Adcroft branch in
`_bc_ke_and_pressure_gradients`; the resulting pressure gradient enters
momentum with sign `-grad(p')/rho_0` at line 4519.

## What the pressure operator actually does

Depth is positive downward. At a face it selects the shallower centroid
`z_f=min(z_W,z_E)` and compares
`p_E-g rho'_E(z_E-z_f)` with `p_W-g rho'_W(z_W-z_f)`.
See `latlon_cgrid_operators.py:3732` and `:3832`. This is a constant-density
shift, not a vertical reconstruction of density through the shifted interval.

It has **no division by partial-cell thickness**. For the supplied pair of
bottom thicknesses the centroid separation is half their difference, not
the entire bathymetric cliff. Faces beneath the shallower column are closed
by the 3-D active-cell masks (`ocean_pe_latlon_cgrid.py:4477`).

Analytically, consider a common cell-top depth, shallower thickness `h`, deeper
thickness `H`, and horizontally uniform linear density `rho'=a+s*z`.
The true horizontal pressure gradient is zero. With the deeper column east,
the implemented residual is

```
grad_x(p') = g*s*h*(H-h)/(4*dx).
```

This is an algebraic derivation, not a measurement of Marmara's density profile.
It tends to zero as `h` tends to zero. Conversely, for vertically constant
but horizontally different densities, the same implementation gives the
correct nonzero hydrostatic pressure difference at the common depth. A large
salinity contrast alone therefore does not establish a spurious PGF.

The existing `smc03` selector reconstructs vertical density slopes and
integrates both columns to the common depth. Existing tests already demonstrate
that it removes the linear-density rest-state error. It also changes pressure
quadrature on full-cell faces when vertical density slopes differ horizontally;
switching to it is not generally bitwise neutral for healthy runs.

## Competing mechanisms and missing discriminator

- **Partial-cell PGF truncation error:** present in code and covered by existing
  manufactured tests; possible seed, unverified explanation of the growth.
  Fixed forcing alone does not demonstrate an exponential feedback loop.
- **Physical adjustment to the unresolved density front plus unstable transport:**
  still possible. The matching run's step-40 log reports finite temperature
  extrema at levels 5 and 6 in the same region, rather than at the cited bottom
  layer. These are late damage observations, not proof of the initiating term.
- **PV/KE or vertical-momentum feedback over variable thickness:** still possible.
  Unlike the PGF, these operators do contain thickness divisions. No local
  term budget or velocity-maximum depth was provided or recovered.
- **Explicit advective/diffusive CFL:** lower-timestep failures weaken a simple
  fixed-speed CFL explanation, but do not exclude an evolving local CFL failure.
  No matched physical-time tendency/CFL comparison was recovered.
- **Initial NaNs:** excluded by the supplied initial-condition measurement.
- **Global barotropic origin:** disfavored by localized onset, but a stable
  global kinetic-energy fraction cannot rule out a local barotropic response.
- **Northern fold or partition boundary:** the reported location is away from
  the fold and inside band 10 in the log; this does not certify all halo code.

The existing public `LatLonCGridOceanModel.tendencies_with_diagnostics`
(`ocean_model_latlon_cgrid.py:3962`) supplies the component momentum budget.
The next necessary input is an initial or pre-runaway state with the velocity
maximum's level. Evaluate that budget at the growing faces, separate PGF from
KE (the current diagnostic combines them), and compare otherwise identical
Adcroft and SMC03 evaluations. A frozen-state PGF difference measures forcing,
not growth-rate attribution; the latter requires a controlled trajectory or
linearized perturbation test. No new long integration was launched.

## Validation and limits

CPU, `JAX_ENABLE_X64=1`, existing tests:

```
pytest tests/ocean/unit/test_pgf_smc03_phase3.py \
       tests/ocean/unit/test_partial_cells_phase3b.py -q
14 passed, 1 skipped in 35.28s
```

These validate existing operators, not an ORCA12 cure. No new regression is
claimed; no fix-reversion test exists because no fix was made. No new module,
parameter, selector, constant, or pytree field was introduced.

The specified NEMO source directory is absent here; no NEMO execution-path
equivalence is claimed. GitHub issue access failed with a network connection
error; nothing was posted. The recovered snapshot filename denotes a partially
written final snapshot, not a verified initial/pre-runaway checkpoint.

Default choice: retain `adcroft` pending attribution. The available alternative
is explicit `smc03`, which addresses the demonstrated rest-state error but is
**not verified to unblock this lane**. This retained default and its known
error are flagged deliberately, not presented as a completed fix.
