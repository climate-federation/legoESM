# Adversarial physics review — tropopause-refined sigma coordinate (round 1)

You are an independent ADVERSARIAL physics/numerics reviewer for legoESM (JAX
differentiable ESM). Find every bug, sign error, unit inconsistency,
broken-gradient pattern, conservation violation, hidden uniform-grid
assumption, and test-quality defect in the change below. Cite file:line.
If you believe a candidate concern is NOT a bug, say so and explain why.

Repo root is the `--cd` directory. The change is UNCOMMITTED: `git diff` +
the untracked `tests/unit/test_tropopause_refined_sigma.py` show it all.
You have full repo access — please GREP, don't guess.

## Motivation (measured, this campaign)

AMIP MPAS runs have a +17.8 K tropical warm bias at 100 hPa and TTL water
vapour 20x ERA5. Root cause: `create_sigma_coordinate` was UNIFORM IN SIGMA
(`jnp.linspace(0.01, 1, n+1)`), so at nlev=30 every layer is ~33 hPa and the
model's coldest tropical level lands at the ~26 hPa LID instead of ~100 hPa
— no tropopause cold point, no cold trap, warm wet TTL.
Adding levels does not fix the distribution: L40 (uniform sigma, dt=75)
spiked to 92 m/s at day 39 and BLEW UP at day 46 at dt=60 (100% NaN,
levels 0-11 = top-down failure, T_min 175 K precursor).

## The change

1. `packages/core/legoesm/grids/vertical.py`: NEW
   `tropopause_refined_sigma_half(n_levels, sigma_top=0.01, sigma_refine=0.12,
   refine=3.0, width=0.45)` — half levels placed by the equidistribution
   principle: quantiles of a density
   `d(sigma) = 1 + (refine-1)*exp(-0.5*((ln s - ln sigma_refine)/width)^2)`
   integrated in SIGMA space (trapezoid cumulative on a 20001-pt float64 host
   grid, `np.interp` inversion, endpoints pinned).
2. Same file: `create_sigma_coordinate` gains
   `tropopause_refine=1.0, sigma_refine=0.12, refine_width=0.45`;
   `tropopause_refine == 1.0` keeps the literal linspace (bit-identical
   default).
3. `packages/coupler/legoesm/driver/config.py`: `GridConfig.tropopause_refine`
   + `validate_strict` bound [1, 6] + sigma-only guard.
4. `packages/coupler/legoesm/driver/model_driver.py`: sigma branch passes it.
5. `scripts/run/run_amip.py`: `--tropopause-refine` flag.
6. `tests/unit/test_tropopause_refined_sigma.py`: 15 tests.

## Deployment context (all arms use `--vertical-coord sigma`, MPAS dycore)

- Production century: `--nlev 30 --dt 75 --vertical-coord sigma` (uniform)
- Stability arm now running: same + `--tropopause-refine 3.0`, 60 days
- Blown-up arm: `--nlev 40 --dt 75/60 --vertical-coord sigma` (UNIFORM,
  24.4 hPa layers everywhere, failure spanned levels 0-11 = 10-302 hPa)
- All arms: `--convection bechtold --gravity-wave-drag mcfarlane+hines
  --hard-saturation-adjustment --hard-sat-ice-curve
  --mpas-qv-smooth-del2-m2s 2e5`

## Measured grid trade (nlev=30, refine=3, p_s=985 hPa)

```
 k   p_f uni   dp uni |   p_f ref   dp ref
  0     26.10    32.51 |     29.74    39.78
  1     58.61    32.51 |     61.92    24.59
  2     91.11    32.51 |     82.84    17.25
  3    123.62    32.51 |     98.92    14.91
  4    156.12    32.51 |    113.46    14.18
  5    188.63    32.51 |    127.69    14.27
  6    221.13    32.51 |    142.30    14.94
  7    253.64    32.51 |    157.83    16.12
  8    286.14    32.50 |    174.82    17.86
 ...
 29    968.75    32.51 |    963.83    42.34
```
TTL full levels in 70-200 hPa: 4 -> 8. thinnest 32.5 -> 14.2 hPa.
top layer 32.5 -> 39.8 hPa (THICKER). lowest 32.5 -> 42.3 hPa (+30%).

## What I have already established numerically (verify or refute these too)

### CONFIRMED F1 — the #930 vertical biharmonic filter breaks column
conservation on ANY non-uniform grid.
`packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:148`
`vertical_del4_T_tendency` builds a fourth difference on the LEVEL INDEX with
edge-padded outer Laplacian, so `sum_k tend_k == 0` exactly. The docstring
(line ~186) and the tracer call site (line ~579-585) claim this
"conserves column moisture to machine precision". That is only true when
`dsigma` is constant. The physically conserved quantity is
`sum_k tend_k * dsigma_k`. Measured (nu_vert4_T = 2.0e-6 1/s, p_s = 985 hPa,
2-delta checkerboard of +-5 K / +-1 g/kg):

```
sigma L30 uniform   : enthalpy   +0.0000 W/m2   moisture  +0.00000 mm/day
sigma L30 refine=3  : enthalpy   -7.8587 W/m2   moisture  -0.13517 mm/day
sigma L30 refine=6  : enthalpy  -17.9331 W/m2   moisture  -0.30845 mm/day
hybrid L40 str=2.0  : enthalpy   -0.9884 W/m2   moisture  -0.01700 mm/day
hybrid L40 str=2.5  : enthalpy   -1.5326 W/m2   moisture  -0.02636 mm/day
```
Linear in amplitude: a +-1 K checkerboard at refine=3 gives -1.57 W/m2.
So the defect is PRE-EXISTING on the hybrid coordinate, but refine=3 makes
it ~8x worse than any shipped configuration, on a century run whose target
TOA imbalance is < 1 W/m2.

Candidate fix verified: subtract the mass-weighted mean,
`tend -= (tend*dsigma).sum()/dsigma.sum()`. Leaves the uniform grid changed
by max 1.4e-20 K/s (below float32 ULP => bit-identical in practice) and
drives the refined-grid residual to 8e-21 (machine zero).
QUESTION FOR YOU: is mean-removal the right fix, or should the operator be
rewritten in mass-flux form `-(F_{k+1/2}-F_{k-1/2})/dsigma_k`? The flux form
changes the effective RATE by 1/dsigma and is therefore NOT
backward-compatible for existing uniform/hybrid runs. Argue the trade.

### CONFIRMED F2 — `tropopause_refine` is silently DROPPED by the AMIP config
adapter. `packages/coupler/legoesm/driver/config.py:2377` (`from_amip_config`)
and `to_amip_config` (~2591) do not carry the field. Measured:
```
config_to_dict/from_dict        : 3.0  OK
to_amip_config/from_amip_config : 1.0  *** LOST (silently reverts to uniform)
```
`packages/coupler/legoesm/driver/checkpoint.py:42 _config_from_dict_auto`
routes AMIP-format JSON through `from_amip_config`. Assess the real blast
radius: which live paths deserialize through the AMIP adapter?

### CONFIRMED F3 — no restart guard on the vertical level DISTRIBUTION.
`model_driver.py:4842 load_checkpoint` MPAS branch guards SHAPES only
(`nCells`, `nlev`). Uniform-L30 and refine-3-L30 have identical shapes, so a
checkpoint written on one grid restarts silently on the other, reinterpreting
every profile. Precedent for the right fix is in the SAME function's save
side: `model_driver.py:~4332 physstate_meta_conv_scheme` ("shape checks alone
cannot catch a cross-scheme restore"). The MPAS npz stores no vertical
coordinate at all.

### CONFIRMED F4 — grid smoothness. Max adjacent-layer thickness ratio:
```
refine=1.0: 1.0000   refine=3.0: 1.6178   refine=6.0: 2.4061
HYBRID L40 stretching=2.0: 1.0492   HYBRID L40 stretching=2.5: 1.0635
```
Every shipped grid is <= 1.066; refine=3 is 1.62 and refine=6 is 2.41.
Operational practice keeps adjacent-layer ratios <~ 1.2-1.3. Is the
`validate_strict` upper bound of 6 defensible? What IS the right bound?

### REFUTED (check my work) — vertical advective CFL is NOT binding.
`vertical_advection` (vertical.py:490) is first-order upwind on
`dsigma_full = diff(sigma_full)`; `sigma_dot ~ omega/p_s`. Measured
CFL = |sigma_dot| dt / min(dsigma_full):
```
dt=75 refine=3 omega=0.5 Pa/s -> 0.026   omega=5.0 Pa/s -> 0.264
dt=60 refine=3 omega=5.0 Pa/s -> 0.211   dt=75 refine=6 omega=5.0 -> 0.392
```
The `nu_vert4_T` filter is index-based, so its explicit limit
(16*nu*dt = 2.4e-3) does not change with refinement at all. Both GWD schemes
cap acceleration by `min(umcfac*U/dt, tndmax)`, which is dp-INDEPENDENT
(hines.py:283-297, mcfarlane.py:544-552). So I find no explicit vertical
stability constraint that binds at refine<=6 for dt in {60, 75}.
DISPROVE THIS if you can — name any term in the MPAS sigma path or the
physics whose stability limit scales with 1/dsigma or 1/dp.

### CONFIRMED F6 — an unsupported causal claim in the docstring.
`vertical.py:262-264` says the refinement's "side benefit: the top layer
THICKENS (33 -> 40 hPa), the opposite of the thin-top configuration that blew
L40 up at day 46." But L40 was UNIFORM sigma with 24.4 hPa layers EVERYWHERE
and its failure spanned levels 0-11 = 10-302 hPa. refine=3 places 14.2-17.9
hPa layers at 99-175 hPa — INSIDE that band and 42% thinner than anything L40
had. The claim only covers k=0. Is the comment defensible as written?

## Please additionally hunt (grep, do not assume)

- Anything else assuming uniform sigma: unweighted level averages,
  `dsigma[0]`, hard-coded layer thickness, plev/CMOR interpolation, the
  `mpas_qv_smooth_del2_m2s` smoother, the hard-saturation adjustment /
  ice-curve drain, the ice-skin and land-boundary code, GWD launch-level
  selection, radiation layer/level construction, the Bechtold convection
  scheme's entrainment or cloud-top detection.
- `create_sigma_coordinate` ignores `GridConfig.p_top_Pa` entirely
  (`model_driver.py:1002-1005` never passes it; sigma_top is hardcoded 0.01).
  Is that a real defect and does the new `validate_strict` message
  ("hybrid ... has its own `stretching`") mislead?
- Numerics of the construction: 20001-pt trapezoid + linear interp. I measure
  max placement error 1.1e-8 in sigma (1e-5 hPa) vs a 4e6-pt reference, and
  0 non-monotone cases over
  n in {10,20,30,40,60,80,137} x refine in {1.0001,1.5,3,6} x
  width in {0.02..3} x sigma_refine in {0.0101..0.999} x
  sigma_top in {1e-5,1e-3,1e-2}, in float64 AND after the float32 storage
  cast. Simmons-Burridge alpha stays in (0.016, 0.65) in all of them.
  Find a parameter combination that breaks it if one exists.
- Differentiability: `tropopause_refined_sigma_half` is host-side numpy,
  called once at setup and frozen into `SigmaCoordinate`. Confirm this cannot
  break `jax.grad`/`jit`/`vmap` through the dycore, and confirm there is no
  path where `tropopause_refine` becomes a traced value.
- Test quality: are the pinned numbers non-vacuous? My mutation test
  (replace the density with flat / Gaussian-in-sigma / inverted) shows the
  suite catches all three, but `test_top_layers_not_thinned`
  (`ref[0] >= uni[0]`) passes trivially on the "flat" mutant. What else is
  weak?
- The equidistribution is in SIGMA but the density bump is Gaussian in LOG
  sigma. Is that mathematically coherent, and is the resulting level
  distribution what the docstring claims?

## Full source of the changed hunks

<<<DIFF>>>

## The new test file

<<<TESTS>>>
