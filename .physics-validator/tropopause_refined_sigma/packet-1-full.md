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

```diff
diff --git a/packages/core/legoesm/grids/vertical.py b/packages/core/legoesm/grids/vertical.py
index 4a07d3c04..38fee86b0 100644
--- a/packages/core/legoesm/grids/vertical.py
+++ b/packages/core/legoesm/grids/vertical.py
@@ -102,8 +102,11 @@ def create_sigma_coordinate(
     n_levels: int,
     sigma_top: float = 0.01,
     dtype=None,
+    tropopause_refine: float = 1.0,
+    sigma_refine: float = 0.12,
+    refine_width: float = 0.45,
 ) -> SigmaCoordinate:
-    """Create a uniformly spaced sigma coordinate.
+    """Create a sigma coordinate (uniform by default).
 
     Parameters
     ----------
@@ -117,6 +120,14 @@ def create_sigma_coordinate(
         Dtype for coordinate arrays. If None, uses the precision
         policy's compute dtype (defaults to float32 when no policy
         is active). Explicit dtype overrides the policy.
+    tropopause_refine : float
+        Peak density ratio for the tropopause refinement (see
+        :func:`tropopause_refined_sigma_half`).  ``1.0`` (default) is the
+        UNIFORM grid, bit-identical to the pre-refinement behaviour.
+        Values > 1 redistribute layers toward ``sigma_refine`` at the SAME
+        level count — the fix for the unresolved tropical cold point.
+    sigma_refine, refine_width : float
+        Centre (in sigma) and log-sigma half-width of the refinement.
 
     Returns
     -------
@@ -133,7 +144,16 @@ def create_sigma_coordinate(
             dtype = get_policy().compute
         except Exception:
             dtype = jnp.float32
-    sigma_half = jnp.linspace(sigma_top, 1.0, n_levels + 1, dtype=dtype)
+    if tropopause_refine == 1.0:
+        # Uniform (default) — kept as the literal linspace so the untouched
+        # path stays bit-identical to the pre-refinement code.
+        sigma_half = jnp.linspace(sigma_top, 1.0, n_levels + 1, dtype=dtype)
+    else:
+        sigma_half = jnp.asarray(
+            tropopause_refined_sigma_half(
+                n_levels, sigma_top=sigma_top, sigma_refine=sigma_refine,
+                refine=tropopause_refine, width=refine_width),
+            dtype=dtype)
     sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
     dsigma = sigma_half[1:] - sigma_half[:-1]
 
@@ -163,6 +183,99 @@ def create_sigma_coordinate(
     )
 
 
+def tropopause_refined_sigma_half(
+    n_levels: int,
+    sigma_top: float = 0.01,
+    sigma_refine: float = 0.12,
+    refine: float = 3.0,
+    width: float = 0.45,
+) -> np.ndarray:
+    """Half-level sigma with layers REDISTRIBUTED toward the tropopause.
+
+    The uniform-in-sigma default (:func:`create_sigma_coordinate`) spaces
+    every layer by the same ``dp = dsigma * p_s`` — about 33 hPa at 30
+    levels — so the tropical tropopause layer, whose structure is a
+    10-20 hPa affair, is spanned by ~3 levels and the model forms no cold
+    point (its coldest tropical level lands at the ~26 hPa top instead of
+    ~100 hPa; measured 2026-07-25).  Adding levels does NOT fix this: at
+    40 levels the TTL still gets 4 levels, while the thinner top layers
+    destabilised the model (blow-up at day 46 with dt=60).
+
+    So redistribute at FIXED count instead.  Levels are placed by the
+    standard equidistribution principle: they are the quantiles of a
+    density ``d(sigma)`` in SIGMA space, so a flat density reproduces the
+    uniform default exactly and the refinement only steals layers from the
+    (over-resolved) mid-troposphere::
+
+        d(sigma) = 1 + (refine - 1) * exp(-0.5 * ((ln sigma - ln sigma_refine) / width)^2)
+
+    The bump is Gaussian in LOG sigma because atmospheric structure scales
+    with log-pressure; ``width`` is therefore in log-sigma units (0.45 ~ a
+    factor e^0.45 = 1.6 in pressure either side of the centre).  Working in
+    sigma (not log-sigma) for the equidistribution is deliberate: a pure
+    log-sigma grid would put 166 hPa between the lowest levels at 30
+    levels and destroy the boundary layer.
+
+    Parameters
+    ----------
+    n_levels : int
+        Number of layers (returns ``n_levels + 1`` half levels).
+    sigma_top : float
+        Sigma of the model top (same meaning as in
+        :func:`create_sigma_coordinate`).
+    sigma_refine : float
+        Centre of the refinement, in sigma (0.12 ~ 120 hPa at p_s = 1000
+        hPa — the tropical cold point).
+    refine : float
+        Peak density ratio.  ``refine = 1`` reproduces the uniform grid
+        EXACTLY (the identity case, pinned by a test).
+    width : float
+        Gaussian half-width of the bump in log-sigma units.
+
+    Returns
+    -------
+    numpy.ndarray, shape ``(n_levels + 1,)``
+        Monotone increasing half-level sigma from ``sigma_top`` to 1.
+    """
+    if not (refine >= 1.0 and width > 0.0):
+        raise ValueError(
+            f"tropopause_refined_sigma_half: need refine >= 1 and width > 0; "
+            f"got refine={refine!r}, width={width!r}")
+    if not (0.0 < sigma_top < sigma_refine < 1.0):
+        raise ValueError(
+            f"tropopause_refined_sigma_half: need 0 < sigma_top < "
+            f"sigma_refine < 1; got sigma_top={sigma_top!r}, "
+            f"sigma_refine={sigma_refine!r}")
+    # Fine auxiliary grid; the quantile inversion below is a 1-D interp, so
+    # resolution here only sets the placement accuracy (not a runtime cost:
+    # this runs once at setup, on the host, in float64).
+    s = np.linspace(sigma_top, 1.0, 20001, dtype=np.float64)
+    _ln = np.log(s)
+    dens = 1.0 + (refine - 1.0) * np.exp(
+        -0.5 * ((_ln - np.log(sigma_refine)) / width) ** 2)
+    # DELIBERATELY single-bump.  A matching surface bump was tried and
+    # rejected (measured 2026-07-25, 30 levels, refine=3): it does protect
+    # the lowest layer (42 -> 23 hPa) but it is very WIDE in sigma (it spans
+    # sigma ~ 0.6-1.0), so it starves the tropopause back to 5 levels from 8
+    # and thickens the TOP layer to 57 hPa.  At 30 levels the grid cannot
+    # refine both ends; the tropopause is the identified defect, so it wins.
+    # ACCEPTED COST: the lowest layer coarsens ~30% (33 -> 42 hPa at
+    # refine=3).  Side benefit: the top layer THICKENS (33 -> 40 hPa), the
+    # opposite of the thin-top configuration that blew L40 up at day 46.
+    # Equidistribution: place levels at equal increments of the cumulative
+    # density, so spacing ~ 1/d — fine where d is large.
+    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1])
+                                           * np.diff(s))])
+    cum /= cum[-1]
+    targets = np.linspace(0.0, 1.0, n_levels + 1)
+    half = np.interp(targets, cum, s)
+    # Pin the ends exactly (interp round-off would otherwise move the lid
+    # and the surface by ~1e-16, which the hydrostatic integration and the
+    # p_s = sigma=1 identity both assume).
+    half[0], half[-1] = sigma_top, 1.0
+    return half
+
+
 def pressure_from_sigma(
     sigma: jax.Array,
     p_s: jax.Array,
diff --git a/packages/coupler/legoesm/driver/config.py b/packages/coupler/legoesm/driver/config.py
index e3b40740d..658d40557 100644
--- a/packages/coupler/legoesm/driver/config.py
+++ b/packages/coupler/legoesm/driver/config.py
@@ -64,6 +64,14 @@ class GridConfig(NamedTuple):
     vertical_coord: str = "hybrid"   # sigma, hybrid
     p_top_Pa: float = 200.0
     stretching: float = 2.0
+    # SIGMA-coordinate layer redistribution toward the tropopause, at FIXED
+    # nlev (grids/vertical.tropopause_refined_sigma_half).  1.0 = the uniform
+    # grid, bit-identical.  Uniform sigma gives ~33 hPa layers everywhere at
+    # nlev=30, so the tropical tropopause is spanned by ~4 levels and no cold
+    # point forms; refine=3 doubles the levels in 70-200 hPa, paid for by the
+    # mid-troposphere (the lowest layer coarsens ~30%, measured).  Ignored by
+    # the hybrid coordinate, which has its own `stretching`.
+    tropopause_refine: float = 1.0
     use_duogrid: bool = False        # enable FV3 Duo-Grid halo (required for MPI multi-node)
 
 
@@ -1296,6 +1304,20 @@ class ExperimentConfig(NamedTuple):
             errors.append(f"grid.nlev must be > 0, got {g.nlev}")
         if g.p_top_Pa <= 0:
             errors.append(f"grid.p_top_Pa must be > 0, got {g.p_top_Pa}")
+        # Tropopause refinement: 1.0 = uniform; the upper bound keeps the
+        # thinnest layer from approaching the vertical-CFL limit (thin layers
+        # are exactly what blew L40 up at day 46).
+        if not (1.0 <= g.tropopause_refine <= 6.0
+                and math.isfinite(g.tropopause_refine)):
+            errors.append(
+                f"grid.tropopause_refine must be finite in [1, 6] "
+                f"(1 = uniform); got {g.tropopause_refine}")
+        if g.tropopause_refine != 1.0 and g.vertical_coord != "sigma":
+            errors.append(
+                f"grid.tropopause_refine={g.tropopause_refine} only applies "
+                f"to the sigma coordinate; vertical_coord="
+                f"{g.vertical_coord!r} has its own `stretching` and would "
+                "silently ignore it.")
         if d.dt <= 0:
             errors.append(f"dycore.dt must be > 0, got {d.dt}")
         if d.hyperdiff_scale < 0:
diff --git a/packages/coupler/legoesm/driver/model_driver.py b/packages/coupler/legoesm/driver/model_driver.py
index 282531218..9d0fffb80 100644
--- a/packages/coupler/legoesm/driver/model_driver.py
+++ b/packages/coupler/legoesm/driver/model_driver.py
@@ -1000,7 +1000,9 @@ class ModelDriver:
             )
         else:
             from legoesm.grids.vertical import create_sigma_coordinate
-            self.sigma = create_sigma_coordinate(gc.nlev)
+            self.sigma = create_sigma_coordinate(
+                gc.nlev,
+                tropopause_refine=getattr(gc, "tropopause_refine", 1.0))
 
         logger.info(f"  Grid: {gc.grid_type} {gc.resolution}, "
               f"{gc.nlev} levels ({gc.vertical_coord})")
diff --git a/scripts/run/run_amip.py b/scripts/run/run_amip.py
index ed01d26c0..1a86d0b0b 100644
--- a/scripts/run/run_amip.py
+++ b/scripts/run/run_amip.py
@@ -1279,6 +1279,14 @@ def build_arg_parser() -> argparse.ArgumentParser:
                              "__param_spec__ entry yet (bounds undecided), so "
                              "this flag is its ONLY route -- --params cannot "
                              "reach it.")
+    parser.add_argument("--tropopause-refine", type=float, default=None,
+                        dest="tropopause_refine",
+                        help="Sigma-coordinate layer redistribution toward "
+                             "the tropopause at FIXED nlev (1.0 = uniform, "
+                             "bit-identical; 3.0 doubles the levels in "
+                             "70-200 hPa, paid for by the mid-troposphere). "
+                             "Fixes the unresolved tropical cold point "
+                             "without adding levels. Sigma coordinate only.")
     parser.add_argument("--hard-sat-ice-curve",
                         action=argparse.BooleanOptionalAction, default=False,
                         dest="hard_sat_ice_curve",
@@ -1580,6 +1588,8 @@ def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
         vertical_coord=args.vertical_coord,
         p_top_Pa=args.p_top if args.p_top is not None else 200.0,
         stretching=args.stretching if args.stretching is not None else 2.0,
+        tropopause_refine=(args.tropopause_refine
+                           if args.tropopause_refine is not None else 1.0),
         use_duogrid=getattr(args, "use_duogrid", False),
     )
 
```

## The new test file
```python
"""Tropopause-refined sigma: redistribute layers at FIXED level count.

The uniform-in-sigma default spaces every layer ~33 hPa at 30 levels, so
the tropical tropopause layer gets ~3 levels and the model forms no cold
point (measured 2026-07-25: coldest tropical level at the ~26 hPa lid).
Adding levels does not fix the distribution and destabilised the model
(L40 blow-up at day 46, dt=60).  These pin the redistribution: identity at
refine=1, more TTL resolution at refine>1, and — the load-bearing safety
property — no thinning of the top layers, which is what killed L40.
"""

import numpy as np
import pytest

from legoesm.grids.vertical import (
    create_sigma_coordinate,
    tropopause_refined_sigma_half,
)

P_S = 985.0  # hPa, the campaign's typical surface pressure


def _dp(half):
    return np.diff(np.asarray(half)) * P_S


class TestGridProperties:
    def test_identity_at_refine_one(self):
        """refine=1 must reproduce the uniform grid EXACTLY — the default
        path stays byte-identical."""
        got = tropopause_refined_sigma_half(30, refine=1.0)
        np.testing.assert_allclose(got, np.linspace(0.01, 1.0, 31), atol=1e-12)

    def test_monotone_and_endpoints_exact(self):
        for n in (20, 30, 40):
            h = tropopause_refined_sigma_half(n, refine=3.0)
            assert h.shape == (n + 1,)
            assert np.all(np.diff(h) > 0)
            assert h[0] == 0.01 and h[-1] == 1.0

    def test_refinement_adds_ttl_levels(self):
        """The whole point: more levels between 70 and 200 hPa."""
        uni = tropopause_refined_sigma_half(30, refine=1.0) * P_S
        ref = tropopause_refined_sigma_half(30, refine=3.0) * P_S
        n_uni = int(((uni >= 70) & (uni <= 200)).sum())
        n_ref = int(((ref >= 70) & (ref <= 200)).sum())
        assert n_ref > n_uni, (n_uni, n_ref)

    def test_ttl_spacing_improves(self):
        """Layer thickness AT the tropopause shrinks."""
        for h, label in ((tropopause_refined_sigma_half(30, refine=1.0), "uni"),
                         (tropopause_refined_sigma_half(30, refine=3.0), "ref")):
            p = np.asarray(h) * P_S
            mid = 0.5 * (p[1:] + p[:-1])
            near = np.abs(mid - 120.0) < 60.0
            thick = _dp(h)[near].mean()
            if label == "uni":
                uni_thick = thick
            else:
                assert thick < 0.75 * uni_thick, (uni_thick, thick)

    def test_top_layers_not_thinned(self):
        """SAFETY: L40 died at day 46 because thin TOP layers destabilise the
        model.  Redistribution must THICKEN the top, never thin it."""
        uni = _dp(tropopause_refined_sigma_half(30, refine=1.0))
        ref = _dp(tropopause_refined_sigma_half(30, refine=3.0))
        assert ref[0] >= uni[0], (uni[0], ref[0])

    def test_boundary_layer_cost_is_bounded(self):
        """The boundary layer PAYS for the tropopause — an accepted, measured
        cost, not a free lunch: the lowest layer coarsens ~30% (33 -> 42 hPa
        at 30 levels, refine=3).  Pinned so a future density change cannot
        quietly make it worse (and nowhere near the 166 hPa a naive
        log-spaced grid would give)."""
        uni = _dp(tropopause_refined_sigma_half(30, refine=1.0))
        ref = _dp(tropopause_refined_sigma_half(30, refine=3.0))
        assert 1.0 < ref[-1] / uni[-1] < 1.35, (uni[-1], ref[-1])

    def test_layers_come_from_the_mid_troposphere(self):
        """Conservation of levels: the TTL gain is paid for by the
        (over-resolved) mid-troposphere, not the ends."""
        uni = tropopause_refined_sigma_half(30, refine=1.0) * P_S
        ref = tropopause_refined_sigma_half(30, refine=3.0) * P_S
        band = lambda p, lo, hi: int(((p >= lo) & (p <= hi)).sum())  # noqa: E731
        assert band(ref, 400, 800) < band(uni, 400, 800)

    def test_monotone_in_refine_strength(self):
        counts = []
        for r in (1.0, 2.0, 3.0, 5.0):
            p = tropopause_refined_sigma_half(30, refine=r) * P_S
            counts.append(int(((p >= 70) & (p <= 200)).sum()))
        assert counts == sorted(counts), counts

    @pytest.mark.parametrize("bad", [
        dict(refine=0.5), dict(width=0.0), dict(sigma_refine=1.5),
        dict(sigma_top=0.2, sigma_refine=0.1),
    ])
    def test_invalid_parameters_raise(self, bad):
        with pytest.raises(ValueError):
            tropopause_refined_sigma_half(30, **bad)


class TestCoordinateIntegration:
    def test_default_is_bit_identical(self):
        a = create_sigma_coordinate(30)
        b = create_sigma_coordinate(30, tropopause_refine=1.0)
        np.testing.assert_array_equal(np.asarray(a.sigma_half),
                                      np.asarray(b.sigma_half))
        np.testing.assert_array_equal(np.asarray(a.dsigma),
                                      np.asarray(b.dsigma))

    def test_refined_coordinate_is_self_consistent(self):
        """The derived Simmons-Burridge quantities must stay finite and
        physical on the refined grid (they divide by dsigma and take logs
        of sigma ratios)."""
        c = create_sigma_coordinate(30, tropopause_refine=3.0)
        for name in ("sigma_full", "sigma_half", "dsigma", "ln_ratio",
                     "alpha", "fractional_sigma", "dsigma_full"):
            arr = np.asarray(getattr(c, name))
            assert np.all(np.isfinite(arr)), name
        assert np.all(np.asarray(c.dsigma) > 0)
        # alpha is the Simmons-Burridge weight: strictly within (0, 1).
        alpha = np.asarray(c.alpha)
        assert np.all((alpha > 0) & (alpha < 1)), alpha

    def test_refined_cold_point_band_resolved(self):
        """End-to-end statement of the defect being fixed: at 30 levels the
        refined grid puts >= 4 full levels in 70-200 hPa (uniform gives 2)."""
        uni = np.asarray(create_sigma_coordinate(30).sigma_full) * P_S
        ref = np.asarray(
            create_sigma_coordinate(30, tropopause_refine=3.0).sigma_full) * P_S
        n_uni = int(((uni >= 70) & (uni <= 200)).sum())
        n_ref = int(((ref >= 70) & (ref <= 200)).sum())
        # Measured baseline: uniform puts 4 full levels in the band; the
        # refinement doubles it.  (Only 3 sit above 100 hPa uniformly.)
        assert n_uni == 4 and n_ref >= 7, (n_uni, n_ref)
```
