You are an independent ADVERSARIAL physics/numerics reviewer for legoESM.
Be efficient: budget at most ~15 tool calls, then WRITE THE REVIEW.
Do NOT re-print the diff or the tests in your answer — I have them.

SCOPE: the UNCOMMITTED change in the working tree (`git diff` +
untracked `tests/unit/test_tropopause_refined_sigma.py`): a new
tropopause-refined sigma vertical coordinate.
`packages/core/legoesm/grids/vertical.py::tropopause_refined_sigma_half`
places half-levels as quantiles of a density
`d(s) = 1 + (refine-1)*exp(-0.5*((ln s - ln 0.12)/0.45)^2)` integrated in
SIGMA (20001-pt trapezoid, np.interp inversion, endpoints pinned).
`create_sigma_coordinate(..., tropopause_refine=1.0)` keeps the literal
linspace. `GridConfig.tropopause_refine` bounded [1,6], sigma-only.
`run_amip.py --tropopause-refine`.

DEPLOYMENT: MPAS hydrostatic dycore, `--vertical-coord sigma --nlev 30
--dt 75`, 100-year AMIP. Grid at refine=3, p_s=985 hPa: layers
39.8, 24.6, 17.3, 14.9, 14.2, 14.3, 14.9, 16.1, 17.9, ... 42.3 hPa
(thinnest 14.2 hPa at 113 hPa; uniform is 32.5 hPa everywhere).
The L40 run that blew up was UNIFORM sigma, 24.4 hPa layers everywhere,
failure spanned levels 0-11 = 10-302 hPa.

OUTPUT FORMAT — a compact list. For each item:
`[CONFIRMED|REFUTED|UNCERTAIN] <file>:<line> — <one-paragraph argument>`
Then a final "VERDICT: is refine=3 safe for a 100-year run? " paragraph.

=== MY FINDINGS, ALREADY MEASURED. Verify or refute each. ===

F1 CONFIRMED. `primitive_eq_mpas.py:148 vertical_del4_T_tendency` is a
fourth difference on the LEVEL INDEX with edge-padded outer Laplacian, so
`sum_k tend_k == 0` exactly. Its docstring and the tracer call site
(~line 579-585) claim it "conserves column moisture to machine precision".
That holds only for constant `dsigma`. The conserved quantity is
`sum_k tend_k * dsigma_k`. Measured (nu_vert4_T=2e-6 1/s, p_s=985 hPa,
2-delta checkerboard +-5 K / +-1 g/kg):
  sigma L30 uniform : +0.0000 W/m2 , +0.00000 mm/day
  sigma L30 refine=3: -7.8587 W/m2 , -0.13517 mm/day
  sigma L30 refine=6: -17.933 W/m2 , -0.30845 mm/day
  hybrid L40 str=2.0: -0.9884 W/m2 , -0.01700 mm/day  (PRE-EXISTING!)
Linear in amplitude (+-1 K -> -1.57 W/m2 at refine=3).
Candidate fix: subtract the mass-weighted mean
`tend -= (tend*dsigma).sum()/dsigma.sum()`; measured to change the uniform
grid by <=1.4e-20 K/s (below float32 ULP) and to zero the refined residual.
QUESTION: mean-removal vs rewriting the operator in mass-flux form
`-(F_{k+1/2}-F_{k-1/2})/dsigma_k` (which changes the effective rate by
1/dsigma => NOT backward compatible for existing uniform/hybrid runs)?
Which is right? Does mean-removal do anything physically objectionable
(e.g. inject a uniform tendency where the filter should be local)?

F2 CONFIRMED. `driver/config.py:2377 from_amip_config` and `to_amip_config`
do not carry `tropopause_refine`. Measured round trip returns 1.0 (uniform).
`driver/checkpoint.py:42 _config_from_dict_auto` routes AMIP-format JSON
through it. Assess the real blast radius: which LIVE paths deserialize
through the AMIP adapter, and does any of them feed a restarted run's grid?

F3 CONFIRMED. `model_driver.py:4842 load_checkpoint` MPAS branch guards
SHAPES only. Uniform-L30 and refine-3-L30 have identical shapes, so a
checkpoint silently restarts on the wrong grid. Precedent for the fix is
in the same function's save side (`physstate_meta_conv_scheme`, ~line 4332
— "shape checks alone cannot catch a cross-scheme restore"). Agree?

F4 CONFIRMED. Max adjacent-layer thickness ratio: refine=1 -> 1.000,
refine=3 -> 1.618, refine=6 -> 2.406. Every SHIPPED grid is <= 1.066
(hybrid L40 stretching 2.0/2.5). Is the validate_strict upper bound of 6
defensible? What bound would you set, and on what criterion?

F5 REFUTED (check me). Vertical advective CFL is not binding.
`vertical.py:490 vertical_advection` is 1st-order upwind on
`dsigma_full = diff(sigma_full)`; sigma_dot ~ omega/p_s. Measured
CFL = |sigma_dot|*dt/min(dsigma_full): dt=75, refine=3, omega=5 Pa/s
(extreme for a ~500 km cell) -> 0.264; refine=6 -> 0.392; dt=60 -> 0.211.
`nu_vert4_T` is index-based so its limit (16*nu*dt = 2.4e-3) does not
change with refinement at all. Both GWD schemes cap acceleration by
`min(umcfac*U/dt, tndmax)` which is dp-INDEPENDENT (hines.py:283-297,
mcfarlane.py:544-552). CHALLENGE: name ANY term in the MPAS sigma path or
in the physics (bechtold convection, hard-saturation adjustment +
ice-curve, mpas qv del2 smoother, turbulence/PBL, radiation, microphysics
sedimentation) whose stability limit scales like 1/dsigma or 1/dp. If you
find one, quantify it at dsigma_min = 0.0144, dt = 75 s.

F6 CONFIRMED. `vertical.py:262-264` claims the refinement's "side benefit:
the top layer THICKENS ... the opposite of the thin-top configuration that
blew L40 up at day 46." But L40 was UNIFORM sigma with 24.4 hPa layers
EVERYWHERE, and refine=3 puts 14.2-17.9 hPa layers at 99-175 hPa — inside
L40's 10-302 hPa failure band and 42% thinner than anything L40 had. The
claim covers only k=0. Defensible as written?

=== ALSO HUNT (grep, do not assume) ===
1. Any OTHER uniform-sigma assumption: unweighted level averages/means,
   `dsigma[0]`, hardcoded layer thickness, plev interpolation, the CMOR
   feed, `mpas_qv_smooth_del2_m2s`, the hard-saturation adjustment / ice
   curve drain, ice-skin, land boundary, GWD launch-level selection,
   radiation layer construction, bechtold entrainment / cloud-top search.
2. `create_sigma_coordinate` never receives `GridConfig.p_top_Pa`
   (`model_driver.py:1002-1005`); sigma_top is hardcoded 0.01. Real defect?
   Does the new validate_strict message mislead by implying sigma is fully
   specified by `tropopause_refine`?
3. Differentiability/JIT: `tropopause_refined_sigma_half` is host-side
   numpy run once at setup. Confirm no `jax.grad`/`jit`/`vmap` breakage and
   no path where `tropopause_refine` becomes traced.
4. Test quality. I mutation-tested the density (flat / Gaussian-in-sigma
   instead of log-sigma / inverted) and the suite catches all three, but
   `test_top_layers_not_thinned` (`ref[0] >= uni[0]`) passes trivially on
   the "flat" mutant. What else is weak or vacuous? What test is MISSING?
5. Is equidistribution in SIGMA with a density that is Gaussian in LOG
   SIGMA mathematically coherent, and does the resulting distribution match
   what the docstring claims?
6. Construction numerics: I measure max placement error 1.1e-8 in sigma vs
   a 4e6-point reference, and 0 non-monotone cases over
   n in {10,20,30,40,60,80,137} x refine in {1.0001,1.5,3,6} x width in
   {0.02,0.05,0.1,0.45,1,3} x sigma_refine in {0.0101..0.999} x sigma_top
   in {1e-5,1e-3,1e-2}, in float64 AND after the float32 storage cast.
   Find a breaking combination if one exists.
