# SI3 lane 3 — rung 3.3 stress replay and trajectory preregistration

Issue: climate-federation/legoESM #1699

Oracle root:
`/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv2d_rhg/final`

This measurement plan is committed before the rung-3.3 stress replay, any
candidate step after completed step 1, and the final candidate restart score.
The immutable normalized pointwise bar remains `1e-15`; the replay ownership
discriminator is at most two ULP at each compared assignment.

## Stress-debt discriminator

The independent replay starts from the oracle `kt=1` entry frame: U/V are the
STEP_ENTRY_CURRENT values and `stress1_i`, `stress2_i`, and `stress12_i` are
the CARRIED_PREVIOUS_STEP values registered by `icestp.F90:154-171`.  Tracer
mass/concentration comes from the same frame.  The replay transcribes the
executed source statements in `icedyn_rhg_evp.F90:189-741` using scalar NumPy
fp64 operations and the exact NEMO loop extents; it does not call the legoESM
stress/deformation/divergence helpers.

At subcycle 1 it checkpoints, in order: F shear (`:392-399`); T shear-square,
divergence, tension, delta and P/delta (`:401-427`); recomputed divergence and
tension, adaptive alpha, T/T stresses and beta (`:430-470`); F alpha,
P/delta and stress12 (`:472-491`); stress divergence and cross velocities
(`:495-516`); and the odd U-then-V velocity pair plus halo (`:638-741`).  It
reports maximum ULP distance for each checkpoint against the production JAX
arm run for exactly one subcycle from those identical inputs.  It then runs
the same replay for 100 subcycles and compares the five carries against both
the production candidate and oracle entry `kt=2`.

Preregistered classification:

* **RE-ASSOCIATION** only if every one-subcycle stress checkpoint differs from
  the source-ordered replay by at most two ULP and the 100-subcycle replay's
  oracle stress residual is in the already measured `1e-14` class.  The
  report must give the per-stress one-subcycle ULP counts and the 100-cycle
  normalized errors.
* **IMPLEMENTATION OWNER** if a one-subcycle checkpoint exceeds two ULP.  Stop
  at the first operand in the source order above and name its array, cell,
  source statement, absolute/normalized error, and ULP distance.  Do not run
  the trajectory before that discrepancy is resolved or explicitly retained
  as DEBT.
* **UNMEASURED** if an independent replay cannot be made to share exact input
  bytes, loop extents, or periodic-halo mapping with the production arm.

The velocity explanation is a separate measured calculation.  At the
maximizing stress-error cell the report applies the exact `:495-510` discrete
divergence to `(candidate stress - oracle stress)`, then propagates that force
through the active `:667-668`/`:719-720` momentum denominator.  It prints the
predicted velocity perturbation and the observed velocity error.  A small
stress-relative error alone is not accepted as an explanation.

## Full trajectory and restart

Only after the replay classification, advance the single fp64 card from cold
entry through all 485 completed steps.  Compare each candidate completed step
`n` against oracle entry frame `kt=n+1` for `n=1..484`; compare completed step
485 against the final ice restart.  The first row above `1e-15`, in registry
order U, V, stress1, stress2, stress12, then transported tracers, is the fixed
first-divergence result.  The gate reports the requested growth rows at
completed steps 1, 10, 50, 100, 200, and 485 for `u_ice`, `v_ice`, all three
stresses, `a_i`, `v_i`, and `v_s`.

Final restart coverage is discovery-driven.  Every prognostic ordinary field,
all three EVP stresses, and all five moments for each of the 16 active Prather
families (80 moment arrays) must be scored or cause a hard failure.  The
restart loader must also demonstrate split continuation and red controls for
a missing, retyped, and perturbed stress and moment.  `snwice_mass` and its
before level retain their existing explicit disposition; they may not vanish
from the registry.

## Documented phenomenology

The shipped `tests/ICE_ADV2D/EXPREF/README:48-55,64-66` documents four
qualitative observations relevant to this resolved card: a square
concentration and Gaussian volume are advected; Prather conserves maxima but
creates side lobes; rheology calculates velocity under constant ice-atmosphere
stress; and ice below the 1 kg m-2 mass threshold moves at the zero ocean
velocity.  The README gives no tolerance or quantitative side-lobe definition.

Therefore the gate will report, for each model separately, the concentration
and volume maximum series, negative/local-extremum census, moving/nonmoving
mass census, and velocity range.  Maximum conservation and side-lobe
documentation conformance remain **MEASURED-UNCLASSIFIED** unless the README
itself supplies a numerical predicate; the mass-threshold statement is checked
exactly against the source predicate at `icedyn_rhg_evp.F90:319-329,575-579,
626-630,681-685,733-737`.  No endpoint-equality predicate will be revived.

## End-of-task ASKED / UNASKED choice list

**ASKED:** independent NEMO-written-order stress replay; two-ULP discriminator;
arithmetic explanation for stress-versus-velocity labels; all 485 boundaries;
specified growth table; discovery-driven final restart including 80 moments
and three stresses; shipped README phenomenology on both models; fp64 CPU;
binding controls; honest labels; explicit-path commits and bundle; no push and
no shipped-NEMO modification.

**UNASKED:** no tolerance change; no landfast, multi-category, thermodynamics,
ridging/rafting, coupled-ocean, or nonzero-Coriolis card; no analytic oracle;
no claim that a qualitative README sentence is numerically confirmed without
a documented predicate; no attribution from magnitude alone.
