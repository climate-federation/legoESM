# T30 active port protocol (2026-11-25)

T30 ports only spectral NH paired mass transport onto main d575b68710's existing
TracerMassHooks. Main already implements MPAS/LES conservative stages; do not
port the T29 shared adapter or MPAS/LES changes. The sections below are T29
history and measurement provenance, including superseded routes. T30's task
and Claude's carry tiebreak supersede their commit/runtime ordering: STATIC
ONLY here, no PBS/pytest/JAX imports. Claude reviews the final port before any
commit; the user runs runtime checks on another HPC. Current-branch numerical
results, drift, AD/JIT, reversion and byte equality remain UNVERIFIED.

The flat/nonflat inventory and constant-drift fixtures and numerical bars below
remain unchanged. Main's original T21/20-step defect fixture must also conserve
at <=1e-12. Reuse existing census provenance and planted violations; preserve
main's ocean momentum operator selections. No new ocean or MPAS/LES method.

T30 exact pre-code votes D5-D8: Codex YES + GLM YES, after Claude's independent
mass carry / pure face-rename tiebreak. Optional mass/with_mass callbacks retain
the default grid arithmetic. The spectral-local (zero-q state, mass_hat) carry
uses existing RK/finish_stage; no parallel integrator. Evaluation-state faces
are recomputed in finish_stage: instrumentation expects 3 stages, 2 blends and
10 recoveries (3 slow, 3 acoustic-base, 3 face-evaluation, 1 final), replacing
the historical adapter's 7. All scientific fixtures/tolerances are retained;
no current-snapshot runtime or bitwise-trajectory claim follows from algebra.

---

# T29: height-coordinate acoustic tracer mass, phase 0 preregistration

Status: phase 0 completed at ccd817f21c (artifacts in
`/glade/derecho/scratch/pg2328/t29_phase0/retry_import`); MPAS/LES phase 1
completed locally at 52b658b634 after full-diff GLM/Claude SHIP and PBS checks
(receipts in `/glade/derecho/scratch/pg2328/t29_phase1`). Spectral quadrature
instrument committed at 3dfbd2623f and measured on PBS. Option 1 completed
locally at b80cb95d65, preserving nonflat legacy transport. The later
2026-11-23 USER terrain-adoption decision at the end of this document
supersedes the earlier flat-only disposition. The new snapshot awaits
full-diff SHIP and PBS validation; prior passes are historical.
No push, PR, issue edit or merge.

## Claims and distinct statistics

Extend `diag_hybrid_tracer_smoothing.py --all-vertical-paths`; preserve the
existing ocean rows and paired-column function. Port only the T24 height
section from `wt_latlon_tracer` commit `4cd050d083`, original lines 598–690,
with its original seeded-column and MPAS preamble. No model changes.

1. **Reproduction:** the old per-column paired product-rule statistic is
   `abs(sum(m*dq+q*dm))/sum(abs(m*dq)+abs(q*dm))`, maximum over columns.
   Finite steps use the exact layer product increment as `dq`, `m=1, dm=0`.
   It is a gross tendency/increment residual, never an inventory-loss fraction.
   Confirm iff values round to issue #2641 at two significant digits:

   | Original explicit fixture | Slow | Acoustic only | Full step |
   |---|---:|---:|---:|
   | MPAS NH | .026 | .45 | .033 |
   | Spectral NH | .049 | .45 | .058 |
   | LES centered / van Leer | .029 / .0017 | .085 | .024 |

   LES direct serial/local-halo operators and native slow callers must reproduce
   their corresponding slow values. Local halo rows certify no full halo step.
   Any mismatch refutes reproduction: STOP, retain artifacts, no physics fix.
2. **New full-step budget:** per tracer, `R=sum(V*rho_new*q_new) -
   sum(V*rho_old*q_old) - B`, `relative=abs(R)/sum(abs(V*rho_old*q_old))`.
   `V=area*J*dz`; native `mesh.areaCell`, `GaussianGrid.grid_area`,
   `PlaneGrid.grid_area`, with no uniform-grid proxy. Height indexing is
   top-to-bottom, thickness positive; boundary content `B` positive into domain.
   Closed lids and sphere/periodic plane give `B=0`. Assert input/output lids.
   Inventory bar: **CONSERVES iff relative <= 1e-12**, otherwise **LEAKS**.
   Signed R is retained. Pre-fix model rows report classifications, not asserts.
   New budgets need not numerically equal old gross-increment residuals.
3. **Controls:** each initial dataset/tracer gets identity (expected R=0),
   a conservative internal transfer (expected R=0), and the same amount added
   only at receiver (expected R=+A). Donor is first largest positive content in
   flattened order, receiver first largest remaining positive content, distinct.
   `A=.01*(V*rho*q)_donor`; updates are `-A/(V*rho)_donor` and
   `+A/(V*rho)_receiver`. Assert `abs(R-expected)/initial_abs_inventory<=1e-12`
   for each control; planted source must exceed the bar and classify LEAKS.
   Reject zero/nonfinite content, unresolvable sources, fewer than two positive
   cells, fp32, incompatible/empty shapes, nonfinite arrays and nonpositive V/rho.
   No denominator floor, alternate fixture, dropped cell or fp32 conversion.

## Locked fixtures and provenance

Exact T24 baseline: `_make_mesh(level=2)` (five Lloyd iterations),
`_column(seed=0,ncol=6)`; MPAS 10 height levels/H=30000 m, profile
`q[0,:10]+q[0,20:30]`, w=sin(linspace(0,pi,11)), final lid set to zero.
Gaussian truncation 10, same height profiles; inventory q/rho reconstructed
from actual initial spectral state, as for final state. LES existing
`_setup_with_tracers(1)` gives 8x8x4, 200 m spacing, H=2000 m, passive,
explicit, no mass fixer/coriolis/moist buoyancy; q=`q[0,22:26]`, w sine over
five interfaces with final lid zero. Exactly one full step, dt=1 s, all cores.
Defaults are inherited unchanged for exact reproduction and saved in full
resolved config dictionaries; this is not a production-default decision.

Save initial/final rho/q, native area/J/dz/V, states and geometry arrays in
`height_inputs.npz`, member shapes/dtypes plus whole-file SHA256 in provenance.
Keep ocean `inputs.npz` intact. Stamp git SHA, script/model/helper/fixture
paths+SHA256, dt, precision policy fp64, CPU, PBS_JOBID, lane and completion.
Partial output stays incomplete on failure. Old statistic and new global
inventory statistic have distinct row labels; verdicts are per tracer only.

## Genuine MPI lane and planned execution

All runtime (including tests) follows SHIP+commit, on PBS CPU,
`develop@desched1`, account `UCLB0017`, submitted through `ssh derecho2`.
With `JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu`, from this worktree:

```sh
python scripts/validate/diag_hybrid_tracer_smoothing.py --all-vertical-paths --out /glade/derecho/scratch/pg2328/t29_phase0/serial
mpiexec -n 2 python scripts/validate/diag_hybrid_tracer_smoothing.py --all-vertical-paths --height-halo-mpi --out /glade/derecho/scratch/pg2328/t29_phase0/halo_mpi
python -m pytest tests/unit/test_diag_hybrid_tracer_smoothing.py
```

Explicitly distinct output paths; old overwrite semantics retained, no CLI
claim to prevent directory collisions. MPI lane measures only genuine halo
full step with `(n_ranks_y,n_ranks_x)=(2,1)`, local 4x8 owned cells, halo from
existing `horizontal_advection_halo_width`. Exactly two ranks required before
JAX/mpi4jax/collectives; any lane exception prints traceback and calls Abort(1).
Single-rank serial delegation is not measured. No pytest skip for wrong size.

Each rank builds identical tiny deterministic T24 global fixture and slices
all seven fields using `scatter_plane_field`. Record and compare per-rank
SHA256 of contiguous NumPy bytes for every global initial field before stepping.
Root mismatch reaches lane-level Abort, so peers cannot hang in later gathers.
Each rank uses existing local grid/model/flat metric, unchanged hc/config;
`step_halo` runs once. Gather actual initial/final rho/q with existing helper;
root computes global budgets and controls, writes artifacts, and records each
rank's layout/source/fixture hashes/PBS/precision. No root-array scatter protocol.
MPI runtime incompatibility: STOP, no substitute fixture or serial fallback.
Serial-vs-halo parity tolerance belongs to phase 1; no extra phase-0 threshold.

Direct tests use fp64 V=[2,3,5], rho=[1,2,4], two q columns
[.001,.002,.004] and [.004,.002,.001]. Include signed positive/negative sources,
explicit incoming-boundary closure, unequal-volume conservative transfer and
intentionally wrong equal-delta-q transfer (donor2, receiver1, delta_q=.01*q[2])
that must classify LEAKS with independently signed expected content change.
Reject invalid inputs as above. Controls fail hard before any model claim.

## Phase 1 corrected design and pre-runtime registration (2026-11-23)

The authoritative corrected reference is WRF Fig 3.1 step 7: in EVERY existing
selected dynamics RK stage, after its acoustic loop, advance rho*q with that
stage's exact time-integrated continuity mass flux, then recover q with the
resulting actual rho. Not one Euler update per large step and not per acoustic
substep. Existing SSP3/34/54 density and tracer-mass leaves use identical
coefficients. No generic integrator coefficient or production config changes.
Spectral remains operational HOLD3-0 until its separate quadrature measurement;
no spectral code or claim in this implementation.

Predictions BEFORE runtime: MPAS explicit and LES explicit/SI/SI-horizontal,
serial and genuine 2-rank halo, have each closed-domain full-step tracer inventory
residual <=1e-12 relative initial absolute inventory (each tracer separately).
Greater residual REFUTES closure; no tolerance relaxation. Constant q=0.01 stays
constant to absolute1e-12; JIT/eager and halo/serial rho/q match rtol/atol1e-12.
Only source-free/fixer-free budgets carry these invariant claims: existing SGS
and physics tendencies retain their evaluation-state q-source semantics, using
ACTUAL rho (SD1 decided2-1); density sponge/hyperdiff and post-step fixer keep
existing order and are outside this source-free invariant claim.

For SSP54 composite RHS, reconstructions alone divide by the carried sum of RHS
weights; mass/source tendencies retain their b weights. Reconstructions come
from the RHS evaluation q, not the independently blended RK base. All transient
reconstruction/normalizer/scratch leaves reset to zero at stage boundaries.
Vertical/horizontal exports include (1+beta) wherever continuity does, and export
before subsequent moist-buoyancy changes to returned w. Test nonzero beta0.1.
MPAS native centred reconstruction has no vertical selector; LES retains both
centered/van_leer vertical and upwind1/centered/van_leer horizontal, with existing
weno5 tracer->van_leer mapping. No new smoothing, floor, clamp or limiter.

Frozen phase-0 census geometry/profiles/dt1 remain unchanged. Retire ONLY the two
pre-fix MPAS/LES full-step paired reproduction targets; report those rows without
pretending their per-column statistic is a global budget. Keep the other14
legacy reproduction checks, original ocean rows, spectral rows, all12 inventory
controls, original provenance. Global inventory acceptance now asserts the fixed
MPAS/LES serial and genuine-MPI paths CONSERVE at the existing1e-12 bar.

Tests add nonuniform rho/q/horizontal winds to the exact T24 passive fixtures;
separately reuse existing stretched geometry4x4x20,H20000,dz_sfc50,dxdy2000,w0.5.
Mass/constant matrices span every selected RK and acoustic mode and both LES
vertical selectors. Frozen horizontally uniform rho/u periodic smooth-q transport
must match the selected existing RK applied to the SAME discrete operator at
rtol/atol1e-12, except SSP54/SI-horizontal as disposed below. Planted Euler must
differ by >1e-12; if insensitive, STOP rather than silently changing dt. This is
a discrete update check, not an assertion about the vertical acoustics' temporal
order. There is no frozen/source-corrected MPAS polynomial surrogate (F2). Source-only sub-floor actual rho
1e-10, nonzero S0.01, identity dynamics must produce q+dt*S for every RK.
FP32 gradients through nonzero q and w must be finite AND nonzero; fp64 finite
 differences reuse existing NH check_grads tolerances1e-3/1e-6 where affordable.

All runs PBS CPU develop@desched1,UCLB0017 after full-diff GLM+Claude SHIP. Required
checks: touched MPAS NH and LES unit suites, genuine2-rank closure/parity matrix,
census serial+MPI, named task ratchets. Run only full-step closure test selections
against phase0 committed code in separate temporary checkout to show reversion
FAILS; source-only/new-helper ImportError is NOT counted as non-vacuity evidence.
Record actual summary lines and source hashes. No test/model execution has yet
occurred on this phase1 diff; this registration does not claim a passing result.


## Claude SSP54/SI-horizontal disposition, before any phase-1 runtime

KEEP normalized final reconstruction; no held MPAS/LES variant. For uniform
rho/u periodic centered advection, first four independent scalar SSP54 stages
supply q2/q3/q4. The expected final tracer is
`a52*q2+a53*q3+a54*q4 + dt*L((b53*q3+b54*q4)/B)`, B=b53+b54.
Read coefficients from the existing SSP54 table, not a copied new table.
This must pass the same1e-12 comparison and reject single-evaluation Euler.
All other RK/acoustic combinations retain the ordinary scalar SSP comparison.
The existing DYNAMICS advances the last acoustic loop over dt with B-weighted
slow forcing; tracer follows that same continuity flux, preserving the pairing.
This source-confirmed discrepancy from ordinary SSP54 is a separate follow-up,
not fixed here. Its size/order in nonuniform density is UNMEASURED/PLAUSIBLE;
the constant-density uniform-flow fixture cannot quantify that density error.

F1 genuine-MPI fixture adds y-face v=cos(2*pi*y/ny), rho_prime increment
0.01*rho_ref*cos(2*pi*y/ny), and q factor(1+0.1*sin(2*pi*y/ny)) to the same
T24/E3 fixture. Decompose exactly2x1 and assert nonzero seam v/tracer mass.
Matrix all3 RK/all3 acoustics/both vertical selectors, dt1. For each case,
measure nonconstant inventory and constant-q inventory plus serial rho/q parity
with the already approved bars. Exceptions Abort(1) rather than hanging peers.
FP32 nonzero q/w gradients span all LES acoustic modes/both vertical selectors;
cheap fp64 reverse check_grads reuses existing1e-3/1e-6 tolerances, no new bar.


Full-diff review hardening (GLM Vote A): acceptance requires at least one
matching full-step global-inventory/full-domain row for EACH fixed path, and
nonempty per-tracer verdicts all CONSERVES; neither absent rows nor an empty
requested-path set can pass. Serial/MPI use the same helper. Planted controls
remove/mislabel a row, insert one leaking tracer, or empty the verdict/path list;
each must fail. MPI exceptions print traceback, Abort(1), then re-raise; the
original handler already makes failure fatal, never a swallowed success.

## Spectral horizontal quadrature instrument, before measurement

Votes Q1-Q3 YES3-0 (Codex/GLM/Claude); Q4 YES Codex/GLM, Claude pending.
This additive instrument changes no model, config, CLI, legacy census fixture,
baseline target or acceptance threshold. Commit only after complete-diff GLM
and Claude SHIP; then run existing `--all-vertical-paths` on PBS CPU fp64.

Measure horizontal continuity only on the SAME Gaussian truncation10 grid,
10-level height coordinate and flat terrain used by the existing T24 census.
The exact census initial state has zero horizontal winds: label it
`census_identity`, require exact-zero norms and report ZERO_UNSCORED with
relative=null. It cannot support a nonvacuous quadrature conclusion.
Separately use the unmodified existing
`TestAliasingStabilityNH()._rough_nh_state(grid,hc,n_tracers=1,seed=1)` from
`tests/unit/test_spectral_dealiasing.py`, labelled `supplemental_rough_seed1`.
Pin that fixture source and store its complete initial state, grid/metric,
native areas, density, velocities, actual/expected tendencies and divergence
in the existing hashed height_inputs artifact pipeline; retain resolved config.

Evaluate actual `spectral_nh_slow_tendencies` with unchanged SpectralNHConfig
and physics_tendency=None. Read current source: horizontal density continuity
is flux form; sponge/hyperdiffusion do not change density; final de-aliasing
retains n=0. Independently call existing `vordiv_from_uv_3d` on J*rho*u/v,
using the model's sanitized rho and exact velocity/cosine clipping convention.
Compare actual density tendency to canonical negative divergence/J projected
back through analysis, the existing de-aliasing mask and synthesis, at existing
rtol/atol1e-12. No copied divergence formula or claim of bitwise identity.

For EACH level, record signed sum(area*J*actual_drho), its denominator
sum(area*abs(J*actual_drho)), and their absolute ratio; also record canonical
raw sum(area*div), sum(area*abs(div)), ratio, and complex n=m=0 coefficient.
Use native Gaussian grid_area, no cosine proxy, denominator floor or vertical
averaging. Supplemental norms must be finite and strictly positive at EVERY
level. CONFIRMS horizontal quadrature iff every actual/canonical relative
residual <=1e-12 and the continuity comparison passes; otherwise REFUTES and
STOP with partial artifacts. Zero identity is never scored as AT_BAR.

Control on each nonzero supplemental divergence level: add spatial constant
0.01*max(abs(div)); verify seed amplitude finite/nonzero before perturbation.
The SAME predicate must classify EVERY planted level DEBT (>1e-12), and the
signed added integral must equal offset*sum(area) at rtol1e-12, atol0.
Direct helper tests use unequal area[[2,3,5]], field[[[1,2],[1,2],[-1,-2]]]:
both weighted sums zero, unweighted sums nonzero; constant perturbation fails.
Exact-zero norms return relative=null; invalid shape/fp32/nonfinite/nonpositive
area are rejected. No empirical quadrature result yet. This is not finite-step
rho*q closure, aliasing accuracy or spectral-method SHIP.

## Spectral phase-1 OPTION1, pre-runtime registration (2026-11-23)

Claude `claude_answer_specdesign.md` selected OPTION1: transport M=J*rho*q
with the SAME native projection P=S*A as density, recover q_hat=A(PM/(J*rho)),
keep the public q_hat state. P1-P6 and amendments P7-P11 have Codex+GLM YES;
full-diff GLM and Claude SHIP are still required. No new prognostic state,
production config, RK coefficients, limiter, floor or inventory correction.
Native horizontal quadrature prerequisite already passed at instrument commit
3dfbd2623f: `/glade/derecho/scratch/pg2328/t29_spectral_instrument/measurement_receipt.json`.
That result certifies horizontal quadrature, not this finite-step fix.

The shared split-explicit adapter uses optional representation callbacks;
existing grid callers retain their arithmetic. Spectral transient mass is SH
coefficients A(J*rho*S(q_hat)), so mass and density use identical SSP3 blends.
The reconstruction normalizer is REAL even when mass coefficients are complex.
Each RK stage transports q with the pre-mask horizontal continuity fluxes and
its exact time-integrated vertical acoustic rho*w flux, then recovers the
existing public q coefficients. Both explicit and semi-implicit acoustic
variants export the exact flux used by their backward density update. Flux is
positive upward on top-to-bottom levels, giving bottom minus top in mass RHS.
Use density continuity's exact native oc2/dmu divergence (A1 amendment,
Codex+GLM YES; Claude required) and the existing horizontal tendency mask; vertical
mass and q recovery use full native P, not an extra tendency mask. No spectral
vertical selector exists; reuse centered height faces. Standalone old slow and
acoustic diagnostics retain their defaults; the active paired model suppresses
all old slow tracer advection. Sources use ACTUAL evaluation-state density;
source/fixer-free invariant scope and the anchored fixer ordering stay unchanged.

KNOWN DIFFERENCE: final SH q projection generally loses pointwise/local mass
agreement. Certify GLOBAL inventory <=1e-12 relative per tracer and constant
q=0.01 to absolute1e-12 on flat J. Never claim local equality or a nonflat-J
invariant from that flat result. Public rest initializer still allocates a zero
dummy tracer for n_tracers=0; do not change its state. Truly empty tracer axes
retain the original integrator path. Check empty-axis bytes and zero-dummy
nontracer-field bytes against pre-fix3dfbd2623f; no state-dependent zero switch.

Tests and supplemental census reuse the existing rough NH seed1 fixture on the
exact census Gaussian truncation10 / hc10 / H30000 grid. Add0.01 to synthesized
q before analysis and require positivity; pack q and2q. Add the original census
sine w with exact zero lids; require nonzero horizontal density variation,
vor/div and w. Config explicitly sets g=constants.g, hyperdiff0/order2,
sponge0/width10000, n_acoustic_substeps6, small_earth1, rotatingTrue,
semi_implicitFalse/True, dealiasing0.667, fix_massFalse, anchorFalse; dt1.
Measure actual represented initial/final rho and q with native area*J*dz.
Reuse existing global_inventory_budget and controls, no surrogate mass statistic.
Keep the original census fixture and resolved default config unchanged; it now
also must CONSERVE. Retire ONLY its superseded spectral full-step legacy5.8e-2
target; retain the old standalone spectral slow/acoustic reproduction targets.

Test native orthogonality per level/tracer:
`sum(area*f*P(g)) = sum(area*P(f)*P(g))`, with g=J*rho*q, f=actual rho AND
f=J*rho*q0 (require maxabs(f-Pf)>1e-12 for the second case). Normalize by
sum(area*abs(f*Pg)), strictly positive before division; require <=1e-12.
Plant +0.01*maxabs(Pg) in ONLY lhs Pg; amplitude must be nonzero and the SAME
predicate must fail at EVERY level/tracer. No aggregate cancellation verdict.

Observe the actual shared helper through unchanged SSP3: all3 acoustic stage
mass budgets, both existing RK blends, and all7 native mass-to-q recoveries
must close1e-12 on flat J. Check stage/blend recon/normalizer reset to zero.
Retain mass leaves and signed per-tracer budgets in the test observations;
print signed stage/blend/recovery inventory pairs in test output. Instrument
module-local split_explicit aliases before invoking the helper, under disable_jit
for Python observations only. Separately require JIT/eager and _step_on_cpu vs
_step_jit rho/q parity at rtol/atol1e-12 for both acoustic variants. Verify raw
acoustic density increment from exported flux at rtol/atol1e-12, exact zero lid
flux, nonzero interior flux. Reversion selects ONLY full-step inventory tests
against3dfbd2623f plus the reviewed test file; each must fail the INVENTORY
assertion, never count imports/helper absence as evidence.

Native spectral transforms retain fp64/complex128 as required by the existing
backend contract. Test finite/nonzero FP32 INPUT gradients in q/w amplitudes
through the actual new full step; do not call that fully-fp32 spectral arithmetic.
Both variants also get cheap scalar FP64 check_grads order1/reverse using the
existing1e-3/1e-6 tolerances and squared synthesized-q loss. No backend or
precision-policy default changes. Unsupported gradient behavior is a gate.

Nonflat J: the exact same supplemental state/grid/hc changes ONLY the metric to
that returned by existing dcmip25_tc2_init_spectral(grid,n_levels10,subcase'a').
Label 'supplemental_TC2a_metric', not a TC2a integration; save source parameters,
metric, full initial/final states, resolved configs and actual inventory arrays.
Require finite J>0, maxabs(J-1)>1e-12 and ptp(J)>1e-12; never substitute a fixture
if that guard fails. This supersedes the initially proposed low-resolution TC1
metric before any implementation/run; the TC1 lane remains in the report.
Historical nonflat gate (superseded by the user disposition below): measure
EACH tracer at the SAME1e-12 bar, both acoustic variants subject to immediate
STOP on the first failure. Persist each budget and arrays BEFORE its assertion.
That gate fired on the draft explicit case; the subsequent authorized old/new
comparison measured both acoustic variants. It did not authorize a correction.
The user then selected the flat-only option below; preserve that failure as
evidence, not as an acceptance assertion for the unchanged legacy terrain path.

Static results-change/terrain inventory: matrix spectral TC1 mountain and TC2a
gap mountain; bench_spectral_nh TC1 mountain (existing flat fallback on initializer
exception, unchanged); supplied nonflat metrics in the component factory. The
spectral-vs-C-grid comparison benchmark uses flat terrain. Matrix spectral TC3
is explicitly skipped before initialization for missing microphysics callback;
do not list it as an active terrain integration. Factory defaults to flat if no
metric supplied. Initializers TC1/TC2 carry a zero dummy tracer; field impact is
not measured. Existing post-step dry-mass fixers remain outside closure claims.

After exact full-diff GLM+Claude SHIP, run on the authorized PBS CPU protocol:
spectral unit/dealiasing suites, census/direct helper tests, touched shared-helper
MPAS/LES regressions and genuine2-rank matrix, task ratchets, spectral reversion,
tracer-free cross-snapshot byte comparisons with planted controls. Keep the
already confirmed exact pre-existing hydrostatic-MPAS deselection explicit;
new model failures/disagreements/nonflat debt require fresh ask_claude and STOP.
Local fix commit only after runtime evidence and required final-diff SHIP.
No spectral fix runtime evidence exists at this registration.

A1 source qualification: physics mass sources are re-projected and masked;
source-only q+dt*S holds up to projection, preserving the global (n=0,m=0) mode.


## User option 1: exact flat dispatch; terrain debt deferred (2026-11-23)

This disposition supersedes only the historical nonflat conservation STOP above.
It keeps every conservation bar, fixture, control and flat acceptance predicate.
The paired spectral tracer-mass method runs only if J == 1 everywhere, checked
once by exact NumPy comparison at model construction and stored as a Python
bool. There is no new config/default or traced geometry branch. Nonflat J uses
the existing unfixed spectral tracer method; option 2 is a separate research
step. Empty tracer axes retain the legacy integrator on either geometry.

KNOWN DIFFERENCE / RULE 0 debt: on the SAME supplemental TC2a-metric fixture,
the legacy explicit relative residual was 6.9208461950594359e-7 and the legacy
semi-implicit residual was 6.9398988303003302e-7, each for q and 2q. Signed
residuals were [80486342064, 160972684128] and [80707915680, 161415831360].
The all-terrain draft instead gave 1.6751283622535413e-5 explicit and
1.6751283343384507e-5 semi-implicit. These are prior-snapshot measurements,
not results from the new geometry dispatch. Receipt:
`/glade/derecho/scratch/pg2328/t29_terrain_compare/comparison_receipt.json`.
The q/2q relative residuals are not independent evidence. No physical cause or
terrain-integration impact has been established. This is not a TC2a integration.

Keep BOTH nonflat acoustic census inventory rows, using the SAME statistic and
1e-12 bar and retaining their measured signed/relative residuals and verdicts.
Annotate them report_only_nonflat_legacy, method legacy, RULE 0 debt; option 2.
Flat supplemental rows are required_flat_conservation, method paired, and must
CONSERVE as before. Save arrays/rows before any acceptance assertion. All
instrument controls, quadrature, reproduction and other fixed-path inventory
assertions remain required. No assertion of nonflat conservation, no rescale,
no correction, no dropped terrain lane. Terrain lanes above are retained legacy
behavior; flat spectral-vs-C-grid and factory-flat lanes select the new method.
Numerical impact and current-snapshot byte equality still require measurement.

O1-O4: Codex YES + GLM YES, VOTED 2-0 + Claude pending; exact detail proposal and
vote are in the T29 scratch handoff. O2 dispatch tests patch the actual module
integrator aliases with distinct sentinels for both acoustics, flat/TC2a metrics,
and a single native-fp64 ULP departure from J=1. Call _split_step eagerly and
through fresh jax.jit closures; require exact Python bool and correct selection.
Invert the cached bool as a planted violation and require the SAME dispatch
assertion to fail in both modes. The ULP metric is selection-only, not integrated.

After full-diff GLM review and fresh Claude SHIP, run the existing PBS CPU checks
including reversion, full regressions/ratchets, and guarded byte comparisons.
O4 uses a separate detached HEAD3dfbd2623f baseline with only the reviewed test
file overlay, exact source/fixture/arm/import pins and PBS3.12 AST pins. Reuse
repository _same_bytes and the existing tracer-free collector/checker pattern;
review the exact adapted scratch source before execution. Compare ALL seven
SpectralNHState input/output leaves for nonflat packed positive q on BOTH
acoustics and step/_step_on_cpu, including shapes, dtypes and bytes. Require
identical inputs/configs/geometry; changed-model-hash, wrong-root, same-source,
and nonzero one-ULP payload controls must fire. Flat positive q must differ from
baseline in tracer output as a nonvacuity control. Truly empty tracer axes on
flat/nonflat require full-leaf equality; zero-dummy q requires all NONTRACER
leaves equal and exact zero q in each arm. No requirement of signed-zero byte
equality for the dummy q leaf. Current-snapshot shared-helper MPAS/LES byte and
regression checks are still required. Prior passes do not discharge new checks.
Any model failure/disagreement triggers the standing Claude gate; no local
commit until the required review/runtime checks pass. No outward actions.


## USER terrain adoption, 2026-11-23 (supersedes flat-only option 1 above)

The user chose projected-denominator recovery for NONFLAT spectral NH tracers
now. This replaces the earlier nonflat legacy/report-only instructions above;
those remain historical, not the active acceptance protocol. Nonempty tracers
use the paired RK mass path on both geometries. Exact flat J, detected once at
construction, retains the committed b80cb95d65 recovery expression and operation
order. Nonflat recovery is q_hat = A(P(M) / P(J*rho)), with the same full native
unmasked P = synthesis(analysis). No force-flat override, floor, clip, new
config/default, or changed source/fixer ordering. Empty tracer axes still need
the legacy integrator, so its branch is retained.

Historical research evidence (PBS8291236, native fp64, dt=1 s, source/fixer-free;
rough Earth-grid state with TC2a metric, NOT a TC2a integration): each q/2q
rough tracer inventory residual was 7.429348830111536e-15 explicit and
3.714674415055768e-15 semi-implicit; constant inventory residual
1.3999651841602016e-15 in both. Each constant tracer's max relative drift from
q0=0.01 was 9.59150033702505e-7 explicit and 9.553411842860848e-7 semi-implicit.
Committed flat controls were 1.6653345369377348e-14 and 1.7520707107365752e-14.
Minimum P(J*rho) at FINAL recovery: 0.046434189131528796 explicit and
0.04643384275006408 semi-implicit. Intermediate recovery positivity UNMEASURED.
Provenance: /glade/derecho/scratch/pg2328/t29_option2_measurement/measurement_receipt.json
and terrain/measurements.json, flat_control/measurements.json there. These are
research-snapshot measurements, not a pass for this production draft. Packed
q/2q and duplicated constant tracers are not independent fixtures.

KNOWN DIFFERENCE: uniform tracer is not preserved over terrain. PLAUSIBLE,
not isolated, cause: density's terrain update is not mass-consistent with the
tracer recovery. Follow-up cure specified by the user: store J*rho as spectral
density; Claude will file that issue. No long-time stability, general terrain,
full TC2a, source/fixer, or intermediate positivity certification is claimed.

Pre-registered current-snapshot checks (A1-A5, Codex YES + GLM YES each;
Claude pending; exact detail vote in T29 scratch t29_terrain_adoption):

- Both acoustics, both geometries: the existing rough and constant fixtures
  must CONSERVE each tracer at the unchanged global inventory bar <=1e-12.
  Nonflat census rows are required_nonflat_conservation with method
  paired_projected_density; retain measured signed/relative residuals and
  controls, save arrays and rows BEFORE acceptance. No hardcoded verdict.
- Terrain constant drift: assert each case/tracer at its full measured value
  above, rtol=1e-6, atol=1e-12 on the relative-drift statistic. Never a blanket
  upper bound. Flat constant assertion remains unchanged.
- Revert ONLY spectral production source to b80cb95d65 in a separate worktree
  with the reviewed test overlay. BOTH nonflat inventory tests must fail at
  the rough global-inventory assertion, not imports/config or a missing API.
- Extend existing JIT/eager/CPU parity, finite/nonzero fp32-input q/w gradients,
  and fp64 check_grads to both geometries, preserving all existing tolerances.
- Pin construction-time exact geometry bool on flat, TC2a and one-ULP metrics;
  nonempty tracers select paired, empty axes legacy, both acoustics. Swap
  actual integrator-alias sentinels as a planted violation, eager/fresh JIT.
- After full-diff GLM review and fresh Claude SHIP, run the option-1 PBS CPU
  suites, census, reversion, MPI and ratchets. Keep only the two exact
  main-reproduced deselections; model failure/disagreement triggers Claude STOP.
- Then-current guarded spectral byte adapters reuse the existing 24-case
  capture/pins/comparator, BASE b80cb95d65 plus the reviewed test overlay.
  Identical inputs/configs/geometry; all seven flat output leaves byte-equal,
  including positive and zero tracers; nonflat empty all seven byte-equal;
  all nontracer outputs byte-equal; nonflat positive tracer output MUST change;
  nonflat zero q numerical==0 only (no signed-zero byte claim there).
  Wrong-root, same-source and nonzero one-ULP controls must fire. Generate AST
  pins under PBS Python3.12 from frozen hash-checked source; baseline/candidate
  fixture ASTs equal. Exact adapter source needs Codex+GLM review before run.
- MPAS/LES shared helper source must remain byte-identical to b80cb95d65, with
  its SHA recorded before the full-diff gate, and option-1 suites/ratchets rerun.
  A5's omission of new MPAS/LES byte captures applies ONLY while shared sources
  are unchanged; any shared-source change requires their full adapters again.

RESULTS CHANGE: both acoustic variants for nonempty spectral tracers on
nonflat terrain: matrix TC1 mountains and TC2a gap mountain; spectral NH
benchmark TC1 mountain (existing initialization-error fallback is flat);
component factory when supplied a nonflat metric. Matrix spectral TC3 remains
skipped before initialization; spectral-vs-C-grid benchmark and factory default
are flat. These are paths reached if tracers are populated; this is not a claim
that each runner initializes tracers or has been integrated here. Flat recovery
unchanged in source; empirical byte equality still required.

Full-diff Claude SHIP is required BEFORE runtime, and commit only after the
required checks pass. No push, PR, issue edit or merge. All current-snapshot
runtime evidence remains UNVERIFIED until those checks actually run.
