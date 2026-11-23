# T29: height-coordinate acoustic tracer mass, phase 0 preregistration

Status: phase 0 completed at ccd817f21c (artifacts in
`/glade/derecho/scratch/pg2328/t29_phase0/retry_import`); MPAS/LES phase 1
completed locally at 52b658b634 after full-diff GLM/Claude SHIP and PBS checks
(receipts in `/glade/derecho/scratch/pg2328/t29_phase1`). Spectral quadrature
instrument below awaits full-diff SHIP, commit, then PBS measurement.
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
