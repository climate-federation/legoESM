# T29: height-coordinate acoustic tracer mass, phase 0 preregistration

Status: preregistration, no model edits, no baseline run. Phase-0 probe must
receive GLM full-diff review and Claude SHIP, then be committed before running.
No push, PR, issue edit or merge. Phase 1 requires a separate full-diff gate.

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

Phase 1 remains unimplemented: reference method uses acoustic-averaged
continuity fluxes for once-per-large-step m*q transport at final RK stage,
all applicable variants, remove duplicated slow vertical transport, preserve
existing face selectors. Unsupported core/variant design: STOP with evidence.
Separate fix preregistration, variant/AD/JIT/parity/reversion tests and required
ratchets follow measured baseline. No phase-0 number claimed before controls
and reproduction pass on the committed instrument.
