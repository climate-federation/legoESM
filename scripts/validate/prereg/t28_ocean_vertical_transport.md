# T28 preregistration — before production fix

Issue #2642; base a87671b16c53bc2052c38ed1532fb5ddd40d65a1.
Ocean-only port of the committed PR2597 census on
fix/latlon-tracer-vertical-conservation, same pathname; paired_column_budget
is byte-identical. Existing atmospheric smoothing mode is preserved unchanged; atmospheric census modes
are not imported or certified.

## Frozen measurement and verdict

CPU PBS develop@desched1, UCLB0017, t2_env.sh with T21_W=this worktree;
PrecisionPolicy.fp64(). No integration, production configuration or knob change.
Measure each backend's explicit tracer binding (CD-grid and spectral), original
_column(32,seed=0), plus 32 layers geometrically spaced 5--300 m, T=2+18 exp(-z/500),
S=34+0.7(1-exp(-z/1000)), J=[0.97,1,1.03], w=1e-5*[1,-0.7,0.4]*sin(2*pi*k/32).
Closed lids, k increasing downward, w positive upward.
Use actual coordinate dz_ref after interface reconstruction, not input rounding.
Statistic per column: abs(sum(h*dq+q*dh))/sum(abs(h*dq)+abs(q*dh));
dh=diff(w), the vertical split of continuity only. No denominator floor.
CONFIRM fix: every explicit tracer row <=1e-12; REFUTE: any row >1e-12 or nonfinite.
Before fix expect LEAKS; do not assume the historical 0.111 value on new fixtures.
Positive control: original backward-Euler fixed-h flux solve, dt=600 s,
finite product increment must <=1e-12. Negative control: add 1e-3 K/s to surface
T tendency; must >1e-12. Empty/vacuous non-control records refuse a verdict.
Save producer SHA, source hashes, patch, CPU/precision/job and actual inputs.

## Implementation and checks (D1/D2/D4 A; D3 Claude B)

Reuse flux_form_vertical_tracer_advection for tracer and volume flux divergence.
h*dq=-div(w*q)+q*div(w); dh=-div(w). Partial-cell faces gated by active cells;
h=dz_ref*J or h_partial*J, existing thickness floor retained. Healthy layers only:
no exactness claim at the floor. Upwind selection retained, no new scheme knob.
Extract old body into public advective_vertical_momentum_ocean and preserve both
backends' momentum bitwise. Delete private spectral helper; explicit public
tracer/momentum callers; census spectral row measures actual tracer binding.
Test nlev=1, nonuniform z-star/J, partial cells, constant tracer, fp32 finite AD,
eager/JIT parity; show conservation tests fail on isolated original base.
Compare saved base/fixed momentum bitwise. No full-model, horizontal transport,
climate, NEMO fidelity, momentum conservation or stability claim from this budget.
