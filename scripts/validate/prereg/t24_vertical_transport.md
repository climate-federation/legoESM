# T24 vertical tracer transport census — preregistration

Before measurement: extend `diag_hybrid_tracer_smoothing.py` using
`--all-vertical-paths --out DIR`; preserve its original smoothing mode.
Run only PBS develop@desched1, UCLB0017, CPU, fp64 policy.

Instantaneous per-column budget: `sum(m*dq + q*dm)`, scaled by
`sum(abs(m*dq)+abs(q*dm))`. Pressure uses `m=dp`, `dm=-diff(F)`;
sigma uses the equivalent unit-surface-pressure weight `dsigma`.
Height uses `m=rho*J*dz`, with `dm` obtained from the native acoustic
density JVP at dt=0, tangent 1, one substep (including off-centering).
Closed lids and positive-up w. Acoustic-only finite updates are also scored.
This is a vertical subsystem diagnostic, not total continuity.

Finite updates use exact `m_new*q_new-m_old*q_old`, scaled by the sum
of absolute layer product increments. Report separately from instantaneous
budgets; these are finite changes, not an estimate of a semidiscrete operator.
Each column is scored separately; no horizontal cancellation or area average.
Zero numerator/denominator gives zero; nonfinite model output invalidates run.
CONSERVES iff maximum relative residual <= 1e-12; otherwise LEAKS.
Nonzero lid flux invalidates the closed-boundary fixture, not a leak verdict.

Known-conserving controls: native hybrid SB and smooth van Leer. Negative
control: native advective hybrid. Planted violation: add `0.01*q*rate` to
nonzero interior layer 24, where rate is each column's maximum interior
`abs(F_face)/min(dp_above,dp_below)`. All four must classify as registered.

Coverage: native pressure tracer callers on MPAS/latlon/spectral hybrid and
sigma, standalone prescribed-wind MPAS/latlon/spectral/cube sigma; MPAS and
spectral NH slow callers plus full split steps; LES serial caller/full step
and both serial/halo local vertical operators; actual FV3 Lagrangian remap
on its existing native fixture/window; explicit and implicit ocean vertical
operators and spectral ocean (report only). Native vmap traces are observed
using restored spies and live callbacks. Local halo columns make no MPI claim.
The table is incomplete if a listed route did not execute. A full height fix
requiring new slow/acoustic time coupling is a task-mandated design STOP.

User approved SB as the spectral production default; FV3 SB only if it leaks.
Other design/default decisions remain reserved to user. No ocean edits.
