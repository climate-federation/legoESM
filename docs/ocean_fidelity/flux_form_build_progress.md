# Flux-form momentum build — progress log

Spec: `docs/ocean_fidelity/flux_form_build_spec.md`. Branch: `matching_Veros_oracle`.
One dated entry per iteration: what changed, the exact gate command + result, commit hash,
every micro-decision. Newest at the bottom.

## Gate status
- [ ] F1 existing paths bit-identical
- [ ] F2 dispatch discipline (ValueError on unknown momentum_advection/momentum_flux_scheme)
- [ ] F3 zero-velocity ⇒ zero
- [ ] F4 uniform-flow analytic ⇒ zero
- [ ] F5 momentum conservation (periodic domain)
- [ ] F6 differentiability
- [ ] F7 idealized-gyre stability
- [ ] F8 regression lock (flux_form golden case)
- [ ] F9 oracle confirmation (informational)

---
