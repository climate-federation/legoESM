# Prognostic EKE build — progress log

Spec: `docs/ocean_fidelity/eke_build_spec.md`. Branch: `matching_Veros_oracle`.
One dated entry per iteration: what changed, the exact gate command + result, commit hash,
every micro-decision. Newest at the bottom.

## Gate status
- [ ] E1 EKE closure module (pure) + EKEConfig + direct unit tests
- [ ] E2 dispatch (prognostic kappa_GM mode; ValueError on unknown)
- [ ] E3 positivity (E >= 0 / E_min)
- [ ] E4 budget closure (advection + iso-diffusion conserve integral-E)
- [ ] E5 differentiability
- [ ] E6 state threading + zero-behaviour-when-OFF (existing bit-identical)
- [ ] E7 idealized channel (E spins up bounded; kappa_GM responds)
- [ ] E8 regression lock (EKE-active golden case)
- [ ] E9 oracle confirmation (informational) + recipe adoption

---
