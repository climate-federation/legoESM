# EKE mixing-length (`eke_len`) variant — progress log

Spec: `docs/ocean_fidelity/eke_len_build_spec.md`. Branch: `matching_Veros_oracle`.
One dated entry per gate: what changed, the exact gate command + result, commit
hash, every micro-decision. Newest at the bottom. Adversarial review per gate
(physics-validator agent + `/code-review`; the codex CLI it normally drives is
absent on this machine).

## Gate status
- [ ] L1 pure mixing-length functions (rhines + deformation radius + composite) + config + unit tests
- [ ] L2 wire `int_N_dz` + β + scheme dispatch into the coupling (default bit-identical)
- [ ] L3 differentiability (grad through the rhines length finite + nonzero)
- [ ] L4 oracle confirmation (reproduce Veros `eke_len`/`L_rossby`/`L_rhines`/`K_gm` to machine precision)
- [ ] L5 recipe adoption (flip EKE on in `build_acc_model_config`)
- [ ] L6 regression lock + measure-first free-run re-check

---
