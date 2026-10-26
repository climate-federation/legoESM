# Preregistration: carried Kmm Treguier bundle, round 69

Date: 2026-08-30. Frozen before implementation and replay. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 68 directly measures both inputs to the coefficient sums as divergent:
`e3w(Kmm)` maximum normalized error `1.3319403821e-7`, and `rn2b`
`1.7941006975e-7`. NEMO `ldftra.F90:747-767` reads both at the same `Kmm`
time level as the preceding exact `ldf_slp` call. legoESM already carries the
certified step-entry `(rn2b,e3w_Kmm)` bundle into `compute_nemo_native_slopes`
but drops it before `compute_treguier_kappa_gm_nemo_native`, which recomputes
both at the later geometry.

Extend the Treguier native builder with optional `pn2_override` and
`e3w_override`, using the same nlev/nlev-1 restoration and validation as the
slope builder. Pass the existing `native_slope_pn2/native_slope_e3w` pair from
the production dispatch. Defaults remain `None`; every non-DINO path is byte
pinned. The DINO path must carry the pair together. Tests must prove each
single override is a non-inert planted arm, the coupled override uses both,
invalid depths fail, generic output is byte-identical, and JIT/grad stay finite.

Replay rows `e3w`, `rn2b`, `zn`, `zah`, `zhw`, `zRo`, `zaeiw`, `aeiu`, then
8.8--8.10 under unchanged bars. Promote only consecutive passing rows.

Frozen round-68 receipt SHA-256:
`8e9a2ee7ee1969da86ee2ef24198047d6ea83ee14842753b528271453efaf9a0`.
