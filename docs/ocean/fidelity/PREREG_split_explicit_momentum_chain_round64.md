# Preregistration: live QCO tracer face thickness, round 64

Date: 2026-08-30. Frozen before implementation and replay. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 63 corrects the row-8.3 instrument, makes 8.3 and 8.4 bit-exact, and
stops at 8.5: NEMO's live `e3u(Kmm)` differs in all 9,758 U columns, maximum
normalized error `7.4551634675e-5`. The production tracer mass flux still
uses generic min-face `h_u_old/h_v_old`, whereas the already-certified literal
Kmm cycle and WZV path build raw live QCO face geometry from Kmm SSH.

Under the existing `wzv_call2_evaluation="nemo_literal"` scope, build live
Kmm `e3u/e3v` with `nemo_qco_live_face_thicknesses` from bridge-carried
`e3t_0`, raw 3-D masks, `hu_0/hv_0`, and metric areas. Map native east/north
faces once to the redundant legoESM layout and use those thicknesses for
`mass_flux_u/v = e3{u,v}(Kmm) * {u,v}_corrected * mask`. The literal velocity
and thickness are one coupled tracer-entry composition. Generic/off paths
must retain their old statements byte-for-byte.

Tests must cover the live-thickness formula, nonzero separation from the
generic min-face plant, JIT/autodiff, and generic byte identity. Rerun the
round-63 direct-cycle scorer with unchanged bars. Rows 8.3--8.5 must be at
bar; stop at the first later row. If 8.5 remains red, do not promote it.

Frozen round-63 receipt SHA-256:
`3a1a25ca328761b1bcbeb87953751a3a15b1ac00852b2ff62fd4223d107d24e0`.
