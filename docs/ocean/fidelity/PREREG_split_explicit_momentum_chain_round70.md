# Preregistration: Kmm surface W thickness, round 70

Date: 2026-08-30. Frozen before implementation and replay. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 69 makes carried `rn2b` bit-exact and reduces the row-8.8 maximum error
from `3.86313e-7` to `9.66343e-10`, but `e3w(Kmm)` remains red at
`5.19566e-9`. The carried `e3w_Kmm` bundle contains only NEMO levels 2:jpk;
the native helper restores level 1 from its later recomputed geometry. NEMO
`depth_e3.F90:64` defines `e3w(1)=2*(gdept(1)-gdepw(1))`; DINO's uniform top
cell makes this identical to carried `e3t(1,Kmm)`, already present in the
same `TKEEntryN2Bundle`.

When the bundle has nlev-1 W slots, prepend `e3t_Kmm[...,0]` before passing it
to the Treguier builder. Preserve full-nlev inputs unchanged and reject every
other depth. This changes only the already-scoped carried-step-entry DINO
path. Replay direct `e3w`, the entire coefficient ladder, and rows 8.8--8.10
under unchanged bars. The later-geometry surface reconstruction remains a
red planted control via round 69.

Frozen round-69 receipt SHA-256:
`bd109e2869333a31b8b6a8410d3ec10f24acae94353ed1f35d60900ae5b62eb2`.
