# Preregistration: exact mesh-reference Kmm surface e3w, round 71

Date: 2026-08-30. Frozen before implementation and replay. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 70 refutes and retracts the `e3t(1,Kmm)` proxy: row 8.8 worsened to
`8.7793e-4`. NEMO `depth_e3.F90:64` defines the surface W thickness from
T-depth, while the DINO raw mesh already carries a distinct `e3w_0`.
legoESM's canonical `nemo_e3w_from_live_gdept(..., interior=False)` implements
the exact mesh-reference rule `e3w_0*(1+r3t)` (`eos.py:787-801`).

At step-entry bundle construction, compute that full live mesh-reference W
field with the same reciprocal `r3t` already used by eosbn2 and carry only its
surface slot as a new optional `e3w_surface_Kmm` field. Prepend it to the
existing certified interior `e3w_Kmm` only for the GM coefficient call.
Existing bundle constructors remain valid through a `None` default; `None`
retains the historical reconstruction. Tests must distinguish the exact raw
mesh surface from the refuted `e3t` proxy and retain generic byte identity.

Replay direct `e3w`, all coefficient operands, and rows 8.8--8.10 under the
same bars. Round 70's worsening is a mandatory planted-control receipt.

Frozen round-70 receipt SHA-256:
`d831bbcb89e06c8795a085103041293becedb8ac79c438f1302b44296b3314bc`.
