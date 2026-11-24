# ORCA2 round 252 preregistration — direct post-drag V boundary

Date: 2026-10-12. Frozen base: `93eadc7de`. Scope: OMT-4, both
**independent** and **given NEMO's entry** labels. This round splits the
override-free `post_drag_v` boundary named by round 251 before any held
external-mode association is re-scored. Measurement is offline from the
production passive trace and the admitted slow-forcing record. No deck,
selector, carried-state contract, stabiliser, sea-ice field, or
`unmeasured_features` entry may change.

## Compiled source order

The executing oracle build forms the V depth mean without a surface-mask
association at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stp2d.f90:194-210`, calls the
baroclinic drag correction at `stp2d.f90:218-221`, and only then adds wind at
`stp2d.f90:223-233`. The drag routine constructs the signed coefficient and
bottom residual, then adds their product without a surface-mask multiply at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:1426-1464`.
The later two-dimensional Coriolis statement applies NEMO's resolved surface
V mask at `dynspg_ts.f90:286-324`.

legoESM's current direct path applies its compact `state.v_mask` after the
depth mean, after the drag addition, and again after the Coriolis subtraction
(`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:6383-6416,
6551-6563,7284-7288`). The round-251 record says that compact mask differs
from NEMO's resolved surface V mask on exactly the same 35 northern-fold
faces as the `post_drag_v` debt. Those three associations must be split rather
than treated as one inferred mask fix.

## Frozen predictions and falsifiers

1. **R252-P1 — instrument and records.** The inherited slow-forcing and
   external-substep twins remain admitted, both labels agree, CPU/JIT/fp64 is
   stamped, and the passive trace changes no completed-state bit. No forcing
   override is active. Any failed admission, label disagreement, dirty tree,
   wrong backend/precision, override, or moved state invalidates the split.
2. **R252-P2 — direct post-drag split.** The executing pre-drag depth mean,
   signed drag coefficient, bottom residual, literal drag increment, and
   *unmasked* drag addition are bit-exact on the 35 owner faces. The first
   non-bit operation is legoESM's compact surface-V-mask association after the
   drag addition: compact versus resolved mask is 35/35 unequal, and applying
   the compact mask reproduces round 251's `1.6557659420864476e-6 m s^-2`
   maximum at zero ULP. Any earlier unequal output, a non-exact unmasked
   addition, or a compact-mask replay that does not reproduce `post_drag_v`
   **REFUTES** this prediction and becomes the retained finding.
3. **R252-P3 — one resolved mask source.** Reusing the existing D114 resolved
   mesh builder for the slow-forcing U/V surface masks makes the post-drag and
   final slow-V outputs exact on all 35 faces, with zero off-support changes
   above the 2e-10 floor. A second mask builder, a card-specific fallback, or a
   result that needs any changed arithmetic beyond mask provenance refutes the
   candidate.
4. **R252-P4 — private unit.** Only after P3 closes its own boundary, re-score
   the two held NEMO associations with the complete round-217 unit. The
   kt=1-stage-1 real V fold-identity violations fall to zero and the unit
   completes kt=5 and kt=8 under both labels. If the direct boundary closes but
   this fails, the mask statement remains independently eligible while the
   association unit stays private under pending Decision 116.
5. **R252-P5 — landing.** A model statement lands only after its own exact
   replay, OMT-1..4, independent rung 0, given-entry rung 10, GYRE, DINO,
   tanks, moved-row, month, citation, and test gates pass under Decision 96.
   Otherwise all package changes are removed and the round is HELD with the
   next first boundary named.
6. **R252-P6 — controls.** Claim-label coverage, production D114 route,
   admission, passivity, override absence, source order, 35-face support,
   unmasked-addition replay, compact-mask replay, resolved-mask closure, and
   one-bit plants must each refuse when violated.

## Stop conditions

No in-executable observer and no NEMO acquisition are allowed: the required
operands already exist. No floor, clip, damping, alternate configuration, or
partial mask source may land. Decision 116 remains pending and is not
re-emitted.
