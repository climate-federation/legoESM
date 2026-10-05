# Preregistration — ORCA2 round 157 external-mode exchange split

Date: 2026-10-05. Frozen base: `8a7ac4e5f`.
Every ORCA2 number in this round is **independent**: hierarchy rung 0 starts
from its own climatological T/S, zero velocity, and zero sea surface. No
configuration, initial state, forcing, carried-state form, stabiliser,
sea-ice selector, or `unmeasured_features` entry may change.

## Source order and question

Round 156 narrowed the unsafe complete external-mode association to its U and
V velocity outputs. The source unit is the seven-field call at compiled
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:770-779`.
This np2 record resolves `jpni=2`, `jpnj=1`, `nn_hls=2`, and
`ln_nnogather=.true.`. Its compiled double-precision `lbc_lnk` first performs
east/west MPI exchange (`lbclnk.f90:1960-1992`) and only then dispatches the
north-fold exchange (`lbclnk.f90:2104-2114`). The T-pivot no-gather branch
explicitly sends the extra duplicated U/V lines (`lbcnfd.f90:1561-1574`,
`:1629-1669`) and conditionally overwrites them (`lbcnfd.f90:1750-1767`).

This round splits those two geometry operations for the U and V outputs and
walks their first consumers through the 65 external substeps. Each component
is a private measurement arm only; neither a one-field association nor one
half of its exchange is a NEMO program unit eligible to land.

## Frozen protocol

1. Reuse the admitted rung-0 entry, slow forcing, raw barotropic history,
   raw face depths, unmasked V transport, and materialised V transport from
   rounds 146-156. Run CPU production JIT in fp64/libm.
2. Add one fail-closed private selector with exactly four values in compiled
   order: `u_cyclic`, `u_fold`, `v_cyclic`, `v_fold`. Unknown values and any
   overlap with the complete or one-field arms refuse.
3. For each of the 65 external substeps, compare production with the selected
   component immediately after association and at the next transport,
   continuity, and SSH boundaries. Record the first unequal substep, cell
   count, maximum absolute difference, and argmax. No magnitude-only equality
   can hide signed-zero differences.
4. Validate the instrument before attribution: composing U cyclic then U fold
   must equal the existing U-only associated image; composing V cyclic then V
   fold must equal the existing V-only image, bit for bit. The ordinary
   observer must reproduce the unobserved production state bit for bit.
5. If a component can be followed safely to kt=1 stage 1, compare its
   T/S/u/v/ssh/uu_b/vv_b state with unchanged production at that boundary.
   Do not continue a component past the first non-finite row.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R157-P1 | The split instrument is passive. | Unselected production is array-identical to round 156 and its rung-0 ladder moves 0/200 rows. | Any default-path bit or ladder row moves. |
| R157-P2 | U cyclic is the first source-ordered non-vacuous U operation. | `u_cyclic` changes U immediately after association at an earlier source boundary than `u_fold`. | `u_cyclic` is exact or only `u_fold` changes U. |
| R157-P3 | The U fold is independently non-vacuous on the T-pivot row. | `u_fold` changes at least one U bit at the association boundary. | The component is exact. |
| R157-P4 | V cyclic is a no-op in the compact global representation. | `v_cyclic` stays bit-exact at every scored boundary while its plant fires. | Any V or downstream bit moves. |
| R157-P5 | The V north-fold operation owns the dominant finite growth seen in round 156. | `v_fold` is non-bit after association and its kt=1 stage-1 T maximum is within one ULP of the V-only arm's `0.1774972822409795 K`. | It is exact, becomes non-finite before that boundary, or the maximum differs by more than one ULP. |
| R157-P6 | The four-way split is complete. | U and V component compositions are bit-identical to their existing one-field associated images on every substep. | Either composition differs or a selector is vacuous without being rejected. |

## Controls and terminal rule

One-ULP, signed-zero, selector-registry, overlap, and component-composition
plants must all refuse. A V-cyclic no-op is not credited as a control; its
known-answer plant must perturb an active boundary value and be detected.

Production remains unchanged. The round is **HELD** after naming the first
non-bit exchange operation and its first consumer. A complete production
candidate remains ineligible while it becomes non-finite at kt=8.

ASKED choices: none. UNASKED choices: empty.
