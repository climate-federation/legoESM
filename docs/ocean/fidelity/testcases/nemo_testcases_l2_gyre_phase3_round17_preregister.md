# GYRE phase 3 round 17 preregistration — stage-1 HPG/Krhs chain

Date: 2026-09-04

Starting tip: `8b6ec4efe31e08d0f8db9c1e1378f7049ffc2409`

Shared source-rounding convergence commit: `2a7b1f7ae`

Canonical helper source: ice-lane commit `86a8eb21d18` (the earlier supplied
`e2ad2c30629` reference was corrected by the user).  Its executable body and
the former ocean body are identical; the hermetic prd pin and the production
EOS JIT-parity gate remain bit-exact/0 ulp after the ownership swap.

## Oracle execution order

At kt=1 from rest, `stp2d.F90:126-128` calls EOS and `dyn_hpg` first and stores
the HPG result in `uu/vv(:,:,:,Krhs)`.  It then calls `dyn_ldf` at `:130-131`,
`dyn_vor` at `:144-146`, `wzv` at `:155`, and, for GYRE's live vector-invariant
`np_VEC_c2` branch, `dyn_keg` and `dyn_zad` at `:159-165`.  The dumped initial
velocity is identically zero, so lateral diffusion, Coriolis/relative
vorticity, kinetic-energy gradient, and vertical advection are identically
zero.  Therefore the completed stage-1 3-D momentum `Krhs` is HPG alone.

NEMO `hpg_sco` evaluates, in source order, `zcoef0=-grav*0.5` at
`dynhpg.F90:340`; the surface `zhpi/zhpj` at `:343-350`; surface terrain
corrections `zuap/zvap` at `:351-355`; and `Krhs=zhpi+zuap` at `:357-360`.
For levels 2 through `jpkm1`, it accumulates `zhpi/zhpj` at `:367-375`, forms
the level-local terrain correction at `:376-380`, and writes the sum at
`:381-384`.  Metric reciprocals `r1_e1u/r1_e2v` and masks are the resolved
oracle operands, not reconstructed alternatives.

## Prediction and falsifiers

**Predicted owner:** missing per-source-statement materialization in the shared
`hpg_sco` recurrence.  The one-variable arm changes only association by applying
the shared `nemo_source_round` after each multiplication, addition/subtraction,
and written assignment while preserving the NEMO operand order and reciprocal
forms.  No card or public scheme switch may select the old path.

CONFIRM only if all of the following hold against Oracle V2 under production
JIT/CPU/fp64/libm:

1. the direct `hpg_sco` output and stage-1 `Krhs` become bit-exact in every wet
   U/V cell;
2. the depth mean, post-drag/post-wind slow forcing, external-mode substeps,
   `un_adv/vn_adv`, and stage-1 Kaa velocity become bit-exact in order; and
3. an ablation restoring the pre-fix recurrence reproduces the old nonzero
   `Krhs` residual, while a planted live HPG operand exits nonzero.

REFUTE ownership if stage-1 `Krhs` remains non-bit-exact.  The first differing
source operand or intermediate becomes the next boundary; no downstream label
is allowed.  If HPG/Krhs clears but the depth mean remains nonzero, apply the
already measured NEMO-literal reduction as the next one-variable arm and score
it independently.  Only if the complete stage-1 chain clears may the run
proceed to the kt=1..10 sweep, cross-card gates, stage-2 Kaa, and stage-3
transports.

The public implementation is the NEMO identity.  The old recurrence may exist
only in a private diagnostic hook for the one-variable falsifier.  Every new
gate must include a live planted control that exits nonzero.
