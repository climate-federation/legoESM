# ORCA2 round 331 preregistration — remaining TKE family localization

Frozen 2026-10-24 after source inspection and before running a round-331
production-JIT score or reading any new numerical payload.  The round-330
receipt's admitted totals (901 returned-energy, 931 post-TKE-viscosity, and
925 consumed-viscosity inequalities) are prior evidence, not measurements by
this round.

## Source-read discriminator

The record build holds the surface TKE value in `en(:,:,1)` and solves only
`2:jpkm1` (`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:272-276`
and `:411-477`).  The card selects its separate `nemo_z0` surface row and
literal matrix/solve path; no different formula, index, or branch is visible
in the source comparison.

Before `zdf_tke`, NEMO calls `zdf_sh2`, `zdf_drg`, and `zdf_mxl` in that order
(`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90:318-336`).  On this
deck, `nn_etau=1`, so TKE never reads `zdf_mxl`'s `nmln`; the only such read is
the inactive `nn_etau=2` arm (`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:498-510`).
Linear bottom drag is initialized once and `zdf_drg('BOTTOM')` does not update
it when `l_zdfdrg` is false; the top call adds the recorded implicit ice drag,
but `zdf_tke` reads neither coefficient except through its already scored
bottom/cavity boundary branch.  The live shear field remains an unrecorded TKE
input.

## Frozen predictions and falsifiers

- R331-P1 **CONFIRM:** a CPU fp64 production-JIT re-score on the landed
  round-330 card reproduces the admitted totals: returned `en` 901, post-TKE
  `avm` 931, consumed `avm` 925, and zero inequalities at zero-based `k=28`.
  It reports exact counts for every level and partitions the residual by
  surface (`k=0`), interior (`1<=k<28`), and deepest (`k=28`) families.
  **REFUTE:** any total or deepest-family count differs; reconcile before
  recording a new number.
- R331-P2 **CONFIRM:** the compiled surface boundary and the card's selected
  literal path have no transcription difference; therefore no surface-BC arm
  is proposed from source reading alone.  **REFUTE:** a different constant,
  operand, index, solve bound, or executing branch is found; replay that one
  literal difference against the admitted final `en` before proposing it.
- R331-P3 **CONFIRM:** `zdf_mxl` is inactive for the executing `nn_etau=1`
  TKE arm, and the linear-bottom `zdf_drg` call does not rebuild `rCdU_bot`.
  **REFUTE:** the resolved selectors or an executing `zdf_tke` read show
  otherwise.
- R331-P4 **CONFIRM:** the admitted records cannot distinguish `zdf_sh2` from
  later inside-TKE matrix/RHS/solve statements.  Extend the reviewed
  round-328 acquisition by importing its helpers and recording both-rank
  `zdf_sh2`, `zdf_drg`, and `zdf_mxl` outputs plus inside-`zdf_tke` boundaries
  after the boundary conditions, after matrix/RHS construction, and after the
  solve.  Its read-only preflight and every planted violation pass/fail as
  specified; stop for the operator-run record.  **REFUTE:** the old records
  already make one cited source statement uniquely replayable and that arm
  makes every affected row bit-identical or strictly better under Decision
  122/139.

No configuration choice is authorized.  A package change is eligible only
after one cited executing statement owns a family and passes all seven pairs,
both labels, every other executing card, and the N=1 reverse/forward gradient
gate.
