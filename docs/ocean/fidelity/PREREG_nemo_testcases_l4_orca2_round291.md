# ORCA2 round 291 preregistration — stage-1 V correction operand split

Date: 2026-10-16

Round 290 proved that the independent OMT-4 kt=1 stage-1 raw V update is at
the campaign floor on fold source row 146 and that the following barotropic
correction creates the full 1,440/5,400 debt.  This round splits that
correction in NEMO's compiled order:

1. carried external-mode target `vv_b(:,:,Kaa)`;
2. ascending vertical sum `SUM(e3v_3d*vv(:,:,:,Kaa))`;
3. stored reference-depth reciprocal `r1_hv_0`;
4. `zvb = vv_b - sum*r1_hv_0`;
5. `vv(Kaa) = vv(Kaa) + zvb*vmask`.

The statements are
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:521-528`
and `:547-550`.  The reference thickness, depth and reciprocal originate at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/domzgr.f90:184-188` and
`domain.f90:193-215`.

Every value is **independent OMT-4, kt=1 stage 1**.  The instrument reuses the
round-290 passive trace, the admitted rank-complete round-280 `pre_v`,
`ext_vb`, and `cor_v` streams, and the production card's raw NEMO face bundle.
No executable observer, NEMO run, model edit, card change, configuration
choice, carried-state change, or landing is authorised.

## Frozen predictions and falsifiers

1. **R291-P1 — admission and passivity.**  The round-280 record re-admits as
   `PASS_R280_OMT4_STAGE1_ADMISSION`; every traced completed-state leaf remains
   array-identical to the ordinary run; CPU production JIT and fp64/libm are
   printed.  Any failure refuses all numbers.
2. **R291-P2 — carried target.**  Candidate `target_v` on source row 146 is at
   the `2e-10 m/s` floor against recorded `ext_vb`.  A valued difference makes
   the external-mode target the first operand and refutes the predicted mask
   owner.
3. **R291-P3 — depth-weighted sum and reciprocal.**  The candidate reference
   V thickness and `r1_hv_0` reproduce the raw NEMO bundle on the source row;
   using each side's matched `pre_v`, the ascending depth-weighted sums and
   their products agree within the floor.  The first valued row among
   thickness, sum, reciprocal, or product owns the boundary and refutes the
   predicted mask owner.
4. **R291-P4 — scalar correction.**  The source-ordered candidate scalar
   correction `target_v - sum*r1_hv_0` is at the floor against the correction
   inferred from NEMO's matched `cor_v-pre_v` on wet levels.  Replaying it with
   the candidate's stage mask reproduces the live corrected V exactly.  A
   valued scalar mismatch names the first preceding operand; a failed
   candidate self-replay refuses the instrument.
5. **R291-P5 — final mask boundary.**  The compact production stage mask is
   the first valued operand: it differs from NEMO's raw resolved `vmask` on
   1,319 source-row level slots, and replacing only that mask makes the
   corrected source row valued-exact.  A different count or a non-closing
   resolved-mask replay refutes this prediction.  The statement remains
   **HELD** because round 286 already proved its source-exact route exposes a
   cancelling error and refuses the independent rung-0 trajectory at kt=4.
6. **R291-P6 — non-vacuity.**  Passivity, source order, target, thickness,
   reciprocal, scalar-correction, mask, self-replay, and one-bit plants must
   each make the gate refuse.  A green plant refutes the instrument.

The outcome is measurement-only and **HELD**.  Decision 125's completed-stage
top-V exchange and the resolved stage mask remain off, Decision 126 remains
pending and unchanged, and sea ice remains exactly as the shipped card carries
it.

Independent review: operator, as required for this lane.
