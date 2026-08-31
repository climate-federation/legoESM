# Preregistration: DINO standalone initialization-geometry repair

Date: 2026-08-30. Session:
`01a053d4-8e9f-7212-bbdb-19ba2d64e140`.

Status: **FROZEN BEFORE IMPLEMENTATION OR RESCORE.** This is the repair round
licensed by `dino_ic_euler_peel_v2`, whose first over-bar row was
`input_wet_mask`. No GPU, NEMO execution, MPI execution, or long standalone
arm is part of this protocol.

## Question and fixed ordering

Can legoESM's standalone `nemo_dino_kamm_mlf` construction reproduce the
geometry NEMO passes to `usr_def_istate`, without reading a restart or mesh at
runtime? Repair and score in this order:

1. physical-core topology (`tmask`),
2. T-point latitude (`gphit`),
3. three-dimensional at-rest T depth (`gdept_0`),
4. resolved case-4 T/S initialization, and
5. only if every initialization row passes, the existing Euler and filtered-
   carry rows against `RUN_KT2`.

Any earlier failure stops attribution. Later rows may be printed for debugging
but must be stamped `CONDITIONAL_AFTER_<FIRST_FAILURE>` and cannot own the gap.

## Source ownership frozen before the repair

- **Topology.** DINO selects `ln_Iperio=.true.` (`RUN_KT2/namelist_cfg:65`).
  `usrdef_zgr.F90:471-479` derives `k_bot/k_top` from positive analytic
  bathymetry and applies the model boundary condition; the 195 x 48 physical
  core has no artificial western land column. legoESM currently calls
  `partial_periodic_seam_wall_latlon` at
  `packages/ocean/legoesm/ocean/experiments/dino.py:2982-2992`, even when the
  already-resolved `nemo_faithful_grid` selector is true.
- **Latitude.** `usrdef_hgr.F90:96` forms integer T-row index `ztj`; line 106
  evaluates `ASIN(TANH(rn_e1_deg*rad*ztj))/rad`. legoESM has the same integer
  placement at `packages/core/legoesm/grids/latlon.py:691-708`, but lowers the
  transcendental through JAX; the frozen v2 receipt measured 5,472 exact
  mismatches, maximum `5.684341886080802e-14` degree. The repair reuses
  `nemo_faithful_grid`: only that static construction uses scalar source-order
  host libm, while the general Mercator default remains unchanged.
- **Depth.** `usrdef_zgr.F90:108-120` calls `zgr_sco_mi96` with
  `zflat=zHmax`. `zgr_lib.F90:161-169` first builds the reference ladder and
  locates the `rn_hco` join; lines 173-181 rebuild the 3-D profile below it;
  lines 184-189 run `depth_to_e3` then `e3_to_depth`. legoESM's
  `dino_lat_lon_vertical` at
  `packages/ocean/legoesm/ocean/experiments/dino.py:2699-2708` retains only the
  first mi96 ladder, and `standalone_20y.py:371-386` broadcasts it. The frozen
  v2 receipt measured a 104.96931566119792 m maximum depth error.

Repository search found and will extend, not duplicate:

- `create_mercator_grid(..., equator_on_tpoint=True)` for integer-index
  Mercator placement;
- `create_levy_stretched_z_star(..., analytic_t_depths=True)` for the first
  mi96 pass;
- `dino_masked_zco_coordinate` / `create_partial_cell_coordinate` for the
  full-cell staircase; and
- the committed `ic_euler_peel.py` scorer and `nemo_istate_case4`
  source-order oracle transcription.

No existing implementation of the `rn_hco` second mi96 pass was found. The
minimum public addition is a reusable NEMO-mi96 join option in the existing
vertical constructor, not a second DINO-only ladder implementation. The
physical value is a resolved DINO field, `z_coordinate_transition_depth_m`,
set explicitly to the executed namelist's 1000 m on both DINO oracle cards.

## Frozen rows and bars

Use the same core extraction and inputs as `dino_ic_euler_peel_v2`. All inputs
remain SHA-256 stamped. No common-mask substitution is legal for topology.

| rung | row | active set | pass bar |
|---|---|---|---|
| G0 | `input_wet_mask` | all 195 x 48 x 35 cells | zero mismatches |
| G1 | `input_latitude_deg` | all 195 x 48 T points | bitwise equality |
| G2 | `input_t_depth_m` | NEMO wet T cells | bitwise equality |
| I0 | `resolved_T_nemo_wet` | NEMO wet T cells | max abs <= 1e-15 degC |
| I1 | `resolved_S_nemo_wet` | NEMO wet T cells | max abs <= 1e-15 PSU |

The existing depth-only profile rows retain their 1e-15 pointwise bar. If
geometry passes but source-order profile evaluation alone exceeds it, that is
the next initialization owner and Euler remains withheld; the bar is not
relaxed.

Euler rows retain the v2 1e-15 bars and exact NEMO raw dumps. They are legal
only when G0-G2, I0-I1, and both depth-only profile rows pass. The step-2
restart before-fields remain a filtered carry receipt, not the raw step-1
endpoint.

## Controls and outcomes

Required controls:

1. the old seam-wall path must fail G0;
2. the old JAX Mercator evaluator must fail exact G1 on this oracle;
3. the old first-pass-only ladder must fail G2 by more than 100 m;
4. the existing one-ULP field plant and wrong-core plant must fire; and
5. a geometry-pass gate must refuse to execute/issue Euler rows when any
   initialization row is planted over bar.

Outcome labels:

- `INIT_GEOMETRY_CONFIRMED`: G0-G2, I0-I1, and both profile rows pass.
- `INIT_GEOMETRY_PARTIAL_<FIRST_FAILURE>`: any initialization row fails;
  report the first row and do not attribute Euler.
- `EULER_SCORED`: initialization is confirmed and all Euler/carry rows are
  emitted with their frozen bars. This label does not imply they pass.

A fresh standalone year is worth issuing only if initialization is
`INIT_GEOMETRY_CONFIRMED`, the Euler rows have been legally scored, and no
claim-admission blocker remains. Otherwise the handoff contains no GPU launch
block and names the first remaining repair.
