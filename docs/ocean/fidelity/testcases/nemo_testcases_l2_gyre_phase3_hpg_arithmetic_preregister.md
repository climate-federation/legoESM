# NEMO testcase lane 2 GYRE — stage-2 HPG arithmetic preregistration

Date: 2026-09-01

Session: `ea650f83-28b8-4c68-b0cc-809a9fd417de`

Parent tracer/interleave landing: `553b2ac60` plus the uncommitted collapsed
WS tracer/interleave work measured by
`legoesm_phase3_literal_hpg_kt2_gate.json`.

## Registered boundary

The production stage-2 baroclinic RHS differs from the oracle by
`6.210838231905309e-17` U and `6.826983728936255e-17` V.  Those tendency rows
are inside the `1e-15` tendency bar, but multiplication by the resolved
`rDt=7200 s` produces `4.4718035621766294e-13` and
`4.915428269481632e-13` raw-Kaa errors.  Therefore the integrated state remains
DEBT; an operator-level AT-BAR row is not promoted to a whole-stage match.

Vorticity and advection contributions are already below `4.59e-26` absolute.
The remaining RHS discrepancy is consequently registered at HPG arithmetic,
not assigned to those near-null terms.

## Oracle source and one-variable arm

The selected `hpg_sco` recurrence evaluates the surface face expression as
`zcoef0 * r1_e1u * (east_product - west_product)` and the slope correction as
`-zcoef0 * (east_rhd + west_rhd) * (east_depth - west_depth) * r1_e1u`
(`dynhpg.F90:340-360`), then repeats the same association while accumulating
`zhpi/zhpj` from level 2 through `jpkm1` (`:367-390`).  Under `key_RK3` this is
the first assignment into `Krhs` (`:357-363,383-388`).

The repository search found the canonical `nemo_sco` implementation, but its
current face construction calls `gradient_*_cgrid` and
`interp_cell_to_*face`; those helpers divide and average before the final
products.  The registered arm changes only this association to the literal
source order, retaining identical rho, e3w, gdept, metrics, masks, and RK
program.  Because NEMO exposes no HPG-arithmetic switch, a supported result is
folded into the existing `nemo_sco` identity with no public selector.

## Confirm/refute and controls

- CONFIRM: the direct HPG/RHS discrepancy moves at residual scale and the
  stage-2 corrected Kaa reaches `<=1e-15` normalized absolute error.
- REFUTE as the stage owner: movement is below one tenth of the registered
  corrected-Kaa residual, or corrected Kaa remains DEBT.
- The existing nonuniform-rho unit oracle is evaluated with a planted
  association violation, and the full gate's planted `+1` HPG/RHS controls
  must still fire.
- fp64 state and geometry dtypes remain mandatory.  All later stage-3 terms
  and kt=2 trajectory fields remain UNMEASURED until this first boundary is
  classified.
