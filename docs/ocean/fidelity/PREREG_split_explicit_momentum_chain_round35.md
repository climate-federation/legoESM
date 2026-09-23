# Preregistration: corrected raw dyn_zdf dispatch boundary, round 35

Date: 2026-08-30. Frozen before numerical execution. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

## Correction and question

Rounds 33--34 are retracted because their fed arm double-subtracted the
barotropic mean and compared different representations. This round changes no
physics and substitutes no operand. It runs one unmodified production day-180
step and observes the first U and V calls to
`implicit_vertical_diffusion_ocean_momentum_dispatch` at
`ocean_model_latlon_cgrid.py:7582-7680`. The captured dispatch input and raw
return both precede production's mean readdition at `:7550-7557`, matching
NEMO's barotropic-free `dyn_zdf` stage-8 representation
(`stpmlf.F90:398-407`; `dynzdf.F90:137-214,305-371`).

## Ordered ladder and bars

The reference operand pack is the already-committed construction in
`zdf_chain_end.py:314-413`:

1. literal selector, `rDt=5400 s`, and U/V wet mask;
2. RHS: `Kbb + rDt*Krhs - barotropic(Kaa)`, surface stress, and implicit
   barotropic bottom-drag RHS contribution;
3. adjacent-T-point face `avm`;
4. live cell thickness `e3u/e3v(Kaa)`;
5. live interface thickness `e3uw/e3vw(Kmm)`;
6. bottom-only implicit drag diagonal;
7. raw dispatch return versus NEMO stage-8 `naa_B`.

RHS and raw output use the ACCUMULATING `1e-12` class bar. Coefficients,
metrics, masks, scalars, and diagonal use the POINTWISE `1e-15` bar. The first
strictly failing input operand owns the row-4 assembly interval and later rows
are ordered-blocked. If all inputs pass and the raw output fails, that
contradicts the bound chain-end kernel receipt and is INVALID pending diagnosis.
If an input fails but output passes, the difference is bounded cancellation:
row 4 may be promoted only with the failing operand recorded as debt. If every
input and output passes, row 4 is AT BAR and row 5 opens.

## Admission and controls

The scorer binds the checkout, entry restart, retained stream manifest, every
consumed dump, active NEMO sources, production modules, itself, and this
preregistration. It requires a tracked-clean checkout, checkout-first
`PYTHONPATH`, CPU/fp64, lane `d180`, and the exported session ID. Exactly two
calls, U then V, with the declared stagger shapes are required. The hook must
be restored and an identity call must be unchanged. A `2x` class-bar plant,
one-cell roll, wet NaN, wrong `rDt`, and output plant must each fire. Missing
or changed hashes, nonfinite captures, or ambiguous call order are INVALID.
No SLOT is allocated: the production step and all oracle operands use existing
day-180 dumps.
