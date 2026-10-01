# Preregistration: dyn_zdf RHS wind-placement peel, round 36

Date: 2026-08-30. Frozen before numerical execution. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

## Question and source order

Round 35 localizes row 4 first to the raw dispatch RHS. NEMO constructs
`Kbb+rDt*Krhs`, removes `uu_b/vv_b`, and adds the barotropic bottom-drag RHS
term at DINO `dynzdf.F90:137-178`; only after the matrix forward-factor sweep
does it add surface stress to level 1 at `:340-373`. Production's default
explicit stress is already in the state before the baroclinic strip at
`ocean_model_latlon_cgrid.py:7334-7354,7275-7296`. This round asks whether
stripping the wind's depth mean before the solve owns the RHS difference.

## Additive arms and bars

One unmodified CPU production step captures the public mixing-entry U/V and
the raw dispatch inputs. From those same arrays and the existing dumps, score:

1. `B`, the production raw RHS (round-35 identity repeat);
2. `N`, the non-wind production RHS after subtracting the actual wind's
   baroclinic contribution and its bottom-diagonal mean coupling, against the
   corresponding NEMO no-wind RHS;
3. `W`, the actual full top-cell wind deposit against NEMO
   `poststress-prestress`;
4. `D_actual`, the counterfactual that restores the actual wind depth mean
   (and its drag-diagonal coupling), equivalent to adding the actual wind
   after the strip;
5. `D_oracle`, `D_actual` with only the full wind deposit substituted by
   NEMO's dump.

All RHS arms use the ACCUMULATING `1e-12` bar. Placement is the majority owner
if `D_actual` removes at least 90% of the baseline U normalized-RMS error.
It fully owns the U RHS only if `D_actual` is AT BAR; if `D_oracle` alone is
AT BAR, ownership is wind placement plus wind arithmetic. A red `N` records
independent upstream RHS debt and prevents full row-4 promotion even if the
90% removal bar passes. V is a zero-wind falsifier: its placement arms must be
bit-identical to baseline and any V change invalidates the decomposition.

## Admission and controls

Bind the official round-35 SHA, checkout, entry restart, retained manifest,
all streams, NEMO DINO `dynzdf.F90`, production model and stress helper,
scorer, and this registration. Require CPU/fp64, tracked-clean checkout,
checkout-first `PYTHONPATH`, session ID, exactly one public momentum entry and
two raw dispatch calls with restoration. Algebraically reconstruct baseline
from `N + stripped(W) - diag*mean(W)` to the accumulating bar. Require the V
zero-wind control, a same-JAX identity, a `2x` bar plant, roll, and wet NaN to
fire. No SLOT is allocated; all oracle data are retained.
