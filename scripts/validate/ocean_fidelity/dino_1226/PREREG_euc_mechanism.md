# Pre-registration — DINO equatorial-undercurrent mechanism probe

Registered before computing any legoESM-versus-NEMO vertical-mixing statistic
for this probe and before stepping either arm.

Upstream finding: `dino_regional_audit_result.md` reports a 28--32% deficit in
the 5--26 m equatorial zonal-velocity shear and a wind-driven-layer zero
crossing 9--11% (3.0--3.4 m) deeper in legoESM at days 90/180/270/360.  The
audit labels vertical mixing only PLAUSIBLE and prescribes the matched-state
coefficient comparison performed here.

## State, code, and provenance

Both arms start from the control member's NEMO day-180 restart
`DINO_00005760_restart.nc`, the same restart used by the verdict-year campaign.
The restart's T, S, u, v, before levels, TKE `en`, and closure-only `avm_k` are
read through the committed DINO twin loader.  The legoESM recipe is the shipped
`nemo_dino_kamm_mlf` card, with NEMO's own ladders (`LEGOESM_NEMO_E3T=both`),
fp64, the bridged start, and the corrected seasonal clock.  The NEMO comparison
target is the control run from this same restart.

The artifact must record, after reading rather than merely requesting them:

* legoESM launch SHA and dirty state;
* SHA-256 of every imported sibling reducer/loader;
* NEMO source/config SHA and dirty state, DINO namelist SHA-256, NEMO binary
  SHA-256, and day-180 restart SHA-256;
* observed recipe, ladder mode, seasonal phase, state dtype, and CUDA device.

A tree or input that changes between launch and artifact write is fatal.

## Leg O — offline matched-state coefficient discriminator

At the equator row (T-row 99), compare legoESM's closure-only vertical
viscosity computed from the bridged NEMO day-180 state with NEMO restart
`avm_k`.  The shared interface mapping is legoESM interface `i` to NEMO W-level
`jk=i+2`.  Report both the equator row and the nested E10 rows 89--109, but the
equator row decides.

The scored depth interval is the interfaces with NEMO `gdepw_1d` in
`[10, 40] m`, spanning the observed 28--35 m undercurrent top.  Every aggregate
uses interface control-volume thickness weights; no layer-count average may
decide a verdict.  Let

```
NRMS = sqrt(sum(w * (A_lego - avm_k_NEMO)^2)
            / sum(w * avm_k_NEMO^2))
```

over wet equatorial cells and scored interfaces.  Also report each interface's
thickness-weighted mean coefficient and ratio.

* **CONFIRM_CLOSURE_DIFFERENCE**: `NRMS >= 0.25` and at least two consecutive
  scored interfaces have a mean ratio outside `[0.75, 1.25]` in the same
  direction.
* **REFUTE_CLOSURE_DIFFERENCE**: `NRMS <= 0.10` and every scored-interface mean
  ratio lies in `[0.90, 1.10]`.
* **UNRESOLVED**: everything between those bars.

This leg names only whether the closures deliver materially different
coefficients at a matched state.  It does not by itself claim that viscosity
owns the trajectory difference.

## Leg R — short substitution response

Run one 10-day legoESM pair on `CUDA_VISIBLE_DEVICES=0` from the identical
day-180 state:

* **BASE**: shipped card, unchanged;
* **NEMO_AVM**: identical, except the momentum-viscosity output passed to the
  implicit vertical solve is replaced at every step by NEMO's day-180
  closure-only `avm_k`, mapped to legoESM interfaces.  Tracer diffusivity,
  prognostic TKE update, EVD trigger, forcing, and every configuration field
  remain BASE.  The replacement is a fixed coefficient field, so the result is
  explicitly a short impulse/retention discriminator, not a production model.

Daily outputs are scored against the NEMO control member at the same elapsed
day.  The primary statistic is the equator-row undercurrent-top depth: the
shallowest linearly interpolated sign change of zonal-mean `u` in the top
100 m.  The co-primary guard is the 5--26 m shear used by the regional audit.
Profile summaries and integrated differences use the campaign thicknesses.

At day 10 define core-depth closure

```
Cz = 1 - abs(z_NEMO_AVM - z_NEMO) / abs(z_BASE - z_NEMO).
```

Define shear closure analogously from absolute shear errors.

* **CONFIRM_VISCOSITY_DRIVER**: `Cz >= 0.50`, the core moves toward NEMO, and
  the shear error does not worsen by more than 10% of BASE's absolute error.
* **REFUTE_VISCOSITY_DRIVER**: `Cz <= 0.10`, or the core moves away from NEMO.
* **UNRESOLVED**: `0.10 < Cz < 0.50`, or a core-depth confirmation fails the
  shear guard.

If BASE's day-10 core-depth error is below 0.5 m, the response denominator is
too small and Leg R is **UNMEASURABLE**, not a pass.  No transport integral can
override either profile verdict.

## Controls

All controls run before a verdict is emitted and are demonstrated able to fail.

1. Exact day-0 identity of T, S, u, v and bridged before/TKE state on the shared
   wet mask; any mismatch is fatal.
2. BASE and NEMO_AVM inputs/configs hash-identical before the coefficient hook.
3. The hook must leave K_v and returned TKE bit-identical while changing A_v;
   a deliberate wrong interface shift must fail the known-depth mapping.
4. Wet/dry mask plant: `1e6` on a dry coefficient cell moves every scored
   profile by exactly zero; the same plant on a wet equatorial cell must move
   it.
5. BASE day-10 u must reproduce the previously committed/saved verdict-year
   control trajectory within its storage precision; otherwise no response
   verdict is emitted.
6. Profile reduction uses wet-cell zonal means and refuses an empty level.
7. A deliberate 2x coefficient plant must alter the first-step implicit
   momentum result; this proves the substitution path is live.

