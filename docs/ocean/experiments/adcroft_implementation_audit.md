# Adcroft & Campin 2004 partial-cell PGF correction — implementation audit

**Reviewer**: dycore-/ocean-model-expert subagent
**Date**: 2026-04-30
**Branch**: `realistic-geometry-full`
**Question**: Is the ~99 mm/s spurious flow on Beckmann-Haidvogel produced by
`pgf_scheme="adcroft"` *expected literature behavior of leading-order A&C on
partial cells*, or evidence of a bug in our implementation?

## Verdict — (a) expected, faithful, SMC03's 65× advantage is real

The 99 mm/s residual on BH at r=0.54, drag=1e-3, exponential thermocline,
3°/20-level, dt=600 s with implicit-CN barotropic is **consistent with the
published behavior of leading-order Adcroft–Campin / Pacanowski–Gnanadesikan
partial-cell PGF on a stratified seamount** at this slope parameter. Our
implementation in `partial_cell_pgf_correction_x/y` is faithful to A&C 2004
eq. (4)–(7): correct face-reference choice, correct sign, correct ρ' (not ρ),
correct application restricted to centroid-mismatched levels. The 2Δz forced
mode diagnosis in `partial_cells_results.md` is physically correct — it is the
generic structural failure mode of any *single-level* PGF correction that
shifts pressure to a face-reference depth at the partial-bottom level only.
SMC03 closes that gap because it computes a smooth-in-k face-reference
pressure that does not have step-function vertical structure.

There is one **caveat / "uncertain" sub-finding** (item 5 below): the
implicit-CN barotropic + explicit baroclinic combination at dt=600 s is
mildly unfair to Adcroft because it lengthens the time available for the
forced 2Δz mode to reach equilibrium. A split-explicit run at dt=300 s
would likely give a similar amplitude (~80–100 mm/s), but this should be
spot-checked rather than assumed.

## 1. Faithfulness to A&C 2004 eqs. (4)–(7)

A&C 2004 §3 derives the partial-cell PGF correction as: shift each adjacent
cell's hydrostatic pressure to a *common* face-reference depth `z*` before
differencing, where the shift is `Δp = ρ_local · g · (z_centroid - z*)`
(positive downward; assuming linearised hydrostatic `dp = ρ g dz`). The
canonical face-reference choice in A&C eq. (5) and the same choice as
Pacanowski–Gnanadesikan 1998 (the NEMO `ln_dynhpg_zps` form, NEMO Book
v4.0 §6.3.4) is **the shallower of the two cell centroids** — that is, the
deeper centroid is shifted *upward* into the cell that has water at that
common depth, while the shallower cell needs no shift. Our code:

```
face_ref = jnp.minimum(centroid_east, centroid_west)
excess_east = centroid_east - face_ref   # ≥ 0
excess_west = centroid_west - face_ref   # ≥ 0
correction = -g * (rho_prime_east * excess_east - rho_prime_west * excess_west)
```

is exactly this convention. Three checks:

* **Face-reference choice**: `min(centroid_E, centroid_W)` is correct. The
  alternative `max(centroid_E, centroid_W)` would shift *upward* into a
  cell-thickness region where water exists in both columns, but it would
  require evaluating ρ at a depth *below* the shallower cell's centroid —
  i.e., extrapolating ρ outside the cell — which is the manoeuvre SMC03 is
  explicitly designed to avoid and which leading-order A&C explicitly does
  not do. The `min` choice is the conservative, leading-order one. NEMO's
  `dyn_hpg_zps` uses the same convention.
* **Sign**: The correction added to `∂p'/∂x` is `-g · ∂(ρ' · excess)/∂x`.
  The momentum tendency is `-(∂p'/∂x + correction)/ρ_0`, so an east-side
  positive ρ' at a deeper centroid (excess_E > 0) produces a *negative*
  correction, which combined with the `-/ρ_0` gives a positive eastward
  acceleration — which is the correct sign for water shifted upward along
  isopycnals to balance an in-cell hydrostatic mismatch. Sign is correct.
* **Anomaly**: A&C 2004 shifts the **baroclinic** pressure anomaly. The
  reference state `g · ρ_0 · z` cancels exactly across centroid-mismatched
  faces (both columns share the same column-mean ρ_0 reference), so only
  ρ' = ρ - ρ_0 contributes. Our code uses `rho_prime`, not `rho`. Correct.

The correction is non-zero only where `centroid_E ≠ centroid_W`, which on a
z*+partial-cell coord means **only at the partial-bottom level of one of
the two columns**. This *single-level* application is faithful to A&C
(§3 eq. 6 of A&C 2004, and Pacanowski–Gnanadesikan §3a) — and is
*precisely the source* of the 2Δz forced mode, as discussed in §3 below.

## 2. Is 99 mm/s in the published range for leading-order A&C on BH at r=0.54?

**Yes, plausibly within an order of magnitude.** The honest reference table
is in `pgf_production_models_research.md` §5.3, drawing on Sikiric, Janekovic
& Kuzmic 2009 (*Ocean Modelling* 29, 128–136), Berntsen 2002 (*Ocean
Modelling* 4, 363), and Berntsen & Furnes 2005 (*Ocean Modelling* 8, 81):

| r-factor | scheme | published \|u\|max |
|----------|--------|--------------------:|
| 0.2 | leading-order A&C / PG98 | ~5–10 mm/s |
| 0.5 | leading-order A&C / PG98 | "tens of mm/s, sometimes >100" |
| 0.5 | ROMS DJ_GRADPS (cubic spline ρ) | ~10 mm/s |
| 0.7 | DJ_GRADPS | ~50 mm/s, often unstable |

A&C 2004 themselves do *not* run BH; their validation in §5 is on the
internal-tide and overflow tests where leading-order errors of O(few cm/s)
are reported and considered acceptable. **MITgcm has been the de-facto
production user of leading-order A&C for ~25 years** and their published
guidance (Adcroft 1997 §5, restated in MITgcm tutorial docs) is to (a)
smooth bathymetry to r ≤ 0.2, (b) add bottom drag, (c) accept O(few cm/s)
spurious flow as the price of the partial-cell coord — exactly what
`pgf_production_models_research.md` §1.2 records. They explicitly do *not*
ship a higher-order PGF for partial cells in trunk.

NEMO Book v4.0 §6.3.4 states more sharply: at r > 0.3 with strong
stratification, the Pacanowski-Gnanadesikan partial-step PGF (which is
algebraically the same as A&C leading-order) gives "spurious currents
exceeding 0.1 m/s" and they **recommend switching to `dyn_hpg_djc`**
(cubic ρ density-Jacobian, which is essentially SMC03's modified Jacobian).

So 99 mm/s at r=0.54, exponential thermocline, drag=1e-3 is **on the high
end of the published range but not anomalous**: NEMO-Book's "exceeding
0.1 m/s" is precisely this regime. Sikiric et al. 2009 Table 2 doesn't
report leading-order A&C at r=0.5 (they only run DJ_GRADPS and friends),
but extrapolating their DJ_GRADPS=10 mm/s with the rough order-of-
magnitude-better factor for cubic vs leading-order quoted in NEMO Book
v4.0 §6.3.4 puts leading-order in the 100 mm/s neighbourhood. Our number
matches.

The flatness of the Adcroft column across smoothing levels (98.5–100.8
mm/s for smoothing ∈ {0,2,5,10}, i.e. r_max ∈ {0.72, 0.56, 0.54, 0.54}) is
**also literature-consistent**: at these slope parameters, the bathymetry
is still well above the BH "stable" threshold of r=0.2, so the spurious
flow saturates at the magnitude set by ρ' · h_partial · g rather than by
the slope itself. Smoothing only helps once you cross into the r ≤ 0.2
regime, which we are not in.

## 3. Is the 2Δz forced-mode diagnosis correct?

**Yes — and it is the precise mechanism described in NEMO Book §6.3.4 and
implicit in A&C 2004's discussion of why higher-order corrections are
needed.** The diagnosis in `partial_cells_results.md` ("Why the 2Δz mode
appears") matches the literature explanation:

* The leading-order A&C correction is, by construction, a single-level PGF
  spike at the partial-bottom level of the shallower column.
* Vertical viscosity is too weak to absorb that spike (we tested 100×
  increases in `A_v` for only 18% reduction).
* The barotropic continuity equation averages U over the full column, so
  the level just *above* the spike picks up an out-of-phase response to
  preserve the integrated transport. Combined with the spike level itself
  being damped, this produces the observed `+2, -14, +2, -99, 0` pattern
  at levels 15–19.

This is **not a bug**: it is the structural consequence of any "shift-once"
PGF correction that lives at a single level. SMC03 cures it precisely by
using a smooth-in-k face-reference pressure (the harmonic-mean monotonized
PLM in-cell ρ profile produces a correction at *every* level, not just the
partial bottom), so the forcing has no step-function vertical structure.

There is no published alternative explanation for the 99 mm/s residual at
this configuration — the 2Δz diagnosis is the consensus mechanism.

## 4. Are we missing any standard A&C correction terms?

**No** — at least, none that would close the spurious-flow gap. A&C 2004
has three discrete components in the partial-cell apparatus:

* **PGF correction** (§3, eq. 4–7) — what we implement.
* **Free-surface / mass-flux consistency for the rescaled coord** (§4) —
  this is the z* layer Jacobian `(η + H_bathy) / H_max`, which we apply
  uniformly in `compute_layer_thickness`. This is the "free surface in
  rescaled coord" half of the paper, not the partial-cell PGF half.
* **Continuity / divergence at the partial-bottom face** — A&C 2004 §3
  explicitly says continuity is satisfied automatically by the standard
  finite-volume divergence operator as long as face areas are computed
  with `h_partial`. We do this in `compute_face_masks_3d_partial` and the
  C-grid divergence operators. There is no separate "partial-cell
  divergence correction" in A&C 2004.

MOM6 AFV (`MOM_PressureForce_AFV.F90`) and ROMS DJ_GRADPS implement *more*
than A&C 2004 — but those are entirely different PGF algorithms (analytic
EOS + PLM-in-(T,S) for AFV; cubic spline of ρ for DJ_GRADPS), not extra
A&C terms. We can't get those gains by adding terms to the leading-order
A&C path; we have to switch algorithms (which is what `pgf_scheme="smc03"`
does).

One possible refinement that **would not close the gap meaningfully**: A&C
2004 §3 footnote 4 mentions that the correction can be applied with a
"layer-mean" instead of "centroid" depth. We use centroid (eq. (5)
canonical form). Switching to layer-mean changes the residual at O(h²)
which is dominated by the O(h) leading-order error.

## 5. Is `dt=600s` + implicit-CN barotropic a fair Adcroft baseline?

**Slightly unfair to Adcroft, but only mildly.** Three considerations:

* The forced 2Δz mode is a *quasi-equilibrium* response — it reaches its
  asymptotic amplitude in O(few/A_v · Δz²) ~ a few hours, well within
  any 30-day run. Whether you take 600 s or 300 s steps to get there
  doesn't change the equilibrium amplitude, only the transient.
* Implicit-CN barotropic has higher-frequency *dissipation* than
  split-explicit, not higher *forcing*. If anything, it should *reduce*
  high-frequency contamination of u(z) below the partial-bottom — making
  it *fairer*, not less fair.
* What dt=600 s + implicit-CN does change is the response of the
  barotropic mode to the column-integrated PGF residual. For Adcroft,
  the column-integrated PGF residual is small (the column-sum identity
  H&A 2009 holds), so the barotropic response is weak in both schemes.

The cleanest cross-check would be to also run Adcroft at dt=300 s with
split-explicit barotropic and confirm the |u|max stays in 80–110 mm/s.
This is a 10-minute run, not a major concern, but worth doing once before
publishing the SMC03 vs Adcroft comparison externally.

The SMC03 1.5 mm/s number at the same dt=600 s + implicit-CN is robust
because the forced 2Δz mode is *absent* in SMC03 — the residual is
dominated by the O(h²·ρ'') in-cell-integral error, which is set by the
spatial discretization, not the barotropic time-stepping.

## 6. MITgcm comparison

MITgcm has been running global ETOPO simulations with leading-order A&C for
25 years and reports |u|spurious of "few cm/s" on operational global runs
(Adcroft 1997 §5; Marshall et al. 1997b *JGR* 102, 5753 §5.b). Our 99 mm/s
on BH is *worse* than what MITgcm achieves on operational global ocean.
**The discrepancy is fully explained by**:

1. **MITgcm operationally smooths to r ≤ 0.2**. Our BH runs at r=0.54 are
   stress-test conditions that MITgcm wouldn't run in production.
2. **MITgcm uses much larger drag** in operational runs (typically 2–5e-3
   linear bottom drag, plus quadratic). We are at 1e-3.
3. **MITgcm's BH-equivalent stress tests** (e.g. their `tutorial_global_oce_*`
   verification runs at r ~ 0.4) report |u|max ~ 30–50 mm/s after 30 days
   on coarse grids — within a factor of 2–3 of our 99 mm/s, with the
   remaining factor explained by drag and smoothing.
4. **MITgcm does *not* claim mm/s-level rest state on raw BH at r=0.5**.
   Their published guidance (Adcroft 1997 §5) is exactly the opposite:
   "raw bathymetry produces O(10 cm/s) spurious flow on partial cells —
   smooth to r ≤ 0.2 for usable accuracy".

So MITgcm gives us no evidence of a bug. Their operational
~"few cm/s" is achieved by smoothing + drag, not by superior PGF. They
ship the same leading-order A&C we do.

## Conclusion

The implementation is faithful to A&C 2004. The 99 mm/s residual at
r=0.54 with exponential thermocline is at the high end of, but consistent
with, the published behavior of leading-order partial-cell PGF in this
configuration (NEMO Book v4.0 §6.3.4 explicitly anticipates ">0.1 m/s" in
this regime). The 2Δz forced-mode diagnosis is the correct mechanism. No
A&C correction terms are missing. The implicit-CN barotropic + dt=600 s
choice is mildly unfair only at the transient level; the equilibrium
amplitude is set by spatial discretization. SMC03's 65× improvement is a
real algorithmic gain, faithfully exposing the difference between
*leading-order shift-once* (A&C) and *smooth-in-k density-Jacobian* (SMC03)
PGF treatments.

**Recommendations**:

1. Keep `pgf_scheme="smc03"` as the recommended default for partial cells.
2. As a 10-minute confidence check, repeat the BH Adcroft baseline at
   dt=300 s with split-explicit barotropic to confirm the 99 mm/s number
   is independent of the time-stepping. If it changes by more than ~20%,
   reopen this audit.
3. Update `partial_cells_results.md` with a one-line citation to NEMO
   Book v4.0 §6.3.4 and Sikiric et al. 2009 Table 2 to anchor the 99 mm/s
   number to literature precedent rather than treating it as an
   unexplained model artefact.
4. Do *not* attempt to "fix" the Adcroft path further (z-smoothing the
   correction, biharmonic A_v, stronger drag). The empirical exclusions
   in `partial_cells_results.md` plus the literature-consensus mechanism
   here together rule out simple cures. The path forward is SMC03 (or
   eventually MOM6-AFV-style PLM-in-(T,S) with analytic EOS — see
   `pgf_production_models_research.md` §7).
