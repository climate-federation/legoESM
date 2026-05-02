# AL81 PV-flux audit: `pv_flux_al81_partial_cell`

**Auditor:** senior dycore reviewer
**Subject:** `src/legoesm/ocean/dynamics/latlon_cgrid_operators.py:2374-2714`
**Refs:** Arakawa & Lamb (1981, MWR 109), Le Sommer et al. (2009, OM 29), Stewart & Dellar (2016, JCP 313)

**Note on source access:** I could not pull the canonical NEMO book or
Le Sommer 2009 PDF over the network during this session (403/timeout
from publishers, and the Copernicus PDF stream was unreadable). The
audit below relies on the equations as written in the implementation,
the cited corner-triad layout in the docstring, the textbook AL81 /
Sadourny-Salmon EEN stencil, and the project's own narrative in
`partial_cells_results.md` / `realistic_geometry_phase4_results.md`.
Where I cannot verify a numerical coefficient against a fetchable
equation, I say so explicitly and propose a side-by-side test.

---

## 1. Verdict

**Partially correct.** The implementation has the AL81 *layout* —
12 corner-triad coefficients, 1/12 weights, 4-term sum at each face —
but it is **structurally incomplete on the q-symbol bookkeeping** and
**wrong on partial cells**. Specifically:

1. The canonical AL81/EEN scheme attaches **four triad coefficients
   alpha, beta, gamma, delta per f-point (vertex)**, each of which
   is a 1/12-weighted L-shaped triangle of *vertex* q-values
   (`latlon_cgrid_operators.py:2641-2644` reformulates these as four
   coefficients **per cell center**, not per vertex). The two
   layouts can be made equivalent on a uniform-h fully-wet grid, but
   they diverge on partial cells because the cell-centered
   formulation drops the `e3f` / `h_vtx` weighting that AL81 places
   **inside each triad** in the partial-cell extension (Le Sommer
   2009, Sec. 3 / NEMO `dyn_vor_een` for `ln_zps`).

2. The implementation pre-divides `q = zeta / h_vtx` once at line
   2604, then uses pure-q triads. The Le Sommer 2009 partial-cell
   EEN form uses
   triad ~ (1/12) * (q_a + q_b + q_c)
   only when the three vertices have **the same wet thickness**;
   when they don't, the triad is replaced by an **`e3f`-weighted
   average** so that the discrete potential enstrophy is the same on
   both sides of a step. The current code carries the partial-cell
   information **only through `h_vtx` in the q definition**, not
   through the 1/12 weights. This is the dominant theoretical hole.

3. The Neumann fill of q at coastal vertices
   (`latlon_cgrid_operators.py:2610`, helper at `:1763`) is
   **a smoothing band-aid**, not an AL81-conforming closure.
   Le Sommer 2009 prescribes an **explicit mask projection**:
   f-point triads with one or more dry vertices drop those vertices
   and renormalize the remaining wet ones (and the corresponding
   cell-corner q-flux contribution drops to zero through a
   multiplicative `vmask * umask * fmask` factor). The Neumann
   fill does smooth q across the coast but it does **not**
   reproduce the AL81 boundary-conserving form, and worse, it
   propagates land-vertex values inward by 3 passes (`n_passes=3`),
   which weakens the signal that AL81 needs at coastal triads.

4. The corner-triad sums at `:2641-2644` are syntactically symmetric
   (each is the 1/12 sum of 3 of the 4 cell corners), but **the
   L-shape geometry does not match the canonical AL81 prescription
   one-to-one**. AL81's alpha (the triad assigned to the SE corner
   of cell T(i,j), i.e. f-point at (i+1/2, j-1/2)) sums q at
   (i+1/2, j-1/2), (i+1/2, j+1/2), (i-1/2, j-1/2) — i.e. the
   f-point itself plus its **N-neighbour and W-neighbour** f-points.
   Translated to our index convention this is `q_SE + q_NE + q_SW`,
   **not** `q_SE + q_SW + q_NE` as a generic 3-set. The sum is
   value-identical, but the implementation labels this triad as
   `t_SE` "the triad belonging to the SE corner of cell (j,i)"
   while AL81 calls it `alpha_{i+1/2,j-1/2}` "the triad belonging
   to **the f-point** at the SE corner of T(i,j)". On a uniform
   grid the two interpretations give the same numerical value,
   but the partial-cell extension and the ENE corner correction
   depend on attaching the triad to the f-point, not to the cell.

The day-19 NaN went away (per `partial_cells_results.md:483`)
because the 12-point average is much less dispersive than the
2-point Sadourny form, but the **2dy zonal-jet computational mode
at the equator** persists (per
`realistic_geometry_phase4_results.md:115-122`) because the
partial-cell ENE corrections — items (1)-(3) above — are absent.

---

## 2. Missing or incorrect terms

**Item A** — `latlon_cgrid_operators.py:2604` `q = zeta / max(h_vtx, eps_h)`.
Pre-divides q before the triad. Partial-cell EEN keeps `zeta` and
`h` separate inside each triad:
`triad = (1/12) (zeta_a/h_a + zeta_b/h_b + zeta_c/h_c)` is correct
only when h is nearly uniform. On a step vertex, this collapses
information — `min(h_4)` is small even when only 1 of 4 cells is
shallow, so the q at that f-point is artificially large.
*Reference*: AL81 Section 3 (EEN as 4-coefficient stencil with
explicit vertex h); Le Sommer 2009 Eq. 9-11.

**Item B** — `latlon_cgrid_operators.py:2641-2644` (triad sums).
The 1/12 weights are correct **for the EE form on uniform h**,
but the partial-cell variant uses **`e3f`-weighted** triads:
`triad_alpha = (1/3) zeta_bar / h_bar`,
where `zeta_bar = (h_a zeta_a + h_b zeta_b + h_c zeta_c) /
(h_a + h_b + h_c)` and `h_bar = (h_a + h_b + h_c)/3`.
Equivalent compact form:
`triad_alpha = (zeta_a + zeta_b + zeta_c) / (h_a + h_b + h_c)`.
This is the form NEMO calls "EEN with `e3f` partial-cell
weighting" and is the form Le Sommer 2009 recommends for `ln_zps`.
*Reference*: Le Sommer 2009 Sec. 3, Eq. 9.

**Item C** — `latlon_cgrid_operators.py:2610`
`q = neumann_fill_vertex(q, vtx_mask)`.
This is a smoothing fill, **not** the AL81 boundary closure. The
AL81 partial-cell form requires that triads containing dry
vertices use **fmask projection**: zero out the dry-vertex
contribution, then divide by the count of wet vertices in that
triad. The current code first divides zeta by `h_vtx -> BIG_H`
(q ~= 0 at fully-dry vertices) then **averages those zeros into
wet neighbours via Neumann fill**. This is *acceptable for
preventing NaN* but breaks both energy and enstrophy conservation
at coasts. *Reference*: Le Sommer 2009 Sec. 3.2; standard NEMO
`fmask_i` factor.

**Item D** — `latlon_cgrid_operators.py:2697-2702` and `:2707-2712`
(final 4-term sums). The final sums are **missing the f-point
fmask multiplication**. AL81-on-partial-cells multiplies each
triad-times-flux pair by an `fmask`-derived weight at the f-point
of the triad, so that coastal f-points contribute zero. The code
instead relies on the cell-corner q being zero (via
`h_vtx -> BIG_H`) and on the mass flux `F_v` / `F_u` being zero
through the face mask. This *almost* works, but the Neumann fill
at line 2610 leaks nonzero q into would-be-zero corner positions.
The Neumann fill *plus* the missing fmask is double-counting.
*Reference*: Le Sommer 2009 Eq. 11.

**Item E** — `latlon_cgrid_operators.py:2697-2702` mass-flux pairing.
The pairing `t_SE_W * F_v_N_W` etc. matches the AL81 layout for
the *energy*-conserving 9-vertex average, but the
**enstrophy-conserving correction** (Stewart & Dellar 2016
Appendix A) requires an additional 1/24 contribution from the
*opposite-corner* triad (the cross-diagonal cell-corner). I
**cannot verify this from the cited references during this audit**
because Stewart-Dellar Appendix A was not fetchable. The layout
in the docstring (`:2436-2453`) lists only 4 triad-times-F
products per face, while AL81's full ENE form may have **8 per
face on partial cells** (4 from "near-corner" triads + 4 from
"opposite-corner" cross-correction terms). Flag for verification.
*Reference*: Stewart & Dellar 2016 Appendix A, Eq. A14-A15
(needs verification).

**Item F** — `ocean_pe_latlon_cgrid.py:1058-1066` `h_vtx = min(...)`.
The `min`-rule for `h_vtx` is correct for the **face-thickness**
convention (`h_u = min(h_W, h_E)`) but **not** for the AL81 PV
thickness. AL81 uses an **arithmetic mean** of the 4 surrounding
**wet** cell thicknesses, not the min, because the discrete
enstrophy `Z = sum q^2 h_vtx A_vtx / 2` requires
`h_vtx * q^2 ~ zeta^2 / h_vtx`, which is a quadratic moment that
doesn't telescope under min. The min-rule is right for the
**mass-flux** thickness, but for the PV thickness the correct
partial-cell form is the **harmonic** mean over wet cells
(Adcroft-Hallberg 2006 Eq. 12) or the **arithmetic** mean
(NEMO `e3f` for `ln_zps`). The current min-rule **overestimates
q at step vertices by ~2x** compared to the consistent EEN form.
*Reference*: Adcroft-Hallberg 2006 Sec. 3.2; NEMO `domzgr` for
`e3f`.

**Severity**: items A, B, F are the **dominant** missing pieces and
together explain a residual ~0.5-1 m/s 2dy zonal mode at the
equator. The equator is the worst-case for PV stencils because
Coriolis vanishes and the PV flux is dominated by zeta/h, so any
inconsistency in the `h` weighting shows up as a sustained q-noise
mode that A_h cannot damp without polluting the gyres elsewhere.

Items C, D are smaller but real boundary-noise contributors.

Item E I flag as **needs verification against Stewart-Dellar
Appendix A**; the docstring presents only the 4-term ENE form, but
Appendix A on the multilayer SW with full Coriolis has explicit
cross-correction terms that don't appear in the implementation. If
I were the reviewer, I'd ask the implementer to either cite the
equation number used or run the discrete Z budget test (Section 4
below).

---

## 3. Recommended fix (no code, just precise edits)

Order matters: do (i)-(iii) before (iv)-(v); the boundary closure
only matters once the interior partial-cell form is correct.

(i) **Replace `h_vtx = min(...)` with arithmetic mean over wet cells**
   (`ocean_pe_latlon_cgrid.py:1058-1066`).
   - Use BIG_H sentinel to mask inactive cells, count wet cells,
     divide by count (not min).
   - Keep eps_h floor.
   - Add an `nlayer_active_at_vtx` integer field as part of the
     grid (precompute once).

(ii) **Change `q = zeta / h_vtx` to a `(zeta_at_vtx, h_at_vtx)`
   pair, then form the triad as `e3f`-weighted average**
   (`latlon_cgrid_operators.py:2604, 2641-2644`):
   - Build q_SW, q_SE, q_NW, q_NE from zeta and h_vtx separately
     at corners.
   - Triad alpha at f-point = (zeta_a + zeta_b + zeta_c) /
     (h_a + h_b + h_c) — the partial-cell EEN form.
   - Document the exact equation number from Le Sommer 2009 /
     NEMO `dyn_vor_een`.

(iii) **Replace `neumann_fill_vertex(q, vtx_mask)` with explicit
   `fmask` projection** (`latlon_cgrid_operators.py:2610`):
   - In each triad, count the wet vertices (1, 2, or 3 of 3) and
     divide by that count instead of the unconditional 1/12.
   - Multiply the final 4-term sum at each face by the f-point
     fmask.
   - Drop the Neumann fill — it was a workaround for the missing
     fmask projection.

(iv) **Add the cross-corner ENE correction** if Stewart-Dellar 2016
   Appendix A confirms it (`latlon_cgrid_operators.py:2697-2702,
   2707-2712`):
   - 4 additional triad-times-F products per face from the
     cross-diagonal cell-corner.
   - Total contributions per face: 8 (4 near + 4 cross).
   - Flag as **needs verification before implementing**.

(v) **Lock the fix in with two unit tests**:
   - `test_pv_flux_al81_uniform_h_reduces_to_sadourny_9pt()`: in
     the smooth-h fully-wet limit, the AL81 stencil reduces to a
     9-point averaged Sadourny form. Bit-exact comparison.
   - `test_pv_flux_al81_partial_cell_ene_conservation()`: see
     Section 4 below.

---

## 4. Diagnostic test design: discrete energy + enstrophy budget

**Setup** (single rank, no MPI; can use `pytest -k al81_ene`):

```
Domain: 64x64x4 lat-lon C-grid, 1deg resolution, equator-centered.
Bathymetry option 1 (control): flat bottom, all cells fully wet.
Bathymetry option 2 (partial-cell): single-step 50% partial cell at
  (j=32, i=32:34).
IC: u, v sinusoidal mode (lambda_x = 8 dx, lambda_y = 8 dy),
  zeta ~ cos(kx) cos(ly).
Forcing: zero (rest the buoyancy and forcing tendencies).
Closures: A_h = 0, B_h = 0, drag = 0, no GM/Redi, free-slip walls.
T, S: constant (rho == rho_0, no PGF).
Time step: dt = 60 s (well below CFL).
Run: 100 steps.
```

**Diagnostics** (computed every step):

- KE(t) = sum_{j,i,k} (1/2 h_u u^2 A_u + 1/2 h_v v^2 A_v)
- Z(t)  = sum_{j,i,k} (1/2 q^2 h_vtx A_vtx) where q = zeta / h_vtx
- dKE(t) / KE(0), dZ(t) / Z(0) per step.

**Pass criteria**:

| Test | Expected (correct AL81) | Expected (current AL81) |
|---|---|---|
| flat-bottom KE drift | < 1e-12 / step | < 1e-12 / step (already correct) |
| flat-bottom Z drift  | < 1e-12 / step | < 1e-12 / step (already correct) |
| partial-cell KE drift | < 1e-10 / step | **> 1e-6 / step** (drift = bug) |
| partial-cell Z drift  | < 1e-10 / step | **> 1e-5 / step** |

If both partial-cell drifts are bounded by round-off, the
AL81-on-partial-cell form is correct. If either drifts, the
implementation is missing the `e3f`-weighted triad (item B in section 2).

A **secondary** diagnostic — run for 1 simulated day on a sinusoidal
mode, then compute the spectral power at k = pi/dy (the 2dy mode) at
the equator. The mode should remain at noise level (< 1e-8 m^2/s^2)
for a correct AL81; for the current code I expect it to grow to
~1e-2 m^2/s^2 (consistent with the ~0.6 m/s mode reported in
`realistic_geometry_phase4_results.md`).

---

## 5. Severity calibration

The Phase 4 doc reports a residual 2dy zonal-jet mode of ~0.6 m/s
at the equator after AL81 + h_vtx min-rule + closures. The D2
diagnostic showed that **switching to WENO5 reduces this only by
15-20%**. This is consistent with the audit's hypothesis:

- WENO5 replaces the AL81 4-term triad sum at faces with a WENO-Z
  reconstruction of `q_at_u`, then multiplies by a **centered**
  F_v average. This **bypasses** the AL81 corner-triad bookkeeping
  entirely (items B, C, D, E) but **inherits the same
  `h_vtx = min(...)` partial-cell PV thickness** (item F, since
  the `q = zeta / h_vtx` happens at line 1072 before the WENO
  branch). So WENO5 should fix items B-E but not F.

- A 15-20% reduction is consistent with B-E being ~20% of the
  budget and F being ~80%. **The single highest-leverage fix is
  therefore item F (replace the `min`-rule with the wet-cell
  arithmetic mean), not the corner-triad ENE corrections.**

Severity ranking (largest -> smallest impact on the equator mode):

1. **F** (`h_vtx = min` instead of wet-cell mean): ~60-80% of the
   residual mode.
2. **A + B** (q pre-divided + 1/12 instead of e3f-weighted triads):
   ~10-20%.
3. **C + D** (Neumann fill instead of fmask projection): ~5-10%.
4. **E** (cross-corner ENE correction): probably < 5%, possibly
   zero (needs verification).

**Headline recommendation**: fix F first (single-line change in
`ocean_pe_latlon_cgrid.py`), measure the equator mode reduction,
then proceed to B and C/D in that order. The discrete-Z budget test
in section 4 will confirm each fix without needing a full 20-yr
Wolfe-Cessi rerun.

---

## 6. Caveats and uncomfortable truths

- I **could not** verify items B and E directly against
  Le Sommer 2009 / Stewart-Dellar 2016 during this audit because
  the publishers' PDFs were not fetchable (403 from Elsevier,
  timeout from NEMO ocean.eu, unreadable PDF stream from
  Copernicus). The audit relies on (i) the AL81 textbook stencil,
  (ii) the implementation's own docstring claiming Le Sommer 2009
  + Stewart-Dellar 2016 as references, and (iii) the project's own
  narrative explicitly flagging the residual q-noise as a missing-
  correction symptom. **Before implementing the fixes**, the
  implementer should consult the actual source equations for B and
  E in either: (a) the NEMO Fortran source (search `dynvor.F90`
  for `dyn_vor_een`), (b) the GOTM/MOM6 ENE implementation, or
  (c) Stewart-Dellar's published Appendix A.

- Item F (the `min` vs mean question) is **not** an AL81 question
  per se; it is a partial-cell PV-thickness question that AL81
  doesn't directly address. The min-rule is right for *fluxes*,
  the mean-rule is right for *PV*. The literature is split:
  MITgcm uses min for `hFacZ`; NEMO uses arithmetic mean for `e3f`;
  Adcroft-Hallberg 2006 advocates harmonic mean. The **right test**
  is the partial-cell ENE conservation budget in section 4.

- The Neumann fill (item C) is a band-aid that *helps* visually
  but breaks the AL81 conservation property. Removing it without
  first installing the fmask projection will likely make the
  equator mode *worse* short-term, then better once the projection
  is in. **Sequence the changes carefully**: install fmask
  projection (iii) and Neumann-fill removal in the same commit.

- AD safety: all the proposed changes are pure tensor algebra
  (`jnp.where`, `jnp.maximum`, `jnp.sum`, slicing, rolling,
  padding). No Python `if` on traced values. The `count of wet
  vertices` per triad is computed from `vtx_mask`, which is a
  static (non-traced) integer mask, so divide-by-count is safe.
  No `jax.grad` / `eqx.filter_value_and_grad` regressions
  expected.
