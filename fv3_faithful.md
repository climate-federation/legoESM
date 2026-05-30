# FV3-faithful cubed-sphere — change log (shrunk @ iter 16)

Goal: cube FV core faithful to GFDL FV3 across atmosphere+ocean (SW→AMIP/OMIP),
matching MPAS/ico + lat-lon FV, **zero cube edge artifacts**, visual + quantitative,
codex-reviewed. CPU only (`JAX_PLATFORMS=cpu`; Metal broken). Per-iter detail in
git `8f693690..HEAD`. Oracle `../Code/FV3/...` STILL TCC-blocked (copy to
`/tmp/fv3oracle` for line-by-line fidelity).

## ✅ HEADLINE: the cube PE atmosphere edge imprint is ROOT-CAUSED + FIXED
**Root cause (FV3-faithfulness gap):** `_step_cell_centre`/`tendencies()`/diffusion
converted cell-centre winds → D-grid corners by treating the **vector** winds as
**scalars**, blending face-local (u,v) across panel seams WITHOUT rotation. FV3
rotates them. ⇒ winds 167×/83× rough at seams → vorticity 229× rough → the
growing spurious v on a balanced jet. Ruled out FIRST (not the cause): damping,
halo/duogrid, IC/perturbation, the PGF (Lin-PGF test = no effect).
**Fix** (7 sites, commits ac6a8f58…4fd71710; 6 codex reviews → APPROVE):
`center_to_dgrid_vector` for the 3 wind/tendency lifts; `pad_halo_vector_4d` for
the diffusion inner+outer halos; duogrid args. Regression test (4deb66ae).
**Validated:**
| metric | before → after |
|--------|----------------|
| baroclinic v_rms@2d | 4.95 → **0.60** (stops growing) |
| **convergence** | anti-converged 3.24→3.91 → **converges** 0.688→0.535→0.282 (accelerating ⇒ ~2nd order ⇒ very close to latlon at AMIP res) |
| mass drift | 2.6e-11 → **1.1e-15** (10⁴×) |
| gravity_wave max\|v\| | 22.4 (outlier) → **19.5 ≈ ico 20** |
| v-field (visual) | wavenumber-4 panel blocks → **smooth zonal bands** |
| vs latlon-hybrid | cube **13× cleaner** (latlon-hybrid has its own equatorial artifact) |
ALL fast PE cube cases PASS; HS climate physical (286K); SW W2 sentinel 13/13.

## ✅ OCEAN cube fix (codex-approved, commits 5f425042, 47d21e83)
Was NaN-blowing-up; defaulted cube→FC-Gram backend in `_create_ocean_setup`,
decoupled FC from density-test heavy diffusion (`cube_fc_light_diffusion`), opted
IGW out. rest_state ×4 NaN→PASS. (FC over-damps fast barotropic waves — right for
density/rest, wrong for pure-wave cases.)

## Verdict on the 3 user-flagged SW visual concerns
1. cosine-bell day-1 = bulk PPM-limiter diffusion in the panel INTERIOR (cube/ico
   L2 1.4×), NOT a grid artifact; high-lat scatter = cosmetic native render.
2. W5 propagates ≈ latlon/MPAS by day 15; ~12% day-1 deficit (damping). lowering
   damping blows up.
3. cross-grid longitude alignment NOW TRULY FIXED (commits 2d04a168, 882b4063,
   418cc7de; codex re-review: "no real correctness issues", MPAS re-regrid
   identity max-err 0.0). Was: latlon stored native 72x144 under a 360-pt label
   (gross tens-of-deg translation); gaussian/spectral kept native ~64 lon →
   (181,64) narrower+shifted; npz lon + icosa weights were node-centered while
   cube data is CELL-centered (half-cell drift). Fix: one canonical canvas
   (`_canvas_lat`/`_canvas_lon`, == regridding.py:457) for EVERY regrid + npz
   label + plot axis. SW cosine_bell/W5/W2 all 4 grids co-located within 1 cell
   (visual confirmed). Also fixed a HARD bug: `run_cosine_bell` imported a
   non-existent `_conservation_accumulator` → every latlon SW case ERRORED before
   stepping (latlon SW was silently 0% running). Now 16/16 SW PASS all grids.
   Regression: `tests/test_latlon_regrid_alignment.py` (9 tests).

## Residuals (all distinct from the now-fixed panel-edge artifact)
- baroclinic + topography (rest_state mountain) = SMOOTH CONVERGING gradient-
  truncation (Lin-PGF tested = no help ⇒ not the PGF operator; only the FC-spectral
  gradient build — large/uncertain — shrinks them at coarse C36).
- SW cube W2 (0.09) = separate FV3Edge path, untouched.

## Other fixes
- `2b409ad8` native cube velocity snapshot plots were BLANK (size mismatch) — the
  visual-inspection tool itself; now render `u_cc_east/v_cc_north`.
- `81a5d021` AST-guard regression fix; SW/`LEGOESM_*` probe env knobs (inert).

## Open / next (separate / polish / breadth)
- FC-spectral gradient build (shrinks the smooth truncation residual at coarse res;
  large, uncertain; cube already very close at AMIP res).
- MPI vector packed-halo (single-device unaffected). latlon-hybrid equatorial
  artifact (latlon path, separate). Broad HS/AMIP cross-grid climate (latlon HS +
  ico baroclinic pathologically SLOW on CPU — not bugs, just slow).
- `DONE` withheld: smooth residual + separate items remain; the panel-edge artifact
  (the user's core concern) IS eliminated + the cube is a consistent FV3-faithful
  discretization.
