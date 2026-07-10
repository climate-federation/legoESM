# Pseudo-incompressible vs spectral LES: the "too smooth" gap (2026-07-10)

The pseudo-incompressible plane LES core produced visibly **smoother** (over-diffused)
turbulence than the pseudo-spectral core. Diagnosis + fix below.

## Root cause — WENO5 momentum advection (not a bug)

The pseudo core advects **momentum** with upwind-biased **WENO5** (`plane_fd_advection`),
which carries inherent high-wavenumber (~k⁶) numerical dissipation that acts directly on
the resolved velocity → it smooths the LES eddies. The spectral core advects momentum
with the **energy-conserving rotational form** (zero numerical dissipation, only 3/2
dealiasing). No over-diffusion *bug* exists — the effect is by construction (confirmed by
codex review + the SGS length scale / single-application checks).

## The viz comparison was unfair (controlled A/B)

Matched-config A/B (48³, neutral geostrophic, same forcing), resolved w-variance `ww_max`:

| SGS (matched) | pseudo (weno5) | spectral | note |
|---|---|---|---|
| vreman | 0.133 | 0.148 | **already agree** |
| LASD | 0.153 | 0.27–0.34 | pseudo capped by WENO5 |

So at matched SGS the cores agree under vreman. The snapshot "too smooth" impression came
from an **unfair comparison**: the viz ran spectral with **LASD-dynamic SGS at 48³** vs
pseudo with **vreman at 40³** (different SGS *and* resolution). The real residual is that
WENO5's numerical diffusion becomes the *limiting* dissipation once the SGS ν_t is small
(LASD) — capping the pseudo core's resolved energy (0.153 vs spectral ~0.27–0.34).

## Fix — a `momentum_scheme` knob (default weno5, bit-identical)

Added less-/non-dissipative **momentum** advection options (scalars keep monotone WENO5):

- **`weno7`, `weno9`** — wire the existing `core/weno.py` `weno7_z`/`weno9_z`; less upwind
  dissipation than weno5 (proven: `weno9 < weno7 < weno5` dissipation, see the
  non-dissipation test). Still upwind ⇒ self-stabilizing.
- **`central`** — 2nd-order central, **non-dissipative** (mirrors the spectral rotational
  form). Sharpest, but carries no numerical dissipation ⇒ relies on the SGS for 2Δ control.

`momentum_scheme` config field (validated), driver `--momentum-scheme`. Default `weno5` ⇒
byte-identical. The MPI y-halo width is now derived from the scheme reach
(`_halo_width`: weno5=3, weno7=4, weno9=5) instead of hard-coded 3.

## Stability / usage tradeoff (measured)

Non-dissipative sharpness trades numerical stability:

- **`central` alone overshoots** — at 48³ it becomes 2Δ-noise-dominated (hi-k energy
  fraction ~0.6, `ww` inflated to ~0.47) because the SGS ν_t is too weak to control the
  grid mode. `central` + biharmonic `hyperdiff` **fails** (NaN — the coeff needed is above
  the biharmonic's own CFL stability window, same lesson as the fine-res gabls1 work).
- **`central` + a CFL-unlimited VELOCITY de-noiser — SOLVED.** Added
  `momentum_shapiro_coeff` (blend) + `momentum_shapiro_order` (Shapiro order): a per-step
  `[1,2,1]`-family horizontal low-pass on u,v,w. It **commutes with the C-grid divergence**
  (horizontal periodic convolution), so `div(H·u)=H(div u)=0` ⇒ divergence-free is
  preserved with **no re-projection** (proven to 1e-10). A single `[1,2,1]` (order 1)
  **over-damps** (its passband response <1 compounds over thousands of steps, eroding the
  eddies to laminar), so a **high-order** Shapiro (`order=8`, response `1−sin¹⁶` ≈ flat in
  the passband) is required. Result (LASD, 48³): `central` + `momentum_shapiro_coeff≈0.3–0.5,
  order=8` gives **ww 0.28–0.30, hi-k 0.008–0.03 — matching spectral (0.32) and clean**
  (vs raw central 0.47/0.59). This is the FD analogue of the spectral core's
  non-dissipative-advection + sharp-filter recipe. Use in f64.
- **`weno7`/`weno9`** are sharper than weno5 and self-stabilizing in **f64**, but their
  reduced dissipation makes them **f32-fragile** (they can NaN on marginal cases where
  weno5 holds). Use f64.

## Recommendation

- **Closest match to spectral (f64):** `momentum_scheme="central"` +
  `momentum_shapiro_coeff≈0.4`, `momentum_shapiro_order=8` — non-dissipative advection +
  the CFL-unlimited velocity de-noiser reproduces the spectral core's recipe and lands on
  its resolved energy (ww ≈ 0.29 vs 0.32).
- **Simpler, self-stabilizing (f64):** `momentum_scheme="weno7"` (or `weno9`) — sharper
  than weno5 with no de-noiser to tune (but f32-fragile).
- The cores already agree at **matched SGS + resolution** with the default `weno5` —
  reproduce the spectral result by using the same SGS (e.g. both LASD) and grid, not by
  comparing mismatched configs (the original "too smooth" artifact).

Codex adversarial review: no defects beyond the MPI halo-width fix (applied). Tests:
`test_plane_fd_advection.py` (invariants for all 6 schemes + the non-dissipation ordering).
