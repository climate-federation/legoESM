# Plane-LES resolved-turbulence notes (laminar-collapse fix)

Diagnosis + fix for the `scripts/run/run_les_plane.py` boundary-layer LES not
resolving turbulence (validated against the **jax-alfa** oracle; Bou-Zeid LASD
SGS, see `_compute_scale_dependent_dynamic_smag_cs_plane`).

## Symptom
GABLS1/Wangara runs collapsed to a 1-D column: `max|w| ~ 0.02 m/s`,
`wvar ~ 1e-4 m²/s²` — i.e. **zero resolved turbulence**, plus (early versions)
the geostrophic wind spinning down to < 1 m/s.

## Root causes (all fixed)
1. **Double vertical mixing.** The dycore 3-D SGS
   (`sgs_vertical_diffusion=True`) already carries the resolved vertical SGS
   flux; the driver's `_apply_pbl_column` added a *second* full-column Louis K
   (≤ 20 m²/s) every step → homogenised the column and erased the eddies. Fixed:
   the column is now **surface-flux-only** — a thin `K_sl = κ u_* z f(Ri)
   e^(−z/40 m)` that deposits the surface stress/heat/moisture through the
   under-resolved surface layer, plus a small uniform `K_floor` for the column
   2Δz mode. Resolved eddies + the 3-D SGS set the mixed-layer structure.
2. **Deep surface coupling spun the wind down.** `h_sl = 100 m` spread the
   surface drag through a deep layer (8 → < 1 m/s). Fixed: `h_sl = 40 m` (thin).
3. **Explicit vertical-SGS CFL blow-up with LASD.** `_vertical_K_diffusion_full`
   is explicit forward-Euler (`K < 0.5 dz²/dt ≈ 45 m²/s` near the surface).
   With `--scale-dependent` the Mason κz wall cap was off ⇒ uncapped near-wall
   `K = Cs·Δ` overshot the limit and blew up at ~0.08 h. Fixed: keep
   `smagorinsky_wall_damping=True` even for LASD (β still governs aloft).
4. **w-damping annihilates the turbulence (the decisive one).** The
   semi-implicit acoustic **off-centring** `β` and the **`si_w_filter`** damp `w`
   on *every* acoustic substep. `w'` is the carrier of BL turbulence, so any
   nonzero value makes `w'` decay monotonically regardless of seed/dissipation:
   the seed dies before the mean shear can amplify it. With
   `--off-centering 0 --si-w-filter 0` (and a smaller `dt` for the vertical
   acoustic CFL) `w'` instead **grows and sustains**.

## IC that lets turbulence spin up
- **Log-law mean wind** for the neutral case (shear present from t=0 ⇒ immediate
  resolved shear production) instead of a uniform profile.
- **Large-scale (spectral low-pass) velocity seed** — a white-noise seed is
  dominated by 2Δ energy that the biharmonic hyperdiff annihilates in a few
  steps; a smooth seed survives to be amplified.

## Working recipe (neutral ABL)
```bash
JAX_ENABLE_X64=1 .venv/bin/python scripts/run/run_les_plane.py --case neutral \
    --scale-dependent --off-centering 0 --si-w-filter 0 --hyperdiff 0.001 \
    --dt 0.05
```
Confirmed: `w'` grows (`max|w|` 0.044 → 0.062 m/s, `wvar` 4e-4 → 7e-4) over 0.1 h
on a 24³ CPU run — qualitatively opposite to the earlier monotonic decay.

## Status / remaining
- Mechanism fixed; turbulence **sustains** instead of decaying.
- Reaching a statistically-steady BL (`w' ~ 0.4 u_*`, log-law, resolved
  spectra) needs **multi-hour simulated time** = long GPU runs (CPU does
  ~21 steps/s). Quantitative oracle comparison (mean profiles, variances,
  Cs²(z), spectra vs jax-alfa) is the next step on GPU.
- The SBL (GABLS1) is the hardest case (weak, intermittent turbulence at
  Δ=12.5 m); expect it to need the finest grid + longest spin-up.
