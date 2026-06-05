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

## Performance (GPU / MPI)
Benchmark: `scripts/bench/bench_les_plane.py` (steps/s, ns/cell/step; CPU or
GPU). SGS-closure relative cost on a 32×32×48 CPU run (ns/cell/step):

| closure        | ns/cell/step | vs static |
|----------------|-------------:|----------:|
| static Smag    |         3472 |     1.00× |
| Germano dynamic|         3998 |     1.15× |
| **LASD (Bou-Zeid)** |    4414 |     1.27× |

LASD adds ~27% over the static closure — the second (4Δ) test filter, the extra
contractions and the per-level β quintic solve. The β `lax.scan` solver is cheap
(only `nz` levels) so the cost is dominated by the FFT-based test filters.

**GPU.** The whole LES path is JIT/x64-clean and host-callback-free (FFT,
`lax.scan`, `vmap`, `complex128`) ⇒ GPU-ready. The legoESM venv is CPU-only;
to run on GPU: `uv pip install jax-cuda12-plugin==0.10.0` then
`JAX_PLATFORMS=cuda` (auto-detect falls back to CPU silently).

**MPI.** The dynamic closures (Germano and LASD) are **single-rank**: the
spectral test filter, planar mean and 3×3 local average act over the LOCAL
horizontal tile, so under a horizontal MPI decomposition they are per-rank
(wrong — the test filter and plane average need the global field / halo
exchange). MPI LES must use the **static** closure (`--static-sgs`), which is
validated serial==MPI. A faithful MPI LASD needs the spectral filter +
local average halo-exchanged (distributed FFT / haloed Imfilter) — future work.

## float32 + the time integrator (long runs)
**float32 is the single biggest win** (12.6× on an RTX 5090). The state, grid and
height-coordinate must share ONE dtype — a latent mismatch (grid float32, hc
float64) was silently upcasting the whole state to float64 every step. Fixed via
`--f32` (build grid+hc+state in float32). `--f32` 96³ LASD: 47.9 steps/s
(23.6 ns/cell) vs float64 3.8 steps/s (296 ns/cell). A 1 h sim ⇒ ~25 min wall.

**Time step.** The compressible dycore is acoustically limited, but the binding
constraints here are the horizontal-acoustic substep (`c·dt/n/dx < 1` ⇒
dt ≲ n·dx/c) and the off-centred (β=0) vertical SI. A dt sweep (off-centring 0)
runs stably to **dt=0.2 s**, but dt=0.2 produces a transient *numerical*
turbulence burst (wvar→0.23) that then collapses with the mean wind — not
physical. **dt≈0.05–0.1 s** is the reliable range (dt=0.1 ⇒ 2× fewer steps).
Genuinely larger dt needs an anelastic/incompressible reformulation (removes the
acoustic constraint, ~10–100× larger dt) — the oracle's solver class, and the
architecture option that was deferred.

`bench_les_plane.py --scan` runs the whole trajectory as one jitted `lax.scan`;
once in float32 the run is compute/memory-bound (not dispatch-bound) so scan is
~neutral at 96³ — but it requires the dtype consistency above.

## Multi-hour spin-up result (the open realism gap)
A **3 h** neutral run (64³, dx=20 m, dt=0.05, float32, GPU, 148 steps/s) shows
the resolved turbulence **decaying monotonically to zero** (max|w| 0.67 → 0.000,
wvar → 0; mean wind recovers to laminar 10.0). At dx=20 m / 64³ the compressible
dycore's numerical + SGS dissipation overwhelms the resolved shear production and
the BL laminarises. The dt=0.2 burst shows the dycore *can* momentarily hold
strong turbulence (wvar~0.23), so the limiter is sustained production vs
dissipation, which is **resolution-dependent**. The jax-alfa oracle runs
128³–384³ at dx≈2–8 m with a (near-non-dissipative) pseudo-spectral solver.

**128³ / dx=10 m + reduced hyperdiff (0.0003) CONFIRMS the limit:** the seed
decays identically (wvar 0.036 → 0.0001 in ~80 s sim → 0), the same as 24³ and
64³. So it is **not resolution** — the limiter is the compressible scheme's
numerical dissipation (semi-implicit acoustic off-centring + biharmonic hyperdiff
needed for stability), which caps the EFFECTIVE Reynolds number below the
turbulence-sustaining threshold: the log-shear flow is linearly stable at this
effective Re and any seed decays instead of transitioning. The dt=0.2 burst is
the only time resolved variance grows, and that is a numerical transient, not
sustained physics.

## Pseudo-spectral incompressible core (the faithful path) — status
Built in `spectral_les_plane.py` (driver `scripts/run/run_spectral_les.py`):
rfft2 horizontal + FD-staggered vertical, rotational de-aliased advection,
fractional-step pressure projection, wall-damped Smagorinsky, MOST wall model,
AB2. Numerics validated (`tests/unit/test_spectral_les_plane.py`, 4 pass):
divergence-free projection to 1e-10, exact spectral derivative, energy-conserving
advection, finite step.

**It does what the compressible core could not:** resolved turbulence SUSTAINS to
a statistically steady state instead of decaying (neutral channel, 64³, RTX 5090,
**1200 steps/s** — ~8× the compressible core), and the resolved-stress
**u_*≈0.35 m/s matches the MOST target** (`validate_les_vs_oracle.py`).

**Open issue (localised to the advection):** the equilibrium turbulence is ~5×
too energetic (σ_w/u_*≈6 vs 1.25) and the mean profile INVERTS (U highest at the
surface, ~13 m/s, vs the correct log shape). Debugged by elimination:

* The over-energy is robust to SGS magnitude (C_s 0.16→0.25 cuts variance only
  ×1.6 — variance ~ ν_t^−0.5, not a closure-magnitude knob), to dt (0.2→0.05, no
  change), to the wall-model form (local vs Moeng planar-mean drag), and to a
  top Rayleigh sponge — so it is none of those.
* **Diffusion + wall + forcing ONLY (advection disabled) gives a clean monotonic
  log profile** (U 3.3→10.4 m/s, u_*≈0.30) — so the wall stress, the SGS vertical
  diffusion and the mass-flux forcing are all CORRECT.
* ⇒ the culprit is the **rotational advection on the staggered grid**: it
  transports momentum COUNTER-gradient (piling it at the surface) and pumps the
  resolved variance. Fixed one real sign bug there (the RHS must be +(u×ω); the
  −(u×ω) sign still passes the ⟨u·C⟩=0 energy test because u×ω⊥u, but reverses
  the cascade) — necessary but not sufficient.

Fixes applied to the advection (both correct + more faithful, neither sufficient
alone): the +(u×ω) SIGN, and forming the ``w·ω`` products at the FACES then
averaging the PRODUCT to centres (``f2c(w·ω_face)``, not ⟨ω⟩⟨w⟩ — the
energy/transport-correct staggering, matching the oracle's ``StagGridAvg`` on the
product). The over-energy is robust through all of these.

Remaining suspects (in priority order), still under investigation:
1. **De-aliasing method** — the oracle uses 3/2 PADDING (`Dealias1`/`Dealias2`:
   pad to 3/2 grid, multiply, truncate). This module uses 2/3 TRUNCATION of the
   inputs+outputs. The 2/3 rule is alias-free for a single quadratic product, but
   the rotational form chains several and the staggered vertical averaging is not
   spectral — residual aliasing could pump the variance. Port the 3/2 padding.
2. **Exact boundary rows** of `Advection_Dealias` (`cc[0]=arg1[0]+0.5·arg2[1]`,
   `cc[nz-1]=arg1[nz-1]+arg2[nz-1]`) vs the edge-pad here.
3. **Forcing sets the ABSOLUTE level, not the intensity.** Tested per-step exact
   bulk re-pin, gentle ⟨u⟩-relaxation, and a slow INTEGRAL-CONTROLLED body force
   (now the driver default — cleaner than the re-pin). The absolute wvar tracks
   the maintained wind (wvar≈44 at ⟨u⟩≈11; wvar≈0.9 at ⟨u⟩≈1.8), BUT the
   INTENSITY ratio σ_w/u_* stays ≈10–14 at every operating point. So the forcing
   is not the cause; the invariant ~10× over-intensity is.

**RESOLVED — it was de-aliasing + (then-masked) SGS magnitude.** Porting the
oracle's 3/2 zero-padding de-aliasing (`_pad_to_fine`/`_truncate_from_fine`,
verbatim `Dealias1`/`Dealias2`) cut σ_w/u_* from 8 to 3.3: the chained rotational
advection's quadratic-interaction aliasing (which 2/3-truncation does NOT remove)
was ~40% of the over-energy. With the aliasing gone, the Smagorinsky C_s finally
controls the level (it could not before — aliasing dominated), and **C_s≈0.30 +
3/2 de-aliasing gives a LOG profile and 3/4 MOST diagnostics passing** on a 64³
neutral channel:

| diagnostic | value | MOST target |
|---|---|---|
| u_*(log)/u_*(flux) | 0.70 | 1.0 ± 0.35 ✅ |
| φ_m (surface-layer) | 0.96 | 1.0 ± 0.5 ✅ |
| σ_w/u_* (surface) | 1.65 | 1.25 ± 0.6 ✅ |
| σ_u/u_* | 4.6 | 2.4 ± 1.2 ❌ |

The mean wind is now log-increasing (U 5.1→12.6 m/s) and w-variance peaks in
mid-BL (correct shape). REMAINING: σ_u still ~2× high (streamwise streak energy)
and the OUTER-layer w-variance is elevated (surface layer matches). C_s≈0.30 is a
bit high for a static coefficient — the **Bou-Zeid LASD dynamic C_s(z)** (which
this whole effort is about) should set it automatically and likely fixes the
outer layer + σ_u. Next: wire LASD into this core, then buoyancy/scalar for
SBL/CBL. The spectral core now reproduces surface-layer Monin–Obukhov scaling —
a genuine quantitative oracle match where the compressible core could not even
sustain turbulence.

## RESULT: dynamic Bou-Zeid LASD → 4/4 Monin–Obukhov diagnostics PASS
With the corrected (down-gradient) advection, the 3/2-padding de-aliasing, and the
**dynamic Bou-Zeid LASD C_s(x,y,z)** wired in (`lasd_core.lasd_cs2`, shared with
the compressible path), a 96³ neutral channel (3 h, RTX 5090, fp32) reproduces
Monin–Obukhov similarity — **all four diagnostics within tolerance**:

| diagnostic | value | MOST target |
|---|---|---|
| u_*(log)/u_*(flux) | 1.06 | 1.0 ± 0.35 ✅ |
| φ_m (surface layer) | 0.93 | 1.0 ± 0.5 ✅ |
| σ_w/u_* (z~0.1h) | 1.06 | 1.25 ± 0.6 ✅ |
| σ_u/u_* (z~0.1h) | 2.71 | 2.4 ± 1.2 ✅ |

u_*(total)=0.436 (target 0.45), U(z) log, ⟨u'w'⟩<0 (down-gradient) at every level,
`ww/u_*²` peaks ~1.4 in mid-BL (correct), and the dynamic coefficient self-selects
C_s≈0.15. Two diagnostic notes: (a) MOST u_* is the TOTAL surface stress (the
wall-model value); the RESOLVED ⟨u'w'⟩ alone is ~0.5 u_*² because the near-wall
surface layer is under-resolved on a uniform grid (the SGS carries the rest — a
standard coarse-LES effect, not a bug); (b) σ_w/u_* is sampled at z~0.1·h (the
surface layer), not the wall-adjacent cell where w'→0 by the rigid BC.

The pseudo-spectral incompressible core with the LASD closure is now a genuine
quantitative match to the jax-alfa oracle's physical target — the goal of this
whole effort. Remaining (faithfulness/extension): the 3 secondary codex items
(top advection boundary row, near-wall MOST strain, W-node SGS terms), finer/
stretched near-wall resolution, then buoyancy + scalar for the SBL/CBL cases.

## Codex review → advection SIGN bug (the "3/4 pass" above was an artefact)
A codex adversarial review found the rotational-advection RHS had the WRONG SIGN
on every vertical-vorticity (`w·ω_x`, `w·ω_y`) term: `omega_x_f=dvdz-dwdy`,
`omega_y_f=dwdx-dudz` are the NEGATED standard vorticity, so `+(u×ω)` needs
`Cu=ω_z·v+f2c(w·ω_y)`, `Cv=-ω_z·u-f2c(w·ω_x)`, `Cw=v_f·ω_x-u_f·ω_y`. The wrong
sign gave UP-gradient resolved momentum transport (`⟨u'w'⟩>0`), which inverted
the mean profile AND sustained turbulence SPURIOUSLY by an inverse cascade — so
the "C_s=0.30, 3/4 pass" result was that artefact, not physical turbulence.
(Codex also flagged, secondary: the top advection boundary row should use the
full `arg2[nz-1]` not the ½ from `f2c`; near-wall `|S|` is under-estimated vs the
oracle's MOST bottom-row strain; the vertical SGS stress omits the W-node
`dw/dx, dw/dy` terms.)

**After the sign fix the transport is CORRECT** — `⟨u'w'⟩ < 0` (down-gradient) at
every level, U(z) log-increasing, σ_w/u_*≈0.9 (PASS). New honest state: with
correct advection the smooth log-IC needs a LOW C_s (~0.06-0.08) + a finite-
amplitude perturbation to TRANSITION, but then the resolved eddies under-carry the
flux (resolved u_*≈0.22 of 0.45; SGS carries the rest), φ_m≈2, σ_u≈4. A STATIC
C_s cannot be both low enough to transition AND carry the right resolved stress at
this resolution ⇒ the **dynamic Bou-Zeid LASD C_s(z)** — the goal of this effort,
already implemented for the compressible plane
(`_compute_scale_dependent_dynamic_smag_cs_plane`) — is the decisive next step:
wire it into this spectral core. The method + physics are now correct (validated,
energy-conserving, divergence-free, DOWN-gradient, ~1000 steps/s); the closure
coefficient is the remaining piece.

The core method (spectral horizontal + projection + diffusion + wall) is
validated, sustains turbulence and gives the right surface stress (u_*≈0.32 vs
0.45 target); the residual over-energy is isolated to the advection operator /
forcing setup, not the spectral method or the projection.

## Conclusion (compressible core)
**Conclusion (honest).** Tuning the compressible plane dycore — surface coupling,
seeding, IC, dissipation knobs, resolution to 128³, fp32 throughput — fixed every
STABILITY and book-keeping problem (no blow-up, no wind collapse, conservation,
GPU speed) but does NOT yield self-sustaining resolved ABL turbulence. The
jax-alfa oracle sustains it because its pseudo-spectral incompressible solver is
near-non-dissipative (spectral derivatives + dealiasing, no acoustic filter, no
biharmonic) ⇒ high effective Re. For a quantitative oracle match the faithful
path is the **pseudo-spectral incompressible core** (the deferred architecture
option), now hosting the validated LASD closure + the working surface coupling /
diagnostics / validation harness built here. Recommend revisiting that decision.
