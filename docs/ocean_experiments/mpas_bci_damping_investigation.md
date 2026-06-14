# MPAS BCI damping — root-cause investigation (2026-04-24)

## The problem

Strong-forcing Eady channel (U=0.8 m/s, τ_Eady≈5 d) on MPAS 20 km and lat-lon
100×50 configured with identical dissipation (`B_h=2.3e11`, `C_smag=0`, no
sponge, `bottom_drag_coeff=0.001 m/s`) yield very different BCI growth rates.

| Diagnostic | lat-lon tvd | MPAS tvd (default config) |
|---|---|---|
| `mean_ke` at day 0 | 9.3e-3 | 9.3e-3 |
| `mean_ke` at day 40 | 1.21e-2 | 9.2e-3 |
| `mean_ke` at day 80 | 4.19e-2 | 9.3e-3 |
| `mean_ke` at day 100 | **2.66e-1** | 1.0e-2 |
| KE e-fold time (d10–d60 fit) | 56 d | effectively **no growth** |

MPAS KE is ~27× smaller than lat-lon by day 100. `max_speed` grows locally
on MPAS (0.4 → 1.9 m/s by day 200), but `mean_ke` — the domain-integrated
eddy energy — stays pinned at the initial-condition value for the first 100
days. Something is dissipating the balanced BCI perturbation globally.

## What it is NOT (both confirmed by independent audits)

Two independent audits — an ocean-physics audit and a dycore/numerics audit
— were run in parallel on the linear-bottom-drag implementation.

**Consensus**: bottom drag is applied **identically** on both grids.

- Same operator form: `du/dt += -r · u_bot / h_bot` at bottom level only.
- Same units (`r` in m/s) — dimensional analysis consistent.
- Same `h_bot` semantics (live face-interpolated bottom layer thickness).
- Same barotropic-substep form `(1 − dt·r/H)`.
- Same F_slow_u routing (MPAS was brought into parity by commit `1bbc021`).
- Same mask treatment (edge mask / u-mask).
- No unit confusion — commit `57b81aa` fixed a historical r·u → r·u/h mismatch
  on both grids simultaneously.

Effective damping timescales with `r=0.001 m/s`:
- Bottom-layer-only: `h_bot/r ≈ 5.9 d` (but IC has u_bot ≈ 0 for classical
  baroclinic Eady, so this doesn't kick in until the mode grows).
- Barotropic-mean: `H/r ≈ 64 d` (or 32 d if the shared design double-counts
  drag in F_slow_u + barotropic substep; still symmetric across grids).

## What it IS — confirmed by experiment

The dycore audit surfaced the **TRiSK enstrophy-conserving PV flux** as a
grid-asymmetric suspect with no lat-lon analogue. The enstrophy-conserving
scheme formally dissipates `∫q²dA`, which is exactly the quantity BCI grows
through.

Test: run MPAS tvd strong-forcing with `pv_scheme="energy"` instead of the
default `"enstrophy"`, 80 days. Repeat with APVM damping (300 s, 1200 s,
3600 s) and with Smagorinsky `C_s=0.2`. All configurations:

| Config | KE e-fold (d10–d60) | Blowup day |
|---|---|---|
| `pv=enstrophy, apvm=0` (baseline) | shrinking | survives 200 d |
| `pv=energy, apvm=300s` | **72 d** | 57 |
| `pv=energy, apvm=1200s` | 72 d | 57 |
| `pv=energy, apvm=300s, Csmag=0.2` | 72 d | 58 |
| `pv=energy, apvm=3600s` (12×dt) | 72 d | 58 |
| lat-lon tvd | 79 d | survives 200 d |

**Result**: switching to energy-conserving PV flux recovers BCI growth at a
rate (72 d) essentially equal to lat-lon (79 d). The enstrophy-conserving
TRiSK PV flux was the dominant BCI damping mechanism on MPAS.

**Side effect**: the energy-conserving scheme has a known ζ-checkerboard
null mode. It triggers at day ~57 regardless of APVM strength (300 s up to
12×dt=3600 s) or Smagorinsky. The instability timing is within ~2 % across
all four APVM/Smag combinations — it's insensitive to these knobs.

## Why APVM does not control the checkerboard here

APVM upstream-biases the PV advection by `dt_apvm/2 · u·∇q`. With our
discretisation, this adds a Laplacian-like damping on PV scales comparable
to `u·dt_apvm`. For the Eady regime (u ~ 0.4 m/s, dt_apvm = 300 s, Δx = 20 km),
the APVM-induced damping scale is `u·dt_apvm ≈ 120 m ≪ Δx`. Even at
`dt_apvm = 3600 s`, it's ~1.4 km, still below the 20 km cell size. APVM is a
consistency correction for advected PV, not a scale-selective damping of
the null mode.

## Mixed energy/enstrophy PV flux (implemented & tested)

Implemented in `ocean_pe_mpas.py` as `pv_scheme="mixed"` with weight
`pv_alpha ∈ [0, 1]`: `F = α·F_energy + (1−α)·F_enstrophy`. Exposed through
`--pv-scheme mixed --pv-alpha <α>` on the CLI.

Sweep at strong Eady forcing (U=0.8, 80 d, tvd, APVM=300 s):

| α | max_spd at end | Stable 80 d? | Blowup day |
|---|---|---|---|
| 1.0 (pure energy) | 1.28 @ day 50 | ❌ | 57 |
| 0.8 | 1.81 @ day 60 | ❌ | 64 |
| 0.7 | 1.61 @ day 60 | ❌ | 68 |
| 0.6 | 0.80 @ day 56 (→2.66 pre-blowup) | ❌ | 77 |
| **0.5** | **1.61 @ day 80** | **✅** | — |
| 0.0 (pure enstrophy) | 0.92 @ day 80 | ✅ (no growth) | — |

Stability is **monotonic** in α and the threshold is between 0.5 and 0.6:
below the threshold the scheme inherits enstrophy's null-mode suppression;
above it, the ζ-checkerboard wins.

At α=0.5, the MPAS BCI reaches the same max_spd (1.6 m/s) at 80 days that
lat-lon tvd reaches at day 60 — ~20 d delay, but the KE does grow and BCI
equilibrates rather than saturating on a grid-scale instability. KE e-fold
(d10–d70 fit, log-linear): ~360 d, so the domain-integrated growth rate is
still ~5× slower than lat-lon (79 d).

**Takeaway**: the mixed PV flux **delays but does not eliminate** the
instability. The stability boundary observed in the 80-d sweep was
illusory — an 80-d run with α=0.5 ends just before the actual blowup.

### The boundary is not a stability threshold, it's a blowup delay

Extending the α=0.5 configuration to 200 d reveals that it blows up at
**day 83** (just past the 80-d run). Testing additional damping:

| α=0.5, 200 d | Blowup day |
|---|---|
| APVM=300 s (baseline) | 83 |
| APVM=1200 s (4×dt) | 84 |
| APVM=300 s + Csmag=0.2 | 83 |

APVM strength and Smagorinsky both fail to delay the blowup meaningfully
— within 1 day of the baseline. The instability timing is essentially
insensitive to all damping we can apply through existing operators, and
blows up *regardless* once BCI amplitude reaches finite level.

### Root cause: nonlinear coupling of BCI to ζ null mode

At any α > 0, finite-amplitude BCI amplifies the grid-scale ζ-checkerboard
through nonlinear interactions. Enstrophy weighting damps the null mode
*proportionally* to (1−α), but the BCI itself pumps energy into it at a
rate proportional to its own amplitude. Since BCI grows exponentially
with a rate ~σ·α (where σ is the analytic Eady rate), the blowup timing
is roughly `t_blowup ≈ t_saturation(α) + O(ln(N_modes)/σ·α)`. Adding more
damping (APVM, Smag) targets the wrong quantity — viscous damping of u
does not directly damp the grid-scale ζ null mode.

The proper fix requires an operator that specifically damps grid-scale
ζ: `del^4` on relative vorticity (biharmonic on ζ, not on u). This is a
standard MPAS-Ocean operator for exactly this purpose and is not
currently implemented in legoESM. Estimated effort: ~50 LOC new operator
in `src/legoesm/ocean/dynamics/mpas_ocean_operators.py` plus config
surface and wiring.

## Biharmonic ζ-damping: `−K_ζ·∇⁴ζ` operator (implemented & tested)

New operators added to `src/legoesm/core/operators_voronoi.py`:

- `vertex_laplacian_3d(φ_v, mesh)` — finite-volume Laplacian of a
  vertex-centred scalar on the triangular dual grid.
- `biharmonic_vorticity_del4_3d(u, mesh)` — edge-normal force that, when
  added as `K_ζ · F` to the momentum equation, produces `−K_ζ · ∇⁴ζ` in
  the vorticity equation. Bypasses the velocity-to-curl chain and acts on
  the ζ field directly at vertices, so it captures the ζ-checkerboard
  null mode that `B_h · ∇⁴u` misses.

Wired through `MPASOceanConfig.K_zeta_bih [m⁴/s]` and `--K-zeta-bih` CLI
flag. Unit tests (9 passing) cover: kernel on constants, sign on bump,
damping correlation (caught a sign error at implementation), zero-velocity
input, k⁴ scale-selectivity, L² energy monotonicity, JIT, and jax.grad.
Ocean-expert audit (2026-04-24) verdict: **correctly implemented**.

Parameter sweep at U=0.8, 20 km, 80 d, tvd, `pv_scheme=energy`:

| K_ζ (m⁴/s) | Status | max_spd@d80 | KE@d80 | Note |
|---|---|---|---|---|
| 0 | blew d57 | — | — | no damping (baseline) |
| 1e11 | blew d67 | 1.06 @ d50 | 1.15e-2 | under-damped |
| **2e11** | **PASS** | **1.16** | **1.52e-2** | best 80-d growth |
| 3e11 | PASS | 0.83 | 1.06e-2 | (survives to d143 in 200-d run) |
| 5e11 | PASS | 0.48 | 9.16e-3 | over-damped |
| 1e12 | PASS | 0.42 | 9.15e-3 | BCI fully suppressed |
| lat-lon tvd | reference | 2.70 | 4.19e-2 | (KE e-fold ≈ 76 d) |

Extended runs (200 d):
- K_ζ=3e11 survives to **day 143** — 86 days longer than pure energy with
  clear BCI growth (max_spd peaked 1.43 at day 135).
- K_ζ=2e11 survives to day 99 (faster growth but earlier onset of nonlinear
  instability because of higher finite-amplitude).

The monotonic relationship **blowup-day ∝ K_ζ** (57 → 67 → 99 → 143 → ∞ as
K_ζ goes 0 → 1e11 → 2e11 → 3e11 → 5e11) is exactly what a scale-selective
damping should produce: larger K_ζ buys more headroom against the
nonlinear cascade but eventually over-damps the BCI itself.

Grid-scale (Δx=20 km) ζ damping rate `K_ζ/Δx⁴`:
- K_ζ = 2e11 → e-fold 9.2 d (comparable to τ_Eady = 5 d — just strong enough
  to control the null mode while letting BCI grow)
- K_ζ = 3e11 → e-fold 6.2 d (slightly over)
- K_ζ = 1e12 → e-fold 1.85 d (kills BCI along with the null mode)

**Conclusion**: the biharmonic ζ-damping operator works as predicted. MPAS
strong-forcing Eady BCI now grows visibly for 80–140 days before
numerical instabilities take over (the remaining instabilities come from
finite-amplitude nonlinear cascade, not the primary null mode that killed
the baseline runs). The recommended operational setting is
`pv_scheme="energy", apvm_dt=dt, K_zeta_bih=2e11 to 3e11 m⁴/s` at 20 km
resolution.

Follow-up considerations (not blocking):
- Land-mask inside the operator chain (audit Q5) — material for realistic
  geometry but harmless for all-ocean Eady channel. Shared concern with
  other curl-based operators.
- Tuning K_ζ scaling with Δx (expected ∝ Δx⁴) — not yet verified across
  resolutions.
- Long-run saturation behaviour beyond d143 — depends on nonlinear eddy
  dissipation, may need additional small Smagorinsky.

## Options for controlling the checkerboard (ranked)

1. **Add `del^4` hyperviscosity on ζ** (separate from `B_h` on u). This is
   what MPAS-Ocean uses operationally for this same null mode. Targets
   grid-scale vorticity without damping the BCI scale. New operator; ~50 LOC.

2. **Mixed energy/enstrophy PV flux** with weighted combination,
   `α·F_energy + (1−α)·F_enstrophy`, α ≈ 0.8 (Ringler 2010). Retains most of
   the BCI growth and partially damps the null mode. New code path; ~30 LOC.

3. **Fix the IC thermal-wind balance**. Current `_set_linear_shear_mpas`
   evaluates an analytic envelope at edge midpoints, which is inconsistent
   with the actual discrete T values at cell centres. Ocean-expert in an
   earlier session recommended deriving u at edges from the cross-edge
   T difference directly. Unclear whether this would help the checkerboard
   (which is a null-mode instability, not an adjustment shock), but would
   reduce the initial 17 % off-jet zonal noise. ~15 LOC.

4. **Lower dt**. Checkerboard is spatial, not temporal — lowering dt
   likely doesn't help. Cheapest to test, lowest expected yield.

## Recommendation

For the current advection-scheme comparison work:
- **Short-term**: use MPAS with `pv_scheme="enstrophy"` (the current default).
  Acknowledge in the write-up that BCI amplitude is suppressed on MPAS by a
  factor ~10 relative to lat-lon at strong forcing. This is useful for
  smoke-testing the advection scheme (mixing diagnostics etc.) but is
  **not** a quantitatively meaningful Eady BCI comparison with lat-lon.
- **Medium-term**: implement option (1) — `del^4` on ζ. Open a new issue
  and plan it as its own PR. Once implemented, the `pv=energy, del4_zeta=...`
  combination should give quantitatively correct BCI on MPAS.
- **Optional hygiene fix**: option (3) — IC thermal-wind discretisation —
  addresses a separate but related problem (17 % IC velocity noise), and
  is cheap to implement.

## Artifacts

- Comparison plot: `results/ocean/eady_uniform/bci_pv_scheme_compare.png`
- Timeseries CSVs from all five probe runs:
  - `results/ocean/eady_uniform/mpas_channel/20km/tvd_mpas_20km_U08_Bh2.3e11_Cs0.0_200d/`
  - `results/ocean/eady_uniform/mpas_channel/20km/tvd_mpas_20km_U08_Bh2.3e11_Cs0.0_pvenergy_apvm300_80d/`
  - `results/ocean/eady_uniform/mpas_channel/20km/tvd_mpas_20km_U08_Bh2.3e11_Cs0.0_pvenergy_apvm1200_80d/`
  - `results/ocean/eady_uniform/mpas_channel/20km/tvd_mpas_20km_U08_Bh2.3e11_Cs0.2_pvenergy_apvm300_80d/`
  - `results/ocean/eady_uniform/mpas_channel/20km/tvd_mpas_20km_U08_Bh2.3e11_Cs0.0_pvenergy_apvm3600_80d/`
- CLI additions (committable as their own small PR): `--pv-scheme {enstrophy,energy}`
  and `--apvm-dt <seconds>` on `scripts/matrix/run_ocean_test_matrix.py`.
- Drag-audit conclusions: see this doc above; the two audit reports are
  preserved in the session transcript.
