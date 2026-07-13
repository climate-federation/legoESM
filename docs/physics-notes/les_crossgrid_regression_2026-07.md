# Cross-core dry-ABL LES regression sweep (GPU, 2026-07-09)

Re-ran the dry-ABL LES suite (neutral / ekman / gabls1 / wangara) on **each of
the three plane LES dycores** ("grid types") on the RTX 5090, float32, to check
for issues/regressions. LES is plane-only (doubly-periodic); the three cores are
the analogue of the global "grid types".

- **spectral-incompressible** (`spectral_les_plane`) — production, pseudo-spectral,
  drivers `run_spectral_les.py` (neutral/ekman), `run_spectral_sbl.py` (gabls1),
  `run_spectral_cbl.py` (convective).
- **compressible plane** (`compressible_euler_plane`) — legacy, `run_les_plane.py`.
- **pseudo-incompressible** (`pseudo_incompressible_plane`) — newest, matrix-free
  BiCGSTAB Poisson. Had NO `scripts/run` driver → built `run_pseudo_les.py` this pass.

Runs are **reduced-config regression runs** (moderate grid, ~0.3–1 h sim), NOT
full validation equilibria — enough to see turbulence sustainment, u\*, stability,
and crashes, compared to each core's documented behaviour.

## Verdict

| core | neutral | ekman | gabls1 | wangara |
|------|---------|-------|--------|---------|
| **spectral** (prod) | ✅ u\*=0.42 | ✅ u\*=0.49 | ✅ u\*=0.185 (not collapsed) | ✅ CBL σ_w/w\*=0.65 |
| **compressible** (legacy) | ⚠️ laminar (u\*≈0.001) | ⚠️ laminar | ❌ NaN (acoustic burst) | ⚠️ laminar (dt≤0.5); ❌ NaN at default dt=1.0 |
| **pseudo-incompr.** (new) | ✅ u\*=0.455 sustains | ✅ u\*=0.463 sustains | ✅ f64 / ❌ f32 NaN | ✅ f64 / ❌ f32 NaN |

✅ = runs + physically reasonable; ⚠️ = runs but turbulence dead (expected for the
legacy core); ❌ = NaN / crash.

**No NEW regressions found.** Every failure reproduces a *documented* pathology.
The one genuinely new result is that the pseudo core sustains turbulence on the
neutral/ekman cases (matches its design goal) but NaNs on the buoyant cases in
float32.

## Spectral-incompressible (production) — all pass, no regression

| case | result | documented target |
|------|--------|--------------------|
| neutral | u\*_total 0.418, u\*_res 0.283, sustains | ~0.436 (target 0.45) |
| ekman | u\*_total 0.490, sustains | ~0.45 |
| gabls1 (1 h, reduced) | u\*=0.185, h=128 m, jet 8.02@318 m super-geostrophic, Δθ +1.35 K | u\*=0.256 @ 9 h/256³ (Beare 0.26–0.30) |
| cbl | z_i=508 m, w\*=1.0, σ_w/w\*=0.65, ⟨w'θ'⟩/Q0 sfc +0.88 / z_i −0.12, well-mixed | σ_w/w\*≈0.6 |

gabls1 u\* is below the documented 0.256 purely because this is a 1 h / 48²×64 run
vs the documented 9 h / 256²×100 equilibrium (shallower BL, less spun-up) — correct
direction, not a regression. Throughput 131 steps/s @48³ f32.

## Compressible plane (legacy) — reproduces documented "cannot sustain turbulence"

- **neutral / ekman**: run clean but laminarise (max|w|≈0.002–0.003 m/s, u\*≈0.001).
  Turbulence dead — the documented numerical-dissipation ceiling (acoustic
  off-centring + biharmonic hyperdiff cap effective Re below transition).
- **gabls1**: **NaN at t≈0.1 h.** Trajectory is the documented *destructive
  acoustic burst*: mean wind craters 8→0.17 m/s while max|w| runs away
  0.006→2.4→NaN (burst onset at acoustic CFL≳0.42). The tiny 0.01 h smoke was too
  short to reach it.
- **wangara**: **NaN at step 1** with the case default `dt=1.0` (vertical acoustic
  CFL ≈ 2.1 at dz_sfc=20). At `dt≤0.5` it runs (laminar, max|w|≈0.013).

Net: legacy core is unusable for sustained ABL LES (known). Not a regression.

## Pseudo-incompressible (new driver) — sustains neutral/ekman; NaN on buoyant cases in f32

Built `scripts/run/run_pseudo_les.py` (+ `tests/unit/test_run_pseudo_les_cli.py`),
mirroring the other cores, reusing the `pip.*` core API and the case pattern from
`scripts/validate/validate_bl_new_vs_spectral.py`. Codex adversarial review: **no
HIGH findings**; 3 MED + 2 LOW applied (collocate C-grid faces before moments;
compile warm-up no longer advances the reported state; MOST stability-corrected
u\* for gabls1; per-run case-dict copy; strengthened CLI test).

- **neutral / ekman**: ✅ **sustain turbulence** — wvar 0.12 (growing), max|w|≈1.2,
  u\*_wall 0.455 / 0.463 ≈ MOST target 0.45, jet ~10 m/s. This is the core's design
  win vs the compressible core's collapse. ~49 steps/s @40³ f32.
- **gabls1 / wangara**: ❌ **NaN in float32.** These are the two BUOYANT cases
  (stratified θ + surface heat flux); neutral/ekman have no active buoyancy and
  survive. It is **not** a clean WENO5-β-cancellation (the initial hypothesis) —
  the scheme dependence flips between cases (32³/0.12 h controls):

  | case | f32 weno5 | f32 van_leer | f64 weno5 |
  |------|-----------|--------------|-----------|
  | gabls1 | finite (this small/short config) | **NaN** | finite |
  | wangara | **NaN** | finite | — |

  gabls1 f32-weno5 is finite at 32³/0.12 h but NaN at 40³×64 **even short**, so
  the trigger is **grid** (finer dz), not duration. **Precision is the clean
  lever** — controlled test at the exact sweep grid, changing only f32→f64:

  | case (sweep grid) | f32 weno5 | f64 weno5 |
  |-------------------|-----------|-----------|
  | gabls1 40³×64 | NaN (short + long) | **finite, wvar=0.099, turbulent** |
  | wangara 40³×48 | NaN | **finite, wvar=0.247, turbulent** |

  **f64 fixes both** — they run finite AND sustain turbulence. So the pseudo core
  handles **all 4 dry cases in f64**; in **f32 only the non-buoyant** cases
  (neutral/ekman) are robust. This matches `validate_bl_new_vs_spectral.py`
  defaulting to f64 and the core being WIP.

  Root cause is the **documented fine-res stable-BL 2Δ grid-noise** (memory
  phase2c: "fine-res GABLS1-COOLING long run still collapses"), NOT the f32
  WENO5 β-cancellation (already fixed in `core/weno.py` common-shift). The
  in-core biharmonic de-noiser `hyperdiff_coeff` (now exposed as `--hyperdiff`)
  helps **partially**: `--hyperdiff 1e4` makes **wangara** f32 finite + turbulent
  (1e3 too weak, 1e5 over-damps → unstable), but **gabls1 at its fine dx=10 grid
  cannot be de-noised in f32** — the explicit-biharmonic stability window
  (coeff ≲ 500 at dt=0.2) is below what suppresses the 2Δ noise. gabls1 f32 fine
  ⇒ f64 (or coarsen / smaller dt). The driver warns on `--f32` + a buoyant case.

## Actions

1. **Pseudo buoyant LES f32 — now unlocked by a CFL-unlimited de-noiser.** Added
   a per-step `[1,2,1]` horizontal Shapiro low-pass on the SCALARS (θ, tracers),
   `shapiro_coeff` / driver `--shapiro`. Being a *multiplicative filter*
   (response cos²(kΔ/2) ∈ [0,1]) it is not CFL-limited like the biharmonic
   `--hyperdiff`, so it suppresses the fine-res 2Δ θ-noise at any resolution.
   Verified at the sweep config where f32 gabls1 NaN'd: `--shapiro 0.05` too weak
   (NaN), **`--shapiro 0.1` runs finite + turbulent + stable** (u\*≈0.32,
   u\*_res≈0.13, dθ/dz>0, super-geostrophic jet). Scalars-only ⇒ the velocity
   projection is untouched (div-free preserved); `shapiro_coeff=0` is bit-identical
   default. **Caveat: gabls1 f32 stays MARGINAL** even with the de-noiser — a later
   `--shapiro 0.15` recorded run tipped to NaN where `0.1` held (f32 GPU
   non-determinism on a marginal case). **f64 is the reliable path** for the
   stratified/buoyant pseudo cases; `--shapiro` is the fast-but-marginal f32 option.
   `--hyperdiff 1e4` also fixes wangara. Driver warns on `--f32` + buoyant case.
2. **Compressible core gated research-only.** It never sustains LES turbulence
   (relaminarises; gabls1 acoustic-burst NaN), so `run_les_plane.py` now **refuses
   to run without `--research-only`** — it exits with a warning pointing to the
   spectral / pseudo-incompressible cores, so its laminar/NaN output can't be
   mistaken for LES. Also **fixed** the wangara default `dt=1.0`→`0.5` (was
   step-1 NaN, vertical acoustic CFL ≈ 2.1; dt=0.5 verified clean). CLI tests pass.
3. New `scripts/run/run_pseudo_les.py` (+ CLI test) gives the pseudo core parity
   with the other two — codex-reviewed (no HIGH), 5 findings applied.

## No regressions

Spectral (production) unchanged and on-target; compressible failures are the
documented relaminarisation + acoustic burst; pseudo f32-buoyant NaN is a new
*characterisation* of a WIP core, not a regression (no prior f32 buoyant baseline).
