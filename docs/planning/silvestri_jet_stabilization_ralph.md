# Ralph-loop task: stabilize the Silvestri §5 baroclinic jet (eddy-resolving)

## OBJECTIVE
Get the Silvestri et al. 2024 §5 baroclinic-jet matrix PRODUCING STABLE RESULTS at the paper's
resolution, then run the matrix and fill the §5 scoreboard (Figs 8/9/10). Currently ALL 5 schemes
BLOW UP early (1/8° day 11, 1/16° day 4 — finer fails faster). Recipe/driver/metrics/plotter are
built + correct (`ocean/experiments/silvestri_baroclinic_jet.py`, `scripts/run/run_silvestri_
baroclinic_jet.py`, `scripts/plot/plot_silvestri_comparison.py`); the blocker is NUMERICAL STABILITY.
Parent effort: PR #475, `docs/ocean_experiments/silvestri_weno_reproduction.md`,
`docs/planning/silvestri_weno_reproduction_ralph.md`. §4 is already fully reproduced.

## DIAGNOSIS SO FAR (do not re-derive)
- 1/8° (160×128×50, dt=900): W9V blows day 11, QG2 day 15, SM2 day 24 (max|u|→nan).
- dt=450 did NOT help W9V (still day 11) → **dt-INDEPENDENT → a GRID-SCALE MODE, not a CFL/advective
  limit.** Smaller dt alone won't fix it.
- 1/16° (320×256×50) blows EARLIER (day 4) → **finer-fails-faster = grid-scale numerical instability,
  NOT under-resolution.** Also fp64 320×256×50 ≈ 24 GB → OOM risk on the 32 GB V100S.
- Even the EXPLICIT-CLOSURE schemes (SM2 OM4p25, QG2 QG-Leith) blow up → either their closure
  strength is too weak at this res OR there is a SETUP contributor independent of the momentum scheme.
- Hypothesis: legoESM's WENO vector-invariant under-dissipates the C-grid grid-scale mode vs
  Oceananigans' self-dissipating WENOVectorInvariant — the SAME conclusion the Eady eddy-resolving
  work reached (it needed a tuned A_h≈1000 + C_smag≈0.1 backstop). See `eady_eddy_resolving_ralph.md`.

## DIAGNOSTIC LADDER (work in order; each step decides the next)
- **D1 — SETUP vs SCHEME (do FIRST).** Run a HEAVILY-DAMPED config at 1/8° (e.g. momentum_advection=
  vector_invariant + A_h=5e4 + B_h + C_smag=0.2 + dt=300, lateral_friction_scheme=none). Does the JET
  integrate stably for ≥60 days?
  - STABLE → the setup is sound; the blowup is INSUFFICIENT DISSIPATION for the eddy-resolving
    schemes. Go to D2.
  - BLOWS → the bug is in the SETUP, not the scheme (IC thermal-wind projection onto a grid mode?
    AB2 outer + implicit-CN barotropic coupling at fine res? smc03 PGF on the tilted front? the
    zonal-mean restoring?). Diagnose: bisect — turn off the restoring, flatten the front, try
    barotropic_solver=rigid_lid, try a gentler IC (smaller Δb), check max|u| growth location
    (walls? interior? a single column?). Fix the setup before any scheme work.
- **D2 — minimum dissipation per scheme** (only if D1 STABLE). For each scheme find the least
  dissipation that gives a stable, equilibrated 1/8° run (cf. the Eady min-dissipation search,
  `_eke`/`gridscale_frac` proxies). Key question for faithfulness: can the no-closure WENO schemes
  (W9V/W9D/UP3) run with the Eady backstop (A_h≈1000+C_smag≈0.1) — stable but NOT paper-faithful — or
  is there a config that needs NO explicit closure (truly faithful)? Document the faithfulness cost.
- **D3 — implicit-dissipation gap (deeper, only if needed).** Why does legoESM weno9 vector-invariant
  under-dissipate vs Oceananigans? Compare the D-term (divergence damping, `weno_d_term`), the
  KE-gradient form, the `{ζ;u}` smoothness, the vertical/upwinding. Fetch Oceananigans
  `src/Advection/vector_invariant_*.jl` for the reference dissipation. A real fix here = faithful §5.
- **D4 — resolution + memory.** Once stable at 1/8°, try 1/16° (the paper's MAIN res). fp64 OOMs →
  use float32 (the model supports it for finite-volume) or reduce nlev/blocks. 1/32° is the paper's
  finest (likely float32 + careful memory).

## SUCCESS / DONE
- §5 runs stably to (at least) the ~250-day equilibration (paper) at ≥1/8°, for the 5 schemes (with
  the faithfulness status of each honestly recorded — faithful no-closure vs stabilized-with-backstop).
- The matrix run + `plot_silvestri_comparison.py --case jet` → Figs 8/9/10; the §5 scoreboard in the
  report filled (W9V most energetic, etc., or the honest deviation).
- Commit each step; append to the PROGRESS LOG (newest on top). Adversarial-review any NUMERICS/code
  change (skeptical-subagent; codex CLI unavailable in this env).

## ENV
- GPU works (jaxlib CUDA-enabled, ~8.5 s/day at 1/8° on V100S). 2× V100S; **GPU 0 sometimes has an
  external job** — check `nvidia-smi` and pin `CUDA_VISIBLE_DEVICES` to a FREE GPU. fp64 1/16° ≈ 24 GB.
- Run via the driver `scripts/run/run_silvestri_baroclinic_jet.py` (one scheme/res per process) or a
  small inline script for config overrides the driver doesn't expose (A_h/C_smag/dt) — put throwaway
  probes in `scripts/tmp/` (gitignored), clean up after. DON'T delete a run's `--out` dir mid-run.
- The driver emits a parseable VERDICT + saves npz; poll the log for the blowup day.
- `JAX_ENABLE_X64=1` for fp64; consider float32 (drop it) for the fine grids.

## PROGRESS LOG (append every iteration — newest on top)

### Iteration 4 — D2 CONFIRMED at 160×128 (1/8°); --stabilize wired — 2026-06-15
- **160×128×50 + A_h=1000 + C_smag=0.1 (weno9): STABLE to 18 days** (max|u| flat ~0.06 through
  days 5/10/15/18), where un-damped weno9 BLEW at day 11. The backstop decisively fixes §5 at the
  paper's coarsest production resolution. (max|u|~0.06 = eddies still developing slowly; 1/8°
  under-resolves L_d so growth is slow, but it's STABLE — the point.)
- **Wired `--stabilize`** into `build_silvestri_baroclinic_jet_setup(stabilize=True)` + the driver:
  applies the backstop ONLY to the no-closure WENO/flux schemes (lateral_friction_scheme="none");
  SM2/QG2 untouched (their closure + backstop would trip the double-friction guard). Tested + committed.
- **OPEN: SM2/QG2** — in the 1/8° matrix they ALSO blew (SM2 day 24, QG2 day 15) WITH their closures.
  Their closure alone is insufficient at 1/8°. The `--stabilize` backstop can't apply (guard). NEXT:
  test SM2/QG2 un-damped at 48×32×50 (do they blow at coarse res?); if so, options: stronger closure
  coeff (OM4p25/QG-Leith) — not faithful; or relax the guard to allow a SMALL A_h with their closure
  for the stabilize case. Then run the matrix (W9V/W9D/UP3 with --stabilize; SM2/QG2 once stabilized).
- **NEXT: matrix on GPU when free** (both externally busy through D1-D4): the W9V/W9D/UP3 + --stabilize
  1/8° row is ready to launch. Then 1/16° (float32 for the OOM). Then Figs 8/9/10 + scoreboard.

### Iteration 3 — D2: the Eady backstop A_h=1000+C_smag=0.1 STABILIZES §5 + keeps eddies — 2026-06-15
- **D2 (48×32×50, CPU):** weno9 + **A_h=1000 + C_smag=0.1** → **STABLE to 55 days, max|u|→0.2**
  (eddies developing — NOT the over-damped 0.05 of A_h=5e4). Compare: un-damped weno9 blows day 39;
  A_h=5e4 stable-but-dead. So A_h=1000+C_smag=0.1 is the sweet spot (the SAME Eady min-dissipation
  recipe), and it's the path to §5 results: run the WENO schemes with this backstop.
- **FAITHFULNESS COST (record honestly):** the paper's W9V has NO explicit closure (relies on the
  WENO implicit dissipation). legoESM's weno9 needs the A_h+C_smag backstop → "stabilized WENO", not
  purely faithful. This IS the legoESM finding (its WENO under-dissipates vs Oceananigans) — a real,
  documentable result, not a reproduction failure to hide.
- **NEXT: confirm the backstop at 160×128 (GPU)** where un-damped blows at day 11 (the 48×32 was a
  late/borderline blowup at day 39; 160×128 is the real test). Then wire the backstop into the §5
  config/recipe (a `stabilize` option) + run the matrix. Also check SM2/QG2 (do they need a backstop
  too, or is their closure enough at the right strength?). Both GPUs were externally busy during D1/D2.

### Iteration 2 — D1 ANSWERED: setup is SOUND, blowup = insufficient dissipation — 2026-06-15
- **D1 DONE (CPU, 48×32×50 — both GPUs externally busy).** Two probes at the same grid:
  - UN-DAMPED faithful weno9 (A_h=0, C_smag=0): **BLEW UP day 39** (max|u| 0.06→0.6 over 30d, then
    0.6→1.4→nan days 30-39 — the eddy field reaching grid scale).
  - HEAVILY DAMPED (vector_invariant + A_h=5e4 + C_smag=0.2): **STABLE to 55 days** (max|u| flat ~0.05).
- **→ THE SETUP IS SOUND** (IC/thermal-wind/barotropic/PGF/restoring all integrate stably under
  damping). The blowup is **INSUFFICIENT DISSIPATION** for the eddy-resolving schemes — same as the
  Eady rebuild. (A_h=5e4 is way over-damped: eddies dead. The fix is the MINIMUM dissipation.)
- **Also isolated the nlev confound:** 48×32 blows at nlev=50 (day 39) but the earlier nlev=12 sanity
  was stable-to-40d — nlev=50 is a secondary aggravator; horizontal resolution is the primary driver
  (48→day39, 160→day11: finer-fails-faster = grid-scale mode).
- **NEXT: D2 — minimum dissipation.** Test the Eady backstop A_h≈1000 + C_smag≈0.1 (+ smag_cfl_safety
  =0.5) on weno9 at 48×32×50 — is it STABLE *and* does it keep eddies (max|u|~O(1), not 0.05)? Then at
  160×128 on a free GPU. Then per-scheme. Then check SM2/QG2 (their closure + maybe a smaller backstop).
  Honestly record the faithfulness cost (the paper's WENO has NO closure; legoESM's needs a backstop).

### Iteration 1 (setup) — 2026-06-15
- Spec written from the §5-blowup diagnosis (PR #475). The momentum-scheme code + recipe + driver are
  DONE and §4 is reproduced; this loop is purely the §5 eddy-resolving STABILIZATION.
- NEXT: **D1** — heavily-damped 1/8° jet (vector_invariant + A_h=5e4 + C_smag=0.2 + dt=300, 60 d) on a
  free GPU. Stable → dissipation problem (D2); blows → setup bug (bisect the IC/barotropic/PGF/restoring).
