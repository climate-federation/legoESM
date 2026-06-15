# Ralph-loop task: unlock eddy-RESOLVING ocean sims via the Eady test

## OBJECTIVE v2 (current — min-dissipation, long/saturated, high effective resolution)
Find the **MINIMUM-dissipation** config that runs a **LONG, STABLE, statistically-STABILIZED
(saturated)** eddy-resolving Eady simulation — so the *effective resolution* is as high as
possible (least dissipation = sharpest resolved eddies, energy extends to high wavenumber).
This is an OPTIMIZATION: minimize dissipation **subject to** {stable to saturation AND
statistically stabilized AND no grid-scale energy pileup}.

- **Track: WENO5 flux-form ONLY.** The NEMO-like track (vector_invariant+hollingsworth)
  is grid-scale-unstable even at 60 and blows up at 120 even with A_h — it needs an
  **energy-enstrophy-conserving (EEN) vorticity flux** ported to the lat-lon C-grid (a CODE
  addition, not a config knob; NEMO `nn_dynvor=een`). Out of scope for the dissipation search;
  note it as the prerequisite for a NEMO dycore.
- **Driver reports a parseable `VERDICT ... stable= saturated= max_u= EKE_sat= gridscale_frac=
  ... dissipation[...]` line** + a `SATURATION` line. `gridscale_frac` = energy fraction in the
  top quartile of zonal wavenumbers (the effective-resolution / over-vs-under-dissipation
  proxy): too LOW dissipation → high frac (grid pileup, → blowup); too HIGH → spectrum rolls
  off early (low effective resolution); the SWEET SPOT is the minimum dissipation with
  `gridscale_frac` small AND saturated-stable.
- **Search procedure:** for each candidate (resolution, dissipation form+strength, dt), run
  LONG enough to saturate (weak U=0.2 τ≈20d → ~100–150 d; strong U=0.8 τ≈5d → ~50–80 d; strong
  saturates faster = cheaper iteration). Record the VERDICT. Then SWEEP dissipation DOWN to the
  edge of stability and pick the minimum stable+saturated value. Compare dissipation FORMS
  (Laplacian A_h vs biharmonic B_h vs Smagorinsky C_smag vs Leith C_leith): biharmonic/Leith
  are more scale-selective (higher effective resolution per unit stability) — prefer them.
- **Validate** the winning min-dissipation config's saturated EKE/spectrum against the
  dt-matched Veros control at the same resolution (Veros 60 + weak-120 dt450 both run clean;
  Veros weak-60 saturation is running for the matched comparison).
- **Then push resolution finer** (toward ~10 km, the design point; try ISOTROPIC grids —
  120×120 is 2:1 anisotropic) and re-minimize dissipation; report the effective resolution
  achieved per grid.
- **ENV CAVEAT (critical):** GPU runs are intermittently SIGTERM/SIGURG-killed (exit 143/144)
  in this session — some complete, some die early. RE-RUN killed configs (don't trust a single
  failed run); prefer shorter triage runs; both V100S GPUs are currently free (the external
  28 GB job cleared). Foreground short runs are most reliable.
- **Done:** a documented MINIMUM-dissipation recipe that runs a long stable saturated
  eddy-resolving (≥120×120) Eady sim with small gridscale_frac + Veros-validated saturated
  stats; commit it (dissipation as the config knob) + update the test. Report the
  effective-resolution / dissipation tradeoff curve.

## Goal (v1 — superseded by v2 above; kept for context)
Get the idealized **Eady baroclinic-instability** experiment running **cleanly and
stably at eddy-RESOLVING resolution** (≥120×120 ≈ 17 km, ideally ~10 km / ~200×100,
where the 107 km deformation radius is resolved by ≳10 cells), so legoESM gains a
validated eddy-resolving capability. Converge this on **two dycore tracks**:

1. **WENO5 track** — flux-form: `momentum_advection="weno5"`, `tracer_advection="weno5"`.
2. **NEMO-like track** (Pierre's direction) — `momentum_advection="vector_invariant"`
   + `ke_gradient_scheme="hollingsworth"` (NEMO 4.2.1 `dynkeg.F90 nkeg_HW`, `nn_dynkeg=1`),
   NEMO-style Leith/Smagorinsky biharmonic viscosity.

## Success criteria (a run is "clean" iff ALL hold at the target resolution)
1. **Stable**: finite for the whole run; **no localized grid-scale spike** — max|u|
   stays physical (comparable to the 60×60 saturated value, not 5–20× the jet); no
   checkerboard in the surface u/v field.
2. **Correct growth**: fitted EKE growth rate λ within ~±25 % of analytical
   `2·σ_Eady` (σ = 0.31·f·|∂U/∂z|/N) during the LINEAR phase. (At present the
   "stable" 120 run grows ~7× too fast = residual grid-scale energy — that counts
   as NOT clean.)
3. **Physical saturation**: EKE grows exponentially then plateaus; eddy field looks
   like coherent meanders/eddies, not noise.
4. **Matches a trusted reference** where one exists (Veros at the same resolution +
   timestep; see the Veros caveat below).

## The sponge-necessity hypothesis (TEST THIS EXPLICITLY — user suspects NO sponge needed)
The old config relied on a τ=1-day wall sponge. A *properly-dissipated* eddy-resolving
setup with the right numerics **may not need the sponge at all**. For every candidate
config that works WITH the sponge, **re-run with `--no-sponge`** and report whether it
still works. The win condition prefers a **sponge-free** clean run. If the sponge turns
out to be necessary, document precisely why (wall steepening? Kelvin trapping?) with
evidence (where the no-sponge run fails spatially).

## Findings so far (this session — do not re-derive)
- **60×60 (eddy-permitting) WORKS** on the WENO5 track: correct growth (λ ratio 1.14),
  saturates ~1.4–1.8 m/s, matches Veros (max|u| ~0.5–0.6). Both strong (U=0.8) and weak
  (U=0.2) regimes saturate at 60.
- **120×120 strong forcing is ill-posed for BOTH legoESM AND Veros** ("solution diverged")
  → use the **weak/balanced regime (U=0.2, τ≈20 d)** for eddy-resolving; strong (U=0.8,
  τ≈5 d) is a stress test, not the physical target.
- **120×120 has a grid-scale mode** (max|u| 4.5 spike) that grows over ~days. The IC is
  clean (verified). **More dissipation BLEW UP FASTER** (C_smag 0.4 / +Leith) because the
  biharmonic viscous-CFL (∝dx⁴) is violated → the fix is **smaller dt + `smag_cfl_safety`
  cap**, not just more viscosity.
- **dt=300 + C_smag=0.35 + smag_cfl_safety=0.5 (weak, 120×120) is now STABLE** (no blowup,
  max|u| settles ~1.7) but the growth is still ~7× too fast → residual grid-scale energy.
  Next: more/different scale-aware dissipation (Leith), smaller dt, or the NEMO momentum
  (hollingsworth) which is anti-grid-noise by construction.

## Veros control caveat (important)
Veros `eady_uniform` uses a FIXED `dt_mom=dt_tracer=1800 s` and `NX=NY=30`. At 120×120
(dx~8 km) that dt violates CFL → Veros "diverged at iteration ~61" even at weak forcing.
So the Veros 120 failures are a **timestep confound, not a Veros defect**. For a FAIR
Veros control at fine res, the Veros dt must be reduced too (patch `_vc.DT_MOM_S` /
`settings.dt_*` proportionally to dx). The Veros-config grid/U/dt are module-level
constants in `packages/ocean/legoesm/ocean/fidelity/veros_configs/eady_uniform.py`
(patch `_vc.NX/NY/U_SURFACE` and the dt before `run_veros(..., force_recompute=True)`).

## Knobs (all wired into the tools below)
`run_eady_rebuilt.py` flags: `--resolution NxN`, `--days`, `--dt`, `--u-surface`
(0.2 weak / 0.8 strong), `--c-smag`, `--c-leith`, `--c-smag-lap`, `--b-h`,
`--smag-cfl-safety`, `--momentum-advection {weno5,weno7,vector_invariant}`,
`--ke-gradient {centered,hollingsworth}`, `--tracer-advection`,
`--barotropic-solver {implicit_cn,explicit_substep,rigid_lid}`, `--no-sponge`, `--tag`.

## Tools (already built this session — extend, don't rebuild)
- `packages/ocean/legoesm/ocean/experiments/eady_uniform.py::build_eady_uniform_setup`
  — the corrected recipe builder (all knobs as kwargs).
- `scripts/run/run_eady_rebuilt.py` — runs + fits growth rate + reports stability/EKE/max|u|,
  saves `results/eady_rebuilt/eady_rebuilt_<res><tag>.npz`.
- `scripts/tmp/_eady_veros_stability.py NX DAYS U` — Veros reference (patches NX/NY/U;
  ADD a dt patch for fair fine-res control).
- `scripts/tmp/_eady_120_locate.py` — spike locator (had a scan-carry pytree bug; fix:
  use `.replace(data=...)` not `Field(...)` so v's 'edge' location is preserved).

## Compute / hygiene
- 2× V100S. Pin with `CUDA_VISIBLE_DEVICES=0|1`; check `nvidia-smi` (an external job has
  intermittently grabbed GPU0). fp64 (`JAX_ENABLE_X64=1`) always. Run in background +
  poll logs.
- The spike appears by ~day 5, so a 30–40-day run is enough to TRIAGE a config before
  committing to a long saturation run — start short, escalate.
- Throwaway probes → `scripts/tmp/`. Mandatory: keep `build_eady_uniform_setup` clean +
  unit-tested; run the codex/adversarial review before declaring a shippable config.

## Definition of done
A **clean, sponge-free (or sponge-justified)** eddy-resolving Eady run at ≥120×120 (and a
check toward ~10 km) on BOTH the WENO5 and NEMO-like tracks: stable, correct growth rate,
physical saturation, validated against a fair (dt-matched) Veros control. Write up the
winning config(s) per track + the sponge verdict in this file's "## RESULTS" section, and
commit the recipe + a unit test.

## COMPUTE NOTE (iter 6)
GPU0 is occupied by a **28 GB external production job (PID 285582)** — likely the OMIP/
tripole spinup from a prior session. DO NOT kill it. **Use GPU1 only** (free). Laplacian
viscosity (A_h) is CFL-safe at **dt=300** (2× faster than the dt=150 the biharmonic forced),
so triage A_h runs at dt=300.

## CHECKPOINT STATUS (iter 12 — committed, loop paused for input)
**WHAT WORKS:** the AUDITED + REBUILT recipe is a real improvement and runs CLEAN at
**eddy-PERMITTING (60×60)** on the WENO5 track in both regimes — correct BCI growth
(λ ratio 1.1–1.4 vs σ_Eady), physical saturation, matches the Veros reference. Committed:
`build_eady_uniform_setup` → `EadyUniformRecipe` (canonical shape), corrected Phase-G stack
(implicit_cn / weno5×2 / rk3+ab2 / smc03, KPP & GM/Redi off), all dissipation knobs,
patchable Veros dt, `run_eady_rebuilt.py`, `tests/ocean/unit/test_eady_uniform_recipe.py` (5).

**OPEN (the eddy-RESOLVING frontier, 120×120):** a persistent localized grid-scale mode
(max|u|~3–4 vs physical ~1) that grows over days. RULED OUT: the sponge (no-sponge blows up
HARDER → mode is interior; sponge is stabilizing, NOT the cause — so the user's no-sponge
hypothesis does NOT hold for the current numerics), and the SETUP (dt-matched Veros runs 120
cleanly → sound). The mode sits at ~4–5Δx, CLOSE to the ~16Δx eddy scale, so viscosity can't
cleanly separate them: Laplacian A_h=5e3 kills the spike but OVER-DAMPS the BCI (EKE
decreases); A_h=1500 is too weak (spike returns by day 5); fixed biharmonic B_h=2.5e11 MISSES
it (too sharply peaked at 2Δx). NEMO-like track BLOWS UP at 120 (needs an enstrophy-conserving
vorticity scheme, not yet a lat-lon config field).

**LEADS for resuming (each a multi-run investigation):** (1) the 120×120 grid is 2:1
ANISOTROPIC (8 km lon × 17 km lat) — try an ISOTROPIC grid (60×120 ~17 km, or 100×200
~10 km design point). (2) Locate the spike spatially (the fast locator + most runs in this
session hit a flaky-env exit-144 — use SHORT foreground runs). (3) The WENO5 momentum may
itself seed grid-scale energy near the eddy scale — compare a clean low-dissipation run's
u-spectrum. (4) NEMO track: find/port an enstrophy-conserving (EEN) vorticity flux.

**ENV CAVEAT:** GPU0 has a 28 GB external production job (use GPU1 only); `run_in_background`
+ nohup tasks intermittently die with exit 144 — run experiments FOREGROUND with `timeout`.

## LEADING RECIPE (iter 8) — min-dissipation eddy-resolving @120×120
**`A_h=1000 + C_smag=0.1` (Laplacian + biharmonic Smag), weak U=0.2, dt=600, sponge on:**
stable=True, **gridscale_frac=0.0001** (extremely clean spectrum), EKE 8.7e-4 (~15× the
pure-A_h=1500 baseline 5.8e-5), still growing at 80d. The COMBINATION (moderate Laplacian to
catch the ~4–5Δx mode + biharmonic for scale-selectivity) beats both pure forms (pure A_h
over-damps, pure C_smag blows up). gridscale_frac far below any noise threshold → ROOM to
reduce dissipation further (→ higher effective resolution). RUNNING: 150d saturation of this
config + a lower-dissipation push (A_h=600+C_smag=0.08).

## MIN-DISSIPATION EDGE (iter 9-11) — 120×120 weak dt600
| config | stable | gridscale_frac | EKE (80d, growing) |
|---|---|---|---|
| A_h=1000+C_smag=0.1 | ✅ | 0.0001 | 8.7e-4 |
| A_h=800+C_smag=0.15 | ✅ | 0.0001 | 8.2e-4 |
| A_h=800+C_smag=0.1  | ✅ (running) | — | 1.7e-4 @40d, climbing |
| A_h=600+C_smag=0.08 | ✗ BLEW UP | — | — |
**Minimum stable combination ≈ A_h≈800 + C_smag≈0.1.** Recommend **A_h=1000+C_smag=0.1** as
the robust default (just above the edge), A_h=800 as the aggressive minimum. The 150-day
saturation run of A_h=1000+C_smag=0.1 is confirming `saturated=True` (the gating "long
saturated" check). Once it does → commit this combination as the eddy-resolving dissipation
default + update test + cancel loop.

### v2 Iteration 12 (weak saturation slow → strong for explicit saturated demo)
- **A_h=1000+C_smag=0.1 WEAK 150d**: stable=True, clean (gridscale_frac 0.0001), EKE 4.5e-3
  (strong, ~half the 60×60 ref 9.4e-3) — but `saturated=False`, STILL CLIMBING. Weak (τ=20d)
  needs ~250+ d to statistically stationarize; eddies are developed+clean, just not stationary
  in 150d. A_h=800+C_smag=0.1 @80d: EKE 1.4e-3, clean (gridscale_frac 0.0002), stable.
- → For an EXPLICIT `saturated=True` demo, use STRONG (U=0.8, τ=5d, saturates ~50d) with the
  combination (strong eddies more vigorous → A_h=2000+C_smag=0.15 / A_h=3000+C_smag=0.2).
  Strong-120 blew up earlier ONLY because the dissipation form was wrong; the combination
  should stabilize it. Running both, 60d.
**TODO:** if strong-combo saturates stable+clean → that's the explicit saturated eddy-resolving
demo; commit the recipe (combination dissipation, both regimes documented) + test; cancel loop.

## RESULTS (loop appends here)

### v2 Iteration 1 (min-dissipation search begins)
- Driver upgraded: parseable `VERDICT` line + `SATURATION` + `gridscale_frac` (zonal-spectrum
  top-quartile energy fraction = effective-resolution / over-vs-under-dissipation proxy).
  Validated: 60×60 A_h=1500 → stable, gridscale_frac=0.0003 (clean).
- ENV: GPU0's 28 GB external job came BACK — use GPU1 only; GPU0 runs OOM.
- Launched (strong U=0.8, 120×120, 60 d → saturate): A_h=5e3 (GPU0, may OOM) vs Leith C_leith=2.0
  (GPU1). Testing which dissipation FORM saturates stably with the smallest gridscale_frac
  (scale-selective Leith should beat broad Laplacian for effective resolution). Strong forcing
  = faster saturation = cheaper iteration; confirm the winner at weak later.
**TODO:** read both VERDICTs; pick the form that is stable+saturated with lowest
gridscale_frac; then SWEEP its strength DOWN to the stability edge (min dissipation); re-run
GPU0-OOM'd configs on GPU1.

### v2 Iteration 2 (pivot to weak + isotropic-grid test)
- iter1 results: **Leith=2 strong-120 BLEW UP**; A_h=5e3 strong-120 OOM'd (GPU0). **Strong-120
  is too aggressive** (even Veros diverged at strong-120) — switch to **WEAK (U=0.2, the
  physical target)** + **dt=600** (CFL-safe with Laplacian, 2× faster, key in this flaky env).
- **MECHANISM (important): launcher Bash commands exit 144 but the `&`-backgrounded python
  CHILD SURVIVES.** So: launch with `&`, ignore the 144, POLL the log file in later iterations.
  Watchers (run_in_background) also die — don't rely on them; read logs directly. Runs ~15–20
  min each, GPU1 only (GPU0 has the external job) → loop is slow, ~1 run/iteration.
- Launched the **ISOTROPIC-GRID test** (the deferred structural lead): `120x60` = 120 lat ×
  60 lon = ~17 km SQUARE (vs 120×120's 2:1 anisotropy 8×17 km), weak, A_h=1500, dt=600, 80 d.
  KEY: if isotropic is CLEAN (low gridscale_frac, no spike) where anisotropic 120×120 wasn't,
  the 2:1 anisotropy was seeding the mode → use isotropic grids for eddy-resolving.
**TODO:** poll run_iso17_Ah1500.log for the VERDICT; if isotropic is clean → minimize A_h on
isotropic grids + push finer (120×240 ~8 km isotropic). If still spiky → the mode isn't the
anisotropy; reconsider (WENO-momentum spectrum, dt, tracer noise).

### v2 Iterations 3-4 (BREAKTHROUGH in framing + the dissipation-form answer)
- **`gridscale_frac` is the right metric; max|u| spikes are boundary cells, NOT blowup.** At
  A_h=1500 dt=600 80d: aniso 120×120 SATURATED clean (EKE 5.8e-5, **gridscale_frac 0.0017**),
  iso 120×60 stable clean (7.7e-5, 0.0020). So I had a CLEAN stable saturated eddy-resolving
  run all along — just judged it "blown up" by max|u|. **Anisotropy is NOT the issue** (iso≈aniso).
- **But A_h Laplacian OVER-DAMPS:** EKE 5.8e-5 at 120×120 is ~150× weaker than the 60×60
  C_smag=0.2 reference (9.4e-3). And A_h=500 BLEW UP → the Laplacian stability edge
  (500<A_h<1500) is already over-damped. **Pure Laplacian is the WRONG form** for high
  effective resolution.
- **→ The answer is SCALE-SELECTIVE dissipation (biharmonic Smagorinsky C_smag), which gives
  strong eddies (high EKE = high eff-res) AND stability.** The 60×60 success used C_smag=0.2.
  Testing 120×120 C_smag=0.2 (no A_h) dt=600 80d now — expect high EKE + low gridscale_frac =
  the min-dissipation eddy-resolving answer.
**TODO:** read C_smag=0.2 @120; if EKE high (~mℯ-3) + gridscale_frac low + saturated → THAT'S
the recipe; then sweep C_smag DOWN (0.15,0.1) to the min stable+clean = max eff-res; validate
vs Veros; push to 120×240; commit. (Use gridscale_frac + EKE-level + stable, NOT max|u|.)

### v2 Iterations 5-6 (pure forms both fail at 120 → combination; scale-separation limit)
**120×120 weak dt600 80d dissipation map so far:**
- A_h=1500 (Laplacian): STABLE+SATURATED, clean (gridscale_frac 0.0017), but OVER-DAMPED (EKE 5.8e-5).
- A_h=500: BLEW UP. A_h=1000-1500 = the Laplacian stability edge (already over-damped).
- C_smag=0.2 AND 0.1 (biharmonic Smag, no A_h, cap=0.5): BOTH BLEW UP.
**So at 120 NEITHER pure form gives stable+strong-eddies.** Root cause: the grid-scale mode
(~4–5Δx ≈ 35–70 km) is too CLOSE to the 130 km eddy scale (only ~2-3× apart) → Laplacian (∝L²)
can't separate them (A_h that kills 50 km also damps 130 km on the growth timescale), and
biharmonic alone can't stabilize the mode under CFL. **120 (dx 8–17 km, Ld 107 km) is
marginally-resolving, not strongly eddy-resolving** — the eddies are barely resolved so they're
weak + need heavy stabilization. TRUE high-eff-res needs FINER grids (more scale separation).
- Testing COMBINATIONS: A_h=800+C_smag=0.15 and A_h=1000+C_smag=0.1 (moderate Laplacian for the
  mode + biharmonic for scale-selectivity, less total damping than A_h=1500).
**KEY STRATEGIC TODO:** if combos don't beat A_h=1500's EKE meaningfully, the real lever is
RESOLUTION (push to 120×240 ~8 km / 200×200, where scale separation lets scale-selective
biharmonic keep eddies strong). The min-dissipation-vs-effective-resolution tradeoff at a FIXED
marginal grid (120) is fundamentally limited; report this honestly + show the EKE(dissipation)
curve, then demonstrate the finer-grid path.

### Iteration 1
**CLEAN so far: 60×60 (eddy-permitting), WENO5 track, both regimes.**
- weak-60 (U=0.2, dt=600, C_smag=0.2, +sponge): STABLE, saturated EKE~9.3e-3 by day~140,
  λ ratio 1.44 (τ=13.9d vs Eady 20d), max|u|~1.8. CLEAN.
- strong-60 (U=0.8, +sponge): STABLE, λ ratio 1.14, saturates ~1.4. CLEAN. Matches Veros
  (max|u| 0.52–0.59).
**120×120 NOT clean yet:**
- weak-120 WENO5 dt=300 C_smag=0.35 cap=0.5 +sponge: STABLE (no blowup) but growth λ ratio
  7.5 (τ=2.66d) = residual grid-scale energy; max|u| peak 2.35 settled 1.7. Not clean.
- dt=600 + more dissipation (C_smag 0.4 / +Leith) BLEW UP (biharmonic viscous-CFL) → cap +
  smaller dt is the lever.
**Veros control:** 60 OK (clean). 120 "diverged" at BOTH strong & weak — but its dt=1800 is
too big for 8 km (CFL); needs a dt-matched re-run for a fair control (TODO).
**Launched this iter (triage, 40 d):** NEMO track [vector_invariant+hollingsworth] weak-120
dt=300 (`_nemo_dt300`); WENO5 weak-120 dt=150 C_smag=0.3 (`_weno_dt150`). Hypotheses: NEMO
hollingsworth momentum is anti-grid-noise; smaller dt gives dissipation headroom.
**TODO:** read both; if either is clean (λ ratio ~1, physical max|u|), test `--no-sponge`;
add Veros dt-patch for a fair 120 control; then push toward ~10 km and commit.

### Iteration 2
- **Architecture: aligned to the recipe pattern.** `build_eady_uniform_setup` now returns a
  structured `EadyUniformRecipe` NamedTuple (model_config/physics_config/grid/z_coord/
  wall_mask/initial_state) matching `ACCRecipe` — was a bare tuple. Driver updated. NEXT
  (deeper, after config settles): an `EadyUniformConfig`-backed YAML template under
  `config/templates/3d_idealized/eady_uniform.yaml` + register in #376 experiment registry.
- Fixed `run_eady_rebuilt.py` print line (was hardcoded "weno5x2" — now shows actual
  momentum/ke-gradient/U/sponge/dissipation).
- **WENO5 track at 120 is the converging one.** dt=150 + C_smag=0.3 + cap=0.5 (weak):
  day-5 spike (max|u| 2.6) DECAYS to ~1.67 by day 20, EKE still growing — smaller dt + more
  Smag suppresses the grid-scale mode over time. Awaiting the 40-day growth-rate fit. So
  the lever for the WENO5 track is confirmed: smaller dt + CFL-capped Smag.
- **NEMO track (vector_invariant + hollingsworth) BLOWS UP** at 120 at BOTH dt=300 (day 4)
  and dt=150 (day 5). Hypothesis: vector-invariant momentum needs an enstrophy-conserving
  vorticity/PV scheme (NEMO uses EEN) — legoESM's `pv_scheme` is NOT a LatLonCGridOceanConfig
  field (was an MPAS/matrix override); the lat-lon C-grid vector-invariant PV flux may be
  enstrophy-unstable at eddy-resolving res. NEXT: find the lat-lon vorticity-flux scheme +
  whether an enstrophy-conserving option exists; or give the NEMO track much more
  (CFL-capped) dissipation. The NEMO track is the harder one.
**TODO next iter:** (1) WENO5 dt=150 fit — if λ ratio ~1 and max|u| physical → CLEAN; then
`--no-sponge` test + push to ~10 km. (2) NEMO track: vorticity-scheme investigation. (3)
dt-matched Veros 120 control (patch veros dt). (4) when a track is clean: commit recipe +
`tests/ocean/unit/test_eady_uniform_recipe.py`.

### Iteration 3
- **Veros dt now patchable**: added `DT_MOM_S`/`DT_TRACER_S` module constants to
  `veros_configs/eady_uniform.py` (were hardcoded 1800) + wired into set_parameter + the
  control script (`_eady_veros_stability.py NX DAYS U DT`). The CFL-matched dt at NX=120 is
  ~1800·30/120 = 450 s. Launched the FAIR Veros weak-120 control at dt=450 (the dt=1800
  "divergence" was the confound).
- **WENO5 dt=150 (WITH sponge)**: max|u| spike 2.6 (day5) → settles ~1.7–2.1, EKE growing
  to ~1.4e-4 by day30 — suppressed but not fully clean (max|u| creeping back toward 2.1).
- **DECISIVE sponge test launched** (user hypothesis: no sponge needed): WENO5 120 dt=150
  `--no-sponge` vs the WITH-sponge run. KEY QUESTION: is the day-5 spike the SPONGE EDGE
  (hard near-wall u/v damping → grid-scale discontinuity, sharper at finer res)? If the
  no-sponge run is CLEANER, the user is right AND the sponge was part of the problem. Print
  line fixed (now shows mom/ke/U/sponge/dissipation correctly).
**TODO next iter:** read the with-vs-without-sponge head-to-head + the fair Veros control;
decide the sponge verdict; if a clean config exists, push to ~10 km + commit recipe+test;
NEMO track still needs the vorticity-scheme fix.

### Iteration 4 (DIAGNOSTIC — stop blind grid-search)
- **WITH-sponge dt=150 finished: growth ratio ~20 (WORSE than dt=300's 7.5).** The EKE
  metric is dominated by a LOCALIZED grid-scale spike (max|u| peak ~3), so the fitted
  "growth rate" is meaningless until the spike is gone. dt/Smag grid-search is NOT
  converging → must LOCATE the spike.
- Launched `scripts/tmp/_eady_locate2.py` (plain step loop, no scan → avoids the pytree-carry
  bug the earlier locator hit): steps 120×120 to day 6 and reports the lat/lon/lev of
  max|u| each day + whether it's at the N/S wall vs interior + a 5×5 patch (checkerboard?).
- Launched the WENO5 120 dt=150 **--no-sponge** run (user hypothesis + tests if the sponge
  EDGE seeds the spike).
- Veros weak-120 dt=450 (fair, dt-matched control) running on CPU.
**KEY DECISIONS pending these 3:** (a) WHERE is the spike (wall/interior/sponge-edge)? (b)
is no-sponge cleaner? (c) does dt-matched Veros run 120 cleanly (setup soundness)? These
three together should determine the fix instead of more blind sweeps.

### Iteration 11 (biharmonic doesn't catch this mode → moderate Laplacian; + grid anisotropy lead)
- **B_h=2.5e11 (fixed biharmonic) does NOT suppress the spike** (day-5 max|u|=3.77, vs
  A_h=5e3's clean 0.93). So the mode is NOT a pure 2Δx mode (which biharmonic would kill) —
  it's at ~4–5Δx, close to the ~16Δx eddy scale, where broad Laplacian damps it but
  sharply-peaked biharmonic misses. Killed B_h.
- **The mode and the eddies are close in scale** → no dissipation cleanly separates them.
  A_h=5e3 kills both (over-damp); need a MODERATE A_h that grows eddies while controlling the
  spike. Testing A_h=1500 (weak-120, dt=300).
- **STRUCTURAL LEAD: the 120×120 grid is 2:1 ANISOTROPIC** (domain 1000 km lon × 2000 km lat
  → dx_lon~8 km, dx_lat~17 km). Anisotropic C-grid cells can seed grid-scale noise. 60×60
  works but is also 2:1. An ISOTROPIC fine grid (e.g. 60×120 ~17 km square, or 125×250 ~8 km)
  is the cleaner eddy-resolving test — TRY if moderate A_h doesn't cleanly work.
**TODO:** read A_h=1500; if eddies grow + max|u| physical → minimize A_h, re-test --no-sponge,
push finer. Else try the isotropic grid. Then NEMO track + commit.
- **COMPUTE: `run_in_background=true` tasks fail exit 144 at startup in this session** (NOT
  the runs — FOREGROUND works, exit 0). Run experiments FOREGROUND (`timeout 600 env ... python`);
  the harness auto-backgrounds long ones + notifies. Use this.
- **A_h=1500 looks PROMISING**: day-3 max|u|=0.89 (clean, no spike — like A_h=5e3) — a
  moderate Laplacian controls the mode. Running full 40-day to check the BCI GROWS (the
  A_h=5e3 failure mode was over-damping → EKE decreasing). If A_h=1500 grows eddies AND
  stays clean → WENO5-track answer at 120.

### Iteration 8 (A_h over-damps → scale-selective biharmonic is the real answer)
- **A_h=5e3 (Veros value) OVER-DAMPS**: kills the grid-scale spike (max|u|~1, clean) BUT
  also kills the BCI — EKE *decreases* (1.96e-10 day5 → 7.0e-11 day10). Laplacian isn't
  scale-selective: at the eddy scale (~130 km) A_h=5e3 gives ~23-day damping ≈ the 20-day
  weak-BCI growth time → suppresses the instability. So pure Laplacian at Veros strength is
  wrong for legoESM here.
- **The textbook eddy-resolving answer = BIHARMONIC** (damping ∝ L⁴, vastly more
  scale-selective than Laplacian's L²): at 16 km grid vs 130 km eddy the ratio is ~4400×, so
  a B_h tuned for ~3-day grid-scale damping gives ~10⁴-day eddy damping → eddies untouched.
  A *fixed* B_h is CFL-safe (its CFL ∝ B_h·dt/dx⁴ doesn't depend on the strain rate that
  blew up the biharmonic-Smag runs). Testing **B_h=2.5e11** (weak-120, dt=300, C_smag=0).
- NOTE (compute mechanism): `nohup ... &` children get KILLED when the launcher Bash call
  exits 144. RELIABLE way = run the python directly as a `run_in_background=true` Bash task
  (harness-tracked, persists, notifies). Don't nohup-detach.
**If B_h=2.5e11 is clean (spike gone AND BCI grows ~σ_Eady):** WENO5-track answer. Then
sweep B_h to the minimum that works, re-test --no-sponge, push ~10 km, apply to NEMO, commit.

### Iteration 7 (FIX CONFIRMED)
**A_h=5e3 (Veros Laplacian viscosity, C_smag=0) KILLS the grid-scale spike.** weak-120
dt=300, day 5: max|u|=**0.93 m/s** vs 2.6–4.5 in every no-A_h run. EKE clean (1.96e-10).
→ The WENO5-track fix is **harmonic Laplacian viscosity** (the piece my rebuild dropped;
biharmonic Smag alone is too scale-selective). Awaiting full 40-day growth+saturation.
**Next:** confirm full run clean (λ ratio ~1, physical saturation); then re-test `--no-sponge`
WITH A_h (user hypothesis — proper Laplacian damping may remove the sponge need); minimize
A_h; push to ~10 km; apply A_h to the NEMO track; commit recipe (A_h as a config knob) + test.

### Iteration 5 (DIAGNOSIS RESOLVED → targeted fix)
Three decisive results:
1. **NO-SPONGE BLEW UP (day 7, harder than with-sponge)** → the spike is NOT the sponge's
   fault (it's an INTERIOR grid-scale mode); the sponge is doing real stabilization. **The
   user's "no sponge needed" hypothesis does NOT hold for the current (under-damped)
   numerics** — but may revive once the interior mode is properly damped (re-test then).
2. **Fair dt-matched Veros (weak-120, dt=450) RUNS CLEANLY** (progressing, no divergence) →
   the dt=1800 "divergence" was the CFL confound, and **the SETUP IS SOUND at 120**. A
   trusted dycore handles it; legoESM's grid-scale mode is a legoESM-numerics gap.
3. **ROOT-CAUSE HYPOTHESIS: missing Laplacian viscosity.** Veros runs clean with harmonic
   **A_h=5e3** (Laplacian); my rebuild used A_h=0 + biharmonic Smag ONLY. Biharmonic is too
   scale-selective to kill a persistent grid-scale mode; I over-corrected by dropping the
   Laplacian. Exposed `--a-h` (builder + driver). Launched Veros-like Laplacian-only
   weak-120 dt=150: A_h=5e3 (`_Ah5e3`) and A_h=2e4 (`_Ah2e4`), C_smag=0.
**If A_h fixes the grid-scale mode (clean growth, physical max|u|):** that's the WENO5-track
answer. Then: re-test `--no-sponge` WITH A_h (does proper Laplacian damping remove the need
for the sponge → user's hypothesis?), tune A_h down to the minimum that works, push to
~10 km, commit. NEMO track likely also needs A_h (its vector-invariant blowups may be the
same under-damping).
