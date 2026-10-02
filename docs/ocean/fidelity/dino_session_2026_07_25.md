# DINO NEMO-oracle — session record, 2026-07-24/25

*Companion to `dino_handoff_2026_07.md` (the standing program status). This file records
one long session: what shipped, what was learned, and — importantly — five attributions
that were made and then retracted, with the reason each died. Branch
`feat/nemo-dino-topo-bridge`; PR #1313 (merged) and PR #1343 (open). Issues #1226, #1317.*

## What shipped (all reviewed, committed, pushed)

**Model-correctness fixes** (these change results):
- **Halo-strip domain fix** — NEMO 5 files are haloless; reading with `nn_hls=0` gives the
  true 199×52 domain. Cell census exact (342,134 wet + face masks). Resolved the SSH
  smoothness deficit outright (ratio 0.51 → 1.00).
- **Sub-seafloor tracer leak** — `pn2` masked at construction per `eosbn2.F90:1467`, 3-D MLD
  masks. Poison gate landed: dry-cell values must produce exactly zero wet response.
- **Barotropic vvl flux-depth gap** — the substep continuity flux now uses face depths from
  the mid-step AB3-extrapolated ssh (`dynspg_ts.F90:556-609`), with a per-substep
  NEMO-sequence oracle test.
- **GM bolus ~3× too strong** — lego's Treguier κ_GM was built from simplified cell-centred
  slopes instead of NEMO's `ldf_eiv` W-point `wslpi/wslpj` summed over the full column
  (`ldftra.F90:643-726`). κ_GM correlation 0.28, mean 646 vs 196 m²/s. Fixed; tracer
  advection tendency vs NEMO 0.93 → 0.99.
- **From-rest Euler-start crash** — once `nemo_before`/`nemo_burchard` were on the card, a
  fresh analytic IC crashed at step 0 (guards read `T_before`/`u_before`, which don't exist).
  Fixed by seeding `before := now` at cold start — NEMO's own convention
  (`istate.F90:97-99,135-137` sets `Kmm := Kbb`).

**NEMO zdftke gap closure** — every DIFF/PARTIAL row of the Phase-1 alignment table on
#1317: T3 exact surface `avm(jk=1)` (plus a double-counted surface coupling inflating TKE
~2×), T4 Burchard now×before shear, T6 explicit buoyancy sink, T8/T13 `rn2b` operands, T15
per-column bottom BC, T18/T19 mixing-length floors, T21 ceiling, T23 MAX-only backgrounds.
T11 (coefficient vintage) deliberately not built — later measurement showed it isn't the lever.

**Instruments** (in `scripts/validate/ocean_fidelity/dino_1226/` + `ocean/fidelity/`):
- `kamm_twin_90d.py` — state-initialized twin with a **hard day-0 gate** (bit-identical to
  NEMO's restart, non-rest), `--bridge-tke`, `--bridge-before`, `--vmix-scheme`, `--save-3d`.
- `heat_discriminator.py` — redistribution-vs-loss vertical drift discriminator.
- `mode_projection.py` — shared checkerboard-mode projector.
- `box_heat_budget.py` + runners — online box heat-budget accumulator; VERTMIX measured
  **directly** (implicit-realized), residual reported separately as a closure diagnostic.
- `tendency_probe.py` fixed to mirror production (it was dropping the cos-lat Redi override).

## Certification status

| metric | value | vs NEMO |
|---|---|---|
| ACC year 1 (from rest) | 53.2 Sv | 0.887× (NEMO 60.0) |
| ACC year 5 (from rest) | 66.9 Sv | 0.734× (NEMO 91.1) |
| SST corr / bias (y5) | 0.996 / −0.22 K | — |
| SSH small-scale ratio (y5) | 0.87 | 1.00 |

The ACC deficit is a **spin-up-rate** deficit, not a static offset: lego starts within 11% and
grows at ~44% of NEMO's rate. NEMO itself is far from equilibrium at year 5 — its own 20-year
run climbs 60 → 91 → 145 Sv, still +1.8 Sv/yr at year 20, heading toward the paper's ~206 Sv
(Kamm et al. 2025, R1). So year-5 ratios compare two spin-up trajectories, not two equilibria.

Momentum budget is **fully closed and faithful**: wind input (1.000), topographic form stress
at the sill including wall cells (corr 0.9999, ratio 1.017), bottom drag (`r_eff` ratio 1.000,
sink 1.006), all momentum tendencies 0.99–1.0. Bottom drag is ~0.1% of wind input in both models.

## Five attributions made and retracted

Each was announced, then killed — none by physics, all by measurement artifacts.

1. **"Vertical mixing 8× too weak."** Died: NEMO's `ttrd_zdf + ttrd_evd` double-counts.
   `ttrd_evd` is a *non-additive re-diagnostic* of a term already inside `ttrd_zdf`
   (`zdfphy.F90:313/323`, `zdfevd.F90:79-102`); NEMO's own budget closes to 0.4% *without* it
   and overshoots by exactly `ttrd_evd` *with* it. NEMO's source even flags it:
   `trdtra.F90:132 "!!gm Gurvan, verify the jptra_evd trend please !"`.
2. **"Two compensating defects (false convective trigger + dead ambient closure)."** Died with
   (1), and separately: the trigger cross-tabulation showed **99.97% agreement** with NEMO
   (6 false positives, 1 missed, n=23,608), with lego's convective K_v exactly 100 m²/s =
   `rn_evd` where it fires. The "28–40% spurious convection" was measured against in-situ N²,
   a different stability criterion, not NEMO's actual trigger.
3. **"Redi 53% too strong."** Died: the diagnostic probe omitted the cos-lat κ_Redi override
   that production applies. Corrected, Redi matches at 0.96–0.9997. Production was never affected.
4. **"Deep-band convergence carried by vertical advection."** Died: the H/V split is a
   *bookkeeping convention*, identical in both codes (both fold the GM bolus into the advecting
   transport before splitting, both limit per-direction). Individual components differ by up to
   47×; ADV_TOTAL — the only fair comparison — matches to 0.6–2 W/m². I had read a within-model
   bucket as a cross-model attribution.
5. **"Composition/ordering difference drives the divergence."** Died: the cross-model single-step
   residual (1.3996e-03 K) is numerically *identical* to lego's **own** probe residual, while
   NEMO's own is 1.28e-08 K — so the signal arises entirely within lego, with no NEMO data
   involved. A probe limitation, not a cross-model finding.

## Rules earned (the durable output)

- **NEMO's `trddyn`/`trdtra` trends are integrator bookkeeping, not physics buckets.** Never
  quote a trend comparison without first proving closure against `ttrd_tot`. Three separate
  traps: `vtrd_zdf` is ~98.6% barotropic-strip bookkeeping; `ttrd_zdf` folds in the isoneutral
  vertical contribution; `ttrd_evd` is non-additive. **Every** unclosed trend comparison this
  session produced a false finding.
- **A residual bucket cannot be an attribution.** If a term is computed as
  `TOTAL − (other terms)`, a signal in it is "that term plus all other terms' error."
- **Explicit flux is the wrong diagnostic for an implicitly-applied K.** At `K·dt/dz² ≫ 1` the
  backward-Euler solve saturates: explicit −108 vs realized −5.93 W/m², an 18× error. Always
  realize the flux through an actual implicit solve.
- **Verify agent claims against the filesystem.** Two agents narrated background work that
  hadn't started; others narrated confusingly while their work was real.
- **Compare like with like, and check the instrument before the physics.** Three of the five
  retractions were instrument bugs, two were convention mismatches. None were physics.

## Structural facts established (not attributions — just true)

- **NEMO composes additively/simultaneously**: every `tra_*` routine takes `Kbb`/`Kmm` as
  `INTENT(in)` and accumulates only into `ts(:,:,:,:,Nrhs)` (zeroed `stpmlf.F90:336`);
  `tra_zdf` (:370) applies the single state update. No routine reads what another wrote.
- **lego composes sequentially**: `_leapfrog_step` runs `_step_impl` twice (Nnn advective,
  Nbb dissipative) then base-shifts; within one call, GM/Redi is layered on the already-updated
  field (`ocean_model_latlon_cgrid.py:3696, 3784`). This is *deliberate*, matching NEMO's
  forward-in-time Nbb diffusion.
- Whether that structural difference produces materially different output is **untested**. It is
  an O(Δt²) commutator effect — invisible per-step, potentially significant over 57,600 steps.
  A sound magnitude estimate needs an operator norm, not tendency magnitudes; an ad hoc proxy
  overshot by 180× and was rejected.

## Open, with the next step specified

1. **The ACC deficit has no attributed cause.** Every operator compared on a clean footing
   matches; no validated budget shows a single carrier. The remaining differences are smaller
   than the instruments' own error budgets.
2. **The composition question is open.** To close it: build a probe that reconstructs lego's own
   multi-stage step from its true intermediate `_step_impl` outputs, *then* compare cross-model.
3. Minor: the `>1000 m` band's NEMO closure residual (6.3%) is the least-tight of the three;
   `box_budget_5y.npz` stores per-term totals as scalars (per-term re-windowing impossible —
   fixed in the newer accumulator).

## Assets on disk (regenerable, not committed)

NEMO runs under `~/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/`: `RUN_1Y`, `RUN_5Y`, `RUN_20Y`
(yearly dumps, the ACC growth curve), `RUN_90D_TWIN`, `RUN_STEPDUMP` (per-step restarts with
trends), `RUN_90D_CST`, `RUN_90D_CST_NOGM`. Session scratchpad holds the twin/budget npz series
and plots; all regenerable from the committed harness scripts.
