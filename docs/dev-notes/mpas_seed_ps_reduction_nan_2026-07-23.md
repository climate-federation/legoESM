# MPAS moist-AMIP day-1 NaN regression — bisection dossier (2026-07-23)

## Symptom

Every moist MPAS AMIP run on main (post 2026-07-18) produces NaN in all
prognostics within the first simulated day. Invariant across: dt 100/120/240 s,
nlev 20/40, SCVT level 4/5, convection sbm/bechtold, precision fp32/fp64,
IC era5/uniform, SST source ICON-pool bc files / PCMDI-AMIP-1-1-10 lat-lon.
The identical configuration ran 10+ healthy days on 2026-07-08 code
(`/scratch/b/b309178/evap_sweep_mpas_1783497599`). The cube lane is unaffected
only because the `legoesm_ap` checkout predates the breaking merge.

## Conviction chain (all probes: L4/L20/dt100/bechtold/fp32, 3 days, ERA5 IC,
full CMIP6 forcing; jobs under `/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs/bisect_*`)

LIVE (3/3 days): `7b61d8ac2`, `c1b3a9099` (θ-form), `3bd176034` (vert4 damper),
`c4d679c15` (flux-form continuity), `a16e1493a` (reorg), `d4555befb` (#1146),
`eb374ea45` (#1175), `cb834b520` (#1174), `e233fdce9` (#1168), `135444c05` (#1178).

DEAD (NaN day 1): `5cf43680b` (#1180, twice — independent worktrees),
`f267f6fc6` (#1211), `e74903ae2` (#1247), `8c3d48693` (#1279), main
(`ad7c1b1a5`) with era5 IC, uniform IC, and PCMDI SST.

→ breaking merge: **`4ed72f8c3` #1179 perf/les-mpi-solver-remediation**
(#1178 LIVE, #1179 DEAD). Despite the title, the merge carried AMIP driver
"audit fixes".

File-level grafts onto live #1178: `grids/topography.py` LIVE,
`scripts/run/run_amip.py` LIVE, **`driver/model_driver.py` DEAD**.
Hunk-level graft (fB2): **the MPAS-init hunk alone reproduces the NaN** —
passing `phis` into `held_suarez_init_mpas` so the seed
`p_s = p_ref·exp(−phis/(R_d·T_init))` (≈677 hPa over Tibet) instead of the
old flat-p_s-then-patch-phis.

Control (fB3): phis field kept at construction, seed p_s left FLAT → LIVE.
**The hydrostatic p_s reduction of the seed state is the poison, not the code
path.**

Fix validation (wD): main + revert-to-patch-in → LIVE (matches the live-probe
trajectory bit-for-bit in the printed diagnostics).

## Open mechanism question (follow-up)

The final pre-run states print identically live-vs-dead (the ERA5 overlay
replaces u, T, p_s, phis, q_v), yet the run dies — the seed p_s leaks through
a channel the overlay does not replace. Only visible seed-derived divergence:
the default RH-based moisture init line (q_v 10.15 → 10.46 g/kg; the
saturation helper itself is capped and sane). Suspects for the leak: a
statically-derived reference tied to the seed p_s (conservation target,
reference profile, or a JIT-time bake), or a non-overlaid tracer/aux field.
This likely hides a REAL latent dycore/driver bug: a hydrostatically balanced
initial state must not be less stable than an unbalanced one. Root-cause
before re-attempting the audit's p_s reduction; the audit concern (startup
pressure shock for `ic='default'`) remains real but bounded/benign.

## Second, independent finding: hybrid vertical coordinate at nlev=40

With the seed-p_s revert in place, moist MPAS at **nlev=40 + hybrid**
(`vertical_coord=hybrid`, p_top 200 Pa, stretching 2.0 — the production
defaults) STILL NaNs at day 1, on level-4 AND level-5 meshes, era5 AND uniform
IC, rrtmg AND gray radiation, dt 100-240.  Discriminators (L4-or-L5/L40/dt100
unless noted): hybrid dt100 DEAD; hybrid **dt50 LIVE**; **sigma dt100 LIVE**
(4/4 days); nlev=20 hybrid dt100 LIVE (5/5).  So the constraint is
hybrid-coordinate-specific and dt-scaling (Δσ-halving halves the stable dt) —
a stiffness in the MPAS hybrid vertical transport / mass-flux path at thin
upper-level Δp, NOT a generic vertical CFL (sigma at the same nlev/dt is
fine) and NOT physics-specific (gray dies too).  The July dry L5/L40 run was
stable because dry dynamics drive far weaker vertical mass flux.
Follow-up: root-cause the MPAS hybrid vertical numerics at nlev>=40 (compare
`vertical_advection_hybrid` / `compute_mass_flux_from_cumsum` stability vs the
sigma branch); until then MPAS production configs use nlev 20-30 (hybrid) or
sigma at L40, with the dt ladder: hybrid L20@dt100 validated, hybrid L40 needs
dt<=50, sigma L40@dt100 validated.

## Third finding: slow-onset top-of-model blowup (the ORIGINAL day-11 disease)

With both fixes above, moist MPAS AMIP still NaNs after ~1.5-2 weeks:
L5/L20/dt100 at day 14 (chained AND unbroken — bit-identical trajectories, so
the restart path is exonerated), L5/L30/dt75 (PCMDI SST pilot) at day 10, and
historically L4/L20/dt100 at day 11 (Jul-8 sweep).  Forensics on the day-5 vs
day-10 checkpoints: the growing mode is LOCALIZED at the model top (k=0) in
the EDGE WINDS — worst-cell vertical-Nyquist |Δ²u| doubles per 5 days
(23.6→52.8 m/s) while the global p99 grows only ~10%, plus steady top-level
warming (~0.45 K/day).  Classic rigid-lid gravity-wave accumulation.  The
hydrostatic MPAS PE has NO top sponge (the NH MPAS dycore has one;
``--sponge``/#836 was latlon-only and a silent no-op on MPAS).  Dry runs
survive because moist convection is the wave source.

FIX ATTEMPT (sponge): ``_run_mpas`` now applies the #836 sin²-profile Rayleigh
decay to the MPAS edge winds as a config-gated post-step multiply
(``--sponge``).  REFUTED as the cure: sponged runs die EARLIER (day 8-9 vs
9-14) — the sponge perturbs the trajectory but the instability is elsewhere.
(Kept as an option; a lid sponge is standard practice regardless.)

## ROOT CAUSE LOCALIZED (2026-07-23 02:3x): moist-orographic 2Δσ checkerboard
## at ONE cell, amplified to global NaN by the GLOBAL conservation fixer

Instrumentation chain that found it:
1. ``JAX_DEBUG_NANS`` restart from a day-8 checkpoint — caught NaN only at the
   eager daily diag (no op-level info; the step is one jit).
2. Sub-daily blowup capture: a debug ``--config`` YAML with ``diag_days: 0.01``
   (argparse ``type=int`` validates only CLI strings — config defaults inject
   floats unvalidated) → finiteness check + dump every ~8 steps.  Result: the
   NaN is GLOBAL (all 10242 cells) within <=9 steps of onset — physically
   impossible by advection → a GLOBAL operation broadcasts it: the
   ``fix_mass``/conservation fixer's global-sum rescale (one NaN cell → NaN
   integral → every cell NaN in one step).  This is why every earlier autopsy
   was 100%-NaN and why onset days looked chaotic (any single-cell event
   anywhere kills the planet within a step).
3. Fixers OFF (``--no-conservation-fixer --no-fix-mass``) → NO NaN; instead
   the physical-bounds guard fires: T in [115.9, 400.5] K — a FINITE dump with
   the anomaly intact: **cell 7480, 36.3N 80.1E (Tibetan plateau, p_s=654
   hPa): vertical T profile [213.6 217.6 312.8 175.3 400.5 115.9 237.3 200.2
   231.4 ...] — a +-140 K 2Δσ checkerboard at levels 2-7 over a smooth lower
   column, with a 66 g/kg (!) q_v spike feeding latent-heat amplification.**

This is the #930 vertical checkerboard re-emerging under moist-orographic
forcing.  The purpose-built damper ``mpas_nu_vert4_T`` defaults to 2e-6 1/s —
evidently ~an order too weak for this regime (filter stability headroom is
huge: nu <= 1/(16 dt) ~ 6e-4).  The lid sponge missed it because the action is
at sigma ~0.2-0.4, below sponge_sigma_top=0.15.

FIX CANDIDATE: ``--mpas-nu-vert4-t 2e-5`` (and 1e-4 as the strong arm),
validated by 25-day warm-started probes.  Follow-ups: (a) the conservation
fixer should be NaN-guarded (skip the rescale when the global integral is
non-finite — a one-line jnp.where — so a local event can never poison the
globe and the bounds guard reports the TRUE origin), (b) the q_v ~66 g/kg
super-saturation pooling at the checkerboard hot levels deserves its own
bound/diagnosis, (c) revisit the vert4 default for moist MPAS configs.

## Codex adversarial review verdict (2026-07-23 ~06:30, gpt-5.6-terra xhigh)

**F1 (CRITICAL, CONFIRMED): the MPAS ERA5 branch never attached ERA5 moisture
to the prognostic state** — it wrote `self.tracers["q_v"]` (a dead driver
dict) while the dycore advects `state.tracers`, still holding the moist-init
RH taper `rh_init·q_sat(seed_T, seed_p_s)·σ²`.  The spectral branch has the
re-attach; MPAS was missing it.  This RESOLVES the "seed p_s leaks into ERA5
runs" mystery: seed q_v scales with q_sat(seed p_s), so the seed-p_s choice
modulated the latent FUEL at the runaway cells — the seed-p_s revert was a
symptom treatment (less fuel), and every "ERA5" run to date integrated taper
moisture (CWV ~52 start) instead of real ERA5 (~23).  FIXED on main
(model_driver MPAS branch now mirrors the spectral re-attach).  Decisive
probes: F1-fixed, NO qcap, at both grave configs (sigma-L40/40d vs day-24;
sigma-L30-PCMDI/70d vs day~51) — if they clear, the moisture fix alone
stabilizes and qcap retires to an optional guard.

Other confirmed findings (see the physics-validator report for full detail):
F2 qcap cannot fire at the hot-phase pool (q_sat smooth-caps at 1 kg/kg) —
it fires post-cooling, performing the detonation itself in one step;
F3 qcap omits the psychrometric temperature feedback (over-condenses,
discontinuous at the 1.1 threshold); F5 the seed-p_s revert leaves
`ic=default` PGF-unbalanced over terrain (empirical stopgap, caveat added);
F6 the tracer del-4 column-conservation claim FALSE under hybrid (holds for
uniform-sigma); F7 del-4 + positivity clip = small uncompensated moisture
source (no moisture fixer on `_run_mpas`) — quantify drift or add a fixer
before trusting the hydrological cycle; F4 qcap's p_full is pure-sigma
(fine in-scope, blocks hybrid); F8 qcap stales the held radiation cache
between radiation solves; F9 qcap is not JIT/AD-compatible (eager-loop only);
F10 no-q_c edge case loses water; F11 number-tracer units labels wrong.
Mechanism wording correction: hot-phase immunity is caused by the SIGN of
q_v−q_sat (q_sat rises with T then caps), NOT by the sigmoid sharpness; the
sharpness independently under-condenses moderate supersaturation.
Cleared: sponge sign/broadcast/mass-ordering; qcap enthalpy factor
(c_pd·T+L_v·q_v conserved when q_c present); vert4 default activation is
documented-not-accidental.

## Status

- Revert applied to `packages/coupler/legoesm/driver/model_driver.py`
  (working tree, this checkout) with a pointer to this dossier.
- Codex adversarial review NOT yet run (codex commands unavailable in the
  driving session) — owed before the revert is pushed to main; the empirical
  validation above (10 live / 8 dead probes + fix probe) is the interim
  evidence base.
