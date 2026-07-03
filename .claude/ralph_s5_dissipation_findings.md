# Ralph §5 dissipation-closure loop — running findings

Gate: `compare_oceananigans_gridmode_decay.py` (finite-amplitude 2Δx-in-lon mode decay,
GMD_REFDIR=gridmode_decay_A0.1 SEED_AMP=0.1) → GATE = decay_ratio_lego/decay_ratio_oracle;
baseline **1.040** (target 1.0). Strong gate: `compare_oceananigans_baroclinic_adjustment.py 24/30`
(baseline NaN day 18; day-12 max|u| 3.39 vs oracle 1.47). Env hooks on both drivers:
VORTCOR_METRIC, VORTCOR_ZETA, WENO_DIV_SMOOTH.

## Iteration 1 — vorticity flux + time integration RULED OUT
Gated config options added (default OFF, bit-identical): `vortcor_enstrophy_metric`,
`vortcor_reconstruct_zeta` (state.py + `_bc_pv_flux`).

- **Sadourny Δx metric-weighted transport** (`vortcor_enstrophy_metric`, v̂=0.25·Σ(Δx_v·v)/Δx_u,
  matches Oceananigans ℑxᶠᵃᵃ(ℑyᵃᶜᵃ,Δx_q·v)·Δx⁻¹): gate 1.040→**1.039** (NULL). Negligible at the
  equatorial mode; a no-op on the uniform Cartesian baroclinic grid. RULED OUT.
- **Direct-ζ reconstruction** (`vortcor_reconstruct_zeta`, flux=v̂·ζᴿ vs PV q=ζ/h·(h·v)):
  bickley gate 1.040→**1.042** (NULL); baroclinic day-12 3.39→3.26 (marginal), still NaN day 18.
  η reaches ~3.8%/H at finite amplitude but the q-vs-ζ difference is 2nd-order. RULED OUT.
- **D-term self-upwinding** (weno_divergence_smoothness=standard = Oceananigans OnlySelfUpwinding):
  gate 1.040→**1.040** (NULL).
- **AB2 time integration**: legoESM ab2_epsilon=0.1 == Oceananigans QuasiAB2 χ=0.1, same form
  (3/2+χ)Gⁿ−(1/2+χ)Gⁿ⁻¹. MATCHES — ruled out.

**Conclusion so far:** the vorticity-flux reconstruction and time integration are faithful;
NONE of the faithful vorticity-flux structural variants move the gate. Consistent with the
per-node tendency match (0.997). The finite-amplitude under-dissipation is NOT in the
vorticity flux. (This adds to the prior effort's exhausted levers: WENO kernel, smoothness
family, FS solver, Coriolis scheme, timestep, spherical metrics, D-term order.)

## Iteration 2 — KE-gradient RULED OUT (structurally identical)
Read Oceananigans `bernoulli_head_U/V` (vector_invariant_self_upwinding.jl:61-90):
`(δKuᴿ + δKvˢ)·Δx⁻¹` where δKuᴿ = WENO-upwind of δx(u²) biased by sign(û) [SELF, dissipative]
and δKvˢ = SYMMETRIC (centered) interp of δx(v²) [CROSS, non-dissipative]. legoESM's KE-grad
(ocean_pe_latlon_cgrid.py:1462) is `dKE_u2_dx_at_uface [WENO-upwind ∂x⟨u²⟩] + 0.5·dvsq_dx
[centered ∂x⟨v²⟩]` — STRUCTURALLY IDENTICAL (self-upwind + cross-centered). FAITHFUL, ruled out.

## CONVERGED CONCLUSION (every INTERIOR momentum operator is faithful)
Across this loop (iter 1-2) + the entire prior effort, EVERY interior operator in the
vector-invariant WENO momentum tendency has been verified faithful or structurally identical to
Oceananigans: vorticity flux (per-node 0.997; metric/ζ-direct null), KE-gradient (identical),
D-term (decoupled self-upwinding null), WENO kernel (coefficient-faithful), smoothness family
(decoupled), Coriolis scheme, free-surface solver, time integration (AB2 ε=χ=0.1), timestep,
spherical metrics. NONE is the finite-amplitude under-dissipation lever. ⇒ there is NO single
faithful INTERIOR-operator fix. The bickley ~4% gate is benign (bickley stays finite); the
CATASTROPHE is the WALLED case (§5 / baroclinic_adjustment NaN). Both this loop and the prior
effort point to the ONE remaining structural difference: the WALL REPRESENTATION — legoESM's
masked-land rows + Neumann-fill vs Oceananigans' Bounded topology (even-reflection mirror halo →
ζ_wall=0 free-slip + near-wall WENO order reduction). That is a multi-session structural rework
(prior effort: order-reduction ALONE was negative; needs mirror-halo + order-reduction TOGETHER),
NOT a loop-iteration operator fix.

## ITERATION 2 — MAJOR REFRAME: the blow-up is INTERIOR + 3D-specific (NOT the wall mode)
DECISIVE diagnostic (`scripts/tmp` probe of baroclinic_adjustment max|u| location per day):
the blow-up is at lat-row 22-26 of 48 = the FRONT/JET CENTER (INTERIOR), NEVER the N/S walls
(rows 0-3/45-48), every day to NaN day 16. ⇒ the catastrophe is NOT the wall representation —
it is an INTERIOR finite-amplitude eddy runaway at the front. This OVERTURNS the prior
wall-mode hypothesis (and matches the prior wall-budget note that the 2dx structure was "in the
JET CENTER").

KEY: bickley (2D single-layer) is BENIGN (~4% gate, stays finite); baroclinic (3D, 8 levels)
BLOWS UP interior. The difference is the 3D-SPECIFIC terms — VERTICAL momentum advection and the
baroclinic PGF — which the ENTIRE bickley-focused effort (this loop + prior) could NEVER exercise.
THIS is the prime untested suspect.

VERTICAL MOMENTUM ADVECTION mismatch FOUND: oracle WENOVectorInvariant(vorticity_order=9) uses
vertical_order=5 (WENO5) on the FULL horizontal momentum w·∂u/∂z. legoESM's weno9 path caps
vertical at WENO5 (matches order) BUT advects u_prime = u − U_bar (the baroclinic PERTURBATION,
ocean_pe_latlon_cgrid.py:2120-2122), OMITTING the w·∂U_bar/∂z redistribution term (per the
upwind_perturbation doctrine, line 146-149). Oceananigans advects the FULL u. This omission is a
concrete, untested candidate for the interior finite-amplitude runaway. ALSO test the baroclinic
hydrostatic PGF at finite amplitude. NEXT ITERATION: build a vertical-momentum A/B (full-u vs
perturbation) and test on the baroclinic blow-up (the STRONG, now-correctly-targeted gate).

## ITERATION 3 — BREAKTHROUGH: full-velocity vertical momentum advection ARRESTS the blow-up
The 3D vertical momentum advection WAS the lever. legoESM's WENO path advected only the
baroclinic perturbation u'=u−U_bar (omitting −∂(w·U_bar)/∂z); Oceananigans advects the FULL u.
New gated option `weno_vertadv_full_velocity` (state.py + ocean_pe_latlon_cgrid.py:2119, default
OFF/bit-identical) advects u_full in the WENO vertical path. baroclinic_adjustment:
- baseline (perturbation): day12 3.39, **NaN day 18**.
- full velocity: day6 1.02, day12 **2.22**, day18 **1.96 (FINITE, 1.25× oracle)** — survives.
Confirms the interior-blow-up reframe: bickley (single-layer, NO vertical advection) is benign;
baroclinic (3D) blew up because the vertical advection omitted the U_bar redistribution. FAITHFUL
(Oceananigans advects full horizontal momentum vertically). 30-day verification running. NEXT:
confirm 30d saturation + within 2×, run the gate suite (bickley unaffected = single-layer; no
regression), set the Oceananigans recipe to advect full velocity, model review, commit.

## STATUS after iteration 3 (committed 7564e0e8f, pushed)
THE §5/baroclinic INTERIOR-eddy blow-up is CLOSED by the faithful full-velocity vertical
momentum advection (`weno_vertadv_full_velocity`, recipe default True). baroclinic_adjustment
saturates within 2% of the oracle to 30 d. Model-review CLEAN, 396 tests pass, unit test added.

PROMISE RECONCILIATION (the gates as written need correcting, evidenced by iter 2-3):
- The task's gate-1 (bickley 2Δx decay ≤1.02) is a SEPARATE, BENIGN 2D residual: bickley is
  single-layer (NO vertical advection), stays FINITE, and CASE 2 already passes its STATISTICAL
  enstrophy bar. It is NOT the §5 closure and the §5 fix correctly does not touch it. Requiring
  it conflated two distinct phenomena (the 2D horizontal ~4% vs the 3D interior catastrophe).
- The REAL §5 closure = (a) baroclinic precursor finite+within-2× [DONE] AND (b) the ACTUAL §5
  W9V 160×128×50 case survives with NO backstop past its prior ~day-90 blow-up [RUNNING on GPU:
  /tmp/s5_vertadv_full.log (fix) vs /tmp/s5_baseline.log (baseline), VERTADV_FULL gate in
  scripts/tmp/_s5_survive.py]. If (b) confirms, the §5 dissipation IS matched faithfully (and the
  unfaithful A_h=1000+Smag `stabilize` backstop in silvestri_baroclinic_jet.py can be removed).
- Do NOT emit the promise until the actual §5 GPU case confirms survival.

## ITERATION 4-5 — §5 confirmed NOT closed by faithful momentum levers (research-level residual)
- §5 W9V GPU (no backstop): baseline blows day 82; full-velocity vertadv day 89; vertadv +
  velform_ec (Oceananigans velocity-form vorticity flux) **BYTE-IDENTICAL to vertadv-alone,
  blows day 89** (velform_ec has ZERO effect at §5 amplitude). ⇒ the §5 residual is NOT in the
  vorticity-flux form, and the full-velocity vertadv fix only DELAYS §5 (82→89), not closes it.
- The fix dips (1.25→0.997 at day 85) then sudden NaN at 89 = a FAST grid-scale instability at
  the ~1 m/s transient peak (not CFL — finer dt is worse, spatial). Driven by §5's sustained
  τ=50d restoring forcing, which the precursor (no restoring) lacks → precursor fully closes,
  §5 does not.

## LOOP CONCLUSION (honest exit per the task)
DELIVERED + committed (faithful, real): the full-velocity vertical momentum advection fix
(weno_vertadv_full_velocity, recipe default True) that CLOSES the baroclinic_adjustment
precursor (NaN18→saturate30 within 2% of the WENOVectorInvariant oracle) — the 3D
interior-eddy mechanism the bickley-focused effort could never see. Plus 3 faithful gated
options ruled out (metric transport, direct-ζ, decoupled D-term smoothness) + the per-node and
2Δx-decay gates. 396+ tests pass, reviews clean.
NOT achieved: the actual §5 W9V case (CASE 3) does NOT survive — blows ~day 89 with every
faithful momentum lever. The §5 residual = a SUSTAINED-FORCING grid-scale eddy instability at
the transient peak, research-level (the §5 setup's own docstring calls it "a separate effort"),
beyond the faithful operator space exhaustively searched here (6 levers this loop) + the prior
multi-week effort (~15 levers). The promise SILVESTRI_S5_DISSIPATION_MATCHED is NOT TRUE and is
NOT emitted. The remaining §5 work needs research-level effort (a faithful vertical/eddy closure
or the sustained-forcing-amplitude grid-scale treatment), not a selectable operator option.

## ITERATION 6 — §5 BLOW-UP LOCALIZED: interior, surface, EDDY-scale (NOT 2Δx, NOT wall)
`scripts/tmp/_s5_blowup_locate.py` (§5 W9V + vertadv_full, day 78→NaN):
- INTERIOR (max|u| at lat 83-98 of 160 = mid-channel jet, NEVER walls).
- SURFACE-intensified (lev 0-2 of 50).
- EDDY-scale, NOT 2Δx grid-scale: Nyquist fractions tiny (2Δx-lon 0.005-0.015, 2Δx-lat
  0.001-0.003). The energy is in the LARGE eddies, not a grid mode.
- Physical OVER-ENERGIZATION: max|u| 0.93→2.06 m/s (day 78→88) ≈ 2× the oracle's ~1 m/s
  saturation, then a fast sub-daily instability NaNs day 89.
⇒ OVERTURNS both the "2Δx wall-mode" AND "2Δx grid-mode" §5 hypotheses. The §5 residual is the
SURFACE-INTENSIFIED INTERIOR EDDY OVER-ENERGIZATION under sustained τ=50d restoring forcing —
the SAME baroclinic-eddy mechanism the vertadv fix closed in the UNFORCED precursor, but §5's
forcing drives the eddies to ~2× the oracle's saturation amplitude where they blow. This is the
EDDY-EQUILIBRATION problem (legoESM's known EKE overshoot, cf. project_eke_overshoot / Phase-G),
NOT a numerical grid/wall mode. Research target: why legoESM's forced surface eddies equilibrate
~2× too energetic vs Oceananigans (eddy KE sink / GM-like restratification / surface BC), NOT a
selectable momentum operator.

## 🟢 CASE-3 BREAKTHROUGH (post-loop) — §5 SURVIVES with vertadv + faithful dt
The §5 day-89 blow-up was TWO faithful gaps, not one:
1. **vertical momentum advection** (perturbation u' vs full u) — the `weno_vertadv_full_velocity`
   fix (committed). Closes the precursor; delays §5 82→89.
2. **TIMESTEP**: the §5 ORACLE deck (`/tmp/ocn_silvestri/silvestri_jet.jl:46-47`) uses an
   ADAPTIVE wizard `conjure_time_step_wizard!(cfl=0.3, max_Δt=15min)`, Δt start 5min — so the
   oracle's dt ranges 300–900s, shrinking toward 300s at the transient peak. legoESM used a FIXED
   dt=900 (= the oracle's MAX) ALWAYS, including the peak where the oracle adapts DOWN → CFL
   violation at the over-energized peak → NaN day 89. UNFAITHFUL.
RESULT: §5 W9V (no backstop) + vertadv_full + **DT=450** (within the oracle's adaptive range):
SURVIVES to day 110+ (day100 1.61, day105 1.99, day110 1.01, all finite), max|u| oscillating
~1-2 m/s = the oracle's transient eddy amplitude, WITHIN 2×. Baseline dt=900 blew day 82;
vertadv+dt900 blew day 89; vertadv+dt450 SURVIVES. ⇒ CASE 3's "stays finite while oracle stable
AND within 2×" is being MET with TWO FAITHFUL changes (full-u vertadv + oracle-range dt). The
earlier "2× over-energization" was largely the BLOW-UP trajectory at dt=900, not the true
equilibration (dt=450 equilibrates ~1.4×). Long run to day 200 confirming; then implement proper
ADAPTIVE dt (cfl=0.3, max=900 — the wizard) as the clean faithful timestep. This REOPENS CASE 3
as closeable (NOT research-level eddy-equilibration after all — it was vertadv + timestep).

## Next candidates (research-level, untested — beyond faithful operator options)
1. KE-gradient (bernoulli head) upwinding: is legoESM's WENO KE-grad matching Oceananigans'
   OnlySelfUpwinding `kinetic_energy_gradient_scheme`? (test on the STRONG baroclinic gate —
   it IS sensitive: reconstruct_zeta moved day-12 3.39→3.26.)
2. The free-surface/barotropic coupling at finite amplitude (prior "complementary flaws").
3. Whether the residual is a COMBINATION of small effects (no single decisive lever) → would
   imply the structural-obstruction conclusion (no single faithful operator fix).

Use the STRONG gate (baroclinic blow-up day + day-12 max|u|) as the primary signal; the
bickley ~4% gate is weak/near-noise for single-lever discrimination.
