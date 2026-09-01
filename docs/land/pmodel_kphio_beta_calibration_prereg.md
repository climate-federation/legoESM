# Pre-registration — P-model kphio / beta_cost calibration on the EC panel

Written BEFORE the sweep runs (research branch, PR #1698). Everything below is
fixed at this commit; deviations are reported as deviations.

## Why this is a diagnostic-mode calibration, not a free-running one

The 14-site ladder measured the P-model bundle raising GPP bias by **+1.4 to
+2.7 umol m-2 s-1 under prescribed observed soil moisture** and by **+2.7 to
+4.2 free-running** — the same shift, roughly 1.7x larger when the soil is
prognostic. The soil-stress multiplier is applied identically on both paths
(`two_leaf_canopy.py`: `Vc3_leaf_stressed = Vc3_leaf * fStress_vcmax`, and the
p_model branch feeds that same product), so the amplification is not a wiring
asymmetry: what differs is the SOURCE of the root-zone wetness (observed vs
modelled). The free-running arm is less water-stressed than the observed-soil
arm — visible with the PRESCRIBED capacities too (prognostic GPP bias +0.03 vs
diagnostic -1.12) — and the P model's larger capacities multiply that
pre-existing error.

**Therefore fitting kphio against free-running GPP would absorb a soil-stress
error into a photosynthetic parameter** (GLM design review, independently
reached; codex concurred on the protocol requirements). The calibration is run
in DIAGNOSTIC mode, where the soil is prescribed from observations and the
canopy parameters are the only free quantities. The free-running amplification
is recorded as a SEPARATE open finding, not something this calibration fixes.

## Fixed protocol

| item | value |
|---|---|
| arm | diagnostic, two-leaf, `capacity_scheme=p_model`, `g1_source=p_model`, `stomatal_model=medlyn` (the ladder's `cap_g1`) |
| drivers | `jf3423/DifferBESS/data/sitelevel/nc/<SITE>_driver_v2.nc`, full record |
| soil | `--soil auto` (per-site texture), `transpiration_stress=beta_theta` HELD FIXED |
| scored | `score_valid` mask as written by `run_ec_site` (observed forcing only) |
| free parameters | `land.p_model.kphio`, `land.p_model.beta_cost` (both tier 2) |
| held fixed | every other parameter, including the C4 pair `kphio_c4` / `beta_cost_c4` |

## Grid (screen, not a final calibration)

* `kphio`: 0.040, 0.055, 0.070, **0.081785**, 0.100 (spec bounds 0.02–0.15)
* `beta_cost`: 60, 92, **146**, 224, 344 — log-spaced, ratio ~1.53 (bounds 30–500)
* The default pair (0.081785, 146) is included as the **reproducibility
  control**: it must reproduce the published `cap_g1` numbers.

## Objective (fixed now, no post-hoc reweighting)

Per training site, per flux, normalised RMSE `nRMSE = RMSE / sd(obs)`; score =
mean over the 7 training sites of `0.5*nRMSE(GPP) + 0.5*nRMSE(LE)`.

* LE is IN the objective, not a guard: the least-cost slope is what LE
  constrains, so a GPP-only objective leaves `beta_cost` weakly identified.
* Mean (not median) of per-site normalised scores, and every per-site number is
  reported — a median over 7 sites is the 4th-ranked site and hides per-PFT
  trade-offs.
* Bias is reported separately per site; it is the systematic term inside RMSE.

## Split (site identities fixed here; hold-out never used to refine anything)

* **Training (7)**: US-MMS (DBF), FI-Hyy (ENF), DE-Gri (GRA), US-Ne1 (CRO),
  US-Ton (SAV), US-Whs (SHR), FR-Pue (EBF)
* **Hold-out (7)**: DE-Hai, US-Oho (DBF), US-NR1, DE-Obe (ENF), US-Var (GRA),
  DE-Geb (CRO), US-SRM (SAV)

Both halves span forest / grass / crop / savanna / shrub and both climates
(the panel is 10 boreal-class + 4 temperate; training takes 4/3, hold-out 6/1 —
recorded here as a known imbalance rather than silently balanced away).

## Stopping / interpretation rules, fixed in advance

1. **Edge winner = UNRESOLVED, not calibrated.** If the best point sits on a
   grid boundary in either parameter, the result is reported as "optimum
   outside the screened range" and no tuned value is recommended.
2. **A tuned pair is recommended only if it beats the default on the
   HELD-OUT half too.** Training-only improvement is reported as overfitting.
3. **C4 dilution is stated, not corrected**: four sites carry C4 fractions
   (US-Var, US-Whs, US-SRM, US-Ton) and their C4 pathway keeps published
   parameters, so this is a C3 calibration scored on a partly-C4 panel.
4. Per-site regressions are reported even when the mean improves.


---

# Amendment — Stage 2: dry-site P-hydro ladder (pre-registered before it runs)

Both design reviewers refused the first stage-2 sketch. Recorded here with the
changes it forced.

**Refused**: sweeping `gamma_cost` x `gplant_mol` and calling it
"site-appropriate hydraulics". Those two are partly compensatory (they both
modulate how aggressively the plant spends water), while `root_biomass_gm2`
sets the soil-to-root resistance and `minlwp_mpa` sets the dry cutoff — so the
slice was an assumption, and we hold no per-site hydraulic trait data that
would justify the "site-appropriate" label.

**Replaced by**: a one-at-a-time SENSITIVITY SCREEN over all four P-hydro
parameters (`scripts/cluster/ec_site/run_phydro_sensitivity.sbatch`) at two dry
sites, and the 2-D ladder slice is then chosen from the MEASURED response
instead of asserted. The ladder is labelled a behaviour screen of P-hydro's
hydraulic parameters, not a site-specific calibration.

**Hold-out protection**: the screen runs only on US-Whs and US-Ton, which are
stage-1 TRAINING sites. US-SRM and US-Var are stage-1 hold-outs and are
excluded from every stage-2 run until the stage-1 verdict is fixed and
published; only then may they enter the dry-site ladder.

**What "P-hydro pays" means, fixed in advance**: at the dry sites, some
parameter setting must beat the `beta_theta` anchor on the same objective as
stage 1 (`0.5*nRMSE(GPP) + 0.5*nRMSE(LE)`, per-site normalised so the
Mediterranean sites' larger absolute fluxes cannot dominate) at a MAJORITY of
the dry sites, not merely on the multi-site mean. Anything else is reported as
"does not pay at these sites with generic hydraulics".

**Stated caveats, not corrected here**:
* The `beta_theta` anchor is implicitly tuned on these ecosystems and has
  absorbed compensating errors; P-hydro must get the whole supply chain right
  with generic traits. A mixed result is the physically expected outcome of an
  uncalibrated-vs-tuned comparison, not by itself a defect in the scheme.
* Whole-plant conductance genuinely spans this sweep's whole range across shrub
  / oak-savanna / Mediterranean-evergreen types, so a single global value makes
  this a screen and never a verdict on P-hydro's fidelity.
* Switching to P-hydro replaces the stress ARCHITECTURE (it removes the
  empirical stress and re-routes capacity and slope through the hydraulic
  optimum), so it is a one-SWITCH ladder but not a single-mechanism test.
* The discriminating measurement neither stage runs is predicted leaf water
  potential against observed predawn / midday values — that is what would
  separate "P-hydro is mis-parameterised" from "P-hydro is wrong here".
  Recorded as the recommended next instrument.
