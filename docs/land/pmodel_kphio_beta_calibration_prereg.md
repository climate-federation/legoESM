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


## Stage-2 sensitivity screen — RESULT (measured, 2 dry training sites)

One-at-a-time response of the prognostic P-hydro arm at US-Whs + US-Ton
(`diagnostics/phydro_sens/ladder_skill.csv`). Range = spread of the low/high
perturbation around the default.

| parameter | GPP RMSE range | LE bias range | rank |
|---|---|---|---|
| `gplant_mol` (5e-4 .. 2e-2) | **1.17** | **14.3** | 1 |
| `minlwp_mpa` (-5 .. -0.5) | 0.77 | 9.0 | 2 |
| `root_biomass_gm2` (50 .. 2000) | 0.74 | 10.1 | 3 |
| `gamma_cost` (0.1 .. 10) | 0.23 | 3.7 | 4 (weakest) |

**The assumed slice was wrong.** `gamma_cost` — half of the originally proposed
grid — is the LEAST informative of the four, exactly as both reviewers
predicted. The ladder slice is therefore **`gplant_mol` x `minlwp_mpa`**, chosen
from this measurement rather than asserted.

**Substantive finding (PLAUSIBLE, 2 sites):** LOWERING whole-plant conductance
from the 4e-3 default to 1e-3 improves dry-site skill on both fluxes at once —
GPP RMSE 2.75 -> 1.89 and bias +1.40 -> +0.47, LE bias +27.8 -> +19.2 W/m2.
Direction is physically sensible (the default lets a semiarid plant move too
much water), and it is the largest single-parameter improvement seen anywhere in
this campaign. It sits near the low edge of the screened range, so the ladder
extends `gplant_mol` down to its spec bound (5e-4) rather than stopping at 1e-3.

**Sequencing (hold-out protection, unchanged):** the dry-site ladder runs only
after the stage-1 verdict is fixed and published, because two of the five dry
sites (US-SRM, US-Var) are stage-1 hold-outs.


## Stage-1 screen result + compensation diagnostic (measured)

* 5x5 screen: best point (kphio=0.040, beta=60) — the LOW-LOW corner, so
  pre-registered rule 1 fires: **UNRESOLVED**, no tuned value recommended
  (training improvement +6.5%; US-Whs and FR-Pue regress at that point).
* **Compensation diagnostic (GLM-proposed, run on the same outputs): a
  GPP-only re-ranking AND an LE-only re-ranking each independently pick the
  SAME corner (0.040, 60).** Under the interpretation fixed before running it,
  divergence would have meant kphio was being spent to buy LE at GPP's
  expense; CONVERGENCE means both fluxes genuinely prefer lower capacities —
  the corner-seeking is NOT flux-trading compensation, and the calibration
  remains legitimate. The extension grid (kphio down to 0.020, beta down to
  30, previous corner as overlap control) is the running next probe.
* Open mechanism question (PLAUSIBLE, unmeasured): the preferred kphio is
  heading toward roughly HALF the rpmodel default — a factor suggestive of a
  light-accounting convention mismatch between rpmodel's incident-light kphio
  (absorptance folded in) and the host canopy's own absorption profile.
  To be checked against the extension result, not assumed.


## Extension result (measured; full 7x7 discovered grid)

* Joint objective: optimum **(kphio=0.040, beta=60), INTERIOR** on the
  extended axes; overlap control reproduced the screen objective exactly
  (0.9135). Improvement vs default: joint +6.5%, GPP-only +13.9%.
* GPP-only re-ranking: same interior point — genuine optimum, not
  flux-trading.
* LE-only re-ranking: floor-seeks (kphio=0.020 boundary). Explained by a
  PRE-EXISTING, arm-independent latent-heat bias: the production baseline
  already carries LE bias +41 W/m2 at FR-Pue and +32 at US-MMS (ladder CSV,
  base_bb arm) — no photosynthetic parameter can remove it, so LE alone keeps
  pushing kphio down. The recommendation rests on the joint/GPP interior
  agreement; the LE residual is an open finding on the HARNESS/soil-evap side,
  logged separately.
* Pending: pre-registered rule 2 — (0.040, 60) vs default on the 7 held-out
  sites (array 9597644). No recommendation until that returns.


## STAGE-1 FINAL VERDICT (hold-out evaluated; closed)

**No tuned value is recommended.** The interior training optimum
(kphio=0.040, beta=60) does NOT beat the default on the held-out half
(0.8389 vs 0.8370) — pre-registered rule 2 fires: the +6.5% training
improvement (+13.9% GPP-only) was overfitting to the training sites'
composition. Defaults (0.081785, 146) stand.

Reading (PLAUSIBLE): the diagnostic-mode GPP overshoot is real but
heterogeneous across sites, and a single GLOBAL scalar kphio cannot reduce it
without degrading other sites in equal measure. The two live follow-up levers,
in order: (1) the factor-of-two light-convention question (mechanism check on
the leaf-level Iabs/kphio absorptance convention vs the host canopy's own
absorption — a code-reading + oracle exercise, no GPU); (2) per-PFT or
per-climate kphio (a shape='n_pft' spec change - a real scope decision, not
taken unilaterally).

## Stage-2 dry-site ladder — grid FIXED from the measured screen

Slice: `gplant_mol` x `minlwp_mpa` (ranked 1 and 2 by the sensitivity screen;
gamma_cost ranked last and is dropped). Directions from the screen: LOWER
gplant improved both fluxes (best screened point 1e-3, near the screened edge);
SHALLOWER minlwp (-1.0) improved GPP. Grid:

* `gplant_mol`: 5e-4 (spec floor), 1e-3, 2e-3, 4e-3 (default)
* `minlwp_mpa`: -1.0, -1.5, -2.0 (default)
* anchor arm: `transpiration_stress=beta_theta`, same P-model bundle
* sites: US-SRM, US-Whs, US-Ton, US-Var, FR-Pue (hold-outs released — the
  stage-1 verdict above is fixed and published before any stage-2 run touches
  them)
* objective and "pays" rule as pre-registered in the stage-2 amendment
  (majority of dry sites, per-site normalised).
