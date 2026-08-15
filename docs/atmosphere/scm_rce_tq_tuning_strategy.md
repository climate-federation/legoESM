# SCM-RCE convection tuning against CRM **temperature and humidity** profiles

Status: REVIEWED (codex on the strategy, codex on the code, GLM on both) and
IMPLEMENTED. The sections below are the plan as first written; **"What the
review changed" at the end is what is actually being run**, and where the two
disagree the later section wins.

## The ask

For each convection scheme available in legoESM, tune **all** of its exposed
parameters inside the single-column model in the RCEMIP1 configuration so the
equilibrium **temperature and water-vapour profiles** match the CRM reference.
Report the T and q RMSE ranking of every scheme **before and after** tuning, and
plot the profiles.

## What already exists (do not rebuild)

* `scripts/run/run_scm_rce_campaign.py` — SCM-RCE evaluation, reference
  extraction, derivative-free tuner (`tune_category_winner`), profile/ranking
  CSV + PNG writers.
* `scripts/run/run_scm_rce_convection_intercomparison.py` — the 10-scheme
  driver, per-scheme checkpoints with a protocol signature, merge gate.
* `packages/ml/legoesm/training/scm_rce_metrics.py` — the shared normalized
  profile-RMSE arithmetic used by both the derivative-free and the AD path.
* `scripts/cluster/scm_rce_paper/convtune_arms.sbatch` — the pinned 10-arm
  SLURM array with its preflight gates.
* Reference: `results/rcemip_ref_sam300`, the **external** RCEMIP archive
  SAM_CRM RCE_small300 run (Wing et al. 2018). Never our own CRM output.

## The one thing that is wrong for this ask

The tuner currently minimises `combined`:

```
combined = sqrt( (T_rmse^2 + qv_rmse^2 + cloud_rmse^2 + w*precip_rmse^2) / (3+w) ),  w = 1
```

each term being a mass-weighted RMSE normalised by the **reference's own**
mass-weighted standard deviation.

MEASURED on the 2026-08-13 anchor-off arm (10 schemes): `cloud_rmse` spans
1.38–18.63 while `T_rmse` spans 0.14–0.46, so the **condensate term carries
87–100 % of the sum of squares**. T contributes 0.01–2.3 % and q_v 0.008–3.4 %
for nine of ten schemes. The shipped ranking is a condensate ranking. Tuning
under it does not tune temperature or humidity, which is exactly what is asked
for here.

## Change 1 — a `thermo` tuning objective

Add a third member to `TUNE_OBJECTIVES`:

```
thermo = sqrt( (T_term^2 + qv_term^2) / 2 )
```

No condensate, no precipitation. Two candidate normalisations, and this is the
first question for the reviewers:

* **(a) reference-std** — reuse the existing `normalized_profile_rmse`, i.e.
  divide by the CRM's own mass-weighted std of each variable. Commensurable by
  construction, consistent with every other score in the repo, but the T-vs-q
  weight is then whatever the reference's spread happens to be.
* **(b) fixed physical tolerances** — 1 K and 1 g/kg, as
  `score_subcloud_jax` already does, with the stated reason that a declared
  tolerance says what "wrong by one unit" means instead of discovering it.

## Change 2 — score humidity where humidity lives

A mass-weighted **absolute** q_v RMSE is a boundary-layer metric. q_v spans
roughly three decades between the surface (~18 g/kg) and the upper troposphere
(~0.01 g/kg), and mass weighting adds another factor favouring low levels, so
the free-troposphere humidity — the field that actually distinguishes convection
schemes, and the one RCEMIP intercomparisons report — contributes a negligible
fraction of the metric. Candidate fixes, in order of how much machinery they
add:

1. score `q_v` as is (status quo; free troposphere invisible);
2. score `log(q_v + floor)`;
3. score relative humidity, which is bounded, O(1) at every level, and is the
   RCEMIP-standard humidity diagnostic;
4. report all of them and tune on one.

Second question for the reviewers: which of these is the defensible target, and
does it change what "close to the CRM" means in a way a reader would object to?

## Change 3 — tune *all* the parameters

`tune_category_winner` currently calls `build_trainable_params(tier="extended")`
= `tunable_tier` 1–2. "All parameters" means `tier="aggressive"` = 1–3.

Tier 0 stays fixed **deliberately**: by this repo's own parameter-hygiene
contract tier 0 is numerics floors, regularisers, smoothing widths, measurement
conventions, and anything iteration-coupled. Those change the numerics, not the
physics, and several are coupled to a substep count. Tuning them would make the
result a statement about the discretisation.

This needs `--tune-tier` threaded through the driver, into the checkpoint
signature (otherwise an `extended` checkpoint is silently reused for an
`aggressive` request), and into the sbatch's parameter-count preflight so the
budget is computed at the tier actually used.

## Protocol (pinned, one variable per arm)

Identical to the 2026-08-13 anchor-off arm except for the objective and the
tier, so the before/after and the cross-scheme comparison are controlled:

| setting | value |
|---|---|
| reference | `results/rcemip_ref_sam300` (external SAM_CRM RCE_small300) |
| length / dt | 100 days / 600 s |
| analysis window | last 5 days |
| microphysics | morrison, hard saturation adjustment |
| radiation | RRTMGP |
| subsidence solve | `implicit_flux` |
| convection substeps | 10 |
| microphysics substeps | 30 |
| BL SST anchor | OFF (`--bl-anchor-top-m -1`) |
| tuning budget | 12 evals per tunable parameter, clamped [48, 240] |
| seed | fixed, so a shorter budget is nested inside a longer one |

## What the search actually is

`_candidate_values` spends the first `1 + 2*n_params` evaluations on a
deterministic one-at-a-time sweep at the 25 % and 75 % points of each
parameter's range, then draws jointly at random (log-uniform for any range
spanning >= 2 decades). Acceptance is best-so-far. That is a *random* search,
not an optimiser: with ~20 parameters and 240 evaluations the high-dimensional
schemes will remain under-converged, and any resulting ranking is partly a
statement about search budget. Third question for the reviewers: is a cheap
local refinement (coordinate descent around the incumbent for the last ~25 % of
the budget) worth adding, or does it just buy over-fitting to a 5-day window?

## Known structural limits, to be stated in the results, not hidden

* **kuo cannot convect in a single column** — it needs a large-scale moisture
  convergence operator, so `convection/integration.py` deliberately disables it
  rather than substituting a proxy. Its row is the no-convection baseline
  (CONFIRMED: bit-identical score to `convection=none`).
* **dca exposes 0 tunable parameters** at `extended` tier; whether `aggressive`
  changes that is to be measured, not assumed.
* The column has a **fixed 300 K SST** and **no large-scale forcing**, so no
  scheme can be rewarded or punished through an SST feedback.
* A 5-day analysis window at day 95-100 is an *approximate* equilibrium; drift
  diagnostics are reported per scheme and a scheme whose tuned optimum is a
  drifting column (emanuel did this in the previous arm: P 0.87 vs E 2.14
  mm/day) must be labelled, because its profile score is not a statement about
  a settled state.

## Deliverables

1. `profiles_all_convection.png` — already produced by the driver: CRM black,
   a-priori red dashed, tuned red solid, T / q_v / condensate panels per scheme.
2. **NEW** a paired before/after ranking figure with **two panels**: T RMSE in K
   and q_v RMSE in g/kg (physical units, not the normalised score), each scheme
   a before→after segment, sorted by the tuned value.
3. The merged CSV, which already carries `apriori_T_rmse_K`, `tuned_T_rmse_K`,
   `apriori_qv_rmse_g_kg`, `tuned_qv_rmse_g_kg`.

## Falsification, declared before running

* If the `thermo` objective is doing anything, tuned `T_rmse` and `qv_rmse` must
  fall for schemes that have live parameters. If a scheme's T and q RMSE do
  **not** fall while its combined score does, the objective was not wired in.
* `kuo` and any 0-parameter scheme must show tuned == a-priori EXACTLY. A moved
  value there means the tuner is perturbing something it should not.
* A scheme whose tuned T/q RMSE improves while its condensate RMSE explodes is
  a real trade-off to report, not a bug — but it must be visible in the figure.

---

## What the review changed

Three reviews ran before any tuning compute was spent: codex on the strategy,
GLM on the strategy, codex on the implementation. Each of the following is a
change they forced, with the measurement that settled it.

### 1. The humidity variable is `log q`, not RH and not absolute `q_v`

The probe (`scripts/validate/scm_rce_tq_objective_probe.py`, job 9417296)
imposed a uniform 10 % relative humidity error at every level of the SAM_CRM
reference and reported what fraction of each candidate metric's sum of squares
came from aloft:

| metric | share above 2 km | above 5 km | above 10 km |
|---|---|---|---|
| absolute `q_v` | 26.4 % | **1.72 %** | 0.0006 % |
| relative humidity | 66.5 % | 27.9 % | 4.3 % |
| `log q` | 80.6 % | 56.6 % | 28.6 % |
| (column mass share) | 80.6 % | 56.6 % | 28.6 % |

An absolute `q_v` RMSE is a boundary-layer metric — 1.7 % of its leverage lies
above 5 km against a 56.6 % mass share. That is the original defect.

RH fixes most of it but carries a cost GLM quantified: `RH = q_v/q_sat(T)` and
`dln q_sat/dT ≈ 0.06 /K` at 300 K rising to ~0.12 /K near the cold point, so a
5 % RH tolerance ALSO demands ~0.8 K near the surface and ~0.4 K aloft. It
re-weights temperature by 2–4×, unevenly, and makes a compensating
warm-and-moist bias the exact optimum.

`log q` has neither problem: no saturation curve, so no liquid/ice phase
convention and no coupling to temperature, and its leverage is distributed
exactly like the column mass. It is the default; `rh` stays selectable and both
terms are reported for every run.

### 2. `physical`, not `aggressive`

`aggressive` is neither a superset nor a subset of "the scheme's physics". It
ADMITS four `category == "numerics"` parameters (a sigmoid layer-edge width, a
mass-flux normalisation scale, an MSE search offset) and EXCLUDES the CAPE
trigger of eight of the ten schemes — tier 0 solely because the trigger sigmoid
saturates and its AD gradient vanishes, which is irrelevant to a gradient-free
search. The `physical` set is aggressive minus `numerics`, minus Kain-Fritsch's
grid-length scaling (meaningless in a single column), plus the tier-0 parameters
whose spec reference marks them AD-unreachable. The resulting set is pinned
per scheme by a test.

### 3. The mask comes from the reference's cold point

A fixed 100 hPa bound sat near the RCEMIP-300 K cold point by coincidence. If
the cold point falls just below it the objective scores tropopause levels where
a radiatively-controlled temperature bias no convection parameter can fix turns
into a large humidity error. The bound is now the reference's own cold-point
pressure plus a buffer, computed from the reference alone so every scheme is
scored on one identical domain.

### 4. Two seeds, a local refinement, and a measured noise floor

The search remains a random draw plus a greedy coordinate refinement; codex
quantified what that is worth at 19–25 parameters (draws ~0.76 of each range
apart; the chance of landing in a 20 %-wide box around an optimum is ~1e-11).
It is therefore reported as what it is. Three things make that reportable
rather than hidden:

* **the refinement stage** turns the best random draw into a point no single
  parameter move improves;
* **two independent seeds**, compared by
  `scripts/validate/scm_rce_tq_seed_agreement.py`, which reports rank churn,
  pair swaps, and — the part a score table cannot show — whether the two seeds
  landed on DISTANT parameter values with similar scores, i.e.
  non-identifiability;
* **the score's own noise floor**, the spread of the objective across the last
  four non-overlapping analysis windows of the same run. An improvement smaller
  than that is not a result, and nothing previously measured it.

### 5. Labelling

The CSV and the markdown summary sorted a tuned table by the historical
combined score and described the search as "extended-tier" whatever had been
run. Both now rank by, and name, the objective and parameter set recorded in
each row's own checkpoint signature; unstamped checkpoints are no longer
accepted on the skip path, and the ranking figure refuses to mix objectives.

### Still open, stated rather than fixed

* **No energetics in the objective.** Precipitation, OLR and cloud radiative
  effect are reported but not minimised, so a scheme can match T and humidity
  with wrong energetics. `P-E` is on every row for exactly this reason.
* **No held-out SST.** A 295 K / 305 K RCEMIP case would test generalisation;
  it costs another integration per incumbent and is not run here.
* **One CRM.** Tuning to SAM optimises SAM-emulation, including SAM's own
  structural biases.
* **A spec inconsistency found in passing**: `KainFritschConfig`'s
  `cape_threshold` appears in BOTH the `excluded` block and the `params` block
  of its `__param_spec__`. The registry honours `excluded` (measured: zero
  tier-0 entries for that scheme), so nothing here is affected, but the
  duplicate should be removed.
