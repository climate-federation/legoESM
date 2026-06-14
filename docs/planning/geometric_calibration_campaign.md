# GEOMETRIC calibration campaign — protocol draft (for discussion)

**Status:** proposal, 2026-06-11 (Dhruv + Claude session). Companion to the
staged GEOMETRIC plan (Torres et al. 2025 reference notes) and the 2026-06-11
ocean differentiability audit. Key protocol decisions recorded in §5.

## 1. Goal

Calibrate the GEOMETRIC mesoscale closure (`EKEConfig.closure="geometric"`,
`GeometricConfig`, Torres et al. 2025 JAMES) so the coarse model's
**equilibrated mean state and variability** match a reference — directly
attacking the post-Phase-G climate residual (EKE overshoot ~2.4×,
too-barotropic vertical structure, abyssal drift), which was pinned to
eddy–mean equilibration, not dycore numerics.

Calibration targets (all paper-provenanced fields of `GeometricConfig`,
with the paper's own sampled ranges as priors/bounds):

| param | default | paper range | role |
|---|---|---|---|
| `alpha` | 0.04 | Appendix E | eddy efficiency, κ_gm = α·∫E dz / ∫M²/N dz |
| `c_eps_geometric` | 0.022 | 0.001–0.1 | EKE dissipation |
| `kappa_u` | 1500 m²/s | 500–5000 | barotropic production B_T |
| `kappa_e` | 500 m²/s | (low sensitivity) | EKE lateral diffusion |
| `rossby_factor` | 0.4 | — | dissipation length R_d prefactor |

Plus, at a later stage, a **spatial** α(x, y) (low-rank basis — see §5).

## 2. Method: division of labor (EKI primary, adjoint complementary)

Informed by Perezhogin, Adcroft & Zanna (arXiv:2604.06398) — ETKI
calibration of an NN eddy closure in MOM6 DG/NW2 — and by our own adjoint
campaigns (demo-B parameter recovery; GM/Redi `adjoint_stabilization`;
2026-06 differentiability audit + fix PRs #415/#416/#418).

**ETKI (Ensemble Transform Kalman Inversion) is the primary optimizer for
the scalar parameters.** Rationale:
- The objective is an **equilibrated statistic**: with the exact-IC protocol
  (§3) the informative averaging windows are O(800 days) — far beyond any
  validated adjoint horizon (30 days), and ETKI is demonstrably robust to
  the chaotic-averaging noise at that range.
- GEOMETRIC's prognostic-EKE feedback partially self-compensates κ
  perturbations (measured for the analogous `c_k`: near-zero short-window
  sensitivity), so short-window adjoint losses see little of the signal.
- For ≤6 scalars, ensemble cost is trivial at 2° (each 10-yr ACC run is
  hours on one V100; a full ne=100 × 5-iteration calibration ≈ a few
  GPU-days, and the *protocol* can be iterated dozens of times like
  Perezhogin did in DG before scaling up).

**The adjoint stack is used where it is uniquely strong:**
1. **Design/identifiability** (Stage 0): exact parameter sensitivities of
   candidate observables at the prior point — ranks observables, exposes
   sloppy parameter directions, chooses the ETKI observation vector. One
   stabilized adjoint run replaces a sensitivity ensemble.
2. **The window-vs-equilibrium consistency check** (the kill question for
   any short-window gradient use, §4).
3. **High-dimensional stages** (§5): spatial α(x, y) and joint
   IC+parameter problems where ensembles thin out.

Requirements carried over from the audit (non-negotiable for any adjoint
work): `PrecisionPolicy.fp64()`, `adjoint_stabilization="stop_gradient_slopes"`,
windows ≤ 30 days unless re-validated, FD spot-checks at campaign
milestones.

## 3. Protocol elements adopted from Perezhogin et al.

1. **Exact-IC, no-equilibration evaluation.** Initialize every ensemble
   member from a (filtered/bridged) snapshot of the *truth's* spun-up
   attractor; evaluate the loss as a time-average over the window WITHOUT
   waiting for re-equilibration (they showed spin-up can go to zero when
   the IC is exact; only the averaging length matters). Our attractor-drop
   bridge (halo strip, AB2-history seeding, consistent-ψ) is exactly this
   machinery, already validated against Veros.
2. **Observation vector = spatial maps of time-mean + temporal std,
   equal-weighted, normalized; NO domain-integrated scalars.** For us:
   time-mean T, S, η (or interface-equivalent) + their temporal std + the
   (depth-integrated) EKE field. Transport (ACC) is a *held-out* metric,
   not an observable — both Perezhogin (integrated metrics disrupt the
   loss balance) and our own κ/BSF degeneracy finding agree.
3. **ETKI specifics:** `EnsembleKalmanProcesses.jl`'s ETKI or a ~20-line
   in-repo port; ne = 100 (200 if members blow up), R = I on normalized
   observables, scheduler step δt, initial ensemble = priors perturbed
   ~25% (sigmoid/log-transformed to the paper ranges; start interior, not
   at bounds — the AIMIP saturation lesson).
4. **Early stopping against structural error.** Expect the
   loss-vs-physical-metrics divergence after ~3–5 iterations (their Fig.
   2k); monitor held-out metrics (transport, stratification profiles,
   abyssal T drift) every iteration and stop at the plateau. With a
   different-closure truth (§4 Stage 1) this is what keeps fitted
   parameters meaningful rather than error-absorbing.
5. **Cheap-config-first.** All protocol iteration (loss composition,
   window lengths, priors, ensemble size) happens on the 2° ACC channel.
   Window/spin-up splits are hyperparameters to optimize there, per their
   explicit recommendation.

And one element adopted from Pierre's AIMIP stack (see
`packages/ml/legoesm/training/`): **a per-parameter gradient/effect
liveness audit before any compute** — the dead-DOF lesson
(`trainable_constraints_for_scheme`, `test_physics_params_audit.py`
pattern). Every `GeometricConfig` knob must demonstrably reach the
tendencies (finite, nonzero effect in a designed-to-activate column/state)
before entering θ.

## 4. Stages

### Stage 0 — twin calibration on the ACC channel (protocol shakedown)

Truth = the model itself at known "true" GEOMETRIC parameters (spun-up 2°
ACC recipe, GEOMETRIC on). Perturb parameters (2× / paper-range priors),
recover with ETKI under the §3 protocol.

Gates / deliverables:
- ETKI recovers the scalars to within ensemble spread; loss decreases
  monotonically to the noise floor set by the averaging window.
- **Adjoint identifiability table**: dJ/dθ for each candidate observable
  (mean maps, std maps, EKE field, transport) at 5/30-day windows,
  fp64 + stabilization — fixes the observation vector for Stage 1.
- **Window-vs-equilibrium sign test** (the kill question for gradient
  methods here): compare 30-day window gradients against 10-yr forward
  perturbation pairs per parameter. Outcome documented either way — it
  decides whether adjoints may shortcut Stage-1 iterations or remain
  design-only.
- Hyperparameter choices frozen: averaging window (start 800 days,
  sweep down), spin-up split (start 0), ne, δt.

Estimated cost: O(10) GPU-days total including protocol iteration.

### Stage 1 — oracle calibration (the climate-residual experiment)

Truth = the **Veros oracle ACC attractor** (the Phase-G reference), ICs =
bridged oracle snapshots. Calibrate GEOMETRIC scalars to the §3
observation vector; held-out metrics = ACC transport, stratification,
abyssal drift, EKE level (the 2.4× overshoot is the success criterion).

Caveats stated up front:
- Veros runs Eden–Greatbatch, not GEOMETRIC → fitted parameters are
  *effective* values absorbing closure-structure differences. Early
  stopping + held-out metrics bound the damage; the result is still the
  relevant one ("can calibrated GEOMETRIC close the equilibration
  residual?").
- If Stage 1 succeeds, the scientifically clean follow-on is an
  **eddy-resolving legoESM reference** (1/4° ACC à la Meunier) as truth —
  bigger compute, to be scoped only after Stage-1 results.

### Stage 2 — transfer + spatial stage

- Frozen Stage-1 parameters into global_4deg (transfer-matrix machinery);
  verify residual movement; then production configs — **gated on running
  the step-gradient-matrix probe (PR #418) on the exact production stack
  first** (the audit's stated coverage gap).
- **Spatial α(x, y)** via the adjoint: low-rank basis (reuse the AIMIP
  `aimip_spatial` Legendre×Fourier pattern + its initialization/lr-scale
  lessons), trained with the windowed-adjoint machinery; ETKI-calibrated
  scalars as the baseline/prior. This is where the differentiable stack
  is the only practical tool.

## 5. Decisions (2026-06-12; revisit with Pierre if he objects)

Originally posed as open questions; decided to keep the campaign moving.

1. **`TrainableOceanParams` lives in the shared training framework**
   (`packages/ml/legoesm/training/`), extending
   `trainable_constraints_for_scheme` and the `test_physics_params_audit`
   liveness-test pattern to ocean schemes. One dead-DOF mechanism for the
   whole repo, not a parallel ocean-local system (consistent with the
   recipe-architecture direction: extend the existing modules).
2. **ETKI: minimal in-repo JAX/numpy implementation.** The ETKI update +
   ensemble-shrink rule is ~20 lines; no Julia dependency in the
   environment. API mirrors `EnsembleKalmanProcesses.jl` (θ-ensemble in,
   g-ensemble in, updated ensemble out; δt scheduler explicit) so a swap
   to the package remains trivial if its diagnostics are wanted later.
   Home: `packages/ml/legoesm/training/etki.py` + unit test (linear
   forward model: ETKI must reproduce the analytic posterior-mean
   trajectory).
3. **Loss recipe: start from the Perezhogin loss, not the AIMIP one.**
   Equal-weighted normalized mean+std maps (Eq. 9-style, R = I) is the
   proven design for THIS problem class; the AIMIP recipes (bias
   penalties, CRPS, multi-lead) are forecast-error tooling and are
   consulted only if Stage-0 exposes a need (their documented
   over-correction failure modes are reason for restraint).
4. **Eddy-resolving ACC reference: deferred until Stage-1 results.**
   Stage 1 against the Veros oracle answers the residual question first;
   the eddy-resolving truth is scoped only if Stage 1 succeeds and the
   effective-parameter caveat needs lifting for publication.
5. **Ensemble execution: local-first via the experiment-manifest system.**
   Stage-0/1 members are tiny (exact-IC, ~800-day windows at 2°: ~3-4 min
   GPU each ⇒ a full ne=100 × 5-iteration calibration ≈ 30 GPU-hours),
   so the 2× V100S box with `CUDA_VISIBLE_DEVICES` pinning (one member
   per GPU, sequential batches) suffices; manifest entries (#376/#381)
   for provenance. A thin SLURM array under `scripts/cluster/` becomes
   the scale-out path only at Stage 2 / eddy-resolving scoping.

## 6. References / provenance

- Torres et al. 2025 (JAMES, 10.1029/2025MS005394) — closure + parameter
  provenance (`GeometricConfig` docstrings carry page/equation refs).
- Perezhogin, Adcroft & Zanna, arXiv:2604.06398 — ETKI protocol, exact-IC
  trick, loss design, early-stopping behavior.
- `.physics-validator/{diff_veros_demos,adjoint_oracle_match,gm_adjoint_stab,
  ocean_diff_audit}/RESULTS.md` (session worktree scratch) — adjoint
  validity envelope, κ-identifiability, stabilization, audit findings.
- Fix series: PR #415 (limiter eps-underflow NaN family), #416 (rigid-lid
  jit cache), #418 (CI step-gradient matrix + fp64 policy docs), issue
  #417 (MPAS lazy-cache analog).
