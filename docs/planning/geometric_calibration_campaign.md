# GEOMETRIC calibration campaign — protocol draft (for discussion)

**Status:** proposal, 2026-06-11 (Dhruv + Claude session). Companion to the
staged GEOMETRIC plan (Torres et al. 2025 reference notes) and the 2026-06-11
ocean differentiability audit. Open questions for Pierre at the bottom.

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

## 5. Open questions (for Pierre)

1. Should `TrainableOceanParams` (GeometricConfig knobs + future KPP/GM
   scalars) live in the AIMIP training framework
   (`packages/ml/legoesm/training/`) with `trainable_constraints_for_scheme`
   extended to ocean schemes — or stay ocean-local? (Proposal: shared
   framework; the dead-DOF filter and audit-test pattern should be the
   single mechanism.)
2. ETKI implementation: call `EnsembleKalmanProcesses.jl`, or keep a
   minimal Python ETKI in-repo (update rule is ~20 lines; avoids a Julia
   dependency; loses the package's schedulers/diagnostics)?
3. AIMIP status: which loss-recipe version is current "best" and are the
   bias-penalty / multi-lead lessons stable enough to cite as defaults?
4. Eddy-resolving ACC reference (Stage-1 follow-on): worth scoping now
   (storage + GPU budget), or after Stage-1 results?
5. Ensemble execution: ETKI members are embarrassingly parallel single-GPU
   jobs — run via the experiment-manifest system (#376/#381) or a thin
   SLURM array under `scripts/cluster/`?

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
