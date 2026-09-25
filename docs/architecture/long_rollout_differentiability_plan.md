# Long-rollout differentiability: what to take from astronomix (arXiv:2607.18176)

Status: REVIEWED AND LARGELY REFUTED. Do not build P1 or P2. The original
proposal is kept below the review so the refuted claims stay visible.

## MEASURED, 2026-09-17 — the full-physics lane, and the receipt does not reproduce

Jobs 9829660 (sweep) and 9829661 (NaN trap, `JAX_DEBUG_NANS=1`), git 3da050246,
lat-lon 32x8, dt=600 s, fp32 arrays, one initial condition, adjoint taken with
respect to the initial state. Two scheme sets, never pooled.

Max abs adjoint vs rollout length (steps); 144 steps = 24 h, the horizon the
receipt names:

| scheme set | friction | 1 | 8 | 36 | 72 | 144 |
|---|---|---|---|---|---|---|
| today (rrtmgp, mass flux, Kessler) | off | 1.6e-2 | 1.27e+4 | 6.46e+4 | 3.27e+3 | 4.48e+3 |
| today | on | 1.6e-2 | 1.27e+4 | 4.11e+4 | 5.84e+3 | 4.88e+4 |
| June-like (gray, SBM, no microphysics) | off | 1.5e-2 | 6.33e+0 | 1.27e+3 | 4.39e+4 | 3.06e+13 |
| June-like | on | 1.5e-2 | 7.69e+0 | 1.97e+3 | 2.28e+3 | 1.11e+10 |

The forward was finite in every arm at every horizon.

**THE RECEIPT DOES NOT REPRODUCE.** The trap job ran the same 24 h rollout with
JAX's NaN detector armed, in both scheme sets, and exited clean: no NaN is
created anywhere in the backward pass. The claim that set this lane's 6 h
training horizon — "1-step grad is fine; 144-step is NaN" — is not a property of
the current code. Caveat, stated plainly: this is today's code, not the June
code, and many fixes have landed since; it is 32x8 on one initial condition.
What is established is that the horizon cap is no longer justified by the thing
it cites.

**The physics, not the friction, controls adjoint growth.** Today's scheme set
has a BOUNDED adjoint: it jumps once in the first eight steps and then sits
between 1e3 and 1e5 out to 24 h, with no exponential branch at all. The
June-like set — gray radiation, SBM convection, and crucially NO microphysics —
grows at 0.28 per step and reaches 3e13. Kessler and the mass-flux scheme are
doing the damping.

**Rayleigh friction is a lever only in the stripped-down set.** It buys a factor
2750 at 24 h in the June-like arms (3.06e13 -> 1.11e10) and nothing usable in
today's arms (4.48e3 -> 4.88e4, non-monotonic, i.e. noise). On the neural lane
it bought a factor 2 against a 0.44/step growth rate. So "add the missing
damping" would not have fixed any lane that matters.

**Finite is not the same as usable.** 3e13 is a number, but it is a terrible
conditioning number for an optimiser. Today's 4.5e3 is the healthy case. Any
decision to lengthen the training horizon should be made on the conditioning,
not on the absence of a NaN.

A live defect on main blocked all of this for two rounds:
`physics_pipeline.compute_radiation_core` passes `conv_cloud_coeff` AND
`Nc_default` to `build_cloud_config`, which has never accepted either; the
caller gained them on 2026-08-10 (8e76f1fc5, a "restore" commit). Any run
reaching the radiation branch with a cloud scheme raised `TypeError`. No AMIP
run has been launched from this checkout since 1 July, which is why nobody hit
it. Both added, with a contract test that parses the caller's keyword list out
of the source and asserts the builder accepts all twenty — it caught the second
one by itself, on its first real run.

## MEASURED, 2026-09-16 — the neural lane's adjoint, damping on vs off

Job 9811480, git 3da050246, lat-lon 32x8, dt=600 s, fp32 arrays (x64 flag on,
`precision=fp32`), one initial condition (2015 day 0), loss = the lane's own
`combined_loss` against its 6 h ERA5 target. Differentiated with respect to the
INITIAL STATE, whose every leaf is live. One variable between arms: the
Rayleigh profile handed to the segment builder.

| steps | 1 | 2 | 8 | 36 | 72 | 144 |
|---|---|---|---|---|---|---|
| max abs adjoint, no damping | 1.64e-2 | 1.64e-2 | 4.70e-2 | 1.19e+1 | 9.05e+7 | 6.10e+21 |
| max abs adjoint, damped | 1.64e-2 | 1.64e-2 | 4.70e-2 | 8.50e+0 | 5.26e+6 | 2.92e+21 |

The forward stayed finite at every horizon (loss 1.8131e3 to 1.8135e3).

**CONFIRMED: the adjoint grows exponentially.** Clean log-linear growth at
0.44/step measured over both 36->72 and 72->144, i.e. an e-folding time of
about two steps (~23 min). No NaN at 144 steps; extrapolating the same rate,
fp32 overflow lands near 230 steps (~38 h).

**REFUTED: Rayleigh friction is not the lever.** The damped arm is a factor 2.1
lower after 144 steps and its fitted growth rate differs by 2.5% (0.370 vs
0.379 per step). The "undamped dycore" fix would not have moved this.

**RETRACTED from earlier in this session.** I argued from the fp32 overflow
budget (~88.7 e-folds) that an adjoint e-folding time short enough to overflow
inside 24 h was implausible for a hydrostatic GCM by about two orders of
magnitude. The measurement says the opposite: the observed rate is of exactly
the order that argument dismissed. The arithmetic was right and the physical
prior attached to it was wrong.

Note the instrument earned itself here: at 144 steps an L2 gradient norm would
have squared 6.1e21, overflowed fp32, and reported a blow-up that the data does
not contain. The max-norm was adopted on a reviewer's warning before the run.

STILL OPEN: the classical (full-physics) lane, which is the one the June
receipt is about, and the untruncated NaN trap. Both were blocked by a live
defect on main -- `physics_pipeline` passes `conv_cloud_coeff` to
`build_cloud_config`, which has not accepted that argument since 2026-08-10
(8e76f1fc5), so every radiation step with a cloud scheme raised `TypeError`.
Fixed, with a contract test that reads the caller's keyword list from source.
Jobs 9816183 / 9816184 rerunning.

## Review outcome (codex + GLM-5.2 + independent Claude, all three run before any code)

**Verdict: one item survives (P4), and it must be repaired before it is trusted.**

### The motivating receipt is confounded — this is the finding that matters

The plan was built on one observation: the lat-lon primitive-equation adjoint
returns NaN through a >6 h chain while the forward stays finite (job 8533825),
read as chaotic adjoint growth. It is not established that it is.
`training_driver.py:90-96` documents a DIFFERENT cause of exactly that
signature on exactly that stack — a missing Rayleigh-friction profile,
"#797 bug 11": "without it the forward 6 h rollout stays finite but the
720-step ADJOINT through the undamped dycore returns NaN gradients".
`run_aimip_latlon.py` never passes `fric_decay`, so `training_driver.py:98-99`
substitutes ones, i.e. no damping. Only `scale_build.py:770-781` passes it.
VERIFIED by reading both call sites. The NaN measurement was taken in the one
configuration the repo itself documents as NaN-producing for an unrelated and
already-diagnosed reason. Chaotic adjoint growth is UNPROVEN here.

### Per-item verdicts

**P1 multiple shooting — DO NOT BUILD.** Four independent kills:
- Structural: production supervises against a full ERA5 state at EVERY lead, so
  every shooting node is observed. That is precisely the condition under which
  multiple shooting degenerates to what truncated BPTT already computes; the
  paper's gain comes from its seven UNOBSERVED interior nodes and a single
  terminal observation. Our configuration is the paper's short-horizon case,
  where the paper itself reports multiple shooting is not advantageous.
- Conditioning: lifting at the 24 h lead nodes (`run_aimip_headtohead_t106.py`,
  216 steps at dt=400) leaves every segment carrying a 24 h adjoint — four
  times past the horizon the repo says kills it. Identical per-segment
  conditioning to truncated BPTT.
- Dimension, not memory: the plan named optimizer memory as the dominant risk.
  It is 0.1-0.3 GB at the resolutions actually trained — a non-issue. The real
  cost is that the trainer's control is SIX scalars
  (`trainable_params.py::DEFAULT_TRAINABLE`; CLAUDE.md's "8 params" is stale),
  and lifting two T106/L8 states adds ~1.5e7 variables. Ratio ~2.5e6 : 1. The
  penalty term then minimises by moving states, and the six parameters become
  numerically invisible.
- Framing: truncated BPTT IS multiple shooting at zero penalty with segment
  starts clamped to the free run; hard-continuity multiple shooting IS full
  backprop. The method is a dial between two things already operated here, and
  its far endpoint is the failure it was proposed to cure.

The plan also inverted the paper: astronomix starts its segments on a
consistent forward trajectory so defects begin at ZERO. ERA5 nodes start the
defects at the model's 6 h drift — worse, not better. And ERA5 lifted states
are not valid restarts: `era5_to_state.py` seeds tke/qke/ice/number
concentrations as zeros, and `run_aimip_latlon.py:315-322` deliberately zeroes
the held radiation state because otherwise observed fluxes leak into the score.
Lifting at ERA5 nodes reintroduces that leak at every interior node.

**P2 kink-immune gradient certification — DROP, it is already built and ours is
better.** `tests/grids/fv3_gate_helpers.py:423` `assert_fd_truncation_scaling`
already certifies against a finite-difference ladder and asserts the observed
truncation ORDER, with the non-vacuity control the plan proposed to write
(`test_fv3_gate_helpers.py:249`, red against a `custom_vjp` whose backward is
50% too large). One-sided checks at limiter switching surfaces exist in
`test_fv3_nh_core.py` and `test_fv3_tp_core.py`, and they are STRONGER than the
paper's: they check each side against the ANALYTIC branch derivative, whereas
the paper's `min(|g_AD-g_F|, |g_AD-g_B|)` accepts whichever side happens to
agree and therefore cannot detect AD taking the wrong branch. At the hard sites
the primary gate is not finite differences at all but the FD-free adjoint
identity <Jv,w>=<v,J^T w> (`fv3_gate_helpers.py:280-324`), which runs ON the
limiter surface. Residue: promote `_one_sided` from one test module into
`fv3_gate_helpers.py`, ~30 lines. Not a work item.

Also FALSE in the original plan: the pressure-gradient kernels were named as
uncertified. `fv3_pgrad.py:190-215` says order-2 `check_grads` IS expected to
hold and reports reverse mode CONFIRMED by three gates. `fv3_nh_core.py:56-60`
names exactly one non-smooth site, not a wide set.

**P3 analytic functional-derivative convergence gate — LAST, and weaker than
claimed.** Genuinely absent in the resolution direction. But the justification
was over-claimed: a finite-difference check tests whether AD equals the
derivative of the discrete map, and if it does and the map is order p then the
gradient is order p — so P3 is a more SENSITIVE instrument, not a test of a
different property. Neither production dycore has a nominal spatial order to
assert against (FV3 is PPM, ~2-3; the spectral core has no algebraic order),
and a periodic-plane test cannot certify cubed-sphere halo VJPs.

**P4 wire the gradient-horizon diagnostic — DO THIS, after repairing it.**
Inventory confirmed: `grad_horizon.py` has no production consumer, and the
curriculum's pushforward metadata is still unwired. Two instrument defects must
be fixed first, or the sweep will lie:
- `grad_horizon.py:61-62` filters non-finite samples out of the growth-rate
  fit, so the sweep silently discards the blow-up it exists to find.
- `grad_horizon.py:49` passes a Python int horizon, forcing a retrace and
  recompile per horizon point; reverse-mode radiation compile is documented at
  >10 h at true T106. Run at the lat-lon default resolution or with radiation
  as forcing, never at T106.

### The measurements that decide everything, in order

1. Adjoint amplification A(6 h) vs A(24 h) from the repaired sweep, on real
   checkpoints, **with `fric_decay` ON and OFF**. The on/off pair is what
   separates chaotic growth from the #797 damping defect, and nobody has run it.
   Log max|adjoint| immediately before the NaN: huge-then-overflow means a
   growth problem; NaN at small norm means a kink-VJP bug and P1 is moot either
   way.
2. If and only if growth survives that: at a horizon where full backprop is
   still finite (12 h = two 6 h segments), compute the PARAMETER gradient three
   ways — full backprop as reference, truncated BPTT, multiple shooting — and
   report cosine similarity and relative norm against the reference over >=8
   initial conditions. If truncated BPTT scores >~0.99 there is no bias for P1
   to remove and P1 dies before a single GPU-hour. Note the original plan scored
   initial-condition error, which is not the production currency; the production
   problem is parameter estimation.

### Corrections to the text below

- The function is `multi_step_rollout_loss` (`training_driver.py:317`), not
  `_multi_step_loss`. No such symbol exists.
- "Truncated BPTT live, default ON for the lat-lon stack" is FALSE.
  `run_aimip_latlon.py:606` defaults `--multi-step-hours` to empty, so the
  single-rollout branch is taken and the `stop_gradient` at
  `training_driver.py:393` never executes. VERIFIED. There is no truncation
  bias in the default lane to fix.
- "Same per-segment adjoint cost as truncation but unbiased at convergence" is
  PLAUSIBLE, not measured: the paper compares multiple shooting against SINGLE
  shooting and contains no truncated-BPTT arm.
- "Checkpointing is a solved problem here" overstates. Three square-root
  checkpoint implementations coexist (`checkpoint_schedule.py:134`,
  `compiled_segments.py:88`, `dycore_rollout.py:118`) — a duplication finding in
  its own right — and `dycore_rollout.py:95-101` REFUSES the binomial schedule,
  so the training rollout cannot use it.
- Host offload covers selected matmul residuals, not full states.

### Reuse target the plan missed

If weak-constraint machinery is ever wanted, it is nearly present:
`da/control_vector.py:300-444` already resolves an `(n_window, *field_shape)`
time-distributed control with one slice per step, and
`da/cost_function.py:261-341` already runs a windowed rollout injecting it.
That is the weak-constraint forcing formulation minus the multiplier and
penalty terms. It is tested but not exported from `da/__init__.py`.

---

# ORIGINAL PROPOSAL (superseded by the review above; refuted claims retained)


Status: PROPOSAL, not started. Nothing below has been implemented or measured
in legoESM.

## The paper, in one paragraph

Storcks, Thuerey & Buck, *Per Astronomix ad Astra: High-Order Differentiable
(Magneto)hydrodynamics with Energy-Conserving Self-Gravity*. A JAX
finite-difference/finite-volume MHD solver built to be differentiated. Most of
the paper is about the solver (5th-order constrained transport, 4th-order
self-gravity, Pallas kernels, GPU scaling) and is irrelevant to us. Four things
in it are directly relevant to long differentiable rollouts, and they are the
only things this plan considers.

## What legoESM already has (pre-implementation search, RULE 4)

Searched `packages/`, `src/`, `tests/`, `scripts/` for: `shooting`,
`augmented lagrangian`, `continuity defect`, `checkpoint`/`remat`,
`rollout`/`bptt`, `check_grads`, `4dvar`/`weak constraint`.

| capability | where | state |
|---|---|---|
| binomial/Treeverse + sqrt-N checkpointing, host offload | `training/checkpoint_schedule.py` | built; consumed by 4D-Var cost function |
| adjoint-norm-vs-horizon sweep, growth-rate estimate, ceiling check | `training/grad_horizon.py` | built; **no production consumer — tests only** |
| rollout curriculum ladder + pushforward metadata | `training/curriculum.py` | ladder wired; pushforward metadata **not wired** (documented open item in `run_aimip.py`) |
| truncated BPTT across multi-step segments | `training/training_driver.py::_multi_step_loss` | live, default ON for the lat-lon stack |
| strong-constraint incremental 4D-Var | `da/incremental.py` + `da/{control_vector,cost_function,minimizer}.py` | built |
| multiple shooting / lifted-state optimization | — | **absent, zero hits** |
| gradient certification at non-smooth kernels | `fv3_nh_core.py`, `fv3_pgrad.py`, `fv3_tp_core.py` | **explicitly NOT certified**: comments say order-2 `check_grads` "is not expected to hold" at the C^0 sites |

So checkpointing is a solved problem here and the paper adds nothing to it. The
gap is elsewhere.

## The fact that makes this paper relevant to us

`training_driver.py` already documents our own type-1 failure:

> The lat-lon primitive-equation ADJOINT explodes to NaN through a >~6 h
> differentiable chain even when the forward is finite (job 8533825) — so full
> backprop-through-time over a 24 h+ multi-step rollout is unusable on this
> stack.

Appendix G of the paper is the theory for exactly that: the adjoint obeys a
homogeneous linear equation `∂_t p = −A(t)ᵀ p`, so `‖p(0)‖ ~ exp(λ_max T)‖p(T)‖`
even when the nonlinear forward stays bounded — the saturation that bounds the
state does not bound the adjoint. Our mitigation is truncated BPTT
(`stop_gradient` on the carry between segments). That is a **biased** gradient:
it deletes the cross-segment sensitivity instead of stabilising it.

The paper's mitigation is multiple shooting, which has the same per-segment
adjoint cost as truncation but is unbiased at convergence.

## P1 — Multiple shooting (lifted segment-start states) as an unbiased alternative to truncated BPTT

**What the paper did.** 2D Kelvin–Helmholtz, 256², recovering a 10-parameter
initial velocity perturbation from a terminal velocity observation. Split
`[0,T]` into `M=8` segments; the segment-start states `s_1..s_{M-1}` become
optimization variables alongside the control `p`; continuity is imposed as
`r_j = S_h(s_j) − s_{j+1} = 0` through an augmented Lagrangian

`L = ½‖H(S_h(s_{M-1})) − y‖² + Σ_j [⟨λ_j, r_j⟩ + (ρ/2)‖r_j‖²]`

optimized by alternating inner Adam sweeps at fixed `(λ, ρ)` with outer updates
`λ_j ← λ_j + ρ r_j` and `ρ ← 1.6ρ` whenever the total defect stops decreasing.
Segments are initialized from a globally consistent forward trajectory, so the
defects start at zero.

**Result, with provenance.** 16 cold initializations per configuration, Adam,
identical step budgets and initializations for both arms.
- Short horizon `T = 20 t_g` (~730 steps): single shooting wins; multiple
  shooting is *not* advantageous.
- Long horizon `T = 60 t_g` (~2200 steps, ~×40 tangent growth): multiple
  shooting gets ~3 orders of magnitude lower terminal loss and ~1 order lower
  initial-condition error, while single shooting stalls.

The 3-orders figure is measured on the lifted trajectory whose segments are not
yet continuous — the paper says so itself and names the IC error (1 order) as
the feasible, directly comparable metric. **Quote the 1 order, not the 3.**

**Why it should transfer.** Our multi-step trainer already chains segments with
per-lead ERA5 targets. Lifting the segment starts is nearly free here: the
lifted variables have a natural, physically consistent initialization (the ERA5
analysis at each lead), so defects start small rather than at zero-by-cold-start.
This is weak-constraint 4D-Var (Trémolet 2006, cited by the paper) applied to
the training window, and `da/` already owns control-vector, cost-function and
minimizer machinery to reuse.

**Cost and risk, stated up front.**
- Memory: adds `M−1` full model-state pytrees to the optimizer, ×3 with Adam
  moments. This is the dominant risk and must be sized before any build.
- The continuity penalty introduces `ρ` and its growth factor — new tunables,
  and a norm choice for `‖r_j‖` (unweighted L2 over a multi-variable ESM state
  is physically meaningless; needs per-variable scaling).
- The comparison against truncated BPTT is only meaningful if both arms are run
  at identical horizon, data, optimizer and step budget.

**Proposed first test — a twin experiment, not AMIP.** Shallow water or SCM
scale, recover a known perturbation from a terminal observation, single
variable changed between arms (truncated BPTT vs multiple shooting), 8+ seeds,
report IC error. Refutes if IC error does not separate at the horizon where our
adjoint is already known to blow up.

## P2 — Kink-immune gradient certification at non-smooth kernels

**What the paper did.** Sod shock tube, `N=200`, 250 *fixed* timesteps (fixed so
the FD baseline is not contaminated by the step count changing with the
parameters). `∇J` is not continuous, so central differencing averages across
the jump while AD returns the derivative of one branch of the discrete map.
They therefore certify AD against the **minimum of the two one-sided finite
differences** per parameter, `min(|g_AD − g_F|, |g_AD − g_B|)`, and show it
decays at the one-sided truncation rate `O(h)` down to `7.2e-6` at `h=1e-6`.

**Why we need it.** Our own source says the gradient at the C^0 sites in the
FV3 kernels is not certified because `check_grads(order=2)` — central
differences — cannot hold there. The consequence is that the gradient through
our limiters, positivity fixes and pressure-gradient kernels is currently
**unverified**, not verified-and-approximate. The same applies to saturation
adjustment and convection triggers.

**Proposal.** One shared helper (`min` over the two one-sided FDs, fixed step
count, tolerance as a function of `h`) plus its non-vacuity test (must go red
against a deliberately wrong VJP), then apply it at the sites currently
carrying a "not expected to hold" comment.

Cheapest item here, and it retires an existing hole rather than adding surface.

## P3 — Analytic functional-derivative convergence gate

**What the paper did.** Linearized acoustics with a closed-form Fourier-space
functional derivative of `J = ½⟨S_t U_0, S_t U_0⟩`. Sweeping resolution
`N ∈ {16..256}`, the AD gradient converges to the exact analytic gradient **at
the spatial order of the scheme**: `O(N^-5)` for their 5th-order FD solver
(3.4e-2 → 1.1e-6), `O(N^-2)` for the FV solver (6.2e-1 → 7.2e-3). They note they
are not aware of this test being used elsewhere in the differentiable-simulator
literature.

**Why it is stronger than FD.** An FD check only certifies that AD is consistent
with the *discrete* solver map — it cannot see an order-reducing defect in the
backward pass. An order-convergence test against the continuum adjoint can. For
a cubed-sphere/duogrid stack whose halo exchanges carry custom VJPs, that is
precisely the class of bug we would otherwise ship silently.

**Proposal.** Linearized gravity-wave or acoustic problem on a periodic plane
grid, closed-form adjoint, resolution sweep, assert the observed order matches
the scheme's nominal order. Plane grid first — the sphere adds metric terms that
would need their own derivation.

## P4 — Wire the gradient-horizon diagnostic into the trainers

`grad_horizon.py` measures adjoint-norm growth versus window length and
estimates the growth rate, and nothing in production calls it. Our curriculum
horizons are hand-set hours. Proposal: run the sweep once per training
configuration, log the measured growth rate, and derive (or at least bound) the
curriculum ladder from the measured usable horizon rather than from a guess.

This is the cheapest of the four and is the measurement that would tell us
whether P1 is worth building at all.

## Explicitly not taken from the paper

- Pallas kernel generation, GPU scaling, the 4th-order self-gravity scheme, the
  5th-order constrained-transport MHD scheme — solver work, unrelated.
- Tangent reorthonormalization on segments (Benettin et al. 1980) — the paper
  mentions it as a way to control numerical degeneracy of exponentially growing
  tangents but does not implement it. No evidence to act on.
- Their smooth flux blending toward Lax–Friedrichs. It is designed for
  smoothness but the paper never measures its effect on gradients, and our
  limiter set is different.

## Open questions for the user (no work starts before these are answered)

1. Which of P1–P4, and in what order? Recommendation: P4 (measure) → P2 (cheap,
   closes an existing hole) → P1 (the real lever) → P3.
2. P1's twin experiment needs a venue: shallow water on the plane, SCM, or the
   lat-lon primitive-equation stack where the NaN was actually observed?
3. P1 introduces new tunables (penalty weight, its growth factor, the norm used
   for the continuity defect). These are unasked choices and need to be decided,
   not defaulted.
