# SWG online-learning round-10 review against Maddison (2026)

Date: 2026-10-15

## Executive verdict

The pilot reproduces Maddison's central *method*--differentiate a deterministic
trajectory loss through the coarse model--but it is not yet a reproduction of
the experiment that produced Maddison's stable result.  It stops at the two
window lengths that were unstable in Maddison, omits from its loss the first
interior wall ring in the boundary region where Maddison later diagnosed strong
learned
dissipation, uses a different prognostic model and closure operator, trains on
only two years with one seed, and evaluates from exact rest.  The falling
validation loss therefore establishes successful optimization, not
a-posteriori fidelity.

The strongest causal hypothesis for the positive closure work is the omitted
wall ring, closely followed by stopping before the four-day window.  The
round-10 spatial result supports the wall hypothesis, but does not yet prove
it: `implied_subgrid_work.py` compares a model launched from *rebalanced* SSH
with a stored truth endpoint evolved from the original SSH.  Its quantity is
the one-hour correction demanded by this training contract, not a pure
matched-initial-state sub-grid acceleration.  It also has not been projected
onto the non-divergent potential-force class that the network can represent.
Pilot 2 is the right direct intervention, but its physical interpretation needs
that inexpensive decomposition.

My pre-result prediction is that including the ring will make the trained
closure's mean work negative and keep the 20-year runs near or below the
`nu=10` control.  Confidence is moderate rather than high: the negative ring
work is larger than the positive interior work, but the finite-difference
diagnostic contains the rebalancing mismatch and components outside the
closure's operator range.  I did not inspect any round-11 output when making
this prediction.

### Evidence and terminology

The comparison uses the supplied full texts of Maddison (2026) and Brolly
(2026), repository commit `816058407`, the round-7--9 receipts, and the
read-only pilot/round-10 artifacts under
`swgonl_rounds/round9_pilot/`.  Code references below are repository
`file:line` references at that commit.  Artifact references name a JSON key
rather than pretending that generated JSON has a stable source line.

The round-10 probes load four critical files from the diagnosis worktree rather
than this checkout (`scripts/validate/swgonl_round10/energy_budget.py:46-49`).
I compared its driver, online library, SWE RK3 implementation, and excluded-ring
configuration byte-for-byte with this checkout; all four match.  I also ran
the source-term calculation from this checkout on CPU/fp64 and independently
reduced the stored per-sample JSON, as detailed in the probe audit.

## Settings that differ

The table deliberately omits nominal agreements: batch size 10; five epochs at
each curriculum stage; network carry with optimizer reset; deterministic zero
noise; unfiltered *model* vorticity against a filtered/coarse target in the
template objective; and AdamW with intended Keras defaults.  The lane makes the
last item more reproducible by pinning `lr=1e-3`, weight decay `0.004`, betas
`0.9/0.999`, epsilon-hat `1e-7`, a constant schedule, and no clipping
(`config/swg_online/pilot_template_ring_excluded.json:31-48`;
`packages/ml/legoesm/training/swg_online.py:1526-1586`), whereas Maddison does
not identify the Keras version or spell out those values [Maddison 2026,
Section 2.6].  That is an uncertainty in exact reproduction, not evidence of a
different intended optimizer.

| Setting | Maddison (2026) | This lane | Consequence |
|---|---|---|---|
| Prognostic model, coarse resolution, and time integration | Rigid-lid, one-layer streamfunction/vorticity equation on a 65 x 65, 62.5 km grid; `dt=30 min`; Arakawa Jacobian and CNAB2, with drag/viscosity implicit and the CNN explicit [Sections 2.1 and 2.6; Equations 1--4; Table 1]. | Explicit-free-surface, one-layer NEMO shallow water at 50 km, 500 m and `dt=400 s`, advanced by the certified three-stage RK3 card (`docs/ocean/fidelity/testcases/nemo_testcases_l1_swg_online_design.md:15-20`; `packages/ocean/legoesm/ocean/dynamics/swe_rk3.py:140-149`). | This is a method transfer, not a solver reproduction.  Free-surface gravity modes, C-grid/QCO geometry, and stage coupling can change what the closure must learn. |
| Truth resolution and filter | 2049 x 2049, 1.95 km, `dt=2 min`; instantaneous vorticity is Gaussian-filtered with standard deviation 62.5 km, then sampled at 62.5 km [Section 2.3; Table 1]. | 6.25 km truth; standard deviation 50 km (eight fine cells) is applied to fine finite-volume volume and transports with wall parity, followed by exact conservative restriction (`nemo_testcases_l1_swg_online_design.md:34-54,58-63`). | The reference resolves the 7.94 km Munk scale much less strongly, and the operators are not mathematically identical: primitive extensives are filtered here, vorticity there. |
| Training start coarse-graining | Each input is exactly the filtered/coarse vorticity at the window's first record [Sections 2.3--2.4]. | The filtered/restricted stored `(SSH,U,V)` is used, but SSH alone is iteratively rebalanced; velocities are unchanged (`packages/ml/legoesm/training/swg_online.py:964-969,1172-1183`). | The model start is no longer exactly the stored truth state that generated its later targets.  This is the main confound in calling the one-hour residual “sub-grid forcing.” |
| Target coarse-graining | The later target is the same directly filtered/downsampled vorticity record as the start [Sections 2.3--2.4]. | Targets are unmodified stored right endpoints 1 through `24w` hours later (`packages/ml/legoesm/training/swg_online.py:1020-1025`; `scripts/run/run_swg_online_training.py:779-785`). | Rebalancing is applied only to the model initial state, not to the truth trajectory. |
| Equation 7 support and boundary ring | `W=dx_C^2` on every non-boundary vorticity point and zero only on the boundary [Section 2.6, Equation 7].  Thus the first interior ring is included. | Pilot 1 also zeros the first interior F ring; pilot 2 restores the Equation 7 support (`packages/ml/legoesm/training/swg_online.py:1294-1311`; `config/swg_online/pilot_template_ring_excluded.json:5-7`; `config/swg_online/pilot_template_ring_weighted.json:3-7`).  The rest of the Equation 7 prefactor and hourly/window reduction is literal (`packages/ml/legoesm/training/swg_online.py:1336-1373`). | Pilot 1 removed 14--20% of the no-closure loss and, according to the one-hour training residual, removed the only net energy-sink region (`nemo_testcases_l1_swg_online_design.md:193-201`).  This is the most directly aligned difference. |
| Window lengths and curriculum extent | Every integer window from 1 through 60 days, five epochs each; the network is carried and AdamW reset [Sections 2.4 and 2.6]. | The full authorized lane is only 1, 2, 4, 8 days; both pilots stop after 1 and 2 (`nemo_testcases_l1_swg_online_design.md:203-210`; `config/swg_online/pilot_template_ring_weighted.json:44-51`). | The pilot stops before Maddison's transition: 1/2 days unstable, 3 days stable but over-energized, and 4 days the best compromise [Sections 3.2--3.6 and 5]. |
| Training volume and held-out splits | Twelve post-spinup years sampled hourly (`105121` records); first 80% (about 9.6 years) train and the contiguous last 20% validate; no independent test set is reported [Sections 2.3--2.4]. | The pilot trains on days `[731,1461)` (730 days), validates on `[4265,5111)` after a long separation, and pins an independent test year `[5141,5506)`; only one seed (`config/swg_online/pilot_template_ring_excluded.json:14-31`).  The intended full train split is 3,504 days (`nemo_testcases_l1_swg_online_design.md:123-127`). | The pilot uses 20.8% of its intended training split and about one fifth of Maddison's training duration.  Its validation is more separated and it has a real test set, but sampling error and seed dependence are much larger. |
| Training arithmetic | Online training is fp32; other calculations are fp64 [Section 2.1]. | The card, data, network parameters, rollouts, and gradients are fp64 (`packages/ml/legoesm/training/swg_online.py:295-331,630-641`; `scripts/run/run_swg_online_training.py:328-372`). | This should improve numerical fidelity but is another non-identical optimization landscape; Maddison did not study the fp32 effect. |
| Closure variable and Equation 6 scaling | A scalar CNN maps normalized relative vorticity directly to a local vorticity tendency `S_theta`; exact scales are `1/(|beta|L)` and `|tau0| pi/(rho0 D L)` [Section 2.1, Equations 5--6]. | Four channels (SSH, T-centred U/V, mask) map to an F-point transport potential.  A boundary-zero clockwise rotated gradient yields live-face acceleration and a divergence-free transport force.  Equation 6 is transferred through a shallow-water grid-scale analogy, not used directly (`swg_online.py:295-360,381-436`; `swe_rk3.py:73-107`). | The hypothesis class and conditioning variables differ.  The potential enforces useful conservation and wall-normal-force constraints, but may not represent all of the diagnosed endpoint residual; the scale mostly controls conditioning, not the final sign of work. |
| Network architecture and size | Seven width-5 convolutional layers, 32 filters, ELU, residual/skip paths from the input and every layer, trainable skip scale/bias; 154,857 weights [Section 2.5]. | Five width-3 convolutions (four hidden plus output), 16 hidden channels, ELU, no residual connections, no final bias; 7,696 parameters and an 11-cell/550 km receptive field (`nemo_testcases_l1_swg_online_design.md:148-154`; `config/swg_online/pilot_template_ring_excluded.json:8-12`). | The lane has about 20 times fewer parameters, a different receptive field, and no explicit identity/linear skip channel. |
| Where the learned acceleration enters | `S_theta` is evaluated as an explicit vorticity-tendency term in CNAB2 (CNAB1 on the first step) [Section 2.1, Equations 4--5]. | The potential is evaluated from each RK stage state and its acceleration is appended to both momentum RHS arrays at every stage (`packages/ml/legoesm/training/swg_online.py:1447-1467`; `packages/ocean/legoesm/ocean/dynamics/swe_rk3.py:282-293,400-416`). | Identical checkpoint values do not imply identical discrete feedback or stability. |
| Free-run start, length, and sampling | Each reported calculation has a 12-year spinup followed by 12 diagnostic years, hourly composite-trapezoid sampling [Section 3.1]; the paper does not explicitly state the coarse runs' initial field beyond this spinup protocol. | Each checkpoint is integrated 7,300 days (20 years) from the card's exact rest state, with daily snapshots and daily means (`nemo_testcases_l1_swg_online_design.md:258-270`; `config/swg_online/pilot_template_ring_excluded.json:50-55`).  Round 10 adds diagnostic three-year runs from rebalanced truth day 4265. | Exact rest is a substantially more out-of-distribution launch than a long spun-up evaluation, but it cannot explain positive work already measured on training-window states. |
| Reported stability and energy behavior | `w=1,2` become numerically unstable; `w=3` is stable but has excessive large-scale/EKE and is the sole reported net energy source; all later cases have net-negative CNN work, generally more negative with window length.  `w=4` best matches the filtered spectrum/mean/EKE [Sections 3.2, 3.4--3.6; Equation 9; Figures 7--11]. | Pilot 1's 20-year runs remain finite but reach 25--40 times control KE, 6--10 m/s maximum speed, and 24--40 m SSH range.  On fixed training/validation starts every trained checkpoint has positive mean work; final `w=2` is `+1.4321e-9 m2 s-3`, 0.302 of wind work (artifact `step6_closure_work_by_epoch.json`, `per_network.w02_e05`). | “Finite” is weaker than Maddison's statistical stability.  The sign resembles Maddison's too-short/near-threshold closures, not his four-day result. |
| Evaluation metrics | Instantaneous PV, offline forcing and L2 error, KE spectra and large-scale KE, integrated/local/spectral CNN KE forcing, MKE/EKE, mean streamfunction, and EKE maps [Sections 3.1--3.6; Equations 8--9]. | A frozen eight-window gate scores mean face-transport RMSE, KE-spectrum relative L1, wet EKE-map NRMSE, basin-EKE ratio, two top-octave fractions, native grid-noise index, and hard thickness/Courant/finite health (`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_online_evaluation.py:85-92,139-204,873-934`). | The lane has a much sharper held-out pass rule, but one seed is explicitly unscorable by that gate (`nemo_testcase_l1_swg_online_evaluation.py:1332-1357`).  Closure work is diagnostic, not yet part of the gate. |

## What worked for Maddison, and why

1. **The coarse model was deliberately under-dissipated.**  Keeping
   `nu=10 m2 s-1` at 62.5 km left grid noise that the through-model objective
   could learn to remove rather than hiding the resolution gap behind a tuned
   viscosity [Maddison 2026, Section 2.6].  The unparameterized control stayed
   finite but had grid noise and a poor jet [Section 3.1].

2. **The curriculum crossed the instability threshold.**  Offline forcing
   agreement was favorable for `w=1,2` even though those closures were unstable
   in free runs; Maddison therefore explicitly concludes that offline skill was
   a poor predictor of online behavior [Sections 3.2--3.3].  The paper's
   mechanism is that “instability may need to have sufficient time to develop
   and influence the loss” [Section 3.6].  One and two days were too short,
   three days acquired enough dissipation to remain finite but still injected
   too much energy, and four days gave the best spectrum, mean flow, and
   variability compromise [Sections 3.4--3.6 and 5].

3. **The learned closure had the right spatially cancelling energy pattern.**
   Maddison did measure energy behavior, not merely numerical survival.  With
   local work `G=-psi S_theta` [Equation 9], Figure 9 shows `w=3` as a net
   source and all later reported closures as net sinks.  Figure 10 locates
   strong dissipation near the western boundary and dynamic backscatter in the
   interior [Section 3.5].  That is strikingly consistent with round 10's
   negative wall-ring/positive-interior training residual.  It is still an
   inference, not Maddison's ablation: the paper did not remove the first
   interior ring and rerun training.

4. **Four days stayed below the predictability horizon.**  Maddison estimates
   a roughly nine-day leading Lyapunov time.  Longer windows became increasingly
   dissipative and eventually severely suppressed EKE, while some very long
   cases also recovered grid noise [Sections 3.2 and 3.6].  The claimed success
   is therefore a window compromise, not monotonic improvement through 60
   days.

The paper does **not** put an energy/work penalty in Equation 7, report a
closed total-energy drift budget, or impose a sign constraint on `S_theta`.
It does report integrated, local, and spectral closure KE forcing, long-run KE
spectra, MKE and EKE.  Its wall treatment is free-slip `psi=zeta=0` plus CNN
zero padding [Section 2.1]; Equation 7 excludes boundary points but includes
the first interior ring [Section 2.6].  It discusses the western boundary as
the main learned dissipation region [Sections 3.3 and 3.5], but presents no
wall-ring sensitivity experiment.

## What may be missing here

Ranks concern the **positive-work result**, not every possible fidelity error.
Each row gives one discriminating test and a pre-result expectation.  Costs use
the measured pilot profile: about 1.51 GPU-hours for the two training stages
and about 0.67 GPU-hours per 20-year free run
(`nemo_testcases_l1_swg_online_round8_integration_receipt.md:158-168,136-140`).

| Rank | Hypothesis and present evidence | One decisive measurement/run | Cost | Prediction |
|---:|---|---|---:|---|
| 1 | **The excluded first interior wall ring removed the net sink.**  Under the current one-hour training residual, mean work is `-2.7291e-9` in the ring and `+1.1913e-9 m2 s-3` elsewhere; pilot 1 learned `+1.3370e-9`, close to the interior source.  Maddison weighted this ring and found boundary dissipation. | Pilot 2: change only `wall_ring=weighted`; evaluate work on the same 20 fixed starts and complete both 20-year free runs.  Confirm if final mean work is negative and long-run KE is no higher than the `nu=10` control; otherwise refute. | Already running; about 2.8 GPU-h for two stages plus two runs. | Confirm: negative work, a stable attractor at or below control KE.  The sign prediction is stronger than the exact KE-level prediction. |
| 2 | **The curriculum stops before the horizon that taught Maddison stability/dissipation.**  Maddison's 1/2-day closures failed, the 3-day closure was a source, and `w=4` was the first good energy result.  Pilot 1 stops at two days. | Carry pilot 2's final network into one otherwise identical weighted `w=4`, five-epoch stage; score fixed-state work and one 20-year free run.  Confirm if work becomes more negative and KE/EKE move toward truth without a health failure. | About 0.8 GPU-h training plus 0.7 GPU-h free run. | More negative work and lower KE than weighted `w=2`; unlikely variance collapse at four days because it remains below this lane's measured 10.3-day predictability time. |
| 3 | **SSH-only rebalancing makes the diagnosed sink partly an initialization correction.**  The coarse forecast starts from rebalanced SSH, while its truth endpoint belongs to the original stored trajectory (`swg_online.py:964-969,1020-1025`).  Pressure/divergent adjustment is especially liable to concentrate near walls. | On the same 20 starts, difference the target residuals from forecasts launched with stored SSH and rebalanced SSH; report the rebalancing contribution to ring/interior work.  Confirm if it supplies at least half of the present negative ring work; refute below half. | CPU, roughly 5--15 min using the existing one-hour scan. | A material fraction of the apparent ring sink will be rebalancing correction, so the pure model-resolution sink will be weaker than `-2.7291e-9`. |
| 4 | **Part of the demanded correction is outside the scalar-potential closure class (and the Equation 6 analogue may condition it poorly).**  The lane predicts a boundary-zero transport potential, whereas Maddison predicts vorticity tendency directly.  The observed acceleration cosine is only `+0.379`. | In the probe's wet-face L2 inner product, project the corrected one-hour acceleration onto the exact discrete `Phi -> (a_u,a_v)` operator range, then recompute ring/interior work.  Confirm if at least half of the negative ring work lies in the orthogonal component; refute otherwise. | CPU, minutes once the states are loaded; no training. | Some divergent/rebalancing work will be unrepresentable, but a negative representable wall component will remain.  Equation 6 scale alone is unlikely to set the wrong sign because all leaves are live and optimization rapidly changes work. |
| 5 | **Two training years under-sample regimes and walls.**  The pilot uses 730 of the intended 3,504 train days; Maddison uses about 9.6 train years. | Repeat the winning weighted curriculum with `[731,4235)` and no other change, then compare fixed-start work and one free run. | About 4.8 times pilot training: roughly 7.3 GPU-h through `w=2`, plus runs. | Smaller checkpoint/validation variance and better spatial skill, but no sign reversal beyond what weighting/window length already causes. |
| 6 | **The one observed seed may be atypical.**  All 20 sampled starts share the sign, but they share one trained network.  The held-out gate itself refuses fewer than three seeds (`nemo_testcase_l1_swg_online_evaluation.py:873-890`). | Train two additional seeds with the winning configuration and apply the existing three-seed gate. | Two replicas, about 5.6 GPU-h at pilot volume. | Work sign repeats; magnitude and held-out skill vary.  A seed-only explanation is unlikely because positive work appears after epoch 1 and across all sampled states. |
| 7 | **There is no energy constraint.**  Equation 7 can lower trajectory-vorticity error while injecting KE; pilot 1 demonstrates that freedom.  This is a vulnerability, but not a difference that prevented Maddison's success. | A/B train the winning setup with one preregistered mean-work penalty or inequality, keeping every other choice fixed; judge Equation 7 loss and the existing gate, not stability alone. | New reviewed implementation plus about one pilot (2.8 GPU-h). | It forces non-positive work but trades some short-horizon loss; it should be a fallback after supervised ring/window information, not the first repair. |
| 8 | **The template compares unfiltered model vorticity with a filtered target.**  This can load the loss with grid-scale mismatch, but it is an intended agreement with Maddison, and filtering the target normally asks for dissipation rather than a source. | Weighted-ring A/B between `template` and `symmetric`, same seed/data/curriculum, with the existing scorer's operator-specific frozen bars. | One extra pilot, about 2.8 GPU-h. | `symmetric` changes spectral detail but does not by itself cure positive work; rank is low because the paper succeeded with the template asymmetry. |
| 9 | **Brolly's deterministic point-loss degeneracy suppresses variance.**  Brolly proves that long-lead deterministic discrepancy objectives penalize predictive spread, while the energy score's ensemble term avoids this [Brolly 2026, Sections 2.2--2.4, Equations 8--10, Propositions 1--2].  This lane's stochastic interface is dormant (`nemo_testcases_l1_swg_online_design.md:349-359`). | After a stable deterministic baseline exists, train stochastic `S>=2` with the energy score and compare the frozen EKE ratio/spectrum with deterministic training at the same horizon. | At least roughly `S` times rollout work plus new experiment configuration and review. | It should preserve EKE at longer horizons, but it does not explain the present positive mean work: Brolly predicts over-damping/variance loss, the opposite directional symptom. |
| 10 | **Starting free runs from exact rest exaggerates the failure.**  Three-year day-4265 launches plateau at only 1.6--3 times control KE, whereas from-rest runs reach much more extreme states. | Extend both trained day-4265 launches to 20 years and score them with the same diagnostics. | About 1.3 GPU-h for two continuations. | It changes transient severity, not closure-work sign; positive work is already present on training starts.  The high-energy attractor should persist or emerge more slowly. |

The ranking separates Brolly's warning from the current fault.  Brolly's
Proposition 1 says long-lead deterministic point losses push forecasts toward
a climatological center with suppressed spread; his numerical deterministic
closures become overly smooth, while stochastic energy-score closures recover
the spectrum [Brolly 2026, Sections 2.4 and 3.2].  Basin KE 25--40 times a
control is not evidence against that theorem--mean flow and variance are not
the same statistic--but neither is it the predicted signature.  The lane's EKE
ratio gate is the right place to test collapse once a stable baseline exists.

## Audit of the round-10 probes

### Independent checks performed

All commands used `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`, the prescribed
`PYTHONPATH`, and the prescribed virtual-environment Python.  No GPU or
round-11 artifact was read.

1. I reran `energy_budget.make_terms` from this checkout for the first `w=2`
   training-batch start.  It reproduced the artifact exactly:
   closure `+8.572366509010006e-10`, wind
   `+4.807255494841736e-9`, viscosity `-3.414918062656318e-11`,
   drag `-2.9848627496630343e-9 m2 s-3`, and KE
   `0.014924313748314457 m2 s-2`.  The analytic drag identity
   `drag_work=-2 r KE` had residual `1.43e-22 m2 s-3`.
2. Across every raw sample in `step4b_energy_budget_day4265.json`, the maximum
   absolute drag-identity residual was `1.52e-21`.  I independently refit the
   logged closure-work/KE slopes: `1.0561956256357354` for the pooled
   truth-start samples and `0.5505502454102499` for the pooled from-rest first
   300 days, exactly the stored values.
3. I recomputed every stored summary field used here in
   `step6_closure_work_by_epoch.json`; maximum numerical difference was zero.
   I likewise recomputed the selected Step-7 means/medians/minima/maxima with
   zero difference, and verified `ring + interior = total` for every start to
   `1.04e-24 m2 s-3`.
4. I independently refit every Step-5 constant-sign log slope with zero
   difference.  Eight of the ten trained network/state pairs have fitted
   closure exponents `1.2119--1.9222` versus drag exponent 2; two pairs change
   closure-work sign over `lambda=0.25..3` and correctly have no exponent.

These checks reproduce arithmetic, not the physical interpretation of each
quantity.

### `energy_budget.py`: sound for its stated source-work diagnostic

The on/off differences isolate closure, wind, viscosity, and drag in the
model's own stage-3 source program (`energy_budget.py:85-140`).  Division by
the stage-3 `dt` gives acceleration, and `u*a_u+v*a_v` has the stated
`m2 s-3` units.  Signs are correct: wind/closure “on minus off,” viscosity
“with minus A_h=0,” and drag “with minus r=0.”  The exact drag identity above
is a strong sign/unit control.  A zero-potential closure also gives exactly
zero work (`closure_work_by_epoch.py:95-117`).

The scope must remain explicit.  This is the derivative of the probe's
velocity diagnostic
`0.5*(mean_U(u^2)+mean_V(v^2))`, not an exact discrete shallow-water total
energy budget.  It is unweighted by live layer mass, omits gravitational
potential energy, advection/pressure conversion, and evaluates the closure at
full-step entry states rather than integrating work over all three RK stage
states (`energy_budget.py:176-203`).  Those facts do not invalidate the robust
claim “the learned force does positive instantaneous work on sampled flows,”
but they prohibit calling the four reported terms a closed energy budget.

### `implied_subgrid_work.py`: arithmetic sound, SGS interpretation defective

Because initial U/V are identical, `(truth_1h-model_1h)/3600` is a valid
one-hour mean **endpoint-correction acceleration**, with correct units and
sign (`implied_subgrid_work.py:176-179`).  Work uses start velocity, so it is a
linear projection of that correction, not the exact one-hour energy change.
The face-ring masks follow the documented adjoining-T-cell convention
(`:86-105`), and the ring/interior work split is an exact partition because
both use the full-domain face-count denominators (`:173-174,230-232`).  The
controls prove the two steppers agree, self-comparison is zero, and the rebuilt
closure/wind fields reproduce `make_terms` (`:188-214`); their recorded maxima
are `0`, `0`, and `3.55e-15` relative.

The defect is the scientific label.  `batch.initial` contains rebalanced SSH
(`swg_online.py:964-969`), while `batch.target` is read directly from later
stored truth records (`swg_online.py:1020-1025`;
`implied_subgrid_work.py:66-82`).  Thus truth and model do not evolve from the
same complete initial state, contrary to a pure SGS-tendency interpretation.
Moreover, a face-work ring is not identical to an Equation-7 F-vorticity loss
ring, and the residual has not been projected into the potential operator's
range.  The numerical statement is therefore:

> Under the exact training data contract, the one-hour velocity correction has
> mean work `-1.5378e-9 m2 s-3`, decomposed as ring `-2.7291e-9` plus interior
> `+1.1913e-9`, over these 20 starts.

It is **not yet**: “the physical SGS forcing from the same rebalanced state is
a sink entirely in the wall ring.”

### `closure_work_by_epoch.py`: sound measurement, causal title too strong

The script loads the fresh initialization and all ten actual checkpoints,
holds 15 starts fixed, reuses the validated term isolator, and has two capable
controls (`closure_work_by_epoch.py:48-79,82-150`).  It correctly shows the
initial network at only 0.00090 of wind work, epoch-1 `w=1` already at 0.222,
the `w=1` peak at epoch 3 (0.652), and final `w=2` at 0.302, with positive work
on every sampled start.  Loss and work decline together during the `w=2`
stage; during `w=1`, work first rises while loss falls.  The script wisely
reports within-stage correlations because pooled `w=1/w=2` losses are
different objectives (`:157-173`).

What it cannot show is that training *rewarded energy injection*: it measures
checkpoint association, not the dot product between a loss gradient and a
work gradient or a controlled energy-penalty intervention.  The observation
“positive work is learned by epoch 1 and persists” is sound; the causal wording
in the module title is not established.

### `closure_scaling.py`: raw samples sound; threshold label and universal inference defective

Scaling SSH anomalies and U/V by fixed `lambda` and reevaluating the exact
source terms is implemented as documented (`closure_scaling.py:39-73,91-107`).
Drag scales as `lambda^2` to roundoff, the lambda-one controls for matching
network/state pairs are exact, and the stored constant-sign exponents refit
exactly.  These are useful local response measurements.

Two issues prevent the claimed general conclusion.  First, two of ten trained
network/state pairs change closure-work sign over the sampled range; for the
`w=2` network on rest-run day 300 the work is positive through `lambda=2` and
negative at `lambda=3`.  Second,
`closure_exceeds_abs_drag_from_lambda` stores merely the *first sampled hit*
(`closure_scaling.py:104-105`), although the relation need not remain true for
larger lambda.  On that same state it reports `0.25`, while closure exceeds
drag at 0.25 and 0.5 but not at 1.0.  The lambda-one “control” on a no-closure
control trajectory also compares a trained closure's work with the Step-4b
no-closure value zero (`:110-120`), so only its state-dependent wind/drag/KE
parts are a valid cross-probe control.

Finally, the scaled states are fixed-pattern, generally off-attractor states;
a finite-range exponent below two does not prove a trajectory equilibrium or
that “a balance always exists.”  The defensible conclusion is narrower: on
eight constant-sign sampled pairs the learned source grows more slowly than
linear-drag work over this lambda range, while two pairs reverse sign.

**Probe verdict:** `energy_budget.py` and `closure_work_by_epoch.py` are sound
within their stated diagnostics.  `implied_subgrid_work.py` is defective when
reported as a matched-state physical SGS force, though its training-residual
arithmetic and wall partition are sound.  `closure_scaling.py` has sound raw
samples/exponents but a defective “from lambda” derived label and does not
support the universal balance inference.

## Pilot 2 prediction and next two experiments

Pilot 2 changes the single setting most directly implicated by both Maddison
and round 10.  I predict its learned mean closure work will turn negative by
the end of the weighted curriculum and its 20-year runs will remain finite and
settle at or below the `nu=10` control KE, plausibly closer to the sampled
filtered-truth KE (`0.0155 m2 s-2`) than the control (`about 0.022--0.023`).  The reasoning
is the measured sign balance: adding a `-2.73e-9` wall-region correction to a
loss that previously exposed a `+1.19e-9` interior correction reverses the net
supervised energy signal, and Maddison's successful closures learned the same
boundary-sink/interior-source pattern.  The qualification is material: the
diagnosed wall sink may include SSH-rebalancing and unrepresentable components,
so a reduced positive source or a near-zero result would not be surprising.

After pilot 2, in order:

1. **Commit and run the corrected one-hour residual decomposition**: stored-SSH
   versus rebalanced-SSH launches, then exact projection into the potential
   operator range, with the existing ring/interior work table and planted
   controls.  This is CPU-minutes and determines whether pilot 2 learned a
   physical/resolution correction or a data-preparation artifact.
2. **Continue the weighted network to `w=4` for five epochs and run one
   20-year free evaluation**, changing nothing else.  This tests the most
   decisive Maddison result that the present pilot has not attempted.  Only
   after the weighted four-day result should the campaign spend on full data
   and two additional seeds.

No new scientific choice is required to run those two measurements: their
operators, comparison states, window, seed, data subset, and success criteria
are specified above.
