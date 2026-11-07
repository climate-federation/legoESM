# Dual review — CAM6 cloud macrophysics on the spectral (WeatherBench) lane (2026-09-24)

Worktree wt_wbcam6, branch wb/cam6-baseline, on 938855bde. Prompts: `*_prompt.md` siblings. Final reports only; exploration transcripts are not kept.

## Round 1

### codex

**HOLD.** The wiring largely mirrors hydrostatic, but the split factory bypasses the producer gate, and cold-start radiation consumes uninitialized cloud diagnostics.
Ranked by decision impact:
1. **B — Producer gate bypass: confirmed.** **MEASURED:** `cam6_clubb + louis + split_rad=True` builds successfully with routing enabled and a zero cloud-fraction carry. The [split factory](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/packages/ml/legoesm/training/aimip_params.py:1629) constructs non-radiative physics with radiation disabled, then constructs radiation independently; neither checks the combined producer requirement. Validate the resolved configuration before splitting. The current CLUBB deck avoids this combination, but the advertised guard is incomplete.
2. **B — Cold-start zero carry: confirmed.** **MEASURED:** the classical factory’s state-aware seed leaves cloud fraction and deep-convection mass flux zero. **READ-OFF-THE-CODE:** it seeds TKE only; [initial radiation receives that carry](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/packages/ml/legoesm/training/neural_gcm_spectral.py:1917). At cadence six, the resulting heating persists through steps 0–5: three hours at the deck’s timestep. With its `max_random` optics, liquid/deep cloud cover is suppressed; “every cloud” overstates this because diagnosed ice stratus can remain. The producer gate cannot prevent this startup transient.
   Leaving `spectral_amip_rollout` loud is appropriate for this WB-scoped repair. It already lacked CAM6 support, and merely forwarding its fixed sizing carry would introduce persistent zero diagnostics. Keep that limitation explicit.
3. **A — Lag is correct at refreshes.** **MEASURED with a sentinel carry:** cadence one reads successive entering carries; cadence six reads the initial carry, then the carry after six publications. **READ-OFF-THE-CODE:** \(P_{k+1}=F(S_k,P_k)\), while refreshed radiation uses \(R(S_k,P_k)\), matching the hydrostatic accumulator’s read convention. Cadence holds that heating between refreshes; it does not introduce another carry-update lag. `_ps_entry` is evaluated repeatedly for RK tendencies, but its returned carry is persisted once.
4. **C — No rematerialization/None-pytree defect found.** **MEASURED, ad hoc:** real RRTMGP, real checkpoint wrapper, and `eqx.filter_value_and_grad` produced a finite, nonzero derivative with respect to supplied cloud fraction using subcolumn optics. **READ-OFF-THE-CODE:** argument seven remains dynamic; static indices remain grid/vertical coordinate. This establishes carry-to-radiation sensitivity, not full CLUBB-parameter-to-WB-loss qualification. Clipping, ice dominance and Boolean subcolumn masks still limit sensitivity locally.
5. **D — No signature mismatch found.** **READ-OFF-THE-CODE:** current marker setters are radiation builders whose callables accept `phys_state`, including the checkpoint wrapper. Dictionary rebuilding preserves forcing. Radiation currently occupies the first dispatcher position; the later read-only branch lacks direct test coverage.
6. **E — Publication is mechanically consistent.** **READ-OFF-THE-CODE:** every non-`None` dispatch branch defines `conv_out`; registered producers use the common `ConvectionOutput`, whose added fields default to `None`. ZM’s column/interface ordering matches the spectral flattening. **MEASURED:** the new publication and half-publication refusal tests pass; their uniform values do not independently prove column ordering.
7. **F — Scientific decision remains startup treatment.** Deriving the routing flag and forwarding existing fields are mechanical. Accepting zero liquid/deep diagnostics for the first radiation window needs an explicit owner decision. Introducing diagnostic initialization or a warm-up would also be a scientific choice; neither should be silently added.
8. **G — Ratchets do not certify this contract.** The dispatch baseline pins unknown-value rejection, not the exact supported-lane set or refusal wording. New tests cover spectral acceptance and nonhydrostatic refusal, but omit the split-factory producer bypass and cold-start values.
**Validation:** independently reran **46 tests, all passed**, with imports pinned to this worktree. Your larger regression result remains user-reported. No files changed.
**UNVERIFIED:** full WB loss gradients, production stability/memory, and the scientific acceptability of cold-start zero cloud diagnostics. Runtime probes above were ad hoc review checks, not committed campaign measurements.

### GLM

# Adversarial review: spectral cam6_clubb READ-side wiring

**Verdict up front: SHIP**, with one cheap pre-merge hardening I'd insist on (item B-2 below) and three named follow-ups. The diff does unblock a lane that was hard-down, the mirror is faithful where I can trace it, and the tests are genuinely non-vacuous. The residual risk is concentrated in one place: the diff still permits *silent* None/zero cloud paths that it assumes are caught downstream — the exact failure class this work exists to kill.

Where I say "verify on tree," I mean I only have the diff plus the two reference excerpts; those claims need a file read, not my inference.

---

## A. Lag convention — **one step, correctly. Cadence is orthogonal.**

Trace the carry, not the comments:

- Step `k`'s `_rad_refresh` reads `phys_state` entering step `k` = `phys_state_new` published during step `k−1`.
- Step `k−1`'s publication is `_ps_entry(state_{k−1}^{entry}, ..., phys_state_{k−1})` — diagnosed from the **pre-step** state of `k−1`.
- Hydrostatic: radiation runs first in the chain at step `k`, reading the carry published at the end of step `k−1`'s physics, which ran on step `k−1`'s entry state.

Both lanes: radiation refreshing at step `k` sees cloud diagnosed from the state at entry of step `k−1`. **Lag = exactly one step, equivalent.** The pre-step harvest (`_ps_entry` on the carry state, not the integrated state) is what makes the spectral version match despite radiation not literally being "first in a module chain" — the comment in the diff is correct, and the double `_ps_entry` call (harvest + per-stage) does not add lag because both read the same pre-step state.

`rad_update_interval=6` does **not** change this. The refresh at step `k∈{0,6,12,...}` still reads the one-step-lagged carry; between refreshes the *tendency* is stale by design, identically to the hydrostatic `update_interval_steps` caching. The only coupling worth noting: at a refresh, the cloud optical inputs are one state-step older than the T/q/p the same radiation call consumes. That mismatch is inherited from the hydrostatic design, not introduced here.

## B. Zero/None-carry paths — **yes, at least one real one; and the "loud" claim is unpinned.**

Three distinct paths:

1. **First gating window of every fresh rollout — zeros, silently.** `phys0` comes from `phys_state_in`, `seed_phys_state`, or `_ps_init`. The seed seeds shear-equilibrium TKE *moments*; `cloud_fraction`, `conv_mass_flux_up`, `conv_icwmr` are zero-initialized. `_init_rad(phys0)` and the step-0 refresh (step 0 satisfies `0 % 6 == 0`) both read that zero carry → **clear-sky radiation for the entire first gating window** (6 × 900 s). Hydrostatic has the same first-call behavior, so this is parity — *but* hydrostatic production runs are long, while **WB training calls `spectral_rollout` fresh per sample**, so every training rollout re-pays a zero-cloud first window. That's a systematic, not transient, training bias. This is the single highest-impact item in the review. It's a decision (accept parity / seed a diagnostic cloud_fraction in `seed_phys_state` / force the RH path on the init call), and it currently ships silently.

2. **`_cam6_cf_active and phys_state is None` is not guarded in `_physics_fn_core`.** The build-time gate catches flag=False. The runtime None case (stateless caller, `spectral_amip_rollout`, any future non-threading caller) falls through to `cloud_fraction_override=None` and whatever `compute_cloud_properties` does with cam6_clubb and no override. You *assert* it raises on `spectral_amip_rollout` — I cannot confirm from the diff, and **no test pins it**. If the backend instead falls back to RH or zeros, you've rebuilt the silent-clear-sky bug one layer down. One line fixes this:
   ```python
   if _cam6_cf_active and phys_state is None:
       raise ValueError("cam6_clubb requires the lagged phys_state carry ...")
   ```
   This is my pre-merge condition. Note it cannot fire in the stateless scan branch for a legit deck (producer gate requires clubb, which is stateful/marked), so it's safe to make hard.

3. Leaving `spectral_amip_rollout` raising: **right call in direction** (loud > silent), but the loudness is currently an accident of an untested backend path. With the guard from (2), it becomes a guaranteed, correctly-located error. As shipped: probably loud, possibly very quiet.

## C. checkpoint / autodiff — **clean; one memory note.**

- `static_argnums=(1,2)` marks grid/sigma; `phys_state` as 7th positional doesn't disturb it. The wrapper's kwargs→positional plumbing already existed for `forcing`.
- `None` fields in the NamedTuple are empty pytree nodes; the field *names* (hence structure) are static per config, so no re-trace/structure-flip hazard inside `scan`.
- Gradient: `phys_state` rides the scan carry → `cloud_fraction_override` → cloud optics → RRTMGP, under `nothing_saveable` at the step level (recompute) and default policy at the radiation core. Backward through both checkpoints is supported; nothing here severs it. Same differentiability contract as hydrostatic. Note the (desirable, but name-it-under-F) consequence: **training gradients now couple radiation loss → CLUBB carry**, which no spectral baseline ever had (cam6 previously raised).
- Cost note: the inner checkpoint now *saves* the full `phys_state` pytree (clubb_moments `(ncol,15,nlev+1)` + carries) as an input for backward. Modest vs. the state, but it's a real activation-memory delta on the training hot path.

## D. Marker/signature mismatch in combined — **no current case; failure mode is loud.**

`_wants_phys_state_ro` has exactly one producer on this lane (`_make_spectral_pe_radiation`, gated on `use_clubb_cloud_fraction`), and that fn's `physics_fn` signature has `phys_state`. A future producer that marks without the kwarg fails with a `TypeError` at first trace — acceptable. Verified that combined picks the flag up from `config.radiation` (the combined test passes it only via the config). Both dispatcher branches (first-fn `_fwd0`, subsequent `_fwd`) forward; the stateful `with_phys_state` branch already receives it positionally. No stale-key hazard (`{**_fwd, ...}` copy). Clean.

## E. Convection bridge — **logic mirrors hydrostatic; two verify-on-tree items.**

- `conv_out` defined-in-every-branch-when-`conv_fn is not None`: plausible but **not demonstrated by the diff or tests** — both tests fake `_get_convection_fn`. Read the dispatch chain; a fall-through branch under a non-None `conv_fn` would `NameError` at trace.
- Real ZM shapes: the fake pins `(ncol, nlev+1)`; the passing ZM regression suite is the only thing covering the real producer. The `.reshape(ncol, nlev+1)` is a no-op if ZM already returns columns (the adjacent `dT_dt.reshape(n_lat, n_lon, nlev)` implies it does), and the C-order flatten matches radiation's column construction. Consistent — but again, fake-tested only.
- The both-or-neither `ValueError` fires on static None-ness (fine under jit). Older `ConvectionOutput` producers: fields exist because hydrostatic already publishes; an *attribute-less* legacy class would `AttributeError` rather than the clean ValueError — check whether any such producer remains reachable on spectral.

## F. Scientific choices shipped, not asked

1. **First-window zero-cloud spin-up, re-paid every training rollout** (B-1). Must be asked.
2. **Training-gradient coupling radiation→CLUBB carry** now active (C). Correct, but it changes optimization dynamics vs. every existing spectral arm; no baseline exists to regress, so it's a choice to *state*, not defend.
3. **Partial mirror:** the hydrostatic READ side also routes `conv_precip` for the Slingo surrogate (`_conv_cloud_active`); the spectral side reads only `cloud_fraction`/`mass_flux_up`/`icwmr`. If cam6_clubb's CloudConfig keeps `convective_cloud` off (CAM6 uses the ZM deepcu formulation, not Slingo), the omission is consistent — but if any spectral deck can enable `convective_cloud` with clubb-cf, spectral silently diverges from hydrostatic. Verify and pin.
4. **Gray branch flag:** `use_clubb_cloud_fraction=(cloud_scheme == "cam6_clubb")` is added to a `RadiationConfig` that branch never gives a `cloud_scheme` (it defaults). So flag=True in a config that cannot be cam6_clubb — either inert (gray ignores the override) or it routes an override into the default cloud scheme. This is defensive symmetry without a consumer; I'd either pass `cloud_scheme` there or drop the flag from the gray branch.

## G. Ratchets — **weakly pinned.**

- The refusal-message change is matched only by `match="hydrostatic"`, which the *old* message also satisfied. The dispatch-hardening suite passing means nothing pinned the old lane list — fine — but nothing now pins that `spectral_pe` belongs. Strengthen the match to include `spectral_pe`.
- Membership is pinned positively (factory builds + marker asserted), and the producer-gate ValueError is pinned. Adequate floor.

---

## Ranked by decision impact

1. **B-2** — add the explicit `_cam6_cf_active and phys_state is None` raise + a test pinning it (this also converts your `spectral_amip_rollout` claim from assumption to contract). *Pre-merge.*
2. **B-1/F-1** — decide the first-gating-window zero-cloud spin-up for fresh training rollouts; at minimum document it, ideally seed `cloud_fraction` diagnostically in `seed_phys_state`.
3. **F-3** — confirm `convective_cloud` is unreachable for cam6 on spectral, or mirror the `conv_precip` read.
4. **E** — read the dispatch chain for `conv_out` fall-through; one real-ZM shape assertion.
5. **F-4** — gray-branch flag semantics.
6. **G** — strengthen the refusal-message pin.
7. **C memory** — note the saved-`phys_state` activation cost.

**SHIP** — conditional on item 1, which is a two-line change with a test. Everything else is a follow-up, not a redesign. The one thing that would flip me to HOLD is evidence that `compute_cloud_properties` does *not* raise for cam6-without-override; then B-2 stops being hardening and becomes the bug.

## Round 2

### codex

SHIP for the four changes. No new blocking findings; cold-start remains the owner’s decision.
- **Producer gate:** checks the resolved configs before either split physics builder. Replacing both builders with failure sentinels confirmed neither executes before rejection.
- **Missing carry:** raises during tracing inside real `jax.checkpoint`; `None` remains Python `None`, so no tracer-dependent branch. Gray revert is consistent with pre-wiring behavior.
- **Tests:** independently reproduced **9 passed** on this tree and **9 failed** using the identical tightened tests against `938855bde`, with package import paths verified.
Recommendation: **(b) for WB training**, because the initialization transient occupies half of every sample. Choose **(a)** if reproducing production’s cold-start procedure is the controlling requirement. The deciding factor is the intended training regime; any skill benefit from (b) remains unverified. No changes made.

### GLM

## Verdict: SHIP

All four changes hold up under attack. The two round-1 blockers are closed with fail-loud guards at the correct layers, and I found no new defect. Nits below are non-blocking.

### 1. Split-factory gate placement

The gate sits immediately after the `PhysicsConfig` assembly and **before both builds whose pairing it validates** (`make_physics(non_rad_cfg, ...)` and `make_radiation_physics(rad_cfg, ...)`). Config/dataclass construction is pure. The non-split path is independently covered by the combined.py gate, which fires at the very top of `_make_spectral_pe_combined`, before any `tagged_fns` construction. One honest caveat: the function body before line ~1628 isn't in the hunk; if anything between the `rad_cfg` flag derivation (~1345) and the gate does climatology I/O, a refusal wastes that work — a perf nit, not a correctness hole (no persistent state is corrupted). The gate's message matches combined.py's ("cloud-fraction-producing"), so the two entry points fail identically. The only residual uncovered path is a caller hand-building `make_radiation_physics(cam6_cfg, "spectral_pe", use_clubb_cloud_fraction=True)` next to a louis closure built separately — the builder can't see turbulence, same contract shape as the hydrostatic lane. Acceptable.

### 2. Trace-time firing of the missing-carry raise

Yes, it fires, and the reasoning is sound: `jax.checkpoint` does **not** defer the Python body — the wrapper traces *through* `_physics_fn_core` when invoked (eagerly or under jit), so `phys_state is None` and the `raise` execute during that trace. `None` passes through pytree handling as a genuine Python `None`, never a tracer; when the carry *is* present it arrives as a pytree of tracers, `is None` is False, and `getattr`/`.reshape` work on the reassembled container. The empirical proof is in the test design itself: `test_spectral_cam6_radiation_refuses_a_missing_carry` deliberately does **not** apply `_no_remat`, so the raise demonstrably penetrates the real checkpoint wrapper. Even better, in the rollout the stateless case is caught *earlier* than any trace: `_init_rad()` runs at Python level as the scan's init value, so a cam6 stateless caller dies eagerly with the named contract before the scan is built. Also verified the closure chain: `phys_state` captured by the `lax.cond` branch lambdas from the scan carry is a tracer pytree (no spurious raise), and the forwarding is conditional on signature inspection (`_rad_accepts`), so older signatures still bind.

### 3. Gray revert

Clean. The gray branch's `RadiationConfig` no longer sets the flag → default False → the split producer gate reads `rad_cfg.use_clubb_cloud_fraction` = False → inert; the dispatcher's `NotImplementedError` gate only fires when the flag is True; the gray builder path never consults `_cam6_cf_active`. No dangling consumer or producer left on that branch. Pre-wiring behavior restored exactly.

### 4. Tightened tests on pre-wiring

Both tightened tests are among the verified 7/7: the factory test dies at the first builder call (`NotImplementedError` on `spectral_pe` + flag), the bridge test dies on the missing `prog["conv_mass_flux_up"]` key. The `match="spectral_pe"` pin works as intended — dropping `spectral_pe` from the allowed tuple changes the message text and fails the regex. The two *new* round-2 tests are also non-vacuous by construction: pre-wiring, the split factory with cam6+louis builds successfully (flag defaulted False, no builder gate) so `pytest.raises` fails; and the missing-carry call proceeds into the backend instead of raising the named `ValueError`. The arange-pattern bridge test does prove ordering within the `ConvectionOutput` layout contract (bitwise `assert_array_equal` through the reshape); it can't catch a wrong *layout assumption* by the bridge itself, but that assumption is the output type's documented contract.

### Residual nits (non-blocking)

- The call-time raise is keyed on `_cam6_cf_active`, not on `use_clubb_cloud_fraction`. A hand-built `sundqvist + flag on + stateless caller` silently gets `cloud_fraction_override=None` instead of a raise. Unreachable via either factory (flag ⟺ cam6 there); consider widening to the flag someday.
- The "reshape defensively" reads would silently scramble a numel-compatible carry of the wrong shape rather than error. The `PhysicsState` column-form contract makes this theoretical; an explicit shape assert would be cleaner.
- Overlong `NotImplementedError` message line — style only.

### (a) vs (b) — recommendation for the owner

**(a), accept the transient.** The deciding factor is **train/inference symmetry under the WB scoring protocol**: the training target is the production AMIP physics, and the production coupled driver itself cold-starts from the same zero-init carry ("real zero-init cloud_fraction here"). Option (a) makes the WB lane bit-for-bit faithful to the system being emulated *and* guarantees the transient appears identically at train time and at any 6h-rollout inference time — the errors cancel. Option (b) is a forcing sequence that exists nowhere in production, applied to *every* sample, and its own validity is unvalidated (a single diagnostic CLUBB/ZM pass on a raw IC that has seen no convection/microphysics is not the equilibrium carry either) — plus a per-sample physics call inside the trace. Revisit (b) only if the lane is later scored on long continuous rollouts, where a first-window-only transient becomes train-only noise. In the meantime, I'd spend the measurement budget not on changing the code but on bounding the effect once: window-1 all-sky vs clear-sky heating delta under zero-cf, so the acceptance is quantified rather than assumed.

## Round 3

### codex

**HOLD — the warm-up advances prognostic memory beyond filling radiation’s inputs.**
- **A/F — blocking:** [Retaining the entire warmed carry](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/packages/ml/legoesm/training/neural_gcm_spectral.py:1929) produces exactly the carry old step 0 would publish. Step 0 then advances that memory again on the same pre-step atmosphere. This affects CLUBB moments, convection profiles, and—when enabled—GWD spectra and stochastic keys. Its tendencies also use the warmed memory. That extra physics spin-up is a scientific choice beyond “fill the carry first.” Publish only `cloud_fraction`, `conv_mass_flux_up`, and `conv_icwmr`; preserve the remaining seeded fields. The carry matches old step 1’s incoming carry, **not its radiation result**, which uses a different atmosphere and may skip refresh.
- **B/C/D:** The checkpoint is effective by installed JAX/Equinox source: captured traced parameters participate in rematerialization; it does not add a full unrecomputed physics tape. `prevent_cse=True` can cost optimization/performance, but presents no identified correctness problem. The split factory forwards the marker correctly, including WeatherBench evaluation. However, combined non-split execution bypasses this warm-up; hand-built wrappers can lose the marker. An explicitly supplied fresh seed also skips it: `None` distinguishes internally initialized from supplied state, not fresh from resumed state. Document/enforce that initialization contract.
- **E — measured:** **10 tests passed** against this worktree. Removing the warm-up in memory makes the new test fail with `reading rad saw the seed`; the unmarked control passes unchanged. But [the assertion checks CLUBB moments](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/tests/unit/test_spectral_cam6_cloud_routing.py:386), not radiation’s three inputs—it requires the unwanted side effect. Replace it with checks on those inputs, preservation of other carry fields, and chained-carry skipping.
UNVERIFIED: full-rollout gradient execution and peak memory. No files changed.

### GLM

## VERDICT: SHIP

The warm-up is correctly gated, correctly checkpointed, semantically coherent (it extends the lag convention to k=0 with a virtual physics pass at t=0), and the tests are non-vacuous in both directions. Residual items below are nameable, not blocking. Item-by-item:

## A — Warm-up semantics and the double-advance side effect

**What the warm-up publishes.** `_ps_entry(initial_state, …, seed)[1]` = `F(S₀, seed)`. That is *exactly* what the old code's step-0 entry produced (`C₁_old = F(S₀, seed)`), so yes — step 0's radiation now reads what old-code step 1's radiation would have read (as a functional of S₀; the trajectories diverge after step 0 because step 0's radiation differs, which is the point). This is the standard "t=0 diagnostic call" convention and is the natural reading of "fill the carry first." Intended.

**The side effect is real but bounded.** The new carry chain is `C₀ = F(S₀, seed)`, `C₁ = F(S₀, F(S₀, seed))`, then `C_{k+1} = F(S_k, C_k)` as before. So every carry from step 1 on is one `F`-application deeper than the old chain, applied at the *initial* state:

- **Pure diagnostics** (cloud_fraction, mass_flux_up, icwmr, and anything state-dominated): re-applying F at the same state is essentially idempotent — no issue.
- **Prognostic CLUBB moments**: integrated twice at S₀ while the atmosphere advances once. This is a genuine double-advance, but it is *in the spin-up direction* (moment relaxation toward the S₀ equilibrium), the seed already prefers shear-equilibrium TKE, and it decays over the window. Benign.
- **prng_key**: if `_ps_entry` splits the key (stochastic physics), the stream shifts by one draw. A stream offset, not a correctness issue — but it is unnamed in the comment.
- **The atmosphere is untouched**: the lambda keeps `[1]` only, so no tendency is ever double-applied to T/u/v/tracers. This is the load-bearing safety property and it holds by construction.

**Must it publish only the three read fields?** No, and it shouldn't. Selective publication would (a) hardcode CAM6 field names into the generic rollout — bad layering, and (b) produce a chimera carry (seed moments + filled diagnostics from different entries) that step 0's CLUBB entry then evolves incoherently. The full entry is self-consistent and generic. Accept the side effect; it is one extra physics entry at t=0, which is standard practice. The comment already declares the deliberate departure; add one sentence naming the moment/RNG side effect.

## B — Gradient path

`nothing_saveable` means the checkpoint saves zero intermediates and **recomputes** the entry in backward — so no un-remat'd physics tape is added. Cost is one extra physics forward (trace time) plus one extra recompute per backward pass, outside the scan, i.e. O(1) versus n_steps in-scan entries. Symmetric with `step_fn_ckpt`'s own treatment. `prevent_cse=True` is the *safe* direction here (omitting it under remat is what causes the classic CSE/backward-differential artifacts); its only cost is disallowed sharing. One noteworthy improvement: the first window's cloud fraction previously had **zero gradient** w.r.t. CLUBB parameters (seed carry, no param dependence); the warm-up now threads a correct gradient path through it. That is a training-signal change — an improvement, but name it under F.

## C — Marker coverage

- **Raw builder output**: marked when routing is on (`_make_spectral_pe_radiation` sets it) — hand-built callers using it are covered, and its signature accepts `phys_state`.
- **Split wrapper**: covered *by this round's fix #2*; the factory test asserts marker-on-wrapper iff cam6, so removing the re-export fails the suite.
- **Combined path**: the passing dispatcher test proves the marker is set on the combined-built radiation fn (forwarding is gated on it).
- **Eval**: same factory → same marker → warm-up applies identically. GLM's symmetry objection is answered.
- **Residual**: a hand-written rad fn that reads `phys_state` but doesn't set the marker silently gets the seed. That is a caller contract violation the library cannot defend against; crucially the *dangerous* variant — a real cam6 fn reached with no carry at all — raises by design. Also: the non-split combined arm has no rollout-level `rad_physics_fn`, so it gets no warm-up — but there radiation runs every step inside combined, so the transient is one step (900 s), not a 3 h window, same class as a production cold start. Out of the owner's decided scope (the split training lane); name it.

## D — The `phys_state_in is None` discriminator

Correct. The rollout cannot distinguish "previous segment's carry" from "freshly minted seed" — zero-sniffing heuristics are fragile, an explicit flag is API bloat, and "explicit carry = caller authority over the initial carry" is the standard opt-out semantic. The loss case (fresh caller manually passing a zero seed with cam6 silently gets the transient back) exists but requires opting out of the mechanism the documented protocol provides. **Nit**: the skip branch itself is untested — the test comment explicitly declines to test it. Two lines (`phys_state_in=seed` + reading rad → assert moments == seed at first call) would close it.

## E — Non-vacuity

- **Removing the warm-up**: the reading case's first radiation call sees the seed → `allclose(m_first, m0)` True → `(not True) is True` fails. ✓
- **Gate leakage** (warm-up runs unmarked): the non-reading control sees filled moments → fails. ✓ The control genuinely proves the marker half of the gate (the `phys_state_in` half is the untested branch from D).
- **Removing fix #2**: `bool(getattr(rad, "_wants_phys_state_ro", False)) is expect` fails for cam6. ✓
- The shear-seed separation premise is deterministic (fixed state, fixed JAX op sequence), so not flaky. The suite's "10 passed" is consistent with the file's 10 tests.
- Minor: the test proxies "carry filled" via `clubb_moments` rather than the actual read fields; acceptable compositionally, since the factory/dispatcher tests pin that radiation reads `cloud_fraction` from the carry.

## F — Scientific choices beyond the owner's decision (named)

1. **Full-carry warm-up** (all state, incl. prognostic moments and RNG stream, advanced one entry at t=0) rather than surgical fill of the three read fields — accepted per A, but it *is* a choice.
2. **Arm asymmetry**: split training/eval lane warms up; non-split combined arm keeps a one-step transient; `spectral_amip_rollout` cannot run cam6 at all. No train/inference asymmetry is introduced (both get the warm-up), but the lane-to-lane story should stay documented.
3. **First-window cloud fraction = single-pass value from seed moments**, not a spun-up equilibrium — the owner's override of GLM's objection, now deliberate; not re-litigated.
4. **New gradient path** through first-window cloud fraction (B) — a change to the training signal, in the correct direction.

## Residual follow-ups (non-blocking)

- Add the skip-branch assertion (D).
- One comment sentence naming the moment/RNG side effect (A) — the "deliberate departure" paragraph is the right home.
- Pre-existing, out of scope: the init radiation at step 0 is recomputed by the first `_rad_refresh` when `should_refresh(0)` is true — one duplicated radiation call per rollout, unchanged by this round.

**SHIP.**

## Round 4 (final form)

### codex

SHIP.
- A: Decision recorded in rollout comment and test docstring; three-field mentions describe only the rejected alternative.
- B: Supersaturated fixture produces nonzero cloud; identical tracer inputs isolate the warm-up’s moment advance. The first-call 0.123 tag discriminates the chained skip.
- C: No new debris, stale implementation comment, or unused import found. Worktree-pinned CPU run: **10 passed**; diff check clean.

### GLM

**A. Decision recorded — yes, findable, nothing stale.**
The rollout comment block carries all four elements a future reader needs: the decision + date, the transient it prevents (3 h of a 6 h sample cloud-blind at cadence 6), the rejected three-field alternative *with its reason* ("a cloud fraction diagnosed from moments the carry then does not hold"), and the cost sentence GLM asked for (one extra relaxation of every prognostic carry + one stochastic draw, atmosphere unmoved). The test docstring records it too. The three-field variant survives only as the named rejected alternative — correct. The one stale comment that existed (combined.py's "CLUBB-cf routing is hydrostatic-only today") is replaced with text that matches the new code, and the `NotImplementedError` message now lists `spectral_pe`. `init_rad_tendency` is fully gone (definition and both scan-init sites converted to `_init_rad`), so no dangling references.

**B. The test physics holds up.**
- *Supersaturation:* at 280 K, e_s ≈ 9.9 hPa → w_s ≈ 6.2e-3 at the surface, rising to ~0.02 only near ~300 hPa. q_v = 0.02 is ~3× saturation through the lower half of the column. CLUBB's PDF cloud fraction with a supersaturated mean and the seed's finite variance is strictly positive in one pass; `np.any(cf > 0)` at the recorded first call is satisfied for a reason caused by the pass. (And it empirically passed.)
- *Attribution:* the `moments != seed` assertion is clean because the paired non-reading arm runs the **identical** state, physics, and seed and asserts `moments == seed` at the same record instant. The record fires inside `_init_rad(phys0)` — before the scan — so the per-step harvest cannot contaminate either arm. `_with_prescribed_sfc` is common to both arms and provably doesn't touch moments (the non-reading equality pins that). The tracer value therefore cannot be the cause; only the marker-gated warm-up differs.
- *Chained test:* exact array equality against a uniform 0.123 is discriminating in both directions — a leaky gate would let the CLUBB-diagnosed fraction overwrite it, and a dropped carry would give the diagnosed value or None. The `conv_mass_flux_up == seed` assertion in both arms additionally pins that the warm-up doesn't fabricate deepcu inputs in a config with no convection scheme. Genuine.

**C. Sweep of the full diff vs 938855bde.**
No debris found. Specifically: no unused imports in the test file; the 10-passing count reconciles exactly with the file's contents (9 functions, one parametrized ×2) — nothing silently missing from the round-3 truncation; the three gate sites (combined, split factory, radiation builder) use consistent messages and the tests match on the shared phrase; the `_wants_phys_state_ro` re-export on the split `rad_fn` wrapper covers the seam the rollout actually keys on; `_call_rad` omits the kwarg when `phys_state is None`, so markerless/stateless arms bind old signatures; the warm-up uses the same `_ps_entry` the per-step harvest uses, under the same `nothing_saveable` remat — self-consistent by construction. Non-blocking observations, none new in this round: (i) the combined producer gate fires even when `radiation.scheme == "none"` (flag moot) — strictness matching the hydrostatic gate; (ii) flag-on + non-cam6 cloud scheme with no forwarded carry degrades silently to the RH path (only cam6 refuses loudly) — same convention as hydrostatic; (iii) the owner-decision date appears twice in one docstring — cosmetic.

**D. SHIP.** The delta implements the owner's full-pass decision exactly as scoped, the gate is double-tested in both directions with clean controls, and the record-keeping survives a reader who arrives cold.
