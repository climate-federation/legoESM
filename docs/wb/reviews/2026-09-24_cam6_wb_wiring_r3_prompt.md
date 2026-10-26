ROUND 3 re-review, spectral cam6_clubb READ-side wiring, worktree
/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6 (uncommitted on 938855bde).
Rounds 1-2: both reviewers SHIP on the wiring (producer gate, missing-carry
raise, gray revert, tightened tests). The ONE open item was the cold-start
zero cloud-fraction carry in the first radiation window. Reviewers disagreed on
the remedy: codex recommended filling the carry, GLM recommended accepting the
transient (train/inference symmetry; a single pass is not the equilibrium
carry either). OWNER DECIDED 2026-09-24: "Fill the carry first."

THE NEW CHANGE (only this is under review now):
 1. neural_gcm_spectral.py spectral_rollout, stateful branch, after phys0 is
    seeded and anchored and BEFORE the deferred first radiation call:
      if phys_state_in is None and getattr(rad_physics_fn, "_wants_phys_state_ro", False):
          phys0 = jax.checkpoint(lambda ps: _ps_entry(initial_state, grid, sigma_coord, ps)[1],
                                 prevent_cse=True, policy=nothing_saveable)(phys0)
    i.e. ONE non-radiative physics pass on the initial state publishes CLUBB's
    cloud fraction and ZM's mass flux / in-cloud water into the carry before
    radiation first reads it. Gated (a) to a FRESH seed only -- a chained
    caller's carry already holds the previous segment's values -- and (b) to
    a radiation fn that advertises reading the carry, so every other stateful
    arm is byte-identical.
 2. aimip_params.py: the split factory re-exports `_wants_phys_state_ro` from
    the raw radiation builder onto the rad_fn wrapper the rollout receives
    (the marker was on the raw fn, which the rollout never sees).
 3. Tests: the factory test asserts the marker is on the wrapper iff cam6; a
    new rollout test uses the sheared-wind setup of the CLUBB carry test and
    asserts the FIRST radiation call sees moments != seed for a marked rad fn
    and == seed for an unmarked one (fresh-seed path, no phys_state_in).

GLM's symmetry objection, answered with evidence: the WB evaluator
(scripts/validate/run_weatherbench_eval.py) goes through
build_mode_components -> the same make_run_seg(...).raw -> spectral_rollout
with the same marked rad fn, so the warm-up applies identically at scoring.

ATTACK:
 A. Is the warm-up's carry the carry the FIRST STEP would have published
    anyway (i.e. _ps_entry on the pre-step state)? If so, step 0's radiation
    now reads exactly what step 1's radiation would have read under the old
    code -- is that the intended semantics, and does anything else in the
    carry (CLUBB moments, Bechtold profile, GWD spectrum, prng_key) advance
    TWICE relative to the atmosphere as a side effect? Is that side effect
    acceptable or must the warm-up publish ONLY the radiation-read fields
    (cloud_fraction, conv_mass_flux_up, conv_icwmr) and keep the rest at the
    seed?
 B. Gradient path: the warm-up is inside the traced graph and checkpointed
    nothing_saveable. Under eqx.filter_value_and_grad, does it add one
    un-remat'd physics tape, or is the checkpoint effective? Any hazard with
    prevent_cse=True here?
 C. The gate reads rad_physics_fn._wants_phys_state_ro. Any path where the
    rollout receives a rad fn that READS the carry but lacks the marker (the
    combined non-split path; a hand-built caller)? Then the transient
    silently returns.
 D. The chained-caller skip: is `phys_state_in is None` the right discriminator,
    or can a fresh caller pass a seed explicitly and lose the warm-up?
 E. Non-vacuity of the new test: does it really fail if the warm-up is
    removed, and does the unmarked control really prove the gate?
 F. Anything here that is a scientific choice BEYOND the owner's "fill the
    carry first"? Name it.
SHIP or HOLD.
