ROUND 4 (final form) re-review, spectral cam6_clubb READ-side wiring, worktree
/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6 (uncommitted on 938855bde).

HISTORY. Rounds 1-2: both SHIP on the wiring. Round 3 on the carry warm-up
SPLIT: codex HOLD ("publish only the three radiation-read fields; the full
pass is an unasked spin-up of prognostic memory"), GLM SHIP ("the full pass is
self-consistent; a three-field fill is a chimera carry, and hardcodes CAM6
field names into the generic rollout"). Both named it a choice beyond the
owner's "fill the carry first". It was put to the OWNER with both arguments.
OWNER DECISION 2026-09-24: FULL PASS.

WHAT IS UNDER REVIEW NOW (only the delta since round 3):
 1. neural_gcm_spectral.py: the warm-up keeps the WHOLE carry from one
    non-radiative physics pass on the initial state (the three-field variant,
    which was implemented and tested green in between, is gone). The comment
    records the decision, the rejected alternative, and the side effect (one
    extra relaxation step of every prognostic carry and one extra stochastic
    draw, on an atmosphere that has not moved). Gating unchanged: fresh seed
    only, and only for a radiation fn marked _wants_phys_state_ro.
 2. tests: the warm-up test now uses a SUPERSATURATED column (q_v = 0.02) so
    CLUBB diagnoses a non-zero PDF cloud fraction in one pass; asserts for the
    reading arm cloud_fraction > 0 somewhere AND moments != seed (whole carry
    advanced), for the non-reading arm cloud_fraction == 0 AND moments == seed
    (gate held), conv_mass_flux_up == seed in both (no convection scheme in
    that config), and a chained-carry arm (phys_state_in supplied with a tagged
    cloud_fraction 0.123) reaching radiation UNCHANGED (the skip). The three
    factory tests truncated by an editing error in round 3 are restored, with
    the gate test's helper now building the trainable set for the SAME closure
    it requests (it previously tripped an earlier, unrelated refusal).

FOR THE RECORD, not to relitigate: codex's round-3 objection stands as a named
cost in the code comment; GLM's suggestion to add one sentence naming the
moment/RNG side effect is done.

ATTACK, briefly -- this is the final gate before commit:
 A. Is the decision recorded where a future reader will find it (the rollout
    comment, the test docstring)? Anything still describing the three-field
    variant?
 B. Test: does q_v = 0.02 at 280 K actually supersaturate enough levels for a
    non-zero CLUBB fraction on the first pass, and does the assertion `moments
    != seed` hold for a reason connected to the pass (not to the tracer
    change)? Is the chained-carry skip test genuinely discriminating?
 C. Anything in the whole diff (git diff vs 938855bde) that the four rounds
    have not seen: leftover debris, a stale comment, an import no longer used.
 D. One line: SHIP or HOLD. If HOLD, the single thing that must change.
