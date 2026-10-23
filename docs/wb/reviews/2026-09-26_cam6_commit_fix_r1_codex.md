# codex review, CAM6 commit fix, round 1

VERDICT HOLD

1. **[P2] Returned-carry checks still permit broken implementations** — `tests/unit/test_spectral_macmic_subcycle.py:155–178`. Confirmed by executing the assertion helper: `clubb_moments=None` passes against nonempty reference moments because `jax.tree.leaves(None)` makes the `zip` empty. The diagnostic N=1 branch also returns without checking any carry. Assert matching tree structures before comparing leaves, and check diagnostic carry too.

2. **[P2] Sequential reference mishandles per-call inputs** — `tests/unit/test_spectral_macmic_subcycle.py:125,140`. `update_physics_state` clears `surface_wth_override` and `surface_wqv_override`; production deliberately preserves these during intermediate sub-steps. The fixture initializes neither, so removing production’s preservation would pass this reference comparison. Seed nonzero overrides, preserve them explicitly in the reference, and verify every sub-step receives them while the final returned carry clears them.

The saline guard and raw count forwarding look correct. Full test suite and claimed seven mutations were not rerun.
