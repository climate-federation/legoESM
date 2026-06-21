# MPAS run loop does not carry stateful physics across steps

**Status:** open (latent). **Found:** CLUBB port iter 83 (2026-06), codex
adversarial review of an attempted `--clubb-prognostic` AMIP-CLI exposure
(change reverted). **Affects:** `model_driver._run_mpas`.

## Problem

The MPAS run loop starts `_phys_state = None`, and
`update_physics_state(None, ...)` returns `None`
(`physics_state.py`, `update_physics_state` early-return), so ANY
stateful-physics carry — turbulence TKE/qke, prognostic convection profiles,
prognostic-spectral GWD spectrum, prognostic-CLUBB `clubb_moments` — is
DISCARDED and RESEEDED every step. A prognostic scheme run through this loop
"succeeds" but silently is not prognostic.

Additionally, MPAS checkpoints persist only `u/T/p_s/phis/tracers`, so even
with an in-loop carry, restarts would reset the physics state.

Currently masked because MPAS AMIP runs use `--turbulence none`.

## Fix sketch (human-directed PR; run-loop lifecycle + checkpoint-format surgery)

1. Initialize a real `PhysicsState` before the MPAS loop whenever stateful
   physics is active (use the shared `turbulence_scheme_traits` /
   `convection_scheme_traits` predicates).
2. Thread the carry through the loop (the direct `combined.make_physics`
   MPAS path already does this and is integration-tested — see
   `tests/unit/test_clubb_scheme.py` MPAS-driver tests).
3. Persist the carry (incl. `clubb_moments`, `(ncol, 15, nzm)`) in the MPAS
   checkpoint format and restore on restart.

Until then: prognostic schemes on MPAS are runnable/tested via the direct
`combined.make_physics` path; production-CLI exposure stays blocked (the
reverted iter-83 change shows the wiring that would otherwise be needed).
