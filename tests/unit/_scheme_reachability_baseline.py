"""Shrink-only baseline for the SCHEME reachability audit.

Every entry is a column-physics scheme that ``ExperimentConfig.validate_strict``
ACCEPTS but that a production run driver does NOT offer on its CLI -- i.e. a
parameterization the model implements that a user cannot select there.  The
audit asserts the computed unreachable set EQUALS this baseline: wiring a scheme
through goes red until the baseline shrinks, and a NEW unreachable scheme goes
red until it is either wired or consciously listed here with a reason.

This is the STRING-scheme complement of ``_params_reachability_baseline.py``.
That one covers ``--params``, which is ``:float``-only BY DESIGN (see
CLAUDE.md / [[loop-counts-never-trainable]]: only float fields are spec-eligible,
so an iteration count can never reach the trainable collector). The consequence
is that no string scheme field has ANY generic override path -- a scheme is
reachable only if some driver spells out an explicit flag. That is precisely why
these gaps cluster on the scheme axes, and why they need their own ratchet.

Each entry MUST carry a reason. "Narrower" is not automatically drift: a driver
can legitimately decline a scheme that does not FUNCTION in its configuration,
and those exclusions are protective. The bar for adding an entry is that the
exclusion is deliberate and explained -- not that it is merely current.
"""

# (driver, axis, scheme) -> reason
UNREACHABLE_SCHEMES = {
    # --- protective exclusions: the scheme does not function in this driver ---
    ("run_amip", "microphysics", "ml_emulator"):
        "Returns an UNTRAINED network's ~zero tendencies ('suitable for "
        "testing' per its docstring) and run_amip exposes no checkpoint flag, "
        "while _require_full_physics_for_amip would count it ACTIVE -- i.e. "
        "silently disabled microphysics that PASSES the full-physics guard. "
        "Excluding it is protective. Wiring it needs a checkpoint flag first.",
    ("run_amip", "clouds", "resolved"):
        "SAM's CRM convention (cloud fraction = 1 wherever condensate > 0); "
        "raises without explicit q_cloud/q_ice from microphysics. Not a C48 "
        "GCM scheme -- it belongs to the plane/CRM lane, not AMIP.",
    # NOTE: run_coupled DOES offer --clouds resolved and --microphysics
    # ml_emulator. An earlier draft of this baseline assumed it mirrored
    # run_amip's exclusions; the self-test below caught the guess. Whether
    # run_coupled SHOULD offer them is a separate physics question (it has no
    # _require_full_physics_for_amip guard for ml_emulator to fool), tracked
    # rather than silently changed here.

    # --- verified gap, no legitimate use in this driver ---
    ("run_coupled", "clouds", "cam6_clubb"):
        "CAM6 cloud fraction on the CLUBB path is wired on the MPAS combined-"
        "physics lane only (validate_strict requires discretization='mpas'); "
        "run_coupled drives the FV pipeline, where the cloud call threads "
        "neither lat nor the deep-convection carries, so the scheme cannot "
        "function there. Offer it once the pipeline threads them.",
    ("run_coupled", "radiation", "none"):
        "run_coupled has NO --held-suarez-forcing lane (verified), so a "
        "coupled run with radiation off has no energy source at all -- it "
        "would silently produce garbage. run_amip DOES offer 'none' because "
        "its --held-suarez-forcing lane is a dry core and HS forcing is "
        "ADDITIVE to the physics pipeline rather than a replacement.",
}
