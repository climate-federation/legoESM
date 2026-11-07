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
    # ("run_amip", "clouds", "resolved") REMOVED (vote VC31): run_amip now
    # offers --clouds resolved, restricted to --microphysics ifs_cloudsc
    # (CLOUDSC's advected cloud fraction + grid-mean condensate); every other
    # microphysics is refused in run_amip's own argument check.
    ("run_coupled", "radiation", "none"):
        "With radiation off a coupled run gets no solar heating of the "
        "surface or the ocean, so the coupled climate has no energy source; "
        "main's run_coupled refused it and the shared parser (P5a) keeps "
        "that refusal.",
    ("run_coupled", "surface_bulk_scheme", "none"):
        "No surface bulk fluxes means no air-sea exchange, so nothing drives "
        "the ocean; run_coupled refuses it (test_surface_bulk_none).",
}
