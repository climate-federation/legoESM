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
    # run_coupled shares run_amip's parser (composable physics P5a), so it
    # carries exactly run_amip's two protective exclusions above.
    ("run_coupled", "microphysics", "ml_emulator"):
        "Shared parser with run_amip (P5a): same untrained-network exclusion.",
    ("run_coupled", "clouds", "resolved"):
        "Shared parser with run_amip (P5a): same CRM-only-convention exclusion.",
    ("run_coupled", "surface_bulk_scheme", "none"):
        "No surface bulk fluxes means no air-sea exchange, so nothing drives "
        "the ocean; run_coupled refuses it (test_surface_bulk_none).",
}
