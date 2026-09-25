"""The ``PhysicsOutput`` tendency pytree — shared by the physics pipeline and
the atmosphere physics components (e.g. neural physics).

Lives in ``core`` (the shared substrate, importing nothing above it) so a
component can produce/consume it without importing the driver — the
components-must-not-import-driver boundary.  ``driver.physics_pipeline``
re-imports it (it is the pipeline that produces a ``PhysicsOutput``).
"""

from __future__ import annotations

from typing import NamedTuple

import jax


class PhysicsOutput(NamedTuple):
    """Output from a single physics step.

    ``conv_prog`` / ``tke`` / ``qke`` / ``gwd_spectrum`` are UPDATED
    prognostic carries (not tendencies): the caller feeds them back on
    the next step (issue #413).  The three stateful slots default to
    ``None`` — populated only when the corresponding scheme is active
    and the pipeline threads its carry, so legacy 19-field consumers
    are unaffected.
    """
    dT_dt: jax.Array
    dq_v_dt: jax.Array
    dq_c_dt: jax.Array
    dq_r_dt: jax.Array
    precip: jax.Array
    sw_net_sfc: jax.Array
    lw_net_sfc: jax.Array
    sw_up_toa: jax.Array
    lw_up_toa: jax.Array
    sw_down_toa: jax.Array
    du_dt: jax.Array
    dv_dt: jax.Array
    dq_i_dt: jax.Array
    dq_s_dt: jax.Array
    dq_g_dt: jax.Array
    dN_c_dt: jax.Array
    dN_r_dt: jax.Array
    dN_i_dt: jax.Array
    conv_prog: jax.Array
    shflx: jax.Array | None = None   # surface sensible heat flux [W/m2]
    lhflx: jax.Array | None = None   # surface latent heat flux [W/m2]
    tke: jax.Array | None = None     # updated prognostic TKE (ncol, nlev)
    qke: jax.Array | None = None     # updated MYNN-2.5 qke=2*TKE (ncol, nlev)
    gwd_spectrum: jax.Array | None = None  # updated GWD wave-action spectrum
    w_land: jax.Array | None = None  # updated slab-land soil water [kg/m2]
    snow: jax.Array | None = None    # updated slab-land snow water equiv. [kg/m2]
    land_runoff: jax.Array | None = None  # diagnosed land runoff (Hortonian + Dunne) [kg/m2/s]
    # Diagnostic sub-grid LIQUID cloud fraction (ncol, nlev) from a moist
    # higher-order closure's PDF (diagnostic CLUBB).  Carried out of the physics
    # step and back in to compute_radiation_core so radiation can use it instead
    # of the RH grid-scale fraction (use_clubb_cloud_fraction).  None for schemes
    # with no PDF cloud closure => radiation keeps the grid-scale cloud path.
    cloud_fraction: jax.Array | None = None
    # Per-process column budget ledger (diagnostics.process_ledger): the
    # (N_LEDGER, 2) [water kg/m²/s, dry-enthalpy W/m²] global-mean rates of
    # the PHYSICS rows (turbulence/convection/microphysics/radiation/
    # other_physics; the clips/dynamics rows are filled by the segment
    # driver).  ``None`` (default — the static ``budget_ledger`` gate off)
    # keeps the pytree byte-identical to the pre-ledger output.  Appended
    # LAST to preserve positional-construction ABI.
    budget_ledger: jax.Array | None = None
    # Per-column MG2-style CFL sedimentation sub-step count the microphysics
    # REQUIRED this step (``MicrophysicsOutput.sed_substeps_required``); a
    # value above the configured cap means the fall was clamped.  ``None``
    # (sub-stepping off, or a scheme that does not sediment) keeps the pytree
    # byte-identical.  Carried so the finite-volume lanes can report the
    # clamp the way the MPAS loop does.  Appended LAST (positional ABI).
    sed_substeps_required: jax.Array | None = None
