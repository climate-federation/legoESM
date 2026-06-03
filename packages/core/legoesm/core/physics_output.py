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
    """Output from a single physics step."""
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
