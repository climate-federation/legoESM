"""3D Monte-Carlo ray-tracing radiation for plane LES/CRM (``scheme="mc3d"``).

Pure-JAX port of the microhh ``rte-rrtmgp-cpp`` forward ray tracer. See
``docs/specs/mc3d_raytracer.md``. Phase 1: shortwave spatial solver + tallies.
"""

from legoesm.atmosphere.physics.radiation.mc3d.config import MC3DRadiationConfig
from legoesm.atmosphere.physics.radiation.mc3d.knull_grid import (
    MajorantGrid,
    build_majorant_grid,
)
from legoesm.atmosphere.physics.radiation.mc3d.photon_walk import (
    PlaneRTGeometry,
    trace_batch,
    trace_one,
)
from legoesm.atmosphere.physics.radiation.mc3d.parallel import (
    pmean_result,
    shard_keys,
    solve_sw_sharded,
)
from legoesm.atmosphere.physics.radiation.mc3d.raytracer_lw import (
    LWResult,
    solve_lw_monochromatic,
    solve_lw_spectral,
)
from legoesm.atmosphere.physics.radiation.mc3d.raytracer_sw import (
    solve_sw_monochromatic,
)
from legoesm.atmosphere.physics.radiation.mc3d.tally import MC3DResult, tally_batch

__all__ = [
    "LWResult",
    "MC3DRadiationConfig",
    "MC3DResult",
    "MajorantGrid",
    "PlaneRTGeometry",
    "build_majorant_grid",
    "pmean_result",
    "shard_keys",
    "solve_lw_monochromatic",
    "solve_lw_spectral",
    "solve_sw_monochromatic",
    "solve_sw_sharded",
    "tally_batch",
    "trace_batch",
    "trace_one",
]
