"""3D Monte-Carlo ray-tracing radiation for plane LES/CRM (``scheme="mc3d"``).

Pure-JAX port of the microhh ``rte-rrtmgp-cpp`` forward ray tracer. See
``docs/specs/mc3d_raytracer.md``. Phase 1: shortwave spatial solver + tallies.

ORACLE STATUS (tier-2 audit, 2026-10-19).  The sampling kernels -- Mie angle,
free path (sample_tau and the step floor), Rayleigh, Henyey-Greenstein for
1e-3 <= |g| <= 1 - 1e-6 -- and the null-collision majorant grid match the
running microhh device functions (v0.3, fc291af0, fp64) to round-off:
tests/atmosphere/test_mc3d_kernels_oracle.py.  NOT YET FAITHFUL:
the photon transport itself (analog single-deposit estimator vs microhh's
weighted photons with Russian roulette; no diffuse top-of-domain incidence;
photons at the iteration cap counted as escaped), which has not been compared
with the CUDA tracer, and the Henyey-Greenstein guards the oracle does not have: the |g| < 1e-3
isotropic branch (mu = 1 - 2u where the oracle formula tends to 2u - 1) and
the clip of |g| to 1 - 1e-6.
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
