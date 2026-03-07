"""Component coupler for legoESM."""

# Low-level types that don't depend on land/ice (no cycle)
from legoesm.coupler.config import CouplerConfig, TileConfig
from legoesm.coupler.coupling_fields import AtmToSurface, SurfaceToAtm, TileResponse
from legoesm.coupler.tile_fractions import TileFractions, compute_tile_fractions, blend_tiles
from legoesm.coupler.accumulator import (
    FluxAccumulator, accumulate, mean_accumulator, reset_accumulator,
    accumulator_from_flux,
)
from legoesm.coupler.surface_exchange import (
    extract_atm_to_surface, extract_atm_to_surface_nh,
)
from legoesm.coupler.bulk_flux import compute_most_fluxes, psi_m, psi_h
from legoesm.coupler.lake import LakeConfig, LakeState, step_lake


def __getattr__(name):
    """Lazy imports to avoid circular dependency with land/ice modules."""
    if name in ("SurfaceState", "init_surface_state", "make_coupler",
                "ocean_tile_response"):
        from legoesm.coupler.coupler import (
            SurfaceState, init_surface_state, make_coupler, ocean_tile_response,
        )
        _map = {
            "SurfaceState": SurfaceState,
            "init_surface_state": init_surface_state,
            "make_coupler": make_coupler,
            "ocean_tile_response": ocean_tile_response,
        }
        return _map[name]
    raise AttributeError(f"module 'legoesm.coupler' has no attribute {name!r}")


__all__ = [
    "CouplerConfig", "TileConfig",
    "AtmToSurface", "SurfaceToAtm", "TileResponse",
    "TileFractions", "compute_tile_fractions", "blend_tiles",
    "FluxAccumulator", "accumulate", "mean_accumulator", "reset_accumulator",
    "accumulator_from_flux",
    "extract_atm_to_surface", "extract_atm_to_surface_nh",
    "compute_most_fluxes", "psi_m", "psi_h",
    "SurfaceState", "init_surface_state", "make_coupler", "ocean_tile_response",
    "LakeConfig", "LakeState", "step_lake",
]
