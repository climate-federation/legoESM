"""Grid implementations for legoESM."""

from legoesm.grids.protocol import GridProtocol, VerticalCoordProtocol
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.cubed_sphere_cdgrid import (
    CubedSphereCDGrid,
    create_cubed_sphere_cdgrid,
)
from legoesm.grids.halo import pad_halo, pad_halo_vector
from legoesm.grids.gaussian import GaussianGrid, create_gaussian_grid
from legoesm.grids.latlon import (
    LatLonGrid,
    create_latlon_grid,
    LatLonCGridGeometry,
    FoldDescriptor,
    create_latlon_geometry,
    create_beta_plane_cgrid_geometry,
    ensure_geometry,
    create_mercator_grid,
    create_regional_latlon_grid,
    create_stretched_latlon_grid,
)
from legoesm.grids.nesting import (
    NestedLatLonGrid,
    BoundaryInterpWeights,
    create_nested_latlon_grid,
    apply_boundary_interp,
)
# Experimental, staged-not-integrated. Plane NH dycore lands in a
# follow-up PR (CRM rollout, PR2). For now PlaneGrid is consumed only
# by ``plane_operators`` and its direct unit tests.
from legoesm.grids.plane import PlaneGrid, create_plane_grid
from legoesm.grids.capability import (
    instantiate as instantiate_grid,
    capability_matrix,
    supported_extents,
    operator_family,
    validate_runtime,
    available_precision_modes,
    available_integrators,
    ARCHITECTURES,
    COMPONENTS,
    component_complexities,
    validate_complexity,
)
from legoesm.grids.tripole import (
    create_tripole_grid,
    create_synthetic_tripole,
)
from legoesm.grids.vertical import (
    SigmaCoordinate,
    create_sigma_coordinate,
    HybridSigmaPressureCoordinate,
    create_hybrid_coordinate,
    make_hybrid_levels,
    assert_hybrid_valid_for_surface_pressure,
    standard_hybrid_levels,
    hybrid_from_sigma,
)
from legoesm.grids.edge_blending import (
    blend_scalar_cube_edges,
    blend_scalar_cube_edges_2d,
    blend_vector_cube_edges,
)
from legoesm.grids.topography import (
    gaussian_mountain,
    zonal_ridge,
    schaer_mountain,
    land_mask_from_topography,
    phis_from_topography,
    smooth_phis_cubed_sphere,
    smooth_phis_gaussian,
    TopographyConfig,
    load_real_topography,
)
from legoesm.grids.voronoi import (
    VoronoiMesh,
    create_voronoi_mesh,
    create_regional_voronoi_mesh,
    load_mpas_mesh,
)
