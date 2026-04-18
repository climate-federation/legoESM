"""Ocean test matrix TestCase dataclass and test matrix generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from ocean_test_matrix import config


# ===========================================================================
# TestCase dataclass
# ===========================================================================

@dataclass
class TestCase:
    """A single test case in the ocean matrix."""
    case: str               # rest_state, barotropic_gyre, barotropic_double_gyre, etc.
    grid_type: str          # cubed_sphere, latlon, mpas, spectral
    resolution: str         # C24, 36x72, ico3, T21
    duration_days: float
    quick_days: float
    run_kwargs: dict = field(default_factory=dict)

    # Variant groups: cases that share a parent folder in the output tree.
    # Each key is a parent folder name; values are the case names nested under it.
    _VARIANT_GROUPS = {
        "rest_state": {
            "rest_state_stratified_with_land",
            "rest_state_uniform_with_land",
            "rest_state_stratified_no_land",
            "rest_state_uniform_no_land",
        },
        "barotropic_double_gyre": {
            "barotropic_double_gyre",
            "barotropic_double_gyre_sin2",
        },
        "baroclinic_gyre": {
            "baroclinic_gyre",
            "baroclinic_gyre_cos",
        },
    }

    # Flat lookup: case_name -> parent folder (built from _VARIANT_GROUPS)
    _CASE_TO_GROUP = {}
    for _group, _cases in _VARIANT_GROUPS.items():
        for _c in _cases:
            _CASE_TO_GROUP[_c] = _group

    @property
    def output_path(self) -> str:
        group = self._CASE_TO_GROUP.get(self.case)
        if group is not None:
            return f"{group}/{self.case}/{self.grid_type}/{self.resolution}"
        return f"{self.case}/{self.grid_type}/{self.resolution}"


# ===========================================================================
# Test matrix generation
# ===========================================================================

def _build_test_matrix() -> list[TestCase]:
    """Generate the full test matrix from grid x case."""
    matrix: list[TestCase] = []
    res = config.GRID_RESOLUTIONS

    # --- Rest state adjustment (with land): all grids except spectral ---
    for g in config.GRID_TYPES:
        if g == "spectral":
            continue
        matrix.append(TestCase(
            "rest_state_stratified_with_land", g, res[g], 1.0, 0.1))

    # --- Rest state with uniform T/S (with land): isolates barotropic PGF ---
    for g in config.GRID_TYPES:
        if g == "spectral":
            continue
        matrix.append(TestCase(
            "rest_state_uniform_with_land", g, res[g], 1.0, 0.1))

    # --- Rest state adjustment without land: all grids ---
    for g in config.GRID_TYPES:
        matrix.append(TestCase(
            "rest_state_stratified_no_land", g, res[g], 1.0, 0.1))

    # --- Rest state uniform T/S without land: control ---
    for g in config.GRID_TYPES:
        if g == "spectral":
            continue
        matrix.append(TestCase(
            "rest_state_uniform_no_land", g, res[g], 1.0, 0.1))

    # --- Barotropic gravity wave: resolution-matched grids (~384-446 km dx) ---
    bwave_res = {"cubed_sphere": "C24", "latlon": "48x72",
                 "mpas": "ico4", "spectral": "T21"}
    for g in config.GRID_TYPES:
        matrix.append(TestCase(
            "barotropic_wave", g, bwave_res[g], 2.0, 0.2))

    # --- Wind-driven regional barotropic double gyre: regional grids ---
    # cs_regional excluded: ocean init assumes 6-face arrays (TODO: adapt)
    # Two wind profiles: cosine (zero net wind) and sin² (net eastward wind)
    for g in ["mpas_regional", "latlon_regional"]:
        matrix.append(TestCase(
            "barotropic_double_gyre", g, res[g], 30.0, 2.0))
        matrix.append(TestCase(
            "barotropic_double_gyre_sin2", g, res[g], 30.0, 2.0))

    # --- Wind-driven regional baroclinic gyre: regional grids ---
    # Tests Coriolis double-counting fix (#103) with realistic stratification
    # Two wind profiles: sin² (default, net eastward) and cosine (zero net wind)
    for g in ["mpas_regional", "latlon_regional"]:
        matrix.append(TestCase(
            "baroclinic_gyre", g, res[g], 60.0, 5.0))
        matrix.append(TestCase(
            "baroclinic_gyre_cos", g, res[g], 60.0, 5.0))

    # --- Global barotropic wind-driven: latlon, mpas ---
    # (cubed_sphere excluded — face-boundary instability produces unphysical speeds)
    for g in ["latlon", "mpas"]:
        matrix.append(TestCase(
            "global_barotropic_wind", g, res[g], 60.0, 5.0))

    # --- Global barotropic wind, single layer (truly barotropic) ---
    # 1-level eliminates vertical coupling issues; wind and bottom drag
    # act on the same layer → clean Sverdrup-like equilibrium.
    for g in ["latlon", "mpas"]:
        matrix.append(TestCase(
            "global_barotropic_wind_1lev", g, res[g], 60.0, 5.0,
            run_kwargs={"nlev": 1}))

    # --- Geostrophic adjustment: all grids ---
    for g in config.GRID_TYPES:
        matrix.append(TestCase(
            "geostrophic_adjustment", g, res[g], 10.0, 1.0))

    # --- Phillips two-layer baroclinic: all grids ---
    for g in config.GRID_TYPES:
        matrix.append(TestCase(
            "phillips_two_layer", g, res[g], 10.0, 1.0))

    # --- Inertia-Gravity Wave (Bishnu et al. 2024): all grids ---
    for g in config.GRID_TYPES:
        matrix.append(TestCase(
            "inertia_gravity_wave", g, res[g], 2.0, 0.2))

    # --- Lock Exchange (NEMO / Petersen et al. 2015): cubed_sphere, latlon ---
    for g in ["cubed_sphere", "latlon"]:
        matrix.append(TestCase(
            "lock_exchange", g, res[g], 1.0, 0.1))

    # --- Overflow (NEMO / Petersen et al. 2015): cubed_sphere, latlon ---
    for g in ["cubed_sphere", "latlon"]:
        matrix.append(TestCase(
            "overflow", g, res[g], 0.5, 0.1))

    # --- Stommel Gyre Tracer (Hecht et al. 2000): cubed_sphere, latlon, mpas ---
    for g in ["cubed_sphere", "latlon", "mpas"]:
        matrix.append(TestCase(
            "stommel_gyre_tracer", g, res[g], 60.0, 5.0))

    # --- Eady baroclinic instability: zonally periodic channel grids ---
    for g in ["mpas_channel", "latlon_channel"]:
        matrix.append(TestCase(
            "eady_instability", g, res[g], 60.0, 5.0))

    # --- Classical Eady (uniform N², linear shear): 1000x2000 km channel ---
    # 200x100 latlon ≈ 10 km resolution; L_d ≈ 97 km, λ_max ≈ 390 km
    eady_u_res = {"latlon_channel": "200x100", "mpas_channel": "10km"}
    for g in ["latlon_channel", "mpas_channel"]:
        matrix.append(TestCase(
            "eady_uniform", g, eady_u_res[g], 120.0, 10.0))

    # --- ACC channel with Gaussian ridge (Zhang et al. 2024 inspired) ---
    # ~50 km resolution default; exercises z-star with variable bathymetry
    acc_res = {"latlon_channel": "20x36", "mpas_channel": "50km"}
    for g in ["latlon_channel", "mpas_channel"]:
        matrix.append(TestCase(
            "acc_channel", g, acc_res[g], 30.0, 2.0))

    return matrix


TEST_MATRIX = _build_test_matrix()
