"""HYDE 3.2 / 3.3 anthropogenic land-cover source (producer side).

The History Database of the Global Environment (HYDE, Klein Goldewijk et al.
2017, *Earth Syst. Sci. Data*, doi:10.5194/essd-9-927-2017) reconstructs annual
**cropland**, **pasture**, **rangeland** and **built-up** areas from 10 000 BC to
the present.  HYDE carries only the *anthropogenic* land use (no natural-PFT
breakdown and no gross transitions), so the crosswalk overlays these fractions on
the model's existing CLM5 potential-natural-vegetation shape via the shared
:mod:`legoesm.land.surface_data.sources.anthropogenic` overlay.

HYDE grid files store **absolute areas** (km² per cell); dividing by the per-cell
grid area (``area_var``) recovers the fraction of the cell.  Pasture and rangeland
are both managed grass and are summed; built-up maps to the model's ``bare_soil``
proxy (no urban landunit).

**Fidelity note:** HYDE has no transitions, so E_LUC bookkeeping driven by it uses
*net* year-to-year cover change and therefore understates gross-transition
(shifting-cultivation) emissions — see :mod:`legoesm.land.land_use_change`.

Host-side only (NumPy / xarray); NOT traced.
"""

from __future__ import annotations

from legoesm.land.surface_data.sources.anthropogenic import (
    AnthropogenicSourceConfig,
    read_anthropogenic_states,
)

# HYDE variable names (the "baseline" product); overridable per file if needed.
HYDE_CONFIG = AnthropogenicSourceConfig(
    crop_vars=("cropland",),
    pasture_vars=("pasture", "rangeland"),
    urban_vars=("built_up",),
    lat_var="lat",
    lon_var="lon",
    time_var="time",
    year_base=0,             # HYDE grid files carry calendar years directly.
    area_var="garea",        # km² per cell -> fraction.
)


def read_hyde(
    path: str | None = None,
    config: AnthropogenicSourceConfig = HYDE_CONFIG,
    *,
    years: tuple[int, int] | None = None,
    dataset=None,
) -> dict:
    """Read HYDE cropland / pasture / built-up -> ``{lat, lon, years, crop, pasture, urban}``.

    Thin preset over :func:`read_anthropogenic_states`; feed the result to
    :func:`legoesm.land.surface_data.sources.anthropogenic.build_anthropogenic_pft_frac`.
    """
    return read_anthropogenic_states(path, config, years=years, dataset=dataset)
