"""Pongratz et al. anthropogenic land-cover source (producer side).

The Pongratz et al. (2008, *Global Biogeochem. Cycles*,
doi:10.1029/2007GB003153) reconstruction gives annual **cropland** and
**pasture** fractions from AD 800 to 1992, merging historical maps before 1700
with HYDE-style statistics afterward.  Like HYDE it carries only the
anthropogenic fraction and no gross transitions, so the crosswalk overlays crop /
pasture on the CLM5 potential-natural-vegetation shape via the shared
:mod:`legoesm.land.surface_data.sources.anthropogenic` overlay.

Fractions are of the grid cell in [0, 1] (no absolute-area conversion, no urban).

**Fidelity note:** net transitions only -> E_LUC understates gross-transition
emissions (see :mod:`legoesm.land.land_use_change`).

Host-side only (NumPy / xarray); NOT traced.
"""

from __future__ import annotations

from legoesm.land.surface_data.sources.anthropogenic import (
    AnthropogenicSourceConfig,
    read_anthropogenic_states,
)

PONGRATZ_CONFIG = AnthropogenicSourceConfig(
    crop_vars=("crop",),
    pasture_vars=("pasture",),
    urban_vars=(),           # no built-up field.
    lat_var="lat",
    lon_var="lon",
    time_var="time",
    year_base=0,             # calendar years (AD 800-1992).
    area_var=None,           # already fractions of the cell.
)


def read_pongratz(
    path: str | None = None,
    config: AnthropogenicSourceConfig = PONGRATZ_CONFIG,
    *,
    years: tuple[int, int] | None = None,
    dataset=None,
) -> dict:
    """Read Pongratz crop / pasture -> ``{lat, lon, years, crop, pasture, urban}``.

    Thin preset over :func:`read_anthropogenic_states`; feed the result to
    :func:`legoesm.land.surface_data.sources.anthropogenic.build_anthropogenic_pft_frac`.
    """
    return read_anthropogenic_states(path, config, years=years, dataset=dataset)
