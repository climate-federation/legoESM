"""KK10 (Kaplan & Krumhardt) anthropogenic land-cover source (producer side).

The KK10 reconstruction (Kaplan et al. 2011, *The Holocene*,
doi:10.1177/0959683610386983) gives a single annual **total anthropogenic land-
use fraction** (notably high early-Holocene land use relative to HYDE), with no
crop/pasture split, no built-up, and no gross transitions.  The crosswalk
therefore (1) reads the total used fraction and (2) splits it into cropland and
pasture by a fixed share before overlaying it on the CLM5 potential-natural-
vegetation shape via the shared
:mod:`legoesm.land.surface_data.sources.anthropogenic` overlay.

The crop/pasture split is a genuine free choice of the reconstruction (KK10 does
not resolve it), exposed as ``crop_share`` (default 0.5) so a producer can set it
per run rather than baking in a magic value.

**Fidelity note:** net transitions only -> E_LUC understates gross-transition
emissions (see :mod:`legoesm.land.land_use_change`).

Host-side only (NumPy / xarray); NOT traced.
"""

from __future__ import annotations

import numpy as np

from legoesm.land.surface_data.sources.anthropogenic import (
    AnthropogenicSourceConfig,
    read_anthropogenic_states,
)

# KK10 stores one total-used field; it enters read_anthropogenic_states as the
# lone "crop" group and is re-split by read_kk10 afterward.
KK10_CONFIG = AnthropogenicSourceConfig(
    crop_vars=("land_use",),
    pasture_vars=(),
    urban_vars=(),
    lat_var="lat",
    lon_var="lon",
    time_var="time",
    year_base=0,
    area_var=None,           # already a fraction of the cell.
)


def read_kk10(
    path: str | None = None,
    config: AnthropogenicSourceConfig = KK10_CONFIG,
    *,
    crop_share: float = 0.5,
    years: tuple[int, int] | None = None,
    dataset=None,
) -> dict:
    """Read KK10 total anthropogenic fraction, split crop / pasture by ``crop_share``.

    Returns ``{lat, lon, years, crop, pasture, urban}`` (urban all-zero); feed to
    :func:`legoesm.land.surface_data.sources.anthropogenic.build_anthropogenic_pft_frac`.
    """
    if not 0.0 <= crop_share <= 1.0:
        raise ValueError(f"crop_share must be in [0, 1] (got {crop_share})")
    raw = read_anthropogenic_states(path, config, years=years, dataset=dataset)
    total = raw["crop"]                       # the whole used fraction sits here
    raw["crop"] = total * crop_share
    raw["pasture"] = total * (1.0 - crop_share)
    raw["urban"] = np.zeros_like(total)
    return raw
