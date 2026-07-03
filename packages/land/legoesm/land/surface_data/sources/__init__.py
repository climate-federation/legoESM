"""Primary surface-data sources (one module per dataset).

Each module reads a raw dataset and produces fields on the harmonized regular
lat-lon grid for :mod:`legoesm.land.surface_data.schema`.  First source:
:mod:`~legoesm.land.surface_data.sources.hwsd2` (FAO HWSD v2.0 soil).
"""

from __future__ import annotations
