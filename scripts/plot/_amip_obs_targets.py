"""Shared AMIP observational targets + realism acceptance bands.

Single source of the CERES / GPCP / ERA5 global-mean comparison targets and
the mechanical pass/fail acceptance bands used by BOTH AMIP scorecards
(``plot_amip_cmor_diagnostics.py`` on CMOR Amon NetCDF, and
``plot_convection_scorecard.py`` on ``timeseries.npz``).  Kept dependency-light
(pure Python literals, no numpy/xarray/matplotlib) so the timeseries scorecard
can reuse the exact same criteria without pulling the CMOR plotter's heavy I/O
imports.

These are OBSERVATIONAL comparison targets + acceptance criteria (annual global
means from ERA5 / CERES / GPCP, roughly obs uncertainty + CMIP-class model
spread), NOT model physical constants — deliberately not sourced from
``legoesm.constants``.
"""

from __future__ import annotations

# (CMOR var, unit-scale, unit label, Earth observational reference, colormap).
# The unit-scale converts the raw CMOR field to the display/target unit (e.g.
# pr kg/m2/s -> mm/day); the reference value is in the DISPLAY unit.
FIELD_TABLE = (
    ("tas", 1.0, "K", 288.0, "RdBu_r"),
    ("pr", 86400.0, "mm/day", 2.9, "YlGnBu"),
    ("rsut", 1.0, "W/m2", 100.0, "viridis"),
    ("rlut", 1.0, "W/m2", 239.0, "magma"),
    ("clt", 1.0, "%", 67.0, "Blues"),
    ("prw", 1.0, "mm", 24.5, "GnBu"),
    ("hfls", 1.0, "W/m2", 88.0, "YlOrRd"),
    ("hfss", 1.0, "W/m2", 20.0, "YlOrRd"),
)
_ALBEDO_REF = 0.29   # observational planetary albedo (CERES); annotation only.

# Absolute tolerance on the area-weighted global mean of each field vs its
# Earth reference above (DISPLAY unit). Widen/tighten per campaign.
_REALISM_ABS_TOL = {
    "tas": 4.0,     # K
    "pr": 0.6,      # mm/day  (~20% of 2.9)
    "rsut": 12.0,   # W/m2
    "rlut": 10.0,   # W/m2
    "clt": 12.0,    # %
    "prw": 4.0,     # mm
    "hfls": 15.0,   # W/m2
    "hfss": 8.0,    # W/m2
}
_ALBEDO_ABS_TOL = 0.03        # planetary-albedo acceptance band (dimensionless)
_R_TOA_ABS_TOL = 5.0          # |net TOA imbalance| acceptance band [W/m2]
# Fields that MUST be present for a run to be scorecard-eligible at all.
_REQUIRED_FIELDS = ("tas", "pr", "rsut", "rlut")
