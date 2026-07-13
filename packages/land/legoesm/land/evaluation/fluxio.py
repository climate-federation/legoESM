"""Readers and column schema for the CLM-ML offline ``.out`` files.

The offline drivers (``scripts/run/run_chats7_offline.py``,
``scripts/run/run_fluxnet_offline.py``) and the Fortran CLM-ML v2 reference
all write the same whitespace-delimited ``.out`` schema.  This module is the
single definition of that schema plus the loader, so the validator, the
recipe driver and the plotters agree on which column is which variable
(previously duplicated inside ``validate_clm_ml_canopy.py``).

Row writers still live next to the model call in
``scripts/run/run_chats7_offline.py`` (they read the live CLM-ML
``mlcanopy_type`` and pull in ``clm-ml-jax`` internals); this module is
deliberately dependency-light (numpy only) so it can be imported anywhere
for analysis without loading the canopy physics package.

Each schema entry is ``(key, label, unit)`` in column order.  ``time`` /
``zen`` / ``pai`` etc. are included so the column index equals the position
in the list.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Column schema — order MUST match the writers in run_chats7_offline.py
# (_write_flux_row / _write_fsun_row / _write_aux_row) and the Fortran
# CLMml_driver.py::output.
# ---------------------------------------------------------------------------

VarSpec = tuple[str, str, str]  # (key, human label, unit)

FLUX_VARS: list[VarSpec] = [
    ("time", "Julian day", "day"),
    ("rnet", "Net radiation", "W m-2"),
    ("stflx_air", "Air heat storage", "W m-2"),
    ("shflx", "Sensible heat", "W m-2"),
    ("lhflx", "Latent heat", "W m-2"),
    ("gpp", "GPP", "umol m-2 s-1"),
    ("ustar", "Friction velocity", "m s-1"),
    ("swup", "SW upwelling", "W m-2"),
    ("lwup", "LW upwelling", "W m-2"),
    ("tair_top", "Air T (canopy top)", "K"),
    ("G_soil", "Ground heat flux", "W m-2"),
    ("rn_soil", "Net radiation (soil)", "W m-2"),
    ("sh_soil", "Sensible heat (soil)", "W m-2"),
    ("lh_soil", "Latent heat (soil)", "W m-2"),
    ("lh_trans", "Transpiration LH", "W m-2"),
    ("lh_evap", "Evaporation LH", "W m-2"),
    ("beta", "Soil moisture stress", "-"),
    ("stflx_veg", "Veg heat storage", "W m-2"),
]

FSUN_VARS: list[VarSpec] = [
    ("zen", "Solar zenith", "deg"),
    ("sw_vis", "SW VIS", "W m-2"),
    ("pai", "PAI (LAI+SAI)", "m2 m-2"),
    ("laisun", "LAI sunlit", "m2 m-2"),
    ("laisha", "LAI shaded", "m2 m-2"),
    ("swveg_vis", "SW abs veg VIS", "W m-2"),
    ("swvegsun_vis", "SW abs veg VIS sun", "W m-2"),
    ("swvegsha_vis", "SW abs veg VIS sha", "W m-2"),
    ("gpp", "GPP total", "umol m-2 s-1"),
    ("gpp_sun", "GPP sunlit", "umol m-2 s-1"),
    ("gpp_sha", "GPP shaded", "umol m-2 s-1"),
    ("lh_veg", "LH veg", "W m-2"),
    ("lh_sun", "LH sunlit", "W m-2"),
    ("lh_sha", "LH shaded", "W m-2"),
    ("sh_veg", "SH veg", "W m-2"),
    ("sh_sun", "SH sunlit", "W m-2"),
    ("sh_sha", "SH shaded", "W m-2"),
    ("vcmax25_veg", "Vcmax25 veg", "umol m-2 s-1"),
    ("vcmax25_sun", "Vcmax25 sunlit", "umol m-2 s-1"),
    ("vcmax25_sha", "Vcmax25 shaded", "umol m-2 s-1"),
    ("gs_veg", "Stomatal cond veg", "mol m-2 s-1"),
    ("gs_sun", "Stomatal cond sun", "mol m-2 s-1"),
    ("gs_sha", "Stomatal cond sha", "mol m-2 s-1"),
    ("wind_veg", "Wind speed veg", "m s-1"),
    ("wind_sun", "Wind speed sun", "m s-1"),
    ("wind_sha", "Wind speed sha", "m s-1"),
    ("tl_veg", "Leaf T veg", "K"),
    ("tl_sun", "Leaf T sun", "K"),
    ("tl_sha", "Leaf T sha", "K"),
    ("ta_veg", "Air T in canopy", "K"),
    ("ta_sun", "Air T in canopy sun", "K"),
    ("ta_sha", "Air T in canopy sha", "K"),
]

AUX_VARS: list[VarSpec] = [
    ("btran", "Soil moisture stress", "-"),
    ("lsc_top", "Leaf spec. conduct. top", "mmol m-2 s-1 MPa-1"),
    ("psis", "Soil water potential", "MPa"),
    ("lwp_top", "Leaf WP (top)", "MPa"),
    ("lwp_mid", "Leaf WP (mid)", "MPa"),
    ("fracminlwp", "Frac leaves at min LWP", "-"),
]

# tag -> schema, so a recipe/validator can say "flux" and get the columns.
SCHEMAS: dict[str, list[VarSpec]] = {
    "flux": FLUX_VARS,
    "fsun": FSUN_VARS,
    "aux": AUX_VARS,
}


def load_out(path: str | Path) -> np.ndarray:
    """Load a whitespace-delimited ``.out`` file into a 2-D float array.

    Blank lines and any line that does not parse fully as floats (stray
    header/log lines) are skipped.  If rows have inconsistent widths, the
    modal width wins (a trailing partial row from a killed run is dropped).
    Returns an empty ``(0, 0)`` array when nothing parses, so callers can
    treat a missing/empty file uniformly.
    """
    rows: list[list[float]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append([float(x) for x in line.split()])
            except ValueError:
                continue
    if not rows:
        return np.empty((0, 0))
    lengths = {len(r) for r in rows}
    if len(lengths) > 1:
        modal = max(lengths, key=lambda n: sum(1 for r in rows if len(r) == n))
        rows = [r for r in rows if len(r) == modal]
    return np.array(rows, dtype=np.float64)


def load_out_named(
    path: str | Path, tag: str
) -> dict[str, np.ndarray]:
    """Load a ``.out`` file as ``{variable_key: column}`` using ``SCHEMAS``.

    ``tag`` is one of ``{"flux", "fsun", "aux"}``.  Columns beyond the
    schema length are ignored; a file with fewer columns than the schema
    raises (a schema/format mismatch is a real error, not a silent
    truncation).  An empty file yields an empty dict.
    """
    if tag not in SCHEMAS:
        raise ValueError(
            f"Unknown .out tag {tag!r}; known tags: {sorted(SCHEMAS)}"
        )
    specs = SCHEMAS[tag]
    arr = load_out(path)
    if arr.size == 0:
        return {}
    if arr.shape[1] < len(specs):
        raise ValueError(
            f"{Path(path).name}: expected >= {len(specs)} columns for tag "
            f"{tag!r} but found {arr.shape[1]}"
        )
    return {spec[0]: arr[:, i] for i, spec in enumerate(specs)}
