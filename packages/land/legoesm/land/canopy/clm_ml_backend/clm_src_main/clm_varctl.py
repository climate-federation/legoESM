"""
JAX translation of clm_varctl Fortran module.

Run control variables for the CLM land surface model.
Contains global configuration constants used throughout the model,
including the log file unit number and the path to the RSL psihat
look-up tables.

Original Fortran module: clm_varctl
Fortran lines 1-17
"""

import os as _os

# "stdout" log file unit number — Fortran line 13
# Used throughout the model as the target for diagnostic output.
iulog: int = 6


# RSL psihat look-up table file path — Fortran line 14
# Path to the roughness sublayer psihat NetCDF look-up table read by
# MLCanopyTurbulenceMod.LookupPsihatINI.  The file ships as package data inside
# ``multilayer_canopy/data/`` so it is present in a built wheel/sdist, not only
# in an editable source checkout.  Resolve it with importlib.resources (works
# for regular wheel installs, editable installs, and the source tree); fall back
# to the legacy ``src/rsl_lookup_tables/`` layout so an old checkout still runs.
def _resolve_rslfile() -> str:
    try:
        from importlib.resources import files as _files

        _p = _files("legoesm.land.canopy.clm_ml_backend.multilayer_canopy") / "data" / "psihat.nc"
        # netCDF4 needs a real filesystem path; the package installs unzipped, so
        # the traversable is already a concrete path. str() is safe here.
        if _p.is_file():
            return str(_p)
    except (ImportError, ModuleNotFoundError, FileNotFoundError):
        pass
    # Legacy fallback: <src>/rsl_lookup_tables/psihat.nc (two levels up).
    _pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    return _os.path.join(_pkg_root, "rsl_lookup_tables", "psihat.nc")


rslfile: str = _resolve_rslfile()
