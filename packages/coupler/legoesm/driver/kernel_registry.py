"""Registry-driven kernel selection for physics parameterizations.

Replaces the hard-coded ``if/elif`` dispatch chains in the old
``build_physics_pipeline`` with simple dictionaries.

Each registry maps a **scheme name** (str) to a *(module_path, function_name)*
pair.  The lookup is lazy: modules are imported only when the scheme is
actually selected.  This keeps import time low and avoids pulling in unused
backends (e.g. RRTMGP data tables when only gray radiation is needed).

Public API
----------
RADIATION_REGISTRY : dict
    ``{"gray": ..., "rrtmgp": ...}``
CONVECTION_REGISTRY : dict
    ``{"sbm": ..., "dca": ..., "kuo": ..., "mass_flux": ..., "edmf": ...,
       "zhang_mcfarlane": ..., "kain_fritsch": ..., "emanuel": ...,
       "tiedtke": ..., "bechtold": ...}``
MICROPHYSICS_REGISTRY : dict
    ``{"kessler": ..., "sundqvist": ..., "seifert_beheng": ...,
      "morrison": ..., "thompson": ...}``
resolve_kernel(registry, name) -> callable
    Import and return the kernel function for *name*.
"""
from __future__ import annotations

from importlib import import_module
from typing import Callable

# ---------------------------------------------------------------------------
# Registry entries: (module_path, function_name, config_attr_on_parent)
#
#   module_path  — dotted module path, resolved via importlib
#   func_name    — attribute name of the kernel function in that module
#   config_attr  — attribute on the parent config (MicrophysicsConfig, etc.)
#                  that holds the per-scheme sub-config; None if N/A.
# ---------------------------------------------------------------------------

_Entry = tuple  # (module_path: str, func_name: str)


# -- Radiation ---------------------------------------------------------------

RADIATION_REGISTRY: dict[str, _Entry] = {
    "gray": (
        "legoesm.atmosphere.physics.radiation.gray",
        "gray_radiation",
    ),
    "rrtmgp": (
        "legoesm.atmosphere.physics.radiation.rrtmgp_radiation",
        "rrtmgp_radiation",
    ),
}


# -- Convection --------------------------------------------------------------

CONVECTION_REGISTRY: dict[str, _Entry] = {
    "sbm": (
        "legoesm.atmosphere.physics.convection.sbm",
        "sbm_convection",
    ),
    "dca": (
        "legoesm.atmosphere.physics.convection.dca",
        "dca_convection",
    ),
    "kuo": (
        "legoesm.atmosphere.physics.convection.kuo",
        "kuo_convection",
    ),
    "mass_flux": (
        "legoesm.atmosphere.physics.convection.mass_flux",
        "mass_flux_convection",
    ),
    "edmf": (
        "legoesm.atmosphere.physics.convection.mass_flux",
        "edmf_convection",
    ),
    "zhang_mcfarlane": (
        "legoesm.atmosphere.physics.convection.zhang_mcfarlane",
        "zhang_mcfarlane_convection",
    ),
    "kain_fritsch": (
        "legoesm.atmosphere.physics.convection.kain_fritsch",
        "kain_fritsch_convection",
    ),
    "emanuel": (
        "legoesm.atmosphere.physics.convection.emanuel",
        "emanuel_convection",
    ),
    "tiedtke": (
        "legoesm.atmosphere.physics.convection.tiedtke",
        "tiedtke_convection",
    ),
    "bechtold": (
        "legoesm.atmosphere.physics.convection.bechtold",
        "bechtold_convection",
    ),
}


# -- Microphysics ------------------------------------------------------------

MICROPHYSICS_REGISTRY: dict[str, _Entry] = {
    "kessler": (
        "legoesm.atmosphere.physics.microphysics.kessler",
        "kessler_microphysics",
    ),
    "sundqvist": (
        "legoesm.atmosphere.physics.microphysics.sundqvist",
        "sundqvist_microphysics",
    ),
    "seifert_beheng": (
        "legoesm.atmosphere.physics.microphysics.seifert_beheng",
        "seifert_beheng_microphysics",
    ),
    "morrison": (
        "legoesm.atmosphere.physics.microphysics.morrison",
        "morrison_microphysics",
    ),
    "thompson": (
        "legoesm.atmosphere.physics.microphysics.thompson",
        "thompson_microphysics",
    ),
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def resolve_kernel(registry: dict[str, _Entry], name: str) -> Callable:
    """Look up *name* in *registry*, import the module, and return the kernel.

    Parameters
    ----------
    registry : dict
        One of ``RADIATION_REGISTRY``, ``CONVECTION_REGISTRY``, or
        ``MICROPHYSICS_REGISTRY``.
    name : str
        Scheme name, e.g. ``"kessler"`` or ``"gray"``.

    Returns
    -------
    callable
        The kernel function ready to be called.

    Raises
    ------
    KeyError
        If *name* is not in *registry*.
    """
    if name not in registry:
        available = ", ".join(sorted(registry))
        raise KeyError(
            f"Unknown scheme {name!r}. Available: {available}"
        )
    module_path, func_name = registry[name]
    mod = import_module(module_path)
    return getattr(mod, func_name)


def available_schemes(registry: dict[str, _Entry]) -> list[str]:
    """Return sorted list of registered scheme names."""
    return sorted(registry)
