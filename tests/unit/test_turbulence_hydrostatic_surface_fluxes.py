"""The hydrostatic turbulence path must export its surface turbulent fluxes.

It did not.  Only the MPAS/edge-normal construction attached
``shflx_sfc``/``lhflx_sfc`` to the tendency, so on the hydrostatic lane the
CMOR hfls/hfss feed had nothing to read (``evspsbl`` is derived as
``lhflx / L_v``), and a single-column water budget could not see its own
evaporation -- the SCM-RCE budget measured E = 0.0000 mm/day on a column whose
own bulk formula gives 1.559.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = (REPO_ROOT / "packages" / "atmosphere" / "legoesm" / "atmosphere"
       / "physics" / "turbulence" / "integration.py")


def _tendency_constructions(src: str):
    """Every ``HydrostaticTendencies(...)`` call in the module, as AST nodes."""
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "HydrostaticTendencies"):
            out.append({kw.arg for kw in node.keywords})
    return out


def test_every_tendency_construction_exports_the_surface_fluxes():
    """AST, not a string search: both lanes build the object, and a lane that
    silently omits the diagnostic is the defect this file exists for."""
    calls = _tendency_constructions(SRC.read_text())
    assert calls, "no HydrostaticTendencies construction found — test is stale"
    missing = [i for i, kws in enumerate(calls)
               if not {"shflx_sfc", "lhflx_sfc"} <= kws]
    assert not missing, (
        f"{len(missing)} of {len(calls)} HydrostaticTendencies constructions in "
        "turbulence/integration.py omit shflx_sfc/lhflx_sfc; a lane that does "
        "not export them leaves the CMOR hfls/hfss feed empty and makes a "
        "column water budget unable to see its own evaporation")


def test_the_export_is_none_guarded():
    """A scheme with no surface fluxes must stay byte-identical, so the export
    has to be conditional rather than unconditional."""
    src = SRC.read_text()
    assert src.count("None if _lhf") + src.count("None if _lhf_h") >= 2
    assert src.count("None if _shf") + src.count("None if _shf_h") >= 2
