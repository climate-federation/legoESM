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
    """``HydrostaticTendencies(...)`` calls that carry a REAL turbulence result.

    The module also builds zero tendencies on its ``turb_fn is None`` branches
    (turbulence disabled).  Those legitimately have no surface fluxes to
    report, and demanding the keyword there would be demanding a `None` be
    written explicitly -- the first version of this gate did exactly that and
    failed on both of them.  A construction is exempt when every tendency it
    passes is a freshly-zeroed array, which is what those branches build.
    """
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "HydrostaticTendencies"):
            continue
        seg = ast.get_source_segment(src, node) or ""
        if "turb_out" in seg or "_lhf" in seg or "_shf" in seg:
            out.append({kw.arg for kw in node.keywords})
            continue
        # A turbulence-off branch passes ZERO arrays for every tendency. They
        # may be built inline (`jnp.zeros(...)`) or hoisted into locals
        # (`zero_edges`), so test each argument's own source rather than
        # searching the whole call -- the hoisted form slipped through the
        # first version of this heuristic.
        vals = [ast.get_source_segment(src, kw.value) or ""
                for kw in node.keywords if kw.arg != "tracer_tendencies"]
        if vals and all("zero" in v.lower() or v.strip() == "None" for v in vals):
            continue
        out.append({kw.arg for kw in node.keywords})
    return out


def test_the_zero_tendency_branches_are_exempt_and_the_real_ones_are_not():
    """NON-VACUITY of the exemption: it must remove the turbulence-off
    branches and NOTHING else, or the gate stops covering the lanes it was
    written for."""
    src = SRC.read_text()
    total = sum(1 for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "HydrostaticTendencies")
    kept = len(_tendency_constructions(src))
    assert total >= 4, total
    assert kept == 2, (
        f"{kept} of {total} constructions kept; expected exactly the two that "
        "carry a real turbulence result (hydrostatic and MPAS)")


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
