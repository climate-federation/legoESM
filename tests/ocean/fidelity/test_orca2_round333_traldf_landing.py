"""Round 333: tra_ldf consumes the stored stage-3 Kmm T thickness."""

from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
MODEL = REPO / "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py"


def _gm_divisor_expression(source: str) -> str | None:
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "gm_redi_tracer_tendency_latlon"
        and any(keyword.arg == "redi_divisor_thickness_override"
                for keyword in node.keywords)
    ]
    if len(calls) != 1:
        return None
    values = [
        keyword.value for keyword in calls[0].keywords
        if keyword.arg == "redi_divisor_thickness_override"
    ]
    return ast.unparse(values[0]) if len(values) == 1 else None


def test_production_ldf_call_consumes_the_stored_kmm_thickness() -> None:
    expression = _gm_divisor_expression(MODEL.read_text())
    assert expression is not None
    assert "_h_s3_kmm" in expression
    assert "gm_cfg.redi_flux_geometry_stage == 'rk3_kmm'" in expression
    assert "tracer_ldf_diagnostics" in expression


def test_production_call_site_check_fails_for_the_exact_revert() -> None:
    source = MODEL.read_text()
    anchor = """                redi_divisor_thickness_override=(
                    self._nemo_ws_test_hooks.tracer_ldf_diagnostics.get(
                        \"divisor_thickness\")
                    if (isinstance(
                            self._nemo_ws_test_hooks.tracer_ldf_diagnostics,
                            dict)
                        and self._nemo_ws_test_hooks.tracer_ldf_diagnostics.get(
                            \"divisor_thickness\") is not None)
                    else (_h_s3_kmm if gm_cfg.redi_flux_geometry_stage
                          == \"rk3_kmm\" else None)),
"""
    assert anchor in source
    reverted = source.replace(
        anchor,
        """                redi_divisor_thickness_override=(self._nemo_ws_test_hooks.tracer_ldf_diagnostics.get(\"divisor_thickness\") if isinstance(self._nemo_ws_test_hooks.tracer_ldf_diagnostics, dict) else None),
""",
        1,
    )
    expression = _gm_divisor_expression(reverted)
    assert expression is not None
    assert "_h_s3_kmm" not in expression
