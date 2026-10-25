"""Round 333: tra_ldf consumes the stored stage-3 Kmm T thickness."""

from __future__ import annotations

import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    _stored_primal_with_rebuilt_tangent,
)


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


def test_stored_divisor_keeps_its_bits_and_the_finite_rebuilt_tangent() -> None:
    stored = jnp.asarray([1.0, jnp.nextafter(1.0, 2.0)], dtype=jnp.float64)
    rebuilt = jnp.asarray([3.0, 4.0], dtype=jnp.float64)
    stored_dot = jnp.asarray([np.nan, np.nan], dtype=jnp.float64)
    rebuilt_dot = jnp.asarray([5.0, -7.0], dtype=jnp.float64)
    primal, tangent = jax.jvp(
        _stored_primal_with_rebuilt_tangent,
        (stored, rebuilt),
        (stored_dot, rebuilt_dot),
    )
    np.testing.assert_array_equal(
        np.asarray(primal).view(np.uint64), np.asarray(stored).view(np.uint64))
    np.testing.assert_array_equal(np.asarray(tangent), np.asarray(rebuilt_dot))
    gradient = jax.grad(
        lambda live: jnp.sum(_stored_primal_with_rebuilt_tangent(stored, live))
    )(rebuilt)
    np.testing.assert_array_equal(np.asarray(gradient), np.ones(2))
