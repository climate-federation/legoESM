"""Round 333: tra_ldf consumes the safe stored stage-3 Kmm thickness."""

from __future__ import annotations

import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    _nemo_safe_stored_ldf_divisor,
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


def test_production_ldf_call_consumes_the_safe_stored_kmm_thickness() -> None:
    expression = _gm_divisor_expression(MODEL.read_text())
    assert expression is not None
    assert "_nemo_safe_stored_ldf_divisor(_h_s3_kmm, _active_live)" in expression
    assert "gm_cfg.redi_flux_geometry_stage == 'rk3_kmm'" in expression
    assert "tracer_ldf_diagnostics" in expression


def test_production_call_site_check_fails_for_the_exact_revert() -> None:
    source = MODEL.read_text()
    anchor = """                # traldf_iso.f90:301-305/:318-322 reads the already-stored
                # stage-2 e3t(Kmm).  Preserve those wet-cell words, but use an
                # inert unit divisor where NEMO's wet-cell loop does not
                # execute; a rectangular 0/0 followed by masking has a NaN
                # transpose even when its primal is later discarded.
                redi_divisor_thickness_override=(
                    self._nemo_ws_test_hooks.tracer_ldf_diagnostics.get(
                        \"divisor_thickness\")
                    if (isinstance(
                            self._nemo_ws_test_hooks.tracer_ldf_diagnostics,
                            dict)
                        and self._nemo_ws_test_hooks.tracer_ldf_diagnostics.get(
                            \"divisor_thickness\") is not None)
                    else (_nemo_safe_stored_ldf_divisor(
                        _h_s3_kmm, _active_live)
                        if gm_cfg.redi_flux_geometry_stage == \"rk3_kmm\"
                        else None)),
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


def test_safe_stored_divisor_preserves_wet_bits_and_has_finite_transpose() -> None:
    stored = jnp.asarray([2.0, 0.0, 0.0], dtype=jnp.float64)
    active = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64)
    safe = _nemo_safe_stored_ldf_divisor(stored, active)
    np.testing.assert_array_equal(
        np.asarray(safe).view(np.uint64),
        np.asarray([2.0, 1.0, 1.0], dtype=np.float64).view(np.uint64),
    )

    numerator = jnp.asarray([3.0, 0.0, 0.0], dtype=jnp.float64)

    def masked_division(divisor):
        safe_divisor = _nemo_safe_stored_ldf_divisor(divisor, active)
        return jnp.sum((numerator / safe_divisor) * active)

    gradient = jax.grad(masked_division)(stored)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    np.testing.assert_array_equal(
        np.asarray(gradient), np.asarray([-0.75, 0.0, 0.0]))

