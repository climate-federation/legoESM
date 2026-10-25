"""Round 332: retain hpg_sco's dry V source until NEMO's stage mask.

NEMO's active ``hpg_sco`` writes the unmasked acceleration into ``Krhs``
(``dynhpg.f90:400,425``).  Its vector stage masks only later, at
``stprk3_stg.f90:470``.  The sign of that product is live input to the north
fold, so the generic tendency boundary must not mask the completed V RHS
first.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    _rk3_fold_exchange_top_v,
    rk3_stage_velocity_update,
)


REPO = Path(__file__).resolve().parents[3]
TENDENCY_SOURCE = (
    REPO / "packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py"
)


def premature_vector_vmask_writes(source: str) -> list[str]:
    """Find writes that reintroduce the pre-stage V mask at its real site."""

    tree = ast.parse(source)
    matching_branches: list[ast.If] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if ast.unparse(node.test) == "bool(config.rk3_vector_raw_vmask)":
            matching_branches.append(node)
    defects: list[str] = []
    if len(matching_branches) != 1:
        return [f"found {len(matching_branches)} vector-raw-mask branches"]
    writes = []
    for node in ast.walk(ast.Module(body=matching_branches[0].body,
                                    type_ignores=[])):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        if any(isinstance(target, ast.Name) and target.id == "dv_dt"
               for target in targets):
            writes.append(ast.unparse(node))
    defects.extend(writes)
    return defects


def test_production_vector_rhs_reaches_the_stage_without_a_premature_mask():
    assert premature_vector_vmask_writes(TENDENCY_SOURCE.read_text()) == []


def test_production_call_site_check_fails_for_the_exact_revert():
    source = TENDENCY_SOURCE.read_text()
    anchor = """        if bool(config.rk3_vector_raw_vmask):
            # hpg_sco stores its source-native acceleration without a V mask
"""
    assert anchor in source
    reverted = source.replace(
        anchor,
        """        if bool(config.rk3_vector_raw_vmask):
            dv_dt = dv_dt * _literal_vmask
            # hpg_sco stores its source-native acceleration without a V mask
""",
        1,
    )
    assert premature_vector_vmask_writes(reverted) == [
        "dv_dt = dv_dt * _literal_vmask"
    ]


def test_negative_hpg_source_survives_the_stage_mask_and_fold_with_finite_grad():
    mask = jnp.zeros((2, 1, 1), dtype=jnp.float64)
    fold = SimpleNamespace(
        is_active=True,
        perm_v=jnp.asarray([0], dtype=jnp.int32),
        vector_sign_v=-1.0,
    )
    grid = SimpleNamespace(fold=fold)

    def folded(source):
        before = jnp.zeros_like(source)
        staged = rk3_stage_velocity_update(
            before, source, np.float64(1.0), mask,
            vector_form=True, written_order=True,
        )
        return _rk3_fold_exchange_top_v(staged, grid, mask)

    source = jnp.asarray([[[-2.0]], [[0.0]]], dtype=jnp.float64)
    result = np.asarray(folded(source))
    assert result[0, 0, 0] == 0.0 and np.signbit(result[0, 0, 0])
    assert result[1, 0, 0] == 0.0 and not np.signbit(result[1, 0, 0])

    gradient = np.asarray(jax.grad(lambda value: jnp.sum(folded(value)))(source))
    assert np.isfinite(gradient).all()
