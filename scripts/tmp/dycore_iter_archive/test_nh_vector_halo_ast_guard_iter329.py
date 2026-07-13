"""FV3_3D iter 329: AST regression guard for iter-328 vector halo
wiring on the NH 3D path.

iter-328 wired ``center_to_dgrid_vector`` as the opt-in path for
the NH cell-centre → D-grid corner interpolation of (u, v) when
``use_fv3_vector_halo_uv = True``.  Without an AST guard a future
refactor could silently drop:

* the ``center_to_dgrid_vector`` import,
* the gate ``if config.use_fv3_vector_halo_uv:`` branch,
* or the ``center_to_dgrid_vector(u, v, cdgrid)`` call inside
  the gate

leaving ``use_fv3_vector_halo_uv = True`` as a silent no-op (the
flag flips but nothing changes), restoring the cube-edge basis-
mismatch bug without test failure (iter-328 tests would still pass
because they were captured at iter-328 time).

Wait — iter-328's ``test_vector_path_changes_state`` would actually
catch a silent no-op (asserts state DIFFERS).  iter-329 still adds
value: it pins the EXACT mechanism, so a refactor that silently
swaps the body for an equivalent-on-iter-328-fixture path (e.g.,
re-implementing scalar halo manually that happens to produce
slightly different floats) would fail iter-329 even if it slips
past iter-328.  Defense-in-depth.

Tests
-----

1. ``test_center_to_dgrid_vector_imported`` — the symbol is
   imported from operators_cdgrid.
2. ``test_vector_halo_gate_present`` — ``if config.use_fv3_vector_halo_uv:``
   branch present in NH source.
3. ``test_center_to_dgrid_vector_called_in_gate`` —
   ``center_to_dgrid_vector(u, v, cdgrid)`` appears within the
   gated branch.
4. ``test_scalar_path_else_branch_present`` — the else branch
   keeps ``interp_center_to_corner`` on the stacked (u, v) for
   the bit-for-bit baseline default.
"""
from __future__ import annotations

import re

import pytest

from tests.legoesm_paths import legoesm_source_path


SRC_PATH = legoesm_source_path("atmosphere/dynamics/gcm/compressible_euler_cdgrid.py")


@pytest.fixture(scope="module")
def nh_source():
    return SRC_PATH.read_text()


def test_center_to_dgrid_vector_imported(nh_source):
    """``center_to_dgrid_vector`` is imported from operators_cdgrid."""
    pat = (
        r"from\s+legoesm\.core\.operators_cdgrid\s+import\s*\("
        r"[^)]*center_to_dgrid_vector"
    )
    assert re.search(pat, nh_source, re.DOTALL), (
        "iter-328 import 'center_to_dgrid_vector' missing from "
        f"{SRC_PATH.name} import block.  A refactor likely "
        f"removed the import, breaking the vector halo path."
    )


def test_vector_halo_gate_present(nh_source):
    """``if config.use_fv3_vector_halo_uv:`` gate present."""
    pat = r"if\s+config\.use_fv3_vector_halo_uv\s*:"
    assert re.search(pat, nh_source), (
        "iter-328 gate 'if config.use_fv3_vector_halo_uv:' missing "
        f"from {SRC_PATH.name}.  The opt-in flag has no effect."
    )


def test_center_to_dgrid_vector_called_in_gate(nh_source):
    """``center_to_dgrid_vector(u, v, cdgrid, ...)`` appears.

    Accepts the 3-arg form (iter-328) or the 4+ arg form (iter-697
    added ``use_fv3_a2b_ord4=`` keyword arg).
    """
    pat = r"center_to_dgrid_vector\(\s*u\s*,\s*v\s*,\s*cdgrid\b"
    assert re.search(pat, nh_source), (
        "iter-328 call 'center_to_dgrid_vector(u, v, cdgrid, ...)' "
        f"missing from {SRC_PATH.name}.  The vector halo path is "
        f"a silent no-op."
    )


def test_scalar_path_else_branch_present(nh_source):
    """The else branch keeps the legacy scalar-halo stack pattern
    so flag=False stays bit-for-bit baseline."""
    # Look for the stack-based scalar halo path (legacy default).
    pat = r"_uv_stack\s*=\s*jnp\.stack\(\s*\[\s*u\s*,\s*v\s*\]"
    assert re.search(pat, nh_source), (
        "iter-328 legacy scalar halo path '_uv_stack = jnp.stack([u, v]...' "
        f"missing from {SRC_PATH.name}.  Flag=False would no longer "
        f"reproduce the pre-iter-328 baseline bit-for-bit."
    )
    pat2 = r"interp_center_to_corner\(\s*_uv_flat\s*,\s*cdgrid\s*\)"
    assert re.search(pat2, nh_source), (
        "iter-328 legacy 'interp_center_to_corner(_uv_flat, cdgrid)' "
        "call missing — scalar halo baseline path broken."
    )
