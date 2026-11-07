"""The ledger WRITER must ship the reduction its rows were built for (#1354).

The reader is tested directly in ``test_mpas_ledger_report.py``, but the
writer lives inside ``ModelDriver._run_column``'s diagnostic block, which needs
a full MPAS run to reach. Codex flagged the consequence during review of the
fix: *"No test locks this writer/tracker identity."*

That identity is the whole point. The rows are differenced against the
energy-budget tracker's ``dE/dt``, which is area-weighted; shipping ANY other
weights -- a fresh ``grid.areaCell`` read, a recomputed cell area, a
post-mask variant -- would silently reintroduce the mismatch this fix exists
to remove, while still looking area-weighted to the reader.

So these are AST assertions over the symbol that actually runs, and each is
paired with the specific defect it would catch.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

_SRC = (pathlib.Path(__file__).resolve().parents[2]
        / "packages" / "coupler" / "legoesm" / "driver" / "model_driver.py")


def _run_mpas_body() -> ast.FunctionDef:
    """The enclosing function, named -- not the module, and not a wrapper.

    A test that inspects source must name the symbol that RUNS; asserting
    against the whole file would pass on a copy of this block living in some
    other lane.
    """
    tree = ast.parse(_SRC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_run_column":
            return node
    pytest.fail("ModelDriver._run_column not found -- the pin has lost its target")


def _ledger_savez(fn: ast.FunctionDef) -> ast.Call:
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute)
                and node.func.attr == "savez"):
            continue
        if any(isinstance(kw.arg, str) and kw.arg == "ledger_rates"
               for kw in node.keywords):
            return node
    pytest.fail("the budget-ledger savez call is gone from _run_column")


def _area_kw_dict(fn: ast.FunctionDef) -> dict[str, str]:
    """The ``{"area_cell": ..., "area_hash48": ...}`` literal, as key -> code.

    Read STRUCTURALLY rather than by matching source text: ``ast.unparse``
    normalises string quoting, so a text assertion on ``"area_cell"`` fails
    against its own re-rendered ``'area_cell'`` -- which is exactly how the
    first version of this test failed while the code was correct.
    """
    for node in ast.walk(fn):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        if "area_cell" in keys:
            return {k.value: ast.unparse(v)
                    for k, v in zip(node.keys, node.values)
                    if isinstance(k, ast.Constant)}
    pytest.fail("no dict literal carrying 'area_cell' inside _run_column")


def test_writer_ships_weights_and_their_fingerprint():
    """Without both, the reader cannot establish the reduction and refuses."""
    fn = _run_mpas_body()
    call = _ledger_savez(fn)
    assert any(kw.arg is None for kw in call.keywords), (
        "the ledger savez has no **kwargs expansion, so the conditional "
        "area_cell/area_hash48 pair cannot be reaching the artifact")
    d = _area_kw_dict(fn)
    assert d.get("area_cell") == "_area_w_led", (
        "the shipped weights are no longer the snapshot taken next to the "
        f"tracker call; got {d.get('area_cell')!r}")
    assert d.get("area_hash48") == "np.asarray(_content_hash48(_area_w_led))", (
        "the fingerprint must be computed from the SAME array that is "
        "shipped, or an override can be validated against weights the "
        f"artifact does not contain; got {d.get('area_hash48')!r}")


def test_shipped_weights_are_the_trackers_own():
    """The identity codex asked for: one source, not two lookalikes.

    ``self.diagnostics._area_w`` is what the energy budget receives as
    ``area_weights``. Reading ``grid.areaCell`` here instead would look
    right and be wrong -- the tracker applies its own masking.
    """
    src = ast.unparse(_run_mpas_body())
    assert "_area_w_led = self.diagnostics._area_w" in src, (
        "the ledger no longer snapshots the tracker's own weights")
    assert "areaCell" not in src.split("_area_w_led")[1][:400], (
        "a raw areaCell read appeared next to the ledger weights; the two "
        "are not interchangeable")


def test_writer_refuses_a_weight_count_that_disagrees_with_its_rows():
    """The invariant GLM called the missing one: fail at the source."""
    src = ast.unparse(_run_mpas_body())
    assert "_area_w_led.size != _led_rates.shape[0]" in src, (
        "the writer-side shape check is gone; a rank/reorder mistake would "
        "now reach the artifact and be diagnosed downstream, if at all")


def test_weights_are_copied_not_aliased():
    """np.savez serialises at call time; an aliased buffer can be mutated
    between the tracker call and the save (GLM)."""
    src = ast.unparse(_run_mpas_body())
    assert ".ravel().copy()" in src, (
        "the ledger weights are no longer copied, so a later mask or regrid "
        "can make the artifact disagree with the dE/dt it is compared to")
