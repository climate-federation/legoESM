"""#2525: a run ending on a year boundary keeps its window means in BOTH files.

The annual snapshot drains the window-mean accumulators.  When the final step
is also a year boundary, draining it inside the loop left ``snapshot_final``
with no window means.  ``main`` now skips the in-loop annual write on the final
step and writes the year file after the loop with the same ``extra`` dict as
``snapshot_final``.  An end-to-end year of integration is not affordable in a
unit test, so this reads ``main``'s own AST (the function that runs).
"""
from __future__ import annotations

import ast
import inspect


def _main_tree():
    import scripts.run.run_omip_core2 as core2
    return ast.parse(inspect.getsource(core2.main))


def _year_writes(tree):
    """(inside_loop, If node) for every ``if`` whose test reads steps_per_year
    and whose body writes a snapshot."""
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)
             and ast.unparse(n.target) == "step"]
    in_loop = {id(n) for lp in loops for n in ast.walk(lp)}
    return [(id(n) in in_loop, n) for n in ast.walk(tree)
            if isinstance(n, ast.If) and "steps_per_year" in ast.unparse(n.test)
            and any(isinstance(c, ast.Call) and ast.unparse(c.func) == "_save_snapshot"
                    for b in n.body for c in ast.walk(b))]


def test_in_loop_annual_snapshot_skips_the_final_step():
    in_loop = [n for inside, n in _year_writes(_main_tree()) if inside]
    assert in_loop, "no annual snapshot block found inside the step loop"
    for n in in_loop:
        assert "step != n_steps" in ast.unparse(n.test), ast.unparse(n.test)


def test_final_year_file_shares_the_final_snapshot_extra():
    tree = _main_tree()
    # (the lax.scan lane's while-loop year write carries no accumulators)
    after = [n for inside, n in _year_writes(tree) if not inside
             and "n_steps % steps_per_year" in ast.unparse(n.test)]
    assert len(after) == 1, "expected one post-loop year write"
    src = ast.unparse(after[0])
    assert "_save_snapshot" in src and "extra=_final_extra" in src, src
    assert "step=n_steps" in src and "day=_final_day" in src, src
    finals = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
              and ast.unparse(n.func) == "_save_snapshot"
              and n.args[1:2] and ast.unparse(n.args[1]) == "'final'"]
    assert finals and any(
        any(k.arg == "extra" and ast.unparse(k.value) == "_final_extra"
            for k in c.keywords) for c in finals)
