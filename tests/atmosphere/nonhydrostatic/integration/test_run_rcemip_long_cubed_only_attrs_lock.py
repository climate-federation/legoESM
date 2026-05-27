"""AST-locked check that ``_CUBED_ONLY_ATTRS`` matches the actual
cubed-sphere-only argparse registrations.

iter-294 (Codex iter-293 round-4 residual note): the iter-291
cubed-sphere-only misuse detection relies on
``_CUBED_ONLY_ATTRS`` being kept in sync with the CLI flags
that semantically apply only to ``--grid cubed_sphere``. If a
future PR adds a new ``--cubed-*`` or surface-flux ``--sfc-*``
flag and forgets to extend ``_CUBED_ONLY_ATTRS``, the silent-
ignore bug returns (passing the flag with --grid mpas does
nothing, no error).

This AST-based test parses scripts/run_rcemip_long.py and asserts
the ``_CUBED_ONLY_ATTRS`` literal exactly matches the set of
parser-flag attribute names that begin with the cubed-sphere or
surface-flux prefixes. A new flag triggers the lock.
"""
from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
DRIVER = REPO_ROOT / "scripts" / "run_rcemip_long.py"


# Prefixes that conceptually mean "cubed-sphere-only" in this
# driver. n-cubed-sphere is the grid-resolution knob; sfc-* are
# the bulk surface-flux knobs (cubed-sphere is the only grid
# wired for them today); cubed-* are dycore-tuning knobs.
_CUBED_PREFIXES = ("n_cubed_sphere", "sfc_", "cubed_")


def _extract_cubed_only_attrs_tuple(tree: ast.Module) -> tuple[str, ...]:
    """Return the literal value of the module-level
    ``_CUBED_ONLY_ATTRS`` assignment."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "_CUBED_ONLY_ATTRS":
                    if not isinstance(node.value, ast.Tuple):
                        raise AssertionError(
                            "_CUBED_ONLY_ATTRS RHS is not a tuple "
                            f"literal: {ast.dump(node.value)}"
                        )
                    return tuple(
                        elt.value for elt in node.value.elts
                        if isinstance(elt, ast.Constant)
                    )
    raise AssertionError(
        "module-level _CUBED_ONLY_ATTRS tuple not found in "
        f"{DRIVER}"
    )


def _extract_add_argument_dest_names(tree: ast.Module) -> set[str]:
    """Walk every ``p.add_argument(...)`` call and return the
    argparse-derived ``dest`` attr name (from the first positional
    arg, stripping ``--`` prefix and substituting ``-`` for ``_``).
    """
    dests: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match p.add_argument(...) or parser.add_argument(...).
        func = node.func
        if not (isinstance(func, ast.Attribute)
                and func.attr == "add_argument"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            flag = first.value
            if flag.startswith("--"):
                dest = flag[2:].replace("-", "_")
                dests.add(dest)
    return dests


def test_cubed_only_attrs_matches_add_argument_registrations():
    """Lock that ``_CUBED_ONLY_ATTRS`` includes every parser dest
    starting with ``n_cubed_sphere``, ``sfc_``, or ``cubed_``. A
    future ``--cubed-hyperdiff`` (or similar) added without
    extending the tuple fires this test instead of slipping past
    the iter-291 misuse detection.
    """
    tree = ast.parse(DRIVER.read_text())
    cubed_only_attrs = set(_extract_cubed_only_attrs_tuple(tree))
    all_dests = _extract_add_argument_dest_names(tree)
    cubed_prefix_dests = {
        d for d in all_dests
        if any(d.startswith(p) or d == p.rstrip("_")
               for p in _CUBED_PREFIXES)
    }
    missing = cubed_prefix_dests - cubed_only_attrs
    extra = cubed_only_attrs - cubed_prefix_dests
    assert not missing, (
        f"_CUBED_ONLY_ATTRS missing dest(s): {sorted(missing)}. "
        f"Either add to the tuple or rename the CLI flag to a "
        f"non-cubed/sfc prefix."
    )
    assert not extra, (
        f"_CUBED_ONLY_ATTRS contains attrs that don't correspond "
        f"to any p.add_argument() registration: {sorted(extra)}. "
        f"Stale tuple entries (CLI flag was renamed/removed?)."
    )
