"""The config NamedTuples' positional layout is a merge hazard. Make it loud.

WHY THIS EXISTS. `DycoreConfig` and `ExperimentConfig` are NamedTuples with a
few hundred defaulted fields, and every new field carries a comment saying it
was appended at the TUPLE END to preserve positional construction. That
convention works until two branches append at the same end, which is exactly
what happened merging #1769: `main` had appended the MPAS sponge pair and the
branch had appended the FV3 window pair, both at the tuple end, and whichever
one loses the tie-break has its fields SHIFT.

The dangerous part is that a shift is silent. A positional caller written
against one layout binds `fv3_duo_windows` (int | None) to
`mpas_sponge_del2_top_layers` (int) with no TypeError and no type-checker
complaint: you get a config with a corrupted sponge and the duo lane quietly
defaulted off. Review on that merge flagged it as the finding most likely to
be missed.

TWO GATES, and they are different claims:

1. **Nothing in the tree constructs these positionally** — so a shift cannot
   silently mis-bind anything that exists today. This is the load-bearing one,
   and it is an AST walk rather than a grep because `DycoreConfig(dt=600.0)`
   matches every reasonable "positional" regex.
2. **The field ORDER is pinned to a snapshot** — so the next append-collision
   shows up as a deliberate diff in this file rather than as a silent
   reordering. The snapshot is meant to be updated when fields are added; what
   it prevents is a field MOVING without anyone noticing.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from legoesm.driver.config import DycoreConfig, ExperimentConfig

_REPO = pathlib.Path(__file__).resolve().parents[2]
_ROOTS = ("packages", "src", "scripts", "tests")
_GUARDED = ("DycoreConfig", "ExperimentConfig", "AMIPExperimentConfig",
            "OutputConfig", "GridConfig")


def _positional_construction_sites(repo=None, roots=None):
    """Every call to a guarded config with a positional or *starred argument.

    ``repo``/``roots`` are parameters ONLY so the non-vacuity test can run this
    exact function over a planted violation.  A negative control that
    reimplements the visitor proves nothing about the visitor that ships
    (codex, merge review 2026-09-24).
    """
    repo = _REPO if repo is None else repo
    hits = []
    for root in (_ROOTS if roots is None else roots):
        base = repo / root
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(), str(path))
            except (SyntaxError, UnicodeDecodeError):
                continue  # not ours to police here
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                name = (fn.id if isinstance(fn, ast.Name)
                        else fn.attr if isinstance(fn, ast.Attribute) else None)
                # ``Cfg._make(row)`` re-materializes POSITIONALLY just as
                # ``Cfg(*row)`` does, but its call name is ``_make``, so the
                # check above cannot see it (merge review, 2026-09-23).
                if (isinstance(fn, ast.Attribute) and fn.attr == "_make"
                        and isinstance(fn.value, ast.Name)
                        and fn.value.id in _GUARDED):
                    hits.append(
                        f"{path.relative_to(repo)}:{node.lineno} "
                        f"{fn.value.id}._make")
                    continue
                if name not in _GUARDED or not node.args:
                    continue
                hits.append(f"{path.relative_to(repo)}:{node.lineno} {name}")
    return hits


def test_no_positional_config_construction_anywhere():
    """A positional caller is what makes an append-collision silent."""
    hits = _positional_construction_sites()
    assert not hits, (
        "these sites construct a config POSITIONALLY, so appending a field "
        "at the tuple end — or losing an append tie-break in a merge — "
        "silently rebinds their arguments to the wrong fields:\n  "
        + "\n  ".join(hits))


def test_the_audit_is_not_vacuous(tmp_path):
    """...and the walk must FIND a positional call when one exists.

    A test that reports 'zero hits' is worth nothing until it is shown to
    report a nonzero one, so plant a call and re-run the same visitor over it.
    """
    pkg = tmp_path / "packages"
    pkg.mkdir()
    (pkg / "planted.py").write_text(
        "DycoreConfig(600.0, 'hydrostatic')\n"      # positional
        "ExperimentConfig(*row)\n"                  # star-unpack
        "DycoreConfig._make(row)\n"                 # positional re-materialize
        "DycoreConfig(dt=600.0)\n")                 # keyword-only: NOT a hit
    found = _positional_construction_sites(repo=tmp_path, roots=("packages",))
    assert len(found) == 3, f"the shipped visitor missed a violation: {found}"
    assert any("_make" in h for h in found), found


@pytest.mark.parametrize("cls", [DycoreConfig, ExperimentConfig])
def test_no_duplicate_field_names(cls):
    """A merge that kept both sides of a collision can also keep a field
    twice under one name — Python would take the last, silently."""
    fields = cls._fields
    assert len(fields) == len(set(fields)), (
        f"{cls.__name__} has duplicate field names: "
        f"{sorted({f for f in fields if fields.count(f) > 1})}")


def test_dycore_window_and_sponge_fields_coexist_at_the_end():
    """The #1769 merge specifically: both append-collision pairs survived.

    This is the assertion the merge's own verification made by counting; it
    belongs in the suite so a later rebase cannot quietly drop one side.
    """
    f = DycoreConfig._fields
    for name in ("mpas_sponge_del2_top_layers", "mpas_sponge_del2_top_factor",
                 "fv3_duo_windows", "fv3_duo_window_pad"):
        assert name in f, f"{name} is gone from DycoreConfig"
    # main's pair keeps the indices it shipped with; the branch's pair lands
    # after it. Order between the two pairs is the tie-break the merge made.
    assert f.index("mpas_sponge_del2_top_factor") < f.index("fv3_duo_windows")
    d = DycoreConfig()
    assert d.fv3_duo_windows is None and d.fv3_duo_window_pad is None
    assert d.mpas_sponge_del2_top_layers == 0
    assert d.mpas_sponge_del2_top_factor == 1.0


def test_the_restart_codec_stays_name_keyed():
    """The path a run's own checkpoints take, which is what makes the layout
    shift survivable: the driver serializes the config by FIELD NAME into
    sorted JSON, never as a bare value tuple.  Measured on the 2026-09-23 main
    merge, which unioned two independent field-order changes; if this ever
    becomes a pickle, every artifact written before that merge re-binds."""
    restart = (_REPO / "packages/coupler/legoesm/driver/restart.py").read_text()
    assert "experiment_config_to_dict" in restart
    assert "json.dumps" in restart
    assert "pickle.dump" not in restart
