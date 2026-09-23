"""The driver configs are name-keyed everywhere, so field ORDER is not an ABI.

Both reviewers of the 2026-09-23 main merge raised the same hazard: a
``NamedTuple`` re-materializes POSITIONALLY on ``pickle.loads`` and on
``cls._make(row)`` / ``cls(*row)``, regardless of how it was constructed.  The
merge unions two field-order changes (main inserted five ``cloud_cap_floor_*``
fields mid-tuple; the campaign branch moved ``cld_macmic_num_steps`` to the
tail), so an artifact written before it and re-materialized positionally after
it would silently re-bind every field from the first change onward -- scalars
into scalar slots, no exception.

The measurement that settles it is that NOTHING re-materializes these configs
positionally: the restart codec goes through ``experiment_config_to_dict`` /
``_asdict`` into sorted JSON.  That was true when it was measured by hand; this
test is the gate that keeps it true, per the repo rule that a checked property
without a gate is prose.
"""
import ast
import pathlib

import pytest

_CONFIGS = {"ExperimentConfig", "DycoreConfig", "AMIPExperimentConfig",
            "OutputConfig", "GridConfig"}
_ROOT = pathlib.Path(__file__).resolve().parents[2]
_TREES = [_ROOT / "packages", _ROOT / "scripts", _ROOT / "src"]


def _sources():
    for tree in _TREES:
        if tree.is_dir():
            yield from sorted(tree.rglob("*.py"))


def _offences(path):
    try:
        tree = ast.parse(path.read_text(), str(path))
    except (SyntaxError, UnicodeDecodeError):
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # cls(*row) / cls(positional, ...)
        name = getattr(func, "id", None)
        if name in _CONFIGS:
            if any(isinstance(a, ast.Starred) for a in node.args):
                yield node.lineno, f"{name}(*...) star-unpack"
            elif node.args:
                yield node.lineno, f"{name}(...) with {len(node.args)} positional arg(s)"
        # cls._make(row)
        if (isinstance(func, ast.Attribute) and func.attr == "_make"
                and getattr(func.value, "id", None) in _CONFIGS):
            yield node.lineno, f"{func.value.id}._make(...)"


def test_no_positional_rematerialization_of_driver_configs():
    found = [f"{p.relative_to(_ROOT)}:{line}: {what}"
             for p in _sources() for line, what in _offences(p)]
    assert not found, (
        "driver configs must stay name-keyed -- field order is not an ABI and "
        "the merge history has reordered it:\n  " + "\n  ".join(found))


def test_the_gate_is_not_vacuous(tmp_path):
    """A planted violation of each shape must be caught."""
    for snippet, needle in (
            ("ExperimentConfig(1, 2)", "positional"),
            ("ExperimentConfig(*row)", "star-unpack"),
            ("DycoreConfig._make(row)", "_make"),
    ):
        planted = tmp_path / "planted.py"
        planted.write_text(snippet + "\n")
        hits = list(_offences(planted))
        assert hits, f"gate missed {snippet!r}"
        assert any(needle in what for _, what in hits), (snippet, hits)


def test_the_restart_codec_is_name_keyed():
    """The path the running arms' checkpoints actually take."""
    restart = (_ROOT / "packages/coupler/legoesm/driver/restart.py").read_text()
    assert "experiment_config_to_dict" in restart
    assert "json.dumps" in restart
    assert "pickle.dump" not in restart


@pytest.mark.parametrize("cls_name", sorted(_CONFIGS))
def test_configs_expose_asdict(cls_name):
    """Name-keyed serialization has to exist for the rule above to be usable."""
    from legoesm.driver import config as _cfg
    from legoesm.forcing import amip_config as _amip
    cls = getattr(_cfg, cls_name, None) or getattr(_amip, cls_name, None)
    if cls is None:
        pytest.skip(f"{cls_name} not exported")
    assert hasattr(cls, "_asdict") and hasattr(cls, "_fields")
