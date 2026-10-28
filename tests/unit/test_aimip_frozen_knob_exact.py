"""AIMIP classical knobs: a frozen knob runs at exactly its declared value (no
edge pull-in); a trained knob keeps the inward pull-in for its start (user
2026-10-28).  Decks declare the Sundqvist rain rate explicitly."""
from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest
import yaml

from legoesm.training.aimip_params import (
    AIMIP_CLASSICAL_CONSTRAINTS,
    AIMIPClassicalParams,
    aimip_declared_knobs,
    declare_sundqvist_rate,
)

jax.config.update("jax_enable_x64", True)
_C = {c.name: c for c in AIMIP_CLASSICAL_CONSTRAINTS}
_ROOT = Path(__file__).resolve().parents[2]


def _value(params, name):
    return float(params.as_dict()[name])


def _ulps(a, b, limit=8):
    """float64 ulps between a and b, counted with nextafter (``limit`` + 1 if
    further apart)."""
    a, b, n = float(a), float(b), 0
    while a != b and n <= limit:
        a, n = float(np.nextafter(a, b)), n + 1
    return n


def test_ulp_counter():
    x = 1.5e-4
    assert _ulps(x, x) == 0
    assert _ulps(x, np.nextafter(np.nextafter(x, 1.0), 1.0)) == 2
    assert _ulps(x, x * (1 + 1e-7)) > 8


def test_frozen_rain_rate_runs_at_its_declared_value_trained_starts_pulled_in():
    c = _C["sundqvist_auto_rate"]
    others = [n for n in _C if n != c.name]
    frozen = AIMIPClassicalParams.from_defaults(
        trainable=others, declared={c.name: 1.5e-4})
    trained = AIMIPClassicalParams.from_defaults(
        trainable=[c.name], declared={c.name: 1.5e-4})
    # the float64 logit round trip costs at most 2 ulps (user 2026-10-28)
    assert _ulps(_value(frozen, c.name), 1.5e-4) <= 2
    pulled = c.min_val + 0.05 * (c.max_val - c.min_val)
    np.testing.assert_allclose(_value(trained, c.name), pulled, rtol=1e-6)


def test_frozen_knob_at_a_bound_runs_at_the_bound():
    c = _C["sundqvist_auto_rate"]
    p = AIMIPClassicalParams.from_defaults(trainable=[], declared={c.name: c.min_val})
    assert _value(p, c.name) == c.min_val


def test_declared_values_are_checked():
    with pytest.raises(ValueError, match="unknown"):
        AIMIPClassicalParams.from_defaults(trainable=[], declared={"nope": 1.0})
    c = _C["sundqvist_auto_rate"]
    with pytest.raises(ValueError, match="outside"):
        AIMIPClassicalParams.from_defaults(trainable=[], declared={c.name: 2 * c.max_val})


def test_every_sundqvist_aimip_deck_declares_the_rain_rate():
    decks = [p for p in (_ROOT / "config").rglob("*.yaml")
             if "aimip_microphysics" in p.read_text()]
    hits = 0
    for p in decks:
        d = yaml.safe_load(p.read_text()) or {}
        d = {**d, **(d.get("cfg_overrides") or {})}
        if d.get("aimip_microphysics") == "sundqvist":
            hits += 1
            assert aimip_declared_knobs(d) == {"sundqvist_auto_rate": 1.5e-4}, p
    assert hits == 57


def test_generators_name_the_rain_rate_only_for_sundqvist():
    assert declare_sundqvist_rate({"aimip_microphysics": "sundqvist"})[
        "aimip_sundqvist_auto_rate"] == 1.5e-4
    assert "aimip_sundqvist_auto_rate" not in declare_sundqvist_rate(
        {"aimip_microphysics": "kessler"})



# --- Checkpoints (user 2026-10-28): frozen knobs are stored as float64 and
# come back as the exact Python float; no fp32 rounding at build or on disk.

def _frozen_rate_params():
    c = _C["sundqvist_auto_rate"]
    return c, AIMIPClassicalParams.from_defaults(
        trainable=[n for n in _C if n != c.name], declared={c.name: 1.5e-4})


def test_a_frozen_knob_is_built_as_its_float64_logit():
    from legoesm.training.aimip_params import _exact_logit
    c, p = _frozen_rate_params()
    assert p.raw_values[c.name] == _exact_logit(1.5e-4, c.min_val, c.max_val)
    assert type(p.raw_values[c.name]) is float


def test_frozen_knobs_round_trip_a_checkpoint_exactly(tmp_path):
    from legoesm.ml.training import load_checkpoint, save_checkpoint
    c, p = _frozen_rate_params()
    path = tmp_path / "c.eqx"
    save_checkpoint(p, path)
    _, template = _frozen_rate_params()
    back = load_checkpoint(template, path)
    frozen = [k for k, v in p.raw_values.items() if type(v) is float]
    assert c.name in frozen
    for k in frozen:
        assert type(back.raw_values[k]) is float, k
        assert back.raw_values[k] == p.raw_values[k], k


def test_a_frozen_checkpoint_loads_into_an_all_array_template(tmp_path):
    from legoesm.ml.training import load_checkpoint, save_checkpoint
    c, p = _frozen_rate_params()
    path = tmp_path / "c.eqx"
    save_checkpoint(p, path)
    template = AIMIPClassicalParams.from_defaults()
    back = load_checkpoint(template, path)
    dt = template.raw_values[c.name].dtype
    assert back.raw_values[c.name].dtype == dt
    assert float(back.raw_values[c.name]) == float(
        np.asarray(p.raw_values[c.name], dtype=dt))


# --- Knob record beside each checkpoint (user 2026-10-28, R1 + Q): frozen set
# and ranges; a TRAINED knob never narrows silently into a template.

def _with_policy(policy, fn):
    from legoesm.core.precision import get_policy, set_policy
    old = get_policy()
    set_policy(policy)
    try:
        return fn()
    finally:
        set_policy(old)


def test_save_writes_a_versioned_knob_record(tmp_path):
    import json

    from legoesm.ml.checkpoint_io import (
        KNOB_RECORD_FORMAT, KNOB_RECORD_VERSION, knob_record_path,
    )
    from legoesm.ml.training import save_checkpoint
    c, p = _frozen_rate_params()
    save_checkpoint(p, tmp_path / "c.eqx")
    rec = json.loads(knob_record_path(tmp_path / "c.eqx").read_text())
    assert (rec["format"], rec["version"]) == (KNOB_RECORD_FORMAT, KNOB_RECORD_VERSION)
    assert c.name in rec["frozen"]
    assert rec["ranges"][c.name] == [c.min_val, c.max_val]
    assert set(rec["ranges"]) == set(_C)


def _trained_fp64_checkpoint(tmp_path):
    from legoesm.core.precision import PrecisionPolicy
    from legoesm.ml.training import save_checkpoint
    p = _with_policy(PrecisionPolicy.fp64(),
                     lambda: AIMIPClassicalParams.from_defaults(trainable=list(_C)))
    assert p.raw_values["sundqvist_auto_rate"].dtype == np.float64
    save_checkpoint(p, tmp_path / "c.eqx")
    return tmp_path / "c.eqx"


def test_a_trained_fp64_knob_does_not_narrow_into_an_fp32_template(tmp_path):
    from legoesm.core.precision import PrecisionPolicy
    from legoesm.ml.checkpoint_io import CheckpointStructureError
    from legoesm.ml.training import load_checkpoint
    path = _trained_fp64_checkpoint(tmp_path)
    tmpl = _with_policy(PrecisionPolicy.fp32(), AIMIPClassicalParams.from_defaults)
    with pytest.raises(CheckpointStructureError, match="would narrow"):
        load_checkpoint(tmpl, path)


def test_a_trained_fp32_knob_widens_into_an_fp64_template(tmp_path):
    from legoesm.core.precision import PrecisionPolicy
    from legoesm.ml.training import load_checkpoint, save_checkpoint
    p = _with_policy(PrecisionPolicy.fp32(),
                     lambda: AIMIPClassicalParams.from_defaults(trainable=list(_C)))
    save_checkpoint(p, tmp_path / "c.eqx")
    tmpl = _with_policy(PrecisionPolicy.fp64(), AIMIPClassicalParams.from_defaults)
    back = load_checkpoint(tmpl, tmp_path / "c.eqx")
    v = back.raw_values["sundqvist_auto_rate"]
    assert v.dtype == np.float64
    assert float(v) == float(p.raw_values["sundqvist_auto_rate"])


def test_without_a_record_a_dtype_change_is_an_error(tmp_path):
    from legoesm.ml.checkpoint_io import CheckpointStructureError, knob_record_path
    from legoesm.ml.training import load_checkpoint, save_checkpoint
    c, p = _frozen_rate_params()
    save_checkpoint(p, tmp_path / "c.eqx")
    knob_record_path(tmp_path / "c.eqx").unlink()
    with pytest.raises(CheckpointStructureError, match="no knob record"):
        load_checkpoint(AIMIPClassicalParams.from_defaults(), tmp_path / "c.eqx")
    _, frozen_tmpl = _frozen_rate_params()   # same layout: nothing to classify
    back = load_checkpoint(frozen_tmpl, tmp_path / "c.eqx")
    assert back.raw_values[c.name] == p.raw_values[c.name]


def test_a_record_with_other_ranges_is_an_error(tmp_path):
    import json

    from legoesm.ml.checkpoint_io import CheckpointStructureError, knob_record_path
    from legoesm.ml.training import load_checkpoint, save_checkpoint
    c, p = _frozen_rate_params()
    save_checkpoint(p, tmp_path / "c.eqx")
    side = knob_record_path(tmp_path / "c.eqx")
    rec = json.loads(side.read_text())
    rec["ranges"][c.name] = [c.min_val, 2 * c.max_val]
    side.write_text(json.dumps(rec))
    with pytest.raises(CheckpointStructureError, match="other AIMIP knob ranges"):
        load_checkpoint(_frozen_rate_params()[1], tmp_path / "c.eqx")


def test_optimizer_moment_copies_carry_no_record():
    import equinox as eqx

    from legoesm.ml.checkpoint_io import knob_record
    c, p = _frozen_rate_params()
    moments = eqx.tree_at(lambda m: m.raw_values, p,
                          {k: (None if type(v) is float else v)
                           for k, v in p.raw_values.items()})
    assert moments.knob_record() is None
    assert knob_record((p, moments))["frozen"] == p.knob_record()["frozen"]


def test_raw_loaders_check_the_record_ranges(tmp_path):
    """Loaders that call equinox directly (trainer resume, EMA, WB) run the
    same range check first."""
    import json

    from legoesm.ml.checkpoint_io import check_knob_record, knob_record_path
    from legoesm.ml.training import save_checkpoint
    c, p = _frozen_rate_params()
    path = tmp_path / "c.eqx"
    save_checkpoint((p, 0), path)
    check_knob_record(path, (p, 0))
    side = knob_record_path(path)
    rec = json.loads(side.read_text())
    rec["ranges"][c.name] = [c.min_val, 2 * c.max_val]
    side.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="other AIMIP knob ranges"):
        check_knob_record(path, (p, 0))
    side.unlink()
    check_knob_record(path, (p, 0))
