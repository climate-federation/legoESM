"""Self-test for the cross-grid config-parity gate (provably non-vacuous).

Synthetic-violation doctrine: the gate must (a) pass on config-identical
manifests, (b) FAIL on a planted value drift, (c) catch a drift hidden behind
the flat<->nested alias (the bottom_drag_scheme signature that motivated the
tool), and (d) fold layout/one-sided differences instead of crying wolf.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "validate"))

from omip_config_parity import main as parity_main  # noqa: E402


def _manifest(tmp_path, name, runtime):
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(
        {"config": {"config_kind": name,
                    "resolved_config": {"grid": name.split("_")[0],
                                        "runtime_config": runtime}}}))
    return str(p)


_BASE = {
    "pgf_scheme": "smc03",
    "tracer_advection": "superbee",
    "S_ref": 35.0,
}


def test_identical_configs_pass(tmp_path):
    a = _manifest(tmp_path, "tripole_a", dict(_BASE))
    b = _manifest(tmp_path, "mpas_b", dict(_BASE))
    assert parity_main([a, b]) == 0


def test_value_drift_fails(tmp_path):
    a = _manifest(tmp_path, "tripole_a", dict(_BASE))
    b = _manifest(tmp_path, "mpas_b", {**_BASE, "tracer_advection": "tvd"})
    assert parity_main([a, b]) == 1


def test_aliased_drift_caught(tmp_path):
    # lat-lon nests the drag under bottom_drag.*; MPAS keeps it flat.  A
    # differing value must be caught THROUGH the alias (the real mpas8 bug).
    a = _manifest(tmp_path, "tripole_a",
                  {**_BASE,
                   "bottom_drag": {"bottom_drag_scheme": "nemo_quadratic"}})
    b = _manifest(tmp_path, "mpas_b",
                  {**_BASE, "bottom_drag_scheme": "legacy"})
    assert parity_main([a, b]) == 1


def test_aliased_equal_folds(tmp_path):
    a = _manifest(tmp_path, "tripole_a",
                  {**_BASE,
                   "bottom_drag": {"bottom_drag_scheme": "nemo_quadratic"}})
    b = _manifest(tmp_path, "mpas_b",
                  {**_BASE, "bottom_drag_scheme": "nemo_quadratic"})
    assert parity_main([a, b]) == 0


def test_one_sided_class_field_folds(tmp_path):
    # A knob that exists on only one config class is a field-set difference,
    # not a value drift — must not fail the gate.
    a = _manifest(tmp_path, "tripole_a",
                  {**_BASE, "weno_d_term": True})
    b = _manifest(tmp_path, "mpas_b", dict(_BASE))
    assert parity_main([a, b]) == 0


def test_declared_difference_passes_but_strict_fails(tmp_path):
    a = _manifest(tmp_path, "tripole_a",
                  {**_BASE, "physics": {"vertical_mixing": {"scheme": "tke"}}})
    b = _manifest(tmp_path, "mpas_b",
                  {**_BASE, "physics": {"vertical_mixing": {"scheme": "kpp"}}})
    assert parity_main([a, b]) == 0
    assert parity_main([a, b, "--strict-declared"]) == 1
