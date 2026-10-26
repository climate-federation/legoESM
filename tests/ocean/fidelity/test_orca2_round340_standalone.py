from __future__ import annotations

import inspect
import struct
from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean import bulk_flux_omip as bulk
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round340_standalone_gate as gate,
)
from tests.ocean.fidelity.test_nemo_testcase_l4_orca2_round337_surface import (
    _production_surface_case,
)

jax.config.update("jax_enable_x64", True)


def _k0():
    return bulk.NemoStandaloneK0Config(1, False, True, 0, False)


def _run_with_wind(u_wind):
    args, s0, ice_if, s2 = _production_surface_case()
    return bulk.nemo_orca2_k0_surface_card(
        u_wind, *args[1:5], *args[8:],
        k0_config=_k0(), ice_if=ice_if, s0=s0, s2=s2,
    )


def test_k0_is_explicit_and_calls_production_composition(monkeypatch):
    for parameter in inspect.signature(bulk.NemoStandaloneK0Config).parameters.values():
        assert parameter.default is inspect.Parameter.empty
    for field, wrong in (
        ("nn_ice", 0), ("ln_drgice_imp", True), ("ln_mxl0", False),
        ("nn_mxlice", 2), ("ln_icebergs", True),
    ):
        values = _k0()._asdict()
        values[field] = wrong
        with pytest.raises(ValueError, match="Decision-175 K0"):
            bulk.validate_nemo_standalone_k0(bulk.NemoStandaloneK0Config(**values))

    args, s0, ice_if, s2 = _production_surface_case()
    called = {"count": 0}
    original = bulk.nemo_blk_oce_ncar

    def planted(*call_args, **kwargs):
        called["count"] += 1
        return original(*call_args, **kwargs)

    monkeypatch.setattr(bulk, "nemo_blk_oce_ncar", planted)
    result = bulk.nemo_orca2_k0_surface_card(
        *args[:5], *args[8:], k0_config=_k0(), ice_if=ice_if, s0=s0, s2=s2)
    assert called["count"] == 1
    assert result.bulk.fr_i is not None
    assert result.s2 is not None


def test_composed_card_gradient_n1_reverse_equals_forward():
    args, *_ = _production_surface_case()
    u = args[0]
    direction = jnp.linspace(0.5, 1.0, u.size, dtype=jnp.float64).reshape(u.shape)

    def objective(value):
        result = _run_with_wind(value)
        return jnp.sum(result.surface_forcing.q_net)

    primal, forward = jax.jvp(objective, (u,), (direction,))
    reverse = jnp.vdot(jax.grad(objective)(u), direction)
    assert np.isfinite(np.asarray(primal))
    assert np.isfinite(np.asarray(forward))
    assert np.isfinite(np.asarray(reverse))
    np.testing.assert_allclose(reverse, forward, rtol=2.0e-14, atol=2.0e-14)


def _boundaries(value=0.0):
    return OrderedDict(
        (name, OrderedDict((field, np.asarray([value], dtype=np.float64))
                           for field in ("a", "b")))
        for name in gate.SURFACE_ORDER
    )


def test_surface_first_then_unchanged_rung10_bar_and_plants():
    expected = _boundaries()
    actual = _boundaries()
    rung_actual = OrderedDict(T=np.zeros(2), S=np.ones(2))
    rung_expected = OrderedDict(T=np.zeros(2), S=np.ones(2))
    result = gate.score_composed_card(actual, expected, rung_actual, rung_expected)
    assert result["status"] == "PASS_S4_EXACT"

    for boundary in gate.SURFACE_ORDER:
        planted = _boundaries()
        planted[boundary]["a"][0] = np.nextafter(0.0, np.inf)
        with pytest.raises(gate.GateError, match="surface exact bar failed"):
            gate.score_composed_card(
                planted, expected, rung_actual, rung_expected)
    rung_actual["T"][0] = np.nextafter(0.0, np.inf)
    with pytest.raises(gate.GateError, match="rung-10 bar failed"):
        gate.score_composed_card(actual, expected, rung_actual, rung_expected)


def test_no_nemo_exchange_record_api():
    signature = inspect.signature(bulk.nemo_orca2_k0_surface_card)
    assert all("record" not in name and "nemo" not in name
               for name in signature.parameters if name != "k0_config")
    gate.validate_k0_selectors(gate.K0_SELECTORS)


def test_native_record_parser_checksum_adjacent_controls(tmp_path):
    path = tmp_path / "oracle_r340_surface_k00000001_b00_r0000.bin"
    header = (
        b"NEMO_R340_S4_1  "
        + struct.pack("=7i", 1, 1, 0, 0, 2, 2, 64)
        + struct.pack("=4i", 1, 2, 1, 2)
    )
    path.write_bytes(header + np.zeros(68, dtype="=f8").tobytes())
    assert gate.inspect_exact_record(path) == {
        "kind": "surface", "kt": 1, "kid": 0, "rank": 0,
    }

    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(gate.GateError, match="truncated binary64 payload"):
        gate.inspect_exact_record(path)
    path.write_bytes(header + np.full(68, np.nan, dtype="=f8").tobytes())
    with pytest.raises(gate.GateError, match="non-finite"):
        gate.inspect_exact_record(path)


def test_native_state_time_level_plant(tmp_path):
    path = tmp_path / "oracle_r340_state_k00000001_s0_r0000.bin"
    # jpi=jpj=jpk=jpts=1 gives four binary64 payload words.
    data = (
        b"NEMO_R340_STATE1"
        + struct.pack("=10i", 1, 1, 0, 9, 0, 1, 1, 1, 1, 64)
        + np.zeros(4, dtype="=f8").tobytes()
    )
    path.write_bytes(data)
    with pytest.raises(gate.GateError, match="time level moved"):
        gate.inspect_exact_record(path)
