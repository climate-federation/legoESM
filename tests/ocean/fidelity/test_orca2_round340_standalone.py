from __future__ import annotations

import inspect
import os
import struct
from collections import OrderedDict, namedtuple
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean import bulk_flux_omip as bulk
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round340_standalone_gate as gate,
)
from tests.ocean.fidelity.test_nemo_testcase_l4_orca2_round337_surface import (
    _production_surface_case,
)

jax.config.update("jax_enable_x64", True)
set_policy(PrecisionPolicy.fp64(transcendentals="libm"))


def _k0():
    return bulk.NemoStandaloneK0Config(1, False, True, 0, False)


def _run_with_wind(u_wind):
    args, s0, ice_if, s2 = _production_surface_case()
    one = jnp.ones_like(s2.surface_mask)
    return bulk.nemo_orca2_k0_surface_card(
        u_wind, *args[1:5], *args[8:],
        k0_config=_k0(), ice_if=ice_if, s0=s0, s2=s2,
        chl=one, geothermal_heat_flux_mwm2=one,
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
        *args[:5], *args[8:], k0_config=_k0(), ice_if=ice_if, s0=s0, s2=s2,
        chl=jnp.ones_like(s2.surface_mask),
        geothermal_heat_flux_mwm2=jnp.ones_like(s2.surface_mask))
    assert called["count"] == 1
    assert result.bulk.fr_i is not None
    assert result.s2 is not None
    np.testing.assert_array_equal(
        np.asarray(result.surface_forcing.ice_concentration),
        np.asarray(result.ice_if.fr_i)[1:-1, 1:-1],
    )
    np.testing.assert_array_equal(
        np.asarray(result.surface_forcing.chl), np.ones(s2.surface_mask.shape))
    np.testing.assert_array_equal(
        np.asarray(result.surface_forcing.geothermal_heat_flux_mwm2),
        np.ones(s2.surface_mask.shape),
    )


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
        (f"kt01/{name}", OrderedDict((field, np.asarray([value], dtype=np.float64))
                           for field in ("a", "b")))
        for name in gate.SURFACE_ORDER
    )


def test_surface_first_then_unchanged_rung10_bar_and_plants():
    expected = _boundaries()
    actual = _boundaries()
    rung_expected = OrderedDict(
        (f"kt{kt:02d}/{checkpoint}", OrderedDict(
            T=np.zeros(2), S=np.ones(2), u=np.zeros(2), v=np.zeros(2), ssh=np.zeros(2)))
        for kt in range(1, 11) for checkpoint in ("entry", "stage1", "stage2", "stage3")
    )
    rung_actual = OrderedDict(
        (name, OrderedDict((field, value.copy()) for field, value in fields.items()))
        for name, fields in rung_expected.items()
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(gate, "load_oracle_boundaries", lambda _root: expected)
    monkeypatch.setattr(gate, "load_oracle_state_ladder", lambda _root: rung_expected)
    result = gate.score_composed_card(actual, Path("unused"), rung_actual)
    assert result["status"] == "PASS_S4_EXACT"

    for boundary in expected:
        planted = _boundaries()
        planted[boundary]["a"][0] = np.nextafter(0.0, np.inf)
        with pytest.raises(gate.GateError, match="surface exact bar failed"):
            gate.score_composed_card(
                planted, Path("unused"), rung_actual)
    # The established rung-10 bar accepts a one-ULP non-bit row.
    rung_actual["kt01/entry"]["T"][0] = np.nextafter(0.0, np.inf)
    assert gate.score_composed_card(actual, Path("unused"), rung_actual)[
        "rung10"]["kt01/entry/T"]["classification"] == "AT_BAR_NOT_EXACT"
    rung_actual["kt01/entry"]["T"][0] = 2.0e-15
    with pytest.raises(gate.GateError, match="rung-10 bar failed"):
        gate.score_composed_card(actual, Path("unused"), rung_actual)
    monkeypatch.undo()


def test_no_nemo_exchange_record_api():
    signature = inspect.signature(bulk.nemo_orca2_k0_surface_card)
    assert all("record" not in name and "nemo" not in name
               for name in signature.parameters if name != "k0_config")
    gate.validate_k0_selectors(gate.K0_SELECTORS)


def test_native_record_parser_checksum_adjacent_controls(tmp_path):
    path = tmp_path / "oracle_r340_surface_k00000001_b01_r0000.bin"
    header = (
        b"NEMO_R340_S4_3  "
        + struct.pack("=7i", 3, 1, 1, 0, 2, 2, 64)
    )
    path.write_bytes(header + np.zeros(28, dtype="=f8").tobytes())
    assert gate.inspect_exact_record(path) == {
        "kind": "surface", "kt": 1, "kid": 1, "rank": 0,
    }

    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(gate.GateError, match="truncated binary64 payload"):
        gate.inspect_exact_record(path)
    path.write_bytes(header + np.full(28, np.nan, dtype="=f8").tobytes())
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


def test_s0f_v3_payload_contains_owned_cells_only(tmp_path):
    path = tmp_path / "oracle_r340_fld_k00000001_c001_f001_r0000.bin"
    header = (
        b"NEMO_R340_FLDRD1"
        + struct.pack("=9i", 3, 1, 1, 1, 0, 0, 1, 2, 64)
        + b"tair".ljust(34)
        + struct.pack("=4i", 1, 2, 3, 4)
        + struct.pack("=d", 0.0)
        + struct.pack("=11i", 1, 1, 1, 4, 5, 1, 2, 3, 2, 4, 6)
        + np.arange(6, dtype="=f8").tobytes()
        + struct.pack("=14i", 1, 1, 1, 1, 1, 4, 5, 1, 2,
                      2, 3, 2, 4, 12)
        + np.arange(12, dtype="=f8").tobytes()
    )
    path.write_bytes(header)
    fields = gate.read_exact_record(path)["fields"]
    assert fields["fnow"].shape == (3, 2, 1)
    assert fields["fdta"].shape == (3, 2, 1, 2)


def test_composed_card_steps_a_three_dimensional_ocean_state():
    """Production binding plant: the result is a 3-D ocean step, not q_net."""

    deck = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
    if not deck.exists():
        pytest.skip("ORCA2 immutable input deck is not installed")
    card = bulk.build_nemo_orca2_k0_ocean_card(deck)
    shape = (2, 3)
    one = jnp.ones(shape, dtype=jnp.float64)
    zero = jnp.zeros(shape, dtype=jnp.float64)
    State = namedtuple(
        "State", "T S u v eta H_bathy land_mask u_before v_before")
    tiny_e3t = jnp.ones(shape + (2,), dtype=jnp.float64)
    recipe = card.ocean.recipe
    card = card._replace(ocean=card.ocean._replace(recipe=recipe._replace(
        z_coord=recipe.z_coord._replace(nemo_e3t_0=tiny_e3t))))
    state = State(
        Field(jnp.ones(shape + (2,), dtype=jnp.float64)),
        Field(34.0 * jnp.ones(shape + (2,), dtype=jnp.float64)),
        Field(jnp.zeros((2, 4, 2), dtype=jnp.float64)),
        Field(jnp.zeros((3, 3, 2), dtype=jnp.float64)),
        Field(zero), Field(one), Field(one), None, None,
    )
    _, old_s0, old_iif, s2 = _production_surface_case()
    means = bulk.NemoSurfaceMeans(*(zero for _ in range(7)))
    s0 = old_s0._replace(
        prior=means, u_before=zero, v_before=zero, temperature_now=zero,
        salinity_now=zero, ssh_now=zero, ssh_inverse_barometer=zero,
        e3t_reference=one, r3t_now=zero, surface_mask=one,
        shortwave_fraction_top=one,
    )
    iif = old_iif._replace(
        fdta_before=0.25 * one, fdta_after=0.75 * one,
        sss_m=zero, sst_m=zero, ts_surface=zero, ff_f=one,
        tmask_surface=one, smask0=one,
    )
    fields = bulk.NemoStandaloneAtmosphereFields(
        6 * one, -2 * one, 286 * one, 0.006 * one, 100800 * one,
        125 * one, 310 * one, 2.3e-5 * one, 3.0e-6 * one,
    )
    pair = bulk.NemoStandaloneFieldPair(zero, zero, 0, -10, 10)
    s0f = bulk.NemoStandaloneS0FInputs(fields, fields, 0, -10, 10, pair, pair)

    class ThreeDimensionalOcean:
        def step(self, live, dt, *, freshwater, surface_forcing, **_):
            increment = surface_forcing.q_net[..., None] * jnp.asarray(dt) * 1.0e-12
            return live._replace(T=Field(live.T.data + increment))

    def objective(wind):
        moved = fields._replace(u_wind=wind)
        forcing = s0f._replace(before=moved, after=moved)
        result = bulk.nemo_orca2_k0_ocean_step(
            card, state, s0f=forcing, ice_if=iif, s0=s0, s2=s2,
            chl=one, geothermal_heat_flux_mwm2=one,
            model=ThreeDimensionalOcean(), nemo_kt=1,
        )
        assert result.ocean.T.data.ndim == 3
        return jnp.sum(result.ocean.T.data)

    direction = jnp.linspace(0.5, 1.0, 6, dtype=jnp.float64).reshape(shape)
    _, forward = jax.jvp(objective, (fields.u_wind,), (direction,))
    reverse = jnp.vdot(jax.grad(objective)(fields.u_wind), direction)
    assert np.isfinite(np.asarray(forward)) and np.isfinite(np.asarray(reverse))
    np.testing.assert_allclose(reverse, forward, rtol=2.0e-14, atol=2.0e-14)

    Trace = namedtuple("Trace", "state_after stage_outputs")

    class TenStepOcean:
        def __init__(self):
            self.calls = 0

        def step(self, live, dt, *, freshwater, surface_forcing, **_):
            self.calls += 1
            increment = surface_forcing.q_net[..., None] * jnp.asarray(dt) * 1.0e-12
            after = live._replace(T=Field(live.T.data + increment))
            stage = (after.u.data, after.v.data, after.T.data,
                     after.S.data, after.eta.data)
            return Trace(after, (stage, stage, stage))

    model = TenStepOcean()
    controls = {
        "s0f": s0f, "ice_if": iif, "s0": s0, "s2": s2,
        "chl": one, "geothermal_heat_flux_mwm2": one,
    }
    replay = gate.run_composed_card(
        card, state, (controls.copy() for _ in range(10)), model=model)
    assert model.calls == 10
    assert len(replay["rung10"]) == 40
    assert sum(len(row) for row in replay["rung10"].values()) == 200
    assert "kt01/S0_ENTRY" in replay["surface_actual"]
    assert "kt01/S1_TURB" in replay["surface_actual"]
    assert "kt10/S3_EXIT_IIF" in replay["surface_actual"]
    assert replay["state_after"].T.data.ndim == 3


@pytest.mark.skipif(
    os.environ.get("ORCA2_R340_REAL_SMOKE") != "1",
    reason="operator-only real rung-10 CPU smoke",
)
def test_real_rung10_composed_card_ten_steps_cpu():
    """M4 plant: real card + real model + deck sources, never the 2x3 stub."""

    deck = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
    source = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
        "round5/acquisition/orca1ice_surface_entry_every_step_a_np2")
    k0 = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
        "round339/k0_smoke/orca2_k0_10step_np2")
    card = bulk.build_nemo_orca2_k0_ocean_card(deck)
    controls = gate.build_real_composed_inputs(
        card, deck, source / "input_files.sha256", k0 / "ice_cover_clim.nc")
    assert [int(item["s0f"].isecsbc) for item in controls] == [
        10800 * kt for kt in range(1, 11)]
    np.testing.assert_array_equal(
        np.asarray(controls[0]["s0f"].before.u_wind),
        np.asarray(controls[1]["s0f"].before.u_wind))
    assert not np.array_equal(
        np.asarray(controls[0]["s0f"].before.u_wind),
        np.asarray(controls[2]["s0f"].before.u_wind))
    assert not np.array_equal(
        np.asarray(controls[0]["s0f"].before.qsr_down),
        np.asarray(controls[8]["s0f"].before.qsr_down))
    replay = gate.run_composed_card(card, card.ocean.recipe.initial_state, controls)
    assert len(replay["surface_steps"]) == 10
    assert replay["surface_actual"]
    assert "kt01/S0_F/c001/f001/votemper" in replay["surface_actual"]
    assert "kt01/S0_F/c002/f006/scale_cri" in replay["surface_actual"]
    assert "kt01/S0_F/c004/f001/U_10_MOD" in replay["surface_actual"]
    assert "kt01/S0_F/c009/f002/vosaline" in replay["surface_actual"]
    assert "kt10/S0_F/c002/f002/vosaline" in replay["surface_actual"]
    assert replay["surface_actual"][
        "kt01/S0_F/c001/f001/votemper"]["fnow"].shape == (148, 180, 31)
    assert replay["surface_actual"][
        "kt01/S0_F/c004/f001/U_10_MOD"]["isecsbc"].tolist() == [10800, 10800]
    assert len(replay["rung10"]) == 40
    for leaf in jax.tree_util.tree_leaves(replay["state_after"]):
        if leaf is not None:
            assert np.isfinite(np.asarray(leaf)).all()


@pytest.mark.skipif(
    os.environ.get("ORCA2_R340_REAL_GRADIENT") != "1",
    reason="operator-only real composed-card CPU gradient",
)
def test_real_composed_card_n1_reverse_equals_forward():
    """M4: one real surface+ocean step, reverse AD against a forward JVP."""

    deck = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
    source = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
        "round5/acquisition/orca1ice_surface_entry_every_step_a_np2")
    k0 = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
        "round339/k0_smoke/orca2_k0_10step_np2")
    card = bulk.build_nemo_orca2_k0_ocean_card(deck)
    controls = gate.build_real_composed_inputs(
        card, deck, source / "input_files.sha256", k0 / "ice_cover_clim.nc")
    control = controls[0]
    state = card.ocean.recipe.initial_state
    direction = jnp.asarray(1.0, dtype=jnp.float64)

    def objective(delta):
        atmosphere = control["s0f"].before
        moved = atmosphere._replace(u_wind=atmosphere.u_wind + delta)
        s0f = control["s0f"]._replace(before=moved, after=moved)
        result = bulk.nemo_orca2_k0_ocean_step(
            card, state, s0f=s0f, ice_if=control["ice_if"],
            s0=control["s0"], s2=control["s2"], chl=control["chl"],
            geothermal_heat_flux_mwm2=control["geothermal_heat_flux_mwm2"],
            sponge=control["sponge"], nemo_kt=1,
        )
        return jnp.sum(result.ocean.T.data)

    zero = jnp.asarray(0.0, dtype=jnp.float64)
    primal, forward = jax.jvp(objective, (zero,), (direction,))
    reverse = jax.grad(objective)(zero) * direction
    assert np.isfinite(np.asarray(primal))
    assert np.isfinite(np.asarray(forward))
    assert np.isfinite(np.asarray(reverse))
    assert float(np.abs(np.asarray(forward))) > 0.0
    np.testing.assert_allclose(reverse, forward, rtol=2.0e-12, atol=2.0e-12)
