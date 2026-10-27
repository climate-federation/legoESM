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
    return bulk.NemoStandaloneK0Config(
        1, False, True, 0, False, "nemo_k0_istate"
    )


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
        ("initial_state", "si3_iceistate"),
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


def test_item1_north_fold_t_pivot_stress_lbc_and_vtauv():
    """Item 1: right-half T-pivot fold (91..179), row -2 north fold, and vmask[-1]."""
    from legoesm.ocean.forcing import nemo_surface

    deck = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
    if not deck.exists():
        pytest.skip("ORCA2 immutable input deck is not installed")
    card = bulk.build_nemo_orca2_k0_ocean_card(deck)
    state = card.ocean.recipe.initial_state
    mask = np.asarray(state.land_mask.data, dtype=np.float64)
    u_mask = np.asarray(state.u_mask.data, dtype=np.float64)
    v_mask = np.asarray(state.v_mask.data, dtype=np.float64)
    fold = card.ocean.recipe.grid.fold
    stress_lbc, tmask_east, tmask_north, umask, vmask = (
        gate._build_stress_lbc_and_masks(mask, u_mask, v_mask, fold)
    )
    ny, nx = mask.shape
    half = nx // 2
    center = np.arange(ny * nx, dtype=np.int64).reshape(ny, nx)
    perm_t = np.asarray(fold.perm_T, dtype=np.int64)
    perm_v = np.asarray(fold.perm_v, dtype=np.int64)

    # Right-half-only fold/sign on T-pivot row -1 (lbcnfd.f90:619-628).
    np.testing.assert_array_equal(
        np.asarray(stress_lbc.center_index)[-1, :half + 1],
        center[-1, :half + 1],
    )
    np.testing.assert_array_equal(
        np.asarray(stress_lbc.center_index)[-1, half + 1:],
        center[-1, perm_t[half + 1:]],
    )
    np.testing.assert_array_equal(
        np.asarray(stress_lbc.center_sign)[-1, :half + 1],
        np.ones(half + 1, dtype=np.float64),
    )
    np.testing.assert_array_equal(
        np.asarray(stress_lbc.center_sign)[-1, half + 1:],
        -np.ones(nx - (half + 1), dtype=np.float64),
    )
    # Row -1 north neighbor folds from row -2 (lbcnfd.f90:588-606).
    np.testing.assert_array_equal(
        np.asarray(stress_lbc.north_index)[-1],
        center[-2, perm_t],
    )
    np.testing.assert_array_equal(
        np.asarray(stress_lbc.north_sign)[-1],
        -np.ones(nx, dtype=np.float64),
    )
    # V-grid mask on row -1 folds from row -2 (dommsk.f90:232, lbcnfd.f90:688-706).
    np.testing.assert_array_equal(vmask[-1], vmask[-2, perm_v])

    arm_a = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
        "round340/s4_acquisition/orca2_s4_exact_a_np2"
    )
    if arm_a.exists():
        def join_surface(kid: int):
            r0 = gate.read_exact_record(
                arm_a / f"oracle_r340_surface_k00000001_b{kid:02d}_r0000.bin"
            )
            r1 = gate.read_exact_record(
                arm_a / f"oracle_r340_surface_k00000001_b{kid:02d}_r0001.bin"
            )
            return gate._join_rank_fields([r0, r1])

        s3_exit = join_surface(5)
        s2_lbc = join_surface(10)
        s2_exit = join_surface(11)
        utau_c, vtau_c, utau_e, vtau_n = (
            nemo_surface.nemo_t_stress_lbc_neighbors(
                jnp.asarray(s3_exit["utau"]),
                jnp.asarray(s3_exit["vtau"]),
                stress_lbc,
            )
        )
        utau_u, vtau_v = nemo_surface.nemo_t_stress_to_uv(
            utau_c, vtau_c, utau_e, vtau_n,
            jnp.asarray(mask), jnp.asarray(tmask_east), jnp.asarray(tmask_north),
            jnp.asarray(umask), jnp.asarray(vmask),
        )
        assert gate.exact_score(np.asarray(utau_c), s2_lbc["utau"])["unequal"] == 0
        assert gate.exact_score(np.asarray(vtau_c), s2_lbc["vtau"])["unequal"] == 0
        assert gate.exact_score(np.asarray(utau_u), s2_exit["utauU"])["unequal"] == 0
        assert gate.exact_score(np.asarray(vtau_v), s2_exit["vtauV"])["unequal"] == 0


def test_item2_even_kt_surface_boundaries_retain_prior_s2_exits():
    """Item 2: on even kt, NEMO retains kt-1 S2_EXIT across kid=3..10 and IIF."""
    args, s0, ice_if, s2 = _production_surface_case()
    shape = s2.surface_mask.shape
    one = jnp.ones(shape, dtype=jnp.float64)
    # Give s2 nonzero runoff, restoring, and FWB so tail differs from pre/post/ssr.
    s2 = s2._replace(
        runoff_now=2.5e-4 * one,
        sss_target=35.5 * one,
        freshwater_budget_correction=jnp.asarray(1.2e-5, dtype=jnp.float64),
    )
    step1 = bulk.nemo_orca2_k0_surface_card(
        *args[:5], *args[8:], k0_config=_k0(), ice_if=ice_if, s0=s0, s2=s2,
        chl=one, geothermal_heat_flux_mwm2=one,
    )
    s0_kt2 = s0._replace(kt=2)
    step2 = bulk.nemo_orca2_k0_surface_card(
        *args[:5], *args[8:], k0_config=_k0(), ice_if=ice_if, s0=s0_kt2, s2=s2,
        chl=one, geothermal_heat_flux_mwm2=one, prior_surface=step1,
    )
    State = namedtuple("State", "T")
    state_before = State(Field(jnp.zeros(shape + (2,), dtype=jnp.float64)))
    atmosphere = bulk.NemoStandaloneAtmosphereFields(*(one for _ in range(9)))
    pair = bulk.NemoStandaloneFieldPair(one, one, 0, -10, 10)
    controls = {
        "s0f": bulk.NemoStandaloneS0FInputs(
            atmosphere, atmosphere, 0, -10, 10, pair, pair),
        "ice_if": ice_if,
        "s2": s2,
    }
    frames_kt1 = gate._surface_boundaries_for_step(1, state_before, controls, step1)
    frames_kt2 = gate._surface_boundaries_for_step(2, state_before, controls, step2)

    s2_exit_kt1 = frames_kt1["kt01/S2_EXIT"]
    # Sanity check: on odd kt=1, S1_EXIT (pre) and S2_EXIT (tail) differ.
    assert not np.array_equal(
        frames_kt1["kt01/S1_EXIT"]["emp"], s2_exit_kt1["emp"]
    )
    assert not np.array_equal(
        frames_kt1["kt01/S1_EXIT"]["qns"], s2_exit_kt1["qns"]
    )
    # On even kt=2, kid=3..10 retain kt=1 S2_EXIT values for all non-mean fields.
    mean_names = set(gate.OUTER_FULL_FIELDS[:7])
    for kid in range(3, 11):
        boundary = f"kt02/{gate.OUTER_BOUNDARIES[kid]}"
        for name, actual in frames_kt2[boundary].items():
            if name in mean_names:
                continue
            np.testing.assert_array_equal(
                actual, s2_exit_kt1[name], err_msg=f"{boundary}/{name}"
            )
    for boundary in ("kt02/S3_COVER_IIF", "kt02/S3_EXIT_IIF"):
        for name in ("emp", "utau", "vtau", "qsr", "qns", "sfx", "taum", "wndm"):
            np.testing.assert_array_equal(
                frames_kt2[boundary][name],
                s2_exit_kt1[name],
                err_msg=f"{boundary}/{name}",
            )


def test_item3_shared_allocate_extent_gate_and_admit_order(tmp_path):
    """Item 3: shared NEMO-ALLOCATE extent gate and same_program_observers order."""
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round339_common as round339,
    )

    k0_run = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
        "round339/k0_smoke/orca2_k0_10step_np2"
    )
    if k0_run.exists():
        ppsrc = k0_run / "build_sources/ppsrc/nemo"
        verified = gate.verify_r340_allocated_extents(
            ppsrc, k0_run / "ocean.output"
        )
        assert verified["nn_hls"] == 2
        assert set(verified["units"]) == {
            "sbc_oce", "sbcrnf", "sbcssr", "dom_oce", "oce",
        }
        # Plant a non-qns A2D(0) variable (taum) changed to (jpi,jpj); the old
        # qns-only inline regex missed this, whereas the shared gate catches it.
        planted_ppsrc = tmp_path / "ppsrc"
        planted_ppsrc.mkdir()
        for unit in gate.R340_ALLOCATED_EXTENTS:
            text = (ppsrc / f"{unit}.f90").read_text(encoding="utf-8")
            if unit == "sbc_oce":
                import re
                text = re.sub(
                    r"taum\s*\(\s*Nis0-\(0\):Nie0\+\(0\)\s*,\s*Njs0-\(0\):Nje0\+\(0\)\s*\)",
                    "taum(jpi,jpj)",
                    text,
                    count=1,
                )
            (planted_ppsrc / f"{unit}.f90").write_text(text, encoding="utf-8")
        with pytest.raises(gate.GateError, match="ALLOCATE extent moved for taum"):
            gate.verify_r340_allocated_extents(
                planted_ppsrc, k0_run / "ocean.output"
            )
    else:
        sample = (
            "SUBROUTINE alloc()\n"
            " ALLOCATE( qns(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0)), &\n"
            "  & taum(jpi,jpj) )\n"
            "END SUBROUTINE alloc\n"
        )
        with pytest.raises(round339.GateError, match="ALLOCATE extent moved for taum"):
            round339.validate_allocated_extents(
                sample,
                {"qns": gate.A2D0_EXTENT, "taum": gate.A2D0_EXTENT},
            )

    run_sh = Path(
        "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round340_s4_acquisition/run.sh"
    ).read_text(encoding="utf-8")
    assert "ALLOCATE\\(\\s*qns" not in run_sh
    assert "verify_r340_allocated_extents" in run_sh
    admit_pair_pos = run_sh.index("admit_pair(){")
    admit_pair_body = run_sh[admit_pair_pos:run_sh.index("\n}\n", admit_pair_pos)]
    assert admit_pair_body.index("check_same_program_build") < admit_pair_body.index("admit_one")
    admit_existing_pos = run_sh.index("admit_existing_route(){")
    admit_existing_body = run_sh[
        admit_existing_pos:run_sh.index("\n}\n", admit_existing_pos)
    ]
    assert "check_same_program_build" in admit_existing_body
    assert admit_existing_body.index("check_same_program_build") < admit_existing_body.index("admit_one")


def test_scoring_driver_reproducibility_control_and_composed_card_flow(
    tmp_path, monkeypatch
):
    """Direct test for nemo_testcase_l4_orca2_round340_score_composed_card."""
    import json
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round340_score_composed_card as score_driver,
    )

    k0 = score_driver.explicit_k0_config()
    assert k0 == bulk.NemoStandaloneK0Config(
        1, False, True, 0, False, "nemo_k0_istate"
    )

    if score_driver.DEFAULT_ARM_A.exists() and score_driver.DEFAULT_ARM_B.exists():
        repro = score_driver.score_reproducibility_control(
            score_driver.DEFAULT_ARM_A, score_driver.DEFAULT_ARM_B
        )
        assert repro["status"] == "PASS_REPRODUCIBILITY_EXACT"
        assert repro["surface_unequal"] == 0
        assert repro["surface_count"] == 123824304
        assert repro["rung10_unequal"] == 0
        assert repro["rung10_rows"] == 200

    expected_surface = _boundaries()
    rung_expected = OrderedDict(
        (
            f"kt{kt:02d}/{checkpoint}",
            OrderedDict(
                T=np.zeros(2),
                S=np.ones(2),
                u=np.zeros(2),
                v=np.zeros(2),
                ssh=np.zeros(2),
            ),
        )
        for kt in range(1, 11)
        for checkpoint in ("entry", "stage1", "stage2", "stage3")
    )

    def clone_surface():
        return OrderedDict(
            (b, OrderedDict((k, v.copy()) for k, v in fields.items()))
            for b, fields in expected_surface.items()
        )

    def clone_rung():
        return OrderedDict(
            (c, OrderedDict((k, v.copy()) for k, v in fields.items()))
            for c, fields in rung_expected.items()
        )

    state = {"surface": clone_surface(), "rung10": clone_rung()}
    DummyRecipe = namedtuple("DummyRecipe", "initial_state")
    DummyOcean = namedtuple("DummyOcean", "recipe")
    DummyCard = namedtuple("DummyCard", "ocean k0")
    dummy_card = DummyCard(DummyOcean(DummyRecipe("init")), k0)

    monkeypatch.setattr(
        score_driver,
        "score_reproducibility_control",
        lambda arm_a, arm_b: {
            "status": "PASS_REPRODUCIBILITY_EXACT",
            "arm_a": str(arm_a),
            "arm_b": str(arm_b),
            "surface_unequal": 0,
            "surface_count": 34,
            "rung10_unequal": 0,
            "rung10_count": 400,
            "rung10_rows": 200,
        },
    )
    monkeypatch.setattr(
        bulk, "build_nemo_orca2_k0_ocean_card", lambda _deck: dummy_card
    )
    monkeypatch.setattr(bulk, "_validate_k0_ocean_card", lambda _card: None)
    monkeypatch.setattr(
        gate,
        "build_real_composed_inputs",
        lambda _card, _deck, _manifest, _ice: ("ctrl",) * 10,
    )
    monkeypatch.setattr(
        gate,
        "run_composed_card",
        lambda _card, _init, _ctrls, model=None: {
            "surface_actual": state["surface"],
            "rung10": state["rung10"],
        },
    )
    monkeypatch.setattr(
        gate, "load_oracle_boundaries", lambda _root: expected_surface
    )
    monkeypatch.setattr(
        gate, "load_oracle_state_ladder", lambda _root: rung_expected
    )

    out_pass = tmp_path / "pass.json"
    rc = score_driver.main(["--output-json", str(out_pass)])
    assert rc == 0
    payload = json.loads(out_pass.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS_S4_EXACT"
    assert payload["composed_card"]["surface"]["unequal"] == 0
    assert payload["composed_card"]["rung10"]["classifications"]["BIT_EXACT"] == 200

    # Plant a surface discrepancy and verify strict mode fails while --report-only
    # still writes the full surface + 200-row rung-10 table.
    state["surface"] = clone_surface()
    state["surface"]["kt01/S0_EXIT"]["a"][0] = 0.25
    state["rung10"] = clone_rung()
    state["rung10"]["kt01/entry"]["ssh"][0] = 0.25
    with pytest.raises(gate.GateError, match="surface exact bar failed") as exc_info:
        score_driver.run_and_score_real_composed_card(strict=True)
    assert exc_info.value.report is not None
    assert exc_info.value.report["status"] == "FAIL_S4_SURFACE"
    assert (
        exc_info.value.report["composed_card"]["surface"]["first_non_bit_row"]
        == "kt01/S0_EXIT/a"
    )
    assert (
        exc_info.value.report["composed_card"]["rung10"]["first_over_bar_row"]
        == "kt01/entry/ssh"
    )
    out_fail = tmp_path / "fail.json"
    assert score_driver.main(["--output-json", str(out_fail)]) == 1
    assert (
        score_driver.main(["--output-json", str(out_fail), "--report-only"])
        == 0
    )
    fail_payload = json.loads(out_fail.read_text(encoding="utf-8"))
    assert fail_payload["status"] == "FAIL_S4_SURFACE"
    assert (
        fail_payload["composed_card"]["surface"]["by_boundary"]["kt01/S0_EXIT"][
            "unequal"
        ]
        == 1
    )


def test_resume7_k0_istate_ssh_and_dry_cell_signed_zero_keep_si3_bit_identical():
    """Fixes 1 & 2: K0 istate starts from ssh=0 and preserves dry-cell -0.0 on T; SI3 stays bit-identical."""
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe

    sig = inspect.signature(recipe.build_orca2_istate_for_mode)
    assert sig.parameters["initial_state"].default is inspect.Parameter.empty
    with pytest.raises(ValueError, match="unknown ORCA2 initial_state"):
        recipe.build_orca2_istate_for_mode(
            "unused", "unused", np.ones((1, 1, 1), dtype=bool),
            f_t=np.ones((1, 1)), e1e2t=np.ones((1, 1)),
            initial_state="unknown_mode",
        )

    deck = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
    if not deck.exists():
        pytest.skip("ORCA2 immutable input deck is not installed")

    si3_card = recipe.build_orca2_zps_card(deck)
    k0_card = bulk.build_nemo_orca2_k0_ocean_card(deck)
    assert k0_card.k0.initial_state == "nemo_k0_istate"

    active = np.asarray(si3_card.recipe.z_coord.is_active, dtype=bool)
    wet_surface = active[..., 0]
    si3_eta = np.asarray(si3_card.recipe.initial_state.eta.data, dtype=np.float64)
    k0_eta = np.asarray(k0_card.ocean.recipe.initial_state.eta.data, dtype=np.float64)
    assert int(wet_surface.sum()) == 16433
    assert np.allclose(si3_eta[wet_surface], -0.093435058726849693, rtol=0.0, atol=1e-16)
    assert np.all(k0_eta == 0.0) and not np.any(np.signbit(k0_eta))

    si3_t = np.asarray(si3_card.recipe.initial_state.T.data, dtype=np.float64)
    k0_t = np.asarray(k0_card.ocean.recipe.initial_state.T.data, dtype=np.float64)
    si3_s = np.asarray(si3_card.recipe.initial_state.S.data, dtype=np.float64)
    k0_s = np.asarray(k0_card.ocean.recipe.initial_state.S.data, dtype=np.float64)
    # Wet cells are bit-identical between SI3 and K0 initial T and S.
    assert np.array_equal(si3_t[active].view(np.uint64), k0_t[active].view(np.uint64))
    assert np.array_equal(si3_s.view(np.uint64), k0_s.view(np.uint64))
    # SI3 keeps +0.0 on all dry cells; K0 preserves -0.0 where raw T < 0 before tmask multiply.
    assert int(np.sum(np.signbit(si3_t[~active]))) == 0
    assert int(np.sum(np.signbit(k0_t[~active]))) == 151917
    assert int(np.sum(np.signbit(k0_t[..., 0][~wet_surface]))) == 4668

    # Plant SI3's initial_state onto the K0 card and verify _validate_k0_ocean_card refuses.
    planted_eta = k0_card._replace(
        ocean=k0_card.ocean._replace(
            recipe=k0_card.ocean.recipe._replace(
                initial_state=k0_card.ocean.recipe.initial_state._replace(
                    eta=si3_card.recipe.initial_state.eta
                )
            )
        )
    )
    with pytest.raises(ValueError, match="ssh = 0.0"):
        bulk._validate_k0_ocean_card(planted_eta)

    planted_t = k0_card._replace(
        ocean=k0_card.ocean._replace(
            recipe=k0_card.ocean.recipe._replace(
                initial_state=k0_card.ocean.recipe.initial_state._replace(
                    T=si3_card.recipe.initial_state.T
                )
            )
        )
    )
    with pytest.raises(ValueError, match="ptsd \\* tmask signed zeros"):
        bulk._validate_k0_ocean_card(planted_t)

    arm_a = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
        "round340/s4_acquisition/orca2_s4_exact_a_np2"
    )
    if arm_a.exists():
        r0 = gate.read_exact_record(arm_a / "oracle_r340_state_k00000001_s0_r0000.bin")
        r1 = gate.read_exact_record(arm_a / "oracle_r340_state_k00000001_s0_r0001.bin")
        entry_a = gate._join_rank_fields([r0, r1])
        assert gate.exact_score(k0_t, entry_a["T"][..., :30])["unequal"] == 0
        assert gate.exact_score(k0_s, entry_a["S"][..., :30])["unequal"] == 0
        assert gate.exact_score(k0_eta, entry_a["ssh"])["unequal"] == 0


def test_resume7_unmasked_s1_turb_and_non_nemo_caller_bit_identity():
    """Fix 3: _nemo_ncar_turbulent is unmasked over A2D(0); non-NEMO callers stay bit-identical."""
    args, *_ = _production_surface_case()
    # Evaluate nemo_blk_oce_ncar on a 2x3 grid with mixed wet (1.0) and dry (0.0) mask
    # and distinct dry-cell air/surface temperatures to prove S1_TURB diagnostics are unmasked.
    mask = jnp.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=jnp.float64)
    all_wet = jnp.ones_like(mask)
    sst_c = jnp.asarray([[15.0, -5.0, 12.0], [-10.0, 18.0, -8.0]], dtype=jnp.float64)
    ssu = jnp.asarray([[0.1, -0.2, 0.05], [0.15, -0.1, 0.2]], dtype=jnp.float64)
    ssv = jnp.asarray([[-0.05, 0.1, -0.15], [0.2, 0.05, -0.1]], dtype=jnp.float64)
    t_air = jnp.asarray([[286.0, 260.0, 285.0], [255.0, 288.0, 265.0]], dtype=jnp.float64)
    s1_args = (args[0], args[1], t_air, args[3], args[4], sst_c, ssu, ssv, *args[8:12])
    res_masked = bulk.nemo_blk_oce_ncar(
        *s1_args, mask, *args[13:], nn_ice=0, ice_if=None, s0=None, s2=None
    )
    res_unmasked = bulk.nemo_blk_oce_ncar(
        *s1_args, all_wet, *args[13:], nn_ice=0, ice_if=None, s0=None, s2=None
    )
    for field in ("theta_air", "ssq", "Cd", "Ch", "Ce", "rhoa", "wndm"):
        np.testing.assert_array_equal(
            np.asarray(getattr(res_masked, field)).view(np.uint64),
            np.asarray(getattr(res_unmasked, field)).view(np.uint64),
            err_msg=field,
        )
    dry = np.asarray(mask) == 0.0
    for field in ("tsk", "sensible", "latent", "evap", "qsr", "qns", "emp", "utau", "vtau", "taum"):
        assert np.all(np.asarray(getattr(res_masked, field))[dry] == 0.0), field

    # M6: verify non-NEMO callers (air_sea_fluxes, ncar_transfer_coefficients, blk_oce_2 default)
    # never call _nemo_ncar_turbulent and remain bit-identical.
    u10, v10, t_air, q_air, slp = s1_args[:5]
    sst_k = sst_c + 273.15
    ssq = bulk.seawater_q_sat(sst_k, slp)
    theta_air, p10 = bulk.potential_air_temperature_10m(t_air, q_air, slp)
    theta_sst = bulk.exner_potential_temperature(sst_k, slp)
    wind = jnp.sqrt(u10 ** 2 + v10 ** 2 + 1.0e-12)
    cd, ch, ce, ub = bulk.ncar_transfer_coefficients(
        theta_air, q_air, theta_sst, ssq, wind, nb_iter=5
    )
    tau_x, tau_y, sh, lh, evap = bulk.air_sea_fluxes(
        u10, v10, t_air, q_air, sst_k, slp_Pa=slp, algo="ncar"
    )
    rho = bulk.rho_air_moist(t_air, q_air, p10)
    urho = ub * jnp.maximum(rho, 1.0)
    np.testing.assert_array_equal(
        np.asarray(tau_x).view(np.uint64),
        np.asarray(-urho * cd * u10).view(np.uint64),
    )
    np.testing.assert_array_equal(
        np.asarray(sh).view(np.uint64),
        np.asarray(urho * ch * (theta_air - theta_sst) * bulk.moist_air_cp(q_air)).view(np.uint64),
    )


def test_resume7_s0f_tradmp_year_seam_slots_match_oracle():
    """Fix 4: damping_call S0_F frames use sf_tsd's (nbb=1, naa=2) slots after istate's kt=1 call."""
    deck = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
    source = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
        "round5/acquisition/orca1ice_surface_entry_every_step_a_np2"
    )
    k0 = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
        "round339/k0_smoke/orca2_k0_10step_np2"
    )
    if not (deck.exists() and source.exists() and k0.exists()):
        pytest.skip("ORCA2 inputs or round-339 smoke artifacts are not installed")

    card = bulk.build_nemo_orca2_k0_ocean_card(deck)
    controls = gate.build_real_composed_inputs(
        card, deck, source / "input_files.sha256", k0 / "ice_cover_clim.nc"
    )
    # kt=1 c001 (istate) has nbb=2, naa=1; kt=1 c009 (tra_dmp) has nbb=1, naa=2.
    c001_t = controls[0]["_s0f_actual"]["kt01/S0_F/c001/f001/votemper"]
    c009_t = controls[0]["_s0f_actual"]["kt01/S0_F/c009/f001/votemper"]
    assert c001_t["nbb"].tolist() == [2.0, 2.0]
    assert c001_t["naa"].tolist() == [1.0, 1.0]
    assert c001_t["nrec"].tolist() == [1.0, 1339200.0, 12.0, -1339200.0] * 2
    assert c009_t["nbb"].tolist() == [1.0, 1.0]
    assert c009_t["naa"].tolist() == [2.0, 2.0]
    assert c009_t["nrec"].tolist() == [12.0, -1339200.0, 1.0, 1339200.0] * 2
    np.testing.assert_array_equal(c001_t["fdta"][..., 0], c009_t["fdta"][..., 1])
    np.testing.assert_array_equal(c001_t["fdta"][..., 1], c009_t["fdta"][..., 0])

    arm_a = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
        "round340/s4_acquisition/orca2_s4_exact_a_np2"
    )
    if arm_a.exists():
        for kt, control in enumerate(controls, start=1):
            call = 9 if kt == 1 else (6 if kt % 2 else 2)
            for f_idx, var in enumerate(("votemper", "vosaline"), start=1):
                r0 = gate.read_exact_record(
                    arm_a / f"oracle_r340_fld_k{kt:08d}_c{call:03d}_f{f_idx:03d}_r0000.bin"
                )
                r1 = gate.read_exact_record(
                    arm_a / f"oracle_r340_fld_k{kt:08d}_c{call:03d}_f{f_idx:03d}_r0001.bin"
                )
                expected = gate._join_rank_fields([r0, r1])
                actual = control["_s0f_actual"][
                    f"kt{kt:02d}/S0_F/c{call:03d}/f{f_idx:03d}/{var}"
                ]
                for key, exp_val in expected.items():
                    assert gate.exact_score(actual[key], exp_val)["unequal"] == 0, (
                        f"kt{kt:02d}/S0_F/c{call:03d}/f{f_idx:03d}/{var}/{key}"
                    )



