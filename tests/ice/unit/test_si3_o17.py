"""Structural and fail-closed tests for the new O17 production entry."""

from pathlib import Path
import struct

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.ice import sea_ice
from legoesm.ice import si3_o17 as o17_mod
from legoesm.ice.config import SeaIceConfig, build_si3_orca2_o17_config
from legoesm.ice.si3_o17 import SI3O17Card, SI3O17LiveForcing, SI3O17StepState, si3_o17_step
from legoesm.ice.si3_orca2_state import SI3ORCA2StateConfig

from scripts.validate.ocean_fidelity.testcases import (
    nemo_si3_o17_ad_probe,
    nemo_si3_o17_composition_scorer,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o17_ad_probe import (
    DEFAULT_MASTER,
    _case,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o17_acquisition.preflight import (
    ORDER,
    preflight,
)

REPO = Path(__file__).resolve().parents[3]


def test_o17_typed_pytree_boundary_and_explicit_card() -> None:
    assert SI3O17LiveForcing._fields[:4] == ("kt", "sss_m", "sst_m", "rn_amax_2d")
    assert not (
        {"redistribution_state", "correction_state", "thermo", "fluxes"}
        & set(SI3O17LiveForcing._fields)
    )
    assert SI3O17StepState._fields == ("ice", "aevp")
    assert SI3O17Card._fields[-1] == "core2_precipitation_closure"
    assert callable(si3_o17_step)


def test_o17_card_refuses_core2_closure_before_any_inference() -> None:
    card = SI3O17Card(*([None] * (len(SI3O17Card._fields) - 1)), True)
    with pytest.raises(ValueError, match="core2_precipitation_closure=False"):
        build_si3_orca2_o17_config(card)


def test_o17_call_order_preflight_and_swap_drop_plants() -> None:
    report = preflight(REPO)
    assert report["call_order"] == list(ORDER)
    assert report["steps"] == 100
    assert any("o4_orca2_prather_gate" in name for name in report["imported_component_tools"])
    for plant in ("swap-rheology-advection", "drop-redistribution"):
        with pytest.raises(SystemExit, match="STATUS PLANT-FIRED"):
            preflight(REPO, plant)


def test_o17_public_production_dispatch_calls_composition(monkeypatch) -> None:
    state_cfg = SI3ORCA2StateConfig.for_orca2_sas_ice(ln_icedyn=True, ln_icethd=True)
    card = object()
    config = SeaIceConfig(
        dynamics="none",
        transport="none",
        thermo_scheme="si3_orca2_o2",
        ice_constants=state_cfg.ice_constants,
        si3_orca2=state_cfg,
        core2_precipitation_closure=False,
        si3_o17=card,
    )
    entry = SI3O17StepState(None, None)
    forcing = SI3O17LiveForcing(1, *([None] * (len(SI3O17LiveForcing._fields) - 1)))
    expected = (entry, object())
    calls = []

    def planted(step_entry, live_forcing, actual_card):
        calls.append((step_entry, live_forcing, actual_card))
        return expected

    monkeypatch.setattr(sea_ice, "si3_o17_step", planted)
    actual = sea_ice.step_sea_ice(entry, forcing, None, None, None, config, 0.0, state_cfg.rDt_ice)
    assert actual == expected
    assert calls == [(entry, forcing, card)]


def test_o17_record_reader_is_self_describing_and_rejects_truncation(tmp_path) -> None:
    assert nemo_si3_o17_composition_scorer.VALID_LABELS == ("INDEPENDENT", "GIVEN-NEMO-ENTRY")
    record = tmp_path / "o17_kind22_step0001_rank0000.bin"
    field = b"v_i".ljust(32, b" ")
    end = b"__END__".ljust(32, b" ")
    payload = (
        b"O17S0001" + struct.pack("=i8i", 1, 22, 1, 0, 6, 5, 5, 10, 5)
        + field + struct.pack("=i4i", 2, 2, 3, 0, 0)
        + np.arange(6, dtype=np.float64).reshape(2, 3, order="F").tobytes(order="F")
        + end + struct.pack("=i4i", -1, 0, 0, 0, 0)
    )
    record.write_bytes(payload)
    parsed = nemo_si3_o17_composition_scorer.read_record(record)
    assert parsed["meta"]["kind"] == 22
    np.testing.assert_array_equal(parsed["fields"]["v_i"], np.arange(6).reshape(2, 3, order="F"))
    record.write_bytes(payload[:143])  # one byte short of v_i's six float64 values
    with pytest.raises(ValueError, match="truncated"):
        nemo_si3_o17_composition_scorer.read_record(record)


def test_o17_unified_acquisition_routes_real_scorer_only_after_one_master_run() -> None:
    run = (REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_acquisition/run.sh").read_text()
    scorer = (REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py").read_text()
    assert run.count('"$master" --run "$evidence"') == 1
    assert "nemo_si3_o4_orca2_prather_acquisition/run.sh" not in run
    assert "ACQUISITION_NEEDED" not in scorer
    assert "si3_o17_step(entry, forcing, card)" in scorer
    assert 'touch "$EVIDENCE/RUN_COMPLETE"' in (
        REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o16_master_acquisition/run.sh"
    ).read_text()


def test_o17_step_executes_active_ice_eager_jit_and_numerical_plants() -> None:
    """Run the real composition and prove order/fold/forcing features alter numbers."""
    entry, forcing, card = _case(DEFAULT_MASTER)
    card = card._replace(aevp=card.aevp._replace(n_subcycles=1))

    def run(step_entry, dynamic_forcing, **plants):
        live = SI3O17LiveForcing(1, *dynamic_forcing)
        # O6 redistribution has its own admitted eager/JIT numerical suite;
        # omit that already-covered kernel so this composition gate can reach
        # and compile the O7 forcing boundary on the shared test host.
        plants.setdefault("_plant_drop_redistribution", True)
        return si3_o17_step(
            step_entry, live, card, _test_return_surface_probe=True, **plants
        )

    dynamic = forcing[1:]
    with jax.disable_jit():
        eager = run(entry, dynamic)

    compiled = jax.jit(run)(entry, dynamic)
    for got, want in zip(compiled, eager, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=0.0, atol=1.0e-12, equal_nan=True)
    assert np.count_nonzero(np.asarray(eager[2])) > 0

    with jax.disable_jit():
        swapped = run(entry, dynamic, _plant_swap_rheology_advection=True)
        no_fold = run(entry, dynamic, _plant_drop_deformation_fold=True)
    assert any(
        not np.array_equal(np.asarray(got), np.asarray(want))
        for got, want in zip(swapped[:2], eager[:2], strict=True)
    )
    assert any(
        not np.array_equal(np.asarray(got), np.asarray(want))
        for got, want in zip(no_fold[4:6], eager[4:6], strict=True)
    )
    half_frq = forcing._replace(surface=forcing.surface._replace(frq_m=forcing.surface.frq_m * 0.5))
    with jax.disable_jit():
        half_frq_fluxes = run(entry, half_frq[1:])[6:8]
    assert any(
        not np.array_equal(np.asarray(got), np.asarray(want))
        for got, want in zip(half_frq_fluxes, eager[6:8], strict=True)
    )

    with jax.disable_jit():
        tail_eager = o17_mod._thermodynamic_tail(entry.ice, card)
    tail_compiled = jax.jit(lambda state: o17_mod._thermodynamic_tail(state, card))(entry.ice)
    np.testing.assert_allclose(
        np.asarray(tail_compiled.oa_i), np.asarray(tail_eager.oa_i), rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_array_equal(np.asarray(tail_compiled.u_ice), np.asarray(tail_eager.u_ice))
    np.testing.assert_array_equal(np.asarray(tail_compiled.v_ice), np.asarray(tail_eager.v_ice))

    with pytest.raises(ValueError, match="ln_ice_embd=F"):
        si3_o17_step(entry, forcing, card._replace(ln_ice_embd=True))
