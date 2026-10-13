"""Direct tests for the SWG_DG round-6 backend/rank discriminator."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_PATH = (Path(__file__).resolve().parents[3]
         / "scripts/validate/ocean_fidelity/testcases/"
         "nemo_testcase_l1_swg_dg_round6.py")
_SPEC = importlib.util.spec_from_file_location("swg_dg_round6", _PATH)
round6 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(round6)


def test_bit_summary_is_word_exact_for_signed_zero_and_one_ulp():
    reference = np.asarray([0.0, 1.0], dtype=np.float64)
    candidate = np.asarray([-0.0, np.nextafter(1.0, np.inf)], dtype=np.float64)
    row = round6.bit_summary(reference, candidate)
    assert row["bit_unequal"] == 2
    assert row["first_unequal_cell"] == [0]
    assert not row["bit_identical"]


def test_coordinate_assembly_is_exact_and_rejects_conflicting_overlap():
    lon = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    lat = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    left = (lon[:, :1], lat[:, :1], np.asarray([[1.0], [2.0]]))
    right = (lon[:, 1:], lat[:, 1:], np.asarray([[3.0], [4.0]]))
    assembled = round6.assemble_coordinate_tiles_exact(
        (left, right), lon, lat)
    np.testing.assert_array_equal(assembled, [[1.0, 3.0], [2.0, 4.0]])
    bad_overlap = (
        lon[:, :1], lat[:, :1], np.asarray([[1.0], [9.0]]))
    with pytest.raises(RuntimeError, match="conflicting rank overlap"):
        round6.assemble_coordinate_tiles_exact(
            (left, bad_overlap, right), lon, lat)


def test_json_writer_handles_numpy_scalars_and_refuses_overwrite(tmp_path):
    path = tmp_path / "receipt.json"
    round6._write_json(path, {"count": np.int64(3)})
    assert path.read_text() == '{\n  "count": 3\n}\n'
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        round6._write_json(path, {"count": np.int64(4)})


def test_exposure_classifier_distinguishes_zero_from_magnitude():
    names = (
        "zonal_difference", "numerator", "area_product",
        "transport_divergence", "hdiv", "e3_hdiv_product",
        "e3_hdiv", "zhdiv", "ssh_increment",
    )
    nonzero = {name: {"nonzero": True} for name in names}
    assert round6._classify_exposure(nonzero, nonzero) == (
        "same_nonzero_chain_different_magnitude")
    zero = {name: {"nonzero": True} for name in names}
    zero["numerator"] = {"nonzero": False}
    assert round6._classify_exposure(zero, nonzero) == (
        "zero_on_shipped_nonzero_on_maddison")


def test_tiny_statement_bisect_reproduces_production_stage_word_for_word():
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.dynamics.swe_rk3 import SWERK3StageEntry
    from legoesm.ocean.fidelity.nemo_testcase_recipe import SWG_MADDISON_NAMELIST, build_swg_card

    jax.config.update("jax_enable_x64", True)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    tiny = SWG_MADDISON_NAMELIST._replace(
        nn_SWG=1, rn_Dt_s=50.0, nn_itend=10, nn_write=10, nn_stock=10)
    card = build_swg_card(tiny)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    shape = card.recipe.land_mask.shape
    jj, ii = np.indices(shape, dtype=np.float64)
    mask = np.asarray(card.recipe.land_mask, dtype=np.float64)
    fields = {
        name: jnp.zeros(shape, dtype=jnp.float64)
        for name in SWERK3StageEntry._fields
    }
    fields["u_k"] = jnp.asarray(mask * (1.0e-4 + ii * 1.0e-7))
    fields["v_k"] = jnp.asarray(mask * (-2.0e-4 + jj * 1.0e-7))
    entry = SWERK3StageEntry(**fields)
    production = model.replay_swe_rk3_stage(entry, card.dt_s, 1, entry)
    probes = round6.build_statement_bisect(entry, card)
    np.testing.assert_array_equal(
        np.asarray(probes["baseline"]), np.asarray(production.ssh_k))
    assert tuple(probes) == round6.PROBE_NAMES
    assert np.any(np.asarray(probes["zonal_difference"]) != 0.0)
