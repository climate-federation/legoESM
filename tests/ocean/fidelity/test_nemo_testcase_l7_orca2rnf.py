"""Lane-B L7: certified V5 plus ORCA2 river-mouth heat mixing."""

from __future__ import annotations

import inspect
import pickle
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.core.source_rounding import nemo_source_round
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
from legoesm.ocean.physics.vertical_mixing.config import RiverMouthMixingConfig
from legoesm.ocean.physics.vertical_mixing.k_profiles import (
    _nemo_river_mouth_nkrnf,
    compute_vertical_K_profiles,
    nemo_river_mouth_heat_diffusivity,
)
from tests.ocean.fidelity.test_nemo_testcase_v1_orca2tke import (
    CARD_FILES,
    CASES,
    INPUTS,
    ORCA2_DECK,
    _root,
    _values,
)

V5 = CASES / "nemo_testcase_v5_orca2havtb"
RUNOFF = INPUTS / "runoff_core_monthly.nc"
IWM = INPUTS / "zdfiwm_forcing_orca2.nc"
TOOL = CASES / "nemo_testcase_l7_orca2rnf"
V5_RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                 "gridladder_vmix_rounds/round15/oracle_v5_r15/twin_a")


@pytest.fixture(autouse=True)
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


@pytest.fixture(scope="module")
def cards(tmp_path_factory):
    needed = (*CARD_FILES, RUNOFF.name, IWM.name)
    if not all((INPUTS / name).is_file() for name in needed):
        pytest.skip("no ORCA2 inputs")
    base = tmp_path_factory.mktemp("l7-rnf")
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    try:
        v5_root = _root(base, V5, "v5")
        l7_root = _root(base, V5, "l7")
        for root in (v5_root, l7_root):
            (root / IWM.name).symlink_to(IWM)
            (root / RUNOFF.name).symlink_to(RUNOFF)
        out = (recipe.build_orca2v5_havtb_card(v5_root),
               recipe.build_orca2l7_rnf_card(l7_root), l7_root)
    finally:
        set_policy(before)
    return out


@pytest.mark.skipif(not ORCA2_DECK.is_file(), reason="no NEMO tree")
def test_l7_deck_block_and_static_mask_are_orca2s_exact_selection():
    deck = _values(ORCA2_DECK)
    assert {key: deck[key] for key in recipe._ORCA2L7_RNF_FIELDS} == (
        recipe._ORCA2L7_RNF_FIELDS)
    assert "".join(deck["namsbc_rnf.sn_cnf"].split()) == (
        "'runoff_core_monthly',0.,'socoefr0',.false.,.true.,"
        "'yearly','','',''")
    with netCDF4.Dataset(RUNOFF) as ds:
        field = ds.variables["socoefr0"]
        assert field.dimensions == ("y", "x")
        values = np.asarray(field[:], dtype=np.float64)
    assert values.shape == (148, 180)
    assert np.count_nonzero(values) == 0


def test_l7_is_v5_bits_plus_explicit_coordinate_selection(cards):
    v5, l7, _ = cards
    assert pickle.dumps(v5.recipe.model_config) == pickle.dumps(l7.recipe.model_config)
    assert v5.dt_s == l7.dt_s and v5.n_steps == l7.n_steps
    assert l7.recipe.z_coord.nemo_rnf_mouth_config == RiverMouthMixingConfig(
        enabled=True, depth_m=15.0, diffusivity_m2_s=1.0e-3)
    mask = np.asarray(l7.recipe.z_coord.nemo_rnf_mouth_mask)
    assert mask.shape == (148, 180) and np.count_nonzero(mask) == 0
    assert v5.recipe.z_coord.nemo_rnf_mouth_config is None
    assert v5.recipe.z_coord.nemo_rnf_mouth_mask is None
    assert _nemo_river_mouth_nkrnf(l7.recipe.z_coord, 15.0) == 2
    assert recipe._orca2g1_selector_failures(l7) == []


def test_selector_refuses_missing_mask_and_each_wrong_scalar(cards):
    _, l7, _ = cards
    z = l7.recipe.z_coord
    planted = (
        z._replace(nemo_rnf_mouth_mask=None),
        z._replace(nemo_rnf_mouth_config=z.nemo_rnf_mouth_config._replace(
            enabled=False)),
        z._replace(nemo_rnf_mouth_config=z.nemo_rnf_mouth_config._replace(
            depth_m=14.0)),
        z._replace(nemo_rnf_mouth_config=z.nemo_rnf_mouth_config._replace(
            diffusivity_m2_s=2.0e-3)),
    )
    for changed in planted:
        card = l7._replace(recipe=l7.recipe._replace(z_coord=changed))
        assert recipe._orca2g1_selector_failures(card) == ["ln_rnf_mouth"]


@pytest.mark.skipif(not jax.config.x64_enabled, reason="binary64 value contract")
def test_river_statement_is_exact_jitted_traced_heat_only_and_nonvacuous():
    h = float.fromhex
    avt = jnp.asarray([[[h("0x1.0000000000001p-14"), h("0x1.8p-13"),
                         h("0x1.4p-12")],
                        [h("0x1.2p-15"), h("0x1.1p-13"),
                         h("0x1.9p-12")]]], dtype=jnp.float64)
    rnf = jnp.asarray([[h("0x1.4p-2"), h("0x1.8p-1")]], dtype=jnp.float64)
    wet = jnp.asarray([[[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]], dtype=jnp.float64)

    @jax.jit
    def selected(a, m, w):
        return nemo_river_mouth_heat_diffusivity(
            a, m, w, rn_avt_rnf=1.0e-3, nkrnf=3)

    got = selected(avt, rnf, wet)
    sr = nemo_source_round
    increment = sr(sr(sr(jnp.asarray(2.0) * jnp.asarray(1.0e-3))
                         * rnf[..., None]) * wet)
    want = sr(avt + jnp.where(jnp.arange(3) < 2, increment, 0.0))
    assert np.array_equal(np.asarray(got), np.asarray(want))
    assert not np.array_equal(np.asarray(got), np.asarray(avt))  # removal plant
    # Momentum and salt have no call site: the selected kernel returns one
    # heat channel, and its positional call writes only K_v_total.
    source = inspect.getsource(compute_vertical_K_profiles)
    call = source.index("K_v_total = nemo_river_mouth_heat_diffusivity(")
    assert call < source.index("conv = physics_config.convection")
    assert call < source.index("if _ddm_cfg is not None and _ddm_cfg.enabled")
    assert call < source.index("if (iwm_cfg is not None and iwm_cfg.enabled")
    grad = jax.grad(lambda a: jnp.sum(selected(a, rnf, wet)))(avt)
    assert np.isfinite(np.asarray(grad)).all()


@pytest.mark.skipif(not jax.config.x64_enabled, reason="binary64 value contract")
def test_real_zero_mask_is_exact_identity_but_position_plant_fires(cards):
    _, l7, _ = cards
    base = jnp.linspace(1.0e-6, 4.0e-5, 3 * 4 * 30,
                        dtype=jnp.float64).reshape(3, 4, 30)
    wet = jnp.ones_like(base)
    zero = jnp.zeros(base.shape[:-1], dtype=base.dtype)
    actual = jax.jit(lambda a, m: nemo_river_mouth_heat_diffusivity(
        a, m, wet, rn_avt_rnf=1.0e-3,
        nkrnf=_nemo_river_mouth_nkrnf(l7.recipe.z_coord, 15.0)))(base, zero)
    assert np.array_equal(np.asarray(actual), np.asarray(base))

    # EVD replaces an unstable cell.  River-before-EVD and river-after-EVD
    # therefore differ: this plant makes moving the statement observable.
    synthetic = jnp.ones(base.shape[:-1], dtype=base.dtype)
    before = nemo_river_mouth_heat_diffusivity(
        base, synthetic, wet, rn_avt_rnf=1.0e-3, nkrnf=2)
    evd_after = before.at[0, 0, 0].set(0.1)
    evd_before = base.at[0, 0, 0].set(0.1)
    moved = nemo_river_mouth_heat_diffusivity(
        evd_before, synthetic, wet, rn_avt_rnf=1.0e-3, nkrnf=2)
    assert not np.array_equal(np.asarray(evd_after), np.asarray(moved))


@pytest.mark.skipif(not V5_RECORD.is_dir(), reason="no admitted V5 record")
@pytest.mark.skipif(not jax.config.x64_enabled, reason="binary64 value contract")
def test_one_step_v5_versus_l7_real_mask_is_bit_identical():
    """Production jitted step: actual record is identity; synthetic plant above acts."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    sys.path[:0] = [str(CASES), str(CASES / "nemo_testcase_g5_orca2ice")]
    import g5_records as g5

    l7 = recipe.build_orca2l7_rnf_card(V5_RECORD)
    run = l7.recipe
    base_z = run.z_coord._replace(
        nemo_rnf_mouth_config=None, nemo_rnf_mouth_mask=None)
    fresh, surface = g5.nemo_supplied_forcing(l7, V5_RECORD, 1)
    surface = surface._replace(
        ice_mean_thickness_hm_i=g5.read_v1_hm_i(V5_RECORD, 1))

    def step(z_coord):
        model = LatLonCGridOceanModel(
            run.grid, z_coord, run.model_config, iwm_forcing=run.iwm_forcing)
        return jax.device_get(jax.jit(
            lambda state, fw, sf: model.step(
                state, l7.dt_s, freshwater=fw, surface_forcing=sf)
        )(run.initial_state, fresh, surface))

    base, selected = step(base_z), step(run.z_coord)
    for name in ("T", "S", "u", "v", "eta"):
        assert np.array_equal(
            np.asarray(getattr(base, name).data).view(np.uint64),
            np.asarray(getattr(selected, name).data).view(np.uint64)), name


def test_acquisition_tool_is_guarded_and_observer_controls_fire(tmp_path):
    run = TOOL / "run.sh"
    instrument = TOOL / "instrument_zdfphy.py"
    reader = TOOL / "l7_records.py"
    assert subprocess.run(["bash", "-n", str(run)]).returncode == 0
    text = run.read_text()
    for required in (
        "L7_OPERATOR_ACK=I_ACKNOWLEDGE_NEMO_BUILD_AND_RUN",
        "--admit-existing", "checklist=7", "trap cleanup EXIT",
        'check_values "$REAL_LOG"', 'check_values "$target_deck"',
        '--check-acting "$SMOKE/$RECORD"', "steps <= 10",
        "smoke -> writer-free boundary -> twins A/B/C",
    ):
        assert required in text, required
    assert "| grep -q" not in text
    assert "/tmp/" not in text
    result = subprocess.run(
        [sys.executable, str(reader), "--selftest"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    source = Path(
        "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/OCE/ZDF/zdfphy.F90")
    if not source.is_file():
        pytest.skip("no NEMO source")
    good = tmp_path / "zdfphy.f90"
    result = subprocess.run(
        [sys.executable, str(instrument), str(source), str(good)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    rendered = good.read_text()
    assert rendered.index("l7_closure(:,:,:)") < rendered.index(
        "avt(ji,jj,jk) = avt(ji,jj,jk) + 2._wp")
    river = rendered.index("l7_river(:,:,:)")
    evd = rendered.index("IF( ln_zdfevd )", river)
    evd_ddm = rendered.index("l7_evd_ddm(:,:,:)", evd)
    iwm = rendered.index("IF( ln_zdfiwm )", evd_ddm)
    final = rendered.index("WRITE(l7_unit) 'IWM_AVT", iwm)
    assert river < evd < evd_ddm < iwm < final
    for plant in ("--plant-after-continuation", "--plant-short-literal",
                  "--plant-raw-observer", "--plant-after-evd"):
        result = subprocess.run(
            [sys.executable, str(instrument), plant, str(source),
             str(tmp_path / "plant.f90")], capture_output=True, text=True)
        assert result.returncode != 0, plant
