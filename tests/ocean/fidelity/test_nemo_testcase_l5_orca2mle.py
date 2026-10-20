"""Lane B L5: certified G9 plus ORCA2 mixed-layer eddies."""

from __future__ import annotations

import importlib.util
import pickle
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe

from tests.ocean.fidelity.test_nemo_testcase_g9_orca2tadv import (
    CARD_FILES, CASES, G9, INPUTS, ORCA2, _root, _values,
)
from tests.ocean.fidelity.test_nemo_testcase_v1_orca2tke import _differs, _flat


L5 = CASES / "nemo_testcase_l5_orca2mle"
RUN = L5 / "run.sh"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INSTRUMENT = _module("instrument_tramle", L5 / "instrument_tramle.py")


@pytest.fixture(autouse=True, scope="module")
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


def _l5_root(base: Path) -> Path:
    root = _root(base, G9)
    path = root / "namelist_cfg"
    text = path.read_text()
    old = "   ln_mle      = .false."
    assert text.count(old) == 1
    path.write_text(text.replace(old, "   ln_mle      = .true."))
    return root


def test_l5_deck_is_exactly_g9_plus_ln_mle(tmp_path):
    root = _l5_root(tmp_path)
    g9, l5 = _values(G9 / "namelist_cfg"), _values(root / "namelist_cfg")
    changed = {key: (g9.get(key), l5.get(key)) for key in set(g9) | set(l5)
               if g9.get(key) != l5.get(key)}
    assert changed == {"namtra_mle.ln_mle": (False, True)}
    assert (root / "namelist_ice_cfg").read_bytes() == (
        G9 / "namelist_ice_cfg").read_bytes()
    recipe.read_orca2g1_deck(root, "ORCA2L5-mle")
    with pytest.raises(ValueError, match="namtra_mle"):
        recipe.read_orca2g1_deck(root, "ORCA2G9-tadv")


@pytest.mark.skipif(not ORCA2.is_dir(), reason="no NEMO tree")
def test_l5_resolved_namtra_mle_is_exact_orca2_program():
    cfg, ref = _values(ORCA2 / "namelist_cfg"), _values(ORCA2 / "namelist_ref")
    got = {key: cfg.get(key, ref[key])
           for key in recipe._ORCA2L5_MLE_RESOLVED_FIELDS}
    assert got == recipe._ORCA2L5_MLE_RESOLVED_FIELDS


@pytest.mark.parametrize("old,new", [
    ("   ln_mle      = .true.", "   ln_mle      = .false."),
    ("   ln_mle      = .true.", "   ln_mle      = .true.\n   rn_ce = 0.07"),
])
def test_reader_refuses_each_mle_deck_plant(tmp_path, old, new):
    root = _l5_root(tmp_path)
    path = root / "namelist_cfg"
    text = path.read_text(); assert text.count(old) == 1
    path.write_text(text.replace(old, new))
    with pytest.raises(ValueError, match="namtra_mle"):
        recipe.read_orca2g1_deck(root, "ORCA2L5-mle")


def test_l5_acquisition_is_fail_closed_and_numeric():
    assert subprocess.run(["bash", "-n", str(RUN)]).returncode == 0
    text = RUN.read_text()
    for required in (
        "check_log_values", "6.0000000000000000E-002", "bad-float-log",
        "smoke_2step", "ref_100step", "completed_steps",
        "REF_TARGET_NAME=ORCA2L5_MLE_R21_REF", "--admit-existing",
        "steps=$((completed < 10 ? completed : 10))",
        'audit_run "$TARGET/BLD/bin/nemo.exe" "$TWIN_B"',
        'audit_run "$TARGET/BLD/bin/nemo.exe" "$TWIN_C"',
        "altered deck plant accepted", "altered log plant accepted",
        "incomplete smoke plant accepted", "checklist=7",
        "commit the exact acquisition tool",
    ):
        assert required in text, required
    assert "| grep -q" not in text and "| grep -Fq" not in text


@pytest.mark.skipif(
    not (ORCA2.parent / "BLD/ppsrc/nemo/tramle.f90").is_file(),
    reason="no compiled NEMO tree")
def test_l5_observer_zero_first_defined_domain():
    source = (ORCA2.parent / "BLD/ppsrc/nemo/tramle.f90").read_text()
    instrumented = INSTRUMENT.instrument(source)
    INSTRUMENT.validate(instrumented)
    assert "zfield(:,:,:) = 0._wp" in instrumented
    assert "IF( llwet ) zfield(ji,jj,jk) = pfield(ji,jj,jk)" in instrumented
    planted = instrumented.replace("l5_canon_3d(pFu,'U')", "pFu", 1)
    with pytest.raises(ValueError, match="canonical"):
        INSTRUMENT.validate(planted)


@pytest.fixture(scope="module")
def cards(tmp_path_factory):
    if not all((INPUTS / name).is_file() for name in CARD_FILES):
        pytest.skip("no ORCA2 inputs")
    base = tmp_path_factory.mktemp("l5")
    (base / "g9").mkdir()
    (base / "l5").mkdir()
    return (
        recipe.build_orca2g9_tadv_card(_root(base / "g9", G9)),
        recipe.build_orca2l5_mle_card(_l5_root(base / "l5")),
    )


def test_l5_card_changes_only_existing_mle_selector(cards):
    g9, l5 = cards
    c9, c5 = g9.recipe.model_config, l5.recipe.model_config
    changed = {key for key in set(_flat(c9)) | set(_flat(c5))
               if _differs(_flat(c9).get(key), _flat(c5).get(key))}
    assert changed and all(key == "physics.mle" or key.startswith("physics.mle.")
                           for key in changed)
    mle = c5.physics.mle
    assert (mle.ce, mle.lat_ref_deg, mle.rho_c_mle, mle.mld_uv,
            mle.no_mle_in_convection, mle.through_fct) == (
        0.06, 20.0, 0.01, "min", False, True)
    assert c5.gm_redi is None and c5.tracer_advection == "fct2"
    for name in ("grid", "z_coord", "initial_state"):
        assert pickle.dumps(getattr(g9.recipe, name)) == pickle.dumps(
            getattr(l5.recipe, name)), name
    assert recipe._orca2g1_selector_failures(g9) == []
    assert recipe._orca2g1_selector_failures(l5) == []


@pytest.mark.parametrize("change", [
    {"ce": 0.07}, {"lat_ref_deg": 21.0}, {"rho_c_mle": 0.02},
    {"mld_uv": "avg"}, {"no_mle_in_convection": True},
    {"through_fct": False},
])
def test_l5_selector_refuses_each_mle_wiring_plant(cards, change):
    l5 = cards[1]
    cfg = l5.recipe.model_config
    physics = cfg.physics._replace(mle=cfg.physics.mle._replace(**change))
    planted = l5._replace(recipe=l5.recipe._replace(
        model_config=cfg._replace(physics=physics)))
    assert recipe._orca2g1_selector_failures(planted) == ["ln_mle"]


def test_l5_model_requires_fct_ws_route(cards):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    run = cards[1].recipe
    LatLonCGridOceanModel(run.grid, run.z_coord, run.model_config)
    with pytest.raises(ValueError, match="through_fct"):
        LatLonCGridOceanModel(
            run.grid, run.z_coord,
            run.model_config._replace(tracer_advection="none"))


def test_l5_jitted_traced_x64_transport_is_exact_and_nonzero():
    from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
    from legoesm.ocean.eos import make_eos_fn
    from legoesm.ocean.physics.lateral_mixing.mle import MLEConfig
    from legoesm.ocean.physics.lateral_mixing.mle_latlon_cgrid import (
        mle_tracer_tendency_latlon_cgrid,
    )
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(3, 4)
    geom = ensure_geometry(grid)
    zc = create_ocean_z_star(3, 300.0, 100.0, 100.0)
    mask = jnp.ones((3, 4), dtype=jnp.float64)
    um, vm = compute_face_masks(mask, grid)
    J = jnp.ones_like(mask)
    T = jnp.array([[[18., 18., 4.], [18., 18., 4.], [8., 8., 4.], [8., 8., 4.]]]*3)
    S = jnp.full_like(T, 35.0)
    cfg = MLEConfig(no_mle_in_convection=False, through_fct=True)

    def transport(t, s):
        rho = make_eos_fn()(t, s, jnp.zeros_like(t))
        n2 = jnp.zeros((*t.shape[:-1], t.shape[-1]-1), dtype=t.dtype)
        return mle_tracer_tendency_latlon_cgrid(
            t, s, rho, n2, mask, um, vm, zc, J, geom, cfg,
            return_horizontal_transport=True)

    u, v = jax.jit(transport)(T, S)
    assert jax.config.jax_enable_x64 and get_policy().compute == np.dtype(np.float64)
    ubits, vbits = np.asarray(u).view(np.uint64), np.asarray(v).view(np.uint64)
    # Exact fp64 witnesses; removing the MLE diagnosis or its face wiring
    # changes both the count and the source-ordered sum bits.
    assert np.count_nonzero(ubits) == 18
    assert np.count_nonzero(vbits) == 0
    assert np.asarray(jnp.sum(u)).view(np.uint64).item() == 0


G9_RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                 "gridladder_rounds/round46/oracle_g9_r46/twin_a")


@pytest.mark.skipif(not G9_RECORD.is_dir(), reason="no admitted G9 record")
def test_l5_jitted_traced_x64_step_consumes_mle_transport(cards):
    """The production L5 step changes exactly when its staged wiring acts."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    import sys
    sys.path[:0] = [str(CASES), str(CASES / "nemo_testcase_g5_orca2ice")]
    import g5_records as g5

    l5 = cards[1]
    run = l5.recipe
    fw, sf = g5.nemo_supplied_forcing(l5, G9_RECORD, 1)
    configs = (
        run.model_config,
        run.model_config._replace(
            physics=run.model_config.physics._replace(mle=None)),
    )
    outputs = []
    for cfg in configs:
        model = LatLonCGridOceanModel(run.grid, run.z_coord, cfg)
        outputs.append(jax.device_get(jax.jit(
            lambda state, fresh, surface: model.step(
                state, l5.dt_s, freshwater=fresh, surface_forcing=surface)
        )(run.initial_state, fw, sf)))
    counts = tuple(int(np.count_nonzero(
        np.asarray(getattr(outputs[0], name).data).view(np.uint64)
        != np.asarray(getattr(outputs[1], name).data).view(np.uint64)))
        for name in ("T", "S", "u", "v", "eta"))
    print("L5 staged MLE exact unequal counts", counts)
    assert counts == (24588, 21880, 0, 0, 0)
