"""Lane B L5: certified G9 plus ORCA2 mixed-layer eddies."""

from __future__ import annotations

import importlib.util
import hashlib
import inspect
import pickle
import subprocess
import sys
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
STATEMENT_REF = _module("mle_statement_reference", L5 / "mle_statement_reference.py")
CERTIFIED_VALUES = _module(
    "certified_card_values", L5 / "certified_card_values.py"
)
RECORDS = _module(
    "l5_records",
    CASES / "nemo_testcase_l3_orca2eiv" / "l3_records.py",
)
REVIEW_CAPTURE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds/"
    "round21/review_reproduced_codex_r21/l5_capture.npz")


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
    not (ORCA2.parents[2] / "src/OCE/TRA/tramle.F90").is_file(),
    reason="no NEMO source tree")
def test_l5_observer_zero_first_defined_domain():
    source = (ORCA2.parents[2] / "src/OCE/TRA/tramle.F90").read_text()
    instrumented = INSTRUMENT.instrument(source)
    INSTRUMENT.validate(instrumented)
    assert "zfield(:,:,:) = 0._wp" in instrumented
    assert "IF( llwet ) zfield(ji,jj,jk) = pfield(ji,jj,jk)" in instrumented
    planted = instrumented.replace("l5_canon_3d(pFu,'U')", "pFu", 1)
    with pytest.raises(ValueError, match="canonical"):
        INSTRUMENT.validate(planted)
    planted = instrumented.replace(
        INSTRUMENT.BEFORE + INSTRUMENT.BODY_ANCHOR,
        INSTRUMENT.BODY_ANCHOR + INSTRUMENT.BEFORE,
        1,
    )
    with pytest.raises(ValueError, match="nested"):
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
    assert (mle.ce, mle.lat_ref_deg, mle.rho_c_mle, mle.ref_depth_m,
            mle.max_grid_scale_m, mle.mld_uv, mle.no_mle_in_convection,
            mle.through_fct, mle.nemo_association) == (
        0.06, 20.0, 0.01, 10.0, 111.0e3, "min", False, True, True)
    assert c5.gm_redi is None and c5.tracer_advection == "fct2"
    for name in ("grid", "z_coord", "initial_state"):
        assert pickle.dumps(getattr(g9.recipe, name)) == pickle.dumps(
            getattr(l5.recipe, name)), name
    assert recipe._orca2g1_selector_failures(g9) == []
    assert recipe._orca2g1_selector_failures(l5) == []


@pytest.mark.parametrize("change", [
    {"ce": 0.07}, {"lat_ref_deg": 21.0}, {"rho_c_mle": 0.02},
    {"ref_depth_m": 11.0}, {"max_grid_scale_m": 110.0e3},
    {"mld_uv": "avg"}, {"no_mle_in_convection": True},
    {"through_fct": False}, {"nemo_association": False},
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


def test_l5_rotation_rate_is_card_constant_and_legacy_call_is_unchanged(cards):
    from legoesm.ocean.physics.lateral_mixing.mle import mle_coefficient

    omega = cards[1].recipe.model_config.constants.Omega
    nemo = np.float64(mle_coefficient(
        0.06, 20.0, omega=omega, nemo_association=True))
    legacy = np.float64(mle_coefficient(0.06, 20.0))
    assert nemo.view(np.uint64).item() == 0x3FCECB14B9A0B306
    assert legacy.view(np.uint64).item() == 0x3FCECB34931005E8


@pytest.mark.skipif(not REVIEW_CAPTURE.is_file(), reason="no preserved real L5 capture")
def test_l5_real_operands_match_statement_reference_bit_for_bit(cards):
    from legoesm.ocean.physics.lateral_mixing.mle import mle_mld_and_buoyancy
    from legoesm.ocean.physics.lateral_mixing.mle_latlon_cgrid import (
        mle_tracer_tendency_latlon_cgrid,
    )

    assert hashlib.sha256(REVIEW_CAPTURE.read_bytes()).hexdigest() == (
        "12ff643eec84e03e53683b62513ebca0efc628526d7ad610316110bba2799dd6"
    )
    d = np.load(REVIEW_CAPTURE)
    ref_u, ref_v, ref_mld, ref_bm = STATEMENT_REF.nemo_mle_transport_reference(d)
    expected_hashes = {
        "u": "0e93b06e5d92951ac5a6462f9c03287684a72b30bdf315c4813d316987ab105e",
        "v": "2b3a390c267a476e7d2e5c309590af42bf8026196c4ab9a0c67f1403d4f5f3b4",
        "mld": "210aaa6af659cec45ff0987286053d1604ef0fe9244d041547d66e3ac10f0226",
        "bm": "73d547f15e707453f14b09a6cf60d05e045e5bcbad4ec4f1948ee9bc9c95e332",
    }
    for name, array in (("u", ref_u), ("v", ref_v), ("mld", ref_mld), ("bm", ref_bm)):
        assert hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() == expected_hashes[name]

    run = cards[1].recipe
    cfg = run.model_config
    rho, hk, wet = (jnp.asarray(d[k]) for k in ("in2", "in6", "is_active"))
    mld, bm, _ = mle_mld_and_buoyancy(
        rho, hk, wet,
        z_faces=jnp.abs(run.z_coord.z_half_ref),
        z_centers_ref=jnp.abs(run.z_coord.t_depth_ref),
        rho_c_mle=cfg.physics.mle.rho_c_mle,
        ref_depth_m=cfg.physics.mle.ref_depth_m,
        rho0=cfg.constants.rho_0,
        grav=cfg.constants.g,
        nemo_association=True,
    )
    wet2 = np.asarray(d["in3"]) > 0.0
    got_mld = np.where(wet2, np.asarray(mld), 0.0)
    got_bm = np.where(wet2, np.asarray(bm), 0.0)
    assert np.array_equal(got_mld.view(np.uint64), ref_mld.view(np.uint64))
    assert np.array_equal(got_bm.view(np.uint64), ref_bm.view(np.uint64))

    T, S, mask, um, vm = (jnp.asarray(d[k]) for k in
                           ("in0", "in1", "in3", "in4", "in5"))
    n2 = jnp.zeros((*T.shape[:-1], T.shape[-1] - 1), dtype=T.dtype)
    got_u, got_v = mle_tracer_tendency_latlon_cgrid(
        T, S, rho, n2, mask, um, vm, run.z_coord,
        jnp.asarray(d["stretch_half0"]), run.grid, cfg.physics.mle,
        h_k=hk, rho0=cfg.constants.rho_0, grav=cfg.constants.g,
        omega=cfg.constants.Omega, return_horizontal_transport=True,
    )
    got_u, got_v = np.asarray(got_u), np.asarray(got_v)
    assert np.array_equal(got_u.view(np.uint64), ref_u.view(np.uint64))
    assert np.array_equal(got_v.view(np.uint64), ref_v.view(np.uint64))
    assert np.count_nonzero(got_u) == 8502
    assert np.count_nonzero(got_v) == 8764
    assert np.count_nonzero(got_v[-1]) == 16

    def transport_loss(stretch):
        u_transport, v_transport = mle_tracer_tendency_latlon_cgrid(
            T, S, rho, n2, mask, um, vm, run.z_coord,
            stretch, run.grid, cfg.physics.mle,
            h_k=hk, rho0=cfg.constants.rho_0, grav=cfg.constants.g,
            omega=cfg.constants.Omega, return_horizontal_transport=True,
        )
        return jnp.sum(u_transport * u_transport) + jnp.sum(
            v_transport * v_transport)

    stretch_gradient = np.asarray(jax.jit(jax.grad(transport_loss))(
        jnp.asarray(d["stretch_half0"])))
    assert stretch_gradient.shape[1] == 180
    assert np.all(np.isfinite(stretch_gradient))
    assert np.count_nonzero(stretch_gradient) > 0


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
    with pytest.raises(ValueError, match="stage rn2 operand"):
        mle_tracer_tendency_latlon_cgrid(
            T, S, make_eos_fn()(T, S, jnp.zeros_like(T)),
            jnp.zeros((*T.shape[:-1], T.shape[-1] - 1), dtype=T.dtype),
            mask, um, vm, zc, J, geom,
            cfg._replace(no_mle_in_convection=True),
            return_horizontal_transport=True,
        )


def test_l5_closed_south_v_metric_has_finite_live_stretch_gradient():
    """The NEMO metric ratio must not differentiate through south-row 0/0."""
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
    zero_south = geom._replace(
        dx_v=geom.dx_v.at[0].set(0.0),
        dy_v=geom.dy_v.at[0].set(0.0),
    )
    safe_south = zero_south._replace(dy_v=zero_south.dy_v.at[0].set(1.0))
    zc = create_ocean_z_star(3, 300.0, 100.0, 100.0)
    mask = jnp.ones((3, 4), dtype=jnp.float64)
    um, vm = compute_face_masks(mask, grid)
    T = jnp.array([[[18., 18., 4.], [18., 18., 4.],
                    [8., 8., 4.], [8., 8., 4.]]]*3)
    S = jnp.full_like(T, 35.0)
    rho = make_eos_fn()(T, S, jnp.zeros_like(T))
    n2 = jnp.zeros((*T.shape[:-1], T.shape[-1] - 1), dtype=T.dtype)
    cfg = MLEConfig(
        no_mle_in_convection=False, through_fct=True,
        nemo_association=True)
    stretch = jnp.ones_like(mask)

    def transport(jacobian, geometry):
        return mle_tracer_tendency_latlon_cgrid(
            T, S, rho, n2, mask, um, vm, zc, jacobian, geometry, cfg,
            omega=grid.omega,
            return_horizontal_transport=True)

    guarded = jax.jit(lambda x: transport(x, zero_south))(stretch)
    reference = jax.jit(lambda x: transport(x, safe_south))(stretch)
    for got, expected in zip(guarded, reference):
        np.testing.assert_array_equal(
            np.asarray(got).view(np.uint64),
            np.asarray(expected).view(np.uint64))

    def loss(jacobian):
        u_transport, v_transport = transport(jacobian, zero_south)
        return jnp.sum(u_transport * u_transport) + jnp.sum(
            v_transport * v_transport)

    gradient = jax.jit(jax.grad(loss))(stretch)
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.count_nonzero(np.asarray(gradient)) > 0


G9_RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                 "gridladder_rounds/round46/oracle_g9_r46/twin_a")
L5_RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "gridladder_vmix_rounds/round21/oracle_l5_r21/twin_a")
G5_CONTROL_RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/"
    "round31/oracle_g5_r31/twin_a")
PHASE3 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")
ORCA2_INPUT_DECK = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
OTHER_ORCA2_PROGRAM_ROOTS = {
    "ORCA2G5-ice": G5_CONTROL_RECORD,
    "ORCA2G6-vec": PHASE3 / "gridladder_rounds/round36/oracle_g6_r36/twin_a",
    "ORCA2G7-drg": PHASE3 / "gridladder_rounds/round41/oracle_g7_r41/twin_a",
    "ORCA2G8-ldf": PHASE3 / "gridladder_rounds/round45/oracle_g8_r45/twin_a",
    "ORCA2G9-tadv": PHASE3 / "gridladder_rounds/round46/oracle_g9_r46/twin_a",
    "ORCA2V1-tke": PHASE3 / "gridladder_vmix_rounds/round1/oracle_v1_r1/twin_a",
    "ORCA2V2-evd": PHASE3 / "gridladder_vmix_rounds/round12/oracle_v2_r12/twin_a",
    "ORCA2V3-ddm": PHASE3 / "gridladder_vmix_rounds/round13/oracle_v3_r13/twin_a",
    "ORCA2V4-iwm": PHASE3 / "gridladder_vmix_rounds/round14/oracle_v4_r14/twin_a",
    "ORCA2V5-havtb": PHASE3 / "gridladder_vmix_rounds/round15/oracle_v5_r15/twin_a",
    "ORCA2L1-bbl": PHASE3 / "gridladder_vmix_rounds/round16/oracle_l1_r16/twin_a",
    "ORCA2L2-iso": PHASE3 / "gridladder_vmix_rounds/round17/oracle_l2_r17/twin_a",
    "ORCA2L3-eiv": PHASE3 / "gridladder_vmix_rounds/round20/oracle_l3_r20/twin_a",
}
OTHER_CONFIG_PICKLE_HASHES_AT_9039513 = {
    "ORCA2G5-ice": "a18a471582b3ae85ded987caf928db93f651e70e8150bed6825c76f311583cc8",
    "ORCA2G6-vec": "de77253d194cc9e8edb56ddafad1491cd591adcfb77ea321255a5c84503a1baa",
    "ORCA2G7-drg": "8664ff1901ab99e88399496769bc65eb10f92963ae6310b93c77982c03fe8f4d",
    "ORCA2G8-ldf": "e31d8a0b36e4627c3eb61debf7f828ed3cfea76f73fe99bbcd02d6fdbe200d1e",
    "ORCA2G9-tadv": "89e98e7fe5fb28aa20c794237665f214beb479e980c9e512236031a208c04c21",
    "ORCA2V1-tke": "48d5eb1e6b3d09197839e9d1028e195a70abcdbf1fbb2f0ae942a314b824f678",
    "ORCA2V2-evd": "824ee5f546f5106d051476574c69b63d55c4c97c1079ea9b45f3a8107f6f449b",
    "ORCA2V3-ddm": "06d89fb358e9846f1828db6d7a7a2d962e243ef308d3132d06c128f89417af1c",
    "ORCA2V4-iwm": "68cd8ed2f399392b3ab7b34c77d73ee1114789af89a37d5370fb24aba41a1b97",
    "ORCA2V5-havtb": "68cd8ed2f399392b3ab7b34c77d73ee1114789af89a37d5370fb24aba41a1b97",
    "ORCA2L1-bbl": "d0ddd05f2f294b707fa5176b0416aef3556673e4f85b072e52264fc5c0838091",
    "ORCA2L2-iso": "71c21ad9604116b2eaa6e09164199d42dffdfb32ae6cae4a9964f1fe0402f453",
    "ORCA2L3-eiv": "131db293410a90d7cd0e79cfa23e8ee02f79c28e3a7f9cbb1284db233e20b1f4",
    "LOCK_EXCHANGE-zco": "1b1fcbbde395d226d48ddd0ba2cab00ecd1ebf2ecb4a68e3809d3ea2638a6d19",
    "OVERFLOW-zps": "40c46c0f3f3054184518a7da2f43302a68f55118251cdc369200b5aa27f10214",
    "GYRE-zco": "ae6aed2e3c3c5e2d428ee4d3ada3db7b58aee1d0a80b1da2737dcdebee370afe",
    "ORCA2-zps": "6bf153d931cce0d9ca41d879bf9462dab8a85488f1dbc03681d20947c0c50a7f",
    "VORTEX-zco": "4d4cb267e633879ba888edf67786913e6e256975c77a8cc5e234186f64d3031b",
    "VORTEX_VEC-zco": "2daba59aa11e98dc952a012bf1f45fdd7cadfa1998a97fb421729b92052a2c12",
    "VORTEX-15km-zco": "4d4cb267e633879ba888edf67786913e6e256975c77a8cc5e234186f64d3031b",
    "VORTEX_VEC-15km-zco": "2daba59aa11e98dc952a012bf1f45fdd7cadfa1998a97fb421729b92052a2c12",
    "VORTEX-10km-zco": "4d4cb267e633879ba888edf67786913e6e256975c77a8cc5e234186f64d3031b",
    "VORTEX_VEC-10km-zco": "2daba59aa11e98dc952a012bf1f45fdd7cadfa1998a97fb421729b92052a2c12",
    "VORTEX_SMT-zps": "4d4cb267e633879ba888edf67786913e6e256975c77a8cc5e234186f64d3031b",
    "VORTEX_SMT_VEC-zps": "2daba59aa11e98dc952a012bf1f45fdd7cadfa1998a97fb421729b92052a2c12",
    "VORTEX_SMT1_VEC-zps": "5be0648aa6eea1817346d040d9ef991bf875f9b640b31ebc2792cacc2096a4ee",
    "VORTEX_SMT2_VEC-zps": "d538d4a146e74470b655bb1d496cef05a3cb2b347b6710b3eeb3d01316b6a7db",
    "VORTEX_SMT3_VEC-zps": "ff71c950ec7e90955123ca6de525ab6892d677392c9aea945c8382a449a1a019",
    "VORTEX_SMT4_VEC-zps": "ef32c0cee68bf111d0e31e36558e4a79522032cf952ae673a44a74b899e26a85",
}

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
    assert counts == (24584, 21878, 0, 0, 0)


@pytest.mark.skipif(not L5_RECORD.is_dir(), reason="no admitted L5 record")
def test_l5_record_jitted_traced_x64_transport_and_wiring(cards):
    """The jitted fp64 L5 step matches its record and needs every L5 wire."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSL5MLETransportTrace,
        _NEMOWSRK3TestHooks,
    )
    import sys
    sys.path[:0] = [str(CASES), str(CASES / "nemo_testcase_g5_orca2ice")]
    import g5_records as g5
    from nemo_testcase_phase3_trajectory_gate import comparison_masks

    l5 = cards[1]
    run = l5.recipe
    assert jax.config.jax_enable_x64
    assert not jax.config.jax_disable_jit
    assert get_policy().compute == np.dtype(np.float64)
    fw, sf = g5.nemo_supplied_forcing(l5, L5_RECORD, 1)

    def compiled_step(model):
        return jax.device_get(jax.jit(
            lambda state, fresh, surface: model.step(
                state, l5.dt_s, freshwater=fresh, surface_forcing=surface)
        )(run.initial_state, fw, sf))

    observed = compiled_step(LatLonCGridOceanModel(
        run.grid, run.z_coord, run.model_config,
        iwm_forcing=run.iwm_forcing,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_l5_mle_transport=True),
    ))
    assert isinstance(observed.mass_flux_w, _NEMOWSL5MLETransportTrace)
    witness_paths = [
        L5_RECORD.parent / f"twin_{label}" / "oracle_l5_mle_kt00000001.bin"
        for label in "abc"
    ]
    witness_bytes = [path.read_bytes() for path in witness_paths]
    assert witness_bytes[0] == witness_bytes[1] == witness_bytes[2]
    record = RECORDS.read(L5_RECORD / "oracle_l5_mle_kt00000001.bin")
    nlev = run.z_coord.n_levels
    record_shape = tuple(int(record[name]) for name in ("nx", "ny", "nz"))
    record_wet = RECORDS.defined_masks(
        L5_RECORD / "mesh_mask.nc", record_shape)

    for point, card_slice in (("u", (slice(None), slice(1, None))),
                              ("v", (slice(1, None), slice(None)))):
        wet = np.ascontiguousarray(np.swapaxes(
            record_wet[point.upper()][2:-2, 2:-2, :nlev], 0, 1))
        for boundary in ("before", "delta", "after"):
            name = f"{boundary}_{point}"
            raw = np.asarray(record[name])[2:-2, 2:-2]
            assert not np.any(raw[..., nlev:]), name
            ref = np.ascontiguousarray(np.swapaxes(raw[..., :nlev], 0, 1))
            got = np.asarray(getattr(observed.mass_flux_w, name))[card_slice]
            for label, mask in (("wet", wet), ("dry", ~wet)):
                classes = comparison_masks(ref, got, mask)
                assert not np.any(classes["normal_difference"]), (name, label)
                assert not np.any(classes["flushed_subnormal"]), (name, label)
            if boundary == "delta":
                assert np.count_nonzero(ref[wet]) > 0, point
                assert not np.any(ref[~wet].view(np.uint64)), point

    stage = g5.read_step(L5_RECORD, 1, [L5_RECORD], nlev)
    for field, key in (("T", "3_tn_all_aa"), ("S", "3_sn_all_aa")):
        ref = stage[key]
        got = np.asarray(getattr(observed, field).data)
        classes = comparison_masks(ref, got, np.ones(ref.shape, dtype=bool))
        assert not np.any(classes["normal_difference"]), field

    # Each plant removes a distinct L5 production wire after construction,
    # then executes the same jitted step with traced state and forcing leaves.
    cfg = run.model_config
    mle = cfg.physics.mle
    plants = {
        "without_mle": None,
        "without_stage3_fct_route": mle._replace(through_fct=False),
        "without_nemo_association": mle._replace(nemo_association=False),
    }
    for name, planted_mle in plants.items():
        planted = LatLonCGridOceanModel(
            run.grid, run.z_coord, cfg, iwm_forcing=run.iwm_forcing)
        planted.config = cfg._replace(
            physics=cfg.physics._replace(mle=planted_mle))
        after = compiled_step(planted)
        counts = {}
        for field, key in (("T", "3_tn_all_aa"), ("S", "3_sn_all_aa")):
            ref = stage[key]
            got = np.asarray(getattr(after, field).data)
            classes = comparison_masks(ref, got, np.ones(ref.shape, dtype=bool))
            counts[field] = int(classes["normal_difference"].sum())
        print(name, counts)
        assert counts["T"] > 0 and counts["S"] > 0, name


@pytest.mark.skipif(not G5_CONTROL_RECORD.is_dir(), reason="no admitted G5 record")
def test_round21_config_census():
    """Pin every non-L5 resolved configuration pickle."""

    expected_cards = {
        "ORCA2G1-flat", "ORCA2G1b-vec", "ORCA2G2-zps", "ORCA2G3-ts",
        "ORCA2G4-sbc", "ORCA2G5-ice", "ORCA2G6-vec", "ORCA2G7-drg",
        "ORCA2G8-ldf", "ORCA2G9-tadv", "ORCA2V1-tke", "ORCA2V2-evd",
        "ORCA2V3-ddm", "ORCA2V4-iwm", "ORCA2V5-havtb", "ORCA2L1-bbl",
        "ORCA2L2-iso", "ORCA2L3-eiv", "ORCA2L5-mle",
    }
    assert set(recipe._ORCA2G_JPK) == expected_cards
    assert set(recipe._ORCA2L_MLE_CASES) == {"ORCA2L5-mle"}
    assert {case for case, on in recipe._ORCA2G_TADV.items() if on} == {
        "ORCA2G9-tadv", "ORCA2L3-eiv", "ORCA2L5-mle"}
    non_ladder = {
        "LOCK_EXCHANGE-zco", "OVERFLOW-zps", "GYRE-zco", "ORCA2-zps",
        "VORTEX-zco", "VORTEX_VEC-zco", "VORTEX-15km-zco",
        "VORTEX_VEC-15km-zco", "VORTEX-10km-zco",
        "VORTEX_VEC-10km-zco", "VORTEX_SMT-zps", "VORTEX_SMT_VEC-zps",
        "VORTEX_SMT1_VEC-zps", "VORTEX_SMT2_VEC-zps",
        "VORTEX_SMT3_VEC-zps", "VORTEX_SMT4_VEC-zps",
    }
    dispatch = inspect.getsource(recipe.build_nemo_testcase_card)
    assert all(f'"{case}"' in dispatch for case in non_ladder)
    assert non_ladder.isdisjoint(recipe._ORCA2L_MLE_CASES)

    # Full resolved-model-configuration pickle census against the pre-review
    # tip.  These are configuration hashes, not compiled step programs.
    assert set(OTHER_CONFIG_PICKLE_HASHES_AT_9039513) == (
        set(OTHER_ORCA2_PROGRAM_ROOTS) | non_ladder)
    for case, expected in OTHER_CONFIG_PICKLE_HASHES_AT_9039513.items():
        if case in OTHER_ORCA2_PROGRAM_ROOTS:
            root = OTHER_ORCA2_PROGRAM_ROOTS[case]
        elif case == "ORCA2-zps":
            root = ORCA2_INPUT_DECK
        else:
            root = None
        if root is not None and not root.is_dir():
            pytest.skip(f"missing immutable control deck {root}")
        card = recipe.build_nemo_testcase_card(case, deck_root=root)
        digest = hashlib.sha256(
            pickle.dumps(card.recipe.model_config)).hexdigest()
        assert digest == expected, case
        assert getattr(
            getattr(card.recipe.model_config, "physics", None), "mle", None
        ) is None, case



@pytest.mark.parametrize("case", tuple(CERTIFIED_VALUES.EXPECTED))
def test_round21_certified_card_two_step_values(case):
    """Run each certified card in a fresh compiler process and pin its bits."""
    root = CERTIFIED_VALUES.ROOTS[case]
    if not root.is_dir():
        pytest.skip(f"missing immutable certified record {root}")
    result = subprocess.run(
        [sys.executable, str(L5 / "certified_card_values.py"), case],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == f"{case} {CERTIFIED_VALUES.EXPECTED[case]}"
