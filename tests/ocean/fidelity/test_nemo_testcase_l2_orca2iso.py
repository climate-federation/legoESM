"""Lane B L2: certified G5 plus ORCA2's standard isoneutral diffusion."""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe

from tests.ocean.fidelity.test_nemo_testcase_v1_orca2tke import (
    CARD_FILES,
    CASES,
    G5,
    INPUTS,
    ORCA2_DECK,
    _differs,
    _flat,
    _root,
    _values,
)

L2 = CASES / "nemo_testcase_l2_orca2iso"
RUN = L2 / "run.sh"
REFUSED_SMOKE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds/"
    "round17/oracle_l2_r17_refused1/smoke_2step")
L2_RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds/"
    "round17/oracle_l2_r17")
L2_WALK_RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "gridladder_vmix_rounds/round18/oracle_l2_r18_ldf_walk/"
    "oracle_l2_ldf_walk_kt00000001.bin")
L2_OBSERVER = L2_RECORD / "observer_ldf_slopes"
needs_l2_record = pytest.mark.skipif(
    not (L2_RECORD / "twin_a/oracle_step_entry_kt00000001.bin").is_file(),
    reason="the operator's admitted L2 record is not on this machine")


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DECK = _module("l2_deck", L2 / "l2_deck.py")
RECORDS = _module("l2_records", L2 / "l2_records.py")


@pytest.fixture(scope="module")
def ladder():
    sys.path.insert(0, str(CASES))
    return _module(
        "l2_ladder", CASES / "nemo_testcase_g2_orca2bathy_ladder.py")


@pytest.fixture(autouse=True)
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


def _l2_root(base: Path) -> Path:
    root = _root(base, G5, "l2")
    (root / "namelist_cfg").write_text(DECK.render_text(
        (root / "namelist_cfg").read_text()))
    return root


def test_l2_deck_is_g5_plus_the_complete_resolved_namtra_ldf_group_only(tmp_path):
    root = _l2_root(tmp_path)
    g5, l2 = _values(G5 / "namelist_cfg"), _values(root / "namelist_cfg")
    changed = {k: (g5.get(k), l2.get(k)) for k in set(g5) | set(l2)
               if g5.get(k) != l2.get(k)}
    assert changed == {k: (g5.get(k), value)
                       for k, value in recipe._ORCA2L2_LDF_FIELDS.items()
                       if g5.get(k) != value}
    assert {k: v for k, v in l2.items() if k.startswith("namtra_ldf.")} == (
        recipe._ORCA2L2_LDF_FIELDS)
    assert l2["namtra_eiv.ln_ldfeiv"] is False
    with pytest.raises(ValueError, match="certified G5"):
        DECK.render_text(DECK.render_text((G5 / "namelist_cfg").read_text()))


@pytest.mark.skipif(not ORCA2_DECK.is_file(), reason="no NEMO tree")
def test_l2_namtra_ldf_values_resolve_from_orca2_cfg_plus_reference():
    cfg = _values(ORCA2_DECK)
    ref = _values(ORCA2_DECK.with_name("namelist_ref"))
    resolved = {key: cfg.get(key, ref[key]) for key in recipe._ORCA2L2_LDF_FIELDS}
    assert resolved == recipe._ORCA2L2_LDF_FIELDS
    # Full ORCA2 enables EIV; L2 deliberately does not copy that later rung.
    assert cfg["namtra_eiv.ln_ldfeiv"] is True


def test_reader_refuses_l2_group_on_g5_and_off_group_on_l2(tmp_path):
    root = _l2_root(tmp_path)
    with pytest.raises(ValueError, match="namtra_ldf"):
        recipe.read_orca2g1_deck(root, "ORCA2G5-ice")
    with pytest.raises(ValueError, match="namtra_ldf"):
        recipe.read_orca2g1_deck(G5, "ORCA2L2-iso")
    text = (root / "namelist_cfg").read_text()
    (root / "namelist_cfg").write_text(text.replace("rn_sw_triad     = 1", "rn_sw_triad     = 0"))
    with pytest.raises(ValueError, match="namtra_ldf"):
        recipe.read_orca2g1_deck(root, "ORCA2L2-iso")


@pytest.fixture(scope="module")
def cards(tmp_path_factory):
    if not all((INPUTS / name).is_file() for name in CARD_FILES):
        pytest.skip("no ORCA2 inputs")
    base = tmp_path_factory.mktemp("l2")
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    try:
        return (
            recipe.build_orca2g5_ice_card(_root(base, G5, "g5")),
            recipe.build_orca2l2_iso_card(_l2_root(base)),
        )
    finally:
        set_policy(before)


def test_l2_card_is_g5_plus_shared_isoneutral_config_only(cards):
    g5, l2 = cards
    a, b = _flat(g5.recipe.model_config), _flat(l2.recipe.model_config)
    changed = {k for k in set(a) | set(b) if _differs(a.get(k), b.get(k))}
    assert changed and all(k == "gm_redi" or k.startswith("gm_redi.") for k in changed)
    assert recipe._orca2l2_isoneutral_selected(l2.recipe.model_config.gm_redi)
    assert l2.recipe.model_config.gm_redi == recipe._orca2_isoneutral_gm_redi(
        0.5 * recipe._SMT3_RN_UD * recipe._SMT3_RN_LD,
        slope_prd_evaluation="nemo_eos80_literal", north_fold=True,
        redi_flux_reference_thickness_evaluation="nemo_native",
        redi_flux_geometry_stage="rk3_kmm", slope_uv_source_rounding=True)
    assert l2.recipe.model_config.gm_redi.slope_prd_geometry_stage == "before_step"
    assert l2.recipe.model_config.gm_redi.kappa_GM == 0.0
    assert l2.recipe.model_config.tracer_advection == "none"
    assert l2.recipe.model_config.bbl_diffusive_option == 0
    s5, s2 = g5.recipe.initial_state, l2.recipe.initial_state
    assert [name for name in s5._fields if _differs(
        getattr(getattr(s5, name), "data", getattr(s5, name)),
        getattr(getattr(s2, name), "data", getattr(s2, name)))] == []
    assert recipe._orca2g1_selector_failures(l2) == []
    assert recipe._orca2g1_selector_failures(g5) == []


def test_selector_refuses_each_l2_isoneutral_wiring_or_nonzero_eiv(cards):
    _, l2 = cards
    gm = l2.recipe.model_config.gm_redi
    for change in ({"msc_stabilize": False},
                   {"kappa_redi_lat_scaling": False},
                   {"kappa_redi_horizontal_evaluation": "cosine_scaled"},
                   {"slope_prd_geometry_stage": "current_step"},
                   {"slope_prd_evaluation": "nemo_literal"},
                   {"nemo_slope_north_fold": False},
                   {"redi_flux_reference_thickness_evaluation": "tpoint_alias"},
                   {"redi_flux_geometry_stage": "step_entry"},
                   {"slope_uv_source_rounding": False},
                   {"kappa_GM": 1.0}):
        bad = l2.recipe.model_config._replace(gm_redi=gm._replace(**change))
        failures = recipe._orca2g1_selector_failures(
            l2._replace(recipe=l2.recipe._replace(model_config=bad)))
        assert "ln_traldf_lap+iso+msc" in failures or "ln_ldfeiv=F" in failures


def test_l2_mode20_coefficient_is_the_masked_nemo_metric_value(cards):
    from legoesm.grids.latlon import ensure_geometry
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        nemo_iso_face_masks,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        static_kappa_redi_override,
    )

    _, l2 = cards
    run, state = l2.recipe, l2.recipe.initial_state
    ku, kv = static_kappa_redi_override(run.model_config.gm_redi, run.grid)
    geom = ensure_geometry(run.grid)
    want_u = 0.5 * recipe._SMT3_RN_UD * np.maximum(
        np.asarray(geom.dx_u[:, 1:]), np.asarray(geom.dy_u[:, 1:]))
    want_v = 0.5 * recipe._SMT3_RN_UD * np.maximum(
        np.asarray(geom.dx_v[1:, :]), np.asarray(geom.dy_v[1:, :]))
    np.testing.assert_array_equal(np.asarray(ku), want_u)
    np.testing.assert_array_equal(np.asarray(kv), want_v)
    um, vm, _ = nemo_iso_face_masks(
        state.u_mask.data, state.v_mask.data, run.z_coord.is_active,
        grid=run.grid, north_fold=True)
    masked = np.concatenate([
        (np.asarray(ku)[..., None] * np.asarray(um)).ravel(),
        (np.asarray(kv)[..., None] * np.asarray(vm)).ravel(),
    ])
    wet = masked[masked != 0.0]
    assert wet.min() < wet.max()
    assert min(want_u.min(), want_v.min()) < 140.0
    assert max(want_u.max(), want_v.max()) > 2000.0
    assert np.any(np.asarray(ku) != run.model_config.gm_redi.kappa_Redi)
    if (L2_RECORD / "twin_a/mesh_mask.nc").is_file():
        import netCDF4
        with netCDF4.Dataset(L2_RECORD / "twin_a/mesh_mask.nc") as ds:
            nemo_u = 0.009 * np.maximum(ds["e1u"][0], ds["e2u"][0])
            nemo_v = 0.009 * np.maximum(ds["e1v"][0], ds["e2v"][0])
        np.testing.assert_array_equal(np.asarray(ku), nemo_u)
        np.testing.assert_array_equal(np.asarray(kv), nemo_v)
        assert (float(nemo_v.min()), float(nemo_u.max())) == (
            134.99730994125315, 2001.5806220738168)


@pytest.fixture(scope="module")
def l2_native_slopes(cards):
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.eos import (
        make_eos_fn,
        nemo_roquet_density_anomaly_ratio,
    )
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        compute_nemo_native_slopes,
    )
    from legoesm.ocean.vertical import compute_ocean_jacobian

    _, l2 = cards
    run, state = l2.recipe, l2.recipe.initial_state
    z_coord, cfg = run.z_coord, run.model_config.gm_redi
    T, S = state.T.data, state.S.data
    eta, H_bathy = state.eta.data, state.H_bathy.data
    jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)
    active = state.land_mask.data[..., None] * z_coord.is_active
    # stprk3's rank-5 EOS overload reads the shared gdept_1d ladder, including
    # in a partial bottom cell (eosbn2.f90:357), then applies the column r3t.
    gdept = z_coord.t_depth_ref[None, None, :] * jacobian[..., None]
    expected_prd = nemo_roquet_density_anomaly_ratio(
        T, S, jnp.zeros_like(gdept), rho0=run.model_config.rho_0,
        geometric_depth_m=gdept, tmask=active)
    rho = run.model_config.rho_0 * (1.0 + expected_prd)
    bundle = LatLonCGridOceanModel(
        run.grid, z_coord, run.model_config)._tke_step_entry_n2_bundle(state)
    result = compute_nemo_native_slopes(
        rho, T, S, state.land_mask.data, state.u_mask.data,
        state.v_mask.data, z_coord, run.grid, cfg,
        make_eos_fn("nemo_eos80", rho0=run.model_config.rho_0),
        rho_0=run.model_config.rho_0, g=run.model_config.constants.g,
        active_3d=z_coord.is_active,
        jacobian=jacobian, eta=eta, H_bathy=H_bathy,
        prd_jacobian=jacobian,
        pn2_override=bundle.rn2b, e3w_override=bundle.e3w_Kmm,
        return_diagnostics=True)
    return l2, result, expected_prd, gdept, active


def test_l2_slope_density_is_eos80_not_the_simplified_eos(l2_native_slopes):
    from legoesm.ocean.eos import NemoSEOSConfig, nemo_seos_prd_literal

    l2, result, expected_prd, gdept, active = l2_native_slopes
    got = np.asarray(result[4]["prd"])
    np.testing.assert_array_equal(got, np.asarray(expected_prd))
    # Independent kt-1 values written by NEMO immediately after
    # stprk3.f90:188.  These anchors make this test fail if the production
    # path and the expected-array helper are changed together.
    assert got[1, 49, 0] == 0.0015760654114520545
    assert got[80, 7, 24] == 0.0017764720643707044
    assert got[104, 48, 0] == -7.81552393591678e-06
    state = l2.recipe.initial_state
    simplified = np.asarray(nemo_seos_prd_literal(
        state.T.data, state.S.data, gdept,
        NemoSEOSConfig(rho0=l2.recipe.model_config.rho_0))) * np.asarray(active)
    assert np.max(np.abs(got - simplified)) > 1.0e-4


def test_l2_north_fold_opens_faces_crosses_pivot_and_folds_slopes(
        l2_native_slopes):
    l2, result, _, _, _ = l2_native_slopes
    uslp, vslp, wslpi, wslpj, diagnostics = result
    fold = l2.recipe.grid.fold
    raw_vmask = np.asarray(
        l2.recipe.z_coord.nemo_een_barotropic.vmask)[..., :vslp.shape[-1]]
    np.testing.assert_array_equal(np.asarray(diagnostics["vmask"]), raw_vmask)
    assert np.count_nonzero(raw_vmask[-1]) == 1319
    assert np.max(np.abs(np.asarray(diagnostics["zgrv"][-1]))) > 0.0
    perm_v, perm_t = np.asarray(fold.perm_v), np.asarray(fold.perm_T)
    np.testing.assert_array_equal(
        np.asarray(vslp[-1]), fold.vector_sign_v * np.asarray(vslp[-2])[perm_v])
    half = l2.recipe.grid.n_lon // 2
    for slope in (wslpi, wslpj):
        a = np.asarray(slope)
        np.testing.assert_array_equal(
            a[-1, half + 1:], -a[-1, perm_t[half + 1:]])
    # U uses the shared compact-U pivot helper; applying it again is idempotent.
    import jax.numpy as jnp
    from legoesm.grids.operators_latlon_cgrid import fold_pivot_half_row
    u_redundant = jnp.concatenate([uslp[:, -1:], uslp], axis=1)
    np.testing.assert_array_equal(
        np.asarray(fold_pivot_half_row(u_redundant, fold, "U")[:, 1:]),
        np.asarray(uslp))
    # NEMO kt-1 pivot anchors, after the associated sign=-1 lbc_lnk call at
    # ldfslp.f90:338.  Structural self-fold checks alone would miss a jointly
    # wrong source row or stagger permutation.
    assert np.asarray(uslp)[-1, 29, 1] == -7.82621254877704e-06
    assert np.asarray(vslp)[-1, 29, 1] == -4.4775705127014186e-06
    assert np.asarray(wslpi)[-1, 39, 20] == 0.00020415286535305747
    assert np.asarray(wslpj)[-1, 39, 20] == -0.00024243406967063396


def test_l2_carried_slope_n2_resolves_orca2s_eos80_branch(cards, monkeypatch):
    from legoesm.ocean import eos as eos_mod
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    _, l2 = cards
    run = l2.recipe
    model = LatLonCGridOceanModel(run.grid, run.z_coord, run.model_config)
    seen = []
    real = eos_mod.compute_buoyancy_frequency_nemo_bn2

    def spy(*args, **kwargs):
        seen.append(kwargs["source_rounded"])
        return real(*args, **kwargs)

    monkeypatch.setattr(eos_mod, "compute_buoyancy_frequency_nemo_bn2", spy)
    bundle = model._tke_step_entry_n2_bundle(run.initial_state)
    assert bundle is not None
    assert bundle.rn2.shape[-1] == run.initial_state.T.data.shape[-1] - 1
    assert np.isfinite(np.asarray(bundle.rn2)).all()
    assert seen == [True]


def test_l2_observer_dump_has_committed_reproducer_and_pinned_producers():
    patch = L2 / "ldfslp_slope_dump.F90.patch"
    if not (L2_OBSERVER / "vmix_r17_slopes.bin").is_file():
        pytest.skip("round-17 write-only slope observer is not on this machine")

    def sha(path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    assert sha(patch) == "a8d8223cc3be43c58e440f30c8d4924afe4c8a362e64c93e66eeb88aaf324f8f"
    assert sha(L2_OBSERVER / "vmix_r17_slopes.bin") == \
        "e8b07d7f8dfffdbbe69020f125eb628af9529ee8079595a5a5e65572023d5724"
    assert sha(L2_OBSERVER / "nemo_observer.exe") == \
        "91064e46d04610c47e2786affea83ead177e24cc07cb88e7a3fb18cd651745c5"
    provenance = (L2_OBSERVER / "PROVENANCE.txt").read_text()
    assert "8d0e6d42e7c19dca848734179584198d2735b154ab3fbea02a7f693129dc53de" in provenance
    assert "37fc7d2e294e8c4f42e398675fb99c13a8294c39bca037199637afe6d525168d" in provenance


@pytest.fixture(scope="module")
def l2_production_ldf_trace(cards):
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    sys.path.insert(0, str(CASES / "nemo_testcase_g5_orca2ice"))
    import g5_records

    _, l2 = cards
    forcing, surface = g5_records.nemo_supplied_forcing(
        l2, L2_RECORD / "twin_a", 1)
    model = LatLonCGridOceanModel(
        l2.recipe.grid, l2.recipe.z_coord, l2.recipe.model_config,
        iwm_forcing=l2.recipe.iwm_forcing)
    # The hook is write-only and intentionally unavailable to public config.
    # Install it after construction, as the ladder's other private observers do.
    model._nemo_ws_test_hooks = _NEMOWSRK3TestHooks(
        tracer_ldf_diagnostics={})
    trace = jax.device_get(model.step(
        l2.recipe.initial_state, dt=l2.dt_s, freshwater=forcing,
        surface_forcing=surface))
    steps = g5_records.read_step(
        L2_RECORD / "twin_a", 1,
        (L2_RECORD / "twin_b", L2_RECORD / "twin_c"), 30)
    return l2, trace.ldf_diagnostics, steps


def _observer_slopes():
    with (L2_OBSERVER / "vmix_r17_slopes.bin").open("rb") as handle:
        ni, nj, nk = map(int, np.fromfile(handle, np.int32, 3))
        n3, n2 = ni * nj * nk, ni * nj

        def a3():
            return np.fromfile(handle, np.float64, n3).reshape(
                (ni, nj, nk), order="F").transpose(1, 0, 2)

        def a2(dtype=np.float64):
            return np.fromfile(handle, dtype, n2).reshape(
                (ni, nj), order="F").T

        _ = a3(), a3(), a2(), a2(), a2(np.int32), a2(), a3(), a3()
        names = ("uslp", "vslp", "wslpi", "wslpj")
        return {name: a3()[2:-2, 2:-2, :30] for name in names}


@needs_l2_record
@pytest.mark.skipif(not L2_WALK_RECORD.is_file(), reason="no admitted L2 walk")
@pytest.mark.slow
def test_l2_production_face_geometry_and_slopes_match_admitted_values(
        l2_production_ldf_trace):
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.vertical import (
        compute_ocean_jacobian,
        nemo_qco_live_face_thicknesses,
        nemo_qco_resolved_mesh_operands,
    )
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        nemo_iso_a33_e3w,
    )

    l2, diag, steps = l2_production_ldf_trace
    run, state = l2.recipe, l2.recipe.initial_state
    nlev = state.T.data.shape[-1]
    u3 = jnp.broadcast_to(
        state.u_mask.data[..., None], (*state.u_mask.data.shape, nlev))
    v3 = jnp.broadcast_to(
        state.v_mask.data[..., None], (*state.v_mask.data.shape, nlev))
    ops = nemo_qco_resolved_mesh_operands(
        run.z_coord, run.grid, u3, v3, state.T.data.dtype, nlev)
    mean_ssh_faces = nemo_qco_live_face_thicknesses(
        steps["2_ssh_aa"], run.z_coord, ops.e3u_0, ops.e3v_0,
        ops.umask3, ops.vmask3, grid=run.grid)
    got = tuple(np.asarray(diag[name]) for name in ("e3u_flux", "e3v_flux"))
    shape3 = (184, 152, 31)
    count3 = int(np.prod(shape3))
    direct_faces = []
    for index in (7, 8):
        full = np.memmap(
            L2_WALK_RECORD, dtype="=f8", mode="r",
            offset=68 + index * count3 * 8, shape=shape3, order="F")
        direct_faces.append(np.asarray(
            full[2:-2, 2:-2, :30]).transpose(1, 0, 2).copy())
    for actual, want in zip(got, mean_ssh_faces, strict=True):
        np.testing.assert_array_equal(actual, np.asarray(want))
    assert jax.config.x64_enabled
    assert tuple(hashlib.sha256(a.tobytes()).hexdigest() for a in got) == (
        "fe9697432f56aec0ae5375cef68d23a16f62ffbbbbe24b4079b3d75bd2a6c252",
        "eae07bc5b76ad220ef0026a27c2be796650fdee9daace911741d9fe233678134",
    )
    # Against NEMO's dumped e3u/e3v_live: 36 U and 20 V faces differ, each
    # within one ULP (normalized <=1e-15, the D127 bar).
    for actual, want, n_ulp in zip(got, direct_faces, (36, 20), strict=True):
        assert np.count_nonzero(actual != want) == n_ulp
        assert np.max(np.abs(actual - want) / np.max(np.abs(want))) <= 1e-15

    # Both plants have fixed, measured value effects: aliasing face reference
    # thickness changes every face cell; using step-entry SSH changes only the
    # live Kmm stretch.  These are values, not graph/signature assertions.
    e3t0 = run.z_coord.nemo_e3t_0[..., :nlev]
    alias = nemo_qco_live_face_thicknesses(
        steps["2_ssh_aa"], run.z_coord, e3t0, e3t0,
        ops.umask3, ops.vmask3, grid=run.grid)
    entry = nemo_qco_live_face_thicknesses(
        steps["e_ssh_bb"], run.z_coord, ops.e3u_0, ops.e3v_0,
        ops.umask3, ops.vmask3, grid=run.grid)
    assert tuple(np.count_nonzero(a != np.asarray(b))
                 for a, b in zip(got, alias, strict=True)) == (18803, 18300)
    assert tuple(np.count_nonzero(a != np.asarray(b))
                 for a, b in zip(got, entry, strict=True)) == (413030, 415175)

    # traldf_iso_a33 uses e3w_1d*(1+r3t(Kmm)), where Kmm is stage 2's stored
    # N+1/2 ratio.  The production diagnostic comes through the jitted model
    # step above; the admitted NEMO operands below are independent values.
    count2 = 184 * 152
    arrays2_offset = 68 + 40 * count3 * 8
    r3t_full = np.memmap(
        L2_WALK_RECORD, dtype="=f8", mode="r", offset=arrays2_offset,
        shape=shape3[:2], order="F")
    r3t = np.asarray(r3t_full[2:-2, 2:-2].T).copy()
    e3w_1d = np.memmap(
        L2_WALK_RECORD, dtype="=f8", mode="r",
        offset=arrays2_offset + 11 * count2 * 8, shape=(31,))
    nemo_e3w = np.asarray(e3w_1d[:30])[None, None, :] * (
        1.0 + r3t[..., None])
    np.testing.assert_array_equal(
        np.asarray(diag["zfw_operands"]["e3w_above"]), nemo_e3w)

    entry_jacobian = compute_ocean_jacobian(
        state.eta.data, state.H_bathy.data, run.z_coord)
    entry_e3t = run.z_coord.dz_ref[None, None, :] * entry_jacobian[..., None]
    entry_e3w = np.asarray(nemo_iso_a33_e3w(
        run.z_coord, entry_e3t, entry_jacobian, state.T.data.dtype))
    entry_delta = np.abs(entry_e3w - nemo_e3w)
    assert np.count_nonzero(entry_delta) == 492990
    assert np.max(entry_delta) == 5.102821413176116

    observer = _observer_slopes()
    for name in ("uslp", "vslp", "wslpi", "wslpj"):
        np.testing.assert_array_equal(np.asarray(diag[name]), observer[name])


@needs_l2_record
@pytest.mark.slow
def test_l2_kt1_record_requires_metric_eos80_and_north_fold(ladder):
    """One NEMO record, three planted omissions; each must cross D127."""
    root = L2_RECORD / "twin_a"
    common = dict(
        arm="independent", kt_from=1, kt_to=1,
        restart_roots=(L2_RECORD / "twin_b", L2_RECORD / "twin_c"),
        allow_dirty=True, case="ORCA2L2-iso")

    def statuses(out):
        return {row["name"].rsplit(".", 1)[-1]: row["status"]
                for row in out["per_kt"][0]["rows"]}

    def tracer_values(out):
        rows = {row["name"].rsplit(".", 1)[-1]: row
                for row in out["per_kt"][0]["rows"]}
        return tuple((rows[name]["n_unequal"], rows[name]["max_abs"],
                      rows[name]["rms"])
                     for name in ("T", "S"))

    selected = ladder.run(root, **common)
    selected_status = statuses(selected)
    assert all(selected_status[name] == "AT-BAR"
               for name in ("ssh", "uu_b", "vv_b", "u", "v", "T", "S"))
    assert tracer_values(selected) == (
        (2, 7.105427357601002e-15, 4.3618394145414156e-17),
        (2, 7.105427357601002e-15, 6.156559646205488e-17),
    )
    controls = (
        ("without_l2_metric", "l2_metric_selected", (
            (16433, 0.020541617987626548, 0.0006725116923845033),
            (16433, 0.020566462820472253, 0.00042662880934269434))),
        ("without_l2_eos80", "l2_eos80_selected", (
            (16429, 0.004663857636465485, 0.0001388908523019663),
            (16429, 0.0014607050156243417, 3.852705067169812e-05))),
        ("without_l2_north_fold", "l2_north_fold_selected", (
            (140, 0.0013241514577646551, 1.595933265099751e-05),
            (140, 0.008011780370249966, 9.00824649513366e-05))),
        ("without_l2_kmm_face_geometry", "l2_kmm_face_geometry_selected", (
            (16432, 5.456048228147381e-05, 5.273630166024779e-07),
            (16433, 2.605229774843565e-05, 3.3306669931064525e-07))),
    )
    for flag, stamp, expected in controls:
        omitted = ladder.run(root, **common, **{flag: True})
        assert omitted[stamp] is False
        rows = statuses(omitted)
        assert rows["T"] == "DEBT" or rows["S"] == "DEBT", flag
        assert tracer_values(omitted) == expected, flag


def test_shared_constructor_keeps_vortex_smt3_selection_unchanged():
    smt3 = recipe.build_vortex_smt_zps_card("vector", "smt3")
    assert smt3.recipe.model_config.gm_redi == recipe._orca2_isoneutral_gm_redi(270.0)


def test_acting_checker_requires_wet_ldf_and_tracer_only_entry(monkeypatch, tmp_path):
    shape = (2, 3, 2)
    zero = np.zeros(shape)
    base_stage = {f"{where}_{tr}": zero.copy()
                  for where in ("before_ldf", "after_ldf", "Kaa")
                  for tr in ("T", "S")}
    l2_stage = {k: v.copy() for k, v in base_stage.items()}
    l2_stage["after_ldf_T"][0, 1, 0] = 1.0
    l2_stage["Kaa_T"][0, 1, 0] = 1.0
    entry0 = {"T": zero.copy(), "S": zero.copy(), "u": zero.copy(),
              "v": zero.copy(), "ssh": np.zeros(shape[:2])}
    entry2 = {k: v.copy() for k, v in entry0.items()}
    entry2["T"][0, 1, 0] = 1.0
    monkeypatch.setattr(RECORDS, "read_tracer_stage3",
                        lambda path: l2_stage if "l2" in str(path) else base_stage)
    monkeypatch.setattr(RECORDS, "read_entry",
                        lambda path, case: entry2 if case == "ORCA2L2-iso" else entry0)

    class Var:
        def __getitem__(self, key):
            return np.array([[2, 2, 0], [2, 2, 2]])

    class Dataset:
        variables = {"bottom_level": Var()}
        def __enter__(self): return self
        def __exit__(self, *args): return False

    monkeypatch.setattr(RECORDS.netCDF4, "Dataset", lambda path: Dataset())
    result = RECORDS.check_l2(Path("l2"), Path("g5"), tmp_path / "domain")
    assert result["pass"] and result["ldf_wet_nonzero"] == 1
    l2_stage["after_ldf_T"][0, 2, 0] = 1.0                 # dry-column plant
    assert not RECORDS.check_l2(Path("l2"), Path("g5"), tmp_path / "domain")["pass"]


def test_acting_checker_requires_exact_kt1_barotropic_frame_when_requested(
        monkeypatch, tmp_path):
    shape = (1, 1, 1)
    zero = np.zeros(shape)
    stages = {f"{where}_{tr}": zero.copy()
              for where in ("before_ldf", "after_ldf", "Kaa")
              for tr in ("T", "S")}
    l2_stages = {key: value.copy() for key, value in stages.items()}
    l2_stages["after_ldf_T"][0, 0, 0] = 1.0
    l2_stages["Kaa_T"][0, 0, 0] = 1.0
    entries = {name: zero.copy() for name in ("T", "S", "u", "v", "ssh")}
    l2_entries = {key: value.copy() for key, value in entries.items()}
    l2_entries["T"][0, 0, 0] = 1.0
    monkeypatch.setattr(RECORDS, "read_tracer_stage3",
                        lambda path: l2_stages if "l2" in str(path) else stages)
    monkeypatch.setattr(RECORDS, "read_entry",
                        lambda path, case: l2_entries if case == "ORCA2L2-iso" else entries)

    barotropic = {
        "Kaa": 3,
        "uu_b": np.ones(shape),
        "vv_b": np.ones(shape),
        "un_adv": np.ones(shape),
        "vn_adv": np.ones(shape),
    }
    monkeypatch.setattr(
        RECORDS, "read_bt_frame",
        lambda path, expect_step: {key: value.copy() if hasattr(value, "copy") else value
                                   for key, value in barotropic.items()})

    class Var:
        def __init__(self, value): self.value = value
        def __getitem__(self, key): return self.value

    class Dataset:
        def __init__(self, path):
            if "domain" in str(path):
                self.variables = {"bottom_level": Var(np.ones((1, 1), dtype=int))}
        def __enter__(self): return self
        def __exit__(self, *args): return False

    monkeypatch.setattr(RECORDS.netCDF4, "Dataset", Dataset)
    result = RECORDS.check_l2(
        Path("l2"), Path("g5"), tmp_path / "domain", check_barotropic=True)
    assert result["pass"] and result["barotropic_checked"]

    def planted_frame(path, expect_step):
        out = {key: value.copy() if hasattr(value, "copy") else value
               for key, value in barotropic.items()}
        if "l2" in str(path):
            out["uu_b"][0, 0, 0] = 2.0
        return out

    monkeypatch.setattr(RECORDS, "read_bt_frame", planted_frame)
    assert not RECORDS.check_l2(
        Path("l2"), Path("g5"), tmp_path / "domain",
        check_barotropic=True)["pass"]


@pytest.mark.skipif(not REFUSED_SMOKE.is_dir(), reason="no refused L2 smoke record")
def test_corrected_acting_checker_accepts_refused_smoke_and_rejects_kt1_plant(
        monkeypatch):
    result = RECORDS.check_l2(
        REFUSED_SMOKE, RECORDS.G5_ROOT, RECORDS.DOMAIN,
        check_barotropic=True)
    assert result["pass"]
    assert result["barotropic_unequal"] == 0

    original = RECORDS.read_bt_frame

    def planted_frame(path, *, expect_step):
        out = original(path, expect_step=expect_step)
        if Path(path).parent == REFUSED_SMOKE:
            out["uu_b"] = out["uu_b"].copy()
            out["uu_b"].flat[0] += 1.0
        return out

    monkeypatch.setattr(RECORDS, "read_bt_frame", planted_frame)
    planted = RECORDS.check_l2(
        REFUSED_SMOKE, RECORDS.G5_ROOT, RECORDS.DOMAIN,
        check_barotropic=True)
    assert not planted["pass"]
    assert planted["barotropic_unequal"] == 1


def test_run_script_reuses_g5_binaries_and_is_fail_closed():
    text = RUN.read_text()
    assert subprocess.run(["bash", "-n", str(RUN)]).returncode == 0
    assert "makenemo" not in text
    for token in (
        "ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/bin/nemo.exe",
        "ORCA2G5_OMIP_L4_R31REF_FIX1/BLD/bin/nemo.exe",
        "for m in ldfslp.o ldftra.o ldfc1d_c2d.o traldf.o traldf_iso.o trazdf.o",
        "L2_DECK_SHA", "G5_STAGE_SHA", 'check_acting "$SMOKE"',
        'check_acting "$SMOKE" barotropic', "--require-barotropic",
        'check_acting "$TWIN_A"', "ln_traldf_msc += +T$", "ln_ldfeiv += +F$",
        "commit the exact acquisition tool before running it",
    ):
        assert token in text, token
