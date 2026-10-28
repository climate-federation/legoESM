"""Direct unit and master-record tests for the SI3 O8--O14 0-ULP scoring driver."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.testcases.nemo_si3_o8_o12_master_scorer import (
    DEFAULT_MASTER_ROOT,
    FieldScoreAccumulator,
    O13_CONSERVED_FIELDS,
    PINNED_PROJECTION_SHA256,
    PINNED_RECORDS_INVENTORY_SHA256,
    PINNED_RESOURCE_USAGE_SHA256,
    PINNED_RUN_MANIFEST_SHA256,
    flip_1ulp,
    score_o14,
    check_o13_conservation,
    score_o8_to_o12,
    score_o15_o16,
    verify_kind15_translation_evidence,
    verify_master_admission_manifests,
)


def test_field_score_accumulator_catches_1ulp_and_signed_zero() -> None:
    """Prove 0-ULP comparison catches both a 1-ULP normal change and -0.0 vs +0.0."""
    acc = FieldScoreAccumulator()
    base = np.array([[1.25, -0.5], [0.0, 3.75]], dtype=np.float64)
    acc.update("f_exact", base, base.copy(), kt=1, category=1)
    assert acc.summary()["status"] == "PASS_0_ULP"
    assert acc.summary()["total_unequal"] == 0

    pert_1ulp = flip_1ulp(base, (1, 1))
    acc.update("f_1ulp", pert_1ulp, base, kt=4, category=2)
    s1 = acc.stats["f_1ulp"]
    assert s1["unequal"] == 1
    assert s1["total"] == 4
    assert s1["max_ulp"] == 1
    assert s1["first_non_bit"] == {
        "kt": 4,
        "category": 2,
        "cell": [1, 1],
        "field": "f_1ulp",
    }

    neg_zero = base.copy()
    neg_zero[1, 0] = -0.0
    acc.update("f_signed_zero", neg_zero, base, kt=2, category=3)
    s0 = acc.stats["f_signed_zero"]
    assert s0["unequal"] == 1
    assert s0["first_non_bit"] == {
        "kt": 2,
        "category": 3,
        "cell": [1, 0],
        "field": "f_signed_zero",
    }
    assert acc.summary()["status"] == "NON_BIT_EXACT"


def test_master_admission_manifests_and_plants() -> None:
    """Verify the admitted master record manifests and fail-closed manifest/namelist plants."""
    if not DEFAULT_MASTER_ROOT.is_dir():
        pytest.skip(f"admitted master record not mounted at {DEFAULT_MASTER_ROOT}")
    info = verify_master_admission_manifests(DEFAULT_MASTER_ROOT)
    assert info["run_manifest_sha256"] == PINNED_RUN_MANIFEST_SHA256
    assert info["resource_usage_sha256"] == PINNED_RESOURCE_USAGE_SHA256
    assert info["records_inventory_sha256"] == PINNED_RECORDS_INVENTORY_SHA256
    assert info["projection_manifest_sha256"] == PINNED_PROJECTION_SHA256
    assert len(info["rung_record_digests"]["O8"]) == 200
    assert len(info["rung_record_digests"]["O9"]) == 2115
    assert len(info["rung_record_digests"]["O10"]) == 1600
    assert len(info["rung_record_digests"]["O11"]) == 1100
    assert len(info["rung_record_digests"]["O12"]) == 200
    assert len(info["rung_record_digests"]["O13"]) == 200
    assert len(info["rung_record_digests"]["O14"]) == 200

    with pytest.raises(ValueError, match="STATUS PLANT-FIRED.*run_manifest.json SHA-256 mismatch"):
        verify_master_admission_manifests(DEFAULT_MASTER_ROOT, plant="manifest-digest")

    with pytest.raises(
        ValueError,
        match="STATUS PLANT-FIRED.*scalar header rn_sinew=0.75 disagrees with resolved namelist",
    ):
        score_o8_to_o12(
            DEFAULT_MASTER_ROOT,
            steps=(1,),
            plant="namelist-mismatch",
            verify_manifests=False,
        )


def test_score_o8_to_o12_representative_steps_0_ulp() -> None:
    """Score rungs O8--O13 across steps (1, 2, 100), including all Picard steps."""
    if not DEFAULT_MASTER_ROOT.is_dir():
        pytest.skip(f"admitted master record not mounted at {DEFAULT_MASTER_ROOT}")
    report = score_o8_to_o12(
        DEFAULT_MASTER_ROOT,
        steps=(1, 2, 100),
        verify_manifests=True,
    )
    assert report["status"] == "PASS_ALL_O8_O13_0_ULP"
    assert report["steps_scored"] == 3
    rungs = report["rungs"]

    assert rungs["O8"]["score"]["status"] == "PASS_0_ULP"
    assert rungs["O8"]["score"]["total_unequal"] == 0
    assert rungs["O8"]["score"]["first_non_bit"] is None

    for sub in (
        "zdf_exit",
        "dh_exit",
        "temp1_exit",
        "composed_category_stages",
        "picard_final_kind17",
        "pre_category_gather_kind21",
    ):
        assert rungs["O9"][sub]["status"] == "PASS_0_ULP"
        assert rungs["O9"][sub]["total_unequal"] == 0
        assert rungs["O9"][sub]["first_non_bit"] is None
    assert rungs["O9"]["picard_final_kind17"]["iconv_matches"] == 15

    for sub in ("sal_exit", "temp2_exit", "unpadded_o10_composed_spot_check"):
        assert rungs["O10"][sub]["status"] == "PASS_0_ULP"
        assert rungs["O10"][sub]["total_unequal"] == 0
        assert rungs["O10"][sub]["first_non_bit"] is None
    assert rungs["O10"]["unpadded_o10_composed_spot_check"]["categories_checked"] == 5

    assert rungs["O11"]["score"]["status"] == "PASS_0_ULP"
    assert rungs["O11"]["score"]["total_unequal"] == 0
    assert rungs["O11"]["score"]["first_non_bit"] is None
    assert rungs["O11"]["production_column_step_arrays_o9_o11"]["status"] == "PASS_0_ULP"
    assert rungs["O11"]["production_column_step_arrays_o9_o11"]["total_unequal"] == 0
    assert rungs["O11"]["production_column_step_arrays_o9_o11"]["first_non_bit"] is None

    assert rungs["O12"]["score"]["status"] == "PASS_0_ULP"
    assert rungs["O12"]["score"]["total_unequal"] == 0
    assert rungs["O12"]["score"]["first_non_bit"] is None
    assert rungs["O12"]["production_step_ponds_si3_lev"]["status"] == "PASS_0_ULP"
    assert rungs["O12"]["production_step_ponds_si3_lev"]["total_unequal"] == 0
    assert rungs["O12"]["production_step_ponds_si3_lev"]["first_non_bit"] is None

    assert rungs["O13"]["score"]["status"] == "PASS_0_ULP"
    assert rungs["O13"]["score"]["total_unequal"] == 0
    assert rungs["O13"]["score"]["first_non_bit"] is None
    assert rungs["O13"]["conservation"]["status"] == "PASS_CONSERVATION"
    assert rungs["O13"]["conservation"]["frames_checked"] == 3
    assert set(rungs["O13"]["conservation"]["fields"]) == set(O13_CONSERVED_FIELDS)


def test_score_o14_representative_steps_0_ulp_and_plants() -> None:
    """Score every O14 EXIT field through production and prove both plants fire."""
    if not DEFAULT_MASTER_ROOT.is_dir():
        pytest.skip(f"admitted master record not mounted at {DEFAULT_MASTER_ROOT}")
    report = score_o14(DEFAULT_MASTER_ROOT, steps=(1, 2, 100), verify_manifests=True)
    assert report["status"] == "PASS_O14_0_ULP"
    assert report["steps_scored"] == 3
    assert report["score"]["status"] == "PASS_0_ULP"
    assert report["score"]["total_unequal"] == 0
    assert report["score"]["first_non_bit"] is None

    with pytest.raises(ValueError, match=r"STATUS PLANT-FIRED.*O14 non-bit-exact.*'field': 'v_i'"):
        score_o14(DEFAULT_MASTER_ROOT, steps=(1,), plant="o14-1ulp", verify_manifests=False)
    with pytest.raises(ValueError, match=r"STATUS PLANT-FIRED.*O14 non-bit-exact"):
        score_o14(
            DEFAULT_MASTER_ROOT,
            steps=(1,),
            plant="o14-model-bypass",
            verify_manifests=False,
        )


def test_score_o15_and_kind20_owned_interior_representative_steps_0_ulp() -> None:
    if not DEFAULT_MASTER_ROOT.is_dir():
        pytest.skip(f"admitted master record not mounted at {DEFAULT_MASTER_ROOT}")
    report = score_o15_o16(DEFAULT_MASTER_ROOT, steps=(1, 2, 100))
    assert report["status"] == "PASS_ALL_O15_O16_0_ULP"
    assert report["scores"]["O15"]["total_elements"] == 3 * 10_069_920
    assert report["scores"]["O15"]["total_unequal"] == 0
    assert report["scores"]["O16_aggregation"]["total_unequal"] == 0
    assert report["scores"]["O16_flux"]["total_unequal"] == 0
    # NEMO writes ji=ntsi..ntei, jj=ntsj..ntej: 180*148 cells for
    # each of utau, vtau, and the already-interior taum record.
    tau = report["scores"]["O16_kind20_owned_interior"]
    assert tau["total_elements"] == 3 * 79_920
    assert tau["total_unequal"] == 0


@pytest.mark.parametrize(
    ("plant", "field"),
    [
        ("o15-strip-transfer-barriers", "a_i"),
        ("o16-1ulp", "qsr"),
        ("o16-tau-1ulp", "utau"),
    ],
)
def test_o15_barrier_and_kind20_score_plants_fail(plant: str, field: str) -> None:
    if not DEFAULT_MASTER_ROOT.is_dir():
        pytest.skip(f"admitted master record not mounted at {DEFAULT_MASTER_ROOT}")
    with pytest.raises(ValueError, match=rf"STATUS PLANT-FIRED.*'field': '{field}'"):
        score_o15_o16(DEFAULT_MASTER_ROOT, steps=(1,), plant=plant)


def test_kind15_versioned_translation_evidence_and_wrong_permutation_plant() -> None:
    if not DEFAULT_MASTER_ROOT.is_dir():
        pytest.skip(f"admitted master record not mounted at {DEFAULT_MASTER_ROOT}")
    evidence = verify_kind15_translation_evidence(DEFAULT_MASTER_ROOT)
    assert evidence["status"] == "KIND15_VERSIONED_TRANSLATION_PROVED"
    assert evidence["writer_tail_order"][2:5] == ["qsr", "sfx", "cloud_fra"]
    assert evidence["admitted_schema_tail_order"][2:5] == ["cloud_fra", "qsr", "sfx"]
    assert evidence["cross_kind_fields_checked"] == 36
    assert evidence["qsr_matches_qsr_oce"] is True
    assert evidence["cloud_fra_matches_pp_cldf"] == 0.81
    assert evidence["translation_applied"] is True
    assert evidence["translation_version"] == 1
    assert evidence["translation_kind"] == 15
    with pytest.raises(ValueError, match="kind-15 qsr permutation evidence failed"):
        verify_kind15_translation_evidence(DEFAULT_MASTER_ROOT, plant=True)


def test_o16_model_arithmetic_plant_fails() -> None:
    if not DEFAULT_MASTER_ROOT.is_dir():
        pytest.skip(f"admitted master record not mounted at {DEFAULT_MASTER_ROOT}")
    with pytest.raises(ValueError, match=r"STATUS PLANT-FIRED.*O16_flux"):
        score_o15_o16(DEFAULT_MASTER_ROOT, steps=(1,), plant="o16-model-mutation")


@pytest.mark.parametrize(
    ("plant", "expected_pattern"),
    [
        ("o8-1ulp", r"STATUS PLANT-FIRED.*O8 non-bit-exact: unequal=1/53280.*'field': 'ht_i_new'"),
        ("o9-1ulp", r"STATUS PLANT-FIRED.*O9 non-bit-exact.*'field': 't_su'"),
        ("o10-1ulp", r"STATUS PLANT-FIRED.*O10 non-bit-exact.*'field': 'sz_i'"),
        ("o11-1ulp", r"STATUS PLANT-FIRED.*O11 non-bit-exact.*'field': 'a_i'"),
        ("o12-1ulp", r"STATUS PLANT-FIRED.*O12 non-bit-exact.*'field': 'a_ip'"),
        ("o13-1ulp", r"STATUS PLANT-FIRED.*O13 non-bit-exact.*'field': 'a_i'"),
        ("o13-model-mutation", r"STATUS PLANT-FIRED.*O13 non-bit-exact"),
        ("o13-dropped-transfer", r"STATUS PLANT-FIRED.*O13 conservation failed for v_i"),
        ("swapped-exit", r"STATUS PLANT-FIRED.*O8 non-bit-exact"),
    ],
)
def test_one_ulp_exit_field_plants_caught_per_rung(
    plant: str, expected_pattern: str
) -> None:
    """Prove that exit or model plants in every rung O8--O13 are caught."""
    if not DEFAULT_MASTER_ROOT.is_dir():
        pytest.skip(f"admitted master record not mounted at {DEFAULT_MASTER_ROOT}")
    with pytest.raises(ValueError, match=expected_pattern):
        score_o8_to_o12(
            DEFAULT_MASTER_ROOT,
            steps=(1,),
            plant=plant,
            verify_manifests=False,
        )


def test_production_call_sites_route_through_o8_o12_and_fail_when_reverted(
    monkeypatch,
) -> None:
    """Verify M4: duplicates deleted, production entry points call certified O8-O12 functions, and revert plants fail."""
    import jax.numpy as jnp
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.core.field import Field
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ice import bitz_lipscomb as bl
    from legoesm.ice import ponds as ponds_mod
    from legoesm.ice import si3_thd_da as da_mod
    from legoesm.ice import si3_thd_pnd as pnd_mod
    from legoesm.ice.c1d_omip_l3 import build_c1d_omip_l3_card
    from legoesm.ice.config import (
        MeltPondConfig,
        SI3LateralMeltConfig,
        SI3PondConfig,
        SeaIceConfig,
        SnowConfig,
    )
    from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG
    from legoesm.ice.sea_ice import step_sea_ice
    from legoesm.ice.state import SI3ColumnState, init_dynamic_ice_state

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))

    # 1. Duplicate helpers must not exist on bitz_lipscomb + AST arithmetic guard:
    for deleted_name in (
        "_si3_zdf_bl99_step",
        "_dh_step",
        "_c1d_dh_via_core",
        "_piecewise_remap",
        "_nemo_snow_enthalpy_remap",
        "_nemo_ice_enthalpy_remap",
    ):
        assert not hasattr(bl, deleted_name), f"duplicate {deleted_name} still exists on bitz_lipscomb"
    _assert_no_duplicate_o8_o12_arithmetic()
    # 2. Production step_sea_ice(thermo_scheme='si3_bl99') -> si3_column_step_arrays calls
    #    si3_ice_thd_zdf_bl99, si3_ice_thd_dh, si3_ice_thd_temp (x2), si3_ice_thd_sal(nn_icesal=2),
    #    si3_ice_thd_da (when lateral_melt=True), and si3_ice_thd_frazil.
    card = build_c1d_omip_l3_card()
    cfg_c1d = card.config._replace(
        si3=card.config.si3._replace(
            lateral_melt=True,
            namthd_da=SI3LateralMeltConfig(rn_beta=1.0, rn_dmin=8.0),
        )
    )
    n_col = 4
    sss_c1d = jnp.full(n_col, 34.0, dtype=jnp.float64)
    s_bulk_c1d = jnp.full(n_col, 5.0, dtype=jnp.float64)
    s_lay_c1d = bl.option2_salinity_profile(s_bulk_c1d, sss_c1d)
    ice_c = card.config.ice_constants
    t_ice_c1d = jnp.broadcast_to(jnp.linspace(265.0, 271.0, 3), (n_col, 3))
    e_i_c1d = bl.ice_enthalpy_from_temperature(t_ice_c1d, s_lay_c1d, ice_c)
    e_s_c1d = bl.snow_enthalpy_from_temperature(jnp.full((n_col, 3), 266.0), ice_c)

    def _fld(v, layered=False):
        dims = ("column", "layer") if layered else ("column",)
        return Field(jnp.asarray(v, jnp.float64), name="x", dims=dims, units="1")

    state_c1d = SI3ColumnState(
        _fld(jnp.full(n_col, 0.85)),
        _fld(jnp.full(n_col, 1.8)),
        _fld(jnp.full(n_col, 0.15)),
        _fld(jnp.full(n_col, 268.0)),
        _fld(e_i_c1d, True),
        _fld(e_s_c1d, True),
        _fld(s_bulk_c1d),
        _fld(s_lay_c1d, True),
        _fld(jnp.zeros(n_col)),
    )
    z1 = jnp.zeros(n_col, dtype=jnp.float64)
    forcing_c1d = bl.SI3SurfaceForcing(
        qns_ice=jnp.full(n_col, 35.0, dtype=jnp.float64),
        qsr_ice=jnp.full(n_col, 80.0, dtype=jnp.float64),
        dqns_ice=jnp.full(n_col, -4.5, dtype=jnp.float64),
        qtr_ice_top=jnp.full(n_col, 15.0, dtype=jnp.float64),
        t_bottom=jnp.full(n_col, 271.35, dtype=jnp.float64),
        sss=sss_c1d,
        evaporation=jnp.full(n_col, 1.0e-6, dtype=jnp.float64),
        snow_precipitation=jnp.full(n_col, 2.0e-5, dtype=jnp.float64),
        qprec_ice=jnp.full(n_col, -1.1e8, dtype=jnp.float64),
        qcn_ice_bottom=z1,
        qsb_ice_bottom=jnp.full(n_col, 10.0, dtype=jnp.float64),
        fhld=jnp.full(n_col, 5.0, dtype=jnp.float64),
        qlead=jnp.full(n_col, -200.0, dtype=jnp.float64),
        frq_m=jnp.ones(n_col, dtype=jnp.float64),
        sst_m=jnp.full(n_col, 1.5, dtype=jnp.float64),
    )

    called_stages: list[tuple[str, object]] = []
    real_zdf = bl.si3_ice_thd_zdf_bl99
    real_dh = bl.si3_ice_thd_dh
    real_temp = bl.si3_ice_thd_temp
    real_sal = bl.si3_ice_thd_sal
    real_da = da_mod.si3_ice_thd_da
    real_frazil = bl.si3_ice_thd_frazil
    real_do = bl.si3_ice_thd_do

    monkeypatch.setattr(
        bl,
        "si3_ice_thd_zdf_bl99",
        lambda **kw: (called_stages.append(("zdf", kw["t_i"].shape[-1])), real_zdf(**kw))[1],
    )
    monkeypatch.setattr(
        bl,
        "si3_ice_thd_dh",
        lambda **kw: (called_stages.append(("dh", kw["nn_icesal"])), real_dh(**kw))[1],
    )
    monkeypatch.setattr(
        bl,
        "si3_ice_thd_temp",
        lambda **kw: (called_stages.append(("temp", kw["e_i"].shape[-1])), real_temp(**kw))[1],
    )
    monkeypatch.setattr(
        bl,
        "si3_ice_thd_sal",
        lambda **kw: (called_stages.append(("sal", kw["namelist"]["nn_icesal"])), real_sal(**kw))[1],
    )
    monkeypatch.setattr(
        da_mod,
        "si3_ice_thd_da",
        lambda **kw: (called_stages.append(("da", kw["nn_icesal"])), real_da(**kw))[1],
    )
    monkeypatch.setattr(
        bl,
        "si3_ice_thd_frazil",
        lambda *a, **kw: (called_stages.append(("frazil", kw["ln_frazil"])), real_frazil(*a, **kw))[1],
    )
    monkeypatch.setattr(
        bl,
        "si3_ice_thd_do",
        lambda **kw: (called_stages.append(("do", kw["nn_icesal"])), real_do(**kw))[1],
    )

    next_state_c1d, _ = step_sea_ice(
        state_c1d, forcing_c1d, z1, z1, z1, cfg_c1d, 1.0, card.dt_seconds
    )
    assert called_stages == [
        ("zdf", 3),
        ("dh", 2),
        ("temp", 3),
        ("sal", 2),
        ("temp", 3),
        ("da", 2),
        ("frazil", False),
        ("do", 2),
    ]

    # Reverting the production O14 call must change the production result.
    with monkeypatch.context() as mp:
        mp.setattr(
            bl,
            "si3_ice_thd_do",
            lambda **kw: bl.SI3DOOutputs(
                a_i=kw["a_i"],
                v_i=kw["v_i"],
                sv_i=kw["sv_i"],
                e_i=kw["e_i"],
                szv_i=kw["szv_i"],
                at_i=jnp.sum(kw["a_i"], axis=-1),
                sfx_opw=kw["sfx_opw"],
                wfx_opw=kw["wfx_opw"],
                hfx_thd=kw["hfx_thd"],
                hfx_opw=kw["hfx_opw"],
            ),
        )
        rev_do_state, _ = step_sea_ice(
            state_c1d, forcing_c1d, z1, z1, z1, cfg_c1d, 1.0, card.dt_seconds
        )
        assert not np.array_equal(
            np.asarray(next_state_c1d.h_ice.data).view(np.uint64),
            np.asarray(rev_do_state.h_ice.data).view(np.uint64),
        )

    # Reverting the wiring of si3_ice_thd_sal inside si3_ice_thd_o10_category_stages must fail:
    with monkeypatch.context() as mp:
        mp.setattr(
            bl,
            "si3_ice_thd_sal",
            lambda **kw: bl.SI3SalinityOutputs(
                kw["sz_i"],
                kw["sfx_bri"],
                kw["sfx_res"],
                jnp.zeros_like(kw["h_i"], dtype=jnp.int32),
                jnp.zeros_like(kw["h_i"], dtype=jnp.int32),
                jnp.zeros_like(kw["h_i"]),
                jnp.zeros_like(kw["h_i"]),
                kw["s_i"],
                kw["s_i"],
            ),
        )
        rev_state_c1d, _ = step_sea_ice(
            state_c1d, forcing_c1d, z1, z1, z1, cfg_c1d, 1.0, card.dt_seconds
        )
        assert not np.array_equal(
            np.asarray(next_state_c1d.S_layers.data).view(np.uint64),
            np.asarray(rev_state_c1d.S_layers.data).view(np.uint64),
        )

    # 3. Production step_sea_ice with ponds.scheme='si3_lev' -> step_ponds(scheme='si3_lev')
    #    calls si3_ice_thd_pnd -> si3_pnd_lev -> si3_ice_perm_eff, and fails when reverted.
    namthd_pnd = SI3PondConfig(
        rn_himin=0.1,
        rn_apnd_min=0.15,
        rn_apnd_max=0.85,
        rn_pnd_flush=0.1,
        ln_pnd_lids=True,
    )
    shape_2d = (2, 2)
    t_init = SeaIceConfig().T_freeze_ocean
    base_pnd = init_dynamic_ice_state(
        shape_2d, S_ice_init=2.5, T_ice_init=t_init, pond_scheme="si3_lev"
    )
    state_pnd = base_pnd._replace(
        h_ice=base_pnd.h_ice.replace(data=jnp.full(shape_2d, 1.4, dtype=jnp.float64)),
        T_ice=base_pnd.T_ice.replace(data=jnp.full(shape_2d, t_init, dtype=jnp.float64)),
        concentration=base_pnd.concentration.replace(data=jnp.full(shape_2d, 0.80, dtype=jnp.float64)),
        h_snow=base_pnd.h_snow.replace(data=jnp.full(shape_2d, 0.05, dtype=jnp.float64)),
        S_ice=base_pnd.S_ice.replace(data=jnp.full(shape_2d, 2.5, dtype=jnp.float64)),
        pond_area=base_pnd.pond_area.replace(data=jnp.full(shape_2d, 0.15, dtype=jnp.float64)),
        pond_depth=base_pnd.pond_depth.replace(data=jnp.full(shape_2d, 0.12, dtype=jnp.float64)),
    )
    z22 = jnp.zeros(shape_2d, dtype=jnp.float64)
    forcing_pnd = AtmToSurface(
        sw_down=z22,
        lw_down=z22,
        precip_total=z22,
        precip_snow=z22,
        T_lowest=jnp.full(shape_2d, t_init, dtype=jnp.float64),
        q_lowest=z22,
        u_lowest=z22,
        v_lowest=z22,
        p_lowest=jnp.full(shape_2d, 100000.0, dtype=jnp.float64),
        p_surface=jnp.full(shape_2d, 101325.0, dtype=jnp.float64),
        rho_lowest=jnp.full(shape_2d, 1.25, dtype=jnp.float64),
        cos_zenith=z22,
        co2_ppmv=jnp.full(shape_2d, 400.0, dtype=jnp.float64),
        has_radiation=z22,
        has_precipitation=z22,
        z_lowest=jnp.full(shape_2d, 10.0, dtype=jnp.float64),
    )
    ocean_sst = jnp.full(shape_2d, t_init, dtype=jnp.float64)
    cfg_pnd = SeaIceConfig(
        emissivity_ice=0.0,
        Cd_ice=0.0,
        Ch_ice=0.0,
        Ce_ice=0.0,
        ocean_heat_transfer_coeff=0.0,
        snow=SnowConfig(enabled=True, flooding=False),
        ponds=MeltPondConfig(enabled=True, scheme="si3_lev", namthd_pnd=namthd_pnd),
        ice_constants=NEMO_SI3_CONSTANTS_CONFIG,
    )

    pnd_calls: list[str] = []
    real_thd_pnd = pnd_mod.si3_ice_thd_pnd
    real_pnd_lev = pnd_mod.si3_pnd_lev
    real_perm_eff = pnd_mod.si3_ice_perm_eff

    monkeypatch.setattr(
        ponds_mod,
        "si3_ice_thd_pnd",
        lambda **kw: (pnd_calls.append("si3_ice_thd_pnd"), real_thd_pnd(**kw))[1],
    )
    monkeypatch.setattr(
        pnd_mod,
        "si3_pnd_lev",
        lambda **kw: (pnd_calls.append("si3_pnd_lev"), real_pnd_lev(**kw))[1],
    )
    monkeypatch.setattr(
        pnd_mod,
        "si3_ice_perm_eff",
        lambda *a, **kw: (pnd_calls.append("si3_ice_perm_eff"), real_perm_eff(*a, **kw))[1],
    )

    next_state_pnd, _ = step_sea_ice(
        state_pnd, forcing_pnd, ocean_sst, z22, z22, cfg_pnd, U_min=1.0, dt=3600.0
    )
    assert pnd_calls == ["si3_ice_thd_pnd", "si3_pnd_lev", "si3_ice_perm_eff"]

    # Reverting the wiring of si3_ice_thd_pnd in step_ponds must fail:
    with monkeypatch.context() as mp:
        mp.setattr(
            ponds_mod,
            "si3_ice_thd_pnd",
            lambda **kw: pnd_mod.SI3PondOutputs(
                at_i=jnp.sum(kw["a_i"], axis=-1),
                a_ip=kw["a_ip"],
                v_ip=kw["v_ip"],
                h_ip=kw["h_ip"],
                v_il=kw["v_il"],
                h_il=kw["h_il"],
                a_ip_frac=kw["a_ip_frac"],
                wfx_pnd=kw["wfx_pnd"],
            ),
        )
        rev_state_pnd, _ = step_sea_ice(
            state_pnd, forcing_pnd, ocean_sst, z22, z22, cfg_pnd, U_min=1.0, dt=3600.0
        )
        assert not np.array_equal(
            np.asarray(next_state_pnd.pond_depth.data).view(np.uint64),
            np.asarray(rev_state_pnd.pond_depth.data).view(np.uint64),
        )


_KNOWN_ARITH_CALLS = frozenset({
    "nemo_source_round",
    "sr",
    "_minsd",
    "_maxsd",
    "_div_const",
    "_sum0",
    "ice_enthalpy_from_temperature",
    "snow_enthalpy_from_temperature",
    "ice_temperature_from_enthalpy",
    "snow_temperature_from_enthalpy",
    "option2_salinity_profile",
    "p07_conductivity",
})

_CANONICAL_O8_O12_KERNELS: dict[str, set[str]] = {
    "packages/ice/legoesm/ice/bitz_lipscomb.py": {
        "_si3_zdf_bl99_core",
        "si3_ice_thd_zdf_bl99",
        "_cumulative_remap",
        "_si3_ice_thd_dh_core",
        "si3_ice_thd_dh",
        "si3_ice_thd_temp",
        "_sal_gd",
        "_sal_fl",
        "si3_ice_thd_sal",
        "si3_ice_thd_o10_category_stages",
        "si3_column_step_arrays",
        "_si3_column_thd_da",
        "si3_ice_thd_do",
        "si3_ice_thd_frazil",
    },
    "packages/ice/legoesm/ice/si3_thd_da.py": {"si3_ice_thd_da"},
    "packages/ice/legoesm/ice/si3_thd_pnd.py": {
        "si3_ice_thd_pnd",
        "si3_pnd_lev",
        "si3_ice_perm_eff",
    },
    "packages/ice/legoesm/ice/itd.py": {"si3_ice_itd_rem"},
}


def _assert_no_duplicate_o8_o12_arithmetic(
    source_overrides: dict[str, str] | None = None,
) -> None:
    """AST-level guard forbidding duplicated O8--O12 arithmetic in ice/ and testcases/."""
    import ast
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    overrides = source_overrides or {}

    def _read_src(rel_path: str) -> str:
        if rel_path in overrides:
            return overrides[rel_path]
        return (repo_root / rel_path).read_text(encoding="utf-8")

    def _count_arith(stmts: list[ast.stmt]) -> int:
        return sum(
            isinstance(n, (ast.BinOp, ast.UnaryOp, ast.Compare))
            for s in stmts
            for n in ast.walk(s)
        )

    def _norm_stmts(stmts: list[ast.stmt]):
        var_map: dict[str, int] = {}

        def _norm(node):
            if isinstance(node, list):
                return tuple(_norm(x) for x in node)
            if not isinstance(node, ast.AST):
                return node
            if isinstance(node, ast.Name):
                if node.id in _KNOWN_ARITH_CALLS:
                    return ("CallName", node.id)
                return ("Var", var_map.setdefault(node.id, len(var_map)))
            if isinstance(node, ast.Attribute):
                return ("Attr", node.attr)
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float, bool)):
                    return ("Const", type(node.value).__name__, node.value)
                return ("Const", type(node.value).__name__)
            fields = []
            for fname, fval in ast.iter_fields(node):
                if fname in (
                    "ctx",
                    "lineno",
                    "col_offset",
                    "end_lineno",
                    "end_col_offset",
                    "type_comment",
                ):
                    continue
                fields.append((fname, _norm(fval)))
            return (type(node).__name__, tuple(fields))

        return _norm(stmts)

    def _iter_blocks(func_node: ast.FunctionDef):
        for node in ast.walk(func_node):
            for attr in ("body", "orelse"):
                blk = getattr(node, attr, None)
                if isinstance(blk, list) and len(blk) >= 4:
                    yield blk

    canon_windows: dict[object, tuple[str, str, int]] = {}
    for rel, fnames in _CANONICAL_O8_O12_KERNELS.items():
        tree = ast.parse(_read_src(rel), filename=rel)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in fnames:
                for blk in _iter_blocks(node):
                    for i in range(len(blk) - 3):
                        win = blk[i : i + 4]
                        if _count_arith(win) >= 5:
                            sig = _norm_stmts(win)
                            canon_windows.setdefault(
                                sig, (rel, node.name, win[0].lineno)
                            )

    scan_files = [
        *sorted(
            str(p.relative_to(repo_root))
            for p in (repo_root / "packages/ice/legoesm/ice").glob("*.py")
        ),
        "scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py",
        "scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py",
        "scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py",
    ]
    for rel in scan_files:
        tree = ast.parse(_read_src(rel), filename=rel)
        canon_set = _CANONICAL_O8_O12_KERNELS.get(rel, set())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name not in canon_set:
                for blk in _iter_blocks(node):
                    for i in range(len(blk) - 3):
                        win = blk[i : i + 4]
                        if _count_arith(win) >= 5:
                            sig = _norm_stmts(win)
                            if sig in canon_windows:
                                c_rel, c_fn, c_ln = canon_windows[sig]
                                raise AssertionError(
                                    f"duplicate O8-O12 arithmetic window in {rel}:{node.name}:{win[0].lineno} "
                                    f"matches canonical kernel {c_rel}:{c_fn}:{c_ln}"
                                )

    # AST structure and delegation checks on nemo_si3thd_phase2_gate.py:
    phase2_rel = "scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py"
    phase2_tree = ast.parse(_read_src(phase2_rel), filename=phase2_rel)
    phase2_funcs = {
        n.name: n for n in phase2_tree.body if isinstance(n, ast.FunctionDef)
    }
    assert "_c1d_dh_via_core" not in phase2_funcs, (
        "duplicate _c1d_dh_via_core still exists in nemo_si3thd_phase2_gate.py"
    )

    def _called_names(fn_node: ast.AST) -> set[str]:
        out: set[str] = set()
        for n in ast.walk(fn_node):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name):
                    out.add(n.func.id)
                elif isinstance(n.func, ast.Attribute):
                    out.add(n.func.attr)
        return out

    for remap_name in ("_nemo_snow_enthalpy_remap", "_nemo_ice_enthalpy_remap"):
        r_fn = phase2_funcs[remap_name]
        loops = [n for n in ast.walk(r_fn) if isinstance(n, (ast.For, ast.While, ast.If))]
        binops = [n for n in ast.walk(r_fn) if isinstance(n, ast.BinOp)]
        assert not loops, f"{remap_name} contains duplicate control flow: {loops}"
        assert len(binops) <= 2, f"{remap_name} contains duplicate arithmetic ({len(binops)} BinOps)"
        assert "_cumulative_remap" in _called_names(r_fn), (
            f"{remap_name} does not delegate to _cumulative_remap"
        )

    zdf_fn = phase2_funcs["_si3_zdf_bl99_step"]
    assert "si3_ice_thd_zdf_bl99" in _called_names(zdf_fn), (
        "_si3_zdf_bl99_step does not delegate to si3_ice_thd_zdf_bl99"
    )

    dh_fn = phase2_funcs["_dh_step"]
    assert "_call_prod_dh" in _called_names(dh_fn), (
        "_dh_step does not delegate to _call_prod_dh -> si3_ice_thd_dh"
    )
    assert "si3_ice_thd_dh" in _called_names(phase2_funcs["_call_prod_dh"]), (
        "_call_prod_dh does not delegate to si3_ice_thd_dh"
    )
    for n in ast.walk(dh_fn):
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name):
            assert n.test.id not in (
                "_nemo_dh_ice_content_accumulation",
                "_nemo_dh_snow_update",
            ), f"duplicate {n.test.id} branch still exists in _dh_step"

    abl_fn = phase2_funcs["_si3_column_step_arrays_ablation"]
    abl_calls = _called_names(abl_fn)
    assert "si3_column_step_arrays" in abl_calls, (
        "_si3_column_step_arrays_ablation does not delegate to si3_column_step_arrays"
    )
    assert "si3_ice_thd_sal" in abl_calls, (
        "_si3_column_step_arrays_ablation does not delegate to si3_ice_thd_sal"
    )
    abl_binops = [n for n in ast.walk(abl_fn) if isinstance(n, ast.BinOp)]
    assert len(abl_binops) <= 3, (
        f"_si3_column_step_arrays_ablation contains duplicate O8/O10 arithmetic ({len(abl_binops)} BinOps)"
    )


def test_si3_column_step_arrays_requires_explicit_si3_and_c1d_reads_namelist(
    tmp_path,
) -> None:
    """Verify missing SI3 config raises ValueError (no silent nn_icesal default) and C1D reads nn_icesal from its namelist."""
    from pathlib import Path
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ice import bitz_lipscomb as bl
    from legoesm.ice.c1d_omip_l3 import (
        C1D_OMIP_L3_NAMELIST_ICE_CFG,
        build_c1d_omip_l3_card,
    )
    from legoesm.ice.config import SI3ThermoConfig
    from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))

    card = build_c1d_omip_l3_card()
    assert card.config.si3.salinity_scheme == 2

    z1 = jnp.zeros(1, dtype=jnp.float64)
    z3 = jnp.zeros((1, 3), dtype=jnp.float64)
    st = bl.SI3ColumnArrays(
        jnp.asarray([0.85], dtype=jnp.float64),
        jnp.asarray([1.8], dtype=jnp.float64),
        jnp.asarray([0.15], dtype=jnp.float64),
        jnp.asarray([268.0], dtype=jnp.float64),
        z3,
        z3,
        jnp.asarray([5.0], dtype=jnp.float64),
        z3,
        z1,
    )
    forcing = bl.SI3SurfaceForcing(
        qns_ice=z1,
        qsr_ice=z1,
        dqns_ice=z1,
        qtr_ice_top=z1,
        t_bottom=jnp.asarray([271.35], dtype=jnp.float64),
        sss=jnp.asarray([34.0], dtype=jnp.float64),
        evaporation=z1,
        snow_precipitation=z1,
        qprec_ice=z1,
        qcn_ice_bottom=z1,
        qsb_ice_bottom=z1,
        fhld=z1,
        qlead=z1,
    )

    with pytest.raises(
        ValueError,
        match=r"si3_column_step_arrays requires an explicit SI3ThermoConfig \(si3\); no default",
    ):
        bl.si3_column_step_arrays(st, forcing, 3600.0, NEMO_SI3_CONSTANTS_CONFIG)

    with pytest.raises(
        ValueError,
        match=r"si3_column_step_arrays requires an explicit SI3ThermoConfig \(si3\); no default",
    ):
        bl.si3_column_step_arrays(
            st,
            forcing,
            3600.0,
            NEMO_SI3_CONSTANTS_CONFIG,
            si3=SI3ThermoConfig(salinity_scheme=None),  # type: ignore[arg-type]
        )

    with pytest.raises(
        ValueError,
        match=r"_si3_column_thd_da requires an explicit SI3ThermoConfig \(si3\); no default",
    ):
        bl._si3_column_thd_da(st, st, forcing, None, 3600.0, NEMO_SI3_CONSTANTS_CONFIG)

    # Verify C1D actually parses nn_icesal from its namelist file:
    orig_nml = Path(C1D_OMIP_L3_NAMELIST_ICE_CFG).read_text(encoding="utf-8")
    nml_scheme4 = tmp_path / "namelist_ice_cfg_4"
    nml_scheme4.write_text(
        orig_nml.replace("nn_icesal        = 2", "nn_icesal        = 4"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="salinity_scheme=4"):
        build_c1d_omip_l3_card(namelist_ice_path=nml_scheme4)

    nml_missing = tmp_path / "namelist_ice_cfg_missing"
    nml_missing.write_text(
        orig_nml.replace("nn_icesal        = 2", "! nn_icesal omitted"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="nn_icesal"):
        build_c1d_omip_l3_card(namelist_ice_path=nml_missing)


def test_no_duplicate_o8_o12_arithmetic_ast_guard_and_plants(monkeypatch) -> None:
    """Verify the AST no-duplicate-arithmetic guard passes on repo and fails on planted duplicates."""
    from pathlib import Path
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ice import bitz_lipscomb as bl
    from scripts.validate.ocean_fidelity.testcases import nemo_si3thd_phase2_gate as phase2

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    _assert_no_duplicate_o8_o12_arithmetic()

    repo_root = Path(__file__).resolve().parents[3]
    phase2_rel = "scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py"
    phase2_src = (repo_root / phase2_rel).read_text(encoding="utf-8")

    # Plant 1: a renamed 4-statement arithmetic window from production O14
    # injected into nemo_si3thd_phase2_gate.py is caught by the AST detector.
    planted_window = (
        "\ndef _planted_dup_do(sr, jnp, constants, dtype, zs_newice, t_bo, eps10):\n"
        "    ztmelts = sr(-jnp.asarray(constants.liquidus_slope, dtype=dtype) * zs_newice)\n"
        "    t_bo_c = sr(t_bo - jnp.asarray(constants.T0, dtype=dtype))\n"
        "    capped_t_bo = _minsd(-eps10, t_bo_c)\n"
        "    latent_ratio = sr(ztmelts / capped_t_bo)\n"
        "    return latent_ratio\n"
    )
    with pytest.raises(
        AssertionError,
        match=r"duplicate O8-O12 arithmetic window in .*nemo_si3thd_phase2_gate\.py:_planted_dup_do",
    ):
        _assert_no_duplicate_o8_o12_arithmetic({phase2_rel: phase2_src + planted_window})

    # Plant 2: re-introducing a loop into _nemo_snow_enthalpy_remap is caught.
    planted_remap = phase2_src.replace(
        "    return bl._cumulative_remap(\n",
        "    for _k in range(3):\n        pass\n    return bl._cumulative_remap(\n",
        1,
    )
    with pytest.raises(
        AssertionError,
        match=r"_nemo_snow_enthalpy_remap contains duplicate control flow",
    ):
        _assert_no_duplicate_o8_o12_arithmetic({phase2_rel: planted_remap})

    # Plant 3: re-introducing an `if _nemo_dh_ice_content_accumulation:` branch into _dh_step is caught.
    planted_dh = phase2_src.replace(
        "    h_i0, h_s0 = state.h_ice, state.h_snow\n",
        "    if _nemo_dh_ice_content_accumulation:\n        pass\n    h_i0, h_s0 = state.h_ice, state.h_snow\n",
        1,
    )
    with pytest.raises(
        AssertionError,
        match=r"duplicate _nemo_dh_ice_content_accumulation branch still exists in _dh_step",
    ):
        _assert_no_duplicate_o8_o12_arithmetic({phase2_rel: planted_dh})

    # Plant 4: an O13 boundary/melt arithmetic window copied outside the one
    # production remapper is caught as well.
    itd_rel = "packages/ice/legoesm/ice/itd.py"
    itd_src = (repo_root / itd_rel).read_text(encoding="utf-8")
    planted_o13 = itd_src + (
        "\ndef _planted_o13_duplicate(h_i, h_i_b, a_i, remap, eps10, hi_max, h_r_first, h_l_first, sr):\n"
        "    dh0 = sr(h_i[..., 0] - h_i_b[..., 0])\n"
        "    melt = remap & (a_i[..., 0] > eps10) & (dh0 < 0.0)\n"
        "    dh0_melt = _si3_o13_fused_melt_min(neg_dh0=sr(-dh0), hi_max=hi_max[1], h_r=h_r_first)\n"
        "    eta_max = sr(dh0_melt - h_l_first)\n"
        "    return melt, eta_max\n"
    )
    with pytest.raises(
        AssertionError,
        match=r"duplicate O8-O12 arithmetic window in .*itd\.py:_planted_o13_duplicate",
    ):
        _assert_no_duplicate_o8_o12_arithmetic({itd_rel: planted_o13})

    # Runtime verification that phase2_gate helpers delegate to production kernels in bitz_lipscomb:
    calls: list[str] = []
    orig_remap = bl._cumulative_remap
    monkeypatch.setattr(
        bl,
        "_cumulative_remap",
        lambda *a, **kw: (calls.append("_cumulative_remap"), orig_remap(*a, **kw))[1],
    )
    h4 = jnp.asarray([[0.01, 0.05, 0.05, 0.05]], dtype=jnp.float64)
    e4 = jnp.full((1, 4), 1.0e8, dtype=jnp.float64)
    phase2._nemo_snow_enthalpy_remap(h4, e4)
    phase2._nemo_ice_enthalpy_remap(h4, e4)
    assert calls == ["_cumulative_remap", "_cumulative_remap"]
@pytest.mark.parametrize(
    ("selector", "bad_value", "expected_pattern"),
    [
        ("ln_icedO", False, r"STATUS PLANT-FIRED.*record identity ln_icedO=0.0.*preregistered value 1.0"),
        ("ln_leadhfx", False, r"STATUS PLANT-FIRED.*record identity ln_leadhfx=0.0.*preregistered value 1.0"),
        ("rn_hinew", 0.15, r"STATUS PLANT-FIRED.*record identity rn_hinew=0.1.*preregistered value 0.15"),
    ],
)
def test_score_o14_refuses_preregistered_selector_mismatches(
    selector: str, bad_value: object, expected_pattern: str
) -> None:
    from scripts.validate.ocean_fidelity.testcases.nemo_si3_o8_o12_master_scorer import (
        DEFAULT_PPSRC,
        _verify_record_scalars_against_namelist,
        parse_resolved_ice_namelist,
        read_record_verified,
        record_name,
    )

    if not DEFAULT_MASTER_ROOT.is_dir():
        pytest.skip(f"admitted master record not mounted at {DEFAULT_MASTER_ROOT}")
    run_dir = DEFAULT_MASTER_ROOT / "record_100step"
    namelist = parse_resolved_ice_namelist(
        run_dir / "output.namelist.ice", run_dir / "output.namelist.dyn", ppsrc=DEFAULT_PPSRC
    )
    scalars = read_record_verified(
        run_dir / "o8_o16_master_records" / record_name(12, 1, 0)
    )["scalars"]
    namelist[selector] = bad_value
    with pytest.raises(ValueError, match=expected_pattern):
        _verify_record_scalars_against_namelist(scalars, namelist, 12)


def test_o14_receipt_rehashes_every_cited_file_digest_and_mismatch_plant() -> None:
    import hashlib
    import re
    from pathlib import Path

    receipt = Path("docs/ice/fidelity/si3_r23_o14_score_receipt.md")
    text = receipt.read_text(encoding="utf-8")
    match = re.search(
        r"<!-- BEGIN REHASHABLE_SHA256 -->(.*?)<!-- END REHASHABLE_SHA256 -->",
        text,
        flags=re.S,
    )
    assert match is not None
    entries = re.findall(r"^- `(/[^`]+)` SHA-256 `([0-9a-f]{64})`$", match.group(1), re.M)
    assert len(entries) == 5
    missing = [raw_path for raw_path, _ in entries if not Path(raw_path).is_file()]
    if missing:
        pytest.skip(f"receipt evidence files are not mounted: {missing}")

    def verify(items):
        for raw_path, expected in items:
            path = Path(raw_path)
            assert path.is_file(), path
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected

    verify(entries)
    planted = list(entries)
    planted[0] = (planted[0][0], "0" * 64)
    with pytest.raises(AssertionError):
        verify(planted)


def test_o13_p4_conservation_and_each_dropped_transfer_plant() -> None:
    """P4: a_i/sv_i included; every conserved inventory catches a dropped receipt."""
    import copy
    import jax

    from tests.ice.unit.test_si3_o13_itd_remap import _call_kwargs, _inputs, _model

    before = _inputs()
    after = {
        key: np.asarray(value)
        for key, value in jax.jit(_model)(_call_kwargs(before)).items()
    }
    report = check_o13_conservation(before, after, frame="synthetic")
    assert report["status"] == "PASS_CONSERVATION"
    assert set(report["fields"]) == set(O13_CONSERVED_FIELDS)

    for field in O13_CONSERVED_FIELDS:
        planted = copy.deepcopy(after)
        arr = planted[field].copy()
        index = np.unravel_index(np.argmax(np.abs(arr)), arr.shape)
        assert arr[index] != 0.0, field
        arr[index] = 0.0
        planted[field] = np.asarray(arr)
        with pytest.raises(ValueError, match=rf"STATUS PLANT-FIRED.*{field}"):
            check_o13_conservation(before, planted, frame=f"drop-{field}")


def test_o13_receipt_digests_and_scope_ratchet() -> None:
    """Checklist 17: re-hash every O13 receipt file and pin gate scope."""
    import hashlib
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    master = DEFAULT_MASTER_ROOT
    score_path = master.parents[1] / "round22/o13_master_score.json"
    paths = {
        master / "manifest/run_manifest.json": "bfb0b71d1500b7bc91c68a9e39c8af7a654b139258c0635ad5b1f25284b31fd0",
        master / "manifest/resource_usage.json": "b30a5febe9248e5008999cb419953dad40c78658542a3c7221d301825270967c",
        master / "manifest/records.sha256": "fbd3cb4d3450f428d8444dcb145bec3c1a832554e3e7e05d7f5b6a159daadd85",
        master / "manifest/per_rung/O13.json": "a94d126ddaee3194a053ef437d762707dff93045287392d4e0586aaee995a4aa",
        master / "manifest/per_rung/O14.json": "8130e5ac516cf8c1549badbab2caf90cf5cce03f3167c317a10532b836572cd6",
        score_path: "4e95174e554347a3b5dad598af506b42bfe2295f03b1bf0f5425940697cf6cbe",
        Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_SAS_ICE_O8O16_REF/BLD/obj/iceitd.o"): "2642ba5ec85fc283fc288d25ade003a837a3fe6f8cccc0f32b7f3bb3d89caa71",
        Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/iceitd.f90"): "22baac67260037d4cde180d48ab1938e730b57f492c7d01eb6ade93c9403e7ba",
    }
    missing = [path for path in paths if not path.is_file()]
    if missing:
        pytest.skip(f"O13 receipt artifacts not mounted: {missing}")

    receipt = (root / "docs/ice/fidelity/si3_r22_o13_receipt.md").read_text(encoding="utf-8")
    for path, expected in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        assert expected in receipt, path

    score = json.loads(score_path.read_text(encoding="utf-8"))
    assert score["status"] == "PASS_O13_0_ULP"
    assert score["steps_scored"] == 100
    assert score["rungs"]["O13"]["score"]["total_unequal"] == 0
    assert score["rungs"]["O13"]["conservation"]["frames_checked"] == 100
    assert tuple(score["rungs"]["O13"]["conservation"]["fields"]) == tuple(sorted(O13_CONSERVED_FIELDS))
    assert _CANONICAL_O8_O12_KERNELS["packages/ice/legoesm/ice/itd.py"] == {"si3_ice_itd_rem"}
