"""Direct unit and master-record tests for the SI3 O8--O12 0-ULP scoring driver."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.testcases.nemo_si3_o8_o12_master_scorer import (
    DEFAULT_MASTER_ROOT,
    FieldScoreAccumulator,
    PINNED_PROJECTION_SHA256,
    PINNED_RECORDS_INVENTORY_SHA256,
    PINNED_RESOURCE_USAGE_SHA256,
    PINNED_RUN_MANIFEST_SHA256,
    flip_1ulp,
    score_o8_to_o12,
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
    """Score rungs O8, O9, O10, O11, O12 across steps (1, 2, 100) including all Picard steps."""
    if not DEFAULT_MASTER_ROOT.is_dir():
        pytest.skip(f"admitted master record not mounted at {DEFAULT_MASTER_ROOT}")
    report = score_o8_to_o12(
        DEFAULT_MASTER_ROOT,
        steps=(1, 2, 100),
        verify_manifests=True,
    )
    assert report["status"] == "PASS_ALL_O8_O12_0_ULP"
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


@pytest.mark.parametrize(
    ("plant", "expected_pattern"),
    [
        ("o8-1ulp", r"STATUS PLANT-FIRED.*O8 non-bit-exact: unequal=1/53280.*'field': 'ht_i_new'"),
        ("o9-1ulp", r"STATUS PLANT-FIRED.*O9 non-bit-exact.*'field': 't_su'"),
        ("o10-1ulp", r"STATUS PLANT-FIRED.*O10 non-bit-exact.*'field': 'sz_i'"),
        ("o11-1ulp", r"STATUS PLANT-FIRED.*O11 non-bit-exact.*'field': 'a_i'"),
        ("o12-1ulp", r"STATUS PLANT-FIRED.*O12 non-bit-exact.*'field': 'a_ip'"),
        ("swapped-exit", r"STATUS PLANT-FIRED.*O8 non-bit-exact"),
    ],
)
def test_one_ulp_exit_field_plants_caught_per_rung(
    plant: str, expected_pattern: str
) -> None:
    """Prove that a 1-ULP perturbation in one exit field of each rung O8..O12 is caught."""
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

    # 1. Duplicate helpers must not exist on legoesm.ice.bitz_lipscomb
    for deleted_name in (
        "_si3_zdf_bl99_step",
        "_dh_step",
        "_c1d_dh_via_core",
        "_piecewise_remap",
        "_nemo_snow_enthalpy_remap",
        "_nemo_ice_enthalpy_remap",
    ):
        assert not hasattr(bl, deleted_name), f"duplicate {deleted_name} still exists on bitz_lipscomb"

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
    ]

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
