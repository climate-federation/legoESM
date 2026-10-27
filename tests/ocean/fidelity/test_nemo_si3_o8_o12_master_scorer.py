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

    assert rungs["O12"]["score"]["status"] == "PASS_0_ULP"
    assert rungs["O12"]["score"]["total_unequal"] == 0
    assert rungs["O12"]["score"]["first_non_bit"] is None


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
