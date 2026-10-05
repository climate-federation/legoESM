"""ORCA2 round 9 — the READ lateral viscosity coefficient and its gate.

The card's builder transcribes NEMO ``ldf_dyn_init`` under ``nn_ahm_ijk_t =
-30`` (``ldfdyn.f90:348-353``): the coefficient is read whole from
``eddy_viscosity_3D.nc``, completed by the lateral boundary exchange for its own
grid-point nature (``iom.f90:958-975``) and masked over levels one to ``jpkm1``
(``ldfdyn.f90:388-396``).  These tests run on a tiny synthetic file, so they do
not need the deck or the record.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round9_ldf_coefficient_gate as gate,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_orca2_ldf_dyn_coefficients,
)

NLAT, NLON, NLEV = 6, 8, 3


def _write_viscosity(path, ahmt, ahmf):
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("t", 1)
        ds.createDimension("z", ahmt.shape[-1])
        ds.createDimension("y", ahmt.shape[0])
        ds.createDimension("x", ahmt.shape[1])
        for name, values in (("ahmt_3d", ahmt), ("ahmf_3d", ahmf)):
            var = ds.createVariable(name, "f4", ("t", "z", "y", "x"))
            var[0] = np.moveaxis(values, -1, 0)


def _fields(tmp_path):
    rng = np.random.default_rng(9)
    ahmt = rng.integers(1, 40, size=(NLAT, NLON, NLEV)).astype(np.float64)
    ahmf = rng.integers(1, 40, size=(NLAT, NLON, NLEV)).astype(np.float64)
    path = tmp_path / "eddy_viscosity_3D.nc"
    _write_viscosity(path, ahmt, ahmf)
    return path, ahmt, ahmf


def test_builder_masks_and_maps_the_f_points_onto_the_vertex_layout(tmp_path):
    path, raw_t, raw_f = _fields(tmp_path)
    rng = np.random.default_rng(3)
    tmask = (rng.random((NLAT, NLON, NLEV)) > 0.3).astype(np.float64)
    fmask = rng.choice([0.0, 0.5, 1.0, 2.0], size=(NLAT, NLON, NLEV))

    ahmt, ahmf = build_orca2_ldf_dyn_coefficients(path, tmask, fmask)

    # T points: the file value, masked, on the card's own axis order.
    np.testing.assert_array_equal(ahmt, raw_t * tmask)
    assert ahmf.shape == (NLAT + 1, NLON + 1, NLEV)
    # vertex[j, i] is NEMO's F point (j-1, (i-1) mod n_lon)
    for j in range(1, NLAT + 1):
        for i in range(NLON + 1):
            np.testing.assert_array_equal(
                ahmf[j, i], (raw_f * fmask)[j - 1, (i - 1) % NLON])
    # the wrap column duplicates column zero, and the south row has no NEMO
    # source at all
    np.testing.assert_array_equal(ahmf[:, 0], ahmf[:, NLON])
    assert np.all(ahmf[0] == 0.0)


def test_builder_keeps_only_the_cards_levels(tmp_path):
    """NEMO's last level is never masked and legoESM has no such level."""
    rng = np.random.default_rng(11)
    full = rng.random((NLAT, NLON, NLEV + 1))
    path = tmp_path / "eddy_viscosity_3D.nc"
    _write_viscosity(path, full, full)
    tmask = np.ones((NLAT, NLON, NLEV))
    ahmt, ahmf = build_orca2_ldf_dyn_coefficients(path, tmask, tmask)
    assert ahmt.shape[-1] == NLEV
    assert ahmf.shape[-1] == NLEV


def test_builder_refuses_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_orca2_ldf_dyn_coefficients(
            tmp_path / "absent.nc", np.ones((2, 2, 1)), np.ones((2, 2, 1)))


def test_the_mask_is_not_vacuous(tmp_path):
    """A dry cell must actually be zeroed, or the masking proves nothing."""
    path, raw_t, _ = _fields(tmp_path)
    tmask = np.ones((NLAT, NLON, NLEV))
    tmask[2, 3, 1] = 0.0
    ahmt, _ = build_orca2_ldf_dyn_coefficients(path, tmask, tmask)
    assert raw_t[2, 3, 1] != 0.0
    assert ahmt[2, 3, 1] == 0.0


def _fold_consistent_pair():
    """A file that already satisfies both compiled T-pivot rules."""
    nx, ny, nz = gate.GLOBAL_NX, gate.GLOBAL_NY, 2
    rng = np.random.default_rng(5)
    ahmt = rng.random((ny, nx, nz))
    right = np.arange(nx // 2 + 1, nx)
    ahmt[-1, right] = ahmt[-1, nx - right]
    ahmf = rng.random((ny, nx, nz))
    ahmf[-1] = ahmf[-2, ::-1]
    return ahmt, ahmf


def test_fold_consistency_accepts_a_file_the_exchange_would_not_change():
    ahmt, ahmf = _fold_consistent_pair()
    result = gate.fold_consistency(ahmt, ahmf)
    assert result["exchange_is_identity_on_owned_cells"] is True
    assert result["t_point_row_unequal"] == 0
    assert result["f_point_row_unequal"] == 0


@pytest.mark.parametrize("break_field", ["t", "f"])
def test_fold_consistency_refuses_when_the_exchange_would_move_a_cell(break_field):
    """Non-vacuity: omitting NEMO's exchange is only legal while this holds."""
    ahmt, ahmf = _fold_consistent_pair()
    if break_field == "t":
        ahmt[-1, gate.GLOBAL_NX - 3, 0] += 1.0
    else:
        ahmf[-1, 7, 0] += 1.0
    with pytest.raises(gate.GateError, match="NOT fold-consistent"):
        gate.fold_consistency(ahmt, ahmf)


def test_score_reports_the_worst_cell():
    a = np.zeros((2, 3))
    b = np.zeros((2, 3))
    b[1, 2] = 4.0
    row = gate.score(a, b)
    assert row["bit_identical"] is False
    assert row["unequal"] == 1
    assert row["worst_cell"] == [1, 2]
    assert gate.score(a, a)["bit_identical"] is True


# --- round 8's OPEN item 3: the other half of the tripolar fold row ----------

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_phase2n_een_operand_gate as een,
)


def _write_rank_block(path, magic, owned):
    """One rank's instrumented dump: owned cells filled, halo slots empty."""
    local = np.zeros((een.LOCAL_NY, een.LOCAL_NX, een.NZ))
    local[een.HALO:-een.HALO, een.HALO:-een.HALO, :] = owned
    payload = local.transpose(1, 0, 2).ravel(order="F")
    path.write_bytes(magic.ljust(16).encode("ascii")
                     + b"\0" * 40 + payload.tobytes())


def _write_per_rank(tmp_path, *, halo_leak=False):
    rng = np.random.default_rng(8)
    halves = {}
    for name, stem, magic in (
        ("e3f_0vor", "oracle_een_e3f0vor", "NEMO_L4_E3F0_1"),
        ("live_e3f_vor", "oracle_een_e3fvor", "NEMO_L4_E3FV_1"),
    ):
        blocks = []
        for rank in (0, 1):
            owned = rng.random((148, 90, een.NZ)) + 1.0
            blocks.append(owned)
            path = tmp_path / f"{stem}_kt00000001_r{rank:04d}.bin"
            _write_rank_block(path, magic, owned)
            if halo_leak and rank == 1:
                raw = bytearray(path.read_bytes())
                raw[56:64] = np.float64(3.0).tobytes()
                path.write_bytes(bytes(raw))
        halves[name] = np.concatenate(blocks, axis=1)
    return halves


def test_per_rank_record_joins_both_halves_of_the_fold_row(tmp_path):
    halves = _write_per_rank(tmp_path)
    recorded, coverage = een._stitch_per_rank(tmp_path)
    for name, expected in halves.items():
        assert recorded[name].shape == (148, een.GLOBAL_NX, een.ACTIVE_NZ)
        np.testing.assert_array_equal(
            recorded[name], expected[..., :een.ACTIVE_NZ])
    # each rank is checked in its own right, not just the join
    assert set(coverage) == {
        "e3f_0vor_rank0", "e3f_0vor_rank1",
        "live_e3f_vor_rank0", "live_e3f_vor_rank1",
    }
    assert all(row["halo_slots_all_zero"] for row in coverage.values())


def test_per_rank_record_refuses_a_halo_slot_that_is_not_empty(tmp_path):
    """Non-vacuity: the emptiness of the halo is asserted, not assumed.

    Round 8 retracted a count that read those zeroed slots as NEMO values;
    the guard that prevents it must still be able to fire.
    """
    _write_per_rank(tmp_path, halo_leak=True)
    with pytest.raises(een.GateError, match="halo slot is non-zero"):
        een._stitch_per_rank(tmp_path)
