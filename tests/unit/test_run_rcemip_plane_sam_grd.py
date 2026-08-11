"""``--sam-grd`` must actually put the column on gSAM's grd levels (#1562).

The flag parsed and the run started, but ``args.sam_grd`` was never read: a
"SAM oracle grid" run silently used the uniform/stretched column instead, and
every profile comparison against that oracle was then made across two different
vertical grids — with no error and the flag echoed back in the log.

These are behavioural checks on the builder, not on argparse: a CLI round-trip
only proves the string is stored, which is exactly the thing that was already
true while the flag did nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "scripts" / "run")):
    if p not in sys.path:
        sys.path.insert(0, p)

jax.config.update("jax_enable_x64", True)

P_SFC_RCEMIP = 101480.0


def _driver():
    import run_rcemip_plane
    return run_rcemip_plane


def _theta_ref_fn():
    """The driver's own reference profile — a constant-theta default would put
    exner_ref<0 near the top of a 33 km column and NaN the reference state."""
    m = _driver()
    return lambda z: m._rcemip_theta_profile(
        z, T_sfc=300.0, q_sfc=m.WING_Q_SFC_DEFAULT)


def _args(tmp_path, grd_lines, nlev=25, sounding=None):
    grd = tmp_path / "grd"
    grd.write_text(grd_lines)
    return SimpleNamespace(sam_grd=str(grd), nlev=nlev, sounding=sounding)


# RCEMIP1's own shape: a single column of scalar heights, fewer rows than the
# run's level count, stretched near the surface then uniform.
_ONE_COL = "\n".join(f"{z:.1f}" for z in
                     (37.0, 112.0, 200.0, 300.0, 420.0, 570.0, 760.0,
                      1000.0, 1300.0, 1700.0, 2200.0, 2800.0, 3300.0,
                      3800.0, 4300.0)) + "\n"
# GATE_IDEAL's shape: the 3-column "z idx spacing" form. Deliberately NOT
# uniformly spaced — a uniform fixture is reproducible by
# create_height_coordinate, which would make the non-vacuity check below pass
# whether or not the file was read.
_THREE_COL_Z = (25.0, 75.0, 125.0, 200.0, 300.0, 450.0, 650.0, 900.0)
_THREE_COL = "\n".join(
    f"{z:.1f} {k + 1:d} {(z - _THREE_COL_Z[k - 1]) if k else z:.1f}"
    for k, z in enumerate(_THREE_COL_Z)) + "\n"


@pytest.mark.parametrize("grd_text", [_ONE_COL, _THREE_COL])
def test_sam_grd_levels_reach_the_height_coordinate(tmp_path, grd_text):
    """The interfaces of the built column ARE the file's, not a rebuild."""
    m = _driver()
    from legoesm.atmosphere.forcing.sam_case_forcing import read_sam_grd

    n_rows = len([ln for ln in grd_text.splitlines() if ln.strip()])
    args = _args(tmp_path, grd_text, nlev=n_rows)
    hc = m.build_sam_grd_height_coord(args, P_SFC_RCEMIP, _theta_ref_fn())

    expected = read_sam_grd(args.sam_grd, n_levels=n_rows).z_half
    assert hc.n_levels == n_rows
    np.testing.assert_allclose(np.asarray(hc.z_half), expected, rtol=0, atol=0)
    # Non-vacuous: the oracle grid must NOT be reproducible by the analytic
    # families the flag claims to override, or this test would pass unwired.
    uniform = m.create_height_coordinate(
        n_rows, H=float(hc.H), p_sfc=P_SFC_RCEMIP,
        theta_ref_fn=_theta_ref_fn())
    assert not np.allclose(np.asarray(hc.z_half), np.asarray(uniform.z_half))


def test_cell_centres_are_a_quarter_of_the_second_difference_off_sam(tmp_path):
    """The interfaces are exact; the CENTRES are not — pin the exact relation
    rather than a metres number that only holds for one file.

    ``read_sam_grd`` places interfaces at SAM's ``zi(k)=(z(k-1)+z(k))/2``, and
    ``HeightCoordinate`` re-derives ``z_full`` as the midpoint of those. The
    composition gives ``z_full[k] - z_sam[k] = (z[k-1] - 2 z[k] + z[k+1]) / 4``:
    a quarter of the SECOND difference, i.e. zero wherever the spacing is
    uniform and proportional to how fast it changes where it is not. That is
    why ``read_sam_grd``'s own NOTE can quote ~1.5 m for RCEMIP1's mild
    stretch while a sharply stretched column is off by far more.
    """
    m = _driver()
    from legoesm.atmosphere.forcing.sam_case_forcing import read_sam_grd

    n_rows = len(_ONE_COL.split())
    args = _args(tmp_path, _ONE_COL, nlev=n_rows)
    hc = m.build_sam_grd_height_coord(args, P_SFC_RCEMIP, _theta_ref_fn())

    sam_z = read_sam_grd(args.sam_grd, n_levels=n_rows).z_full_bottom_up
    # HeightCoordinate stores top-to-bottom; SAM's z_full is bottom-to-top.
    offset = np.asarray(hc.z_full)[::-1] - sam_z
    # Interior: a quarter of the second difference.
    predicted = (sam_z[:-2] - 2.0 * sam_z[1:-1] + sam_z[2:]) / 4.0
    np.testing.assert_allclose(offset[1:-1], predicted, atol=1e-9)
    # Surface: zi(0)=0 is not the midpoint of two scalar levels, so the ends
    # follow their own rules — asserted rather than masked out.
    np.testing.assert_allclose(offset[0], (sam_z[1] - 3.0 * sam_z[0]) / 4.0,
                               atol=1e-9)
    # Top: the extrapolated interface makes the centre land exactly on SAM's.
    np.testing.assert_allclose(offset[-1], 0.0, atol=1e-9)
    # Non-vacuous: the offset must be REAL on a stretched grid, or this test
    # would also pass against a coordinate that reproduced SAM exactly and the
    # docstring's caveat would be unfounded.
    assert np.abs(offset).max() > 0.0

    # The INTERIOR term must VANISH on a uniformly spaced column, or "second
    # difference" is the wrong description of it. The surface term does NOT
    # vanish there — for z[k] = (k+1) dz it is (z[1] - 3 z[0])/4 = -dz/4 —
    # which is exactly why the ends are asserted separately above.
    uni = tmp_path / "grd_uniform"
    dz = 50.0
    uni.write_text("\n".join(f"{dz * k:.1f}" for k in range(1, 9)) + "\n")
    hc_u = m.build_sam_grd_height_coord(
        SimpleNamespace(sam_grd=str(uni), nlev=8, sounding=None),
        P_SFC_RCEMIP, _theta_ref_fn())
    sam_u = read_sam_grd(str(uni), n_levels=8).z_full_bottom_up
    off_u = np.asarray(hc_u.z_full)[::-1] - sam_u
    np.testing.assert_allclose(off_u[1:-1], 0.0, atol=1e-9)
    np.testing.assert_allclose(off_u[0], -dz / 4.0, atol=1e-9)


def test_sam_grd_extends_a_short_file_to_nlev(tmp_path):
    """SAM's setgrid.f90 rule: keep adding the last spacing. RCEMIP1 ships 25
    rows for a 74-level run, so a driver that could not extend would be
    unusable on the very file the help text names."""
    m = _driver()
    args = _args(tmp_path, _ONE_COL, nlev=40)
    hc = m.build_sam_grd_height_coord(args, P_SFC_RCEMIP, _theta_ref_fn())
    assert hc.n_levels == 40
    dz = np.asarray(hc.dz)
    # Top-to-bottom storage: the extension is uniform at the last spacing.
    assert np.allclose(dz[:10], dz[0])


def test_sam_grd_and_sounding_together_are_refused(tmp_path):
    """Both flags choose the vertical coordinate. Letting either win silently
    is the #1562 failure wearing a different hat."""
    m = _driver()
    args = _args(tmp_path, _ONE_COL, sounding="/nonexistent/snd")
    with pytest.raises(SystemExit, match="both choose the vertical"):
        m.build_sam_grd_height_coord(args, P_SFC_RCEMIP, _theta_ref_fn())


def test_missing_grd_file_is_refused(tmp_path):
    m = _driver()
    args = SimpleNamespace(sam_grd=str(tmp_path / "absent"), nlev=25,
                           sounding=None)
    with pytest.raises(SystemExit, match="file not found"):
        m.build_sam_grd_height_coord(args, P_SFC_RCEMIP, _theta_ref_fn())


def test_main_routes_sam_grd_ahead_of_the_analytic_builders():
    """The builder existing is not the fix — ``main`` has to call it. Guard the
    dispatch itself, naming the symbol that runs."""
    import inspect
    src = inspect.getsource(_driver().main)
    assert "build_sam_grd_height_coord(" in src
    # ...and ahead of both analytic builders, so --sam-grd is not outranked.
    i = src.index("build_sam_grd_height_coord(")
    for later in ("create_stretched_height_coordinate(",
                  "create_height_coordinate("):
        assert i < src.index(later)
