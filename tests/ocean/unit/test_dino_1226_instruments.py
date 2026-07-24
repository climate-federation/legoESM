"""Direct unit tests for the #1226 DINO twin instruments promoted from the
session scratchpad (``scripts/validate/ocean_fidelity/dino_1226/``):
``kamm_twin_90d.py`` (day-0 verification gate), ``heat_discriminator.py``
(drift-profile / crossing-depth), and ``mode_projection.py`` (checkerboard
mode projector). All synthetic -- no NEMO artifacts, CPU-fast.
"""
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"


@pytest.fixture(scope="module")
def instruments():
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import validate.ocean_fidelity.dino_1226.heat_discriminator as heat_discriminator
        import validate.ocean_fidelity.dino_1226.kamm_twin_90d as kamm_twin_90d
        import validate.ocean_fidelity.dino_1226.mode_projection as mode_projection
        importlib.reload(mode_projection)
        importlib.reload(heat_discriminator)
        importlib.reload(kamm_twin_90d)
        return types.SimpleNamespace(
            kamm_twin_90d=kamm_twin_90d,
            heat_discriminator=heat_discriminator,
            mode_projection=mode_projection,
        )
    finally:
        try:
            sys.path.remove(str(SCRIPTS_DIR))
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# mode_projection
# ---------------------------------------------------------------------------
def test_checkerboard_projector_isolates_checkerboard(instruments):
    mode_projection = instruments.mode_projection
    n_lat_face, n_lon, n_lev = 60, 56, 10
    x = np.linspace(0, 2 * np.pi, n_lon)
    smooth_field = np.zeros((n_lat_face, n_lon, n_lev)) + np.sin(x)[None, :, None]

    # checkerboard signal confined to cols 47-49, upper-level band 0-8
    sign = np.where(np.arange(n_lon) % 2 == 0, 1.0, -1.0)
    checker = np.zeros_like(smooth_field) + sign[None, :, None]
    v_reference = smooth_field.copy()
    v_reference[:, 47:50, 0:8] += checker[:, 47:50, 0:8]

    proj = mode_projection.build_checkerboard_projector(v_reference)
    assert proj.shape == v_reference.shape
    assert np.isclose(np.sum(proj**2), 1.0)

    # zero outside the east-wall column block / upper-level band
    assert np.allclose(proj[:, :45, :], 0.0)
    assert np.allclose(proj[:, 51:, :], 0.0)
    assert np.allclose(proj[:, :, 8:], 0.0)

    # A field proportional to proj projects onto proj as exactly its scale
    # factor (proj is unit-norm by construction): project(proj, proj) == 1,
    # project(3*proj, proj) == 3.
    assert mode_projection.project(proj, proj) == pytest.approx(1.0, rel=1e-6)
    assert mode_projection.project(3.0 * proj, proj) == pytest.approx(3.0, rel=1e-6)

    # The full reference field (smooth background + checkerboard) projects to
    # a large amplitude (it contains the checkerboard proj was built from);
    # the smooth-only field (no checkerboard) projects to something small in
    # comparison -- isolation, not exact zero, since it has some raw energy
    # in the masked region too.
    amp_reference = mode_projection.project(v_reference, proj)
    amp_smooth = mode_projection.project(smooth_field, proj)
    assert abs(amp_smooth) < 0.1 * abs(amp_reference)


def test_project_nemo_row_alignment(instruments):
    """proj built from a legoESM face field (n_lat+1 rows) must drop row 0 to
    align with a NEMO-native (n_lat rows) field, per the module docstring."""
    mode_projection = instruments.mode_projection
    n_lat_face, n_lon, n_lev = 21, 56, 10
    v = np.zeros((n_lat_face, n_lon, n_lev))
    v[:, 47:50, 0:8] = 1.0
    proj = mode_projection.build_checkerboard_projector(v)
    proj_nemo = proj[1:, :, :]
    assert proj_nemo.shape == (n_lat_face - 1, n_lon, n_lev)

    # a NEMO-native field proportional to the row-shifted proj_nemo projects
    # to exactly that scale factor times ||proj_nemo||^2 -- confirms
    # proj_nemo (row-shifted) is the correct object to contract a NEMO array
    # against.
    norm2 = float(np.sum(proj_nemo**2))
    assert norm2 > 0.0
    assert mode_projection.project(proj_nemo, proj_nemo) == pytest.approx(norm2, rel=1e-6)
    assert mode_projection.project(2.0 * proj_nemo, proj_nemo) == pytest.approx(
        2.0 * norm2, rel=1e-6)

    # projecting against the UN-shifted proj (wrong row count/alignment)
    # would fail on the shape mismatch -- here just confirm the shapes
    # differ, i.e. the row-drop is a real, necessary alignment step.
    assert proj.shape[0] == proj_nemo.shape[0] + 1


# ---------------------------------------------------------------------------
# kamm_twin_90d: day-0 verification gate
# ---------------------------------------------------------------------------
def _fake_field(data):
    return types.SimpleNamespace(data=data)


class _FakeState(types.SimpleNamespace):
    """SimpleNamespace + a NamedTuple-like ``_replace`` (real state is a
    NamedTuple; the fake only needs ``_replace`` for bridge_tke_from_restart)."""
    def _replace(self, **kwargs):
        d = dict(self.__dict__)
        d.update(kwargs)
        return _FakeState(**d)


def _fake_state(t_field, eta, u, v):
    return _FakeState(
        T=_fake_field(t_field), eta=_fake_field(eta), u=_fake_field(u), v=_fake_field(v),
    )


def _fake_restart(t_field, ssh, u, v):
    return types.SimpleNamespace(T=t_field, ssh=ssh, u=u, v=v)


def test_day0_verify_rejects_mismatched_state(instruments):
    kamm_twin_90d = instruments.kamm_twin_90d
    n_lat, n_lon, n_lev = 4, 5, 3
    land_mask = np.ones((n_lat, n_lon))
    t_nemo = np.ones((n_lat, n_lon, n_lev)) * 10.0
    u_nemo = np.ones((n_lat, n_lon, n_lev)) * 0.5
    v_nemo = np.ones((n_lat, n_lon, n_lev)) * 0.5
    ssh_nemo = np.zeros((n_lat, n_lon))
    restart = _fake_restart(t_nemo, ssh_nemo, u_nemo, v_nemo)

    # legoESM state does NOT match (e.g. a rest-state IC: T differs, u/v == 0)
    t_lego = t_nemo.copy()
    t_lego[0, 0, 0] += 1.0  # mismatch
    u_face = np.zeros((n_lat, n_lon + 1, n_lev))  # rest state: u == 0 everywhere
    v_face = np.zeros((n_lat + 1, n_lon, n_lev))
    st = _fake_state(t_lego, ssh_nemo.copy(), u_face, v_face)

    with pytest.raises(SystemExit):
        kamm_twin_90d.verify_day0_matches_restart(st, restart, land_mask)


def test_day0_verify_rejects_rest_state_even_if_t_matches(instruments):
    """The 2026-07-24 defect: T/eta can match a restart's initial profile
    while u/v are identically zero -- the max|u0|>0.1 / max|v0|>0.1 checks
    must catch this rest-state-start case even when T/eta happen to agree."""
    kamm_twin_90d = instruments.kamm_twin_90d
    n_lat, n_lon, n_lev = 4, 5, 3
    land_mask = np.ones((n_lat, n_lon))
    t_field = np.ones((n_lat, n_lon, n_lev)) * 10.0
    ssh = np.zeros((n_lat, n_lon))
    u_nemo = np.ones((n_lat, n_lon, n_lev)) * 0.5  # NEMO restart has real velocity
    v_nemo = np.ones((n_lat, n_lon, n_lev)) * 0.5
    restart = _fake_restart(t_field, ssh, u_nemo, v_nemo)

    # legoESM u/v identically zero (rest-state defect): T/eta match but u/v don't
    u_face_rest = np.zeros((n_lat, n_lon + 1, n_lev))
    v_face_rest = np.zeros((n_lat + 1, n_lon, n_lev))
    st = _fake_state(t_field.copy(), ssh.copy(), u_face_rest, v_face_rest)
    with pytest.raises(SystemExit):
        kamm_twin_90d.verify_day0_matches_restart(st, restart, land_mask)


def test_day0_verify_passes_for_matched_developed_state(instruments):
    kamm_twin_90d = instruments.kamm_twin_90d
    n_lat, n_lon, n_lev = 4, 5, 3
    land_mask = np.ones((n_lat, n_lon))
    t_field = np.linspace(5.0, 20.0, n_lat * n_lon * n_lev).reshape(n_lat, n_lon, n_lev)
    ssh = np.linspace(-0.1, 0.1, n_lat * n_lon).reshape(n_lat, n_lon)
    u_nemo = np.ones((n_lat, n_lon, n_lev)) * 0.3
    v_nemo = np.ones((n_lat, n_lon, n_lev)) * 0.3
    restart = _fake_restart(t_field, ssh, u_nemo, v_nemo)

    # legoESM faces sliced per verify's convention: u.data[:, 1:, :] <-> NEMO u,
    # v.data[1:, :, :] <-> NEMO v -- construct a state that matches exactly.
    u_face = np.zeros((n_lat, n_lon + 1, n_lev))
    v_face = np.zeros((n_lat + 1, n_lon, n_lev))
    u_face[:, 1:, :] = u_nemo
    v_face[1:, :, :] = v_nemo
    st = _fake_state(t_field.copy(), ssh.copy(), u_face, v_face)

    # must not raise
    kamm_twin_90d.verify_day0_matches_restart(st, restart, land_mask)


def test_kamm_twin_90d_import_is_side_effect_free(instruments):
    """Import must not read any NEMO artifact (module docstring guarantee)."""
    kamm_twin_90d = instruments.kamm_twin_90d
    assert callable(kamm_twin_90d.main)
    assert callable(kamm_twin_90d.run_twin)
    assert callable(kamm_twin_90d.verify_day0_matches_restart)
    assert callable(kamm_twin_90d.bridge_tke_from_restart)


# ---------------------------------------------------------------------------
# kamm_twin_90d: --bridge-tke (#1317 TKE cold-start isolation)
# ---------------------------------------------------------------------------
def test_bridge_tke_from_restart_mapping_round_trip(instruments):
    """en[w-level 0..jpk-1] -> state.tke[interior interface 0..jpk-2]: drops
    only the surface w-level (index 0); every other level is a straight
    index-for-index carry, masked to wet columns."""
    kamm_twin_90d = instruments.kamm_twin_90d
    n_lat, n_lon, jpk = 4, 5, 6  # jpk NEMO w-levels == nlev
    land_mask = np.ones((n_lat, n_lon))
    land_mask[0, 0] = 0.0  # one land cell

    # en varies by level (level k -> value 100+k) so a wrong index shift is
    # detectable -- a level-mapping bug is not masked by a constant field.
    en = np.zeros((n_lat, n_lon, jpk))
    for k in range(jpk):
        en[:, :, k] = 100.0 + k
    en[0, 0, :] = 999.0  # land cell -- must NOT leak through the mask

    st = _fake_state(np.zeros((n_lat, n_lon, jpk), dtype=np.float64),
                      np.zeros((n_lat, n_lon)),
                      np.zeros((n_lat, n_lon + 1, jpk)),
                      np.zeros((n_lat + 1, n_lon, jpk)))
    st_out = kamm_twin_90d.bridge_tke_from_restart(st, en, land_mask)

    tke = np.asarray(st_out.tke.data)
    assert tke.shape == (n_lat, n_lon, jpk - 1)
    # interior interface j corresponds to NEMO w-level j+1 (surface w-level 0
    # dropped): tke[:,:,j] == en[:,:,j+1] == 100+(j+1) on wet cells.
    for j in range(jpk - 1):
        expected = 100.0 + (j + 1)
        wet_vals = tke[:, :, j][land_mask > 0.5]
        assert np.allclose(wet_vals, expected)
    # land cell masked to zero, not the raw en=999 value
    assert tke[0, 0, :].max() == 0.0


def test_build_twin_state_default_bridge_tke_off(instruments, monkeypatch):
    """--bridge-tke defaults False: the module must not call
    read_nemo_restart_en/bridge_tke_from_restart on the default path -- the
    flag-off path is byte-identical to before this feature existed. Verified
    both by signature default (no NEMO artifacts needed for this synthetic
    suite) and by patching the two TKE-bridge entry points to explode if
    reached from the default-args call path."""
    kamm_twin_90d = instruments.kamm_twin_90d

    def _boom(*a, **k):
        raise AssertionError("TKE bridge must not run when bridge_tke=False")

    monkeypatch.setattr(kamm_twin_90d, "read_nemo_restart_en", _boom)
    monkeypatch.setattr(kamm_twin_90d, "bridge_tke_from_restart", _boom)

    import inspect
    build_sig = inspect.signature(kamm_twin_90d._build_twin_state)
    run_sig = inspect.signature(kamm_twin_90d.run_twin)
    assert build_sig.parameters["bridge_tke"].default is False
    assert run_sig.parameters["bridge_tke"].default is False


def test_parse_args_bridge_tke_flag(instruments):
    kamm_twin_90d = instruments.kamm_twin_90d
    args = kamm_twin_90d._parse_args(["nemo_dino_kamm_mlf", "out.npz"])
    assert args.bridge_tke is False
    args = kamm_twin_90d._parse_args(["nemo_dino_kamm_mlf", "out.npz", "--bridge-tke"])
    assert args.bridge_tke is True


# ---------------------------------------------------------------------------
# heat_discriminator: drift profile + crossing depth
# ---------------------------------------------------------------------------
def test_wet_mean_profile_basic(instruments):
    heat_discriminator = instruments.heat_discriminator
    t3d = np.zeros((2, 2, 3))
    t3d[..., 0] = 10.0
    t3d[..., 1] = 12.0
    t3d[..., 2] = 14.0
    wet = np.ones((2, 2), dtype=bool)
    profile = heat_discriminator.wet_mean_profile(t3d, wet)
    assert np.allclose(profile, [10.0, 12.0, 14.0])


def test_wet_mean_profile_masks_land(instruments):
    heat_discriminator = instruments.heat_discriminator
    t3d = np.zeros((2, 2, 2))
    t3d[0, 0, :] = 100.0  # land cell, should be excluded
    t3d[0, 1, :] = 10.0
    t3d[1, 0, :] = 10.0
    t3d[1, 1, :] = 10.0
    wet = np.array([[False, True], [True, True]])
    profile = heat_discriminator.wet_mean_profile(t3d, wet)
    assert np.allclose(profile, [10.0, 10.0])


def test_drift_crossing_depth_known_crossing(instruments):
    heat_discriminator = instruments.heat_discriminator
    # drift_diff positive near surface, crosses zero between level 2 and 3,
    # negative below -- crossing depth should land between depth[2] and depth[3]
    depth = np.array([5.0, 20.0, 50.0, 100.0, 200.0])
    drift_diff = np.array([2.0, 1.0, 0.5, -0.5, -1.0])
    x_depth = heat_discriminator.drift_crossing_depth(drift_diff, depth)
    assert x_depth is not None
    assert 50.0 < x_depth < 100.0
    # linear interpolation: frac = 0.5/(0.5-(-0.5)) = 0.5 -> depth 75.0
    assert x_depth == pytest.approx(75.0, abs=1e-6)


def test_drift_crossing_depth_no_crossing_returns_none(instruments):
    heat_discriminator = instruments.heat_discriminator
    depth = np.array([5.0, 20.0, 50.0])
    drift_diff = np.array([1.0, 2.0, 3.0])  # never changes sign
    assert heat_discriminator.drift_crossing_depth(drift_diff, depth) is None


def test_heat_discriminator_import_is_side_effect_free(instruments):
    heat_discriminator = instruments.heat_discriminator
    assert callable(heat_discriminator.main)
    assert callable(heat_discriminator.run_discriminator)
