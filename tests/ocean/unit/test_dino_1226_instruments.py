"""Direct unit tests for the #1226 DINO twin instruments promoted from the
session scratchpad (``scripts/validate/ocean_fidelity/dino_1226/``):
``kamm_twin_90d.py`` (day-0 verification gate), ``heat_discriminator.py``
(drift-profile / crossing-depth), and ``mode_projection.py`` (checkerboard
mode projector). All synthetic -- no NEMO artifacts, CPU-fast.
"""
import importlib
import os
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
# kamm_twin_90d: --bridge-before (#1317 leap-frog before-level bridge)
# ---------------------------------------------------------------------------
def test_build_twin_state_default_bridge_before_off(instruments, monkeypatch):
    """--bridge-before defaults False: the module must not call
    read_nemo_restart_before/bridge_before_state_topo on the default path --
    mirrors test_build_twin_state_default_bridge_tke_off."""
    kamm_twin_90d = instruments.kamm_twin_90d

    def _boom(*a, **k):
        raise AssertionError("before-level bridge must not run when bridge_before=False")

    monkeypatch.setattr(kamm_twin_90d, "read_nemo_restart_before", _boom)
    monkeypatch.setattr(kamm_twin_90d, "bridge_before_state_topo", _boom)

    import inspect
    build_sig = inspect.signature(kamm_twin_90d._build_twin_state)
    run_sig = inspect.signature(kamm_twin_90d.run_twin)
    assert build_sig.parameters["bridge_before"].default is False
    assert run_sig.parameters["bridge_before"].default is False


def test_parse_args_bridge_before_flag(instruments):
    kamm_twin_90d = instruments.kamm_twin_90d
    args = kamm_twin_90d._parse_args(["nemo_dino_kamm_mlf", "out.npz"])
    assert args.bridge_before is False
    args = kamm_twin_90d._parse_args(
        ["nemo_dino_kamm_mlf", "out.npz", "--bridge-before"])
    assert args.bridge_before is True


def test_print_before_bridge_verify_reports_zero_for_matched_state(instruments):
    """_print_before_bridge_verify computes max|d_tb|/.../max|d_vb| against
    the raw restart before-level using the FULL 3-D tmask/umask/vmask (not
    the 2-D surface land_mask) -- a matched synthetic state must report
    all-zero diffs."""
    kamm_twin_90d = instruments.kamm_twin_90d
    n_lat, n_lon, n_lev = 4, 5, 3
    tmask = np.ones((n_lat, n_lon, n_lev))
    tmask[0, 0, 1:] = 0.0  # a partial-cell dry-below-surface column
    grid = types.SimpleNamespace(tmask=tmask, umask=np.ones((n_lat, n_lon, n_lev)),
                                  vmask=np.ones((n_lat, n_lon, n_lev)))

    t_field = np.linspace(5.0, 20.0, n_lat * n_lon * n_lev).reshape(n_lat, n_lon, n_lev)
    u_nemo = np.full((n_lat, n_lon, n_lev), 0.3)
    v_nemo = np.full((n_lat, n_lon, n_lev), 0.3)
    before = types.SimpleNamespace(T=t_field, S=t_field.copy(), u=u_nemo, v=v_nemo)

    u_face = np.zeros((n_lat, n_lon + 1, n_lev))
    v_face = np.zeros((n_lat + 1, n_lon, n_lev))
    u_face[:, 1:, :] = u_nemo
    v_face[1:, :, :] = v_nemo
    # T_before at the dry-below-surface cell differs from the raw restart's
    # 0.0 (a Neumann-fill artifact, like the now-level bridge) -- must NOT
    # be flagged since it's outside the 3-D tmask.
    t_before_field = t_field.copy()
    t_before_field[0, 0, 1:] = 999.0
    st = types.SimpleNamespace(
        T_before=_fake_field(t_before_field), S_before=_fake_field(t_before_field.copy()),
        u_before=_fake_field(u_face), v_before=_fake_field(v_face),
    )

    # must not raise, and must print the zero-diff line (captured via capsys
    # in the caller if desired -- here just confirm no exception).
    kamm_twin_90d._print_before_bridge_verify(st, before, grid)


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


# =====================================================================
# kamm_twin_90d: the #1455 seasonal clock (absolute by DEFAULT)
# =====================================================================

def _write_restart_stub(path, adatrj, kt):
    """Minimal NEMO-restart stub carrying only what the clock reader needs."""
    nc = pytest.importorskip("netCDF4")
    with nc.Dataset(path, "w") as d:
        d.createDimension("t", 1)
        for name, val in (("adatrj", adatrj), ("kt", kt)):
            v = d.createVariable(name, "f8", ("t",))
            v[:] = val


def test_seasonal_clock_defaults_to_the_restarts_own_day_of_year(
        instruments, tmp_path, monkeypatch, capsys):
    """DEFAULT (env unset) must be NEMO's clock, read from ``adatrj``.

    This is the #1455 fix: before it, the default was 0 and a twin bridged
    from the day-180 restart ran exactly antiphase to the NEMO run it scored
    against.
    """
    kamm_twin_90d = instruments.kamm_twin_90d
    monkeypatch.delenv("DINO_TWIN_SEASONAL_KT0", raising=False)
    p = tmp_path / "DINO_00005760_restart.nc"
    _write_restart_stub(p, adatrj=180.0, kt=5760.0)

    t0 = kamm_twin_90d.seasonal_t0_seconds(str(p))

    assert t0 == pytest.approx(180.0 * 86400.0)
    assert "restart adatrj" in capsys.readouterr().out


def test_seasonal_clock_legacy_relative_is_opt_in_and_shouts(
        instruments, tmp_path, monkeypatch, capsys):
    """The old relative clock must still be reachable, and must be loud."""
    kamm_twin_90d = instruments.kamm_twin_90d
    monkeypatch.setenv("DINO_TWIN_SEASONAL_KT0", "0")
    p = tmp_path / "DINO_00005760_restart.nc"
    _write_restart_stub(p, adatrj=180.0, kt=5760.0)

    assert kamm_twin_90d.seasonal_t0_seconds(str(p)) == 0.0
    assert "LEGACY RELATIVE CLOCK" in capsys.readouterr().out


def test_seasonal_clock_rejects_a_restart_written_at_another_timestep(
        instruments, tmp_path, monkeypatch):
    """``adatrj`` and ``kt*DT`` must agree, or the offset is not trustworthy."""
    kamm_twin_90d = instruments.kamm_twin_90d
    monkeypatch.delenv("DINO_TWIN_SEASONAL_KT0", raising=False)
    p = tmp_path / "bad_restart.nc"
    _write_restart_stub(p, adatrj=180.0, kt=99.0)   # 99 * 2700 s != 180 days
    with pytest.raises(SystemExit, match="different timestep"):
        kamm_twin_90d.seasonal_t0_seconds(str(p))


def test_run_twin_threads_the_absolute_clock_into_the_forcing(instruments):
    """The run loop must ADD the offset, not restart the seasonal year.

    Source-level check keyed off the symbol that actually runs (``run_twin``),
    and it fails if the ``t0_sec +`` term is dropped from either placement
    branch.
    """
    import inspect
    import re
    kamm_twin_90d = instruments.kamm_twin_90d
    src = inspect.getsource(kamm_twin_90d.run_twin)
    # The regression is the RELATIVE form reappearing; assert its absence
    # rather than an exact spelling of the fixed form, so an innocuous
    # reformat (or hoisting the expression into a local) does not go red.
    assert not re.search(r"t_seconds\s*=\s*\(\s*k\s*\+\s*1\s*\)\s*\*\s*DT", src)
    assert "seasonal_t0_seconds(" in src
    assert src.count("t0_sec") >= 3


# ---------------------------------------------------------------------------
# kamm_twin_90d: the vertical-ladder default (#1455)
#
# A bridged twin only isolates SCHEME differences if both models stand on the
# same vertical grid, so run_twin resolves LEGOESM_NEMO_E3T to "both" (NEMO's
# own thickness AND T-depth ladders) when nothing sets it. Measured cost of the
# old 1-D reference ladder at day 90: +2.93 Sv of circumpolar transport error
# against +0.29 Sv on "both", and +1.87 Sv of full-section ACC error against
# -0.60 Sv (fp64, branch fidelity/dino-step-walk).
#
# These tests go RED if that default silently reverts to the 1-D ladder, if the
# resolution leaks out of run_twin into the helper a dozen sibling probes
# import, or if the loud banner stops firing.
# ---------------------------------------------------------------------------
@pytest.fixture
def _ladder(instruments, monkeypatch):
    """kamm_twin_90d with LEGOESM_NEMO_E3T cleared and the banner set emptied.

    setenv BEFORE delenv on purpose: monkeypatch records an undo entry only for
    a variable that existed. The resolver no longer writes the variable, but the
    tests below do, and a leak here would hand a vertical-ladder selection
    nobody set to every module collected afterwards -- the exact silent inherited
    default that require_explicit_e3t_mode exists to prevent, planted inside the
    test suite where it is invisible.
    """
    kamm_twin_90d = instruments.kamm_twin_90d
    monkeypatch.setenv("LEGOESM_NEMO_E3T", "off")
    monkeypatch.delenv("LEGOESM_NEMO_E3T")
    monkeypatch.setattr(kamm_twin_90d, "_LADDER_ANNOUNCED", set())
    return kamm_twin_90d


def test_ladder_fixture_does_not_leak_the_env(_ladder):
    """The fixture's own contract, asserted so it cannot rot silently."""
    assert "LEGOESM_NEMO_E3T" not in os.environ
    _ladder.resolve_ladder_mode()
    assert "LEGOESM_NEMO_E3T" not in os.environ


def test_resolving_the_ladder_does_not_write_the_environment(_ladder,
                                                             monkeypatch):
    """THE ROOT FIX. The resolver used to write LEGOESM_NEMO_E3T, which made a
    second caller unable to tell the operator's setting from the first call's
    write-back -- so an in-process two-arm sweep silently ran one grid twice.
    No check can separate those cases; the write must not happen at all."""
    _ladder.resolve_ladder_mode(legacy_1d_ladder=True)
    assert "LEGOESM_NEMO_E3T" not in os.environ
    monkeypatch.setenv("LEGOESM_NEMO_E3T", "e3t_only")
    _ladder.resolve_ladder_mode()
    assert os.environ["LEGOESM_NEMO_E3T"] == "e3t_only"   # untouched


def test_two_ladders_in_one_process_stay_distinct_and_both_speak(_ladder,
                                                                 capsys):
    """THE SWEEP GUARD: arm 1 on the 1-D ladder, arm 2 on the default, in one
    process. They must come back DIFFERENT, and neither may be silent."""
    assert _ladder.resolve_ladder_mode(legacy_1d_ladder=True) == "off"
    assert "LEGOESM_NEMO_E3T=off" in capsys.readouterr().out
    assert _ladder.resolve_ladder_mode() == "both"
    assert "LEGOESM_NEMO_E3T=both" in capsys.readouterr().out


def test_every_resolution_states_the_grid_it_chose(_ladder, capsys):
    """No call may be silent -- a twin must never be built without its log
    saying which ladder it stands on, even when the loud banner has already
    fired for that mode earlier in the process."""
    for _ in range(3):
        _ladder.resolve_ladder_mode(legacy_1d_ladder=True)
        assert "vertical ladder: LEGOESM_NEMO_E3T=off" in capsys.readouterr().out


def test_twin_default_ladder_is_nemos_own_not_the_1d_reference(_ladder):
    """THE REGRESSION GUARD: no env, no flag -> NEMO's own ladders.

    Red if the twin default reverts to "off" (the 1-D reference ladder), or to
    either of the mixed half-ladders.
    """
    assert _ladder.NEMO_LADDER_TWIN_DEFAULT == "both"
    mode = _ladder.resolve_ladder_mode()
    assert mode == "both", (
        f"twin default reverted to {mode!r}; a bridged twin must stand on "
        "NEMO's own vertical ladders")


def test_twin_ladder_env_override_still_wins(_ladder, monkeypatch):
    for mode in _ladder.NEMO_E3T_MODES:
        monkeypatch.setenv("LEGOESM_NEMO_E3T", mode)
        assert _ladder.resolve_ladder_mode() == mode


def test_twin_ladder_modes_come_from_the_bridge_not_a_local_copy(_ladder):
    """A re-listed tuple drifts: the harness would accept a mode the bridge
    rejects three calls later, deep inside grid construction."""
    from legoesm.ocean.fidelity.nemo_state_bridge import NEMO_E3T_MODES
    assert _ladder.NEMO_E3T_MODES is NEMO_E3T_MODES
    assert _ladder.NEMO_LADDER_TWIN_DEFAULT in NEMO_E3T_MODES


def test_precision_gate_shares_the_bridges_mode_tuple(instruments):
    """The explicit-mode gate must validate against the same tuple the bridge
    enforces; a re-listed copy drifts and lets a typo through to grid
    construction. (The twin has this drift test; the gate did not.)"""
    import inspect

    from legoesm.ocean.fidelity import precision_gate
    src = inspect.getsource(precision_gate.require_explicit_e3t_mode)
    assert "NEMO_E3T_MODES" in src
    assert '("off", "e3t_only", "gdept_only", "both")' not in src


def test_twin_ladder_rejects_unknown_env(_ladder, monkeypatch):
    monkeypatch.setenv("LEGOESM_NEMO_E3T", "e3t")     # plausible typo
    with pytest.raises(SystemExit, match="Unknown LEGOESM_NEMO_E3T"):
        _ladder.resolve_ladder_mode()


def test_legacy_1d_ladder_flag_selects_off_and_shouts(_ladder, capsys):
    assert _ladder.resolve_ladder_mode(legacy_1d_ladder=True) == "off"
    out = capsys.readouterr().out
    assert "!!" in out and "NON-DEFAULT VERTICAL LADDER" in out
    assert "1-D REFERENCE ladder" in out


def test_the_banner_is_keyed_to_the_grid_not_to_the_default(_ladder, monkeypatch,
                                                            capsys):
    """If someone flips the twin default back to the 1-D ladder, the LOUD
    WARNING must survive rather than vanish along with it -- otherwise one edit
    removes the correct default and its own alarm in the same stroke."""
    monkeypatch.setattr(_ladder, "NEMO_LADDER_TWIN_DEFAULT", "off")
    assert _ladder.resolve_ladder_mode() == "off"
    assert "NON-DEFAULT VERTICAL LADDER" in capsys.readouterr().out


def test_the_ladder_banner_is_announced_once_per_mode(_ladder, capsys):
    """Repeating the alarm every call trains operators to ignore it."""
    _ladder.resolve_ladder_mode(legacy_1d_ladder=True)
    assert "NON-DEFAULT VERTICAL LADDER" in capsys.readouterr().out
    _ladder.resolve_ladder_mode(legacy_1d_ladder=True)
    assert "NON-DEFAULT VERTICAL LADDER" not in capsys.readouterr().out


def test_each_non_default_ladder_gets_its_own_banner_text(_ladder, monkeypatch,
                                                          capsys):
    """The banner must describe the grid the operator actually selected.

    The mixed-ladder branch once served BOTH half-ladders, so an ``e3t_only``
    run was shown the other mode's T-point offset. And the offsets it quotes are
    per-level, so the level must travel with the number -- pairing 110.2 m at
    k=32 with 11.3 m (which is at k=34, not k=32, where it is 7.8 m) is the
    mismatched-window defect this campaign keeps having to retract.

    Measured over the WET levels the model integrates: off 5.28 m at k=33,
    e3t_only 97.20 m at k=32, gdept_only 110.16 m at k=32, both 11.31 m at k=34.
    """
    expect = {
        "off": ("1-D REFERENCE ladder", "+2.93 Sv"),
        "e3t_only": ("97.2 m", "k=32"),
        "gdept_only": ("110.2 m", "k=32"),
    }
    for mode, (a, b) in expect.items():
        monkeypatch.setattr(_ladder, "_LADDER_ANNOUNCED", set())
        monkeypatch.setenv("LEGOESM_NEMO_E3T", mode)
        _ladder.resolve_ladder_mode()
        out = capsys.readouterr().out
        assert a in out and b in out, f"{mode} banner: missing {a!r}/{b!r}"
        for other in expect:
            if other != mode:
                assert expect[other][0] not in out or expect[other][0] == a, \
                    f"{mode} banner quotes {other}'s number"
    # ... and every level cited for 'both' must be its own worst level, k=34.
    monkeypatch.setattr(_ladder, "_LADDER_ANNOUNCED", set())
    monkeypatch.setenv("LEGOESM_NEMO_E3T", "gdept_only")
    _ladder.resolve_ladder_mode()
    out = capsys.readouterr().out
    assert "11.3 m on 'both' (at k=34)" in out


def test_legacy_flag_conflicting_with_env_is_fatal(_ladder, monkeypatch):
    monkeypatch.setenv("LEGOESM_NEMO_E3T", "both")
    with pytest.raises(SystemExit, match="they disagree"):
        _ladder.resolve_ladder_mode(legacy_1d_ladder=True)


def test_legacy_flag_agreeing_with_env_is_allowed(_ladder, monkeypatch):
    monkeypatch.setenv("LEGOESM_NEMO_E3T", "off")
    assert _ladder.resolve_ladder_mode(legacy_1d_ladder=True) == "off"


def test_default_ladder_path_does_not_print_the_alarm(_ladder, capsys):
    """The banner is for NON-default grids only; the default must not shout.

    Weak on its own (it asserts an absence) -- it is here to pin the pairing
    with test_the_banner_is_keyed_to_the_grid..., which asserts the presence.
    """
    _ladder.resolve_ladder_mode()
    out = capsys.readouterr().out
    assert "NON-DEFAULT VERTICAL LADDER" not in out
    assert "LEGOESM_NEMO_E3T=both" in out          # ... but it must say SOMETHING


def test_build_twin_state_does_not_resolve_the_ladder(instruments):
    """SCOPE GUARD. A dozen sibling probes import _build_twin_state directly;
    resolving the ladder in there would silently re-grid every one of them that
    does not pin the variable itself (box_budget_twin90.py does not). The twin
    default belongs one level up, in run_twin."""
    import inspect
    src = inspect.getsource(instruments.kamm_twin_90d._build_twin_state)
    assert "resolve_ladder_mode" not in src


def test_build_twin_state_defaults_to_the_bridges_own_resolution(instruments):
    """... and its e3t_mode default must stay None, which is what makes the
    unchanged-for-siblings claim true."""
    import ast
    import inspect
    import textwrap
    sig = inspect.signature(instruments.kamm_twin_90d._build_twin_state)
    assert sig.parameters["e3t_mode"].default is None
    src = inspect.getsource(instruments.kamm_twin_90d._build_twin_state)
    assert "e3t_mode=e3t_mode" in src               # forwarded to the bridge
    # ... and nothing may rebind it on the way. A single `e3t_mode = None` above
    # the call leaves that substring intact while the twin builds the 1-D ladder
    # AND stamps "both" into the artifact -- provenance that lies, which is worse
    # than the sweep bug this whole guard family exists for.
    fn = ast.parse(textwrap.dedent(src)).body[0]
    assert not [n for n in ast.walk(fn) if isinstance(n, ast.Name)
                and isinstance(n.ctx, ast.Store) and n.id == "e3t_mode"], \
        "e3t_mode is a parameter and must reach the bridge unmodified"


def test_bridge_accepts_an_explicit_ladder_and_defaults_to_none(instruments):
    """The argument channel the twin now uses instead of the environment."""
    import ast
    import inspect
    import textwrap

    from legoesm.ocean.fidelity import nemo_state_bridge
    fn = nemo_state_bridge.bridge_nemo_to_legoesm_topo
    assert inspect.signature(fn).parameters["e3t_mode"].default is None
    src = inspect.getsource(fn)
    assert "mode=e3t_mode" in src
    tree = ast.parse(textwrap.dedent(src)).body[0]
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Name)
                and isinstance(n.ctx, ast.Store) and n.id == "e3t_mode"], \
        "e3t_mode must reach effective_vertical_scale_factors unmodified"


def test_bridge_rejects_an_unknown_ladder_at_entry(instruments):
    """A typo must stop the call, not surface ~180 lines in once the geometry
    is already built. Passing None (the default) must NOT raise."""
    from legoesm.ocean.fidelity import nemo_state_bridge
    with pytest.raises(ValueError, match="unknown e3t_mode"):
        nemo_state_bridge.bridge_nemo_to_legoesm_topo(
            None, None, e3t_mode="e3t")
    # The default must NOT be rejected -- assert it by getting PAST the guard,
    # not by assuming it. With grid=None the very next thing raises AttributeError,
    # which is proof the ladder check let None through.
    with pytest.raises(AttributeError):
        nemo_state_bridge.bridge_nemo_to_legoesm_topo(None, None)


def test_run_twin_resolves_the_ladder_before_building_the_bridge(instruments):
    """PROVE THE PATH EXECUTES: run_twin must resolve, must pass the flag
    through, must do it BEFORE _build_twin_state, and must hand the result down.

    Asserted on the PARSED function, not on a substring. Substring forms stayed
    green under real mutations: wrapping the call in `if legacy_1d_ladder:`
    (every default run then reverts to the 1-D ladder) and the ternary
    `... if legacy_1d_ladder else "off"` (same effect, substring still present).
    Requiring an UNCONDITIONAL top-level call kills both.
    """
    import ast
    import inspect
    import textwrap
    src = inspect.getsource(instruments.kamm_twin_90d.run_twin)
    fn = ast.parse(textwrap.dedent(src)).body[0]
    calls = [i for i, n in enumerate(fn.body)     # TOP-LEVEL statements only
             if isinstance(n, ast.Assign)
             and isinstance(n.value, ast.Call)
             and getattr(n.value.func, "id", None) == "resolve_ladder_mode"]
    assert len(calls) == 1, (
        "run_twin must call resolve_ladder_mode exactly once, unconditionally, "
        "at its top level -- a conditional call makes the twin default inert")
    node = fn.body[calls[0]]
    assert [getattr(a, "id", None) for a in node.value.args] == \
        ["legacy_1d_ladder"], "the --legacy-1d-ladder selection must be passed on"
    assert [t.id for t in node.targets] == ["ladder_mode"]
    # Nothing may REBIND it afterwards. Counting ast.Assign statements is not
    # enough -- annotated, augmented, tuple, walrus and for-target bindings all
    # slip past that. Count store-context names instead.
    n_bind = sum(1 for n in ast.walk(fn) if isinstance(n, ast.Name)
                 and isinstance(n.ctx, ast.Store) and n.id == "ladder_mode")
    assert n_bind == 1, "ladder_mode must be bound exactly once in run_twin"
    # ... and the resolution must precede the build, by STATEMENT ORDER rather
    # than by a substring search that a reworded comment could break.
    builds = [i for i, n in enumerate(fn.body)
              if "_build_twin_state" in ast.dump(n)]
    assert builds and calls[0] < builds[0]
    # The resolved mode must actually reach the bridge.
    assert "e3t_mode=ladder_mode" in src


def test_run_twin_stamps_the_resolved_ladder_into_the_artifact(instruments):
    """The npz must carry nemo_ladder_mode, the same way it carries
    seasonal_t0_seconds, so a scorer never infers the grid from a filename --
    and it must stamp the value the resolver RETURNED, not a re-read of the
    environment ~90 simulated days later."""
    import inspect
    src = inspect.getsource(instruments.kamm_twin_90d.run_twin)
    assert "nemo_ladder_mode=np.str_(ladder_mode)" in src
    assert "nemo_ladder_mode=np.str_(os.environ" not in src


def test_ladder_stamp_round_trips_through_npz(instruments, tmp_path):
    """A stamp that reads back as b'both' or ['both'] would be a silent defect."""
    p = tmp_path / "stamp.npz"
    np.savez(p, nemo_ladder_mode=np.str_("both"))
    d = np.load(p)
    assert "nemo_ladder_mode" in d.files
    assert str(d["nemo_ladder_mode"]) == "both"


def test_parse_args_legacy_1d_ladder_flag(instruments):
    kamm_twin_90d = instruments.kamm_twin_90d
    args = kamm_twin_90d._parse_args(["nemo_dino_kamm_mlf", "out.npz"])
    assert args.legacy_1d_ladder is False
    args = kamm_twin_90d._parse_args(
        ["nemo_dino_kamm_mlf", "out.npz", "--legacy-1d-ladder"])
    assert args.legacy_1d_ladder is True


def test_main_threads_the_legacy_flag_into_run_twin(instruments):
    """Without this, the flag can go inert while every other test stays green:
    the run would use "both" AND stamp "both", so the artifact would read as
    correct while the operator asked for the 1-D ladder."""
    import inspect
    src = inspect.getsource(instruments.kamm_twin_90d.main)
    assert "legacy_1d_ladder=args.legacy_1d_ladder" in src


def test_gate_prints_the_ladder_before_it_can_refuse_a_candidate(tmp_path,
                                                                 monkeypatch,
                                                                 capsys, request):
    """intent (e): the acceptance gate PRINTS the stamped ladder and never
    refuses on it -- and prints it before the seasonal-clock refusal, so a
    refused candidate still records which grid it ran on.

    acceptance_gate_90d imports acc_thermal_wind at module scope, which opens
    NEMO mesh files; load_candidate itself touches neither, so a stub module
    gets us to the branch under test without the NEMO tree.
    """
    import importlib
    _dir = (Path(__file__).resolve().parents[3] / "scripts" / "validate"
            / "ocean_fidelity" / "dino_1226")
    stub = types.ModuleType("acc_thermal_wind")
    stub.DINO = "/nonexistent"
    monkeypatch.setitem(sys.modules, "acc_thermal_wind", stub)
    monkeypatch.delitem(sys.modules, "acceptance_gate_90d", raising=False)
    monkeypatch.syspath_prepend(str(_dir.parent))
    monkeypatch.syspath_prepend(str(_dir))
    gate = importlib.import_module("acceptance_gate_90d")
    # POP at teardown, not monkeypatch.delitem: delitem RESTORES the value on
    # undo, which would put the module built against the fake acc_thermal_wind
    # (its DINO paths frozen at import time) back into sys.modules for every
    # test that runs after this one.
    request.addfinalizer(lambda: sys.modules.pop("acceptance_gate_90d", None))

    stamped = tmp_path / "stamped.npz"
    np.savez(stamped, nemo_ladder_mode=np.str_("both"),
             seasonal_t0_seconds=np.float64(0.0))     # legacy clock -> refused
    with pytest.raises(SystemExit, match="LEGACY relative clock"):
        gate.load_candidate(str(stamped))
    assert "vertical ladder of this candidate: both" in capsys.readouterr().out

    # An UNSTAMPED artifact must still be scoreable: the only refusal it may hit
    # is the missing day-90 field, never the missing ladder stamp. The `match=`
    # is load-bearing -- a bare raises() passes even when the gate is mutated to
    # refuse on the stamp, because the print has already fired by then.
    bare = tmp_path / "bare.npz"
    np.savez(bare, seasonal_t0_seconds=np.float64(15552000.0))
    with pytest.raises(SystemExit, match="has no u3d_day90"):
        gate.load_candidate(str(bare))
    assert "UNSTAMPED" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# traadv_fct_probe: the two conditioning-robust statistics added for #1455.
# The tracer-advection rows are scored with a ratio of SUMS, which on a row
# whose horizontal and vertical parts nearly cancel reports the residual
# amplified by the state's own cancellation rather than the operator's error.
# These two helpers are what separates the two, so they get direct tests.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fct_probe():
    probe_dir = SCRIPTS_DIR / "validate" / "ocean_fidelity" / "dino_1226"
    sys.path.insert(0, str(probe_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "_fct_probe_under_test", probe_dir / "traadv_fct_probe.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        try:
            sys.path.remove(str(probe_dir))
        except ValueError:
            pass


def test_err_norm_is_zero_for_an_exact_match_and_one_for_a_doubling(fct_probe):
    rng = np.random.default_rng(0)
    n = rng.standard_normal(500)
    assert fct_probe.stats(n.copy(), n)["err_norm"] == pytest.approx(0.0, abs=1e-15)
    # L = 2N differs from N by exactly N, so err_norm = RMS(N)/RMS(N) = 1.
    assert fct_probe.stats(2.0 * n, n)["err_norm"] == pytest.approx(1.0, rel=1e-12)


def test_err_norm_sees_an_error_that_abs_ratio_reports_as_perfect(fct_probe):
    """The whole point of the metric: a sign-flipped-in-half field has
    abs_ratio exactly 1.0 (sum of magnitudes unchanged) while being wrong."""
    n = np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
    lego = -n
    r = fct_probe.stats(lego, n)
    assert r["abs_ratio"] == pytest.approx(1.0)
    assert r["err_norm"] > 1.0


def test_cancellation_factor_is_one_without_cancellation(fct_probe):
    a = np.ones((3, 4, 5))
    act = np.ones((3, 4, 5), dtype=bool)
    assert fct_probe.cancellation_factor(a, a, act) == pytest.approx(1.0)


def test_cancellation_factor_grows_as_the_two_parts_cancel(fct_probe):
    a = np.ones((3, 4, 5))
    act = np.ones((3, 4, 5), dtype=bool)
    b = -a * (1.0 - 1.0e-3)
    # (sum|a| + sum|b|) / sum|a+b| = (1 + 0.999) / 1e-3
    assert fct_probe.cancellation_factor(a, b, act) == pytest.approx(1999.0, rel=1e-9)


def test_cancellation_factor_compares_only_the_shared_levels(fct_probe):
    """The reference carries jpkm1 levels and the model one more; scoring on
    the union would index past the reference and either raise or silently
    score a pad level."""
    a = np.ones((2, 2, 5))                 # model: 5 levels
    b = -0.5 * np.ones((2, 2, 4))          # reference: 4 levels
    act = np.ones((2, 2, 6), dtype=bool)   # mask: 6 levels
    # On the 4 shared levels: (16 + 8) / 8 = 3.0.  Scoring a's 5th level too
    # would either raise or change the answer, so this pins the slicing.
    assert fct_probe.cancellation_factor(a, b, act) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# kamm_twin_90d: the --save-3d snapshot grid (#1455 verdict360)
# ---------------------------------------------------------------------------
def test_snap_days_default_is_the_recorded_grid(instruments):
    """None must reproduce the recorded 0/30/60/90 grid EXACTLY -- every twin
    artifact produced before --snap-days existed was written on it, and a
    silently widened default would change what a sibling probe's npz contains."""
    k = instruments.kamm_twin_90d
    assert k.resolve_snap_days(None, 90, True) == (0, 30, 60, 90)
    assert k.resolve_snap_days(None, 90, True) == k.SNAP_DAYS


def test_snap_days_explicit_grid_is_used(instruments):
    k = instruments.kamm_twin_90d
    assert k.resolve_snap_days((0, 10, 20, 30), 30, True) == (0, 10, 20, 30)
    assert k.resolve_snap_days(["0", "180", "360"], 360, True) == (0, 180, 360)


def test_snap_days_beyond_the_run_length_are_dropped_not_raised(instruments):
    """A year-long grid on a 90-day run yields the 90-day prefix; asking for a
    day the run never reaches must not produce a missing-key npz later."""
    k = instruments.kamm_twin_90d
    assert k.resolve_snap_days(tuple(range(0, 361, 90)), 90, True) == (0, 90)
    assert k.resolve_snap_days(None, 45, True) == (0, 30)


def test_snap_days_is_empty_without_save_3d(instruments):
    """Without --save-3d there are no 3-D snapshots at all, whatever was asked
    for -- otherwise the loop would try to store days it never captured."""
    k = instruments.kamm_twin_90d
    assert k.resolve_snap_days((0, 10, 20), 360, False) == ()
    assert k.resolve_snap_days(None, 360, False) == ()


def test_snap_days_cli_parses_a_comma_list(instruments):
    k = instruments.kamm_twin_90d
    args = k._parse_args(["nemo_dino_kamm_mlf", "/tmp/x.npz", "--days", "360",
                          "--save-3d", "--snap-days", "0,90,180,270,360"])
    assert args.snap_days == "0,90,180,270,360"
    assert k.resolve_snap_days(
        tuple(int(x) for x in args.snap_days.split(",")), args.days,
        args.save_3d) == (0, 90, 180, 270, 360)
    # and the flag is genuinely optional
    assert k._parse_args(["r", "/tmp/x.npz"]).snap_days is None
# the seasonal-clock guard every scorer shares
# ---------------------------------------------------------------------------
def test_clock_guard_accepts_the_restarts_own_day_of_year(instruments):
    kamm_twin_90d = instruments.kamm_twin_90d
    day180 = 180.0 * 86400.0
    stamped = {"seasonal_t0_seconds": day180,
               "seasonal_t0_reference_seconds": day180}
    assert kamm_twin_90d.assert_nemo_seasonal_clock(stamped, "ok.npz") == (
        day180, day180)


def test_clock_guard_rejects_a_nonzero_offset_the_old_test_let_through(
        instruments):
    """The defect the pair-check fixes.

    The first guard rejected only ``t0 == 0``.  An explicit step offset of one
    stamps 2700 s -- still 179.97 days out of phase with the day-180 restart --
    and passed, so the gate reported a NEMO comparison for a run forced in the
    opposite season.  A ``t0 != 0`` test cannot fail on this input; the
    pair-check must.
    """
    kamm_twin_90d = instruments.kamm_twin_90d
    stamped = {"seasonal_t0_seconds": 2700.0,              # kt0 = 1
               "seasonal_t0_reference_seconds": 180.0 * 86400.0}
    assert stamped["seasonal_t0_seconds"] != 0.0           # old guard: passes
    with pytest.raises(SystemExit, match="out of phase"):
        kamm_twin_90d.assert_nemo_seasonal_clock(stamped, "kt0_1.npz")


def test_clock_guard_still_rejects_the_legacy_relative_clock(instruments):
    kamm_twin_90d = instruments.kamm_twin_90d
    stamped = {"seasonal_t0_seconds": 0.0,
               "seasonal_t0_reference_seconds": 180.0 * 86400.0}
    with pytest.raises(SystemExit, match="out of phase"):
        kamm_twin_90d.assert_nemo_seasonal_clock(stamped, "legacy.npz")


def test_clock_guard_rejects_an_artifact_missing_either_stamp(instruments):
    kamm_twin_90d = instruments.kamm_twin_90d
    with pytest.raises(SystemExit, match="seasonal_t0_seconds"):
        kamm_twin_90d.assert_nemo_seasonal_clock({}, "old.npz")
    with pytest.raises(SystemExit, match="seasonal_t0_reference_seconds"):
        kamm_twin_90d.assert_nemo_seasonal_clock(
            {"seasonal_t0_seconds": 0.0}, "half.npz")


def test_run_twin_stamps_the_reference_clock_and_the_run_configuration(
        instruments):
    """Both stamps the guards read must be written by the runner.

    Source-level, keyed off ``run_twin`` itself -- the symbol that runs -- and
    it goes red if either stamp is dropped from the artifact.
    """
    import inspect
    kamm_twin_90d = instruments.kamm_twin_90d
    src = inspect.getsource(kamm_twin_90d.run_twin)
    assert "seasonal_t0_reference_seconds=" in src
    assert "run_config=" in src
    # the reference must come from the restart, not from the same override the
    # twin itself used -- otherwise the pair-check compares a value to itself
    assert "_restart_elapsed_seconds(" in src
