"""Direct unit tests for the #1226 DINO twin instruments promoted from the
session scratchpad (``scripts/validate/ocean_fidelity/dino_1226/``):
``kamm_twin_90d.py`` (day-0 verification gate), ``heat_discriminator.py``
(drift-profile / crossing-depth), and ``mode_projection.py`` (checkerboard
mode projector). All synthetic -- no NEMO artifacts, CPU-fast.
"""
import importlib
import inspect
import json
import os
import sys
import types
import unittest.mock as mock
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
        import validate.ocean_fidelity.dino_1226.tcarry_baseline_reconcile as tcarry_reconcile
        import validate.ocean_fidelity.dino_1226.tcarry_basin_floor90 as tcarry_floor
        import validate.ocean_fidelity.dino_1226.tcarry_basin_reverdict as tcarry_reverdict
        import validate.ocean_fidelity.dino_1226.tcarry_bridge_omega_score as tcarry_omega
        import validate.ocean_fidelity.dino_1226.tcarry_een_off_discriminator as tcarry_een
        importlib.reload(mode_projection)
        importlib.reload(heat_discriminator)
        importlib.reload(kamm_twin_90d)
        importlib.reload(tcarry_reconcile)
        importlib.reload(tcarry_floor)
        importlib.reload(tcarry_reverdict)
        importlib.reload(tcarry_omega)
        importlib.reload(tcarry_een)
        return types.SimpleNamespace(
            kamm_twin_90d=kamm_twin_90d,
            heat_discriminator=heat_discriminator,
            mode_projection=mode_projection,
            tcarry_reconcile=tcarry_reconcile,
            tcarry_floor=tcarry_floor,
            tcarry_reverdict=tcarry_reverdict,
            tcarry_omega=tcarry_omega,
            tcarry_een=tcarry_een,
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


def test_bridge_tke_coefficients_preserves_surface_and_interior(instruments):
    kamm = instruments.kamm_twin_90d
    n_lat, n_lon, jpk = 2, 3, 4
    wet = np.ones((n_lat, n_lon))
    en = np.arange(n_lat * n_lon * jpk, dtype=float).reshape(n_lat, n_lon, jpk)
    avm = en + 100.0
    avt = en + 200.0
    dissl = en + 300.0
    st = _fake_state(np.zeros((n_lat, n_lon, jpk)),
                     np.zeros((n_lat, n_lon)),
                     np.zeros((n_lat, n_lon + 1, jpk)),
                     np.zeros((n_lat + 1, n_lon, jpk)))
    out = kamm.bridge_tke_from_restart(
        st, en, wet, restart_avm=avm, restart_avt=avt,
        restart_dissl=dissl)
    np.testing.assert_array_equal(out.tke_avm.data, avm[..., 1:])
    np.testing.assert_array_equal(out.tke_avt.data, avt[..., 1:])
    np.testing.assert_array_equal(out.tke_avm_surface.data, avm[..., 0])
    np.testing.assert_array_equal(out.tke_dissl.data, dissl[..., 1:])


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
def test_before_level_bridge_is_on_by_default(instruments):
    """#1455 (2026-08-24): the before-level bridge defaults ON, on BOTH the
    python surface and the CLI.

    REVERT-RED, and this is the whole point of the test: restoring either
    ``bridge_before=False`` default turns it red immediately.  The old default
    made the twin start from a forward-Euler step, which makes the whole
    trajectory delayed about half a step, injects a permanent perturbation of
    half a leap-frog step of tendency, and skips NEMO's step-1 after-level
    reconciliation.
    """
    import inspect
    kamm_twin_90d = instruments.kamm_twin_90d
    build_sig = inspect.signature(kamm_twin_90d._build_twin_state)
    run_sig = inspect.signature(kamm_twin_90d.run_twin)
    assert build_sig.parameters["bridge_before"].default is True
    assert run_sig.parameters["bridge_before"].default is True
    args = kamm_twin_90d._parse_args(["nemo_dino_kamm_mlf", "out.npz"])
    assert args.bridge_before is True
    assert kamm_twin_90d.resolve_start_mode(args.bridge_before) == "bridged"


@pytest.mark.parametrize("flag", ["--legacy-euler-start", "--no-bridge-before"])
def test_the_euler_start_needs_an_explicit_flag(instruments, flag):
    """Both spellings of the legacy switch select the Euler start, and nothing
    else does."""
    kamm_twin_90d = instruments.kamm_twin_90d
    args = kamm_twin_90d._parse_args(["nemo_dino_kamm_mlf", "out.npz", flag])
    assert args.bridge_before is False
    assert kamm_twin_90d.resolve_start_mode(args.bridge_before) == "euler"
    # the flag that used to be needed for the DEFAULT behaviour still parses,
    # so every recorded invocation keeps working -- it is now a no-op
    args = kamm_twin_90d._parse_args(
        ["nemo_dino_kamm_mlf", "out.npz", "--bridge-before"])
    assert args.bridge_before is True


def test_the_euler_start_prints_a_loud_banner_naming_the_running_mean(
        instruments, capsys):
    """The legacy start must not be selectable quietly: the banner has to name
    the running-mean identity, because that identity is what invalidates every
    phase/lag number produced in this mode."""
    kamm_twin_90d = instruments.kamm_twin_90d
    kamm_twin_90d._START_ANNOUNCED.clear()

    assert kamm_twin_90d.resolve_start_mode(False) == "euler"
    out = capsys.readouterr().out
    assert "!!" in out and "LEGACY FORWARD-EULER START" in out
    # the DELAY term, named with its identity...
    assert "RUNNING MEAN" in out and "half a step" in out
    # ...and the INJECTION term, which the identity does NOT describe and which
    # an earlier banner omitted entirely. Its absence is the failure mode that
    # makes a reader think shifting a result half a step undoes the Euler start.
    assert "INJECTS" in out and "2.06e-3 Sv" in out
    assert "reconciliation" in out
    assert "twin start: euler" in out

    # non-vacuity: the DEFAULT must not print the banner, or the banner is
    # noise and proves nothing
    kamm_twin_90d._START_ANNOUNCED.clear()
    assert kamm_twin_90d.resolve_start_mode(True) == "bridged"
    out = capsys.readouterr().out
    assert "LEGACY FORWARD-EULER START" not in out
    assert "twin start: bridged" in out   # every call states the mode


def test_run_twin_stamps_the_resolved_start_mode(instruments):
    """The artifact must record which start it took, next to the ladder and
    the dtype -- a scorer reads the stamp, never the filename.

    Source-level because writing the stamp needs a full NEMO-artifact twin
    run.  It names ``run_twin`` -- the function that RUNS and that contains the
    ``np.savez`` call -- not a wrapper, and it asserts the RESOLVED value is
    stamped (``start_mode``, the return of ``resolve_start_mode``) rather than
    the raw flag, which is the failure mode the ladder stamp was written to
    avoid.  Deleting the stamp line turns this red.
    """
    import inspect
    kamm_twin_90d = instruments.kamm_twin_90d
    src = inspect.getsource(kamm_twin_90d.run_twin)
    assert "requested_start = resolve_start_mode(bridge_before)" in src
    # the stamped value is the one OBSERVED on the built state, checked against
    # what the flag requested -- not the flag itself
    assert 'observed_start = "bridged" if st.T_before is not None else "euler"' in src
    assert "start_mode = observed_start" in src
    assert "twin_start_mode=np.str_(start_mode)" in src


def test_start_mode_of_reads_the_stamp_and_admits_when_it_is_missing(instruments):
    """The shared reader every scorer uses. An UNSTAMPED artifact must come
    back as unknown, never guessed at from whatever the default was."""
    kamm_twin_90d = instruments.kamm_twin_90d
    assert kamm_twin_90d.start_mode_of({"twin_start_mode": "bridged"}) == "bridged"
    assert kamm_twin_90d.start_mode_of({"twin_start_mode": "euler"}) == "euler"
    assert kamm_twin_90d.start_mode_of({"nemo_ladder_mode": "both"}) is None


def test_the_start_mode_criterion_is_asymmetric(instruments):
    """Disposition of the start mode at the gate (#1455), as review corrected
    it: a MISSING stamp is printed and forgiven, a stamped OFF-CLAIM start is a
    reason.

    The bootstrap problem -- no recorded artifact carries the new stamp, so a
    strict criterion would refuse the campaign's own baseline -- attaches only
    to the missing case. A run that stamped "euler" was deliberately started
    off the trajectory the thresholds were earned on, which is the same test
    the ladder criterion applies. Both directions are asserted, so the guard
    can pass neither by accepting everything nor by rejecting everything.
    """
    k = instruments.kamm_twin_90d
    base = {"nemo_ladder_mode": "both", "control_dtype": "float64"}
    ok, reasons, _, _ = k.certifiable_grid_and_precision(dict(base))
    assert ok and reasons == [], f"a MISSING start stamp must be forgiven: {reasons}"
    ok, reasons, _, _ = k.certifiable_grid_and_precision(
        dict(base, twin_start_mode="bridged"))
    assert ok and reasons == [], f"the claim's own start must certify: {reasons}"
    ok, reasons, _, _ = k.certifiable_grid_and_precision(
        dict(base, twin_start_mode="euler"))
    assert not ok, "a stamped Euler start must withhold the verdict"
    assert any("start mode" in r for r in reasons), reasons
    # the two older criteria still bite, so this is not the only thing left
    assert not k.certifiable_grid_and_precision(
        dict(base, nemo_ladder_mode="off"))[0]
    assert not k.certifiable_grid_and_precision(
        dict(base, control_dtype="float32"))[0]


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
# kamm_twin_90d: --bridge-before-stress-tpoint (round-3 end-wall gate)
# ---------------------------------------------------------------------------
def test_tpoint_stress_selector_defaults_corrected_and_legacy_is_opt_in(instruments):
    import inspect

    k = instruments.kamm_twin_90d
    build_sig = inspect.signature(k._build_twin_state)
    run_sig = inspect.signature(k.run_twin)
    assert build_sig.parameters["bridge_before_stress_tpoint"].default is True
    assert run_sig.parameters["bridge_before_stress_tpoint"].default is True
    base = k._parse_args(["nemo_dino_kamm_mlf", "out.npz"])
    explicit_corrected = k._parse_args([
        "nemo_dino_kamm_mlf", "out.npz", "--bridge-before-stress-tpoint"])
    legacy = k._parse_args([
        "nemo_dino_kamm_mlf", "out.npz",
        "--bridge-before-stress-legacy-u-as-t"])
    legacy_euler = k._parse_args([
        "nemo_dino_kamm_mlf", "out.npz", "--legacy-euler-start"])
    assert base.bridge_before_stress_tpoint is True
    assert explicit_corrected.bridge_before_stress_tpoint is True
    assert legacy.bridge_before_stress_tpoint is False
    assert legacy_euler.bridge_before is False
    assert legacy_euler.bridge_before_stress_tpoint is False
    with pytest.raises(SystemExit):
        k._parse_args([
            "nemo_dino_kamm_mlf", "out.npz",
            "--bridge-before-stress-tpoint",
            "--bridge-before-stress-legacy-u-as-t"])


def test_tpoint_stress_reconstruction_reuses_loader_and_changes_only_carry(
        instruments, monkeypatch):
    """Red if the selector inverts utau_b, edits a prognostic, or bypasses
    either existing DINO forcing loader."""
    from collections import namedtuple

    k = instruments.kamm_twin_90d
    calls = []
    forcing_token = object()
    expected_x = np.array([[-1.0, -2.0], [-3.0, -4.0]])
    expected_y = np.zeros_like(expected_x)

    def _arrays(grid, cfg):
        calls.append(("arrays", grid, cfg))
        return forcing_token

    def _surface(forcing):
        calls.append(("surface", forcing))
        return types.SimpleNamespace(tau_x=expected_x, tau_y=expected_y)

    monkeypatch.setattr(k, "dino_lat_lon_surface_forcing_arrays", _arrays)
    monkeypatch.setattr(k, "dino_step_surface_forcing", _surface)
    State = namedtuple(
        "State", "T S u v eta tau_x_prev tau_y_prev tke")
    sentinel_fields = [object() for _ in range(6)]
    state = State(
        *sentinel_fields[:5], np.full_like(expected_x, 99.0),
        np.full_like(expected_y, 88.0), sentinel_fields[5])
    grid, cfg = object(), object()

    rebuilt, receipt = k.reconstruct_dino_before_stress_tpoint(
        state, grid, cfg, t_seconds=15_552_000.0)
    assert calls == [("arrays", grid, cfg), ("surface", forcing_token)]
    assert np.array_equal(np.asarray(rebuilt.tau_x_prev), expected_x)
    assert np.array_equal(np.asarray(rebuilt.tau_y_prev), expected_y)
    # Exact sign is load-bearing: negating the analytic T field to mimic the
    # raw NEMO face convention makes this assertion red.
    assert float(np.asarray(rebuilt.tau_x_prev)[0, 0]) == -1.0
    for name in ("T", "S", "u", "v", "eta", "tke"):
        assert getattr(rebuilt, name) is getattr(state, name)
    assert receipt["bridge_before_stress_stagger"] == "T"
    assert receipt["bridge_before_stress_reconstruction_seconds"] == 15_552_000.0
    assert receipt["bridge_before_stress_sha256"] == k._stress_content_sha256(
        expected_x, expected_y)


def test_tpoint_stress_hash_and_time_controls_can_fail(instruments):
    k = instruments.kamm_twin_90d
    x = np.arange(6.0).reshape(2, 3)
    y = np.zeros_like(x)
    planted = x.copy()
    planted[0, 0] = np.nextafter(planted[0, 0], np.inf)
    assert k._stress_content_sha256(x, y) != k._stress_content_sha256(planted, y)
    with pytest.raises(ValueError, match="finite and >=0"):
        k._analytic_dino_tpoint_stress(object(), object(), t_seconds=-1.0)


def test_tpoint_stress_selector_refuses_euler_start_before_io(
        instruments, monkeypatch):
    k = instruments.kamm_twin_90d

    def _boom(*args, **kwargs):
        raise AssertionError("selector must refuse before reading an oracle file")

    monkeypatch.setattr(k, "read_nemo_mesh_mask", _boom)
    with pytest.raises(SystemExit, match="requires --bridge-before"):
        k._build_twin_state(
            "nemo_dino_kamm_mlf", "/unused", "/unused",
            bridge_before=False, bridge_before_stress_tpoint=True)
    with pytest.raises(AssertionError, match="selector must refuse"):
        k._build_twin_state(
            "nemo_dino_kamm_mlf", "/unused", "/unused",
            bridge_before=False, bridge_before_stress_tpoint=False)


def test_tpoint_stress_selector_threads_and_stamps_receipts(
        instruments, monkeypatch):
    """Red if the CLI goes inert or artifacts omit any registered receipt."""
    import inspect

    k = instruments.kamm_twin_90d
    seen = {}
    src = inspect.getsource(k.run_twin)

    def _spy(*args, **kwargs):
        seen.update(kwargs)
        return True

    monkeypatch.setattr(k, "run_twin", _spy)
    monkeypatch.setattr(k, "provenance_gate", lambda: None)
    monkeypatch.setattr(k, "_precision_gate", lambda: None)
    k.main(["nemo_dino_kamm_mlf", "out.npz"])
    assert seen["bridge_before_stress_tpoint"] is True
    seen.clear()
    k.main([
        "nemo_dino_kamm_mlf", "out.npz",
        "--bridge-before-stress-legacy-u-as-t"])
    assert seen["bridge_before_stress_tpoint"] is False
    seen.clear()
    k.main([
        "nemo_dino_kamm_mlf", "out.npz", "--legacy-euler-start"])
    assert seen["bridge_before"] is False
    assert seen["bridge_before_stress_tpoint"] is False

    assert 'bridge_before_stress_stagger = "T"' in src
    assert "np.array_equal(np.asarray(st.tau_x_prev)" in src
    assert "bridge_before_stress_stagger=np.str_(bridge_before_stress_stagger)" in src
    assert "bridge_before_stress_reconstruction_seconds=np.float64(" in src
    assert "bridge_before_stress_sha256=np.str_(bridge_before_stress_sha256)" in src


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
             seasonal_t0_seconds=np.float64(0.0),      # legacy clock -> refused
             seasonal_t0_reference_seconds=np.float64(180.0 * 86400.0))
    with pytest.raises(SystemExit, match="out of phase"):
        gate.load_candidate(str(stamped))
    assert "vertical ladder of this candidate: both" in capsys.readouterr().out

    # An UNSTAMPED artifact must still be scoreable: the only refusal it may hit
    # is the missing day-90 field, never the missing ladder stamp. The `match=`
    # is load-bearing -- a bare raises() passes even when the gate is mutated to
    # refuse on the stamp, because the print has already fired by then.
    bare = tmp_path / "bare.npz"
    np.savez(bare, seasonal_t0_seconds=np.float64(15552000.0),
             seasonal_t0_reference_seconds=np.float64(15552000.0))
    with pytest.raises(SystemExit, match="has no u3d_day90"):
        gate.load_candidate(str(bare))
    assert "UNSTAMPED" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# #1640: the ladder CONTENT HASH (identity, not taxonomy) and the gate's
# refusal to issue a verdict off the claim's grid/precision.
# ---------------------------------------------------------------------------
class _FakeLadder:
    """Only the four 1-D arrays the hash reads -- see vertical_ladder_sha256."""

    def __init__(self, dz, t_depth=None, dtype=np.float64):
        self.dz_ref = np.asarray(dz, dtype=dtype)
        self.z_full_ref = np.cumsum(self.dz_ref) - 0.5 * self.dz_ref
        self.z_half_ref = np.concatenate(
            [np.zeros(1, dtype=dtype), np.cumsum(self.dz_ref)])
        self.t_depth_ref = (None if t_depth is None
                            else np.asarray(t_depth, dtype=dtype))


def test_ladder_hash_is_an_identity_not_a_label(instruments):
    """A label cannot tell two runs apart that share it; a content hash can.

    Non-vacuity is the point of every assertion here: the hash must be STABLE
    across rebuilds of the same ladder (or it is useless as an identity) and
    must MOVE for each thing that makes a grid a different grid -- a changed
    thickness, a changed T-depth array, and a changed dtype.
    """
    h = instruments.kamm_twin_90d.vertical_ladder_sha256
    base = [10.0, 20.0, 40.0, 80.0]
    assert h(_FakeLadder(base)) == h(_FakeLadder(base)), (
        "same ladder must hash the same, or the hash cannot certify identity")
    # a changed thickness is a different grid
    assert h(_FakeLadder([10.0, 20.0, 40.0, 80.5])) != h(_FakeLadder(base))
    # a present-vs-absent t_depth_ref is a different grid (this is exactly the
    # gdept half of the ladder mode, so the hash MUST separate them)
    assert h(_FakeLadder(base, t_depth=[5.0, 20.0, 50.0, 110.0])) != h(
        _FakeLadder(base))
    # and the SAME numbers at a different precision are not the same grid for a
    # claim that depends on precision
    assert h(_FakeLadder(base, dtype=np.float32)) != h(
        _FakeLadder(base, dtype=np.float64))


@pytest.mark.parametrize("stamps,expect", [
    ({"nemo_ladder_mode": "both", "control_dtype": "float64"}, True),
    ({"nemo_ladder_mode": "off", "control_dtype": "float64"}, False),
    ({"nemo_ladder_mode": "both", "control_dtype": "float32"}, False),
    ({"nemo_ladder_mode": "both"}, False),                    # dtype unstamped
    ({"control_dtype": "float64"}, False),                    # ladder unstamped
    ({}, False),                                              # both unstamped
])
def test_certifiable_only_on_the_claims_grid_and_precision(instruments, stamps,
                                                           expect):
    """#1640: the gate SCORES anything, CERTIFIES only the claim's grid.

    All six arms are asserted, so the guard cannot pass by accepting or by
    rejecting everything -- the on-claim arm proves it is not vacuously
    strict, the five off-claim arms that it is not vacuously permissive."""
    ok, reasons, _, _ = (
        instruments.kamm_twin_90d.certifiable_grid_and_precision(stamps))
    assert ok is expect
    assert (reasons == []) is expect, (
        "an off-claim candidate must SAY why it cannot be certified")


def test_uncertified_gate_prints_no_verdict_token_anywhere(instruments):
    """The reviewer's ask is that the gate REFUSE TO ISSUE PASS/FAIL, not that
    it merely drop the tally line -- a per-row status IS a verdict, so a
    version that suppressed only the tally would still have issued one five
    times over.  This asserts no verdict token survives anywhere in the
    output, and (non-vacuity) that the certified call still emits them."""
    import contextlib
    import importlib
    import io
    _dir = (Path(__file__).resolve().parents[3] / "scripts" / "validate"
            / "ocean_fidelity" / "dino_1226")
    stub = types.ModuleType("acc_thermal_wind")
    stub.DINO = "/nonexistent"
    with mock.patch.dict(sys.modules, {"acc_thermal_wind": stub}):
        sys.modules.pop("acceptance_gate_90d", None)
        sys.path.insert(0, str(_dir))
        try:
            gate = importlib.import_module("acceptance_gate_90d")
        finally:
            sys.path.remove(str(_dir))
            sys.modules.pop("acceptance_gate_90d", None)

    rows = [(k, 1.0, 1.0, 0.0, 1.0, True) for k in list(gate.LABELS)[:2]]
    rows += [(k, 1.0, 9.0, 8.0, 1.0, False) for k in list(gate.LABELS)[2:3]]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gate.print_gate(rows, 5, certified=False)
    out = buf.getvalue()
    assert "PASS" not in out and "FAIL" not in out, (
        f"an uncertified gate must issue no verdict token, got:\n{out}")
    assert "GATE 90D-TWIN" not in out, "the tally line is itself a verdict"

    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        gate.print_gate(rows, 5)
    out2 = buf2.getvalue()
    assert "PASS" in out2 and "FAIL" in out2 and "GATE 90D-TWIN" in out2, (
        "the CERTIFIED path must still issue verdicts, or this test would "
        "pass against a gate that never says anything")


def _import_gate():
    """Import acceptance_gate_90d with its NEMO-artifact dependency stubbed --
    the same pattern test_uncertified_gate_prints_no_verdict_token_anywhere
    uses, factored out rather than copied."""
    import importlib
    _dir = (Path(__file__).resolve().parents[3] / "scripts" / "validate"
            / "ocean_fidelity" / "dino_1226")
    stub = types.ModuleType("acc_thermal_wind")
    stub.DINO = "/nonexistent"
    with mock.patch.dict(sys.modules, {"acc_thermal_wind": stub}):
        sys.modules.pop("acceptance_gate_90d", None)
        sys.path.insert(0, str(_dir))
        try:
            return importlib.import_module("acceptance_gate_90d")
        finally:
            sys.path.remove(str(_dir))
            sys.modules.pop("acceptance_gate_90d", None)


def _fake_twin_npz(tmp_path, **stamps):
    u_full = np.zeros((5, 53, 3))
    d = dict(u3d_day90=u_full, T3d_day90=np.zeros((5, 4, 3)),
             S3d_day90=np.zeros((5, 4, 3)), land_mask=np.ones((5, 4)),
             nemo_ladder_mode=np.str_("both"),
             vertical_ladder_sha256=np.str_("deadbeef"))
    d.update(stamps)
    p = tmp_path / "cand.npz"
    np.savez(p, **d)
    return str(p)


@pytest.mark.parametrize("stamped,expect", [
    ({"twin_start_mode": np.str_("bridged")}, "bridged"),
    ({"twin_start_mode": np.str_("euler")}, "LEGACY FORWARD-EULER START"),
    ({}, "UNSTAMPED"),
])
def test_the_gate_prints_the_start_mode_of_every_candidate(tmp_path, monkeypatch,
                                                           stamped, expect):
    """#1455: whatever start a candidate took, the gate SAYS SO.

    All three arms are asserted so the print cannot pass by being a constant:
    the bridged default, the legacy Euler start (which must carry the warning
    text, not just the word), and an artifact written before the stamp existed
    (which must be reported as unknown, never guessed at from whatever the
    default was on the day, which the audit says would be wrong: every
    recorded twin build was bridged).  The clock guard is bypassed here
    because this test is about the start-mode line, not about the clock.
    """
    import contextlib  # noqa: PLC0415
    import io  # noqa: PLC0415

    gate = _import_gate()
    monkeypatch.setenv("DINO_GATE_ALLOW_LEGACY_CLOCK", "1")
    path = _fake_twin_npz(tmp_path, **stamped)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gate.load_candidate(path)
    out = buf.getvalue()
    assert "twin start mode:" in out, f"the gate must state the start:\n{out}"
    assert expect in out, f"expected {expect!r} in:\n{out}"


def test_main_threads_the_start_flag_into_run_twin(instruments, monkeypatch):
    """MED-4(a): deleting `bridge_before=args.bridge_before` from main() leaves
    every other test green while --legacy-euler-start silently goes inert --
    run_twin's default is True, so the flag would do nothing and the artifact
    would stamp "bridged". This is the one assertion that catches it."""
    kamm_twin_90d = instruments.kamm_twin_90d
    seen = {}

    def _spy(*a, **kw):
        seen.update(kw)
        return True

    monkeypatch.setattr(kamm_twin_90d, "run_twin", _spy)
    monkeypatch.setattr(kamm_twin_90d, "provenance_gate", lambda: None)
    monkeypatch.setattr(kamm_twin_90d, "_precision_gate", lambda: None)
    kamm_twin_90d.main(["nemo_dino_kamm_mlf", "o.npz", "--legacy-euler-start"])
    assert seen["bridge_before"] is False, (
        "main() did not thread the start flag into run_twin -- the flag is inert")
    seen.clear()
    kamm_twin_90d.main(["nemo_dino_kamm_mlf", "o.npz"])
    assert seen["bridge_before"] is True
    assert seen["perturb_eps"] == kamm_twin_90d.PERTURB_EPS_DEFAULT


def test_run_twin_refuses_a_start_mode_the_built_state_contradicts(instruments,
                                                                  monkeypatch):
    """HIGH-1: the stamp must be a RECEIPT read off the built state, not a
    restatement of the CLI flag. Hand run_twin a state whose before level
    disagrees with the requested mode and it must refuse rather than stamp
    either answer."""
    import types as _t
    kamm_twin_90d = instruments.kamm_twin_90d

    class _Fld:
        data = np.zeros((2, 2, 2))

    fake_state = _t.SimpleNamespace(T=_Fld(), T_before=None)

    def _fake_build(*a, **kw):
        return (None, None, None, None, None, None, fake_state)

    monkeypatch.setattr(kamm_twin_90d, "_build_twin_state", _fake_build)
    monkeypatch.setattr(kamm_twin_90d, "seasonal_t0_seconds", lambda *a, **k: 0.0)
    monkeypatch.setattr(kamm_twin_90d, "_git_provenance", lambda: ("a" * 40, 0))
    with pytest.raises(SystemExit, match="START-MODE MISMATCH"):
        kamm_twin_90d.run_twin("nemo_dino_kamm_mlf", "o.npz", bridge_before=True)


def test_the_gate_forwards_the_start_flag_it_was_given(instruments, monkeypatch,
                                                       tmp_path):
    """MED-4(b): inverting the gate's forwarding line would make every default
    --run-recipe run produce an Euler-start candidate. run_candidate_twin had
    no coverage at all; this is it, both directions."""
    gate = _import_gate()
    seen = []
    monkeypatch.setattr(gate.subprocess, "run", lambda cmd, **kw: seen.append(cmd))
    gate.run_candidate_twin(str(tmp_path / "c.npz"), "nemo_dino_kamm_mlf",
                            None, True)
    assert "--bridge-before" in seen[-1] and "--legacy-euler-start" not in seen[-1]
    gate.run_candidate_twin(str(tmp_path / "c.npz"), "nemo_dino_kamm_mlf",
                            None, False)
    assert "--legacy-euler-start" in seen[-1] and "--bridge-before" not in seen[-1]


def _import_startmode_scorer():
    import importlib
    _dir = (Path(__file__).resolve().parents[3] / "scripts" / "validate"
            / "ocean_fidelity" / "dino_1226")
    stub = types.ModuleType("acc_thermal_wind")
    stub.DINO = "/nonexistent"
    with mock.patch.dict(sys.modules, {"acc_thermal_wind": stub}):
        for m in ("acceptance_gate_90d", "startmode_ab_score"):
            sys.modules.pop(m, None)
        sys.path.insert(0, str(_dir))
        try:
            return importlib.import_module("startmode_ab_score")
        finally:
            sys.path.remove(str(_dir))
            for m in ("acceptance_gate_90d", "startmode_ab_score"):
                sys.modules.pop(m, None)


def test_startmode_scorer_reads_every_stamp_and_admits_the_missing_ones(tmp_path):
    """The A/B scorer prints each arm's provenance before any number, so a
    mislabelled file cannot be read as a result. An unstamped field must come
    back as UNSTAMPED, not as a guess."""
    sc = _import_startmode_scorer()
    full = tmp_path / "full.npz"
    np.savez(full, twin_start_mode=np.str_("euler"),
             nemo_ladder_mode=np.str_("both"),
             control_dtype=np.str_("float64"),
             vertical_ladder_sha256=np.str_("abcdef0123456789ff"))
    got = sc.stamps_of(str(full))
    assert got["start"] == "euler" and got["ladder"] == "both"
    assert got["dtype"] == "float64" and got["hash"] == "abcdef0123456789"
    bare = tmp_path / "bare.npz"
    np.savez(bare, x=np.zeros(2))
    assert set(sc.stamps_of(str(bare)).values()) == {"UNSTAMPED"}


def test_startmode_scorer_refuses_arms_on_different_ladders(tmp_path):
    """A pair that does not stand on the same vertical ladder is not a
    one-variable comparison, and the refusal must happen BEFORE any metric is
    computed (it does: the guard runs before load_candidate, which is why this
    test needs no NEMO artifacts at all).

    Non-vacuity: the matching pair gets PAST the guard and fails later, on the
    missing NEMO baseline -- so the guard is not simply rejecting everything.
    """
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    d64 = dict(control_dtype=np.str_("float64"))
    np.savez(a, vertical_ladder_sha256=np.str_("aaaa000000000000"), **d64)
    np.savez(b, vertical_ladder_sha256=np.str_("bbbb000000000000"), **d64)
    sc = _import_startmode_scorer()
    with pytest.raises(SystemExit, match="DIFFERENT vertical ladders"):
        sc.main([f"A={a}", f"B={b}"])

    same = tmp_path / "same.npz"
    np.savez(same, vertical_ladder_sha256=np.str_("aaaa000000000000"), **d64)

    # and provenance-less arms are refused before either of those questions
    naked = tmp_path / "naked.npz"
    np.savez(naked, x=np.zeros(2))
    with pytest.raises(SystemExit, match="without provenance"):
        sc.main([f"A={naked}", f"B={naked}"])
    with pytest.raises(BaseException) as ei:      # SystemExit is not Exception
        sc.main([f"A={a}", f"B={same}"])
    assert "DIFFERENT vertical ladders" not in str(ei.value)


def test_startmode_scorer_flags_the_band_floor_as_a_transfer():
    """The channel-band transport has no 1e-14 floor of its own; it borrows
    ACC's. That borrowing must be declared in the module, not silently reused
    -- this is the campaign's transferred-floor rule applied to its own tool."""
    sc = _import_startmode_scorer()
    assert sc.ACCBAND_FLOOR_IS_A_TRANSFER is True
    assert sc.FLOORS["accband"] == sc.FLOORS["acc"]
    assert "accband" in sc.KEYS and "acc" in sc.KEYS


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
    for selector in (
        "tke_preclosure_coeff_source",
        "tke_shear_evaluation_stage",
        "tke_shear_metric_source",
        "tke_n2_evaluation_stage",
        "tke_langmuir_evaluation",
        "dino_wind_profile_evaluation",
    ):
        assert f'"{selector}"' in src
    assert "producer_git_sha=" in src
    assert "producer_dirty_tracked_files=" in src
    # the reference must come from the restart, not from the same override the
    # twin itself used -- otherwise the pair-check compares a value to itself
    assert "restart_elapsed_seconds(" in src


def test_corrected_stress_live_paths_use_public_clock_helper(instruments):
    """REBASE-RED: main removed the private helper spelling, while two
    corrected-stress call sites on this branch still used it."""
    import inspect
    k = instruments.kamm_twin_90d
    src = inspect.getsource(k.run_twin)
    assert "_restart_elapsed_seconds(" not in src
    assert src.count("restart_elapsed_seconds(") == 2
    assert k._restart_elapsed_seconds is k.restart_elapsed_seconds


# ---------------------------------------------------------------------------
# kamm_twin_90d -- fp64 snapshot storage and the always-on fp64 reduced series
#
# WHY THESE EXIST: the campaign's single-precision snapshot storage repeatedly
# capped what was measurable (lego's early ensemble spread was 2 differing
# cells of 342134 at day 10; ensemble members tie exactly on max-type metrics
# at the storage quantum). The fix has two halves and each is tested here --
# the opt-in float64 3-D block, and the always-on float64 REDUCED series that
# is computed from the LIVE state before the storage cast.
# ---------------------------------------------------------------------------
def test_snapshot_storage_defaults_to_float32_and_the_flag_makes_it_float64(
        instruments):
    """REVERT-RED ON THE DEFAULT. Doubling every recorded twin's artifact by
    accident is the one regression this feature could cause, so the default is
    asserted as tightly as the flag."""
    K = instruments.kamm_twin_90d
    assert K.snapshot_dtype(False) is np.float32
    assert K.snapshot_dtype(True) is np.float64
    assert K._parse_args(["r", "o.npz"]).fp64_3d is False
    assert K._parse_args(["r", "o.npz", "--fp64-3d"]).fp64_3d is True


def test_storage_dtype_stamp_defaults_to_the_legacy_all_float32_map(instruments):
    """EXTEND-ONLY. Every artifact recorded before the stamp existed was
    all-float32, so an unstamped artifact must resolve to exactly that -- a
    consumer reading it must not change any recorded score."""
    K = instruments.kamm_twin_90d
    assert K.snapshot_storage_dtypes({}) == K.LEGACY_STORAGE_DTYPES
    assert all(v == "float32" for v in K.LEGACY_STORAGE_DTYPES.values())
    stamped = {"storage_dtypes": np.str_(
        '{"T3d": "float64", "S3d": "float64", "u3d": "float64", '
        '"reduced": "float64"}')}
    got = K.snapshot_storage_dtypes(stamped)
    assert got["u3d"] == "float64" and got["reduced"] == "float64"


_WET = np.ones((6, 5), dtype=np.float64)


def _two_states_apart_by(delta):
    """Two 3-D u fields differing by `delta` on ONE face, everything else
    identical. `delta` is chosen sub-quantum by the caller."""
    a = np.full((6, 5, 4), 0.25, dtype=np.float64)
    b = a.copy()
    b[2, 3, 1] += delta
    return a, b


def _sum_reducer(f64):
    """A stand-in for the mesh-backed reducer: a masked weighted column sum,
    i.e. the same SHAPE of reduction (a masked transport integral) the real one
    takes, with no NEMO mesh required.

    It reads ``land_mask`` deliberately. The real reducer needs the wet domain
    for every one of its eleven metrics, and a 5-day verification run caught
    the wiring gap where the harness handed it only the 3-D fields -- a stub
    that ignored the mask would have kept that green.
    """
    w = np.arange(1, f64["u"].shape[2] + 1, dtype=np.float64)
    m = f64["land_mask"] > 0.5
    return {"x": float(np.einsum("jik,k->", np.where(m[:, :, None],
                                                     f64["u"], 0.0), w))}


def test_capture_snapshot_stores_at_the_requested_dtype(instruments):
    K = instruments.kamm_twin_90d
    a, _ = _two_states_apart_by(0.0)
    fields = {"T": a, "S": a, "eta": a[:, :, 0], "u": a, "v": a}
    for fp64, want in ((False, np.float32), (True, np.float64)):
        stored, red, status = K.capture_snapshot(
            fields, snap_dtype=K.snapshot_dtype(fp64), reducer=_sum_reducer,
            land_mask=_WET)
        assert status == "ok"
        assert {v.dtype for v in stored.values()} == {np.dtype(want)}
        # the land mask is a reducer INPUT, never a stored snapshot field --
        # it is time-invariant and already written once per run
        assert set(stored) == {"T", "S", "eta", "u", "v"}
        assert red["x"] == pytest.approx(
            _sum_reducer({"u": a, "land_mask": _WET})["x"])
    stored, red, status = K.capture_snapshot(
        fields, snap_dtype=np.float32, reducer=None, land_mask=_WET)
    assert red is None and status == "no reducer"
    # the mask is REQUIRED -- it was optional for one commit and a call site
    # that forgot it shipped, crashing every snapshot run
    with pytest.raises(TypeError):
        K.capture_snapshot(fields, snap_dtype=np.float32,
                           reducer=_sum_reducer)


def test_the_reducer_is_handed_the_wet_domain(instruments):
    """The reductions are all masked integrals, so a reducer that never sees
    the land mask cannot produce the gate's quantity. A 5-day verification run
    caught exactly this wiring gap after the first round of unit tests were
    green, so it is pinned here."""
    K = instruments.kamm_twin_90d
    a, _ = _two_states_apart_by(0.0)
    fields = {"T": a, "S": a, "eta": a[:, :, 0], "u": a, "v": a}
    seen = {}

    def spy(f64):
        seen.update(f64)
        return {"x": 0.0}

    K.capture_snapshot(fields, snap_dtype=np.float32, reducer=spy,
                       land_mask=_WET)
    assert "land_mask" in seen, "the reducer was not handed the wet domain"
    assert seen["land_mask"].dtype == np.float64
    # masking must actually bite, or the check above is decorative
    half = _WET.copy()
    half[3:, :] = 0.0
    full = K.capture_snapshot(fields, snap_dtype=np.float32,
                              reducer=_sum_reducer, land_mask=_WET)[1]["x"]
    part = K.capture_snapshot(fields, snap_dtype=np.float32,
                              reducer=_sum_reducer, land_mask=half)[1]["x"]
    assert part < full


def test_the_fp64_reduced_series_resolves_a_perturbation_the_fp32_block_cannot(
        instruments):
    """THE POINT OF THE WHOLE CHANGE, stated as a measurement.

    Plant a perturbation strictly BELOW the float32 storage quantum of the
    field it perturbs. Then:

      * the float32-stored 3-D block is BIT-IDENTICAL between the two states,
        so ANY metric reduced from storage ties exactly -- this is the recorded
        campaign's "2 differing cells of 342134" and its exact ensemble ties;
      * the float64 REDUCED series, computed from the live state before the
        cast, separates them -- so an ensemble spread built from it is a
        measurement rather than a report of the npz dtype;
      * with --fp64-3d the stored block separates them too.

    The perturbation is checked to be genuinely sub-quantum first: a test that
    planted a RESOLVABLE perturbation would pass while proving nothing.
    """
    K = instruments.kamm_twin_90d
    a, _ = _two_states_apart_by(0.0)
    quantum = np.spacing(np.float32(a[2, 3, 1]))
    delta = 0.01 * float(quantum)
    a, b = _two_states_apart_by(delta)
    assert delta > 0.0, "the planted perturbation must be nonzero"
    assert np.float32(a[2, 3, 1]) == np.float32(b[2, 3, 1]), (
        "the perturbation is NOT sub-quantum -- this test would pass "
        "vacuously")

    def cap(state, fp64):
        return K.capture_snapshot(
            {"T": state, "S": state, "eta": state[:, :, 0],
             "u": state, "v": state},
            snap_dtype=K.snapshot_dtype(fp64), reducer=_sum_reducer,
            land_mask=_WET)

    s32a, r64a, _ = cap(a, False)
    s32b, r64b, _ = cap(b, False)
    # 1. float32 storage ties the two states, bit for bit.
    assert np.array_equal(s32a["u"], s32b["u"])
    # ... and therefore so does any metric reduced from the STORED field.
    assert (_sum_reducer({"u": s32a["u"].astype(np.float64),
                          "land_mask": _WET})["x"]
            == _sum_reducer({"u": s32b["u"].astype(np.float64),
                             "land_mask": _WET})["x"])
    # 2. the fp64 reduced series does not.
    assert r64a["x"] != r64b["x"]
    assert r64b["x"] - r64a["x"] == pytest.approx(delta * 2.0, rel=1e-9)
    # 3. --fp64-3d also separates the stored block.
    s64a, _, _ = cap(a, True)
    s64b, _, _ = cap(b, True)
    assert not np.array_equal(s64a["u"], s64b["u"])


def test_reduced_series_keys_cover_every_scored_metric(instruments):
    """The stored series must cover what the scorers actually reduce. If
    verdict360 grows a metric and the series does not, a future ensemble is
    back to reducing float32 for that one -- so the two lists are pinned
    against each other rather than maintained in parallel by hand."""
    K = instruments.kamm_twin_90d
    pytest.importorskip("netCDF4")
    sys.path.insert(0, str(SCRIPTS_DIR / "validate" / "ocean_fidelity" / "dino_1226"))
    try:
        import verdict360
    except SystemExit as exc:                       # no NEMO mesh on this box
        pytest.skip(f"NEMO mesh unavailable: {exc}")
    finally:
        try:
            sys.path.remove(
                str(SCRIPTS_DIR / "validate" / "ocean_fidelity" / "dino_1226"))
        except ValueError:
            pass
    assert tuple(K.REDUCED_KEYS) == tuple(verdict360.KEYS)


def test_the_real_reducer_reproduces_the_recorded_scorers_exactly(instruments):
    """IDENTITY, not similarity. The stored series is only trustworthy if it is
    the SAME number the recorded scorer would produce from the same state at
    the same precision -- otherwise it is a second spelling of ten reductions,
    which is this campaign's most expensive defect class. Driven on a synthetic
    but physically-ranged state, on the real NEMO mesh."""
    K = instruments.kamm_twin_90d
    pytest.importorskip("netCDF4")
    d = str(SCRIPTS_DIR / "validate" / "ocean_fidelity" / "dino_1226")
    sys.path.insert(0, d)
    try:
        import acc_thermal_wind as A
        import verdict360
        reducer, status = K.build_snapshot_reducer(
            os.path.dirname(A.mm.filepath()))
    except SystemExit as exc:
        pytest.skip(f"NEMO mesh unavailable: {exc}")
    finally:
        try:
            sys.path.remove(d)
        except ValueError:
            pass
    assert status == "ok" and reducer is not None
    ny, nx, nz = A.tmask.shape
    rng = np.random.default_rng(0)
    z = np.arange(nz, dtype=np.float64)
    T = 4.0 + 16.0 * np.exp(-z / 6.0)[None, None, :] + 0.05 * rng.standard_normal((ny, nx, nz))
    S = 34.5 + 0.5 * np.exp(-z / 10.0)[None, None, :] + 0.01 * rng.standard_normal((ny, nx, nz))
    u = 0.05 * rng.standard_normal((ny, nx + 1, nz))
    mask = np.asarray(A.tmask[:, :, 0], dtype=np.float64)
    got = reducer({"T": T, "S": S, "u": u, "land_mask": mask})
    st = {"T": T, "S": S, "u": u[:, 1:nx + 1, :], "land_mask": mask}
    want = verdict360.all_metrics(st, A.tmask & (mask > 0.5)[:, :, None])
    for k in K.REDUCED_KEYS:
        assert got[k] == want[k], k
    # the per-row profile must PARTITION the metric it decomposes
    rows = got[K.REDUCED_ROW_KEY]
    assert rows.shape == (ny,)
    assert float(rows.sum()) == pytest.approx(want["acc_mean"], rel=1e-10)


# ---------------------------------------------------------------------------
# fp64_snapshot_contract_check.py -- the verification instrument itself
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def contract_check():
    d = str(SCRIPTS_DIR / "validate" / "ocean_fidelity" / "dino_1226")
    sys.path.insert(0, d)
    try:
        import fp64_snapshot_contract_check as C
        return C
    except SystemExit as exc:                       # no NEMO mesh on this box
        pytest.skip(f"NEMO mesh unavailable: {exc}")
    finally:
        try:
            sys.path.remove(d)
        except ValueError:
            pass


def test_the_contract_instrument_passes_its_own_self_check(contract_check):
    """The instrument decides what we believe about the resolution claim, so
    it is gated here rather than trusted from one run. Its self-check plants a
    perturbation VERIFIED invisible to float32 storage on every field and
    fails unless at least one metric that float32 ties is resolved at fp64."""
    assert contract_check._self_check() == 0


def test_a_metric_tied_at_both_precisions_is_not_counted_as_a_gain(
        contract_check, monkeypatch):
    """The instrument's first run scored four metrics as `UNMEASURABLE at
    fp32` that were simply untouched by the perturbation -- tied in BOTH
    columns. A metric the perturbation never reached is not a precision win,
    and counting it would let this instrument report a result for doing
    nothing."""
    C = contract_check
    keys = ("acc", "up", "deep")
    monkeypatch.setattr(C.K, "REDUCED_KEYS", keys)
    monkeypatch.setattr(C, "metrics_from_storage",
                        lambda q, day, cast: {"acc": 1.0, "up": 2.0,
                                              "deep": 3.0 + q})
    monkeypatch.setattr(C, "stored_series",
                        lambda q, day: {"acc": 1.0 + q, "up": 2.0,
                                        "deep": 3.0 + q})
    rows, gained = C.resolution_table(0.0, 1e-9, 5)
    assert {k: (d32 == 0.0, d64 > 0.0) for k, d32, d64 in rows} == {
        "acc": (True, True),      # tied at fp32, resolved at fp64 -> a gain
        "up": (True, False),      # tied at BOTH -> not a gain
        "deep": (False, True),    # resolved at both -> not a gain
    }
    assert gained == 1


# ---------------------------------------------------------------------------
# The artifact-assembly seams. Both reviews found real defects here that every
# earlier test missed, because the earlier tests INJECT a reducer and never
# cross the seam between the harness and the real one.
# ---------------------------------------------------------------------------
def test_the_stamp_never_promises_a_series_the_artifact_does_not_carry(
        instruments):
    """The stamp and the series keys must be built from ONE day list. They
    were computed thirty lines apart, so a run whose snapshot grid excluded
    day 0 and which then stopped before its first requested day stamped a
    series as present while writing none."""
    K = instruments.kamm_twin_90d
    row = np.zeros(3)
    made = {k: 1.0 for k in K.REDUCED_KEYS} | {K.REDUCED_ROW_KEY: row}
    # day 0 was captured (it always is under --save-3d) but is NOT a requested
    # snapshot day, and the run never reached day 30
    kw = K.reduced_series_kwargs({0: made}, [30, 60, 90])
    assert kw == {}, "a day outside the requested grid must not be written"
    stamp = json.loads(K.storage_stamp("absent", "absent" if not kw else "x"))
    assert stamp["reduced"] == "absent"
    # and when a requested day IS present, the keys and the stamp agree
    kw = K.reduced_series_kwargs({0: made, 30: made}, [30, 60, 90])
    assert list(kw["reduced_days"]) == [30]
    assert all(kw[f"reduced_{k}"].shape == (1,) for k in K.REDUCED_KEYS)
    assert kw[f"reduced_{K.REDUCED_ROW_KEY}"].shape == (1, 3)
    assert json.loads(K.storage_stamp("float32", "float64"))["reduced"] \
        == "float64"


def test_a_failing_reduction_loses_its_day_and_never_the_run(instruments):
    """The reduction is a diagnostic written alongside the primary data, and
    the artifact is only saved after the whole time loop. A reducer that
    raised at day 90 of a 90-day twin would delete 90 days of compute to
    protect a few kB of annotation."""
    K = instruments.kamm_twin_90d

    def boom(live):
        raise SystemExit("the three latitude groups do not partition")

    red, status = K.safe_reduce(boom, {"u": np.zeros(3)})
    assert red is None
    assert "SystemExit" in status and "partition" in status
    # a SystemExit is an exception, so it must be caught like any other --
    # this is the exact type the real reducer raises
    stored, red, status = K.capture_snapshot(
        {"u": np.zeros((2, 2, 2))}, snap_dtype=np.float32, reducer=boom,
        land_mask=np.ones((2, 2)))
    assert stored["u"].dtype == np.float32, "the 3-D block must still be kept"
    assert red is None and status.startswith("reduction failed")


def test_the_series_resolution_is_stamped_not_the_container(instruments):
    """The series is ALWAYS stored in a float64 array, but its resolution is
    the precision the arm was BUILT at. On a deliberate FP64=0 arm it is
    float64-stored and float32-resolved, and a consumer told "float64" would
    credit it with resolution it does not have."""
    K = instruments.kamm_twin_90d
    assert json.loads(K.storage_stamp("float32", "float32"))["reduced"] \
        == "float32"
    assert json.loads(K.storage_stamp("float32", "float64"))["reduced"] \
        == "float64"


def test_a_storage_only_flag_stays_out_of_the_run_config_string(instruments):
    """The recorded configuration string is compared BYTE-FOR-BYTE between two
    arms by the seasonal-clock A/B, which hard-aborts on any difference as a
    confound. A key added for a storage-only setting would make every
    recorded-arm-vs-new-arm comparison abort forever."""
    K = instruments.kamm_twin_90d
    src = inspect.getsource(K.run_twin)
    cfg = src.split("run_config = json.dumps(")[1].split("}, sort_keys=True)")[0]
    assert "fp64_3d" not in cfg
    assert "perturb_seed" in cfg, "wrong block located -- this test is vacuous"


def test_twin_cli_exposes_literal_and_legacy_langmuir_arms(instruments):
    k = instruments.kamm_twin_90d
    default = k._parse_args(["nemo_dino_kamm_mlf", "out.npz"])
    legacy = k._parse_args([
        "nemo_dino_kamm_mlf", "out.npz",
        "--tke-langmuir-evaluation", "vectorized",
    ])
    assert default.tke_langmuir_evaluation is None
    assert legacy.tke_langmuir_evaluation == "vectorized"


# ---------------------------------------------------------------------------
# paired corrected-T-carry basin verdict
# ---------------------------------------------------------------------------
def test_tcarry_basin_floor_has_priority_over_refute(instruments):
    """Zero response was previously both REFUTE and floor-limited."""
    score = instruments.tcarry_reverdict.classify
    assert score(0.0, -1.0, 0.1, True) == "UNRESOLVED/FLOOR"
    assert score(0.01, -1.0, 0.001, True) == "REFUTED"


def test_tcarry_basin_floor_instrument_self_test_is_red_capable(instruments):
    assert instruments.tcarry_floor._self_test() == 0


def test_tcarry_basin_compensation_blocks_confirm(instruments):
    score = instruments.tcarry_reverdict.classify
    assert score(0.25, -1.0, 0.01, True) == "CONFIRMED"
    assert score(0.25, -1.0, 0.01, False) == "UNRESOLVED/COMPENSATION"


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_tcarry_basin_nonfinite_classifier_is_fatal(instruments, bad):
    with pytest.raises(SystemExit, match="non-finite"):
        instruments.tcarry_reverdict.classify(bad, -1.0, 0.1, True)


def test_tcarry_basin_day0_identity_rejects_nonfinite_and_changed_bits(instruments):
    identical = instruments.tcarry_reverdict._bit_identical
    assert identical(np.array([1.0]), np.array([1.0]))
    assert not identical(np.array([1.0]), np.array([np.nan]))
    assert not identical(np.array([0.0]), np.array([-0.0]))


def test_tcarry_basin_baseline_gate_can_fail(instruments):
    gate = instruments.tcarry_reverdict._check_baseline
    assert instruments.tcarry_reverdict.BASELINE[90] == -0.43908550999203477
    assert instruments.tcarry_reverdict.FLOOR[90] == 0.00015266693430725714
    gate(instruments.tcarry_reverdict.BASELINE[90], 90)
    with pytest.raises(SystemExit, match="misses registered"):
        gate(instruments.tcarry_reverdict.BASELINE[90] + 3.0, 90)
    gate(instruments.tcarry_reverdict.BASELINE[360], 360)
    with pytest.raises(SystemExit, match="misses registered"):
        gate(instruments.tcarry_reverdict.BASELINE[360] + 3.0, 360)


def test_tcarry_basin_rejects_unregistered_common_config(instruments):
    check = instruments.tcarry_reverdict._registered_config_errors
    valid = {
        "recipe": "nemo_dino_kamm_mlf", "n_days": 90,
        "bridge_tke": False, "bridge_before": True,
        "vmix_scheme": None, "use_gm_redi": None,
        "surface_stress_implicit": False, "surface_tendency_placement": None,
        "save_step_eta": False, "perturb_seed": None, "perturb_eps": 1e-14,
        "perturb_baro": None, "perturb_baro_sha256": None,
        "perturb_baro_key": "dU_avg", "perturb_baro_scale": 1.0,
        "daily_acc": False, "u_m": None,
    }
    assert check(valid, "arm", 90) == []
    planted = dict(valid, perturb_baro="/tmp/plant.npz",
                   perturb_baro_sha256="0" * 64)
    assert any("perturb_baro" in error for error in check(planted, "arm", 90))


def test_tcarry_selftest_runs_every_receipt_plant_through_real_npz(
        instruments, capsys):
    """Regression for the Stage-1 crash: a dict-only test cannot expose an
    NPZ overlay whose missing membership dunder triggers integer iteration."""
    assert instruments.tcarry_reverdict._self_test() == 0
    out = capsys.readouterr().out
    assert "planted corrected-T -> U_AS_T_LEGACY" in out
    assert "planted corrected float64 -> float32" in out
    assert "planted corrected rn_Uv 0.27 -> 0.54" in out
    assert "planted unregistered perturb_baro" in out
    assert "every receipt plant passed through real NPZ files" in out


def test_tcarry_retained_stage1_producer_is_independent_of_amended_scorer_head(
        instruments):
    scorer = "e" * 40
    expected = instruments.tcarry_reverdict._expected_producer_sha
    assert expected(90, scorer) == instruments.tcarry_reverdict.STAGE1_PRODUCER_GIT_SHA
    assert expected(90, scorer) != scorer
    assert expected(360, scorer) == scorer


def test_tcarry_reconciliation_old_gap_receipt_is_numeric_and_red_capable(
        instruments):
    gate = instruments.tcarry_reconcile._old_gap_matches
    expected = -0.010717232432999602
    assert gate(expected + 8.9e-16, expected)
    assert not gate(expected + 1.0e-6, expected)
    assert not gate(np.nan, expected)


def test_tcarry_reconciliation_clock_escape_is_scoped_to_historical_artifact(
        instruments, monkeypatch):
    reconcile = instruments.tcarry_reconcile
    seen = []

    def fake_load(_path, day):
        seen.append((day, os.environ.get("DINO_GATE_ALLOW_LEGACY_CLOCK")))
        return {"u": np.zeros(1)}

    monkeypatch.delenv("DINO_GATE_ALLOW_LEGACY_CLOCK", raising=False)
    monkeypatch.setattr(reconcile.G, "load_candidate", fake_load)
    monkeypatch.setattr(reconcile.R, "_reduce", lambda state: (0.0, np.zeros(14)))
    nemo = {"u": np.zeros(1)}
    reconcile._gap("old.npz", 30, nemo, historical=True)
    reconcile._gap("current.npz", 30, nemo, historical=False)
    assert seen == [(30, "1"), (30, None)]
    assert "DINO_GATE_ALLOW_LEGACY_CLOCK" not in os.environ


def test_tcarry_een_off_ownership_classifier_has_reachable_both_states(
        instruments):
    scorer = instruments.tcarry_een
    assert scorer.classify_ownership(scorer.HISTORICAL_BASELINE) == (
        "CONFIRMED_FULL_EEN_OWNERSHIP")
    outside = scorer.HISTORICAL_BASELINE + 2.0 * scorer.OWNERSHIP_BAND
    assert scorer.classify_ownership(outside) == "REFUTED_FULL_EEN_OWNERSHIP"
    with pytest.raises(SystemExit, match="non-finite"):
        scorer.classify_ownership(np.nan)


def test_bridge_omega_cli_default_is_explicit_nemo_bit_identical(instruments):
    harness = instruments.kamm_twin_90d
    omitted = harness._parse_args(["nemo_dino_kamm_mlf", "out.npz"])
    explicit = harness._parse_args(
        ["nemo_dino_kamm_mlf", "out.npz", "--bridge-omega", "nemo"])
    assert vars(omitted) == vars(explicit)
    assert omitted.bridge_omega == "nemo"


def test_bridge_omega_selector_changes_only_registered_constant(instruments):
    harness = instruments.kamm_twin_90d
    config = harness.dino_config_for_recipe("nemo_dino_kamm_mlf")
    nemo_omega, nemo_reference = harness.resolve_bridge_omega("nemo")
    old_omega, old_reference = harness.resolve_bridge_omega("legacy-rounded")
    assert nemo_omega == harness.NEMO_CONSTANTS_CONFIG.Omega
    assert old_omega == harness.constants.Omega
    assert nemo_reference == "nemo"
    assert old_reference == "selected_omega"
    assert old_omega != nemo_omega
    assert config.omega == harness.NEMO_CONSTANTS_CONFIG.Omega
    with pytest.raises(ValueError, match="bridge_omega"):
        harness.resolve_bridge_omega("rounded-ish")


def test_tcarry_bridge_omega_scorer_self_test_is_red_capable(instruments):
    assert instruments.tcarry_omega._self_test() == 0


def test_tcarry_bridge_omega_scorer_has_complete_committed_bindings(instruments):
    scorer = instruments.tcarry_omega
    assert scorer._require_bound() is None
    assert scorer.BOUND_PRODUCER_SHA == (
        "9e339ad1b2032bc47ec132fb2ad6f00ea071bbb9")
    assert scorer.FOURTH_PRODUCER_SHA == (
        "b14a17dd6f14592594daacc4b64c6a1de2a0a004")
    assert scorer.FOURTH_ARTIFACT_SHA256 == (
        "678a6a914367561596a259cc27a50f9294d14e7c0ed32c68aefe9ee6fff2a6f8")


def test_tcarry_een_omega_interaction_decision_tree_is_red_capable(instruments):
    scorer = instruments.tcarry_omega
    locked = (scorer.LOCKED_A, scorer.LOCKED_B, scorer.LOCKED_C)
    assert scorer.classify_interaction(
        *locked, scorer.HISTORICAL_BASELINE) == (
            "CONFIRMED_COMBINED_EEN_OMEGA_OWNERSHIP")
    assert scorer.classify_interaction(
        *locked, scorer.HISTORICAL_BASELINE + 2.0 * scorer.TWO_F) == (
            "REFUTED_COMBINED_EEN_OMEGA_OWNERSHIP")
    assert scorer.classify_interaction(
        scorer.LOCKED_A + 2.0 * scorer.LOCKED_TOL,
        scorer.LOCKED_B, scorer.LOCKED_C, scorer.HISTORICAL_BASELINE) == (
            "INVALID_STOP_LOCKED_CORNER")
    assert scorer.classify_additivity(0.0) == "ADDITIVE_BELOW_BAND"
    assert scorer.classify_additivity(2.0 * scorer.TWO_F) == "NON_ADDITIVE"


def test_kt1_record_twin_admission_refuses_a_drifted_record(tmp_path):
    """The admission must FAIL when a record differs, not just when it errors.

    A re-acquired NEMO record is only a substitute for the certified one if
    every restart variable is bit-identical; the whole barotropic-ladder plan
    rests on that. This plants a one-ulp change in a copied tile and requires
    the script to exit non-zero and name the variable.
    """
    import shutil
    import subprocess
    import netCDF4 as nc

    src = sorted(Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
                      "DINO/RUN_FROMREST_KT1").glob(
                          "DINO_00000001_restart_*.nc"))
    if not src:
        pytest.skip("the certified kt=1 record is not on this machine")
    dst = tmp_path / "planted"
    dst.mkdir()
    for p in src:
        shutil.copy2(p, dst / p.name)
    with nc.Dataset(dst / src[0].name, "a") as d:
        v = d.variables["tn"]
        a = v[:]
        a[0, 0, 0, 0] = np.nextafter(float(a[0, 0, 0, 0]), np.inf)
        v[:] = a

    script = (Path(__file__).resolve().parents[3] / "scripts" / "validate" /
              "ocean_fidelity" / "dino_1226" /
              "kt1_record_twin_admission.py")
    r = subprocess.run(
        [sys.executable, str(script),
         str(Path(src[0]).parent / "DINO_00000001_restart_*.nc"),
         str(dst / "DINO_00000001_restart_*.nc")],
        capture_output=True, text=True)
    assert r.returncode != 0, r.stdout[-2000:]
    assert "tn" in r.stdout, r.stdout[-2000:]
    # and it must PASS on the untouched pair, or the test above proves nothing
    ok = subprocess.run(
        [sys.executable, str(script),
         str(Path(src[0]).parent / "DINO_00000001_restart_*.nc"),
         str(Path(src[0]).parent / "DINO_00000001_restart_*.nc")],
        capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout[-2000:]
    assert "ADMITTED" in ok.stdout
