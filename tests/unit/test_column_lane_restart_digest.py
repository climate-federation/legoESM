"""A restart on the column loop reproduces the continuous run, step by step,
on the per-step restart-set digest (``LEGOESM_DIGEST_EVERY_STEPS``): MPAS
and the FV3-duo column lane, land on, restart on a radiation boundary (as
D-REX-1 requires of every checkpoint).  The instrument is validated first: the digest writer sees a
change in each part and nothing else; two continuous runs of one config
agree; a planted 1e-12 K change of one soil leaf is seen.

Preconditions raise ``RuntimeError`` so only the final digest equality (and
the persisted clock origin, and the D-REX-3/4 refusals) can fail the test."""
import os
import types

import numpy as np
import pytest

from tests.atmosphere.hydrostatic.unit.test_fv3_duo_column import (  # noqa: F401
    _drop_compiled_graphs, _land_stress_cfg)
from tests.unit.test_mpas_multilayer_land_port import (
    DT, _LS_KW, ONE_STEP_DAYS, THREE_STEPS_DAYS, TWO_STEPS_DAYS, _build_driver,
    _patch_land_loaders)

_PARTS = ("state", "phys", "land", "landio", "ice", "sfc", "forcing",
          "carry", "cmor")


def _require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def _drive(fn, *a, **k):
    """Call a driver entry point; an assertion failing inside the driver is
    a broken precondition, never the expected digest mismatch."""
    try:
        return fn(*a, **k)
    except AssertionError as exc:
        raise RuntimeError(f"driver assertion in {fn.__name__}") from exc


def _digests(outdir) -> dict[int, str]:
    """``{abs_step: digest line}`` of ``digest_steps_r0.log``; a step logged
    twice is an error (a duplicate could hide a mismatching row)."""
    lines = {}
    with open(os.path.join(str(outdir), "digest_steps_r0.log")) as f:
        for line in f:
            step, rest = line.split(" ", 1)
            k = int(step.removeprefix("step="))
            _require(k not in lines, f"step {k} digested twice in {outdir}")
            lines[k] = rest.strip()
    return lines


def _parts(line) -> dict[str, tuple[str, int]]:
    """``{part: (sha16, n_leaves)}`` of one digest line; every part present."""
    out = {}
    for tok in line.split():
        k, v = tok.split("=", 1)
        sha, n = v.split("/")
        out[k] = (sha, int(n))
    _require(tuple(out) == _PARTS, tuple(out))
    return out


def _restarted_vs_continuous(cont, start_step, resumed):
    """Resumed steps (must start at start_step+1, continuous log gapless)
    and the subset whose digest line differs from the continuous run."""
    c, r = _digests(cont), _digests(resumed)
    steps = sorted(r)
    _require(steps and steps[0] == start_step + 1, (steps, start_step))
    _require(sorted(c) == list(range(1, steps[-1] + 1)), sorted(c))
    return steps, [k for k in steps if c[k] != r[k]]


def _mismatch(cont, resumed, step):
    c, r = _parts(_digests(cont)[step]), _parts(_digests(resumed)[step])
    return f"step {step}: " + ", ".join(
        f"{k} {c[k]} != {r[k]}" for k in c if c[k] != r[k])


def test_digest_writer_sees_each_part_and_only_that_part(tmp_path):
    """The writer, directly: for EVERY part, changing one value changes that
    part's digest and no other; so does a shape change and a dtype change
    with identical bytes; a None part is 0 leaves."""
    from legoesm.driver.model_driver import ModelDriver
    me = types.SimpleNamespace(_output_dir=tmp_path, _mpi_rank=0)
    base = {k: {"a": np.arange(6.0).reshape(2, 3) + i,
                "b": np.arange(3, dtype=np.int64)}
            for i, k in enumerate(_PARTS)}
    cases = [("base", base)]
    for k in _PARTS:
        v = base[k]["a"].copy()
        v[1, 2] += 1e-12
        cases.append((f"{k}:value", {**base, k: {**base[k], "a": v}}))
    cases.append(("state:shape", {**base, "state": {
        **base["state"], "a": base["state"]["a"].reshape(3, 2)}}))
    cases.append(("state:dtype", {**base, "state": {
        **base["state"], "b": base["state"]["b"].view(np.float64)}}))
    assert cases[-1][1]["state"]["b"].tobytes() == base["state"]["b"].tobytes()
    cases.append(("cmor:none", {**base, "cmor": None}))
    for i, (_, parts) in enumerate(cases, start=1):
        ModelDriver._write_step_digest(me, i, parts)
    log = {i: _parts(line) for i, line in _digests(tmp_path).items()}
    assert sorted(log) == list(range(1, len(cases) + 1))
    ref = log[1]
    assert all(ref[k][1] == 2 for k in _PARTS)
    assert log[len(cases)]["cmor"] == ("e3b0c44298fc1c14", 0)
    for i, (name, _) in enumerate(cases[1:], start=2):
        part = name.split(":")[0]
        got = log[i]
        assert got[part][0] != ref[part][0], name
        assert {k: v for k, v in got.items() if k != part} == \
            {k: v for k, v in ref.items() if k != part}, name


def test_restart_comparator_on_synthetic_logs(tmp_path):
    """The comparator: resumed log starts at start_step+1, continuous log is
    gapless, no step twice, one differing line reported."""
    def write(name, lines):
        (tmp_path / name).mkdir()
        with open(tmp_path / name / "digest_steps_r0.log", "w") as f:
            f.writelines(f"step={k} state=a{k}/1 phys=b/1\n" for k in lines)
    write("c", [1, 2, 3])
    write("r", [3])
    assert _restarted_vs_continuous(tmp_path / "c", 2, tmp_path / "r") == \
        ([3], [])
    write("r_early", [2, 3])
    with pytest.raises(RuntimeError):
        _restarted_vs_continuous(tmp_path / "c", 2, tmp_path / "r_early")
    write("c_gap", [1, 3])
    with pytest.raises(RuntimeError):
        _restarted_vs_continuous(tmp_path / "c_gap", 2, tmp_path / "r")
    write("c_dup", [1, 2, 3, 3])
    with pytest.raises(RuntimeError, match="twice"):
        _restarted_vs_continuous(tmp_path / "c_dup", 2, tmp_path / "r")
    (tmp_path / "r_diff").mkdir()
    with open(tmp_path / "r_diff" / "digest_steps_r0.log", "w") as f:
        f.write("step=3 state=a3/1 phys=ZZ/1\n")
    assert _restarted_vs_continuous(tmp_path / "c", 2, tmp_path / "r_diff") \
        == ([3], [3])


def _mpas_continuous(tmp_path, name, days, monkeypatch, nudge=None):
    _patch_land_loaders(monkeypatch)
    d = _drive(_build_driver, str(tmp_path / name), days, **_LS_KW,
               land_stress_from_land=True)
    if nudge is not None:
        d._land_ml_state = d._land_ml_state._replace(
            T_soil=d._land_ml_state.T_soil + nudge)
    _require(_drive(d.run) == "COMPLETED", name)
    return d


def test_two_continuous_runs_agree_and_a_soil_nudge_is_seen(
        tmp_path, monkeypatch):
    """Two continuous MPAS runs of one config digest identically at every
    step; a 1e-12 K nudge of the soil temperature leaf changes the land
    digest at step 1 and, through the one-step land handoff, the
    atmosphere's at step 2."""
    monkeypatch.setenv("LEGOESM_DIGEST_EVERY_STEPS", "1")
    _mpas_continuous(tmp_path, "c1", TWO_STEPS_DAYS, monkeypatch)
    _mpas_continuous(tmp_path, "c2", TWO_STEPS_DAYS, monkeypatch)
    _mpas_continuous(tmp_path, "c3", TWO_STEPS_DAYS, monkeypatch, nudge=1e-12)
    c1, c2, c3 = (_digests(tmp_path / n) for n in ("c1", "c2", "c3"))
    assert sorted(c1) == [1, 2] and c1 == c2
    p1, p3 = _parts(c1[1]), _parts(c3[1])
    assert all(p1[k][1] > 0 for k in ("state", "phys", "land", "landio",
                                       "sfc", "forcing")), p1
    assert p3["land"] != p1["land"] and p3["land"][1] == p1["land"][1]
    assert _parts(c3[2])["state"] != _parts(c1[2])["state"]


def test_mpas_restart_matches_the_continuous_digest(tmp_path, monkeypatch):
    """MPAS, land stress on, gray radiation: resume from the step-2
    checkpoint; step 3 digests equal the continuous run's.  The digest's
    forcing part carries the lagged shflx/lhflx the CAPE-diurnal forcing
    reads (on by default), so dropping D-REX-2 fails here.  The forcing-clock
    origin is checked as written and adopted: in a same-config chain the
    reconstructed origin snaps to the same float, so the digest alone cannot
    see it."""
    monkeypatch.setenv("LEGOESM_DIGEST_EVERY_STEPS", "1")
    _mpas_continuous(tmp_path, "c", THREE_STEPS_DAYS, monkeypatch)
    _mpas_continuous(tmp_path, "a", TWO_STEPS_DAYS, monkeypatch)
    ckpt = sorted((tmp_path / "a").glob("checkpoint_day_*.npz"))[-1]
    dB = _drive(_build_driver, str(tmp_path / "b"), ONE_STEP_DAYS, **_LS_KW,
                land_stress_from_land=True)
    step, day = _drive(dB.load_checkpoint, ckpt)
    _require(step == 2, step)
    _require(_drive(dB.run, start_step=step, start_day=day) == "COMPLETED",
             "b")
    steps, bad = _restarted_vs_continuous(tmp_path / "c", step, tmp_path / "b")
    _require(steps == [3], steps)
    assert not bad, _mismatch(tmp_path / "c", tmp_path / "b", bad[0])
    with np.load(ckpt) as z:
        assert "clock_day_origin" in z.files
        origin = float(z["clock_day_origin"])
    assert origin == dB._clock_day_origin_ckpt == dB._clock_day_origin


@pytest.mark.slow
def test_duo_column_lane_restart_matches_the_continuous_digest(
        tmp_path, monkeypatch):
    """FV3-duo column lane, RRTMGP every 4 steps, land on: the day-1
    checkpoint (step 144 at the CFL-clamped 600 s; a radiation boundary, as
    D-REX-1 requires of every checkpoint) resumes bitwise over the remaining
    144 steps -- 108 of them held-radiation steps."""
    from legoesm.driver.model_driver import ModelDriver
    monkeypatch.setenv("LEGOESM_DIGEST_EVERY_STEPS", "1")

    def build(name, days):
        d = tmp_path / name
        d.mkdir()
        cfg = _land_stress_cfg(d, monkeypatch, stress=None, days=days,
                               dt=600.0, rad_update_steps=4)
        cfg = cfg._replace(output=cfg.output._replace(checkpoint_days=1))
        drv = ModelDriver(cfg, output_dir=d)
        _drive(drv.setup)
        return drv, d

    for name, days in (("c", 2), ("a", 1)):
        _require(_drive(build(name, days)[0].run) == "COMPLETED", name)
    ckpt = tmp_path / "a" / "checkpoint_day_0001.npz"
    _require(ckpt.exists(), ckpt)
    dB, d = build("b", 1)
    step, day = _drive(dB.load_checkpoint, ckpt)
    _require(step == 144 and step % 4 == 0, step)
    _require(_drive(dB.run, start_step=step, start_day=day) == "COMPLETED",
             "b")
    steps, bad = _restarted_vs_continuous(tmp_path / "c", step, d)
    _require(steps == list(range(145, 289)), steps)
    assert not bad, _mismatch(tmp_path / "c", d, bad[0])


@pytest.mark.slow
def test_latlon_column_lane_restart_matches_the_continuous_digest(
        tmp_path, monkeypatch):
    """Lat-lon column lane (P3), RRTMGP every 4 steps, land on: the day-1
    checkpoint (step 48 at 1800 s, a radiation boundary as D-REX-1 requires)
    resumes bitwise over the remaining 48 steps -- 36 of them held-radiation
    steps -- on the full restart set, native face winds included."""
    from legoesm.driver.model_driver import ModelDriver
    from tests.atmosphere.hydrostatic.unit.test_latlon_column import (
        _latlon_land_stress_cfg)
    monkeypatch.setenv("LEGOESM_DIGEST_EVERY_STEPS", "1")

    def build(name, days):
        d = tmp_path / name
        d.mkdir()
        cfg = _latlon_land_stress_cfg(d, monkeypatch, stress=None, days=days,
                                      dt=1800.0, rad_update_steps=4)
        cfg = cfg._replace(output=cfg.output._replace(checkpoint_days=1))
        drv = ModelDriver(cfg, output_dir=d)
        _drive(drv.setup)
        return drv, d

    for name, days in (("c", 2), ("a", 1)):
        _require(_drive(build(name, days)[0].run) == "COMPLETED", name)
    ckpt = tmp_path / "a" / "checkpoint_day_0001.npz"
    _require(ckpt.exists(), ckpt)
    dB, d = build("b", 1)
    step, day = _drive(dB.load_checkpoint, ckpt)
    _require(step == 48 and step % 4 == 0, step)
    _require(_drive(dB.run, start_step=step, start_day=day) == "COMPLETED",
             "b")
    steps, bad = _restarted_vs_continuous(tmp_path / "c", step, d)
    _require(steps == list(range(49, 97)), steps)
    assert not bad, _mismatch(tmp_path / "c", d, bad[0])


@pytest.mark.usefixtures("allow_version_drift")  # hand-built checkpoints, no manifest beside them
def test_mpas_restart_refuses_a_start_day_the_saved_origin_does_not_imply(
        tmp_path, monkeypatch):
    """D-REX-4: resuming the step-2 checkpoint with a start day one day off
    the day its recorded forcing-clock origin implies is refused, naming both
    days -- so the resumed run READS the saved origin (ignoring it, nothing
    could disagree).  The same checkpoint with the origin stripped keeps the
    previous behaviour: the shifted start day runs (origin reconstructed)."""
    _mpas_continuous(tmp_path, "a", TWO_STEPS_DAYS, monkeypatch)
    ckpt = sorted((tmp_path / "a").glob("checkpoint_day_*.npz"))[-1]
    dB = _drive(_build_driver, str(tmp_path / "b"), ONE_STEP_DAYS, **_LS_KW,
                land_stress_from_land=True)
    step, day = _drive(dB.load_checkpoint, ckpt)
    _require(step == 2 and dB._clock_day_origin_ckpt is not None, step)
    with pytest.raises(ValueError, match="implied by the checkpoint's "
                       "forcing-clock origin") as exc:
        dB.run(start_step=step, start_day=day + 1.0)
    assert repr(day + 1.0) in str(exc.value)
    assert repr(dB._clock_day_origin_ckpt + step * DT / 86400.0) in \
        str(exc.value)
    # The saved origin DRIVES the clock: plant one 1.25 d off the config
    # start day (fractional, so an origin rebuilt from floor(start day) is
    # 0.25 d off) and resume with the day it implies plus 5e-7 d (inside the
    # guard's tolerance).  The one resumed step asks for exactly the planted
    # origin + start_step*dt; an origin from cfg.start_day is 1.25 d off, one
    # rebuilt from the start day 5e-7 d off, a clock one step late 1 dt off.
    with np.load(ckpt) as z:
        origin = float(z["clock_day_origin"])
        planted_origin = origin + 1.25
        planted = tmp_path / "planted" / ckpt.name
        planted.parent.mkdir()
        np.savez(planted, **{**{k: z[k] for k in z.files},
                             "clock_day_origin": np.float64(planted_origin)})
    dP = _drive(_build_driver, str(tmp_path / "p"), ONE_STEP_DAYS, **_LS_KW,
                land_stress_from_land=True)
    step, day = _drive(dP.load_checkpoint, planted)
    _require(dP._clock_day_origin_ckpt == planted_origin, "origin not planted")
    seen = []
    cal = dP._calendar_for_radiation
    dP._calendar_for_radiation = lambda d: (seen.append(float(d)), cal(d))[1]
    assert _drive(dP.run, start_step=step,
                  start_day=day + 1.25 + 5e-7) == "COMPLETED"
    _require(seen, "the step loop never asked for a calendar time")
    assert set(seen) == {planted_origin + step * DT / 86400.0}, seen
    assert dP._clock_day_origin == planted_origin
    legacy = tmp_path / "legacy" / ckpt.name
    legacy.parent.mkdir()
    with np.load(ckpt) as z:
        np.savez(legacy, **{k: z[k] for k in z.files
                            if k != "clock_day_origin"})
    dC = _drive(_build_driver, str(tmp_path / "c"), ONE_STEP_DAYS, **_LS_KW,
                land_stress_from_land=True)
    step, day = _drive(dC.load_checkpoint, legacy)
    _require(dC._clock_day_origin_ckpt is None, "origin not stripped")
    assert _drive(dC.run, start_step=step, start_day=day + 1.0) == "COMPLETED"


def test_mpas_mid_day_end_without_daily_forcing_still_checkpoints(
        tmp_path, monkeypatch, capsys):
    """Daily external forcing OFF (gray radiation): a run ending mid-day
    (step 2 at 300 s) still writes its final checkpoint, unwarned."""
    d = _mpas_continuous(tmp_path, "a", TWO_STEPS_DAYS, monkeypatch)
    _require(d.config.radiation == "gray" and not d.config.nc_from_aerosol,
             "daily external forcing would be on")
    ckpts = sorted((tmp_path / "a").glob("checkpoint_day_*.npz"))
    assert len(ckpts) == 1
    with np.load(ckpts[0]) as z:
        assert int(z["step"]) == 2
    assert "NOT checkpointed" not in capsys.readouterr().err


@pytest.mark.slow
@pytest.mark.usefixtures("allow_version_drift")  # hand-built checkpoints, no manifest beside them
def test_duo_daily_external_forcing_restart_needs_a_day_boundary(
        tmp_path, monkeypatch, capsys):
    """D-REX-3, FV3-duo column lane, RRTMGP every 4 steps, the daily
    external-ozone cache on.  A 148-step run (600 s) ends mid-day: its final
    checkpoint is NOT written (it would overwrite the whole-day one under the
    same name), a warning says so, and the whole-day checkpoint (step 144)
    survives and resumes bitwise.  A mid-day checkpoint (written by hand at
    step 148) is refused before any step, naming the step -- also with its
    forcing-clock origin stripped (origin reconstructed from the start day)."""
    from legoesm.driver.model_driver import ModelDriver
    monkeypatch.setenv("LEGOESM_DIGEST_EVERY_STEPS", "1")

    def build(name, n_steps):
        d = tmp_path / name
        d.mkdir()
        cfg = _land_stress_cfg(d, monkeypatch, stress=None,
                               days=(n_steps * 600.0 + 1.0) / 86400.0,
                               dt=600.0, rad_update_steps=4,
                               ozone_forcing="external")
        cfg = cfg._replace(output=cfg.output._replace(checkpoint_days=1))
        drv = ModelDriver(cfg, output_dir=d)
        _drive(drv.setup)
        _require(drv._ozone_ext_active, "external ozone not active")
        return drv, d

    dC, _ = build("c", 148)
    capsys.readouterr()
    _require(_drive(dC.run) == "COMPLETED", "c")
    whole = tmp_path / "c" / "checkpoint_day_0001.npz"
    assert sorted((tmp_path / "c").glob("checkpoint_day_*.npz")) == [whole]
    with np.load(whole) as z:
        assert int(z["step"]) == 144
    assert "final partial-day state was NOT checkpointed" in \
        capsys.readouterr().err
    dB, d = build("b", 4)
    step, day = _drive(dB.load_checkpoint, whole)
    _require(step == 144, step)
    _require(_drive(dB.run, start_step=step, start_day=day) == "COMPLETED",
             "b")
    steps, bad = _restarted_vs_continuous(tmp_path / "c", step, d)
    _require(steps == [145, 146, 147, 148], steps)
    assert not bad, _mismatch(tmp_path / "c", d, bad[0])
    # The mid-day checkpoint the run no longer writes, from its end state.
    dC._output_dir = tmp_path / "mid"
    dC._output_dir.mkdir()
    _drive(dC.save_checkpoint, 148, dC.config.start_day + 148 * 600.0 / 86400.0)
    mid = tmp_path / "mid" / "checkpoint_day_0001.npz"
    dM, _ = build("m", 4)
    step, day = _drive(dM.load_checkpoint, mid)
    _require(step == 148, step)
    with pytest.raises(ValueError, match="restart at step 148 .* not on a "
                       "day boundary while daily external forcing"):
        dM.run(start_step=step, start_day=day)
    legacy = tmp_path / "legacy" / "checkpoint_day_0001.npz"
    legacy.parent.mkdir()
    with np.load(mid) as z:
        np.savez(legacy, **{k: z[k] for k in z.files
                            if k != "clock_day_origin"})
    dN, _ = build("n", 4)
    step, day = _drive(dN.load_checkpoint, legacy)
    _require(step == 148 and dN._clock_day_origin_ckpt is None,
             "origin not stripped")
    with pytest.raises(ValueError, match="restart at step 148 .* not on a "
                       "day boundary while daily external forcing"):
        dN.run(start_step=step, start_day=day)
