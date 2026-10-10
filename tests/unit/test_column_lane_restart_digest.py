"""A restart on the column loop reproduces the continuous run, step by step,
on the per-step restart-set digest (``LEGOESM_DIGEST_EVERY_STEPS``): MPAS
and the FV3-duo column lane, land on, restart at a physics-but-not-radiation
boundary.  The instrument is validated first: the digest writer sees a
change in each part and nothing else; two continuous runs of one config
agree; a planted 1e-12 K change of one soil leaf is seen.

Preconditions raise ``RuntimeError`` so the strict xfail (restart exactness
pending) can only absorb the final digest equality."""
import os
import types

import numpy as np
import pytest

from tests.atmosphere.hydrostatic.unit.test_fv3_duo_column import (  # noqa: F401
    _drop_compiled_graphs, _land_stress_cfg)
from tests.unit.test_mpas_multilayer_land_port import (
    _LS_KW, ONE_STEP_DAYS, THREE_STEPS_DAYS, TWO_STEPS_DAYS, _build_driver,
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


_REX = pytest.mark.xfail(
    strict=True, raises=AssertionError, reason=(
        "R-ex pending (composable physics design): a column-loop restart "
        "re-seeds the land boundary outputs and the held radiation slots "
        "and keeps checkpoint-only carry leaves, so every digest part "
        "differs from the first resumed step on MPAS and the duo "
        "(measured 2026-10-10, jobs 10315955/10315961)"))


@_REX
def test_mpas_restart_matches_the_continuous_digest(tmp_path, monkeypatch):
    """MPAS, land stress on, gray radiation: resume from the step-2
    checkpoint; step 3 digests equal the continuous run's."""
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


@pytest.mark.slow
@_REX
def test_duo_column_lane_restart_matches_the_continuous_digest(
        tmp_path, monkeypatch):
    """FV3-duo column lane, RRTMGP every 5 steps, land on: the day-1
    checkpoint (step 144 at the CFL-clamped 600 s, not a radiation step)
    resumes bitwise over the remaining 144 steps."""
    from legoesm.driver.model_driver import ModelDriver
    monkeypatch.setenv("LEGOESM_DIGEST_EVERY_STEPS", "1")

    def build(name, days):
        d = tmp_path / name
        d.mkdir()
        cfg = _land_stress_cfg(d, monkeypatch, stress=None, days=days,
                               dt=600.0, rad_update_steps=5)
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
    _require(step == 144 and step % 5 and (step - 1) % 5, step)
    _require(_drive(dB.run, start_step=step, start_day=day) == "COMPLETED",
             "b")
    steps, bad = _restarted_vs_continuous(tmp_path / "c", step, d)
    _require(steps == list(range(145, 289)), steps)
    assert not bad, _mismatch(tmp_path / "c", d, bad[0])
