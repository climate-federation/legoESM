"""Oracle match: OpenIFS 49r1 CLOUDSC (``microphysics.ifs_cloudsc.cloudsc``, hard forms).

Fixture ``oracle_fixtures/ifs_cloudsc_49r1.npz``: 25 of the CLOUDSC-dwarf's 100
columns x 137 levels (dt 3600 s), chosen by stage-activity coverage (vote V22a), and
the outputs of RUNNING OpenIFS 8f6f722f73 (cycle 49r1) arpifs/phys_ec/cloudsc.F90 with
its 49r1 SUCLDP settings (gfortran 8.5, -O1 -ffp-contract=off, fp64), in two arms:
``base`` (convective detrainment absent, votes V3c/V4b) and ``detr`` (file PLUDE with
the FOEALFA liquid/ice split, vote V18a).  Six columns also carry the write-only stage
dumps of the instrumented oracle (one per CLOUDSC section, V13).
Built by scripts/validate/oracle/cloudsc_ifs/{build49.sh,instrument_cloudsc.py,
make_fixture.py}.

Metric: per field max|ours - oracle| / max|oracle| (cloudsc_io.rel); tolerance 1e-12.
The port runs WITHOUT an outer ``jax.jit``: XLA's jit contracts ``a + b*c`` into an FMA
that gfortran (-ffp-contract=off) does not form, which flips threshold tests that the
oracle evaluates on exact values (measured: za = 0.9999999999999999 vs 1.0).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

import legoesm.atmosphere.physics.microphysics.ifs_cloudsc as M
from legoesm.atmosphere.physics._ifs_thermo import IFSThermo

ROOT = Path(__file__).resolve().parents[2]
_FIX = Path(__file__).parent / "oracle_fixtures" / "ifs_cloudsc_49r1.npz"
_TOL = 1.0e-12
_ARMS = ("base", "detr")
_STAGES = ("s1", "s32", "s33", "s34", "s41", "s42", "s43", "s44", "s45", "s47", "s48",
           "s412", "s413", "s414", "s49", "s415", "s521", "s522", "s6")


def _io():
    spec = importlib.util.spec_from_file_location(
        "cloudsc_io", ROOT / "scripts/validate/oracle/cloudsc_ifs/cloudsc_io.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


io = _io()
rel = io.rel


@pytest.fixture(scope="module")
def fix():
    return dict(np.load(_FIX))


def setup(fix, arm, cols=slice(None)):
    cfg = M.IFSCloudscConfig(**{f: fix[f"cfg_{f}"].item() for f in M.IFSCloudscConfig._fields})
    th = IFSThermo(**{f: float(fix[f"const_{f}"]) for f in IFSThermo._fields})
    r = {k[len(arm) + 4:]: v for k, v in fix.items() if k.startswith(f"{arm}_in_")}
    r = {k: (v[cols] if np.ndim(v) else float(v)) for k, v in r.items()}
    return cfg, th, io.to_inputs(r), float(r["PTSPHY"])


def port_rel_all(fix, arm):
    """field -> rel(port, oracle); the spread harness imports this (control C4)."""
    cfg, th, x, dt = setup(fix, arm)
    po = io.port_fields(M.cloudsc(cfg, th, x, dt, hard=True), cfg.ncldtop)
    return {k: rel(po[k], fix[f"{arm}_out_{k}"]) for k in io.OUT_FIELDS}


def test_settings_are_49r1_defaults(fix):
    cfg = M.IFSCloudscConfig(**{f: fix[f"cfg_{f}"].item() for f in M.IFSCloudscConfig._fields})
    assert cfg == M.IFSCloudscConfig()


@pytest.mark.parametrize("arm", _ARMS)
def test_whole_call_matches_oracle(fix, arm):
    errs = port_rel_all(fix, arm)
    bad = {k: v for k, v in errs.items() if not v <= _TOL}
    assert not bad, bad


def test_detr_arm_exercises_detrainment(fix):
    assert np.max(np.abs(fix["detr_out_plude"])) > 0 and np.all(fix["base_out_plude"] == 0)


@pytest.fixture(scope="module")
def snaps(fix):
    res = {}
    for arm in _ARMS:
        ncs = fix[f"{arm}_stage_s1_zsolqa"].shape[1]
        cfg, th, x, dt = setup(fix, arm, slice(0, ncs))
        res[arm] = M.cloudsc(cfg, th, x, dt, hard=True, stages=True)[1]
    return res


@pytest.mark.parametrize("arm", _ARMS)
@pytest.mark.parametrize("stage", _STAGES)
def test_stage_matches_oracle(fix, snaps, arm, stage):
    assert set(snaps[arm]) == set(_STAGES)
    want = {k[len(f"{arm}_stage_{stage}_"):] for k in fix if k.startswith(f"{arm}_stage_{stage}_")}
    assert want and set(snaps[arm][stage]) == want, (stage, sorted(want ^ set(snaps[arm][stage])))
    for fld, pv in snaps[arm][stage].items():
        ov = fix[f"{arm}_stage_{stage}_{fld}"]
        e = rel(np.asarray(pv), ov)
        assert e <= _TOL, (stage, fld, e)
