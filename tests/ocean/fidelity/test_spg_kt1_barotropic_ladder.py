"""The kt=1 barotropic ladder, and the record audit it opens with.

Two things are under test.

The AUDIT must actually detect a raced record.  It is the reason two rungs of
the ladder read UNMEASURED instead of carrying a number, so an audit that
quietly said "coherent" would put fabricated numbers on the board.  The check
here is the closed identity the audit rests on: with no freshwater flux, the
first substep from rest reduces to ``un_e = rDt_e * zu_frc * ssumask``, so two
coherent streams MUST satisfy it and this record's do not.

The TABLE must be able to fail, and must be able to pass.  ``--plant`` moves
one cell of every rung that currently reads AT BAR; all four must flip to
DEBT.  Planting into a rung that is already DEBT would prove nothing.

The audit must also FAIL CLOSED: pointed at a directory with no record it must
say INDETERMINATE, never "coherent".  An audit that cannot look is not an
audit that found nothing.

Skipped (never silently passed) when NEMO's record or netCDF4 is absent.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import subprocess
import sys

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_GATE = os.path.join(_REPO, "scripts", "validate", "ocean_fidelity", "dino_1226",
                     "spg_kt1_barotropic_ladder.py")
_RUN = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/"
        "RUN_FROMREST_KT1")

_needs_oracle = pytest.mark.skipif(
    not glob.glob(os.path.join(_RUN, "mesh_mask_*.nc")),
    reason=f"NEMO from-rest record not here: {_RUN}")
_needs_nc = pytest.mark.skipif(
    importlib.util.find_spec("netCDF4") is None,
    reason="netCDF4 not installed: the audit cannot read the oracle mesh, so "
           "a failure here would not be a regression")


def _run(*extra):
    env = dict(os.environ, JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu")
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(_REPO, "packages", p) for p in
         ("core", "ocean", "atmosphere", "coupler", "ice", "land", "ml", "tools")]
        + [os.path.join(_REPO, "src"), env.get("PYTHONPATH", "")])
    return subprocess.run([sys.executable, _GATE, *extra],
                          capture_output=True, text=True, env=env, cwd=_REPO)


@_needs_oracle
@_needs_nc
def test_audit_reports_this_record_as_raced():
    r = _run("--audit-only")
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    assert "VERDICT: RACED" in r.stdout, r.stdout[-4000:]
    # the closed identity is the evidence, not the verdict line
    assert "closed identity" in r.stdout
    assert "cells unequal 625/650" in r.stdout, r.stdout[-4000:]


@_needs_oracle
@_needs_nc
def test_audit_names_two_streams_from_different_ranks():
    """Two dumps whose land patterns cannot both belong to one tile."""
    r = _run("--audit-only")
    lines = [ln for ln in r.stdout.splitlines() if "land pattern" in ln]
    assert len(lines) == 2, r.stdout[-4000:]
    import re
    sets = [set(int(x) for x in re.findall(r"\d+", ln.split("ranks")[1]))
            for ln in lines]
    assert sets[0] and sets[1]
    assert sets[0].isdisjoint(sets[1]), lines


@_needs_oracle
@_needs_nc
def test_this_script_never_writes_inside_the_oracle():
    """It used to, and a reviewer found it still reachable.

    The removed ``--emit-runsh`` wrote a re-run that edited
    ``cfgs/DINO/MY_SRC/dynspg_ts.F90`` in place behind an exit trap and then
    ran ``makenemo -r DINO -n DINO``, overwriting the oracle's own BLD tree
    and ``bin/nemo.exe`` -- which no trap restores.  The acquisition is now a
    makenemo CONFIG COPY that refuses to touch either path.  This asserts on
    the SOURCE, because the flag is gone and no invocation can exercise it.
    """
    body = open(_GATE).read()
    assert "--emit-runsh" not in body
    assert "emit_runsh" not in body
    assert "makenemo" not in body.split('"""', 2)[2], \
        "this script must not build NEMO outside its docstring"
    assert "nemo_dino_kt1_rankdump/run.sh" in body, \
        "the pointer to the safe acquisition was lost"


def test_audit_fails_closed_when_there_is_nothing_to_audit(tmp_path):
    """An audit that cannot look must not report "coherent"."""
    r = _run("--audit-only", "--run-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr[-2000:]
    assert "INDETERMINATE" in r.stdout, r.stdout[-3000:]
    assert "coherent --" not in r.stdout


@pytest.mark.slow
@_needs_oracle
@_needs_nc
def test_plant_flips_every_rung_that_is_at_bar():
    """Non-vacuity.  Needs a model step, so it is marked slow.

    "GATE FAIL" is NOT the assertion -- rungs R0a/R0b/R4* are already DEBT, so
    the gate fails either way and that check could never catch a no-op plant.
    The assertion is that the four rows which read AT BAR without the plant
    read DEBT with it, one by one.
    """
    plain = _run()
    planted = _run("--plant")
    at_bar = {ln.split("   ")[0].strip() for ln in plain.stdout.splitlines()
              if ln.rstrip().endswith("AT BAR")}
    assert len(at_bar) == 4, plain.stdout[-6000:]
    for name in at_bar:
        hit = [ln for ln in planted.stdout.splitlines()
               if ln.startswith(name)]
        assert len(hit) == 1, (name, planted.stdout[-6000:])
        assert hit[0].rstrip().endswith("DEBT"), hit[0]
