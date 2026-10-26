"""The backend-parity job, and what it is allowed to call a pass.

Three defects codex found in the evidence for this lane's headline claim.

1. The job that produces the parity table always exited zero. It printed the
   worst backend-to-backend difference and stopped: a one-percent wind error
   and bit-identical agreement produced the same job state. A missing result
   file was also success.
2. It scored the EAGER step while the model deploys the COMPILED one.
   Compilation is not neutral in this port -- fused multiply-adds and
   reassociation can flip an upwind selector bit, which is why one of its
   recurrences had to be unrolled -- so the published parity covered a lane
   nobody runs.
3. The array comparator checked that the two lanes had non-finite values in
   the SAME CELLS and then excluded those cells. A NaN in one lane and an
   infinity in the other therefore compared equal.

These are static checks: running the parity job needs the pinned Fortran
oracle output, which is not in the tree.
"""

import ast
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_JOB_DIR = _ROOT / "scripts" / "cluster" / "fv3_native"
_JOB = _JOB_DIR / "full_step_backend_parity.sbatch"
_BATCHED_JOB = _JOB_DIR / "full_step_batched_parity.sbatch"
_JOBS = [_JOB, _BATCHED_JOB]
_ARM_CEILINGS = [(_JOB, "MAX_REL"),
                 (_BATCHED_JOB, "BATCHED_VS_LOOP_MAX_REL")]
_ARM_WARNING = "ARM_TO_ARM_WARN_REL"
_JIT_SELECTORS = [(_JOB,
                   '[ "$BK" = jax ] && [ "$JIT" = 1 ] && JFLAG="--jit"'),
                  (_BATCHED_JOB,
                   '[ "$JIT" = 1 ] && JFLAG="--jit"')]
_SCORER = _ROOT / "scripts" / "validate" / "fv3_native" / "full_step_oracle_parity.py"


@pytest.mark.parametrize(("job", "arm_ceiling"), _ARM_CEILINGS)
def test_the_parity_job_can_report_a_failure(job, arm_ceiling):
    body = job.read_text()
    arm_args = re.findall(r'EXTRA="([^"\n]*)"', body)
    assert arm_args and all("--max-rel" in args for args in arm_args), (
        "the job compares the two backends and exits zero whatever it finds, "
        "so the parity table it produces cannot distinguish agreement from a "
        "one-percent error")
    assert re.search(r"--backend [^\n]*\$EXTRA", body), (
        "the per-arm oracle ceiling is defined but not passed to the scorer")
    assert f'os.environ["{arm_ceiling}"]' in body, (
        "the job does not gate its two arms against each other")
    assert "if worst[0] > bound:" in body, (
        "the arm-to-arm ceiling is read but does not gate the comparison")
    assert re.search(r"^if not math\.isfinite\(bound\):$", body, re.MULTILINE), (
        "a NaN or infinite value is not a ceiling")
    assert "exit $BKRC" in body, "the job discards its comparison's status"
    assert "SystemExit(0)" not in body, (
        "a missing or empty result file still exits zero, i.e. 'nothing was "
        "compared' is reported as 'the backends agree'")


@pytest.mark.parametrize(("job", "arm_ceiling"), _ARM_CEILINGS)
def test_the_arm_ceiling_and_warning_band_are_both_pinned(job, arm_ceiling):
    """The ceiling stopped being a refusal when the user chose its value.

    It was a refusal while nobody had picked a number, because a comparison
    with no ceiling reports a number and certifies nothing. On 2026-09-11 the
    user chose 1 deliberately -- there is no established tolerance for
    arm-to-arm agreement yet and a guessed tight number would fire on
    rounding -- so 1 is now an ASKED value, not a silent default, and the
    warning band carries the number that actually gets tightened first.
    Both stay overridable; this pins that neither drifts unnoticed.
    """
    body = job.read_text()
    assert f'export {arm_ceiling}="${{{arm_ceiling}:-1}}"' in body, (
        f"{arm_ceiling} no longer defaults to the value the user chose")
    assert ('export ARM_TO_ARM_WARN_REL="${ARM_TO_ARM_WARN_REL:-1.4e-4}"'
            in body), (
        "the warning band no longer defaults to the 2026-09-11 measurement, "
        "so a run that quietly worsened would stop saying so")


@pytest.mark.parametrize(("job", "arm_ceiling"), _ARM_CEILINGS)
def test_arm_delta_has_required_loose_ceiling_and_warning(job, arm_ceiling,
                                                          tmp_path):
    body = job.read_text()
    assert f'export {arm_ceiling}="${{{arm_ceiling}:-1}}"' in body
    assert _ARM_WARNING in body, (
        "the comparison no longer reads a warning band at all")
    delimiter = "PYEOF" if job == _BATCHED_JOB else "EOF"
    program = body.rsplit(f"<<'{delimiter}'\n", 1)[1].split(
        f"\n{delimiter}", 1)[0]
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text(json.dumps({"metric": 1.0, "batched": False,
                                "compiled": False}))
    right.write_text(json.dumps({"metric": 1.5, "batched": True,
                                 "compiled": False}))
    env = os.environ.copy()
    env[arm_ceiling] = "1"
    env[_ARM_WARNING] = "1.4e-4"
    result = subprocess.run(
        [sys.executable, "-", str(left), str(right), "0"], input=program,
        env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert ("WARN: arm-to-arm worst=3.333e-01 exceeds warning "
            "band=1.400e-04" in result.stdout)
    assert "ceiling=1 is deliberately loose and expected to be tightened" in (
        result.stdout)


@pytest.mark.parametrize("job", _JOBS)
def test_nonhydrostatic_oracle_ceiling_tracks_boundary_floor_fix(job):
    body = job.read_text()
    assert re.search(r'nh\)\s+EXTRA="--nh --max-rel 3e-6"', body)
    if job == _JOB:
        assert re.search(
            r'nhmoist\)\s+EXTRA="--nh --moist --tracers --max-rel 3e-6"',
            body)


@pytest.mark.parametrize(("job", "arm_ceiling"), _ARM_CEILINGS)
def test_arm_delta_scan_excludes_bookkeeping(job, arm_ceiling, tmp_path):
    body = job.read_text()
    delimiter = "PYEOF" if job == _BATCHED_JOB else "EOF"
    full_program = body.rsplit(f"<<'{delimiter}'\n", 1)[1].split(
        f"\n{delimiter}", 1)[0]
    program = full_program.split("\nimport math, os\n", 1)[0]
    left = {"batched": False, "compiled": False,
            "residuals": {"face": {"field": {
                "rel": 1.0, "vacuous_constant_ic": False,
                "cells_over_10pct": 1}}}}
    right = {"batched": True, "compiled": False,
             "residuals": {"face": {"field": {
                 "rel": 1.5, "vacuous_constant_ic": True,
                 "cells_over_10pct": 1}}}}
    paths = [tmp_path / "left.json", tmp_path / "right.json"]
    for path, payload in zip(paths, (left, right)):
        path.write_text(json.dumps(payload))
    result = subprocess.run([sys.executable, "-", *map(str, paths), "0"],
                            input=program, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "at residuals.face.field.rel" in result.stdout
    env = os.environ.copy()
    env[arm_ceiling] = str(np.finfo(float).tiny)
    result = subprocess.run([sys.executable, "-", *map(str, paths), "0"],
                            input=full_program, env=env,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "exceeds" in result.stdout
    for invalid in ("nan", "inf"):
        env[arm_ceiling] = invalid
        result = subprocess.run(
            [sys.executable, "-", *map(str, paths), "0"],
            input=full_program, env=env, capture_output=True, text=True)
        assert result.returncode != 0
        assert "must be finite" in result.stdout
    for path, payload in zip(paths, (
            {"batched": False, "compiled": False},
            {"batched": True, "compiled": False})):
        path.write_text(json.dumps(payload))
    result = subprocess.run([sys.executable, "-", *map(str, paths), "0"],
                            input=program, capture_output=True, text=True)
    assert result.returncode != 0
    assert "no shared numeric" in result.stdout


@pytest.mark.parametrize("job", _JOBS)
def test_a_dead_backend_stops_the_job(job):
    body = job.read_text()
    assert 'exit "$RC"' in body, (
        "a backend that died leaves no result file; the job then found no "
        "shared metrics and finished successfully anyway")


def test_the_scorer_can_run_the_compiled_path():
    src = _SCORER.read_text()
    assert '"--jit"' in src, (
        "the scorer has no way to run the compiled step, so a parity claim "
        "about the shipped solver cannot be measured at all")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_make_jax_step")
    called = {n.func.attr for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "make_fv_dynamics_step_jit" in called, (
        "the jax adapter never builds the compiled step the model deploys")
    assert "fv_dynamics_step" in called, (
        "the eager path is gone; the established score would no longer be "
        "reproducible")


def test_main_forwards_the_batched_arm_to_the_jax_step():
    tree = ast.parse(_SCORER.read_text())
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    call = next(n for n in ast.walk(main)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "_make_jax_step")
    passed = {kw.arg: kw.value for kw in call.keywords}.get("batched")
    assert (isinstance(passed, ast.Attribute)
            and isinstance(passed.value, ast.Name)
            and passed.value.id == "args" and passed.attr == "batched"), (
        "main must forward args.batched; silently ignoring --batched would "
        "certify the wrong arm, which is worse than crashing")


def test_the_compiled_flag_is_refused_on_the_other_backend():
    """A flag that silently does nothing is how a lane goes unmeasured."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "full_step_oracle_parity", _SCORER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["full_step_oracle_parity"] = mod
    spec.loader.exec_module(mod)
    with pytest.raises(SystemExit, match="backend jax"):
        mod.main(["--backend", "numpy", "--jit"])


def test_batched_flag_is_refused_on_the_numpy_backend():
    """Face batching is a JAX implementation arm, not a NumPy option."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "full_step_oracle_parity", _SCORER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["full_step_oracle_parity"] = mod
    spec.loader.exec_module(mod)
    with pytest.raises(SystemExit, match="only in the JAX dynamics lane"):
        mod.main(["--backend", "numpy", "--batched"])


def test_batched_arm_dispatch_and_compiled_cache_identity(monkeypatch):
    """Both dispatches receive the arm, and compiled arms cannot alias."""
    import importlib.util
    import sys

    from legoesm.core import fv3_cgrid_phase_3d, fv3_duo_stepper, fv3_dynamics

    spec = importlib.util.spec_from_file_location(
        "full_step_oracle_parity", _SCORER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["full_step_oracle_parity"] = mod
    spec.loader.exec_module(mod)

    monkeypatch.setattr(fv3_duo_stepper, "build_jax_duo_stepper_context",
                        lambda ctx: ctx)
    monkeypatch.setattr(fv3_cgrid_phase_3d, "state_3d_to_jax",
                        lambda state: {"x": np.stack([f["x"] for f in state])})
    seen = []

    def result(jstate, jpress, jq):
        return {"state": jstate, "press": jpress, "q": jq}

    def eager(_ctx, jstate, jpress, *, q, **kw):
        seen.append(("eager", kw["batched"]))
        return result(jstate, jpress, q)

    def builder(_ctx, _km, **kw):
        seen.append(("jit", kw["batched"]))
        return lambda jstate, jpress, jq, **_dyn: result(jstate, jpress, jq)

    monkeypatch.setattr(fv3_dynamics, "fv_dynamics_step", eager)
    monkeypatch.setattr(fv3_dynamics, "make_fv_dynamics_step_jit", builder)
    state = [{"x": np.zeros(1)} for _ in range(6)]
    press = [{k: np.zeros(1) for k in ("ps", "pe", "peln", "pk", "pkz")}
             for _ in range(6)]
    q = [[np.zeros(1)] for _ in range(6)]

    mod._make_jax_step({}, batched=True)({}, state, press, q=q, km=1)
    keys = []
    for batched in (False, True):
        step = mod._make_jax_step({}, jit=True, batched=batched)
        step({}, state, press, q=q, km=1)
        keys.append(next(iter(inspect.getclosurevars(step).nonlocals["_cache"])))

    assert seen == [("eager", True), ("jit", False), ("jit", True)]
    assert keys[0] != keys[1]
    assert ("batched", False) in keys[0]
    assert ("batched", True) in keys[1]


def test_two_lanes_that_failed_differently_do_not_compare_equal():
    """The comparator's own defect, exercised.

    A tripwire left as NaN in one lane and leaked as an infinity in the other
    sits in the same cell. The masks match, so the cell was skipped and two
    fields that had diverged into different failure modes passed.
    """
    import importlib.util
    import sys

    stepper = _ROOT / "tests" / "grids" / "test_fv3_duo_stepper.py"
    spec = importlib.util.spec_from_file_location("_fv3_duo_stepper", stepper)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_fv3_duo_stepper"] = mod
    spec.loader.exec_module(mod)
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([1.0, np.inf, 3.0])
    with pytest.raises(AssertionError, match="non-finite VALUES differ"):
        mod._cmp(a, b, "field", 0.0)

    # and two lanes that failed the SAME way still compare fine
    mod._cmp(a, np.array([1.0, np.nan, 3.0]), "field", 0.0)


@pytest.mark.parametrize(("job", "selector"), _JIT_SELECTORS)
def test_the_job_certifies_the_compiled_path(job, selector):
    """Both jobs expose the compiled arm without changing their defaults."""
    body = job.read_text()
    assert ('JIT="${JIT:-0}"' in body and selector in body
            and re.search(r"--backend [^\n]*\$JFLAG", body)), (
        "the parity job cannot opt in to scoring the compiled solver")


def _load_scorer():
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        "full_step_oracle_parity", _SCORER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["full_step_oracle_parity"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_second_tracer_zero_is_the_default_and_modulated_is_the_port_field():
    """--tracer2 (2026-09-13): 'zero' keeps the cold-start deck's vacuous
    liq_wat; 'modulated' builds the same field the model's
    n_tracers=2 IC builds, on a tiny fake ctx; anything else is refused."""
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        lon_modulated_tracer)
    mod = _load_scorer()
    n, ng, km = 3, 1, 2
    m = n + 2 * ng
    rng = np.random.default_rng(1)
    sphum6 = []
    for _ in range(6):
        q = np.zeros((m, m, km))
        q[ng:ng + n, ng:ng + n, :] = rng.uniform(0.1, 1.0, (n, n, km))
        sphum6.append(q)
    lon6 = [rng.uniform(0, 2 * np.pi, (m, m)) for _ in range(6)]
    ctx = {"n": n, "ng": ng, "gs6": [{"agrid_lon": lo} for lo in lon6]}
    zero = mod.build_port_tracer_ic(sphum6)
    assert all(np.array_equal(f[0], s) and not f[1].any()
               for f, s in zip(zero, sphum6))
    modl = mod.build_port_tracer_ic(sphum6, ctx=ctx, second="modulated")
    for t in range(6):
        np.testing.assert_array_equal(modl[t][0], sphum6[t])
        np.testing.assert_array_equal(
            modl[t][1], lon_modulated_tracer(sphum6[t], lon6[t], n, ng, 1))
        assert modl[t][1][ng:ng + n, ng:ng + n, :].all()
    with pytest.raises(ValueError, match="needs the grid ctx"):
        mod.build_port_tracer_ic(sphum6, second="modulated")
    with pytest.raises(ValueError, match="unknown second tracer"):
        mod.build_port_tracer_ic(sphum6, ctx=ctx, second="shifted")


def _fake_deck(tmp_path, name, tracers, dnats_lines):
    d = tmp_path / name
    d.mkdir()
    (d / "field_table").write_text("".join(
        f' "TRACER", "atmos_mod", "{t}"\n           "units", "kg/kg" /\n'
        for t in tracers))
    (d / "input.nml").write_text(" &fv_core_nml\n" + "".join(
        f"       dnats = {v}\n" for v in dnats_lines) + " /\n")
    return str(d)


def test_deck_tracers_come_from_field_table_order_and_the_last_dnats(tmp_path):
    """The pinned deck's nml sets dnats twice (0 then 1); Fortran takes
    the LAST, so rainwat is inert and sphum/liq_wat advect. A deck adding
    cl/cl2 BEFORE rainwat advects four."""
    mod = _load_scorer()
    plain = _fake_deck(tmp_path, "plain", ["sphum", "liq_wat", "rainwat"], [0, 1])
    assert mod.read_deck_tracers(plain) == (("sphum", "liq_wat"), ("rainwat",))
    term = _fake_deck(tmp_path, "term",
                      ["sphum", "liq_wat", "cl", "cl2", "rainwat"], [0, 1])
    assert mod.read_deck_tracers(term) == (("sphum", "liq_wat", "cl", "cl2"),
                                           ("rainwat",))
    mod.resolve_deck_tracers(term, term)
    assert mod.ADVECTED_TRACERS == ("sphum", "liq_wat", "cl", "cl2")
    assert mod.NR_TRACERS == 4
    with pytest.raises(SystemExit, match="refusing to score across tracer sets"):
        mod.resolve_deck_tracers(plain, term)


def test_port_tracer_ic_follows_the_deck_names_including_the_terminator_pair(
        tmp_path):
    from legoesm.core.fv3_native_dcmip16_ic import dcmip16_terminator_six_face
    from legoesm.grids.factory import create_fv3_duo_grid
    mod = _load_scorer()
    term = _fake_deck(tmp_path, "term",
                      ["sphum", "liq_wat", "cl", "cl2", "rainwat"], [1])
    mod.resolve_deck_tracers(term, term)
    grid = create_fv3_duo_grid(12)
    ctx = grid.ctx_np
    n, ng, km = grid.n, grid.ng, 2
    m = n + 2 * ng
    sphum6 = [np.full((m, m, km), 1e-3) for _ in range(6)]
    q = mod.build_port_tracer_ic(sphum6, ctx=ctx)
    pairs = dcmip16_terminator_six_face(ctx, km)
    for t in range(6):
        assert len(q[t]) == 4
        np.testing.assert_array_equal(q[t][0], sphum6[t])
        assert not q[t][1].any()
        np.testing.assert_array_equal(q[t][2], pairs[t][0])
        np.testing.assert_array_equal(q[t][3], pairs[t][1])
    with pytest.raises(ValueError, match="cl/cl2 need the grid ctx"):
        mod.build_port_tracer_ic(sphum6)
    odd = _fake_deck(tmp_path, "odd", ["sphum", "dust", "rainwat"], [1])
    mod.resolve_deck_tracers(odd, odd)
    with pytest.raises(ValueError, match="has no port IC"):
        mod.build_port_tracer_ic(sphum6, ctx=ctx)


def test_a_lon_dependent_extra_scalar_breaks_the_zonal_mirror_tie():
    """Job 9766086: on the bump-free faces the derived map picked a
    transform at random and the terminator cl came out at rel 1.0 on
    three faces. The ambiguity is a MIRROR ALONG THE FACE'S LATITUDE
    INDEX on the rotated faces: the DCMIP jet is hemispherically
    symmetric, so the jet component (v there, varying along i) is
    invariant under i-reversal, and the component the reversal negates
    (u) is ~0. A synthetic face built that way, with an extra scalar that
    is NOT symmetric along i: the chosen transform must match the extra,
    and the returned cost must still be the prognostic residual."""
    mod = _load_scorer()
    n, km = 6, 2
    sym = np.cos(np.linspace(-1.0, 1.0, n))            # symmetric along i
    # v is staggered along i (n+1 rows): its OWN symmetric profile, not a
    # duplicated last row -- codex 2026-09-14: the duplicated row made v
    # asymmetric, so the prognostics alone already picked the mirror and
    # the test proved nothing about the extra scalar
    sym_v = np.cos(np.linspace(-1.0, 1.0, n + 1))
    v = np.tile(sym_v[:, None, None], (1, n, km))       # (n+1, n, km)
    u = np.zeros((n, n + 1, km))
    pt = np.tile(sym[:, None, None], (1, n, km)) + 300.0
    delp = np.full((n, n, km), 1.0e4)
    cl = np.linspace(0.0, 4e-6, n)[:, None, None] * np.ones((n, n, km))
    port = {"u": u, "v": v, "pt": pt, "delp": delp, "cl": cl}

    def as_oracle(a):  # port (i, j, k) -> oracle stored (k, j, i)
        return np.ascontiguousarray(np.moveaxis(a.transpose(1, 0, 2), -1, 0))
    # oracle tile = the port face mirrored along i ('fi': u -> -u, which
    # is invisible at u == 0); the prognostic fields cannot see the
    # mirror, cl can
    orc = {k: as_oracle(mod.DIHEDRAL["fi"](a) * (-1.0 if k == "u" else 1.0))
           for k, a in port.items()}
    r, transposed, nm, su, sv, per = mod.score_pair(port, orc)
    assert nm == "fi" and not transposed, (nm, transposed, per)
    assert per["cl"] < 1e-15
    assert r < 1e-15  # cost = prognostic residual, unchanged by the extra
    # and WITHOUT the extra the prognostic fields alone accept BOTH
    orc_plain = {k: a for k, a in orc.items() if k != "cl"}
    port_plain = {k: a for k, a in port.items() if k != "cl"}
    r0, _, nm0, _, _, per0 = mod.score_pair(port_plain, orc_plain)
    assert r0 < 1e-15 and nm0 in ("id", "fi")
    # NON-VACUITY: every prognostic field, u and v included, must accept
    # BOTH the identity and the mirror at the floor -- otherwise the
    # prognostics decide and the extra scalar is never needed
    for nm_t in ("id", "fi"):
        f = mod.DIHEDRAL[nm_t]
        su, sv = mod.DIHEDRAL_SIGNS[nm_t]
        got = {
            "u": mod.rel(su * f(port_plain["u"]), mod.oracle_ij(orc_plain["u"], False)),
            "v": mod.rel(sv * f(port_plain["v"]), mod.oracle_ij(orc_plain["v"], False)),
            "pt": mod.rel(f(port_plain["pt"]), mod.oracle_ij(orc_plain["pt"], False)),
            "delp": mod.rel(f(port_plain["delp"]), mod.oracle_ij(orc_plain["delp"], False)),
        }
        assert max(got.values()) < 1e-15, (nm_t, got)
    # and with the extra, the WRONG transform must score badly on it
    f = mod.DIHEDRAL["id"]
    assert mod.rel(f(port["cl"]), mod.oracle_ij(orc["cl"], False)) > 0.1
    # tie bookkeeping: two candidates fit without the extra, one with it
    assert per0["_ties"] == 2, per0
    assert per["_ties"] == 1, per


def test_terminator_tracer_floor_is_derived_from_its_cancellation():
    from legoesm.core.fv3_native_dcmip16_ic import TERM_QCLY
    mod = _load_scorer()
    want = 4.0 * np.finfo(np.float64).eps / (TERM_QCLY / 0.25)
    assert mod.TRACER_IC_MAX_REL["cl"] == want == mod.TRACER_IC_MAX_REL["cl2"]
    assert 1e-11 < want < 1e-10          # ~5.5e-11; measured IC residual 6.9e-12
    assert mod.TRACER_SCALE["cl2"] == TERM_QCLY
    assert "sphum" not in mod.TRACER_IC_MAX_REL   # sphum keeps the 1e-12 floor


def test_cl2_one_step_ceiling_is_twice_the_measured_ulp_envelope():
    """User call 2026-09-14: cl2 gates at 2x the one-ulp branch-flip
    envelope measured by tracer_ulp_sensitivity.py (job 9767368), on the
    qcly scale; every other tracer keeps the single --max-rel."""
    from legoesm.core.fv3_native_dcmip16_ic import TERM_QCLY
    mod = _load_scorer()
    assert mod.TRACER_STEP_MAX_REL == {
        "cl2": 2.0 * mod.CL2_ULP_ENVELOPE_ABS / TERM_QCLY}
    assert 1e-3 < mod.TRACER_STEP_MAX_REL["cl2"] < 2e-3
    assert mod.CL2_ULP_ENVELOPE_ABS == 3.297e-09


def test_arm_ksplit_nsplit_must_match_the_step_deck(tmp_path):
    """--k-split/--n-split used to be taken on faith. A deck whose
    namelist says k_split=2 refuses an arm asked to run k_split=1, before
    any oracle file is read; the last assignment wins as Fortran reads it."""
    mod = _load_scorer()
    decks = []
    for name in ("ic", "step"):
        d = tmp_path / name; d.mkdir()
        (d / "field_table").write_text(' "TRACER", "atmos_mod", "sphum"\n /\n')
        (d / "input.nml").write_text(
            " &fv_core_nml\n       k_split = 1\n       k_split = 2\n"
            "       n_split = 8\n       dnats = 0\n /\n")
        decks.append(str(d))
    assert mod._nml_int(mod._nml_text(decks[1]), "k_split") == 2
    with pytest.raises(SystemExit, match="deck has k_split=2"):
        mod.main(["--ic-run", decks[0], "--step-run", decks[1], "--k-split", "1",
                  "--n-split", "8", "--backend", "numpy"])


def test_nml_value_takes_the_last_assignment_on_a_line_too():
    """codex 2026-09-14: `k_split = 1, k_split = 2` on ONE line is legal
    Fortran and the last wins; re.search took the first."""
    mod = _load_scorer()
    text = " &fv_core_nml\n   k_split = 1, k_split = 2\n   n_split = 8 ! n_split = 3\n /\n"
    assert mod._nml_int(text, "k_split") == 2
    assert mod._nml_int(text, "n_split") == 8
    assert mod._nml_int("k_split = 3\nk_split = 4\n", "k_split") == 4
    # value on the NEXT line (codex round 2): the last assignment is 2
    assert mod._nml_int("k_split = 1\nk_split =\n   2\n", "k_split") == 2
    assert mod._nml_int("k_split = 1 ! k_split =\n 5\n", "k_split") == 1


def test_ulp_probe_pins_are_guarded_and_documented():
    """codex round 2: every pinned number in tracer_ulp_sensitivity.py has a
    config guard and a re-measurement path; the band is a named constant."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tracer_ulp_sensitivity",
        _ROOT / "scripts" / "validate" / "fv3_native" / "tracer_ulp_sensitivity.py")
    u = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(u)
    assert u.PINNED_CONFIG == (48, 5, 8, 1920.0)
    assert u.ENVELOPE_BAND == (0.5, 1.2)
    assert u.ORACLE_EDGE_NONCONSTANCY == 1.8177e-4
    assert callable(u.measure_oracle_edge_nonconstancy)
    with pytest.raises(SystemExit, match="pinned to C48"):
        u.main(["--assert-envelope", "--n", "24"])
    with pytest.raises(SystemExit, match="go together"):
        u.main(["--oracle-ic-run", "x"])


def test_bijection_is_chosen_on_the_extra_scalar_when_tiles_are_identical():
    """Six port faces with IDENTICAL prognostic fields (as the zonally
    symmetric base state makes them) and a distinct extra scalar each; the
    oracle tiles are a shuffle of them. Without the extra the pairing is a
    permutation-order accident and the runner-up bijection scores 1.0x the
    winner; with the extra the true shuffle is recovered and the margin is
    large. The REPORTED worst stays the prognostic cost."""
    mod = _load_scorer()
    n, km = 6, 2
    rng = np.random.default_rng(3)
    u = np.zeros((n, n + 1, km)); v = np.zeros((n + 1, n, km))
    pt = np.full((n, n, km), 300.0); delp = np.full((n, n, km), 1.0e4)

    def as_oracle(a):
        return np.ascontiguousarray(np.moveaxis(a.transpose(1, 0, 2), -1, 0))
    port = [{"u": u, "v": v, "pt": pt, "delp": delp,
             "cl": rng.uniform(1e-6, 4e-6, (n, n, km))} for _ in range(6)]
    shuffle = [3, 0, 5, 1, 4, 2]           # oracle tile t = port face shuffle[t]
    orc = [{k: as_oracle(a) for k, a in port[shuffle[t]].items()} for t in range(6)]
    cost, meta, perm, worst, per_field, _ = mod.derive_face_map(port, orc)
    assert [shuffle[perm[pf]] for pf in range(6)] == list(range(6)), perm
    assert worst < 1e-15
    assert per_field[0][perm[0]]["_bijection_margin"] > 1e3
    plain_p = [{k: a for k, a in f.items() if k != "cl"} for f in port]
    plain_o = [{k: a for k, a in f.items() if k != "cl"} for f in orc]
    _, _, _, worst0, pf0, _ = mod.derive_face_map(plain_p, plain_o)
    assert worst0 < 1e-15
    assert pf0[0][0]["_bijection_margin"] == 1.0
