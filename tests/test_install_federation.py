"""Unit tests for ``scripts/experiment/install_federation.py``.

The helper exists because plain ``pip`` cannot resolve the uv-workspace
inter-member deps (``legoesm-core~=0.1.0`` etc. are unpublished — only
``[tool.uv.sources]`` maps them, and pip ignores that).  These tests pin the
pure resolution logic — the federation DAG read from the real
``packages/*/pyproject.toml``, the transitive closure (incl. extras), and the
emitted pip command — without touching the network or installing anything.
"""
from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "experiment" / "install_federation.py"


def _load():
    spec = importlib.util.spec_from_file_location("install_federation", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load()


def test_all_members_discovered_from_disk():
    # Every packages/<m> with a pyproject is a member; core leads.
    on_disk = {p.name for p in (REPO / "packages").iterdir()
               if (p / "pyproject.toml").is_file()}
    assert set(mod.ALL_MEMBERS) == on_disk
    assert mod.ALL_MEMBERS[0] == "core"


def test_member_deps_match_pyproject():
    # member_deps() must mirror the legoesm-* hard deps actually declared.
    for m in mod.ALL_MEMBERS:
        with (REPO / "packages" / m / "pyproject.toml").open("rb") as fh:
            proj = tomllib.load(fh)["project"]
        declared = {
            d for s in proj.get("dependencies", [])
            if (d := mod._dist_to_member(s)) and d != m
        }
        assert mod.member_deps(m) == declared, m


def test_core_depends_on_nothing():
    assert mod.member_deps("core") == set()


def test_component_closure_is_core_only():
    # The four Earth-system components are mutually independent (contract #2):
    # standalone install pulls ONLY core.
    for comp in ("atmosphere", "ocean", "land", "ice"):
        assert mod.transitive_closure([comp]) == ["core", comp]


def test_orchestration_cluster_pulls_whole_cycle():
    # coupler/ml/tools are a mutual cycle: any one drags all members in.
    closure = mod.transitive_closure(["coupler"])
    assert set(closure) == set(mod.ALL_MEMBERS)
    assert closure[0] == "core"  # deterministic, core-first


def test_extras_expand_closure():
    # atmosphere[ml] needs legoesm-ml installed too (and ml's cycle) — without
    # this the editable command would emit atmosphere[ml] with no source for
    # legoesm-ml.  Bare atmosphere must NOT pull ml.
    bare = mod.transitive_closure(["atmosphere"])
    assert bare == ["core", "atmosphere"]  # exact closed set
    with_ml = mod.transitive_closure(["atmosphere"], {"atmosphere": ("ml",)})
    assert "ml" in with_ml
    # atmosphere[ml] -> ml, and ml's cycle drags in the whole federation.
    assert set(with_ml) == set(mod.ALL_MEMBERS)


def test_declared_extras_reflect_pyproject():
    # atmosphere declares only [ml]; the four components are otherwise narrow;
    # the meta package carries the aggregate extras (dev/all/...).
    assert "ml" in mod.declared_extras("atmosphere")
    assert "ml" not in mod.declared_extras("land")
    assert "ml" not in mod.declared_extras("ice")
    assert {"dev", "all"} <= mod.declared_extras("meta")


def test_extras_do_not_apply_transitively():
    # Extras only expand the originally-requested members, never members pulled
    # in transitively (matches pip's "extras apply to named specs only").
    # tools is pulled transitively by ml; tools' own extras must not fire.
    closure = mod.transitive_closure(["ml"], {"ml": ()})
    assert set(closure) == set(mod.ALL_MEMBERS)


def test_aliases_normalize():
    assert mod._normalize("cryosphere") == "ice"
    assert mod._normalize("legoesm-atmosphere") == "atmosphere"
    assert mod._normalize("ATM") == "atmosphere"


def test_unknown_member_raises():
    with pytest.raises(SystemExit):
        mod._normalize("banana")


def test_editable_command_for_component():
    cmd = mod.build_editable_cmd(["core", "atmosphere"], {}, include_meta=False)
    assert cmd[:4] == [mod.sys.executable, "-m", "pip", "install"]
    assert "-e" in cmd
    assert str(REPO / "packages" / "core") in cmd
    assert str(REPO / "packages" / "atmosphere") in cmd
    # No meta when include_meta is False.
    assert str(REPO) not in cmd


def test_editable_command_attaches_extras_only_to_requested():
    cmd = mod.build_editable_cmd(
        ["core", "atmosphere"], {"atmosphere": ("ml",)}, include_meta=False
    )
    joined = " ".join(cmd)
    assert str(REPO / "packages" / "atmosphere") + "[ml]" in joined
    # core must NOT get the extra (it has no [ml] extra).
    assert str(REPO / "packages" / "core") + "[" not in joined


def test_editable_command_includes_meta():
    cmd = mod.build_editable_cmd(["core"], {"meta": ("dev",)}, include_meta=True)
    assert f"{REPO}[dev]" in " ".join(cmd)


def test_wheelhouse_command_uses_find_links_and_dist_names():
    wh = Path("/tmp/wh")
    cmd = mod.build_wheelhouse_cmd(["atmosphere"], {"atmosphere": ("ml",)}, wh)
    assert "--find-links" in cmd
    assert str(wh) in cmd
    assert "legoesm-atmosphere[ml]" in cmd
    # Dist names, never editable paths, in wheelhouse mode.
    assert "-e" not in cmd


def test_wheelhouse_command_includes_meta_for_all():
    # Regression: wheelhouse --all must emit the root meta spec (legoesm[dev]),
    # not just the eight members — else the dev tools + console script are missing.
    wh = Path("/tmp/wh")
    cmd = mod.build_wheelhouse_cmd(
        list(mod.ALL_MEMBERS), {"meta": ("dev",)}, wh, include_meta=True
    )
    assert "legoesm[dev]" in cmd
    cmd_nometa = mod.build_wheelhouse_cmd(["atmosphere"], {}, wh, include_meta=False)
    assert not any(c == "legoesm" or c.startswith("legoesm[") for c in cmd_nometa)


def test_canon_pep685():
    assert mod._canon("legoesm_Core") == "legoesm-core"
    assert mod._canon("legoesm.atmosphere") == "legoesm-atmosphere"
    assert mod._dist_to_member("legoesm_core~=0.1.0") == "core"


def test_main_drops_undeclared_extra_but_keeps_declared(capsys):
    # atmosphere land --extras ml: land has no ml extra -> warned + dropped, but
    # atmosphere keeps [ml]; command still emitted (rc 0 on dry-run).
    rc = mod.main(["atmosphere", "land", "--extras", "ml", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr()
    assert "does not declare extra(s) ml" in out.err
    assert "packages/atmosphere[ml]" in out.out


def test_main_errors_when_no_target_declares_extra():
    # land alone with --extras ml: nothing to attach -> hard error (argparse exit 2).
    with pytest.raises(SystemExit):
        mod.main(["land", "--extras", "ml", "--dry-run"])
    with pytest.raises(SystemExit):
        mod.main(["atmosphere", "--extras", "bogus", "--dry-run"])


def test_extras_matching_is_canonical(capsys):
    # PEP 685: --extras ML (or M_L) must match a declared `ml`, not be dropped.
    rc = mod.main(["atmosphere", "--extras", "ML", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr()
    assert "does not declare" not in out.err
    assert "packages/atmosphere[ml]" in out.out  # emitted canonical


def test_main_all_wheels_includes_meta(monkeypatch, capsys):
    # End-to-end through main()'s --wheels branch: --all --extras dev must emit the
    # root meta spec legoesm[dev], not just the members.  AND dry-run must perform
    # NO build side effects — build_wheels() must not be called and no temp
    # wheelhouse created (regression: build used to run before the dry-run guard).
    def _boom(*a, **k):
        raise AssertionError("build_wheels must not run on a dry-run")

    def _no_tmp(*a, **k):
        raise AssertionError("no TemporaryDirectory on a dry-run")

    monkeypatch.setattr(mod, "build_wheels", _boom)
    monkeypatch.setattr(mod.tempfile, "TemporaryDirectory", _no_tmp)
    rc = mod.main(["--all", "--extras", "dev", "--wheels", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "--find-links <temp-wheelhouse>" in out
    assert "legoesm[dev]" in out


def test_main_single_wheels_dry_run_no_build(monkeypatch, capsys):
    # Same regression guard for the non-all single-component wheels dry-run.
    monkeypatch.setattr(
        mod, "build_wheels",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no build on dry-run")))
    monkeypatch.setattr(
        mod.tempfile, "TemporaryDirectory",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no tmp on dry-run")))
    rc = mod.main(["atmosphere", "--wheels", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "legoesm-atmosphere" in out
    assert "--find-links" in out


def test_dist_to_member_parsing():
    assert mod._dist_to_member("legoesm-core~=0.1.0") == "core"
    assert mod._dist_to_member("legoesm-ml~=0.1.0; extra == 'ml'") == "ml"
    assert mod._dist_to_member("legoesm-atmosphere[ml]~=0.1.0") == "atmosphere"
    assert mod._dist_to_member("jax>=0.4.35") is None
    assert mod._dist_to_member("legoesm-nonsuch") is None  # not a real member
