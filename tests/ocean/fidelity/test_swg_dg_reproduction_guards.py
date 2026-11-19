"""Direct tests for the cross-deck and short-ladder reproduction guards."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import textwrap
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest
import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "scripts/validate/ocean_fidelity/testcases"
FIXTURES = ROOT / "tests/ocean/fixtures/swg_dg"
PACKAGE = ROOT / "scripts/data/nemo_swg_double_gyre"
DOCS = ROOT / "docs/ocean/fidelity/testcases"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


deck = _load("swg_dg_deck_guard_test",
             TOOLS / "nemo_testcase_l1_swg_dg_deck_guard.py")
short = _load("swg_dg_short_ladder_test",
              TOOLS / "nemo_testcase_l1_swg_dg_short_ladder.py")
inventory = _load("swg_dg_namelist_inventory_test",
                  TOOLS / "nemo_testcase_l1_swg_namelist_inventory.py")


def _rehash(run: Path) -> None:
    (run / "manifest.sha256").write_text("".join(
        f"{hashlib.sha256((run / name).read_bytes()).hexdigest()}  {name}\n"
        for name in ("namelist_ref", "namelist_cfg")))


def _fixture_run(tmp_path: Path, *, echo: str | None = None,
                 reference: str | None = None,
                 config: str | None = None) -> Path:
    """A NEMO run directory holding the reference run's three deck files."""
    run = tmp_path / "nemo_run"
    run.mkdir()
    (run / "output.namelist.dyn").write_text(
        (FIXTURES / "output.namelist.dyn").read_text() if echo is None else echo)
    (run / "namelist_ref").write_text(
        (FIXTURES / "namelist_ref").read_text()
        if reference is None else reference)
    (run / "namelist_cfg").write_text(
        (FIXTURES / "namelist_cfg.maddison_3yr").read_text()
        if config is None else config)
    (run / "ocean.output").write_text(
        "NEMO completed; nambdy values are not printed by this executable\n")
    _rehash(run)
    return run


def _edit(text: str, pattern: str, replacement: str) -> str:
    edited, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    assert count == 1, (pattern, count)
    return edited


def _echo() -> str:
    return (FIXTURES / "output.namelist.dyn").read_text()


def _reference() -> str:
    return (FIXTURES / "namelist_ref").read_text()


def _config() -> str:
    return (FIXTURES / "namelist_cfg.maddison_3yr").read_text()


def test_gpu_entry_points_request_the_cuda_platform_name(monkeypatch):
    # JAX 0.10 refuses JAX_PLATFORMS=gpu on a CUDA-only install (it also tries ROCm).
    checkout = _load("swg_dg_checkout_test", TOOLS / "nemo_testcase_checkout.py")
    assert checkout.JAX_PLATFORMS_FOR == {"cpu": "cpu", "gpu": "cuda"}
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    short._configure_backend("gpu", "0")
    assert os.environ["JAX_PLATFORMS"] == "cuda"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    short._configure_backend("cpu", None)
    assert os.environ["JAX_PLATFORMS"] == "cpu"


def test_gpu_short_ladder_loads_its_helpers_after_selecting_the_gpu(
        monkeypatch, tmp_path):
    """The scorer selects the GPU, then loads round7 and composition.

    Emulated on CPU: JAX is initialised here, so reporting ``gpu`` is what an
    initialised CUDA backend does.  A helper that bootstraps at import with
    the CPU default refuses ("requested cpu, selected gpu"); the scorer must
    instead reach its first input, the missing one-rank restart.
    """
    import jax
    from legoesm.core.precision import get_policy, set_policy
    from legoesm.ocean.fidelity import provenance

    for name in ("JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    # The scorer stamps its commit; a development tree is legitimately dirty.
    monkeypatch.setattr(provenance, "git_sha", lambda **_: "0" * 40)
    policy = get_policy()
    try:
        with pytest.raises(RuntimeError, match="missing single-rank restart"):
            short.run(tmp_path / "no_ladder", composition="outer-scan",
                      backend="gpu", gpu_device="0", plant=False)
        # Control: the CPU form reaches the same first input.
        monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
        with pytest.raises(RuntimeError, match="missing single-rank restart"):
            short.run(tmp_path / "no_ladder", composition="outer-scan",
                      backend="cpu", gpu_device=None, plant=False)
    finally:
        set_policy(policy)


def test_resolved_nemo_and_legoesm_decks_cover_every_compiled_read(tmp_path):
    result = deck.run(_fixture_run(tmp_path), 1095)
    assert result["compiled_inventory_groups"] == 27
    assert result["compiled_inventory_fields"] == 336
    assert result["card_compared_fields"] == 147
    assert result["nemo_only_pinned_fields"] == 189
    assert result["card_rows"] == 158
    assert result["comparison_rows"] == 347
    assert [row["field"] for row in result["differences"]] == [
        "sn_cfctl.l_runstat", "nn_stock"]
    assert all(row["allowlisted"] and row["reason"]
               for row in result["differences"])
    assert result["all_non_allowlisted_values_equal"]
    assert result["nemo_sources"]["ocean_output_cross_checks"] == {
        "nambdy": {
            "status": "not_printed",
            "matched_fields": [],
            "note": (
                "NEMO's ocean.output contains no machine-readable value for "
                "this echo-omitted group; the content-hashed namelist_ref/"
                "namelist_cfg precedence is authoritative"),
        }}
    nambdy_sources = {
        source for name, source in result["resolved_sources"].items()
        if name.startswith("nambdy.")}
    assert nambdy_sources == {"namelist_ref"}
    pinned = [row for row in result["rows"] if row["kind"] != "card"]
    assert all(row["equal"] and row["reason"] for row in pinned)
    assert {row["field"].split(".")[0] for row in pinned} == {
        "nammpp", "namtsd", "namsbc_rnf", "namc1d", "namclo", "namsbc_apr",
        "namsbc_blk", "namsbc_cpl", "namsbc_flx", "namsbc_fwb",
        "namsbc_iif", "namsbc_ssr", "namsbc_wave"}
    assert sorted(result["card_only_fields"]) == sorted(deck.CARD_ONLY)
    assert len(deck.CARD_ONLY) == 19


def test_cross_deck_guard_rejects_changed_echoed_physical_value(tmp_path):
    echo = _echo().replace(
        "RN_TAU= 0.10000000000000001", "RN_TAU= 0.11000000000000000")
    with pytest.raises(RuntimeError, match="rn_tau_pa"):
        deck.run(_fixture_run(tmp_path, echo=echo), 1095)


def test_cross_deck_guard_rejects_changed_unechoed_ln_bdy(tmp_path):
    reference = _edit(_reference(), r"^(\s*ln_bdy\s*=\s*)\.false\.",
                      r"\1.true.")
    with pytest.raises(RuntimeError, match="ln_bdy nemo=True"):
        deck.run(_fixture_run(tmp_path, reference=reference), 1095)


def test_cross_deck_guard_refuses_compiled_field_missing_from_every_source(
        tmp_path):
    reference = _edit(_reference(), r"^\s*ln_bdy\s*=.*\n", "")
    with pytest.raises(RuntimeError,
                       match="unresolved compiled-read field nambdy.ln_bdy"):
        deck.run(_fixture_run(tmp_path, reference=reference), 1095)


def test_cross_deck_guard_applies_nemo_cfg_over_ref_precedence(tmp_path):
    reference = _edit(_reference(), r"^(\s*ln_bdy\s*=\s*)\.false\.",
                      r"\1.true.")
    config = _config() + "&nambdy\n ln_bdy = .false.\n/\n"
    result = deck.run(
        _fixture_run(tmp_path, reference=reference, config=config), 1095)
    assert result["resolved_sources"]["nambdy.ln_bdy"] == "namelist_cfg"


# The review's plants (nn_hls, jpni, ln_tsd_init, rn_hrnf changed or deleted;
# the wind stress deleted) plus the other ways a field can escape the guard.
_ECHO_PLANTS = {
    "nn_hls_changed": (r"^ NN_HLS=2 ", " NN_HLS=1 ", r"nammpp\.nn_hls nemo=1"),
    "nn_hls_deleted": (r"^ NN_HLS=.*\n", "",
                       "echoed group nammpp lacks compiled-read field nn_hls"),
    "jpni_changed": (r"^ JPNI=0 ", " JPNI=8 ", r"nammpp\.jpni nemo=8"),
    "jpni_deleted": (r"^ JPNI=.*\n", "",
                     "echoed group nammpp lacks compiled-read field jpni"),
    "ln_tsd_init_changed": (r"^ LN_TSD_INIT=F,", " LN_TSD_INIT=T,",
                            r"namtsd\.ln_tsd_init nemo=True"),
    "ln_tsd_init_deleted": (
        r"^ LN_TSD_INIT=.*\n", "",
        "echoed group namtsd lacks compiled-read field ln_tsd_init"),
    "rn_hrnf_changed": (r"^ RN_HRNF=  15\.0+ ", " RN_HRNF=  99.000000000000000 ",
                        r"namsbc_rnf\.rn_hrnf nemo=99\.0"),
    "rn_hrnf_deleted": (
        r"^ RN_HRNF=.*\n", "",
        "echoed group namsbc_rnf lacks compiled-read field rn_hrnf"),
    "rn_tau_deleted": (
        r"^ RN_TAU=.*\n", "",
        "echoed group namusr_def lacks compiled-read field rn_tau"),
    "extra_echoed_field": (r"^&NAMUSR_DEF\n", "&NAMUSR_DEF\n RN_FOO=1          ,\n",
                           r"has field\(s\) \['rn_foo'\] outside"),
    "unparsed_echo_line": (r"^&NAMMPP\n", "&NAMMPP\n garbage\n",
                           "group nammpp has line"),
}


@pytest.mark.parametrize("plant", sorted(_ECHO_PLANTS))
def test_deck_guard_refuses_every_echo_plant(tmp_path, plant):
    pattern, replacement, refusal = _ECHO_PLANTS[plant]
    echo = _edit(_echo(), pattern, replacement)
    with pytest.raises(RuntimeError, match=refusal):
        deck.run(_fixture_run(tmp_path, echo=echo), 1095)


def test_deck_guard_refuses_an_extra_echoed_group(tmp_path):
    echo = _echo() + "&NAMSBC_BLK\n LN_NCAR=T,\n /\n"
    with pytest.raises(RuntimeError, match="echoed groups differ"):
        deck.run(_fixture_run(tmp_path, echo=echo), 1095)


def test_deck_guard_pins_an_unechoed_group_override(tmp_path):
    config = _config() + "&namsbc_blk\n rn_zqt = 3.\n/\n"
    with pytest.raises(RuntimeError, match=r"namsbc_blk\.rn_zqt nemo=3\.0"):
        deck.run(_fixture_run(tmp_path, config=config), 1095)


def test_deck_guard_refuses_a_repeated_group_block(tmp_path):
    config = _config() + "&nammpp\n/\n"
    with pytest.raises(RuntimeError, match="repeats the nammpp group"):
        deck.run(_fixture_run(tmp_path, config=config), 1095)


def _parsed_fixture():
    parse = deck.parse_namelist_groups
    return (parse(FIXTURES / "output.namelist.dyn"),
            parse(FIXTURES / "namelist_ref"),
            parse(FIXTURES / "namelist_cfg.maddison_3yr"))


def test_deck_guard_refuses_a_read_that_is_neither_compared_nor_registered():
    registry = deck.load_nemo_only_registry()
    registry["fields"].pop("nammpp.nn_hls")
    namelist = deck._load_driver().resolved_daily_namelist(1095)  # noqa: SLF001
    with pytest.raises(RuntimeError, match=r"neither compared .*nammpp\.nn_hls"):
        deck.compare_resolved(*_parsed_fixture(), namelist, registry=registry)


def test_deck_guard_refuses_a_card_field_without_a_read_or_registration(
        monkeypatch):
    monkeypatch.delitem(deck.CARD_ONLY, "key_RK3")
    namelist = deck._load_driver().resolved_daily_namelist(1095)  # noqa: SLF001
    with pytest.raises(RuntimeError, match=r"card field\(s\) neither .*key_RK3"):
        deck.compare_resolved(*_parsed_fixture(), namelist)


def test_deck_guard_selects_float64_itself(tmp_path):
    """Run from a shell without JAX_ENABLE_X64: the guard must still pass."""
    run = _fixture_run(tmp_path)
    env = {key: value for key, value in os.environ.items()
           if key != "JAX_ENABLE_X64"}
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, str(TOOLS / "nemo_testcase_l1_swg_dg_deck_guard.py"),
         "--nemo-run-dir", str(run), "--days", "1095",
         "--json", str(tmp_path / "deck_guard.json")],
        env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == (
        "SWG_DG_DECK_GUARD PASS compiled_reads=336 groups=27 "
        "card_compared=147 nemo_only_pinned=189 comparison_rows=347 "
        "allowed_differences=2")
    report = json.loads((tmp_path / "deck_guard.json").read_text())
    assert report["jax_enable_x64"].startswith("selected by the checkout bootstrap")


def test_import_time_bootstrap_does_not_make_collection_order_dependent():
    """Importing a tool as a library (pytest collection) in a float32 process
    must not raise; only a command-line run enforces float64."""
    env = {key: value for key, value in os.environ.items()
           if key != "JAX_ENABLE_X64"}
    env["JAX_PLATFORMS"] = "cpu"
    library = subprocess.run(
        [sys.executable, "-c",
         "import importlib.util, sys\n"
         "import jax\n"
         "assert not jax.config.jax_enable_x64\n"
         "spec = importlib.util.spec_from_file_location('guard', sys.argv[1])\n"
         "guard = importlib.util.module_from_spec(spec)\n"
         "spec.loader.exec_module(guard)\n"
         "print('IMPORTED')\n",
         str(TOOLS / "nemo_testcase_l1_swg_dg_deck_guard.py")],
        env=env, capture_output=True, text=True)
    assert library.returncode == 0, library.stderr
    assert library.stdout.strip().splitlines()[-1] == "IMPORTED"
    enforced = subprocess.run(
        [sys.executable, "-c",
         "import sys\n"
         "import jax\n"
         "sys.path.insert(0, sys.argv[1])\n"
         "from nemo_testcase_checkout import bootstrap_checkout\n"
         "bootstrap_checkout(sys.argv[2], enforce_x64=True)\n",
         str(TOOLS), str(TOOLS / "nemo_testcase_l1_swg_dg_deck_guard.py")],
        env=env, capture_output=True, text=True)
    assert enforced.returncode != 0
    assert "JAX x64 is disabled" in enforced.stderr


@pytest.mark.parametrize("relative", (
    "nemo_testcase_l1_swg_dg_restart_identity.py",
    "nemo_testcase_l1_swg_dg_climate.py",
))
def test_annual_and_climate_tools_select_float64_themselves(tmp_path, relative):
    """The same misleading float32 refusal, one step later, is closed too.

    Without x64 these tools reached the card's geometry check in float32 and
    refused with 'grid differs'; they must now get past the bootstrap and
    stop only at their first missing input.
    """
    env = {key: value for key, value in os.environ.items()
           if key != "JAX_ENABLE_X64"}
    env["JAX_PLATFORMS"] = "cpu"
    code = (
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location('tool', sys.argv[1])\n"
        "tool = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(tool)\n"
        "import jax\n"
        "from legoesm.ocean.fidelity.nemo_testcase_recipe import "
        "SWG_MADDISON_NAMELIST, build_swg_card\n"
        "build_swg_card(SWG_MADDISON_NAMELIST)\n"
        "print('X64', jax.config.jax_enable_x64)\n")
    result = subprocess.run(
        [sys.executable, "-c", code, str(TOOLS / relative)],
        env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "X64 True"


_SYNTHETIC_PPSRC = {
    "main.f90": """
        MODULE main_mod
        CONTAINS
           SUBROUTINE nemo_gcm
              CALL init_a
              IF( ln_x ) CALL init_b       ! guarded calls still count
              zz = fun_f( 1 )
              CALL gen( 2 )
              WRITE(numout,*) 'READ(numnam_ref, namz) is only text here'
           END SUBROUTINE nemo_gcm
        END MODULE main_mod
    """,
    "a.f90": """
        MODULE a_mod
        CONTAINS
           SUBROUTINE init_a
              NAMELIST/nama/ first, second, &   ! continued
                 &           third
              READ(numnam_ref(MAX(INDEX(numnam_ref,"&nama "),1):),nama,IOSTAT=ios);CALL ctl(ios)
              IF(lwm) WRITE( numond, nama )
           END SUBROUTINE init_a
        END MODULE a_mod
    """,
    "b.f90": """
        MODULE b_mod
           NAMELIST/namb/ only_field
        CONTAINS
           SUBROUTINE init_b
              READ( numnam_cfg, namb, IOSTAT=ios )
           END SUBROUTINE init_b
           SUBROUTINE orphan
              NAMELIST/namc/ never_read
              READ( numnam_ref, namc )
           END SUBROUTINE orphan
        END MODULE b_mod
    """,
    "c.f90": """
        MODULE c_mod
           INTERFACE gen
              MODULE PROCEDURE gen_impl
           END INTERFACE
        CONTAINS
           SUBROUTINE gen_impl( k )
              NAMELIST/namd/ dee
              READ( numnam_ref, NML=namd )
              CALL load_nml
           END SUBROUTINE gen_impl
           INTEGER FUNCTION fun_f( k )
              NAMELIST/namf/ eff
              READ( numnam_cfg, namf )
              fun_f = k
           END FUNCTION fun_f
           SUBROUTINE load_nml
              NAMELIST/nl1/ itest
              READ( cltest, nl1 )
           END SUBROUTINE load_nml
        END MODULE c_mod
    """,
}


def _synthetic_tree(root: Path, files: dict[str, str]) -> Path:
    root.mkdir()
    for name, text in files.items():
        (root / name).write_text(textwrap.dedent(text))
    return root


def test_inventory_derives_every_reachable_read_and_nothing_else(tmp_path):
    derived = inventory.derive_read_inventory(
        _synthetic_tree(tmp_path / "ppsrc", _SYNTHETIC_PPSRC))
    groups = derived["groups"]
    assert sorted(groups) == ["nama", "namb", "namd", "namf"]
    assert groups["nama"]["fields"] == ["first", "second", "third"]
    assert groups["nama"]["read_sites"] == [["a.f90", "init_a", 7]]
    assert groups["nama"]["echo_written"]
    assert groups["namb"] == {"fields": ["only_field"],
                              "read_sites": [["b.f90", "init_b", 6]],
                              "echo_written": False}
    assert groups["namd"]["read_sites"] == [["c.f90", "gen_impl", 9]]
    assert groups["namf"]["read_sites"] == [["c.f90", "fun_f", 14]]
    assert derived["excluded_reads"] == [{
        "group": "nl1", "file": "c.f90", "routine": "load_nml", "line": 19,
        "unit": "cltest"}]
    assert derived["field_count"] == 6


def test_inventory_refuses_a_namelist_read_it_cannot_resolve(tmp_path):
    files = dict(_SYNTHETIC_PPSRC)
    files["a.f90"] = files["a.f90"].replace(
        "IF(lwm) WRITE( numond, nama )", "READ( numnam_cfg, namz )")
    with pytest.raises(ValueError, match="unresolved namelist READ"):
        inventory.derive_read_inventory(
            _synthetic_tree(tmp_path / "ppsrc", files))


def test_inventory_cli_refuses_a_build_that_reads_differently(
        tmp_path, capsys):
    reference = tmp_path / "reference.json"
    tree = _synthetic_tree(tmp_path / "ppsrc", _SYNTHETIC_PPSRC)
    reference.write_text(json.dumps(inventory.derive_read_inventory(tree)))
    assert inventory.main(["--source-root", str(tree),
                           "--reference", str(reference)]) == 0
    assert capsys.readouterr().out.startswith(
        "SWG_DG_NAMELIST_INVENTORY PASS groups=4 fields=6 excluded_reads=1")
    (tree / "b.f90").write_text((tree / "b.f90").read_text().replace(
        "NAMELIST/namb/ only_field", "NAMELIST/namb/ only_field, new_field"))
    assert inventory.main(["--source-root", str(tree),
                           "--reference", str(reference)]) == 2
    assert "namb.fields" in capsys.readouterr().err


@pytest.mark.skipif(
    "SWG_DG_PPSRC" not in os.environ,
    reason=("set SWG_DG_PPSRC to a built tests/SWG_DG/BLD/ppsrc/nemo tree to "
            "re-derive the committed compiled-read inventory"))
def test_live_build_derives_the_committed_inventory():
    derived = inventory.derive_read_inventory(Path(os.environ["SWG_DG_PPSRC"]))
    assert inventory.inventory_differences(
        derived, inventory.load_reference_inventory()) == []


def test_short_ladder_scores_every_step_and_field_and_localizes_plant():
    masks = {name: np.ones((2, 2), dtype=bool) for name in "TUV"}
    restarts = {
        step: {name: np.full((2, 2), float(step))
               for name in ("sshn", "un", "vn")}
        for step in range(1, 11)
    }
    trajectory = [
        {name: np.full((2, 2), float(step))
         for name in ("ssh_m", "u_m_s", "v_m_s")}
        for step in range(1, 11)
    ]

    def exact(reference, candidate, mask, *, name):
        unequal = int(np.count_nonzero(
            np.asarray(reference).view(np.uint64)[mask]
            != np.asarray(candidate).view(np.uint64)[mask]))
        return {"name": name, "bit_identical": unequal == 0,
                "bit_unequal": unequal, "max_abs": float(
                    np.max(np.abs(reference[mask] - candidate[mask])))}

    rows = short.score_trajectory(
        restarts, trajectory, masks, exact_row=exact)
    assert len(rows) == 30
    assert all(row["bit_identical"] for row in rows)
    trajectory[3]["v_m_s"][1, 0] = np.nextafter(4.0, np.inf)
    rows = short.score_trajectory(
        restarts, trajectory, masks, exact_row=exact)
    debts = [row for row in rows if not row["bit_identical"]]
    assert [(row["step"], row["field"], row["bit_unequal"])
            for row in debts] == [(4, "v_m_s", 1)]


def test_short_ladder_reference_comparator_localizes_departure(tmp_path, capsys):
    baseline = json.loads(short.REFERENCE_BASELINE.read_text())
    paths = {}
    for composition in ("outer-scan", "materialized"):
        form = baseline["forms"][composition]
        signatures = short._expected_signatures(form)
        rows = []
        for step in range(1, 11):
            for field in ("ssh_m", "u_m_s", "v_m_s"):
                bit, numeric, signed, maximum = signatures[(step, field)]
                rows.append({
                    "step": step, "field": field,
                    "bit_unequal": bit, "numeric_unequal": numeric,
                    "signed_zero_only": signed, "max_abs": maximum,
                })
        payload = {
            "schema": "swg-dg-short-restart-ladder-v1",
            "composition": composition,
            "all_exact": form["all_exact"],
            "rows": rows,
        }
        path = tmp_path / f"{composition}.json"
        path.write_text(json.dumps(payload))
        paths[composition] = path

    args = [
        "--compare-reference", "--outer-report", str(paths["outer-scan"]),
        "--materialized-report", str(paths["materialized"]),
    ]
    assert short.main(args) == 0
    output = capsys.readouterr().out
    assert "outer-scan same as reference" in output
    assert "materialized same as reference" in output

    materialized = json.loads(paths["materialized"].read_text())
    planted = next(row for row in materialized["rows"]
                   if row["step"] == 4 and row["field"] == "v_m_s")
    planted["bit_unequal"] += 1
    paths["materialized"].write_text(json.dumps(materialized))
    assert short.main(args) == 1
    output = capsys.readouterr().out
    assert "materialized departs from reference at step=4 field=v_m_s" in output


_EXACT_PIN = re.compile(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+-]+)")


def _exact_pins(path: Path) -> dict[str, str]:
    """Every requirement line must be ``name==version``; nothing looser."""
    pins = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        match = _EXACT_PIN.fullmatch(line)
        assert match, f"{path.name}: not an exact pin: {line!r}"
        name = canonicalize_name(match.group(1))
        assert name not in pins, f"{path.name}: {name} pinned twice"
        pins[name] = match.group(2)
    return pins


def test_environment_reconstruction_files_are_machine_readable_and_pinned():
    explicit = (PACKAGE / "nemo-build-linux-64-explicit.txt").read_text()
    assert "# platform: linux-64" in explicit
    assert "@EXPLICIT" in explicit
    urls = [line for line in explicit.splitlines() if line.startswith("https://")]
    assert len(urls) == 122
    assert all("conda.anaconda.org/conda-forge/" in url for url in urls)
    assert all(re.fullmatch(r"https://.+#[0-9a-f]{32}", url) for url in urls)
    unhashed_urls = [url.rsplit("#", 1)[0] for url in urls]
    assert hashlib.sha256(
        ("\n".join(unhashed_urls) + "\n").encode()).hexdigest() == (
            "deed23744c963e5c5787cd0731517a3aebd6475872a3c8938c3566eb0d80d56f")
    assert any("openmpi-5.0.10-h67ed482_1" in url for url in urls)
    assert any("netcdf-fortran-4.6.3-mpi_openmpi_h8f6c4f5_0" in url
               for url in urls)

    readable = yaml.safe_load(
        (PACKAGE / "nemo-build-environment.yml").read_text())
    assert readable["name"] == "swg-dg-nemo-build"
    assert "openmpi=5.0.10" in readable["dependencies"]
    assert "netcdf-fortran=4.6.3" in readable["dependencies"]

    environment = json.loads(
        (DOCS / "swg_double_gyre_environment.json").read_text())
    python = environment["python"]
    main_file = PACKAGE / "python-gpu-requirements.txt"
    selftest_file = PACKAGE / "python-selftest-requirements.txt"
    # Exact pins equal to the committed reference versions, in files whose
    # bytes are the reviewed ones: a changed version, a loosened specifier,
    # an added or a removed pin all fail here.
    for path, key, digest in (
            (main_file, "packages", "requirements_file_sha256"),
            (selftest_file, "selftest_packages",
             "selftest_requirements_file_sha256")):
        pins = _exact_pins(path)
        assert pins == {canonicalize_name(name): version
                        for name, version in python[key].items()}
        assert hashlib.sha256(path.read_bytes()).hexdigest() == python[digest]
    pinned = _exact_pins(main_file) | _exact_pins(selftest_file)
    assert len(_exact_pins(main_file)) == 40
    assert len(_exact_pins(selftest_file)) == 5
    for name, version in (("jax", "0.10.0"), ("jax-cuda12-plugin", "0.10.0"),
                          ("nvidia-cuda-runtime-cu12", "12.9.79"),
                          ("numpy", "2.4.4"), ("typing-extensions", "4.15.0"),
                          ("certifi", "2026.4.22"), ("pytest", "9.0.3"),
                          ("pyyaml", "6.0.3")):
        assert pinned[name] == version

    # The environment running this test IS the one the pins describe, and the
    # pins are a closed set under installed requirement metadata.
    installed = {name: metadata.version(name) for name in pinned}
    assert installed == pinned
    pending = list(pinned)
    visited = set()
    missing_required_edges = []
    while pending:
        parent = pending.pop()
        if parent in visited:
            continue
        visited.add(parent)
        for raw in metadata.distribution(parent).requires or ():
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            child = canonicalize_name(requirement.name)
            if child not in pinned:
                missing_required_edges.append((parent, child, raw))
            else:
                pending.append(child)
    assert visited == set(pinned)
    assert missing_required_edges == []

    assert environment["nemo"]["repository"] == (
        "https://forge.nemo-ocean.eu/nemo/nemo.git")
    assert environment["hardware"]["minimum_gpu_memory_bytes"] == 32 * 2**30
    assert environment["host_requirements"]["glibc_minimum"] == "2.34"

    costs = json.loads((DOCS / "swg_double_gyre_reference_costs.json").read_text())
    assert costs["nemo_three_year"]["wall_seconds"] == 22_918
    assert (costs["legoesm_gpu_three_year_daily_mean"]["wall_seconds"]
            == 8_117)
    assert costs["legoesm_cpu_one_day"]["wall_seconds"] == pytest.approx(
        233.93357684463263)
    assert all(
        len(costs[name]["source_artifact_sha256"]) == 64
        for name in ("nemo_three_year", "nemo_three_year_fresh_rebuild",
                     "legoesm_gpu_three_year_daily_mean",
                     "legoesm_cpu_one_day", "movie_year1"))


def selftest_row(entry: dict) -> str:
    """One guide row of the committed self-test measurement."""
    result = f"{entry['passed']} passed"
    if entry["skipped"]:
        result += f", {entry['skipped']} skipped"
    if entry["failed"]:
        result += f", {entry['failed']} failed"
    return (f"| `{Path(entry['path']).name}` | {result} | "
            f"{entry['wall_seconds']:.0f} s | "
            f"{entry['peak_rss_kb'] / 1048576:.1f} GiB |")


def climate_row(window: str, model: str, metrics: dict) -> str:
    def cell(stats):
        return f"{stats['mean']:.4g} ({stats['min']:.4g}–{stats['max']:.4g})"
    return (f"| {window} | {model} | {cell(metrics['basin_mean_ke_m2_s2'])} | "
            f"{cell(metrics['basin_mean_eke_m2_s2'])} |")


def test_stranger_runbook_matches_its_committed_measurements():
    runbook = (DOCS / "swg_double_gyre_reproduction.md").read_text()
    required = (
        "https://github.com/climate-federation/legoESM",
        "--branch fidelity/swg-double-gyre-standalone",
        'test "$(git -C "$LEGOESM_ROOT" rev-parse HEAD)" = "$LEGOESM_COMMIT"',
        "https://forge.nemo-ocean.eu/nemo/nemo.git",
        "dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796",
        "nemo-build-linux-64-explicit.txt",
        "python-gpu-requirements.txt",
        "python-selftest-requirements.txt",
        "python=3.13.0=h9ebbce0_101_cp313",
        "typing-extensions==4.15.0",
        "certifi==2026.4.22",
        "glibc 2.34 or newer",
        "ldd --version | head -1",
        "GNU binutils `ar`",
        "580.126.20",
        "`>=575.57.08`",
        "`>=525`",
        "LD_LIBRARY_PATH",
        "cuDNN 9.21.0 support matrix",
        "Nobody has yet run",
        "--python \"$PY\"",
        "--conda-prefix \"$NEMO_PREFIX\" --execute",
        "--short-ladder dg_ladder",
        "--composition outer-scan",
        "--composition materialized",
        "--backend gpu --gpu-device 0 --composition outer-scan",
        "--compare-reference",
        "SWG_DG_NAMELIST_INVENTORY PASS groups=27 fields=336 excluded_reads=2",
        ("SWG_DG_DECK_GUARD PASS compiled_reads=336 groups=27 "
         "card_compared=147 nemo_only_pinned=189 comparison_rows=347 "
         "allowed_differences=2"),
        "NEMO_RESTART_TILES_MATCH_REFERENCE tiles=96",
        "matches_2026_10_13_reference_build",
        "SWG_DG_CLIMATE_REPORT_WRITTEN window=1-365",
        "agreement_test=none",
        "414,545",
        "nemo_testcases_l1_swg_dg_round7_receipt.md",
        "reference-only evidence",
        "Exactness policy (decided by the maintainer on 2026-10-13)",
        "recorded debt",
        "What the 2026-10-13 fresh reproduction covered",
        "peak_bytes_in_use",
        "100 GiB",
        "SWG_DG_RESTART_IDENTITY PASS day=365",
        "SWG_DG_MOVIE COMPLETE frames=365",
        "cross-toolchain acceptance bar",
        ("Across both windows, the measured maxima are\n"
         "`2.286230926795696e-11` for kinetic energy and\n"
         "`1.402270660699756e-11` for eddy kinetic energy."),
        "full-restart identity",
    )
    for text in required:
        assert text in runbook, text
    for text in ("352 passed", "pending the user", "pending user",
                 "PASS year 1", "export PATH=", "agrees to `1e-11`",
                 "sole source of", "/data/abyssal", "/home/dbalwada",
                 "SELFTEST_", "CLIMATE_TABLE"):
        assert text not in runbook, text
    assert re.findall(r"^## ([1-9])\. ", runbook, re.MULTILINE) == list(
        "123456789")

    costs = json.loads((DOCS / "swg_double_gyre_reference_costs.json").read_text())
    for name in ("nemo_three_year", "nemo_three_year_fresh_rebuild",
                 "legoesm_gpu_three_year_daily_mean", "movie_year1"):
        assert f"{costs[name]['wall_seconds']:,} s" in runbook, name

    for window, name in (
            ("days 1–365", "swg_dg_climate_year1_reference_report.json"),
            ("days 366–1095", "swg_dg_climate_years2_3_reference_report.json")):
        report = json.loads((DOCS / name).read_text())
        for model in ("NEMO", "legoESM"):
            assert climate_row(window, model, report["metrics"][model]) in runbook

    selftest = json.loads((DOCS / "swg_dg_selftest_reference.json").read_text())
    loop = re.search(r"for test_file in \\\n(.*?); do", runbook, re.S).group(1)
    listed = re.findall(r"(tests/\S+\.py)", loop)
    assert listed == [entry["path"] for entry in selftest["files"]]
    for entry in selftest["files"]:
        assert selftest_row(entry) in runbook, entry["path"]
    total = selftest["total"]
    assert (f"| Total | {total['passed']} passed, {total['skipped']} skipped |"
            in runbook)
