"""Unit tests for the CMIP6/input4MIPs forcing downloader
(``scripts/data/download_cmip6_forcing.py``).

The DOWNLOAD itself (curl to ESGF) is a network side effect and is not
exercised here; these tests pin the pure resolution logic — Solr-doc URL
picking, latest-version selection, plan/dest-path construction, and the channel
registry — with the ESGF Solr query MOCKED, so a code regression in the
resolver is caught without touching the network.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_PY = _REPO / "scripts" / "data" / "download_cmip6_forcing.py"


def _mod():
    spec = importlib.util.spec_from_file_location("download_cmip6_forcing", _PY)
    m = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves sys.modules[cls.__module__].
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def test_registry_is_wellformed():
    """Every channel spec is complete; non-consumable channels explain the
    adapter they need (no silent 'download then it mysteriously fails')."""
    m = _mod()
    assert set(m.FORCINGS) == {
        "sst_sic", "ozone", "volcanic", "ghg", "solar", "aerosol"}
    for key, spec in m.FORCINGS.items():
        assert spec.channel == key
        assert spec.source_id and spec.variable_ids and spec.deck_flags
        # A channel that is NOT consumed raw MUST document its adapter.
        if not spec.directly_consumable:
            assert spec.adapter_note, f"{key} needs an adapter_note"
    # The two channels cross-checked against in-repo code are marked verified.
    assert m.FORCINGS["sst_sic"].verified and m.FORCINGS["ozone"].verified
    # SST/SIC + ozone + volcanic are consumed raw; ghg/solar/aerosol are not.
    assert m.FORCINGS["sst_sic"].directly_consumable
    assert not m.FORCINGS["ghg"].directly_consumable


def test_pick_https_requires_exact_httpserver_service():
    """_pick_https matches the service field EXACTLY == HTTPServer, so an HTTPS
    OPeNDAP/Globus-GUI URL is NOT mistaken for a downloadable file."""
    m = _mod()
    doc = {"url": [
        "http://esgf.example/thredds/dodsC/x.nc|application/opendap|OPENDAP",
        "https://g-abc.data.globus.org/x.nc|application/netcdf|HTTPServer",
    ]}
    assert m._pick_https(doc) == "https://g-abc.data.globus.org/x.nc"
    # HTTPS but the service is OPeNDAP on a globus host -> rejected (the old
    # 'data.globus.org substring' heuristic wrongly accepted this).
    assert m._pick_https(
        {"url": ["https://g.data.globus.org/x.nc|application/x-netcdf|OPENDAP"]}
    ) is None
    assert m._pick_https({"url": ["gsiftp://x|.|GridFTP"]}) is None
    assert m._pick_https({}) is None


def test_latest_version_keeps_only_newest():
    """A Solr File search can return several versions; only the newest is kept
    so a fetch never mixes vintages."""
    m = _mod()
    docs = [
        {"title": "a", "version": "20190101"},
        {"title": "b", "version": "20200101"},
        {"title": "c", "version": 20200101},   # int form, same newest
    ]
    kept = m._latest_version(docs)
    assert {d["title"] for d in kept} == {"b", "c"}
    assert m._latest_version([]) == []


def test_solr_query_tries_next_node_on_empty(monkeypatch):
    """A node that answers EMPTY is not authoritative (discovery is
    partitioned) — _solr_query tries the next node and returns its non-empty
    docs, rather than concluding 'not found' from the first empty answer."""
    m = _mod()
    import json as _json
    import types

    calls = []

    def fake_run(cmd, capture_output=True, text=True):
        url = cmd[-1]
        calls.append(url)
        node = url.split("?")[0]
        if node == m.ESGF_SEARCH_NODES[0]:
            body = {"response": {"docs": []}}          # first node: empty
        else:
            body = {"response": {"docs": [{"title": "hit.nc"}]}}  # next: has it
        return types.SimpleNamespace(returncode=0, stdout=_json.dumps(body))

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    monkeypatch.setattr(m.time, "sleep", lambda *_a: None)
    docs = m._solr_query(m.FORCINGS["ozone"], "vmro3")
    assert docs == [{"title": "hit.nc"}]
    # It actually consulted more than the first (empty) node.
    assert any(m.ESGF_SEARCH_NODES[1] in u for u in calls)


def test_resolve_files_uses_mocked_solr(monkeypatch):
    """resolve_files turns Solr docs into (filename, url, size), newest only."""
    m = _mod()
    fake_docs = [
        {"title": "tosbcs_old.nc", "version": "20180101", "size": 10,
         "url": ["https://g-x.data.globus.org/tosbcs_old.nc|.|HTTPServer"]},
        {"title": "tosbcs_new.nc", "version": "20200101", "size": 4242,
         "url": ["https://g-x.data.globus.org/tosbcs_new.nc|.|HTTPServer"]},
    ]
    monkeypatch.setattr(m, "_solr_query", lambda spec, var, search_url=None: fake_docs)
    out = m.resolve_files(m.FORCINGS["sst_sic"], "tosbcs")
    assert out == [("tosbcs_new.nc",
                    "https://g-x.data.globus.org/tosbcs_new.nc", 4242)]


def test_build_plan_reports_incomplete_channel_not_silent(monkeypatch, tmp_path):
    """A variable ESGF has no file for is reported in ``missing`` (channel is
    INCOMPLETE) — NOT silently dropped into a success (codex review). Files are
    grouped under <out>/<channel>/."""
    m = _mod()

    def fake_query(spec, variable_id, search_url=None):
        if variable_id == "siconcbcs":
            return []   # pretend ESGF returns nothing for this variable
        return [{"title": f"{variable_id}.nc", "version": "1", "size": 7,
                 "url": [f"https://g.data.globus.org/{variable_id}.nc"
                         f"|application/netcdf|HTTPServer"]}]

    monkeypatch.setattr(m, "_solr_query", fake_query)
    plan, missing = m.build_plan([m.FORCINGS["sst_sic"]], tmp_path)
    assert len(plan) == 1
    spec, var, url, size, dest = plan[0]
    assert var == "tosbcs"
    assert dest == tmp_path / "sst_sic" / "tosbcs.nc"
    # The unresolved SIC variable is surfaced, not swallowed.
    assert missing == [("sst_sic", "siconcbcs")]


def test_build_plan_shared_file_satisfies_multiple_vars(monkeypatch, tmp_path):
    """A single combined file (SAME url) serving both tosbcs and siconcbcs
    satisfies BOTH variables — planned ONCE, neither reported missing (the
    combined PCMDI SST/SIC file must not read as a self-collision)."""
    m = _mod()
    shared = [{"title": "amip_bcs.nc", "version": "1", "size": 9,
               "url": ["https://g.data.globus.org/amip_bcs.nc"
                       "|application/netcdf|HTTPServer"]}]
    monkeypatch.setattr(
        m, "_solr_query", lambda spec, variable_id, search_url=None: shared)
    plan, missing = m.build_plan([m.FORCINGS["sst_sic"]], tmp_path)
    assert len(plan) == 1 and not missing   # one download, nothing missing
    assert [p[4] for p in plan] == [tmp_path / "sst_sic" / "amip_bcs.nc"]


def test_build_plan_different_source_filename_conflict(monkeypatch, tmp_path):
    """Same filename from DIFFERENT sources (different url) is a real conflict:
    the second is dropped and, having no other file, recorded incomplete."""
    m = _mod()

    def fake_query(spec, variable_id, search_url=None):
        # Both vars claim 'clash.nc' but from different URLs.
        return [{"title": "clash.nc", "version": "1", "size": 1,
                 "url": [f"https://g.data.globus.org/{variable_id}/clash.nc"
                         f"|application/netcdf|HTTPServer"]}]

    monkeypatch.setattr(m, "_solr_query", fake_query)
    plan, missing = m.build_plan([m.FORCINGS["sst_sic"]], tmp_path)
    assert len(plan) == 1                          # first kept
    assert len({p[4] for p in plan}) == 1          # no overwrite
    assert missing == [("sst_sic", "siconcbcs")]   # the dropped one is flagged


def test_build_plan_conflict_marks_incomplete_even_with_other_files(
        monkeypatch, tmp_path):
    """A variable that plans one file AND loses another to a different-source
    conflict is STILL incomplete (a dropped time slice) — planned_here>0 must
    not hide the missing slice (codex review)."""
    m = _mod()

    def fake_query(spec, variable_id, search_url=None):
        if variable_id == "tosbcs":
            return [{"title": "clash.nc", "version": "1", "size": 1,
                     "url": ["https://g.data.globus.org/A/clash.nc"
                             "|application/netcdf|HTTPServer"]}]
        # siconcbcs: one unique file + one that clashes (different url) with the
        # already-planned tosbcs clash.nc.
        return [
            {"title": "sic_only.nc", "version": "1", "size": 1,
             "url": ["https://g.data.globus.org/B/sic_only.nc"
                     "|application/netcdf|HTTPServer"]},
            {"title": "clash.nc", "version": "1", "size": 1,
             "url": ["https://g.data.globus.org/B/clash.nc"
                     "|application/netcdf|HTTPServer"]},
        ]

    monkeypatch.setattr(m, "_solr_query", fake_query)
    plan, missing = m.build_plan([m.FORCINGS["sst_sic"]], tmp_path)
    # siconcbcs planned sic_only.nc but lost clash.nc -> still reported missing.
    assert ("sst_sic", "siconcbcs") in missing


def test_solr_query_inconclusive_when_a_node_fails(monkeypatch):
    """If one node answers empty but another FAILS, the result is inconclusive
    (raise) — we must not report a definitive 'not found' from a partial view."""
    m = _mod()
    import json as _json
    import types

    def fake_run(cmd, capture_output=True, text=True):
        node = cmd[-1].split("?")[0]
        if node == m.ESGF_SEARCH_NODES[0]:
            return types.SimpleNamespace(   # clean empty
                returncode=0, stdout=_json.dumps({"response": {"docs": []}}))
        return types.SimpleNamespace(returncode=7, stdout="")   # network fail

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    monkeypatch.setattr(m.time, "sleep", lambda *_a: None)
    with pytest.raises(RuntimeError, match="inconclusive"):
        m._solr_query(m.FORCINGS["ozone"], "vmro3")


def test_incomplete_channel_dry_run_exits_nonzero(monkeypatch, tmp_path, capsys):
    """--dry-run returns non-zero when a selected channel is incomplete, so a
    scripted caller can't mistake a partial plan for a full one."""
    m = _mod()

    def fake_query(spec, variable_id, search_url=None):
        if variable_id == "siconcbcs":
            return []
        return [{"title": f"{variable_id}.nc", "version": "1", "size": 3,
                 "url": [f"https://g.data.globus.org/{variable_id}.nc"
                         f"|application/netcdf|HTTPServer"]}]

    monkeypatch.setattr(m, "_solr_query", fake_query)
    rc = m.main(["--channels", "sst_sic", "--out-dir", str(tmp_path),
                 "--dry-run"])
    assert rc == 1
    assert "INCOMPLETE" in capsys.readouterr().err


def test_cli_list_and_dry_run(monkeypatch, tmp_path, capsys):
    """--list exits 0; a COMPLETE --dry-run resolves a plan (mocked) without
    downloading and returns 0."""
    m = _mod()
    assert m.main(["--list"]) == 0
    assert "input4MIPs forcing channels" in capsys.readouterr().out

    monkeypatch.setattr(
        m, "_solr_query",
        lambda spec, var, search_url=None: [
            {"title": f"{var}.nc", "version": "1", "size": 3,
             "url": [f"https://g.data.globus.org/{var}.nc"
                     f"|application/netcdf|HTTPServer"]}])
    # download_one must NOT be called in a dry run.
    monkeypatch.setattr(m, "download_one", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("download_one called during --dry-run")))
    rc = m.main(["--channels", "ozone", "--out-dir", str(tmp_path), "--dry-run"])
    assert rc == 0   # ozone has one variable (vmro3), fully resolved
    assert "Plan: 1 files" in capsys.readouterr().out


def test_download_one_skips_when_size_matches(monkeypatch, tmp_path):
    """download_one skips an existing file whose size matches, and never shells
    out to curl in that case (the size-verify guard)."""
    m = _mod()
    dest = tmp_path / "x.nc"
    dest.write_bytes(b"1234")            # 4 bytes
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("curl called though size already matched")))
    assert m.download_one("https://x", dest, expected_size=4) is True


def test_out_dir_required_without_list():
    m = _mod()
    assert m.main(["--channels", "ozone"]) == 2   # missing --out-dir
