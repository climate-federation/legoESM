"""The named DEPHY-format field-campaign cases, and the file each one names.

These cases are downloaded rather than vendored, so the two things that can go
wrong silently are a registry entry pointing at the wrong file and a variable
rename that changes what a number MEANS. Both are pinned here.

The load tests skip when the cache is empty; populate it with

    python scripts/data/fetch_dephy_cases.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
xr = pytest.importorskip("xarray")

from legoesm.atmosphere.forcing.scm.dephy_scm import (  # noqa: E402
    DEPHY_SCM_CASES,
    load_dephy_case,
    resolve_dephy_case_path,
)

_ROOT = Path(__file__).resolve().parents[4]


def _cached(name: str) -> bool:
    return Path(resolve_dephy_case_path(name)).is_file()


ALL_CASES = sorted(DEPHY_SCM_CASES)


# --- registry hygiene -------------------------------------------------------

def test_unknown_case_is_rejected_by_both_entry_points():
    """Dispatch hardening: a typo must not resolve to some other case."""
    with pytest.raises(ValueError, match="Unknown DEPHY case"):
        resolve_dephy_case_path("not_a_case")
    with pytest.raises(ValueError, match="Unknown DEPHY case"):
        load_dephy_case("not_a_case")


@pytest.mark.parametrize("name", ALL_CASES)
def test_filename_is_the_one_the_url_serves(name):
    """A registry whose filename and URL disagree caches under a name nothing
    ever looks for, so the case reads as permanently missing."""
    spec = DEPHY_SCM_CASES[name]
    assert spec.url.rsplit("/", 1)[-1] == spec.filename
    assert spec.url.startswith("https://")


@pytest.mark.parametrize("name", ALL_CASES)
def test_every_case_declares_its_provenance(name):
    """A case with no reference cannot be checked against anything."""
    spec = DEPHY_SCM_CASES[name]
    for field in ("reference", "regime", "campaign", "note"):
        assert getattr(spec, field).strip(), f"{name}: empty {field}"


def test_cache_path_follows_the_shared_forcing_root(monkeypatch, tmp_path):
    """One environment variable must move every forcing file this repo reads,
    the gSAM decks and these alike."""
    monkeypatch.setenv("LEGOESM_LES_FORCING", str(tmp_path))
    got = Path(resolve_dephy_case_path("mpace"))
    assert got == tmp_path / "dephy" / DEPHY_SCM_CASES["mpace"].filename


def test_missing_file_names_the_fetch_command(monkeypatch, tmp_path):
    """The error a user actually hits first has to say what to run."""
    monkeypatch.setenv("LEGOESM_LES_FORCING", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="fetch_dephy_cases.py"):
        load_dephy_case("mpace")


# --- the cases themselves ---------------------------------------------------

@pytest.mark.parametrize("name", ALL_CASES)
def test_every_cached_case_loads_with_physical_profiles(name):
    if not _cached(name):
        pytest.skip(f"{name} not cached; run scripts/data/fetch_dephy_cases.py")
    case = load_dephy_case(name, nlev=32)
    T = np.asarray(case.T_profile)
    assert T.shape == (32,)
    assert np.all(np.isfinite(T))
    assert 150.0 < T.min() and T.max() < 340.0, (name, T.min(), T.max())
    if case.q_v_profile is not None:
        q = np.asarray(case.q_v_profile)
        assert np.all(np.isfinite(q)) and q.min() >= 0.0 and q.max() < 0.1
    assert case.duration_seconds > 0.0
    assert 0.0 < case.sigma_top < 1.0
    assert -90.0 <= case.latitude_deg <= 90.0


@pytest.mark.skipif(not _cached("mpace"), reason="mpace not cached")
def test_mpace_is_the_klein_2009_arctic_case():
    """Guards against the registry pointing at a different campaign: M-PACE is
    at 71.75 N over a 274.01 K ocean, 12 h from 17 UTC on 9 October 2004."""
    case = load_dephy_case("mpace", nlev=32)
    assert case.latitude_deg == pytest.approx(71.75, abs=0.01)
    assert case.surface_type == "ocean"
    assert case.duration_seconds == pytest.approx(12.0 * 3600.0)
    assert "2004-10-09" in case.start_date


@pytest.mark.skipif(not _cached("sandu_ref"), reason="sandu_ref not cached")
def test_sandu_is_a_multi_day_subtropical_transition():
    """The Sc-to-Cu transition is defined by following the boundary layer for
    days, not hours; a few-hour file would be a different case."""
    case = load_dephy_case("sandu_ref", nlev=32)
    assert case.duration_seconds > 48.0 * 3600.0
    assert 15.0 < case.latitude_deg < 40.0


@pytest.mark.skipif(
    not all(_cached(n) for n in ("sandu_ref", "sandu_fast", "sandu_slow")),
    reason="sandu composites not cached")
def test_the_three_sandu_composites_are_different_cases():
    """REF/FAST/SLOW must not all resolve to the same file.

    All three pairs, not just one: with only REF-vs-FAST checked, SLOW could
    resolve to REF and the test would still pass.
    """
    import itertools
    cases = {n: load_dephy_case(n, nlev=24)
             for n in ("sandu_ref", "sandu_fast", "sandu_slow")}
    for a, b in itertools.combinations(cases, 2):
        sst_a = float(np.asarray(cases[a].forcing.T_s(48 * 3600.0)))
        sst_b = float(np.asarray(cases[b].forcing.T_s(48 * 3600.0)))
        assert abs(sst_a - sst_b) > 0.1, (
            f"{a} and {b} reach the same sea-surface temperature at 48 h "
            f"({sst_a:.2f} vs {sst_b:.2f} K); the composites differ by how "
            "fast the surface warms, so identical values mean both names "
            "resolved to one file")


# --- COMBLE: the renamed file must mean the same thing ----------------------

@pytest.mark.skipif(not _cached("comble"), reason="comble not cached")
def test_comble_rename_lands_the_right_variable_at_the_right_level():
    """Elementwise, not "the range looks plausible".

    For every model level the loaded temperature must lie between the raw
    file's own temperatures at the two levels that bracket that level's
    pressure. That is interpolation-agnostic -- it does not re-implement the
    loader -- but it fails if the rename grabbed the wrong variable, the wrong
    time slice, or mapped the column onto the wrong vertical coordinate,
    because any of those puts values outside the local bracket.
    """
    raw = xr.open_dataset(resolve_dephy_case_path("comble"), decode_times=False)
    assert "pa" not in raw.variables and "pressure" in raw.variables, (
        "the raw file is expected to use the intercomparison's own names")
    for var in ("pressure", "temp", "u", "v"):
        extra = {d: n for d, n in zip(raw[var].dims, raw[var].shape)
                 if d != "lev" and n != 1}
        assert not extra, (
            f"{var} has non-singleton non-level dimensions {extra}; "
            "flattening below would mix samples from different times or "
            "locations into the bracket")
    p_raw = np.asarray(raw["pressure"].values).ravel()
    order = np.argsort(p_raw)
    p_raw = p_raw[order]

    case = load_dephy_case("comble", nlev=48)
    p_scm = np.asarray(case.pressure_full, dtype=np.float64)
    for var, loaded in (("temp", np.asarray(case.T_profile)),
                        ("u", np.asarray(case.u_profile)),
                        ("v", np.asarray(case.v_profile))):
        raw_v = np.asarray(raw[var].values).ravel()[order]
        for k, p_k in enumerate(p_scm):
            if not (p_raw[0] <= p_k <= p_raw[-1]):
                continue                       # outside the file; clamped
            j = int(np.searchsorted(p_raw, p_k))
            lo, hi = sorted((raw_v[max(j - 1, 0)], raw_v[min(j, raw_v.size - 1)]))
            assert lo - 1e-6 <= loaded[k] <= hi + 1e-6, (
                var, k, p_k, loaded[k], lo, hi)


@pytest.mark.skipif(not _cached("comble"), reason="comble not cached")
def test_comble_has_its_surface_temperature_forcing():
    """The rename that actually mattered.

    COMBLE's whole physics is heat and moisture picked up from a sea surface
    that warms down the trajectory. The loader maps a prescribed skin
    temperature only from `ts_forc`, and this file calls the series `ts`, so
    before the rename the case came back with NO surface forcing at all and
    still integrated to finite, innocuous-looking profiles. The published
    specification holds the skin temperature at 247.0 K for the first two
    hours, so that value is the check.
    """
    case = load_dephy_case("comble", nlev=32)
    assert case.forcing.prescribe == "T_s"
    t0 = float(np.asarray(case.forcing.T_s(0.0)))
    assert t0 == pytest.approx(247.0, abs=0.5), t0
    late = float(np.asarray(case.forcing.T_s(18.0 * 3600.0)))
    assert late > t0 + 5.0, (
        f"the surface must warm down the trajectory: {t0:.1f} -> {late:.1f} K")


@pytest.mark.skipif(not _cached("comble"), reason="comble not cached")
def test_comble_loads_under_strict():
    """The file ships ERA5 nudging profiles with every nudging switch at 0.

    Renaming them to the DEPHY names would make the loader treat reference
    data as an active forcing it cannot represent -- and `strict=True` would
    then refuse a case that is perfectly runnable. Loading strictly is the
    non-vacuous version of that claim.
    """
    case = load_dephy_case("comble", nlev=24, strict=True)
    assert case.unsupported == ()


def test_a_rename_whose_source_is_missing_is_an_error(tmp_path):
    """Skipping an absent source silently un-does the conversion, and the file
    then reads through the loader's defaults while still producing a column."""
    from legoesm.atmosphere.forcing.scm.dephy_scm import load_dephy_scm_case
    if not _cached("mpace"):
        pytest.skip("mpace not cached")
    with pytest.raises(ValueError, match="does not contain"):
        load_dephy_scm_case(resolve_dephy_case_path("mpace"),
                            rename={"no_such_variable": "pa"})


@pytest.mark.skipif(not _cached("mpace"), reason="mpace not cached")
def test_a_rename_that_would_overwrite_is_an_error():
    from legoesm.atmosphere.forcing.scm.dephy_scm import load_dephy_scm_case
    with pytest.raises(ValueError, match="overwrite"):
        load_dephy_scm_case(resolve_dephy_case_path("mpace"),
                            rename={"ta": "qv"})


# --- the forcing has to be CONNECTED, not merely present -------------------

@pytest.mark.parametrize("name", ALL_CASES)
def test_forcing_channels_carry_something(name):
    """A Lagrangian column with its forcing silently unwired drifts on its
    initial condition and still looks finite. Each case is checked for the
    channels its own specification defines."""
    if not _cached(name):
        pytest.skip(f"{name} not cached")
    case = load_dephy_case(name, nlev=32)
    f = case.forcing
    live = []
    for chan in ("u_geo", "v_geo", "subsidence_w", "theta_adv", "qv_adv"):
        fn = getattr(f, chan, None)
        if fn is None:
            continue
        if np.abs(np.asarray(fn(0.0))).max() > 0.0:
            live.append(chan)
    for chan in ("w_th_s", "w_qv_s", "T_s"):
        fn = getattr(f, chan, None)
        if fn is not None and abs(float(np.asarray(fn(0.0)))) > 0.0:
            live.append(chan)
    assert live, f"{name}: every forcing channel is absent or identically zero"
    assert f.prescribe in ("fluxes", "T_s"), (
        f"{name}: no surface forcing at all ({f.prescribe!r}); a column with "
        "no surface boundary is not the case")


@pytest.mark.skipif(not _cached("mpace"), reason="mpace not cached")
def test_mpace_wind_forcing_gap_is_declared_not_hidden():
    """M-PACE nudges its wind, which SCMForcing cannot represent, so the loaded
    column has no wind forcing and no Coriolis. That is a real limitation of
    this case here; it is pinned so it cannot quietly become a surprise."""
    case = load_dephy_case("mpace", nlev=24)
    assert case.forcing.u_geo is None and case.forcing.v_geo is None
    assert case.forcing.f_c == 0.0
    assert "nud" in " ".join(case.unsupported).lower()
    note = DEPHY_SCM_CASES["mpace"].note.lower()
    assert "nudging" in note and "no wind forcing" in note


# --- the fetch script -------------------------------------------------------

def _fetcher():
    path = _ROOT / "scripts" / "data" / "fetch_dephy_cases.py"
    spec = importlib.util.spec_from_file_location("_fetch_dephy", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_fetch_dephy"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_fetcher_lists_without_downloading(capsys):
    assert _fetcher().main(["--list"]) == 0
    out = capsys.readouterr().out
    for name in ALL_CASES:
        assert name in out


def test_fetcher_rejects_an_unknown_case():
    """A selector matching nothing is a hard error, never a silent no-op."""
    with pytest.raises(SystemExit, match="unknown case"):
        _fetcher().main(["--only", "not_a_case"])


# --- the cases have to RUN, not just parse ---------------------------------

def _physics_for(case):
    """Louis turbulence + Morrison, with the bulk heat coefficient zeroed when
    the case prescribes its surface fluxes (the SCM refuses both at once,
    because it would double-count the deck's sensible heat)."""
    from legoesm.atmosphere.physics import (
        ConvectionConfig, GravityWaveDragConfig, MicrophysicsConfig,
        PhysicsConfig, RadiationConfig, TurbulenceConfig,
    )
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="louis"),
        microphysics=MicrophysicsConfig(scheme="morrison"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    if case.forcing.prescribe == "fluxes":
        turb = cfg.turbulence
        sub = turb.louis
        turb = turb._replace(
            louis=sub._replace(surface=sub.surface._replace(Ch_neutral=0.0)))
        cfg = cfg._replace(turbulence=turb)
    return cfg


@pytest.mark.parametrize("name", ALL_CASES)
def test_every_cached_case_integrates(name):
    """A case that loads but cannot be stepped is not a usable case.

    Ten minutes of model time with real physics. This is an INTEGRATION smoke
    test and nothing more: plenty of sign and unit errors stay finite and
    bounded over ten minutes in a forced boundary layer, so a pass here is not
    evidence the forcing is right. What checks the forcing is
    `test_forcing_channels_carry_something` and the per-case value tests.
    """
    if not _cached(name):
        pytest.skip(f"{name} not cached; run scripts/data/fetch_dephy_cases.py")
    jax.config.update("jax_enable_x64", True)
    case = load_dephy_case(name, nlev=32, dt=60.0)
    scm = case.create_scm(physics_config=_physics_for(case), dt=60.0)
    state, _ = scm.run(nsteps=10, save_every=10)
    T = np.asarray(state.T.data).ravel()
    assert np.all(np.isfinite(T)), name
    assert 150.0 < T.min() and T.max() < 340.0, (name, T.min(), T.max())


@pytest.mark.skipif(not _cached("mpace"), reason="mpace not cached")
def test_mpace_forms_cloud_because_it_is_a_cloudy_boundary_layer():
    """M-PACE's initial state is a cloud-topped mixed layer, so a column that
    stays dry has lost the case. Distinguishes a real load from a plausible
    one: the profile can look fine and still be the wrong quantity."""
    jax.config.update("jax_enable_x64", True)
    case = load_dephy_case("mpace", nlev=32, dt=60.0)
    scm = case.create_scm(physics_config=_physics_for(case), dt=60.0)
    state, _ = scm.run(nsteps=10, save_every=10)
    q_c = np.asarray(state.tracers["q_c"].data)
    assert float(q_c.max()) > 1.0e-5, (
        f"expected liquid in the M-PACE mixed layer, got {float(q_c.max()):.2e}")


@pytest.mark.parametrize("name", ALL_CASES)
def test_every_case_pins_an_immutable_source(name):
    """A branch URL serves whatever the branch says today.

    A forcing file that changes under a fixed case name is an uncontrolled
    comparison, so the URL is pinned to a commit and the bytes to a digest.
    """
    spec = DEPHY_SCM_CASES[name]
    assert len(spec.sha256) == 64 and all(c in "0123456789abcdef"
                                          for c in spec.sha256), spec.sha256
    assert "/master/" not in spec.url and "/main/" not in spec.url, (
        f"{name}: {spec.url} points at a branch, not a commit")


def test_a_corrupted_cached_file_is_reported(tmp_path, monkeypatch, capsys):
    """A cached file whose bytes no longer match the registry must not be used
    silently -- the case would run, and be a different experiment."""
    monkeypatch.setenv("LEGOESM_LES_FORCING", str(tmp_path))
    dest = tmp_path / "dephy" / DEPHY_SCM_CASES["mpace"].filename
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"not a netcdf file")
    assert _fetcher().main(["--only", "mpace"]) == 1
    assert "does not match the registry checksum" in capsys.readouterr().out


def test_a_download_whose_checksum_is_wrong_is_discarded(tmp_path):
    """The digest check must reject the file, not just warn about it."""
    mod = _fetcher()
    dest = tmp_path / "case.nc"
    src = tmp_path / "served.bin"
    src.write_bytes(b"wrong content")
    with pytest.raises(OSError, match="checksum mismatch"):
        mod._download(src.as_uri(), dest, expected_sha="0" * 64)
    assert not dest.exists(), "a mismatched download must not reach the cache"
    assert not list(tmp_path.glob("*.partial"))


_SANDU_NUD = frozenset({"ta_nud", "theta_nud", "thetal_nud", "qv_nud",
                        "qt_nud", "rv_nud", "rt_nud"})
_DECLARED_GAPS = {
    "sandu_ref": _SANDU_NUD, "sandu_fast": _SANDU_NUD,
    "sandu_slow": _SANDU_NUD,
    "mpace": frozenset({"ua_nud", "va_nud"}),
    "armcu": frozenset(), "comble": frozenset(),
}


@pytest.mark.parametrize("name", ALL_CASES)
def test_unsupported_channels_match_what_the_note_declares(name):
    """A case's known gap has to be written down, and ONLY that gap.

    The loader reports forcing channels it cannot represent. The set is
    compared exactly: a case that grows a new unrepresentable channel is a case
    quietly running different physics from its specification, and matching on
    "contains nud" would not notice.
    """
    if not _cached(name):
        pytest.skip(f"{name} not cached")
    case = load_dephy_case(name, nlev=16)
    got = {u.split(":", 1)[0].strip() for u in case.unsupported}
    assert got == set(_DECLARED_GAPS[name]), (name, sorted(got))
    note = DEPHY_SCM_CASES[name].note.lower()
    if _DECLARED_GAPS[name]:
        assert "known limitation" in note, (
            f"{name} has unrepresentable forcing channels {sorted(got)} but "
            "its note does not declare them")
    else:
        assert "known limitation" not in note


@pytest.mark.parametrize("name,required", [
    ("comble", ("u_geo", "v_geo", "T_s")),
    ("mpace", ("subsidence_w", "theta_adv", "qv_adv", "w_th_s", "w_qv_s")),
    ("sandu_ref", ("u_geo", "v_geo", "subsidence_w", "T_s")),
    ("armcu", ("u_geo", "theta_adv", "qv_adv", "w_th_s", "w_qv_s")),
])
def test_each_case_gets_the_channels_its_specification_defines(name, required):
    """Per case, not "at least one channel is live".

    A single live channel -- usually the surface -- would otherwise cover for a
    geostrophic wind or a subsidence profile that never got wired.
    """
    if not _cached(name):
        pytest.skip(f"{name} not cached")
    case = load_dephy_case(name, nlev=32)
    for chan in required:
        fn = getattr(case.forcing, chan, None)
        assert fn is not None, f"{name}: {chan} is absent"
        val = np.abs(np.asarray(fn(0.0)))
        assert np.all(np.isfinite(val)) and val.max() > 0.0, (
            f"{name}: {chan} is identically zero")


@pytest.mark.skipif(not _cached("comble"), reason="comble not cached")
def test_comble_surface_temperature_follows_the_whole_trajectory():
    """Not just the first sample.

    A mapping that grabbed only the first point of the series would reproduce
    the 247 K start and silently drop the warming to open water that drives the
    entire case -- the same shape of failure as the missing surface forcing
    this rename fixed.
    """
    raw = xr.open_dataset(resolve_dephy_case_path("comble"), decode_times=False)
    series = np.asarray(raw["ts"].values).ravel()
    t = np.asarray(raw["time"].values).ravel()
    case = load_dephy_case("comble", nlev=24)
    for k in (0, len(t) // 2, len(t) - 1):
        got = float(np.asarray(case.forcing.T_s(float(t[k] - t[0]))))
        assert got == pytest.approx(float(series[k]), abs=0.05), (
            k, got, float(series[k]))
    assert series[-1] - series[0] > 20.0, (
        "the trajectory must warm substantially, or the check above is weak")
