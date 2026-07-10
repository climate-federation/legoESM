"""Unit tests for the LMIP experiment config schema + override system."""

import pytest
import yaml

from legoesm.land.lmip_config import apply_overrides, validate_config


def _minimal():
    """Smallest legal config — every required field present."""
    return {
        "grid": {"type": "latlon", "resolution": 4},
        "physics": {"land_mode": "multilayer",
                    "surface_scheme": "simple_seb",
                    "bulk_scheme": "constant"},
        "forcing": {"source": "synthetic", "year_start": 2000, "year_end": 2000},
        "surfdata": {"path": "/tmp/sd.nc"},
        "time": {"dt": 3600.0, "n_steps": 4, "start_doy": 0.0},
        "output": {"tapes": [{"name": "step", "freq": "step",
                              "average": "inst", "vars": ["T_sfc"]}]},
    }


def test_minimal_config_validates():
    cfg = validate_config(_minimal())
    assert cfg.grid["type"] == "latlon"
    assert cfg.forcing["k_neighbors"] == 4                # default applied
    assert cfg.restart["from"] == ""                       # default applied
    assert cfg.land_frac_min == 0.5                        # default applied


def test_missing_required_field_raises():
    bad = _minimal()
    del bad["grid"]["resolution"]
    with pytest.raises(ValueError, match="grid.resolution"):
        validate_config(bad)


def test_bad_grid_type_raises():
    bad = _minimal()
    bad["grid"]["type"] = "voronoi"
    with pytest.raises(ValueError, match="grid.type"):
        validate_config(bad)


def test_land_cover_dataset_defaults_to_clm5():
    cfg = validate_config(_minimal())
    assert cfg.surfdata["land_cover_dataset"] == "clm5"     # default applied


def test_bad_land_cover_dataset_raises():
    bad = _minimal()
    bad["surfdata"]["land_cover_dataset"] = "not_a_dataset"
    with pytest.raises(ValueError, match="land_cover_dataset"):
        validate_config(bad)


def test_land_use_change_bookkeeping_accepted():
    ok = _minimal()
    ok["land_use_change"] = {"scheme": "bookkeeping"}
    validate_config(ok)                                    # no raise


def test_bad_land_use_change_scheme_raises():
    bad = _minimal()
    bad["land_use_change"] = {"scheme": "bogus"}
    with pytest.raises(ValueError, match="land_use_change.scheme"):
        validate_config(bad)


def test_land_use_change_unknown_key_raises():
    # A typo'd knob must be a hard error (else it is silently dropped -> defaults).
    bad = _minimal()
    bad["land_use_change"] = {"scheme": "bookkeeping", "clear_brun_frac": 0.9}
    with pytest.raises(ValueError, match="unknown key"):
        validate_config(bad)


def test_simple_seb_plus_most_rejected_at_config_time():
    bad = _minimal()
    bad["physics"]["bulk_scheme"] = "most"
    with pytest.raises(ValueError, match="simple_seb"):
        validate_config(bad)


def test_source_cru_jra_requires_data_dir():
    # source and data_dir must agree — the driver picks real-vs-synthetic from
    # data_dir, so source='cru_jra' with an empty data_dir would silently run
    # synthetic forcing instead of the reanalysis the user asked for.
    bad = _minimal()
    bad["forcing"]["source"] = "cru_jra"          # but no data_dir
    with pytest.raises(ValueError, match="data_dir"):
        validate_config(bad)


def test_source_synthetic_rejects_data_dir():
    bad = _minimal()
    bad["forcing"]["data_dir"] = "/glade/forcing"  # contradicts source='synthetic'
    with pytest.raises(ValueError, match="synthetic"):
        validate_config(bad)


def test_source_cru_jra_with_data_dir_ok():
    good = _minimal()
    good["forcing"]["source"] = "cru_jra"
    good["forcing"]["data_dir"] = "/glade/forcing"
    cfg = validate_config(good)
    assert cfg.forcing["source"] == "cru_jra"


def test_year_end_before_year_start_rejected():
    bad = _minimal()
    bad["forcing"]["year_start"] = 2005
    bad["forcing"]["year_end"] = 2000
    with pytest.raises(ValueError, match="year_end"):
        validate_config(bad)


def test_missing_output_tapes_rejected():
    bad = _minimal()
    bad["output"] = {"tapes": []}
    with pytest.raises(ValueError, match="output.tapes"):
        validate_config(bad)


def test_apply_overrides_creates_nested_paths():
    base = _minimal()
    merged = apply_overrides(base, [
        "forcing.year_start=1980",
        "forcing.year_end=1984",
        "physics.bulk_scheme=constant",
        "time.n_steps=8760",
    ])
    assert merged["forcing"]["year_start"] == 1980
    assert merged["forcing"]["year_end"] == 1984
    assert merged["time"]["n_steps"] == 8760
    validate_config(merged)                                # still legal


def test_apply_overrides_yaml_parses_bools_and_lists():
    base = _minimal()
    merged = apply_overrides(base, [
        "output.tapes=[{name: monthly, freq: monthly, average: mean, vars: [T_sfc]}]",
    ])
    assert merged["output"]["tapes"][0]["freq"] == "monthly"


def test_apply_overrides_rejects_malformed_spec():
    with pytest.raises(ValueError, match="expected"):
        apply_overrides(_minimal(), ["forcing.year_start-1980"])   # missing '='


def test_apply_overrides_creates_new_key():
    base = _minimal()
    merged = apply_overrides(base, ["restart.from=/tmp/prev.npz"])
    assert merged["restart"]["from"] == "/tmp/prev.npz"
