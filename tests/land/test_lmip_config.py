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
    assert cfg.land_frac_min == 0.0                        # default: any surfdata land (no arbitrary cutoff)


def test_missing_required_field_raises():
    bad = _minimal()
    del bad["grid"]["resolution"]
    with pytest.raises(ValueError, match="grid.resolution"):
        validate_config(bad)


def test_bad_grid_type_raises():
    bad = _minimal()
    bad["grid"]["type"] = "octahedral_reduced_gaussian"
    with pytest.raises(ValueError, match="grid.type"):
        validate_config(bad)


@pytest.mark.parametrize(
    "alias", ["voronoi", "icosahedral", "ico", "mpas", "mpas_voronoi"])
def test_voronoi_aliases_accepted(alias):
    """The SCVT mesh is a legal spin-up target, under every alias it goes by.

    A coupled run on that mesh needs its land state spun up on the SAME mesh:
    the restart loader compares column counts and refuses a mismatch rather
    than interpolating, so a spin-up on any other grid produces a state that
    run cannot load.  The driver has always been able to BUILD the mesh; this
    schema was the only thing refusing to ask for it.
    """
    cfg = _minimal()
    cfg["grid"]["type"] = alias
    assert validate_config(cfg).grid["type"] == alias


def test_simple_seb_plus_most_now_accepted():
    # land/stable fixed SimpleSEB cold-start stability (amip_sota runs this
    # pairing), so the prior blanket rejection is gone.  (main re-adds a
    # rejection each sync; this branch keeps the pairing accepted — see the
    # merge note.  LMIP production uses two_leaf_canopy+most, unaffected either way.)
    cfg_in = _minimal()
    cfg_in["physics"]["bulk_scheme"] = "most"        # simple_seb + most
    cfg = validate_config(cfg_in)
    assert cfg.physics["bulk_scheme"] == "most"


def test_physics_knob_defaults():
    cfg = validate_config(_minimal())
    p = cfg.physics
    assert p["stomatal_model"] == "ball_berry"       # canopy leaf conductance model
    assert p["stomata_enabled"] is False             # SimpleSEB Jarvis off (amip_sota)
    assert p["snow_albedo_feedback"] is True
    assert p.get("vc_max25") is None                 # None -> per-PFT / land default


def test_freeze_thaw_defaults_off():
    # Off by default = bit-identical sensible-only soil heat (no behaviour change
    # for existing configs that don't mention it).
    cfg = validate_config(_minimal())
    assert cfg.physics["enable_freeze_thaw"] is False


def test_freeze_thaw_override_on():
    ok = _minimal()
    ok["physics"]["enable_freeze_thaw"] = True
    cfg = validate_config(ok)
    assert cfg.physics["enable_freeze_thaw"] is True


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


def test_freeze_thaw_non_bool_raises():
    bad = _minimal()
    bad["physics"]["enable_freeze_thaw"] = "yes"     # must be a real bool
    with pytest.raises(ValueError, match="enable_freeze_thaw"):
        validate_config(bad)


def test_bad_stomatal_model_raises():
    bad = _minimal()
    bad["physics"]["stomatal_model"] = "jarvis"      # not a canopy leaf model
    with pytest.raises(ValueError, match="stomatal_model"):
        validate_config(bad)


def test_stomatal_calibration_bounds():
    bad = _minimal()
    bad["physics"]["vc_max25"] = 1000.0              # absurd
    with pytest.raises(ValueError, match="vc_max25"):
        validate_config(bad)
    ok = _minimal()
    ok["physics"].update(vc_max25=60.0, g1=9.0, gs_max=0.3)
    cfg = validate_config(ok)
    assert cfg.physics["vc_max25"] == 60.0


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


# --------------------------------------------------------------------------
# Surface albedo values (generic names, config-supplied)
# --------------------------------------------------------------------------
def test_albedo_defaults_to_an_empty_block():
    """No albedo block = the uncalibrated LandAlbedoConfig defaults, so an
    EXISTING config reproduces its baseline byte-for-byte."""
    p = validate_config(_minimal()).physics
    assert p["albedo"] == {}
    assert p["glacier_albedo_vis"] is None and p["glacier_albedo_nir"] is None


def test_albedo_accepts_landalbedoconfig_field_values():
    cfg_in = _minimal()
    cfg_in["physics"]["albedo"] = {"alpha_snow_max": 0.8077,
                                   "snow_depth_crit": 15.4229}
    p = validate_config(cfg_in).physics
    assert p["albedo"]["alpha_snow_max"] == 0.8077


def test_albedo_rejects_unknown_field():
    """Dispatch hardening: a typo'd field name must raise, never be silently
    dropped — the run would carry DIFFERENT physics than the config declares."""
    bad = _minimal()
    bad["physics"]["albedo"] = {"alpha_snow_mx": 0.8}      # plausible typo
    with pytest.raises(ValueError, match="unknown field"):
        validate_config(bad)


def test_albedo_rejects_nonfinite_value():
    bad = _minimal()
    bad["physics"]["albedo"] = {"alpha_snow_max": float("nan")}
    with pytest.raises(ValueError, match="finite"):
        validate_config(bad)


def test_glacier_pair_must_come_together_and_be_a_fraction():
    half = _minimal()
    half["physics"]["glacier_albedo_vis"] = 0.8
    with pytest.raises(ValueError, match="together"):
        validate_config(half)
    bad = _minimal()
    bad["physics"].update(glacier_albedo_vis=1.5, glacier_albedo_nir=0.6)
    with pytest.raises(ValueError, match="glacier_albedo_vis"):
        validate_config(bad)
    ok = _minimal()
    ok["physics"].update(glacier_albedo_vis=0.8178, glacier_albedo_nir=0.6178)
    p = validate_config(ok).physics
    assert (p["glacier_albedo_vis"], p["glacier_albedo_nir"]) == (0.8178, 0.6178)


def test_retired_selector_keys_fail_loudly():
    """An old config carrying the retired 2026-08 selector keys must raise —
    silently ignoring them would run it uncalibrated."""
    for key in ("albedo_calibration", "root_calibration"):
        bad = _minimal()
        bad["physics"][key] = "amip_multilayer"
        with pytest.raises(ValueError, match="retired"):
            validate_config(bad)


# --------------------------------------------------------------------------
# Vertical soil grid
# --------------------------------------------------------------------------
def test_soil_grid_defaults_match_soilgridconfig():
    """Schema defaults must track SoilGridConfig() so a config with no soil keys
    reproduces the 8-layer / 6.375 m baseline exactly."""
    from legoesm.land.soil_grid import SoilGridConfig
    p = validate_config(_minimal()).physics
    assert p["soil_n_layers"] == SoilGridConfig().n_layers
    assert p["soil_depth_m"] == SoilGridConfig().total_depth
    assert p["soil_growth_factor"] == SoilGridConfig().growth_factor


def test_soil_grid_amip_parity_accepted():
    cfg_in = _minimal()
    cfg_in["physics"].update(soil_n_layers=10, soil_depth_m=3.0)
    p = validate_config(cfg_in).physics
    assert (p["soil_n_layers"], p["soil_depth_m"]) == (10, 3.0)


@pytest.mark.parametrize("key,bad", [
    ("soil_n_layers", 0),          # a zero-layer column is not a column
    ("soil_n_layers", 500),        # absurd
    ("soil_n_layers", True),       # bool is not an int layer count
    ("soil_n_layers", 8.5),        # non-integer
    ("soil_depth_m", -1.0),        # negative depth
    ("soil_depth_m", 0.05),        # implausibly shallow (0 means "default")
    ("soil_growth_factor", 0.5),   # shrinking layers with depth
    ("soil_growth_factor", 9.0),   # out of range
])
def test_soil_grid_rejects_nonsense(key, bad):
    cfg_in = _minimal()
    cfg_in["physics"][key] = bad
    with pytest.raises(ValueError, match=key):
        validate_config(cfg_in)


def test_pft_root_lists_default_to_absent():
    """No lists = the scalar MultiLayerLandConfig values, bit-identical to the
    pre-existing behaviour."""
    p = validate_config(_minimal()).physics
    assert all(p[k] is None for k in
               ("root_depth_per_pft", "theta_wp_per_pft", "theta_fc_per_pft"))


def _root_lists():
    n = 17
    return {"root_depth_per_pft": [0.1 + 0.1 * i for i in range(n)],
            "theta_wp_per_pft": [0.05 + 0.003 * i for i in range(n)],
            "theta_fc_per_pft": [0.17 + 0.006 * i for i in range(n)]}


def test_pft_root_lists_accepted_together():
    cfg_in = _minimal()
    cfg_in["physics"].update(_root_lists())
    p = validate_config(cfg_in).physics
    assert len(p["root_depth_per_pft"]) == 17


def test_pft_root_lists_rejected_when_partial():
    """The builders gather all three tables at the dominant PFT — a partial
    specification silently mixing per-PFT and scalar values must raise."""
    bad = _minimal()
    bad["physics"]["root_depth_per_pft"] = _root_lists()["root_depth_per_pft"]
    with pytest.raises(ValueError, match="together"):
        validate_config(bad)


@pytest.mark.parametrize("mutate,match", [
    (lambda r: r["root_depth_per_pft"].pop(), "17"),            # wrong length
    (lambda r: r["root_depth_per_pft"].__setitem__(0, 50.0), "root_depth"),
    (lambda r: r["theta_wp_per_pft"].__setitem__(0, 0.5), "wp"),  # wp >= fc
])
def test_pft_root_lists_reject_nonsense(mutate, match):
    bad = _minimal()
    r = _root_lists()
    mutate(r)
    bad["physics"].update(r)
    with pytest.raises(ValueError, match=match):
        validate_config(bad)
