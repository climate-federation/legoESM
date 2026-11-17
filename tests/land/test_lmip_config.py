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


def test_pmodel_switch_keys_default_and_validate():
    cfg = validate_config(_minimal())
    assert cfg.physics["canopy_capacity_scheme"] == "prescribed"
    assert cfg.physics["canopy_g1_source"] == "table"
    bad = _minimal()
    bad["physics"]["canopy_capacity_scheme"] = "optimum"
    with pytest.raises(ValueError, match="canopy_capacity_scheme"):
        validate_config(bad)
    bad2 = _minimal()
    bad2["physics"]["canopy_g1_source"] = "xi"
    with pytest.raises(ValueError, match="canopy_g1_source"):
        validate_config(bad2)


def test_pmodel_switches_that_would_be_dropped_are_refused():
    """calibrated_land_physics replaces the surface scheme, and the slab +
    simple_seb config carries no stomata block: either would run the baseline
    under a P-model label, so the config is refused."""
    import pytest
    from legoesm.land.lmip_config import validate_config
    base = {"grid": {"type": "latlon", "resolution": 4},
            "forcing": {"source": "synthetic", "data_dir": "",
                        "year_start": 2000, "year_end": 2000},
            "surfdata": {"path": "sd.nc"},
            "time": {"dt": 3600.0, "n_steps": 4, "start_doy": 0.0}}
    pm = {"canopy_capacity_scheme": "pmodel", "stomatal_model": "medlyn"}
    with pytest.raises(ValueError, match="calibrated_land_physics"):
        validate_config({**base, "physics": {
            "land_mode": "multilayer", "surface_scheme": "two_leaf_canopy",
            "bulk_scheme": "most", "calibrated_land_physics": True, **pm}})
    with pytest.raises(ValueError, match="silently inert"):
        validate_config({**base, "physics": {
            "land_mode": "slab", "surface_scheme": "simple_seb",
            "bulk_scheme": "most", **pm}})


def test_co2_file_defaults_to_empty_and_must_be_a_string():
    import copy
    from pathlib import Path
    tmpl = Path(__file__).resolve().parents[2] / "templates/land/biophysics/smoke_4deg.yaml"
    base = yaml.safe_load(tmpl.read_text())
    base["forcing"].pop("co2_file", None)
    assert validate_config(copy.deepcopy(base)).forcing["co2_file"] == ""
    base["forcing"]["co2_file"] = None                  # what `-o forcing.co2_file=` yields
    assert validate_config(copy.deepcopy(base)).forcing["co2_file"] == ""
    base["forcing"]["co2_file"] = 412
    with pytest.raises(ValueError, match="co2_file"):
        validate_config(base)


def test_optimality_wrapper_fills_the_switches_and_refuses_disagreement():
    """physics.optimality fills the four optimality switches (pmodel |
    phydro); a switch also given must agree; the P-hydro arm's three 'phydro'
    switches must come together."""
    import copy
    from legoesm.land.lmip_config import OPTIMALITY_SWITCHES, validate_config
    base = {
        "grid": {"type": "latlon", "resolution": 4},
        "physics": {"land_mode": "multilayer", "surface_scheme": "two_leaf_canopy",
                    "bulk_scheme": "most"},
        "forcing": {"source": "synthetic", "data_dir": "", "year_start": 2000, "year_end": 2000},
        "surfdata": {"path": "x.nc"},
        "time": {"dt": 3600.0, "n_steps": 24, "start_doy": 0.0},
        "output": {"tapes": [{"name": "d", "freq": "daily", "average": "mean", "vars": ["GPP"]}]},
    }
    for arm, switches in OPTIMALITY_SWITCHES.items():
        cfg = copy.deepcopy(base); cfg["physics"]["optimality"] = arm
        if arm == "pmodel":
            cfg["physics"]["pmodel_acclimation_scheme"] = "han"
        phys = validate_config(cfg).physics
        assert "optimality" not in phys
        for k, v in switches.items():
            assert phys[k] == v, (arm, k)
    cfg = copy.deepcopy(base); cfg["physics"]["optimality"] = "phydro"
    cfg["physics"]["stomatal_model"] = "medlyn"
    with pytest.raises(ValueError, match="set one or the other"):
        validate_config(cfg)
    cfg = copy.deepcopy(base); cfg["physics"]["optimality"] = "wue"
    with pytest.raises(ValueError, match="optimality"):
        validate_config(cfg)
    # Half an arm by hand is refused; the whole arm by hand is accepted.
    cfg = copy.deepcopy(base); cfg["physics"]["transpiration_stress"] = "phydro"
    with pytest.raises(ValueError, match="together"):
        validate_config(cfg)
    cfg = copy.deepcopy(base); cfg["physics"].update(OPTIMALITY_SWITCHES["phydro"])
    assert validate_config(cfg).physics["canopy_capacity_scheme"] == "phydro"


def test_pmodel_acclimation_key_defaults_validates_and_refuses_inert_use():
    """physics.pmodel_acclimation defaults to the original 24 h scheme; the
    subdaily scheme is accepted only where it acts (a P-model switch on the
    multilayer two-leaf canopy, not the P-hydro arm)."""
    import copy
    from legoesm.land.lmip_config import validate_config
    base = {
        "grid": {"type": "latlon", "resolution": 4},
        "physics": {"land_mode": "multilayer", "surface_scheme": "two_leaf_canopy",
                    "bulk_scheme": "most", "optimality": "pmodel",
                    "pmodel_acclimation_scheme": "han"},
        "forcing": {"source": "synthetic", "data_dir": "", "year_start": 2000, "year_end": 2000},
        "surfdata": {"path": "x.nc"},
        "time": {"dt": 3600.0, "n_steps": 24, "start_doy": 0.0},
        "output": {"tapes": [{"name": "d", "freq": "daily", "average": "mean", "vars": ["GPP"]}]},
    }
    assert validate_config(copy.deepcopy(base)).physics["pmodel_acclimation_scheme"] == "han"
    none = copy.deepcopy(base); none["physics"]["optimality"] = "none"
    none["physics"].pop("pmodel_acclimation_scheme")
    assert "pmodel_acclimation_scheme" not in validate_config(none).physics
    ok = copy.deepcopy(base); ok["physics"]["pmodel_acclimation_scheme"] = "mengoli"
    assert validate_config(ok).physics["pmodel_acclimation_scheme"] == "mengoli"
    bad = copy.deepcopy(base); bad["physics"]["pmodel_acclimation_scheme"] = "noon"
    with pytest.raises(ValueError, match="pmodel_acclimation_scheme"):
        validate_config(bad)
    inert = copy.deepcopy(base); inert["physics"].pop("optimality")
    inert["physics"]["pmodel_acclimation_scheme"] = "mengoli"
    with pytest.raises(ValueError, match="optimality='pmodel' only"):
        validate_config(inert)
    ph = copy.deepcopy(base); ph["physics"]["optimality"] = "phydro"
    ph["physics"]["pmodel_acclimation_scheme"] = "mengoli"
    with pytest.raises(ValueError, match="optimality='pmodel' only"):
        validate_config(ph)
    seb = copy.deepcopy(base); seb["physics"]["surface_scheme"] = "simple_seb"
    seb["physics"].pop("optimality")
    seb["physics"].update(canopy_capacity_scheme="pmodel", stomata_enabled=True,
                          pmodel_acclimation_scheme="mengoli")
    with pytest.raises(ValueError, match="two-leaf canopy only"):
        validate_config(seb)


def test_electron_transport_model_key_defaults_and_refuses_without_pmodel():
    import copy
    from legoesm.land.lmip_config import validate_config
    base = {
        "grid": {"type": "latlon", "resolution": 4},
        "physics": {"land_mode": "multilayer", "surface_scheme": "two_leaf_canopy",
                    "bulk_scheme": "most", "optimality": "pmodel",
                    "pmodel_acclimation_scheme": "han"},
        "forcing": {"source": "synthetic", "data_dir": "", "year_start": 2000, "year_end": 2000},
        "surfdata": {"path": "x.nc"},
        "time": {"dt": 3600.0, "n_steps": 24, "start_doy": 0.0},
        "output": {"tapes": [{"name": "d", "freq": "daily", "average": "mean", "vars": ["GPP"]}]},
    }
    assert validate_config(copy.deepcopy(base)).physics["electron_transport_model"] == "nrh"
    ok = copy.deepcopy(base); ok["physics"]["electron_transport_model"] = "smith"
    assert validate_config(ok).physics["electron_transport_model"] == "smith"
    bad = copy.deepcopy(base); bad["physics"]["electron_transport_model"] = "pmodel_phi0"
    with pytest.raises(ValueError, match="renamed 'smith'"):
        validate_config(bad)
    nopm = copy.deepcopy(base); nopm["physics"].pop("optimality")
    nopm["physics"].pop("pmodel_acclimation_scheme")
    nopm["physics"]["electron_transport_model"] = "smith"
    with pytest.raises(ValueError, match="canopy_capacity_scheme='pmodel'"):
        validate_config(nopm)


def test_pmodel_stomatal_model_requires_pmodel_capacities_and_no_slope():
    import copy
    from legoesm.land.lmip_config import validate_config
    base = {
        "grid": {"type": "latlon", "resolution": 4},
        "physics": {"land_mode": "multilayer", "surface_scheme": "two_leaf_canopy",
                    "bulk_scheme": "most", "stomatal_model": "pmodel",
                    "canopy_capacity_scheme": "pmodel", "canopy_g1_source": "table"},
        "forcing": {"source": "synthetic", "data_dir": "", "year_start": 2000, "year_end": 2000},
        "surfdata": {"path": "x.nc"},
        "time": {"dt": 3600.0, "n_steps": 24, "start_doy": 0.0},
        "output": {"tapes": [{"name": "d", "freq": "daily", "average": "mean", "vars": ["GPP"]}]},
    }
    assert validate_config(copy.deepcopy(base)).physics["stomatal_model"] == "pmodel"
    bad = copy.deepcopy(base); bad["physics"]["canopy_g1_source"] = "pmodel"
    with pytest.raises(ValueError, match="pmodel|g1_source"):
        validate_config(bad)
    bad2 = copy.deepcopy(base); bad2["physics"]["canopy_capacity_scheme"] = "prescribed"
    with pytest.raises(ValueError, match="pmodel"):
        validate_config(bad2)


def test_sandoval_kphio_needs_its_file_and_the_file_needs_sandoval():
    import copy
    base = {
        "grid": {"type": "latlon", "resolution": 4},
        "physics": {"land_mode": "multilayer", "surface_scheme": "two_leaf_canopy",
                    "bulk_scheme": "most", "optimality": "pmodel",
                    "pmodel_acclimation_scheme": "han"},
        "forcing": {"source": "synthetic", "data_dir": "", "year_start": 2000, "year_end": 2000},
        "surfdata": {"path": "x.nc"},
        "time": {"dt": 3600.0, "n_steps": 24, "start_doy": 0.0},
        "output": {"tapes": [{"name": "d", "freq": "daily", "average": "mean", "vars": ["GPP"]}]},
    }
    c = validate_config(copy.deepcopy(base))
    assert c.physics["intrinsic_quantum_yield"] == "temperature"
    assert c.surfdata["sandoval_file"] == ""
    ok = copy.deepcopy(base); ok["physics"]["intrinsic_quantum_yield"] = "sandoval"
    ok["surfdata"]["sandoval_file"] = "sv.nc"
    _ok = validate_config(ok)
    assert _ok.surfdata["sandoval_file"] == "sv.nc"
    assert _ok.raw["params"]["land.p_model.phi0_peak"] == 1.0 / 9.0  # Sandoval peak recorded
    nofile = copy.deepcopy(base); nofile["physics"]["intrinsic_quantum_yield"] = "sandoval"
    with pytest.raises(ValueError, match="sandoval_file"):
        validate_config(nofile)
    inert = copy.deepcopy(base); inert["surfdata"]["sandoval_file"] = "sv.nc"
    with pytest.raises(ValueError, match="inert"):
        validate_config(inert)
    nopm = copy.deepcopy(base); nopm["physics"].pop("optimality")
    nopm["physics"].pop("pmodel_acclimation_scheme")
    nopm["physics"]["intrinsic_quantum_yield"] = "sandoval"
    nopm["surfdata"]["sandoval_file"] = "sv.nc"
    with pytest.raises(ValueError, match="silently inert"):
        validate_config(nopm)
    typo = copy.deepcopy(base); typo["physics"]["intrinsic_quantum_yield"] = "sandovl"
    with pytest.raises(ValueError, match="intrinsic_quantum_yield"):
        validate_config(typo)


def test_params_block_defaults_empty_and_must_be_a_mapping():
    from legoesm.land.p_model import PHI0_PEAK_DEFAULT
    # The only entry a bare config records is the quantum-yield peak of the
    # chosen phi0 option (so the resolved config carries the value it ran with).
    _peak = {"land.p_model.phi0_peak": PHI0_PEAK_DEFAULT["temperature"]}
    assert validate_config(_minimal()).raw["params"] == _peak
    c = _minimal(); c["params"] = {"land.two_leaf_canopy.vcmax_light_frac": 1.0}
    assert validate_config(c).raw["params"] == {
        "land.two_leaf_canopy.vcmax_light_frac": 1.0, **_peak}
    # an explicit peak wins over the option default
    c = _minimal(); c["params"] = {"land.p_model.phi0_peak": 0.07}
    assert validate_config(c).raw["params"] == {"land.p_model.phi0_peak": 0.07}
    bad = _minimal(); bad["params"] = [1.0]
    with pytest.raises(ValueError, match="params must be a mapping"):
        validate_config(bad)


def _two_leaf_base(**physics):
    import copy
    base = {
        "grid": {"type": "latlon", "resolution": 4},
        "physics": {"land_mode": "multilayer", "surface_scheme": "two_leaf_canopy",
                    "bulk_scheme": "most"},
        "forcing": {"source": "synthetic", "data_dir": "", "year_start": 2000, "year_end": 2000},
        "surfdata": {"path": "x.nc"},
        "time": {"dt": 3600.0, "n_steps": 24, "start_doy": 0.0},
        "output": {"tapes": [{"name": "d", "freq": "daily", "average": "mean", "vars": ["GPP"]}]},
    }
    base["physics"].update(physics)
    return copy.deepcopy(base)


@pytest.mark.parametrize("physics,hint", [
    ({"optimality": "least_cost"}, "renamed 'pmodel'"),
    ({"canopy_capacity_scheme": "p_model"}, "renamed 'pmodel'"),
    ({"canopy_g1_source": "p_model"}, "renamed 'pmodel'"),
    ({"stomatal_model": "pmodel_chi"}, "renamed 'pmodel'"),
    ({"optimality": "pmodel", "pmodel_acclimation_scheme": "ema_24h"}, "renamed 'han'"),
    ({"optimality": "pmodel", "pmodel_acclimation_scheme": "subdaily_noon"}, "renamed 'mengoli'"),
    ({"optimality": "pmodel", "pmodel_acclimation": "han"}, "renamed physics.pmodel_acclimation_scheme"),
    ({"canopy_light_response": "clm"}, "renamed physics.electron_transport_model"),
    ({"electron_transport_model": "clm"}, "renamed 'nrh'"),
    ({"electron_transport_model": "pmodel_phi0"}, "renamed 'smith'"),
    ({"pmodel_kphio_method": "temperature"}, "renamed physics.intrinsic_quantum_yield"),
])
def test_retired_switch_strings_are_refused_with_the_new_name(physics, hint):
    from legoesm.land.lmip_config import validate_config
    with pytest.raises(ValueError, match=hint):
        validate_config(_two_leaf_base(**physics))


def test_optimality_arms_pin_stomata_hours_and_days():
    """The four arms resolve to the table of docs/land/optimality_switches.md."""
    from legoesm.land.lmip_config import validate_config
    want = {
        ("none",): dict(canopy_capacity_scheme="prescribed", canopy_g1_source="table",
                        stomatal_model="ball_berry", pmodel_acclimation_hours="24h"),
        ("pmodel", "han"): dict(canopy_capacity_scheme="pmodel", canopy_g1_source="pmodel",
                                stomatal_model="medlyn", pmodel_acclimation_hours="24h"),
        ("pmodel", "mengoli"): dict(canopy_capacity_scheme="pmodel", canopy_g1_source="table",
                                    stomatal_model="pmodel", pmodel_acclimation_hours="noon"),
        ("phydro",): dict(canopy_capacity_scheme="phydro", canopy_g1_source="table",
                          stomatal_model="phydro", pmodel_acclimation_hours="24h"),
    }
    for arm, exp in want.items():
        phys = {"optimality": arm[0]}
        if len(arm) > 1:
            phys["pmodel_acclimation_scheme"] = arm[1]
        got = validate_config(_two_leaf_base(**phys)).physics
        for k, v in exp.items():
            assert got[k] == v, (arm, k, got[k])
        assert got["pmodel_acclimation_days"] == 15.0
        assert got["electron_transport_model"] == "nrh"
        assert got["intrinsic_quantum_yield"] == "temperature"
    # han admits ball_berry, refuses leuning / pmodel; mengoli pins pmodel
    ok = validate_config(_two_leaf_base(optimality="pmodel", pmodel_acclimation_scheme="han",
                                        stomatal_model="ball_berry")).physics
    assert ok["stomatal_model"] == "ball_berry"
    for sm in ("leuning", "pmodel", "phydro"):
        with pytest.raises(ValueError, match="admits stomatal_model"):
            validate_config(_two_leaf_base(optimality="pmodel", pmodel_acclimation_scheme="han",
                                           stomatal_model=sm))
    with pytest.raises(ValueError, match="admits stomatal_model"):
        validate_config(_two_leaf_base(optimality="pmodel", pmodel_acclimation_scheme="mengoli",
                                       stomatal_model="medlyn"))
    # hours / days given explicitly override the arm defaults
    got = validate_config(_two_leaf_base(optimality="pmodel", pmodel_acclimation_scheme="han",
                                         pmodel_acclimation_hours="daytime",
                                         pmodel_acclimation_days=7)).physics
    assert got["pmodel_acclimation_hours"] == "daytime" and got["pmodel_acclimation_days"] == 7.0
    with pytest.raises(ValueError, match="pmodel_acclimation_days"):
        validate_config(_two_leaf_base(optimality="pmodel", pmodel_acclimation_scheme="han",
                                       pmodel_acclimation_days=0))
    # smith electron transport is admitted under the phydro arm too
    got = validate_config(_two_leaf_base(optimality="phydro",
                                         electron_transport_model="smith")).physics
    assert got["electron_transport_model"] == "smith"
