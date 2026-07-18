"""Tests for the run_dino.py ``--config`` YAML loader + the committed configs.

The loader maps a YAML of run parameters onto argparse defaults (so explicit
CLI flags still win), rejects unknown keys, and coerces Path-typed fields.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "run_dino", _REPO / "scripts" / "run" / "run_dino.py")
rd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rd)


def _write(tmp_path, text):
    c = tmp_path / "cfg.yaml"
    c.write_text(text)
    return str(c)


def test_config_loads_as_defaults(tmp_path, monkeypatch):
    cfg = _write(tmp_path,
                 "grid: mpas\ndays: 42\nsnapshot_every_days: 5\n"
                 "mpas_eq_visc_boost: 5.0\noutput_dir: results/x\n")
    monkeypatch.setattr(sys, "argv", ["run_dino", "--config", cfg])
    args = rd._parse_args()
    assert args.grid == "mpas"
    assert args.days == 42
    assert args.snapshot_every_days == 5
    assert args.mpas_eq_visc_boost == 5.0
    # Path-typed field coerced from the YAML string (else .mkdir() would crash).
    assert isinstance(args.output_dir, Path)
    assert str(args.output_dir) == "results/x"


def test_cli_overrides_config(tmp_path, monkeypatch):
    cfg = _write(tmp_path, "grid: mpas\ndays: 42\n")
    monkeypatch.setattr(sys, "argv",
                        ["run_dino", "--config", cfg, "--days", "7"])
    args = rd._parse_args()
    assert args.days == 7        # explicit CLI flag wins
    assert args.grid == "mpas"   # still taken from the config


def test_unknown_key_raises(tmp_path, monkeypatch):
    cfg = _write(tmp_path, "grid: mpas\nbogus_key: 1\n")
    monkeypatch.setattr(sys, "argv", ["run_dino", "--config", cfg])
    with pytest.raises(SystemExit):
        rd._parse_args()


def test_committed_configs_parse_and_are_valid(monkeypatch):
    cfgs = sorted((_REPO / "scripts" / "experiment" / "dino").glob("*.yaml"))
    assert cfgs, "no committed DINO configs found"
    for y in cfgs:
        monkeypatch.setattr(sys, "argv", ["run_dino", "--config", str(y)])
        args = rd._parse_args()        # raises if a key is unknown
        assert args.grid in ("latlon", "mpas")
        assert args.days > 0
        assert isinstance(args.output_dir, Path)


def test_barotropic_solver_flag_parses(monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["run_dino", "--barotropic-solver", "rigid_lid"])
    args = rd._parse_args()
    assert args.barotropic_solver == "rigid_lid"


def test_barotropic_solver_default_none(monkeypatch):
    """Default None → main() keeps DINOConfig.barotropic_solver (implicit_cn)."""
    monkeypatch.setattr(sys, "argv", ["run_dino"])
    args = rd._parse_args()
    assert args.barotropic_solver is None


def test_barotropic_coriolis_flag_parses(monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["run_dino", "--barotropic-coriolis", "een"])
    args = rd._parse_args()
    assert args.barotropic_coriolis == "een"


def test_barotropic_coriolis_default_none(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_dino"])
    assert rd._parse_args().barotropic_coriolis is None


def test_barotropic_coriolis_rejects_bad_choice(monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["run_dino", "--barotropic-coriolis", "bogus"])
    with pytest.raises(SystemExit):
        rd._parse_args()


def test_nemo_faithful_grid_flag_parses(monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["run_dino", "--grid", "latlon", "--nemo-faithful-grid"])
    args = rd._parse_args()
    assert args.nemo_faithful_grid is True


def test_nemo_faithful_grid_default_false(monkeypatch):
    # Opt-in: absent flag keeps the legoESM [-50,0]/198×50 grid.
    monkeypatch.setattr(sys, "argv", ["run_dino"])
    assert rd._parse_args().nemo_faithful_grid is False


def test_barotropic_solver_via_config(tmp_path, monkeypatch):
    cfg = _write(tmp_path, "barotropic_solver: rigid_lid\n")
    monkeypatch.setattr(sys, "argv", ["run_dino", "--config", cfg])
    args = rd._parse_args()
    assert args.barotropic_solver == "rigid_lid"


def test_barotropic_solver_rejects_bad_choice(monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["run_dino", "--barotropic-solver", "bogus"])
    with pytest.raises(SystemExit):
        rd._parse_args()


def test_momentum_advection_flag_parses(monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["run_dino", "--momentum-advection", "weno7"])
    args = rd._parse_args()
    assert args.momentum_advection == "weno7"


def test_momentum_advection_default_none_and_rejects_bad(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_dino"])
    assert rd._parse_args().momentum_advection is None
    monkeypatch.setattr(sys, "argv",
                        ["run_dino", "--momentum-advection", "bogus"])
    with pytest.raises(SystemExit):
        rd._parse_args()


def test_coriolis_scheme_flag_parses(monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["run_dino", "--coriolis-scheme", "matsuno_split"])
    args = rd._parse_args()
    assert args.coriolis_scheme == "matsuno_split"


def test_coriolis_scheme_default_none_and_rejects_bad(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_dino"])
    assert rd._parse_args().coriolis_scheme is None
    monkeypatch.setattr(sys, "argv",
                        ["run_dino", "--coriolis-scheme", "leapfrog"])
    with pytest.raises(SystemExit):
        rd._parse_args()


def test_barotropic_slow_forcing_ab2_flag_parses(monkeypatch):
    """Tri-state: None (card value) / 'on' / 'off' — the FE-barotropic-Coriolis
    bisect lever (the 'oceananigans'-card blowup discriminator)."""
    monkeypatch.setattr(sys, "argv", ["run_dino"])
    assert rd._parse_args().barotropic_slow_forcing_ab2 is None
    monkeypatch.setattr(
        sys, "argv", ["run_dino", "--barotropic-slow-forcing-ab2", "off"])
    assert rd._parse_args().barotropic_slow_forcing_ab2 == "off"
    monkeypatch.setattr(
        sys, "argv", ["run_dino", "--barotropic-slow-forcing-ab2", "1"])
    with pytest.raises(SystemExit):
        rd._parse_args()


def test_rigid_lid_dt_mom_ratio_flag_parses(monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["run_dino", "--barotropic-solver", "rigid_lid",
                         "--rigid-lid-dt-mom-ratio", "6"])
    args = rd._parse_args()
    assert args.rigid_lid_dt_mom_ratio == 6.0


def test_rigid_lid_dt_mom_ratio_default_none(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_dino"])
    args = rd._parse_args()
    assert args.rigid_lid_dt_mom_ratio is None


def test_eos_flag_parses(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_dino", "--eos", "nemo_seos"])
    args = rd._parse_args()
    assert args.eos == "nemo_seos"


def test_eos_default_none(monkeypatch):
    """Default None → main() keeps DINOConfig.eos (wright)."""
    monkeypatch.setattr(sys, "argv", ["run_dino"])
    args = rd._parse_args()
    assert args.eos is None


def test_eos_via_config(tmp_path, monkeypatch):
    cfg = _write(tmp_path, "eos: nemo_seos\n")
    monkeypatch.setattr(sys, "argv", ["run_dino", "--config", cfg])
    args = rd._parse_args()
    assert args.eos == "nemo_seos"


def test_eos_rejects_bad_choice(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_dino", "--eos", "wrightt"])
    with pytest.raises(SystemExit):
        rd._parse_args()


def test_eos_replaces_dino_config(monkeypatch):
    """The --eos flag must reach DINOConfig.eos (the cfg-replace round-trip)."""
    import dataclasses
    from legoesm.ocean.experiments.dino import DINOConfig
    cfg = DINOConfig()
    assert cfg.eos == "wright"          # default
    cfg = dataclasses.replace(cfg, eos="nemo_seos")
    assert cfg.eos == "nemo_seos"


def test_recipe_flag_parses(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_dino", "--recipe", "mitgcm"])
    args = rd._parse_args()
    assert args.recipe == "mitgcm"


def test_recipe_default_none(monkeypatch):
    """Default None → main() leaves DINOConfig untouched (no recipe overlay)."""
    monkeypatch.setattr(sys, "argv", ["run_dino"])
    args = rd._parse_args()
    assert args.recipe is None


def test_recipe_via_config(tmp_path, monkeypatch):
    cfg = _write(tmp_path, "recipe: veros\n")
    monkeypatch.setattr(sys, "argv", ["run_dino", "--config", cfg])
    args = rd._parse_args()
    assert args.recipe == "veros"


def test_recipe_rejects_bad_choice(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_dino", "--recipe", "bogus_model"])
    with pytest.raises(SystemExit):
        rd._parse_args()


def test_recipe_overlays_dino_config(monkeypatch):
    """The --recipe value must reach DINOConfig via dino_config_for_recipe
    (the cfg-overlay round-trip main() performs)."""
    from legoesm.ocean.experiments.dino import (
        DINOConfig, dino_config_for_recipe,
    )
    assert DINOConfig().eos == "wright"                 # bare default
    mit = dino_config_for_recipe("mitgcm")
    assert mit.eos == "unesco80" and mit.momentum_advection == "flux_form"


def test_tke_momentum_visc_bg_flag_parses(monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["run_dino", "--tke-momentum-visc-bg", "1.2e-4"])
    args = rd._parse_args()
    assert args.tke_momentum_visc_bg == 1.2e-4


def test_tke_momentum_visc_bg_default_none(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_dino"])
    args = rd._parse_args()
    assert args.tke_momentum_visc_bg is None


def test_A_v_bg_effective_floors_only_tke():
    """The TKE momentum-viscosity floor must apply ONLY to vmix='tke', leaving
    kpp/constant at the paper A_v_bg (they are stable there)."""
    import dataclasses
    from legoesm.ocean.experiments.dino import DINOConfig
    base = DINOConfig()
    assert base.A_v_bg == 1.2e-4 and base.tke_momentum_visc_bg == 5.0e-4
    # kpp/constant: paper A_v_bg, no floor.
    assert dataclasses.replace(base, vmix_scheme="kpp").A_v_bg_effective == 1.2e-4
    assert dataclasses.replace(
        base, vmix_scheme="constant").A_v_bg_effective == 1.2e-4
    # tke: floored to the stabilizer (max with A_v_bg).
    assert dataclasses.replace(
        base, vmix_scheme="tke").A_v_bg_effective == 5.0e-4
    # A higher A_v_bg override still wins (it's a floor, not an override).
    assert dataclasses.replace(
        base, vmix_scheme="tke", A_v_bg=2.0e-3).A_v_bg_effective == 2.0e-3
    # And the floor can be lowered to run TKE at the unstable paper viscosity.
    assert dataclasses.replace(
        base, vmix_scheme="tke",
        tke_momentum_visc_bg=1.2e-4).A_v_bg_effective == 1.2e-4


def test_evd_momentum_flag_parses(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_dino", "--evd-momentum", "off"])
    args = rd._parse_args()
    assert args.evd_momentum == "off"


def test_evd_momentum_default_none(monkeypatch):
    """Default None -> main() keeps DINOConfig.evd_on_momentum (True)."""
    monkeypatch.setattr(sys, "argv", ["run_dino"])
    args = rd._parse_args()
    assert args.evd_momentum is None


def test_evd_momentum_replaces_config():
    import dataclasses
    from legoesm.ocean.experiments.dino import DINOConfig
    cfg = DINOConfig()
    assert cfg.evd_on_momentum is True      # NEMO nn_evdm=1 default
    cfg = dataclasses.replace(cfg, evd_on_momentum=False)
    assert cfg.evd_on_momentum is False


def test_bottom_drag_scheme_flag_parses(monkeypatch):
    """--bottom-drag-scheme parses (DINOConfig.bottom_drag_scheme wiring);
    unknown values are rejected by argparse choices."""
    monkeypatch.setattr(sys, "argv", ["run_dino.py"])
    assert rd._parse_args().bottom_drag_scheme is None    # keep field default
    monkeypatch.setattr(sys, "argv", [
        "run_dino.py", "--bottom-drag-scheme", "nemo_quadratic"])
    assert rd._parse_args().bottom_drag_scheme == "nemo_quadratic"
    monkeypatch.setattr(sys, "argv", [
        "run_dino.py", "--bottom-drag-scheme", "nemo_typo"])
    with pytest.raises(SystemExit):
        rd._parse_args()
    from legoesm.ocean.experiments.dino import DINOConfig
    assert DINOConfig().bottom_drag_scheme == "legacy"


def test_vmix_choices_include_richardson_and_catke(monkeypatch):
    # codex fix: richardson/catke are wired in _dino_vertical_mixing_config, so
    # --vmix must accept them (else the recipe fallback override is unusable).
    for scheme in ("kpp", "tke", "constant", "richardson", "catke"):
        monkeypatch.setattr(sys, "argv", ["run_dino.py", "--vmix", scheme])
        assert rd._parse_args().vmix == scheme


def test_vmix_rejects_unknown(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_dino.py", "--vmix", "bogus"])
    with pytest.raises(SystemExit):
        rd._parse_args()
def test_r1_exact_preset(monkeypatch):
    """--preset r1_exact loads the DINO_R1 exactness preset; explicit flags
    still override on top; every already-exact oracle selection asserted."""
    from legoesm.ocean.experiments.dino import dino_r1_exact_config
    cfg = dino_r1_exact_config()
    # oracle selections (DINO_R1/EXP00/namelist_cfg)
    assert cfg.eos == "nemo_seos"                       # nameos ln_seos
    assert cfg.vmix_scheme == "tke"                     # ln_zdftke
    assert cfg.bottom_drag_scheme == "nemo_quadratic"   # namdrg ln_non_lin
    assert cfg.use_gm_redi is False                     # ln_ldfeiv=.false.
    assert cfg.A_h_floor == 0.0                         # no legoESM floor
    assert cfg.A_h_eq_boost == 1.0                      # no legoESM boost
    assert cfg.tracer_advection == "fct2"
    assert cfg.vertical_coordinate == "masked_zco"
    assert cfg.pgf_quadrature == "nemo_trapezoid"
    assert cfg.forcing_annual_cycle is True          # ln_ann_cyc
    # stabilizer TKE viscosity floor OFF -> effective A_v == avm0 exactly
    assert cfg.tke_momentum_visc_bg == cfg.A_v_bg
    assert cfg.A_v_bg_effective == cfg.A_v_bg == 1.2e-4  # rn_avm0
    assert cfg.K_v_bg == 1.2e-5                          # rn_avt0
    assert cfg.K_conv == 100.0                           # rn_evd
    assert cfg.evd_on_momentum is True                   # nn_evdm=1
    assert cfg.dt == 2700.0                              # rn_Dt
    assert cfg.U_M == 0.27 and cfg.U_T == 0.027          # rn_Uv / rn_Ud
    # overrides still win
    cfg2 = dino_r1_exact_config(vmix_scheme="kpp")
    assert cfg2.vmix_scheme == "kpp" and cfg2.eos == "nemo_seos"
    # CLI wiring
    monkeypatch.setattr(sys, "argv", ["run_dino.py", "--preset", "r1_exact"])
    args = rd._parse_args()
    assert args.preset == "r1_exact"


def test_r1_exact_preset_flows_to_model_config():
    """The preset's selections reach the built LatLonCGridOceanConfig."""
    from legoesm.ocean.experiments.dino import (
        dino_r1_exact_config, dino_lat_lon_grid, dino_lat_lon_model_config,
    )
    cfg = dino_r1_exact_config()
    g = dino_lat_lon_grid(cfg, n_lon=12)
    mc, _phys = dino_lat_lon_model_config(g, cfg, physics=True)
    assert mc.eos == "nemo_seos"
    assert mc.bottom_drag.bottom_drag_scheme == "nemo_quadratic"
    assert mc.bottom_drag.bottom_drag_cd0 == cfg.C_d_bottom
    assert mc.tracer_advection == "fct2"
    # EIV off but iso-neutral Redi-only ON (ln_traldf_iso + msc):
    assert mc.gm_redi is not None
    assert mc.gm_redi.kappa_GM == 0.0            # no bolus transport
    assert mc.gm_redi.implicit_K33 is True       # MSC
    assert mc.K_h == 0.0                         # no iso-level double-count
    assert mc.lateral_viscosity.A_h_floor == 0.0
    assert mc.lateral_viscosity.A_h_eq_boost == 1.0


def test_allow_multiyear_flag(monkeypatch):
    """--days > 365 requires --allow-multiyear (GPU/SLURM opt-out)."""
    monkeypatch.setattr(sys, "argv", ["run_dino.py", "--days", "720"])
    args = rd._parse_args()
    assert args.days == 720 and args.allow_multiyear is False
    monkeypatch.setattr(sys, "argv",
                        ["run_dino.py", "--days", "720", "--allow-multiyear"])
    assert rd._parse_args().allow_multiyear is True
