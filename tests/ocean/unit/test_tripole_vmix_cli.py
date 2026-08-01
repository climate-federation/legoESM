"""--tripole-vmix wiring: NEMO zdftke closure on the OMIP tripole path.

Gates for scripts/run/run_omip_core2.py's ``--tripole-vmix {none,tke,kpp}``:

  * ``orca1_zdftke_config()`` maps the ORCA1 ``&namzdf_tke`` namelist value by
    value onto :class:`TKEConfig` (rn_ediff/rn_ediss/rn_emin/rn_emin0/nn_pdl/
    nn_mxl/ln_lc/rn_lc/nn_etau/rn_efr/nn_htau);
  * ``build_tripole_vmix_config``: "none" reproduces the legacy
    ``VerticalMixingConfig(scheme="none")`` object EXACTLY (byte-identical
    default), "tke"/"kpp" attach the closure, unknown raises (dispatch
    hardening), and ``--iwm`` composes ADDITIVELY with the closure (NEMO's
    zdfphy order) instead of clobbering it;
  * the ``build_tripole`` keyword exists and defaults to "none";
  * ``nemo_etau_injection`` broadcasts the nn_htau=1 "latitude" profile
    against full 2-D-horizontal columns (the model threads 2-D
    ``grid.lat_T`` degrees — a 1-D row-latitude cannot right-broadcast);
  * a tiny lat-lon C-grid model with the ORCA1 TKE config (etau latitude
    mode) traces + steps finitely through the implicit vertical-mixing
    fallback — the end-to-end regression gate for the lat_T threading fix.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def _runner():
    from scripts.run import run_omip_core2 as r
    return r


# ---------------------------------------------------------------------------
# ORCA1 namelist -> TKEConfig mapping
# ---------------------------------------------------------------------------


def test_orca1_zdftke_namelist_mapping():
    """Every mapped &namzdf_tke value, against the RUN_REF namelist."""
    from legoesm import constants
    r = _runner()
    cfg = r.orca1_zdftke_config()
    assert cfg.c_k == 0.1                       # rn_ediff (ref default)
    assert cfg.c_eps == 0.7                     # rn_ediss (ref default)
    assert cfg.tke_background == 1.0e-6         # rn_emin
    assert cfg.tke_surface_min == 1.0e-4        # rn_emin0
    assert cfg.tke_mxl_choice == 3              # nn_mxl=3 (lup/ldown + ln_mxl0)
    # nn_pdl=1: Pr = clamp(Ri/ri_cri, 1, 10), ri_cri = 2/(2+ediss/ediff) = 2/9
    assert cfg.prandtl_mode == "richardson"
    assert cfg.prandtl_ri_coeff == pytest.approx(4.5)
    assert cfg.lc is True                       # ln_lc
    assert cfg.lc_coeff == 0.25                 # rn_lc (namelist_cfg override)
    assert cfg.etau_mode == "below_ml"          # nn_etau = 1
    assert cfg.etau_frac == 0.08                # rn_efr (namelist_cfg override)
    assert cfg.etau_htau_mode == "latitude"     # nn_htau = 1 (ref default)
    # &namzdf rn_avm0/rn_avt0 backgrounds live INSIDE the closure floors:
    # compute_K_from_tke's max(closure, kappaM_min/kappaH_min) IS NEMO's
    # avm = max(closure, avmb) / avt = max(pdl*avt, avtb) composition
    # (zdftke.F90:715,723); the model-level additive A_v/K_v are forced
    # molecular by build_tripole so nothing is double-counted.
    assert cfg.kappaM_min == 1.2e-4             # rn_avm0
    assert cfg.kappaH_min == 1.2e-5             # rn_avt0
    # ln_zdfiwm: zdfiwm_init forces avmb/avtb to molecular — the wave field
    # is the interior background.
    cfg_iwm = r.orca1_zdftke_config(iwm_enabled=True)
    assert cfg_iwm.kappaM_min == constants.nu_ocean_molecular
    assert cfg_iwm.kappaH_min == 1.0e-10
    assert cfg_iwm._replace(
        kappaM_min=cfg.kappaM_min, kappaH_min=cfg.kappaH_min) == cfg
    assert cfg.bg_diff_scale == 0.0             # nn_avb = 0 (no depth profile)
    # rn_ebb=67.83 has no config field: it is tke.py's module constant.
    from legoesm.ocean.physics.vertical_mixing import tke as tke_mod
    assert tke_mod._NEMO_TKE_EBB == 67.83
    # NEMO integrates en prognostically -> the ORCA1 card default (2026-07-24).
    assert cfg.prognostic is True
    assert cfg.n2_mode == "insitu"
    # prognostic carry uses the vertical TKE solve only (no horizontal en
    # advection): advection_scheme stays "none" (Veros vs.dtke path off).
    assert cfg.advection_scheme == "none"


# ---------------------------------------------------------------------------
# build_tripole_vmix_config dispatch + iwm composition
# ---------------------------------------------------------------------------


def test_builder_none_matches_legacy_object():
    """Default 'none' reproduces the pre-flag VerticalMixingConfig EXACTLY,
    so --tripole-vmix none (and the default) stays byte-identical."""
    from legoesm.ocean.physics.vertical_mixing.config import (
        VerticalMixingConfig,
    )
    r = _runner()
    assert r.build_tripole_vmix_config("none") == VerticalMixingConfig(
        scheme="none")


def test_builder_tke_attaches_namelist_mapping():
    r = _runner()
    vm = r.build_tripole_vmix_config("tke")
    assert vm.scheme == "tke"
    assert vm.tke == r.orca1_zdftke_config()
    # iwm stays at its (disabled) default when not requested.
    assert vm.iwm.enabled is False


def test_builder_kpp_defaults():
    from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
    r = _runner()
    vm = r.build_tripole_vmix_config("kpp")
    assert vm.scheme == "kpp"
    assert vm.kpp == KPPConfig()


def test_builder_unknown_raises():
    r = _runner()
    with pytest.raises(ValueError, match="tripole-vmix"):
        r.build_tripole_vmix_config("tk")   # typo must not run other physics


def test_builder_iwm_composes_with_closure():
    """--iwm must ADD onto the closure (NEMO zdfphy: zdf_tke then zdf_iwm
    adds onto avt/avm), never clobber the scheme back to 'none'."""
    from legoesm.ocean.physics.vertical_mixing.internal_wave_mixing import (
        IWMConfig,
    )
    r = _runner()
    iwm = IWMConfig(enabled=True)
    vm = r.build_tripole_vmix_config("tke", iwm=iwm)
    assert vm.scheme == "tke"               # closure survives
    # under iwm the TKE floors flip to the zdfiwm_init molecular backgrounds
    assert vm.tke == r.orca1_zdftke_config(iwm_enabled=True)
    assert vm.iwm is iwm                    # additive wave K attached
    # legacy behaviour: iwm alone rides scheme="none"
    vm0 = r.build_tripole_vmix_config("none", iwm=iwm)
    assert vm0.scheme == "none" and vm0.iwm is iwm
    # a DISABLED iwm is not attached (defaults preserved, namzdf floors)
    vm1 = r.build_tripole_vmix_config("tke", iwm=IWMConfig(enabled=False))
    assert vm1.iwm.enabled is False
    assert vm1.tke == r.orca1_zdftke_config(iwm_enabled=False)


def test_orca1_zdftke_surface_bc_override():
    """--tke-surface-bc: None keeps the TKEConfig default (veros_flux, the
    flagged gap); 'nemo_dirichlet' selects NEMO's en(1)=rn_ebb|tau|/rho0
    Dirichlet BC; an unknown value raises (not a silent fallthrough)."""
    r = _runner()
    # 2026-07-24: the ORCA1 card DEFAULT is now the NEMO Dirichlet BC (the
    # flagged veros_flux gap is closed); --tke-surface-bc veros_flux reverts.
    assert r.orca1_zdftke_config().surface_bc == "nemo_dirichlet"       # default
    assert r.orca1_zdftke_config(surface_bc=None).surface_bc == "nemo_dirichlet"
    vf = r.orca1_zdftke_config(surface_bc="veros_flux")
    assert vf.surface_bc == "veros_flux"
    # ONLY the surface BC changes — every other leaf is byte-identical.
    assert vf._replace(surface_bc="nemo_dirichlet") == r.orca1_zdftke_config()
    with pytest.raises(ValueError, match="surface_bc"):
        r.orca1_zdftke_config(surface_bc="dirichlet")     # typo must raise


def test_builder_tke_surface_bc_threads():
    """build_tripole_vmix_config threads --tke-surface-bc onto the closure and
    it composes additively with --iwm (the surface BC rides the same config)."""
    from legoesm.ocean.physics.vertical_mixing.internal_wave_mixing import (
        IWMConfig,
    )
    r = _runner()
    vm = r.build_tripole_vmix_config("tke", tke_surface_bc="nemo_dirichlet")
    assert vm.scheme == "tke"
    assert vm.tke.surface_bc == "nemo_dirichlet"
    assert vm.tke == r.orca1_zdftke_config(surface_bc="nemo_dirichlet")
    # default (None) leaves the closure byte-identical to the plain mapping
    assert r.build_tripole_vmix_config("tke").tke == r.orca1_zdftke_config()
    # composes with iwm (floors flip to molecular, surface BC still applied)
    vmi = r.build_tripole_vmix_config(
        "tke", iwm=IWMConfig(enabled=True), tke_surface_bc="nemo_dirichlet")
    assert vmi.tke == r.orca1_zdftke_config(
        iwm_enabled=True, surface_bc="nemo_dirichlet")


def test_tke_surface_bc_requires_tke_closure():
    """--tke-surface-bc with a non-TKE closure raises (dispatch hardening):
    the surface-TKE BC does not exist for 'none'/'kpp', so silently ignoring
    it would run a different config than the flag implies."""
    r = _runner()
    for vmix in ("none", "kpp"):
        with pytest.raises(ValueError, match="requires --tripole-vmix tke"):
            r.build_tripole_vmix_config(vmix, tke_surface_bc="nemo_dirichlet")


def test_build_tripole_keyword_surface_bc_defaults_none():
    """The build_tripole pass-through exists and defaults to None (unset =
    no behaviour change for existing runs)."""
    r = _runner()
    params = inspect.signature(r.build_tripole).parameters
    assert "tke_surface_bc" in params
    assert params["tke_surface_bc"].default is None


def test_orca1_zdftke_mxl_choice_override():
    """--tke-mxl-choice: None keeps the card value (now 3 = NEMO nn_mxl=3,
    lup/ldown sweeps + ln_mxl0 anchor); 2 reverts to Veros; 4 selects NEMO
    nn_mxl=2 (single length); unknown raises."""
    r = _runner()
    # 2026-07-24: card DEFAULT is now nn_mxl=3 (ln_mxl0 anchor); =2 reverts.
    assert r.orca1_zdftke_config().tke_mxl_choice == 3                # default
    assert r.orca1_zdftke_config(mxl_choice=None).tke_mxl_choice == 3
    c2 = r.orca1_zdftke_config(mxl_choice=2)
    assert c2.tke_mxl_choice == 2
    # ONLY the mixing-length choice changes; every other leaf byte-identical.
    assert c2._replace(tke_mxl_choice=3) == r.orca1_zdftke_config()
    # 4 = NEMO nn_mxl=2 (the value the ORCA1 namelist actually runs): the same
    # lup/ldown sweeps as 3 but a SINGLE length, l_eps = l_k = min(lup,ldn).
    # It was in the `bad` list until it was implemented.
    c4 = r.orca1_zdftke_config(mxl_choice=4)
    assert c4.tke_mxl_choice == 4
    assert c4._replace(tke_mxl_choice=3) == r.orca1_zdftke_config()
    for bad in (1, 0, 5):
        with pytest.raises(ValueError, match="mxl_choice"):
            r.orca1_zdftke_config(mxl_choice=bad)
    # composes with surface_bc (both overrides apply, independent)
    both = r.orca1_zdftke_config(surface_bc="veros_flux", mxl_choice=2)
    assert both.tke_mxl_choice == 2 and both.surface_bc == "veros_flux"


def test_builder_tke_mxl_choice_threads():
    """build_tripole_vmix_config threads --tke-mxl-choice onto the closure.
    Uses the NON-default 2 so a broken forward would be caught (default is 3)."""
    r = _runner()
    vm = r.build_tripole_vmix_config("tke", tke_mxl_choice=2)
    assert vm.tke.tke_mxl_choice == 2
    assert vm.tke == r.orca1_zdftke_config(mxl_choice=2)
    assert r.build_tripole_vmix_config("tke").tke.tke_mxl_choice == 3  # default
    # off-tke closure rejects (dispatch hardening), like the other knobs
    for vmix in ("none", "kpp"):
        with pytest.raises(ValueError, match="tke-mxl-choice"):
            r.build_tripole_vmix_config(vmix, tke_mxl_choice=3)


def test_orca1_zdftke_prognostic_override():
    """--tke-prognostic: None keeps the card value (now True = NEMO prognostic
    Mode-A en); False reverts to diagnostic Mode-B. Composes with other knobs."""
    r = _runner()
    # 2026-07-24: card DEFAULT is now prognostic=True (NEMO Mode-A); False reverts.
    assert r.orca1_zdftke_config().prognostic is True                # default
    assert r.orca1_zdftke_config(prognostic=None).prognostic is True
    cd = r.orca1_zdftke_config(prognostic=False)
    assert cd.prognostic is False
    # ONLY prognostic changes; every other leaf byte-identical.
    assert cd._replace(prognostic=True) == r.orca1_zdftke_config()
    # composes with surface_bc + mxl_choice (all three independent overrides)
    allc = r.orca1_zdftke_config(surface_bc="veros_flux", mxl_choice=2,
                                 prognostic=False)
    assert (allc.prognostic is False and allc.tke_mxl_choice == 2
            and allc.surface_bc == "veros_flux")


def test_builder_tke_prognostic_threads():
    """build_tripole_vmix_config threads --tke-prognostic onto the closure.
    Uses the NON-default False so a broken forward is caught (default is True)."""
    r = _runner()
    vm = r.build_tripole_vmix_config("tke", tke_prognostic=False)
    assert vm.tke.prognostic is False
    assert vm.tke == r.orca1_zdftke_config(prognostic=False)
    assert r.build_tripole_vmix_config("tke").tke.prognostic is True  # default
    for vmix in ("none", "kpp"):
        with pytest.raises(ValueError, match="tke-prognostic"):
            r.build_tripole_vmix_config(vmix, tke_prognostic=True)


def test_tke_card_knobs_require_tripole_tke():
    """--tke-eice / --tke-surface-bc are applied ONLY in the tke branch of
    build_tripole_vmix_config; they are silently discarded on every other
    (grid, tripole_vmix) context.  The guard must raise on ALL THREE discard
    paths and NOT fire when the full tripole+tke context holds or the knobs
    are unset (no false positive)."""
    r = _runner()
    # the ONLY valid context: tripole + tke closure -> no raise
    r._validate_tke_card_grid("tripole", "tke", tke_eice=3,
                              tke_surface_bc="nemo_dirichlet")
    # unset knobs: allowed on any (grid, vmix) -> no regression
    r._validate_tke_card_grid("mpas", "none")
    r._validate_tke_card_grid("tripole", "none")
    r._validate_tke_card_grid("latlon_bathy", "kpp")
    # discard path 1: wrong grid (build_tripole never runs)
    for grid in ("mpas", "latlon_bathy", "cubed_sphere"):
        with pytest.raises(SystemExit, match="tke-surface-bc"):
            r._validate_tke_card_grid(grid, "tke", tke_surface_bc="nemo_dirichlet")
        with pytest.raises(SystemExit, match="tke-eice"):
            r._validate_tke_card_grid(grid, "tke", tke_eice=1)
    # discard path 2: tripole but vmix none (attach block skipped) — codex HIGH
    with pytest.raises(SystemExit, match="tke-surface-bc"):
        r._validate_tke_card_grid("tripole", "none",
                                  tke_surface_bc="nemo_dirichlet")
    # discard path 3: tripole but vmix kpp (knob not applied in kpp branch)
    with pytest.raises(SystemExit, match="tke-eice"):
        r._validate_tke_card_grid("tripole", "kpp", tke_eice=1)
    # --tke-mxl-choice + --tke-prognostic are guarded the same way (all paths)
    r._validate_tke_card_grid("tripole", "tke", tke_mxl_choice=3,
                              tke_prognostic=True)                   # allowed
    for grid, vmix in (("mpas", "tke"), ("latlon_bathy", "tke"),
                       ("tripole", "none"), ("tripole", "kpp")):
        with pytest.raises(SystemExit, match="tke-mxl-choice"):
            r._validate_tke_card_grid(grid, vmix, tke_mxl_choice=3)
        with pytest.raises(SystemExit, match="tke-prognostic"):
            r._validate_tke_card_grid(grid, vmix, tke_prognostic=True)


def test_main_wires_tke_card_guard_before_builders(monkeypatch):
    """main() must CALL the guard, and BEFORE the grid builders run — proving
    the wiring + ordering the helper test alone cannot (codex MED).  A tripwire
    on build_mpas_ocean fires only if the guard were removed/misordered."""
    import sys
    r = _runner()

    def _boom(*a, **k):                    # must never be reached
        raise AssertionError("build_mpas_ocean ran before the guard rejected")

    monkeypatch.setattr(r, "build_mpas_ocean", _boom, raising=False)
    monkeypatch.setattr(
        sys, "argv",
        ["run_omip_core2.py", "--grid", "mpas", "--mesh", "/nonexistent.nc",
         "--tke-surface-bc", "nemo_dirichlet"])
    with pytest.raises(SystemExit, match="tke-surface-bc"):
        r.main()


def test_build_tripole_rejects_unknown_vmix_at_entry():
    """''/None/typos must not silently run as 'none' at the programmatic
    surface (argparse choices only guard the CLI).  The guard fires before
    any grid/mesh work, so dummy args suffice."""
    r = _runner()
    for bad in ("", None, "tk"):
        with pytest.raises(ValueError, match="tripole_vmix"):
            r.build_tripole(75, 6000.0, "/nonexistent/mesh.nc",
                            tripole_vmix=bad)


def test_build_tripole_keyword_defaults_none():
    """The pass-through wiring exists and defaults byte-identical."""
    r = _runner()
    params = inspect.signature(r.build_tripole).parameters
    assert "tripole_vmix" in params
    assert params["tripole_vmix"].default == "none"


# ---------------------------------------------------------------------------
# nn_htau=1 "latitude" penetration on 2-D-horizontal columns
# ---------------------------------------------------------------------------


def test_etau_latitude_broadcasts_2d_columns():
    """h_tau = max(0.5, min(30, 45|sin lat|)) with 2-D (n_lat, n_lon) lat_deg
    against (n_lat, n_lon, nlev-1) columns — the shape contract the model's
    grid.lat_T threading relies on."""
    from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
    from legoesm.ocean.physics.vertical_mixing.tke import (
        _NEMO_TKE_EMIN0, nemo_etau_injection,
    )
    n_lat, n_lon, nk = 3, 4, 5
    cfg = TKEConfig(etau_mode="below_ml", etau_htau_mode="latitude",
                    etau_frac=0.08)
    e = jnp.zeros((n_lat, n_lon, nk))
    taum = jnp.zeros((n_lat, n_lon))
    depth_w = jnp.asarray([5.0, 15.0, 30.0, 60.0, 120.0])
    lat = jnp.broadcast_to(
        jnp.asarray([0.0, 45.0, 80.0])[:, None], (n_lat, n_lon))
    out = np.asarray(nemo_etau_injection(
        e, taum, depth_w, cfg, rho_0=1026.0, lat_deg=lat))
    assert out.shape == (n_lat, n_lon, nk)
    base = cfg.etau_frac * _NEMO_TKE_EMIN0
    htau = np.maximum(0.5, np.minimum(30.0, 45.0 * np.abs(
        np.sin(np.deg2rad(np.asarray([0.0, 45.0, 80.0]))))))
    for j, h in enumerate(htau):
        want = base * np.exp(-np.asarray(depth_w) / h)
        np.testing.assert_allclose(out[j, 0], want, rtol=1e-12)
        np.testing.assert_allclose(out[j, -1], want, rtol=1e-12)


def test_model_step_orca1_tke_latitude_mode():
    """End-to-end: a tiny lat-lon C-grid model (n_lat != n_lon) running the
    ORCA1 zdftke mapping (etau latitude mode + lc) traces + steps finitely
    through the implicit-solve K-profile fallback.  With the legacy 1-D
    ``grid.lat`` threading this raises a broadcast error at trace time."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.surface_forcing.config import (
        SurfaceForcingConfig,
    )
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig

    r = _runner()
    n_lat, n_lon = 6, 12          # distinct so a 1-D lat mis-broadcast trips
    grid = create_latlon_grid(n_lat, n_lon)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=4000.0, land_lat_threshold=80.0)
    physics = OceanPhysicsConfig(
        vertical_mixing=r.build_tripole_vmix_config("tke"),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,
    )
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4, bottom_drag_r=1.0e-3, n_barotropic_substeps=8,
        implicit_vertical_mixing=True, physics=physics,
        enable_runtime_checks=False)
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    out = model.step(state, 900.0)
    for name in ("T", "S", "u", "v", "eta"):
        arr = np.asarray(getattr(out, name).data)
        assert np.isfinite(arr).all(), f"non-finite {name}"
