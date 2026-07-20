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
    assert cfg.tke_mxl_choice == 2              # nn_mxl = 2 (closest)
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
    # Structural conventions follow the validated DINO recipe (defaults).
    assert cfg.prognostic is False
    assert cfg.n2_mode == "insitu"
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


def test_builder_kpp_honors_overrides():
    """tripole-vmix kpp threads --kpp-ri-crit/--kpp-cv/--kpp-eice through the
    SAME _kpp_vmix_override as the mpas/latlon builders, so a tripole-KPP run
    can hold KPP params byte-identical to an MPAS-KPP run (the same-scheme
    cross-grid pair isolating the grid effect)."""
    from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
    r = _runner()
    vm = r.build_tripole_vmix_config(
        "kpp", kpp_ri_crit=0.15, kpp_eice=3)
    assert vm.scheme == "kpp"
    assert vm.kpp.Ri_crit == 0.15
    assert vm.kpp.eice == 3
    # untouched fields keep scheme defaults (no re-derived override logic)
    assert vm.kpp.Cv == KPPConfig().Cv
    # and it matches the mpas/latlon override object exactly
    assert vm == r._kpp_vmix_override(0.15, None, 3)


def test_builder_kpp_overrides_rejected_on_other_closures():
    """KPP overrides on 'none'/'tke' would be silently dropped -> reject."""
    r = _runner()
    for vmix in ("none", "tke"):
        with pytest.raises(ValueError, match="tripole-vmix kpp"):
            r.build_tripole_vmix_config(vmix, kpp_ri_crit=0.15)
        with pytest.raises(ValueError, match="tripole-vmix kpp"):
            r.build_tripole_vmix_config(vmix, kpp_eice=3)


def test_builder_tke_eice_rejected_on_non_tke_closures():
    """--tke-eice only reaches the TKE closure; 'none'/'kpp' must reject
    (symmetric to the KPP-override reject), pointing at --kpp-eice."""
    r = _runner()
    for vmix in ("none", "kpp"):
        with pytest.raises(ValueError, match="tripole-vmix tke"):
            r.build_tripole_vmix_config(vmix, tke_eice=3)


def test_build_tripole_validates_closure_overrides_before_mesh():
    """codex-HIGH regression: with the DEFAULT --tripole-vmix none the
    optional-physics block in build_tripole is skipped, so the closure-mismatch
    rejects must run UNCONDITIONALLY (and before any mesh work) — a
    --tke-eice/--kpp-* override on the default closure raises the closure
    ValueError, NOT a silent no-op (and not a mesh FileNotFoundError, proving
    the validation precedes the eORCA load)."""
    r = _runner()
    bogus = "/nonexistent/mesh_that_must_never_be_opened.nc"
    with pytest.raises(ValueError, match="tripole-vmix tke"):
        r.build_tripole(75, 6000.0, bogus, tripole_vmix="none", tke_eice=3)
    with pytest.raises(ValueError, match="tripole-vmix kpp"):
        r.build_tripole(75, 6000.0, bogus, tripole_vmix="none", kpp_eice=3)
    with pytest.raises(ValueError, match="tripole-vmix kpp"):
        r.build_tripole(75, 6000.0, bogus, tripole_vmix="tke",
                        kpp_ri_crit=0.15)


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
