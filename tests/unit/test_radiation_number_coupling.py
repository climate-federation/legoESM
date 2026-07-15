"""Droplet/ice NUMBER → radiation effective-radius coupling must be wired on
EVERY grid and for EVERY microphysics scheme, not just the cubed-sphere NH path.

Background
----------
``compute_cloud_properties`` derives the M2005 PSD liquid/ice effective radii
from the cloud-droplet number ``N_c`` (per-VOLUME [#/m³], tracer slot 6) and ice
number ``N_i`` (per-MASS [#/kg], tracer slot 8).  Originally only
``model_type='nonhydrostatic'`` extracted those slots; the plane CRM, MPAS NH,
hydrostatic (lat-lon / cubed-sphere) and spectral-PE radiation paths silently
passed ``n_cloud=n_ice=None`` ⇒ RRTMGP fell back to a FIXED r_eff regardless of
the droplet number — unfaithful to SAM whenever a double-moment scheme
(Morrison / Seifert-Beheng) was active.

This module pins the coupling for the whole {grid × microphysics} matrix:

* GRID coverage — every ``model_type`` that ``make_radiation_physics`` dispatches:
  - array-slot states (``plane``, ``nonhydrostatic``, ``mpas_nh``): the radiation
    ``physics_fn`` is run with a spy on ``_call_radiation_backend`` and we assert
    the captured ``n_cloud`` / ``n_ice`` equal tracer slots 6 / 8 for a 9-slot
    (double-moment) state and are ``None`` for a 3-slot (single-moment) state;
  - dict-tracer states (``hydrostatic`` / lat-lon, ``spectral_pe``): the shared
    ``_extract_tracer_columns`` helper is exercised directly (those paths forward
    its 5-tuple verbatim to the backend).
* MICROPHYSICS coverage — the slot layout IS the scheme contract: single-moment
  ``kessler`` carries 3 slots (no number ⇒ constant r_eff), double-moment
  ``morrison`` / ``seifert_beheng`` carry 9 slots incl. N_c (6) and N_i (8).
  ``test_scheme_slot_contract`` ties the two together so the matrix is explicit.

The spy intercepts AFTER the per-path tracer extraction, so a cheap ``gray``
radiation config is used — no RRTMGP k-distribution load — yet the
extraction→backend wiring under test is exactly what runs under RRTMGP.
"""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.physics.radiation import integration as rad_int
from legoesm.atmosphere.physics.radiation.config import (
    GrayRadiationConfig,
    RadiationConfig,
)
from legoesm.atmosphere.physics.radiation.integration import (
    _extract_tracer_columns,
    make_radiation_physics,
)

jax.config.update("jax_enable_x64", True)

# Sentinel number values placed in slots 6 (N_c, per-volume #/m³) and 8 (N_i,
# per-mass #/kg) so a successful capture is unambiguous (≠ q_v/q_c/q_i magnitudes).
_NC_VAL = 1.0e8
_NI_VAL = 5.0e3
_GRAY = RadiationConfig(scheme="gray", gray=GrayRadiationConfig())


@pytest.fixture
def backend_spy(monkeypatch):
    """Patch ``_call_radiation_backend`` to capture the n_cloud / n_ice kwargs
    the per-grid radiation ``physics_fn`` extracts, returning a minimal output
    (``heating_rate`` shaped like the input ``T``) so the path completes."""
    captured: dict = {}

    def _fake_backend(**kw):
        captured["n_cloud"] = kw.get("n_cloud")
        captured["n_ice"] = kw.get("n_ice")
        captured["q_cloud"] = kw.get("q_cloud")
        captured["q_ice"] = kw.get("q_ice")
        T = kw["T"]
        return SimpleNamespace(heating_rate=jnp.zeros_like(T))

    monkeypatch.setattr(rad_int, "_call_radiation_backend", _fake_backend)
    return captured


# --------------------------------------------------------------------- #
# Per-grid state builders (3-slot single-moment vs 9-slot double-moment) #
# --------------------------------------------------------------------- #


def _plane_state(n_slots):
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        make_flat_plane_terrain_metric,
        make_rest_state,
    )
    from legoesm.grids.plane import create_plane_grid
    from legoesm.grids.vertical import create_height_coordinate

    ny = nx = 4
    nlev = 8
    grid = create_plane_grid(nx=nx, ny=ny, nlev=nlev, dx=4_000.0,
                             dy=4_000.0, dtype=jnp.float64)
    hc = create_height_coordinate(nlev, H=20_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    rest = make_rest_state(grid, hc, dtype=jnp.float64)
    tracers = jnp.zeros((ny, nx, nlev, n_slots), dtype=jnp.float64)
    tracers = tracers.at[..., 0].set(0.01)        # q_v
    tracers = tracers.at[..., 1].set(1.0e-3)      # q_c
    if n_slots > 3:
        tracers = tracers.at[..., 3].set(5.0e-4)  # q_i
        tracers = tracers.at[..., 6].set(_NC_VAL)  # N_c
        tracers = tracers.at[..., 8].set(_NI_VAL)  # N_i
    state = rest._replace(tracers=rest.tracers.replace(data=tracers))
    return state, grid, hc, tm


def _nonhydro_state(n_slots):
    from legoesm.core.field import Field
    from legoesm.core.state import NonHydrostaticState
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import (
        compute_terrain_metric,
        create_height_coordinate,
    )

    n = 4
    nlev = 8
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, 20_000.0)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    d3 = ("face", "x", "y", "level")
    dw = ("face", "x", "y", "level_half")
    d2 = ("face", "x", "y")
    dtr = ("face", "x", "y", "level", "tracer")
    tracers = jnp.zeros((6, n, n, nlev, n_slots), dtype=jnp.float64)
    tracers = tracers.at[..., 0].set(0.01)
    tracers = tracers.at[..., 1].set(1.0e-3)
    if n_slots > 3:
        tracers = tracers.at[..., 3].set(5.0e-4)
        tracers = tracers.at[..., 6].set(_NC_VAL)
        tracers = tracers.at[..., 8].set(_NI_VAL)
    z = jnp.zeros((6, n, n, nlev), jnp.float64)
    state = NonHydrostaticState(
        u=Field(z, name="u", dims=d3, units="m/s"),
        v=Field(z, name="v", dims=d3, units="m/s"),
        w=Field(jnp.zeros((6, n, n, nlev + 1), jnp.float64), name="w",
                dims=dw, units="m/s"),
        theta_prime=Field(z, name="theta_prime", dims=d3, units="K"),
        rho_prime=Field(z, name="rho_prime", dims=d3, units="kg/m^3"),
        phis=Field(jnp.zeros((6, n, n), jnp.float64), name="phis",
                   dims=d2, units="m^2/s^2"),
        tracers=Field(tracers, name="tracers", dims=dtr, units="kg/kg"),
    )
    return state, grid, hc, tm


def _mpas_state(n_slots):
    from legoesm.core.field import Field
    from legoesm.core.state import MPASNonHydrostaticState
    from legoesm.grids.vertical import (
        compute_terrain_metric,
        create_height_coordinate,
    )
    from legoesm.grids.voronoi import create_voronoi_mesh

    nlev = 8
    mesh = create_voronoi_mesh(2, lloyd_iterations=5)
    hc = create_height_coordinate(nlev, 20_000.0)
    tm = compute_terrain_metric(jnp.zeros((mesh.nCells,)), hc)
    dc = ("nCells", "nlev")
    dw = ("nCells", "nlev_half")
    dtr = ("nCells", "nlev", "tracer")
    tracers = jnp.zeros((mesh.nCells, nlev, n_slots), dtype=jnp.float64)
    tracers = tracers.at[..., 0].set(0.01)
    tracers = tracers.at[..., 1].set(1.0e-3)
    if n_slots > 3:
        tracers = tracers.at[..., 3].set(5.0e-4)
        tracers = tracers.at[..., 6].set(_NC_VAL)
        tracers = tracers.at[..., 8].set(_NI_VAL)
    z = jnp.zeros((mesh.nCells, nlev), jnp.float64)
    state = MPASNonHydrostaticState(
        # MPAS carries edge-normal velocity ``u`` (nEdges) only — no ``v``.
        u=Field(jnp.zeros((mesh.nEdges, nlev), jnp.float64), name="u",
                dims=("nEdges", "nlev"), units="m/s"),
        w=Field(jnp.zeros((mesh.nCells, nlev + 1), jnp.float64), name="w",
                dims=dw, units="m/s"),
        theta_prime=Field(z, name="theta_prime", dims=dc, units="K"),
        rho_prime=Field(z, name="rho_prime", dims=dc, units="kg/m^3"),
        phis=Field(jnp.zeros((mesh.nCells,), jnp.float64), name="phis",
                   dims=("nCells",), units="m^2/s^2"),
        tracers=Field(tracers, name="tracers", dims=dtr, units="kg/kg"),
    )
    return state, mesh, hc, tm


_ARRAY_SLOT_GRIDS = {
    "plane": _plane_state,
    "nonhydrostatic": _nonhydro_state,
    "mpas_nh": _mpas_state,
}


# --------------------------------------------------------------------- #
# GRID coverage — array-slot dycores (plane / cubed-sphere NH / MPAS NH) #
# --------------------------------------------------------------------- #


class TestArraySlotGridCoupling:
    @pytest.mark.parametrize("model_type", list(_ARRAY_SLOT_GRIDS))
    def test_double_moment_9slot_passes_number(self, model_type, backend_spy):
        """9-slot (Morrison / Seifert-Beheng) state ⇒ radiation backend receives
        n_cloud = slot 6 (N_c) and n_ice = slot 8 (N_i)."""
        state, grid, hc, tm = _ARRAY_SLOT_GRIDS[model_type](9)
        physics_fn = make_radiation_physics(_GRAY, model_type=model_type)
        physics_fn(state, grid, hc, tm)
        assert backend_spy["n_cloud"] is not None, model_type
        assert backend_spy["n_ice"] is not None, model_type
        assert float(jnp.max(backend_spy["n_cloud"])) == pytest.approx(_NC_VAL)
        assert float(jnp.max(backend_spy["n_ice"])) == pytest.approx(_NI_VAL)

    @pytest.mark.parametrize("model_type", list(_ARRAY_SLOT_GRIDS))
    def test_single_moment_3slot_passes_none(self, model_type, backend_spy):
        """3-slot (kessler / single-moment) state ⇒ no number tracers ⇒
        n_cloud / n_ice are None (radiation falls back to constant r_eff)."""
        state, grid, hc, tm = _ARRAY_SLOT_GRIDS[model_type](3)
        physics_fn = make_radiation_physics(_GRAY, model_type=model_type)
        physics_fn(state, grid, hc, tm)
        assert backend_spy["n_cloud"] is None, model_type
        assert backend_spy["n_ice"] is None, model_type
        # q_cloud (slot 1) is still wired — only the NUMBER is absent.
        assert backend_spy["q_cloud"] is not None, model_type


# --------------------------------------------------------------------- #
# GRID coverage — dict-tracer dycores (hydrostatic / lat-lon, spectral) #
# --------------------------------------------------------------------- #


class TestDictTracerExtraction:
    """``hydrostatic`` (lat-lon + cubed-sphere) and ``spectral_pe`` forward the
    ``_extract_tracer_columns`` 5-tuple verbatim to the backend, so testing the
    helper pins those two paths."""

    def _tracers(self, with_number):
        ncol, nlev = 6, 8
        t = {
            "q_v": jnp.full((ncol, nlev), 0.01),
            "q_c": jnp.full((ncol, nlev), 1.0e-3),
            "q_i": jnp.full((ncol, nlev), 5.0e-4),
        }
        if with_number:
            t["N_c"] = jnp.full((ncol, nlev), _NC_VAL)
            t["N_i"] = jnp.full((ncol, nlev), _NI_VAL)
        return SimpleNamespace(tracers=t, T=SimpleNamespace(
            data=jnp.zeros((ncol, nlev)))), ncol, nlev

    def test_double_moment_keys_extracted(self):
        state, ncol, nlev = self._tracers(with_number=True)
        qv, qc, qi, nc, ni = _extract_tracer_columns(state, ncol, nlev)
        assert nc is not None and ni is not None
        assert float(jnp.max(nc)) == pytest.approx(_NC_VAL)
        assert float(jnp.max(ni)) == pytest.approx(_NI_VAL)
        assert nc.shape == (ncol, nlev) and ni.shape == (ncol, nlev)

    def test_single_moment_keys_absent_is_none(self):
        state, ncol, nlev = self._tracers(with_number=False)
        qv, qc, qi, nc, ni = _extract_tracer_columns(state, ncol, nlev)
        assert nc is None and ni is None
        assert qc is not None and qi is not None   # condensate still wired


# --------------------------------------------------------------------- #
# MICROPHYSICS coverage — the slot count IS the scheme contract         #
# --------------------------------------------------------------------- #


def test_scheme_slot_contract():
    """Tie the grid tests (which key off slot count) to the microphysics
    schemes. Single-moment (kessler / sundqvist) need ≤3 slots → no number →
    constant r_eff. EVERY double-moment scheme uses the canonical 9-slot layout
    (q_v,q_c,q_r,q_i,q_s,q_g,N_c,N_r,N_i — p3 reuses slots 4/5 for q_rim/B_rim
    but still keeps N_c at 6 / N_i at 8), so the radiation paths' slot-6/8
    extraction is valid for all of them."""
    from legoesm.atmosphere.physics.microphysics.integration import (
        _PLANE_MIN_TRACER_SLOTS,
    )
    assert _PLANE_MIN_TRACER_SLOTS["kessler"] <= 3
    assert _PLANE_MIN_TRACER_SLOTS["sundqvist"] <= 3
    for scheme in ("morrison", "seifert_beheng", "thompson", "p3",
                   "ml_emulator"):
        assert _PLANE_MIN_TRACER_SLOTS[scheme] == 9, scheme


def test_ice_condensate_reaches_rrtmgp_iwp():
    """Morrison-style ice (q_i>0) must reach RRTMGP as a NONZERO ice water path,
    and the ice number (N_i) must drive the M2005 effective radius rather than
    the constant fallback.  This locks the q_ice -> IWP -> rrtmgp end of the
    chain (the array-slot grid tests assert n_ice, not q_ice/IWP)."""
    from legoesm.atmosphere.physics.clouds.cloud_fraction import (
        compute_cloud_properties,
    )
    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    from legoesm import constants

    ncol, nlev = 4, 6
    T = jnp.full((ncol, nlev), 250.0)        # below freezing: ice regime
    p = jnp.full((ncol, nlev), 5.0e4)
    q_v = jnp.full((ncol, nlev), 1.0e-4)
    dp = jnp.full((ncol, nlev), 1.0e4)
    q_i = jnp.full((ncol, nlev), 5.0e-4)
    cfg = CloudConfig(scheme="sundqvist")

    # q_i with NO number -> IWP nonzero (= q_i*dp/g), constant-fallback r_eff.
    kw0 = compute_cloud_properties(
        T=T, p_full=p, q_v=q_v, dp=dp, config=cfg, q_ice=q_i).to_rrtmg_kwargs()
    assert float(jnp.min(kw0["cloud_path_ice"])) > 0.0
    exp_iwp = 5.0e-4 * 1.0e4 / constants.g
    assert float(jnp.max(kw0["cloud_path_ice"])) == pytest.approx(exp_iwp, rel=1e-6)

    # Adding the ice NUMBER changes the ice effective radius (M2005 PSD active).
    kw1 = compute_cloud_properties(
        T=T, p_full=p, q_v=q_v, dp=dp, config=cfg, q_ice=q_i,
        n_ice=jnp.full((ncol, nlev), 1.0e5)).to_rrtmg_kwargs()
    assert bool(jnp.all(jnp.isfinite(kw1["cloud_r_eff_ice"])))
    assert float(jnp.max(jnp.abs(
        kw1["cloud_r_eff_ice"] - kw0["cloud_r_eff_ice"]))) > 0.0


def test_number_extraction_ad_connected(monkeypatch):
    """The slot-6/8 extraction must be AD-connected (not a stop-gradient): with
    a backend whose heating depends on n_cloud, grad of the radiation tendency
    w.r.t. the tracer field is FINITE and NON-ZERO at slot 6 (N_c) — proving the
    droplet number actually flows through to radiation, not just that it's
    finite under gray (which ignores it)."""
    def _nc_backend(**kw):
        T = kw["T"]
        nc = kw.get("n_cloud")
        # heating proportional to the droplet number so grad flows through it
        hr = jnp.zeros_like(T) if nc is None else (1.0e-12 * nc)
        return SimpleNamespace(heating_rate=hr.reshape(T.shape))

    monkeypatch.setattr(rad_int, "_call_radiation_backend", _nc_backend)
    state, grid, hc, tm = _plane_state(9)
    physics_fn = make_radiation_physics(_GRAY, model_type="plane")
    base = state.tracers.data

    def loss(tr):
        s = state._replace(tracers=state.tracers.replace(data=tr))
        t = physics_fn(s, grid, hc, tm)
        return jnp.sum(t.dtheta_prime_dt.data ** 2)

    g = jax.grad(loss)(base)
    assert g.shape == base.shape
    assert bool(jnp.all(jnp.isfinite(g)))
    # Gradient must be non-zero at the N_c slot (6) — the number truly couples.
    assert float(jnp.max(jnp.abs(g[..., 6]))) > 0.0
